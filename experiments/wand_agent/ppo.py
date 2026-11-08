"""PPO (clipped objective, GAE) for one policy on all four levels, optionally with memory.

Structure follows CleanRL's ppo.py and ppo_atari_lstm.py (MIT licence,
https://github.com/vwxyzjn/cleanrl): collect num_steps from every game, compute
GAE advantages backwards, then run a few epochs of minibatch updates on that
rollout and discard it.

Conventions used throughout:
  ends[t]  True when action t ended its game. The observation after it already
           belongs to the next game, so neither the advantage nor the critic's
           estimate crosses that boundary.
  first[t] True when observation t is the first of a game. With memory (LSTM),
           the memory of that game is zeroed before observation t is read.
"""
import argparse
import json
import math
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .toy import MemoryCue
from .vector import GymVector, VectorEnv

ACTIVATIONS = {'tanh': nn.Tanh, 'relu': nn.ReLU}


def _layer(inputs, outputs, std=math.sqrt(2)):
    """Orthogonal weights, zero bias (one of PPO's standard details)."""
    layer = nn.Linear(inputs, outputs)
    nn.init.orthogonal_(layer.weight, std)
    nn.init.zeros_(layer.bias)
    return layer


def _body(sizes, activation):
    layers = []
    for inputs, outputs in zip(sizes[:-1], sizes[1:]):
        layers += [_layer(inputs, outputs), ACTIVATIONS[activation]()]
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Policy (logits) and critic (value) from the same observation.

    shared=False: two separate MLPs. shared=True: one trunk, two output heads.
    lstm=H > 0: shared trunk -> LSTM with H units -> both heads, as in CleanRL's
    LSTM agent. The LSTM state (h, c) is the agent's memory between steps.
    """

    def __init__(self, obs_dim, n_actions, hidden=(128, 128), activation='relu', shared=False, lstm=0):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f'Unknown activation {activation!r}')
        shared = bool(shared or lstm)
        self.meta = dict(obs_dim=int(obs_dim), n_actions=int(n_actions), hidden=list(hidden),
                         activation=activation, shared=shared, lstm=int(lstm))
        sizes = [obs_dim, *hidden]
        if shared:
            self.body = _body(sizes, activation)
        else:
            self.actor_body = _body(sizes, activation)
            self.critic_body = _body(sizes, activation)
        features = sizes[-1]
        if lstm:
            self.lstm = nn.LSTM(features, lstm)
            for name, param in self.lstm.named_parameters():
                if 'bias' in name:
                    nn.init.zeros_(param)
                else:
                    nn.init.orthogonal_(param, 1.0)
            features = lstm
        # Small actor weights: every action starts almost equally likely.
        self.actor = _layer(features, n_actions, std=0.01)
        self.critic = _layer(features, 1, std=1.0)

    @property
    def recurrent(self):
        return self.meta['lstm'] > 0

    def initial_state(self, n):
        """Empty memory for n games (None without LSTM)."""
        if not self.recurrent:
            return None
        return torch.zeros(1, n, self.meta['lstm']), torch.zeros(1, n, self.meta['lstm'])

    def sequence(self, obs, state=None, first=None):
        """Run T steps of N games in order.

        obs [T, N, D], first [T, N] -> logits [T, N, A], value [T, N], memory after step T.
        Without LSTM the steps are independent and state stays None.
        """
        steps, n = obs.shape[:2]
        flat = obs.reshape(steps * n, -1)
        if self.meta['shared']:
            actor_features = critic_features = self.body(flat)
        else:
            actor_features, critic_features = self.actor_body(flat), self.critic_body(flat)
        if self.recurrent:
            if first is None:
                first = torch.zeros(steps, n, dtype=torch.bool)
            h, c = state
            inputs = actor_features.reshape(steps, n, -1)
            outputs = []
            for t in range(steps):
                # Zero the memory of games whose observation t starts a new game.
                keep = (~first[t]).float().view(1, n, 1)
                out, (h, c) = self.lstm(inputs[t:t + 1], (h * keep, c * keep))
                outputs.append(out)
            actor_features = critic_features = torch.cat(outputs).reshape(steps * n, -1)
            state = (h, c)
        return (self.actor(actor_features).reshape(steps, n, -1),
                self.critic(critic_features).reshape(steps, n), state)

    def step(self, obs, state=None, first=None):
        """One step of N games: obs [N, D] -> logits [N, A], value [N], new memory."""
        logits, value, state = self.sequence(obs[None], state, None if first is None else first[None])
        return logits[0], value[0], state

    def forward(self, obs):
        """Memoryless policy only: obs [B, D] -> logits [B, A], value [B]."""
        if self.recurrent:
            raise RuntimeError('A model with memory needs step() or sequence() and a memory state')
        logits, value, _ = self.step(obs)
        return logits, value


def save_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f'w/{k}': v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, kind=np.array('ppo'), meta=np.array(json.dumps(model.meta)), **arrays)
    temporary.replace(path)


def load_model(path):
    with np.load(path, allow_pickle=False) as data:
        if str(data['kind']) != 'ppo':
            raise ValueError(f'{path} is not a PPO checkpoint')
        meta = json.loads(str(data['meta']))
        state = {k[2:]: torch.from_numpy(data[k].copy()) for k in data.files if k.startswith('w/')}
    if not all(torch.isfinite(v).all() for v in state.values()):
        raise ValueError('Checkpoint has nonfinite weights')
    model = ActorCritic(**meta)
    model.load_state_dict(state, strict=True)
    return model.eval()


def compute_gae(rewards, values, ends, next_value, gamma, lam):
    """Advantages and critic targets for a rollout of shape [T, N].

    delta(t) = r(t) + gamma * V(next state) - V(s(t)), with V(next state) = 0 when
    action t ended the game; next_value[N] is the critic's estimate after the
    last step (the bootstrap). A(t) = delta(t) + gamma * lam * A(t+1), computed
    from the end, also cut at game ends. Returns (advantages, advantages + values).
    """
    steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    following = torch.zeros_like(next_value)
    for t in reversed(range(steps)):
        continues = 1.0 - ends[t].float()
        next_values = next_value if t == steps - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_values * continues - values[t]
        following = delta + gamma * lam * continues * following
        advantages[t] = following
    return advantages, advantages + values


def _minibatches(model, rollout, c, rng):
    """Yield (logits, values, pick) for each minibatch of one epoch.

    pick(x) selects the entries of a [T, N] rollout tensor that belong to the
    minibatch, in the same order as logits and values. Without memory a
    minibatch is any set of transitions. With memory it is a set of whole games
    columns, replayed in order from the memory saved before the rollout.
    """
    obs = rollout['obs']
    steps, n = rollout['actions'].shape
    if model.recurrent:
        per = n // c['num_minibatches']
        order = rng.permutation(n)
        for start in range(0, n, per):
            columns = torch.from_numpy(order[start:start + per])
            state = tuple(s[:, columns] for s in rollout['state'])
            logits, values, _ = model.sequence(obs[:, columns], state, rollout['first'][:, columns])
            yield logits.reshape(-1, logits.shape[-1]), values.reshape(-1), \
                lambda x, columns=columns: x[:, columns].reshape(-1)
    else:
        size = steps * n
        per = size // c['num_minibatches']
        flat_obs = obs.reshape(size, -1)
        order = rng.permutation(size)
        for start in range(0, size, per):
            index = torch.from_numpy(order[start:start + per])
            logits, values = model(flat_obs[index])
            yield logits, values, lambda x, index=index: x.reshape(-1)[index]


def ppo_update(model, optimizer, rollout, c, rng):
    """Several epochs of clipped-objective minibatch updates on one rollout.

    rollout: dict of [T, N] tensors (obs is [T, N, D]): obs, actions, logp,
    advantages, returns, values, first; plus 'state', the memory before step 0.
    """
    stats = defaultdict(list)
    for _ in range(c['update_epochs']):
        for logits, value, pick in _minibatches(model, rollout, c, rng):
            dist = Categorical(logits=logits)
            logp = dist.log_prob(pick(rollout['actions']))
            log_ratio = logp - pick(rollout['logp'])
            ratio = log_ratio.exp()
            with torch.no_grad():
                stats['approx_kl'].append(((ratio - 1) - log_ratio).mean().item())
                stats['clipfrac'].append(((ratio - 1).abs() > c['clip_coef']).float().mean().item())

            adv = pick(rollout['advantages'])
            if c['norm_adv']:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            # Pessimistic (larger) of the two losses: no gain for moving ratio beyond 1 +/- clip.
            policy_loss = torch.max(-adv * ratio,
                                    -adv * ratio.clamp(1 - c['clip_coef'], 1 + c['clip_coef'])).mean()

            target, old_values = pick(rollout['returns']), pick(rollout['values'])
            if c['clip_vloss']:
                clipped = old_values + (value - old_values).clamp(-c['clip_coef'], c['clip_coef'])
                value_loss = 0.5 * torch.max((value - target) ** 2, (clipped - target) ** 2).mean()
            else:
                value_loss = 0.5 * ((value - target) ** 2).mean()

            entropy = dist.entropy().mean()
            loss = policy_loss - c['ent_coef'] * entropy + c['vf_coef'] * value_loss
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), c['max_grad_norm'])
            optimizer.step()
            stats['policy_loss'].append(policy_loss.item())
            stats['value_loss'].append(value_loss.item())
            stats['entropy'].append(entropy.item())
        if c['target_kl'] is not None and np.mean(stats['approx_kl'][-c['num_minibatches']:]) > c['target_kl']:
            break
    result = {k: float(np.mean(v)) for k, v in stats.items()}
    returns, values = rollout['returns'], rollout['values']
    variance = returns.var().item()
    result['explained_variance'] = (float('nan') if variance == 0
                                    else 1 - (returns - values).var().item() / variance)
    return result


def make_envs(c):
    if c['env'] == 'wod':
        levels = [level for _ in range(c['envs_per_level']) for level in c['levels']]
        return VectorEnv(levels, seed=c['seed'], visit_bonus=c['visit_bonus'], bonus_cap=c['bonus_cap'])
    if c['env'] == 'memory':
        return MemoryCue(c['num_envs'], seed=c['seed'], delay=c['delay'])
    return GymVector(c['env'], c['num_envs'], seed=c['seed'])


def check_config(c):
    positive = ('steps', 'num_steps', 'num_minibatches', 'update_epochs', 'save_every', 'log_every')
    if any(c[k] <= 0 for k in positive):
        raise ValueError(f'These settings must be positive: {positive}')
    if c['env'] == 'wod' and (c['envs_per_level'] <= 0 or not c['levels']):
        raise ValueError('Use at least one level and one game per level')
    if not (0 <= c['gamma'] <= 1 and 0 <= c['gae_lambda'] <= 1 and c['clip_coef'] > 0 and c['learning_rate'] > 0):
        raise ValueError('Invalid gamma, gae_lambda, clip_coef or learning_rate')
    if c.get('lstm', 0) < 0:
        raise ValueError('lstm must be 0 (off) or the number of LSTM units')


def progress(window):
    """Mean return and win rate per level over the most recent finished games."""
    out = {}
    for level, games in window.items():
        out[level] = dict(games=len(games), mean_return=round(float(np.mean([g['native_return'] for g in games])), 3))
        if 'success' in games[0]:
            out[level]['success'] = round(float(np.mean([g['success'] for g in games])), 3)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, default=Path('configs/ppo.json'))
    p.add_argument('--out', type=Path, default=Path('runs/ppo'))
    p.add_argument('--steps', type=int, help='Override total game steps')
    p.add_argument('--seed', type=int)
    p.add_argument('--levels', nargs='+', help='Override levels (MiniHack only)')
    p.add_argument('--envs-per-level', type=int)
    p.add_argument('--lstm', type=int, help='LSTM units (0 = no memory)')
    a = p.parse_args()
    c = json.loads(a.config.read_text())
    c.setdefault('lstm', 0)
    for key, value in (('steps', a.steps), ('seed', a.seed), ('levels', a.levels),
                       ('envs_per_level', a.envs_per_level), ('lstm', a.lstm)):
        if value is not None:
            c[key] = value
    try:
        check_config(c)
    except ValueError as error:
        p.error(str(error))

    torch.set_num_threads(1)
    torch.manual_seed(c['seed'])
    rng = np.random.default_rng(c['seed'])
    envs = make_envs(c)
    n, steps = envs.num_envs, c['num_steps']
    batch_size = n * steps
    units = n if c['lstm'] else batch_size  # with memory, minibatches are made of whole games
    if units % c['num_minibatches']:
        envs.close()
        p.error(f"num_minibatches must divide {'the number of games' if c['lstm'] else 'num_envs * num_steps'}")
    a.out.mkdir(parents=True, exist_ok=False)  # never overwrite earlier evidence
    iterations = max(1, c['steps'] // batch_size)
    c.update(num_envs=n, batch_size=batch_size, iterations=iterations)
    (a.out / 'config.json').write_text(json.dumps(c, indent=2) + '\n')

    model = ActorCritic(envs.obs_dim, envs.n_actions, c['hidden'], c['activation'], c['shared'], c['lstm'])
    optimizer = torch.optim.Adam(model.parameters(), lr=c['learning_rate'], eps=1e-5)
    rollout = dict(obs=torch.zeros(steps, n, envs.obs_dim), actions=torch.zeros(steps, n, dtype=torch.long),
                   logp=torch.zeros(steps, n), rewards=torch.zeros(steps, n), values=torch.zeros(steps, n),
                   ends=torch.zeros(steps, n, dtype=torch.bool), first=torch.zeros(steps, n, dtype=torch.bool))

    window = defaultdict(lambda: deque(maxlen=100))
    global_step, status, started = 0, 'failed', time.monotonic()
    try:
        obs = torch.as_tensor(envs.reset())
        state = model.initial_state(n)
        first = torch.ones(n, dtype=torch.bool)
        with (a.out / 'episodes.jsonl').open('w', buffering=1) as episodes, \
                (a.out / 'metrics.jsonl').open('w', buffering=1) as metrics:
            for iteration in range(1, iterations + 1):
                if c['anneal_lr']:
                    optimizer.param_groups[0]['lr'] = c['learning_rate'] * (1 - (iteration - 1) / iterations)
                # The update replays every game of this rollout from this memory.
                rollout['state'] = None if state is None else tuple(s.clone() for s in state)

                # 1. Collect: the current policy plays num_steps in every game.
                for t in range(steps):
                    with torch.no_grad():
                        logits, value, state = model.step(obs, state, first)
                        dist = Categorical(logits=logits)
                        action = dist.sample()
                    rollout['obs'][t], rollout['first'][t] = obs, first
                    rollout['actions'][t], rollout['logp'][t], rollout['values'][t] = action, dist.log_prob(action), value
                    next_obs, rewards, ends, infos = envs.step(action.numpy())
                    global_step += n
                    rollout['rewards'][t] = torch.from_numpy(rewards)
                    rollout['ends'][t] = first = torch.from_numpy(ends)
                    for info in infos:
                        if info is not None:
                            episodes.write(json.dumps(dict(global_step=global_step, **info)) + '\n')
                            window[info['level']].append(info)
                    obs = torch.as_tensor(next_obs)

                # 2. Advantages, bootstrapping from the critic after the last step.
                with torch.no_grad():
                    next_value = model.step(obs, state, first)[1]
                    rollout['advantages'], rollout['returns'] = compute_gae(
                        rollout['rewards'], rollout['values'], rollout['ends'], next_value, c['gamma'], c['gae_lambda'])

                # 3. Learn from this rollout, then it is discarded.
                stats = ppo_update(model, optimizer, rollout, c, rng)

                seconds = time.monotonic() - started
                row = dict(iteration=iteration, global_step=global_step, seconds=round(seconds, 1),
                           sps=round(global_step / seconds), learning_rate=optimizer.param_groups[0]['lr'],
                           rollout_reward=float(rollout['rewards'].sum() / n), **stats)
                metrics.write(json.dumps(row) + '\n')
                if iteration % c['log_every'] == 0 or iteration == iterations:
                    print(json.dumps(dict(step=global_step, sps=row['sps'], entropy=round(stats['entropy'], 3),
                                          approx_kl=round(stats['approx_kl'], 4),
                                          explained_variance=round(stats['explained_variance'], 3),
                                          recent=progress(window))), flush=True)
                if global_step // c['save_every'] != (global_step - batch_size) // c['save_every']:
                    save_model(model, a.out / f'step-{global_step:09d}.npz')
        status = 'finished'
    except KeyboardInterrupt:
        status = 'interrupted'
        print('Interrupted; saving the latest policy.', flush=True)
    finally:
        envs.close()
        save_model(model, a.out / 'latest.npz')
        summary = dict(status=status, steps=global_step, seconds=round(time.monotonic() - started, 1),
                       recent=progress(window))
        (a.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
