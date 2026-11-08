"""PPO (clipped objective, GAE) for one policy on all four levels.

Structure follows CleanRL's ppo.py (MIT licence, https://github.com/vwxyzjn/cleanrl):
collect num_steps from every game, compute GAE advantages backwards, then run
a few epochs of minibatch updates on that rollout and discard it.

Convention used throughout: ends[t] is True when action t ended its game. The
observation after it already belongs to the next game, so neither the advantage
nor the critic's estimate crosses that boundary.
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
    """Policy (11 logits) and critic (1 value) from the same observation.

    shared=False: two separate MLPs. shared=True: one trunk, two output heads.
    """

    def __init__(self, obs_dim, n_actions, hidden=(128, 128), activation='relu', shared=False):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f'Unknown activation {activation!r}')
        self.meta = dict(obs_dim=int(obs_dim), n_actions=int(n_actions), hidden=list(hidden),
                         activation=activation, shared=bool(shared))
        sizes = [obs_dim, *hidden]
        if shared:
            self.body = _body(sizes, activation)
        else:
            self.actor_body = _body(sizes, activation)
            self.critic_body = _body(sizes, activation)
        # Small actor weights: every action starts almost equally likely.
        self.actor = _layer(sizes[-1], n_actions, std=0.01)
        self.critic = _layer(sizes[-1], 1, std=1.0)

    def forward(self, obs):
        """Return (logits[B, n_actions], value[B])."""
        if self.meta['shared']:
            actor_features = critic_features = self.body(obs)
        else:
            actor_features, critic_features = self.actor_body(obs), self.critic_body(obs)
        return self.actor(actor_features), self.critic(critic_features).squeeze(-1)

    def value(self, obs):
        return self(obs)[1]


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


def ppo_update(model, optimizer, batch, c, rng):
    """Several epochs of clipped-objective minibatch updates on one rollout."""
    obs, actions, old_logp, advantages, returns, old_values = batch
    size = len(actions)
    minibatch = size // c['num_minibatches']
    stats = defaultdict(list)
    for _ in range(c['update_epochs']):
        order = rng.permutation(size)
        for start in range(0, size, minibatch):
            index = torch.from_numpy(order[start:start + minibatch])
            logits, value = model(obs[index])
            dist = Categorical(logits=logits)
            logp = dist.log_prob(actions[index])
            log_ratio = logp - old_logp[index]
            ratio = log_ratio.exp()
            with torch.no_grad():
                stats['approx_kl'].append(((ratio - 1) - log_ratio).mean().item())
                stats['clipfrac'].append(((ratio - 1).abs() > c['clip_coef']).float().mean().item())

            adv = advantages[index]
            if c['norm_adv']:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            # Pessimistic (larger) of the two losses: no gain for moving ratio beyond 1 +/- clip.
            policy_loss = torch.max(-adv * ratio,
                                    -adv * ratio.clamp(1 - c['clip_coef'], 1 + c['clip_coef'])).mean()

            target = returns[index]
            if c['clip_vloss']:
                clipped = old_values[index] + (value - old_values[index]).clamp(-c['clip_coef'], c['clip_coef'])
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
    variance = returns.var().item()
    result['explained_variance'] = (float('nan') if variance == 0
                                    else 1 - (returns - old_values).var().item() / variance)
    return result


def make_envs(c):
    if c['env'] == 'wod':
        levels = [level for _ in range(c['envs_per_level']) for level in c['levels']]
        return VectorEnv(levels, seed=c['seed'], visit_bonus=c['visit_bonus'], bonus_cap=c['bonus_cap'])
    return GymVector(c['env'], c['num_envs'], seed=c['seed'])


def check_config(c):
    positive = ('steps', 'num_steps', 'num_minibatches', 'update_epochs', 'save_every', 'log_every')
    if any(c[k] <= 0 for k in positive):
        raise ValueError(f'These settings must be positive: {positive}')
    if c['env'] == 'wod' and (c['envs_per_level'] <= 0 or not c['levels']):
        raise ValueError('Use at least one level and one game per level')
    if not (0 <= c['gamma'] <= 1 and 0 <= c['gae_lambda'] <= 1 and c['clip_coef'] > 0 and c['learning_rate'] > 0):
        raise ValueError('Invalid gamma, gae_lambda, clip_coef or learning_rate')


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
    a = p.parse_args()
    c = json.loads(a.config.read_text())
    for key, value in (('steps', a.steps), ('seed', a.seed), ('levels', a.levels), ('envs_per_level', a.envs_per_level)):
        if value is not None:
            c[key] = value
    try:
        check_config(c)
    except ValueError as error:
        p.error(str(error))
    a.out.mkdir(parents=True, exist_ok=False)

    torch.set_num_threads(1)
    torch.manual_seed(c['seed'])
    rng = np.random.default_rng(c['seed'])
    envs = make_envs(c)
    n, steps = envs.num_envs, c['num_steps']
    batch_size = n * steps
    if batch_size % c['num_minibatches']:
        p.error('num_envs * num_steps must be divisible by num_minibatches')
    iterations = max(1, c['steps'] // batch_size)
    c.update(num_envs=n, batch_size=batch_size, iterations=iterations)
    (a.out / 'config.json').write_text(json.dumps(c, indent=2) + '\n')

    model = ActorCritic(envs.obs_dim, envs.n_actions, c['hidden'], c['activation'], c['shared'])
    optimizer = torch.optim.Adam(model.parameters(), lr=c['learning_rate'], eps=1e-5)
    buf_obs = torch.zeros(steps, n, envs.obs_dim)
    buf_actions = torch.zeros(steps, n, dtype=torch.long)
    buf_logp, buf_rewards, buf_values = (torch.zeros(steps, n) for _ in range(3))
    buf_ends = torch.zeros(steps, n, dtype=torch.bool)

    window = defaultdict(lambda: deque(maxlen=100))
    global_step, status, started = 0, 'failed', time.monotonic()
    try:
        obs = torch.as_tensor(envs.reset())
        with (a.out / 'episodes.jsonl').open('w', buffering=1) as episodes, \
                (a.out / 'metrics.jsonl').open('w', buffering=1) as metrics:
            for iteration in range(1, iterations + 1):
                if c['anneal_lr']:
                    optimizer.param_groups[0]['lr'] = c['learning_rate'] * (1 - (iteration - 1) / iterations)

                # 1. Collect: the current policy plays num_steps in every game.
                for t in range(steps):
                    with torch.no_grad():
                        logits, value = model(obs)
                        dist = Categorical(logits=logits)
                        action = dist.sample()
                    buf_obs[t], buf_actions[t], buf_logp[t], buf_values[t] = obs, action, dist.log_prob(action), value
                    next_obs, rewards, ends, infos = envs.step(action.numpy())
                    global_step += n
                    buf_rewards[t] = torch.from_numpy(rewards)
                    buf_ends[t] = torch.from_numpy(ends)
                    for info in infos:
                        if info is not None:
                            episodes.write(json.dumps(dict(global_step=global_step, **info)) + '\n')
                            window[info['level']].append(info)
                    obs = torch.as_tensor(next_obs)

                # 2. Advantages, bootstrapping from the critic after the last step.
                with torch.no_grad():
                    advantages, returns = compute_gae(buf_rewards, buf_values, buf_ends, model.value(obs),
                                                      c['gamma'], c['gae_lambda'])

                # 3. Learn from this rollout, then it is discarded.
                batch = (buf_obs.reshape(-1, envs.obs_dim), buf_actions.reshape(-1), buf_logp.reshape(-1),
                         advantages.reshape(-1), returns.reshape(-1), buf_values.reshape(-1))
                stats = ppo_update(model, optimizer, batch, c, rng)

                seconds = time.monotonic() - started
                row = dict(iteration=iteration, global_step=global_step, seconds=round(seconds, 1),
                           sps=round(global_step / seconds), learning_rate=optimizer.param_groups[0]['lr'],
                           rollout_reward=float(buf_rewards.sum() / n), **stats)
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
