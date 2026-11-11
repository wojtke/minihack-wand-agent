"""PPO (clipped objective, GAE) for one policy on all four levels, optionally with
memory (LSTM), a curiosity bonus (RND) and self-imitation of won games (SIL).

Structure follows CleanRL's ppo.py, ppo_atari_lstm.py and ppo_rnd_envpool.py
(MIT licence, https://github.com/vwxyzjn/cleanrl): collect num_steps from every
game, compute GAE advantages backwards, then run a few epochs of minibatch
updates on that rollout and discard it.

Conventions used throughout:
  ends[t]  True when action t ended its game. The observation after it already
           belongs to the next game, so neither the advantage nor the critic's
           estimate of the game's reward crosses that boundary.
  first[t] True when observation t is the first of a game. With memory (LSTM),
           the memory of that game is zeroed before observation t is read.
  values   [..., K]: K = 1 critic head for the game's reward, K = 2 with RND
           (second head for the curiosity bonus).
"""
import argparse
import hashlib
import json
import math
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .rnd import RND
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
    """Policy (logits) and critic (values) from the same observation.

    shared=False: two separate MLPs. shared=True: one trunk, two output heads.
    lstm=H > 0: shared trunk -> LSTM with H units -> both heads, as in CleanRL's
    LSTM agent. The LSTM state (h, c) is the agent's memory between steps.
    value_heads: 1, or 2 when the critic also predicts the curiosity bonus.
    """

    def __init__(self, obs_dim, n_actions, hidden=(128, 128), activation='relu', shared=False, lstm=0,
                 value_heads=1):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f'Unknown activation {activation!r}')
        shared = bool(shared or lstm)
        self.meta = dict(obs_dim=int(obs_dim), n_actions=int(n_actions), hidden=list(hidden),
                         activation=activation, shared=shared, lstm=int(lstm), value_heads=int(value_heads))
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
        self.critic = _layer(features, value_heads, std=1.0)

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

        obs [T, N, D], first [T, N] -> logits [T, N, A], values [T, N, K], memory after step T.
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
                self.critic(critic_features).reshape(steps, n, -1), state)

    def step(self, obs, state=None, first=None):
        """One step of N games: obs [N, D] -> logits [N, A], values [N, K], new memory."""
        logits, values, state = self.sequence(obs[None], state, None if first is None else first[None])
        return logits[0], values[0], state

    def forward(self, obs):
        """Memoryless policy only: obs [B, D] -> logits [B, A], values [B, K]."""
        if self.recurrent:
            raise RuntimeError('A model with memory needs step() or sequence() and a memory state')
        logits, values, _ = self.step(obs)
        return logits, values


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

    pick(x) selects the entries of a [T, N, ...] rollout tensor that belong to
    the minibatch, in the same order as logits and values. Without memory a
    minibatch is any set of transitions. With memory it is a set of whole game
    columns, replayed in order from the memory saved before the rollout.
    """
    obs = rollout['obs']
    steps, n = rollout['actions'].shape
    parts = c['num_minibatches']
    if model.recurrent:
        for columns in np.array_split(rng.permutation(n), min(parts, n)):
            columns = torch.from_numpy(columns)
            state = tuple(s[:, columns] for s in rollout['state'])
            logits, values, _ = model.sequence(obs[:, columns], state, rollout['first'][:, columns])
            yield (logits.reshape(-1, logits.shape[-1]), values.reshape(-1, values.shape[-1]),
                   lambda x, columns=columns: x[:, columns].reshape(-1, *x.shape[2:]))
    else:
        flat_obs = obs.reshape(steps * n, -1)
        for index in np.array_split(rng.permutation(steps * n), parts):
            index = torch.from_numpy(index)
            logits, values = model(flat_obs[index])
            yield logits, values, lambda x, index=index: x.reshape(-1, *x.shape[2:])[index]


def _value_loss(values, old_values, targets, c):
    """Sum over critic heads of 0.5 * mean squared error, optionally clipped as in PPO."""
    if c['clip_vloss']:
        clipped = old_values + (values - old_values).clamp(-c['clip_coef'], c['clip_coef'])
        errors = torch.max((values - targets) ** 2, (clipped - targets) ** 2)
    else:
        errors = (values - targets) ** 2
    return 0.5 * errors.mean(0).sum()


def ppo_update(model, optimizer, rollout, c, rng, rnd=None):
    """Several epochs of clipped-objective minibatch updates on one rollout.

    rollout: dict of [T, N] tensors: obs [T, N, D], actions, logp, advantages,
    returns [T, N, K], values [T, N, K], first; 'state', the memory before
    step 0; and next_obs [T, N, D] when rnd (the curiosity module) is given.
    """
    stats = defaultdict(list)
    for _ in range(c['update_epochs']):
        for logits, values, pick in _minibatches(model, rollout, c, rng):
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
            value_loss = _value_loss(values, pick(rollout['values']), pick(rollout['returns']), c)
            entropy = dist.entropy().mean()
            loss = policy_loss - c['ent_coef'] * entropy + c['vf_coef'] * value_loss
            if rnd is not None:
                rnd_loss = rnd.loss(pick(rollout['next_obs']))
                loss = loss + rnd_loss
                stats['rnd_loss'].append(rnd_loss.item())
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g['params']], c['max_grad_norm'])
            optimizer.step()
            stats['policy_loss'].append(policy_loss.item())
            stats['value_loss'].append(value_loss.item())
            stats['entropy'].append(entropy.item())
        if c['target_kl'] is not None and np.mean(stats['approx_kl'][-c['num_minibatches']:]) > c['target_kl']:
            break
    result = {k: float(np.mean(v)) for k, v in stats.items()}
    # How much of the spread of the game-reward targets the critic explains.
    returns, values = rollout['returns'][..., 0], rollout['values'][..., 0]
    variance = returns.var().item()
    result['explained_variance'] = (float('nan') if variance == 0
                                    else 1 - (returns - values).var().item() / variance)
    return result


class SelfImitation:
    """Self-Imitation Learning (Oh et al., 2018) on won games only.

    Every finished game whose info says success is stored with its discounted
    training returns R(t). After each PPO update the policy is pushed towards
    the stored actions where R(t) exceeds the critic's estimate V(s(t)):
        loss = -log pi(a|s) * max(R - V, 0) + value_coef * 0.5 * max(R - V, 0)^2
    so a rare win is learned from many times instead of once. Each level keeps
    its own buffer of at most `capacity` transitions (oldest games dropped) and
    gets an equal share of every batch, so thousands of short Easy wins cannot
    crowd out a few Medium ones. Without memory (MLP) only.
    """

    def __init__(self, num_envs, gamma, capacity):
        self.gamma, self.capacity = gamma, capacity
        self.current = [[] for _ in range(num_envs)]
        self.levels = {}  # level -> dict(games=deque, size=int, flat=joined arrays or None)
        self.wins = 0

    def record(self, obs, actions, rewards, ends, infos):
        for i in range(len(self.current)):
            self.current[i].append((obs[i].numpy(), int(actions[i]), float(rewards[i])))
            if ends[i]:
                if infos[i] is not None and infos[i].get('success'):
                    self._add(infos[i].get('level'), self.current[i])
                self.current[i] = []

    def _add(self, level, game):
        returns, following = np.zeros(len(game), np.float32), 0.0
        for t in reversed(range(len(game))):
            following = game[t][2] + self.gamma * following
            returns[t] = following
        buffer = self.levels.setdefault(level, dict(games=deque(), size=0, flat=None))
        buffer['games'].append((np.stack([g[0] for g in game]), np.array([g[1] for g in game]), returns))
        buffer['size'] += len(game)
        buffer['flat'] = None
        self.wins += 1
        while buffer['size'] > self.capacity and len(buffer['games']) > 1:
            buffer['size'] -= len(buffer['games'].popleft()[0])

    @property
    def games(self):
        return sum(len(b['games']) for b in self.levels.values())

    def sample(self, batch, rng):
        parts = []
        for buffer, share in zip(self.levels.values(), np.array_split(np.arange(batch), len(self.levels))):
            if buffer['flat'] is None:  # joined again only after this level's buffer changed
                buffer['flat'] = tuple(np.concatenate([g[k] for g in buffer['games']]) for k in range(3))
            index = rng.integers(0, len(buffer['flat'][0]), len(share))
            parts.append([x[index] for x in buffer['flat']])
        return tuple(torch.from_numpy(np.concatenate(x)) for x in zip(*parts))


def sil_update(model, optimizer, sil, c, rng):
    stats = defaultdict(list)
    if not sil.levels:
        return {}
    for _ in range(c['sil_updates']):
        obs, actions, returns = sil.sample(c['sil_batch'], rng)
        logits, values = model(obs)
        gap = (returns - values[:, 0]).clamp_min(0)
        logp = Categorical(logits=logits).log_prob(actions)
        policy_loss = -(logp * gap.detach()).mean()
        value_loss = 0.5 * (gap ** 2).mean()
        loss = c['sil_coef'] * (policy_loss + c['sil_value_coef'] * value_loss)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g['params']], c['max_grad_norm'])
        optimizer.step()
        stats['sil_policy_loss'].append(policy_loss.item())
        stats['sil_positive'].append((gap > 0).float().mean().item())
    result = {k: float(np.mean(v)) for k, v in stats.items()}
    result.update(sil_games=sil.games, sil_wins=sil.wins)
    return result


def make_envs(c):
    if c['env'] == 'wod':
        return VectorEnv(c['games'], seed=c['seed'], visit_bonus=c['visit_bonus'], bonus_cap=c['bonus_cap'],
                         wand_bonus=c['wand_bonus'], death_penalty=c['death_penalty'],
                         align_bonus=c['align_bonus'], gamma=c['gamma'], macro=c['macro'])
    if c['env'] == 'memory':
        return MemoryCue(c['num_envs'], seed=c['seed'], delay=c['delay'])
    return GymVector(c['env'], c['num_envs'], seed=c['seed'])


def check_config(c):
    positive = ('steps', 'num_steps', 'num_minibatches', 'update_epochs', 'save_every', 'log_every')
    if any(c[k] <= 0 for k in positive):
        raise ValueError(f'These settings must be positive: {positive}')
    if c['env'] == 'wod' and not c['games']:
        raise ValueError('Use at least one game')
    if not (0 <= c['gamma'] <= 1 and 0 <= c['gae_lambda'] <= 1 and c['clip_coef'] > 0 and c['learning_rate'] > 0):
        raise ValueError('Invalid gamma, gae_lambda, clip_coef or learning_rate')
    if c['lstm'] < 0:
        raise ValueError('lstm must be 0 (off) or the number of LSTM units')
    if c['rnd'] and not 0 < c['rnd_update_proportion'] <= 1:
        raise ValueError('rnd_update_proportion must be in (0, 1]')
    if c['sil'] and c['lstm']:
        raise ValueError('Self-imitation (sil) is implemented without memory only (lstm 0)')
    if c['macro'] and c['env'] != 'wod':
        raise ValueError('Macro-actions exist only for MiniHack (env wod)')


def progress(window):
    """Mean return and win rate per level over the most recent finished games."""
    out = {}
    for level, games in window.items():
        out[level] = dict(games=len(games), mean_return=round(float(np.mean([g['native_return'] for g in games])), 3))
        if 'success' in games[0]:
            out[level]['success'] = round(float(np.mean([g['success'] for g in games])), 3)
    return out


DEFAULTS = dict(lstm=0, wand_bonus=0.0, death_penalty=0.0, align_bonus=0.0, macro=False,
                rnd=False, rnd_ext_coef=2.0, rnd_int_coef=1.0, rnd_gamma=0.99,
                rnd_update_proportion=0.25, rnd_hidden=256, rnd_output=64,
                sil=False, sil_coef=1.0, sil_value_coef=0.01, sil_updates=4, sil_batch=512, sil_capacity=50000)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, default=Path('configs/ppo.json'))
    p.add_argument('--out', type=Path, default=Path('runs/ppo'))
    p.add_argument('--steps', type=int, help='Override total game steps')
    p.add_argument('--seed', type=int)
    p.add_argument('--games', nargs='+', help='MiniHack: one level name per parallel game, e.g. Easy Medium Medium')
    p.add_argument('--levels', nargs='+', help='MiniHack: shorthand with --envs-per-level for equal shares')
    p.add_argument('--envs-per-level', type=int, default=1)
    p.add_argument('--lstm', type=int, help='LSTM units (0 = no memory)')
    p.add_argument('--rnd', action=argparse.BooleanOptionalAction, help='Curiosity bonus (RND) on or off')
    p.add_argument('--wand-bonus', type=float, help='Training reward for picking up a wand (shaping)')
    p.add_argument('--death-penalty', type=float, help='Training penalty for dying')
    p.add_argument('--align-bonus', type=float, help='Potential for holding a wand in line with a monster (shaping)')
    p.add_argument('--macro', action=argparse.BooleanOptionalAction, help='Zap macro-actions instead of z and f')
    p.add_argument('--sil', action=argparse.BooleanOptionalAction, help='Self-imitation of won games on or off')
    p.add_argument('--ent-coef', type=float, help='Entropy bonus coefficient')
    p.add_argument('--init-from', type=Path, help='Start from the weights of a PPO checkpoint with the same network '
                   '(curriculum); the optimizer and the RND networks start fresh')
    a = p.parse_args()
    c = {**DEFAULTS, **json.loads(a.config.read_text())}
    if a.levels:
        c['games'] = [level for _ in range(a.envs_per_level) for level in a.levels]
    for key, value in (('steps', a.steps), ('seed', a.seed), ('games', a.games), ('lstm', a.lstm), ('rnd', a.rnd),
                       ('wand_bonus', a.wand_bonus), ('death_penalty', a.death_penalty),
                       ('align_bonus', a.align_bonus), ('macro', a.macro), ('sil', a.sil), ('ent_coef', a.ent_coef)):
        if value is not None:
            c[key] = value
    if a.init_from:
        c['init_from'] = dict(path=str(a.init_from), sha256=hashlib.sha256(a.init_from.read_bytes()).hexdigest())
    try:
        check_config(c)
    except ValueError as error:
        p.error(str(error))
    a.out.mkdir(parents=True, exist_ok=False)  # never overwrite earlier evidence

    torch.set_num_threads(1)
    torch.manual_seed(c['seed'])
    rng = np.random.default_rng(c['seed'])
    envs = make_envs(c)
    n, steps = envs.num_envs, c['num_steps']
    batch_size = n * steps
    iterations = max(1, c['steps'] // batch_size)
    c.update(num_envs=n, batch_size=batch_size, iterations=iterations)
    (a.out / 'config.json').write_text(json.dumps(c, indent=2) + '\n')

    heads = 2 if c['rnd'] else 1
    model = ActorCritic(envs.obs_dim, envs.n_actions, c['hidden'], c['activation'], c['shared'], c['lstm'], heads)
    if a.init_from:
        start = load_model(a.init_from)
        if start.meta != model.meta:
            p.error(f'--init-from network {start.meta} differs from this run {model.meta}')
        model.load_state_dict(start.state_dict())
        model.train()
    parameters = list(model.parameters())
    rnd = None
    if c['rnd']:
        rnd = RND(envs.obs_dim, c['rnd_hidden'], c['rnd_output'], c['rnd_gamma'], c['rnd_update_proportion'])
        parameters += list(rnd.predictor.parameters())
    optimizer = torch.optim.Adam(parameters, lr=c['learning_rate'], eps=1e-5)
    sil = SelfImitation(n, c['gamma'], c['sil_capacity']) if c['sil'] else None
    rollout = dict(obs=torch.zeros(steps, n, envs.obs_dim), next_obs=torch.zeros(steps, n, envs.obs_dim),
                   actions=torch.zeros(steps, n, dtype=torch.long), logp=torch.zeros(steps, n),
                   rewards=torch.zeros(steps, n), values=torch.zeros(steps, n, heads),
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
                        logits, values, state = model.step(obs, state, first)
                        dist = Categorical(logits=logits)
                        action = dist.sample()
                    rollout['obs'][t], rollout['first'][t] = obs, first
                    rollout['actions'][t], rollout['logp'][t], rollout['values'][t] = action, dist.log_prob(action), values
                    next_obs, rewards, ends, infos = envs.step(action.numpy())
                    global_step += n
                    rollout['rewards'][t] = torch.from_numpy(rewards)
                    rollout['ends'][t] = first = torch.from_numpy(ends)
                    if sil is not None:
                        sil.record(obs, action, rewards, ends, infos)
                    for info in infos:
                        if info is not None:
                            episodes.write(json.dumps(dict(global_step=global_step, **info)) + '\n')
                            window[info['level']].append(info)
                    obs = rollout['next_obs'][t] = torch.as_tensor(next_obs)

                # 2. Advantages, bootstrapping from the critic after the last step.
                extra = {}
                with torch.no_grad():
                    next_values = model.step(obs, state, first)[1]
                    advantages, returns = compute_gae(rollout['rewards'], rollout['values'][..., 0], rollout['ends'],
                                                      next_values[:, 0], c['gamma'], c['gae_lambda'])
                    if rnd is not None:
                        # The bonus does not stop at game ends: novelty carries over to the next game.
                        rnd.update_obs_stats(rollout['next_obs'])
                        bonus, raw = rnd.intrinsic_reward(rollout['next_obs'])
                        int_advantages, int_returns = compute_gae(
                            bonus, rollout['values'][..., 1], torch.zeros_like(rollout['ends']),
                            next_values[:, 1], c['rnd_gamma'], c['gae_lambda'])
                        advantages = c['rnd_ext_coef'] * advantages + c['rnd_int_coef'] * int_advantages
                        returns = torch.stack([returns, int_returns], -1)
                        extra = dict(bonus_mean=float(bonus.mean()), bonus_raw_mean=float(raw.mean()),
                                     bonus_by_game=[round(float(x), 4) for x in bonus.mean(0)])
                    else:
                        returns = returns[..., None]
                rollout['advantages'], rollout['returns'] = advantages, returns

                # 3. Learn from this rollout, then it is discarded.
                stats = ppo_update(model, optimizer, rollout, c, rng, rnd)
                if sil is not None:
                    stats.update(sil_update(model, optimizer, sil, c, rng))

                seconds = time.monotonic() - started
                row = dict(iteration=iteration, global_step=global_step, seconds=round(seconds, 1),
                           sps=round(global_step / seconds), learning_rate=optimizer.param_groups[0]['lr'],
                           rollout_reward=float(rollout['rewards'].sum() / n), **stats, **extra)
                metrics.write(json.dumps(row) + '\n')
                if iteration % c['log_every'] == 0 or iteration == iterations:
                    shown = dict(step=global_step, sps=row['sps'], entropy=round(stats['entropy'], 3),
                                 approx_kl=round(stats['approx_kl'], 4),
                                 explained_variance=round(stats['explained_variance'], 3))
                    if rnd is not None:
                        shown['bonus_raw'] = round(extra['bonus_raw_mean'], 4)
                    print(json.dumps(dict(**shown, recent=progress(window))), flush=True)
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
