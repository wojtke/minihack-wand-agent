"""CPU Double DQN: one network, uniform replay, equal steps on four levels."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from .env import AgentEnv
from .features import DIM
from .native import LEVELS, KEYS
from .network import QNetwork, save


class Replay:
    def __init__(self, size):
        self.obs = np.empty((size, DIM), np.float32)
        self.next_obs = np.empty_like(self.obs)
        self.actions = np.empty(size, np.int64)
        self.rewards = np.empty(size, np.float32)
        self.done = np.empty(size, np.float32)
        self.index = self.count = 0

    def add(self, obs, action, reward, next_obs, done):
        i = self.index
        self.obs[i], self.actions[i], self.rewards[i] = obs, action, reward
        self.next_obs[i], self.done[i] = next_obs, done
        self.index = (i+1) % len(self.actions)
        self.count = min(self.count+1, len(self.actions))

    def sample(self, rng, batch_size):
        i = rng.integers(self.count, size=batch_size)
        return tuple(torch.from_numpy(x[i]) for x in
                     (self.obs, self.actions, self.rewards, self.next_obs, self.done))


def double_dqn_target(online, target, next_obs, rewards, done, gamma):
    with torch.no_grad():
        actions = online(next_obs).argmax(dim=1, keepdim=True)
        values = target(next_obs).gather(1, actions).squeeze(1)
        return rewards + gamma*(1-done)*values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=Path('configs/dqn.json'))
    p.add_argument('--out', type=Path, default=Path('runs/dqn'))
    p.add_argument('--steps', type=int, help='Override config for a short smoke run')
    p.add_argument('--seed', type=int)
    p.add_argument('--no-bonus', action='store_true', help='Train on native reward only')
    a = p.parse_args()
    c = json.loads(a.config.read_text())
    if a.steps is not None: c['steps'] = a.steps
    if a.seed is not None: c['seed'] = a.seed
    if a.no_bonus: c['visit_bonus'] = 0.0
    if any(c[k] <= 0 for k in ('steps', 'batch_size', 'buffer_size', 'train_every', 'target_every', 'save_every')):
        p.error('Step counts, replay size and batch size must be positive')
    if not 0 <= c['epsilon_end'] <= c['epsilon_start'] <= 1 or not 0 < c['epsilon_fraction'] <= 1:
        p.error('Invalid epsilon schedule')
    if not 0 <= c['gamma'] <= 1 or c['learning_starts'] < 0 or c['learning_rate'] <= 0:
        p.error('Invalid learning settings')
    # Refuse accidental overwrites of previous evidence.
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out/'config.json').write_text(json.dumps(c, indent=2)+'\n')
    torch.set_num_threads(1)
    torch.manual_seed(c['seed'])
    rng = np.random.default_rng(c['seed'])
    seed_rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(c['seed']).spawn(4)]
    def episode_seed(i): return int(seed_rngs[i].integers(0, 100_000_000))
    q, target = QNetwork(), QNetwork()
    target.load_state_dict(q.state_dict())
    optimizer = torch.optim.Adam(q.parameters(), lr=c['learning_rate'])
    replay = Replay(c['buffer_size'])
    envs, observations, seeds = [], [], []
    step = updates = 0
    loss_value = None
    started = time.monotonic()
    interrupted = False
    status = "failed"
    try:
        for i, level in enumerate(LEVELS):
            env = AgentEnv(level, c['visit_bonus'], c['bonus_cap'])
            envs.append(env)
            seeds.append(episode_seed(i))
            observations.append(env.reset(seeds[-1]))
        with (a.out/'episodes.jsonl').open('w', buffering=1) as episodes:
            for step in range(1, c['steps']+1):
                i = (step-1) % len(envs)
                epsilon = c['epsilon_start'] + (c['epsilon_end']-c['epsilon_start'])*min(
                    1., (step-1)/max(1, c['steps']*c['epsilon_fraction']))
                action = int(rng.integers(len(KEYS))) if rng.random() < epsilon else q.action(observations[i])
                next_obs, reward, done, metrics = envs[i].step(action)
                replay.add(observations[i], action, reward, next_obs, done)
                observations[i] = next_obs
                if done:
                    episodes.write(json.dumps(dict(global_step=step, seed=seeds[i], **metrics))+'\n')
                    seeds[i] = episode_seed(i)
                    observations[i] = envs[i].reset(seeds[i])
                if step >= c['learning_starts'] and step % c['train_every'] == 0 and replay.count >= c['batch_size']:
                    obs, actions, rewards, following, ended = replay.sample(rng, c['batch_size'])
                    expected = double_dqn_target(q, target, following, rewards, ended, c['gamma'])
                    values = q(obs).gather(1, actions[:, None]).squeeze(1)
                    loss = F.smooth_l1_loss(values, expected)
                    if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(q.parameters(), 10.)
                    optimizer.step()
                    updates += 1
                    loss_value = float(loss.item())
                if step % c['target_every'] == 0: target.load_state_dict(q.state_dict())
                if step % c['save_every'] == 0:
                    save(q, a.out/f'step-{step:09d}.npz')
                if step % 5000 == 0:
                    print(json.dumps(dict(step=step, updates=updates, epsilon=epsilon,
                                          loss=loss_value, seconds=round(time.monotonic()-started, 1))), flush=True)
        status = "finished"
    except KeyboardInterrupt:
        status = "interrupted"
        interrupted = True
        print('Interrupted; saving the latest policy.', flush=True)
    finally:
        for env in envs: env.close()
        save(q, a.out/'latest.npz')
        summary = dict(status=status, steps=step, updates=updates, interrupted=interrupted, last_loss=loss_value,
                       seconds=round(time.monotonic()-started, 2))
        (a.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        print(json.dumps(summary), flush=True)


if __name__ == '__main__': main()
