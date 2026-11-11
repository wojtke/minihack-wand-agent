"""Several games stepped together, split across worker processes.

On-policy methods such as PPO collect a short rollout from many games at once:
the network chooses actions for all games in one batch, then every game takes
one step in parallel. A game that ends is reset immediately inside its worker,
so the observation returned for it is already the first one of the next game,
and `done` marks the boundary. The finished game's metrics come back in `infos`.

Each game draws its episode seeds from its own generator, derived from one
master seed, so a run is fully reproducible. Training seeds lie in
[0, TRAIN_SEED_LIMIT); evaluate.py uses seeds from 100000000 upwards.

By default every game slot always plays its own level. After
set_level_weights(p), each slot draws the level of its next game from p
instead (adaptive level mix); the first game of each slot keeps its level.
"""
import multiprocessing as mp
import signal

import numpy as np

from .env import MACRO_NAMES, AgentEnv
from .features import DIM, ON_ITEM_DIM
from .native import KEYS, LEVELS

TRAIN_SEED_LIMIT = 100_000_000


class Game:
    """One game slot with its own seed generator and automatic reset.

    It keeps one AgentEnv per level it has played, so it can switch levels
    between games when level weights are set.
    """

    def __init__(self, level, seed_sequence, options=None):
        self.options = options or {}
        self.envs = {}
        self.level = level
        self.env = self._env(level)
        self.rng = np.random.default_rng(seed_sequence)
        # Separate stream: without level weights the episode seeds stay the same as before.
        self.level_rng = np.random.default_rng(seed_sequence.spawn(1)[0])
        self.weights = None
        self.seed = None

    def _env(self, level):
        if level not in self.envs:
            self.envs[level] = AgentEnv(level, **self.options)
        return self.envs[level]

    def set_weights(self, weights):
        self.weights = weights

    def reset(self):
        if self.weights is not None:
            self.level = LEVELS[self.level_rng.choice(len(LEVELS), p=self.weights)]
            self.env = self._env(self.level)
        self.seed = int(self.rng.integers(0, TRAIN_SEED_LIMIT))
        return self.env.reset(self.seed)

    def step(self, action):
        obs, reward, done, metrics = self.env.step(action)
        info = None
        if done:
            info = dict(seed=self.seed, **metrics)
            obs = self.reset()
        return obs, reward, done, info

    def close(self):
        for env in self.envs.values():
            env.close()


def _worker(pipe, args):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    games = [Game(*a) for a in args]
    try:
        while True:
            command, data = pipe.recv()
            if command == 'step':
                pipe.send([game.step(action) for game, action in zip(games, data)])
            elif command == 'reset':
                pipe.send([game.reset() for game in games])
            elif command == 'weights':
                for game in games:
                    game.set_weights(data)
            elif command == 'close':
                break
    finally:
        for game in games:
            game.close()
        pipe.close()


class VectorEnv:
    """Batched interface: reset() -> obs[N, DIM]; step(actions[N]) -> obs, rewards, dones, infos.

    levels gives each game slot its level (and the number of slots); infos
    report the level each finished game was actually played on.

    Games are split evenly across `workers` processes (default: one per game, at
    most 8). Each message carries the whole group, so several games per process
    cost fewer round trips. processes=False runs all games in this process, one
    after another; results are identical, which is useful for tests and debugging.
    """

    def __init__(self, levels=LEVELS * 2, seed=0, visit_bonus=0.0, bonus_cap=2.0,
                 processes=True, workers=None, **options):
        """options: further AgentEnv settings (wand_bonus, death_penalty, align_bonus, gamma, macro, on_item)."""
        self.levels = tuple(levels)
        unknown = set(self.levels) - set(LEVELS)
        if unknown:
            raise ValueError(f'Unknown levels: {sorted(unknown)}')
        seeds = np.random.SeedSequence(seed).spawn(len(self.levels))
        options = dict(visit_bonus=visit_bonus, bonus_cap=bonus_cap, **options)
        args = [(level, s, options) for level, s in zip(self.levels, seeds)]
        self.n_actions = len(MACRO_NAMES) if options.get('macro') else len(KEYS)
        self.obs_dim = ON_ITEM_DIM if options.get('on_item') else DIM
        self.processes = processes
        self.closed = False
        if processes:
            workers = min(len(args), workers or 8)
            if workers <= 0:
                raise ValueError('Use at least one worker')
            # Contiguous groups keep the order of games equal to the order of levels.
            bounds = np.linspace(0, len(args), workers + 1).round().astype(int)
            self.groups = [args[i:j] for i, j in zip(bounds[:-1], bounds[1:])]
            ctx = mp.get_context('spawn')
            self.pipes, self.workers = [], []
            for group in self.groups:
                parent, child = ctx.Pipe()
                worker = ctx.Process(target=_worker, args=(child, group), daemon=True)
                worker.start()
                child.close()
                self.pipes.append(parent)
                self.workers.append(worker)
        else:
            self.games = [Game(*a) for a in args]


    @property
    def num_envs(self):
        return len(self.levels)

    def set_level_weights(self, weights):
        """From now on, every game slot draws the level of its next game from weights (one per LEVELS)."""
        weights = np.asarray(weights, np.float64)
        if weights.shape != (len(LEVELS),) or (weights < 0).any() or not np.isclose(weights.sum(), 1):
            raise ValueError(f'Expected {len(LEVELS)} nonnegative weights that sum to 1, got {weights}')
        weights = weights / weights.sum()
        if self.processes:
            for pipe in self.pipes:
                pipe.send(('weights', weights))
        else:
            for game in self.games:
                game.set_weights(weights)

    def reset(self):
        if self.processes:
            for pipe in self.pipes:
                pipe.send(('reset', None))
            rows = [row for pipe in self.pipes for row in pipe.recv()]
        else:
            rows = [game.reset() for game in self.games]
        return np.stack(rows)

    def step(self, actions):
        actions = [int(a) for a in actions]
        if len(actions) != self.num_envs:
            raise ValueError(f'Expected {self.num_envs} actions, got {len(actions)}')
        if self.processes:
            start = 0
            for pipe, group in zip(self.pipes, self.groups):
                pipe.send(('step', actions[start:start + len(group)]))
                start += len(group)
            results = [result for pipe in self.pipes for result in pipe.recv()]
        else:
            results = [game.step(action) for game, action in zip(self.games, actions)]
        obs, rewards, dones, infos = zip(*results)
        return (np.stack(obs), np.asarray(rewards, np.float32),
                np.asarray(dones, bool), list(infos))

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.processes:
            for pipe in self.pipes:
                try:
                    pipe.send(('close', None))
                except (BrokenPipeError, EOFError):
                    pass
            for worker in self.workers:
                worker.join(timeout=10)
                if worker.is_alive():
                    worker.terminate()
        else:
            for game in self.games:
                game.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class GymVector:
    """The VectorEnv interface for any Gymnasium task, in this process.

    Used to check the learning code on a task with a known solution (CartPole).
    Truncation counts as the end of a game, as in our MiniHack tasks.
    """

    def __init__(self, env_id, num_envs, seed=0):
        import gymnasium as gym
        self.env_id = env_id
        self.envs = [gym.make(env_id) for _ in range(num_envs)]
        self.rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(num_envs)]
        self.obs_dim = int(np.prod(self.envs[0].observation_space.shape))
        self.n_actions = int(self.envs[0].action_space.n)
        self.num_envs = num_envs
        self.seeds = [None] * num_envs
        self.totals = [0.0] * num_envs
        self.lengths = [0] * num_envs

    def _reset(self, i):
        self.seeds[i] = int(self.rngs[i].integers(0, TRAIN_SEED_LIMIT))
        self.totals[i], self.lengths[i] = 0.0, 0
        obs, _ = self.envs[i].reset(seed=self.seeds[i])
        return np.asarray(obs, np.float32).ravel()

    def reset(self):
        return np.stack([self._reset(i) for i in range(self.num_envs)])

    def step(self, actions):
        rows, rewards, dones, infos = [], [], [], []
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, term, trunc, _ = env.step(int(action))
            self.totals[i] += float(reward)
            self.lengths[i] += 1
            done = bool(term or trunc)
            info = None
            if done:
                info = dict(level=self.env_id, seed=self.seeds[i], native_return=self.totals[i],
                            steps=self.lengths[i])
                obs = self._reset(i)
            rows.append(np.asarray(obs, np.float32).ravel())
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        return np.stack(rows), np.asarray(rewards, np.float32), np.asarray(dones, bool), infos

    def close(self):
        for env in self.envs:
            env.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
