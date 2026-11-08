"""A task with a known answer for checking memory: remember a cue.

The first observation of a game shows a cue (left or right). The next `delay`
observations are blank. Then the last observation says "answer now", and the
agent wins (+1) only if its action matches the cue; any other step gives 0.
A policy without memory sees the same blank or "answer" observation whatever
the cue was, so it can do no better than guessing: 50% wins. A working LSTM
policy should approach 100%.

Same interface as VectorEnv: reset() -> obs[N, 3]; step(actions) -> obs,
rewards, ends, infos, with automatic reset of finished games.
"""
import numpy as np


class MemoryCue:
    obs_dim = 3
    n_actions = 2

    def __init__(self, num_envs, seed=0, delay=5):
        if delay < 1:
            raise ValueError('Use a delay of at least one blank step')
        self.num_envs, self.delay = num_envs, delay
        self.rng = np.random.default_rng(seed)
        self.cue = np.zeros(num_envs, int)
        self.t = np.zeros(num_envs, int)

    def _observe(self, i):
        obs = np.zeros(3, np.float32)
        if self.t[i] == 0:
            obs[self.cue[i]] = 1.0
        if self.t[i] == self.delay:
            obs[2] = 1.0
        return obs

    def _reset(self, i):
        self.cue[i], self.t[i] = self.rng.integers(2), 0
        return self._observe(i)

    def reset(self):
        return np.stack([self._reset(i) for i in range(self.num_envs)])

    def step(self, actions):
        rows, rewards, ends, infos = [], [], [], []
        for i, action in enumerate(actions):
            if self.t[i] == self.delay:
                won = bool(int(action) == self.cue[i])
                rewards.append(float(won))
                ends.append(True)
                infos.append(dict(level='memory', native_return=float(won), steps=self.delay + 1, success=won))
                rows.append(self._reset(i))
            else:
                self.t[i] += 1
                rewards.append(0.0)
                ends.append(False)
                infos.append(None)
                rows.append(self._observe(i))
        return np.stack(rows), np.asarray(rewards, np.float32), np.asarray(ends, bool), infos

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
