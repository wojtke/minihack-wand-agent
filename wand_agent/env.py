"""Shared observation interface; optional map-independent exploration reward."""
import numpy as np
from .features import DIM, PublicMemory
from .native import HORIZONS, KEYS, make_env, text
from .terminal import FinalObservation


class AgentEnv:
    def __init__(self, level, visit_bonus=0.0, bonus_cap=2.0):
        if visit_bonus < 0 or bonus_cap < 0:
            raise ValueError('Bonuses must be nonnegative')
        self.level, self.visit_bonus, self.bonus_cap = level, visit_bonus, bonus_cap
        self.native = FinalObservation(make_env(level))
        assert ''.join(chr(int(a)) for a in self.native.unwrapped.actions) == KEYS
        assert self.native.unwrapped._max_episode_steps == HORIZONS[level]
        self.active = False

    def reset(self, seed):
        self.raw, _ = self.native.reset(seed=seed)
        self.memory = PublicMemory()
        self.elapsed = 0
        self.total = self.bonus_total = 0.0
        self.visited = {tuple(map(int, self.raw['blstats'][:2]))}
        self.active = True
        return self.memory.encode(self.raw)

    def step(self, action):
        if not self.active:
            raise RuntimeError('Reset before stepping a completed game')
        key = KEYS[int(action)]
        message = text(self.raw['message'])
        if 'What do you want to' not in message and 'In what direction?' not in message:
            self.memory.command = key if key in 'zf' else None
        self.memory.previous_mode = self.memory.mode(self.raw)
        self.raw, reward, term, trunc, info = self.native.step(int(action))
        self.elapsed += 1
        self.memory.previous_action = int(action)
        done = bool(term or trunc)
        bonus = 0.0
        if not done:
            point = tuple(map(int, self.raw['blstats'][:2]))
            if point not in self.visited:
                bonus = min(self.visit_bonus, max(0., self.bonus_cap-self.bonus_total))
                self.visited.add(point)
        self.total += float(reward)
        self.bonus_total += bonus
        self.active = not done
        # Native action limits define a finite task. Ended episodes never bootstrap.
        obs = np.zeros(DIM, np.float32) if done else self.memory.encode(self.raw, self.elapsed)
        metrics = dict(level=self.level, steps=self.elapsed, native_reward=float(reward),
                       native_return=self.total, bonus=self.bonus_total,
                       success=info['end_status'].name == 'TASK_SUCCESSFUL',
                       end_status=info['end_status'].name)
        return obs, float(reward)+bonus, done, metrics

    def close(self):
        self.native.close()
