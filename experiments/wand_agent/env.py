"""Shared observation interface; optional training-only rewards and zap macro-actions.

All extra rewards default to 0 and never enter native_return or evaluation:
- visit_bonus: +visit_bonus per newly visited position, capped at bonus_cap per game.
- wand_bonus: shaping from the potential "holds a wand" (Ng et al., 1999):
  reward wand_bonus * (has_wand(next) - has_wand(now)), i.e. +wand_bonus when a
  wand enters the inventory and -wand_bonus if it leaves. On Easy the wand is
  there from the start, so it adds nothing; per game it adds at most +wand_bonus.
- death_penalty: -death_penalty when the game ends in death. Natively death
  gives 0 while every wasted action costs -0.01, so dying early beats surviving
  to the action limit; this removes that incentive.
- align_bonus: potential-based shaping F = gamma * Phi(next) - Phi(now) with
  Phi = align_bonus if the player holds a wand and a visible monster is on the
  same row, column or diagonal (a straight line a wand ray can travel), else 0.
  Phi is 0 after the game ends, so the shaping sums to at most -Phi(start) over
  a game and cannot be farmed by stepping in and out of line.

macro=True replaces the native inputs z and f by eight "zap the wand in
direction d" actions (17 in all: kljhunby, pick up, zap kljhunby). Each one
plays the native inputs z, f, d inside this wrapper and stops early if a prompt
does not appear or the game ends. Every native input still counts against the
action limit and its time penalty, so evaluation is on the same native game.
"""
import numpy as np
from .features import DIM, MONSTER, PublicMemory
from .native import HORIZONS, KEYS, make_env, text
from .terminal import FinalObservation

DIRECTIONS = 'kljhunby'
MACRO_NAMES = tuple(DIRECTIONS + ',') + tuple('z' + d for d in DIRECTIONS)


class AgentEnv:
    def __init__(self, level, visit_bonus=0.0, bonus_cap=2.0, wand_bonus=0.0, death_penalty=0.0,
                 align_bonus=0.0, gamma=0.99, macro=False):
        if min(visit_bonus, bonus_cap, wand_bonus, death_penalty, align_bonus) < 0:
            raise ValueError('Bonuses and penalties must be nonnegative')
        if not 0 <= gamma <= 1:
            raise ValueError('gamma must be in [0, 1]')
        self.level, self.visit_bonus, self.bonus_cap = level, visit_bonus, bonus_cap
        self.wand_bonus, self.death_penalty = wand_bonus, death_penalty
        self.align_bonus, self.gamma, self.macro = align_bonus, gamma, macro
        self.native = FinalObservation(make_env(level))
        assert ''.join(chr(int(a)) for a in self.native.unwrapped.actions) == KEYS
        assert self.native.unwrapped._max_episode_steps == HORIZONS[level]
        self.active = False

    @property
    def n_actions(self):
        return len(MACRO_NAMES) if self.macro else len(KEYS)

    def reset(self, seed):
        self.raw, _ = self.native.reset(seed=seed)
        self.memory = PublicMemory()
        self.elapsed = 0
        self.total = self.bonus_total = self.shaping_total = 0.0
        self.has_wand = has_wand(self.raw)
        self.potential = self._potential()
        self.visited = {tuple(map(int, self.raw['blstats'][:2]))}
        self.active = True
        return self.memory.encode(self.raw)

    def _potential(self):
        return self.align_bonus * float(self.has_wand and aligned(self.raw)) if self.align_bonus else 0.0

    def step(self, action):
        if not self.active:
            raise RuntimeError('Reset before stepping a completed game')
        action = int(action)
        if not self.macro:
            return self._native_step(action)
        name = MACRO_NAMES[action]
        if len(name) == 1:
            return self._native_step(KEYS.index(name))
        # Zap macro: z, then the wand (always inventory letter f), then the direction,
        # each only if the game asks for it.
        total = 0.0
        for key, prompt in (('z', None), ('f', 'What do you want to zap?'), (name[1], 'In what direction?')):
            if prompt is not None and prompt not in text(self.raw['message']):
                break
            obs, reward, done, metrics = self._native_step(KEYS.index(key))
            total += reward
            if done:
                break
        return obs, total, done, metrics

    def _native_step(self, index):
        key = KEYS[index]
        message = text(self.raw['message'])
        if 'What do you want to' not in message and 'In what direction?' not in message:
            self.memory.command = key if key in 'zf' else None
        self.memory.previous_mode = self.memory.mode(self.raw)
        self.raw, reward, term, trunc, info = self.native.step(index)
        self.elapsed += 1
        self.memory.previous_action = index
        done = bool(term or trunc)
        bonus = 0.0
        if not done:
            point = tuple(map(int, self.raw['blstats'][:2]))
            if point not in self.visited:
                bonus = min(self.visit_bonus, max(0., self.bonus_cap-self.bonus_total))
                self.visited.add(point)
        shaping = 0.0
        if self.wand_bonus or self.align_bonus:
            holds = has_wand(self.raw)
            shaping += self.wand_bonus * (int(holds) - int(self.has_wand))
            self.has_wand = holds
        if self.align_bonus:
            potential = 0.0 if done else self._potential()
            shaping += self.gamma * potential - self.potential
            self.potential = potential
        if done and info['end_status'].name == 'DEATH':
            shaping -= self.death_penalty
        self.total += float(reward)
        self.bonus_total += bonus
        self.shaping_total += shaping
        self.active = not done
        # Native action limits define a finite task. Ended episodes never bootstrap.
        obs = np.zeros(DIM, np.float32) if done else self.memory.encode(self.raw, self.elapsed)
        metrics = dict(level=self.level, steps=self.elapsed, native_reward=float(reward),
                       native_return=self.total, bonus=self.bonus_total, shaping=self.shaping_total,
                       success=info['end_status'].name == 'TASK_SUCCESSFUL',
                       end_status=info['end_status'].name)
        return obs, float(reward) + bonus + shaping, done, metrics

    def close(self):
        self.native.close()


def has_wand(raw):
    return any(b'wand' in bytes(row) for row in raw['inv_strs'])


def aligned(raw):
    """A visible monster on the player's row, column or diagonal."""
    x, y = map(int, raw['blstats'][:2])
    monsters = MONSTER[raw['glyphs']]
    monsters[y, x] = False
    ys, xs = np.nonzero(monsters)
    dx, dy = xs - x, ys - y
    return bool(np.any((dx == 0) | (dy == 0) | (np.abs(dx) == np.abs(dy))))

