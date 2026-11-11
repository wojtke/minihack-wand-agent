"""Record one game decision by decision for the HTML viewers (replay, inspect_games).

A recorded game is a JSON-ready dict:
    frames[t]  the public screen before decision t (the last frame is the end of the game)
    steps[t]   decision t: action, its native keys, policy probabilities, critic values
               and the native reward it earned
The map is stored as the first screen plus per-frame changed cells, which keeps
1000-action Pro games small. Recording uses the agent exactly as agents.PPOAgent.act
does, so a greedy recording is the same game evaluate.py plays for that seed.

on_item: checkpoints trained with one extra input (1 when an object lies under the
player, DIM + 1 inputs) are detected from the checkpoint and get that input too.
"""
import inspect

import numpy as np
import torch

from .agents import PPOAgent, QAgent
from .env import MACRO_NAMES, AgentEnv, has_wand
from .features import DIM, MONSTER, OBJECT
from .native import HORIZONS, KEYS, text

WAND, MINOTAUR = ord('/'), ord('H')


def events(raw):
    """Public facts about one screen: wand visible, wand held, minotaur visible."""
    chars, glyphs = raw['chars'], raw['glyphs']
    message = text(raw['message'])
    wands = OBJECT[glyphs] & (chars == WAND)
    # The player's '@' hides a wand underneath; the game then says "You see here a wand".
    wand_visible = bool(wands.any()) or ('wand' in message and 'see here' in message)
    minotaur = MONSTER[glyphs] & (chars == MINOTAUR)
    return dict(wand_visible=wand_visible, has_wand=has_wand(raw), minotaur_visible=bool(minotaur.any()),
                wands=[[int(x), int(y)] for y, x in zip(*np.nonzero(wands))])


@torch.no_grad()
def decide(agent, obs):
    """agent.act(obs) plus what the policy saw in it: probabilities and critic values."""
    if isinstance(agent, PPOAgent):
        logits, values, agent.state = agent.model.step(torch.as_tensor(obs, dtype=torch.float32)[None],
                                                       agent.state, agent.first)
        agent.first = torch.zeros(1, dtype=torch.bool)
        probs = logits.softmax(-1)[0]
        if agent.sample:
            action = int(torch.multinomial(probs, 1, generator=agent.generator).item())
        else:
            action = int(logits.argmax().item())
        return action, dict(probs=probs.tolist(), values=values[0].tolist())
    if isinstance(agent, QAgent):
        q = agent.network(torch.as_tensor(obs, dtype=torch.float32))
        return int(q.argmax().item()), dict(q=q.tolist(), values=[float(q.max())])
    action = agent.act(obs)
    return action, dict(probs=[1.0/len(KEYS)]*len(KEYS), values=[])


def uses_on_item(agent, force=False):
    """True when the agent's network expects the extra "object under the player" input."""
    model = getattr(agent, 'model', None)  # PPO; DQN networks always take DIM inputs
    if getattr(agent, 'on_item', False) or (model is not None and model.meta['obs_dim'] == DIM + 1):
        return True
    if force and (model is not None or isinstance(agent, QAgent)):
        raise ValueError(f'on_item: this checkpoint takes {DIM} inputs, not {DIM + 1}')
    return force


class _OnItem:
    """The on_item input for an AgentEnv that does not compute it itself.

    Mirrors the object memory of features.PublicMemory(on_item=True): objects seen on
    screen, plus the square under the player when the game says "You see here",
    minus a square whose object was just picked up or that no longer shows one.
    """
    def __init__(self):
        self.objects, self.items = set(), set()

    def __call__(self, raw):
        x, y = map(int, raw['blstats'][:2])
        chars, objects = raw['chars'], OBJECT[raw['glyphs']]
        self.objects.update((int(xx), int(yy)) for yy, xx in zip(*np.where(objects)))
        if text(raw['message']).startswith('You see here'):
            self.objects.add((x, y))
        items = set(map(int, raw['inv_letters'][raw['inv_letters'] != 0]))
        if items - self.items:
            self.objects.discard((x, y))
        self.items = items
        self.objects = {(xx, yy) for xx, yy in self.objects
                        if objects[yy, xx] or (xx == x and yy == y) or chars[yy, xx] == 32}
        return np.float32((x, y) in self.objects)


def _env(level, macro, on_item):
    """AgentEnv with the on_item input; computed here if this AgentEnv cannot."""
    if not on_item:
        return AgentEnv(level, macro=macro), None
    if 'on_item' in inspect.signature(AgentEnv).parameters:
        return AgentEnv(level, macro=macro, on_item=True), None
    return AgentEnv(level, macro=macro), _OnItem()


class _Screen:
    """Map glyphs: the first screen in full, then only the cells that changed.

    legend maps every glyph seen to its ASCII character and colour, for the text view.
    """
    def __init__(self, raw):
        self.glyphs = raw['glyphs'].copy()
        self.height, self.width = self.glyphs.shape
        self.first = [int(g) for g in self.glyphs.reshape(-1)]
        self.legend = {}
        self._learn(raw, np.arange(self.glyphs.size))
        self.diffs = []

    def _learn(self, raw, cells):
        glyphs, chars, colors = (raw[k].reshape(-1) for k in ('glyphs', 'chars', 'colors'))
        for index in cells:
            glyph = int(glyphs[index])
            if glyph not in self.legend:
                self.legend[glyph] = [int(chars[index]), int(colors[index])]

    def add(self, raw):
        glyphs = raw['glyphs']
        changed = np.flatnonzero(glyphs != self.glyphs)
        self._learn(raw, changed)
        flat = glyphs.reshape(-1)
        diff = []
        for index in changed:
            diff += [int(index), int(flat[index])]
        self.diffs.append(diff)
        self.glyphs = glyphs.copy()

    def to_dict(self):
        return dict(width=self.width, height=self.height, glyphs=self.first, diffs=self.diffs,
                    legend={str(g): v for g, v in self.legend.items()})


def _blank(raw):
    chars = raw['chars']
    return not np.any((chars != 0) & (chars != 32))


def _frame(env, obs, on_item):
    raw = env.raw
    x, y = map(int, raw['blstats'][:2])
    hp, maximum = map(int, raw['blstats'][[10, 11]])
    frame = dict(x=x, y=y, t=env.elapsed, hp=hp, hp_max=maximum, msg=text(raw['message']), **events(raw))
    if on_item:
        frame['on_item'] = bool(obs[-1])  # the extra input the policy saw before this decision
    return frame


def record_game(agent, level, seed, label='', on_item=False):
    """Play one game with native rewards only and return it as a recorded game.

    on_item=True forces the extra input; otherwise it is used when the checkpoint has it.
    """
    on_item = uses_on_item(agent, on_item)
    env, extra = _env(level, agent.macro, on_item)

    def observe(obs, done=False):
        if extra is None:
            return obs
        return np.append(obs, np.float32(0) if done else extra(env.raw))
    names = list(MACRO_NAMES if agent.macro else KEYS)
    if getattr(agent, 'sample', False):
        # One generator per game seed: a sampled game is repeatable on its own.
        agent.generator.manual_seed(seed)
    try:
        agent.reset()
        obs = observe(env.reset(seed))
        screen = _Screen(env.raw)
        frames, steps = [_frame(env, obs, on_item)], []
        done = False
        while not done:
            before, total = env.elapsed, env.total
            action, info = decide(agent, obs)
            obs, _, done, metrics = env.step(action)
            obs = observe(obs, done)
            name = names[action]
            # A zap macro plays z, f, direction and stops early if a prompt does not appear.
            sequence = 'zf' + name[1] if agent.macro and len(name) == 2 else name
            steps.append(dict(a=action, keys=sequence[:env.elapsed - before],
                              r=round(env.total - total, 4), **{k: _round(v) for k, v in info.items()}))
            if _blank(env.raw):
                # NLE clears the screen when the task is won: keep the last screen and position.
                screen.diffs.append([])
                frames.append(dict(frames[-1], t=env.elapsed, msg='', **{'end': True}))
            else:
                screen.add(env.raw)
                frames.append(_frame(env, obs, on_item))
    finally:
        env.close()
    value_heads = len(steps[0]['values']) if steps else 0
    return dict(label=label, level=level, seed=int(seed), horizon=HORIZONS[level],
                policy='sample' if getattr(agent, 'sample', False) else 'greedy', on_item=on_item,
                actions=names, value_heads=value_heads, result=metrics,
                map=screen.to_dict(), frames=frames, steps=steps, summary=summarize(frames, steps, names, metrics))


def summarize(frames, steps, names, result):
    """Event funnel of one game, with the same definitions as docs/ppo-summary.md."""
    zapped = any(frames[t]['has_wand'] and names[step['a']].startswith('z')
                 for t, step in enumerate(steps))
    first = {}
    for t, frame in enumerate(frames):
        for key in ('wand_visible', 'has_wand', 'minotaur_visible'):
            if frame[key] and key not in first:
                first[key] = t
    if zapped:
        first['zapped'] = next(t for t, step in enumerate(steps)
                               if frames[t]['has_wand'] and names[step['a']].startswith('z'))
    wand_at = next((frame['wands'][0] for frame in frames if frame['wands']), None)
    return dict(wand_seen='wand_visible' in first or 'has_wand' in first, picked='has_wand' in first,
                minotaur_seen='minotaur_visible' in first, zapped=zapped, success=bool(result['success']),
                end_status=result['end_status'], steps=result['steps'], decisions=len(steps),
                native_return=round(result['native_return'], 4),
                visited=len({(f['x'], f['y']) for f in frames}), first=first, wand_at=wand_at,
                end_at=[frames[-1]['x'], frames[-1]['y']])


# Funnel stages of a lost game, from the earliest failure to the latest.
STAGES = ('nie widział różdżki', 'widział różdżkę, nie podniósł', 'ma różdżkę, nie widział minotaura',
          'ma różdżkę i widział minotaura', 'zap z różdżką, bez wygranej')


def stage(summary):
    """Furthest funnel stage a game reached, as a short Polish label."""
    if summary['success']:
        return 'wygrana'
    if summary['zapped']:
        return STAGES[4]
    if summary['picked']:
        return STAGES[3] if summary['minotaur_seen'] else STAGES[2]
    return STAGES[1] if summary['wand_seen'] else STAGES[0]


def _round(values):
    return [round(float(v), 4) for v in values]


def checkpoint_label(path):
    """Short label for a checkpoint path: run name plus training step if known."""
    from pathlib import Path
    path = Path(path)
    if path.stem.startswith('step-'):
        return f'{path.parent.name} @ {int(path.stem[5:]) / 1e6:.1f}M'
    return f'{path.parent.name}/{path.stem}' if path.parent.name else path.stem
