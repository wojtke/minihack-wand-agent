"""Common interface for every policy that plays one game at a time.

evaluate.py and play.py only use this interface:
    agent.reset()      before each game (clears memory, e.g. an LSTM state)
    agent.act(obs)     -> index into KEYS for one 463-feature observation
                          (into env.MACRO_NAMES when agent.macro is True)
Training code may use batched methods of its own; this is the playing side.
"""
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from .env import MACRO_NAMES
from .features import ON_ITEM_DIM
from .native import KEYS
from .network import load as load_q_network


class Agent(Protocol):
    macro: bool  # True: act() indexes env.MACRO_NAMES instead of KEYS
    on_item: bool  # True: observations carry the extra "object under the player" input

    def reset(self) -> None: ...

    def act(self, obs: np.ndarray) -> int: ...


class RandomAgent:
    """Uniform over the 11 inputs: the baseline every method must beat."""
    macro = on_item = False

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def act(self, obs):
        return int(self.rng.integers(len(KEYS)))


class QAgent:
    """Greedy policy of a Double DQN checkpoint (no exploration)."""
    macro = on_item = False

    def __init__(self, network):
        self.network = network

    def reset(self):
        pass

    def act(self, obs):
        return self.network.action(obs)


class PPOAgent:
    """Actor of a PPO checkpoint. Greedy by default; sample=True draws from the policy.

    With an LSTM the agent keeps its memory between act() calls; reset() clears it.
    """

    def __init__(self, model, sample=False, seed=0):
        self.model, self.sample = model, sample
        # A policy with 17 outputs was trained on zap macro-actions (env.MACRO_NAMES).
        self.macro = model.meta['n_actions'] == len(MACRO_NAMES)
        self.on_item = model.meta['obs_dim'] == ON_ITEM_DIM
        self.generator = torch.Generator().manual_seed(seed)
        self.reset()

    def reset(self):
        self.state = self.model.initial_state(1)
        self.first = torch.ones(1, dtype=torch.bool)

    @torch.no_grad()
    def act(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32)[None]
        logits, _, self.state = self.model.step(obs, self.state, self.first)
        self.first = torch.zeros(1, dtype=torch.bool)
        if self.sample:
            return int(torch.multinomial(logits.softmax(-1)[0], 1, generator=self.generator).item())
        return int(logits.argmax().item())


def load_agent(spec, seed=0, sample=False):
    """'random' or a checkpoint path. Checkpoints without a 'kind' entry are DQN."""
    if spec == 'random':
        return RandomAgent(seed)
    path = Path(spec)
    with np.load(path, allow_pickle=False) as data:
        kind = str(data['kind']) if 'kind' in data.files else 'dqn'
    if kind == 'dqn':
        return QAgent(load_q_network(path))
    if kind == 'ppo':
        from .ppo import load_model
        return PPOAgent(load_model(path), sample=sample, seed=seed)
    raise ValueError(f'Unknown checkpoint kind {kind!r} in {path}')
