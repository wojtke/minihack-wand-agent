"""Common interface for every policy that plays one game at a time.

evaluate.py and play.py only use this interface:
    agent.reset()      before each game (clears memory, e.g. an LSTM state)
    agent.act(obs)     -> index into KEYS for one 463-feature observation
Training code may use batched methods of its own; this is the playing side.
"""
from pathlib import Path
from typing import Protocol

import numpy as np

from .native import KEYS
from .network import load as load_q_network


class Agent(Protocol):
    def reset(self) -> None: ...

    def act(self, obs: np.ndarray) -> int: ...


class RandomAgent:
    """Uniform over the 11 inputs: the baseline every method must beat."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def act(self, obs):
        return int(self.rng.integers(len(KEYS)))


class QAgent:
    """Greedy policy of a Double DQN checkpoint (no exploration)."""

    def __init__(self, network):
        self.network = network

    def reset(self):
        pass

    def act(self, obs):
        return self.network.action(obs)


def load_agent(spec, seed=0):
    """'random' or a checkpoint path. Checkpoints without a 'kind' entry are DQN."""
    if spec == 'random':
        return RandomAgent(seed)
    path = Path(spec)
    with np.load(path, allow_pickle=False) as data:
        kind = str(data['kind']) if 'kind' in data.files else 'dqn'
    if kind == 'dqn':
        return QAgent(load_q_network(path))
    raise ValueError(f'Unknown checkpoint kind {kind!r} in {path}')
