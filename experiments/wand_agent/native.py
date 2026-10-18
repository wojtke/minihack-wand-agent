"""Small, reproducible entry point to the unmodified MiniHack tasks."""

from contextlib import chdir
from tempfile import TemporaryDirectory

import gymnasium as gym
import minihack  # noqa: F401 - registers the environments

LEVELS = ("Easy", "Medium", "Hard", "Pro")
KEYS = "kljhunby,zf"
HORIZONS = dict(zip(LEVELS, (50, 150, 400, 1000)))
OBSERVATIONS = ("glyphs", "chars", "colors", "blstats", "message", "inv_strs", "inv_letters")


class MiniHackEnv(gym.Wrapper):
    """Make NLE 1.1 seeding and timeouts follow the Gymnasium API."""

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            # NLE reset(seed=...) alone does not seed the two NetHack RNGs.
            self.unwrapped.seed(seed, seed, reseed=False)
            self.action_space.seed(seed)
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # NLE 1.1 reports its internal step limit as a terminal state.
        if info["end_status"].name == "ABORTED":
            terminated, truncated = False, True
        return obs, reward, terminated, truncated, info


def make_env(variant="Hard", actions="Restricted", *, pixels=False):
    keys = OBSERVATIONS + (("pixel",) if pixels else ())
    # The upstream level compiler's shell script does not quote its working path.
    # A temporary cwd also works when this project lives in Google Drive/My Drive.
    with TemporaryDirectory(prefix="wand-level-") as directory, chdir(directory):
        return MiniHackEnv(gym.make(f"MiniHack-WoD-{variant}-{actions}-v0", observation_keys=keys))


def text(array):
    return array.tobytes().split(b"\0", 1)[0].decode("utf-8", errors="replace")


def action_index(env, key):
    """Convert a NetHack key to this environment's discrete action index."""
    return tuple(map(int, env.unwrapped.actions)).index(ord(key))
