"""Preserve NLE's public screen when it clears arrays during timeout cleanup."""
import gymnasium as gym


class FinalObservation(gym.Wrapper):
    """Reward, action space, task and horizon unchanged; recover a timeout screen.

    NLE reuses arrays and _quit_game overwrites them before step returns. Capture
    only the configured public observation channels, before that cleanup call.
    This depends on NLE 1.1 internals and is covered by native parity tests.
    """
    def __init__(self, env):
        super().__init__(env)
        self.saved = None
        original_quit = self.unwrapped._quit_game

        def capture_then_quit(observation, done):
            self.saved = {key: value.copy() for key, value in
                          self.unwrapped._get_observation(observation).items()}
            return original_quit(observation, done)

        self.unwrapped._quit_game = capture_then_quit

    def step(self, action):
        self.saved = None
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info["end_status"].name == "ABORTED":
            if self.saved is None:
                raise RuntimeError("NLE timeout did not pass through the expected cleanup hook")
            obs = self.saved
        return obs, reward, terminated, truncated, info
