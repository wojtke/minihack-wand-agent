import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from wand_agent.env import AgentEnv
from wand_agent.native import LEVELS, HORIZONS, KEYS, make_env
from wand_agent.network import QNetwork, load, save
from wand_agent.train import double_dqn_target


class AgentTests(unittest.TestCase):
    def test_double_dqn_selection_and_finite_horizon(self):
        # Online selects action 0, target prefers 1: Double DQN must evaluate 0.
        online = lambda x: torch.tensor([[4., 1.], [4., 1.]])
        target = lambda x: torch.tensor([[2., 9.], [2., 9.]])
        values = double_dqn_target(online, target, torch.zeros(2, 1),
                                  torch.tensor([1., 3.]), torch.tensor([0., 1.]), .5)
        torch.testing.assert_close(values, torch.tensor([2., 3.]))

    def test_portable_checkpoint_preserves_values(self):
        q = QNetwork()
        x = torch.randn(8, 463)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'policy.npz'
            save(q, path)
            torch.testing.assert_close(q(x), load(path)(x), rtol=0, atol=0)

    def test_native_parity_and_timeouts_all_levels(self):
        for level in LEVELS:
            env, native = AgentEnv(level), make_env(level)
            try:
                first = env.reset(1234)
                raw, _ = native.reset(seed=1234)
                self.assertEqual(first.shape, (463,))
                np.testing.assert_array_equal(env.raw['glyphs'], raw['glyphs'])
                # A menu input that does not advance game time still counts.
                for step in range(HORIZONS[level]):
                    action = KEYS.index('f')
                    obs, reward, done, metrics = env.step(action)
                    _, native_reward, term, trunc, info = native.step(action)
                    self.assertEqual(reward, native_reward)
                    self.assertEqual(done, term or trunc)
                    self.assertEqual(metrics['end_status'], info['end_status'].name)
                    self.assertTrue(np.isfinite(obs).all())
                    if done: break
                self.assertTrue(done)
                self.assertEqual(step+1, HORIZONS[level])
                self.assertEqual(metrics['end_status'], 'ABORTED')
                self.assertTrue((obs == 0).all())
                self.assertTrue((env.raw['chars'] == ord('@')).any())
            finally:
                env.close(); native.close()

    def test_evaluation_reward_excludes_bonus(self):
        env, bonus_env = AgentEnv('Medium'), AgentEnv('Medium', visit_bonus=.1)
        try:
            env.reset(0); bonus_env.reset(0)
            for key in 'llllllll':
                a, reward, done, metrics = env.step(KEYS.index(key))
                b, boosted, bonus_done, more = bonus_env.step(KEYS.index(key))
                np.testing.assert_array_equal(a, b)
                self.assertEqual(done, bonus_done)
                self.assertEqual(metrics['native_return'], more['native_return'])
                self.assertEqual(metrics['bonus'], 0)
                self.assertGreaterEqual(boosted, reward)
                if done: break
            self.assertGreater(bonus_env.bonus_total, 0)
        finally: env.close(); bonus_env.close()


if __name__ == '__main__': unittest.main()
