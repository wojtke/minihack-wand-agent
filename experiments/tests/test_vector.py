import unittest

import numpy as np

from wand_agent.agents import QAgent, RandomAgent, load_agent
from wand_agent.env import AgentEnv
from wand_agent.features import DIM
from wand_agent.native import HORIZONS, KEYS, LEVELS
from wand_agent.vector import TRAIN_SEED_LIMIT, VectorEnv


def rollout(env, actions):
    """Step a vector env through fixed actions and collect everything it returns."""
    out = [env.reset()]
    for row in actions:
        obs, rewards, dones, infos = env.step(row)
        out.append((obs, rewards, dones, [i and (i['seed'], i['steps'], i['end_status']) for i in infos]))
    return out


def assert_same(test, a, b):
    test.assertEqual(len(a), len(b))
    np.testing.assert_array_equal(a[0], b[0])
    for x, y in zip(a[1:], b[1:]):
        for u, v in zip(x[:3], y[:3]):
            np.testing.assert_array_equal(u, v)
        test.assertEqual(x[3], y[3])


class VectorTests(unittest.TestCase):
    levels = LEVELS * 2
    actions = np.random.default_rng(0).integers(len(KEYS), size=(120, len(LEVELS) * 2))

    def test_same_seed_same_games_in_processes_and_in_process(self):
        with VectorEnv(self.levels, seed=7, processes=False) as serial:
            expected = rollout(serial, self.actions)
        # One game per process, and uneven groups of several games per process.
        for workers in (8, 3):
            with VectorEnv(self.levels, seed=7, workers=workers) as parallel:
                assert_same(self, rollout(parallel, self.actions), expected)

    def test_different_seed_different_games(self):
        with VectorEnv(self.levels, seed=7, processes=False) as a, VectorEnv(self.levels, seed=8, processes=False) as b:
            self.assertFalse(np.array_equal(a.reset(), b.reset()))

    def test_games_of_one_run_use_different_seeds(self):
        with VectorEnv(('Easy',) * 4, seed=3, processes=False) as env:
            env.reset()
            for _ in range(HORIZONS['Easy']):
                _, _, dones, infos = env.step([KEYS.index('f')] * 4)
            self.assertTrue(dones.all())
            seeds = [info['seed'] for info in infos]
            self.assertEqual(len(set(seeds)), 4)
            self.assertTrue(all(0 <= s < TRAIN_SEED_LIMIT for s in seeds))

    def test_finished_game_resets_to_next_seed(self):
        # 'f' asks for an item and never moves, so every Easy game ends at the action limit.
        wait = [[KEYS.index('f')]] * HORIZONS['Easy']
        with VectorEnv(('Easy',), seed=11, processes=False) as env:
            env.reset()
            for i, row in enumerate(wait):
                obs, rewards, dones, infos = env.step(row)
                self.assertEqual(dones[0], i == len(wait) - 1)
            first = infos[0]
            self.assertEqual((first['steps'], first['end_status']), (HORIZONS['Easy'], 'ABORTED'))
            after_reset = obs[0]
            for row in wait:
                _, _, _, infos = env.step(row)
            second_seed = infos[0]['seed']
        self.assertNotEqual(first['seed'], second_seed)
        single = AgentEnv('Easy')
        try:
            np.testing.assert_array_equal(after_reset, single.reset(second_seed))
        finally:
            single.close()

    def test_rejects_wrong_input(self):
        with self.assertRaises(ValueError):
            VectorEnv(('Impossible',), processes=False)
        with VectorEnv(('Easy',), processes=False) as env:
            env.reset()
            with self.assertRaises(ValueError):
                env.step([0, 0])


class AgentInterfaceTests(unittest.TestCase):
    def test_agents_return_valid_inputs(self):
        obs = np.zeros(DIM, np.float32)
        for agent in (RandomAgent(0), load_agent('checkpoints/shared-best.npz')):
            agent.reset()
            self.assertIn(agent.act(obs), range(len(KEYS)))
        self.assertIsInstance(load_agent('checkpoints/shared-best.npz'), QAgent)

    def test_random_agent_is_reproducible(self):
        obs = np.zeros(DIM, np.float32)
        a, b = RandomAgent(5), RandomAgent(5)
        self.assertEqual([a.act(obs) for _ in range(50)], [b.act(obs) for _ in range(50)])


if __name__ == '__main__': unittest.main()
