import unittest

import numpy as np

from wand_agent.agents import QAgent, RandomAgent, load_agent
from wand_agent.env import MACRO_NAMES, AgentEnv
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


class ShapingTests(unittest.TestCase):
    def play(self, level, seed, keys=None, agent=None, **shaping):
        env = AgentEnv(level, **shaping)
        obs, rewards, metrics = env.reset(seed), [], None
        if agent is not None:
            agent.reset()
        for step in range(HORIZONS[level]):
            action = KEYS.index(keys[step]) if keys else agent.act(obs)
            obs, reward, done, metrics = env.step(action)
            rewards.append(reward)
            if done:
                break
        env.close()
        return rewards, metrics

    def test_death_penalty_applies_only_to_death_and_never_to_native_return(self):
        # Zapping up on Easy bounces the ray back: death.
        rewards, metrics = self.play('Easy', 0, 'zfk', death_penalty=1.0, wand_bonus=0.5)
        self.assertEqual(metrics['end_status'], 'DEATH')
        self.assertAlmostEqual(rewards[-1], metrics['native_reward'] - 1.0)
        self.assertAlmostEqual(metrics['shaping'], -1.0)  # wand held from the start: no wand bonus on Easy
        rewards, metrics = self.play('Easy', 0, 'zfl', death_penalty=1.0, wand_bonus=0.5)
        self.assertTrue(metrics['success'])
        self.assertEqual((metrics['shaping'], sum(rewards)), (0.0, metrics['native_return']))

    def test_wand_bonus_paid_once_when_the_wand_is_picked_up(self):
        agent = load_agent('checkpoints/shared-best.npz')
        rewards, metrics = self.play('Medium', 210000000, agent=agent, wand_bonus=0.5)
        self.assertTrue(metrics['success'])
        self.assertAlmostEqual(metrics['shaping'], 0.5)
        self.assertEqual(sum(1 for r in rewards if r >= 0.4), 2)  # pick-up step and the win
        self.assertAlmostEqual(sum(rewards), metrics['native_return'] + 0.5)

    def test_align_potential_telescopes_to_minus_its_start_value(self):
        # Easy seed 0: the wand is held and the minotaur is in line to the east, so Phi(start) = 0.5;
        # with gamma = 1 the shaping of a whole game sums to Phi(end) - Phi(start) = 0 - 0.5.
        rewards, metrics = self.play('Easy', 0, 'zfl', align_bonus=0.5, gamma=1.0)
        self.assertTrue(metrics['success'])
        self.assertAlmostEqual(metrics['shaping'], -0.5)
        self.assertAlmostEqual(sum(rewards), metrics['native_return'] - 0.5)
        rewards, metrics = self.play('Easy', 0, 'zfl', align_bonus=0.5, gamma=0.99)
        self.assertAlmostEqual(metrics['shaping'], 2 * (0.99 * 0.5 - 0.5) - 0.5)


class OnItemTests(unittest.TestCase):
    def test_on_item_input_marks_the_square_with_the_wand(self):
        # Medium seed 0: the wand lies one square east of the start.
        env = AgentEnv('Medium', macro=True, on_item=True)
        flags = [env.reset(0)[-1]] + [env.step(MACRO_NAMES.index(name))[0][-1] for name in ('l', ',', 'l')]
        env.close()
        self.assertEqual(flags, [0.0, 1.0, 0.0, 0.0])  # start, on the wand, picked up, moved on

    def test_on_item_is_off_by_default_and_vector_env_reports_its_size(self):
        env = AgentEnv('Medium')
        self.assertEqual(env.reset(0).shape, (DIM,))
        env.close()
        with VectorEnv(['Medium'], processes=False, on_item=True) as vector:
            self.assertEqual(vector.reset().shape, (1, DIM + 1))
            self.assertEqual(vector.obs_dim, DIM + 1)


class MacroTests(unittest.TestCase):
    def test_zap_macro_plays_z_f_direction_as_native_inputs(self):
        native = AgentEnv('Easy')
        native.reset(0)
        for key in 'zfl':
            _, _, done, expected = native.step(KEYS.index(key))
        native.close()
        env = AgentEnv('Easy', macro=True)
        env.reset(0)
        _, reward, done, metrics = env.step(MACRO_NAMES.index('zl'))
        env.close()
        self.assertTrue(done)
        self.assertEqual(metrics, expected)
        self.assertEqual(metrics['steps'], 3)  # every native input counts against the action limit
        self.assertAlmostEqual(reward, metrics['native_return'])

    def test_zap_macro_without_a_wand_stops_after_z(self):
        env = AgentEnv('Medium', macro=True)
        env.reset(0)
        _, _, done, metrics = env.step(MACRO_NAMES.index('zl'))
        env.close()
        self.assertFalse(done)
        self.assertEqual(metrics['steps'], 1)

    def test_vector_env_reports_macro_action_count(self):
        with VectorEnv(['Easy'], processes=False, macro=True) as env:
            self.assertEqual(env.n_actions, 17)
            env.reset()
            env.step([MACRO_NAMES.index('zl')])
            self.assertEqual(env.games[0].env.elapsed, 3)  # z, f and the direction
