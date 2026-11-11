import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from wand_agent.agents import PPOAgent, load_agent
from wand_agent.features import DIM
from wand_agent.native import KEYS
from wand_agent.ppo import (ActorCritic, SelfImitation, compute_gae, level_mix, load_model, ppo_update,
                            save_model, sil_update)
from wand_agent.rnd import RND, RunningMeanStd

ROOT = Path(__file__).resolve().parents[1]


def column(*values):
    """A rollout of one game: shape [T, 1]."""
    return torch.tensor(values, dtype=torch.float32)[:, None]


class GaeTests(unittest.TestCase):
    def test_easy_win_matches_hand_calculation(self):
        # z, f, l on Easy: two menu inputs cost 0.01, the shot wins. Critic says 0.6, 0.7, 0.9.
        rewards, values = column(-0.01, -0.01, 1.0), column(0.6, 0.7, 0.9)
        ends = torch.tensor([[False], [False], [True]])
        advantages, returns = compute_gae(rewards, values, ends, torch.tensor([5.0]), 0.99, 0.95)
        torch.testing.assert_close(advantages[:, 0], torch.tensor([0.34169, 0.27505, 0.1]), atol=1e-5, rtol=0)
        torch.testing.assert_close(returns, advantages + values)

    def test_lambda_one_gives_return_minus_value_and_zero_gives_delta(self):
        rewards, values = column(-0.01, -0.01, 1.0), column(0.6, 0.7, 0.9)
        ends = torch.tensor([[False], [False], [True]])
        full, _ = compute_gae(rewards, values, ends, torch.tensor([0.0]), 0.99, 1.0)
        returns = torch.tensor([-0.01 - 0.0099 + 0.9801, -0.01 + 0.99, 1.0])
        torch.testing.assert_close(full[:, 0], returns - values[:, 0], atol=1e-6, rtol=0)
        one_step, _ = compute_gae(rewards, values, ends, torch.tensor([0.0]), 0.99, 0.0)
        torch.testing.assert_close(one_step[:, 0], torch.tensor([0.083, 0.181, 0.1]), atol=1e-6, rtol=0)

    def test_game_end_stops_advantage_and_last_step_bootstraps(self):
        # Step 1 ends a game; steps 2-3 belong to the next one, cut off by the rollout.
        rewards, values = column(0.0, 1.0, 0.0, 0.0), column(0.5, 0.5, 0.2, 0.3)
        ends = torch.tensor([[False], [True], [False], [False]])
        advantages, _ = compute_gae(rewards, values, ends, torch.tensor([0.4]), 0.9, 1.0)
        # The next game's values and rewards never reach steps 0-1.
        self.assertAlmostEqual(advantages[1, 0].item(), 1.0 - 0.5)
        self.assertAlmostEqual(advantages[0, 0].item(), 0.9 * 1.0 - 0.5, places=6)
        # The unfinished game uses the critic's estimate after the last step.
        self.assertAlmostEqual(advantages[3, 0].item(), 0.9 * 0.4 - 0.3, places=6)
        self.assertAlmostEqual(advantages[2, 0].item(), 0.9 * 0.9 * 0.4 - 0.2, places=6)

    def test_games_in_different_columns_are_independent(self):
        rewards = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
        values = torch.zeros(2, 2)
        ends = torch.tensor([[True, False], [False, False]])
        advantages, _ = compute_gae(rewards, values, ends, torch.zeros(2), 0.99, 0.95)
        self.assertEqual(advantages[:, 1].abs().sum().item(), 0.0)
        self.assertAlmostEqual(advantages[0, 0].item(), 1.0)


class ModelTests(unittest.TestCase):
    def test_initial_policy_is_nearly_uniform(self):
        torch.manual_seed(0)
        logits, value = ActorCritic(DIM, len(KEYS))(torch.randn(16, DIM))
        self.assertEqual(logits.shape, (16, len(KEYS)))
        self.assertEqual(value.shape, (16, 1))
        probabilities = logits.softmax(-1)
        self.assertLess((probabilities - 1 / len(KEYS)).abs().max().item(), 0.02)

    def test_checkpoint_round_trip_and_agent(self):
        for shared in (False, True):
            model = ActorCritic(DIM, len(KEYS), hidden=(32, 16), shared=shared)
            x = torch.randn(4, DIM)
            with tempfile.TemporaryDirectory() as d:
                path = Path(d) / 'policy.npz'
                save_model(model, path)
                loaded = load_model(path)
                for a, b in zip(model(x), loaded(x)):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                agent = load_agent(path)
                self.assertIsInstance(agent, PPOAgent)
                obs = x[0].numpy()
                self.assertEqual(agent.act(obs), int(model(x[:1])[0].argmax()))
                samples = [load_agent(path, seed=3, sample=True).act(obs) for _ in range(2)]
                self.assertEqual(samples[0], samples[1])

    def test_update_favours_actions_with_positive_advantage(self):
        torch.manual_seed(0)
        model = ActorCritic(4, 3, hidden=(16,))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, eps=1e-5)
        obs = torch.randn(64, 4)
        # Action 0 always turned out better than expected, action 1 always worse.
        actions = torch.tensor([0, 1] * 32)
        advantages = torch.tensor([1.0, -1.0] * 32)
        with torch.no_grad():
            logits, values = model(obs)
            old_logp = torch.log_softmax(logits, -1).gather(1, actions[:, None]).squeeze(1)
        c = dict(num_minibatches=2, update_epochs=4, clip_coef=0.2, norm_adv=True, clip_vloss=True,
                 ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, target_kl=None)
        column = lambda x: x[:, None]
        rollout = dict(obs=obs[:, None], actions=column(actions), logp=column(old_logp),
                       advantages=column(advantages), returns=column(values + 1), values=column(values),
                       first=torch.zeros(64, 1, dtype=torch.bool), state=None)
        stats = ppo_update(model, optimizer, rollout, c, np.random.default_rng(0))
        with torch.no_grad():
            new_logp = torch.log_softmax(model(obs)[0], -1).gather(1, actions[:, None]).squeeze(1)
        change = new_logp - old_logp
        self.assertGreater(change[0::2].mean().item(), 0)
        self.assertLess(change[1::2].mean().item(), 0)
        for key in ('approx_kl', 'clipfrac', 'entropy', 'policy_loss', 'value_loss', 'explained_variance'):
            self.assertIn(key, stats)



class SelfImitationTests(unittest.TestCase):
    def test_stores_only_won_games_with_discounted_returns_and_balances_levels(self):
        sil = SelfImitation(num_envs=2, gamma=0.5, capacity=100)
        obs = torch.zeros(2, 3)
        won, lost = dict(level='Medium', success=True), dict(level='Hard', success=False)
        sil.record(obs, [0, 1], [0.0, 0.0], [False, False], [None, None])
        sil.record(obs, [2, 1], [1.0, 0.0], [True, True], [won, lost])
        self.assertEqual((sil.games, sil.wins, list(sil.levels)), (1, 1, ['Medium']))
        _, actions, returns = sil.levels['Medium']['games'][0]
        np.testing.assert_allclose(returns, [0.5, 1.0])  # 0 + 0.5 * 1, then 1
        np.testing.assert_array_equal(actions, [0, 2])
        for _ in range(50):  # many short Easy wins
            sil.record(obs, [3, 3], [1.0, 1.0], [True, True], [dict(level='Easy', success=True)] * 2)
        _, actions, _ = sil.sample(100, np.random.default_rng(0))
        self.assertEqual(int((actions != 3).sum()), 50)  # half the batch still comes from Medium

    def test_update_raises_probability_of_won_actions_above_the_critic(self):
        torch.manual_seed(0)
        model = ActorCritic(4, 3, hidden=(16,))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, eps=1e-5)
        sil = SelfImitation(num_envs=1, gamma=0.99, capacity=1000)
        obs = torch.randn(1, 4)
        sil.record(obs, [2], [1.0], [True], [dict(level='Medium', success=True)])
        before = torch.log_softmax(model(obs)[0], -1)[0, 2].item()
        c = dict(sil_updates=20, sil_batch=8, sil_coef=1.0, sil_value_coef=0.01, max_grad_norm=0.5)
        stats = sil_update(model, optimizer, sil, c, np.random.default_rng(0))
        after = torch.log_softmax(model(obs)[0], -1)[0, 2].item()
        self.assertGreater(after, before + 0.5)
        self.assertEqual((stats['sil_games'], stats['sil_wins']), (1, 1))


class MemoryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = ActorCritic(5, 3, hidden=(8,), lstm=6)

    def test_memory_is_zeroed_at_the_start_of_a_game(self):
        obs = torch.randn(7, 2, 5)
        first = torch.zeros(7, 2, dtype=torch.bool)
        first[4, 0] = True  # game 0 restarts at step 4; game 1 continues
        state = tuple(torch.randn(1, 2, 6) for _ in range(2))
        logits, values, _ = self.model.sequence(obs, state, first)
        fresh_logits, fresh_values, _ = self.model.sequence(obs[4:, :1], self.model.initial_state(1))
        torch.testing.assert_close(logits[4:, :1], fresh_logits)
        torch.testing.assert_close(values[4:, :1], fresh_values)
        # Without the restart the earlier steps would still influence step 4.
        no_reset, _, _ = self.model.sequence(obs, state, torch.zeros(7, 2, dtype=torch.bool))
        self.assertFalse(torch.allclose(no_reset[4, 0], logits[4, 0]))
        torch.testing.assert_close(no_reset[:, 1], logits[:, 1])

    def test_replayed_rollout_gives_the_probabilities_seen_while_playing(self):
        steps, n = 9, 3
        obs = torch.randn(steps, n, 5)
        first = torch.rand(steps, n) < 0.2
        start = tuple(torch.randn(1, n, 6) for _ in range(2))
        state, played = start, []
        with torch.no_grad():
            for t in range(steps):
                logits, _, state = self.model.step(obs[t], state, first[t])
                played.append(logits)
            replayed, _, final = self.model.sequence(obs, start, first)
        torch.testing.assert_close(torch.stack(played), replayed)
        torch.testing.assert_close(final[0], state[0])
        # A minibatch of some games, as in the update, matches too.
        columns = torch.tensor([2, 0])
        part, _, _ = self.model.sequence(obs[:, columns], tuple(s[:, columns] for s in start), first[:, columns])
        torch.testing.assert_close(part, replayed[:, columns])

    def test_agent_memory_resets_between_games_and_survives_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'policy.npz'
            save_model(self.model, path)
            agent = load_agent(path)
        obs = [torch.randn(5).numpy() for _ in range(4)]
        first_game = [agent.act(o) for o in obs]
        agent.reset()
        self.assertEqual([agent.act(o) for o in obs], first_game)
        with torch.no_grad():
            logits, _, _ = self.model.sequence(torch.tensor(np.stack(obs))[:, None], self.model.initial_state(1))
        self.assertEqual(first_game, logits[:, 0].argmax(-1).tolist())

    def test_memoryless_forward_refuses_a_recurrent_model(self):
        with self.assertRaises(RuntimeError):
            self.model(torch.randn(2, 5))


class RndTests(unittest.TestCase):
    def test_bonus_shrinks_for_observations_seen_often_and_target_stays_fixed(self):
        torch.manual_seed(0)
        rnd = RND(10, hidden=32, output=8, update_proportion=1.0)
        seen, novel = torch.randn(32, 10), torch.randn(32, 10) + 2
        rnd.update_obs_stats(torch.cat([seen, novel]))
        target_before = [w.clone() for w in rnd.target.parameters()]
        optimizer = torch.optim.Adam(rnd.predictor.parameters(), lr=1e-3)
        before = rnd.error(seen).mean().item()
        for _ in range(300):
            optimizer.zero_grad()
            rnd.loss(seen).backward()
            optimizer.step()
        after = rnd.error(seen).mean().item()
        self.assertLess(after, 0.2 * before)
        self.assertGreater(rnd.error(novel).mean().item(), 3 * after)
        for old, new in zip(target_before, rnd.target.parameters()):
            torch.testing.assert_close(old, new)
            self.assertFalse(new.requires_grad)

    def test_running_statistics_match_all_data(self):
        stats, data = RunningMeanStd((3,)), torch.randn(500, 3) * 4 + 1
        for chunk in data.split(77):
            stats.update(chunk)
        torch.testing.assert_close(stats.mean, data.mean(0), atol=1e-3, rtol=0)
        torch.testing.assert_close(stats.var, data.var(0, unbiased=False), atol=1e-2, rtol=1e-3)

    def test_intrinsic_reward_has_rollout_shape_and_is_positive(self):
        rnd = RND(6, hidden=16, output=4)
        obs = torch.randn(5, 3, 6)
        rnd.update_obs_stats(obs)
        bonus, raw = rnd.intrinsic_reward(obs)
        self.assertEqual(bonus.shape, (5, 3))
        self.assertTrue((raw > 0).all() and (bonus > 0).all())

    def test_agent_from_rnd_checkpoint_uses_only_the_policy(self):
        model = ActorCritic(DIM, len(KEYS), hidden=(16,), value_heads=2)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'policy.npz'
            save_model(model, path)
            agent = load_agent(path)
        obs = torch.randn(DIM)
        self.assertEqual(agent.act(obs.numpy()), int(model(obs[None])[0].argmax()))
        self.assertEqual(model(obs[None])[1].shape, (1, 2))


class LevelMixTests(unittest.TestCase):
    def test_shares_follow_loss_rate_with_a_minimum_and_chances_correct_for_game_length(self):
        games = lambda wins, n, steps: [dict(success=i < wins, steps=steps) for i in range(n)]
        window = dict(Easy=games(10, 10, 3), Medium=games(10, 10, 10), Hard=games(5, 10, 50), Pro=games(0, 10, 300))
        target, chance = level_mix(window, ['Easy', 'Medium', 'Hard', 'Pro'], 0.05)
        # Losses 0, 0, 0.5, 1: the 0.8 left after the minimum goes 1/3 to Hard and 2/3 to Pro.
        np.testing.assert_allclose(target, [0.05, 0.05, 0.05 + 0.8 / 3, 0.05 + 1.6 / 3])
        per_step = target / np.array([3, 10, 50, 300])
        np.testing.assert_allclose(chance, per_step / per_step.sum())

    def test_levels_outside_the_mix_get_nothing_and_unplayed_levels_count_as_lost(self):
        window = dict(Medium=[dict(success=True, steps=10)])
        target, chance = level_mix(window, ['Medium', 'Pro'], 0.1)
        np.testing.assert_allclose(target, [0, 0.1, 0, 0.9])
        np.testing.assert_allclose(chance, [0, 0.1, 0, 0.9])  # unknown length: the mean of known ones


class TrainingSmokeTests(unittest.TestCase):
    def run_training(self, config, *extra):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'run'
            subprocess.run([sys.executable, '-m', 'wand_agent.ppo', '--config', config, '--out', str(out), *extra],
                           cwd=ROOT, check=True, capture_output=True, text=True)
            summary = json.loads((out / 'summary.json').read_text())
            metrics = [json.loads(line) for line in (out / 'metrics.jsonl').read_text().splitlines()]
            self.assertTrue((out / 'latest.npz').exists())
            return summary, metrics

    def test_cartpole_short_run(self):
        summary, metrics = self.run_training('configs/ppo-cartpole.json', '--steps', '2048')
        self.assertEqual(summary['status'], 'finished')
        self.assertEqual(summary['steps'], 2048)
        self.assertEqual(len(metrics), 4)

    def test_memory_task_short_run_with_lstm(self):
        summary, metrics = self.run_training('configs/ppo-memory.json', '--steps', '512', '--lstm', '8')
        self.assertEqual((summary['status'], summary['steps'], len(metrics)), ('finished', 512, 2))

    def test_wod_short_run(self):
        # Default mix: one Easy game and two of each other level, 128 steps each.
        summary, metrics = self.run_training('configs/ppo.json', '--steps', '896')
        self.assertEqual((summary['status'], summary['steps'], len(metrics)), ('finished', 896, 1))
        self.assertTrue(np.isfinite(metrics[0]['entropy']))

    def test_wod_short_run_with_curiosity_and_memory(self):
        summary, metrics = self.run_training('configs/ppo.json', '--steps', '1792', '--rnd', '--lstm', '16')
        self.assertEqual((summary['status'], summary['steps'], len(metrics)), ('finished', 1792, 2))
        for row in metrics:
            self.assertGreater(row['bonus_raw_mean'], 0)
            self.assertEqual(len(row['bonus_by_game']), 7)
            self.assertTrue(np.isfinite(row['rnd_loss']))

    def test_wod_short_run_with_adaptive_levels(self):
        summary, metrics = self.run_training('configs/ppo.json', '--steps', '1792', '--adaptive-levels', '--sil')
        self.assertEqual((summary['status'], summary['steps'], len(metrics)), ('finished', 1792, 2))
        for row in metrics:
            self.assertAlmostEqual(sum(row['level_share'].values()), 1)
            self.assertGreaterEqual(min(row['level_share'].values()), 0.05 - 1e-9)
            self.assertAlmostEqual(sum(row['level_chance'].values()), 1)

    def test_init_from_checkpoint_starts_from_its_weights_and_refuses_other_networks(self):
        with tempfile.TemporaryDirectory() as d:
            def train(out, *extra):
                return subprocess.run([sys.executable, '-m', 'wand_agent.ppo', '--config', 'configs/ppo-cartpole.json',
                                       '--steps', '512', '--out', str(Path(d) / out), *extra],
                                      cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(train('first').returncode, 0)
            start = Path(d) / 'first' / 'latest.npz'
            self.assertEqual(train('second', '--init-from', str(start)).returncode, 0)
            config = json.loads((Path(d) / 'second' / 'config.json').read_text())
            self.assertEqual(config['init_from']['path'], str(start))
            # One update moves the weights a little, but far less than a fresh random start would differ.
            first, second = load_model(start), load_model(Path(d) / 'second' / 'latest.npz')
            fresh = ActorCritic(**first.meta)
            moved = sum((a - b).abs().sum() for a, b in zip(first.parameters(), second.parameters()))
            random = sum((a - b).abs().sum() for a, b in zip(first.parameters(), fresh.parameters()))
            self.assertLess(moved, 0.2 * random)
            refused = train('third', '--init-from', str(start), '--lstm', '8')
            self.assertEqual(refused.returncode, 2)
            self.assertIn('differs', refused.stderr)


if __name__ == '__main__':
    unittest.main()
