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
from wand_agent.ppo import ActorCritic, compute_gae, load_model, ppo_update, save_model

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
        self.assertEqual(value.shape, (16,))
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
        stats = ppo_update(model, optimizer, (obs, actions, old_logp, advantages, values + 1, values), c,
                           np.random.default_rng(0))
        with torch.no_grad():
            new_logp = torch.log_softmax(model(obs)[0], -1).gather(1, actions[:, None]).squeeze(1)
        change = new_logp - old_logp
        self.assertGreater(change[0::2].mean().item(), 0)
        self.assertLess(change[1::2].mean().item(), 0)
        for key in ('approx_kl', 'clipfrac', 'entropy', 'policy_loss', 'value_loss', 'explained_variance'):
            self.assertIn(key, stats)


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

    def test_wod_short_run(self):
        summary, metrics = self.run_training('configs/ppo.json', '--steps', '1024')
        self.assertEqual((summary['status'], summary['steps'], len(metrics)), ('finished', 1024, 1))
        self.assertTrue(np.isfinite(metrics[0]['entropy']))


if __name__ == '__main__':
    unittest.main()
