"""Random Network Distillation (Burda et al., 2018): a curiosity bonus.

A fixed, randomly initialised "target" network maps an observation to a vector.
A second "predictor" network is trained to reproduce that vector. On
observations the agent has seen many times the predictor is accurate, so its
error is small; on new observations the error is large. That error is the
intrinsic reward: it pays the agent for reaching situations it has not seen,
even before it ever wins. It never changes evaluation, which uses the game's
own reward only.

Details follow CleanRL's ppo_rnd_envpool.py (MIT licence,
https://github.com/vwxyzjn/cleanrl): observations are normalised by running
statistics and clipped to [-5, 5] before both networks, the bonus is divided by
the running standard deviation of its discounted sum, and the predictor learns
from a random fraction of each minibatch.
"""
import math

import torch
from torch import nn


class RunningMeanStd:
    """Mean and variance of everything seen so far, updated in batches."""

    def __init__(self, shape=()):
        self.mean = torch.zeros(shape)
        self.var = torch.ones(shape)
        self.count = 1e-4

    def update(self, x):
        batch_mean, batch_var, batch_count = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m2 = self.var * self.count + batch_var * batch_count + delta ** 2 * self.count * batch_count / total
        self.var = m2 / total
        self.count = total


def _mlp(sizes):
    layers = []
    for i, (inputs, outputs) in enumerate(zip(sizes[:-1], sizes[1:])):
        layer = nn.Linear(inputs, outputs)
        nn.init.orthogonal_(layer.weight, math.sqrt(2))
        nn.init.zeros_(layer.bias)
        layers.append(layer)
        if i < len(sizes) - 2:
            layers.append(nn.LeakyReLU())
    return nn.Sequential(*layers)


class RND(nn.Module):
    def __init__(self, obs_dim, hidden=256, output=64, gamma=0.99, update_proportion=0.25):
        super().__init__()
        self.target = _mlp([obs_dim, hidden, hidden, output])
        self.predictor = _mlp([obs_dim, hidden, hidden, hidden, output])
        for param in self.target.parameters():
            param.requires_grad_(False)  # the target never learns
        self.gamma, self.update_proportion = gamma, update_proportion
        self.obs_stats = RunningMeanStd((obs_dim,))
        self.return_stats = RunningMeanStd()
        self.discounted = None

    def _prepare(self, obs):
        return ((obs - self.obs_stats.mean) / self.obs_stats.var.sqrt().clamp_min(1e-8)).clamp(-5, 5)

    def update_obs_stats(self, obs):
        self.obs_stats.update(obs.reshape(-1, obs.shape[-1]))

    def error(self, obs):
        """Prediction error per observation: obs [..., D] -> [...]."""
        x = self._prepare(obs)
        return ((self.predictor(x) - self.target(x)) ** 2).sum(-1) / 2

    @torch.no_grad()
    def intrinsic_reward(self, next_obs):
        """Bonus for reaching each observation of a rollout [T, N, D] -> [T, N], normalised."""
        raw = self.error(next_obs)
        # Scale by the spread of the discounted bonus so its size stays stable during training.
        if self.discounted is None:
            self.discounted = torch.zeros(raw.shape[1])
        sums = []
        for t in range(raw.shape[0]):
            self.discounted = self.discounted * self.gamma + raw[t]
            sums.append(self.discounted)
        self.return_stats.update(torch.stack(sums).reshape(-1))
        return raw / self.return_stats.var.sqrt().clamp_min(1e-8), raw

    def loss(self, obs):
        """Predictor loss on a random fraction of the given observations [B, D]."""
        errors = ((self.predictor(self._prepare(obs)) - self.target(self._prepare(obs)).detach()) ** 2).mean(-1)
        mask = (torch.rand(errors.shape[0]) < self.update_proportion).float()
        return (errors * mask).sum() / mask.sum().clamp_min(1.0)
