"""Native-reward evaluation on a separate, explicit seed range.

Works for any agent from agents.load_agent: a checkpoint path or 'random'.
Levels are evaluated in parallel processes; results do not depend on --workers.
"""
import argparse
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path

import torch

from .agents import load_agent
from .env import AgentEnv
from .native import LEVELS


def wilson(wins, games):
    p, z = wins/games, 1.96
    center = (p+z*z/(2*games))/(1+z*z/games)
    margin = z*math.sqrt(p*(1-p)/games+z*z/(4*games*games))/(1+z*z/games)
    return [max(0., center-margin), min(1., center+margin)]


def play_games(agent, level, seeds):
    """Play one game per seed with native reward only; return per-game metrics."""
    env = AgentEnv(level)
    rows = []
    try:
        for seed in seeds:
            agent.reset()
            obs = env.reset(seed)
            done = False
            while not done:
                obs, _, done, metrics = env.step(agent.act(obs))
            rows.append(dict(seed=seed, **metrics))
    finally:
        env.close()
    return rows


def evaluate_level(args):
    model, level, seed_start, games, sample = args
    torch.set_num_threads(1)
    # Same agent seed for every level, so a level's result does not depend on the others.
    agent = load_agent(model, seed=seed_start, sample=sample)
    rows = play_games(agent, level, range(seed_start, seed_start+games))
    wins = sum(row['success'] for row in rows)
    return level, dict(games=games, wins=wins, success_rate=wins/games,
                       wilson_95=wilson(wins, games),
                       native_mean_return=sum(row['native_return'] for row in rows)/games,
                       mean_steps=sum(row['steps'] for row in rows)/games, episodes=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='checkpoints/shared-best.npz', help="Checkpoint path or 'random'")
    p.add_argument('--games', type=int, default=100)
    p.add_argument('--seed-start', type=int, default=100_000_000)
    p.add_argument('--levels', nargs='+', choices=LEVELS, default=list(LEVELS))
    p.add_argument('--workers', type=int, default=len(LEVELS), help='Levels evaluated at once')
    p.add_argument('--sample', action='store_true', help='PPO: draw actions from the policy instead of the most likely one')
    p.add_argument('--out', type=Path, default=Path('runs/evaluation.json'))
    a = p.parse_args()
    if a.games <= 0 or a.seed_start < 100_000_000 or a.seed_start+a.games > 2**31:
        p.error('Use positive games and held-out seeds in [100000000, 2**31)')
    if a.workers <= 0: p.error('Use at least one worker')
    if a.out.exists(): p.error('Output already exists; choose a new --out')
    report = dict(model=a.model, games_per_level=a.games, seed_start=a.seed_start, bonus=0,
                  policy='sample' if a.sample else 'greedy', levels={})
    if a.model != 'random':
        report['checkpoint_sha256'] = hashlib.sha256(Path(a.model).read_bytes()).hexdigest()
    jobs = [(a.model, level, a.seed_start, a.games, a.sample) for level in a.levels]
    with mp.get_context('spawn').Pool(min(a.workers, len(jobs))) as pool:
        for level, result in pool.imap(evaluate_level, jobs):
            report['levels'][level] = result
            print(json.dumps(dict(level=level, **{k: v for k, v in result.items() if k != 'episodes'})),
                  flush=True)
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__': main()
