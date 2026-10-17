"""Native-reward evaluation on a separate, explicit seed range."""
import argparse
import hashlib
import json
from pathlib import Path
import math
import torch
from .env import AgentEnv
from .native import LEVELS
from .network import load


def wilson(wins, games):
    p, z = wins/games, 1.96
    center = (p+z*z/(2*games))/(1+z*z/games)
    margin = z*math.sqrt(p*(1-p)/games+z*z/(4*games*games))/(1+z*z/games)
    return [max(0., center-margin), min(1., center+margin)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, default=Path('checkpoints/shared-best.npz'))
    p.add_argument('--games', type=int, default=100)
    p.add_argument('--seed-start', type=int, default=100_000_000)
    p.add_argument('--out', type=Path, default=Path('runs/evaluation.json'))
    a = p.parse_args()
    if a.games <= 0 or a.seed_start < 100_000_000 or a.seed_start+a.games > 2**31:
        p.error('Use positive games and held-out seeds in [100000000, 2**31)')
    if a.out.exists(): p.error('Output already exists; choose a new --out')
    torch.set_num_threads(1)
    model = load(a.model)
    report = dict(checkpoint_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),
                  games_per_level=a.games, seed_start=a.seed_start, bonus=0, levels={})
    for level in LEVELS:
        env = AgentEnv(level)
        rows = []
        try:
            for seed in range(a.seed_start, a.seed_start+a.games):
                obs = env.reset(seed)
                done = False
                while not done:
                    obs, _, done, metrics = env.step(model.action(obs))
                rows.append(dict(seed=seed, **metrics))
        finally: env.close()
        wins = sum(row['success'] for row in rows)
        result = dict(games=a.games, wins=wins, success_rate=wins/a.games,
                      wilson_95=wilson(wins, a.games),
                      native_mean_return=sum(row['native_return'] for row in rows)/a.games,
                      mean_steps=sum(row['steps'] for row in rows)/a.games)
        report['levels'][level] = dict(**result, episodes=rows)
        print(json.dumps(dict(level=level, **result)), flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__': main()
