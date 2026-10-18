"""Watch the same policy play any level in the terminal."""
import argparse
from pathlib import Path
import time
import torch
from .env import AgentEnv
from .native import LEVELS, KEYS, text
from .network import load


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--level', choices=LEVELS, default='Easy')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--model', type=Path, default=Path('checkpoints/shared-best.npz'))
    p.add_argument('--delay', type=float, default=.15)
    a = p.parse_args()
    if not 0 <= a.seed < 2**31 or a.delay < 0: p.error('Invalid seed or delay')
    torch.set_num_threads(1)
    model, env = load(a.model), AgentEnv(a.level)
    try:
        obs = env.reset(a.seed)
        done = False
        while not done:
            print('\033[2J\033[H', end='')
            print('\n'.join(''.join(map(chr, row)).rstrip() for row in env.raw['chars']))
            action = model.action(obs)
            print(text(env.raw['message']))
            print(f'{a.level} · step {env.elapsed} · next input {KEYS[action]!r}', flush=True)
            time.sleep(a.delay)
            obs, _, done, metrics = env.step(action)
        print(metrics)
    finally: env.close()


if __name__ == '__main__': main()
