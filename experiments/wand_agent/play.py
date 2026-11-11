"""Watch the same policy play any level in the terminal."""
import argparse
import time
import torch
from .env import MACRO_NAMES, AgentEnv
from .native import LEVELS, KEYS, text
from .agents import load_agent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--level', choices=LEVELS, default='Easy')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--model', default='checkpoints/shared-best.npz', help="Checkpoint path or 'random'")
    p.add_argument('--delay', type=float, default=.15)
    a = p.parse_args()
    if not 0 <= a.seed < 2**31 or a.delay < 0: p.error('Invalid seed or delay')
    torch.set_num_threads(1)
    agent = load_agent(a.model, seed=a.seed)
    env = AgentEnv(a.level, macro=agent.macro, on_item=agent.on_item)
    names = MACRO_NAMES if agent.macro else KEYS
    try:
        agent.reset()
        obs = env.reset(a.seed)
        done = False
        while not done:
            print('\033[2J\033[H', end='')
            print('\n'.join(''.join(map(chr, row)).rstrip() for row in env.raw['chars']))
            action = agent.act(obs)
            print(text(env.raw['message']))
            print(f'{a.level} · step {env.elapsed} · next input {names[action]!r}', flush=True)
            time.sleep(a.delay)
            obs, _, done, metrics = env.step(action)
        print(metrics)
    finally: env.close()


if __name__ == '__main__': main()
