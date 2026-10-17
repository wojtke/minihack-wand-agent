"""Native oracle for the browser port. Run from the repository root."""
import json
from pathlib import Path
import torch
from wand_agent.native import LEVELS, KEYS
from wand_agent import serve
from wand_agent.network import load
from wand_agent.env import AgentEnv

ROOT = Path(__file__).resolve().parents[1]

# Add glyphs to the test oracle without changing the production native server.
original_observation = serve.observation
def observation(raw):
    return {**original_observation(raw), 'glyphs': raw['glyphs'].ravel().tolist()}
serve.observation = observation


def main():
    torch.set_num_threads(1)
    model = load(ROOT / 'checkpoints/shared-best.npz')
    cases = []
    for level in LEVELS:
        for seed in [0, 1, 42, 1234, 210000000]:
            agent, game = AgentEnv(level), serve.Game(level, seed)
            try:
                obs = agent.reset(seed)
                states, keys = [game.state], []
                while not game.done:
                    action = model.action(obs)
                    keys.append(KEYS[action])
                    obs, _, _, _ = agent.step(action)
                    states.append(game.step(keys[-1]).copy())
                cases.append(dict(level=level, seed=seed, keys=''.join(keys), states=states, kind='policy'))
            finally:
                agent.close()
                game.close()
        for key, kind in [('f', 'timeout'), ('l', 'move')]:
            game = serve.Game(level, 0)
            try:
                keys, states = [], [game.state]
                while not game.done:
                    keys.append(key)
                    states.append(game.step(key).copy())
                cases.append(dict(level=level, seed=0, keys=''.join(keys), states=states, kind=kind))
            finally:
                game.close()
    # Initial, menu, movement and terminal images support browser pixel checks.
    for case in cases:
        for i, state in enumerate(case['states']):
            if i not in {0, 1, 2, len(case['states']) - 1}:
                state.pop('image', None)
    out = ROOT / '.build/native-traces.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(cases, separators=(',', ':')))
    print('Native traces:', len(cases), 'games,', sum(len(c['keys']) for c in cases), 'actions')


if __name__ == '__main__':
    main()
