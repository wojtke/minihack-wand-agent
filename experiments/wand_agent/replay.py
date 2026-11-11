"""Record one game of one or more checkpoints and write a self-contained HTML viewer.

    python -m wand_agent.replay --model runs/best/latest.npz --level Pro --seed 100000003 --out runs/viz/pro.html

Several --model paths (e.g. runs/best/step-*.npz) play the same level and seed,
and the viewer switches between them, which shows how the agent changed in training.
Greedy games are the same games evaluate.py plays for that seed. With --sample the
action generator is seeded by the game seed, so a sampled game is repeatable on its
own, but it is not the game evaluate.py --sample played (that one shares one
generator across all seeds of a level).
"""
import argparse
import json
from pathlib import Path

import torch

from .agents import load_agent
from .native import LEVELS
from .trajectory import checkpoint_label, record_game
from .viz_html import attach_tiles, page, write

BODY = """<h1 id="title"></h1>
<p class="sub" id="subtitle"></p>
<div class="sprites" id="sprites"></div>
<div id="viewer"></div>
<h2>Wszystkie zapisane gry</h2>
<div class="card tablewrap"><table id="games"></table></div>"""

SCRIPT = r"""
const games = DATA.games;
document.getElementById('title').textContent = DATA.title;
document.getElementById('subtitle').textContent = DATA.subtitle;
spriteLegend(document.getElementById('sprites'));
const viewer = Viewer(document.getElementById('viewer'), games);
const table = document.getElementById('games');
table.innerHTML = '<tr><th>checkpoint</th><th>wynik</th><th class="num">akcje</th><th class="num">zwrot</th><th class="num">odwiedzone pola</th><th>różdżka widziana / podniesiona</th><th>minotaur</th><th>zap z różdżką</th><th></th></tr>';
const yes = v => v ? 'tak' : '–';
games.forEach((g, i) => {
  const s = g.summary, tr = el('tr', {}, table);
  tr.innerHTML = `<td>${g.label}</td><td>${endBadge(s.end_status)}</td><td class="num">${s.steps}</td>
    <td class="num">${s.native_return.toFixed(2)}</td><td class="num">${s.visited}</td>
    <td>${yes(s.wand_seen)} / ${yes(s.picked)}</td><td>${yes(s.minotaur_seen)}</td><td>${yes(s.zapped)}</td>`;
  const b = el('button', {}, el('td', {}, tr), 'pokaż');
  b.onclick = () => { viewer.select(i); window.scrollTo({ top: 0, behavior: 'smooth' }); };
});
"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', nargs='+', required=True, help="Checkpoint path(s) or 'random'")
    p.add_argument('--level', choices=LEVELS, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--sample', action='store_true', help='PPO: draw actions from the policy instead of the most likely one')
    p.add_argument('--on-item', action='store_true',
                   help='Give the policy the extra "object under the player" input (found from the checkpoint by default)')
    p.add_argument('--out', type=Path, required=True, help='HTML viewer to write')
    p.add_argument('--json', type=Path, help='Also write the recorded games as JSON')
    a = p.parse_args()
    if not 0 <= a.seed < 2**31:
        p.error('Invalid seed')
    torch.set_num_threads(1)
    games = []
    for model in a.model:
        agent = load_agent(model, seed=a.seed, sample=a.sample)
        game = record_game(agent, a.level, a.seed, label=checkpoint_label(model) if model != 'random' else 'random',
                           on_item=a.on_item)
        game['model'] = model
        games.append(game)
        s = game['summary']
        print(f"{game['label']}: {s['end_status']} after {s['steps']} actions, return {s['native_return']:.2f}, "
              f"visited {s['visited']} squares", flush=True)
    policy = 'losowanie z rozkładu' if a.sample else 'greedy'
    title = f'{a.level}, seed {a.seed}'
    data = attach_tiles(dict(title=title, subtitle=f"{', '.join(g['label'] for g in games)} · {policy}", games=games), games)
    if a.json:
        write(a.json, json.dumps(games, ensure_ascii=False) + '\n')
    print(write(a.out, page(f'Gra {title}', BODY, data, SCRIPT)))


if __name__ == '__main__':
    main()
