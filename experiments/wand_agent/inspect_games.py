"""Play many evaluation games of one checkpoint on one level and write an HTML report.

    python -m wand_agent.inspect_games --model runs/best/latest.npz --level Pro --games 100 --out runs/viz/pro-games.html

The report has the event funnel (wand seen, picked up, minotaur seen, zap while
holding the wand, win), how games ended, a coverage map over all games (share of
games that visited each square, where the wand was first seen, where games ended)
and a table of games. A few lost games, one per funnel stage where possible, are
embedded with the full step-by-step viewer. Same seeds and greedy policy as evaluate.py.
"""
import argparse
import json
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .agents import load_agent
from .native import LEVELS
from .trajectory import STAGES, record_game, stage
from .viz_html import attach_tiles, page, write


def _play(job):
    model, level, seeds, sample, on_item = job
    torch.set_num_threads(1)
    agent = load_agent(model, seed=seeds[0], sample=sample)
    return [record_game(agent, level, seed, label=f'{level} {seed}', on_item=on_item) for seed in seeds]


def coverage(games):
    """Background terrain glyphs, per-square count of games that visited it, and marker positions."""
    from nle import nethack
    first = games[0]['map']
    width, height = first['width'], first['height']
    background = np.zeros(width * height, np.int64)
    known = np.zeros(width * height, bool)
    visits = np.zeros(width * height, np.int32)
    legend = {}
    for game in games:
        legend.update(game['map']['legend'])
        # The last screen of a game holds everything it has seen.
        final = np.array(game['map']['glyphs'], np.int64)
        for diff in game['map']['diffs']:
            final[diff[0::2]] = diff[1::2]
        # Terrain only (no player, monsters or objects); the first game that saw a square sets it.
        blank = np.array([legend[str(g)][0] in (0, 32) for g in final])
        terrain = ~known & ~blank & np.array([bool(nethack.glyph_is_cmap(int(g))) for g in final])
        background[terrain], known[terrain] = final[terrain], True
        for x, y in {(f['x'], f['y']) for f in game['frames']}:
            visits[y * width + x] += 1
    stone = next((int(g) for g, (c, _) in legend.items() if c == 32), 0)
    background[~known] = stone
    ends = [dict(x=g['summary']['end_at'][0], y=g['summary']['end_at'][1], end=g['summary']['end_status'],
                 seed=g['seed']) for g in games]
    wands = [dict(x=g['summary']['wand_at'][0], y=g['summary']['wand_at'][1], seed=g['seed'])
             for g in games if g['summary']['wand_at']]
    return dict(width=width, height=height, glyphs=background.tolist(), visits=visits.tolist(),
                games=len(games), ends=ends, wands=wands)


def pick_examples(games, count):
    """Lost games, one per (funnel stage, death or limit) in turn, lowest seeds first, then one win if any."""
    if count <= 0:
        return []
    groups = defaultdict(list)
    for game in games:
        groups[stage(game['summary']), game['summary']['end_status']].append(game)
    wins = [g for g in games if g['summary']['success']]
    chosen = []
    losses = count - (1 if wins and count > 1 else 0)
    queues = [groups[s, end] for s in STAGES for end in ('DEATH', 'ABORTED') if groups.get((s, end))]
    while len(chosen) < losses and any(queues):
        for queue in queues:
            if queue and len(chosen) < losses:
                chosen.append(queue.pop(0))
    if wins and len(chosen) < count:
        chosen.append(wins[0])
    return chosen


BODY = """<h1 id="title"></h1>
<p class="sub" id="subtitle"></p>
<div class="sprites" id="sprites"></div>
<div class="grid2">
  <div class="card"><h3>Lejek wydarzeń (odsetek wszystkich gier)</h3><div class="funnel" id="funnel"></div>
    <p class="note">„Zap z różdżką” = wybrana akcja zap, gdy różdżka jest w ekwipunku, w dowolnym kierunku.
    „Różdżka widziana” = znak różdżki na ekranie lub komunikat „You see here … wand”.</p></div>
  <div class="card"><h3>Jak kończą się gry</h3><div class="tiles" id="tiles"></div>
    <h3 style="margin-top:16px">Etap, na którym gra utknęła</h3><div class="funnel" id="stages"></div></div>
</div>
<h2>Pokrycie mapy we wszystkich grach</h2>
<div class="card">
  <div class="controls" style="margin-top:0">
    <label class="toggle"><input type="checkbox" id="l-visits" checked> odsetek gier, które odwiedziły pole</label>
    <label class="toggle"><input type="checkbox" id="l-wands" checked> różdżka zobaczona po raz pierwszy</label>
    <label class="toggle"><input type="checkbox" id="l-ends" checked> miejsce końca gry</label>
    <label class="toggle"><input type="checkbox" id="l-ascii"> ASCII</label>
  </div>
  <div class="legend" id="cov-legend"></div>
  <div class="mapbox" style="position:relative"><canvas id="cov"></canvas><div class="tip" id="cov-tip" style="display:none;position:absolute"></div></div>
  <p class="note">Tło to teren zobaczony w którejkolwiek grze. Na Easy i Medium układ zmienia się między seedami, więc mapa zbiorcza ma tam mniejszy sens.</p>
</div>
<h2>Przykładowe gry</h2>
<p class="sub">Przegrane z różnych etapów lejka (najniższe seedy), plus jedna wygrana, jeśli była.</p>
<div id="viewer"></div>
<h2>Wszystkie gry</h2>
<div class="controls"><label class="toggle">etap <select id="filter"><option value="">wszystkie</option></select></label></div>
<div class="card tablewrap"><table id="games"></table></div>"""

SCRIPT = r"""
const S = DATA.summaries, N = S.length;
document.getElementById('title').textContent = DATA.title;
document.getElementById('subtitle').textContent = DATA.subtitle;
spriteLegend(document.getElementById('sprites'));
function funnelRows(root, rows) {
  for (const [label, n, title] of rows) {
    el('span', { title: title || '' }, root, label);
    const b = el('div', { class: 'bar' }, root); el('i', { style: `width:${(100 * n / N).toFixed(1)}%` }, b);
    el('span', { class: 'v' }, root, `${n} (${Math.round(100 * n / N)}%)`);
  }
}
const count = f => S.filter(f).length;
funnelRows(document.getElementById('funnel'), [
  ['wszystkie gry', N], ['różdżka widziana', count(s => s.wand_seen)], ['różdżka podniesiona', count(s => s.picked)],
  ['minotaur widziany', count(s => s.minotaur_seen)], ['zap z różdżką', count(s => s.zapped)], ['wygrana', count(s => s.success)]]);
const tiles = document.getElementById('tiles');
for (const [status, label] of [['TASK_SUCCESSFUL', 'wygrane'], ['DEATH', 'śmierć'], ['ABORTED', 'limit akcji']]) {
  const t = el('div', { class: 'tile' }, tiles), n = count(s => s.end_status === status);
  el('div', { class: 'n' }, t, `${n}`); const l = el('div', { class: 'l' }, t); l.innerHTML = endBadge(status) + ` ${Math.round(100 * n / N)}% gier`;
}
const t2 = el('div', { class: 'tile' }, tiles);
el('div', { class: 'n' }, t2, (S.reduce((a, s) => a + s.visited, 0) / N).toFixed(0));
el('div', { class: 'l' }, t2, 'średnio odwiedzonych pól');
funnelRows(document.getElementById('stages'), DATA.stages.map(st => [st, count(s => s.stage === st)]).filter(r => r[1] > 0));

// Coverage map.
const C = DATA.coverage, cov = document.getElementById('cov');
const layers = { visits: true, wands: true, ends: true, ascii: !TILES };
document.getElementById('l-ascii').checked = layers.ascii;
const covBox = (() => {
  let x0 = C.width, x1 = 0, y0 = C.height, y1 = 0;
  for (let i = 0; i < C.width * C.height; i++) if (!isBlank(C.glyphs[i]) || C.visits[i] > 0) {
    const x = i % C.width, y = (i / C.width) | 0; x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
  }
  return [Math.max(0, x0 - 1), Math.max(0, y0 - 1), Math.min(C.width - 1, x1 + 1), Math.min(C.height - 1, y1 + 1)];
})();
const lg = document.getElementById('cov-legend');
lg.innerHTML = `<span><span class="sw" style="background:linear-gradient(90deg,rgba(57,135,229,.25),rgba(57,135,229,.85));width:40px"></span>odsetek gier na polu (0–100%)</span>
  <span><span class="sw" style="background:#ffd600"></span>różdżka zobaczona (kółko)</span>
  <span><span class="sw" style="background:#ff5a5a"></span>śmierć (×)</span><span><span class="sw" style="background:#fab219"></span>limit akcji (×)</span>
  <span><span class="sw" style="background:#3fd64f"></span>wygrana (×)</span>`;
let cell = 16;
function drawCoverage() {
  const [x0, y0, x1, y1] = covBox, cols = x1 - x0 + 1;
  cell = boardCell(cov.parentElement, cols);
  const w = cols * cell, h = (y1 - y0 + 1) * cell, cctx = setupCanvas(cov, w, h);
  const px = x => (x - x0) * cell, py = y => (y - y0) * cell;
  drawGlyphs(cctx, C.glyphs, C.width, covBox, cell, layers.ascii);
  if (layers.visits) for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
    const v = C.visits[y * C.width + x]; if (!v) continue;
    cctx.fillStyle = `rgba(57,135,229,${0.25 + 0.6 * v / C.games})`; cctx.fillRect(px(x), py(y), cell, cell);
  }
  if (layers.wands) { cctx.strokeStyle = '#ffd600'; cctx.lineWidth = 2;
    for (const p of C.wands) { cctx.beginPath(); cctx.arc(px(p.x) + cell / 2, py(p.y) + cell / 2, cell * 0.42, 0, 7); cctx.stroke(); } }
  if (layers.ends) for (const p of C.ends) {
    const X = px(p.x) + cell / 2, Y = py(p.y) + cell / 2, r = cell * 0.28;
    cctx.strokeStyle = p.end === 'DEATH' ? '#ff5a5a' : p.end === 'ABORTED' ? '#fab219' : '#3fd64f'; cctx.lineWidth = 2.5;
    cctx.beginPath(); cctx.moveTo(X - r, Y - r); cctx.lineTo(X + r, Y + r); cctx.moveTo(X + r, Y - r); cctx.lineTo(X - r, Y + r); cctx.stroke();
  }
}
if (TILES) TILES.whenReady(drawCoverage);
for (const k of Object.keys(layers)) document.getElementById('l-' + k).onchange = e => { layers[k] = e.target.checked; drawCoverage(); };
const ctip = document.getElementById('cov-tip');
cov.addEventListener('mousemove', ev => {
  const r = cov.getBoundingClientRect(), x = covBox[0] + Math.floor((ev.clientX - r.left) / cell), y = covBox[1] + Math.floor((ev.clientY - r.top) / cell);
  if (x < 0 || y < 0 || x >= C.width || y >= C.height) { ctip.style.display = 'none'; return; }
  const v = C.visits[y * C.width + x], ends = C.ends.filter(p => p.x === x && p.y === y), wands = C.wands.filter(p => p.x === x && p.y === y);
  ctip.innerHTML = `pole (${x}, ${y})<br>odwiedzone w ${v} z ${C.games} gier (${Math.round(100 * v / C.games)}%)` +
    (wands.length ? `<br>różdżka zobaczona tu w ${wands.length} grach` : '') + (ends.length ? `<br>koniec tu w ${ends.length} grach` : '');
  ctip.style.display = 'block';
  const pr = cov.parentElement.getBoundingClientRect();
  ctip.style.left = Math.min(ev.clientX - pr.left + 14, pr.width - ctip.offsetWidth - 4) + 'px'; ctip.style.top = (ev.clientY - pr.top + 14) + 'px';
});
cov.addEventListener('mouseleave', () => { ctip.style.display = 'none'; });
new ResizeObserver(drawCoverage).observe(cov.parentElement);

// Example games and the table of all games.
const examples = DATA.examples, viewerRoot = document.getElementById('viewer');
const viewer = examples.length ? Viewer(viewerRoot, examples) : (viewerRoot.innerHTML = '<div class="empty">Brak przykładów (--examples 0)</div>', null);
const exampleIndex = new Map(examples.map((g, i) => [g.seed, i]));
const filter = document.getElementById('filter');
for (const st of DATA.stages) if (S.some(s => s.stage === st)) el('option', { value: st }, filter, st);
function renderTable() {
  const table = document.getElementById('games');
  table.innerHTML = '<tr><th class="num">seed</th><th>wynik</th><th>etap</th><th class="num">akcje</th><th class="num">zwrot</th><th class="num">odwiedzone pola</th><th>podgląd</th></tr>';
  for (const s of S) {
    if (filter.value && s.stage !== filter.value) continue;
    const tr = el('tr', {}, table);
    tr.innerHTML = `<td class="num">${s.seed}</td><td>${endBadge(s.end_status)}</td><td>${s.stage}</td><td class="num">${s.steps}</td>
      <td class="num">${s.native_return.toFixed(2)}</td><td class="num">${s.visited}</td>`;
    const td = el('td', {}, tr);
    if (exampleIndex.has(s.seed)) {
      const b = el('button', {}, td, 'obejrzyj');
      b.onclick = () => { viewer.select(exampleIndex.get(s.seed)); viewerRoot.scrollIntoView({ behavior: 'smooth' }); };
    } else {
      const cmd = `python -m wand_agent.replay --model ${DATA.model} --level ${DATA.level} --seed ${s.seed}${DATA.sample ? ' --sample' : ''} --out runs/viz/${DATA.level.toLowerCase()}-${s.seed}.html`;
      const c = el('code', { title: 'Kliknij, aby skopiować komendę', style: 'cursor:copy;color:var(--ink-2)' }, td, 'komenda replay');
      c.onclick = () => { navigator.clipboard && navigator.clipboard.writeText(cmd); c.textContent = 'skopiowano'; setTimeout(() => c.textContent = 'komenda replay', 1200); };
    }
  }
}
filter.onchange = renderTable;
renderTable();
"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True, help="Checkpoint path or 'random'")
    p.add_argument('--level', choices=LEVELS, required=True)
    p.add_argument('--games', type=int, default=100)
    p.add_argument('--seed-start', type=int, default=100_000_000)
    p.add_argument('--sample', action='store_true', help='PPO: draw actions from the policy (generator seeded per game)')
    p.add_argument('--workers', type=int, default=2, help='Processes; keep low while training runs')
    p.add_argument('--examples', type=int, default=6, help='Games embedded with the step-by-step viewer')
    p.add_argument('--on-item', action='store_true',
                   help='Give the policy the extra "object under the player" input (found from the checkpoint by default)')
    p.add_argument('--out', type=Path, required=True, help='HTML report to write')
    p.add_argument('--json', type=Path, help='Also write per-game summaries as JSON')
    a = p.parse_args()
    if a.games <= 0 or a.workers <= 0 or not 0 <= a.seed_start < 2**31 - a.games:
        p.error('Use positive games and workers and a valid seed range')
    seeds = list(range(a.seed_start, a.seed_start + a.games))
    chunks = [seeds[i::a.workers] for i in range(a.workers) if seeds[i::a.workers]]
    jobs = [(a.model, a.level, chunk, a.sample, a.on_item) for chunk in chunks]
    with mp.get_context('spawn').Pool(len(jobs)) as pool:
        games = sorted((g for part in pool.map(_play, jobs) for g in part), key=lambda g: g['seed'])
    summaries = []
    for game in games:
        s = dict(seed=game['seed'], stage=stage(game['summary']), **game['summary'])
        summaries.append(s)
    funnel = {key: sum(s[key] for s in summaries) for key in ('wand_seen', 'picked', 'minotaur_seen', 'zapped', 'success')}
    ends = {status: sum(s['end_status'] == status for s in summaries) for status in ('TASK_SUCCESSFUL', 'DEATH', 'ABORTED')}
    print(json.dumps(dict(level=a.level, games=len(games), **funnel, **ends,
                          visited_mean=round(float(np.mean([s['visited'] for s in summaries])), 1))), flush=True)
    examples = pick_examples(games, a.examples)
    for game in examples:
        game['label'] = f"seed {game['seed']}: {stage(game['summary'])}"
    policy = 'losowanie z rozkładu (generator seedowany seedem gry)' if a.sample else 'greedy'
    data = dict(title=f'{a.level}: {len(games)} gier', model=a.model, level=a.level, sample=a.sample,
                subtitle=f'{a.model} · seedy {seeds[0]}–{seeds[-1]} · {policy}',
                summaries=summaries, stages=list(STAGES) + ['wygrana'], coverage=coverage(games), examples=examples)
    attach_tiles(data, games)
    if a.json:
        write(a.json, json.dumps(dict(model=a.model, level=a.level, policy='sample' if a.sample else 'greedy',
                                      funnel=funnel, ends=ends, games=summaries), indent=1) + '\n')
    print(write(a.out, page(f'Gry {a.level}', BODY, data, SCRIPT)))


if __name__ == '__main__':
    main()
