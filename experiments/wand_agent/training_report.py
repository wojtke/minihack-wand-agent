"""HTML report of how one or more PPO runs trained, from their episodes.jsonl and metrics.jsonl.

    python -m wand_agent.training_report runs/combo-macro-s1 runs/combo-macro-s2 runs/combo-macro-sil-s1 --out runs/viz/train.html

Per level: win rate over the last --window finished training games, and how
training games ended (win, death, action limit). PPO statistics per update:
entropy, explained variance, KL, clip fraction, value loss, RND bonus. Runs whose
names differ only in a final -sN are seeds of one configuration; the report can
show their mean and range. The table lists the step at which each level first
reached a threshold and, if a sibling <run>-eval.json exists, its evaluation wins.
Training games use training seeds and the training policy (sampled actions), so
these rates are not evaluation results.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

from .native import LEVELS
from .viz_html import page, write

METRICS = (('entropy', 'Entropia polityki'), ('explained_variance', 'Explained variance krytyka'),
           ('approx_kl', 'Przybliżona KL'), ('clipfrac', 'Clip fraction'), ('value_loss', 'Strata krytyka'),
           ('bonus_mean', 'Bonus RND na krok (po normalizacji)'), ('rollout_reward', 'Nagroda treningowa w batchu'),
           ('sil_positive', 'SIL: odsetek próbek z R > V'), ('sps', 'Kroki na sekundę'))
THRESHOLDS = dict(Easy=0.9, Medium=0.9, Hard=0.5, Pro=0.1)
IGNORED_CONFIG = {'seed', 'iterations', 'batch_size', 'log_every', 'save_every'}


def _read_jsonl(path):
    if not path.exists():
        return []
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _rolling(steps, flags, edges, window):
    """Mean of the last `window` games finished by each edge; None before the first game."""
    cumulative = np.concatenate([[0], np.cumsum(flags)])
    out = []
    for edge in edges:
        i = int(np.searchsorted(steps, edge, side='right'))
        n = min(window, i)
        out.append(None if n == 0 else round(float(cumulative[i] - cumulative[i - n]) / n, 4))
    return out


def _first_reach(steps, flags, window, threshold):
    """Training step at which the rolling rate over `window` games first reaches threshold."""
    if len(flags) < window:
        return None
    cumulative = np.concatenate([[0], np.cumsum(flags)])
    rates = (cumulative[window:] - cumulative[:-window]) / window
    hit = np.flatnonzero(rates >= threshold)
    return int(steps[hit[0] + window - 1]) if hit.size else None


def _binned(xs, ys, edges):
    """Mean of ys within each bin ending at an edge; None for empty bins."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    index = np.searchsorted(edges, xs, side='left')
    out = []
    for b in range(len(edges)):
        values = ys[(index == b) & np.isfinite(ys)]
        out.append(round(float(values.mean()), 6) if values.size else None)
    return out


def load_run(path, edges, window):
    path = Path(path)
    config = json.loads((path / 'config.json').read_text()) if (path / 'config.json').exists() else {}
    summary = json.loads((path / 'summary.json').read_text()) if (path / 'summary.json').exists() else {}
    episodes = _read_jsonl(path / 'episodes.jsonl')
    metrics = _read_jsonl(path / 'metrics.jsonl')
    last = max([e['global_step'] for e in episodes[-1:]] + [m['global_step'] for m in metrics[-1:]] + [0])
    levels, reach, final = {}, {}, {}
    for level in LEVELS:
        games = [e for e in episodes if e.get('level') == level]
        if not games:
            continue
        steps = np.array([g['global_step'] for g in games])
        live = [edge if edge <= last else None for edge in edges]
        rows = {}
        for key, flags in (('win', [g['success'] for g in games]),
                           ('death', [g['end_status'] == 'DEATH' for g in games]),
                           ('limit', [g['end_status'] == 'ABORTED' for g in games])):
            values = _rolling(steps, np.asarray(flags, float), edges, window)
            rows[key] = [v if e is not None else None for v, e in zip(values, live)]
        lengths = np.asarray([g['steps'] for g in games], float)
        rows['length'] = [v if e is not None else None
                          for v, e in zip(_rolling(steps, lengths, edges, window), live)]
        levels[level] = rows
        wins = np.asarray([g['success'] for g in games], float)
        reach[level] = _first_reach(steps, wins, window, THRESHOLDS[level])
        final[level] = dict(games=len(games), win=round(float(wins[-window:].mean()), 4))
    series = {}
    for key, _ in METRICS:
        points = [(m['global_step'], m[key]) for m in metrics if isinstance(m.get(key), (int, float))]
        if points:
            xs, ys = zip(*points)
            series[key] = _binned(xs, ys, edges)
    evaluations = {}
    for suffix, name in (('-eval.json', 'greedy'), ('-eval-sample.json', 'sample')):
        file = path.parent / (path.name + suffix)
        if file.exists():
            report = json.loads(file.read_text())
            evaluations[name] = {level: [r['wins'], r['games']] for level, r in report.get('levels', {}).items()}
    return dict(name=path.name, group=re.sub(r'-s\d+$', '', path.name), config=config,
                status=summary.get('status', 'w trakcie' if not summary else '?'), steps=last,
                games=len(episodes), levels=levels, metrics=series, reach=reach, final=final, eval=evaluations)


def config_differences(runs):
    keys = sorted({k for run in runs for k in run['config']} - IGNORED_CONFIG)
    return [k for k in keys if len({json.dumps(run['config'].get(k)) for run in runs}) > 1]


BODY = """<h1>Przebieg treningu</h1>
<p class="sub" id="subtitle"></p>
<div class="controls" id="modebar" style="margin-top:0">
  <label class="toggle">pokaż <select id="mode"><option value="runs">każdy run osobno</option><option value="groups">średnia seedów i zakres min–max</option></select></label>
</div>
<p class="sub" id="note" style="display:none"></p>
<h2>Odsetek wygranych w grach treningowych</h2>
<div class="grid2" id="win"></div>
<h2>Jak kończą się gry treningowe</h2>
<div class="controls" style="margin-top:0"><label class="toggle">run <select id="endrun"></select></label></div>
<div class="grid2" id="ends"></div>
<h2>Średnia długość gry (akcje natywne)</h2>
<div class="grid2" id="length"></div>
<h2>Statystyki PPO</h2>
<div class="grid3" id="metrics"></div>
<h2>Podsumowanie runów</h2>
<div class="card tablewrap"><table id="table"></table></div>
<p class="sub" style="margin-top:8px;font-size:12px" id="tablenote"></p>"""

SCRIPT = r"""
const runs = DATA.runs, edges = DATA.edges, LV = DATA.levels;
document.getElementById('subtitle').textContent =
  `${runs.length} run(ów) · okno ${DATA.window} ostatnich gier na poziom · gry treningowe (seedy treningowe, akcje losowane z polityki)`;
const groups = [...new Set(runs.map(r => r.group))];
const DASH = [null, '6 3', '2 3', '8 3 2 3', '12 4', '4 2 1 2', '1 2', '10 2 2 2 2 2'];
const seedIndex = new Map();
for (const g of groups) runs.filter(r => r.group === g).forEach((r, i) => seedIndex.set(r.name, i));
const colorOf = g => css(SERIES[groups.indexOf(g) % SERIES.length]);
const multiSeed = groups.some(g => runs.filter(r => r.group === g).length > 1);
const modeSel = document.getElementById('mode');
if (!multiSeed) document.getElementById('modebar').style.display = 'none';
else modeSel.value = 'groups';
if (groups.length > 8) {
  const n = document.getElementById('note'); n.style.display = '';
  n.textContent = `Uwaga: ${groups.length} konfiguracji, a paleta ma 8 kolorów; kolory się powtarzają. Pokaż mniej runów naraz.`;
}
const xFmt = v => fmtNum(v);
const pts = vals => edges.map((x, i) => [x, vals ? vals[i] : null]);

function seriesFor(get) {
  if (modeSel.value === 'groups') {
    return groups.map(g => {
      const members = runs.filter(r => r.group === g).map(get).filter(Boolean);
      const mean = edges.map((_, i) => { const v = members.map(m => m[i]).filter(x => x !== null && x !== undefined); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null; });
      const band = members.length > 1 ? edges.map((x, i) => { const v = members.map(m => m[i]).filter(x => x !== null && x !== undefined); return v.length ? [x, Math.min(...v), Math.max(...v)] : [x, null, null]; }) : [];
      return { name: members.length > 1 ? `${g} (${members.length} seedy)` : g, color: colorOf(g), points: pts(mean), band };
    }).filter(s => s.points.some(p => p[1] !== null));
  }
  return runs.map(r => ({ name: r.name, color: colorOf(r.group), dash: DASH[seedIndex.get(r.name) % DASH.length], width: 1.75, points: pts(get(r)) }))
    .filter(s => s.points.some(p => p[1] !== null));
}
function card(parent, title) { const c = el('div', { class: 'card' }, parent); el('h3', {}, c, title); return el('div', {}, c); }

function render() {
  const win = document.getElementById('win'); win.innerHTML = '';
  const len = document.getElementById('length'); len.innerHTML = '';
  for (const level of LV) {
    lineChart(card(win, level), { series: seriesFor(r => r.levels[level] && r.levels[level].win), yMin: 0, yMax: 1, yFmt: fmtPct, xFmt, height: 200, xLabel: 'krok treningu',
      tipTitle: x => `krok ${fmtNum(x)}` });
    lineChart(card(len, level), { series: seriesFor(r => r.levels[level] && r.levels[level].length), yMin: 0, xFmt, height: 160, xLabel: 'krok treningu',
      tipTitle: x => `krok ${fmtNum(x)}` });
  }
  const met = document.getElementById('metrics'); met.innerHTML = '';
  for (const [key, title] of DATA.metric_names) {
    const s = seriesFor(r => r.metrics[key]);
    if (!s.length) continue;
    lineChart(card(met, title), { series: s, xFmt, height: 160, xLabel: 'krok treningu', tipTitle: x => `krok ${fmtNum(x)}`,
      yMin: key === 'explained_variance' ? undefined : 0, legend: s.length > 1 });
  }
  renderEnds();
}
const endSel = document.getElementById('endrun');
runs.forEach((r, i) => el('option', { value: i }, endSel, r.name));
function renderEnds() {
  const root = document.getElementById('ends'); root.innerHTML = '';
  const r = runs[+endSel.value || 0];
  for (const level of LV) {
    const L = r.levels[level];
    if (!L) continue;
    lineChart(card(root, `${level}: ${r.name}`), { series: [
      { name: 'wygrana', color: css('--good'), points: pts(L.win) },
      { name: 'śmierć', color: css('--critical'), points: pts(L.death) },
      { name: 'limit akcji', color: css('--warning'), points: pts(L.limit) }],
      yMin: 0, yMax: 1, yFmt: fmtPct, xFmt, height: 170, xLabel: 'krok treningu', tipTitle: x => `krok ${fmtNum(x)}` });
  }
}
endSel.onchange = renderEnds;
modeSel.onchange = render;
render();

// Summary table.
const table = document.getElementById('table');
const diff = DATA.config_diff;
let head = '<tr><th>run</th><th>status</th><th class="num">kroki</th><th class="num">gry</th>';
for (const k of diff) head += `<th>${k}</th>`;
for (const l of LV) head += `<th class="num">${l}: wygrane (ostatnie ${DATA.window})</th>`;
for (const l of LV) head += `<th class="num">${l} ≥ ${Math.round(100 * DATA.thresholds[l])}% od kroku</th>`;
head += '<th>ewaluacja greedy</th><th>ewaluacja sample</th></tr>';
table.innerHTML = head;
for (const r of runs) {
  const tr = el('tr', {}, table);
  let row = `<td><span class="sw" style="display:inline-block;width:10px;height:3px;border-radius:2px;margin-right:6px;vertical-align:middle;background:${colorOf(r.group)}"></span>${r.name}</td>
    <td>${r.status}</td><td class="num">${fmtNum(r.steps)}</td><td class="num">${fmtNum(r.games)}</td>`;
  for (const k of diff) row += `<td class="mono">${JSON.stringify(r.config[k] ?? null)}</td>`;
  for (const l of LV) row += `<td class="num">${r.final[l] ? fmtPct(r.final[l].win) : '–'}</td>`;
  for (const l of LV) row += `<td class="num">${r.reach[l] ? fmtNum(r.reach[l]) : '–'}</td>`;
  for (const kind of ['greedy', 'sample']) {
    const e = r.eval[kind];
    row += `<td class="mono">${e ? LV.map(l => e[l] ? e[l][0] : '–').join(' / ') + (e[LV[0]] ? ` z ${e[LV[0]][1]}` : '') : '–'}</td>`;
  }
  tr.innerHTML = row;
}
document.getElementById('tablenote').textContent = `Kolumny konfiguracji pokazują tylko ustawienia, które różnią się między runami. Ewaluacja: wygrane ${LV.join(' / ')} z pliku <run>-eval.json obok katalogu runu.`;
"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('runs', nargs='+', type=Path, help='Run directories (with episodes.jsonl and metrics.jsonl)')
    p.add_argument('--out', type=Path, required=True, help='HTML report to write')
    p.add_argument('--window', type=int, default=100, help='Games per level in the rolling win rate')
    p.add_argument('--bins', type=int, default=300, help='Points per curve')
    a = p.parse_args()
    if a.window <= 0 or a.bins <= 1:
        p.error('Use a positive window and at least 2 bins')
    missing = [str(r) for r in a.runs if not (r / 'episodes.jsonl').exists()]
    if missing:
        p.error(f'No episodes.jsonl in: {", ".join(missing)}')
    last = 0
    for run in a.runs:
        with (run / 'episodes.jsonl').open('rb') as stream:
            lines = stream.read().splitlines()
        if lines:
            last = max(last, json.loads(lines[-1])['global_step'])
    edges = np.linspace(last / a.bins, last, a.bins).round().astype(int).tolist() if last else [0, 1]
    runs = []
    for run in a.runs:
        runs.append(load_run(run, edges, a.window))
        print(f"{run.name}: {runs[-1]['games']} games, {runs[-1]['steps']} steps", flush=True)
    levels = [level for level in LEVELS if any(level in r['levels'] for r in runs)]
    data = dict(runs=runs, edges=edges, levels=levels, window=a.window, thresholds=THRESHOLDS,
                metric_names=METRICS, config_diff=config_differences(runs))
    print(write(a.out, page('Przebieg treningu', BODY, data, SCRIPT)))


if __name__ == '__main__':
    main()
