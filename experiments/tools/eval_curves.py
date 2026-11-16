"""Learning curves from checkpoint evaluations (runs/training/<stage>/<run>/checkpoint-evals/*.json + eval.json).

Writes one static HTML page: win rate per level vs training steps, mean over seeds with a min-max band,
per-seed lines for Pro, the random agent as a dashed line and a table of final results.
    python tools/eval_curves.py [--out runs/viz/agregaty/krzywe-ewaluacji.html]
"""
import argparse
import json
import math
import pathlib
import re
import statistics

LEVELS = ["Easy", "Medium", "Hard", "Pro"]
GROUPS = [  # (label, colour, runs, steps of the final checkpoint)
    ("Adaptacyjny mix, MLP", "#2a78d6", [f"adaptive-30m-s{i}" for i in range(1, 6)], 30),
    ("Stały mix, MLP", "#eb6834", [f"fixedmix-30m-s{i}" for i in range(1, 6)], 30),
    ("Adaptacyjny mix, LSTM 128", "#1baf7a", [f"memory-lstm128-20m-s{i}" for i in range(1, 6)], 20),
]


def load_levels(path):
    d = json.loads(path.read_text())
    return {lv: d["levels"][lv]["wins"] / d["levels"][lv]["games"] for lv in LEVELS}


def run_dir(run):
    found = list(pathlib.Path("runs/training").glob(f"*/{run}"))
    return found[0] if found else None


def load_run(run, final_m):
    """{million steps: {level: win rate}} for one run, or None if the run does not exist."""
    pts = {}
    d = run_dir(run)
    if d is None:
        return None
    for p in (d / "checkpoint-evals").glob("step-*.json"):
        if '"Pro"' not in p.read_text():  # evaluation still running
            continue
        m = round(int(re.search(r"step-(\d+)", p.name).group(1)) / 1e6)
        pts[m] = load_levels(p)
    final = d / "eval.json"
    if final.exists():
        pts[final_m] = load_levels(final)
    return pts or None


def path_d(xy):
    return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in xy)


def panel(title, series, random_rate, xmax, per_seed=False, w=640, h=300):
    """series: [(label, colour, {m: [rates over seeds]}, {seed: {m: rate}})]."""
    l, r, t, b = 46, 12, 12, 34
    sx = lambda m: l + (w - l - r) * m / xmax
    sy = lambda v: t + (h - t - b) * (1 - v)
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="{title}">']
    for v in (0, .25, .5, .75, 1):
        out.append(f'<line x1="{l}" x2="{w-r}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="var(--grid)"/>'
                   f'<text class="ax" x="{l-6}" y="{sy(v)+4:.1f}" text-anchor="end">{int(v*100)}%</text>')
    for m in range(0, xmax + 1, 5):
        out.append(f'<text class="ax" x="{sx(m):.1f}" y="{h-b+16}" text-anchor="middle">{m}</text>')
    out.append(f'<text class="ax" x="{(l+w-r)/2:.1f}" y="{h-4}" text-anchor="middle">kroki treningu [mln]</text>')
    out.append(f'<line x1="{l}" x2="{w-r}" y1="{sy(0):.1f}" y2="{sy(0):.1f}" stroke="var(--axis)"/>')
    if random_rate is not None:
        out.append(f'<line x1="{l}" x2="{w-r}" y1="{sy(random_rate):.1f}" y2="{sy(random_rate):.1f}" '
                   f'stroke="#5c6168" stroke-dasharray="5 4"><title>agent losowy: {random_rate:.1%}</title></line>')
    for label, colour, by_m, by_seed in series:
        ms = sorted(by_m)
        if per_seed:
            for seed, pts in by_seed.items():
                xy = [(sx(m), sy(pts[m])) for m in sorted(pts)]
                out.append(f'<path d="{path_d(xy)}" fill="none" stroke="{colour}" stroke-width="1.2" opacity=".45">'
                           f'<title>{label}, seed {seed}</title></path>')
        else:
            lo = [(sx(m), sy(min(by_m[m]))) for m in ms]
            hi = [(sx(m), sy(max(by_m[m]))) for m in reversed(ms)]
            out.append(f'<path d="{path_d(lo + hi)} Z" fill="{colour}" opacity=".15"/>')
        mean = [(sx(m), sy(statistics.mean(by_m[m]))) for m in ms]
        out.append(f'<path d="{path_d(mean)}" fill="none" stroke="{colour}" stroke-width="2.4"/>')
        for m in ms:
            v = by_m[m]
            out.append(f'<circle cx="{sx(m):.1f}" cy="{sy(statistics.mean(v)):.1f}" r="6" fill="transparent">'
                       f'<title>{label}, {m} mln: średnio {statistics.mean(v):.1%} '
                       f'(min {min(v):.0%}, max {max(v):.0%}, seedy: {len(v)})</title></circle>')
    out.append("</svg>")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/viz/agregaty/krzywe-ewaluacji.html")
    args = ap.parse_args()
    rnd = load_levels(pathlib.Path("runs/eval/random-agent.json"))
    data = []  # (label, colour, {seed: {m: {level: rate}}})
    for label, colour, runs, final_m in GROUPS:
        seeds = {}
        for run in runs:
            pts = load_run(run, final_m)
            if pts:
                seeds[run.rsplit("-s", 1)[1]] = pts
        data.append((label, colour, seeds))
    xmax = 30

    def series(level):
        res = []
        for label, colour, seeds in data:
            by_m = {}
            for pts in seeds.values():
                for m, rates in pts.items():
                    by_m.setdefault(m, []).append(rates[level])
            by_m = {m: v for m, v in by_m.items() if len(v) == len(seeds)}  # only steps every seed reached
            by_seed = {s: {m: r[level] for m, r in pts.items()} for s, pts in seeds.items()}
            res.append((f"{label} ({len(seeds)} seedów)", colour, by_m, by_seed))
        return res

    legend = "".join(f'<span><i class="sw" style="background:{c}"></i>{lab} — seedy {", ".join(sorted(s))}</span>'
                     for lab, c, s in data)
    legend += '<span><i class="sw dash"></i>agent losowy</span>'
    panels = "".join(f'<div class="card"><h3>{lv}</h3>{panel(lv, series(lv), rnd[lv], xmax)}</div>' for lv in LEVELS)
    pro_seeds = panel("Pro, seedy", series("Pro"), rnd["Pro"], xmax, per_seed=True, h=360)

    rows = []
    for label, colour, runs, final_m in GROUPS:
        for lv in ("Hard", "Pro"):
            vals = []
            for run in runs:
                d = run_dir(run)
                f = d / "eval.json" if d else None
                if f and f.exists():
                    vals.append(json.loads(f.read_text())["levels"][lv]["wins"])
            if not vals:
                continue
            sd = statistics.stdev(vals) if len(vals) > 1 else 0
            ci = 1.96 * sd / math.sqrt(len(vals)) if len(vals) > 1 else 0
            rows.append(f"<tr><td>{label}</td><td>{final_m} mln</td><td>{lv}</td><td class=num>{len(vals)}</td>"
                        f"<td class=num>{statistics.mean(vals):.1f}</td><td class=num>{sd:.1f}</td>"
                        f"<td class=num>±{ci:.1f}</td><td>{', '.join(map(str, vals))}</td></tr>")

    html = f"""<!doctype html>
<html lang="pl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Krzywe uczenia</title>
<style>
:root {{ color-scheme: light; --page:#fff; --ink:#202124; --muted:#5c6168; --line:#d9dcdf; --grid:#eceef0; --axis:#c3c6ca; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:var(--page); color:var(--ink); font:15px/1.45 Arial, Helvetica, sans-serif; }}
main {{ max-width:1440px; margin:0 auto; padding:32px 5vw 72px; }}
@media (max-width:760px) {{ main {{ padding:24px 16px 60px; }} }}
h1 {{ font-size:36px; line-height:1.2; font-weight:600; letter-spacing:-.025em; margin:0; }}
h2 {{ font-size:24px; font-weight:600; margin:44px 0 16px; }}
h3 {{ font-size:15px; font-weight:600; margin:0 0 8px; }}
.sub {{ font-size:17px; color:var(--muted); margin:10px 0 22px; }}
.note {{ font-size:13px; color:var(--muted); margin:10px 0 0; }}
.card {{ border-top:1px solid var(--line); padding-top:12px; min-width:0; }}
.grid2 {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(420px, 1fr)); gap:22px 38px; }}
@media (max-width:520px) {{ .grid2 {{ grid-template-columns:1fr; }} }}
.legend {{ display:flex; flex-wrap:wrap; gap:4px 18px; font-size:13px; color:var(--muted); margin:0 0 14px; }}
.sw {{ display:inline-block; width:14px; height:3px; margin-right:6px; vertical-align:middle; }}
.sw.dash {{ background:repeating-linear-gradient(90deg, #5c6168 0 4px, transparent 4px 7px); }}
.ax {{ fill:var(--muted); font:11px Arial, Helvetica, sans-serif; }}
table {{ border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }}
th, td {{ text-align:left; padding:7px 18px 7px 0; border-bottom:1px solid var(--line); white-space:nowrap; font-size:14px; }}
td.num, th.num {{ text-align:right; }}
.tablewrap {{ overflow-x:auto; }}
</style></head><body><main>
<h1>Krzywe uczenia</h1>
<p class="sub">Odsetek wygranych w ewaluacji checkpointów co 1 mln kroków: polityka greedy, 200 gier na poziom, seedy od 100&nbsp;000&nbsp;000 (spoza treningu), tylko natywna nagroda.</p>
<div class="legend">{legend}</div>
<div class="grid2">{panels}</div>
<p class="note">Linia: średnia z seedów. Pas: zakres min–max między seedami. Ostatni punkt to ewaluacja końcowa modelu. Learning rate maleje liniowo do zera na końcu każdego runu, więc punkt 20 mln w runie 30 mln nie odpowiada końcowi runu 20 mln. Hard ma sufit ok. 85% przez błąd środowiska (różdżka w ścianie).</p>
<h2>Pro: pojedyncze seedy</h2>
<div class="legend">{legend}</div>
<div class="card">{pro_seeds}</div>
<h2>Wynik końcowy (wygrane na 200 gier)</h2>
<div class="tablewrap"><table><tr><th>Konfiguracja</th><th>Kroki</th><th>Poziom</th><th class=num>Seedy</th><th class=num>Średnia</th><th class=num>Odch. std.</th><th class=num>95% CI średniej</th><th>Poszczególne seedy</th></tr>
{"".join(rows)}</table></div>
<p class="note">Easy i Medium: 200/200 we wszystkich runach. Agent losowy (1000 gier na poziom): Easy {rnd["Easy"]:.1%}, pozostałe 0%. CI średniej z rozkładu normalnego (1,96·sd/√n); przy 2 seedach jest tylko orientacyjny.</p>
</main></body></html>"""
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(out)


if __name__ == "__main__":
    main()
