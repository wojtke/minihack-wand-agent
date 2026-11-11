"""Self-contained HTML pages for replay, inspect_games and training_report.

Each page is one file: styles, scripts and data inline, no server and no network.
Charts are plain SVG; the game map is a canvas.
"""
import html
import json

CSS = r"""
/* Look of the project website (website/style.css): white page, Arial, thin rules, black game board. */
:root {
  color-scheme: light;
  --page: #fff; --surface: #fff; --panel: #f4f4f4; --ink: #202124; --ink-2: #5c6168; --muted: #5c6168;
  --line: #d9dcdf; --grid: #eceef0; --axis: #c3c6ca; --border: #d9dcdf; --wash: #f1f2f3; --accent: #315b78;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --s5: #e87ba4; --s6: #008300; --s7: #6250d6; --s8: #e34948;
  --good: #0ca30c; --warning: #fab219; --critical: #d03b3b;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink); font: 15px/1.45 Arial, Helvetica, sans-serif; font-synthesis: none; }
main { max-width: 1440px; margin: 0 auto; padding: 32px 5vw 72px; }
@media (max-width: 760px) { main { padding: 24px 16px 60px; } }
h1 { font-size: 36px; line-height: 1.2; font-weight: 600; letter-spacing: -.025em; margin: 0; }
h2 { font-size: 24px; line-height: 1.25; font-weight: 600; letter-spacing: -.015em; margin: 44px 0 16px; }
h3 { font-size: 15px; font-weight: 600; margin: 0 0 8px; }
.sub { font-size: 17px; color: var(--muted); margin: 10px 0 22px; }
.note { font-size: 13px; line-height: 1.45; color: var(--muted); margin: 10px 0 0; }
.card { border-top: 1px solid var(--line); padding-top: 12px; min-width: 0; }
.grid2 { display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 22px 38px; }
.grid3 { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 22px 32px; }
@media (max-width: 520px) { .grid2, .grid3 { grid-template-columns: 1fr; } }
.grid2 > *, .grid3 > *, .viewer > * { min-width: 0; }
code, .mono { font: 12px Menlo, Consolas, monospace; }
button, select { font: 14px Arial, Helvetica, sans-serif; color: var(--ink); background: #fff; border: 1px solid var(--line);
  padding: 6px 12px; cursor: pointer; touch-action: manipulation; }
button:hover, select:hover { background: var(--grid); }
button:focus-visible, select:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; }
button[aria-pressed="true"] { border-color: #333; box-shadow: inset 0 0 0 1px #333; }
button.primary { background: #333; color: #fff; border-color: #333; }
label.toggle { display: inline-flex; gap: 6px; align-items: center; color: var(--muted); cursor: pointer; font-size: 14px; white-space: nowrap; }
label.toggle input { accent-color: #444; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 7px 18px 7px 0; border-bottom: 1px solid var(--line); white-space: nowrap; font-size: 14px; }
th { font-weight: 600; }
td.num, th.num { text-align: right; }
.tablewrap { overflow-x: auto; }
.badge { display: inline-flex; align-items: center; gap: 5px; font-size: 13px; color: var(--ink); white-space: nowrap; font-weight: 400; }
.badge::before { content: ""; width: 8px; height: 8px; border-radius: 50%; background: var(--dot, var(--muted)); }
.b-win { --dot: var(--good); } .b-death { --dot: var(--critical); } .b-limit { --dot: var(--warning); }
.chart { position: relative; width: 100%; min-width: 0; }
.chart svg { display: block; overflow: hidden; }
.tip { position: absolute; pointer-events: none; background: #fff; color: var(--ink); border: 1px solid var(--line);
  padding: 6px 9px; font: 12px/1.45 Arial, Helvetica, sans-serif; box-shadow: 0 2px 10px rgba(0,0,0,.15); white-space: nowrap; z-index: 5; }
.tip .sw, .legend .sw { display: inline-block; width: 10px; height: 3px; border-radius: 2px; margin-right: 6px; vertical-align: middle; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 16px; font-size: 13px; color: var(--muted); margin: 0 0 6px; }
.axis text { fill: var(--muted); font: 11px Arial, Helvetica, sans-serif; font-variant-numeric: tabular-nums; }
.empty { color: var(--muted); font-size: 13px; padding: 16px 0; }
.sprites { display: flex; flex-wrap: wrap; gap: 8px 26px; margin: 0 0 18px; align-items: center; }
.sprites > span { display: flex; gap: 7px; align-items: center; font-size: 14px; color: var(--muted); }
.sprites canvas { width: 24px; height: 24px; image-rendering: pixelated; }

/* Game viewer, laid out like the website's game board. */
.viewer { display: grid; grid-template-columns: minmax(0, 1fr) 300px; gap: 30px; }
@media (max-width: 960px) { .viewer { grid-template-columns: 1fr; } }
.tabs { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }
.board { border: 1px solid var(--line); }
.hud { display: flex; justify-content: space-between; flex-wrap: wrap; gap: 6px 16px; font-size: 14px; padding: 12px; background: var(--panel); }
.hud .badge { font-size: 14px; }
.mapbox { background: #000; padding: 12px; overflow: hidden; position: relative; }
.mapbox canvas { display: block; margin: 0 auto; image-rendering: pixelated; }
.msg { font: 14px/1.5 Menlo, Consolas, monospace; min-height: 46px; padding: 10px 12px; border-top: 1px solid var(--line); }
.controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 12px 0 0; }
.controls input[type=range] { flex: 1 1 200px; min-width: 120px; accent-color: var(--accent); }
.controls button { min-width: 40px; }
.side h3 { margin-top: 18px; }
.side h3:first-child { margin-top: 0; }
.bars { display: grid; grid-template-columns: 82px 1fr 46px; gap: 4px 8px; align-items: center; font-size: 13px; }
.bars .bar { height: 10px; background: var(--grid); position: relative; }
.bars .bar i { position: absolute; left: 0; top: 0; bottom: 0; background: #b9bdc2; }
.bars .chosen { font-weight: 600; }
.bars .chosen .bar i { background: var(--accent); }
.bars span.v { text-align: right; color: var(--muted); font-variant-numeric: tabular-nums; }
.chips { display: grid; gap: 4px; margin: 4px 0; font-size: 14px; }
.chip { color: var(--muted); display: flex; gap: 8px; align-items: center; }
.chip::before { content: ""; width: 9px; height: 9px; border: 1px solid #9aa0a6; border-radius: 50%; }
.chip.on { color: var(--ink); }
.chip.on::before { background: var(--accent); border-color: var(--accent); }
.log { max-height: 200px; overflow: auto; font: 12px/1.5 Menlo, Consolas, monospace; border-top: 1px solid var(--line); }
.log div { padding: 2px 4px; cursor: pointer; }
.log div:hover { background: var(--grid); }
.log div.cur { background: #dce9f1; }
.log b { color: var(--muted); font-weight: 400; display: inline-block; min-width: 42px; }
.kv { display: grid; grid-template-columns: auto 1fr; gap: 4px 14px; font-size: 14px; }
.kv span:nth-child(odd) { color: var(--muted); }
.funnel { display: grid; grid-template-columns: 230px 1fr 90px; gap: 8px 12px; align-items: center; font-size: 15px; }
.funnel .bar { height: 14px; background: var(--grid); position: relative; }
.funnel .bar i { position: absolute; left: 0; top: 0; bottom: 0; background: var(--accent); }
.funnel span.v { text-align: right; font-variant-numeric: tabular-nums; }
@media (max-width: 520px) { .funnel { grid-template-columns: 1fr 1fr 70px; } }
.tiles { display: flex; flex-wrap: wrap; gap: 14px 36px; }
.tile .n { font-size: 27px; font-weight: 500; font-variant-numeric: tabular-nums; }
.tile .l { color: var(--muted); font-size: 14px; }
"""

CHART_JS = r"""
const NS = 'http://www.w3.org/2000/svg';
const SERIES = ['--s1','--s2','--s3','--s4','--s5','--s6','--s7','--s8'];
function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
function el(tag, attrs, parent, text) {
  const e = tag.startsWith('svg:') ? document.createElementNS(NS, tag.slice(4)) : document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) if (v !== undefined && v !== null) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
}
function niceTicks(lo, hi, n) {
  if (!(hi > lo)) { hi = lo + 1; }
  const span = hi - lo, raw = span / Math.max(1, n), mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => span / s <= n) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9; v += step) out.push(+v.toPrecision(12));
  return out;
}
function fmtNum(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return '–';
  const a = Math.abs(v);
  if (a >= 1e6) return (v / 1e6).toFixed(a >= 1e7 ? 0 : 1) + 'M';
  if (a >= 1e4) return (v / 1e3).toFixed(0) + 'k';
  if (a >= 100) return v.toFixed(0);
  if (a >= 1) return v.toFixed(2);
  if (a === 0) return '0';
  return v.toPrecision(2);
}
const fmtPct = v => (v === null || v === undefined) ? '–' : Math.round(v * 100) + '%';

/* Line chart. opt: {series:[{name, color, points:[[x,y|null]], width, dash, band:[[x,lo,hi]]}],
   height, yMin, yMax, xFmt, yFmt, xLabel, markers:[{x, color, label}], onPick(x), cursor, tipTitle(x)} */
function lineChart(root, opt) {
  root.classList.add('chart');
  const state = { cursor: opt.cursor };
  function draw() {
    root.innerHTML = '';
    const multi = opt.series.length > 1 && opt.legend !== false;
    if (multi) {
      const lg = el('div', { class: 'legend' }, root);
      for (const s of opt.series) {
        const it = el('span', {}, lg);
        el('span', { class: 'sw', style: `background:${s.color}` }, it);
        it.appendChild(document.createTextNode(s.name));
      }
    }
    const W = Math.max(240, root.clientWidth || 600), H = opt.height || 180;
    const m = { l: 44, r: 12, t: 8, b: opt.xLabel ? 32 : 22 };
    const xs = [], ys = [];
    for (const s of opt.series) {
      for (const [x, y] of s.points) { xs.push(x); if (y !== null && y !== undefined) ys.push(y); }
      for (const [x, lo, hi] of (s.band || [])) { if (lo !== null) { ys.push(lo, hi); } }
    }
    if (!xs.length) { el('div', { class: 'empty' }, root, 'Brak danych'); return; }
    let x0 = opt.xMin ?? Math.min(...xs), x1 = opt.xMax ?? Math.max(...xs);
    let y0 = opt.yMin ?? Math.min(0, ...ys), y1 = opt.yMax ?? Math.max(...ys);
    if (!(y1 > y0)) y1 = y0 + 1;
    if (!(x1 > x0)) x1 = x0 + 1;
    const X = x => m.l + (x - x0) / (x1 - x0) * (W - m.l - m.r);
    const Y = y => H - m.b - (y - y0) / (y1 - y0) * (H - m.t - m.b);
    const svg = el('svg:svg', { width: W, height: H, viewBox: `0 0 ${W} ${H}` }, root);
    const ax = el('svg:g', { class: 'axis' }, svg);
    const yf = opt.yFmt || fmtNum, xf = opt.xFmt || fmtNum;
    for (const v of niceTicks(y0, y1, 4)) {
      el('svg:line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: css('--grid'), 'stroke-width': 1 }, ax);
      el('svg:text', { x: m.l - 6, y: Y(v) + 4, 'text-anchor': 'end' }, ax, yf(v));
    }
    el('svg:line', { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: css('--axis') }, ax);
    for (const v of niceTicks(x0, x1, Math.max(2, Math.floor((W - m.l) / 90))))
      el('svg:text', { x: X(v), y: H - m.b + 14, 'text-anchor': 'middle' }, ax, xf(v));
    if (opt.xLabel) el('svg:text', { x: W - m.r, y: H - 4, 'text-anchor': 'end' }, ax, opt.xLabel);
    for (const mk of (opt.markers || [])) {
      el('svg:line', { x1: X(mk.x), x2: X(mk.x), y1: m.t, y2: H - m.b, stroke: mk.color, 'stroke-width': 1.5,
        'stroke-dasharray': '3 3', opacity: 0.9 }, svg);
    }
    for (const s of opt.series) {
      if (s.band && s.band.length) {
        let d = '', back = '';
        const pts = s.band.filter(p => p[1] !== null);
        pts.forEach((p, i) => { d += (i ? 'L' : 'M') + X(p[0]).toFixed(1) + ',' + Y(p[2]).toFixed(1); });
        pts.slice().reverse().forEach(p => { back += 'L' + X(p[0]).toFixed(1) + ',' + Y(p[1]).toFixed(1); });
        if (pts.length) el('svg:path', { d: d + back + 'Z', fill: s.color, opacity: 0.15, stroke: 'none' }, svg);
      }
      let d = '', pen = false;
      for (const [x, y] of s.points) {
        if (y === null || y === undefined) { pen = false; continue; }
        d += (pen ? 'L' : 'M') + X(x).toFixed(1) + ',' + Y(y).toFixed(1); pen = true;
      }
      el('svg:path', { d, fill: 'none', stroke: s.color, 'stroke-width': s.width || 2, 'stroke-dasharray': s.dash,
        'stroke-linejoin': 'round', 'stroke-linecap': 'round' }, svg);
      const live = s.points.filter(p => p[1] !== null && p[1] !== undefined);
      if (live.length === 1) el('svg:circle', { cx: X(live[0][0]), cy: Y(live[0][1]), r: 4, fill: s.color }, svg);
    }
    const cur = el('svg:line', { y1: m.t, y2: H - m.b, stroke: css('--ink-2'), 'stroke-width': 1, visibility: 'hidden' }, svg);
    const hov = el('svg:line', { y1: m.t, y2: H - m.b, stroke: css('--muted'), 'stroke-width': 1, visibility: 'hidden' }, svg);
    const tip = el('div', { class: 'tip', style: 'display:none' }, root);
    function setCursor(x) {
      state.cursor = x;
      if (x === undefined || x === null) { cur.setAttribute('visibility', 'hidden'); return; }
      cur.setAttribute('x1', X(x)); cur.setAttribute('x2', X(x)); cur.setAttribute('visibility', 'visible');
    }
    setCursor(state.cursor);
    state.setCursor = setCursor;
    const hit = el('svg:rect', { x: m.l, y: 0, width: W - m.l - m.r, height: H - m.b, fill: 'transparent',
      style: opt.onPick ? 'cursor:pointer' : '' }, svg);
    const xAt = ev => { const r = svg.getBoundingClientRect(); return x0 + (ev.clientX - r.left - m.l) / (W - m.l - m.r) * (x1 - x0); };
    hit.addEventListener('mousemove', ev => {
      const x = xAt(ev);
      hov.setAttribute('x1', X(x)); hov.setAttribute('x2', X(x)); hov.setAttribute('visibility', 'visible');
      tip.innerHTML = '';
      el('div', { style: 'color:var(--ink-2);margin-bottom:2px' }, tip, opt.tipTitle ? opt.tipTitle(x) : xf(x));
      for (const s of opt.series) {
        let best = null;
        for (const p of s.points) if (p[1] !== null && p[1] !== undefined && (best === null || Math.abs(p[0] - x) < Math.abs(best[0] - x))) best = p;
        if (!best) continue;
        const row = el('div', {}, tip);
        el('span', { class: 'sw', style: `background:${s.color}` }, row);
        row.appendChild(document.createTextNode(`${s.name}: ${yf(best[1])}`));
      }
      for (const mk of (opt.markers || [])) if (Math.abs(X(mk.x) - X(x)) < 4) el('div', {}, tip, mk.label);
      tip.style.display = 'block';
      const r = root.getBoundingClientRect(), px = ev.clientX - r.left;
      tip.style.left = Math.min(px + 12, r.width - tip.offsetWidth - 4) + 'px';
      tip.style.top = (ev.clientY - r.top + 12) + 'px';
    });
    hit.addEventListener('mouseleave', () => { tip.style.display = 'none'; hov.setAttribute('visibility', 'hidden'); });
    let dragging = false;
    if (opt.onPick) {
      hit.addEventListener('mousedown', ev => { dragging = true; opt.onPick(xAt(ev)); });
      window.addEventListener('mouseup', () => { dragging = false; });
      hit.addEventListener('mousemove', ev => { if (dragging) opt.onPick(xAt(ev)); });
    }
  }
  draw();
  let timer;
  new ResizeObserver(() => { clearTimeout(timer); timer = setTimeout(draw, 80); }).observe(root);
  return { setCursor: x => (state.setCursor ? state.setCursor(x) : (state.cursor = x)), redraw: draw };
}
"""

VIEWER_JS = r"""
const NH = ['#6c6c6c', '#d03b3b', '#3fae3f', '#b5832a', '#3f6fd8', '#b545b5', '#2fa8b5', '#c8c8c8',
            '#8a8a8a', '#ff8c2a', '#6fe86f', '#ffe14a', '#6f9cff', '#ff7cff', '#6fe8f0', '#ffffff'];
const ARROWS = { k: '↑', j: '↓', h: '←', l: '→', y: '↖', u: '↗', b: '↙', n: '↘' };
function actionLabel(name) {
  if (ARROWS[name]) return `${ARROWS[name]} ${name}`;
  if (name === ',') return 'podnieś ,';
  if (name === 'z') return 'zap z';
  if (name === 'f') return 'wybierz f';
  if (name.length === 2 && name[0] === 'z') return `zap ${ARROWS[name[1]]}`;
  return name;
}
const END = { TASK_SUCCESSFUL: ['wygrana', 'b-win'], DEATH: ['śmierć', 'b-death'], ABORTED: ['limit akcji', 'b-limit'] };
function endBadge(status) { const [t, c] = END[status] || [status, '']; return `<span class="badge ${c}">${t}</span>`; }
const EVENT_COLORS = { wand: '--s4', pick: '--s2', mino: '--s7', zap: '--s8' };

/* MiniHack tiles: DATA.tiles = {png, size, cols, index: {glyph: atlas slot}}; DATA.legend = {glyph: [char, colour]}. */
const TILES = (() => {
  const t = DATA.tiles || null;
  if (!t) return null;
  const img = new Image();
  const waiting = [];
  img.onload = () => { t.ready = true; waiting.splice(0).forEach(f => f()); };
  img.src = 'data:image/png;base64,' + t.png;
  t.img = img;
  t.whenReady = f => (t.ready ? f() : waiting.push(f));
  return t;
})();
const LEGEND = DATA.legend || {};
const isBlank = g => { const e = LEGEND[g]; return !e || e[0] === 32 || e[0] === 0; };

/* Draw map glyphs into ctx. box = [x0, y0, x1, y1] in map cells, cell = pixel size of a square cell. */
function drawGlyphs(ctx, glyphs, width, box, cell, ascii) {
  const [x0, y0, x1, y1] = box;
  ctx.imageSmoothingEnabled = false;
  if (ascii || !TILES || !TILES.ready) {
    ctx.font = `${Math.round(cell * 0.8)}px Menlo, Consolas, monospace`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  }
  for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
    const g = glyphs[y * width + x];
    if (isBlank(g)) continue;
    const px = (x - x0) * cell, py = (y - y0) * cell;
    const slot = TILES && TILES.index[g];
    if (!ascii && TILES && TILES.ready && slot !== undefined) {
      const s = TILES.size;
      ctx.drawImage(TILES.img, (slot % TILES.cols) * s, Math.floor(slot / TILES.cols) * s, s, s, px, py, cell, cell);
    } else {
      const [c, col] = LEGEND[g];
      ctx.fillStyle = NH[col & 15];
      ctx.fillText(String.fromCharCode(c), px + cell / 2, py + cell / 2 + 1);
    }
  }
}
function boardCell(container, cols) {
  const avail = Math.max(160, container.clientWidth - 24);
  return Math.max(8, Math.min(32, Math.floor(avail / cols)));
}
function setupCanvas(canvas, w, h) {
  const dpr = window.devicePixelRatio || 1, ctx = canvas.getContext('2d');
  canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
  canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, w, h);
  return ctx;
}
/* Sprite legend as on the website: player, wand, minotaur, staircase, drawn from the atlas. */
function spriteLegend(root) {
  if (!TILES) return;
  const want = [[64, 'Gracz'], [47, 'Różdżka'], [72, 'Minotaur'], [62, 'Schody']];
  for (const [ch, label] of want) {
    const g = Object.keys(LEGEND).find(k => LEGEND[k][0] === ch && TILES.index[k] !== undefined);
    if (g === undefined) continue;
    const span = el('span', {}, root), c = el('canvas', { width: 16, height: 16 }, span);
    span.appendChild(document.createTextNode(label));
    TILES.whenReady(() => { const ctx = c.getContext('2d'); ctx.imageSmoothingEnabled = false; const s = TILES.size, slot = TILES.index[g];
      ctx.drawImage(TILES.img, (slot % TILES.cols) * s, Math.floor(slot / TILES.cols) * s, s, s, 0, 0, 16, 16); });
  }
}

function prepareGame(g) {
  if (g._ready) return g;
  const { width: W, height: H } = g.map, N = W * H;
  let glyphs = Uint16Array.from(g.map.glyphs);
  g._glyphs = [glyphs];
  for (const d of g.map.diffs) {
    glyphs = glyphs.slice();
    for (let k = 0; k < d.length; k += 2) glyphs[d[k]] = d[k + 1];
    g._glyphs.push(glyphs);
  }
  // Crop to the cells that are ever drawn, plus a margin.
  let x0 = W, x1 = 0, y0 = H, y1 = 0;
  for (const c of g._glyphs) for (let i = 0; i < N; i++) if (!isBlank(c[i])) {
    const x = i % W, y = (i / W) | 0;
    if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
  }
  if (x1 < x0) { x0 = 0; x1 = W - 1; y0 = 0; y1 = H - 1; }
  g._box = [Math.max(0, x0 - 1), Math.max(0, y0 - 1), Math.min(W - 1, x1 + 1), Math.min(H - 1, y1 + 1)];
  g._entropy = g.steps.map(s => s.probs ? -s.probs.reduce((a, p) => a + (p > 0 ? p * Math.log(p) : 0), 0) : null);
  const ev = [], f = g.summary.first || {};
  if (f.wand_visible !== undefined) ev.push({ x: f.wand_visible, k: 'wand', label: 'różdżka widoczna' });
  if (f.has_wand !== undefined && f.has_wand > 0) ev.push({ x: f.has_wand, k: 'pick', label: 'różdżka podniesiona' });
  if (f.minotaur_visible !== undefined) ev.push({ x: f.minotaur_visible, k: 'mino', label: 'minotaur widoczny' });
  g.steps.forEach((s, t) => { if (g.actions[s.a].startsWith('z') && g.frames[t].has_wand) ev.push({ x: t, k: 'zap', label: `zap (${g.actions[s.a]})` }); });
  g._events = ev;
  g._ready = true;
  return g;
}

function Viewer(root, games) {
  const st = { gi: 0, t: 0, playing: false, fps: 8, trail: true, heat: false, ascii: !TILES };
  root.innerHTML = `
    <div class="tabs"></div>
    <div class="viewer">
      <div>
        <div class="board">
          <div class="hud"><span data-k="title"></span><span data-k="status"></span></div>
          <div class="mapbox"><canvas></canvas></div>
          <div class="msg" data-k="msg"></div>
        </div>
        <div class="controls">
          <button data-k="first" title="Początek (Home)">⏮</button>
          <button data-k="prev" title="Krok wstecz (←, Shift: 10)">←</button>
          <button data-k="play" class="primary" title="Odtwarzaj (spacja)">▶</button>
          <button data-k="next" title="Krok dalej (→, Shift: 10)">→</button>
          <button data-k="last" title="Koniec (End)">⏭</button>
          <select data-k="fps" title="Prędkość"><option value="2">2 kroki/s</option><option value="8" selected>8 kroków/s</option><option value="20">20 kroków/s</option><option value="60">60 kroków/s</option></select>
          <input type="range" min="0" value="0" data-k="slider" aria-label="Krok">
        </div>
        <div class="controls">
          <label class="toggle"><input type="checkbox" data-k="trail" checked> ślad ostatnich 40 kroków</label>
          <label class="toggle"><input type="checkbox" data-k="heat"> mapa odwiedzin</label>
          <label class="toggle"><input type="checkbox" data-k="ascii"> ASCII</label>
        </div>
        <div class="grid3" style="margin-top:28px">
          <div class="card"><h3 data-k="v0title">Wartość krytyka</h3><div data-k="v0chart"></div></div>
          <div class="card" data-k="v1card"><h3>Wartość krytyka: bonus RND</h3><div data-k="v1chart"></div></div>
          <div class="card"><h3>Entropia polityki</h3><div data-k="echart"></div></div>
        </div>
        <div class="legend" data-k="evlegend" style="margin-top:10px"></div>
      </div>
      <div class="side">
        <h3>Gra</h3>
        <div class="kv" data-k="kv"></div>
        <h3>Stan</h3>
        <div class="chips" data-k="chips"></div>
        <h3>Rozkład akcji w tym kroku</h3>
        <div class="bars" data-k="bars"></div>
        <h3>Komunikaty gry</h3>
        <div class="log" data-k="log"></div>
      </div>
    </div>`;
  const q = k => root.querySelector(`[data-k="${k}"]`);
  const canvas = root.querySelector('canvas'), mapbox = root.querySelector('.mapbox');
  const tabs = root.querySelector('.tabs');
  q('ascii').checked = st.ascii;
  let charts = [], timer;

  function game() { return prepareGame(games[st.gi]); }

  function renderTabs() {
    tabs.innerHTML = '';
    if (games.length < 2) { tabs.style.display = 'none'; return; }
    tabs.style.display = '';
    games.forEach((g, i) => {
      const b = el('button', { 'aria-pressed': i === st.gi }, tabs);
      b.innerHTML = `${g.label || g.level + ' ' + g.seed} &nbsp;${endBadge(g.result.end_status)}`;
      b.onclick = () => select(i);
    });
  }

  function select(i, t) {
    st.gi = i;
    const g = game();
    st.t = Math.min(t ?? st.t, g.frames.length - 1);
    renderTabs();
    q('slider').max = g.frames.length - 1;
    const markers = g._events.map(e => ({ x: e.x, color: css(EVENT_COLORS[e.k]), label: e.label }));
    const pick = x => go(Math.round(x));
    const tipTitle = x => `decyzja ${Math.round(x)}`;
    const common = { height: 150, markers, onPick: pick, cursor: st.t, tipTitle, xLabel: 'decyzja' };
    const isQ = g.steps.length && g.steps[0].q;
    q('v0title').textContent = isQ ? 'max Q (DQN)' : g.value_heads === 2 ? 'Wartość krytyka: nagroda gry + kształtowanie' : 'Wartość krytyka';
    q('v1card').style.display = g.value_heads === 2 ? '' : 'none';
    charts = [];
    const line = css('--accent');
    if (g.value_heads >= 1)
      charts.push(lineChart(q('v0chart'), { ...common, series: [{ name: 'wartość', color: line, width: 1.5, points: g.steps.map((s, t) => [t, s.values[0]]) }] }));
    else q('v0chart').innerHTML = '<div class="empty">Ten agent nie ma krytyka</div>';
    if (g.value_heads === 2)
      charts.push(lineChart(q('v1chart'), { ...common, series: [{ name: 'wartość RND', color: line, width: 1.5, points: g.steps.map((s, t) => [t, s.values[1]]) }] }));
    if (g._entropy.every(v => v === null)) q('echart').innerHTML = '<div class="empty">Brak rozkładu akcji (DQN wybiera argmax Q)</div>';
    else charts.push(lineChart(q('echart'), { ...common, yMin: 0, series: [{ name: 'entropia', color: line, width: 1.2, points: g.steps.map((s, t) => [t, g._entropy[t]]) }] }));
    const lg = q('evlegend'); lg.innerHTML = '';
    for (const [k, label] of [['wand', 'różdżka widoczna'], ['pick', 'podniesiona'], ['mino', 'minotaur widoczny'], ['zap', 'zap z różdżką']]) {
      const it = el('span', {}, lg); el('span', { class: 'sw', style: `background:${css(EVENT_COLORS[k])}` }, it);
      it.appendChild(document.createTextNode(label));
    }
    const log = q('log'); log.innerHTML = '';
    let prev = '';
    g.frames.forEach((f, t) => {
      if (f.msg && f.msg !== prev) { const d = el('div', { 'data-t': t }, log); d.innerHTML = `<b>${t}</b>`; d.appendChild(document.createTextNode(f.msg)); d.onclick = () => go(t); }
      prev = f.msg;
    });
    if (!log.children.length) el('div', { class: 'empty' }, log, 'Brak komunikatów');
    draw();
  }

  function draw() {
    const g = game(), t = st.t, f = g.frames[t], W = g.map.width;
    const [x0, y0, x1, y1] = g._box, cell = boardCell(mapbox, x1 - x0 + 1);
    const w = (x1 - x0 + 1) * cell, h = (y1 - y0 + 1) * cell;
    const ctx = setupCanvas(canvas, w, h);
    const px = x => (x - x0) * cell, py = y => (y - y0) * cell;
    drawGlyphs(ctx, g._glyphs[t], W, g._box, cell, st.ascii);
    if (st.heat) {
      const counts = new Map(); let max = 1;
      for (let k = 0; k <= t; k++) { const key = g.frames[k].x + ',' + g.frames[k].y; const c = (counts.get(key) || 0) + 1; counts.set(key, c); max = Math.max(max, c); }
      for (const [key, c] of counts) {
        const [x, y] = key.split(',').map(Number);
        ctx.fillStyle = `rgba(57,135,229,${0.2 + 0.6 * Math.log(1 + c) / Math.log(1 + max)})`;
        ctx.fillRect(px(x), py(y), cell, cell);
      }
    }
    if (st.trail) {
      ctx.strokeStyle = 'rgba(255,214,0,0.75)'; ctx.lineWidth = Math.max(2, cell / 7); ctx.lineJoin = 'round'; ctx.lineCap = 'round';
      ctx.beginPath();
      const from = Math.max(0, t - 40);
      for (let k = from; k <= t; k++) { const F = g.frames[k]; const X = px(F.x) + cell / 2, Y = py(F.y) + cell / 2; k === from ? ctx.moveTo(X, Y) : ctx.lineTo(X, Y); }
      ctx.stroke();
    }
    ctx.strokeStyle = '#ffd600'; ctx.lineWidth = 2;
    ctx.strokeRect(px(f.x) + 1, py(f.y) + 1, cell - 2, cell - 2);
    renderPanel();
  }

  function renderPanel() {
    const g = game(), t = st.t, f = g.frames[t], s = g.steps[t], N = g.steps.length;
    q('slider').value = t;
    q('title').innerHTML = `${g.level} · seed ${g.seed} &nbsp;${endBadge(g.result.end_status)}`;
    q('status').textContent = `decyzja ${t} / ${N} · akcje ${f.t} / ${g.horizon} · HP ${f.hp}/${f.hp_max}`;
    q('msg').textContent = f.msg || ' ';
    const kv = q('kv'); kv.innerHTML = '';
    const add = (k, v) => { el('span', {}, kv, k); el('span', {}, kv, v); };
    if (g.label) add('agent', g.label);
    add('polityka', g.policy === 'sample' ? 'losowanie z rozkładu' : 'greedy (argmax)');
    add('wynik gry', `${g.result.native_return.toFixed(2)} w ${g.result.steps} akcjach`);
    if (s) {
      add('akcja', `${actionLabel(g.actions[s.a])}${s.keys.length > 1 ? ' (klawisze ' + s.keys.split('').join(' ') + ')' : ''}`);
      if (s.values.length) add(g.value_heads === 2 ? 'wartości' : 'wartość', s.values.map(v => v.toFixed(3)).join(' / '));
      if (s.r) add('nagroda', s.r.toFixed(2));
    } else add('akcja', 'koniec gry');
    const chips = q('chips'); chips.innerHTML = '';
    for (const [k, label] of [['wand_visible', 'różdżka widoczna'], ['has_wand', 'różdżka w ekwipunku'], ['minotaur_visible', 'minotaur widoczny'],
                              ...('on_item' in f ? [['on_item', 'stoi na przedmiocie (wejście on_item)']] : [])])
      el('span', { class: 'chip' + (f[k] ? ' on' : '') }, chips, label);
    const bars = q('bars'); bars.innerHTML = '';
    if (s) {
      const vals = s.probs || s.q, isQ = !s.probs;
      const lo = isQ ? Math.min(...vals) : 0, hi = isQ ? Math.max(...vals) : 1;
      vals.forEach((p, a) => {
        const cls = a === s.a ? 'chosen' : '';
        el('span', { class: cls }, bars, actionLabel(g.actions[a]));
        const wrap = el('div', { class: cls }, bars); const b = el('div', { class: 'bar' }, wrap);
        el('i', { style: `width:${(100 * (p - lo) / ((hi - lo) || 1)).toFixed(1)}%` }, b);
        el('span', { class: 'v ' + cls }, bars, isQ ? p.toFixed(2) : (p * 100).toFixed(p < 0.1 ? 1 : 0) + '%');
      });
      if (isQ) el('div', { class: 'empty', style: 'grid-column:1/-1;padding:4px 0' }, bars, 'DQN: wartości Q zamiast prawdopodobieństw');
    } else el('div', { class: 'empty', style: 'grid-column:1/-1' }, bars, `Koniec: ${(END[g.result.end_status] || [g.result.end_status])[0]}`);
    for (const d of q('log').children) d.classList.toggle('cur', +d.dataset.t === t);
    for (const c of charts) c.setCursor(t);
  }

  function go(t) {
    const g = game();
    st.t = Math.max(0, Math.min(g.frames.length - 1, t));
    draw();
  }
  function play(on) {
    st.playing = on; q('play').textContent = on ? '⏸' : '▶';
    clearInterval(timer);
    if (on) timer = setInterval(() => { if (st.t >= game().frames.length - 1) play(false); else go(st.t + 1); }, 1000 / st.fps);
  }
  q('first').onclick = () => go(0);
  q('last').onclick = () => go(1e9);
  q('prev').onclick = () => go(st.t - 1);
  q('next').onclick = () => go(st.t + 1);
  q('play').onclick = () => play(!st.playing);
  q('fps').onchange = e => { st.fps = +e.target.value; if (st.playing) play(true); };
  q('slider').oninput = e => go(+e.target.value);
  q('trail').onchange = e => { st.trail = e.target.checked; draw(); };
  q('heat').onchange = e => { st.heat = e.target.checked; draw(); };
  q('ascii').onchange = e => { st.ascii = e.target.checked; draw(); };
  document.addEventListener('keydown', e => {
    if (!root.isConnected || root.offsetParent === null || ['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName) && !['range', 'checkbox'].includes(e.target.type)) return;
    const step = e.shiftKey ? 10 : 1;
    if (e.key === 'ArrowRight') { go(st.t + step); e.preventDefault(); }
    else if (e.key === 'ArrowLeft') { go(st.t - step); e.preventDefault(); }
    else if (e.key === ' ') { play(!st.playing); e.preventDefault(); }
    else if (e.key === 'Home') go(0);
    else if (e.key === 'End') go(1e9);
  });
  new ResizeObserver(() => draw()).observe(mapbox);
  if (TILES) TILES.whenReady(() => draw());
  select(0, 0);
  return { select: (i, t) => { play(false); select(i, t ?? 0); } };
}
"""


def tile_atlas(glyphs):
    """MiniHack's 16x16 tiles for the given glyphs, as one base64 PNG with a glyph -> slot index."""
    import base64
    import io
    import math
    import warnings

    import numpy as np
    from PIL import Image
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        from minihack.tiles import GlyphMapper
    glyphs = sorted({int(g) for g in glyphs})
    if not glyphs:
        return None
    mapper, size, cols = GlyphMapper(), 16, 16
    atlas = np.zeros((math.ceil(len(glyphs) / cols) * size, cols * size, 3), np.uint8)
    for slot, glyph in enumerate(glyphs):
        row, col = divmod(slot, cols)
        atlas[row * size:(row + 1) * size, col * size:(col + 1) * size] = mapper.glyph_id_to_rgb(glyph)
    stream = io.BytesIO()
    Image.fromarray(atlas).save(stream, format='PNG', optimize=True)
    return dict(png=base64.b64encode(stream.getvalue()).decode('ascii'), size=size, cols=cols,
                index={str(g): i for i, g in enumerate(glyphs)})


def attach_tiles(data, games, extra_glyphs=()):
    """Add the shared tile atlas and glyph legend of these recorded games to page data."""
    legend, glyphs = {}, set(extra_glyphs)
    for game in games:
        legend.update(game['map']['legend'])
        glyphs.update(game['map']['glyphs'])
        for diff in game['map']['diffs']:
            glyphs.update(diff[1::2])
    data['legend'] = legend
    data['tiles'] = tile_atlas(glyphs)
    return data


def page(title, body, data, script):
    """One HTML file with the shared style, the chart helpers, data as JSON and a page script."""
    payload = json.dumps(data, separators=(',', ':'), ensure_ascii=False).replace('</', '<\\/')
    return f"""<!doctype html>
<html lang="pl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{CSS}</style>
</head>
<body>
<main>
{body}
</main>
<script type="application/json" id="data">{payload}</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);
{CHART_JS}
{VIEWER_JS}
{script}
</script>
</body>
</html>
"""


def write(path, text):
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return path
