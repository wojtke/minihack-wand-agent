// Builds the static PDF of the presentation with headless Chrome (no npm packages).
// Usage, from the repository root: node prezentacja-pdf/build.mjs [--no-demo]
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync, existsSync } from 'node:fs';
import { extname, join, resolve } from 'node:path';
import { tmpdir } from 'node:os';

const ROOT = resolve(import.meta.dirname, '..');
const OUT = join(ROOT, 'prezentacja-pdf');
const MEDIA = join(OUT, 'media');
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const LIVE = 'https://wojtke.github.io/minihack-wand-agent/#5';
const PDF_NAME = '5 – Iśtok, Jasiński.pdf';
mkdirSync(MEDIA, { recursive: true });

const types = { '.html': 'text/html', '.png': 'image/png', '.mp4': 'video/mp4', '.css': 'text/css', '.js': 'text/javascript', '.json': 'application/json' };
const server = createServer((req, res) => {
  const path = join(ROOT, decodeURIComponent(new URL(req.url, 'http://x').pathname));
  if (!path.startsWith(ROOT) || !existsSync(path)) { res.writeHead(404).end(); return; }
  const body = readFileSync(path), type = types[extname(path)] || 'application/octet-stream';
  // Video seeking needs byte ranges.
  const range = /bytes=(\d*)-(\d*)/.exec(req.headers.range || '');
  if (!range) { res.writeHead(200, { 'content-type': type, 'accept-ranges': 'bytes' }).end(body); return; }
  const start = range[1] ? +range[1] : body.length - +range[2], end = range[1] && range[2] ? +range[2] : body.length - 1;
  res.writeHead(206, { 'content-type': type, 'accept-ranges': 'bytes', 'content-range': `bytes ${start}-${end}/${body.length}` }).end(body.subarray(start, end + 1));
});
await new Promise(r => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}`;

const profile = mkdtempSync(join(tmpdir(), 'wand-pdf-'));
const chrome = spawn(CHROME, ['--headless=new', '--remote-debugging-port=0', `--user-data-dir=${profile}`,
  '--no-first-run', '--autoplay-policy=no-user-gesture-required', '--hide-scrollbars', 'about:blank'], { stdio: ['ignore', 'ignore', 'pipe'] });
const wsUrl = await new Promise((ok, fail) => {
  let buf = '';
  chrome.stderr.on('data', d => { buf += d; const m = buf.match(/DevTools listening on (ws:\S+)/); if (m) ok(m[1]); });
  chrome.on('exit', () => fail(new Error('Chrome exited')));
});
const port = new URL(wsUrl).port;
const target = await (await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, { method: 'PUT' })).json();
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise(r => ws.addEventListener('open', r));
let nextId = 0; const pending = new Map();
ws.addEventListener('message', e => { const m = JSON.parse(e.data); if (pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); } });
const send = (method, params = {}) => new Promise((ok, fail) => {
  const id = ++nextId; pending.set(id, m => m.error ? fail(new Error(`${method}: ${m.error.message}`)) : ok(m.result));
  ws.send(JSON.stringify({ id, method, params }));
});
const evaluate = async expression => {
  const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text);
  return r.result.value;
};
const sleep = ms => new Promise(r => setTimeout(r, ms));
const goto = async (url, settle = 500) => { await send('Page.navigate', { url }); await evaluate(`new Promise(r => document.readyState === 'complete' ? r() : addEventListener('load', r))`); await sleep(settle); };
await send('Page.enable'); await send('Runtime.enable');

// 1. Start, middle and end frames of the random-agent recordings.
await goto(`${base}/prezentacja-pdf/slides.html`);
for (const level of ['easy', 'medium', 'hard', 'pro']) {
  const frames = await evaluate(`(async () => {
    const v = document.createElement('video'); v.muted = true; v.preload = 'auto'; v.src = '/website/assets/random-${level}.mp4';
    document.body.append(v);
    await new Promise((ok, fail) => { v.onloadeddata = ok; v.onerror = () => fail(new Error('video')); });
    const at = [0, v.duration / 2, Math.max(0, v.duration - 0.05)], out = [];
    for (const t of at) {
      await new Promise(ok => { v.onseeked = ok; v.currentTime = t; });
      await new Promise(ok => v.requestVideoFrameCallback ? (v.requestVideoFrameCallback(ok), setTimeout(ok, 500)) : setTimeout(ok, 300));
      const c = document.createElement('canvas'); c.width = v.videoWidth; c.height = v.videoHeight;
      c.getContext('2d').drawImage(v, 0, 0); out.push(c.toDataURL('image/png'));
    }
    return out;
  })()`);
  frames.forEach((d, i) => writeFileSync(join(MEDIA, `random-${level}-${i}.png`), Buffer.from(d.split(',')[1], 'base64')));
}
console.log('random-agent frames: ok');

// 2. Screenshot of the live game (published site), with a few moves played on Hard.
if (!process.argv.includes('--no-demo')) {
  await send('Emulation.setDeviceMetricsOverride', { width: 1100, height: 900, deviceScaleFactor: 2, mobile: false });
  await goto(LIVE, 1500);
  await evaluate(`(async () => {
    const s = document.querySelector('#game-level'); s.value = 'Hard'; s.dispatchEvent(new Event('change', { bubbles: true }));
    document.querySelector('#new-game').click();
    for (let i = 0; i < 200 && document.querySelector('#game-image').hidden; i++) await new Promise(r => setTimeout(r, 100));
    if (document.querySelector('#game-image').hidden) throw new Error('game did not start');
  })()`);
  for (const key of 'lllljjl') {
    await evaluate(`document.querySelector('.keypad button[data-key="${key}"]').click()`);
    await sleep(350);
  }
  await sleep(800);
  // Board and agent-view legend only; the controls are described on the slide.
  const box = await evaluate(`(() => { const a = document.querySelector('#game-board').getBoundingClientRect(), b = document.querySelector('#agent-legend').getBoundingClientRect(); return { x: a.x, y: a.y, width: a.width, height: b.bottom - a.y }; })()`);
  const shot = await send('Page.captureScreenshot', { format: 'png', clip: { ...box, scale: 1 } });
  writeFileSync(join(MEDIA, 'demo.png'), Buffer.from(shot.data, 'base64'));
  await send('Emulation.clearDeviceMetricsOverride');
  console.log('demo screenshot: ok');
}

// 3. PDF, one 1280×720 page per slide.
await goto(`${base}/prezentacja-pdf/slides.html`, 800);
const pdf = await send('Page.printToPDF', { preferCSSPageSize: true, printBackground: true, marginTop: 0, marginBottom: 0, marginLeft: 0, marginRight: 0 });
const pkg = join(OUT, 'out');
mkdirSync(pkg, { recursive: true });
writeFileSync(join(pkg, PDF_NAME), Buffer.from(pdf.data, 'base64'));
console.log(`pdf: ${join(pkg, PDF_NAME)}`);

ws.close(); chrome.kill(); server.close();
