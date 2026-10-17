// Optional Playwright dependency; no browser automation ships to site visitors.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const engines = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const url = process.env.PRESENTATION_URL || 'http://localhost:8000/';
const selected = (process.env.BROWSERS || 'chromium,firefox,webkit').split(',');
const cases = JSON.parse(fs.readFileSync(path.join(root, '.build/native-traces.json')))
  .filter(c => c.kind === 'policy' && c.seed === 0)
  .map(c => ({...c, states: c.states.map(s => s.image ? {image:s.image} : null)}));
const out = path.join(root, '.build/browser-qa');
fs.mkdirSync(out, {recursive:true});
for (const name of selected) {
  const browser = await engines[name].launch({headless:true,
    ...(name === 'chromium' && process.env.CHROME_PATH ? {executablePath:process.env.CHROME_PATH} : {})});
  try {
    const context = await browser.newContext({viewport:{width:1280,height:720}});
    const errors = [], requests = [];
    context.on('request', r => requests.push({url:r.url(),method:r.method()}));
    const page = await context.newPage();
    page.on('pageerror', e => errors.push(e.message));
    await page.goto(url + '#3');
    const ready = () => page.waitForFunction(() => !document.body.classList.contains('game-busy') &&
      document.querySelector('#game-state').textContent.includes('playing'), null, {timeout:45000});
    const key = async value => {
      await page.locator('#game-board').focus();
      await page.keyboard.press(value);
      await page.waitForFunction(() => !document.body.classList.contains('game-busy'));
    };
    await ready();
    assert.equal(await page.locator('#hints').isChecked(), false);
    for (const level of ['Easy','Medium','Hard','Pro']) {
      if (level !== 'Easy') { await page.selectOption('#game-level', level); await ready(); }
      assert.match(await page.locator('#game-state').innerText(), new RegExp('^' + level));
      const initial = await page.locator('#game-image').getAttribute('src');
      await key('w');
      assert.match(await page.locator('#game-counters').innerText(), /^1\//);
      await key('Space'); await ready();
      assert.match(await page.locator('#game-counters').innerText(), /^0\//);
      assert.equal(await page.locator('#game-image').getAttribute('src'), initial);
      await page.locator('#hints').check();
      assert.equal(await page.locator('#hint-text').isVisible(), true);
      await page.locator('#hints').uncheck();
      await page.locator('#new-game').click(); await ready();
      assert.equal(await page.evaluate(() => game.seed), 1);
      await page.screenshot({path:path.join(out, name + '-' + level + '.png')});
    }
    await page.selectOption('#game-level', 'Easy'); await ready();
    for (const value of ['r','f','d']) await key(value);
    assert.match(await page.locator('#game-state').innerText(), /success/);
    assert.match(await page.locator('#game-counters').innerText(), /^3\/50 actions · return 0.98$/);
    for (let i=0; i<8; i++) { await key('Space'); await ready(); }
    for (const [shortcut,native] of Object.entries({w:'k',a:'h',s:'j',d:'l',q:'y',e:'u',z:'b',c:'n',p:',',r:'z',f:'f'})) {
      await key(shortcut);
      assert.equal(await page.evaluate(() => game.last_key), native);
      await key('Space'); await ready();
    }
    assert.equal(page.workers().length, 1);
    // Run real trajectories through the worker and compare decoded RGB pixels
    // with native MiniHack, including terminal screens and prompt states.
    const pixels = await page.evaluate(async cases => {
      const {BrowserGame} = await import('./engine/client.mjs');
      const client = new BrowserGame();
      async function rgba(uri) {
        const bitmap = await createImageBitmap(await (await fetch(uri)).blob());
        const canvas = new OffscreenCanvas(bitmap.width,bitmap.height);
        const context = canvas.getContext('2d'); context.drawImage(bitmap,0,0);
        return {width:bitmap.width,height:bitmap.height,data:context.getImageData(0,0,bitmap.width,bitmap.height).data};
      }
      let checked = 0;
      try {
        for (const test of cases) {
          let state = await client.request('reset',{level:test.level,seed:test.seed});
          for (let i=0;i<test.states.length;i++) {
            if (i) state = await client.request('step',{key:test.keys[i-1]});
            if (!test.states[i]) continue;
            const [got,want] = await Promise.all([rgba(state.image),rgba(test.states[i].image)]);
            if (got.width!==want.width || got.height!==want.height ||
                got.data.some((n,j)=>n!==want.data[j])) throw new Error('Pixel mismatch: '+test.level+' step '+i);
            checked++;
          }
        }
      } finally {client.close();}
      return checked;
    }, cases);
    assert.deepEqual(errors, []);
    const network = requests.filter(r => /^https?:/.test(r.url));
    assert(network.every(r => r.method === 'GET' && new URL(r.url).origin === new URL(url).origin));
    assert(!network.some(r => new URL(r.url).pathname.includes('/api/')));
    // A failed download must leave New game usable for recovery.
    if (name === 'chromium') {
      const failed = await context.newPage();
      await context.route('**/engine/wand.wasm', route => route.abort());
      await failed.goto(url + '#3');
      await failed.waitForFunction(() => document.querySelector('#game-state').textContent.includes('Could not start'), null, {timeout:45000});
      assert.equal(await failed.locator('#new-game').isEnabled(), true);
      await context.unroute('**/engine/wand.wasm');
      await failed.locator('#new-game').click();
      await failed.waitForFunction(() => document.querySelector('#game-state').textContent.includes('playing'), null, {timeout:45000});
      await failed.close();
    }
    console.log(JSON.stringify({browser:name,levels:4,pixelComparisons:pixels,backendRequests:0,passed:true}));
    await context.close();
  } finally {await browser.close();}
}
