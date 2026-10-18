const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const create = require('../../.build/site/engine/wand.js');
const root = path.resolve(__dirname, '../..');
(async () => {
  const {WandGame} = await import('../../.build/site/engine/runtime.mjs');
  const levels = JSON.parse(fs.readFileSync(path.join(root, '.build/site/engine/levels.json')));
  const cases = JSON.parse(fs.readFileSync(path.join(root, '.build/native-traces.json')));
  let count = 0;
  for (const test of cases) {
    const module = await create({print: () => {}, printErr: console.error});
    const game = new WandGame(module, test.level, test.seed, levels[test.level],
      fs.readFileSync(path.join(root, '.build/site/engine', test.level.toLowerCase() + '.nhdat')));
    for (let step = 0; step < test.states.length; step++) {
      const got = step ? game.step(test.keys[step - 1]) : game.state();
      const expected = {...test.states[step]};
      delete got.last_key;
      delete expected.last_key;
      delete expected.image;
      try { assert.deepEqual(got, expected); }
      catch (e) { console.error('Mismatch:', test.level, test.seed, test.kind, 'step', step); throw e; }
      count++;
    }
    console.log(test.level, test.seed, test.kind, game.steps, game.status);
  }
  console.log(JSON.stringify({games: cases.length, states: count, matched: true}));
})().catch(e => {console.error(e); process.exitCode = 1;});
