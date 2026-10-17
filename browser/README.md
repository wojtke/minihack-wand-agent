# Browser engine

The presentation runs the original **NLE 1.1.0 / NetHack** C/C++ engine as
WebAssembly. It uses the pinned MiniHack Wand of Death descriptions and task
rules. There is no game API, remote session, Python runtime or rewritten map
simulation. GitHub Pages serves the files; a Web Worker executes the game on
the visitor's device. Training still uses native Python MiniHack.

## Play or serve locally

Open [slide 3](https://wojtke.github.io/minihack-wand-agent/#3). All four levels
work directly in a current Chrome, Firefox or Safari. The first game downloads
about 4 MB of engine/data files, which the browser can cache. No account is
needed. The site does not train an agent in the browser.

For local website editing, any static HTTP server is enough:

```sh
python3 -m http.server 8000 --directory docs
# Open http://localhost:8000/#3
```

Do not open `index.html` as a `file://` URL: browsers restrict workers and
module loading there. SharedArrayBuffer, special cross-origin headers and
server-side execution are not required.

## Rebuild

Prebuilt artifacts are committed in `docs/engine/`; playing does not need this
toolchain. To rebuild, run the repository's `./setup.sh` first, then install
[Emscripten](https://emscripten.org/docs/getting_started/downloads.html):

```sh
git clone https://github.com/emscripten-core/emsdk.git ../emsdk
.venv/bin/python ../emsdk/emsdk.py install 4.0.10
.venv/bin/python ../emsdk/emsdk.py activate 4.0.10
source ../emsdk/emsdk_env.sh
source .venv/bin/activate
python browser/build.py
```

The build verifies the upstream archive's SHA-256 from `source.json`, makes a
fresh build under `.build/`, compiles the data with the wasm32 utilities, and
exports MiniHack's native tile atlas. It also writes a complete modified NLE
source archive and artifact hashes beside the executable. The source archive
is downloadable as [source.tar.gz](https://wojtke.github.io/minihack-wand-agent/engine/source.tar.gz).
It contains the original notices, port code and modified NLE sources.

Changes dated **2026-10-17**:

- Replace the native assembly context switch with Emscripten fibers and
  Asyncify. The original NLE start/step/yield protocol stays intact.
- Correct two existing C declarations to match their definitions
  (`set_random`, `nle_xputs`); WebAssembly checks function signatures.
- Add a C observation bridge and a WebAssembly CMake configuration. Invoke
  compiled build utilities through Node and remove local paths from binaries.
- Rebuild level/quest data for wasm32. MiniHack's dungeon graph is unchanged;
  its version header is converted from five 64-bit to five 32-bit words.
- Translate the Python reset, automatic prompt handling, action limits,
  termination and reward wrapper into `docs/engine/runtime.mjs`.

There are no gameplay, monster, map, RNG, reward or movement changes. The
browser shortcuts translate to the same 11 native inputs. The tile renderer
uses only public glyphs and crops to observed/remembered cells. Each restart
terminates the previous worker, freeing its heap and resetting native globals.
Game state is in memory only; reloading the page resets it.

## Verify against native MiniHack

```sh
python browser/native-traces.py
node browser/check-parity.cjs

# Optional browser UI and pixel tests:
npm install --no-save playwright
npx playwright install chromium firefox webkit
python3 -m http.server 8000 --directory docs
# In another terminal:
PRESENTATION_URL=http://localhost:8000/ node browser/check-ui.mjs
```

The native oracle records frozen-policy trajectories on five seeds per level,
plus timeouts and movement/death cases: 28 complete games, 8,350 actions and
8,378 states. The comparator checks every public state, reward and outcome,
including glyph IDs. Browser tests check all four levels, keyboard controls,
restarts, error recovery and sample rendered pixels.

On 2026-10-17, all comparisons passed; Chromium, Firefox and WebKit each passed
all four levels and 16 pixel comparisons with no game API requests. These checks establish
parity for the tested trajectories; they do not prove all possible trajectories
or imply that the included policy solves Pro.

## Licenses

The engine, its C bridge/fiber port and the NLE-derived runtime wrapper are
distributed under the **NetHack General Public License**, not the repository's
MIT license. MiniHack task descriptions retain Apache 2.0. The renderer's
artwork retains upstream terms. See [THIRD_PARTY.md](../THIRD_PARTY.md) and
[`licenses/`](../licenses/) for the full notices and source locations.
