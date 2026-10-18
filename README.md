# MiniHack Wand of Death

One shared agent for four MiniHack tasks, with a short interactive introduction to the environment.

**[Presentation and playable game](https://wojtke.github.io/minihack-wand-agent/)** — the original NLE engine runs entirely in the browser. No installation or game server is needed.

```text
experiments/   wand_agent/, configs/, checkpoints/, tests/, pyproject.toml, setup.sh
website/       slides, recordings and assets; browser/ contains the engine port and checks
licenses/      full upstream notices
```

## Setup for training

Requires Python 3.11, [uv](https://docs.astral.sh/uv/getting-started/installation/) and native build tools. On macOS:

```sh
xcode-select --install  # only if Command Line Tools are missing
brew install uv bison flex
export PATH="$(brew --prefix bison)/bin:$(brew --prefix flex)/bin:$PATH"
```

On Ubuntu/Debian (Windows: WSL2), install `build-essential bison flex git libncurses-dev` and uv. Tested on Apple Silicon macOS; Linux/WSL is not yet verified.

```sh
git clone https://github.com/wojtke/minihack-wand-agent.git
cd minihack-wand-agent/experiments
./setup.sh
source .venv/bin/activate
python -m unittest discover -s tests -v
python -m wand_agent.play --level Hard --seed 0
```

## Train and evaluate

Run from `experiments/`, with its environment activated:

```sh
python -m wand_agent.train --steps 6000 --out runs/smoke  # installation/learning check
python -m wand_agent.train --out runs/dqn               # fresh policy, 1M actions
python -m wand_agent.evaluate --model runs/dqn/latest.npz --games 100 --out runs/dqn-eval.json
python -m wand_agent.evaluate --games 100 --out runs/included-eval.json
```

The baseline is **Double DQN**, a **128 × 128 MLP**, uniform replay and equal action budgets across all four levels. [configs/dqn.json](experiments/configs/dqn.json) holds the settings; use `--config`, `--seed 7` or `--no-bonus` for comparisons. Add method implementations to the existing `wand_agent/` package when needed, with their settings in `configs/`; separate output directories keep runs independent.

The policy sees **463 features** from public observations: a local crop, visible monsters, remembered objects/exits, inventory, health, elapsed actions, prompts and previous inputs. This interface-specific encoder has limited memory. It receives no difficulty label, hidden state, route or action mask. Actions are the 11 native inputs `kljhunby,zf`. Success, death and the native horizon end bootstrapping. The only extra reward is **+0.01 per newly visited observed position**, capped at +2 per episode; initial and terminal positions earn no bonus. Evaluation is greedy and uses only native rewards.

Use a separate run directory per method/config/seed and retain the git revision used. Each run saves resolved `config.json`, completed `episodes.jsonl` (seeds, native returns and bonuses), `summary.json` and weights every 50,000 actions/on exit. Progress is printed to stdout; redirect it if you need a console log. Outputs are gitignored and existing run directories are refused. **Weights-only checkpoints cannot resume optimizer/replay state.** Replay uses about 400 MB, plus PyTorch/environments. Training seeds are below 100,000,000; evaluation defaults to 100,000,000 onward. Keep validation and final-test ranges separate and compare multiple training seeds. These are fixed terrains: different seeds do not establish transfer to new maps.

## Included checkpoint

[shared-best.npz](experiments/checkpoints/shared-best.npz) contains six NumPy arrays, loaded without pickle. It comes from **3,670,016 actions in an earlier multi-stage training procedure** with modified rewards, replay and exploration. **This starter does not reproduce that lineage; its default training configuration has only been smoke-tested.** The same weights are used on every level.

| Level | Historical wins / 1,000 games | Historical seeds (inclusive) |
|---|---:|---|
| Easy | 1,000 | 910000000–910000999 |
| Medium | 998 | 910010000–910010999 |
| Hard | 803 | 910020000–910020999 |
| Pro | 0 | 910030000–910030999 |

[Checkpoint metadata](experiments/checkpoints/shared-best.json) records hashes and historical results. A separate [20-game-per-level export check](experiments/checkpoints/export-validation.json), starting at seed 210000000, gave 20/20/12/0 wins; original/exported inference matched over 5,332 transitions. **Pro remains unsolved.** The Easy/Medium/Hard videos show complete wins; Pro shows only the first 60 actions. Recordings are accelerated/cropped. Full-map illustrations are presentation-only and are not given to the policy or player.

## Website and browser engine

From the repository root, Python **3.11.8+** is sufficient to assemble a local copy:

```sh
python3 website/browser/build.py
python3 -m http.server 8000 --directory .build/site
# Open http://localhost:8000/#3
```

The build downloads a versioned engine bundle, verifies its SHA-256 and per-file manifest, and combines it with the tracked website sources. The [Pages workflow](.github/workflows/pages.yml) does the same. Generated files stay in `.build/`; the release pinned in [source.json](website/browser/source.json) must remain available. Serving `website/` directly is insufficient. During play, all engine/data files come from the same static site; no backend or external game API is used.

On slide 3: **WASD + Q/E/Z/C** move/aim, **P** picks up, **R** uses a wand, **F** selects/fires, **Space** restarts the same seed. Hints are optional; each restart replaces the Web Worker and resets the native engine.

To rebuild the engine, first run `experiments/setup.sh`, install/activate [Emscripten 4.0.10](https://emscripten.org/docs/getting_started/downloads.html), then run from the repository root:

```sh
source experiments/.venv/bin/activate
python website/browser/build.py --rebuild
python website/browser/native-traces.py
node website/browser/check-parity.cjs
# Optional UI/pixel checks: install Playwright and its browsers, serve .build/site, then:
PRESENTATION_URL=http://localhost:8000/ node website/browser/check-ui.mjs
```

Rebuilding/parity needs the experiments environment because it uses MiniHack's data, renderer and reference policy; ordinary assembly/Pages does not. A rebuild creates `.build/browser-engine.tar.gz` and prints its checksum. Publish it as a new release asset and update `source.json` before deploying engine changes. The native oracle covers 28 games / 8,350 actions / 8,378 states; browser checks cover all levels, controls, restarts, failure recovery and rendered pixels. This is tested-trajectory parity, not proof for every possible game.

## Sources and licenses

Original agent/presentation code and weights: [MIT](LICENSE). The **engine, C port and NLE-derived runtime wrapper** use the [NetHack General Public License](licenses/NetHack.txt). Complete modified engine source, notices and rebuild instructions are distributed at [engine/source.tar.gz](https://wojtke.github.io/minihack-wand-agent/engine/source.tar.gz) alongside the executable. This is an unofficial modified distribution, not affiliated with upstream authors.

Port changes dated **2026-10-17**: native context switching replaced by Emscripten fibers/Asyncify; two C declarations corrected (`set_random`, `nle_xputs`); C observation bridge/CMake build added with Node-hosted utilities and local-path stripping; level/quest archives rebuilt for wasm32 (dungeon version header converted from five 64-bit to five 32-bit words); Python reset/prompt/reward/termination rules translated into `runtime.mjs`. **2026-10-18**: source layout and release packaging reorganized. Gameplay, maps, RNG, actions and rewards are unchanged.

- **MiniHack**, Facebook, Inc. and affiliates: [pinned source](https://github.com/facebookresearch/minihack/tree/09eaa7ee45c567850668d3743d7fe1853917d98f), [Apache 2.0](licenses/MiniHack-Apache-2.0.txt). Task descriptions, renderer and tile artwork come from MiniHack; screenshots, icons and the tile atlas retain upstream terms.
- **NLE/NetHack**: [NLE 1.1.0 source](https://github.com/NetHack-LE/nle/tree/v1.1.0). Bundled **bzip2**, Julian R Seward ([license](licenses/bzip2.txt)); **libtmt**, Rob King ([BSD notice](licenses/libtmt.txt)); **Emscripten** ([license](licenses/Emscripten.txt)); compiled **LLVM** libraries ([license](licenses/LLVM.txt)) and **musl** ([notices](licenses/musl.txt)). Other installed Python dependencies retain their own licenses.
- Environment references: [MiniHack tasks](https://minihack.readthedocs.io/en/stable/envs/skills/wod.html), [pinned task definitions](https://github.com/facebookresearch/minihack/blob/09eaa7ee45c567850668d3743d7fe1853917d98f/minihack/envs/skills_wod.py), [NLE time penalty](https://github.com/NetHack-LE/nle/blob/v1.1.0/nle/env/tasks.py). Text and code were prepared with AI assistance.
