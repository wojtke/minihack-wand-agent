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
python -m wand_agent.evaluate --model random --games 1000 --out runs/random-eval.json  # baseline
```

`evaluate` and `play` accept a checkpoint path or `random` and use the shared interface in `wand_agent/agents.py`. `wand_agent/vector.py` steps several games in parallel processes for on-policy training.

**PPO** ([wand_agent/ppo.py](experiments/wand_agent/ppo.py), settings in [configs/ppo.json](experiments/configs/ppo.json)) trains one actor-critic on 7 parallel games (one Easy, two of each other level; Easy is a fixed 3-input solution) with clipped updates and GAE. A run writes `config.json`, `episodes.jsonl`, `metrics.jsonl` (entropy, approx. KL, clip fraction, explained variance, losses per update), `summary.json` and checkpoints. `configs/ppo-cartpole.json` checks the learning code on CartPole-v1, where a correct PPO reaches a mean return near 500. `--lstm N` adds memory (an LSTM between the shared trunk and both heads); `configs/ppo-memory.json` is a remember-the-cue task that a policy without memory can only guess (50%) and a working LSTM solves. `--rnd` adds a curiosity bonus (Random Network Distillation): a second critic head learns the bonus, which is logged per update but never used in evaluation. Training-only rewards: `--wand-bonus` (picking up a wand), `--death-penalty`, `--align-bonus` (potential for holding a wand in line with a monster). `--macro` replaces the inputs `z` and `f` by eight "zap in direction d" actions played inside the wrapper (evaluation and play detect such checkpoints by their 17 outputs). `--sil` adds self-imitation of won games; `--init-from` starts from another PPO checkpoint.

```sh
python -m wand_agent.ppo --config configs/ppo-cartpole.json --out runs/ppo-cartpole  # ~20 s
python -m wand_agent.ppo --out runs/ppo                                             # all levels, 10M actions
python -m wand_agent.ppo --levels Easy --envs-per-level 8 --out runs/ppo-easy       # one level
python -m wand_agent.ppo --lstm 128 --out runs/ppo-lstm                             # with memory
python -m wand_agent.ppo --rnd --out runs/ppo-rnd                                   # with curiosity bonus
python -m wand_agent.ppo --config configs/ppo-memory.json --lstm 32 --out runs/memory  # memory check, ~40 s
python -m wand_agent.evaluate --model runs/ppo/latest.npz --games 100 --out runs/ppo-eval.json  # add --sample to draw actions
```

The baseline is **Double DQN**, a **128 × 128 MLP**, uniform replay and equal action budgets across all four levels. [configs/dqn.json](experiments/configs/dqn.json) holds the settings; use `--config`, `--seed 7` or `--no-bonus` for comparisons. Add method implementations to the existing `wand_agent/` package when needed, with their settings in `configs/`; separate output directories keep runs independent.

The policy sees **463 features** from public observations: a local crop, visible monsters, remembered objects/exits, inventory, health, elapsed actions, prompts and previous inputs. This interface-specific encoder has limited memory. It receives no difficulty label, hidden state, route or action mask. Actions are the 11 native inputs `kljhunby,zf`. Success, death and the native horizon end bootstrapping. The only extra reward is **+0.01 per newly visited observed position**, capped at +2 per episode; initial and terminal positions earn no bonus. Evaluation is greedy and uses only native rewards.

Use a separate run directory per method/config/seed and retain the git revision used. Each run saves resolved `config.json`, completed `episodes.jsonl` (seeds, native returns and bonuses), `summary.json` and weights every 50,000 actions/on exit. Progress is printed to stdout; redirect it if you need a console log. Outputs are gitignored and existing run directories are refused. **Weights-only checkpoints cannot resume optimizer/replay state.** Replay uses about 400 MB, plus PyTorch/environments. Training seeds are below 100,000,000; evaluation defaults to 100,000,000 onward. Keep validation and final-test ranges separate and compare multiple training seeds. These are fixed terrains: different seeds do not establish transfer to new maps.

## Watch games and training runs

Three commands write self-contained HTML pages in the style of the website (no server, no network; open the file in a browser). Run from `experiments/`:

```sh
python -m wand_agent.replay --model runs/ppo/latest.npz --level Pro --seed 100000003 --out runs/viz/pro.html
python -m wand_agent.replay --model runs/ppo/step-*.npz runs/ppo/latest.npz --level Hard --seed 100000006 --out runs/viz/hard-steps.html
python -m wand_agent.inspect_games --model runs/ppo/latest.npz --level Pro --games 100 --out runs/viz/pro-games.html
python -m wand_agent.training_report runs/ppo-s1 runs/ppo-s2 runs/ppo-s3 --out runs/viz/train.html
```

- **`replay`** plays one game and shows it step by step: the map in MiniHack tiles (or ASCII) with the recent path or a visit heatmap, the chosen action and the full action distribution, critic values (both heads with `--rnd`), policy entropy, game messages and marks for "wand seen", "wand picked up", "minotaur seen" and "zap while holding the wand". Several `--model` paths play the same seed and the page switches between them, e.g. checkpoints from one run. Greedy games are the games `evaluate` plays for that seed. With `--sample`, the generator is seeded by the game seed, so a sampled game is repeatable on its own but differs from `evaluate --sample`, which shares one generator across a level. `--json` also saves the recorded game. DQN checkpoints show Q-values; `random` works too.
- **`inspect_games`** plays `--games` evaluation seeds in `--workers` processes (default 2) and reports the event funnel, how games ended, the stage where each lost game stopped, and a coverage map over all games: share of games that visited each square, where the wand was first seen and where games ended. A few lost games from different stages (`--examples`) are embedded with the step-by-step viewer; the table gives a `replay` command for the others. The coverage map is most useful on Hard and Pro, whose layout is fixed.
- **`on_item`**: a PPO checkpoint trained with the extra "object under the player" input (464 inputs) is recognised from the checkpoint, and `replay` and `inspect_games` feed it that input; `--on-item` forces it. The viewer then also shows when that input was 1.
- **`training_report`** reads `episodes.jsonl` and `metrics.jsonl`. It plots win rate and ending (win, death, action limit) over the last `--window` training games per level, mean game length, and PPO statistics (entropy, explained variance, KL, clip fraction, value loss, RND bonus, SIL). Runs whose names differ only in a final `-sN` are grouped as seeds (mean and min–max range). The table shows the configuration settings that differ between runs, the step at which each level first reached a threshold, and evaluation wins from a sibling `<run>-eval.json`. Training games use the sampled policy and training seeds, so these rates are not evaluation results.

## Included checkpoint

[shared-best.npz](experiments/checkpoints/shared-best.npz) contains six NumPy arrays, loaded without pickle. It comes from **3,670,016 actions in an earlier multi-stage training procedure** with modified rewards, replay and exploration. **This starter does not reproduce that lineage; its default training configuration has only been smoke-tested.** The same weights are used on every level.

| Level | Historical wins / 1,000 games | Historical seeds (inclusive) |
|---|---:|---|
| Easy | 1,000 | 910000000–910000999 |
| Medium | 998 | 910010000–910010999 |
| Hard | 803 | 910020000–910020999 |
| Pro | 0 | 910030000–910030999 |

[Checkpoint metadata](experiments/checkpoints/shared-best.json) records hashes and historical results. A separate [20-game-per-level export check](experiments/checkpoints/export-validation.json), starting at seed 210000000, gave 20/20/12/0 wins; original/exported inference matched over 5,332 transitions. **Pro remains unsolved.** The Easy/Medium/Hard videos show complete wins; Pro shows only the first 60 actions. Recordings are accelerated/cropped. The random-agent clips on slide 3 come from [experiments/tools/record_random.py](experiments/tools/record_random.py) (uniform over the 11 inputs, seeds fixed in advance); seeds and outcomes are in [website/assets/random-recordings.json](website/assets/random-recordings.json). Full-map illustrations are presentation-only and are not given to the policy or player.

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
