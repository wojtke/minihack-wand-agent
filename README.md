# MiniHack Wand of Death

One shared agent for the Easy, Medium, Hard and Pro tasks, plus a short interactive introduction to the environment.

**[Open the presentation](https://wojtke.github.io/minihack-wand-agent/)** · **[Download the PDF](https://wojtke.github.io/minihack-wand-agent/assets/wand-of-death.pdf)**

GitHub Pages serves the slides, gameplay recordings and PDF. To play the actual MiniHack game, run the local server below.

## Setup

Python 3.11 and [uv](https://docs.astral.sh/uv/getting-started/installation/) are required. MiniHack/NLE compile native code.

macOS prerequisites (Homebrew):

```sh
xcode-select --install  # only if Command Line Tools are missing
brew install uv bison flex
export PATH="$(brew --prefix bison)/bin:$(brew --prefix flex)/bin:$PATH"
```

Ubuntu/Debian prerequisites (Windows: use WSL2):

```sh
sudo apt-get update
sudo apt-get install build-essential bison flex git libncurses-dev
# Install uv using its official instructions linked above.
```

Then, from a terminal:

```sh
git clone https://github.com/wojtke/minihack-wand-agent.git
cd minihack-wand-agent
./setup.sh
source .venv/bin/activate
```

Tested on Apple Silicon macOS. Linux/WSL instructions are provided but have not been tested here. The setup pins MiniHack, NLE, NumPy, Gymnasium and PyTorch; build tools stay inside the virtual environment. Run the commands below from the repository root.

## Play and watch

```sh
python -m wand_agent.serve --open
```

Opens **http://127.0.0.1:8768** with real playable games on slide 3. Click the board: **WASD + Q/E/Z/C** move/aim; **P** picks up; **R** uses a wand; **F** selects/fires; **Space** restarts the same seed. Hints are optional. Stop the server with Ctrl+C. Use `--port 8769` if the port is busy.

Watch the included agent in the terminal:

```sh
python -m wand_agent.play --level Hard --seed 0
```

## Train and evaluate

```sh
# Quick installation/learning check; not a useful trained policy:
python -m wand_agent.train --steps 6000 --out runs/smoke

# Fresh shared policy, one million primitive actions across all four levels:
python -m wand_agent.train --out runs/dqn

# Evaluate any checkpoint with native rewards, no exploration bonus:
python -m wand_agent.evaluate --model runs/dqn/latest.npz --games 100 --out runs/dqn-eval.json

# Evaluate the included checkpoint:
python -m wand_agent.evaluate --games 100 --out runs/included-eval.json
```

The trainer uses **Double DQN**, a **128 × 128 MLP**, uniform replay and equal environment steps per level. Its only reward addition is a capped bonus for visiting new observed positions. Use `--no-bonus` for native rewards only; change the seed with `--seed 7`. Settings are in [configs/dqn.json](configs/dqn.json); details are in [METHOD.md](METHOD.md).

Runs log completed episodes, native returns, bonuses, seeds and learning progress. Policies are saved every 50,000 steps and on exit. Ctrl+C saves the latest policy. Checkpoints contain weights only; optimizer/replay resume is not implemented. Output directories must be new to avoid overwriting results. CPU training needs roughly 400 MB for replay, plus PyTorch and the environments.

## Included agent

The included policy comes from earlier experiments (3,670,016 training actions). **The minimal trainer above is a fresh baseline, not a reproduction of that full training history.**

| Level | Wins / 1,000 held-out games |
|---|---:|
| Easy | 1,000 |
| Medium | 998 |
| Hard | 803 |
| Pro | 0 |

These are the original checkpoint's recorded evaluation results. All levels use the same weights. Pro remains unsolved, and seed variation on these fixed terrains does not establish transfer to new maps. See the [checkpoint notes](checkpoints/README.md).

## Checks and contents

```sh
python -m unittest discover -s tests -v
```

`wand_agent/` contains the environment wrapper, observation encoder, trainer, evaluator and local server. `docs/` is the static website published by GitHub Pages; edit its HTML/CSS/JS directly. The PDF uses print CSS: start the server, open the site and choose **Print → Save as PDF** (landscape, background graphics, no browser headers/footers).

Project code and weights use the MIT license. Game artwork and upstream components retain their own licenses; see [THIRD_PARTY.md](THIRD_PARTY.md). This project is not affiliated with the MiniHack or NetHack authors.
