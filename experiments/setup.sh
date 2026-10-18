#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
command -v uv >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/'; exit 1; }
if [ ! -x .venv/bin/python ]; then uv venv --python 3.11 .venv; fi
uv pip install --python .venv/bin/python cmake==3.31.6 ninja==1.11.1.4
PATH="$PWD/.venv/bin:$PATH" uv pip install --python .venv/bin/python -e .
.venv/bin/python -c 'from wand_agent.native import make_env; e=make_env("Easy"); e.reset(seed=0); e.close(); print("MiniHack ready. Activate with: source .venv/bin/activate")'
