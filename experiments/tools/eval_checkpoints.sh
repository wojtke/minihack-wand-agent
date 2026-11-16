#!/bin/zsh
# Evaluates checkpoints of finished runs for learning curves: the first checkpoint at or after
# every 1M steps, greedy, 200 games per level, 3 evaluations at a time. Finished ones are skipped.
#   zsh tools/eval_checkpoints.sh <run> [<run> ...]        run = folder name, e.g. adaptive-30m-s1
# Output: runs/training/<stage>/<run>/checkpoint-evals/step-*.json
cd "${0:A:h}/.."
source .venv/bin/activate
for name in "$@"; do
  run=(runs/training/*/$name(N/))
  [[ -z $run ]] && { print "no run $name" >&2; continue }
  python - $run <<'EOF'
import pathlib, re, sys
run = pathlib.Path(sys.argv[1])
steps = sorted(int(re.search(r"step-(\d+)", p.name).group(1)) for p in run.glob("step-*.npz"))
for m in range(1, 100):
    c = [s for s in steps if s >= m * 1_000_000]
    if c and c[0] < (m + 0.5) * 1_000_000:
        print(f"{run}/step-{c[0]:09d}.npz {run}/checkpoint-evals/step-{c[0]:09d}.json")
EOF
done | while read ck out; do grep -qs '"Pro"' $out || print "$ck $out"; done |
  xargs -P 3 -L 1 zsh -c 'mkdir -p ${1:h}; python -m wand_agent.evaluate --model $0 --games 200 --out $1 > ${1%.json}.log 2>&1'
print "DONE $(date +%H:%M)"
