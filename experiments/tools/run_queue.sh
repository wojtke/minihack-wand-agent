#!/bin/zsh
# Runs every job of a queue file in order (one training at a time), then evaluates it.
#   zsh tools/run_queue.sh <queue> <stage>
#     queue file: runs/queue/<queue>.txt
#     stage: folder in runs/training/ for the new runs, e.g. 5-adaptacyjny-mix
# A line is: <run name> <ppo arguments...>. The file is re-read after each job, so jobs
# can be appended while the queue runs. Finished jobs (eval.json present in any stage) are skipped.
# Each run ends up in runs/training/<stage>/<run>/ with train.log, eval.json and eval.log.
# QUEUE_DEADLINE=<epoch seconds>: no job starts after that time.
# Progress: runs/queue/status-<queue>.txt (runs/ is gitignored)
cd "${0:A:h}/.."
source .venv/bin/activate
queue=${1:?queue name}
stage=${2:?stage folder in runs/training}
file=runs/queue/$queue.txt
statfile=runs/queue/status-$queue.txt
while true; do
  total=$(grep -cv '^\s*\(#\|$\)' $file)
  done_count=0; next=""
  while read -r name args; do
    [[ -z $name || $name == \#* ]] && continue
    if grep -qs '"Pro"' runs/training/*/$name/eval.json; then
      done_count=$((done_count+1))
    elif [[ -z $next ]]; then next="$name $args"; fi
  done < $file
  if [[ -z $next ]]; then
    print "ALL DONE: $done_count/$total jobs finished at $(date '+%H:%M')" > $statfile
    break
  fi
  if [[ -n $QUEUE_DEADLINE && $(date +%s) -gt $QUEUE_DEADLINE ]]; then
    print "DEADLINE: $done_count/$total jobs finished, no new job started at $(date '+%H:%M')" > $statfile
    break
  fi
  name=${next%% *}; args=${next#* }
  [[ $args == $name ]] && args=""
  run=runs/training/$stage/$name
  print "Finished $done_count/$total jobs. Running #$((done_count+1)): $name (started $(date '+%H:%M')). Log: $run/train.log" > $statfile
  rm -rf $run $run.train.log  # an interrupted job restarts from scratch
  mkdir -p ${run:h}  # ppo.py creates the run folder itself and refuses an existing one
  python -m wand_agent.ppo ${=args} --out $run > $run.train.log 2>&1
  mv $run.train.log $run/train.log
  python -m wand_agent.evaluate --model $run/latest.npz --games 200 --out $run/eval.json > $run/eval.log 2>&1
  print "$(date '+%m-%d %H:%M') finished $name" >> runs/queue/finished.txt
done
