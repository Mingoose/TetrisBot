#!/bin/sh
# Train the three v2 value nets one after another, then run the T-slot probe.
#
#   cd training && nohup caffeinate -i ./run_v2.sh > data/logs/run_v2.out 2>&1 &
#
# Environment overrides: PY (default .venv/bin/python), DEVICE (auto|cpu|mps),
# THREADS (default 8), EXTRA (extra train_value.py args, e.g. "--epochs 1").
# Each step runs even if an earlier one failed; the final notification and
# data/logs/run_v2.out say which steps failed.

cd "$(dirname "$0")" || exit 1
PY=${PY:-.venv/bin/python}
DEVICE=${DEVICE:-auto}
THREADS=${THREADS:-8}
EXTRA=${EXTRA:-}
RUNS="data/runs/teacher-v2 data/runs/teacher-v2b"

# Check the setup before spending an hour on it.
mkdir -p data/logs models
[ -x "$PY" ] || { echo "No Python at $PY — create the venv (see HANDOFF.md)"; exit 1; }
"$PY" -c "import torch, numpy" || { echo "torch/numpy missing in $PY"; exit 1; }
for r in $RUNS; do
  [ -f "$r/meta.json" ] || { echo "Missing $r/meta.json — data not in place (see HANDOFF.md)"; exit 1; }
done

failed=""
step() {
  name=$1; shift
  echo "$(date '+%H:%M:%S') start $name"
  if "$@"; then echo "$(date '+%H:%M:%S') done  $name"
  else echo "$(date '+%H:%M:%S') FAILED $name"; failed="$failed $name"; fi
}

train() {
  out=$1; log=$2; shift 2
  "$PY" -u train_value.py $RUNS --threads "$THREADS" --device "$DEVICE" --out "$out" $EXTRA "$@" > "$log" 2>&1
}

step full  train models/value_v2.pt      data/logs/train-v2.log
step half  train models/value_v2_half.pt data/logs/train-v2-half.log --train-fraction 0.5
step c16   train models/value_v2_c16.pt  data/logs/train-v2-c16.log  --channels 16
step probe-full sh -c "'$PY' probe_value.py models/value_v2.pt > data/logs/probe-v2.txt 2>&1"
step probe-c16  sh -c "'$PY' probe_value.py models/value_v2_c16.pt > data/logs/probe-v2-c16.txt 2>&1"

if [ -z "$failed" ]; then msg="All three models trained"; else msg="Failed:$failed (see data/logs)"; fi
echo "$msg"
osascript -e "display notification \"$msg\" with title \"TetrisBot training finished\" sound name \"Glass\"" 2>/dev/null
