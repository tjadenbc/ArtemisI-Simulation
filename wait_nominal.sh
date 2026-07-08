#!/bin/zsh
# Nominal-ready watcher: launch as a BACKGROUND task alongside a local run (run_mc.py).
# It waits for the nominal artifacts to be captured (a few min into the run) and EXITS,
# firing a completion notification -> the signal to pin the nominal to the cluster and
# submit the small run CONCURRENTLY, without waiting for the full local run to finish
# and without foreground polling. Usage: ./wait_nominal.sh outputs/<name> [timeout_s]
OUTDIR="$1"; TIMEOUT="${2:-900}"
[ -z "$OUTDIR" ] && { echo "usage: $0 outputs/<name> [timeout_s]"; exit 1; }
NR="$OUTDIR/nominal_results.json"; NT="$OUTDIR/nominal_targets.json"; NZ="$OUTDIR/nominal_traj.npz"
elapsed=0
while [ "$elapsed" -lt "$TIMEOUT" ]; do
  if [ -s "$NR" ] && [ -s "$NT" ] && [ -s "$NZ" ]; then
    # brief settle so the writes are complete, then confirm the nominal succeeded
    sleep 2
    ok=$(python3 -c "import json;print(json.load(open('$NR')).get('full_success'))" 2>/dev/null)
    echo "NOMINAL-READY $OUTDIR (full_success=$ok) — pin to cluster + submit the small run now."
    exit 0
  fi
  sleep 8; elapsed=$((elapsed+8))
done
echo "NOMINAL-TIMEOUT $OUTDIR after ${TIMEOUT}s — check the local run."; exit 2
