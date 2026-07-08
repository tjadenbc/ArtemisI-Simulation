#!/bin/zsh
# Fetch a completed cluster run's artifacts back locally so the dashboard can be
# generated WITHOUT re-running the nominal (uses the run's own nominal via
# nominal_traj.npz phase boundaries). Run after a cluster run's results.csv lands.
# Usage: ./fetch_cluster_run.sh outputs/<name>
# Expects an SSH host alias for the cluster (configure it in ~/.ssh/config, or set
# CLUSTER_SSH); the project lives at $AR1_PROJECT on the cluster (relative to the
# remote home dir by default).
set -e
OUTDIR="$1"
[ -z "$OUTDIR" ] && { echo "usage: $0 outputs/<name>"; exit 1; }
HOST="${CLUSTER_SSH:-cluster}"
REMOTE="${AR1_PROJECT:-artemis1_project}/$OUTDIR"   # relative to the remote user's home dir
mkdir -p "$OUTDIR"
for f in results.csv nominal_traj.npz nominal_results.json nominal_targets.json oem_residuals.json; do
  if scp -o ConnectTimeout=25 "$HOST:$REMOTE/$f" "$OUTDIR/$f" 2>/dev/null; then
    echo "  fetched $f"
  else
    echo "  (skip: no $f on cluster)"
  fi
done
echo "done -> $OUTDIR  (generate dashboard: python3 -c \"import artemis_outputs; artemis_outputs.main('$OUTDIR')\")"
