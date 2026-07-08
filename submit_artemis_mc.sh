#!/bin/bash
# Submit a SINGLE-NODE Artemis I Monte Carlo to the SLURM cluster.
#
#   ./submit_artemis_mc.sh <outdir> <n_trials> [seed] [workers]
#
# One node, <workers> processes via run_mc.py (spawn-safe — __main__-guarded, so
# no fork-bomb). 200 trials fit easily on one node (~16 cores x ~50 s/trial on the
# cluster's EPYCs ≈ ~10 min); the sharded setup→shard→merge pipeline is only
# needed for the multi-thousand-trial runs across nodes.
#
# PREP (one-time on the cluster):
#   - copy artemis1.py, lunar_gravity_coeffs.py, run_mc.py into $PROJ
#   - $PY must have numpy / scipy / pandas / matplotlib (a Python 3.11+ venv;
#     artemis1.py is 3.11-compatible)
#
# NOTE: the cluster DEFAULT job time limit is 1 h — we request --time explicitly.
set -euo pipefail

OUTDIR=${1:?usage: submit_artemis_mc.sh <outdir> <n_trials> [seed] [workers]}
NTRIALS=${2:?usage: submit_artemis_mc.sh <outdir> <n_trials> [seed] [workers]}
SEED=${3:-37}
WORKERS=${4:-16}
PY=${AR1_PY:-$HOME/venv/bin/python}       # a Python 3.11+ venv with numpy/scipy/pandas/matplotlib
PROJ=${AR1_PROJECT:-$HOME/artemis1_project}   # must contain artemis1.py, lunar_gravity_coeffs.py, run_mc.py
LOGS=$PROJ/slurm_logs
mkdir -p "$LOGS"
cd "$PROJ"

# BLAS pinned to 1 thread/worker (avoid oversubscription); --export carries it
# into the spawned workers. run_mc.py is __main__-guarded -> spawn-safe.
JOB_ID=$(sbatch --parsable --time=4:00:00 \
    --job-name=art1-mc --nodes=1 --ntasks=1 --cpus-per-task="$WORKERS" \
    --export=ALL,OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,VECLIB_MAXIMUM_THREADS=1,MKL_NUM_THREADS=1 \
    --output="$LOGS/art1-mc-%j.out" \
    --wrap "$PY run_mc.py $OUTDIR $NTRIALS $SEED $WORKERS")
echo "submitted job: $JOB_ID"
echo "monitor:  squeue -u \$USER     logs: $LOGS/art1-mc-$JOB_ID.out"
echo "results:  $OUTDIR/results.csv  (checkpointed every trial; resumable)"
