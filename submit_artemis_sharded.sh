#!/bin/bash
# Submit a SHARDED, MULTI-NODE Artemis I Monte Carlo to the SLURM cluster.
#
#   ./submit_artemis_sharded.sh <outdir> <n_trials> [seed] [n_shards] [workers]
#
# Chains three jobs by sbatch dependency (afterok):
#   1. setup  — 1 task: capture the nominal into <outdir> (no trials).
#   2. shard  — array of <n_shards> tasks, one per node, <workers> cpus each:
#               disjoint strided trial subsets via artemis_cluster.py shard.
#   3. merge  — 1 task: concatenate shard CSVs (sorted/deduped) + gather trials/.
#
# Use this only for LARGE runs (thousands of trials across nodes). For <=~200
# trials the single-node submit_artemis_mc.sh is simpler.
#
# Defaults: seed 37, 17 shards (the cluster's 17 nodes), 16 workers/node ->
# 272 cores. At ~50 s/trial on the EPYCs, 5,000 trials ~= 5000/272*50 ~= 15 min
# of compute (plus queue/setup). Resume-safe: re-submit to fill gaps.
#
# PREP (one-time on the cluster), copy into $PROJ:
#   artemis1.py  lunar_gravity_coeffs.py  artemis_cluster.py
#   ($PY = a Python 3.11+ venv with numpy/scipy/pandas/matplotlib)
#
# NOTE: the cluster DEFAULT job time limit is 1 h — every stage pins --time.
set -euo pipefail

OUTDIR=${1:?usage: submit_artemis_sharded.sh <outdir> <n_trials> [seed] [n_shards] [workers]}
NTRIALS=${2:?usage: submit_artemis_sharded.sh <outdir> <n_trials> [seed] [n_shards] [workers]}
SEED=${3:-37}
NSHARDS=${4:-17}
WORKERS=${5:-16}

PY=${AR1_PY:-$HOME/venv/bin/python}       # a Python 3.11+ venv with numpy/scipy/pandas/matplotlib
PROJ=${AR1_PROJECT:-$HOME/artemis1_project}   # must contain artemis1.py, lunar_gravity_coeffs.py, artemis_cluster.py
LOGS=$PROJ/slurm_logs
mkdir -p "$LOGS"
cd "$PROJ"

# 0a) PRE-SUBMIT PIN GUARD: make a silent nominal RE-DERIVATION
# structurally impossible rather than only catching it after a full run burns. For any run > 25 trials,
# REFUSE TO SCHEDULE unless the laptop-derived nominal is already pinned in THIS EXACT outdir — because
# setup would otherwise re-derive on the cluster (different scipy/numpy can converge a DIFFERENT return
# branch and silently shift the whole fleet). The 25-trial tier is EXEMPT — that's where the reference
# nominal is born. Deliberate re-derive: ALLOW_DERIVE=1 ./submit_artemis_sharded.sh ...  The required
# set is artemis_cluster.py's _NOMINAL_FILES (nominal_results/targets/traj — nominal_targets.json carries
# the OD-filter chol + return-branch targets, so it is branch-sensitive and MUST be pinned). Belt AND
# suspenders: keep diffing the fetched-back nominal against the laptop source after the run lands too.
if [ "$NTRIALS" -gt 25 ] && [ "${ALLOW_DERIVE:-0}" != "1" ]; then
    _MISSING=""
    for _f in nominal_results.json nominal_targets.json nominal_traj.npz; do
        [ -f "$OUTDIR/$_f" ] || _MISSING="$_MISSING $_f"
    done
    if [ -n "$_MISSING" ]; then
        echo "PREFLIGHT_MISSING: '$OUTDIR' is NOT pinned — missing:$_MISSING" >&2
        echo "  REFUSING TO SCHEDULE ($NTRIALS > 25 trials): cluster setup would RE-DERIVE the nominal," >&2
        echo "  which can converge a different branch and silently shift the fleet." >&2
        echo "  FIX: scp the laptop-derived nominal (all 3 files) into \$PROJ/$OUTDIR, then resubmit." >&2
        echo "  Or, to DELIBERATELY re-derive on the cluster: ALLOW_DERIVE=1 $0 $*" >&2
        exit 3
    fi
    echo "PREFLIGHT_OK: '$OUTDIR' pinned (nominal_results/targets/traj present) — setup will skip capture."
elif [ "${ALLOW_DERIVE:-0}" = "1" ]; then
    echo "PREFLIGHT_SKIP: ALLOW_DERIVE=1 — deliberately allowing cluster nominal re-derivation for '$OUTDIR'."
fi

# BLAS pinned to 1 thread/worker (avoid oversubscription); --export carries the
# env into the spawned worker processes on each node.
BLAS="OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,VECLIB_MAXIMUM_THREADS=1,MKL_NUM_THREADS=1"

# 0) SERIALIZE cluster runs: the new
# chain's first job depends on EVERY job already queued/running under this user, so runs never
# interleave — each run gets the full node width with a clean ETA, strictly first-come-first-served.
# %A (array-master id) NOT %i (pending array elements print bracket tokens, invalid in dependency
# lists); afterany NOT afterok (a failed/cancelled prior run must not deadlock the next chain).
EXISTING=$(squeue -u "$USER" -h -o %A | sort -un | paste -sd: -)
SERIAL_DEP=""
[ -n "$EXISTING" ] && SERIAL_DEP="--dependency=afterany:$EXISTING" && echo "serializing behind: $EXISTING"

# 1) setup — nominal capture (serial, light).
SETUP_ID=$(sbatch --parsable --time=2:00:00 $SERIAL_DEP \
    --job-name=art1-setup --nodes=1 --ntasks=1 --cpus-per-task=1 \
    --export=ALL,$BLAS --output="$LOGS/art1-setup-%j.out" \
    --wrap "$PY artemis_cluster.py setup $OUTDIR $NTRIALS $SEED")
echo "setup  job: $SETUP_ID"

# 2) shard — one array task per node; each runs its strided trial subset.
SHARD_ID=$(sbatch --parsable --time=8:00:00 \
    --dependency=afterok:$SETUP_ID \
    --job-name=art1-shard --array=0-$((NSHARDS-1)) \
    --nodes=1 --ntasks=1 --cpus-per-task="$WORKERS" \
    --export=ALL,$BLAS --output="$LOGS/art1-shard-%A_%a.out" \
    --wrap "$PY artemis_cluster.py shard $OUTDIR $NTRIALS $SEED \$SLURM_ARRAY_TASK_ID $NSHARDS $WORKERS")
echo "shard array: $SHARD_ID  (tasks 0-$((NSHARDS-1)))"

# 3) merge — runs only if the whole shard array succeeds.
MERGE_ID=$(sbatch --parsable --time=1:00:00 \
    --dependency=afterok:$SHARD_ID \
    --job-name=art1-merge --nodes=1 --ntasks=1 --cpus-per-task=1 \
    --export=ALL,$BLAS --output="$LOGS/art1-merge-%j.out" \
    --wrap "$PY artemis_cluster.py merge $OUTDIR $NSHARDS")
echo "merge  job: $MERGE_ID"

echo
echo "monitor:  squeue -u \$USER"
echo "logs:     $LOGS/art1-{setup,shard,merge}-*.out"
echo "result:   $OUTDIR/results.csv  (after merge; dashboard: artemis_outputs.main('$OUTDIR'))"
echo
echo "if the shard array partially fails, just re-run this same command —"
echo "setup skips the captured nominal, shards fill only missing trials, merge re-runs."
