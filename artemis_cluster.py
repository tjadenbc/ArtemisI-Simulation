"""Multi-node sharded Monte Carlo driver for artemis1 on a SLURM cluster.

Three stages, chained by submit_artemis_sharded.sh with sbatch dependencies:

  setup  — run the nominal into the master outdir (main_parallel indices=[]):
           writes nominal_results.json / nominal_traj.npz (+ nominal_targets.json
           if any). No trials.
  shard  — one per node: copy the master nominal artifacts into a shard-local
           outdir (so the shard SKIPS the nominal and never writes to a shared
           results.csv), then run the strided trial subset
           shard_id, shard_id+n_shards, ...  with the SAME seed. main_parallel
           pre-generates all n perturbations in trial order, so trial i maps to
           the identical perturbation it would have in a single-node run — a
           sharded run is trial-for-trial comparable to a local one.
  merge  — concatenate shard CSVs (sorted by trial, deduped), gather per-trial
           debug JSONs into master/trials/, leaving the master outdir in exactly
           the layout artemis_outputs.py expects.

The captured targeting state is generalized into ONE
nominal_targets.json (_NOMINAL_TARGETS — e.g. the recovery-zone and DRI-schedule
captures); it is pinned
into each shard with the other nominal artifacts (the CR3BP DRO itself is
computed per-process; SPLASH_TARGET is a module constant). The __main__ guard makes
this spawn-safe (the worker pools main_parallel spawns do NOT re-run the
dispatch — that is the fork-bomb an UNGUARDED launcher would cause).

Resume: re-submitting is safe — setup skips a captured nominal, shards fill only
their missing trials (gap-safe), merge is idempotent.

Usage (normally via submit_artemis_sharded.sh, but callable by hand):
  python3 artemis_cluster.py setup <outdir> <n> <seed>
  python3 artemis_cluster.py shard <outdir> <n> <seed> <shard_id> <n_shards> <workers>
  python3 artemis_cluster.py merge <outdir> <n_shards>
"""
import os
import shutil
import sys

# Master artifacts pinned into each shard so shards skip the nominal and share
# identical (deterministic) state. nominal_targets.json carries the nominal-
# captured targeting state (_NOMINAL_TARGETS, e.g. recovery zone + DRI schedule);
# copied when present.
_NOMINAL_FILES = ("nominal_results.json", "nominal_targets.json", "nominal_traj.npz")


def _shard_dir(outdir, k):
    return os.path.join(outdir, f"shard_{k:02d}")


def do_setup(outdir, n, seed):
    import artemis1
    artemis1.main_parallel(n=n, outdir=outdir, seed=seed, workers=1, indices=[])
    if not os.path.exists(os.path.join(outdir, "nominal_results.json")):
        print("WARNING: setup did not produce nominal_results.json")
    print("setup complete")


def do_shard(outdir, n, seed, shard_id, n_shards, workers):
    sd = _shard_dir(outdir, shard_id)
    os.makedirs(sd, exist_ok=True)
    for f in _NOMINAL_FILES:
        src = os.path.join(outdir, f)
        dst = os.path.join(sd, f)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
    if not os.path.exists(os.path.join(sd, "nominal_results.json")):
        raise SystemExit("shard: master nominal artifacts missing — run setup first")
    import artemis1
    idx = list(range(shard_id, n, n_shards))
    print(f"shard {shard_id}/{n_shards}: {len(idx)} trials "
          f"({idx[:3]}...{idx[-1:]}) on {workers} workers")
    artemis1.main_parallel(n=n, outdir=sd, seed=seed, workers=workers, indices=idx)
    print(f"shard {shard_id} complete")


def do_merge(outdir, n_shards):
    import pandas as pd
    frames = []
    missing = []
    os.makedirs(os.path.join(outdir, "trials"), exist_ok=True)
    for k in range(n_shards):
        sd = _shard_dir(outdir, k)
        csv = os.path.join(sd, "results.csv")
        if not os.path.exists(csv):
            missing.append(k)
            continue
        frames.append(pd.read_csv(csv))
        tdir = os.path.join(sd, "trials")
        if os.path.isdir(tdir):
            for fn in os.listdir(tdir):
                dst = os.path.join(outdir, "trials", fn)
                if not os.path.exists(dst):  # trial_nominal.json identical; first wins
                    shutil.copy2(os.path.join(tdir, fn), dst)
    if missing:
        print(f"WARNING: missing shard results: {missing}")
    if not frames:
        raise SystemExit("merge: no shard results found")
    df = pd.concat(frames, ignore_index=True)
    if "trial" in df.columns:
        df = df.drop_duplicates(subset="trial", keep="first").sort_values("trial")
    df = df.reset_index(drop=True)
    df.to_csv(os.path.join(outdir, "results.csv"), index=False)
    ok = df.get("full_success")
    n = len(df)
    nsucc = int(ok.fillna(False).astype(bool).sum()) if ok is not None else 0
    print(f"merge complete: {n} trials, full_success {nsucc}/{n}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "setup":
        do_setup(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif mode == "shard":
        do_shard(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]),
                 int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]))
    elif mode == "merge":
        do_merge(sys.argv[2], int(sys.argv[3]))
    else:
        raise SystemExit(f"unknown mode {mode!r} (setup|shard|merge)")
