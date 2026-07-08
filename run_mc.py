"""Guarded launcher for an Artemis I Monte Carlo (local OR single cluster node).

ALWAYS launch the parallel MC through this file (or `python3 -c`), NEVER an
unguarded script: macOS/`spawn` re-imports the launcher in every worker, and an
unguarded `main_parallel(...)` call recursively spawns pools (a fork-bomb). The
`if __name__ == "__main__"` guard below makes it spawn-safe.

Usage:
  python3 run_mc.py <outdir> <n_trials> [seed] [workers]

Tip: pin BLAS to one thread per worker to avoid oversubscription with many
workers, e.g.:
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  MKL_NUM_THREADS=1 python3 run_mc.py outputs/artemis1_mc200 200 37 16
"""
import sys


def main():
    if len(sys.argv) < 3:
        print("usage: python3 run_mc.py <outdir> <n_trials> [seed] [workers]")
        raise SystemExit(2)
    outdir = sys.argv[1]
    n = int(sys.argv[2])
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 37
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else None
    import artemis1
    artemis1.main_parallel(n=n, outdir=outdir, seed=seed, workers=workers)


if __name__ == "__main__":
    main()
