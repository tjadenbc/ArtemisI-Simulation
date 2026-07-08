# Artemis I Monte Carlo Simulation

A physics-integrated Monte Carlo that estimates **Artemis I's probability of
mission success** by flying the entire uncrewed mission — launch pad to Pacific
splashdown — as one continuous, numerically-integrated trajectory, thousands of
times.

Every powered manoeuvre is a finite-thrust integration and every coast a
three-body integration under Earth (+J2–J6 near-field) and Moon gravity, using a
real November-2022 lunar ephemeris and a degree-8 GRAIL gravity field, with
faithful renderings of the vehicle's own guidance (a two-segment linear-tangent
PEG ascent, the finite two-burn ICPS trans-lunar injection re-aimed at the real
lunar approach, the OTC/RTC correction chains, a solved powered lunar flyby, DRO
insertion and departure, and a PredGuid-class extended-skip entry at ~11 km/s),
SLS/Orion-era engine reliabilities, and a suite of failure modes sourced to the
historical record and NASA probabilistic-risk assessments. Artemis I flew
**uncrewed**, so there is no crew-survival model: mission success = SLS delivers
Orion **and** Orion returns and splashes down intact.

## Headline result

Across **10,000 trials** (seed 37, distributed on a compute cluster):

| Metric | Estimate |
|---|---|
| **Mission success** | **93.15 %**  (95 % CI 92.64–93.63 %) |

"Mission success" means the launch vehicle delivered Orion and the spacecraft
returned and splashed down intact — even if a recovered in-flight anomaly
occurred along the way. Of the 685 failed missions, the losses split into
~5.5 pp of sourced hardware/systems modes (ESM systems, engine ignitions,
boosters, heat shield, parachutes) and ~1.4 pp of an *honest* insertion/return
dispersion tail (missed approach, DRI/RPF propellant depletion). The fleet
reproduces the flown mission's markers: splash-miss median 4.9 km (real ~4.7),
entry peak-g 4.4 (real ~4), maximum Earth distance 431,418 km (real 432,194),
and mission duration 25.46 d (real ~25.45 d).

The definitive run lives in [`outputs/final/`](outputs/final/).
Open [`outputs/final/dashboard.html`](outputs/final/dashboard.html) in a browser
for the full breakdown — failure-mode decomposition, nominal-vs-flown
cross-check, phase timing vs. Artemis I's flown values, the OEM
trajectory-fidelity meter, and known limitations. A companion
[`outputs/final/realism_audit_artifact.html`](outputs/final/realism_audit_artifact.html)
documents the stage-by-stage fidelity review, and
[`outputs/final/artemis_vs_apollo_writeup.html`](outputs/final/artemis_vs_apollo_writeup.html)
is the matched comparison against the sibling
[Apollo 11 Monte Carlo](https://github.com/tjadenbc/Apollo11-Simulation) — on the
matched vehicle-recovered-intact metric the two are at statistical parity
(Apollo 94.21 % vs Artemis 93.15–94.45 %).

## Repository layout

| Path | What it is |
|---|---|
| `artemis1.py` | The simulation (~6,560 lines): physics, every mission phase, `run_mission()`, and the `main()` / `main_parallel()` Monte Carlo drivers. Feature flags are documented in the constants block near the top. |
| `run_mc.py` | Spawn-safe local launcher: `python3 run_mc.py <outdir> <n_trials> [seed] [workers]`. |
| `od_filter.py` | STM-LinCov orbit-determination covariance primitives for the DSN ground-navigation model. |
| `lunar_gravity_coeffs.py` | Embedded GRAIL GRGM1200A spherical-harmonic coefficients (see data note below). |
| `artemis_outputs.py` | Builds `dashboard.html`, `summary.txt`, the figures, and the OEM trajectory-fidelity meter from a run directory (data-driven). |
| `artemis_cluster.py`, `submit_artemis_sharded.sh`, `submit_artemis_mc.sh` | The SLURM sharding pipeline used to produce the definitive run on a compute cluster. |
| `fetch_cluster_run.sh`, `wait_nominal.sh` | Helpers for the cluster workflow (fetch a completed run's artifacts back; wait on the nominal capture). |
| `diag_audit.py` | Parallel per-trial consistency auditor for a completed run. |
| `data/artemis1_ephemeris/` | The as-flown Orion OEM ephemeris — the ground-truth gauge for the trajectory-fidelity meter (public NASA data product). |
| `outputs/final/` | **The definitive run**: the per-trial results CSV, the nominal trajectory, captured targeting products, the dashboard, the realism-audit report, the Apollo comparison, and figures. |

## Quick start

```bash
pip install -r requirements.txt   # numpy, scipy, pandas, matplotlib (Python 3.11+)
```

Run the Monte Carlo (resumable — checkpoints `results.csv` every trial):

```bash
# parallel via the spawn-safe launcher (recommended — ~10× faster, bit-identical to serial)
python3 run_mc.py outputs/myrun 100 42 10          # <outdir> <n_trials> <seed> <workers>

# or programmatically
python3 -c "import artemis1; artemis1.main_parallel(n=100, outdir='outputs/myrun', seed=42, workers=10)"
```

> **Note:** runs **must** be launched via `run_mc.py`, a `__main__`-guarded
> script, or `python3 -c` — never a stdin heredoc. macOS multiprocessing uses the
> `spawn` start method, which re-imports `__main__` in each worker. Pin BLAS to
> one thread per worker (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
> VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1`) to avoid oversubscription.

Regenerate the dashboard, summary, and figures from a run directory:

```bash
python3 -c "import matplotlib; matplotlib.use('Agg'); import artemis_outputs; artemis_outputs.main('outputs/myrun')"
```

Each trial costs a few minutes on a modern laptop core. The definitive
10,000-trial run was produced on a SLURM cluster via
`./submit_artemis_sharded.sh outputs/<name> <n_trials> <seed> <shards>`.

## Configuration

The model's fidelity features are controlled by flags in the constants block near
the top of `artemis1.py` — all default ON for the fidelity-first configuration,
each documented inline, and turning one OFF is bit-identical to the pre-feature
behaviour. A few flags also expose environment-variable overrides (e.g.
`AR1_PEG2SEG`, `AR1_TLI_ADAPT`, `AR1_OD_FILTER`) for A/B comparison against the
pre-feature lineage.

## Reproducibility

Local serial and parallel drivers are **bit-identical** (per-trial perturbations
are pre-generated in the main process and dispatched by trial index). Cluster
runs are *not* bit-identical to a laptop (different scipy/numpy builds), so each
cluster run is treated as its own population. The captured nominal products
(`nominal_results.json`, `nominal_targets.json`, `nominal_traj.npz`) are pinned
into the run directory so cross-machine numerical skew cannot flip the marginal
nominal-trajectory branches; the sharded pipeline reuses a pinned nominal rather
than re-deriving it.

## Definitive-run trial data

The per-trial debug JSONs for the definitive run (`trial_0.json` …
`trial_9999.json`) are distributed as the release asset
`artemis1_final_trials.tar.gz` rather than committed to the repository. Extract
into `outputs/final/trials/`. Each file is a full per-trial overview — the phase
timeline (per-phase mission-elapsed time and duration) and every outcome field —
complementing the per-trial summary rows in the committed `results.csv`.

## Data provenance

The lunar gravity field (`lunar_gravity_coeffs.py`) is derived from NASA's
**GRAIL GRGM1200A** model; the ephemeris uses the Meeus analytic series and
public NASA mission constants; `data/` holds the as-flown Orion OEM ephemeris
used as the trajectory-fidelity gauge. NASA data products are in the public
domain in the U.S. The MIT license below covers the original source code only —
see `LICENSE` for the third-party data notice.

## Use of generative AI

This project was developed with substantial assistance from Claude (Anthropic),
under the author's direction and review.

## License

MIT — see [`LICENSE`](LICENSE). The license covers the simulation source code;
embedded NASA scientific data carries its own (public-domain) terms, noted in
the license file.
