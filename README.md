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
| **Mission success** | **93.00 %**  (95 % CI 92.48–93.48 %) |

"Mission success" means the launch vehicle delivered Orion and the spacecraft
returned and splashed down intact — even if a recovered in-flight anomaly
occurred along the way. Of the 700 failed missions, the losses split into
~5.5 pp of sourced hardware/systems modes (ESM systems, engine ignitions,
boosters, heat shield, parachutes) and ~1.5 pp of an *honest* insertion/return
dispersion tail (DRI/RPF propellant depletion, residual lunar-impact geometry). The fleet
reproduces the flown mission's markers: splash-miss median 4.9 km (real ~4.7),
entry peak-g 4.4 (real ~4), maximum Earth distance 432,197 km (real 432,194),
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
(Apollo 94.21 % vs Artemis 93.00–94.29 %).

## Repository layout

| Path | What it is |
|---|---|
| `artemis1.py` | The simulation (~6,750 lines): physics, every mission phase, `run_mission()`, and the `main()` / `main_parallel()` Monte Carlo drivers. Feature flags are documented inline where they are defined. |
| `phase2b_bake_finite.py`, `dro_native_solve.py`, `return_joint_solve.py` | The offline solve tools behind the baked targeting constants: the forced-ignition TLI + CA-target re-bake, the ephemeris-native reference-DRO construction, and the joint DDP+RPF return solve (regression / continuation / multi-seed modes). Re-run on any force-model change. |
| `oem_loader.py` | Minimal loader for the as-flown OEM ephemeris in the simulation's frame (used by the solve tools). |
| `run_mc.py` | Spawn-safe local launcher: `python3 run_mc.py <outdir> <n_trials> [seed] [workers]`. |
| `od_filter.py` | STM-LinCov orbit-determination covariance primitives for the DSN ground-navigation model. |
| `lunar_gravity_coeffs.py` | Embedded GRAIL GRGM1200A spherical-harmonic coefficients (see data note below). |
| `artemis_outputs.py` | Builds `dashboard.html`, `summary.txt`, the figures, and the OEM trajectory-fidelity meter from a run directory (data-driven). |
| `artemis_cluster.py`, `submit_artemis_sharded.sh`, `submit_artemis_mc.sh` | The SLURM sharding pipeline used to produce the definitive run on a compute cluster. |
| `fetch_cluster_run.sh`, `wait_nominal.sh` | Helpers for the cluster workflow (fetch a completed run's artifacts back; wait on the nominal capture). |
| `diag_audit.py` | Parallel per-trial consistency auditor for a completed run. |
| `test_check_nominal.py` | Unit tests for the nominal-plausibility gate (`check_nominal`) and the branch-force self-heal. |
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

The model's fidelity features are controlled by flags in `artemis1.py`, each
documented inline where it is defined — every ADOPTED fidelity feature defaults ON
(investigated-and-rejected experiments are kept, default OFF, for the record), and
turning a feature OFF is bit-identical to the pre-feature behaviour. A few flags also expose environment-variable overrides (e.g.
`AR1_PEG2SEG`, `AR1_TLI_ADAPT`, `AR1_OD_FILTER`, `AR1_NATIVE_DRO`,
`AR1_PHASEC_BAKE=legacy`) for A/B comparison against the pre-feature lineage.

## Reproducibility

Local serial and parallel drivers are **bit-identical** (per-trial perturbations
are pre-generated in the main process and dispatched by trial index). Cluster
runs are *not* bit-identical to a laptop: a different CPU/BLAS numerical
environment (Apple ARM vs x86, differing linear-algebra kernels) can round a
marginal accept-gate the other way, so each cluster run is treated as its own
population. The captured nominal products (`nominal_results.json`,
`nominal_targets.json`, `nominal_traj.npz`) can be pinned into the run directory
and the sharded pipeline reuses a pinned nominal rather than re-deriving it.

**Nominal self-heal (`check_nominal`).** So a fresh derive on unfamiliar
hardware cannot silently shift the whole fleet, the nominal is validated against
the definitive run's physical markers (`check_nominal`, run automatically inside
`run_nominal_with_boundaries` and again at cluster `setup`). If the derived
nominal lands off the intended trajectory branch, the derivation is **re-run once
on the same machine with the branch forced** (`AR1_FORCE_PHASEC=1`), which keeps
every hard physics/convergence guard and only bypasses the one marginal quality
gate — so the machine reaches a valid nominal *natively* rather than being pinned
to foreign numbers. A still-implausible result raises and blocks the run (the
cluster `setup` job then fails and the `afterok` shard array never starts). For a
deliberately off-default configuration, set `AR1_SKIP_NOMINAL_CHECK=1` to bypass
the gate; the last resort is to pin a validated nominal into the run directory.
`test_check_nominal.py` unit-tests the gate and the self-heal (including a
synthetic branch-flip that proves the auto-recovery).

**Pinning is a convenience, not a requirement — measured.** In an end-to-end test, a
cluster of a different CPU architecture derived its own nominal from scratch (no pinned
files): it matched the laptop-derived nominal to floating-point noise (every physical
marker within 5e-5), passed `check_nominal` without the self-heal, and — on a
trial-matched 2,000-trial fleet flying identical dispersion draws — agreed with the
pinned definitive on 98.6% of per-trial outcomes (1,866 vs 1,865 successes), the
disagreements being symmetric coin-flips among trials sitting on propellant-margin
edges. The sibling Apollo 11 simulation independently ran the same experiment on its own
architecture with the same result.

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

This work was produced in a sustained collaboration between the author and Claude
(Anthropic), a large language model used across successive versions over the project's
development, and the division of labor was consequential enough to warrant a fuller
statement than the customary disclosure line. The conception is the author's: the
question, the decision to answer it with a physics-integrated Monte Carlo rather than a
reliability fault tree, and the design doctrines that define the model — fidelity first;
as-planned targeting; the vehicle-only mission-success definition for the uncrewed
flight; and the requirement that every stage be validated against the historical record
before being built upon. Claude performed essentially all of the implementation: the
simulation code, the guidance and targeting solvers, the failure model, the cluster
pipeline, the excavation of the historical sources behind the calibrated constants, the
diagnostic investigations, the statistical analysis of the Monte Carlo campaigns, and
the drafting of the project's documentation and manuscript. Design and refinement were
a genuine dialogue between the two: the highest-leverage corrections typically began as
the author's questions or catches — among them the as-planned splashdown-aim doctrine,
the question of whether the nominal should be constructed rather than derived (which
precipitated the ephemeris-native reference orbit and the targeting re-derivation that
made the insertion ledger exact), and the demand that cross-machine reproducibility be
demonstrated rather than assumed — and were then diagnosed and engineered by Claude,
while Claude's technical designs were in turn constrained, redirected, and sometimes
rejected by the author's judgment of what faithfulness required.

The process safeguards matter as much as the division of labor. The author directed the
work throughout; set the validation discipline under which no model change was adopted
without tiered Monte Carlo revalidation; reviewed, verified, and edited all code,
numerical results, physical assumptions, and text; and takes full responsibility for
this project. Errors surfaced late in development — a wrong-family return solution,
stale figure-caption claims, and drifted arithmetic in the manuscript's own numbers —
were caught by exactly that review structure, some by the author's reading and some by
adversarial verification passes the author required. The honest summary is that the
author served as principal investigator — conception, judgment, quality control, and
accountability — while Claude served as the research staff: construction, diagnosis,
analysis, and drafting at a speed and volume no individual could match. The author
could not have completed this project without Claude, and Claude could not have
completed this project without the author.

## License

MIT — see [`LICENSE`](LICENSE). The license covers the simulation source code;
embedded NASA scientific data carries its own (public-domain) terms, noted in
the license file.
