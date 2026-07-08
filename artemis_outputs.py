"""Artemis I dashboard / summary / figure generator.

UNCREWED (mission success only, no crew-survival). Reads an artemis1.py run directory:
    <outdir>/results.csv          (the MC trials)
    <outdir>/nominal_results.json (the nominal mission)
    <outdir>/trials/trial_*.json  (per-trial phase timeline)
and writing:
    <outdir>/dashboard.html       (self-contained: figures base64-inlined)
    <outdir>/summary.txt
    <outdir>/fig_*.png

Data-driven: success rate + Wilson CI, failure-mode decomposition, key-metric
distributions, the per-phase timing table vs the Artemis I as-flown reference
(artemis1.ARTEMIS_PHASE_DUR_S), an Artemis I cross-check, and the known
limitations (incl. the return-model caveats). Run:
    python3 -c "import matplotlib; matplotlib.use('Agg'); import artemis_outputs; artemis_outputs.main('outputs/artemis1_mc200')"
"""
import os, json, glob, base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import artemis1 as A

OUTDIR = "outputs/final"   # seed 37: the definitive run. ENABLE_PEG_2SEG (2-segment PEG: the weak-ascent 5.9% no longer falls back to IGM geometry -> flies the nominal MECO 157 km + the cheap Phase-C arrival) + ENABLE_TLI_PLAN_ADAPT (retry-on-artifact-failure Lambert replan, honest-taxonomy backstop). Mission success 93.15% (9315/10000, Wilson 92.64-93.63). Trajectory-neutral: splash 4.88, peak-g 4.43, max-Earth 431,418, dur 25.46.
# --- OEM fidelity meter: sim NOMINAL trajectory vs the as-flown Orion OEM, per phase boundary ---
_OEM_FILE = "data/artemis1_ephemeris/Post_TLI_Orion_AsFlown_20221213_EPH_OEM.asc"
_JD_LAUNCH = 2459899.78315


def _oem_precession():
    """J2000 -> sim ECI (of-date) frame transform for OEM comparison: IAU precession THEN the main
    IAU-1980 nutation terms (mean-of-date -> true-of-date). Nutation tightens the frame-corrected OEM residual (flyby CA 96 -> 83 km) — the sim's effective frame is
    true-of-date, so the ~14 arcsec nutation aligns it better. Comparison-only (not the physics)."""
    def Rx(a): c, s = np.cos(a), np.sin(a); return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])
    def Rz(a): c, s = np.cos(a), np.sin(a); return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    def Ry(a): c, s = np.cos(a), np.sin(a); return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    T = (_JD_LAUNCH - 2451545.0) / 36525.0; asec = np.pi / 180.0 / 3600.0
    z = (2306.2181*T + 1.09468*T*T) * asec; ze = (2306.2181*T + 0.30188*T*T) * asec
    th = (2004.3109*T - 0.42665*T*T) * asec
    P = Rz(-z) @ Ry(th) @ Rz(-ze)                                    # J2000 -> mean-of-date (precession)
    Om = np.radians(125.04452 - 1934.136261*T)                      # main IAU-1980 nutation terms
    Ls = np.radians(280.4665 + 36000.7698*T); Lm = np.radians(218.3165 + 481267.8813*T)
    dpsi = (-17.20*np.sin(Om) - 1.32*np.sin(2*Ls) - 0.23*np.sin(2*Lm) + 0.21*np.sin(2*Om)) * asec
    deps = (9.20*np.cos(Om) + 0.57*np.cos(2*Ls) + 0.10*np.cos(2*Lm) - 0.09*np.cos(2*Om)) * asec
    eps0 = np.radians(23.439291 - 0.0130042*T)
    N = Rx(-(eps0+deps)) @ Rz(-dpsi) @ Rx(eps0)                      # mean-of-date -> true-of-date (nutation)
    return N @ P


def _oem_load():
    from datetime import datetime
    g, R = [], []; LAUNCH = datetime(2022, 11, 16, 6, 47, 44)
    for ln in open(_OEM_FILE):
        p = ln.split()
        if len(p) == 7 and p[0][:2] == "20" and "T" in p[0]:
            try:
                dt = datetime.strptime(p[0][:19], "%Y-%m-%dT%H:%M:%S")
                fr = float("0"+p[0][19:]) if "." in p[0][19:] else 0.0
                g.append((dt-LAUNCH).total_seconds()+fr); R.append([float(x) for x in p[1:4]])
            except Exception:
                pass
    return np.array(g), np.array(R)


# Phase-boundary functions whose OUTPUT state is compared to the OEM, in flight order.
# Keyed on the artemis1 function NAME (hooked at run time) -> the stage label shown.
_OEM_STAGES = [
    ("phase_icps_tli", "post-TLI"), ("phase_outbound_coast", "outbound"),
    ("phase_outbound_powered_flyby", "OPF"), ("phase_dro_insertion", "DRI"),
    ("phase_dro_coast", "DRO/DDP"), ("phase_dro_departure", "post-DDP"),
    ("phase_return_coast", "return coast"), ("phase_return_powered_flyby", "RPF"),
    ("phase_transearth_coast", "EI"),
]


def _oem_rows_from_boundaries(boundaries, P, g, RR):
    """[(label, state, t_end), ...] -> [[label, get_d, residual_km|None], ...] vs the
    frame-aligned OEM. Points within 5 min of a window edge are clamped (e.g. EI sits
    ~0.02 s past the OEM's last epoch — a float edge, not a gap); post-TLI is ~13 min
    pre-window and stays None."""
    _EDGE_TOL_S = 300.0
    out = []
    for label, s, t in boundaries:
        s = np.asarray(s, float)
        if t < g[0] - _EDGE_TOL_S or t > g[-1] + _EDGE_TOL_S:
            out.append([label, t/86400.0, None])
        else:
            tc = float(np.clip(t, g[0], g[-1]))
            oem = P @ (np.array([np.interp(tc, g, RR[:, k]) for k in range(3)]) * 1e3)
            out.append([label, t/86400.0, float(np.linalg.norm(s[:3]-oem)/1e3)])
    return out


def oem_residuals(outdir):
    """Per-stage residual (km) of the sim NOMINAL vs the as-flown OEM (IAU-precession-aligned).
    Cached to <outdir>/oem_residuals.json; on a cache miss runs the REAL nominal mission once
    (~2 min — the return solve) with the CURRENT default model, hooking each phase-boundary
    function to capture its output state. (The prior version re-ran a standalone phase chain with
    stale near-polar flag overrides + _NOMINAL_TARGETS=None, which broke after post-TLI under the
    Phase-C/snap-free model — leaving only one row.) Returns [[stage, get_d, residual_km|None], ...].
    Pass a truthy AR1_FORCE_OEM env var (or delete the cache) to force a recompute on a stale file."""
    cache = os.path.join(outdir, "oem_residuals.json")
    if os.path.exists(cache) and not os.environ.get("AR1_FORCE_OEM"):
        try:
            cached = json.load(open(cache))
            if len(cached) >= 2:          # a valid multi-stage meter; single-row files are stale -> recompute
                return cached
        except Exception:
            pass
    if not os.path.exists(_OEM_FILE):
        return []
    P = _oem_precession(); g, RR = _oem_load()

    # Preferred source: phase boundaries persisted by the RUN's own nominal capture
    # (nominal_traj.npz, transferred back from the cluster). Uses the run's ACTUAL nominal
    # and skips the ~2.5-min local re-run. Falls through to a live re-run if absent/legacy.
    boundaries = None
    npz = os.path.join(outdir, "nominal_traj.npz")
    if os.path.exists(npz) and not os.environ.get("AR1_FORCE_OEM"):
        try:
            d = np.load(npz, allow_pickle=True)
            if all(k in d for k in ("_boundary_labels", "_boundary_states", "_boundary_t")):
                labels = [str(x) for x in d["_boundary_labels"]]
                states = np.asarray(d["_boundary_states"], float)
                ts = np.asarray(d["_boundary_t"], float)
                boundaries = [(labels[i], states[i], float(ts[i])) for i in range(len(labels))]
        except Exception:
            boundaries = None
    if boundaries is not None:
        out = _oem_rows_from_boundaries(boundaries, P, g, RR)
        try:
            json.dump(out, open(cache, "w"))
        except Exception:
            pass
        return out

    boundaries = []                        # [(stage_label, state, t_end), ...] in flight order
    originals = {}

    def _make_hook(label, orig):
        def _hook(state, t0, perturb=None, *a, **k):
            res = orig(state, t0, perturb, *a, **k)
            try:
                if res.get("success", True) and res.get("state") is not None:
                    boundaries.append((label, np.asarray(res["state"], float),
                                       float(res.get("t_end", t0))))
            except Exception:
                pass
            return res
        return _hook

    try:
        for fnname, label in _OEM_STAGES:
            orig = getattr(A, fnname, None)
            if orig is not None:
                originals[fnname] = orig
                setattr(A, fnname, _make_hook(label, orig))
        A.run_mission(perturb=None)        # the REAL nominal, current defaults, proper setup
    except Exception:
        pass
    finally:
        for fnname, orig in originals.items():
            setattr(A, fnname, orig)

    out = _oem_rows_from_boundaries(boundaries, P, g, RR)
    try:
        json.dump(out, open(cache, "w"))
    except Exception:
        pass
    return out

# Real Artemis I reference values (sourced from NASA / AAS mission references).
REAL = {
    "duration_d": 25.45,            # 25 d 10 h 53 m
    "max_earth_km": 432_194.0,      # 268,563 mi, FD13
    "tli_dv_ms": 3050.0,
    "tli_c3": -1.9,                 # km^2/s^2
    "entry_v_ms": 10_990.0,         # 24,581 mph
    "entry_fpa_deg": -6.5,
    "splash": "off San Diego (planned zone); as-flown ~29.0N/118.3W (weather retarget)",
}


def wilson_ci(k, n, z=1.96):
    """Wilson score 95% CI for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    den = 1 + z*z/n
    c = (p + z*z/(2*n)) / den
    hw = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return (p, max(0.0, c - hw), min(1.0, c + hw))


def phase_timing_stats(outdir):
    """Per-phase mission-duration min/avg/max across MC trials vs the Artemis I
    as-flown reference (artemis1.ARTEMIS_PHASE_DUR_S)."""
    files = glob.glob(os.path.join(outdir, "trials", "trial_*.json"))
    durs = {}
    for f in files:
        if os.path.basename(f) == "trial_nominal.json":
            continue
        try:
            with open(f) as fh:
                rec = json.load(fh)
        except Exception:
            continue
        for ph in rec.get("phase_timeline", []):
            durs.setdefault(ph["phase"], []).append(float(ph["duration_s"]))
    rows = []
    for label in [s[2] for s in A.PHASE_SEGMENTS]:
        vals = durs.get(label, [])
        if not vals:
            continue
        avg = sum(vals) / len(vals)
        ref = A.ARTEMIS_PHASE_DUR_S.get(label)
        pct = ((avg - ref) / ref * 100.0) if ref else None
        rows.append({"phase": label, "n": len(vals), "min": min(vals),
                     "avg": avg, "max": max(vals), "ref": ref, "pct": pct})
    return rows


def _dur(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return "—"
    if s < 3600:
        return f"{s/60:.1f} min"
    if s < 86400:
        return f"{s/3600:.2f} h"
    return f"{s/86400:.2f} d"


def _clean_label(name):
    """Collapse an accidentally-doubled phase prefix (e.g. 'entry_entry_overstress'
    -> 'entry_overstress') so the failure table reads cleanly on older runs whose
    data still carries the duplicated prefix."""
    parts = str(name).split("_")
    if len(parts) >= 2 and parts[0] == parts[1]:
        return "_".join(parts[1:])
    return str(name)


def failure_decomposition(df):
    """Count + rate of each mission_failure category (None = success)."""
    n = len(df)
    mf = df["mission_failure"].fillna("(none — success)")
    vc = mf.value_counts()
    return [(_clean_label(k), int(v), 100.0*v/n) for k, v in vc.items()], n


def compute_timing(outdir):
    """Per-trial wall time (mean/median/range). Prefers results.csv's trial_time_s column
    (always present, even for cluster runs whose per-trial JSONs weren't pulled); falls back
    to the trial JSONs. Returns None if no timing is recorded."""
    ts = []
    rp = os.path.join(outdir, "results.csv")
    if os.path.exists(rp):
        try:
            d = pd.read_csv(rp)
            if "trial_time_s" in d.columns:
                col = d[d["trial"] >= 0]["trial_time_s"] if "trial" in d.columns else d["trial_time_s"]
                ts = [float(x) for x in col.dropna() if float(x) > 0]
        except Exception:
            ts = []
    if not ts:
        for f in glob.glob(os.path.join(outdir, "trials", "trial_*.json")):
            if os.path.basename(f) == "trial_nominal.json":
                continue
            try:
                v = json.load(open(f)).get("trial_time_s")
            except Exception:
                continue
            if isinstance(v, (int, float)) and v == v:
                ts.append(float(v))
    if not ts:
        return None
    return {"n": len(ts), "avg": float(np.mean(ts)), "median": float(np.median(ts)),
            "min": float(np.min(ts)), "max": float(np.max(ts)), "total": float(np.sum(ts))}


def crosscheck_rows(nominal, df):
    """Sim-nominal vs real Artemis I, with a signed %-error column vs the as-flown
    actuals. "—" where a percentage
    is not meaningful (a set input, or a coordinate compared to the planned zone)."""
    def has(k):
        v = nominal.get(k)
        return v is not None and not (isinstance(v, float) and np.isnan(v))
    def f(k, spec, suf=""):
        return (format(nominal[k], spec) + suf) if has(k) else "—"
    def err(k, real_val):
        if not has(k):
            return "—"
        return f"{(nominal[k] - real_val) / max(abs(real_val), 1e-9) * 100:+.1f}%"
    rows = [
        ("Mission duration", f("mission_duration_d", ".2f", " d"), f"{REAL['duration_d']:.2f} d",
         err("mission_duration_d", REAL['duration_d'])),
        ("Max distance from Earth", f("max_earth_distance_km", ",.0f", " km"),
         f"{REAL['max_earth_km']:,.0f} km", err("max_earth_distance_km", REAL['max_earth_km'])),
        ("TLI ΔV", f("tli_dv_ms", ",.0f", " m/s"), f"{REAL['tli_dv_ms']:,.0f} m/s",
         err("tli_dv_ms", REAL['tli_dv_ms'])),
        ("Post-TLI C3", f("post_tli_c3_km2s2", ".2f", " km²/s²"), f"{REAL['tli_c3']:.2f} km²/s²",
         err("post_tli_c3_km2s2", REAL['tli_c3'])),
        ("Entry velocity", f"{REAL['entry_v_ms']:,.0f} m/s (set)", f"{REAL['entry_v_ms']:,.0f} m/s", "—"),
        ("Entry peak-g", f("entry_peak_g", ".1f", " g"), "~4 g (skip entry)", err("entry_peak_g", 4.0)),
        ("Splashdown",
         (f"{abs(nominal['splash_lat']):.1f}°{'N' if nominal['splash_lat'] >= 0 else 'S'}, "
          f"{abs(nominal['splash_lon']):.1f}°{'E' if nominal['splash_lon'] >= 0 else 'W'} "
          f"({'N' if nominal['splash_lat'] >= 0 else 'S'}. Pacific)"
          if has("splash_lat") else "—"),
         REAL["splash"], "—"),
    ]
    return rows


def known_limitations(df=None, nominal=None):
    """Data-driven so the splash region, timeline, and dominant failure stay fresh
    against whatever run is being rendered (see realism_audit_artifact.html)."""
    nom = nominal or {}
    slat = nom.get("splash_lat"); slon = nom.get("splash_lon"); mdur = nom.get("mission_duration_d")
    splash_str = (f"~{abs(slat):.0f}°{'N' if slat >= 0 else 'S'}"
                  + (f", {abs(slon):.0f}°{'W' if slon < 0 else 'E'}" if isinstance(slon, (int, float)) else "")
                  if isinstance(slat, (int, float)) else "the planned zone")
    dur_str = (f"{mdur:.2f} d (≈ Artemis I's 25.45 d)"
               if isinstance(mdur, (int, float)) else "the real 25.45 d")

    def _med(col, fmt="{:.1f}"):
        if df is not None and col in df:
            s = df[col].dropna()
            if len(s):
                return fmt.format(s.median())
        return "—"
    sm_med = _med("splash_miss_km", "{:.1f}"); pg_med = _med("entry_peak_g", "{:.1f}")
    rpf_med = _med("rpf_dv_ms", "{:.0f}"); rz_med = _med("recovery_zone_displacement_km", "{:.2f}")

    dom = ""
    if df is not None:
        fc = df[df["full_success"].fillna(False) == False]["mission_failure"].dropna()
        if len(fc):
            top = fc.value_counts()
            dom = (f" The dominant failure here is `{top.index[0]}` at "
                   f"{100*top.iloc[0]/len(df):.1f}%.")
    return [
        ("Failure-mode probabilities are ESTIMATES",
         "Artemis has flown twice (I 2022, II 2026), both successful — no statistical failure base "
         "exists. Inputs are heritage component reliabilities + NASA PRA targets + foreseeable-problem "
         "analysis (sourced from NASA / AAS references). The success number is only as good as these." + dom),
        ("Navigation is a geometry-derived OD covariance, not a live sequential filter",
         f"The nav covariance is built EMERGENTLY from DSN station geometry + tracking-arc physics via "
         f"an STM-LinCov filter (`ENABLE_OD_FILTER`, default-ON) — it replaced the "
         f"hand-set calibrated-covariance stand-in and delivers the as-flown OD-knowledge envelope, so "
         f"DELIVERY PRECISION is now faithful: recovery-zone displacement ~{rz_med} km median, "
         f"splash-miss ~{sm_med} km median. The remaining approximation is that it is a LINEARIZED-"
         f"COVARIANCE (LinCov) model, not a live sequential EKF re-solving state from synthetic raw "
         f"range/Doppler/ΔDOR observations. Guidance corrections are applied on the estimated state; "
         f"each is flown to convergence, not re-tracked continuously."),
        ("Entry is a PredGuid drag-tracker; the shallow-corridor deep-dip velocity is unmatched",
         f"A PredGuid-class drag tracker — bank modulation + closed-loop crossrange bank reversals + a "
         f"lob-lift downrange trim + an extended-skip long-range branch. Nominal peak ~{pg_med} g ≈ real "
         f"Orion ~4 g, and it reproduces the real TWO-PULSE skip; splash accuracy is MATCHED "
         f"(~{sm_med} km median ≈ real ~4.7). Residual (the HUNTEST wall): at Orion's shallow −5.9° "
         f"corridor the first dip bottoms ~10 km/s vs the real deep ~8–9, and matching that depth needs "
         f"a steeper EI → ~10 g overstress — so peak-g OR deep-dip velocity, not both. Atmosphere is "
         f"static USSA-76 with per-trial ±6% density dispersion + chute-phase wind drift (the ~3.5 km "
         f"splash floor); no time-varying weather."),
        ("DRO is a CR3BP map, not the real elliptical orbit",
         f"The reference DRO is a CR3BP periodic orbit mapped onto the real Moon; the sim force model "
         f"cannot hold the real elliptical DRO (propagating the real insertion state drifts ~63,000 km "
         f"over the 6-day coast — the real orbit is periodic only in higher-fidelity dynamics + OM "
         f"maintenance burns). Accepted consequences: RPF ~{rpf_med} m/s vs the as-flown 292.9 (the "
         f"~4,070 km CR3BP-offset solution-basin premium), flyby periselene ~146 vs 130 km. DRI itself "
         f"is a genuine velocity-match (~94 m/s ≈ 110.6, snap-free); max-Earth is correct (431,418 ≈ "
         f"432,194 km). The return is the epoch-committed PLANNED trans-earth branch (100% of successes), "
         f"not a fallback."),
        ("Nominal targets the PLANNED zone; the as-flown Guadalupe splash is a dispersion scenario",
         f"Per the as-planned doctrine the nominal aims the PLANNED recovery zone (off San Diego, "
         f"{splash_str}), matching the flight plan (timeline {dur_str}). Artemis I actually splashed "
         f"~350 km south near Guadalupe Island (~29°N) after an in-mission WEATHER RETARGET — a discrete "
         f"operational move beyond the fleet's natural dispersion, so it is reproduced as a named re-aim "
         f"scenario, not the nominal. The old Indian-Ocean fallback + region "
         f"offset are CLOSED. Separately, mid-mission burns (OTC/DRI/DDP/RPF/RTC) are impulsive Δv with "
         f"execution-error draws (launch/PRM/TLI/OPF are finite), and the nominal outbound path sits "
         f"~2,091 km from the as-flown OEM — an in-plane orientation difference (zonals nominal vs "
         f"pre-zonals baked targets), the standing path-fidelity item."),
    ]


def generate_plots(df, outdir):
    BLUE, ORANGE, GREEN, RED = "#2b6cb0", "#dd6b20", "#38a169", "#e53e3e"
    n = len(df)
    succ = int(df["full_success"].fillna(False).astype(bool).sum())

    # 1. success / failure donut
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    fail = n - succ
    ax.pie([succ, fail] if fail else [succ], colors=[GREEN, RED] if fail else [GREEN],
           labels=[f"success\n{succ}", f"failure\n{fail}"] if fail else [f"success\n{succ}"],
           wedgeprops=dict(width=0.42), startangle=90)
    ax.set_title(f"Mission outcome (n={n})")
    fig.savefig(os.path.join(outdir, "fig_outcome.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)

    # 1b. failure causes — horizontal bars SORTED BY MISSION STAGE, one colour per stage
    if "mission_failure" in df.columns and fail:
        fc = df.loc[~df["full_success"].fillna(False).astype(bool), "mission_failure"].dropna().value_counts()
        ordered = []
        for ph in PHASE_ORDER:
            modes = sorted([(r, c) for r, c in fc.items() if categorize_failure(r) == ph], key=lambda x: -x[1])
            ordered += [(r, c, ph) for r, c in modes]
        if ordered:
            labels = [f"{r}  ({c}, {100*c/n:.1f}%)" for r, c, _ in ordered]
            counts = [c for _, c, _ in ordered]
            colors = [PHASE_COLORS.get(ph, "#888") for _, _, ph in ordered]
            fig, ax = plt.subplots(figsize=(8.4, max(3.6, 0.34 * len(ordered))))
            y = np.arange(len(ordered))
            ax.barh(y, counts, color=colors, edgecolor="#1d1d1f", linewidth=0.4, alpha=0.92)
            ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel(f"trials (of {n})")
            ax.set_title(f"Failure causes by mission stage — {fail} failed missions")
            from matplotlib.patches import Patch
            seen = [ph for ph in PHASE_ORDER if any(p == ph for _, _, p in ordered)]
            ax.legend(handles=[Patch(facecolor=PHASE_COLORS[ph], label=ph) for ph in seen],
                      loc="lower right", fontsize=7.5, framealpha=0.9)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            fig.savefig(os.path.join(outdir, "fig_failure_modes.png"), dpi=110, bbox_inches="tight")
            plt.close(fig)

    # 2. peak-g histogram
    pg = df["entry_peak_g"].dropna()
    if len(pg):
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        ax.hist(pg, bins=15, color=BLUE, edgecolor="white")
        ax.axvline(A.ENTRY_STRUCTURAL_G, color=RED, ls="--", lw=1.5,
                   label=f"structural limit {A.ENTRY_STRUCTURAL_G:.0f} g")
        ax.set_xlabel("entry peak-g"); ax.set_ylabel("trials"); ax.legend()
        ax.set_title("Entry peak deceleration")
        fig.savefig(os.path.join(outdir, "fig_peakg.png"), dpi=110, bbox_inches="tight")
        plt.close(fig)

    # 3. splash-miss histogram
    sm = df["splash_miss_km"].dropna()
    if len(sm):
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        ax.hist(sm, bins=15, color=ORANGE, edgecolor="white")
        ax.set_xlabel("splash miss vs nominal (km)"); ax.set_ylabel("trials")
        ax.set_title("Splashdown dispersion (entry)")
        fig.savefig(os.path.join(outdir, "fig_splash.png"), dpi=110, bbox_inches="tight")
        plt.close(fig)


def generate_summary(df, nominal, outdir):
    n = len(df)
    succ = int(df["full_success"].fillna(False).astype(bool).sum())
    p, lo, hi = wilson_ci(succ, n)
    fd, _ = failure_decomposition(df)
    lines = []
    lines.append("ARTEMIS I — MONTE CARLO SUMMARY")
    lines.append("=" * 50)
    lines.append(f"Trials: {n}")
    lines.append(f"Mission success: {succ}/{n} = {100*p:.1f}%  "
                 f"(Wilson 95% CI {100*lo:.1f}-{100*hi:.1f}%)")
    lines.append("Definition: SLS delivers Orion AND Orion returns & splashes intact (uncrewed).")
    lines.append("")
    lines.append("Failure decomposition:")
    for name, c, rate in fd:
        lines.append(f"  {name:34s} {c:4d}  ({rate:4.1f}%)")
    lines.append("")
    pg = df["entry_peak_g"].dropna(); sm = df["splash_miss_km"].dropna()
    if len(pg):
        lines.append(f"Entry peak-g:   min {pg.min():.1f}  median {pg.median():.1f}  max {pg.max():.1f}")
    if len(sm):
        lines.append(f"Splash miss:    median {sm.median():,.0f} km  p90 {np.percentile(sm,90):,.0f} km")
    if "max_earth_distance_km" in df:
        me = df["max_earth_distance_km"].dropna()
        if len(me):
            lines.append(f"Max Earth dist: median {me.median():,.0f} km (real Artemis I 432,194)")
    ct = compute_timing(outdir)
    if ct:
        lines.append(f"Compute time:   avg {ct['avg']:,.0f} s/trial ({ct['avg']/60:.1f} min) · "
                     f"median {ct['median']:,.0f} s · range {ct['min']:,.0f}–{ct['max']:,.0f} s "
                     f"· {ct['n']} trials")
    lines.append("")
    lines.append("NOTE: FULLY-FLOWN return (DRO departure -> powered lunar flyby -> RPF -> RTC")
    lines.append("-> 122 km EI, propagated; inbound MCC live). Failure-mode probabilities are")
    lines.append("estimates (heritage + PRA). See the README and realism_audit_artifact.html.")
    txt = "\n".join(lines)
    with open(os.path.join(outdir, "summary.txt"), "w") as f:
        f.write(txt + "\n")
    return txt


# Shared light "system" dashboard theme, inlined so the generator is self-contained
# (no external .css dependency). Class vocabulary: .summary-card / .stats / .stat /
# .plot / .caveat / .context / tr.ph-header.
_DASH_STYLE = r"""
/* MC-dashboard style (Artemis I). Light "system" theme:
   #f5f5f7 canvas, white cards, subtle shadows, uppercase small-caps table
   headers, blue "context" / amber "caveat" callouts. Loaded + inlined by the
   dashboard generator (artemis_outputs.py). Plain CSS, inlined in the
   dashboard style block. (Do NOT write a
   literal close-style tag in this comment — inlined into a real style element
   it would close the tag early and dump the CSS as body text.) */
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui,
                     sans-serif; margin: 0; padding: 24px;
                     background: #f5f5f7; color: #1d1d1f; }
h1 { font-weight: 600; margin-bottom: 4px; font-size: 28px; }
h2 { font-weight: 500; margin-top: 32px; border-bottom: 2px solid #d2d2d7;
       padding-bottom: 8px; font-size: 20px; }
.summary-card { background: white; padding: 18px 24px; border-radius: 12px;
                  box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin: 16px 0; }
.stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;
           margin: 16px 0; }
.stat { background: white; padding: 14px; border-radius: 10px;
          box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
.stat .label { color: #6e6e73; font-size: 12px; text-transform: uppercase;
                  letter-spacing: 0.5px; }
.stat .value { font-size: 26px; font-weight: 600; margin-top: 4px; }
table { border-collapse: collapse; width: 100%; background: white;
          border-radius: 10px; overflow: hidden;
          box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
th, td { padding: 9px 14px; text-align: left;
            border-bottom: 1px solid #e8e8ed; vertical-align: top; }
th { background: #f9f9fb; font-weight: 600; color: #6e6e73; font-size: 12px;
       text-transform: uppercase; letter-spacing: 0.4px; }
tr:last-child td { border-bottom: none; }
tr.ph-header td { background: #f0f0f4; color: #1d1d1f; font-size: 12px;
                    letter-spacing: 0.5px; }
.plot { background: white; padding: 16px; border-radius: 10px; margin: 12px 0;
          box-shadow: 0 1px 3px rgba(0,0,0,0.06); text-align: center; }
.plot img { max-width: 100%; height: auto; }
.caveat { background: #fffbeb; border-left: 4px solid #f59e0b;
            padding: 14px 18px; border-radius: 6px; margin: 14px 0;
            font-size: 14px; line-height: 1.6; }
.context { background: #eff6ff; border-left: 4px solid #3b82f6;
            padding: 14px 18px; border-radius: 6px; margin: 14px 0;
            font-size: 14px; line-height: 1.6; }
code { background: #f3f4f6; padding: 2px 5px; border-radius: 3px;
          font-size: 13px; }
"""


FAILURE_EXPLANATIONS = {
    "outbound_missed_lunar_approach": ("Missed lunar approach", "The outbound trajectory missed the ~130 km powered-flyby corridor and the OTC correction chain could not recover it within the ESM propellant budget."),
    "outbound_lunar_impact_trajectory": ("Lunar-impact trajectory", "A dispersed outbound path crossed onto a lunar-impact trajectory (guard-triggered abort)."),
    "esm_systems_failure": ("ESM systems (catastrophic)", "Catastrophic European Service Module propulsion/power failure — triangulated at 1/50 (Apollo-13 SM analogue × modern improvement, cross-checked vs the Orion PRA)."),
    "tli_icps_ignition_failure": ("TLI / ICPS ignition", "The ICPS (RL10B-2) failed to ignite or complete the Trans-Lunar Injection burn."),
    "launch_srb_failure": ("Launch — SRB", "A five-segment solid-rocket-booster failure during first-stage ascent."),
    "launch_structural_failure_max_q": ("Launch — structural (max-Q)", "The vehicle exceeded its structural g / dynamic-pressure limit during ascent."),
    "entry_heatshield_loss": ("Heatshield char-loss", "Heatshield char-loss/failure during the ~11 km/s skip entry."),
    "parachute_failure": ("Parachute", "A drogue or main parachute failure during descent."),
    "cm_sm_separation_failure": ("CM/SM separation", "The CM/ESM separation bolts failed to cleanly separate the modules before entry."),
    "mmod_strike": ("MMOD strike", "A mission-ending micrometeoroid / orbital-debris strike."),
    "nav_sensor_loss": ("Nav-sensor loss", "Loss of a navigation sensor (star tracker / IMU)."),
    "avionics_radiation_anomaly": ("Avionics radiation", "A radiation-induced avionics anomaly in the deep-space / Van Allen environment."),
    "comm_loss_at_burn": ("Comm loss at burn", "Loss of ground communication at a critical burn."),
    "esm_pressurization": ("ESM pressurization", "An ESM propellant-pressurization system failure."),
    "dro_stationkeeping": ("DRO station-keeping", "A station-keeping failure during the ~6-day Distant Retrograde Orbit stay."),
    "rcs_failure": ("RCS", "A reaction-control-system failure."),
    "thermal_loss": ("Thermal control", "A thermal-control-system failure."),
    "solar_particle_event": ("Solar particle event", "A solar particle event (SPE) exceeded vehicle tolerances."),
}


def explain_failure(reason):
    """Plain-language (title, description) for a mission_failure label. Matches VARIANT/suffixed labels by
    SUBSTRING first (dri_/opf_/ddp_/rpf_ propellant-depletion + otc56, and the per-burn OMS-E ignition set)
    so no suffixed label renders a blank description (avoiding the exact-key-lookup trap)."""
    r = str(reason)
    if "esm_propellant_depleted" in r:
        return ("ESM propellant depleted", "The ESM ran out of usable propellant before completing a burn — a correction-heavy or badly-dispersed trajectory drained the finite budget (the suffix names the burn that starved: OPF / DRI / DDP / RPF / OTC-5-6).")
    if "oms_e_ignition" in r:
        return ("OMS-E ignition", "The Orbital Maneuvering System engine (AJ10 heritage) failed to ignite for a required burn (~2e-3 per attempt).")
    return FAILURE_EXPLANATIONS.get(r, (r.replace("_", " "), ""))


# Mission-phase order for the phase-grouped Failure Analysis (tr.ph-header groups) +
# the per-phase colour ramp for the failure-mode bar chart (one distinct hue per mission stage).
PHASE_ORDER = ["launch", "tli / icps", "outbound coast", "powered flyby (opf)", "dro insertion",
               "dro stay", "dro departure", "return flyby (rpf)", "entry", "descent / splash",
               "spacecraft systems"]
PHASE_COLORS = {
    "launch": "#1f77b4", "tli / icps": "#ff7f0e", "outbound coast": "#2ca02c",
    "powered flyby (opf)": "#d62728", "dro insertion": "#9467bd", "dro stay": "#8c564b",
    "dro departure": "#e377c2", "return flyby (rpf)": "#17becf", "entry": "#393b79",
    "descent / splash": "#bcbd22", "spacecraft systems": "#7f7f7f",
}


def categorize_failure(reason):
    """Map a mission_failure label to its mission PHASE (prefix-based, variant-safe). Cross-cutting
    systems modes (ESM/MMOD/nav/avionics/comm/RCS/thermal/SPE) group under 'spacecraft systems' since
    they can occur across the coast rather than in one phase."""
    r = str(reason)
    if r.startswith("launch_"):   return "launch"
    if r.startswith("tli_"):      return "tli / icps"
    if r.startswith("outbound_"): return "outbound coast"
    if r.startswith("opf_"):      return "powered flyby (opf)"
    if r.startswith("dri_"):      return "dro insertion"
    if r == "dro_stationkeeping": return "dro stay"
    if r.startswith("ddp_"):      return "dro departure"
    if r.startswith("rpf_"):      return "return flyby (rpf)"
    if r.startswith("entry_"):    return "entry"
    if r in ("parachute_failure", "cm_sm_separation_failure"): return "descent / splash"
    return "spacecraft systems"


def generate_dashboard(df, nominal, outdir):
    def img(fn):
        p = os.path.join(outdir, fn)
        if not os.path.exists(p):
            return ""
        d = base64.b64encode(open(p, "rb").read()).decode("ascii")
        return f'<img src="data:image/png;base64,{d}">'

    n = len(df)
    succ = int(df["full_success"].fillna(False).astype(bool).sum())
    p, lo, hi = wilson_ci(succ, n)
    ct = compute_timing(outdir)
    compute_str = (f" · avg {ct['avg']:,.0f} s/trial ({ct['avg']/60:.1f} min)" if ct else "")
    fd, _ = failure_decomposition(df)
    _fails = [(nm, c, rate) for nm, c, rate in fd if not str(nm).startswith("(none")]
    # Failure Analysis table — grouped BY STAGE (tr.ph-header), failures only (success is shown
    # by the donut + the summary stat, not a table row); the stage grouping carries plain-language
    # explanations. Mission Outcome's per-stage breakdown is now the fig_failure_modes.png bar chart.
    fa_rows = ""
    for ph in PHASE_ORDER:
        grp = [(nm, c, rt) for nm, c, rt in _fails if categorize_failure(nm) == ph]
        if not grp:
            continue
        ph_tot = sum(c for _, c, _ in grp)
        fa_rows += (f"<tr class='ph-header'><td colspan='3'><strong>{ph.upper()} — "
                    f"{ph_tot} trials ({100*ph_tot/n:.1f}%)</strong></td></tr>")
        for nm, c, rt in grp:
            title, expl = explain_failure(nm)
            fa_rows += (f"<tr><td style='text-align:right;white-space:nowrap'>{c}"
                        f"<br><span style='color:#6e6e73;font-size:11px'>{rt:.1f}%</span></td>"
                        f"<td><strong>{title}</strong>"
                        f"<br><span style='color:#6e6e73;font-size:11px'>{nm}</span></td>"
                        f"<td>{expl}</td></tr>")
    _sm = df["splash_miss_km"].dropna()
    splash_med = f"{_sm.median():.1f}" if len(_sm) else "—"
    cc_rows = "".join(
        f"<tr><td>{lab}</td><td>{sim}</td><td>{real}</td><td style='text-align:right'>{m}</td></tr>"
        for lab, sim, real, m in crosscheck_rows(nominal, df))
    pt = phase_timing_stats(outdir)
    pt_rows = "".join(
        f"<tr><td>{r['phase']}</td><td>{_dur(r['min'])}</td><td>{_dur(r['avg'])}</td>"
        f"<td>{_dur(r['max'])}</td><td>{_dur(r['ref'])}</td>"
        f"<td style='text-align:right'>{('%+.0f%%'%r['pct']) if r['pct'] is not None else '—'}</td></tr>"
        for r in pt)
    if not pt_rows:
        pt_rows = ("<tr><td colspan='6' class='sub'>Per-phase timeline needs the per-trial JSONs "
                   "(outputs/&lt;run&gt;/trials/), not pulled for this run.</td></tr>")
    kl_rows = "".join(f'<div class="caveat"><strong>{i}. {t}.</strong> {x}</div>'
                      for i, (t, x) in enumerate(known_limitations(df, nominal), 1))
    oem = oem_residuals(outdir)

    def _oemcell(r):
        return f"{r:,.0f} km" if r is not None else "— (beyond OEM)"
    oem_rows = "".join(
        f"<tr><td>{st}</td><td style='text-align:right'>{gd:.2f}</td>"
        f"<td style='text-align:right'>{_oemcell(r)}</td></tr>" for st, gd, r in oem)
    _v = [r for _, _, r in oem if r is not None]
    oem_summary = (f"Closest {min(_v):,.0f} km near the DRO → up to {max(_v):,.0f} km on the "
                   f"transit legs (synthetic outbound + the epoch-committed planned return). The sim reaches the "
                   f"real outcome (returns &amp; splashes intact) but flies a different path; "
                   f"OEM-seeding the outbound is the next path-fidelity step — see Known limitations."
                   if _v else "OEM reference not available.")

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Artemis I — Monte Carlo Dashboard</title>
<style>
{_DASH_STYLE}
</style></head><body>
<h1>Artemis I Physics-Integrated Monte Carlo</h1>
<p style="color:#6e6e73;font-size:14px;line-height:1.5">{n} simulated Artemis I missions — the first
uncrewed flight of SLS Block 1 + Orion. Each trial runs full ODE integration of every mission phase from
Kennedy LC-39B liftoff through Pacific splashdown ({os.path.basename(outdir)}{compute_str}).</p>

<div class="summary-card">
<div class="stats">
  <div class="stat"><div class="label">Trials</div><div class="value">{n}</div></div>
  <div class="stat"><div class="label">Mission Success</div><div class="value">{100*p:.1f}%</div>
    <div style="color:#6e6e73;font-size:12px;margin-top:2px">Wilson 95% CI {100*lo:.1f}–{100*hi:.1f}%</div></div>
  <div class="stat"><div class="label">Splash-miss median</div><div class="value">{splash_med} km</div>
    <div style="color:#6e6e73;font-size:12px;margin-top:2px">real ~4.7 km</div></div>
</div>
</div>

<div class="context">
<strong>What this simulation does.</strong> Each trial samples ascent engine-out / ignition events,
engine + insertion dispersions, orbit-determination nav-knowledge and burn-execution errors, and the
sourced failure model, then runs the full uncrewed Artemis I mission with real-physics ODE integration —
pad to splashdown as ONE continuous trajectory (SLS ascent → ICPS TLI → outbound OTC correction chain →
powered lunar flyby → DRO insertion / 6-day stay / departure → return powered flyby → skip entry →
splashdown). <strong>Uncrewed → success = SLS delivers Orion AND Orion returns &amp; splashes down
intact</strong> (there is no crew-survival model). When a mission fails, the cause is recorded as a
<code>mission_failure</code> value; the Failure Analysis below explains each in plain language.
Failure-mode probabilities are estimates (heritage reliabilities + NASA PRA + foreseeable-problem
analysis) — sourced from NASA / AAS references.
</div>

<h2>Mission Outcome</h2>
<div class="plot">{img('fig_outcome.png')}</div>

<h2>Failure Analysis — what went wrong in the {n - succ} failed missions</h2>
<div class="plot">{img('fig_failure_modes.png')}</div>
<table><tr><th style='text-align:right'>count</th><th>failure mode</th><th>what it means</th></tr>{fa_rows}</table>

<h2>Nominal Trajectory vs Artemis I Actuals</h2>
<div class="context">The simulated NOMINAL (unperturbed) mission against the as-flown Artemis I values
(AAS 23-363 + the post-TLI Orion ephemeris).</div>
<table><tr><th>quantity</th><th>sim (nominal)</th><th>Artemis I</th><th style='text-align:right'>error</th></tr>{cc_rows}</table>
<p style="color:#6e6e73;font-size:13px;margin-top:6px">Error = signed % of the sim nominal vs the Artemis I actual. Entry velocity is a set input (<strong>—</strong>). Splashdown shows no % (a coordinate, not a scalar): the nominal targets the PLANNED zone off San Diego, while Artemis I's as-flown Guadalupe splash was a weather retarget (see Known Limitations).</p>

<h2>Trajectory Fidelity vs the As-Flown OEM</h2>
<div class="context">How far the <em>simulated</em> nominal Orion is from the <em>actual</em> Artemis I
trajectory (NASA's as-flown OEM ephemeris, frame-aligned) at each mission stage. Small = the sim flies
the real path; large = same outcome (returns &amp; splashes intact), different path through space.</div>
<table><tr><th>mission stage</th><th style='text-align:right'>GET (days)</th><th style='text-align:right'>distance from real Orion</th></tr>{oem_rows}</table>
<p style="color:#6e6e73;font-size:13px;margin-top:6px">{oem_summary}</p>

<h2>Phase Timing vs Artemis I</h2>
<table><tr><th>phase</th><th>min</th><th>avg</th><th>max</th><th>Artemis I</th><th style='text-align:right'>Δ avg</th></tr>{pt_rows}</table>

<h2>Known Limitations</h2>
{kl_rows}
<p style="color:#6e6e73;font-size:13px;margin-top:24px">Generated by artemis_outputs.py · pad-to-splashdown is one continuously-propagated trajectory (outbound + FLOWN return, inbound MCC live). See realism_audit_artifact.html.</p>
</body></html>"""
    with open(os.path.join(outdir, "dashboard.html"), "w") as f:
        f.write(html)
    # ARTIFACT-ready copy via html_to_pdf.to_artifact_html: strips the doc wrappers (the claude.ai Artifact host supplies its own),
    # keeps <title> + <style> + body, and warns on any external asset ref (the dashboard is self-contained
    # — figures base64-inlined — so it satisfies the Artifact CSP). Falls back to an inline strip if the
    # shared tool isn't importable. Written IN THE RUN DIR (outputs/<run>/) alongside results.csv — NOT
    # the project root: every run keeps its OWN dashboard + artifact so a later, non-authoritative run
    # can't overwrite the authoritative run's.
    try:
        import html_to_pdf
        html_to_pdf.to_artifact_html(os.path.join(outdir, "dashboard.html"),
                                     os.path.join(outdir, "dashboard_artifact.html"))
    except Exception as e:
        _art = (html.replace('<!doctype html><html><head><meta charset="utf-8">', '')
                    .replace('</head><body>', '').replace('</body></html>', '').strip())
        with open(os.path.join(outdir, "dashboard_artifact.html"), "w") as f:
            f.write(_art)
        print(f"  (shared to_artifact_html unavailable: {e}; used inline strip)")


def main(outdir=OUTDIR):
    df = pd.read_csv(os.path.join(outdir, "results.csv"))
    if "trial" in df.columns:
        df = df[df["trial"] >= 0]
    npath = os.path.join(outdir, "nominal_results.json")
    nominal = json.load(open(npath)) if os.path.exists(npath) else {}
    generate_plots(df, outdir)
    print(generate_summary(df, nominal, outdir))
    generate_dashboard(df, nominal, outdir)
    print(f"\nwrote {outdir}/: dashboard.html, dashboard_artifact.html, summary.txt, fig_*.png (all run-local)")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else OUTDIR)
