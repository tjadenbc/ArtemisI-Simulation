"""Per-run trial audit — PARALLEL programmatic consistency checker.
Reads a diag rows-dump (local) OR a run's trials/*.json (cluster), audits ALL trials by default (the
checker is ~free — pure dict checks fanned across `workers`), and reports per-trial flags + run-level
population flags + the wallclock. Pass a small maxn only to down-sample diverse trials on a huge run.

`audit_trial` encodes consistency rules (flag↔decision, physical bounds, ESM ledger, failure
attribution) so the known bug classes (e.g. the silent OD-nav no-op) are caught automatically; the
human/LLM then reviews the flagged trials. Usage:
    python3 diag_audit.py <rows.json|run_dir> [maxn] [workers]
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, json, time, glob, multiprocessing as mp


def audit_trial(rec):
    """Consistency-rule audit of one trial's _debug+results. Returns {trial, succ, fail, flags[]}."""
    db = rec.get("_debug", {}) or {}
    fl = db.get("flags", {}) or {}
    succ = bool(rec.get("full_success")); fail = rec.get("mission_failure")
    dri = db.get("dri", {}) or {}
    flags = []
    # 1. flag <-> decision consistency (the silent-no-op class)
    if fl.get("ENABLE_OD_NAV") and db.get("dri_override") is False:
        flags.append("OD_NAV on but dri_override=False (no-op / capture missing)")
    if fl.get("ENABLE_DDP_RECOVERY") and db.get("ddp_path") == "1dof":
        flags.append("DDP_RECOVERY on but ddp_path=1dof (recovery solve fell back)")
    if fl.get("ENABLE_RPF_RECOVERY_TARGET") and db.get("rpf_path") == "velmatch_fallback":
        flags.append("RPF_RECOVERY on but rpf velmatch fallback")
    # 2. physical bounds — launch + TLI stages (data lives under the phase dicts)
    ln = db.get("launch", {}) or {}
    tli = db.get("tli", {}) or {}
    mg = ln.get("max_g")
    if mg is not None and mg > 4.5:
        flags.append(f"launch max_g {mg:.1f} > 4.5 (overstress)")
    az = ln.get("launch_azimuth_deg")
    if az is not None and not (70.0 <= az <= 95.0):
        flags.append(f"launch azimuth {az:.1f} out of [70,95]")
    tdv = tli.get("tli_dv_ms")
    if tdv is not None and tli.get("success") and not (2750.0 <= tdv <= 3250.0):
        flags.append(f"tli_dv {tdv:.0f} m/s out of [2750,3250]")
    icm = tli.get("icps_prop_margin_kg")
    if succ and icm is not None and icm < 0.0:
        flags.append(f"ICPS margin {icm:.0f} kg NEGATIVE on a success (ledger)")
    apo = tli.get("post_tli_apogee_km")
    if succ and apo is not None and apo < 300000.0:
        flags.append(f"post-TLI apogee {apo:.0f} km < 300000 (didn't reach Moon)")
    # 2b. physical bounds — return nav
    pa = db.get("peri_alt_km")
    if pa is not None and not (50.0 <= pa <= 600.0):
        flags.append(f"peri_alt {pa:.0f} km out of [50,600]")
    fpa = db.get("ei_fpa")
    if succ and fpa is not None and not (-8.5 <= fpa <= -4.5):
        flags.append(f"ei_fpa {fpa:.2f} out of corridor [-8.5,-4.5]")
    ddpdv = db.get("ddp_dv")
    if ddpdv is not None and ddpdv > 400.0:
        flags.append(f"ddp_dv {ddpdv:.0f} m/s LARGE (>400)")
    rpfdv = db.get("rpf_dv")
    if rpfdv is not None and not (200.0 <= rpfdv <= 520.0):
        flags.append(f"rpf_dv {rpfdv:.0f} m/s out of [200,520]")
    dridv = dri.get("dri_dv_ms")
    if dridv is not None and dridv > 300.0:
        flags.append(f"dri_dv {dridv:.0f} m/s LARGE (>300) — ESM drain")
    # 3. ESM ledger
    esm = rec.get("esm_prop_remaining_kg")
    if succ and esm is not None and esm < 300.0:
        flags.append(f"ESM margin LOW ({esm:.0f} kg) on a success")
    # 4. failure attribution
    if fail and "depleted" in str(fail):
        flags.append(f"FAIL {fail} | dri_dv={round(dridv or -1)} ddp_dv={round(ddpdv or -1)} rpf_dv={round(rpfdv or -1)}")
    # 5. displacement sanity (the OD-nav goal)
    disp = rec.get("recovery_zone_displacement_km")
    if succ and disp is not None and disp > 2000.0:
        flags.append(f"displacement {disp:.0f} km (not collapsed)")
    # 6. cross-stage continuity + fidelity milestones (calibrated to the run; real Artemis I anchors)
    te = db.get("transearth", {}) or {}
    #   ins->TLI-IGNITION ~81 min (real Artemis). Compare to the IGNITION time (tli_ign_t_s), NOT the
    #   tli t_end which is now the finite-burn BURNOUT (~9-14 min later) — comparing burnout to the
    #   ignition reference false-flagged every trial at ~95 min. A real rev-slip adds the ~105-min
    #   parking period -> ~186 min, so the band still catches it. (Falls back to t_end if no ignition time.)
    t_ins = ln.get("t_insertion")
    t_tli_ign = tli.get("tli_ign_t_s")
    if t_ins is not None and t_tli_ign is not None and tli.get("success"):
        m = (t_tli_ign - t_ins) / 60.0
        if not (72.0 <= m <= 105.0):
            flags.append(f"ins->TLI-ign {m:.0f} min out of [72,105] (rev-slip? real ~81)")
    #   max-Earth distance = the DRO-apogee milestone (real Artemis I 432,194 km)
    mE = rec.get("max_earth_distance_km")
    if succ and mE is not None and not (428000.0 <= mE <= 437000.0):
        flags.append(f"max-Earth {mE:.0f} km off the 432,194 km milestone")
    #   EI velocity = lunar-return speed
    eiv = te.get("entry_velocity_ms")
    if succ and eiv is not None and not (10700.0 <= eiv <= 11200.0):
        flags.append(f"EI velocity {eiv:.0f} m/s out of [10700,11200]")
    #   epoch monotonicity: DRO insert < DRO departure < entry interface
    ep = [db.get("dri_t_d"), db.get("t_dep_d"), db.get("t_ei_d")]
    if all(isinstance(x, (int, float)) for x in ep) and not (ep[0] < ep[1] < ep[2]):
        flags.append(f"epochs non-monotone dri/dep/ei = {[round(x,2) for x in ep]}")
    #   failure-truncation consistency: a SUCCESS must have flown the full chain to entry
    if succ and not ((db.get("entry", {}) or {}).get("success")):
        flags.append("marked success but entry phase not successful (attribution)")
    return {"trial": rec.get("trial"), "succ": succ, "fail": fail, "flags": flags}


def _load(src):
    if os.path.isdir(src):
        out = []
        for f in glob.glob(os.path.join(src, "trials", "trial_*.json")):
            if "nominal" in f:
                continue
            try:
                out.append(json.load(open(f)))
            except Exception:
                pass
        return out
    return json.load(open(src))


def _select(rows, maxn):
    """The checker is ~free, so audit ALL trials when the run has <= maxn (the common case: local
    <=25, small cluster 250). For huge runs (>maxn), sample maxn DIVERSE trials: each failure mode,
    then a displacement-spread of successes."""
    if len(rows) <= maxn:
        return rows
    sel, seen = [], set()
    for r in rows:
        if not r.get("full_success"):
            fm = r.get("mission_failure")
            if fm not in seen:
                seen.add(fm); sel.append(r)
    oks = sorted([r for r in rows if r.get("full_success")],
                 key=lambda r: r.get("recovery_zone_displacement_km") or 0.0)
    for r in oks[:: max(1, len(oks) // max(1, maxn - len(sel)))]:   # spread across the displacement range
        if r not in sel:
            sel.append(r)
    return sel[:maxn]


def _population_checks(rows):
    """Run-level checks the per-trial pass can't see: over-accuracy (the fidelity bound — the sim
    must not land TIGHTER than the real ~4.7 km), nav-fragility regressions, gross success-rate drift."""
    import numpy as np
    flags = []
    n = len(rows); ok = [r for r in rows if r.get("full_success")]; k = len(ok)
    rate = 100.0 * k / n if n else 0.0
    def med(key):
        v = np.array([r[key] for r in ok if isinstance(r.get(key), (int, float)) and r[key] == r[key]])
        return float(np.median(v)) if len(v) else None
    sm = med("splash_miss_km"); dp = med("recovery_zone_displacement_km"); pg = med("entry_peak_g")
    # OVER-ACCURACY: real Artemis I splashed ~4.7 km from target; the sim must not BEAT that. The entry
    # is now calibrated to ~4.7 km (ENTRY_NAV_RESIDUAL), so the median SHOULD sit ~4.5-5.0; flag only a
    # median clearly TIGHTER than real (< 4.0 km, below the ~3.5 km physical chute-wind floor) — that
    # would mean the guidance/nav was made artificially tight.
    if sm is not None and sm < 4.0:
        flags.append(f"OVER-ACCURATE: splash_miss median {sm:.1f} km < real ~4.7 km (beats real; fidelity bound)")
    # nav-fragility modes should be ~0 (regression signal)
    nav = [r.get("mission_failure") for r in rows if r.get("mission_failure")
           and any(s in str(r.get("mission_failure")) for s in ("missed_soi", "ddp_no_earth", "depleted", "no_earth_return"))]
    if nav:
        from collections import Counter
        flags.append(f"NAV-FRAGILITY modes present (should be ~0): {dict(Counter(nav))}")
    # gross success-rate drift (band; soft)
    if n >= 100 and not (88.0 <= rate <= 98.0):
        flags.append(f"success rate {rate:.1f}% outside the expected [88,98] band")
    return flags, (rate, sm, dp, pg)


def main():
    t0 = time.time()
    src = sys.argv[1]
    # Default: audit ALL trials (the checker is ~free — pure dict checks, parallel). Pass a small
    # maxn only to down-sample DIVERSE trials on a very large run if the JSON load itself is slow.
    maxn = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    rows = _load(src)
    sel = _select(rows, maxn)
    with mp.Pool(min(workers, max(1, len(sel)))) as pool:
        res = pool.map(audit_trial, sel)
    flagged = [r for r in res if r["flags"]]
    print(f"=== audited {len(sel)}/{len(rows)} trials ({workers}-way parallel) ===")
    pop_flags, (rate, sm, dp, pg) = _population_checks(rows)
    print(f"population: success {rate:.1f}% | splash_miss med {sm} | displacement med {dp} | peak_g med {pg}")
    for f in pop_flags:
        print(f"  !! {f}")
    # Print ONLY the flagged trials (concise even at 250) ...
    for r in flagged:
        print(f"trial {r['trial']:4} succ={str(r['succ'])[:5]:5} fail={str(r['fail'])[:30]:30} FLAGS:")
        for f in r["flags"]:
            print(f"        - {f}")
    # ... plus a grouped summary so a 250-trial sweep is reviewable at a glance.
    if flagged:
        from collections import Counter
        cats = Counter(f.split(" m/s")[0].split(" km")[0].split(" (")[0].strip()
                       for r in flagged for f in r["flags"])
        print("\nflag categories:")
        for cat, c in cats.most_common():
            print(f"  {c:4} x {cat}")
    print(f"\n{len(flagged)}/{len(sel)} trials flagged + {len(pop_flags)} population flags | "
          f"audit wallclock {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
