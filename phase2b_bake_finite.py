"""Phase-2b FORCED-IGNITION TLI RE-BAKE — offline solve tool.

Solves the (ignition GET, post-TLI cutoff velocity) pair so that the PRODUCTION finite
TLI — flown by the sim's own phase chain (launch → PRM → TLI) under the CURRENT default
force model — passes through the REAL Artemis I post-TLI state (frame-corrected OEM:
PHASEC_POSTTLI_T_S / _R_M / _V_M). Re-run whenever the force model changes; this run
re-bakes for the Earth J3–J6 zonals (+ SRP), which postdate the 2026-07-02 bake and
leave the outbound OEM residual ~2,091 km where ~1,413 km is recoverable.

Also re-pushes the Phase-C CA target (the real arrival: OEM @ 4.5 d coasted through
CURRENT sim dynamics to lunar closest approach) -> PHASEC_CA_T_S / PHASEC_CA_R_ECI_M.

Prints the bake block to paste into artemis1.py and writes phase2b_bake_v2.json.
Epoch-arithmetic check (a 400-s bake slip once mis-phased a solved burn ~960 km): every printed epoch is
also shown in days for cross-checking against the constants' day-form comments.

  OMP_NUM_THREADS=1 python3 phase2b_bake_finite.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json, time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import artemis1 as A


def coast(s6, t0, t1, rtol=1e-9):
    """Full-force-model coast (gravity_earth_moon = Earth+zonals gated, Moon, Sun, SRP)."""
    if abs(t1 - t0) < 1e-9:
        return np.asarray(s6, float)
    return solve_ivp(lambda tt, y: np.concatenate([y[3:6], A.gravity_earth_moon(y[:3], tt)]),
                     (t0, t1), np.asarray(s6, float), method="DOP853",
                     rtol=rtol, atol=1e-3, max_step=600.0).y[:, -1]


def main():
    print("flags: zonals", A.ENABLE_EARTH_HIGHER_ZONALS, "| SRP", A.ENABLE_SRP,
          "| solar", A.ENABLE_SOLAR_GRAVITY, "| phasec", A.ENABLE_PHASEC_BPLANE, flush=True)
    # the solve silently degenerates (constant residual / wrong burn model) if these are off
    assert A.ENABLE_PHASEC_BPLANE, "ENABLE_PHASEC_BPLANE must be ON (else the forced branch never runs)"
    assert A.ENABLE_FINITE_TLI and A.ENABLE_ICPS_TLI, "finite production TLI must be ON"
    assert os.environ.get("AR1_PHASEC_BAKE", "") != "legacy", "unset AR1_PHASEC_BAKE for a re-solve"
    R_ref = np.asarray(A.PHASEC_POSTTLI_R_M, float)
    V_ref = np.asarray(A.PHASEC_POSTTLI_V_M, float)
    T_ref = float(A.PHASEC_POSTTLI_T_S)

    # -- the production pre-TLI state: launch -> PRM (once; deterministic nominal) --------
    t0w = time.time()
    L = A.phase_sls_launch(None)
    assert L["success"], L
    P = A.phase_icps_prm(np.asarray(L["state"], float), float(L["t_insertion"]), None)
    assert P["success"], P
    pre_tli = np.asarray(P["state"], float)
    t_pre = float(P["t_end"])
    print(f"launch+PRM ready (+{time.time()-t0w:.0f}s); pre-TLI t={t_pre:.2f}s", flush=True)

    # the old bake = warm start (and the A/B "before" point)
    x0 = np.array([float(A.PHASEC_TLI_IGN_S), *np.asarray(A.PHASEC_TLI_VPOST, float)])

    # -- CA-target re-push FIRST (independent; also the solve's aim point) ----------------
    # OEM @ 4.5 d coasted through CURRENT sim dynamics to lunar closest approach — the real
    # arrival expressed on the sim's own dynamics, i.e. the OTC chain's baked corrector target.
    from oem_loader import load_oem
    g, R, V = load_oem()
    i45 = int(np.argmin(np.abs(g - 4.5 * 86400)))
    s45 = np.concatenate([R[i45] * 1e3, V[i45] * 1e3])
    arc2 = solve_ivp(lambda tt, y: np.concatenate([y[3:6], A.gravity_earth_moon(y[:3], tt)]),
                     (float(g[i45]), float(g[i45]) + 1.2 * 86400), s45, method="DOP853",
                     rtol=1e-10, atol=1e-3, max_step=600.0, dense_output=True)
    dmin2, tca2 = 1e30, None
    for tt in np.linspace(float(g[i45]), float(g[i45]) + 1.2 * 86400, 4000):
        d = np.linalg.norm(arc2.sol(tt)[:3] - A.moon_state(tt)[0])
        if d < dmin2:
            dmin2, tca2 = d, tt
    half = 300.0
    for _ in range(20):                       # SHRINKING bracket: epoch resolved to ~us class
        for tt in np.linspace(tca2 - half, tca2 + half, 61):
            d = np.linalg.norm(arc2.sol(tt)[:3] - A.moon_state(tt)[0])
            if d < dmin2:
                dmin2, tca2 = d, tt
        half = max(half / 5.0, 1e-4)
    ca_r = np.asarray(arc2.sol(tca2)[:3], float)
    old_ca_t = float(A.PHASEC_CA_T_S)
    old_ca_r = np.asarray(A.PHASEC_CA_R_ECI_M, float)
    print(f"CA-target re-push: t {tca2:.3f}s ({tca2/86400:.4f} d; old {old_ca_t/86400:.4f} d, "
          f"dt {tca2-old_ca_t:+.2f}s), |dr| vs old {np.linalg.norm(ca_r-old_ca_r)/1e3:.2f} km, "
          f"alt {(dmin2-A.R_MOON)/1e3:.2f} km (old solve: 146.09)", flush=True)

    # -- the solve: fly the REAL lunar approach --------------------------------------------
    # Objective = CA-POINT miss at the CA epoch (the approach anchor the whole OTC chain
    # targets), NOT the fixed-epoch post-TLI 6-state: that residual is STRUCTURAL (~210 km /
    # ~165 m/s — the sim cannot inject exactly onto the real trajectory from its own parking
    # orbit; the original bake accepted a ~6 deg asymptote residual), and a v1 solve on it
    # walks out of the correct trans-lunar family (measured: periselene 91,386 km). A small
    # warm-start anchor keeps the solve in-basin and ~dv-neutral.
    def fly(x):
        A.PHASEC_TLI_R_IGN_M = None
        A.PHASEC_TLI_IGN_S = float(x[0])
        A.PHASEC_TLI_VPOST = tuple(float(v) for v in x[1:4])
        tli = A.phase_icps_tli(pre_tli.copy(), t_pre, None)
        if not tli.get("success"):
            return None
        s6 = np.asarray(tli["state"][:6], float)
        return tli, coast(s6, float(tli["t_end"]), float(tca2))

    def resid(x):
        out = fly(x)
        if out is None:
            return np.full(7, 1e6)
        _, sCA = out
        anchor = (x - x0) * np.array([1e-3, 1e-4, 1e-4, 1e-4])   # in-basin ridge (negligible bias)
        return np.concatenate([(sCA[:3] - ca_r) / 1e3, anchor])

    r0 = resid(x0)
    print(f"OLD bake under CURRENT model: CA-point miss {np.linalg.norm(r0[:3]):.1f} km", flush=True)

    sol = least_squares(resid, x0, method="trf",
                        diff_step=[2e-6, 5e-6, 5e-6, 5e-6],
                        x_scale=[1.0, 10.0, 10.0, 10.0], xtol=1e-12, ftol=1e-12)
    rf = sol.fun
    t_ign, vpost = float(sol.x[0]), tuple(float(v) for v in sol.x[1:4])
    print(f"NEW bake: CA-point miss {np.linalg.norm(rf[:3]):.1f} km  (nfev {sol.nfev}, +{time.time()-t0w:.0f}s)", flush=True)
    print(f"  shift vs old: dt_ign={t_ign - x0[0]:+.4f} s  |dvpost|={np.linalg.norm(sol.x[1:4]-x0[1:4]):.4f} m/s")
    # acceptance gates: no regression, and the solved ignition must sit inside the production
    # FIRST-REV pass-search window (t_pre+300, t_pre+min(9600, one parking period)] — outside it
    # the fleet would ignite one rev early at the baked r_ign (the wrong-true-anomaly impact class)
    assert np.linalg.norm(rf[:3]) < np.linalg.norm(r0[:3]), "REGRESSION: new CA miss worse than old — do not bake"
    _Es = 0.5 * float(np.dot(pre_tli[3:6], pre_tli[3:6])) - A.MU_EARTH / np.linalg.norm(pre_tli[:3])
    _Tpark = 2 * np.pi * np.sqrt((-A.MU_EARTH / (2 * _Es)) ** 3 / A.MU_EARTH)
    assert t_pre + 300.0 < t_ign < t_pre + min(9600.0, _Tpark), \
        f"solved t_ign {t_ign:.1f} outside the production rev-1 window ({t_pre+300:.0f}, {t_pre+min(9600.0,_Tpark):.0f})"
    # post-TLI-state miss as a REPORT metric (before/after), not the objective
    for tag, xx in (("old", x0), ("new", sol.x)):
        out = fly(xx)
        if out is not None:
            tli_, _ = out
            sT = coast(np.asarray(tli_["state"][:6], float), float(tli_["t_end"]), T_ref)
            print(f"  post-TLI-state miss ({tag}): |dr|={np.linalg.norm(sT[:3]-R_ref)/1e3:.1f} km "
                  f"|dv|={np.linalg.norm(sT[3:6]-V_ref):.1f} m/s  tli_dv {tli_['tli_dv_ms']:.1f}")

    # -- reconstruct the ignition POSITION on the nominal parking orbit -------------------
    park = solve_ivp(lambda tt, y: np.concatenate([y[3:6], A.gravity_earth_moon(y[:3], tt)]),
                     (t_pre, t_ign + 60.0), pre_tli[:6], method="DOP853",
                     rtol=1e-9, atol=1e-3, max_step=60.0, dense_output=True)
    r_ign = tuple(float(v) for v in park.sol(t_ign)[:3])

    # -- approach diagnostics with the solved bake (production-consistent ignition) -------
    A.PHASEC_TLI_IGN_S, A.PHASEC_TLI_VPOST, A.PHASEC_TLI_R_IGN_M = t_ign, vpost, r_ign
    tli = A.phase_icps_tli(pre_tli.copy(), t_pre, None)
    s6, te = np.asarray(tli["state"][:6], float), float(tli["t_end"])
    ts = np.linspace(te, float(A.OPF_GET_S) + 40000.0, 3000)
    arc = solve_ivp(lambda tt, y: np.concatenate([y[3:6], A.gravity_earth_moon(y[:3], tt)]),
                    (te, ts[-1]), s6, method="DOP853", rtol=1e-9, atol=1e-3,
                    max_step=1800.0, dense_output=True)
    dmin, tca = 1e30, None
    for tt in ts:
        d = np.linalg.norm(arc.sol(tt)[:3] - A.moon_state(tt)[0])
        if d < dmin:
            dmin, tca = d, tt
    for _ in range(60):  # refine CA
        for tt in np.linspace(tca - 900, tca + 900, 61):
            d = np.linalg.norm(arc.sol(tt)[:3] - A.moon_state(tt)[0])
            if d < dmin:
                dmin, tca = d, tt
    peri_km = (dmin - A.R_MOON) / 1e3
    print(f"  approach: CA {tca/86400:.4f} d (real 5.257), periselene {peri_km:.0f} km "
          f"(pre-OTC; old bake gave ~257), TLI dv {tli['tli_dv_ms']:.1f} m/s", flush=True)

    # -- bake block ------------------------------------------------------------------------
    ca_out = tuple(float(v) for v in ca_r)
    print("\n=== RE-BAKE (paste into artemis1.py; epochs also in days for the slip check) ===")
    print(f"PHASEC_TLI_IGN_S   = {t_ign!r}   # {t_ign/86400:.6f} d")
    print(f"PHASEC_TLI_R_IGN_M = {r_ign!r}")
    print(f"PHASEC_TLI_VPOST   = {vpost!r}")
    print(f"PHASEC_CA_T_S      = {float(tca2)!r}   # {tca2/86400:.6f} d")
    print(f"PHASEC_CA_R_ECI_M  = {ca_out!r}")
    json.dump(dict(tli_ign_s=t_ign, tli_r_ign_m=r_ign, tli_vpost=vpost,
                   ca_t_s=float(tca2), ca_r_eci_m=ca_out,
                   ca_miss_old_km=float(np.linalg.norm(r0[:3])),
                   ca_miss_new_km=float(np.linalg.norm(rf[:3])),
                   peri_km=float(peri_km), tca_d=float(tca / 86400)),
              open("phase2b_bake_v2.json", "w"), indent=2)
    print("  (saved -> phase2b_bake_v2.json)")


if __name__ == "__main__":
    main()
