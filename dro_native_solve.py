"""Ephemeris-native reference DRO — OFFLINE CONSTRUCTION.

Fits ONE BALLISTIC ARC OF THE SIM'S OWN FULL FORCE MODEL (gravity_earth_moon: Earth+J2
[+gated J3-J6], Moon [+gated harmonics], Sun, SRP) to the as-flown OEM DRO segment over
the stay (DRI-margin -> DDP+margin), by least-squares on the state at the DRI epoch.
The result is ON-MANIFOLD BY CONSTRUCTION — a genuine solution of our dynamics with the
real DRO's elliptical geometry class — which is exactly what the two REJECTED approaches
were not:
  * ENABLE_OEM_DRO_REF (declined 2026-07-03): the raw OEM state is off-manifold — it
    drifts 63,445 km / 380 m/s over the 6-day coast when propagated UNcorrected.
  * ER3BP re-map (deleted 2026-07-04): reshaping the target map cannot fix coast drift.

Geometry sourcing (as-planned doctrine): NO design DRO geometry is published in our
research record, so the OEM DRO arc is used as the GEOMETRY-CLASS anchor — a tagged
AS-FLOWN STAND-IN (the doctrine's plan-not-recoverable rule, same tag as PHASEC_RET_*).
The arc's DYNAMICS are ours alone; the fit only selects which solution of our dynamics
to use. Known, accepted mismatch: the real orbit was maintained (OM-1 ~0.02, OM-3
~13.2 m/s near departure) while our arc is ballistic — node weights TAPER toward the DDP
end where the unmodeled OM-3 concentrates the difference.

Outputs: prints the bake block (anchor epoch + corrected 6-state) for artemis1.py's
ENABLE_NATIVE_DRO provider + writes dro_native_bake.json with fit + validation
diagnostics (node-miss profile, peri/apo class, boundedness over an extended span).

  OMP_NUM_THREADS=1 python3 dro_native_solve.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json, time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import artemis1 as A
from oem_loader import load_oem

T_DRI = float(A.PHASEC_DRI_GET_S)          # 831,884 s = 9.6284 d (as-flown DRI epoch)
T_DDP = float(A.PHASEC_RET_DRD_T_S)        # 1,350,371.52 s = 15.6293 d (as-flown DRD epoch)
MARGIN_S = 0.35 * 86400.0                  # arc extends past both ends for targeting slack
NODE_DT_S = 6 * 3600.0                     # OEM fit nodes every 6 h


def prop(s6, t0, t1, rtol=1e-9, dense=False):
    r = solve_ivp(lambda tt, y: np.concatenate([y[3:6], A.gravity_earth_moon(y[:3], tt)]),
                  (t0, t1), np.asarray(s6, float), method="DOP853",
                  rtol=rtol, atol=1e-3, max_step=1800.0, dense_output=dense)
    return r if dense else r.y[:, -1]


def main():
    t0w = time.time()
    assert A.ENABLE_EARTH_HIGHER_ZONALS and A.ENABLE_SRP and A.ENABLE_SOLAR_GRAVITY, \
        "run under the production default force model"
    g, R, V = load_oem()                                    # km, km/s, sim mean-of-date frame
    lo, hi = T_DRI - MARGIN_S, T_DDP + MARGIN_S
    node_t = np.arange(T_DRI, T_DDP + 1.0, NODE_DT_S)
    node_r = np.array([[np.interp(t, g, R[:, k]) * 1e3 for k in range(3)] for t in node_t])
    # taper: full weight through mid-stay, cosine-fade over the last 2 days (unmodeled OM-3)
    w = np.ones(len(node_t))
    fade = node_t > (T_DDP - 2 * 86400.0)
    w[fade] = 0.5 * (1 + np.cos(np.pi * (node_t[fade] - (T_DDP - 2 * 86400.0)) / (2 * 86400.0)))
    w = np.sqrt(np.maximum(w, 0.05))
    print(f"fit span {lo/86400:.3f} -> {hi/86400:.3f} d; {len(node_t)} nodes, 6 h spacing", flush=True)

    # warm start: the OEM state AT the DRI epoch (off-manifold; the fit corrects it)
    x0 = np.concatenate([[np.interp(T_DRI, g, R[:, k]) * 1e3 for k in range(3)],
                         [np.interp(T_DRI, g, V[:, k]) * 1e3 for k in range(3)]])

    def arc_states(x):
        """States at all nodes for anchor state x at T_DRI (one backward + one forward leg)."""
        fwd = prop(x, T_DRI, node_t[-1], dense=True)
        out = np.empty((len(node_t), 3))
        for i, t in enumerate(node_t):
            out[i] = fwd.sol(t)[:3] if t >= T_DRI else np.nan
        return out

    def resid(x):
        try:
            rr = arc_states(x)
        except Exception:
            return np.full(3 * len(node_t), 1e6)
        if not np.all(np.isfinite(rr)):
            return np.full(3 * len(node_t), 1e6)
        return ((rr - node_r) * w[:, None] / 1e3).ravel()

    r0 = resid(x0)
    print(f"warm start (raw OEM state): weighted node-miss RMS "
          f"{np.sqrt(np.mean(np.sum((r0.reshape(-1,3))**2, axis=1))):.0f} km", flush=True)

    sol = least_squares(resid, x0, method="trf", diff_step=1e-7,
                        x_scale=[1e6, 1e6, 1e6, 1.0, 1.0, 1.0], xtol=1e-14, ftol=1e-12)
    rf = sol.fun.reshape(-1, 3)
    rms = np.sqrt(np.mean(np.sum(rf ** 2, axis=1)))
    per_node = np.linalg.norm((arc_states(sol.x) - node_r) / 1e3, axis=1)
    print(f"fit done: weighted RMS {rms:.0f} km (nfev {sol.nfev}, +{time.time()-t0w:.0f}s)", flush=True)
    print("  raw node-miss profile (km):")
    for i in range(0, len(node_t), 4):
        print(f"    GET {node_t[i]/86400:7.3f} d   miss {per_node[i]:9.1f}")
    print(f"    GET {node_t[-1]/86400:7.3f} d   miss {per_node[-1]:9.1f}  (DDP end, tapered)")

    # -- validation: boundedness + geometry class over the extended span ------------------
    ext = prop(sol.x, T_DRI, hi + 3 * 86400.0, dense=True)          # +3 d past DDP margin
    back = prop(sol.x, T_DRI, lo - 1 * 86400.0, dense=True)         # backward past DRI margin
    dmoon, tgrid = [], np.linspace(lo - 1 * 86400.0, hi + 3 * 86400.0, 4000)
    for t in tgrid:
        s = back.sol(t) if t < T_DRI else ext.sol(t)
        dmoon.append(np.linalg.norm(s[:3] - A.moon_state(t)[0]))
    dmoon = np.array(dmoon) / 1e3
    print(f"  Moon-range over extended span: min {dmoon.min():,.0f} km, max {dmoon.max():,.0f} km "
          f"(real elliptical class ~71,000-94,000; CR3BP was ~73,000 circular)")
    print(f"  bounded: {'YES' if dmoon.max() < 150000 and dmoon.min() > 20000 else 'NO -- REJECT'}")
    # state offsets vs OEM at the two anchor epochs (what the OPF/DRI + return solves absorb)
    for tag, te in (("DRI", T_DRI), ("DDP", T_DDP)):
        so = ext.sol(te) if te >= T_DRI else back.sol(te)
        ro = np.array([np.interp(te, g, R[:, k]) * 1e3 for k in range(3)])
        vo = np.array([np.interp(te, g, V[:, k]) * 1e3 for k in range(3)])
        print(f"  vs OEM at {tag}: |dr| {np.linalg.norm(so[:3]-ro)/1e3:8.1f} km   "
              f"|dv| {np.linalg.norm(so[3:6]-vo):7.2f} m/s")

    ok = bool(dmoon.max() < 150000 and dmoon.min() > 20000 and rms < 5000)
    print("\n=== NATIVE-DRO BAKE (paste into artemis1.py) ===" if ok else
          "\n=== FIT REJECTED (bounds/RMS) — do not bake ===")
    print(f"NATIVE_DRO_T0_S    = {float(T_DRI)!r}   # anchor epoch = as-flown DRI GET ({T_DRI/86400:.6f} d)")
    print(f"NATIVE_DRO_R0_M    = {tuple(float(v) for v in sol.x[:3])!r}")
    print(f"NATIVE_DRO_V0_MS   = {tuple(float(v) for v in sol.x[3:6])!r}")
    json.dump(dict(t0_s=float(T_DRI), r0_m=list(map(float, sol.x[:3])),
                   v0_ms=list(map(float, sol.x[3:6])), rms_km=float(rms),
                   node_t_d=list(np.round(node_t / 86400, 4)),
                   node_miss_km=list(np.round(per_node, 1)),
                   moon_range_km=[float(dmoon.min()), float(dmoon.max())],
                   accepted=ok),
              open("dro_native_bake.json", "w"), indent=2)
    print("  (saved -> dro_native_bake.json)")


if __name__ == "__main__":
    main()
