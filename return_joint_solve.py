"""Joint DDP+RPF return solve — offline solve tool: a 6x6
trust-region Newton over the (DDP, RPF) burn vectors, burns at the REAL epochs
(PHASEC_RET_DRD_T_S / PHASEC_RET_RPF_T_S), from a pre-DRD state s0, targeting the real
trans-earth match state at 21.5 d (PHASEC_RET_MATCH_*). The residual is IDENTICAL to
_phasec_return_plan's _match_resid (position/1e3 km + velocity m/s over full-dynamics
coasts), so the solved vectors drop straight into the PHASEC_RET_DDP_DV / RPF_DV /
S0_REF bake.

Modes:
  python3 return_joint_solve.py ref            # REGRESSION: solve from the baked
      PHASEC_RET_S0_REF with the baked vectors as warm start — must reproduce the
      current bake (validates the recreated tool before trusting it on a new reference).
  python3 return_joint_solve.py s0 <file.json> # solve from {"t": <s0 epoch s>,
      "s0": [x,y,z,vx,vy,vz]} (e.g. the new-reference nominal's coasted pre-DRD state;
      if t < DRD epoch the state is coasted forward to it first). Warm start = baked
      vectors, falling back to the real-vector seeds on non-convergence.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json, sys, time
import numpy as np
import multiprocessing as mp
from scipy.optimize import least_squares
import artemis1 as A

T1 = float(A.PHASEC_RET_DRD_T_S)        # 15.6293 d — DDP/DRD epoch
T2 = float(A.PHASEC_RET_RPF_T_S)        # 19.4097 d — RPF epoch
TM = float(A.PHASEC_RET_MATCH_T_S)      # 21.5 d — trans-earth match epoch
R_M = np.asarray(A.PHASEC_RET_MATCH_R_M, float)
V_M = np.asarray(A.PHASEC_RET_MATCH_V_MS, float)


def match_resid(dv1, dv2, s0):
    """Identical formulation to _phasec_return_plan._match_resid."""
    sA = np.concatenate([s0[:3], s0[3:6] + dv1])
    sB = A._coast_rv(sA, T1, T2)
    sC = A._coast_rv(np.concatenate([sB[:3], sB[3:6] + dv2]), T2, TM)
    return np.concatenate([(sC[:3] - R_M) / 1e3, sC[3:6] - V_M])


_POOL = None


def _remote_resid(args):
    x, s0 = args
    return match_resid(np.asarray(x[:3]), np.asarray(x[3:6]), np.asarray(s0))


def solve(s0, x0, max_nfev=None, pool=None):
    """LM solve of the 6x6 match. With `pool`, the finite-difference Jacobian's 6
    perturbation evaluations run in PARALLEL workers (each residual eval is two ~2 s
    full-dynamics coasts — the Jacobian is the hot loop; the continuation ladder itself
    is inherently serial). max_nfev counts residual-only evaluations (trial steps) when
    an analytic jac is supplied, so ~25 caps a missed solve at a few minutes."""
    def f(x):
        return match_resid(x[:3], x[3:6], s0)
    if pool is None:
        return least_squares(f, x0, method="lm", diff_step=1e-6, xtol=1e-14, ftol=1e-14,
                             max_nfev=max_nfev)

    def pjac(x):
        h = 1e-6 * np.maximum(np.abs(x), 1.0)
        pts = [(x + h[i] * np.eye(6)[i], s0) for i in range(6)]
        base = f(x)
        cols = pool.map(_remote_resid, pts)
        J = np.empty((6, 6))
        for i in range(6):
            J[:, i] = (cols[i] - base) / h[i]
        return J

    return least_squares(f, x0, jac=pjac, method="lm", xtol=1e-14, ftol=1e-14,
                         max_nfev=max_nfev)


def report(sol, s0, tag):
    dv1, dv2 = sol.x[:3], sol.x[3:6]
    r = sol.fun
    print(f"[{tag}] converged: |resid| pos {np.linalg.norm(r[:3]):.3f} km, "
          f"vel {np.linalg.norm(r[3:]):.4f} m/s  (nfev {sol.nfev})")
    print(f"  DDP |dv| {np.linalg.norm(dv1):8.3f} m/s   (as-flown DRD+OM-3 151.5; design 145.2; flown DRD 138.5)")
    print(f"  RPF |dv| {np.linalg.norm(dv2):8.3f} m/s   (as-flown 292.9)")
    print("\n=== bake block ===")
    print(f"PHASEC_RET_DDP_DV        = {tuple(float(v) for v in dv1)!r}")
    print(f"PHASEC_RET_RPF_DV        = {tuple(float(v) for v in dv2)!r}")
    print(f"PHASEC_RET_S0_REF        = {tuple(float(v) for v in s0)!r}")
    json.dump(dict(ddp_dv=list(map(float, dv1)), rpf_dv=list(map(float, dv2)),
                   s0_ref=list(map(float, s0)), resid_pos_km=float(np.linalg.norm(r[:3])),
                   resid_vel_ms=float(np.linalg.norm(r[3:]))),
              open(f"return_joint_{tag}.json", "w"), indent=2)
    print(f"  (saved -> return_joint_{tag}.json)")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "ref"
    x_baked = np.concatenate([np.asarray(A.PHASEC_RET_DDP_DV, float),
                              np.asarray(A.PHASEC_RET_RPF_DV, float)])
    t0 = time.time()
    if mode == "ref":
        s0 = np.asarray(A.PHASEC_RET_S0_REF, float)
        sol = solve(s0, x_baked)
        # regression: must reproduce the current bake (the residual at the baked vectors
        # is ~0 by construction, so the solve should return ~the baked x)
        d1 = np.linalg.norm(sol.x[:3] - np.asarray(A.PHASEC_RET_DDP_DV, float))
        d2 = np.linalg.norm(sol.x[3:6] - np.asarray(A.PHASEC_RET_RPF_DV, float))
        print(f"regression vs current bake: |d DDP| {d1:.4f} m/s, |d RPF| {d2:.4f} m/s "
              f"({'PASS' if max(d1, d2) < 0.5 else 'FAIL — tool does not reproduce the bake'})")
        report(sol, s0, "ref")
    elif mode == "s0":
        d = json.load(open(sys.argv[2]))
        s0_new = np.asarray(d["s0"], float)
        t_s0 = float(d.get("t", T1))
        if t_s0 < T1 - 1.0:
            s0_new = A._coast_rv(s0_new, t_s0, T1)
        # CONTINUATION: a direct solve from a far s0
        # leaves the cheap two-burn family (measured: RPF ~1,959 m/s wrong-family).
        # We hold an EXACT solution at the old PHASEC_RET_S0_REF — walk s0 old->new,
        # re-solving warm-started at each step; adaptive halving keeps every step in-basin.
        s0_old = np.asarray(A.PHASEC_RET_S0_REF, float)
        x = x_baked.copy()
        # measured basin width ~ lam 0.03: start there, grow gently, FAIL FAST on misses
        # (an lm run to its default cap costs ~20 min; max_nfev=120 caps a miss at ~4 min,
        # while warm-started SUCCESSES converge in a handful of evaluations regardless).
        lam, step = 0.0, 0.03
        n_solves = 0
        pool = mp.Pool(6)                      # parallel FD-Jacobian workers
        while lam < 1.0 - 1e-12:
            lam_try = min(1.0, lam + step)
            s0_l = s0_old + lam_try * (s0_new - s0_old)
            sol = solve(s0_l, x, max_nfev=25, pool=pool)
            n_solves += 1
            tight = (np.linalg.norm(sol.fun[:3]) < 0.05 and np.linalg.norm(sol.fun[3:]) < 0.05)
            sane = np.linalg.norm(sol.x[3:6]) < 600.0     # wrong-family guard (as-flown RPF 292.9)
            if tight and sane:
                lam, x = lam_try, sol.x
                print(f"  continuation lam={lam:.4f} ok  DDP {np.linalg.norm(x[:3]):7.2f}  "
                      f"RPF {np.linalg.norm(x[3:6]):7.2f} m/s  (solve #{n_solves}, nfev {sol.nfev})", flush=True)
                step = min(step * 1.3, 0.08)
            else:
                step *= 0.5
                print(f"  continuation lam={lam_try:.4f} MISSED (tight={tight} sane={sane}, "
                      f"nfev {sol.nfev}) -> step {step:.4f}", flush=True)
                if step < 5e-4:
                    raise SystemExit("continuation stalled — basin genuinely disconnected?")
        sol = solve(s0_new, x, pool=pool)
        pool.close()
        report(sol, s0_new, "s0")
    elif mode == "multiseed":
        # Min-dv MULTI-SEED pass from the full native s0: the continuation tracks only the family CONNECTED to the old bake
        # (landed at RPF ~371); the real mission's ~293-class family, if reachable from
        # this state, is a different basin. Seeds: the continued solution + REAL burn
        # vectors extracted from the OEM in ECI (ballistic-gap extraction at the real
        # epochs) + crossovers/scalings. Keep the cheapest converged in-corridor family.
        d = json.load(open(sys.argv[2]))
        s0 = np.asarray(d["s0"], float)
        t_s0 = float(d.get("t", T1))
        if t_s0 < T1 - 1.0:
            s0 = A._coast_rv(s0, t_s0, T1)

        from oem_loader import load_oem
        g, R, V = load_oem()
        def oem_state(t):
            return np.concatenate([[np.interp(t, g, R[:, k]) * 1e3 for k in range(3)],
                                   [np.interp(t, g, V[:, k]) * 1e3 for k in range(3)]])
        def extract_dv(t_burn, w=1800.0):
            """Real burn vector in ECI: coast the pre-burn OEM state across the burn
            window ballistically; dv = OEM post-window velocity minus the coast's."""
            pre = oem_state(t_burn - w)
            coast = A._coast_rv(pre, t_burn - w, t_burn + w)
            return oem_state(t_burn + w)[3:6] - coast[3:6]
        dv1_real = extract_dv(T1)
        dv2_real = extract_dv(T2)
        print(f"OEM-extracted real vectors (ECI): |DDP| {np.linalg.norm(dv1_real):.2f} m/s "
              f"(flown DRD 138.5), |RPF| {np.linalg.norm(dv2_real):.2f} m/s (flown 292.9)", flush=True)

        seeds = []
        try:
            cj = json.load(open("return_joint_s0.json"))
            seeds.append(("continued", np.array(cj["ddp_dv"] + cj["rpf_dv"])))
        except Exception:
            pass
        seeds += [
            ("real-real",       np.concatenate([dv1_real, dv2_real])),
            ("real-real*1.1",   np.concatenate([dv1_real, dv2_real * 1.1])),
            ("real-real*0.9",   np.concatenate([dv1_real, dv2_real * 0.9])),
            ("cont-ddp/real-rpf", None),   # filled below if continued present
            ("real-ddp/cont-rpf", None),
        ]
        if seeds and seeds[0][0] == "continued":
            xc = seeds[0][1]
            seeds[4] = ("cont-ddp/real-rpf", np.concatenate([xc[:3], dv2_real]))
            seeds[5] = ("real-ddp/cont-rpf", np.concatenate([dv1_real, xc[3:6]]))
        seeds = [(n, x) for n, x in seeds if x is not None]

        pool = mp.Pool(6)
        results = []
        for name, x0 in seeds:
            sol = solve(s0, x0, max_nfev=40, pool=pool)
            tight = (np.linalg.norm(sol.fun[:3]) < 0.05 and np.linalg.norm(sol.fun[3:]) < 0.05)
            tot = np.linalg.norm(sol.x[:3]) + np.linalg.norm(sol.x[3:6])
            print(f"  seed {name:20s} -> {'CONV' if tight else 'miss'}  "
                  f"DDP {np.linalg.norm(sol.x[:3]):7.2f}  RPF {np.linalg.norm(sol.x[3:6]):7.2f}  "
                  f"total {tot:7.2f} m/s  (nfev {sol.nfev})", flush=True)
            if tight and np.linalg.norm(sol.x[3:6]) < 600.0:
                results.append((tot, name, sol))
        pool.close()
        if not results:
            raise SystemExit("no seed converged in-corridor — keep the continued bake")
        results.sort(key=lambda r: r[0])
        tot, name, sol = results[0]
        print(f"\nWINNER: seed '{name}' total {tot:.2f} m/s")
        report(sol, s0, "multiseed")
    else:
        raise SystemExit("mode must be 'ref', 's0 <file.json>' or 'multiseed <file.json>'")
    print(f"(+{time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
