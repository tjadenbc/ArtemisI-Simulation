"""OD FILTER — pure DSN information-matrix covariance primitives.

Self-contained (numpy only; no artemis1 import -> no circular dependency). The artemis1 wiring
feeds these functions the nominal vehicle states + station geometry at sampled times over each
tracking arc; they accumulate the batch-at-epoch information matrix and return the emergent
covariance P = (sum H^T R^-1 H)^-1 in ECI.

Observables (state x = [r(m); v(m/s)] in ECI):
  - 2-way Doppler (range-rate): H = [(I - u u^T) v_rel / rho, u]   (velocity block = LOS unit u)
  - sequential range:           H = [u, 0]                          (along-LOS position)
  - Delta-DOR (plane-of-sky):   H = [e1^T/rho, 0; e2^T/rho, 0]      (the two cross-LOS axes)
u = LOS unit (station->vehicle), rho = range, v_rel = v_vehicle - v_station.
"""
import numpy as np


def station_eci(lat_deg, lon_deg, alt_m, t, R_EARTH, OMEGA_E, GMST0):
    """DSN station geodetic (spherical Earth, matching artemis1's latlon_alt_to_eci) -> ECI (r, v).
    theta = OMEGA_E * t + GMST0 (same rotating-frame convention as the integrator)."""
    lat = np.deg2rad(lat_deg); lon = np.deg2rad(lon_deg)
    r = R_EARTH + alt_m
    theta = OMEGA_E * t + GMST0
    lam = lon + theta
    r_s = np.array([r * np.cos(lat) * np.cos(lam),
                    r * np.cos(lat) * np.sin(lam),
                    r * np.sin(lat)])
    v_s = np.cross([0.0, 0.0, OMEGA_E], r_s)
    return r_s, v_s


def visible(r_v, r_s, mask_deg):
    """Vehicle elevation above the station local horizon >= mask?"""
    rn = np.linalg.norm(r_s)
    if rn < 1.0:
        return False
    up = r_s / rn
    los = r_v - r_s
    d = np.linalg.norm(los)
    if d < 1.0:
        return False
    elev = np.rad2deg(np.arcsin(np.clip(np.dot(los, up) / d, -1.0, 1.0)))
    return elev >= mask_deg


def _geom(r_v, v_v, r_s, v_s):
    rho_vec = r_v - r_s
    rho = np.linalg.norm(rho_vec)
    u = rho_vec / rho
    v_rel = v_v - v_s
    return rho_vec, rho, u, v_rel


def H_doppler(r_v, v_v, r_s, v_s):
    _, rho, u, v_rel = _geom(r_v, v_v, r_s, v_s)
    pos = (v_rel - np.dot(v_rel, u) * u) / rho      # (I - u u^T) v_rel / rho
    return np.concatenate([pos, u])                  # (6,)


def H_range(r_v, v_v, r_s, v_s):
    _, _, u, _ = _geom(r_v, v_v, r_s, v_s)
    return np.concatenate([u, np.zeros(3)])          # (6,)


def _transverse_basis(u):
    """Two orthonormal vectors spanning the plane transverse to the LOS unit u (pole-safe)."""
    zc = np.array([0.0, 0.0, 1.0])
    a = np.cross(u, zc)
    if np.linalg.norm(a) < 1e-6:
        a = np.cross(u, np.array([1.0, 0.0, 0.0]))
    e1 = a / np.linalg.norm(a)
    e2 = np.cross(u, e1)
    return e1, e2


def H_ddor(r_v, v_v, r_s, v_s):
    _, rho, u, _ = _geom(r_v, v_v, r_s, v_s)
    e1, e2 = _transverse_basis(u)
    return np.vstack([np.concatenate([e1 / rho, np.zeros(3)]),
                      np.concatenate([e2 / rho, np.zeros(3)])])   # (2,6)


def accumulate_covariance(samples, sigmas, prior_diag, Ls=1.0e3, Vs=1.0e-2):
    """Batch-at-epoch information accumulation -> covariance P (6x6, ECI).

    samples: list of dicts, one per sampled (time, geometry) over the arc:
        {"visible": [(r_s, v_s), ...],   # stations with the vehicle above the mask
         "r_v": r_v, "v_v": v_v,         # nominal vehicle state at that time
         "ddor": bool}                    # apply Delta-DOR at this sample (heavier near flybys)
    sigmas: dict {"doppler_ms":..., "range_m":..., "ddor_rad":...} (1-sigma).
    prior_diag: length-6 a-priori 1-sigma [pos(m)x3, vel(m/s)x3] -> keeps Lambda SPD on poor arcs.
    Ls, Vs: non-dimensionalization scales (position m, velocity m/s) for a well-conditioned inverse.
    """
    Lam = np.zeros((6, 6))
    Lam += np.diag(1.0 / np.asarray(prior_diag, float) ** 2)
    sd2 = sigmas["doppler_ms"] ** 2
    sr2 = sigmas["range_m"] ** 2
    sa2 = sigmas["ddor_rad"] ** 2
    for smp in samples:
        r_v = smp["r_v"]; v_v = smp["v_v"]
        for (r_s, v_s) in smp["visible"]:
            Hd = H_doppler(r_v, v_v, r_s, v_s); Lam += np.outer(Hd, Hd) / sd2
            Hr = H_range(r_v, v_v, r_s, v_s);   Lam += np.outer(Hr, Hr) / sr2
        if smp.get("ddor") and smp["visible"]:
            r_s, v_s = smp["visible"][0]
            Hdd = H_ddor(r_v, v_v, r_s, v_s)
            Lam += Hdd.T @ Hdd / sa2
    # non-dimensionalize, invert, redimensionalize (S diagonal, symmetric)
    S = np.diag([Ls, Ls, Ls, Vs, Vs, Vs])
    Lam_nd = S @ Lam @ S
    P_nd = np.linalg.inv(Lam_nd)
    P = S @ P_nd @ S
    return 0.5 * (P + P.T)


def _grav_gradient(gravity_fn, r, t, h=1.0):
    """G = ∂g/∂r (3x3) by central finite difference of the true gravity model."""
    G = np.zeros((3, 3))
    for k in range(3):
        e = np.zeros(3); e[k] = h
        G[:, k] = (gravity_fn(r + e, t) - gravity_fn(r - e, t)) / (2.0 * h)
    return G


def accumulate_covariance_stm(gravity_fn, t_epoch, s6_epoch, arc_len_s, n_samples, station_fn,
                              sigmas, use_ddor, prior_diag, floor_pos_m=0.0, floor_vel_ms=0.0,
                              q_pos_m=0.0, q_vel_ms=0.0, Ls=1.0e3, Vs=1.0e-2):
    """STM-based LinCov covariance at t_epoch. Propagate the state + state-transition matrix Φ(t,t_epoch)
    BACKWARD over the arc [t_epoch-arc, t_epoch]; a measurement at t_i informs the EPOCH state through
    the dynamics: info += (H_i Φ_i)^T R^-1 (H_i Φ_i). This is what pins VELOCITY (range/Doppler over an
    arc + the dynamics coupling), which the Φ≈I batch cannot. Adds process noise (q_*, grows the older
    measurements' effective error) and a systematic-error FLOOR (delivered OD is systematic-limited).
    Returns P (6x6, ECI). gravity_fn(r, t) -> accel; station_fn(t) -> list of (r_s, v_s)."""
    from scipy.integrate import solve_ivp

    def rhs(t, y):
        r = y[:3]; v = y[3:6]; Phi = y[6:].reshape(6, 6)
        g = gravity_fn(r, t)
        G = _grav_gradient(gravity_fn, r, t)
        A = np.zeros((6, 6)); A[:3, 3:] = np.eye(3); A[3:, :3] = G
        dPhi = A @ Phi
        return np.concatenate([v, g, dPhi.ravel()])

    y0 = np.concatenate([np.asarray(s6_epoch, float), np.eye(6).ravel()])
    t_eval = np.linspace(t_epoch, t_epoch - arc_len_s, n_samples)   # backward from the epoch
    sol = solve_ivp(rhs, (t_epoch, t_epoch - arc_len_s), y0, method="RK45",
                    rtol=1e-8, atol=1e-3, t_eval=t_eval, max_step=arc_len_s / 8.0)
    Lam = np.diag(1.0 / np.asarray(prior_diag, float) ** 2)
    sd2 = sigmas["doppler_ms"] ** 2; sr2 = sigmas["range_m"] ** 2; sa2 = sigmas["ddor_rad"] ** 2
    for i in range(sol.y.shape[1]):
        r_v = sol.y[:3, i]; v_v = sol.y[3:6, i]; Phi = sol.y[6:, i].reshape(6, 6)
        dt = abs(t_epoch - sol.t[i])
        # process-noise inflation of the measurement error at age dt (older -> less informative)
        qp = (q_pos_m * dt / 86400.0) ** 2; qv = (q_vel_ms * dt / 86400.0) ** 2
        vis = station_fn(sol.t[i], r_v)
        for (r_s, v_s) in vis:
            for (Hrow, base) in ((H_doppler(r_v, v_v, r_s, v_s), sd2 + qv),
                                 (H_range(r_v, v_v, r_s, v_s), sr2 + qp)):
                HP = Hrow @ Phi
                Lam += np.outer(HP, HP) / base
        if use_ddor and vis:
            r_s, v_s = vis[0]
            HP = H_ddor(r_v, v_v, r_s, v_s) @ Phi
            Lam += HP.T @ HP / (sa2 + qp / max(1.0, np.linalg.norm(r_v - r_s)) ** 2)
    S = np.diag([Ls, Ls, Ls, Vs, Vs, Vs])
    P = S @ np.linalg.inv(S @ Lam @ S) @ S
    P = 0.5 * (P + P.T)
    if floor_pos_m > 0.0 or floor_vel_ms > 0.0:               # systematic-error floor (add covariance)
        P = P + np.diag([floor_pos_m ** 2] * 3 + [floor_vel_ms ** 2] * 3)
    return P


def rotate_eci_to_ric(P_eci, R, I, C):
    """Rotate a 6x6 ECI covariance into the RIC frame given the RIC basis vectors (rows of M)."""
    M = np.array([R, I, C])            # 3x3, rows = RIC axes in ECI
    B = np.zeros((6, 6)); B[:3, :3] = M; B[3:, 3:] = M
    return B @ P_eci @ B.T


def chol_lower(P):
    """Lower-triangular Cholesky with a tiny SPD nudge for safety; err = L @ unit_normal."""
    P = 0.5 * (P + P.T)
    try:
        return np.linalg.cholesky(P)
    except np.linalg.LinAlgError:
        w = np.linalg.eigvalsh(P)
        P = P + (abs(min(w)) + 1e-12) * np.eye(6)
        return np.linalg.cholesky(P)


# --------------------------------------------------------------------------------------------
def _fd_test():
    """Finite-difference check of the analytic partials against numeric derivatives."""
    rng = np.random.default_rng(0)
    R_EARTH = 6378137.0; OMEGA_E = 7.2921159e-5; GMST0 = 1.234
    r_s, v_s = station_eci(35.4, -116.9, 1000.0, 3600.0, R_EARTH, OMEGA_E, GMST0)
    r_v = np.array([-3.0e8, 1.5e8, 8.0e7]); v_v = np.array([-500.0, 900.0, 300.0])

    def rho_dot(r, v):
        rv = r - r_s; rho = np.linalg.norm(rv); u = rv / rho
        return np.dot(v - v_s, u)

    def rng_f(r, v):
        return np.linalg.norm(r - r_s)

    def angle(r, v, e):
        rv = r - r_s; rho = np.linalg.norm(rv); return np.dot(rv, e) / rho  # small-angle proxy

    x = np.concatenate([r_v, v_v]); eps = np.array([1.0] * 3 + [1e-3] * 3)
    # Doppler
    Ha = H_doppler(r_v, v_v, r_s, v_s); Hn = np.zeros(6)
    for k in range(6):
        dx = np.zeros(6); dx[k] = eps[k]
        Hn[k] = (rho_dot(*np.split(x + dx, [3])[:2] if False else (x + dx)[:3], (x + dx)[3:]) if False else
                 (rho_dot((x + dx)[:3], (x + dx)[3:]) - rho_dot((x - dx)[:3], (x - dx)[3:])) / (2 * eps[k]))
    dop_err = np.abs(Ha - Hn).max()
    # Range
    Ha_r = H_range(r_v, v_v, r_s, v_s); Hn_r = np.zeros(6)
    for k in range(6):
        dx = np.zeros(6); dx[k] = eps[k]
        Hn_r[k] = (rng_f((x + dx)[:3], (x + dx)[3:]) - rng_f((x - dx)[:3], (x - dx)[3:])) / (2 * eps[k])
    rng_err = np.abs(Ha_r - Hn_r).max()
    # DDOR (transverse position rows)
    _, rho0, u0, _ = _geom(r_v, v_v, r_s, v_s); e1, e2 = _transverse_basis(u0)
    Ha_dd = H_ddor(r_v, v_v, r_s, v_s); Hn_dd = np.zeros((2, 6))
    for j, e in enumerate((e1, e2)):
        for k in range(6):
            dx = np.zeros(6); dx[k] = eps[k]
            Hn_dd[j, k] = (angle((x + dx)[:3], None, e) - angle((x - dx)[:3], None, e)) / (2 * eps[k])
    ddor_err = np.abs(Ha_dd - Hn_dd).max()
    print(f"FD partial errors: doppler {dop_err:.2e}  range {rng_err:.2e}  ddor {ddor_err:.2e}")
    # accumulate a small covariance to confirm SPD + anisotropy
    def stn(t): return [station_eci(35.4, -116.9, 1000.0, t, R_EARTH, OMEGA_E, GMST0)]
    samples = [{"r_v": r_v, "v_v": v_v, "visible": stn(t), "ddor": (i % 5 == 0)}
               for i, t in enumerate(np.linspace(0, 25200, 30))]
    P = accumulate_covariance(samples, {"doppler_ms": 1e-4, "range_m": 5.0, "ddor_rad": 5e-9},
                              [1e5, 1e5, 1e5, 1.0, 1.0, 1.0])
    pos_sig = np.sqrt(np.diag(P)[:3]); vel_sig = np.sqrt(np.diag(P)[3:])
    print(f"P pos sigma (m): {pos_sig}  vel sigma (m/s): {vel_sig}")
    print(f"SPD: {np.all(np.linalg.eigvalsh(P) > 0)}  cond: {np.linalg.cond(P):.2e}")
    L = chol_lower(P); print(f"chol OK, L@Lt residual {np.abs(L@L.T - P).max():.2e}")
    assert dop_err < 1e-6 and rng_err < 1e-6 and ddor_err < 1e-6, "FD partials mismatch"
    print("FD test PASSED")


if __name__ == "__main__":
    _fd_test()
