"""Minimal loader for the as-flown Orion OEM ephemeris, in the simulation's frame.

The OEM (data/artemis1_ephemeris/) is EME2000/J2000; the simulation's ECI frame is the
mean equator/equinox OF DATE. load_oem() parses the ephemeris and (by default) rotates
the states through the IAU-1976 precession at the launch epoch (constant over the 25-day
mission to <1 km) so they can be compared to — or used to anchor — simulation states
directly. Used by the offline solve tools (phase2b_bake_finite.py, dro_native_solve.py,
return_joint_solve.py); run from the repository root so the data path resolves.

Returns: (get_s, R_km, V_kms) — GET seconds from liftoff, positions in km, velocities
in km/s, as numpy arrays.
"""
from datetime import datetime
import numpy as np
import artemis1 as A

LAUNCH = datetime(2022, 11, 16, 6, 47, 44)
OEM_F = "data/artemis1_ephemeris/Post_TLI_Orion_AsFlown_20221213_EPH_OEM.asc"


def _precession_p(jd):
    """IAU-1976 precession matrix P with r_meanOfDate = P @ r_J2000 (mean equator/equinox)."""
    T = (jd - 2451545.0) / 36525.0
    asr = np.deg2rad(1.0 / 3600.0)
    zeta = (2306.2181 * T + 0.30188 * T**2 + 0.017998 * T**3) * asr
    z = (2306.2181 * T + 1.09468 * T**2 + 0.018203 * T**3) * asr
    theta = (2004.3109 * T - 0.42665 * T**2 - 0.041833 * T**3) * asr

    def R3(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])

    def R2(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])

    return R3(-z) @ R2(theta) @ R3(-zeta)


def load_oem(precess=True):
    g, R, V = [], [], []
    for ln in open(OEM_F):
        p = ln.split()
        if len(p) == 7 and p[0][:2] == "20" and "T" in p[0]:
            try:
                ts = p[0][:26]
                dt = datetime.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S")
                frac = float("0" + ts[19:]) if "." in ts[19:] else 0.0
                get = (dt - LAUNCH).total_seconds() + frac
                g.append(get)
                R.append([float(x) for x in p[1:4]])
                V.append([float(x) for x in p[4:7]])
            except Exception:
                pass
    g, R, V = np.array(g), np.array(R), np.array(V)     # GET s, km, km/s
    if precess:
        P = _precession_p(A.JD_LAUNCH)
        R = R @ P.T
        V = V @ P.T
    return g, R, V
