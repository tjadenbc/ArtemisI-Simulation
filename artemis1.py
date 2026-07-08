"""
Artemis I Mission Simulator — Real Physics
========================================================================

Physics-integrated Monte Carlo of the **Artemis I** mission (16 Nov 2022 –
11 Dec 2022): the first integrated flight of the Space Launch System (SLS)
and the Orion spacecraft, UNCREWED, on a ~25.5-day flight to a Distant
Retrograde Orbit (DRO) around the Moon and back to a Pacific splashdown.

The whole mission is flown as one continuous, numerically-integrated
trajectory — every powered manoeuvre a finite-thrust integration, every coast
a three-body integration under Earth (+ J2–J6 near-field) and Moon gravity —
using a real November-2022 lunar ephemeris and a degree-8 GRAIL gravity field.
A constants block plus documented feature flags configure the model;
`run_mission()` flies one trial; `main()` / `main_parallel()` are the
deterministic, resumable, shard-able Monte Carlo drivers (per-trial
checkpointing, debug, and phase timeline).

MISSION PROFILE
---------------
  * Launch vehicle: SLS Block 1 — 2× five-segment SRBs + a core stage with
    4× RS-25 engines + the ICPS (Interim Cryogenic Propulsion Stage, 1× RL10)
    — replaces the Saturn V's S-IC / S-II / S-IVB.
  * Spacecraft: Orion = Crew Module (CM) + European Service Module (ESM).
    The ESM main engine (an AJ10-derived Orbital Maneuvering System Engine,
    "OMS-E") does every in-space burn — replaces the Apollo CSM's SPS. There
    is NO Lunar Module, NO descent/ascent, NO rendezvous/docking.
  * Trajectory: a Distant Retrograde Orbit (~70,000 km from the Moon,
    retrograde) reached via an Outbound Powered Flyby (OPF) lunar gravity
    assist, NOT a low lunar orbit + landing. Return is via a DRO-departure
    burn + a Return Powered Flyby (RPF). Two lunar close approaches (~130 km),
    not one capture.
  * Uncrewed: there is NO crew-survival model. MISSION success = SLS delivers
    Orion AND Orion returns and splashes down intact. (Artemis I carried the
    "Moonikin" Campos and the Helga/Zohar dosimetry phantoms, but radiation dose
    is out of scope.)
  * Entry: Orion flies a true SKIP entry (skip guidance) at ~11 km/s to a
    Pacific splashdown off Baja California.
  * Duration: ~25.5 days.

State vector convention: y = [x,y,z, vx,vy,vz, m]
in ECI, SI units (m, m/s, kg).

Run with `python3 -c "import artemis1; artemis1.main_parallel(...)"` — NEVER a
stdin heredoc (macOS `spawn` re-imports __main__).
"""
from __future__ import annotations
import os, time, json
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import od_filter as _odf   # rung-a emergent DSN information-matrix covariance primitives
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# Physical constants  (Earth/Moon)
# ============================================================
G0       = 9.80665
MU_EARTH = 3.986004418e14
R_EARTH  = 6_378_137.0
J2       = 1.0826267e-3
# Higher Earth zonal harmonics J3..J6 (unnormalized, EGM/WGS-84 class) — a force-model COMPLETENESS
# term (ENABLE_EARTH_HIGHER_ZONALS). Effect is negligible for this 25-day lunar trajectory (J3 ~ 1.5e-3
# of J2, and (Re/r)^n makes them irrelevant past the brief near-Earth phases), but they make the Earth
# field faithful beyond J2. Applied only within EARTH_ZONAL_R_MAX_M (they fall as ~1/r^(n+1)).
EARTH_ZONALS       = {3: -2.5327e-6, 4: -1.6196e-6, 5: -2.2730e-7, 6: 5.4069e-7}
EARTH_ZONAL_R_MAX_M = 3.0e7    # J3-J6 negligible beyond ~5 Re; skip the FD-gradient call past this
OMEGA_E  = 7.292115e-5

MU_MOON  = 4.9048695e12
R_MOON   = 1_737_400.0
EM_DIST  = 384_399e3
G_MOON   = MU_MOON / R_MOON**2

# Sun — needed for the DRO leg (see ENABLE_SOLAR_GRAVITY). Apollo's 8-day flight
# stayed deep in the Earth-Moon well where solar third-body force is tiny; an
# Artemis DRO reaches ~432,000 km from Earth for ~25 days, where the Sun's tidal
# pull is ~1-2% of Earth's gravity there — a first-order effect on DRO dynamics.
MU_SUN   = 1.32712440018e20        # m^3/s^2
AU       = 1.495978707e11          # m
# Solar radiation pressure (SRP) — first-order on Orion over the multi-day cislunar coast.
SRP_P0_NM2  = 4.56e-6              # solar radiation pressure at 1 AU (N/m^2)   [sourced]
SRP_CR      = 1.3                  # Orion reflectivity coefficient            ESTIMATE
SRP_AREA_M2 = 40.0                 # effective sun-facing area (CM+ESM+arrays) ESTIMATE
SRP_MASS_KG = 26000.0             # representative Orion mass (fixed A/m)       ESTIMATE

# Idealized circular-Moon parameters — used ONLY when ENABLE_REAL_EPHEMERIS is
# False (legacy/idealized model). Artemis runs with the real ephemeris (default
# ON); the 28.4 deg value is the 1969 figure and is irrelevant in that config.
MOON_INC = np.deg2rad(28.4)
OMEGA_M  = np.sqrt(MU_EARTH / EM_DIST**3)

# ---- Real Nov-2022 lunar ephemeris + launch epoch (flag-gated) --------------
# Artemis I lifted off 2022-11-16 06:47:44 UTC from LC-39B, KSC. When
# ENABLE_REAL_EPHEMERIS is True the Moon position and Earth-rotation (GMST) are
# anchored to this epoch (moon_state/_gmst_rad below, ported from the shared lineage with
# this JD).
ENABLE_REAL_EPHEMERIS = True
JD_LAUNCH = 2459899.78315   # 2022-11-16 06:47:44 UTC (Artemis I liftoff)
# JD check: 2022-11-16 12:00 UTC = JD 2459900.0; liftoff is 5h12m16s earlier
# (-0.21685 d) -> 2459899.78315. Verified against the calendar->JD algorithm.
# ΔT / TT fix: the Meeus lunar/solar series take TDT/TT as their time argument, but JD_LAUNCH+t
# was being fed as UTC — displacing the Moon ~v_moon×ΔT ≈ 71 km ALONG-TRACK. TT−UTC at this epoch is
# EXACT (not an estimate): 32.184 s (TT−TAI) + 37 leap seconds (TAI−UTC, constant since 2017) = 69.184 s.
# Only the EPHEMERIS argument shifts (_jde_tt below); GMST stays on UTC(≈UT1) — Earth rotation is a
# UT phenomenon. OFF = the legacy UTC-fed ephemeris, bit-identical. NOTE: baked ephemeris-derived
# constants (PHASEC_*, _NEAR_POLAR_V_NP OEM-backprop seeds) predate this fix — re-solve when activating
# those paths (the SRP lesson: a shifted perilune throws the hardcoded velocity-match seed).
ENABLE_DELTA_T = True
DELTA_T_S      = 69.184     # TT − UTC, Nov-2022 [exact]


def _jde_tt(t):
    """Ephemeris (TT) Julian date at mission-elapsed time t [s]: the Meeus series argument."""
    return JD_LAUNCH + (t + (DELTA_T_S if ENABLE_DELTA_T else 0.0)) / 86400.0

# --- Moon ephemeris: truncated Meeus series (Astronomical Algorithms ch.47) --
# PORTED VERBATIM from the shared lineage (validated to <1" vs Meeus's worked
# example). The series is epoch-agnostic — only JD_LAUNCH changed for Artemis.
# NOTE (Artemis fidelity): the 56/30-term truncation was tuned for Apollo's
# 8-day flight. A 25.5-day DRO mission is more sensitive to lunar-position error
# accumulated over the longer arc; if validation shows drift, add more terms or
# swap in a JPL DE ephemeris (de440) — flagged as a future improvement.
# Principal periodic terms: (D, M, Mp, F, lon_1e-6deg, dist_1e-3km)
_MOON_LR = [
    (0,0,1,0,6288774,-20905355),(2,0,-1,0,1274027,-3699111),(2,0,0,0,658314,-2955968),
    (0,0,2,0,213618,-569925),(0,1,0,0,-185116,48888),(0,0,0,2,-114332,-3149),
    (2,0,-2,0,58793,246158),(2,-1,-1,0,57066,-152138),(2,0,1,0,53322,-170733),
    (2,-1,0,0,45758,-204586),(0,1,-1,0,-40923,-129620),(1,0,0,0,-34720,108743),
    (0,1,1,0,-30383,104755),(2,0,0,-2,15327,10321),(0,0,1,2,-12528,0),
    (0,0,1,-2,10980,79661),(4,0,-1,0,10675,-34782),(0,0,3,0,10034,-23210),
    (4,0,-2,0,8548,-21636),(2,1,-1,0,-7888,24208),(2,1,0,0,-6766,30824),
    (1,0,-1,0,-5163,-8379),(1,1,0,0,4987,-16675),(2,-1,1,0,4036,-12831),
    (2,0,2,0,3994,-10445),(4,0,0,0,3861,-11650),(2,0,-3,0,3665,14403),
    (0,1,-2,0,-2689,-7003),(2,0,-1,2,-2602,0),(2,-1,-2,0,2390,10056),
    (1,0,1,0,-2348,6322),(2,-2,0,0,2236,-9884),(0,1,2,0,-2120,5751),
    (0,2,0,0,-2069,0),(2,-2,-1,0,2048,-4950),(2,0,1,-2,-1773,4130),
    (2,0,0,2,-1595,0),(4,-1,-1,0,1215,-3958),(0,0,2,2,-1110,0),
    (3,0,-1,0,-892,3258),(2,1,1,0,-810,2616),(4,-1,-2,0,759,-1897),
    (0,2,-1,0,-713,-2117),(2,2,-1,0,-700,2354),(2,1,-2,0,691,0),
    (2,-1,0,-2,596,0),(4,0,1,0,549,-1423),(0,0,4,0,537,-1117),
    (4,-1,0,0,520,-1571),(1,0,-2,0,-487,-1739),(2,1,0,-2,-399,0),
    (0,0,2,-2,-381,-4421),
]
_MOON_B = [
    (0,0,0,1,5128122),(0,0,1,1,280602),(0,0,1,-1,277693),(2,0,0,-1,173237),
    (2,0,-1,1,55413),(2,0,-1,-1,46271),(2,0,0,1,32573),(0,0,2,1,17198),
    (2,0,1,-1,9266),(0,0,2,-1,8822),(2,-1,0,-1,8216),(2,0,-2,-1,4324),
    (2,0,1,1,4200),(2,1,0,-1,-3359),(2,-1,-1,1,2463),(2,-1,0,1,2211),
    (2,-1,-1,-1,2065),(0,1,-1,-1,-1870),(4,0,-1,-1,1828),(0,1,0,1,-1794),
    (0,0,0,3,-1749),(0,1,-1,1,-1565),(1,0,0,1,-1491),(0,1,1,1,-1475),
    (0,1,1,-1,-1410),(0,1,0,-1,-1344),(1,0,0,-1,-1335),(0,0,3,1,1107),
    (4,0,0,-1,1021),(4,0,-1,1,833),
]


def _moon_eci_m(jde):
    """Geocentric Moon position in ECI (equatorial, mean equinox), METERS."""
    T = (jde - 2451545.0) / 36525.0
    d2r = np.deg2rad
    Lp = d2r(218.3164477 + 481267.88123421*T - 0.0015786*T**2 + T**3/538841 - T**4/65194000)
    D  = d2r(297.8501921 + 445267.1114034*T - 0.0018819*T**2 + T**3/545868 - T**4/113065000)
    M  = d2r(357.5291092 + 35999.0502909*T - 0.0001536*T**2 + T**3/24490000)
    Mp = d2r(134.9633964 + 477198.8675055*T + 0.0087414*T**2 + T**3/69699 - T**4/14712000)
    F  = d2r(93.2720950 + 483202.0175233*T - 0.0036539*T**2 - T**3/3526000 + T**4/863310000)
    A1 = d2r(119.75 + 131.849*T); A2 = d2r(53.09 + 479264.290*T); A3 = d2r(313.45 + 481266.484*T)
    E = 1 - 0.002516*T - 0.0000074*T**2
    sl = sr = sb = 0.0
    for d,m,mp,f,cl,cr in _MOON_LR:
        arg = d*D + m*M + mp*Mp + f*F; e = E**abs(m)
        sl += cl*e*np.sin(arg); sr += cr*e*np.cos(arg)
    for d,m,mp,f,cb in _MOON_B:
        sb += cb*(E**abs(m))*np.sin(d*D + m*M + mp*Mp + f*F)
    sl += 3958*np.sin(A1) + 1962*np.sin(Lp - F) + 318*np.sin(A2)
    sb += (-2235*np.sin(Lp) + 382*np.sin(A3) + 175*np.sin(A1-F) + 175*np.sin(A1+F)
           + 127*np.sin(Lp-Mp) - 115*np.sin(Lp+Mp))
    lon = np.deg2rad(np.rad2deg(Lp) + sl/1e6)
    lat = np.deg2rad(sb/1e6)
    dist = (385000.56 + sr/1000.0) * 1000.0   # meters
    eps = np.deg2rad(23.439291 - 0.0130042*T)
    xe = dist*np.cos(lat)*np.cos(lon); ye = dist*np.cos(lat)*np.sin(lon); ze = dist*np.sin(lat)
    return np.array([xe, ye*np.cos(eps)-ze*np.sin(eps), ye*np.sin(eps)+ze*np.cos(eps)])


def _gmst_rad(jd):
    """Greenwich Mean Sidereal Time (radians) at Julian date jd."""
    T = (jd - 2451545.0) / 36525.0
    g = 280.46061837 + 360.98564736629*(jd - 2451545.0) + 0.000387933*T*T - T*T*T/38710000.0
    return np.deg2rad(g % 360.0)

# Greenwich sidereal angle at launch — anchors ECI(vernal-equinox) -> Earth-fixed.
# Zero in legacy mode so theta = OMEGA_E*t is reproduced exactly.
_GMST0 = _gmst_rad(JD_LAUNCH) if ENABLE_REAL_EPHEMERIS else 0.0


def _sun_eci_m(jde):
    """Geocentric Sun position in ECI (equatorial, mean equinox), METERS.
    Low-precision solar coordinates (Meeus ch.25, ~0.01 deg) — ample for the
    Sun's third-body perturbation on the cislunar trajectory. NEW for Artemis
    (the shared lineage had no solar term)."""
    T = (jde - 2451545.0) / 36525.0
    d2r = np.deg2rad
    L0 = 280.46646 + 36000.76983*T + 0.0003032*T*T          # mean longitude, deg
    M  = d2r(357.52911 + 35999.05029*T - 0.0001537*T*T)     # mean anomaly
    C  = ((1.914602 - 0.004817*T - 0.000014*T*T)*np.sin(M)
          + (0.019993 - 0.000101*T)*np.sin(2*M)
          + 0.000289*np.sin(3*M))                            # equation of center, deg
    true_long = d2r(L0 + C)
    v = M + d2r(C)                                            # true anomaly
    e = 0.016708634 - 0.000042037*T - 0.0000001267*T*T
    R = 1.000001018 * (1 - e*e) / (1 + e*np.cos(v)) * AU     # Earth-Sun distance, m
    eps = d2r(23.439291 - 0.0130042*T)
    # Sun ecliptic latitude ~ 0; rotate ecliptic -> equatorial about the x-axis.
    x = R*np.cos(true_long)
    y = R*np.cos(eps)*np.sin(true_long)
    z = R*np.sin(eps)*np.sin(true_long)
    return np.array([x, y, z])


def sun_state(t):
    """Sun ECI position [m] at mission time t (s). Velocity not needed for the
    third-body gravity term."""
    return _sun_eci_m(_jde_tt(t))


# ============================================================
# Vehicle constants — SLS Block 1 + Orion (CM + ESM)
# ============================================================
# Figures below are SOURCED (NASA / AAS references) for the citation behind
# each (NASA SLS/RS-25 references, Wikipedia SLS/ESM/Orion, ESA, Everyday
# Astronaut). A few remain ESTIMATE-grade and are tagged inline.

# --- Solid Rocket Boosters (2× five-segment, Shuttle-derived) ---------------
SRB_COUNT          = 2
SRB_THRUST_SL_N    = 14.6e6               # sea level, EACH (vac 16 MN)  [sourced]
SRB_BURN_TIME_S    = 126.0                # propellant web burnout       [sourced]
SRB_SEP_TIME_S     = 132.0                # SRB separation (000/00:02:12); ~6 s tailoff/sep
                                          # after web burnout — boost integrates to here, SRBs
                                          # produce thrust only until SRB_BURN_TIME_S.   [sourced]
SRB_PROP_EACH_KG   = 631_000.0            # gross 730,000 - dry ~99,000  [derived]
SRB_DRY_EACH_KG    = 99_000.0             # gross 730,000 - prop         [derived]

# --- Core stage (4× RS-25, LH2/LOX) -----------------------------------------
RS25_COUNT         = 4
RS25_THRUST_SL_N   = 1.85e6               # per engine, sea level (7.4 MN total) [sourced]
RS25_THRUST_VAC_N  = 2.28e6               # per engine, vacuum (9.1 MN total)    [sourced]
RS25_ISP_SL_S      = 366.0                # sea level                    [sourced]
RS25_ISP_VAC_S     = 452.0                # vacuum                       [sourced]
CORE_BURN_TIME_S   = 480.0                # liftoff -> MECO (~000/00:08:03) [sourced]
CORE_PROP_KG       = 984_000.0            # LOX 840,000 + LH2 144,000    [sourced]
CORE_DRY_KG        = 97_940.0             # empty mass                   [sourced]
# RS-25 throttle profile (SLS runs 109% RPL = the *_VAC_N/_SL_N thrust above). Two sourced
# throttle behaviors flown on Artemis I (NASA SLS Mission Planner's Guide; Boeing/NSF ascent
# writeups): (1) a "throttle bucket" reducing RS-25 thrust through max-Q to limit aero loads,
# and (2) an axial-acceleration limiter throttling down so steady-state axial g stays <=5.
RS25_ACCEL_LIMIT_G   = 5.0                # axial accel cap; throttle to hold      [sourced]
RS25_MAXQ_BUCKET_PCT = 0.62               # RS-25 level through max-Q (frac of 109% = ~68% RPL, the engine's
                                          # ~67% min-throttle floor). CALIBRATED so core MECO duration matches
                                          # the sourced 000/00:08:03 (483 s); also drops max-Q 37->32 kPa.
RS25_MAXQ_T0_S       = 38.0               # bucket start (s after liftoff)         ~max-Q region
RS25_MAXQ_T1_S       = 90.0               # bucket end (throttle back up)          ~max-Q region

# --- ICPS (Interim Cryogenic Propulsion Stage, 1× RL10B-2) ------------------
ICPS_THRUST_N      = 110_100.0            # RL10B-2 vacuum               [sourced]
ICPS_ISP_S         = 465.5                # vacuum                       [sourced]
ICPS_PROP_KG       = 28_576.0             # gross 32,066 - dry 3,490; usable slightly less [derived]
ICPS_DRY_KG        = 3_490.0              # empty mass                   [sourced]
# ICPS does a Perigee Raise Maneuver (PRM, GET ~00:52:56) then TLI (~01:29:27).
ICPS_TLI_DV_MS     = 3050.0               # approx TLI delta-V           ESTIMATE

# --- Orion: Crew Module (CM) + European Service Module (ESM) ----------------
ORION_CM_KG        = 10_400.0             # CM launch mass               [sourced]
ESM_DRY_KG         = 6_185.0              # ESM dry                      [sourced]
ESM_PROP_KG        = 8_600.0             # ESM usable propellant (MMH/MON-3) [sourced]
ESM_WET_KG         = 15_461.0             # ESM launch (wet) mass        [sourced]
ORION_TOTAL_KG     = ORION_CM_KG + ESM_WET_KG   # ~25,861 kg (excl. LAS, stage adapter)

# ESM main engine = OMS-E (AJ10-190 derived, ex-Space-Shuttle OMS pod).
OMSE_THRUST_N      = 26_600.0             # ~26.6 kN                     [sourced]
OMSE_ISP_S         = 316.0                # MMH/MON-3                    [sourced]
# 8 auxiliary engines (R-4D, 490 N each) + 24 RCS thrusters (220 N each, 6 pods
# of 4) — small trajectory-correction / attitude burns; model later if needed.
ESM_AUX_THRUST_N   = 490.0                # each, ×8                     [sourced]
ESM_RCS_THRUST_N   = 220.0                # each, ×24                    [sourced]

# --- Launch site & ascent (LC-39B, Kennedy Space Center) --------------------
LAUNCH_LAT_DEG     = 28.627               # LC-39B                       [sourced ~]
LAUNCH_LON_DEG     = -80.621
# Launch azimuth: this 90° due-east constant is the ENABLE_LAUNCH_CONTINUITY=OFF
# FALLBACK only. With continuity ON (default), _solve_launch_azimuth() solves the
# Moon-plane azimuth at runtime (78.20° for the ~28.6° inclination ascent).
# VALIDATED vs as-flown Artemis I: there is NO single published SLS
# launch azimuth — it is onboard-computed and launch-second-dependent (AAS 20-591;
# the day's azimuth window spans 2-40°, reference ≈ mid-window). The proper anchors
# are the SLS INSERTION TARGET + direction, which the solve reproduces: 78.20° ->
# insertion 29.6×1806 km (the 1806 km / 975 nmi apogee matches the sourced SLS target
# EXACTLY), inclination 30.35° (inside the documented EM-1 28.5-32° band), shifted
# north of due-east (azimuth < 90°) as AAS 20-591 specifies. Reproducing the
# documented insertion geometry from an INDEPENDENT Moon-plane solve IS the validation.
LAUNCH_AZIMUTH_DEG = 90.0
SLS_LIFTOFF_MASS   = 2_610_000.0          # total at liftoff             [sourced]
LAS_MASS_KG        = 6_300.0              # Launch Abort System jettison ESTIMATE
# Ascent aerodynamics. ENABLE_USSA76_ATM (default ON) supplies a USSA-76 layered
# atmosphere (<32 km) + a Mach-dependent ascent Cd (_cd_mach); the constants below
# are the legacy OFF-fallback (single-exponential atmosphere + constant Cd).
SLS_CD             = 0.5                  # OFF-fallback constant ascent Cd; subsonic ref (ON: _cd_mach)
SLS_AREA           = 90.0                # m^2 aero reference area: 8.4 m core (55 m^2) + SRB/interstage frontal
MAXQ_STRUCTURAL_PA = 50_000.0            # ascent max-q structural limit (~1.56x the validated ~32 kPa nominal)
ENABLE_USSA76_ATM  = True                # USSA-76 layered density (<32 km) + Mach-dependent ascent Cd
# SRBs are modeled as a fixed thrust burned to depletion over SRB_BURN_TIME_S
# (mdot = prop/burn); the sourced thrust(14.6MN)/prop(631t)/burn(126s) are
# jointly over-determined for a constant-thrust solid, so Isp is left implicit.
# MECO target (SOURCED — NASA / AAS references): the SLS Core inserts at a LOW
# 30×1806 km perigee for every launch azimuth, then the ICPS flies a discrete
# Perigee-Raise Maneuver (PRM, at apogee) up to 185 km before TLI. Both ICPS burns
# are modeled explicitly (phase_icps_tli). The exo-atmospheric MECO is flown by
# closed-loop IGM-style linear-tangent guidance (phase_sls_launch, ENABLE_IGM_ASCENT)
# solved to this insertion; the open-loop pitch constants below now govern ONLY the
# in-atmosphere boost (liftoff → SRB sep) gravity turn that hands the IGM a sane
# state (and the full ascent when IGM is OFF / falls back).
INSERTION_APOGEE_KM  = 1806.0             # apogee at Core Sep (975 nmi)    [sourced]
INSERTION_PERIGEE_KM = 30.0               # perigee at Core Sep (16 nmi)    [sourced]
PRM_PERIGEE_KM       = 185.0              # PRM raises perigee to (100 nmi) [sourced]
ASCENT_VERT_S        = 10.0               # vertical hold before pitch-over
ASCENT_PITCH_END_DEG = 82.0               # boost-phase gravity-turn pitch (from vertical)
ASCENT_PITCH_T       = 155.0              # pitch-ramp completion (s). Calibrated (jointly with
                                          # the RS-25 max-Q bucket below) so the boost hands off FLAT at
                                          # SRB sep and core MECO matches real Artemis I on ALL metrics:
                                          # alt 156.7 km, FPA +3.61°, 8207 m/s, 30x1806, dur 482.8 s
                                          # (real 157/+3.62/8208/30x1808/483). Was 240 -> lofted MECO to 307 km.
# --- First-stage TRUE GRAVITY TURN (ENABLE_GRAVITY_TURN) ----------------------------
# Real SLS first stage flies an open-loop attitude profile that approximates a GRAVITY
# TURN (near-zero angle of attack through max-Q). This models it faithfully: a vertical
# rise (ASCENT_VERT_S) -> a brief pitch KICK off vertical -> thrust held ALONG the
# RELATIVE-velocity vector (zero AoA), letting gravity turn the trajectory. Replaces the
# open-loop TIME-ramp pitch (which commands a pitch angle regardless of velocity, i.e.
# nonzero AoA). The kick magnitude is the single shaping knob, calibrated so the
# post-SRB-sep handoff -> core IGM reaches the real 157 km / 30x1806 MECO. The core burn
# (post-SRB-sep) is unchanged (IGM linear-tangent; full PEG is the next step). OFF =
# legacy time-ramp pitch (bit-identical to the validated current model).
ENABLE_GRAVITY_TURN = True   # default ON: faithful zero-AoA first-stage gravity turn
GT_KICK_DEG     = 3.47     # pitch-kick angle off vertical (deg). CALIBRATED so the gravity turn hands off
                           # at SRB sep ~47 km (real ~48) and core MECO matches real Artemis I: alt 157.7 km,
                           # FPA +3.621°, 8206 m/s, 30x1806, dur 483.0 s. Validated 24/24 perturbed (robust).
GT_KICK_DUR_S   = 15.0     # kick duration before switching to thrust-along-velocity (s)

# ============================================================
# Mission geometry / targets
# ============================================================
# Distant Retrograde Orbit: ~64,000 km from the Moon (~40,000 mi), retrograde.
# Max distance from Earth reached 432,194 km (268,563 mi) on FD13, 2022-11-28.
DRO_RADIUS_KM            = 64_000.0       # ~40,000 mi from Moon          [sourced]
DRO_MAX_EARTH_DIST_KM    = 432_194.0      # 268,563 mi, FD13 (Nov 28)     [sourced]
LUNAR_FLYBY_ALT_KM       = 130.0          # OPF ~81 mi / RPF ~80.6 mi over Moon [sourced]
# Outbound Powered Flyby ground-elapsed time (GET 005/05:56:16) — the epoch the
# launch azimuth + TLI + OTC chain all target the Moon to. [sourced]
OPF_GET_S                = 453_376.0
# Lambert aim offset from Moon center (km): aiming at an offset point makes the
# ballistic flyby clear the surface at ~130 km rather than impacting. ~4,670 km
# of impact parameter maps to ~130 km periselene at the ~1 km/s lunar approach
# v_inf. TUNED; precise B-plane + DRO-side targeting is the OPF phase's job.
LUNAR_AIM_OFFSET_KM      = 4_670.0
# DRO insertion: the lunar-relative RETROGRADE speed (m/s) at apoapsis that yields
# a bounded DRO. TUNED empirically — a 2-body lunar value is invalid here because
# 64,000 km is at the Moon's Hill-sphere edge (~61,500 km), so the DRO is a pure
# 3-body structure. ~250 m/s retrograde keeps the orbit bounded (~25,000-56,000 km
# from the Moon over 12 d). A proper CR3BP periodic-DRO target is the refinement.
DRO_INSERT_SPEED_MS      = 250.0

# Powered-flyby / DRO burn delta-Vs (ESM/OMS-E).
# NB: in the FLOWN model these are pragmatic-mode (flag-off) budget charges only — the flown
# phases SOLVE the actual burns. Cross-checked vs the as-flown Artemis I:
OPF_DV_MS   = 88.0     # Outbound Powered Flyby (real FD6, ~130 km flyby; 2128->5102 mph is
#                        gravity-assist+burn, so the burn dV alone isn't sourced) [epoch sourced]
DRI_DV_MS   = 90.0     # DRO Insertion (real FD10 / GET 9.66 d; sim flown SOLVES ~196 m/s) [epoch]
DDP_DV_MS   = 30.0     # DRO Departure (real FD16 / GET 15.66 d; sim flown SOLVES ~161 m/s) [epoch]
RPF_DV_MS   = 250.0    # Return Powered Flyby. REAL burn 3 m 27 s -> ~280 m/s (FD20 / GET 19.43 d).
#                        Sim FLOWN min-dV RPF SOLVES ~324-332 m/s (≈ real ~280); the legacy velocity-
#                        match over-burned to ~596 (2.1x) before the min-dV solver + re-seeded v_np.
RTC_DV_MS   = 50.0     # Return Trajectory Corrections (real: several RTCs from FD15.9; sim flown chain)

# --- Pragmatic return model -------
# The trajectory-perfect return (DDP->RPF->Earth) needs invariant-manifold /
# multiple-shooting targeting on a clean CR3BP DRO — a dedicated optimizer
# project. For the success-probability MC, the RETURN is modeled by its binding
# constraint: the ESM ΔV BUDGET (does Orion have propellant to get home?) plus
# the timeline, ending at a representative entry-interface state. The return
# TRAJECTORY is not propagated (the outbound half is full-fidelity; the return is
# budget + representative-EI). Clearly flagged in each return phase.
ENABLE_PRAGMATIC_RETURN = True

# FLOWN RETURN (the fix for the above): when True, the return is FLOWN as a real
# trajectory — DRO departure (DDP, retrograde-to-Moon -> ~100 km lunar perilune)
# -> return powered flyby (RPF, prograde at perilune, the gravity-assist burn that
# does the work) -> trans-earth coast with a Return Trajectory Correction (RTC)
# that targets the entry FPA -> the REAL 122 km entry interface (no synthetic EI).
# Validated: nominal closes at total return dV ~348 m/s (DDP
# ~193 + RPF ~155, budget 733), entry v ~10,989 m/s, FPA targeted -6.5. Takes
# precedence over ENABLE_PRAGMATIC_RETURN. When False, the pragmatic path runs
# (bit-identical to pre-integration). Entry FPA is hyper-sensitive to the RPF
# (~3.8 deg/m/s near the corridor) so the FINAL FPA target is set by the RTC close
# to Earth (low leverage); the per-trial FPA dispersion enters there.
ENABLE_FLOWN_RETURN = True

# Entry interface (Orion reentry): ~122 km altitude (400,000 ft), ~10.99 km/s
# (24,581 mph, sourced), skip-entry flight-path angle ~-6.5° (representative).
ENTRY_INTERFACE_ALT_KM = 122.0
ENTRY_VELOCITY_MS      = 10_990.0     # [sourced] 24,581 mph
ENTRY_FPA_NOM_DEG      = -5.9         # Orion's SHALLOW skip corridor (real corridor -6.5..-5 deg;
#   -6.5 was the STEEP max-g edge = ~8 g). -5.9 floors at ~4-5 g (≈ real Orion ~4 g). REQUIRES the
#   PredGuid drag-tracker (ENABLE_PREDGUID_ENTRY) to capture from the shallow corridor — the legacy
#   fixed-bank/predictor-corrector skips out here. With PredGuid: peak-g ~5 g, all-land, tight miss
#   (validated entry_harness captest2). Flag OFF + this FPA would skip out; revert to -6.5 for legacy.
PREDGUID_G_REF         = 4.0          # drag-tracker reference g (capture target)
PREDGUID_NOM_CDMAX     = 0.4          # nominal climb-out lift-down cap (sets the zone downrange)
# Orion Crew Module aerodynamics (capsule).
CM_CD                  = 1.35         # hypersonic drag coefficient  ESTIMATE
CM_AREA                = 19.8         # m^2 (~5.0 m diameter)        [sourced ~]
CM_LD                  = 0.27         # hypersonic lift-to-drag      [sourced ~]
ENTRY_BANK_DEG         = 30.0         # partial lift-up: robust single-pass entry,
                                      # ~20 min flight (≈ real), low skip-out risk TUNED
# Orion EDL parachute descent (drogue → mains → splash) + wind drift. The entry integration ends at
# sea level, but below ~drogue deploy the capsule is under chutes (terminal velocity + wind drift) for
# several minutes — physics the entry omits. ENABLE_EDL_DESCENT adds the descent TIME (→ EI→splash
# closer to the real ~20 min) and a per-trial WIND-DRIFT dispersion (~km) on the splash point.
ENABLE_EDL_DESCENT     = True
DROGUE_DEPLOY_ALT_M    = 7600.0       # drogue deploy ~25,000 ft         [sourced ~]
MAIN_DEPLOY_ALT_M      = 2900.0       # main-chute deploy ~9,500 ft      [sourced ~]
DROGUE_DESCENT_MS      = 50.0         # avg drogue-phase descent rate     ESTIMATE
SPLASH_TERMINAL_MS     = 8.9          # splashdown speed under mains (~20 mph) [sourced]
EDL_WIND_SIGMA_MS      = 7.0          # 1σ horizontal wind per component (sea-level) ESTIMATE
ENTRY_STRUCTURAL_G     = 12.0         # CM structural g limit        ESTIMATE
# --- ENTRY terminal-nav residual — the OD-filter observability hook. Once the guidance
# is tightened (Lever A+B) the guidance error collapses below the ~3.5 km chute-wind floor, so the
# splash is ~3.1 km — TIGHTER than the real ~4.7 km (the audit flags "over-accurate"). Real landing
# accuracy is floored by the DELIVERED EI-state OD knowledge error the guidance cannot correct. Model
# it exactly like the DRI OD_NAV_RESIDUAL: a per-trial splash-position offset drawn from the EI OD
# covariance (here a calibrated constant σ; the OD FILTER later REPLACES the constant with the emergent
# EI covariance -> splash becomes RESPONSIVE to nav quality = the un-masking). Calibrated so the fleet
# splash-miss median = the real ~4.7 km (match, NOT beat). OFF -> zero residual (bit-identical).
ENABLE_ENTRY_NAV_RESIDUAL = True
ENTRY_NAV_RESIDUAL_KM  = 2.5          # 1σ per-axis EI-nav->splash residual. CALIBRATED on the validated
#   A+B fleet (floor 3.66 km): σ=2.5 -> combined splash-miss median 4.74 km ≈ the real ~4.7 km (match,
#   not beat). EST-grade (anchored to real accuracy). The OD filter later makes this emergent from the
#   delivered EI covariance -> splash becomes responsive to nav quality (the un-masking hook).
# --- Hypersonic-entry atmospheric DENSITY dispersion -------------------------------------------------
# The hypersonic entry flew a STATIC USSA-76 atmosphere -> the entry-g/splash spread came only from the
# EI-state (FPA) dispersion, leaving the run artificially tight on the atmospheric axis. Real entry-g
# scales with density (g ∝ ρ at the peak), and entry-altitude density varies day-to-day/latitude/solar.
# A per-trial MULTIPLICATIVE density scale (constant over the entry) captures the dominant entry-g
# effect honestly. (Winds aloft are NEGLIGIBLE at ~11 km/s hypersonic; chute-phase wind drift is already
# modeled via EDL_WIND_SIGMA_MS.) When OFF, the scale is 1.0 -> bit-identical to the static atmosphere.
ENABLE_ENTRY_ATM_DISP  = True
ENTRY_DENS_SIGMA       = 0.06         # 1σ density-scale (~6%, entry-altitude variability) ESTIMATE
_ENTRY_DENS_SCALE      = 1.0          # per-trial scale, set at the top of phase_entry from the perturb
#   (An in-RHS g-limiter that rolled bank->0 at high load was tried and removed — it
#   deepened skips and drove shallow EIs to 70+ g. Entry g-control lives entirely in
#   _entry_solve_bank, which rejects banks that overstress or skip out.)
# Representative entry-interface ground point (FIXED, independent of SPLASH_TARGET
# to avoid a circular dependency). The pragmatic EI is built ~here; the nominal
# entry then lands at SPLASH_TARGET (which is set to that nominal splashdown).
EI_GROUND_LAT          = 28.0
EI_GROUND_LON          = -145.5

# Splashdown target. The REAL Artemis I splashed in the Pacific off Baja
# California (~28°N, 117.5°W). BUT in the pragmatic return model the return
# TRAJECTORY/PLANE is not propagated, so the constructed entry-interface state's
# absolute landing location is NOT physical (it lands wherever the representative
# EI geometry sends it — the south Atlantic for the nominal). So SPLASH_TARGET is
# set to the sim's OWN pragmatic nominal splashdown, making `splash_miss_km` a
# meaningful ENTRY DISPERSION metric (spread vs the nominal) rather than a
# distance to the real recovery zone. Re-deriving the true Pacific location needs
# the full-fidelity return (the deferred invariant-manifold work). Real Pacific
# point: (28.0, -117.5).
SPLASH_TARGET = (-21.41, -6.35)    # FLOWN-return nominal splash (S. Atlantic). The
                                   # entry is fixed-bank, so the splash point is set
                                   # by the flown EI state, not this constant — which
                                   # only anchors the splash_miss DISPERSION metric.
                                   # Re-derive whenever the return geometry changes.
                                   # Region is a residual (return departure-phase
                                   # ~0.9 rev). Pragmatic-path nominal: (10.298,-86.476).

# Reference mission length (liftoff -> splashdown), for any uniform-timing draws.
MISSION_REF_DURATION_S = 25.0 * 86400 + 10 * 3600 + 53 * 60   # ~25 d 10 h 53 m

# ============================================================
# Feature flags  (top-of-file)
# ============================================================
# Launch / ascent
ENABLE_SRB_DYNAMICS      = True   # model the two SRBs explicitly (thrust+jettison)
ENABLE_RS25_CORE         = True   # 4× RS-25 core-stage integration to MECO
ENABLE_IGM_ASCENT        = True   # closed-loop IGM-style (linear-tangent) MECO
                                  # guidance to the sourced 30×1806 km insertion
# PEG Newton solve bounds (parametrized for the IGM-fallback diagnosis; defaults = the
# original hardcoded literals -> bit-identical). The clips bound the (χ0, tan-rate) linear-tangent
# family; weak-ascent trials stall the solve at resid ~1.1 (~110 km from the (perigee, r_cut) target).
PEG_STEP_CHI_MAX = 8.0            # max Newton step in χ0 (deg)
PEG_STEP_TR_MAX  = 1.5e-3         # max Newton step in tan-rate
PEG_CHI0_MIN, PEG_CHI0_MAX = 10.0, 70.0     # χ0 search bounds (deg)
PEG_TR_MIN,   PEG_TR_MAX   = -4e-3, 1e-3    # tan-rate search bounds
# TWO-SEGMENT PEG (the PEG-robustness fix): when the single-segment
# 2-DOF Newton STALLS (weak-ascent trials: reachable-set floor ~110 km from the (perigee, r_cut) target;
# bounds-widening proven useless 0/12), retry with a SECOND linear-tangent segment (tan-rate switches at
# t_core0 + PEG2_TBREAK_S; tan(χ) continuous) — one extra steering DOF ≈ real PEG's continuously
# re-solved profile. Physically honest: the weak vehicles are only ~0.5% down on thrust with ~33 t of
# core reserve at MECO — real closed-loop guidance absorbs that; only the sim's rigid 1-segment family
# could not. Ladder order preserves bit-identity for converging trials (2-DOF first; 2-seg ONLY on
# stall; IGM stays the final fallback). Default ON; AR1_PEG2SEG=0 -> pre-fix lineage, bit-identical.
ENABLE_PEG_2SEG = (os.environ.get("AR1_PEG2SEG", "1") == "1")   # DEFAULT ON (adopted in the definitive production run; AR1_PEG2SEG=0 -> pre-fix lineage, bit-identical)
PEG2_TBREAK_S   = 180.0           # rate-switch time after core-phase start (mid-core burn)
ENABLE_PEG_GUIDANCE      = True   # default ON: full PEG (Powered Explicit Guidance): a 2-DOF linear-tangent
                                  # predictor-corrector solving (χ0, χ̇) to the FULL target cutoff
                                  # (orbit energy a_T + ang.mom h_T at the real ~157 km MECO radius,
                                  # ascending), vs the 1-DOF IGM. Goal: faithful one-engine-out-to-orbit.
                                  # The IGM (above) is the convergence fallback. OFF = 1-DOF IGM.
PEG_MECO_ALT_KM          = 157.0  # target MECO radius (real Artemis I core-sep altitude) [sourced]
                                  # OFF = legacy open-loop pitch.
ENABLE_ICPS_TLI          = True   # fly the ICPS perigee-raise (PRM) + TLI burns
ENABLE_FINITE_TLI        = True   # fly the TLI as a STEERED ~18-min finite ICPS burn (vs impulsive). The
                                  # real TLI burns ~18 min; modeling it finitely fixes the post-TLI PHASING
                                  # (the vehicle coasts from BURNOUT, not the ignition point) — verified to
                                  # cut the post-TLI residual vs the OEM. OFF = legacy impulsive (bit-identical).
FINITE_TLI_PROP_ALLOWANCE_KG = 600.0  # extra ICPS propellant for the finite burn's ~47 m/s gravity loss
                                      # (applied only when ENABLE_FINITE_TLI; see phase_icps_tli m0 note)
ENABLE_FINITE_PRM        = True   # default ON. Fly the PRM as a finite ~22 s RL10 burn CENTERED at apogee (vs an
                                  # impulsive Δv). The real PRM is a 22 s burn; at apogee the burn is
                                  # tangential and gravity is radial, so the finite-vs-impulsive gravity
                                  # loss is ~0 — this is a FAITHFULNESS/consistency match (real flown burn,
                                  # consistent with ENABLE_FINITE_TLI), not a meaningful trajectory change.
                                  # OFF = legacy impulsive Δv (bit-identical).
ENABLE_FIRST_REV_TLI     = True   # TLI ignites on the FIRST parking rev (Artemis I
                                  # fired ~37 min after the PRM). OFF = global min over
                                  # ~1.5 revs (the min-dv valley recurs per-rev within
                                  # ~5 m/s, so the global min slipped a rev -> +130%
                                  # phase + platform-fragile near-tie). See phase_icps_tli.
ENABLE_LAUNCH_CONTINUITY = True   # pad-to-splashdown as ONE continuous trajectory
                                  # (TLI ignites from the trial's launched orbit)

# Deep-space targeting / corrections
ENABLE_OUTBOUND_MCC          = True   # Outbound Trajectory Correction chain (OTC-1..n)
# Phase-3 faithful outbound OTC chain (AAS 23-363 Table 1 [ft/s!] + timeline).
# Real Artemis I: burn SLOTS fixed pre-flight (timeline-driven, Apollo precedent), execution CONDITIONAL —
# "correction burns were not modeled in Copernicus prior to flight. These were added as needed throughout
# the mission" (OTC-3/4 design ΔV = 0, as-flown 0.87/0.22 m/s) — EXCEPT the mandatory OTC-1 OMSe CHECKOUT
# (min on-time ≥30 s → ~31–35 m/s ≈ the whole 36 m/s outbound budget; proves the engine before OPF).
# When ON: OTC-1 charges max(solved-correction, min-on-time floor) to the ESM ledger (the checkout's excess
# ΔV was a designed-in trajectory component for the real mission; the sim flies the solved correction and
# charges the honest propellant), then OTC-2/3/4 fire at their real GET slots ONLY if the predicted
# periselene error exceeds the deadband. No RNG draws -> OFF is bit-identical to the legacy single-OTC path.
ENABLE_OTC_CHAIN             = True   # OFF = legacy single always-fire targeting OTC (bit-identical)
OTC1_MIN_ONTIME_S            = 30.0   # OMSe checkout minimum on-time (∆t ≥ 30 s)            [sourced]
OTC_SLOT_GETS_S              = (103_495.0, 365_100.0, 431_790.0)  # OTC-2/3/4 real GETs      [sourced]
OTC_DEADBAND_PERI_KM         = 10.0   # slot fires only if |predicted periselene - target| exceeds this
                                      # (estimate-grade stand-in for the ground-OD "added as needed" call)
# NAV slice-1: burn EXECUTION error (cutoff magnitude + pointing) on the OTC chain burns.
# The other OMS-E burns already consume sourced scalar biases (opf/dri/ddp/rpf_dv_bias_ms + omse_isp_factor);
# the OTC burns applied the solved Δv EXACTLY — the one leg where execution error propagates live to the
# next event (the flyby), and the reason the real OTC-2/3/4 trims existed. With this ON, the slots fire
# EMERGENTLY: OTC-1's ~0.4-0.6 m/s execution scatter re-disperses the periselene by ~100-300 km, the later
# slots detect + trim it (calibration target: as-flown trims 0.22/0.87/0.22 m/s, AAS 23-363 §2c).
# σ anchored to AAS 23-363: "each executed burn was within 2 ft/s (0.61 m/s) of its estimated ΔV" —
# OTC-1 (~94 m/s) draws σ ≈ 0.39 m/s (2 ft/s ≈ 1.6σ). Small trims (< OTC_EXEC_LARGE_MS) use the aux/SMRCS
# branch (real trims flew on aux/RCS — finer impulse floor, coarser pointing). Draws come from a SPAWNED
# child stream (parent untouched) -> flag OFF is BIT-IDENTICAL to the pre-flag lineage. EST-grade.
ENABLE_OTC_EXEC_ERRORS       = True   # OFF = OTC burns execute exactly (the perfect-execution limit)
OMSE_EXEC_MAG_FLOOR_MS       = 0.20   # OMSe cutoff-timing/accelerometer floor (~0.2 s at ~1 m/s²)  [est]
OMSE_EXEC_MAG_FRAC           = 0.002  # proportional thrust/Isp knowledge                           [est]
OMSE_EXEC_POINT_DEG          = 0.25   # attitude/TVC pointing execution (1σ per lateral axis)       [est]
AUX_EXEC_MAG_FLOOR_MS        = 0.02   # aux/SMRCS fine-impulse floor (small trims)                  [est]
AUX_EXEC_MAG_FRAC            = 0.01   #                                                             [est]
AUX_EXEC_POINT_DEG           = 0.50   #                                                             [est]
OTC_EXEC_LARGE_MS            = 3.0    # commanded ΔV ≥ this -> OMSe branch; below -> aux/SMRCS trim
# NAV slice-2: ground-OD KNOWLEDGE error on the OTC chain. The slot solves AND the
# deadband fire/skip decisions run on an ESTIMATED state (true + a per-slot RIC-drawn knowledge error
# whose σ shrinks with the tracking-arc length: ~7 h at OTC-1 -> ~5 d at OTC-4); the commanded Δv then
# EXECUTES on the true state (with slice-1 execution error). This is the real reason trims re-fired:
# the ground corrected its ESTIMATE of the miss, not the truth. σ are EST-grade DSN cislunar OD
# (range/Doppler: km-class position / cm-per-s-class velocity on short arcs, improving with the arc;
# in-track dominant). Draws from a THIRD spawned child stream -> flag OFF bit-identical to the
# slice-1 lineage. Calibration target stays the as-flown trim pattern (§2c).
ENABLE_OTC_OD_ERRORS         = True   # OFF = the ground knows the true state (perfect-OD limit)
OTC_OD_POS_SIGMA_M           = ((1000.0, 2000.0, 1000.0),   # OTC-1 (~7 h arc)   [est]
                                (500.0, 1000.0, 500.0),     # OTC-2 (~1.2 d arc) [est]
                                (300.0, 600.0, 300.0),      # OTC-3 (~4.2 d arc) [est]
                                (300.0, 600.0, 300.0),      # OTC-4 (~5.0 d arc) [est]
                                (300.0, 600.0, 300.0),      # OTC-5 (post-OPF)   [est]
                                (300.0, 600.0, 300.0))      # OTC-6 (pre-DRI)    [est]
OTC_OD_VEL_SIGMA_MS          = ((0.010, 0.020, 0.010),      # [r, in-track, cross] per slot
                                (0.005, 0.010, 0.005),
                                (0.003, 0.006, 0.003),
                                (0.003, 0.006, 0.003),
                                (0.003, 0.006, 0.003),
                                (0.003, 0.006, 0.003))
# Post-OPF conditional trim slots OTC-5/6 (AAS 23-363 §2d: as-flown 0.98 / 2.68 m/s, "added as
# needed") — the real chain's absorber for OPF->DRI transit dispersion (phase-5: without them,
# ~12% of dispersed trials arrived out-of-band at the DRO and paid honest depletion failures).
OTC56_SLOT_GETS_S            = (515_700.0, 745_484.0)       # OTC-5 11-22 06:02:44, OTC-6 11-24 21:52:28
ENABLE_POWERED_FLYBY_TARGETING = True # solve the OPF/RPF gravity-assist burn vectors
ENABLE_OPF_STRADDLE          = True   # phase-4 fix: center the finite OPF burn on periselene (real burn
                                      # straddled CA — max Oberth), vs the legacy start-at-CA. OFF = legacy.
ENABLE_DRO_TARGETING         = True   # solve DRI (insertion) / DDP (departure)
ENABLE_RETURN_MCC            = True   # Return Trajectory Correction chain (RTC-1..n)
ENABLE_RETURN_POLISH         = True   # DEFAULT ON: iterate the Phase-C return linear
                                      # correction to CONVERGENCE (frozen-Jacobian Newton on the 6-DOF
                                      # DDP+RPF match residual) so dispersed trials stay on the
                                      # epoch-committed planned return instead of defecting to the free
                                      # in-plane fallback (the Indian-Ocean basin). The single Jlin step
                                      # left ~20 km match residual on trials past the linear envelope,
                                      # which propagated to a >0.35deg entry-FPA gate failure over the
                                      # 4-d match->EI leg; one-to-few Newton passes drive it to <1 km
                                      # (sub-0.1 m/s burn deltas, no ESM cost). OFF = single-step Jlin
                                      # only (bit-identical to the pre-polish lineage).

# Environment models
ENABLE_REAL_EPHEMERIS = True    # (declared above) real ephemeris + GMST
ENABLE_EARTH_HIGHER_ZONALS = True   # Earth J3-J6 zonals (force-model completeness). Beyond
                                # the J2 already modeled; negligible for a lunar trajectory (near-Earth
                                # phases only) but makes the Earth field faithful. OFF = J2-only
                                # (bit-identical to the prior lineage). See EARTH_ZONALS.
ENABLE_SOLAR_GRAVITY  = True    # NEW for Artemis: Sun third-body term in the
                                # gravity model — first-order for the 25-day DRO
                                # (Apollo's 8-day flight didn't need it). OFF =
                                # Earth+Moon only (the legacy model).
ENABLE_SRP            = True    # solar radiation pressure (anti-sunward; ~tens of km over the
                                # 25-day coast). Fixed A/m (SRP_AREA_M2/SRP_MASS_KG). DEFAULT ON.
                                # The earlier regression was on the LEGACY near-polar
                                # model, whose RPF seed was a hardcoded velocity-match to the no-SRP
                                # _NEAR_POLAR_V_NP (SRP threw it -> RPF 324->491, ddp_no_earth). The
                                # CURRENT snap-free + Phase-C model is ADAPTIVE: the OTC ca_point_trim
                                # targets the baked CA point and the return J_lin+RTC re-solve per trial,
                                # so they ABSORB the SRP shift (impact probe: 128.8 km outbound -> 0 by
                                # DRI; CA 146.10 vs 146.12, DRI/DDP/RPF unchanged, nominal succeeds).
                                # No baked-constant re-derivation needed. OFF = bit-identical (no draws).
ENABLE_LUNAR_HARMONICS = True   # Moon non-spherical figure (gates the field
                                # entirely; matters only near the ~130 km flybys)
ENABLE_LUNAR_SH_FIELD = True    # degree-8 GRAIL field (vs degree-2 closed form)
                                # within ~3,500 km of the Moon
ENABLE_LUNAR_LIBRATION = True  # IAU-2009 lunar rotation for the Moon-fixed frame (true pole +
                                # prime meridian incl. physical librations ±6.7° + obliquity ~1.5°)
                                # vs the tidal-lock approximation. Matters only near the ~130 km
                                # flybys (mis-points the C22/J2 bulge). OFF = tidal-lock (production);
                                # validate against the hypersensitive return before enabling.

# Entry
ENABLE_SKIP_ENTRY_GUIDANCE = True   # Orion skip-entry guidance to the splash target
ENABLE_PREDGUID_ENTRY = True        # PredGuid-style DRAG-TRACKER entry (vs the legacy predictor-
#   corrector): captures the SHALLOW corridor (ENTRY_FPA_NOM_DEG -5.9) at ~floor-g, holds it (capped
#   lift-down kills the dive-back overshoot), nulls range via cdmax. Gives ~5 g (≈ Orion 4 g) vs the
#   legacy 8 g, all-land, tight miss. OFF = legacy guided entry (needs ENTRY_FPA_NOM_DEG back at -6.5).
ENABLE_ENTRY_CROSSRANGE = True      # LATERAL entry control: bank REVERSALS (Apollo/Orion
#   lateral logic) — the bank SIGN is closed-loop on cross-track error to the target instead of a single
#   fixed sign picked once. The fixed sign nulls only DOWNRANGE (range via cdmax/v_exit) and left ~405 km
#   of CROSSRANGE (the nominal splashed ~405 km E of its aim). Reversals null the cross-track: harness
#   405 -> 66 km to the aim, peak-g unchanged (~4.1). OFF = the fixed-sign pick (bit-identical).
ENABLE_ENTRY_TERMINAL_TRIM = True   # DOWNRANGE terminal trim (Lever A): the extended-skip
#   range(v_exit) reachable set is GAP-RIDDLED (chaotic recapture — sparse discrete ranges separated by
#   hundreds-of-km gaps; the aim point often falls IN a gap, so the open-loop v_exit grid settled ~60 km
#   short/long and a finer grid or a secant CANNOT help — you cannot land in a gap). The lob LIFT fraction
#   (lob_cc, the below-cap vertical lift during the shallow lob) is instead a SMOOTH, MONOTONE range knob
#   (~3000 -> 6300 km, zero jumps) that holds ~4.1-4.4 g for lob_cc in [0.6,1.0]. So: a coarse v_exit
#   scan BRACKETS the target energy (at full lob lift), then a BISECTION on lob_cc nulls the downrange ->
#   miss ~0 (harness: downrange 55 -> 0.0 km median / 0.7 max across FPA x target dispersion; 21/21 land;
#   ~23 flies ≈ the old grid; peak-g ~4.4). Also tightens the crossrange deadband to
#   ENTRY_XRANGE_DEADBAND_KM (Lever B). Guidance is RNG-free -> flag OFF = the v_exit grid + 25 km
#   deadband (bit-identical to the prior lineage).
ENTRY_XRANGE_DEADBAND_KM = 5.0      # Lever B: crossrange bank-reversal deadband (was the hard-wired 25 km
#   in _predguid_rhs). The 25 km bang-bang held the last sign with no nulling inside -> an ~8 km crossrange
#   floor; 5 km tightens it to ~2-3 km. Applied only under ENABLE_ENTRY_TERMINAL_TRIM (else the rhs
#   default 25.0 stands -> bit-identical).
ENTRY_LOBCC_MIN = 0.35              # lob_cc bisection floor (below ~0.6 the dive-in raises peak-g; 0.35
#   gives bracketing margin without forcing a high-g solution for in-family targets).
ENABLE_ENTRY_LEAD_COMP = (os.environ.get("AR1_UPGRADE2", "0") == "1")  # UPGRADE 2 — INVESTIGATED + REJECTED
#   (keep OFF). Analytical drag-derivative LEAD compensator in the PredGuid drag-tracker: feeds a
#   PREDICTED drag D_pred = D + tau_lead*dD/dt (dD/dt = D*(-2D/vr - alt_rate/H), the exact derivative of
#   D=0.5*rho*vr^2*cda/m under an exponential atmosphere + drag-dominated decel) into the regulator command,
#   so the loop acts BEFORE the decel peak (anticipatory; intended to hold the shallow -5.9 corridor without
#   peak-g overshoot). Only shapes the CONTROL command u (bank angle) — a_drag/a_lift still use the REAL drag.
#   VERDICT (A/B validated at scale, vs the flag-OFF baseline, at tau=2 AND tau=4): it
#   shaves the peak-g BODY toward the real ~4 g (median -0.10 @tau2 / -0.19 @tau4; p90/p99 down too) but
#   (a) that metric was ALREADY matched (baseline med 4.43 ≈ real ~4 g, no gap to close) and (b) it
#   introduces an INHERENT >5 g overshoot outlier on some chaotically-dispersed entry that tau only
#   RELOCATES, not removes (tau4 spiked trial 72 -> 5.15 g; tau2 fixed 72 but spiked trial 26 -> 5.30 g,
#   i.e. tau2 max is WORSE). Success 90.00% / splash / audit all identical across OFF/tau2/tau4 (outliers
#   benign so far), so no offsetting benefit — degrading the worst case to improve an already-matched body
#   is the wrong trade for a peak-g compensator. Default OFF; OFF -> D_pred=D, bit-identical. AR1_UPGRADE2=1
#   (+ AR1_UPGRADE2_TAU) to re-enable for further study.
ENTRY_LEAD_TAU_S = float(os.environ.get("AR1_UPGRADE2_TAU", "4.0"))  # UPGRADE 2 lead time constant (s).
#   Larger = more anticipation (bigger peak-g body shave) but more overshoot risk on dispersed trajectories.
#   Sweeping it does NOT remove the >5 g outlier, only moves which trial it hits (see the verdict above).

# Failure modeling
ENABLE_SLS_ASCENT_FAILURES = True   # SRB / RS-25 / core / ICPS engine-out + ignition
ENABLE_ESM_SYSTEMS_FAILURES = True  # ESM propulsion/power systems failure analog
ENABLE_HEATSHIELD_FAILURE   = True  # AVCOAT/skip-entry thermal risk (Artemis I saw
                                    # unexpected char loss — model as a tail risk)
# Extended SOURCED failure modes (phase-3; NASA / AAS references).
ENABLE_OMS_E_BURN_FAILURES  = True  # AJ10/OMS-E per-burn ignition failure (OPF/DRI/
                                    # DDP/RPF; ~2e-3 each, AJ10 ~0.998 heritage)
ENABLE_AVIONICS_FAILURES    = True  # radiation power-distribution/avionics anomaly
                                    # (Artemis I: latching limiter opened; ~1e-2,
                                    # mostly recoverable)
ENABLE_PARACHUTE_FAILURE    = True  # Orion EDL parachute (2-of-3 mains) failure
                                    # (~3e-3; elevated by Artemis I char-loss debris)
ENABLE_SEP_FAILURE          = True  # CM/SM separation-bolt failure (Artemis I: 3/4
                                    # bolts eroded but separated; ~1e-3)
ENABLE_FORESEEABLE_FAILURES = True  # foreseeable cislunar modes with NO Artemis data
                                    # (MMOD, nav-sensor loss, ESM pressurization, comm-
                                    # loss-at-burn, RCS, thermal, SPE, DRO station-
                                    # keeping) — sourced by analogue/PRA/foresight,
                                    # all estimate-grade (failure-mode-sourcing-method).
ENABLE_LATE_SYSTEMS_CHECK   = True  # coverage fix: the timeline-struck systems
                                    # modes (ESM catastrophic, avionics, foreseeable set) are
                                    # drawn uniformly over the FULL ~25.45 d mission, but the
                                    # last _esm_check ran only at RPF (~19.4 d) — strikes in the
                                    # final ~6 d (trans-Earth coast -> CM/SM sep, ESM still
                                    # attached; ~24% of the timeline) were SILENTLY dropped,
                                    # making the effective systems rates ~24% low. This adds a
                                    # post-trans-Earth-coast milepost so the full ESM-active
                                    # window is evaluated. Consumes NO new RNG draws (the fracs
                                    # are already drawn) -> OFF = bit-identical to the pre-fix
                                    # lineage; ON only reclassifies late strikes.
ENABLE_EXPOSURE_SCALING     = (os.environ.get("AR1_EXPOSURE_SCALING", "0") == "1")  # DEFAULT OFF;
                                    # AR1_EXPOSURE_SCALING=1 flips it ON for a run (spawn-safe: workers
                                    # inherit the env). A cross-mission comparison convention
                                    # (bit-identical OFF). Re-express the TIME-DRIVEN hazards (ESM
                                    # catastrophic, avionics-radiation, MMOD, nav-sensor, ESM-
                                    # pressurization, RCS, thermal, SPE, DRO station-keeping) as a
                                    # Poisson exposure law P = 1 - exp(-lambda*T) with lambda RE-ANCHORED
                                    # so P(T_ref) reproduces the current sourced per-mission rate exactly
                                    # (`_exposure_scaled_prob`). A VARIANCE refinement, NOT a recalibration.
                                    # NOT applied to per-EVENT modes (OMS-E/TLI ignition, CM-SM sep,
                                    # parachute, comm-at-burn). NATURE-class math (period-neutral); the
                                    # rates stay Artemis-sourced (artifact). Artemis flies ~all of its
                                    # 25.45 d beyond the magnetosphere so the shielding-weighted exposure
                                    # ~= the full duration and T_trial ~= T_ref -> OUTCOME-NEUTRAL within
                                    # Artemis (headline unchanged); the deliverable is the per-time lambda
                                    # for the FAIR Apollo(~8 d)-vs-Artemis(~25 d) comparison. Same draw
                                    # consumed either way -> OFF bit-identical.


def _exposure_scaled_prob(rate, T_exposure_s, T_ref_s):
    """Re-anchored Poisson time-exposure scaling (cross-mission comparison convention):
    P = 1 - exp(-lambda*T) with lambda = -ln(1-rate)/T_ref, i.e. P = 1-(1-rate)**(T/T_ref).
    P(T_ref) == rate EXACTLY (variance refinement, nominal-duration mean preserved); longer
    exposure -> higher P. NATURE-class / period-neutral; `rate` stays the Artemis-sourced input."""
    if rate <= 0.0 or T_ref_s <= 0.0:
        return rate
    frac = T_exposure_s / T_ref_s
    if frac == 1.0:
        return rate                      # exact re-anchor at nominal exposure (bit-stable)
    return 1.0 - (1.0 - rate) ** frac


# ============================================================
# Physics helpers  (mission-agnostic primitives,
# copied so artemis1.py stays standalone; the Sun term is new for Artemis)
# ============================================================

def moon_state(t):
    """Moon position and velocity in ECI [m, m/s] at time t."""
    if ENABLE_REAL_EPHEMERIS:
        r = _moon_eci_m(_jde_tt(t))
        dt = 60.0  # s; velocity by central difference (Moon moves ~13 deg/day)
        v = (_moon_eci_m(_jde_tt(t + dt))
             - _moon_eci_m(_jde_tt(t - dt))) / (2 * dt)
        return r, v
    # Idealized circular Moon (legacy / flag-off only).
    theta = OMEGA_M * t
    ci, si = np.cos(MOON_INC), np.sin(MOON_INC)
    r = EM_DIST * np.array([np.cos(theta), np.sin(theta)*ci, np.sin(theta)*si])
    v = EM_DIST * OMEGA_M * np.array([-np.sin(theta), np.cos(theta)*ci, np.cos(theta)*si])
    return r, v


def _earth_zonal_accel(r):
    """Perturbing acceleration (ECI, m/s^2) from Earth zonal harmonics J3..J6 (J2 is handled inline in
    gravity_earth_moon). Computed as the gradient of the zonal potential U_z = -(mu/rn) Σ_n Jn (Re/rn)^n
    Pn(z/rn) by central finite difference — robust + convention-matched to the inline J2 (verified: the
    same construction reproduces the analytic J2 acceleration). Legendre Pn via Bonnet recursion."""
    def _Uz(rv):
        rn = np.linalg.norm(rv); u = rv[2] / rn
        p0, p1 = 1.0, u; tot = 0.0
        for n in range(2, 7):                         # build P2..P6, accumulate J3..J6
            pn = ((2 * n - 1) * u * p1 - (n - 1) * p0) / n
            p0, p1 = p1, pn
            if n in EARTH_ZONALS:
                tot += EARTH_ZONALS[n] * (R_EARTH / rn) ** n * pn
        return -(MU_EARTH / rn) * tot
    a = np.zeros(3); h = 1.0
    for k in range(3):
        e = np.zeros(3); e[k] = h
        a[k] = (_Uz(r + e) - _Uz(r - e)) / (2.0 * h)
    return a


def gravity_earth_moon(r, t):
    """Earth (with J2 + optional J3-J6 zonals) + Moon + (Artemis) Sun third-body acceleration in ECI."""
    rn = np.linalg.norm(r)
    # Earth point mass
    a = -MU_EARTH * r / rn**3
    # J2
    z2_r2 = (r[2] / rn)**2
    f = 1.5 * J2 * MU_EARTH * R_EARTH**2 / rn**5
    a = a + f * np.array([r[0]*(5*z2_r2-1), r[1]*(5*z2_r2-1), r[2]*(5*z2_r2-3)])
    # Higher Earth zonals J3-J6 (completeness; negligible past the near-Earth phases, so gated by range)
    if globals().get("ENABLE_EARTH_HIGHER_ZONALS", False) and rn < EARTH_ZONAL_R_MAX_M:
        a = a + _earth_zonal_accel(r)
    # Moon third body (the second term keeps this a TOTAL inertial accel:
    # subtract the Moon's pull on the Earth-centered frame origin).
    mr, _ = moon_state(t)
    dr = mr - r
    dr_norm = np.linalg.norm(dr)
    a = a + MU_MOON * dr / dr_norm**3
    a = a - MU_MOON * mr / np.linalg.norm(mr)**3
    # Sun third body (NEW for Artemis — first-order on the DRO leg). Same tidal
    # form: spacecraft pull minus the common-mode pull on the frame origin.
    if ENABLE_SOLAR_GRAVITY or globals().get("ENABLE_SRP", False):
        sr = sun_state(t)
    if ENABLE_SOLAR_GRAVITY:
        ds = sr - r
        a = a + MU_SUN * ds / np.linalg.norm(ds)**3
        a = a - MU_SUN * sr / np.linalg.norm(sr)**3
    if globals().get("ENABLE_SRP", False):     # solar radiation pressure (push anti-sunward)
        d = r - sr; dn = np.linalg.norm(d)
        a = a + (SRP_P0_NM2 * SRP_CR * (SRP_AREA_M2 / SRP_MASS_KG) * (AU / dn)**2) * (d / dn)
    # Moon non-spherical figure: only meaningful near the Moon (falls as ~1/r^4).
    # Reuse the Moon-relative distance to skip the call entirely when far away.
    if dr_norm < 5.0e7:
        a = a + lunar_nonspherical_accel(-dr, t)
    return a


# --- Lunar non-spherical gravity (degree-2 closed form + degree-8 GRAIL) ------
# SOURCED degree-2 figure (Konopliv/GRAIL, reference radius 1738 km). C22 for the
# Moon is only ~1 order below J2 (unlike Earth), so it matters. These are the
# LARGE-SCALE figure, NOT localized mascons (those live at degree >= 50). For
# Artemis this field matters only briefly during the ~130 km powered flybys; the
# DRO sits ~70,000 km out where it is utterly negligible. (No mascon landing
# proxy here — Artemis I does not land.)
J2_MOON   = 2.0323e-4
C22_MOON  = 2.2382e-5
R_MOON_GRAV = 1.738e6   # reference radius matching the coefficient convention
LUNAR_SH_DEGREE = 8     # GRGM1200A truncation (unnormalized recursion safe to ~25)


def _build_lunar_sh_terms(nmax):
    """Denormalize the embedded GRGM1200A 4-pi-normalized coefficients to degree
    nmax and flatten to [(n, m, C, S, f, g)] for the Cunningham recursion.
    Asserts degree-2 consistency with the sourced constants."""
    import math as _m
    from lunar_gravity_coeffs import C_NORM, S_NORM
    terms = []
    for n in range(2, nmax + 1):
        for m in range(0, n + 1):
            Nnm = _m.sqrt((2.0 if m else 1.0) * (2 * n + 1)
                          * _m.factorial(n - m) / _m.factorial(n + m))
            Cn = Nnm * C_NORM[n][m]
            Sn = Nnm * S_NORM[n][m]
            if abs(Cn) < 1e-15 and abs(Sn) < 1e-15:
                continue
            terms.append((n, m, Cn, Sn,
                          float((n - m + 2) * (n - m + 1)), float(n - m + 1)))
    _c20 = next(t for t in terms if t[0] == 2 and t[1] == 0)
    _c22 = next(t for t in terms if t[0] == 2 and t[1] == 2)
    assert abs(-_c20[2] - J2_MOON) < 0.01 * J2_MOON, "SH table C20 vs J2_MOON mismatch"
    assert abs(_c22[2] - C22_MOON) < 0.01 * C22_MOON, "SH table C22 vs C22_MOON mismatch"
    return terms


_LUNAR_SH_TERMS = _build_lunar_sh_terms(LUNAR_SH_DEGREE)
_LUNAR_SH_NMAX = LUNAR_SH_DEGREE


def _lunar_sh_accel_body(x, y, z):
    """Perturbing acceleration (m/s^2, body frame) of the truncated lunar field
    via the Cunningham V/W recursion on UNNORMALIZED coefficients — Cartesian
    in/out, singularity-free at the poles. Sums n>=2 only."""
    import math as _m
    R = R_MOON_GRAV
    r2 = x * x + y * y + z * z
    rho = R * R / r2
    x0 = R * x / r2
    y0 = R * y / r2
    z0 = R * z / r2
    size = _LUNAR_SH_NMAX + 2          # degree-n accel needs V/W at n+1
    V = [[0.0] * size for _ in range(size)]
    W = [[0.0] * size for _ in range(size)]
    V[0][0] = R / _m.sqrt(r2)
    for m in range(size):
        if m > 0:
            V[m][m] = (2 * m - 1) * (x0 * V[m - 1][m - 1] - y0 * W[m - 1][m - 1])
            W[m][m] = (2 * m - 1) * (x0 * W[m - 1][m - 1] + y0 * V[m - 1][m - 1])
        if m + 1 < size:
            V[m + 1][m] = (2 * m + 1) * z0 * V[m][m]
            W[m + 1][m] = (2 * m + 1) * z0 * W[m][m]
        for n in range(m + 2, size):
            V[n][m] = ((2 * n - 1) * z0 * V[n - 1][m]
                       - (n + m - 1) * rho * V[n - 2][m]) / (n - m)
            W[n][m] = ((2 * n - 1) * z0 * W[n - 1][m]
                       - (n + m - 1) * rho * W[n - 2][m]) / (n - m)
    ax = ay = az = 0.0
    for n, m, Cn, Sn, f, g in _LUNAR_SH_TERMS:
        Vn = V[n + 1]
        Wn = W[n + 1]
        if m == 0:
            ax += -Cn * Vn[1]
            ay += -Cn * Wn[1]
        else:
            ax += 0.5 * ((-Cn * Vn[m + 1] - Sn * Wn[m + 1])
                         + f * (Cn * Vn[m - 1] + Sn * Wn[m - 1]))
            ay += 0.5 * ((-Cn * Wn[m + 1] + Sn * Vn[m + 1])
                         + f * (-Cn * Wn[m - 1] + Sn * Vn[m - 1]))
        az += g * (-Cn * Vn[m] - Sn * Wn[m])
    k = MU_MOON / (R * R)
    return k * ax, k * ay, k * az


def _Rz(a):
    c, s = np.cos(a), np.sin(a); return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
def _Ry(a):
    c, s = np.cos(a), np.sin(a); return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])


def _precession_j2000_to_date(jd):
    """IAU equatorial precession matrix mapping a J2000 vector to mean-equinox-of-date
    (the sim's ECI). Used to express the J2000-referenced IAU lunar frame in sim coords."""
    T = (jd - 2451545.0) / 36525.0; asec = np.pi / 180.0 / 3600.0
    zeta = (2306.2181 * T + 0.30188 * T * T + 0.017998 * T ** 3) * asec
    z = (2306.2181 * T + 1.09468 * T * T + 0.018203 * T ** 3) * asec
    th = (2004.3109 * T - 0.42665 * T * T - 0.041833 * T ** 3) * asec
    return _Rz(-z) @ _Ry(th) @ _Rz(-zeta)


# Precession at the launch epoch — constant to arcsec over the ~25 d mission (negligible vs the
# ~6.7 deg libration it complements), so computed once rather than per acceleration call.
_PREC_J2000_TO_DATE = _precession_j2000_to_date(JD_LAUNCH) if ENABLE_REAL_EPHEMERIS else np.eye(3)


def _lunar_pole_W(jd):
    """IAU 2009 lunar rotation — the FULL WGCCRE-2009 series: pole RA a0 (8 terms), Dec d0 (9), and
    prime-meridian W (14), with all 13 libration arguments E1..E13 (radians). Captures the physical
    librations (the sub-Earth point's ±6.7 deg wobble) the tidal-lock frame drops. This IS the complete
    analytic model — sub-arcmin accuracy is its inherent ceiling (going tighter needs DE lunar-libration
    ephemeris, unavailable offline), NOT a truncation. Validated vs HORIZONS."""
    d = jd - 2451545.0; T = d / 36525.0
    E = np.radians(np.array([
        125.045 - 0.0529921 * d, 250.089 - 0.1059842 * d, 260.008 + 13.0120009 * d,
        176.625 + 13.3407154 * d, 357.529 + 0.9856003 * d, 311.589 + 26.4057084 * d,
        134.963 + 13.0649930 * d, 276.617 + 0.3287146 * d, 34.226 + 1.7484877 * d,
        15.134 - 0.1589763 * d, 119.743 + 0.0036096 * d, 239.961 + 0.1643573 * d,
        25.053 + 12.9590088 * d]))
    a0 = (269.9949 + 0.0031 * T - 3.8787 * np.sin(E[0]) - 0.1204 * np.sin(E[1]) + 0.0700 * np.sin(E[2])
          - 0.0172 * np.sin(E[3]) + 0.0072 * np.sin(E[5]) - 0.0052 * np.sin(E[9]) + 0.0043 * np.sin(E[12]))
    d0 = (66.5392 + 0.0130 * T + 1.5419 * np.cos(E[0]) + 0.0239 * np.cos(E[1]) - 0.0278 * np.cos(E[2])
          + 0.0068 * np.cos(E[3]) - 0.0029 * np.cos(E[5]) + 0.0009 * np.cos(E[6]) + 0.0008 * np.cos(E[9])
          - 0.0009 * np.cos(E[12]))
    W = (38.3213 + 13.17635815 * d - 1.4e-12 * d * d + 3.5610 * np.sin(E[0]) + 0.1208 * np.sin(E[1])
         - 0.0642 * np.sin(E[2]) + 0.0158 * np.sin(E[3]) + 0.0252 * np.sin(E[4]) - 0.0066 * np.sin(E[5])
         - 0.0047 * np.sin(E[6]) - 0.0046 * np.sin(E[7]) + 0.0028 * np.sin(E[8]) + 0.0052 * np.sin(E[9])
         + 0.0040 * np.sin(E[10]) + 0.0019 * np.sin(E[11]) - 0.0044 * np.sin(E[12]))
    return np.radians(a0), np.radians(d0), np.radians(W % 360.0)


def _moon_fixed_axes_ofdate(t):
    """IAU Moon-fixed axes (prime-meridian x, in-plane y, pole z) expressed in ECI-of-date.
    True pole + prime meridian (physical librations + obliquity) — the fidelity replacement for
    the tidal-lock construction (x = exact Moon->Earth, z = orbit normal)."""
    a0, d0, W = _lunar_pole_W(_jde_tt(t))
    zt = np.array([np.cos(d0) * np.cos(a0), np.cos(d0) * np.sin(a0), np.sin(d0)])  # pole (J2000)
    node = np.array([-np.sin(a0), np.cos(a0), 0.0])                                # eq ascending node
    xt = node * np.cos(W) + np.cross(zt, node) * np.sin(W)                          # prime meridian
    yt = np.cross(zt, xt)
    P = _PREC_J2000_TO_DATE                                                         # J2000 -> of-date
    return P @ xt, P @ yt, P @ zt


def lunar_nonspherical_accel(p, t):
    """Perturbing acceleration (m/s^2) from the Moon's figure at Moon-relative
    position `p` (ECI-aligned, Moon-centered), at time t. Returns ONLY the
    perturbation on top of the -MU_MOON*p/|p|^3 central term.

    Body frame: the IAU-2009 Moon-fixed frame (true pole + prime meridian, incl.
    librations) when ENABLE_LUNAR_LIBRATION, else the tidal-lock approximation
    (spin axis ~ orbit normal, x-axis ~ Moon->Earth; obliquity ~1.5 deg and
    librations ±6.7 deg neglected — adequate for brief flybys, not mascon-grade).
    """
    if not globals().get("ENABLE_LUNAR_HARMONICS", False):
        return np.zeros(3)
    p_dist = np.dot(p, p)              # squared distance (avoid sqrt)
    if p_dist > 5.0e7**2:              # > 50,000 km from Moon center -> negligible
        return np.zeros(3)
    if globals().get("ENABLE_LUNAR_LIBRATION", False):
        x_b, y_b, z_b = _moon_fixed_axes_ofdate(t)            # IAU true pole + prime meridian
    else:
        mr, mv = moon_state(t)
        z_b = np.cross(mr, mv); z_b = z_b / np.linalg.norm(z_b)   # spin axis ~ orbit normal
        earth_dir = -mr                                            # Moon -> Earth
        x_b = earth_dir - np.dot(earth_dir, z_b) * z_b
        nx = np.linalg.norm(x_b)
        if nx < 1e-6:
            return np.zeros(3)
        x_b = x_b / nx
        y_b = np.cross(z_b, x_b)
    x = np.dot(p, x_b); y = np.dot(p, y_b); z = np.dot(p, z_b)
    r = np.sqrt(x*x + y*y + z*z)
    if r < 1.0:
        return np.zeros(3)
    # Degree-N GRAIL field within ~3,500 km of the Moon; else degree-2 closed form.
    if globals().get("ENABLE_LUNAR_SH_FIELD", False) and r < 3.5e6:
        ax, ay, az = _lunar_sh_accel_body(x, y, z)
        return ax * x_b + ay * y_b + az * z_b
    mu = MU_MOON; R = R_MOON_GRAV
    z2r2 = (z/r)**2
    fz = 1.5 * J2_MOON * mu * R**2 / r**5
    a_c20 = fz * np.array([x*(5*z2r2-1), y*(5*z2r2-1), z*(5*z2r2-3)])
    K = 3.0 * mu * R**2 * C22_MOON
    r5 = r**5; r7 = r**7; d = (x*x - y*y)
    a_c22 = K * np.array([2*x/r5 - 5*x*d/r7, -2*y/r5 - 5*y*d/r7, -5*z*d/r7])
    a_body = a_c20 + a_c22
    return a_body[0]*x_b + a_body[1]*y_b + a_body[2]*z_b


_R_AIR = 287.053         # specific gas constant for dry air [J/(kg K)]
_GAMMA_AIR = 1.4


def _ussa76_low(h):
    """US Standard Atmosphere 1976 (density kg/m^3, temperature K) for 0-32 km — the
    ascent / max-q / drag-relevant band. Standard barometric layers (geopotential ≈
    geometric below 32 km, <0.5% error). Replaces the legacy single-exponential, which
    was ~22% thin near 10 km (0.32 vs USSA-76 0.41) and pulled the modeled max-q low.
    Connects smoothly to the legacy >=25 km segments (USSA-76 0.0395 vs 0.040 at 25 km)."""
    if h < 11_000.0:                         # troposphere, lapse -6.5 K/km
        T = 288.15 - 0.0065 * h
        P = 101_325.0 * (T / 288.15) ** 5.25577
    elif h < 20_000.0:                       # tropopause, isothermal 216.65 K
        T = 216.65
        P = 22_632.06 * np.exp(-(h - 11_000.0) / 6_341.62)
    else:                                    # lower stratosphere (20-32 km), lapse +1.0 K/km
        T = 216.65 + 0.001 * (h - 20_000.0)
        P = 5_474.89 * (T / 216.65) ** (-34.1626)
    return P / (_R_AIR * T), T


def _cd_mach(M):
    """Mach-dependent ascent drag coefficient (referenced to SLS_AREA): subsonic ~0.30,
    transonic peak ~0.55 near M~1.1, supersonic decay to ~0.20. A generic launch-stack
    drag curve — Mach-resolved (vs the legacy constant SLS_CD), no SLS aero DB published."""
    return float(np.interp(M, [0.0, 0.6, 0.8, 1.0, 1.1, 1.3, 2.0, 3.0, 5.0, 12.0],
                              [0.30, 0.30, 0.38, 0.52, 0.55, 0.48, 0.32, 0.25, 0.21, 0.20]))


def atm_temperature(altitude_m):
    """USSA-76 temperature (K) for 0-32 km (for the ascent Mach number); ~constant above."""
    if altitude_m < 32_000.0:
        return _ussa76_low(max(0.0, altitude_m))[1]
    return 228.65


def atm_density(altitude_m):
    """Atmospheric density (kg/m^3). Below 25 km: USSA-76 layered model when
    ENABLE_USSA76_ATM (default), else the legacy single-exponential. >=25 km: the
    exponential segments (unchanged — entry corridor was validated against these).
    Covers the Orion skip-entry corridor (entry interface ~122 km)."""
    if altitude_m < 0:    return 1.225
    if altitude_m < 25_000:
        if globals().get("ENABLE_USSA76_ATM", True):
            return _ussa76_low(altitude_m)[0]
        return 1.225 * np.exp(-altitude_m / 7500.0)
    if altitude_m < 100_000:
        return 0.040 * np.exp(-(altitude_m - 25_000) / 7100.0)
    if altitude_m < 200_000:
        return 1e-6 * np.exp(-(altitude_m - 100_000) / 45_000.0)
    return 0.0


def eci_to_latlon(r, t):
    """ECI position -> (lat, lon) in degrees at time t."""
    theta = OMEGA_E * t + _GMST0
    x = np.cos(theta)*r[0] + np.sin(theta)*r[1]
    y = -np.sin(theta)*r[0] + np.cos(theta)*r[1]
    z = r[2]
    rn = np.sqrt(x*x + y*y + z*z)
    return np.rad2deg(np.arcsin(z/rn)), np.rad2deg(np.arctan2(y, x))


def latlon_alt_to_eci(lat_deg, lon_deg, alt, t):
    """Lat/lon/alt -> ECI position at time t."""
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    theta = OMEGA_E * t + _GMST0
    R = R_EARTH + alt
    xe = R*np.cos(lat)*np.cos(lon)
    ye = R*np.cos(lat)*np.sin(lon)
    ze = R*np.sin(lat)
    x =  np.cos(theta)*xe - np.sin(theta)*ye
    y =  np.sin(theta)*xe + np.cos(theta)*ye
    return np.array([x, y, ze])


# --- Lambert solver (universal variables, Bate-Mueller-White / Vallado) -------
def _stumpff(psi):
    """Stumpff functions c2, c3 for universal variable psi."""
    if psi > 1e-6:
        s = np.sqrt(psi)
        return (1 - np.cos(s))/psi, (s - np.sin(s))/(s*s*s)
    if psi < -1e-6:
        s = np.sqrt(-psi)
        return (1 - np.cosh(s))/psi, (np.sinh(s) - s)/(s*s*s)
    return 0.5 - psi/24.0 + psi*psi/720.0, 1/6.0 - psi/120.0 + psi*psi/5040.0


def lambert_uv(r1, r2, tof, mu=MU_EARTH, prograde=True):
    """Universal-variable Lambert solver. Returns (v1, v2) for a two-body
    transfer from r1 (t=0) to r2 (t=tof) under gravity parameter mu, or None."""
    r1 = np.asarray(r1, float)
    r2 = np.asarray(r2, float)
    r1n = np.linalg.norm(r1)
    r2n = np.linalg.norm(r2)
    cos_dnu = float(np.dot(r1, r2) / (r1n * r2n))
    cos_dnu = max(-1.0, min(1.0, cos_dnu))
    z_cross = float(r1[0]*r2[1] - r1[1]*r2[0])
    if prograde:
        dm = 1.0 if z_cross >= 0 else -1.0
    else:
        dm = -1.0 if z_cross >= 0 else 1.0
    A = dm * np.sqrt(r1n * r2n * (1.0 + cos_dnu))
    if abs(A) < 1e-6:
        return None
    psi_low, psi_up = -4.0*np.pi, 4.0*np.pi*np.pi
    psi = 0.0
    c2, c3 = 0.5, 1.0/6.0
    last_y = None
    for _ in range(200):
        y = r1n + r2n + A*(psi*c3 - 1.0)/np.sqrt(c2)
        if A > 0 and y < 0:
            tries = 0
            while y < 0 and tries < 50:
                psi_low = psi
                psi = 0.5*(psi + psi_up)
                c2, c3 = _stumpff(psi)
                y = r1n + r2n + A*(psi*c3 - 1.0)/np.sqrt(c2)
                tries += 1
            if y < 0:
                return None
        if c2 <= 0:
            psi = 0.5*(psi_low + psi)
            c2, c3 = _stumpff(psi)
            continue
        chi = np.sqrt(y / c2)
        t_iter = (chi**3 * c3 + A * np.sqrt(y)) / np.sqrt(mu)
        if abs(t_iter - tof) < 1e-4:
            last_y = y
            break
        if t_iter < tof:
            psi_low = psi
        else:
            psi_up = psi
        psi = 0.5*(psi_low + psi_up)
        c2, c3 = _stumpff(psi)
        last_y = y
    if last_y is None or last_y <= 0:
        return None
    f = 1.0 - last_y/r1n
    g = A * np.sqrt(last_y/mu)
    gdot = 1.0 - last_y/r2n
    if abs(g) < 1e-9:
        return None
    v1 = (r2 - f*r1) / g
    v2 = (gdot*r2 - r1) / g
    return v1, v2


# ============================================================
# CR3BP Distant Retrograde Orbit (Earth-Moon)  — foundation for the return
# ============================================================
# A real differentially-corrected planar DRO periodic orbit (NOT the tuned
# DRO_INSERT_SPEED proxy used by phase_dro_insertion). It is REQUIRED for the
# RETURN: only the correct DRO phasing relative to the Earth-Moon line lets a
# small DDP + a ~130 km RPF flyby fling Orion back to Earth. VERIFIED:
# from this DRO at the right departure phase, DDP ~156 m/s + an RPF escape reaches
# the Earth-entry corridor (perigee below the surface); other phases do not — the
# return is phase-sensitive exactly like the real mission. Period ~11.95 d (Artemis
# flew ~half a rev). Wired into DRI and the DDP/RPF return targeting via dro_state_eci
# (the snap-free OPF->DRO rendezvous + the flown return).
_CR3BP_MU = MU_MOON / (MU_EARTH + MU_MOON)
_EM_MEAN_MOTION = np.sqrt((MU_EARTH + MU_MOON) / EM_DIST**3)
_CR3BP_DRO = None


def _compute_cr3bp_dro():
    """Compute & cache the planar Earth-Moon CR3BP DRO periodic orbit (rotating
    frame, normalized). Returns dict(period, sol, vy0); sol.sol(tau) -> [x,y,vx,vy].
    Differential correction: secant on the initial vy until the next x-axis (y=0)
    crossing is perpendicular (vx=0), which makes the orbit periodic by symmetry."""
    global _CR3BP_DRO
    if _CR3BP_DRO is not None:
        return _CR3BP_DRO
    mu = _CR3BP_MU
    _rad = (globals().get("NEAR_POLAR_DRO_RADIUS_KM", DRO_RADIUS_KM)
            if (globals().get("ENABLE_NEAR_POLAR_FLOWN", False)
                or globals().get("ENABLE_SNAPFREE_DRO", False)) else DRO_RADIUS_KM)
    x0 = (1 - mu) + _rad * 1000.0 / EM_DIST              # far-side x-axis crossing
    def eom(tau, st):
        x, y, vx, vy = st
        r1 = np.hypot(x + mu, y); r2 = np.hypot(x - 1 + mu, y)
        return [vx, vy,
                2*vy + x - (1-mu)*(x+mu)/r1**3 - mu*(x-1+mu)/r2**3,
                -2*vx + y - (1-mu)*y/r1**3 - mu*y/r2**3]
    def half(vy0):
        def ev(tau, st): return st[1]
        ev.terminal = True; ev.direction = 1
        so = solve_ivp(eom, (1e-3, 6.0), [x0, 0.0, 0.0, vy0], events=ev,
                       rtol=1e-11, atol=1e-12)
        if not len(so.t_events[0]):
            return None, None
        return so.y_events[0][0][2], so.t_events[0][0]
    vy0 = -0.5; f0, _ = half(vy0); vy1 = vy0 * 1.01; f1, thp = half(vy1)
    for _ in range(40):
        if f1 is None or abs(f1 - f0) < 1e-14:
            break
        v2 = vy1 - f1 * (vy1 - vy0) / (f1 - f0)
        vy0, f0 = vy1, f1; vy1 = v2; f1, thp = half(vy1)
        if f1 is not None and abs(f1) < 1e-11:
            break
    period = 2.0 * thp
    sol = solve_ivp(eom, (0.0, period), [x0, 0.0, 0.0, vy1],
                    rtol=1e-11, atol=1e-12, dense_output=True)
    _CR3BP_DRO = {"period": period, "sol": sol, "vy0": vy1}
    return _CR3BP_DRO


_OEM_DRO_REF = None
def _oem_dro_ref():
    """Load + cache the AS-FLOWN OEM DRO trajectory, precession-corrected from J2000 to the sim's
    mean-equinox-of-date ECI, as time-indexed r(t)/v(t) [SI]. This is the real flown DRO used as the
    reference when ENABLE_OEM_DRO_REF (fixes the CR3BP->elliptical mapping residuals)."""
    global _OEM_DRO_REF
    if _OEM_DRO_REF is not None:
        return _OEM_DRO_REF
    from datetime import datetime
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data/artemis1_ephemeris/Post_TLI_Orion_AsFlown_20221213_EPH_OEM.asc")
    launch = datetime(2022, 11, 16, 6, 47, 44)
    g, R, V = [], [], []
    for ln in open(path):
        p = ln.split()
        if len(p) == 7 and p[0][:2] == "20" and "T" in p[0]:
            try:
                dt = datetime.strptime(p[0][:19], "%Y-%m-%dT%H:%M:%S")
                fr = float("0" + p[0][19:]) if "." in p[0][19:] else 0.0
                g.append((dt - launch).total_seconds() + fr)
                R.append([float(x) for x in p[1:4]]); V.append([float(x) for x in p[4:7]])
            except Exception:
                pass
    g = np.asarray(g); R = np.asarray(R) * 1000.0; V = np.asarray(V) * 1000.0     # km -> m
    P = _PREC_J2000_TO_DATE                                                        # J2000 -> mean-of-date
    R = (P @ R.T).T; V = (P @ V.T).T
    _OEM_DRO_REF = {"g": g, "R": R, "V": V}
    return _OEM_DRO_REF


def _oem_dro_state(t):
    """OEM DRO state (r_eci, v_eci) [SI, mean-of-date] at mission time t, linearly interpolated;
    None if t is outside the OEM span."""
    d = _oem_dro_ref(); g = d["g"]
    if t < g[0] or t > g[-1]:
        return None
    R, V = d["R"], d["V"]
    return (np.array([np.interp(t, g, R[:, k]) for k in range(3)]),
            np.array([np.interp(t, g, V[:, k]) for k in range(3)]))


def dro_state_eci(phase, t):
    """Earth-Moon DRO state (ECI [r(3), v(3)], SI) at orbit `phase` in [0,1) and
    mission time t. With ENABLE_OEM_DRO_REF, returns the AS-FLOWN OEM DRO state at time t
    (precession-corrected) when t is within the OEM span — the real flown reference DRO. Otherwise
    maps the cached CR3BP periodic orbit into the instantaneous Earth-Moon plane via the real lunar
    ephemeris (rotating frame -> ECI); the CR3BP-vs-elliptical mismatch is the residual the OEM
    anchor removes. (`phase` is ignored on the OEM path — the OEM state is time-determined.)"""
    if globals().get("ENABLE_OEM_DRO_REF", False):
        _oem = _oem_dro_state(t)
        if _oem is not None:
            return _oem[0], _oem[1]
    dro = _compute_cr3bp_dro()
    x, y, vx, vy = dro["sol"].sol((phase % 1.0) * dro["period"])
    mu = _CR3BP_MU
    rmo, vmo = moon_state(t)
    ex = rmo / np.linalg.norm(rmo)
    ez = np.cross(rmo, vmo); ez = ez / np.linalg.norm(ez)
    ey = np.cross(ez, ex)
    omega = np.cross(rmo, vmo) / np.dot(rmo, rmo)
    r_rel = EM_DIST * ((x - (1 - mu)) * ex + y * ey)
    r_eci = rmo + r_rel
    v_eci = vmo + EM_DIST * _EM_MEAN_MOTION * (vx * ex + vy * ey) + np.cross(omega, r_rel)
    return r_eci, v_eci


# ============================================================
# Mission phases  (Artemis I sequence)
# ============================================================
# Convention: each phase returns a results dict that includes
# at least {"success": bool, "failure_reason": str|None} plus phase-specific
# outcome fields, and (for the integrated phases) the propagated end state and,
# optionally, captured trajectory arrays. run_mission() stitches them together.

_CACHED_LAUNCH_AZIMUTH = None


def _solve_launch_azimuth():
    """Solve the launch azimuth (deg E of N) so the insertion-orbit plane
    CONTAINS the Moon's direction at the flyby epoch (OPF_GET_S). This is the
    lightweight launch-continuity solve: with the plane aligned, the TLI + OTC
    targeting toward the Moon needs only a small plane-change component, so the
    OTC ΔV stays within the ESM's tight budget. Cached (geometry is fixed by the
    launch epoch). The full continuity solve (co-designing launch TIME too, as
    NASA does) is the eventual refinement — here the real launch time is fixed.

    Insertion-plane normal as a function of azimuth a:
        n(a) = sin(a)*cross(r_hat0, east0) + cos(a)*cross(r_hat0, north0)
    Solve n(a)·moon_hat = 0  ->  tan(a) = -(B·m)/(A·m); pick the posigrade root."""
    global _CACHED_LAUNCH_AZIMUTH
    if _CACHED_LAUNCH_AZIMUTH is not None:
        return _CACHED_LAUNCH_AZIMUTH
    lat, lon = np.deg2rad(LAUNCH_LAT_DEG), np.deg2rad(LAUNCH_LON_DEG)
    theta0 = _GMST0
    xe = R_EARTH*np.cos(lat)*np.cos(lon)
    ye = R_EARTH*np.cos(lat)*np.sin(lon)
    ze = R_EARTH*np.sin(lat)
    r0 = np.array([np.cos(theta0)*xe - np.sin(theta0)*ye,
                   np.sin(theta0)*xe + np.cos(theta0)*ye, ze])
    r_hat0 = r0 / np.linalg.norm(r0)
    east0 = np.cross(np.array([0.0, 0.0, 1.0]), r_hat0); east0 /= np.linalg.norm(east0)
    north0 = np.cross(r_hat0, east0)
    A_vec = np.cross(r_hat0, east0)
    B_vec = np.cross(r_hat0, north0)
    moon_hat = moon_state(OPF_GET_S)[0]; moon_hat = moon_hat / np.linalg.norm(moon_hat)
    az = np.arctan2(-float(B_vec @ moon_hat), float(A_vec @ moon_hat))
    if np.sin(az) < 0:            # ensure a posigrade (easterly) heading
        az += np.pi
    _CACHED_LAUNCH_AZIMUTH = float(np.rad2deg(az) % 360.0)
    return _CACHED_LAUNCH_AZIMUTH


def phase_sls_launch(perturb=None, t_liftoff=0.0):
    """SLS Block 1 ascent: pad -> SRB+core boost -> SRB sep + LAS jettison ->
    RS-25 core burn to MECO (propellant depletion) -> core-stage separation,
    leaving Orion+ICPS in the insertion orbit.

    Finite-thrust integration with Earth(+J2)/Moon/Sun gravity, atmospheric drag,
    azimuth-plane steering, and the ENABLE_SLS_ASCENT_FAILURES engine-out modes.

    The in-atmosphere boost (liftoff → SRB sep) uses an open-loop gravity-turn pitch
    program; the exo-atmospheric core burn to MECO is flown by closed-loop IGM-style
    linear-tangent guidance (ENABLE_IGM_ASCENT) solved to the SOURCED 30×1806 km
    insertion (open-loop fallback on solve failure). The launch azimuth (continuity
    branch) aligns the insertion plane with the Moon at the flyby epoch. Returns
    success, failure_reason, state ([r,v,m] at core sep), t_insertion,
    insertion_perigee/apogee_km, meco_mode, igm_chi0_deg, launch_azimuth_deg,
    max_q_pa, max_g, peak_alt_km, trajectory_t/y."""
    perturb = perturb or {}
    result = {"success": True, "failure_reason": None, "engine_failures": [],
              "max_q_pa": 0.0, "max_g": 0.0, "peak_alt_km": 0.0}

    # --- failure-mode draws (consumed only when the flag is on) -------------
    sls_fail = globals().get("ENABLE_SLS_ASCENT_FAILURES", False)
    n_srb_fail  = int(perturb.get("n_srb_failures", 0)) if sls_fail else 0
    n_rs25_fail = int(perturb.get("n_rs25_failures", 0)) if sls_fail else 0
    rs25_fail_time = float(perturb.get("rs25_failure_time_s", 1e9))
    # A five-segment SRB cannot be shut down or throttled; a booster failure is
    # an unsurvivable asymmetric-thrust / case-breach event at liftoff.
    if n_srb_fail > 0:
        result["success"] = False
        result["failure_reason"] = "srb_failure"
        result["engine_failures"].append(("SRB", 0.0))
        return result

    # --- engine dispersions -------------------------------------------------
    rs25_isp_f = perturb.get("rs25_isp_factor", 1.0)
    rs25_thr_f = perturb.get("rs25_thrust_factor", 1.0)
    srb_thr_f  = perturb.get("srb_thrust_factor", 1.0)

    # --- pad initial state (LC-39B) ----------------------------------------
    lat, lon = np.deg2rad(LAUNCH_LAT_DEG), np.deg2rad(LAUNCH_LON_DEG)
    theta0 = OMEGA_E * t_liftoff + _GMST0
    xe = R_EARTH*np.cos(lat)*np.cos(lon)
    ye = R_EARTH*np.cos(lat)*np.sin(lon)
    ze = R_EARTH*np.sin(lat)
    r0 = np.array([np.cos(theta0)*xe - np.sin(theta0)*ye,
                   np.sin(theta0)*xe + np.cos(theta0)*ye, ze])
    v0 = np.cross(np.array([0.0, 0.0, OMEGA_E]), r0)   # Earth-surface velocity at pad
    state = np.concatenate([r0, v0, [SLS_LIFTOFF_MASS]])
    t = t_liftoff

    # --- target plane from the launch azimuth (steer downrange in-plane) ----
    r_hat0 = r0 / np.linalg.norm(r0)
    east0 = np.cross(np.array([0.0, 0.0, 1.0]), r_hat0); east0 /= np.linalg.norm(east0)
    north0 = np.cross(r_hat0, east0)
    # Launch azimuth: solved so the insertion plane contains the Moon at flyby
    # (launch continuity); the LAUNCH_AZIMUTH_DEG constant is the flag-off fallback.
    az_deg = _solve_launch_azimuth() if globals().get("ENABLE_LAUNCH_CONTINUITY", False) else LAUNCH_AZIMUTH_DEG
    if globals().get("ENABLE_PHASEC_BPLANE", False) and globals().get("PHASEC_LAUNCH_AZIMUTH_DEG") is not None:
        az_deg = float(globals()["PHASEC_LAUNCH_AZIMUTH_DEG"])   # Stage-1: parking plane = real post-TLI plane
    az = np.deg2rad(az_deg)
    result["launch_azimuth_deg"] = az_deg
    head0 = np.sin(az)*east0 + np.cos(az)*north0
    n_target = np.cross(r_hat0, head0); n_target /= np.linalg.norm(n_target)
    def downrange(r_hat):
        d = np.cross(n_target, r_hat); n = np.linalg.norm(d)
        return d / n if n > 1e-9 else head0

    # --- pitch program (open-loop gravity turn; angle from local vertical) --
    # Overridable for stage-1 ascent calibration (defaults = the sourced constants).
    _p_vert = float(globals().get("_ASCENT_VERT_OVERRIDE", ASCENT_VERT_S))
    _p_t    = float(globals().get("_ASCENT_PITCH_T_OVERRIDE", ASCENT_PITCH_T))
    _p_end  = float(globals().get("_ASCENT_PITCH_END_OVERRIDE", ASCENT_PITCH_END_DEG))
    def pitch_rad(dt):
        if dt < _p_vert:
            return 0.0
        if dt < _p_t:
            frac = (dt - _p_vert) / (_p_t - _p_vert)
            return np.deg2rad(frac * _p_end)
        return np.deg2rad(_p_end)

    # --- TRUE gravity-turn thrust direction (ENABLE_GRAVITY_TURN) ------------
    # Vertical rise -> a brief pitch KICK off vertical -> thrust ALONG the relative
    # velocity (zero AoA); gravity then turns the trajectory. Single knob = the kick.
    _gt_on   = globals().get("ENABLE_GRAVITY_TURN", False)
    _gt_kick = float(globals().get("_GT_KICK_DEG_OVERRIDE", GT_KICK_DEG))
    _gt_kdur = float(globals().get("_GT_KICK_DUR_OVERRIDE", GT_KICK_DUR_S))
    def gt_dir(dt, r_hat, v_rel):
        if dt < _p_vert:                                   # vertical rise
            return r_hat
        if dt < _p_vert + _gt_kdur:                        # pitch kick: hold a fixed tilt off vertical
            k = np.deg2rad(_gt_kick)
            return np.cos(k) * r_hat + np.sin(k) * downrange(r_hat)
        vn = np.linalg.norm(v_rel)                         # gravity turn: thrust ‖ relative velocity
        return v_rel / vn if vn > 1.0 else r_hat

    # --- RS-25 throttle: max-Q bucket + axial-acceleration limiter (calibratable) --
    _alim_g = float(globals().get("_RS25_ACCEL_LIMIT_OVERRIDE", RS25_ACCEL_LIMIT_G))
    _bk_pct = float(globals().get("_RS25_BUCKET_PCT_OVERRIDE", RS25_MAXQ_BUCKET_PCT))
    _bk_t0  = float(globals().get("_RS25_BUCKET_T0_OVERRIDE", RS25_MAXQ_T0_S))
    _bk_t1  = float(globals().get("_RS25_BUCKET_T1_OVERRIDE", RS25_MAXQ_T1_S))
    def _rs25_bucket(dt, ramp=4.0):              # max-Q throttle bucket (trapezoid, smooth)
        if dt <= _bk_t0 - ramp or dt >= _bk_t1 + ramp:
            return 1.0
        if dt < _bk_t0:
            return 1.0 - (1.0 - _bk_pct) * (dt - (_bk_t0 - ramp)) / ramp
        if dt <= _bk_t1:
            return _bk_pct
        return _bk_pct + (1.0 - _bk_pct) * (dt - _bk_t1) / ramp
    def _rs25_throttled(T_core, m):              # apply axial-accel limiter (<= _alim_g)
        lim = _alim_g * G0 * max(m, 1.0)
        return min(T_core, lim)

    # --- RS-25 thrust / Isp vary with altitude (SL -> vac over 0..80 km) ----
    def rs25_thrust(alt):
        f = min(max(alt/80_000.0, 0.0), 1.0)
        return RS25_THRUST_SL_N + f*(RS25_THRUST_VAC_N - RS25_THRUST_SL_N)
    def rs25_isp(alt):
        f = min(max(alt/80_000.0, 0.0), 1.0)
        return RS25_ISP_SL_S + f*(RS25_ISP_VAC_S - RS25_ISP_SL_S)
    def n_rs25_live(dt):
        return max(0, RS25_COUNT - (n_rs25_fail if dt >= rs25_fail_time else 0))

    srb_mdot_each = SRB_PROP_EACH_KG / SRB_BURN_TIME_S   # deplete over burn time

    def accel(t_now, y, boosting):
        r = y[:3]; v = y[3:6]; m = y[6]
        rn = np.linalg.norm(r); alt = rn - R_EARTH
        result["peak_alt_km"] = max(result["peak_alt_km"], alt/1000.0)
        a_grav = gravity_earth_moon(r, t_now)
        r_hat = r / rn
        dt = t_now - t_liftoff
        ne = n_rs25_live(dt)
        T = ne * rs25_thrust(alt) * rs25_thr_f * _rs25_bucket(dt)   # max-Q throttle bucket
        T = _rs25_throttled(T, m)                                   # 5 g axial-accel limiter
        isp = rs25_isp(alt) * rs25_isp_f
        mdot = T / (isp * G0) if T > 0 else 0.0
        if boosting and dt < SRB_BURN_TIME_S:                       # SRBs thrust to web burnout
            T += 2 * SRB_THRUST_SL_N * srb_thr_f
            mdot += 2 * srb_mdot_each
        # Relative-wind velocity (needed by the gravity turn AND drag)
        v_air = np.cross(np.array([0.0, 0.0, OMEGA_E]), r)
        v_rel = v - v_air; vr = np.linalg.norm(v_rel)
        # Thrust direction: TRUE gravity turn (thrust ‖ relative velocity, zero AoA) or legacy time-ramp pitch
        if _gt_on:
            tdir = gt_dir(dt, r_hat, v_rel)
        else:
            pr = pitch_rad(dt)
            tdir = np.cos(pr) * r_hat + np.sin(pr) * downrange(r_hat)
        a_thrust = T * tdir / max(m, 1.0)
        # Atmospheric drag (relative to the rotating atmosphere)
        rho = atm_density(alt); q = 0.5 * rho * vr * vr
        result["max_q_pa"] = max(result["max_q_pa"], q)
        if vr > 1.0 and rho > 1e-9:
            if globals().get("ENABLE_USSA76_ATM", True):    # Mach-dependent ascent Cd
                cd = _cd_mach(vr / np.sqrt(_GAMMA_AIR * _R_AIR * atm_temperature(alt)))
            else:
                cd = SLS_CD                                  # legacy constant
            a_drag = -q * cd * SLS_AREA * v_rel / (vr * max(m, 1.0))
        else:
            a_drag = np.zeros(3)
        a = a_grav + a_thrust + a_drag
        result["max_g"] = max(result["max_g"], np.linalg.norm(a_thrust + a_drag) / G0)
        return np.concatenate([v, a, [-mdot]])

    # --- boost phase: SRB + core, 0 -> SRB_SEP_TIME_S -----------------------
    # SRBs thrust to web burnout (SRB_BURN_TIME_S); the stack then flies ~6 s on the
    # core alone with the spent (dead-mass) boosters still attached, jettisoned at sep.
    try:
        sol_b = solve_ivp(lambda tt, y: accel(tt, y, True),
                          (t, t + SRB_SEP_TIME_S), state, method="RK45",
                          rtol=1e-7, atol=1e-1, max_step=2.0, dense_output=True)
    except Exception as e:
        result.update(success=False, failure_reason=f"boost_integration_error: {e}")
        return result
    # core propellant remaining at SRB sep = total - (boost mass drop - SRB prop)
    core_burned = (SLS_LIFTOFF_MASS - sol_b.y[6, -1]) - 2 * SRB_PROP_EACH_KG
    core_remaining = CORE_PROP_KG - core_burned
    state = sol_b.y[:, -1].copy(); t = sol_b.t[-1]
    _rsep = np.linalg.norm(state[:3]); _vsep = np.linalg.norm(state[3:6])     # SRB-sep diagnostics
    result["srb_sep_alt_km"] = float((_rsep - R_EARTH) / 1000.0)
    result["srb_sep_speed_ms"] = float(_vsep)
    result["srb_sep_fpa_deg"] = float(np.degrees(np.arcsin(np.dot(state[:3], state[3:6]) / (_rsep * _vsep))))
    # SRB separation + LAS jettison (v1: LAS dropped at SRB sep, ~70 s early)
    state[6] -= 2 * SRB_DRY_EACH_KG + LAS_MASS_KG

    if np.linalg.norm(state[:3]) - R_EARTH < 1000.0:
        result.update(success=False, failure_reason="ascent_underperformance_crash",
                      state=state, t_insertion=t)
        return result
    if result["max_q_pa"] > MAXQ_STRUCTURAL_PA:
        result.update(success=False, failure_reason="structural_failure_max_q",
                      state=state, t_insertion=t)
        return result
    if core_remaining <= 0:
        result.update(success=False, failure_reason="core_propellant_starve_at_sep",
                      state=state, t_insertion=t)
        return result

    # --- core phase: RS-25 only, to MECO -----------------------------------
    # Closed-loop IGM-style guidance (ENABLE_IGM_ASCENT): Lawden linear-tangent
    # steering solved so the exo-atmospheric core burn delivers the SOURCED
    # 30×1806 km insertion (perigee INSERTION_PERIGEE_KM, apogee INSERTION_APOGEE_KM),
    # cutting off at the target apogee (guidance-commanded MECO) at the target
    # core-alone burn time (~CORE_BURN_TIME_S − SRB_BURN_TIME_S). Like the classic
    # S-IVB IGM: a 2-var fsolve on (linear-tangent χ₀, tangent rate) with an
    # open-loop-pitch fallback when the solve fails (engine-out / severe dispersion).
    # Propellant depletion is the backstop on both paths; the orbit check below
    # catches any underperformance.
    meco_mass = state[6] - core_remaining
    target_apo_r = R_EARTH + INSERTION_APOGEE_KM * 1000.0
    target_peri_r = R_EARTH + INSERTION_PERIGEE_KM * 1000.0
    t_core0 = t
    state_core0 = state.copy()

    def ev_apogee(t_now, y):
        r = y[:3]; v = y[3:6]; rn = np.linalg.norm(r)
        E = 0.5 * float(np.dot(v, v)) - MU_EARTH / rn
        if E >= 0:
            return -1.0                      # hyperbolic: don't cut here
        a = -MU_EARTH / (2 * E)
        h = np.linalg.norm(np.cross(r, v))
        ecc = np.sqrt(max(0.0, 1 - (h*h) / (MU_EARTH * a)))
        return a * (1 + ecc) - target_apo_r
    ev_apogee.terminal = True; ev_apogee.direction = 1
    def ev_depletion(t_now, y):
        return y[6] - meco_mass
    ev_depletion.terminal = True; ev_depletion.direction = -1

    def _apsides(y):
        """(perigee_r, apogee_r) of the osculating orbit, or None if hyperbolic."""
        r = y[:3]; v = y[3:6]; rn = np.linalg.norm(r)
        E = 0.5 * float(np.dot(v, v)) - MU_EARTH / rn
        if E >= 0:
            return None
        a = -MU_EARTH / (2 * E)
        h = np.linalg.norm(np.cross(r, v))
        ecc = np.sqrt(max(0.0, 1 - (h*h) / (MU_EARTH * a)))
        return a * (1 - ecc), a * (1 + ecc)

    # Linear-tangent fly: χ measured from the in-plane downrange direction,
    # thrust = cos(χ)·downrange + sin(χ)·r_hat with tan(χ)=tan0+rate·(t−t_core0).
    # Integrates with gravity + (residual) drag + altitude-varying RS-25; cut at
    # the target apogee (else propellant depletion).
    def _fly_lt(chi0_deg, tan_rate):
        tan0 = np.tan(np.deg2rad(chi0_deg))
        def rhs_lt(tt, y):
            r = y[:3]; v = y[3:6]; m = y[6]; rn = np.linalg.norm(r)
            alt = rn - R_EARTH
            a_grav = gravity_earth_moon(r, tt)
            r_hat = r / rn
            ne = n_rs25_live(tt - t_liftoff)
            T = ne * rs25_thrust(alt) * rs25_thr_f * _rs25_bucket(tt - t_liftoff)  # bucket (=1 post-sep)
            T = _rs25_throttled(T, m)                                              # 5 g axial limiter
            isp = rs25_isp(alt) * rs25_isp_f
            mdot = T / (isp * G0) if T > 0 else 0.0
            chi = np.arctan(tan0 + tan_rate * (tt - t_core0))
            tdir = np.cos(chi) * downrange(r_hat) + np.sin(chi) * r_hat
            a_th = T * tdir / max(m, 1.0)
            v_air = np.cross(np.array([0.0, 0.0, OMEGA_E]), r)
            v_rel = v - v_air; vr = np.linalg.norm(v_rel)
            rho = atm_density(alt); q = 0.5 * rho * vr * vr
            if vr > 1.0 and rho > 1e-9:
                a_drag = -q * SLS_CD * SLS_AREA * v_rel / (vr * max(m, 1.0))
            else:
                a_drag = np.zeros(3)
            return np.concatenate([v, a_grav + a_th + a_drag, [-mdot]])
        sol = solve_ivp(rhs_lt, (t_core0, t_core0 + 800.0), state_core0,
                        method="RK45", rtol=1e-7, atol=1e-1, max_step=3.0,
                        events=[ev_apogee, ev_depletion], dense_output=True)
        cut_apogee = len(sol.t_events[0]) > 0
        if cut_apogee:
            yf, tf = sol.y_events[0][0], sol.t_events[0][0]
        elif len(sol.t_events[1]) > 0:
            yf, tf = sol.y_events[1][0], sol.t_events[1][0]
        else:
            yf, tf = sol.y[:, -1], sol.t[-1]
        return yf, float(tf), cut_apogee, sol

    # --- PEG (full Powered Explicit Guidance) predictor -----------------------
    # Same linear-tangent steering as _fly_lt, but the cutoff is the PEG ENERGY
    # condition: integrate until the osculating SMA reaches the target a_T on the
    # ASCENDING branch (vs _fly_lt's apogee-radius cutoff). The 2-DOF corrector then
    # solves (χ0, χ̇) so the cutoff state has the target angular momentum h_T (-> the
    # target eccentricity / 30×1806 orbit) AND the target MECO radius r_T (157 km).
    _peg_aT = 0.5 * (target_peri_r + target_apo_r)
    _peg_eT = (target_apo_r - target_peri_r) / (target_apo_r + target_peri_r)
    _peg_hT = np.sqrt(MU_EARTH * _peg_aT * (1.0 - _peg_eT * _peg_eT))
    _peg_rT = R_EARTH + PEG_MECO_ALT_KM * 1000.0
    def _fly_peg(chi0_deg, tan_rate, tan_rate2=None):
        # OPTIONAL second linear-tangent SEGMENT (ENABLE_PEG_2SEG): for tt > t_break the tangent
        # rate switches to tan_rate2 (tan(χ) continuous at the break). One extra steering DOF ≈
        # real PEG's continuously re-solved profile; the weak-ascent trials' (perigee, r_cut)
        # target lies ~110 km outside the single-segment family's reachable set but
        # inside the two-segment one. tan_rate2=None -> the original single-segment law,
        # bit-identical.
        tan0 = np.tan(np.deg2rad(chi0_deg))
        _tb = t_core0 + PEG2_TBREAK_S
        def _tanchi(tt):
            if tan_rate2 is None or tt <= _tb:
                return tan0 + tan_rate * (tt - t_core0)
            return tan0 + tan_rate * (_tb - t_core0) + tan_rate2 * (tt - _tb)
        def rhs(tt, y):
            r = y[:3]; v = y[3:6]; m = y[6]; rn = np.linalg.norm(r)
            alt = rn - R_EARTH
            a_grav = gravity_earth_moon(r, tt)
            r_hat = r / rn
            ne = n_rs25_live(tt - t_liftoff)
            T = ne * rs25_thrust(alt) * rs25_thr_f * _rs25_bucket(tt - t_liftoff)
            T = _rs25_throttled(T, m)
            isp = rs25_isp(alt) * rs25_isp_f
            mdot = T / (isp * G0) if T > 0 else 0.0
            chi = np.arctan(_tanchi(tt))
            tdir = np.cos(chi) * downrange(r_hat) + np.sin(chi) * r_hat
            a_th = T * tdir / max(m, 1.0)
            v_air = np.cross(np.array([0.0, 0.0, OMEGA_E]), r)
            v_rel = v - v_air; vr = np.linalg.norm(v_rel)
            rho = atm_density(alt); q = 0.5 * rho * vr * vr
            a_drag = (-q * SLS_CD * SLS_AREA * v_rel / (vr * max(m, 1.0))
                      if (vr > 1.0 and rho > 1e-9) else np.zeros(3))
            return np.concatenate([v, a_grav + a_th + a_drag, [-mdot]])
        def ev_sma(tt, y):                                  # osc SMA reaches a_T (energy cutoff)
            r = y[:3]; v = y[3:6]; rn = np.linalg.norm(r)
            E = 0.5 * float(np.dot(v, v)) - MU_EARTH / rn
            return (-MU_EARTH / (2 * E)) - _peg_aT if E < 0 else 1.0
        ev_sma.terminal = True; ev_sma.direction = 1
        sol = solve_ivp(rhs, (t_core0, t_core0 + 900.0), state_core0, method="RK45",
                        rtol=1e-7, atol=1e-1, max_step=3.0,
                        events=[ev_sma, ev_depletion], dense_output=True)
        if len(sol.t_events[0]):                            # reached target energy ascending
            return sol.y_events[0][0], float(sol.t_events[0][0]), True, sol
        if len(sol.t_events[1]):                            # depleted before a_T (e.g. engine-out can't reach)
            return sol.y_events[1][0], float(sol.t_events[1][0]), False, sol
        return sol.y[:, -1], float(sol.t[-1]), False, sol

    def _peg_resid(chi0_deg, tan_rate, tan_rate2=None):
        """(perigee_err, r_cut_err) at the a_T-energy cutoff, in units of 100 km; None if a_T not
        reached. Targeting PERIGEE directly (vs angular momentum) is far better-conditioned — e is
        over-sensitive to h near low eccentricity, so an h-residual leaves a several-km perigee miss."""
        yf, tf, reached, sol = _fly_peg(chi0_deg, tan_rate, tan_rate2)
        if not reached:
            return None, yf, tf, sol
        ap = _apsides(yf)
        if ap is None:
            return None, yf, tf, sol
        rn = np.linalg.norm(yf[:3])
        return np.array([(ap[0] - target_peri_r) / 1e5, (rn - _peg_rT) / 1e5]), yf, tf, sol

    # perigee at the apogee-cutoff as a function of the initial pitch χ₀ (km error
    # vs target). The apogee-energy cutoff (ev_apogee) PINS the target apogee, and
    # for a given apogee the perigee and the burn time are linked by the rocket
    # equation — so there is exactly ONE independent steering DOF. We therefore
    # solve a single linear-tangent coefficient χ₀ (with the tangent rate fixed) to
    # hit the target perigee. (The classic S-IVB IGM solved TWO coefficients because
    # its circular target imposed two conditions — radius AND speed — at an FPA=0
    # cutoff; the SLS elliptical target with an energy cutoff imposes one.)
    IGM_TAN_RATE = globals().get("_IGM_TAN_RATE_OVERRIDE", -0.0015)
    def _peri_err_km(chi0_deg):
        yf, _tf, cut_apogee, _ = _fly_lt(chi0_deg, IGM_TAN_RATE)
        ap = _apsides(yf)
        if ap is None or not cut_apogee:
            return None
        return (ap[0] - target_peri_r) / 1000.0

    # --- PEG solve: 2-DOF damped Newton on (χ0, χ̇) to the full cutoff target ---
    peg_done = False
    if globals().get("ENABLE_PEG_GUIDANCE", False):
        try:
            _tr0 = IGM_TAN_RATE
            # warm-start χ0: coarse scan (tan_rate fixed), pick the smallest angular-momentum error
            cg = None; cbest = 1e30
            for c in [16., 19., 22., 25., 28., 31., 35., 42., 52.]:
                rr, _yf, _tf, _s = _peg_resid(c, _tr0)
                if rr is not None and abs(rr[0]) < cbest:
                    cbest = abs(rr[0]); cg = c
            if cg is None:
                result["peg_fail"] = "no_warmstart"     # all 9 coarse χ0 candidates returned None
            else:
                x = np.array([cg, _tr0]); res, yf, tf, sol_p = _peg_resid(x[0], x[1])
                for _ in range(12):
                    if res is None or np.linalg.norm(res) < 5e-3:   # ~0.5 km each — early stop
                        break
                    J = np.zeros((2, 2)); ok = True
                    for k, (dch, dtr) in enumerate([(0.4, 0.0), (0.0, 2e-4)]):
                        rk, _, _, _ = _peg_resid(x[0] + dch, x[1] + dtr)
                        if rk is None:
                            ok = False; break
                        J[:, k] = (rk - res) / (dch if k == 0 else dtr)
                    if not ok:
                        break
                    try:
                        step = np.linalg.solve(J, -res)
                    except np.linalg.LinAlgError:
                        break
                    step[0] = float(np.clip(step[0], -PEG_STEP_CHI_MAX, PEG_STEP_CHI_MAX))
                    step[1] = float(np.clip(step[1], -PEG_STEP_TR_MAX, PEG_STEP_TR_MAX))
                    improved = False
                    for scale in (1.0, 0.5, 0.25):                 # backtracking damping
                        xn = np.array([float(np.clip(x[0] + scale * step[0], PEG_CHI0_MIN, PEG_CHI0_MAX)),
                                       float(np.clip(x[1] + scale * step[1], PEG_TR_MIN, PEG_TR_MAX))])
                        rn_, yfn, tfn, soln = _peg_resid(xn[0], xn[1])
                        if rn_ is not None and np.linalg.norm(rn_) < np.linalg.norm(res):
                            x, res, yf, tf, sol_p = xn, rn_, yfn, tfn, soln
                            improved = True; break
                    if not improved:
                        break
                if res is not None and np.linalg.norm(res) < 2e-2:   # accept within ~1-2 km
                    apo = _apsides(yf)
                    if apo is not None:
                        pk = (apo[0] - R_EARTH) / 1000.0; ak = (apo[1] - R_EARTH) / 1000.0
                        if abs(pk - INSERTION_PERIGEE_KM) < 30.0 and abs(ak - INSERTION_APOGEE_KM) < 60.0:
                            state = yf.copy(); t = tf; sol_c = sol_p
                            result["meco_mode"] = "peg_lineartangent"
                            result["peg_chi0_deg"] = float(x[0]); result["peg_tan_rate"] = float(x[1])
                            result["peg_resid"] = float(np.linalg.norm(res))
                            peg_done = True
                        else:                       # converged but to the WRONG orbit (diagnostics)
                            result["peg_fail"] = "apsides_reject pk=%.1f ak=%.1f" % (pk, ak)
                    else:
                        result["peg_fail"] = "apsides_none"
                else:                               # Newton stalled short of the acceptance gate
                    result["peg_fail"] = ("newton_stall resid=%.4f" % float(np.linalg.norm(res))
                                          if res is not None else "newton_stall resid=None")
        except Exception as _pe:
            peg_done = False
            result["peg_fail"] = "exception:%s" % type(_pe).__name__

    # --- TWO-SEGMENT PEG retry (ENABLE_PEG_2SEG): only when the 1-segment Newton stalled ---------
    if (not peg_done) and globals().get("ENABLE_PEG_2SEG", False) \
            and globals().get("ENABLE_PEG_GUIDANCE", False):
        try:
            # warm start: the stalled 2-DOF iterate if available, else the scan midpoint
            try:
                _u = np.array([float(x[0]), float(x[1]), float(x[1])])
            except Exception:
                _u = np.array([28.0, IGM_TAN_RATE, IGM_TAN_RATE])
            res2, yf2, tf2, sol2 = _peg_resid(_u[0], _u[1], _u[2])
            for _it in range(10):
                if res2 is None or np.linalg.norm(res2) < 5e-3:
                    break
                J2 = np.zeros((2, 3)); ok2 = True
                for k, (dch, dt1, dt2) in enumerate([(0.4, 0.0, 0.0),
                                                     (0.0, 2e-4, 0.0),
                                                     (0.0, 0.0, 2e-4)]):
                    rk, _, _, _ = _peg_resid(_u[0] + dch, _u[1] + dt1, _u[2] + dt2)
                    if rk is None:
                        ok2 = False; break
                    J2[:, k] = (rk - res2) / (dch or dt1 or dt2)
                if not ok2:
                    break
                step2, *_ = np.linalg.lstsq(J2, -res2, rcond=1e-8)   # min-norm (2 constraints, 3 DOF)
                step2[0] = float(np.clip(step2[0], -PEG_STEP_CHI_MAX, PEG_STEP_CHI_MAX))
                step2[1] = float(np.clip(step2[1], -PEG_STEP_TR_MAX, PEG_STEP_TR_MAX))
                step2[2] = float(np.clip(step2[2], -PEG_STEP_TR_MAX, PEG_STEP_TR_MAX))
                improved2 = False
                for _sc in (1.0, 0.5, 0.25):
                    un = np.array([float(np.clip(_u[0] + _sc * step2[0], PEG_CHI0_MIN, PEG_CHI0_MAX)),
                                   float(np.clip(_u[1] + _sc * step2[1], PEG_TR_MIN, PEG_TR_MAX)),
                                   float(np.clip(_u[2] + _sc * step2[2], PEG_TR_MIN, PEG_TR_MAX))])
                    rn2, yfn, tfn, soln = _peg_resid(un[0], un[1], un[2])
                    if rn2 is not None and np.linalg.norm(rn2) < np.linalg.norm(res2):
                        _u, res2, yf2, tf2, sol2 = un, rn2, yfn, tfn, soln
                        improved2 = True; break
                if not improved2:
                    break
            if res2 is not None and np.linalg.norm(res2) < 2e-2:     # same acceptance gate as 1-seg
                apo2 = _apsides(yf2)
                if apo2 is not None:
                    pk2 = (apo2[0] - R_EARTH) / 1000.0; ak2 = (apo2[1] - R_EARTH) / 1000.0
                    if abs(pk2 - INSERTION_PERIGEE_KM) < 30.0 and abs(ak2 - INSERTION_APOGEE_KM) < 60.0:
                        state = yf2.copy(); t = tf2; sol_c = sol2
                        result["meco_mode"] = "peg_2seg"
                        result["peg_chi0_deg"] = float(_u[0]); result["peg_tan_rate"] = float(_u[1])
                        result["peg_tan_rate2"] = float(_u[2])
                        result["peg_resid"] = float(np.linalg.norm(res2))
                        peg_done = True
            if not peg_done:
                result["peg2_fail"] = ("resid=%.4f" % float(np.linalg.norm(res2))
                                       if res2 is not None else "resid=None")
        except Exception as _pe2:
            result["peg2_fail"] = "exception:%s" % type(_pe2).__name__

    igm_done = peg_done                  # a converged PEG solve short-circuits the IGM (now the fallback)
    if not igm_done and globals().get("ENABLE_IGM_ASCENT", False):
        try:
            from scipy.optimize import brentq
            # χ₀ grid spans the physical region from the in-plane horizontal up.
            # Very horizontal angles over-build energy and escape (perigee err None,
            # filtered). The perigee(χ₀) curve crosses the target TWICE: a low-χ₀
            # LOFTED root whose apogee-energy cutoff fires while the vehicle is
            # already DESCENDING (FPA<0 — it then dives to the 30 km perigee, an
            # unphysical atmospheric pass) and a higher-χ₀ flatter root that hits the
            # target apogee while still ASCENDING (FPA>0 — the real Artemis I MECO,
            # which coasts up to apogee for the PRM). Enumerate ALL bracketed roots
            # and PREFER the ascending one; keep the descending root only as a
            # fallback (perturbed trials where the ascending branch isn't reachable).
            grid = [14.0, 17.0, 20.0, 23.0, 27.0, 32.0, 40.0, 52.0, 65.0]
            ev = [(c, e) for c, e in ((c, _peri_err_km(c)) for c in grid)
                  if e is not None]
            cands = []                                  # (ascending, chi0, yf, tf, sol)
            for (c1, e1), (c2, e2) in zip(ev, ev[1:]):
                if e1 * e2 > 0:
                    continue
                try:
                    chi0_sol = brentq(_peri_err_km, c1, c2, xtol=0.05, maxiter=40)
                except Exception:
                    continue
                yf, tf, cut_apogee, sol_lt = _fly_lt(chi0_sol, IGM_TAN_RATE)
                ap = _apsides(yf)
                if not (cut_apogee and ap is not None):
                    continue
                peri_km = (ap[0] - R_EARTH) / 1000.0
                apo_km = (ap[1] - R_EARTH) / 1000.0
                if not (abs(peri_km - INSERTION_PERIGEE_KM) < 30.0
                        and abs(apo_km - INSERTION_APOGEE_KM) < 60.0):
                    continue
                ascending = float(np.dot(yf[:3], yf[3:6])) > 0.0   # radial velocity > 0
                cands.append((ascending, float(chi0_sol), yf, tf, sol_lt))
            if cands:
                cands.sort(key=lambda c: not c[0])      # ascending-branch roots first
                ascending, chi0_sol, yf, tf, sol_lt = cands[0]
                state = yf.copy(); t = tf; sol_c = sol_lt
                result["meco_mode"] = "igm_lineartangent"
                result["igm_chi0_deg"] = float(chi0_sol)
                result["igm_tan_rate"] = float(IGM_TAN_RATE)
                result["meco_ascending"] = bool(ascending)
                igm_done = True
        except Exception:
            igm_done = False

    if not igm_done:
        # Fallback: open-loop pitch program flown to the apogee/depletion cutoff
        # (also the path when ENABLE_IGM_ASCENT is OFF — the legacy ascent).
        try:
            sol_c = solve_ivp(lambda tt, y: accel(tt, y, False),
                              (t_core0, t_core0 + 800.0), state_core0, method="RK45",
                              rtol=1e-7, atol=1e-1, max_step=3.0,
                              events=[ev_apogee, ev_depletion], dense_output=True)
        except Exception as e:
            result.update(success=False, failure_reason=f"core_integration_error: {e}")
            return result
        t_apo = sol_c.t_events[0][0] if len(sol_c.t_events[0]) else np.inf
        t_dep = sol_c.t_events[1][0] if len(sol_c.t_events[1]) else np.inf
        if t_apo <= t_dep:
            state = sol_c.y_events[0][0].copy(); t = t_apo
            result["meco_mode"] = "guidance_apogee_openloop"
        elif t_dep < np.inf:
            state = sol_c.y_events[1][0].copy(); t = t_dep
            result["meco_mode"] = "propellant_depletion"
        else:
            state = sol_c.y[:, -1].copy(); t = sol_c.t[-1]
            result["meco_mode"] = "timeout"
    # Core-stage separation: the ENTIRE core stage departs — its dry structure AND any
    # UNBURNED residual propellant (the IGM cuts MECO at the target apogee, typically with
    # ~30 t of core prop to spare; that residual rides with the jettisoned core, it is NOT
    # part of the ICPS+Orion stack). Plus the LVSA/MSA adapters. So the post-sep stack is
    # exactly ICPS (wet) + Orion. (The old `state[6] -= CORE_DRY_KG` left the ~30 t residual
    # bolted on -> an 87.7 t stack vs the real ~57.9 t; masked downstream only because
    # phase_icps_tli re-derives m0 from the same ICPS/Orion ledger. Fixing it makes the
    # Phase-1 -> Phase-2 mass handoff continuous; the trajectory/orbit is unaffected.)
    result["meco_mass_kg"] = float(state[6])               # full stack mass at MECO (pre core-sep)
    result["core_residual_kg"] = float(state[6] - meco_mass)  # unburned core prop (rides with the jettisoned core)
    state[6] = ICPS_PROP_KG + ICPS_DRY_KG + ORION_TOTAL_KG

    # --- insertion orbit check ---------------------------------------------
    rn = np.linalg.norm(state[:3]); alt = rn - R_EARTH
    v_actual = np.linalg.norm(state[3:6])
    E = 0.5 * v_actual**2 - MU_EARTH / rn
    if E >= 0:
        result.update(success=False, failure_reason="insertion_escape",
                      state=state, t_insertion=t)
        return result
    a = -MU_EARTH / (2 * E)
    h = np.linalg.norm(np.cross(state[:3], state[3:6]))
    ecc = np.sqrt(max(0.0, 1 - (h*h) / (MU_EARTH * a)))
    perigee = (a*(1-ecc) - R_EARTH) / 1000.0
    apogee = (a*(1+ecc) - R_EARTH) / 1000.0
    result["insertion_perigee_km"] = perigee
    result["insertion_apogee_km"] = apogee
    result["alt_insertion_km"] = alt / 1000.0
    result["t_meco_s"] = t

    # Structural max-q ALSO applies through the core/fallback phase (not just boost): when PEG AND the
    # IGM both fail to converge — a severe engine-out tail — the open-loop gravity-turn fallback flies
    # with no active guidance and can tip over and dive into the atmosphere (q to multi-MPa). Such a
    # vehicle BREAKS UP at max-q, chronologically before it would "underspeed", so the structural mode
    # takes precedence over insertion_underspeed below. (Normal PEG/IGM trajectories are exo-atmospheric
    # in the core -> core max_q ~0, so this never false-fires; the boost-phase breach is caught earlier.)
    if result["max_q_pa"] > MAXQ_STRUCTURAL_PA:
        result.update(success=False, failure_reason="structural_failure_max_q",
                      state=state, t_insertion=t)
        return result

    # The insertion is INTENTIONALLY a low-perigee ~30×1806 km orbit (the ICPS PRM
    # raises the perigee, in phase_icps_tli) — a low perigee is NOT a failure here.
    # MECO is on the ascending branch (climbing to apogee), so it cannot re-enter
    # before the PRM. Failure = the Core failed to deliver the insertion ENERGY
    # (apogee far from 1806 km) or cut off implausibly low in the atmosphere.
    if apogee < 1000.0:
        result.update(success=False, failure_reason="insertion_underspeed",
                      state=state, t_insertion=t)
        return result
    if apogee > 5000.0:
        result.update(success=False, failure_reason="insertion_overshoot",
                      state=state, t_insertion=t)
        return result
    if alt / 1000.0 < 80.0:
        result.update(success=False, failure_reason="insertion_too_low",
                      state=state, t_insertion=t)
        return result

    result["state"] = state
    result["t_insertion"] = t
    result["trajectory_t"] = np.concatenate([sol_b.t, sol_c.t])
    result["trajectory_y"] = np.hstack([sol_b.y, sol_c.y])
    return result


def _fly_finite_tli(sol_coast, t_ign, v_target, v_park, m0, isp, dry_floor, perturb, target_apo_r,
                    nav_dx0=None, nav_ba=None):
    """Fly the ICPS TLI as a STEERED finite burn (vs impulsive). Center an ~18-min RL10 burn at the
    min-ΔV ignition point, steer thrust along the VELOCITY-TO-GO toward the target post-TLI velocity
    (aims the cutoff direction at the lunar arrival), and cut at the TARGET APOGEE (the trans-lunar
    energy) — NOT a perigee-speed target, which over-burns once the vehicle has climbed. Modeling the
    finite burn fixes the post-TLI PHASING (the vehicle coasts onward from BURNOUT, not the impulsive
    ignition point) — the dominant piece of the post-TLI-vs-OEM residual. Returns (state7, t_cut) or None
    on starve. Consumes the same tli_dv_bias_ms (cutoff-apogee bias) + tli_pointing_rad as the impulsive path."""
    g0 = G0
    T = ICPS_THRUST_N * perturb.get("icps_thrust_factor", 1.0)
    c = isp * g0
    mdot = T / c
    dv_nom = float(np.linalg.norm(v_target - v_park))
    m_after = m0 * np.exp(-dv_nom / c)
    tau = (m0 - m_after) / mdot                       # nominal burn duration (~19 min)
    t_start = max(float(sol_coast.t[0]) + 1.0, t_ign - 0.5 * tau)   # center the burn at the ignition point
    y0 = sol_coast.sol(t_start)
    state0 = np.concatenate([y0[:3], y0[3:6], [m0]])
    apo_cut = target_apo_r                                          # cut at the clean trans-lunar target
    dv_bias = float(perturb.get("tli_dv_bias_ms", 0.0))            # applied as a prograde Δv at burnout
    pt = np.asarray(perturb.get("tli_pointing_rad", np.zeros(3)), float)
    has_pt = np.linalg.norm(pt) > 1e-12

    # CONTINUOUS within-burn closed-loop NAV (v2): the guidance steers + cuts on a dead-reckoned
    # ESTIMATE x̂ = x_true + δx(t) that drifts via IMU errors — initial pre-burn knowledge (nav_dx0)
    # plus an accelerometer bias (nav_ba), both RIC at burn start, propagated in CLOSED FORM:
    #   δv(Δt) = δv0 + b_a·Δt ;  δr(Δt) = δr0 + δv0·Δt + ½·b_a·Δt² .  The vehicle flies TRUE dynamics;
    # cutting on the ESTIMATED apogee leaves the TRUE injection off by the nav error (emergent — no
    # discrete offset). nav_dx0 None -> perfect knowledge (v1 / flag OFF), bit-identical to before.
    _nav = nav_dx0 is not None
    if _nav:
        _R, _I, _C = _ric_basis(state0)
        _dr0 = nav_dx0[0]*_R + nav_dx0[1]*_I + nav_dx0[2]*_C
        _dv0 = nav_dx0[3]*_R + nav_dx0[4]*_I + nav_dx0[5]*_C
        _ba  = nav_ba[0]*_R + nav_ba[1]*_I + nav_ba[2]*_C

    def _est(tt):                                      # estimate error (δr, δv) at time tt
        dt = tt - t_start
        return _dr0 + _dv0*dt + 0.5*_ba*dt*dt, _dv0 + _ba*dt

    def rhs(tt, y):
        r = y[:3]; v = y[3:6]; m = y[6]
        vg = v + _est(tt)[1] if _nav else v            # steer along the ESTIMATED velocity
        td = vg / np.linalg.norm(vg)                   # PROGRADE — minimal-loss apogee raising
        if has_pt:
            td = td + np.cross(pt, td); td = td / np.linalg.norm(td)
        return np.concatenate([v, gravity_earth_moon(r, tt) + (T / max(m, 1.0)) * td, [-mdot]])

    def cutoff(tt, y):                                 # ESTIMATED osculating apogee reaches the target
        r = y[:3]; v = y[3:6]
        if _nav:
            dr, dv = _est(tt); r = r + dr; v = v + dv   # guidance cuts on the estimate, not truth
        rn = np.linalg.norm(r)
        E = 0.5 * float(np.dot(v, v)) - MU_EARTH / rn
        if E >= 0:
            return 1e12                                # hyperbolic -> well past target, cut
        a = -MU_EARTH / (2 * E)
        h = np.linalg.norm(np.cross(r, v))
        e = np.sqrt(max(0.0, 1 - (h*h) / (MU_EARTH * a)))
        return a * (1 + e) - apo_cut
    cutoff.terminal = True; cutoff.direction = +1

    def starve(tt, y):
        return y[6] - dry_floor
    starve.terminal = True; starve.direction = -1

    sol = solve_ivp(rhs, (t_start, t_start + 1.6 * tau + 200.0), state0, method="RK45",
                    rtol=1e-8, atol=1e-1, max_step=2.0, events=[cutoff, starve])
    if len(sol.t_events[0]) == 0:
        return None                                    # starved before the guidance cutoff
    yb = sol.y_events[0][0].copy(); t_cut = float(sol.t_events[0][0])
    if dv_bias != 0.0:                                 # cutoff-speed bias (matches the impulsive v_cut bias)
        vh = yb[3:6] / np.linalg.norm(yb[3:6]); yb[3:6] = yb[3:6] + dv_bias * vh
    return yb, t_cut


def phase_icps_prm(state, t0, perturb=None):
    """ICPS Perigee-Raise Maneuver — the FIRST of the two ICPS (RL10B-2) burns
    (split from the old combined phase 2; the SOURCED Artemis I sequence). Coast the LOW-perigee ~30×1806 km insertion ~45 min
    to APOGEE (real PRM at GET 00:52:56) and fly a prograde PRM (~42 m/s, ~22 s)
    raising the perigee to PRM_PERIGEE_KM (185 km) for a stable parking orbit.

    Establishes the SHARED ICPS propellant ledger (m0 wet stack -> dry_floor) and
    threads it to phase_icps_tli via the returned state mass. The RL10 ignition-fail
    draw is checked here (the engine's first firing — if it fails to ignite there is
    no PRM and no TLI). Failure modes: icps_ignition_failure, prm_propellant_depleted,
    prm_coast_error.
    Returns success, failure_reason, state ([r,v,m] post-PRM, at apogee), t_end,
    prm_dv_ms, prm_burn_s, parking_perigee_km, parking_apogee_km."""
    perturb = perturb or {}
    result = {"success": True, "failure_reason": None}

    r = np.asarray(state[:3], float); v = np.asarray(state[3:6], float)
    # ICPS stack = ICPS (full) + Orion; the ~3.8 t of stage adapters in the launch
    # handoff mass are jettisoned (LVSA/MSA/OSA separations) and not flown. The ICPS
    # propellant ledger (m0 wet -> dry_floor) is charged by BOTH the PRM and the TLI;
    # the depletion failure mode trips when the stack mass would fall below the floor.
    # The impulsive TLI omits the ~47 m/s gravity loss of the real ~18-min burn, so the legacy
    # ICPS_PROP_KG (28,576) was calibrated to a lossless TLI. The flown finite burn (ENABLE_FINITE_TLI)
    # incurs that real gravity loss; the real ICPS load accommodated it, so add a small propellant
    # allowance (~600 kg, +2%, within the gross-vs-usable uncertainty) so the finite path keeps the same
    # ~37 m/s margin the impulsive had. Gated -> the impulsive production path stays bit-identical.
    _icps_prop = ICPS_PROP_KG + (FINITE_TLI_PROP_ALLOWANCE_KG if globals().get("ENABLE_FINITE_TLI", False) else 0.0)
    m0 = _icps_prop + ICPS_DRY_KG + ORION_TOTAL_KG         # full wet stack
    dry_floor = ICPS_DRY_KG + ORION_TOTAL_KG               # ICPS empty + Orion
    isp_eff = ICPS_ISP_S * perturb.get("icps_isp_factor", 1.0)
    s = np.concatenate([r, v, [m0]])
    t = float(t0)

    # RL10 fails to ignite -> Orion stranded in the insertion orbit (no PRM, no TLI;
    # the ICPS has no abort-to-DRO option). Sourced draw (see sample_perturbation).
    if perturb.get("icps_ignition_fail", False):
        result.update(success=False, failure_reason="icps_ignition_failure",
                      state=s, t_end=t)
        return result

    # --- Perigee-Raise Maneuver (PRM): coast to apogee, then raise perigee ---
    # Core Sep left the stack in a LOW-perigee ~30×1806 km insertion. The ICPS
    # coasts ~45 min to apogee and fires a discrete PROGRADE PRM raising the perigee
    # to PRM_PERIGEE_KM (185 km) for a stable pre-TLI parking orbit (Artemis I: a
    # 22 s RL10 burn, ~44 m/s). Charged to the ICPS ledger; skipped (ΔV 0) when the
    # insertion perigee is already at/above target (e.g. the legacy open-loop
    # ascent). New failure mode: prm_propellant_depleted.
    def _coast_rhs(tt, y):
        return np.concatenate([y[3:6], gravity_earth_moon(y[:3], tt), [0.0]])
    def _ev_apo(tt, y):
        return float(np.dot(y[3:6], y[:3]))      # r·v = 0 at apogee (descending)
    _ev_apo.terminal = True; _ev_apo.direction = -1
    try:
        sol_pa = solve_ivp(_coast_rhs, (t, t + 4000.0), s, method="RK45",
                           rtol=1e-9, atol=1e-1, max_step=30.0,
                           events=[_ev_apo], dense_output=True)
    except Exception as e:
        result.update(success=False, failure_reason=f"prm_coast_error: {e}",
                      state=s, t_end=t)
        return result
    if len(sol_pa.t_events[0]):
        s = sol_pa.y_events[0][0].copy(); t = float(sol_pa.t_events[0][0])
    else:                                        # fallback: max-radius point
        ts = np.linspace(t, sol_pa.t[-1], 2000); Y = sol_pa.sol(ts)
        i = int(np.argmax(np.linalg.norm(Y[:3], axis=0)))
        s = Y[:, i].copy(); t = float(ts[i])
    r_apo = s[:3].copy(); v_apo = s[3:6].copy()
    rn_apo = np.linalg.norm(r_apo); v_apo_mag = np.linalg.norm(v_apo)
    # prograde burn AT apogee raises the opposite apsis (perigee) to the target.
    r_peri_target = R_EARTH + PRM_PERIGEE_KM * 1000.0
    a_new = 0.5 * (rn_apo + r_peri_target)
    v_apo_new = np.sqrt(MU_EARTH * (2.0 / rn_apo - 1.0 / a_new))
    prm_dv = max(0.0, v_apo_new - v_apo_mag)
    m_pre_tli = m0
    if prm_dv > 1e-6:
        m_after_prm = m0 * np.exp(-prm_dv / (isp_eff * G0))
        if m_after_prm < dry_floor:
            result.update(success=False, failure_reason="prm_propellant_depleted",
                          state=np.concatenate([r_apo, v_apo, [dry_floor]]), t_end=t)
            return result
        if globals().get("ENABLE_FINITE_PRM", False):
            # FLY the PRM as a finite RL10 burn CENTERED at apogee (prograde, ~22 s), cutting when the
            # osculating perigee reaches the 185 km target. Back up ballistically by tau/2 so the burn
            # straddles apogee (a Hohmann perigee-raise is most efficient AT apogee -> ~zero gravity loss).
            mdot = ICPS_THRUST_N / (isp_eff * G0)
            tau = (m0 - m_after_prm) / mdot
            sol_b = solve_ivp(_coast_rhs, (t, t - 0.5 * tau),
                              np.concatenate([r_apo, v_apo, [m0]]),
                              method="RK45", rtol=1e-9, atol=1e-1, max_step=5.0)
            y0 = sol_b.y[:, -1]; t_start = t - 0.5 * tau
            def _rhs_prm(tt, y):
                r = y[:3]; v = y[3:6]; m = y[6]
                vh = v / np.linalg.norm(v)
                return np.concatenate([v, gravity_earth_moon(r, tt) + ICPS_THRUST_N * vh / max(m, 1.0), [-mdot]])
            def _ev_peri(tt, y):                         # cut when osc perigee reaches target
                r = y[:3]; v = y[3:6]; rn = np.linalg.norm(r)
                E = 0.5 * float(np.dot(v, v)) - MU_EARTH / rn
                if E >= 0:
                    return 1.0
                a = -MU_EARTH / (2 * E); h = np.linalg.norm(np.cross(r, v))
                ecc = np.sqrt(max(0.0, 1 - (h * h) / (MU_EARTH * a)))
                return a * (1 - ecc) - r_peri_target
            _ev_peri.terminal = True; _ev_peri.direction = 1
            sol_p = solve_ivp(_rhs_prm, (t_start, t_start + 2.0 * tau), y0, method="RK45",
                              rtol=1e-8, atol=1e-1, max_step=2.0, events=[_ev_peri], dense_output=True)
            if len(sol_p.t_events[0]):
                s = sol_p.y_events[0][0].copy(); t = float(sol_p.t_events[0][0])
            else:                                        # didn't reach target in 2*tau -> use burnout
                s = sol_p.y[:, -1].copy(); t = float(sol_p.t[-1])
            m_pre_tli = float(s[6])
            result["prm_burn_s"] = float((m0 - s[6]) / mdot)
        else:
            v_apo = v_apo + prm_dv * (v_apo / v_apo_mag)
            s = np.concatenate([r_apo, v_apo, [m_after_prm]])
            m_pre_tli = m_after_prm
            result["prm_burn_s"] = float((m0 - m_after_prm) / (ICPS_THRUST_N / (isp_eff * G0)))
    else:
        result["prm_burn_s"] = 0.0
    result["prm_dv_ms"] = float(prm_dv)
    # post-PRM parking-orbit diagnostics (185×1806 km nominal)
    _Epk = 0.5 * float(np.dot(s[3:6], s[3:6])) - MU_EARTH / np.linalg.norm(s[:3])
    _apk = -MU_EARTH / (2 * _Epk)
    _hpk = np.linalg.norm(np.cross(s[:3], s[3:6]))
    _epk = np.sqrt(max(0.0, 1 - (_hpk * _hpk) / (MU_EARTH * _apk)))
    result["parking_perigee_km"] = float((_apk * (1 - _epk) - R_EARTH) / 1000.0)
    result["parking_apogee_km"] = float((_apk * (1 + _epk) - R_EARTH) / 1000.0)
    result["state"] = s
    result["t_end"] = t
    return result


# --- TLI PLAN-ADAPTIVITY (the IGM-fallback fix) ---
# The baked Phase-C aim (PHASEC_TLI_VPOST applied verbatim at the trial's pass of PHASEC_TLI_R_IGN_M)
# assumes the PEG-nominal parking GEOMETRY. Trials whose ascent fell back to IGM insert ~13 km lower
# (same ellipse, shifted phase/argp), meet the baked ignition position ~11-19 km off (PEG: 0.8-1.2 km),
# and the nominal v1 flies a wrong asymptote (CA ~95,000 km -> missed_lunar_approach / depletion;
# 589/10000 trials, 38.5% vs 93.6% success). DESIGN v3 — RETRY-ON-ARTIFACT-FAILURE (v1 Newton-on-v1 and
# v2 gate+choose both A/B-rejected; ballistic prediction cannot decide chain-recoverability — it's
# exec-error luck, and v2 broke the chain-rescued survivors, regressing at scale): every trial
# flies the baked aim first; ONLY an outbound artifact death (missed SOI / lunar impact) triggers ONE
# re-flight of TLI with the min-ΔV Lambert replan (ignition TIME + v1 re-picked on the trial's own
# orbit — the ops replan) + the Phase-C chain with its periselene-floor guard. Non-regressive by
# construction; deterministic; no RNG; flag-OFF bit-identical. Retry logic lives in run_mission;
# the replan branch in phase_icps_tli (_TLI_FORCE_REPLAN); the peri-floor guard in _trim_attempt
# (scoped to RETRY coasts via _TLI_RETRY_COAST — first-pass trials bit-identical to flag-OFF).
ENABLE_TLI_PLAN_ADAPT = (os.environ.get("AR1_TLI_ADAPT", "1") == "1")   # DEFAULT ON (adopted in the definitive production run; AR1_TLI_ADAPT=0 -> pre-fix lineage, bit-identical)
_TLI_RETRY_COAST = False   # per-worker: True only while run_mission flies the post-replan coast
# (Newton-on-v1 variants were built and A/B-REJECTED — both dead ends are recorded in
# the call-site comment in phase_icps_tli: a fixed-epoch
# CA target costs ~200+ m/s of pure re-phasing; a time-free Moon-relative CA target still needs
# ~210-250 m/s because the B-plane displacement from the FORCED ignition point is the expensive lever.
# The adopted replan re-picks ignition TIME + v1 jointly via the existing min-ΔV scan machinery.)

# Per-worker retry hint (the _RETURN_PLAN pattern): set by run_mission ONLY for the second TLI pass
# after an outbound ARTIFACT death (missed SOI / lunar impact) under ENABLE_TLI_PLAN_ADAPT; read by
# phase_icps_tli's forced branch to fly the min-ΔV Lambert replan instead of the baked aim. (The v2
# design routed replanned trials to the legacy periselene-only chain — REJECTED in validation: safe
# flyby radius but the WRONG arrival phase makes DRI/DDP bill 250-360 m/s and the fleet regressed
# 87.2→85.2%; replanned trials must keep the Phase-C ca_point_trim so the arrival geometry — and the
# cheap insertion — is preserved, with the trim's periselene-floor guard preventing its impact mode.)
_TLI_FORCE_REPLAN = False


def phase_icps_tli(state, t0, perturb=None):
    """ICPS Trans-Lunar Injection — the SECOND ICPS (RL10B-2) burn (split from the
    old combined phase 2). From the post-PRM ~185×1806 km parking orbit, coast toward
    the next perigee (real TLI ignition GET 01:29:27, ~37 min after the PRM) and fly
    the TLI: a minimum-ΔV first-rev Lambert burn (a STEERED ~18-min finite burn when
    ENABLE_FINITE_TLI) that raises apogee to the Moon's distance (~3 km/s, near-escape
    C3 ~-2). CONTINUES the shared ICPS ledger established by phase_icps_prm (the PRM
    already drew from it; the input state mass IS the ledger). ICPS then separates,
    leaving Orion on the trans-lunar trajectory.

    With ENABLE_FINITE_TLI + ENABLE_PHASEC_BPLANE (default), the TLI is a STEERED
    finite burn flown to the offline-solved forced post-TLI state (PHASEC_TLI_VPOST at
    the phase-adaptive ignition) that hits the REAL lunar approach, with the outbound
    OTC chain trimming the residual encounter error. Flags OFF, the legacy TLI is a
    COPLANAR prograde impulsive burn to lunar-distance apogee (correct energy/
    propellant; lunar-encounter targeting deferred to the OTC/OPF phases).
    Returns success, failure_reason, state ([r,v,m] post-ICPS-sep), t_end, tli_dv_ms,
    post_tli_apogee_km, post_tli_c3_km2s2, icps_prop_used_kg/margin_kg."""
    perturb = perturb or {}
    result = {"success": True, "failure_reason": None}
    _icps_prop = ICPS_PROP_KG + (FINITE_TLI_PROP_ALLOWANCE_KG if globals().get("ENABLE_FINITE_TLI", False) else 0.0)
    m0 = _icps_prop + ICPS_DRY_KG + ORION_TOTAL_KG         # full wet stack (for the cumulative margin report)
    dry_floor = ICPS_DRY_KG + ORION_TOTAL_KG               # ICPS empty + Orion
    isp_eff = ICPS_ISP_S * perturb.get("icps_isp_factor", 1.0)
    s = np.concatenate([np.asarray(state[:3], float), np.asarray(state[3:6], float), [float(state[6])]])
    m_pre_tli = float(state[6])                            # post-PRM mass = the threaded ICPS ledger
    t = float(t0)

    # --- choose TLI ignition + aim (minimum-ΔV Lambert to the Moon) ---------
    # Coast through ~1.5 parking revs; at sampled points Lambert-target the Moon
    # at the flyby epoch and pick the MINIMUM-ΔV point. That point is the natural
    # TLI ignition (near perigee, phased so the outbound leg heads at the Moon) —
    # it sets both the ignition state and the burn AIM direction. The launch
    # azimuth already aligned the plane, so this ΔV is ~prograde and affordable.
    t_flyby = OPF_GET_S
    r_moon_flyby = moon_state(t_flyby)[0]
    target_apo_r = np.linalg.norm(r_moon_flyby)
    # Aim at an OFFSET from Moon center so the ballistic pass clears the surface
    # (~130 km) instead of impacting; offset is in the insertion plane, perp to
    # the Earth->Moon line. (Precise B-plane / DRO-side targeting is the OPF's job.)
    _h0 = np.cross(s[:3], s[3:6]); _h0 = _h0 / np.linalg.norm(_h0)
    _mhat = r_moon_flyby / np.linalg.norm(r_moon_flyby)
    _off = np.cross(_h0, _mhat); _off = _off / np.linalg.norm(_off)
    aim_point = r_moon_flyby + LUNAR_AIM_OFFSET_KM * 1000.0 * _off
    if globals().get("ENABLE_PHASEC_BPLANE", False) and globals().get("PHASEC_TLI_AIM_ECI") is not None:
        aim_point = np.asarray(globals()["PHASEC_TLI_AIM_ECI"], float)   # real-approach aim (solved to match
        #   the real post-TLI state via the SAME min-ΔV ignition scan -> affordable ΔV, real lunar approach)
    try:
        sol_coast = solve_ivp(lambda tt, y: np.concatenate(
                                  [y[3:6], gravity_earth_moon(y[:3], tt), [0.0]]),
                              (t, t + 9600.0), s, method="RK45",
                              rtol=1e-8, atol=1e-1, max_step=60.0, dense_output=True)
    except Exception as e:
        result.update(success=False, failure_reason=f"tli_coast_error: {e}")
        return result
    def _scan_best(t_lo, t_hi, n, seed=None):
        b = seed
        for tc in np.linspace(t_lo, t_hi, n):
            yc = sol_coast.sol(tc)
            lam = lambert_uv(yc[:3], aim_point, t_flyby - tc, mu=MU_EARTH, prograde=True)
            if lam is None:
                continue
            dv = float(np.linalg.norm(lam[0] - yc[3:6]))
            if b is None or dv < b[0]:
                b = (dv, float(tc), yc.copy(), lam[0].copy())
        return b
    # Coarse scan, then a FINE refinement around the coarse min. The min-ΔV is a
    # sharp valley, so the coarse 240 s sampling lands at slightly different points
    # trial-to-trial (a ~200 m/s ignition-selection JITTER); on the tight ICPS TLI
    # margin that jitter alone caused spurious propellant-depletion failures. The
    # fine pass converges every trial to the true minimum (TLI ΔV clusters at
    # ~3.04 km/s), so only the genuine margin tail depletes.
    #
    # FIRST-REV GATE: the min-ΔV Lambert valley recurs ONCE PER PARKING REV within
    # a few m/s (the rev-0 and rev-1 valleys differ ~5 m/s nominal). Scanning ~1.5
    # revs let the global min slip to the LATER-rev valley for a trivial dv gain,
    # which (a) inflated the insertion->TLI phase ~130% (one extra ~105 min rev) and
    # (b) made the choice platform-FRAGILE — laptop vs cluster scipy flipped the
    # near-tie, so 99% of cluster trials fired TLI a rev late. Artemis I fired on the
    # FIRST parking rev (~37 min after the PRM). So scan only the FIRST rev (one
    # parking period from the PRM); extend to later revs ONLY if rev-1 has no valid
    # Lambert (mirrors the TEI first-rev-wins gate). RNG stream untouched.
    _Es = 0.5 * float(np.dot(s[3:6], s[3:6])) - MU_EARTH / np.linalg.norm(s[:3])
    _Tpark = 2 * np.pi * np.sqrt((-MU_EARTH / (2 * _Es)) ** 3 / MU_EARTH) if _Es < 0 else 9600.0
    _rev1_hi = min(t + 9600.0, t + _Tpark) if ENABLE_FIRST_REV_TLI else (t + 9600.0)
    # Min-ΔV ignition scan (aim_point is the Moon+offset natural aim, OR the Phase C real-approach aim
    # baked into PHASEC_TLI_AIM_ECI — same affordable machinery, the aim sets the lunar approach).
    _forced = (globals().get("ENABLE_PHASEC_BPLANE", False)
               and globals().get("PHASEC_TLI_IGN_S") is not None
               and globals().get("PHASEC_TLI_VPOST") is not None)
    if _forced:
        # Phase C: FORCE the offline-solved (ignition, post-TLI velocity) that flies the REAL lunar
        # approach. The min-ΔV scan can't reproduce a specific full-dynamics trajectory from an aim point
        # (sharp valley -> re-picks a ~min-earlier ignition -> flyby amplifies to tens of deg). Deterministic;
        # MC dispersions layer on top downstream, as with the scanned solution.
        # PHASE-ADAPTIVE ignition (v9): ignite where THIS trial's parking orbit passes nearest the baked
        # NOMINAL ignition position — real guidance timed TLI on orbital criteria, not wall-clock GET
        # (launch-timing dispersion shifts the phase; the baked v_target from the wrong true anomaly was
        # the dominant impact channel). Falls back to the fixed GET if no position is baked.
        _rign = globals().get("PHASEC_TLI_R_IGN_M")
        if _rign is not None:
            _rign = np.asarray(_rign, float)
            _tgrid = np.linspace(t + 300.0, _rev1_hi, 600)
            _dg = np.array([np.linalg.norm(sol_coast.sol(tt)[:3] - _rign) for tt in _tgrid])
            _i0 = int(np.argmin(_dg))
            _tfine = np.linspace(_tgrid[max(0, _i0 - 1)], _tgrid[min(len(_tgrid) - 1, _i0 + 1)], 400)
            _df = np.array([np.linalg.norm(sol_coast.sol(tt)[:3] - _rign) for tt in _tfine])
            _tf = float(_tfine[int(np.argmin(_df))])
        else:
            _tf = float(np.clip(globals()["PHASEC_TLI_IGN_S"], t + 300.0, t + 9600.0))
        _ycf = sol_coast.sol(_tf)
        _v1f = np.asarray(globals()["PHASEC_TLI_VPOST"], float)
        # TLI PLAN-ADAPTIVITY (the IGM-fallback fix): if this trial's achieved ignition point sits far
        # from the baked NOMINAL ignition position (off-geometry parking orbit — the IGM-fallback class
        # inserts ~13 km lower with a shifted phase/argp; measured pass-miss 11-19 km vs PEG 0.8-1.2),
        # the baked v1 flies a wrong asymptote (CA ~95,000 km). Design history (A/B-measured dead ends):
        # Newton-on-v1 at the FORCED ignition point cannot fix it affordably — a fixed-epoch
        # target demands ~200+ m/s of pure re-phasing, and even the time-free Moon-relative CA target
        # needs ~210-250 m/s (B-plane displacement ~93,000 km / t_to_CA) — the ignition POINT itself is
        # the wrong lever. The REAL ops replan re-picks ignition TIME + v1 JOINTLY: fall through to the
        # existing min-ΔV ignition scan + Lambert machinery, which (under Phase-C) already targets the
        # real-approach aim point (aim_point = PHASEC_TLI_AIM_ECI above) from the trial's OWN orbit; the
        # residual approach error is what the OTC chain corrector exists to trim. Guarded: if the scan
        # finds nothing, keep the baked aim (current behavior — never worse).
        if globals().get("_TLI_FORCE_REPLAN", False):
            # RETRY REPLAN (ENABLE_TLI_PLAN_ADAPT, design v3 — retry-on-artifact-failure): the FIRST
            # pass flew the baked aim and the coast ended in an outbound ARTIFACT death (missed SOI /
            # lunar impact) — run_mission re-flies TLI with the ops replan: re-pick ignition TIME + v1
            # JOINTLY via the min-ΔV scan (Phase-C aim point) on the trial's OWN orbit. Design history
            # (all A/B-measured): Newton-on-v1 at the forced point = 200-250 m/s (wrong lever);
            # gate+evaluate-then-choose at TLI = broke the chain-rescued survivors (ballistic baked
            # periselene does NOT predict chain success — 46k km missed while 52-64k survived, it's
            # exec-error luck) -> regressed 87.2->85.2% in validation. Retry-on-failure is non-regressive
            # BY CONSTRUCTION: successes never re-fly; only artifact deaths get the second chance.
            # Guarded: scan failure -> the baked aim again (identical outcome, no worse).
            _rb = _scan_best(t + 600.0, _rev1_hi, 40)
            if _rb is None:
                _rb = _scan_best(t + 600.0, t + 9600.0, 40)    # later revs if rev-1 has no solution
            if _rb is not None:
                _lo = max(t + 600.0, _rb[1] - 400.0); _hi = min(t + 9600.0, _rb[1] + 400.0)
                _rb = _scan_best(_lo, _hi, 33, seed=_rb)
            if _rb is not None:
                best = _rb
                result["tli_adapt"] = "lambert_replan_retry"
            else:
                result["tli_adapt"] = "replan_scan_failed"
                best = (float(np.linalg.norm(_v1f - _ycf[3:6])), _tf, _ycf.copy(), _v1f)
        else:
            best = (float(np.linalg.norm(_v1f - _ycf[3:6])), _tf, _ycf.copy(), _v1f)
    else:
        best = _scan_best(t + 600.0, _rev1_hi, 40)        # FIRST rev only (gated)
        if best is None:                                   # rev-1 has no valid solution
            best = _scan_best(t + 600.0, t + 9600.0, 40)   # fall back to later revs
        if best is not None:
            win = (t + 600.0, t + 9600.0)
            lo = max(win[0], best[1] - 400.0); hi = min(win[1], best[1] + 400.0)
            best = _scan_best(lo, hi, 33, seed=best)
    if best is None:
        result.update(success=False, failure_reason="tli_no_lambert_solution",
                      state=np.concatenate([s[:3], s[3:6], [ORION_TOTAL_KG]]), t_end=t)
        return result
    _dv_ign, t, s, v1_lambert = best
    r_ign = s[:3].copy(); v_ign = s[3:6].copy(); r_ign_n = np.linalg.norm(r_ign)
    v_ign_mag = np.linalg.norm(v_ign)
    s = np.concatenate([r_ign, v_ign, [m_pre_tli]])
    aim_hat = v1_lambert - v_ign; aim_hat = aim_hat / np.linalg.norm(aim_hat)
    result["tli_ignition_dv_ms"] = float(_dv_ign)
    # ignition state diagnostics (for the finite-burn TLI study): position, pre-burn (parking) velocity,
    # impulsive post-burn velocity, and the ignition time.
    result["tli_ign_r_m"] = r_ign.copy(); result["tli_ign_vpark_ms"] = v_ign.copy()
    result["tli_ign_vpost_ms"] = v1_lambert.copy(); result["tli_ign_t_s"] = float(t)

    # --- TLI burn (impulsive Lambert at the ignition point) -----------------
    # The minimum-ΔV ignition sits near the 180°-transfer (Hohmann) point, where
    # the Lambert v1 is mildly off-prograde and a finite Vgo burn is marginal on
    # the tight ICPS budget. v1 models the TLI as IMPULSIVE there: set v = v1, with
    # propellant charged by the rocket equation and the honest depletion failure
    # mode preserved. A finite-thrust flown TLI (back-solved ignition on a flyable
    # ~-170° transfer geometry, like a classic integrated TLI) is the fidelity
    # refinement — the near-Hohmann point makes a forward-flown burn marginal.
    # TLI propellant from the ROCKET EQ on the pre-TLI mass (the PRM already drew
    # from the same ICPS ledger); depletion = the cumulative PRM+TLI burn would put
    # the stack below the dry floor (ICPS empty + Orion).
    if globals().get("ENABLE_FINITE_TLI", False):
        # STEERED finite ~18-min ICPS burn (fixes post-TLI phasing). Same propellant/depletion ledger.
        # Target apogee = the impulsive orbit's apogee (the trans-lunar energy the burn cuts off at).
        _Eimp = 0.5 * float(np.dot(v1_lambert, v1_lambert)) - MU_EARTH / np.linalg.norm(r_ign)
        _aimp = -MU_EARTH / (2 * _Eimp) if _Eimp < 0 else 1e12
        _himp = np.linalg.norm(np.cross(r_ign, v1_lambert))
        _eimp = np.sqrt(max(0.0, 1 - (_himp*_himp) / (MU_EARTH * _aimp))) if _Eimp < 0 else 1.0
        _apo_tgt = _aimp * (1 + _eimp) if _Eimp < 0 else 4.5e8
        # v2 continuous within-burn closed-loop nav: pass the drawn pre-burn estimate error (δx0) +
        # accelerometer bias so the burn steers/cuts on a drifting estimate (the injection error then
        # EMERGES from the cutoff). OFF -> nav_dx0=None -> perfect-knowledge burn (v1 / bit-identical).
        _od_capture("tli_init", t, np.concatenate([r_ign, v_ign]))   # pre-TLI parking state (nominal build)
        _nav_dx0 = _nav_ba = None
        if (globals().get("ENABLE_CLOSED_LOOP_NAV", False)
                and globals().get("ENABLE_CLNAV_CONTINUOUS_TLI", False)
                and perturb.get("tli_nav_unit") is not None):
            _Lt = _od_filter_L("tli_init")
            if _Lt is not None:
                _nav_dx0 = _Lt @ np.asarray(perturb["tli_nav_unit"])   # OD-filter: chol(P_ric) @ unit (emergent)
            else:
                _nav_dx0 = np.asarray(perturb["tli_nav_unit"]) * np.array([*CLNAV_TLI_INIT_POS_SIGMA_M, *CLNAV_TLI_INIT_VEL_SIGMA_MS])
            _nav_ba = np.asarray(perturb["tli_nav_ba"]) * np.array(CLNAV_TLI_ACCEL_BIAS_SIGMA_MS2)
        fin = _fly_finite_tli(sol_coast, t, v1_lambert, v_ign, m_pre_tli, isp_eff, dry_floor, perturb, _apo_tgt,
                              nav_dx0=_nav_dx0, nav_ba=_nav_ba)
        if fin is None:
            result.update(success=False, failure_reason="icps_propellant_depleted",
                          tli_dv_attempted_ms=float(np.linalg.norm(v1_lambert - v_ign)),
                          state=np.concatenate([r_ign, v_ign, [dry_floor]]), t_end=t)
            return result
        s, t = fin                                      # burnout state (incl mass) + cutoff time
    else:
        dv = float(np.linalg.norm(v1_lambert - v_ign))
        m_after_tli = m_pre_tli * np.exp(-dv / (isp_eff * G0))
        if m_after_tli < dry_floor:
            result.update(success=False, failure_reason="icps_propellant_depleted",
                          tli_dv_attempted_ms=dv, state=np.concatenate([r_ign, v_ign, [dry_floor]]), t_end=t)
            return result
        # pointing-error execution residual on the achieved velocity
        v_post = v1_lambert.copy()
        pt_err = np.asarray(perturb.get("tli_pointing_rad", np.zeros(3)), float)
        if np.linalg.norm(pt_err) > 1e-12:
            vmag = np.linalg.norm(v_post); vh = v_post / vmag
            perp = pt_err - np.dot(pt_err, vh) * vh
            v_post = v_post + vmag * perp
        s = np.concatenate([r_ign, v_post, [m_after_tli]])

    # --- diagnostics + ICPS separation -------------------------------------
    icps_prop_used = m0 - s[6]                     # cumulative PRM + TLI
    result["icps_prop_used_kg"] = float(icps_prop_used)
    result["icps_prop_margin_kg"] = float(s[6] - dry_floor)   # mass above the dry floor (allowance-aware;
    #   == ICPS_PROP_KG - icps_prop_used on the impulsive path, so bit-identical there)
    # Delivered TLI ΔV from the rocket equation (TLI burn only, post-PRM mass).
    result["tli_dv_ms"] = float(isp_eff * G0 * np.log(m_pre_tli / s[6]))
    rr = s[:3]; vv = s[3:6]; rn = np.linalg.norm(rr)
    E = 0.5 * float(np.dot(vv, vv)) - MU_EARTH / rn
    result["post_tli_c3_km2s2"] = float(2 * E / 1e6)
    if E < 0:
        a = -MU_EARTH / (2 * E)
        h = np.linalg.norm(np.cross(rr, vv))
        ecc = np.sqrt(max(0.0, 1 - (h*h) / (MU_EARTH * a)))
        result["post_tli_apogee_km"] = float((a*(1+ecc) - R_EARTH) / 1000.0)
    else:
        result["post_tli_apogee_km"] = float("inf")   # escape (shouldn't happen)

    # ICPS separation: the spent ICPS departs, leaving Orion (CM + ESM).
    s[6] = ORION_TOTAL_KG
    result["state"] = s
    result["t_end"] = t
    return result


def _exec_burn_error(dv_vec, units3):
    """Executed Δv for a commanded impulse `dv_vec` under burn-EXECUTION error (nav slice-1):
    cutoff-magnitude error (fixed floor + proportional) along the burn, plus small-angle pointing
    tilt on the two perpendicular axes — scaled by the engine-appropriate σ (OMSe for large burns,
    aux/SMRCS for trims below OTC_EXEC_LARGE_MS, mirroring the real chain's engine selection).
    `units3` = 3 pre-drawn standard normals (mag, tilt-1, tilt-2). Deterministic given the draw."""
    dvm = float(np.linalg.norm(dv_vec))
    if dvm < 1e-9:
        return dv_vec
    if dvm >= OTC_EXEC_LARGE_MS:
        s_mag = OMSE_EXEC_MAG_FLOOR_MS + OMSE_EXEC_MAG_FRAC * dvm
        s_pt = np.deg2rad(OMSE_EXEC_POINT_DEG)
    else:
        s_mag = AUX_EXEC_MAG_FLOOR_MS + AUX_EXEC_MAG_FRAC * dvm
        s_pt = np.deg2rad(AUX_EXEC_POINT_DEG)
    vh = dv_vec / dvm
    a = np.array([0.0, 0.0, 1.0]) if abs(vh[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(vh, a); e1 /= np.linalg.norm(e1)
    e2 = np.cross(vh, e1)
    mag = dvm + float(units3[0]) * s_mag
    return mag * (vh + (float(units3[1]) * e1 + float(units3[2]) * e2) * s_pt)


def phase_outbound_coast(state, t0, perturb=None, duration=None):
    """Translunar coast (Orion alone; ICPS already jettisoned), full 3-body
    integration (Earth+J2+Moon+Sun), with the Outbound Trajectory Correction
    chain when ENABLE_OUTBOUND_MCC, then coast to the outbound lunar close
    approach (handed to phase_outbound_powered_flyby).

    OTC-1 is a Lambert-based targeting correction (ESM burn, charged to ESM
    propellant via the mass) that aims the trajectory at the Moon at the flyby
    epoch — small because the launch azimuth already put the Moon in-plane.
    Returns success, failure_reason, state ([r,v,m] at closest approach), t_end,
    otc1_dv_ms/otc_dv_total_ms, lunar_closest_approach_km/_alt_km.

    ENABLE_OTC_CHAIN (default ON) flies the FAITHFUL Artemis I chain (AAS 23-363): OTC-1 is the MANDATORY OMSe checkout (min on-time
    ≥30 s → the ESM ledger is charged max(solved correction, ~31 m/s floor)), then
    OTC-2/3/4 fire at their real fixed GET slots ONLY if the predicted periselene
    error exceeds OTC_DEADBAND_PERI_KM — the real burns were "added as needed"
    from ground OD; slots in-band are skipped (recorded as 0). OFF = the legacy
    single always-fire targeting OTC (bit-identical; no RNG in either path).
    The aim is at Moon CENTER + B-plane offset (the precise ~130 km flyby B-plane
    aim is the OPF phase's job), so the closest approach lands in the lunar vicinity."""
    perturb = perturb or {}
    result = {"success": True, "failure_reason": None}
    s = np.concatenate([np.asarray(state[:3], float),
                        np.asarray(state[3:6], float), [float(state[6])]])
    t = float(t0)
    t_flyby = OPF_GET_S

    def rhs(tt, y):
        return np.concatenate([y[3:6], gravity_earth_moon(y[:3], tt), [0.0]])

    target_rp = R_MOON + LUNAR_FLYBY_ALT_KM * 1000.0   # ~130 km periselene altitude

    def propagate_to_ca(state_otc, t_start):
        """Coast to lunar closest approach; return (periselene_m, ca_state, t_ca)."""
        t_end_scan = t_flyby + 0.6 * 86400.0
        sol = solve_ivp(rhs, (t_start, t_end_scan), state_otc, method="RK45",
                        rtol=1e-9, atol=1e-1, max_step=600.0, dense_output=True)
        ts = np.linspace(t_start, t_end_scan, 3000)
        Y = sol.sol(ts)
        d = np.array([np.linalg.norm(Y[:3, i] - moon_state(tt)[0]) for i, tt in enumerate(ts)])
        i = int(np.argmin(d))
        lo, hi = ts[max(0, i - 1)], ts[min(len(ts) - 1, i + 1)]
        tf = np.linspace(lo, hi, 400); Yf = sol.sol(tf)
        df = np.array([np.linalg.norm(Yf[:3, k] - moon_state(tt)[0]) for k, tt in enumerate(tf)])
        k = int(np.argmin(df))
        return float(df[k]), Yf[:, k].copy(), float(tf[k])

    otc_dv_total = 0.0
    _chain = globals().get("ENABLE_OTC_CHAIN", False)
    if globals().get("ENABLE_OUTBOUND_MCC", False):
        # OTC-1 ~6 h after TLI (real OTC-1 at GET 07:47). B-plane periselene
        # targeting: a small early correction moves the lunar periselene a lot.
        t_otc1 = t + 6.0 * 3600.0
        try:
            sol = solve_ivp(rhs, (t, t_otc1), s, method="RK45",
                            rtol=1e-9, atol=1e-1, max_step=300.0)
        except Exception as e:
            result.update(success=False, failure_reason=f"otc_coast_error: {e}")
            return result
        s = sol.y[:, -1].copy(); t = t_otc1
        r_moon_fly = moon_state(t_flyby)[0]

        def solve_correction(s_c, t_c, iters):
            """B-plane periselene-targeting Lambert correction from (s_c, t_c).
            1) aim at Moon center, propagate, derive the B-plane direction (the
               miss vector component perpendicular to the lunar approach velocity);
            2) bracket then bisect the offset magnitude to hit target periselene.
            `iters` bisections: 7 = the legacy coarse solve (bit-identical flag-OFF);
            the chain uses 16 (~sub-km B-plane resolution) so a fired trim actually
            settles inside the deadband instead of leaving a ~50 km residual that
            re-fires every later slot. Returns (rp, v1, ca_state, t_ca) or None."""
            def aim_v1(aim):
                lam = lambert_uv(s_c[:3], aim, t_flyby - t_c, mu=MU_EARTH, prograde=True)
                return None if lam is None else lam[0]
            v1c = aim_v1(r_moon_fly)
            if v1c is None:
                return None
            rp0, ca0, tca0 = propagate_to_ca(np.concatenate([s_c[:3], v1c, [s_c[6]]]), t_c)
            vrel = ca0[3:6] - moon_state(tca0)[1]; vrel_hat = vrel / np.linalg.norm(vrel)
            mvec = ca0[:3] - moon_state(tca0)[0]
            b_hat = mvec - np.dot(mvec, vrel_hat) * vrel_hat
            if np.linalg.norm(b_hat) < 1.0:
                b_hat = np.cross(vrel_hat, np.array([0.0, 0.0, 1.0]))
            b_hat = b_hat / np.linalg.norm(b_hat)

            def solve_b(b):
                v1 = aim_v1(r_moon_fly + b * b_hat)
                if v1 is None:
                    return None
                rp, ca, tca = propagate_to_ca(np.concatenate([s_c[:3], v1, [s_c[6]]]), t_c)
                return rp, v1, ca, tca

            best = (rp0, v1c, ca0, tca0)
            b_hi = 6_000e3
            res_hi = solve_b(b_hi); grow = 0
            while res_hi is not None and res_hi[0] < target_rp and grow < 6:
                b_hi *= 1.7; res_hi = solve_b(b_hi); grow += 1
            if res_hi is not None and res_hi[0] >= target_rp:
                best = res_hi
                b_lo = 0.0
                for _ in range(iters):
                    b_mid = 0.5 * (b_lo + b_hi)
                    res = solve_b(b_mid)
                    if res is None:
                        break
                    if res[0] < target_rp:
                        b_lo = b_mid
                    else:
                        b_hi = b_mid; best = res
            return best

        def slot_trim(s_c, t_c, rp_cur):
            """GTLT-style DIFFERENTIAL trim for the OTC-2/3/4 slots: the MINIMUM-norm Δv nulling
            the predicted periselene-radius error. (A Lambert RE-PLAN here charges the accumulated
            conic/3-body mismatch + a fixed arrival time — 30–860 m/s of fictitious ΔV vs the real
            0.2–0.9 m/s trims. The real corrections were differential, not re-plans.) FD sensitivity
            of the full-dynamics periselene to Δv in the local (along, cross, radial) basis, min-norm
            Newton, ≤3 passes, converge to half the deadband. Returns (commanded_dv_vec, (rp,ca,tca))."""
            vt = np.zeros(3); out = None; rp_err = rp_cur - target_rp
            for _p in range(3):
                if abs(rp_err) <= 0.5 * OTC_DEADBAND_PERI_KM * 1000.0:
                    break
                rh = s_c[:3] / np.linalg.norm(s_c[:3])
                vh = s_c[3:6] / np.linalg.norm(s_c[3:6])
                hh = np.cross(rh, vh); hh /= np.linalg.norm(hh)
                basis = (vh, hh, rh)
                _d = 0.02                                    # m/s FD step
                g = np.zeros(3)
                for _k, _e in enumerate(basis):
                    sp = np.concatenate([s_c[:3], s_c[3:6] + _d * _e, [s_c[6]]])
                    rp_k, _ca, _tc = propagate_to_ca(sp, t_c)
                    g[_k] = (rp_k - (rp_err + target_rp)) / _d   # d(periselene)/d(dv_k)  [m per m/s]
                gn2 = float(g @ g)
                if gn2 < 1e-6:
                    break
                step = (-rp_err / gn2) * g                   # min-norm solve of g·Δ = -rp_err
                dvv = step[0] * basis[0] + step[1] * basis[1] + step[2] * basis[2]
                s_c = np.concatenate([s_c[:3], s_c[3:6] + dvv, [s_c[6]]])
                vt = vt + dvv
                rp_new, ca_n, tca_n = propagate_to_ca(s_c, t_c)
                out = (rp_new, ca_n, tca_n)
                rp_err = rp_new - target_rp
            return vt, out

        # NAV slice-2: the ground SOLVES + GATES on an ESTIMATED state (true + per-slot RIC knowledge
        # error, σ shrinking with the tracking arc); the commanded Δv then executes on the TRUE state.
        _od_u = (np.asarray(perturb["otc_od_units"], float)
                 if (_chain and globals().get("ENABLE_OTC_OD_ERRORS", False)
                     and perturb.get("otc_od_units") is not None) else None)

        def _estimated(s_c, slot):
            """Ground-OD estimated state at a slot: true + the slot's knowledge error. OD-filter ON ->
            chol(P_ric) @ unit (emergent, correlated); OFF -> the diagonal σ table (bit-identical)."""
            if _od_u is None:
                return s_c
            _L = _od_filter_L(f"otc{slot}")
            if _L is not None:
                err = _L @ np.asarray(_od_u[slot], float)
            else:
                err = np.concatenate([_od_u[slot, :3] * np.asarray(OTC_OD_POS_SIGMA_M[slot], float),
                                      _od_u[slot, 3:] * np.asarray(OTC_OD_VEL_SIGMA_MS[slot], float)])
            se = s_c.copy(); se[:6] = se[:6] + _clnav_inertial_error(se[:6], err)
            return se

        def _prop_to_epoch(s_c, t_c, t_tgt):
            sol_e = solve_ivp(rhs, (t_c, t_tgt), s_c, method="RK45",
                              rtol=1e-9, atol=1e-1, max_step=600.0)
            return sol_e.y[:, -1]

        def _bplane_miss(s_c, t_c):
            """The classical B-plane miss at the trial's OWN closest approach: the offset of the
            trial's CA point from the real CA point (PHASEC_CA_R_ECI_M), projected ⊥ the flyby-
            relative velocity AT CA. Geometry fully constrained (both B-plane components), arrival
            TIME genuinely free. (v6 lesson: perpendicular-at-a-FIXED-EPOCH is NOT the B-plane —
            it over-relaxed the geometry and scattered periselene 12–11,334 km; v5 lesson: pinning
            the fixed-epoch 3-vec makes timing a constraint and goes near-singular on tails —
            1,757 km miss demanded 3,800–321,000 m/s raw Newton.)"""
            # MOON-RELATIVE comparison (v7 lesson): with timing free, a fixed INERTIAL target is
            # wrong — the Moon moves ~1 km/s, so a trial arriving 30 min late finds the corridor
            # displaced ~1,800 km inertially (v7: 12 impacts, CA scattered to 8,500 km). Compare the
            # trial's CA relative to the Moon AT ITS arrival vs the real CA relative to the Moon AT
            # THE REAL arrival — the classical Moon-frame B-plane.
            # FULL Moon-relative CA-point difference, NO projection (v11): the ⊥-v̂ projection had a
            # DEGENERACY — when (ρ_ca − ρ_tgt) happens to lie along v̂, a huge miss projects to ~zero
            # (trial 7: corrector satisfied at trims≈0 while the true CA sat 1,583 km SUBSURFACE).
            # With the trial's own CA floating, timing is still free; the along-track component is
            # naturally weak/uncontrollable and min-norm lstsq (rcond) leaves it — but the metric can
            # no longer be fooled: a subsurface CA is always ≥~1,700 km from the target.
            _rho_tgt = (np.asarray(globals()["PHASEC_CA_R_ECI_M"], float)
                        - moon_state(float(globals()["PHASEC_CA_T_S"]))[0])
            _rp, _ca, _tca = propagate_to_ca(np.asarray(s_c, float), t_c)
            return ((_ca[:3] - moon_state(_tca)[0]) - _rho_tgt), _rp

        def _trim_attempt(s0, t_c, _cap, _abort):
            """One trust-region Newton attempt at the given step cap / total abort. Improve-or-halve
            guarded; returns (dv_vec, final_miss_m). No-ops (zeros) past the abort envelope.
            PERISELENE-FLOOR guard (RETRY coasts only — _TLI_RETRY_COAST, set by run_mission around
            the post-replan coast): the CA-point metric can be REDUCED by steps whose actual periselene
            dives sub-surface (the improve-or-halve accepts a ~1,500-km residual stop that impacts —
            3/12 in the round-5 A/B); reject candidate steps predicting periselene below the floor.
            Scoped to retries (validation lesson: a fleet-wide guard perturbs marginal FIRST-PASS chains
            — trial 156 flipped success→depletion without ever retrying); first-pass trials are
            bit-identical to flag-OFF by construction."""
            _guard = globals().get("_TLI_RETRY_COAST", False)
            _PFLOOR = R_MOON + 20_000.0
            s_c = np.asarray(s0, float).copy()
            vt = np.zeros(3)
            miss, _rp_cur = _bplane_miss(s_c, t_c)
            for _p in range(6):
                mn = float(np.linalg.norm(miss))
                if mn <= 0.5 * OTC_DEADBAND_PERI_KM * 1000.0:
                    break
                J = np.zeros((3, 3)); _d = 0.02
                for _k in range(3):
                    e = np.zeros(3); e[_k] = _d
                    sp = np.concatenate([s_c[:3], s_c[3:6] + e, [s_c[6]]])
                    J[:, _k] = (_bplane_miss(sp, t_c)[0] - miss) / _d
                dvv, *_ = np.linalg.lstsq(J, -miss, rcond=1e-6)   # min-norm (2 constraints in 3-space)
                _dn = float(np.linalg.norm(dvv))
                if _dn > _cap:
                    dvv = dvv * (_cap / _dn)
                s_try = np.concatenate([s_c[:3], s_c[3:6] + dvv, [s_c[6]]])
                miss_try, rp_try = _bplane_miss(s_try, t_c)
                def _reject(_m, _rp):
                    return (np.linalg.norm(_m) >= mn) or (_guard and _rp < _PFLOOR and _rp < _rp_cur)
                if _reject(miss_try, rp_try):                 # step didn't improve (or dove sub-floor): halve once
                    dvv = 0.5 * dvv
                    s_try = np.concatenate([s_c[:3], s_c[3:6] + dvv, [s_c[6]]])
                    miss_try, rp_try = _bplane_miss(s_try, t_c)
                    if _reject(miss_try, rp_try):             # still worse: stop (keep current)
                        break
                s_c = s_try; vt = vt + dvv; miss = miss_try; _rp_cur = rp_try
            if float(np.linalg.norm(vt)) > _abort:
                return np.zeros(3), float(np.linalg.norm(miss))
            return vt, float(np.linalg.norm(miss))

        def ca_point_trim(s_c, t_c):
            """Phase-C targeting with a RETRY LADDER (v9): the standard attempt (50 m/s steps,
            300 abort), then — only if still out of band — a heavy-dispersion attempt from the
            original state (150 m/s steps, 600 abort; a real ops team facing a 3σ launch trial
            re-plans with whatever ΔV it takes, budget permitting). Returns the best attempt."""
            best = None
            for _cap, _abort in ((50.0, 300.0), (150.0, 600.0)):
                vt, mn = _trim_attempt(s_c, t_c, _cap, _abort)
                if best is None or mn < best[1]:
                    best = (vt, mn)
                if best[1] <= OTC_DEADBAND_PERI_KM * 1000.0:
                    break
            return best

        _od_capture("otc0", t, s)          # OTC-1 epoch (nominal build)
        s_est1 = _estimated(s, 0)
        _phasec = globals().get("ENABLE_PHASEC_BPLANE", False)
        if _chain and _phasec:
            # Phase-C: the forced TLI already flies the REAL lunar approach — OTC-1 must trim it
            # DIFFERENTIALLY to the REAL ARRIVAL POINT (ca_point_trim), NOT Lambert-re-plan it: the
            # re-plan re-targets the chain's own Moon+offset aim, destroying the re-aimed geometry
            # (arrival reverted 5.26 -> 5.08 d; insertion quadrupled).
            vt_cmd, _miss1 = ca_point_trim(s_est1, t)
            v1_f = s_est1[3:6] + vt_cmd
        else:
            best = solve_correction(s_est1, t, 16 if _chain else 7)
            if best is None:
                result.update(success=False, failure_reason="otc_no_lambert", state=s, t_end=t)
                return result
            rp_f, v1_f, ca_f, tca_f = best
        dv = float(np.linalg.norm(v1_f - s_est1[3:6]))   # commanded Δv (in the ground's estimate frame)
        if _chain:
            # OTC-1 is the MANDATORY OMSe CHECKOUT (∆t ≥ 30 s): charge the ledger at least the
            # min-on-time ΔV (~26.6 kN / m × 30 s ≈ 31 m/s; as-flown 35.0 m/s). The checkout's
            # excess over the solved correction was a designed-in trajectory component for the
            # real mission (Copernicus re-optimized WITH it); the sim flies the solved correction
            # and charges the honest propellant.
            # NAV slice-1: the burn EXECUTES imperfectly (cutoff magnitude + pointing,
            # _exec_burn_error) — the re-dispersion the later slots exist to trim.
            _exec_u = (np.asarray(perturb["otc_exec_units"], float)
                       if (globals().get("ENABLE_OTC_EXEC_ERRORS", False)
                           and perturb.get("otc_exec_units") is not None) else None)
            dv_vec = v1_f - s_est1[3:6]                  # the ground's commanded Δv vector
            if _exec_u is not None:
                dv_vec = _exec_burn_error(dv_vec, _exec_u[0])
            v_applied = s[3:6] + dv_vec                  # executes on the TRUE velocity
            dv_exec = float(np.linalg.norm(dv_vec))
            dv_charged = max(dv_exec, OMSE_THRUST_N / max(s[6], 1.0) * OTC1_MIN_ONTIME_S)
            result["otc1_corr_dv_ms"] = dv
        else:
            dv_charged = dv
        m_post = s[6] * np.exp(-dv_charged / (OMSE_ISP_S * G0))   # ESM propellant via mass
        otc_dv_total += dv_charged
        result["otc1_dv_ms"] = dv_charged
        if _chain:
            # Fly ONWARD through the fixed OTC-2/3/4 GET slots (no CA jump): at each real slot,
            # predict the periselene from the trial's actual trajectory and fire a trim ONLY if
            # the error exceeds the deadband — the real burns were "added as needed" from ground
            # OD (OTC-3/4 design ΔV = 0, as-flown 0.87/0.22 m/s). Slot predictions use the TRUE
            # post-execution state (knowledge error is nav slice-2). RNG: spawned units only.
            s = np.concatenate([s[:3], v_applied, [m_post]])
            if _phasec:
                # differential-trim branch has no solve CA — bookkeep the post-burn TRUE trajectory
                last_rp, last_ca, last_tca = propagate_to_ca(s, t)
            else:
                last_rp, last_ca, last_tca = rp_f, ca_f, tca_f
            for _i, t_slot in enumerate(OTC_SLOT_GETS_S, start=2):
                if t_slot <= t or t_slot >= t_flyby - 1800.0:
                    result[f"otc{_i}_dv_ms"] = 0.0
                    continue
                try:
                    sol = solve_ivp(rhs, (t, t_slot), s, method="RK45",
                                    rtol=1e-9, atol=1e-1, max_step=600.0)
                except Exception as e:
                    result.update(success=False, failure_reason=f"otc_coast_error: {e}")
                    return result
                s = sol.y[:, -1].copy(); t = t_slot
                _od_capture(f"otc{_i - 1}", t, s)              # OTC-2/3/4 epoch (nominal build)
                s_est = _estimated(s, _i - 1)                  # the ground's OD estimate at this slot
                if _phasec:
                    # Phase-C slots gate + trim on the B-PLANE miss at the trial's own CA vs the
                    # real arrival point (geometry constrained, timing free)
                    _mv, _ = _bplane_miss(s_est, t)
                    if np.linalg.norm(_mv) <= OTC_DEADBAND_PERI_KM * 1000.0:
                        result[f"otc{_i}_dv_ms"] = 0.0         # in-band -> slot skipped (as real)
                        continue
                    vt_cmd, _ = ca_point_trim(s_est, t)
                else:
                    rp_p, ca_p, tca_p = propagate_to_ca(s_est, t)  # predicted CA (from the ESTIMATE)
                    last_rp, last_ca, last_tca = rp_p, ca_p, tca_p
                    if abs(rp_p - target_rp) <= OTC_DEADBAND_PERI_KM * 1000.0:
                        result[f"otc{_i}_dv_ms"] = 0.0         # in-band -> slot skipped (as real)
                        continue
                    vt_cmd, tr_out = slot_trim(s_est, t, rp_p)  # trim solved on the ESTIMATE
                    if tr_out is None:
                        result[f"otc{_i}_dv_ms"] = 0.0         # corrector no-op -> slot skipped
                        continue
                if float(np.linalg.norm(vt_cmd)) < 1e-6:
                    result[f"otc{_i}_dv_ms"] = 0.0             # corrector no-op -> slot skipped
                    continue
                dv_vec_t = vt_cmd
                if _exec_u is not None:
                    dv_vec_t = _exec_burn_error(dv_vec_t, _exec_u[_i - 1])   # trim executes imperfectly
                dv_t = float(np.linalg.norm(dv_vec_t))
                s = np.concatenate([s[:3], s[3:6] + dv_vec_t,
                                    [s[6] * np.exp(-dv_t / (OMSE_ISP_S * G0))]])
                otc_dv_total += dv_t
                result[f"otc{_i}_dv_ms"] = dv_t
                # bookkeeping CA must reflect the EXECUTED burn (the corrector assumed exact
                # execution) — re-propagate from the post-burn state so the final hand-off and
                # any later slot's deadband see the true trajectory
                rp_t, ca_t, tca_t = propagate_to_ca(s, t)
                last_rp, last_ca, last_tca = rp_t, ca_t, tca_t
            if _od_u is not None or _phasec:
                # the bookkept CA above may be the ESTIMATE's prediction (skipped slots) — or, in
                # Phase-C, a fixed-epoch miss with no CA bookkeeping at all — the hand-off must be
                # the TRUE trajectory's closest approach, so re-propagate the true state once
                last_rp, last_ca, last_tca = propagate_to_ca(s, t)
            # closest approach = the last prediction/solve (already propagated in full dynamics)
            _m_cur = s[6]
            s = last_ca.copy(); s[6] = _m_cur; t = last_tca; ca_m = last_rp
        else:
            # carry the OTC mass reduction onto the closest-approach state
            s = ca_f.copy(); s[6] = m_post; t = tca_f; ca_m = rp_f
    else:
        ca_m, s, t = propagate_to_ca(s, t)
    result["otc_dv_total_ms"] = float(otc_dv_total)

    ca_km = ca_m / 1000.0
    result["lunar_closest_approach_km"] = ca_km
    result["lunar_closest_alt_km"] = ca_km - R_MOON / 1000.0
    if ca_km > 66_000.0:        # never entered the lunar sphere of influence
        result.update(success=False, failure_reason="missed_lunar_approach",
                      state=s, t_end=t)
        return result
    if ca_m < R_MOON:           # periselene below the surface — a lunar-impact trajectory the trim
        # chain could not rescue (phase-5 honesty guard: the OPF must not "burn" from inside the
        # Moon; heavy launch/TLI-dispersion tails end here as an honest failure)
        result.update(success=False, failure_reason="lunar_impact_trajectory",
                      state=s, t_end=t)
        return result
    result["state"] = s
    result["t_end"] = t
    return result


# ============================================================
# Snap-free OPF->DRO rendezvous (ENABLE_SNAPFREE_DRO) — kills the ~127,000 km DRI
# capture "snap". The OPF burn is SOLVED so the post-burn ballistic arc arrives AT a
# real CR3BP DRO state (dro_state_eci) at the insertion epoch; DRI is then the small
# GENUINE velocity match (no teleport onto the orbit). Decoupled from the near-polar
# return: inserts at the cheap basin phase SNAPFREE_DRO_PHASE and the general flown
# return (_solve_flown_return) closes an IN-PLANE (off-Africa) return — the splash
# region + mission timeline are the deferred follow-up. Validated: the basin solve
# closes position to 0 km at OPF ~274 + DRI ~38 m/s; the inserted state is a bounded
# retrograde DRO (61-94k km, stable over the coast); the general return closes (EI FPA
# -5.9, in corridor). OFF = the legacy capture-snap DRO (bit-identical to the pre-flag lineage).
# ============================================================
def _coast_rv(s6, t0, t1):
    """Ballistic coast of a 6-state [r,v] under full Earth+Moon gravity, t0 -> t1."""
    sol = solve_ivp(lambda tt, y: np.concatenate([y[3:6], gravity_earth_moon(y[:3], tt)]),
                    (t0, t1), np.asarray(s6, float), method="RK45",
                    rtol=1e-9, atol=1e-1, max_step=1800.0)
    return sol.y[:, -1]


def _snapfree_cfg():
    """(phi, t4, opf_seed) for the ACTIVE snap-free config: the Phase C real-approach target when
    ENABLE_PHASEC_BPLANE, else the legacy basin. Keeps both paths in one place."""
    if globals().get("ENABLE_PHASEC_BPLANE", False):
        seed = globals().get("PHASEC_OPF_SEED_MS")
        return (float(globals().get("PHASEC_DRO_PHASE", 0.75)),
                float(globals().get("PHASEC_DRI_GET_S", 9.65864583 * 86400.0)),
                np.asarray(seed if seed is not None else (200.0, 0.0, 0.0), float))
    return (float(globals().get("SNAPFREE_DRO_PHASE", 0.222)),
            float(globals().get("SNAPFREE_DRI_GET_S", 7.89 * 86400.0)),
            np.asarray(globals().get("SNAPFREE_OPF_SEED_MS", (263.25, -34.35, -67.28)), float))


def _snapfree_active():
    # Snap-free DRO is gated by ENABLE_SNAPFREE_DRO ALONE (decoupled from ENABLE_PHASEC_BPLANE,
    # which now gates ONLY the launch-azimuth + TLI re-aim). Orthogonal flags: the FULL Phase-C stack =
    # BOTH True; ENABLE_PHASEC_BPLANE alone tests the phase-1/2 TLI re-aim with the legacy DRO + teleport
    # intact. Default-preserving (both default False -> unchanged).
    return globals().get("ENABLE_SNAPFREE_DRO", False)


def _solve_opf_to_dro(peri6, t_p, r_target, t4, seed_dv):
    """Solve the 3-DOF periselene burn so the ballistic coast peri->t4 arrives AT the DRO
    position r_target (snap-free). LM-damped Newton, warm-started from seed_dv (the basin
    OPF burn). Returns dv (3,) in m/s, or None if it cannot close position to <50 km."""
    dv = np.asarray(seed_dv, float).copy()
    def arr_of(d):
        return _coast_rv(np.concatenate([peri6[:3], peri6[3:6] + d]), t_p, t4)[:3]
    arr = arr_of(dv); res = arr - r_target; rn = np.linalg.norm(res); lam = 1e-2
    for _ in range(20):
        if rn < 5000.0:
            return dv
        J = np.zeros((3, 3))
        for k in range(3):
            d2 = dv.copy(); d2[k] += 1.0
            J[:, k] = (arr_of(d2) - arr) / 1.0
        JTJ = J.T @ J; Jtr = J.T @ res; stepped = False
        for _ in range(6):
            try:
                step = np.linalg.solve(JTJ + lam * np.diag(np.diag(JTJ) + 1e-30), Jtr)
            except np.linalg.LinAlgError:
                lam *= 4; continue
            dv2 = dv - step; arr2 = arr_of(dv2); rn2 = np.linalg.norm(arr2 - r_target)
            if rn2 < rn:
                dv, arr, res, rn = dv2, arr2, arr2 - r_target, rn2
                lam = max(lam / 3, 1e-9); stepped = True; break
            lam *= 4
        if not stepped:
            break
    return dv if rn < 50000.0 else None


def _opf_snapfree(s, t, perturb, result):
    """OPF as a snap-free DRO-targeting burn: solve (nominal) / apply cached (perturbed,
    open-loop — the DRI then reabsorbs the small dispersion via OD-nav)."""
    phi, t4, seed = _snapfree_cfg()
    r_target = np.asarray(dro_state_eci(phi, t4)[0], float)
    nt = globals().get("_NOMINAL_TARGETS")
    if nt is None:
        nt = {}; globals()["_NOMINAL_TARGETS"] = nt
    if not perturb:
        # NOMINAL solve with BRANCH CONTROL (v10 lesson): the solve space has coexisting branches
        # (190 vs 395 vs 500+ m/s from the same arrival — the φ-sweep measured them); taking the
        # FIRST convergent seed let a microscopic nominal-arrival shift flip the fleet onto a +200
        # m/s branch (every perturbed trial warm-starts from this cached vector). Evaluate the FULL
        # seed ladder and keep the CHEAPEST converged (OPF + resulting DRI) solution.
        _vrel = s[3:6] - moon_state(t)[1]; _vh = _vrel / max(np.linalg.norm(_vrel), 1e-9)
        _r_dro4, _v_dro4 = dro_state_eci(phi, t4)
        _r_dro4 = np.asarray(_r_dro4, float); _v_dro4 = np.asarray(_v_dro4, float)
        dv = None; _best_cost = None
        for _sd in (np.asarray(seed, float), 150.0 * _vh, -150.0 * _vh,
                    300.0 * _vh, -300.0 * _vh, np.zeros(3)):
            _dvc = _solve_opf_to_dro(s[:6], t, r_target, t4, _sd)
            if _dvc is None:
                continue
            _arr = _coast_rv(np.concatenate([s[:3], s[3:6] + _dvc]), t, t4)
            if np.linalg.norm(_arr[:3] - _r_dro4) > 500e3:
                continue
            _cost = float(np.linalg.norm(_dvc)) + float(np.linalg.norm(_v_dro4 - _arr[3:6]))
            if _best_cost is None or _cost < _best_cost:
                _best_cost = _cost; dv = _dvc
        if dv is None:
            result.update(success=False, failure_reason="opf_snapfree_no_solution", state=s, t_end=t)
            return result
        nt["opf_snapfree_dv"] = dv.copy()
    else:
        # CLOSED-LOOP per-trial re-solve (fleet lesson): flying the cached NOMINAL burn
        # open-loop from a dispersed arrival broke 20% of trials (bad post-OPF orbits -> ESM drain).
        # The real GTLT re-solved the OPF from tracking; do the same, warm-started from the nominal.
        _seed0 = np.asarray(nt.get("opf_snapfree_dv", seed), float)
        dv = _solve_opf_to_dro(s[:6], t, r_target, t4, _seed0)
        if dv is None:
            dv = _seed0                                     # fall back to the open-loop cached burn
        # FLEX-DRO FEASIBILITY DIAGNOSTIC (log-only, does NOT change the flown burn): for this
        # dispersed arrival, does an AFFORDABLE flexible insertion (φ, t) exist, vs the fixed-schedule
        # (φ0, t4) it's charged? Sweep φ×t, re-solve the OPF to each candidate, compute total OPF+DRI
        # velocity-match; log the min. Gated OFF by default (AR1_DRO_FLEX_DIAG) — pure feasibility probe.
        if globals().get("_DRO_FLEX_DIAG", False):
            _best_tot = None; _best = None
            _phi0, _t40, _ = _snapfree_cfg()
            for _phc in np.linspace(_phi0 - 0.15, _phi0 + 0.15, 9):
                _phc = float(_phc % 1.0)
                for _dt in (-0.75 * 86400.0, -0.35 * 86400.0, 0.0, 0.35 * 86400.0, 0.75 * 86400.0):
                    _tc = t4 + _dt
                    if _tc <= t + 3600.0:
                        continue
                    _rc, _vc = dro_state_eci(_phc, _tc)
                    _rc = np.asarray(_rc, float); _vc = np.asarray(_vc, float)
                    _dvc = _solve_opf_to_dro(s[:6], t, _rc, _tc, np.asarray(dv, float))
                    if _dvc is None:
                        continue
                    _arrc = _coast_rv(np.concatenate([s[:3], s[3:6] + _dvc]), t, _tc)
                    if np.linalg.norm(_arrc[:3] - _rc) > 3000e3:
                        continue
                    _tot = float(np.linalg.norm(_dvc)) + float(np.linalg.norm(_vc - _arrc[3:6]))
                    if _best_tot is None or _tot < _best_tot:
                        _best_tot = _tot; _best = (_phc, _dt / 86400.0,
                                                   float(np.linalg.norm(_dvc)), _tot - float(np.linalg.norm(_dvc)))
            # baseline: the FIXED-schedule cost this trial actually pays (OPF |dv| + DRI at φ0/t4)
            _arr0 = _coast_rv(np.concatenate([s[:3], s[3:6] + np.asarray(dv, float)]), t, t4)
            _rd0, _vd0 = dro_state_eci(_phi0, t4)
            _fixed_tot = float(np.linalg.norm(dv)) + float(np.linalg.norm(np.asarray(_vd0, float) - _arr0[3:6]))
            result["dro_flex_fixed_total_ms"] = _fixed_tot
            _log = {"fixed_total_ms": _fixed_tot}
            if _best is not None:
                result["dro_flex_min_total_ms"] = _best_tot
                result["dro_flex_best_phi"] = _best[0]
                result["dro_flex_best_dt_d"] = _best[1]
                result["dro_flex_best_opf_ms"] = _best[2]
                result["dro_flex_best_dri_ms"] = _best[3]
                _log.update(min_total_ms=_best_tot, best_phi=_best[0], best_dt_d=_best[1],
                            best_opf_ms=_best[2], best_dri_ms=_best[3])
            globals()["_DRO_FLEX_LAST"] = _log       # per-worker: harness reads after run_mission
    # execution error: scalar bias along the burn direction (consumes opf_dv_bias_ms)
    dv_exec = dv + perturb.get("opf_dv_bias_ms", 0.0) * dv / max(np.linalg.norm(dv), 1e-6)
    dvmag = float(np.linalg.norm(dv_exec))
    isp = OMSE_ISP_S * perturb.get("omse_isp_factor", 1.0)
    m_after = s[6] * np.exp(-dvmag / (isp * G0))
    if m_after < ORION_TOTAL_KG - ESM_PROP_KG:
        result.update(success=False, failure_reason="esm_propellant_depleted_opf", state=s, t_end=t)
        return result
    result.update(opf_dv_ms=dvmag, esm_prop_used_kg=float(s[6] - m_after),
                  state=np.concatenate([s[:3], s[3:6] + dv_exec, [m_after]]), t_end=t)
    return result


def _transit_slot_trim(s6, t0, r_tgt, t_tgt):
    """Trust-region 3×3 FD Newton nulling r(t_tgt) − r_tgt from (s6, t0), ballistic arcs
    (_coast_rv). The phase-5 fixed-epoch corrector for the post-OPF OTC-5/6 slots — same
    safeguards as the outbound ca_point_trim (30 m/s/step cap, improve-or-halve, 100 m/s
    total abort-to-no-op). Returns (dv_vec, final_miss_m)."""
    s_c = np.asarray(s6, float).copy(); vt = np.zeros(3)
    _se = _coast_rv(s_c, t0, t_tgt)
    _vh = _se[3:6] / max(np.linalg.norm(_se[3:6]), 1e-9)
    def _perp(v):
        return v - np.dot(v, _vh) * _vh
    miss = _perp(_se[:3] - r_tgt)
    for _p in range(6):
        mn = float(np.linalg.norm(miss))
        if mn <= 0.5 * OTC_DEADBAND_PERI_KM * 1000.0:
            break
        J = np.zeros((3, 3)); _d = 0.02
        for _k in range(3):
            e = np.zeros(3); e[_k] = _d
            J[:, _k] = (_perp(_coast_rv(np.concatenate([s_c[:3], s_c[3:6] + e]), t0, t_tgt)[:3]
                        - r_tgt) - miss) / _d
        dvv, *_ = np.linalg.lstsq(J, -miss, rcond=1e-6)   # min-norm on the projected system
        _dn = float(np.linalg.norm(dvv))
        if _dn > 50.0:
            dvv = dvv * (50.0 / _dn)
        s_try = np.concatenate([s_c[:3], s_c[3:6] + dvv])
        miss_try = _perp(_coast_rv(s_try, t0, t_tgt)[:3] - r_tgt)
        if np.linalg.norm(miss_try) >= mn:
            dvv = 0.5 * dvv
            s_try = np.concatenate([s_c[:3], s_c[3:6] + dvv])
            miss_try = _perp(_coast_rv(s_try, t0, t_tgt)[:3] - r_tgt)
            if np.linalg.norm(miss_try) >= mn:
                break
        s_c = s_try; vt = vt + dvv; miss = miss_try
    if float(np.linalg.norm(vt)) > 300.0:
        return np.zeros(3), float(np.linalg.norm(miss))
    return vt, float(np.linalg.norm(miss))


def _dri_snapfree(s, t, perturb, result):
    """DRI as the GENUINE velocity match at the snap-free DRO arrival (no teleport). Fly the
    post-OPF transit through the REAL OTC-5/6 conditional trim slots (deadband-gated, the real
    chain's dispersion absorber — as-flown 0.98/2.68 m/s), coast to the insertion epoch, record
    the residual snap, charge the velocity-match (OD-nav: nominal-cost on schedule for IN-BAND
    arrivals only), insert onto the DRO."""
    phi, t4, _ = _snapfree_cfg()
    r_dro, v_dro = dro_state_eci(phi, t4)
    r_dro = np.asarray(r_dro, float); v_dro = np.asarray(v_dro, float)
    if globals().get("ENABLE_OTC_CHAIN", False):
        _pb = perturb or {}
        _eu = _pb.get("otc_exec_units"); _ou = _pb.get("otc_od_units")
        _isp56 = OMSE_ISP_S * _pb.get("omse_isp_factor", 1.0)
        s6c = np.asarray(s[:6], float).copy(); m_c = float(s[6]); t_c = float(t)
        for _j, t_slot in enumerate(OTC56_SLOT_GETS_S, start=5):
            if t_slot <= t_c or t_slot >= t4 - 3600.0:
                result[f"otc{_j}_dv_ms"] = 0.0
                continue
            s6c = _coast_rv(s6c, t_c, t_slot); t_c = float(t_slot)
            _od_capture(f"otc{_j - 1}", t_slot, s6c)           # OTC-5/6 epoch (nominal build)
            s_est = s6c.copy()
            if (_ou is not None and globals().get("ENABLE_OTC_OD_ERRORS", False)
                    and len(np.atleast_2d(_ou)) > _j - 1):
                _Lo = _od_filter_L(f"otc{_j - 1}")
                if _Lo is not None:
                    _err = _Lo @ np.asarray(_ou)[_j - 1]       # OD-filter: chol(P_ric) @ unit (emergent)
                else:
                    _err = np.concatenate([np.asarray(_ou)[_j-1, :3] * np.asarray(OTC_OD_POS_SIGMA_M[_j-1], float),
                                           np.asarray(_ou)[_j-1, 3:] * np.asarray(OTC_OD_VEL_SIGMA_MS[_j-1], float)])
                s_est = s_est + _clnav_inertial_error(s_est, _err)
            _sep0 = _coast_rv(s_est, t_c, t4)
            _vh0 = _sep0[3:6] / max(np.linalg.norm(_sep0[3:6]), 1e-9)
            _miss0 = _sep0[:3] - r_dro
            _miss0 = _miss0 - np.dot(_miss0, _vh0) * _vh0     # ⊥ flight direction (timing floats)
            if np.linalg.norm(_miss0) <= OTC_DEADBAND_PERI_KM * 1000.0:
                result[f"otc{_j}_dv_ms"] = 0.0
                continue
            _dvv, _ = _transit_slot_trim(s_est, t_c, r_dro, t4)
            if (_eu is not None and globals().get("ENABLE_OTC_EXEC_ERRORS", False)
                    and len(np.atleast_2d(_eu)) > _j - 1 and np.linalg.norm(_dvv) > 1e-9):
                _dvv = _exec_burn_error(_dvv, np.asarray(_eu)[_j-1])
            _dvm = float(np.linalg.norm(_dvv))
            if _dvm < 1e-9:
                result[f"otc{_j}_dv_ms"] = 0.0
                continue
            _m_new = m_c * np.exp(-_dvm / (_isp56 * G0))
            if _m_new < ORION_TOTAL_KG - ESM_PROP_KG:
                result.update(success=False, failure_reason="esm_propellant_depleted_otc56",
                              state=np.concatenate([s6c, [m_c]]), t_end=t_c)
                return result
            s6c = np.concatenate([s6c[:3], s6c[3:6] + _dvv]); m_c = _m_new
            result[f"otc{_j}_dv_ms"] = _dvm
        s = np.concatenate([s6c, [m_c]]); t = t_c
    arr = _coast_rv(s[:6], t, t4)
    snap_km = float(np.linalg.norm(arr[:3] - r_dro) / 1000.0)
    dv = float(np.linalg.norm(v_dro - arr[3:6]) + perturb.get("dri_dv_bias_ms", 0.0))
    nt = globals().get("_NOMINAL_TARGETS")
    if nt is None:
        nt = {}; globals()["_NOMINAL_TARGETS"] = nt
    if not perturb:
        nt["dri_dv_nominal_sf"] = dv
    elif (globals().get("ENABLE_OD_NAV", False) and "dri_dv_nominal_sf" in nt
          and snap_km < 5000.0):
        # insert on schedule at nominal cost — ONLY for IN-BAND arrivals (one dispersed
        # trial arrived 476,000 km off yet "inserted" at nominal cost — a residual mini-teleport).
        # Out-of-band arrivals pay the TRUE velocity match below, which for genuine misses depletes
        # the ESM -> an HONEST insertion failure instead of a hidden rescue.
        dv = float(nt["dri_dv_nominal_sf"] + perturb.get("dri_dv_bias_ms", 0.0))
    isp = OMSE_ISP_S * perturb.get("omse_isp_factor", 1.0)
    prop = s[6] * (1.0 - np.exp(-dv / (isp * G0)))
    if s[6] - prop < ORION_TOTAL_KG - ESM_PROP_KG:
        result.update(success=False, failure_reason="esm_propellant_depleted_dri", state=s, t_end=t4)
        return result
    _od_capture("dri", t4, np.concatenate([r_dro, v_dro]))   # DRI epoch (DRO long-arc OD; nominal build)
    r_ins, v_ins = r_dro.copy(), v_dro.copy()
    if perturb and globals().get("ENABLE_OD_NAV_RESIDUAL", False):
        _Ld = _od_filter_L("dri")
        if _Ld is not None:
            # OD-filter: recover the unit draw (residual / σ), apply chol(P_eci) @ unit (emergent, ECI)
            _u6 = np.concatenate([np.asarray(perturb.get("dri_od_pos_residual_m", np.zeros(3)), float) / OD_NAV_POS_SIGMA_M,
                                  np.asarray(perturb.get("dri_od_vel_residual_ms", np.zeros(3)), float) / OD_NAV_VEL_SIGMA_MS])
            _e6 = _Ld @ _u6
            r_ins = r_ins + _e6[:3]; v_ins = v_ins + _e6[3:6]
        else:
            r_ins = r_ins + np.asarray(perturb.get("dri_od_pos_residual_m", np.zeros(3)), float)
            v_ins = v_ins + np.asarray(perturb.get("dri_od_vel_residual_ms", np.zeros(3)), float)
    dro = _compute_cr3bp_dro()
    result.update(success=True, failure_reason=None,
                  state=np.concatenate([r_ins, v_ins, [s[6] - prop]]), t_end=t4,
                  dri_dv_ms=dv, esm_prop_used_kg=float(prop), dro_snap_km=snap_km,
                  dro_phase=phi, dro_t0=t4, dro_period_s=float(dro["period"] / _EM_MEAN_MOTION),
                  dro_insertion_dist_km=float(np.linalg.norm(arr[:3] - moon_state(t4)[0]) / 1000.0))
    return result


def phase_outbound_powered_flyby(state, t0, perturb=None):
    """Outbound Powered Flyby (OPF): the ESM/OMS-E burn at the ~130 km lunar
    periselene. Modeled (v1) as a finite-thrust RETROGRADE capture burn that drops
    Orion from the hyperbolic flyby onto a lunar ellipse whose apoapsis = the DRO
    radius (~64,000 km); the ~3-day coast to that apoapsis and the actual DRO
    circularization are handled by phase_dro_insertion. The solved ΔV (~150-170
    m/s) and ~2.5-min burn match the real OPF. ESM propellant charged via mass.

    Scope (v1): real OPF is a small trim ON TOP of the gravity assist that sets up
    DRI; here it's the capture burn (the gravity assist itself is in the 3-body
    coast). The retrograde DRO *sense* is inherited from the flyby geometry, not
    enforced — that refinement belongs with a proper CR3BP DRO target at DRI.
    Returns success, failure_reason, state, t_end, opf_dv_ms, esm_prop_used_kg,
    post_opf_periapsis_km/apoapsis_km (lunar-relative)."""
    perturb = perturb or {}
    result = {"success": True, "failure_reason": None}
    s = np.concatenate([np.asarray(state[:3], float),
                        np.asarray(state[3:6], float), [float(state[6])]])
    t = float(t0)

    if perturb.get("oms_e_fail_opf"):           # AJ10/OMS-E ignition failure (sourced)
        result.update(success=False, failure_reason="oms_e_ignition_failure",
                      state=s, t_end=t)
        return result

    if _snapfree_active():   # snap-free DRO-targeting burn (no capture snap); Phase C = real-approach target
        return _opf_snapfree(s, t, perturb, result)

    if not globals().get("ENABLE_POWERED_FLYBY_TARGETING", False):
        result.update(state=s, t_end=t, opf_dv_ms=0.0)
        return result

    # lunar-relative state at periselene (the coast handed us closest approach)
    r_moon, v_moon = moon_state(t)
    r_rel = s[:3] - r_moon; v_rel = s[3:6] - v_moon
    r_rel_n = np.linalg.norm(r_rel); v_rel_n = np.linalg.norm(v_rel)
    r_apo = DRO_RADIUS_KM * 1000.0

    # target captured ellipse: periapsis = current, apoapsis = DRO radius
    a_t = 0.5 * (r_rel_n + r_apo)
    v_p_target = np.sqrt(MU_MOON * (2.0 / r_rel_n - 1.0 / a_t))
    dv_target = (v_rel_n - v_p_target) + perturb.get("opf_dv_bias_ms", 0.0)
    if dv_target <= 0.0:
        # flyby already slow enough to be captured at/below DRO distance
        result.update(state=s, t_end=t, opf_dv_ms=0.0,
                      post_opf_apoapsis_km=float("nan"))
        return result

    # finite-thrust ESM retrograde burn; cut off when lunar apoapsis reaches DRO
    T = OMSE_THRUST_N * perturb.get("omse_thrust_factor", 1.0)
    isp = OMSE_ISP_S * perturb.get("omse_isp_factor", 1.0)
    mdot = T / (isp * G0)
    m_esm_empty = ORION_TOTAL_KG - ESM_PROP_KG    # non-OMS-E-propellant mass floor

    t_burn0 = t
    if globals().get("ENABLE_OPF_STRADDLE", False):
        # STRADDLE the burn on periselene (phase-4 audit — the finite-PRM lesson): the
        # real ~2.5-min OPF was CENTERED on closest approach (max Oberth), while this code ignited AT
        # the CA the coast handed over, burning entirely post-periselene. Back up ballistically by
        # half the estimated burn arc and ignite early; the apoapsis cutoff is unchanged. OFF = the
        # legacy start-at-CA burn, bit-identical.
        tau = s[6] * (1.0 - np.exp(-dv_target / (isp * G0))) / mdot
        sol_b = solve_ivp(lambda tt, y: np.concatenate([y[3:6], gravity_earth_moon(y[:3], tt), [0.0]]),
                          (t, t - 0.5 * tau), s, method="RK45", rtol=1e-9, atol=1e-1, max_step=5.0)
        s = sol_b.y[:, -1].copy(); t_burn0 = t - 0.5 * tau

    def rhs(tt, y):
        rr = y[:3]; vv = y[3:6]; mm = y[6]
        vrel = vv - moon_state(tt)[1]
        tdir = -vrel / np.linalg.norm(vrel)       # retrograde (decelerate)
        a = gravity_earth_moon(rr, tt) + T * tdir / max(mm, 1.0)
        return np.concatenate([vv, a, [-mdot]])

    def ev_cut(tt, y):
        rr = y[:3]; vv = y[3:6]
        rmo, vmo = moon_state(tt)
        rrel = rr - rmo; vrel = vv - vmo; rn = np.linalg.norm(rrel)
        E = 0.5 * float(np.dot(vrel, vrel)) - MU_MOON / rn
        if E >= 0:
            return 1e12                            # still hyperbolic
        a = -MU_MOON / (2 * E)
        h = np.linalg.norm(np.cross(rrel, vrel))
        e = np.sqrt(max(0.0, 1 - (h*h) / (MU_MOON * a)))
        return a * (1 + e) - r_apo
    ev_cut.terminal = True; ev_cut.direction = -1
    def ev_starve(tt, y):
        return y[6] - m_esm_empty
    ev_starve.terminal = True; ev_starve.direction = -1

    try:
        sol = solve_ivp(rhs, (t_burn0, t_burn0 + 1200.0), s, method="RK45",
                        rtol=1e-8, atol=1e-1, max_step=2.0,
                        events=[ev_cut, ev_starve])
    except Exception as e:
        result.update(success=False, failure_reason=f"opf_burn_error: {e}")
        return result
    t_cut = sol.t_events[0][0] if len(sol.t_events[0]) else np.inf
    t_dep = sol.t_events[1][0] if len(sol.t_events[1]) else np.inf
    if t_dep < t_cut:
        result.update(success=False, failure_reason="esm_propellant_depleted_opf",
                      state=sol.y_events[1][0].copy(), t_end=t_dep)
        return result
    if t_cut < np.inf:
        s = sol.y_events[0][0].copy(); t = t_cut
    else:
        s = sol.y[:, -1].copy(); t = sol.t[-1]
        result.update(success=False, failure_reason="opf_capture_not_reached",
                      state=s, t_end=t)
        return result

    # diagnostics
    result["opf_dv_ms"] = float(OMSE_ISP_S * G0 * np.log(state[6] / s[6]))
    result["esm_prop_used_kg"] = float(state[6] - s[6])
    r_moon, v_moon = moon_state(t)
    rrel = s[:3] - r_moon; vrel = s[3:6] - v_moon; rn = np.linalg.norm(rrel)
    E = 0.5 * float(np.dot(vrel, vrel)) - MU_MOON / rn
    if E < 0:
        a = -MU_MOON / (2 * E)
        h = np.linalg.norm(np.cross(rrel, vrel))
        e = np.sqrt(max(0.0, 1 - (h*h) / (MU_MOON * a)))
        result["post_opf_periapsis_km"] = float((a*(1-e) - R_MOON) / 1000.0)
        result["post_opf_apoapsis_km"] = float((a*(1+e)) / 1000.0)
    result["state"] = s
    result["t_end"] = t
    return result


def phase_dro_insertion(state, t0, perturb=None):
    """DRO Insertion (DRI): coast from the OPF periselene, find the epoch/phase
    where Orion's trajectory best matches the real CR3BP DRO (dro_state_eci), and
    CAPTURE onto it — state set to the reference DRO state, ΔV = the velocity match
    charged to ESM. Records dro_phase/dro_t0/dro_period_s for the return search.

    KNOWN FIDELITY GAP (documented capture abstraction): the OPF delivers Orion to
    a lunar ellipse INSIDE the DRO, so capture requires a position "snap" onto the
    reference orbit — reported as `dro_snap_km` (~30,000 km). A faithful insertion
    needs the OUTBOUND to RENDEZVOUS with the DRO (the OPF capture targets a DRO
    state — the symmetric counterpart of the RPF→Earth return targeting). Until
    then DRI charges only the velocity-match ΔV (representative; the snap is free),
    and the resulting DRO + return GEOMETRY are correct (which the return needs).
    Returns success, failure_reason, state, t_end, dri_dv_ms, esm_prop_used_kg,
    dro_insertion_dist_km, dro_snap_km, dro_phase, dro_t0, dro_period_s."""
    perturb = perturb or {}
    result = {"success": True, "failure_reason": None}
    s = np.concatenate([np.asarray(state[:3], float),
                        np.asarray(state[3:6], float), [float(state[6])]])
    t = float(t0)

    if perturb.get("oms_e_fail_dri"):           # AJ10/OMS-E ignition failure (sourced)
        result.update(success=False, failure_reason="oms_e_ignition_failure",
                      state=s, t_end=t)
        return result

    if _snapfree_active():   # genuine velocity-match insertion (no teleport); Phase C = real-approach target
        return _dri_snapfree(s, t, perturb, result)

    def rhs(tt, y):
        return np.concatenate([y[3:6], gravity_earth_moon(y[:3], tt), [0.0]])

    # --- coast and find the (time, DRO phase) that best matches a DRO point -
    # Orion arrives on a lunar ellipse INSIDE the DRO; search the insertion epoch
    # over the next ~5 d for the moment its position is closest to a point on the
    # reference DRO (minimizes the capture position offset / "snap").
    t_scan = t + 5.0 * 86400.0
    try:
        sol = solve_ivp(rhs, (t, t_scan), s, method="RK45",
                        rtol=1e-9, atol=1e-1, max_step=600.0, dense_output=True)
    except Exception as e:
        result.update(success=False, failure_reason=f"dri_coast_error: {e}")
        return result
    ts0 = np.linspace(t, t_scan, 3000)
    Y0 = sol.sol(ts0)
    dmoon = np.array([np.linalg.norm(Y0[:3, i] - moon_state(tt)[0]) for i, tt in enumerate(ts0)])
    result["dro_insertion_dist_km"] = float(dmoon.max() / 1000.0)

    if not globals().get("ENABLE_DRO_TARGETING", False):
        i = int(np.argmax(dmoon))
        result.update(state=Y0[:, i].copy(), t_end=float(ts0[i]), dri_dv_ms=0.0)
        return result

    phs = np.linspace(0.0, 1.0, 120, endpoint=False)
    best = (1e30, None, None, None)                       # (snap, t, phase, orion_state)
    for i in range(0, len(ts0), 12):                      # coarse over time
        ti = float(ts0[i]); ri = Y0[:3, i]
        dd = [np.linalg.norm(dro_state_eci(p, ti)[0] - ri) for p in phs]
        k = int(np.argmin(dd))
        if dd[k] < best[0]:
            best = (dd[k], ti, float(phs[k]), Y0[:, i].copy())
    _, t, p0, s = best
    # OD-NAV: insert on the NOMINAL schedule. The per-trial best-match epoch `t` is how the outbound
    # TLI-pointing error propagates to the return (it shifts the reset DRO state/phase -> the EI
    # displacement). Real Artemis's outbound MCCs delivered to the DRO ON TIME; overriding `t` to the
    # nominal DRI epoch represents that, correcting the propagation. (The position snap absorbs the
    # geometry offset, as the v1 capture abstraction already does.)
    _ntI = globals().get("_NOMINAL_TARGETS")
    _dri_override = False
    if not perturb:                       # nominal: capture the DRI epoch (create the dict if the
        if not isinstance(_ntI, dict):    # driver hasn't yet — it was None here, so the capture was
            _ntI = {}; globals()["_NOMINAL_TARGETS"] = _ntI   # silently skipped = the OD-nav no-op bug
        _ntI["dri_t_nominal"] = float(t)
    elif globals().get("ENABLE_OD_NAV", False) and isinstance(_ntI, dict) and "dri_t_nominal" in _ntI:
        t = float(_ntI["dri_t_nominal"]); _dri_override = True
    _dbg("dri_t_d", float(t) / 86400.0); _dbg("dri_override", _dri_override)
    if globals().get("ENABLE_NEAR_POLAR_FLOWN", False):
        # #23: REPHASE — pick the insertion phase so coasting to the real DDP epoch
        # (NEAR_POLAR_DEP_GET_S) lands at the near-polar departure phase (sets the
        # trans-earth asymptote -> near-polar EI). The larger capture snap is absorbed
        # at the same fidelity level as the min-snap abstraction (the OPF->DRO rendezvous
        # is the documented future refinement).
        _period_s = _compute_cr3bp_dro()["period"] / _EM_MEAN_MOTION
        p0 = (NEAR_POLAR_DEP_PHASE - (NEAR_POLAR_DEP_GET_S - t) / _period_s) % 1.0
    else:
        for _ in range(4):                                # refine the phase (min-snap)
            loc = np.linspace(p0 - 1/120, p0 + 1/120, 11)
            dd = [np.linalg.norm(dro_state_eci(p % 1.0, t)[0] - s[:3]) for p in loc]
            p0 = float(loc[int(np.argmin(dd))] % 1.0)
    r_dro, v_dro = dro_state_eci(p0, t)
    result["dro_snap_km"] = float(np.linalg.norm(r_dro - s[:3]) / 1000.0)

    # ΔV = velocity match; position is SET to the reference DRO state (v1 capture
    # abstraction — the OPF delivers Orion to the DRO vicinity; the residual offset
    # dro_snap_km is absorbed). This yields a clean DRO with the correct geometry,
    # which the return (DDP/RPF) requires.
    dv = float(np.linalg.norm(v_dro - s[3:6]) + perturb.get("dri_dv_bias_ms", 0.0))
    # On-schedule DRI (OD-nav): forcing t->t_nominal makes this velocity-match capture from wherever
    # the perturbed orion is at that epoch (far from the reference DRO) -> a large dri_dv (344-575 m/s)
    # that drains the ESM and starves the RPF (28% rpf-depletion). Real Artemis's outbound MCCs
    # corrected the (tiny, amplified) divergence GRADUALLY/early, delivering to the DRO at ~nominal
    # cost -> so charge the NOMINAL dri_dv here (the faithful on-schedule-insertion cost), not the
    # lumped late capture. Captured from the nominal; perturbed-override trials reuse it (+ exec bias).
    if not perturb and isinstance(_ntI, dict):
        _ntI["dri_dv_nominal"] = dv
    elif _dri_override and isinstance(_ntI, dict) and "dri_dv_nominal" in _ntI:
        dv = float(_ntI["dri_dv_nominal"] + perturb.get("dri_dv_bias_ms", 0.0))
    isp = OMSE_ISP_S * perturb.get("omse_isp_factor", 1.0)
    m_esm_empty = ORION_TOTAL_KG - ESM_PROP_KG
    prop = s[6] * (1.0 - np.exp(-dv / (isp * G0)))
    if s[6] - prop < m_esm_empty:
        result.update(success=False, failure_reason="esm_propellant_depleted_dri",
                      state=s, t_end=t)
        return result

    # OD-nav DRI tracking residual: the nominal ΔV is charged (above), but the ACTUAL inserted state
    # differs from the EXACT reference DRO by the ground-OD's finite state knowledge (not perfect
    # schedule-keeping). Applied only on the OD-nav override path (perturbed); the residual then
    # propagates through the DRO coast + return -> a realistic (not artificially tight) dispersion.
    _od_res = 0.0
    if _dri_override and globals().get("ENABLE_OD_NAV_RESIDUAL", False):
        r_dro = r_dro + np.asarray(perturb.get("dri_od_pos_residual_m", np.zeros(3)), float)
        v_dro = v_dro + np.asarray(perturb.get("dri_od_vel_residual_ms", np.zeros(3)), float)
        _od_res = float(np.linalg.norm(perturb.get("dri_od_pos_residual_m", np.zeros(3))))
    _dbg("dri_od_residual_m", _od_res)
    dro = _compute_cr3bp_dro()
    s = np.concatenate([r_dro, v_dro, [s[6] - prop]])
    result["dri_dv_ms"] = dv
    result["esm_prop_used_kg"] = float(prop)
    result["dro_phase"] = p0                              # for the departure search
    result["dro_t0"] = t
    result["dro_period_s"] = float(dro["period"] / _EM_MEAN_MOTION)
    result["state"] = s
    result["t_end"] = t
    return result


def phase_dro_coast(state, t0, perturb=None, duration=None):
    """Coast in the Distant Retrograde Orbit (~6 days, the DRI->DDP stay), full
    3-body integration. Records the max-Earth-distance milestone (Artemis I set
    the crewed-capable-spacecraft record at ~432,000 km here) and confirms the
    orbit stays bounded. Modeled ballistically — a stable DRO needs little
    station-keeping (a small allowance is a future refinement).
    Returns success, failure_reason, state ([r,v,m] at the DDP point), t_end,
    max_earth_distance_km, dro_lunar_min_km/max_km."""
    perturb = perturb or {}
    result = {"success": True, "failure_reason": None}
    s = np.concatenate([np.asarray(state[:3], float),
                        np.asarray(state[3:6], float), [float(state[6])]])
    t = float(t0)
    if (duration is None and globals().get("ENABLE_NEAR_POLAR_FLOWN", False)
            and not globals().get("ENABLE_SNAPFREE_DRO", False)):
        # #23: depart at the real DDP epoch (GET ~15.6 d), the near-polar departure point
        # — shorter than the legacy 6.0 d FD16 coast, and more faithful to the real DDP.
        # (Snap-free uses the default coast; the general return's offset search finds the departure.)
        duration = max(0.5 * 86400.0, NEAR_POLAR_DEP_GET_S - t)
    dur = duration if duration is not None else ARTEMIS_PHASE_DUR_S["DRO coast"]

    def rhs(tt, y):
        return np.concatenate([y[3:6], gravity_earth_moon(y[:3], tt), [0.0]])
    try:
        sol = solve_ivp(rhs, (t, t + dur), s, method="RK45",
                        rtol=1e-9, atol=1e-1, max_step=1800.0, dense_output=True)
    except Exception as e:
        result.update(success=False, failure_reason=f"dro_coast_error: {e}")
        return result
    ts = np.linspace(t, t + dur, 4000)
    Y = sol.sol(ts)
    d_earth = np.linalg.norm(Y[:3, :], axis=0)
    d_moon = np.array([np.linalg.norm(Y[:3, i] - moon_state(tt)[0]) for i, tt in enumerate(ts)])
    i_far = int(np.argmax(d_earth))
    # Report SURFACE distance (geocentric radius − R_Earth), matching NASA's Artemis I record
    # convention: "268,563 mi / 432,194 km, FD13" == (geocentric max − R_Earth). The sim previously
    # reported the raw geocentric radius, which read ~6,378 km (one Earth radius) HIGH vs that altitude
    # reference — a units-CONVENTION mismatch, NOT a DRO-geometry error. Verified against the OEM: max
    # geocentric = 438,586 km, −R_Earth = 432,207 km = 268,561 mi ≈ the 268,563 mi record; the true
    # geometry residual is only ~−790 km (sim geocentric 437,796 vs OEM 438,586, sim slightly low).
    result["max_earth_distance_km"] = float(d_earth[i_far] / 1000.0 - R_EARTH / 1000.0)
    result["max_earth_distance_get_d"] = float(ts[i_far] / 86400.0)
    result["dro_lunar_min_km"] = float(d_moon.min() / 1000.0)
    result["dro_lunar_max_km"] = float(d_moon.max() / 1000.0)

    # The DRO must stay bounded: not escape the lunar region, not impact.
    if d_moon.max() > 120_000e3 or d_moon.min() < (R_MOON + 50_000.0):
        result.update(success=False, failure_reason="dro_unstable", state=sol.y[:, -1].copy(),
                      t_end=t + dur)
        return result

    result["state"] = sol.y[:, -1].copy()
    result["t_end"] = t + dur
    return result


# ============================================================
# Flown-return targeting helpers (ENABLE_FLOWN_RETURN)
# ============================================================
# DRO departure (DDP, retrograde-to-Moon) -> ~100 km lunar return flyby ->
# return powered flyby (RPF, prograde at perilune) -> trans-earth coast -> the
# REAL 122 km entry interface. Prototyped and validated separately. The solve
# happens once in phase_dro_departure; the plan is stashed in _RETURN_PLAN and the
# downstream return phases replay its segments (one trial per worker -> safe).
from scipy.optimize import brentq as _brentq

_RETURN_PLAN = None
_EI_RADIUS = None   # set lazily (R_EARTH + entry-interface altitude)


def _return_rhs(t, y):
    return np.concatenate([y[3:6], gravity_earth_moon(y[:3], t), [0.0]])


def _coast_dro(r, v, m, t0, dt):
    """Ballistically coast a DRO state by dt seconds (3-body). dt may be NEGATIVE (backward
    propagation) — needed by departure-phasing to depart EARLIER; the offset-search caller
    only passes dt >= 0, for which behavior is unchanged."""
    if dt == 0.0:
        return np.asarray(r, float).copy(), np.asarray(v, float).copy()
    sol = solve_ivp(_return_rhs, (t0, t0 + dt),
                    np.concatenate([r, v, [m]]), method="RK45",
                    rtol=1e-8, atol=1e0, max_step=1800.0)
    return sol.y[:3, -1], sol.y[3:6, -1]


def _perilune_state(r, v, m, t0, days=7.0):
    """Integrate to the FIRST lunar perilune (terminal event rrel·vrel=0, rising =
    distance minimum). Returns (perilune_alt_km, t_peri, state[r,v,m])."""
    def ev(tt, y):
        rmo, vmo = moon_state(tt)
        return float(np.dot(y[:3] - rmo, y[3:6] - vmo))
    ev.direction = 1.0
    ev.terminal = True
    sol = solve_ivp(_return_rhs, (t0, t0 + days * 86400.0),
                    np.concatenate([r, v, [m]]), method="RK45",
                    rtol=1e-7, atol=1e0, max_step=7200.0, events=ev)
    if len(sol.t_events[0]):
        te = sol.t_events[0][0]; ye = sol.y_events[0][0]
    else:
        te = sol.t[-1]; ye = sol.y[:, -1]
    alt = (np.linalg.norm(ye[:3] - moon_state(te)[0]) - R_MOON) / 1000.0
    return alt, te, ye.copy()


def _ei_crossing(r, v, m, t0, days=11.0):
    """Integrate to the FIRST descending crossing of the 122 km entry interface.
    Returns (state[r,v,m], t, fpa_deg, v_ms) or None (perigee stays above EI)."""
    global _EI_RADIUS
    if _EI_RADIUS is None:
        _EI_RADIUS = R_EARTH + ENTRY_INTERFACE_ALT_KM * 1000.0
    def ev(tt, y):
        return float(np.linalg.norm(y[:3]) - _EI_RADIUS)
    ev.direction = -1.0
    ev.terminal = True
    sol = solve_ivp(_return_rhs, (t0, t0 + days * 86400.0),
                    np.concatenate([r, v, [m]]), method="RK45",
                    rtol=1e-8, atol=1e0, max_step=7200.0, events=ev)
    if not len(sol.t_events[0]):
        return None
    te = sol.t_events[0][0]; ys = sol.y_events[0][0]
    rr, vv = ys[:3], ys[3:6]
    fpa = np.degrees(np.arcsin(np.dot(rr, vv) / (np.linalg.norm(rr) * np.linalg.norm(vv))))
    return ys.copy(), te, float(fpa), float(np.linalg.norm(vv))


def _solve_ddp(r, v, m, t0, target_peri_km=100.0):
    """Root-find the retrograde-to-Moon DDP magnitude that drops the next lunar
    perilune to target_peri_km. Returns (dv, post_ddp_v, peri_alt, t_peri,
    peri_state) or None."""
    vdir = (v - moon_state(t0)[1]); vdir = vdir / np.linalg.norm(vdir)
    def f(mag):
        return _perilune_state(r, v - mag * vdir, m, t0)[0] - target_peri_km
    if f(0.0) <= 0.0:
        return None                              # already at/below target unburned
    hi = None
    for mg in (100., 150., 200., 250., 300., 350., 400.):
        if f(mg) < 0.0:
            hi = mg; break
    if hi is None:
        return None
    try:
        dv = _brentq(f, 0.0, hi, xtol=0.3, maxiter=40)
    except Exception:
        return None
    alt, tp, ps = _perilune_state(r, v - dv * vdir, m, t0)
    return float(dv), (v - dv * vdir), alt, tp, ps


def _solve_ddp_recovery(r, v, m, t0, rpf_vec, rec, seed_burn, cap_ms=250.0):
    """OD-nav S1 (v2): 3-DOF DDP solving the departure burn so the PREDICTED SPLASH reaches the
    recovery — targeting the RECOVERY (the displacement source) via the cheap high-leverage DRO
    departure (NOT the over-burning RPF of option-2, and NOT the perilune-POSITION match of v1 which
    let the flyby velocity float -> EI scatter). Each eval: apply the DDP, propagate to perilune,
    apply the nominal RPF vector, propagate to the EI, predict the splash zone; residual = predicted
    splash vs the recovery (+ perilune-altitude reg to use the 3rd DOF). Warm-started from the nominal
    DDP burn; capped (ESM-aware) with a None return (caller falls back to the 1-DOF DDP). Returns
    (dv_mag, post_ddp_v, peri_alt, t_peri, peri_state) or None."""
    rlat, rlon = rec["splash_lat"], rec["splash_lon"]

    def resid(x):
        if np.linalg.norm(x) > cap_ms:                  # ESM cap: don't over-burn the DDP
            return [1.0e3, 1.0e3, 1.0e3]
        alt, tp, st = _perilune_state(r, v + x, m, t0)
        if not np.isfinite(alt) or alt > 5000.0 or alt < 0.0:
            return [1.0e3, 1.0e3, 1.0e3]
        e = _prop_to_ei_full(_moon_burn(st, tp, rpf_vec), tp)
        if e is None:
            return [1.0e3, 1.0e3, 1.0e3]
        zlat, zlon = _zone_of(e, rec)
        return [zlat - rlat, _dlon_deg(zlon, rlon), 0.02 * (alt - 130.0)]
    try:
        sol = _least_squares(resid, np.asarray(seed_burn, float), diff_step=5.0,
                             xtol=1e-6, ftol=1e-6, max_nfev=60)
    except Exception:
        return None
    x = sol.x
    if np.linalg.norm(x) > cap_ms:
        return None
    alt, tp, st = _perilune_state(r, v + x, m, t0)
    e = _prop_to_ei_full(_moon_burn(st, tp, rpf_vec), tp)
    if e is None or not np.isfinite(alt) or alt < 0.0:
        return None
    zlat, zlon = _zone_of(e, rec)
    if _gc_dist_km(zlat, zlon, rlat, rlon) > 1500.0:    # didn't get the splash within 1500 km -> fall back
        return None
    return float(np.linalg.norm(x)), (v + x), alt, tp, st


def _solve_rpf(peri_state, t_peri, fpa_target_deg=-6.5, rpf_hint=None):
    """Root-find the prograde (Moon-relative) RPF magnitude at perilune so the EI
    crossing FPA = fpa_target_deg. Returns (dv, post_rpf_v, ei_state, t_ei, fpa,
    v_ei) or None.

    The EI crossing appears only once the burn drops the Earth perigee below the
    entry interface; at appearance FPA ~ 0 (grazing) and steepens fast (~3.8°/m/s,
    the entry-corridor 'keyhole'). So: (1) a coarse 5 m/s scan locates where the
    crossing appears, then (2) a FINE 0.5 m/s scan from just below appearance
    catches the shallow region and brackets the target FPA for brentq. rpf_hint
    (the nominal RPF) narrows the scan for warm-started perturbed trials."""
    r, v, m = peri_state[:3], peri_state[3:6], peri_state[6]
    vdir = (v - moon_state(t_peri)[1]); vdir = vdir / np.linalg.norm(vdir)
    def cr(mag):
        return _ei_crossing(r, v + mag * vdir, m, t_peri)
    lo_scan = max(40.0, rpf_hint - 25.0) if rpf_hint else 40.0
    hi_scan = (rpf_hint + 25.0) if rpf_hint else 320.0
    # (1) coarse: where does the EI crossing first appear?
    appear = None
    for mg in np.arange(lo_scan, hi_scan, 5.0):
        if cr(float(mg)) is not None:
            appear = float(mg); break
    if appear is None:
        return None
    # (2) fine 0.5 m/s scan from below appearance; bracket the FPA target
    prev_mg = None; bracket = None
    for mg in np.arange(max(lo_scan, appear - 6.0), appear + 30.0, 0.5):
        res = cr(float(mg))
        if res is None:
            continue
        if res[2] <= fpa_target_deg:             # steep enough (more negative)
            if prev_mg is not None:
                bracket = (prev_mg, float(mg))
            break
        prev_mg = float(mg)
    if bracket is None:
        return None
    try:
        dv = _brentq(lambda mg: cr(mg)[2] - fpa_target_deg,
                     bracket[0], bracket[1], xtol=0.005, maxiter=40)
    except Exception:
        return None                  # no clean root (e.g. non-monotonic crossing) ->
                                     # DON'T deliver the steep bracket edge (overstresses);
                                     # let the caller search another departure offset.
    res = cr(float(dv))
    if res is None:
        return None
    ei_state, t_ei, fpa, v_ei = res
    # Verify the solve actually landed in the SURVIVABLE corridor. A steep miss
    # (FPA far below target) overstresses no matter how the entry is flown
    # (max lift can't pull a <-6.85 deg / 11 km/s entry under 12 g — grid-proven),
    # so reject it and let the offset search find a corridor EI. Real ground-tracked
    # nav always delivers a corridor EI, so this is a solver guard, not a risk.
    if abs(fpa - fpa_target_deg) > 0.25:
        return None
    return float(dv), (v + dv * vdir), ei_state, t_ei, fpa, v_ei


# ============================================================
# Closed-loop re-targeting (continuous-nav model)
# ============================================================
# Real Artemis nav re-solves the return continuously from ground tracking to a
# FIXED planned recovery EI. We model that with a multiple-shooting corrector:
# the chaotic part is DDP->flyby, but the trans-earth leg is clean near-Kepler,
# so a 3-DOF RPF at perilune (coarse) + a 3-DOF RTC on the clean leg (fine null)
# deliver Orion to the recovery point (lat/lon) at the entry FPA. Validated in
# a return prototype (Δlon/Δlat ~0, RPF+RTC ~160-250 m/s). Deterministic (least_squares,
# no RNG) -> the perturbation stream is untouched.
from scipy.optimize import least_squares as _least_squares


def _dlon_deg(lon, ref):
    return ((lon - ref + 540.0) % 360.0) - 180.0


def _prop_to_ei_full(state, t0, days=13.0):
    """Propagate to the first descending 122 km EI crossing; return
    (fpa_deg, t_ei, lat, lon, az_deg, ei_state) or None. az is the Earth-relative
    ground-track heading at EI (needed for the per-opportunity recovery zone)."""
    global _EI_RADIUS
    if _EI_RADIUS is None:
        _EI_RADIUS = R_EARTH + ENTRY_INTERFACE_ALT_KM * 1000.0
    def ev(tt, y):
        return float(np.linalg.norm(y[:3]) - _EI_RADIUS)
    ev.direction = -1.0; ev.terminal = True
    sol = solve_ivp(_return_rhs, (t0, t0 + days * 86400.0), state, method="RK45",
                    rtol=1e-8, atol=1e0, max_step=7200.0, events=ev)
    if not len(sol.t_events[0]):
        return None
    te = sol.t_events[0][0]; ys = sol.y_events[0][0]
    rr, vv = ys[:3], ys[3:6]
    fpa = np.degrees(np.arcsin(np.dot(rr, vv) / (np.linalg.norm(rr) * np.linalg.norm(vv))))
    lat, lon = eci_to_latlon(rr, te)
    up = rr / np.linalg.norm(rr)
    east = np.cross(np.array([0.0, 0.0, 1.0]), up); east /= np.linalg.norm(east)
    north = np.cross(up, east)
    v_rel = vv - np.cross(np.array([0.0, 0.0, OMEGA_E]), rr)
    vh = v_rel - np.dot(v_rel, up) * up
    az = np.degrees(np.arctan2(np.dot(vh, east), np.dot(vh, north)))
    return float(fpa), float(te), float(lat), float(lon), float(az), ys.copy()


def _moon_burn(peri, t_peri, vec):
    """Add a 3-DOF burn (Moon-relative prograde/radial/normal) at perilune."""
    rmo, vmo = moon_state(t_peri)
    rrel = peri[:3] - rmo; vrel = peri[3:6] - vmo
    pro = vrel / np.linalg.norm(vrel)
    nrm = np.cross(rrel, vrel); nrm /= np.linalg.norm(nrm)
    rad = np.cross(nrm, pro)
    out = peri.copy()
    out[3:6] = out[3:6] + vec[0] * pro + vec[1] * rad + vec[2] * nrm
    return out


def _earth_burn(state, vec):
    """Add a 3-DOF burn (Earth-relative prograde/radial/normal) on the trans-earth leg."""
    r, v = state[:3], state[3:6]
    pro = v / np.linalg.norm(v)
    nrm = np.cross(r, v); nrm /= np.linalg.norm(nrm)
    rad = np.cross(nrm, pro)
    out = state.copy()
    out[3:6] = out[3:6] + vec[0] * pro + vec[1] * rad + vec[2] * nrm
    return out


def _zone_of(e, rec):
    """Per-opportunity recovery zone (PREDICTED SPLASH): the nominal EI->splash
    vector (downrange rec['D'], track-offset rec['rel']) carried to this EI's
    ground point + heading. (e = _prop_to_ei_full tuple.)"""
    return _gc_dest(e[2], e[3], e[4] + rec["rel"], rec["D"])


def _solve_rpf_recovery(peri, t_peri, fpa_t, rec, rpf0):
    """3-DOF RPF (Moon-rel) so the PREDICTED SPLASH hits the recovery point (FPA +
    zone lat/lon). Returns (rpf_vec, post_rpf_state, ei_tuple) or None."""
    def resid(x):
        e = _prop_to_ei_full(_moon_burn(peri, t_peri, x), t_peri)
        if e is None:
            return [1e3, 1e3, 1e3]
        zlat, zlon = _zone_of(e, rec)
        # FPA weighted x5: the entry corridor (g-load) is safety-critical, so null it
        # tightly before the recovery-point position.
        return [5.0 * (e[0] - fpa_t), zlat - rec["splash_lat"], _dlon_deg(zlon, rec["splash_lon"])]
    _seed = list(rpf0) if np.ndim(rpf0) else [rpf0, 0.0, 0.0]   # 3-vec seed -> near-polar basin
    sol = _least_squares(resid, _seed, diff_step=0.5,
                         xtol=1e-8, ftol=1e-8, max_nfev=150)
    post = _moon_burn(peri, t_peri, sol.x)
    e = _prop_to_ei_full(post, t_peri)
    return (sol.x, post, e) if e is not None else None


from scipy.optimize import minimize as _minimize


def _solve_rpf_mindv(peri, t_peri, fpa_t, rec, seed, maxiter=80):
    """MIN-dV 3-DOF RPF: minimize |rpf| subject to the survivable near-polar corridor (FPA within
    +-0.5 deg of fpa_t) and — if rec given — the predicted splash within GUIDED REACH (~2500 km)
    of the recovery. Relaxing the EXACT zone target (cf _solve_rpf_recovery) recovers the real
    ~280 m/s powered flyby instead of the velocity-match's ~600 (the over-burn that drove the
    ESM-depletion + steep-FPA overstress). Returns (rpf, post_state, ei_tuple) or None."""
    REACH_KM = 2500.0

    def obj(x):
        e = _prop_to_ei_full(_moon_burn(peri, t_peri, x), t_peri)
        if e is None:
            return 5000.0 + float(np.linalg.norm(x))
        fpa, lat = e[0], e[2]
        # SMOOTH FPA centering (a flat +-0.5 corridor let the min-dv land at the cheapest-dv EDGE
        # -> perturbed scattered to -6.0 (no-splash) / -7.0 (overstress); 72% entry failures).
        pen = 2000.0 * (fpa - fpa_t) ** 2
        if rec is None:
            pen += 5.0e4 * max(0.0, lat + 10.0)             # near-polar: EI latitude southern
        else:
            zlat, zlon = _zone_of(e, rec)
            pen += 50.0 * max(0.0, _gc_dist_km(zlat, zlon, rec["splash_lat"], rec["splash_lon"]) - REACH_KM)
        return float(np.linalg.norm(x)) + pen

    def _run(s0):
        sol = _minimize(obj, np.asarray(s0, float), method="Nelder-Mead",
                        options=dict(maxiter=maxiter, xatol=1.0, fatol=1.0))
        post = _moon_burn(peri, t_peri, sol.x)
        e = _prop_to_ei_full(post, t_peri)
        if e is None:                            # accept any valid near-polar return; the RTC chain
            return None                          # centers the FPA (rejecting on FPA forced the
        return (np.asarray(sol.x, float), post, e, float(obj(sol.x)))  # velocity-match fallback)

    best = _run(seed)
    # MULTI-START escalation: a single Nelder-Mead from the warm-start can miss the cheap near-polar
    # basin when the flyby geometry SHIFTS (ENABLE_LUNAR_LIBRATION rotates the lunar figure ~13 deg
    # -> the tidal-lock-tuned seed lands at an invalid EI or an expensive >~420 m/s over-burn ->
    # rpf_esm_propellant_depleted). Try a small scatter of alternate seeds and keep the lowest-
    # objective VALID one. Escalation only fires when the primary is invalid/expensive, so the
    # production (tidal-lock) path converges on the first solve and pays no extra cost.
    seed = np.asarray(seed, float)
    if best is None or best[3] > 420.0:
        for s0 in (0.7 * seed, seed + np.array([0., 0., 90.]), seed + np.array([0., 0., -90.]),
                   seed + np.array([90., 0., 0.]), np.array([320., 0., 0.])):
            r = _run(s0)
            if r is not None and (best is None or r[3] < best[3]):
                best = r
    if best is None:
        return None
    return best[:3]


def _solve_rtc(s_rtc, t_rtc, fpa_t, rec):
    """3-DOF RTC (Earth-rel) on the clean trans-earth leg — fine null of the
    PREDICTED SPLASH to the recovery point. Returns (rtc_vec, ei_tuple) or None."""
    def resid(x):
        e = _prop_to_ei_full(_earth_burn(s_rtc, x), t_rtc)
        if e is None:
            return [1e3, 1e3, 1e3]
        zlat, zlon = _zone_of(e, rec)
        return [5.0 * (e[0] - fpa_t), zlat - rec["splash_lat"], _dlon_deg(zlon, rec["splash_lon"])]
    sol = _least_squares(resid, [0.0, 0.0, 0.0], diff_step=0.2,
                         xtol=1e-10, ftol=1e-10, max_nfev=150)
    e = _prop_to_ei_full(_earth_burn(s_rtc, sol.x), t_rtc)
    return (sol.x, e) if e is not None else None


def _phasec_return_plan(s, t, perturb, nt):
    """Phase-C faithful return: fly the JOINT-SOLVED DDP+RPF (real epochs; RPF 296.5 ≈ as-flown
    292.9) and RTC-fine-null to the recovery (PHASEC_RECOVERY = the AS-PLANNED off-San-Diego proxy
    32.30N/118.16W + the ~6,518 km extended-skip downrange). Per-trial dispersion: linear targeting via the nominal-captured
    sensitivity (dv = baked + J_lin @ (s0 − s0_nom)); the RTC absorbs the nonlinear residual.
    Returns a plan dict (same fields _solve_flown_return produces) or None (caller falls back
    to the FPA-corridor return — never a mission failure)."""
    t1 = float(globals()["PHASEC_RET_DRD_T_S"]); t2 = float(globals()["PHASEC_RET_RPF_T_S"])
    dv1 = np.asarray(globals()["PHASEC_RET_DDP_DV"], float).copy()
    dv2 = np.asarray(globals()["PHASEC_RET_RPF_DV"], float).copy()
    rec = dict(globals()["PHASEC_RECOVERY"])
    fpa_target = ENTRY_FPA_NOM_DEG + (perturb or {}).get("entry_fpa_bias_deg", 0.0)
    if t1 - t > 1.0:
        s0 = _coast_rv(np.asarray(s[:6], float), t, t1)
    elif t - t1 > 3600.0:
        return None                                   # arrived past the departure epoch
    else:
        s0 = np.asarray(s[:6], float).copy()

    def _match_resid(dv1_, dv2_, s0_):
        sA = np.concatenate([s0_[:3], s0_[3:6] + dv1_])
        sB = _coast_rv(sA, t1, t2)
        sC = _coast_rv(np.concatenate([sB[:3], sB[3:6] + dv2_]), t2,
                       float(globals()["PHASEC_RET_MATCH_T_S"]))
        return np.concatenate([
            (sC[:3] - np.asarray(globals()["PHASEC_RET_MATCH_R_M"], float)) / 1e3,
            sC[3:6] - np.asarray(globals()["PHASEC_RET_MATCH_V_MS"], float)])

    s0_ref = np.asarray(globals()["PHASEC_RET_S0_REF"], float)
    if "phasec_ret_Jlin" not in nt:
        # capture the LINEAR TARGETING sensitivity ONCE, around the BAKED REFERENCE (~12
        # propagations): J_lin = −J_dv⁻¹ @ J_s0 maps a pre-DRD state offset to the burn correction.
        r0v = _match_resid(dv1, dv2, s0_ref)
        Jdv = np.zeros((6, 6)); Js0 = np.zeros((6, 6))
        for k in range(6):
            e = np.zeros(6); e[k] = 0.05
            Jdv[:, k] = (_match_resid(dv1 + e[:3], dv2 + e[3:6], s0_ref) - r0v) / 0.05
        for k in range(6):
            e = np.zeros(6); e[k] = 1000.0 if k < 3 else 0.02
            Js0[:, k] = (_match_resid(dv1, dv2, s0_ref + e) - r0v) / e[k]
        try:
            nt["phasec_ret_Jlin"] = -np.linalg.solve(Jdv, Js0)
        except np.linalg.LinAlgError:
            nt["phasec_ret_Jlin"] = None
    if nt.get("phasec_ret_Jlin") is not None:
        # EVERY caller corrects for its offset from the reference — the nominal included (it sits
        # off the reference by code drift since the solve; a perturbed trial adds its dispersion)
        corr = np.asarray(nt["phasec_ret_Jlin"], float) @ (s0 - s0_ref)
        dv1 = dv1 + corr[:3]; dv2 = dv2 + corr[3:6]
    if globals().get("ENABLE_RETURN_POLISH", False):
        # POLISH: the single Jlin step above leaves ~20 km match residual on trials whose pre-DRD
        # offset exceeds the linear envelope; that propagates over the ~4-d match->EI leg into a
        # >0.35deg entry-FPA gate failure -> the trial defected to the free in-plane fallback. Iterate
        # the correction to convergence with a frozen Jacobian J_dv (dr/ddv at the reference, captured
        # ONCE — its own 'not in nt' guard so a PINNED nominal_targets.json, which carries Jlin but not
        # Jdv, re-captures it deterministically at s0_ref; ~6 propagations/process, no RNG). Newton with
        # improve-or-halve on the FULL 6-D norm; ~1-3 passes -> <1 km / <0.01 m/s. Corrections are
        # sub-0.1 m/s so the burn magnitudes (hence the ESM ledger) are unchanged.
        if "phasec_ret_Jdv" not in nt:
            rr = _match_resid(dv1_baked := np.asarray(globals()["PHASEC_RET_DDP_DV"], float),
                              dv2_baked := np.asarray(globals()["PHASEC_RET_RPF_DV"], float), s0_ref)
            Jd = np.zeros((6, 6))
            for k in range(6):
                e = np.zeros(6); e[k] = 0.05
                Jd[:, k] = (_match_resid(dv1_baked + e[:3], dv2_baked + e[3:6], s0_ref) - rr) / 0.05
            nt["phasec_ret_Jdv"] = Jd
        Jd = np.asarray(nt["phasec_ret_Jdv"], float)
        try:
            r = _match_resid(dv1, dv2, s0)
            for _pp in range(4):
                if np.linalg.norm(r[:3]) < 1.0 and np.linalg.norm(r[3:6]) < 0.01:
                    break
                n = float(np.linalg.norm(r))
                step = np.linalg.solve(Jd, -r)
                sn = float(np.linalg.norm(step))
                if sn > 20.0:
                    step = step * (20.0 / sn)          # trust cap (m/s)
                r_try = _match_resid(dv1 + step[:3], dv2 + step[3:6], s0)
                if np.linalg.norm(r_try) >= n:         # improve-or-halve on the FULL residual
                    step = 0.5 * step
                    r_try = _match_resid(dv1 + step[:3], dv2 + step[3:6], s0)
                    if np.linalg.norm(r_try) >= n:
                        break                          # stalled -> keep best; RTC/gate decide
                dv1 = dv1 + step[:3]; dv2 = dv2 + step[3:6]; r = r_try
        except np.linalg.LinAlgError:
            pass                                       # singular -> fall through on the linear correction

    sA = np.concatenate([s0[:3], s0[3:6] + dv1])
    sB = _coast_rv(sA, t1, t2)
    post_rpf_v = sB[3:6] + dv2
    _m = float(s[6]) if len(np.atleast_1d(s)) > 6 else ORION_TOTAL_KG
    e0 = _prop_to_ei_full(np.concatenate([sB[:3], post_rpf_v, [_m]]), t2)
    t_rtc = (e0[1] if e0 is not None else 25.4390 * 86400.0) - 2.5 * 86400.0
    if t_rtc <= t2 + 3600.0:
        return None
    seg = solve_ivp(_return_rhs, (t2, t_rtc), np.concatenate([sB[:3], post_rpf_v, [_m]]),
                    method="RK45", rtol=1e-8, atol=1e0, max_step=7200.0)
    s_rtc = seg.y[:, -1].copy()
    # Phase-C RTC: a fixed-epoch EI-POINT corrector — null r(PHASEC_RET_EI_T_S) − PHASEC_RET_EI_R_M
    # (the real EI-approach vector) with a 3×3 trust-region Newton over the clean 2.5-d arc. The
    # FPA/zone fine-null (_solve_rtc) is a LOCAL trim and converged 9,205 km off for the two-burn
    # family's ~145 km / 7.5 m/s floor; matching the real EI point pins geometry AND arrival time
    # (Earth rotation = splash longitude), and the corridor FPA comes along by inheritance.
    _ei_tgt = np.asarray(globals()["PHASEC_RET_EI_R_M"], float)
    _t_ei_tgt = float(globals()["PHASEC_RET_EI_T_S"])
    def _ei_miss(sv):
        _sol = solve_ivp(_return_rhs, (t_rtc, _t_ei_tgt), sv, method="RK45",
                         rtol=1e-8, atol=1e0, max_step=3600.0)
        return _sol.y[:3, -1] - _ei_tgt
    _rtc_acc = np.zeros(3)
    miss = _ei_miss(s_rtc)
    for _p in range(6):
        mn = float(np.linalg.norm(miss))
        if mn <= 30e3:
            break
        J = np.zeros((3, 3))
        for _k in range(3):
            sv = s_rtc.copy(); sv[3 + _k] += 0.02
            J[:, _k] = (_ei_miss(sv) - miss) / 0.02
        try:
            dvv = np.linalg.solve(J, -miss)
        except np.linalg.LinAlgError:
            break
        _dn = float(np.linalg.norm(dvv))
        if _dn > 10.0:
            dvv = dvv * (10.0 / _dn)
        s_try = s_rtc.copy(); s_try[3:6] = s_try[3:6] + dvv
        miss_try = _ei_miss(s_try)
        if np.linalg.norm(miss_try) >= mn:
            dvv = 0.5 * dvv
            s_try = s_rtc.copy(); s_try[3:6] = s_try[3:6] + dvv
            miss_try = _ei_miss(s_try)
            if np.linalg.norm(miss_try) >= mn:
                break
        s_rtc = s_try; _rtc_acc = _rtc_acc + dvv; miss = miss_try
    if float(np.linalg.norm(_rtc_acc)) > 60.0 or float(np.linalg.norm(miss)) > 500e3:
        return None                                   # corrector diverged / miss too big -> fall back
    _pre_dv = 0.0
    e_fin = _prop_to_ei_full(s_rtc, t_rtc)
    if e_fin is None:
        return None
    rtc_vec = _rtc_acc
    fpa_f, t_ei_f, lat_f, lon_f, az_f, ei_state = e_fin
    zlat, zlon = _zone_of(e_fin, rec)
    # FPA acceptance 0.35->0.50: after the return-polish drives the match residual to
    # ~0, a small IRREDUCIBLE FPA offset remains (the trial's own entry_fpa_bias + the nonlinear
    # match->EI propagation; the position-nulling RTC corrector doesn't remove it). At 0.35 a ~0.2%
    # tail hairline-missed (e.g. 0.3535deg) and defected to the free in-plane fallback (Indian Ocean)
    # despite flying a HEALTHY entry — FPA ~-6.25 sits well inside Orion's -6.5..-5.0 corridor and the
    # drag-tracker absorbs it. 0.50 keeps those trials on the faithful Pacific return; still comfortably
    # inside the corridor (worst case -6.4). The fallback solver's own gate (below) stays 0.35 — it
    # guards a different -7.0deg/15 g overstress mode that 0.50 still rejects.
    # AR1_FORCE_PHASEC (robust-reach / nominal self-heal): bypass ONLY this marginal
    # quality reject, so the intended Phase-C branch is kept when a different CPU/BLAS
    # numerical environment nudges fpa_f/zone across the gate. The HARD non-convergence
    # guards earlier in this function (divergence, e_fin) still reject a genuinely bad
    # solve. Unset (the default) => bit-identical to the pre-flag gate.
    if (os.environ.get("AR1_FORCE_PHASEC", "0") != "1"
            and (abs(fpa_f - fpa_target) > 0.50
                 or _gc_dist_km(zlat, zlon, rec["splash_lat"], rec["splash_lon"]) > 2500.0)):
        return None
    return dict(dep_offset_s=max(0.0, t1 - t), t_dep=t1, rd=s0[:3],
                post_ddp_v=s0[3:6] + dv1, ddp_dv=float(np.linalg.norm(dv1)),
                peri_alt_km=float((np.linalg.norm(sB[:3] - moon_state(t2)[0]) - R_MOON) / 1e3),
                t_peri=t2, peri_r=sB[:3], peri_v=sB[3:6],
                rpf_vec=dv2, rpf_dv=float(np.linalg.norm(dv2)), post_rpf_v=post_rpf_v,
                rtc_dv=float(np.linalg.norm(rtc_vec)) + _pre_dv, rtc_vec=rtc_vec,
                ei_r=ei_state[:3], ei_v=ei_state[3:6], t_ei=t_ei_f, ei_fpa=fpa_f,
                ei_vel=float(np.linalg.norm(ei_state[3:6])))


def _solve_flown_return(r0, v0, m0, t0, perturb, dep_offset_s=None, rpf_hint=None,
                        recovery=None):
    """Solve the flown return from a CLEAN DRO state.

    CLOSED-LOOP (recovery={lat,lon} given): re-target to deliver Orion to that
    FIXED recovery point — pick the departure offset, 3-DOF RPF at perilune
    (coarse), then a 3-DOF RTC on the clean trans-earth leg (fine null). Models
    real continuous ground-tracked nav; the plan carries rpf_vec/rtc_vec/rtc_dv
    and the delivered EI IS the recovery point.

    NOMINAL (recovery None): FPA-corridor solve only (1-DOF RPF) — its EI DEFINES
    the recovery point (captured by phase_dro_departure). rpf_hint warm-starts.
    Returns a plan dict or None (no closing / no-recovery-reach)."""
    fpa_target = ENTRY_FPA_NOM_DEG + perturb.get("entry_fpa_bias_deg", 0.0)
    offsets = ([dep_offset_s] if dep_offset_s is not None
               else [d * 86400.0 for d in (0., 1., 2., 3., 4., 5., 6., 7., 8.)])
    for off in offsets:
        rd, vd = _coast_dro(r0, v0, m0, t0, off)
        td = t0 + off
        ddp = _solve_ddp(rd, vd, m0, td, target_peri_km=100.0)
        if ddp is None:
            continue
        ddp_dv, post_ddp_v, peri_alt, t_peri, peri_state = ddp

        if recovery is None:
            # NOMINAL: FPA-corridor solve; its EI defines the recovery point.
            rpf = _solve_rpf(peri_state, t_peri, fpa_target, rpf_hint=rpf_hint)
            if rpf is None:
                continue
            rpf_dv, post_rpf_v, ei_state, t_ei, ei_fpa, ei_v = rpf
            return dict(dep_offset_s=off, t_dep=td, rd=rd, post_ddp_v=post_ddp_v,
                        ddp_dv=ddp_dv, peri_alt_km=peri_alt, t_peri=t_peri,
                        peri_r=peri_state[:3], peri_v=peri_state[3:6],
                        rpf_vec=np.array([rpf_dv, 0.0, 0.0]), rpf_dv=rpf_dv,
                        post_rpf_v=post_rpf_v, rtc_dv=0.0, rtc_vec=np.zeros(3),
                        ei_r=ei_state[:3], ei_v=ei_state[3:6], t_ei=t_ei,
                        ei_fpa=ei_fpa, ei_vel=ei_v)

        # CLOSED-LOOP: 3-DOF RPF (coarse) -> 3-DOF RTC (fine); target the PREDICTED
        # SPLASH (per-opportunity recovery zone) at the fixed recovery point.
        _rpf0 = rpf_hint if rpf_hint else 165.0
        if (globals().get("ENABLE_PHASEC_BPLANE", False) and rpf_hint is None
                and globals().get("PHASEC_RPF_SEED_MS") is not None):
            _rpf0 = np.asarray(globals()["PHASEC_RPF_SEED_MS"], float)   # real cross-plane family
        rr = _solve_rpf_recovery(peri_state, t_peri, fpa_target, recovery, _rpf0)
        if rr is None:
            continue
        rpf_vec, post_rpf, e_rpf = rr
        t_rtc = e_rpf[1] - 2.5 * 86400.0
        seg = solve_ivp(_return_rhs, (t_peri, t_rtc), post_rpf, method="RK45",
                        rtol=1e-8, atol=1e0, max_step=7200.0)
        rtc = _solve_rtc(seg.y[:, -1], t_rtc, fpa_target, recovery)
        if rtc is None:
            continue
        rtc_vec, e_fin = rtc
        fpa_f, t_ei_f, lat_f, lon_f, az_f, ei_state = e_fin
        zlat, zlon = _zone_of(e_fin, recovery)
        # Accept if the EI is in the FPA corridor AND the recovery is within GUIDED
        # REACH (~2,500 km) — the guided skip entry flies the precise splash from
        # there, so the return need not pin the recovery zone tightly (a tight
        # reject was brittle, failing ~65% of trials as ddp_no_earth).
        zmiss = _gc_dist_km(zlat, zlon, recovery["splash_lat"], recovery["splash_lon"])
        # FPA must stay in the SURVIVABLE corridor (tight — steep FPA overstresses);
        # the recovery only needs to be within GUIDED REACH (loose — the guided
        # entry flies the fine splash). If no such solution, the caller falls back
        # to the FPA-only return (always closes a corridor EI; real nav never fails
        # to return) so this is not a mission failure.
        if abs(fpa_f - fpa_target) > 0.35 or zmiss > 2500.0:
            # FPA reject TIGHTENED 0.7->0.35: a non-converged RTC could leave FPA at
            # ~-7.0 (15.7 g, fatal) yet pass the old 0.7 band -> overstress. 0.35 keeps
            # accepted closed-loop deliveries inside the survivable corridor; marginal
            # solves fall back to the FPA-only return (which now also gates FPA).
            continue
        return dict(dep_offset_s=off, t_dep=td, rd=rd, post_ddp_v=post_ddp_v,
                    ddp_dv=ddp_dv, peri_alt_km=peri_alt, t_peri=t_peri,
                    peri_r=peri_state[:3], peri_v=peri_state[3:6],
                    rpf_vec=rpf_vec, rpf_dv=float(np.linalg.norm(rpf_vec)),
                    post_rpf_v=post_rpf[3:6],
                    rtc_vec=rtc_vec, rtc_dv=float(np.linalg.norm(rtc_vec)),
                    ei_r=ei_state[:3], ei_v=ei_state[3:6], t_ei=t_ei_f,
                    ei_fpa=fpa_f, ei_vel=float(np.linalg.norm(ei_state[3:6])))
    return None


# ============================================================
# Near-polar return ANCHOR — the real Artemis I plane-change return
# ============================================================
# The real Artemis I return uses the powered lunar flyby as a ~66 deg PLANE CHANGE into
# a near-polar trans-earth trajectory (validated against the AROW as-flown OEM). Solving that flyby targeting in the sim's own dynamics
# is a full GMAT-class optimizer (a deferred follow-up). As the interim faithful
# nominal, ANCHOR the return to the validated near-polar flyby PERILUNE state (captured
# from the OEM backprop, rotated into the sim frame): forward-propagating it reproduces
# the real near-polar EI (lat -27.3, az ~north, FPA -6.48) -> 25.44 d mission, Baja-
# region splash. The departure->flyby is abstracted (perilune "snapped", same fidelity
# level as the documented DRO-insertion snap); the EI flight-path angle is perturbed per
# trial by the entry-corridor dispersion. When ENABLE_NEAR_POLAR_ANCHOR is False the
# flown solver (above) is used (bit-identical to pre-anchor).
ENABLE_NEAR_POLAR_ANCHOR = True
_NEAR_POLAR_T_PERI = 19.4154 * 86400.0
_NEAR_POLAR_PERI_R = np.array([275643389.43, 257890552.03, 113399918.22])   # m, sim ECI
_NEAR_POLAR_PERI_V = np.array([1726.584, 1193.816, -148.930])               # m/s
# The near-polar EI STATE (full precision), captured by forward-propagating the
# perilune anchor on the clean trans-earth leg. Anchored DIRECTLY (not re-propagated
# from the perilune): the perilune sits AT the flyby, so forward-propagating it
# re-traverses the sensitive outgoing hyperbola and amplifies any rounding into a wildly
# wrong EI. The EI is the endpoint (on the ground) -> rounding-robust.
_NEAR_POLAR_T_EI = 2197904.1151810004
_NEAR_POLAR_EI_R = np.array([-4408180.394168798, -3733072.0677217045, -2980587.118073787])
_NEAR_POLAR_EI_V = np.array([-2391.057763248037, -3190.75661498883, 10238.557696797638])

# --- FLOWN near-polar return (supersedes the EI anchor when True) ---------------
# The anchor hardcodes the EI; this flies the real plane-change return END-TO-END. Fix
# (validated separately): the sim DRO is the RIGHT PLANE but wrong SIZE+PHASE,
# so (1) resize the DRO to the real Artemis size (DRO_RADIUS_KM 64000->73000; real peri/apo
# 71k/94k), gated in _compute_cr3bp_dro; (2) rephase the DRI so departure (GET ~15.6 d, the
# real DDP epoch) lands at the near-polar DRO phase; (3) DDP nails the real near-polar
# perilune; (4) a 3-DOF RPF VELOCITY-MATCHES the near-polar perilune velocity _NEAR_POLAR_V_NP
# (the ~66 deg plane change -> EI azimuth due-north); (5) the RTC fine-nulls FPA (+ recovery,
# closed-loop) on the clean trans-earth leg -> the INBOUND MCC is LIVE. When True it replaces
# the anchor in phase_dro_departure; default False keeps the validated anchor bit-identical.
ENABLE_NEAR_POLAR_FLOWN  = True
# DRO-only re-architecture: snap-free OPF->DRO rendezvous (real DRI, no ~127,000 km capture snap).
# When ON it OVERRIDES the near-polar return path: inserts at the cheap basin phase SNAPFREE_DRO_PHASE
# (real 73k DRO), the OPF burn is SOLVED so the ballistic arc arrives AT the DRO, DRI is the small
# genuine velocity match, and the general flown return (_solve_flown_return) closes an IN-PLANE return.
# Splash region + mission timeline are the deferred follow-up. OFF = legacy capture-snap DRO + near-polar
# (bit-identical to the pre-flag lineage).
ENABLE_SNAPFREE_DRO      = True   # DEFAULT ON (with ENABLE_PHASEC_BPLANE —
#   the DRI capture-snap teleport is LIFTED: trials genuinely fly to the DRO (arrival residual ~614 km
#   median, honest velocity-match DRI ~94 m/s ≈ real 110.6). OFF = the schedule-snapped reference DRO.
SNAPFREE_DRO_PHASE       = 0.222              # snap-free insertion phase (basin min-total-ΔV cell)
SNAPFREE_DRI_GET_S       = 7.89 * 86400.0     # snap-free DRI epoch
SNAPFREE_OPF_SEED_MS     = (263.25, -34.35, -67.28)   # basin-solved OPF burn (m/s); warm-starts the in-sim solve
ENABLE_OEM_DRO_REF       = False  # reference DRO = the AS-FLOWN OEM DRO segment (precession-corrected,
#   time-interpolated) instead of the CR3BP-mapped periodic orbit. INVESTIGATED, NOT ADOPTED
#   (kept OFF as a diagnostic; OFF = CR3BP map, bit-identical). Finding: the sim's force model CANNOT hold
#   the real DRO — propagating the real OEM insertion state drifts 63,445 km / 380 m/s over the 6-day
#   coast (the real orbit is periodic in higher-fidelity dynamics + the OM-1/OM-3 maintenance burns, not
#   the sim's), and the CR3BP DRO is the wrong shape/size (73,000 km Moon-radius vs the real elliptical
#   87,000-94,000) so re-phasing can't match it (best 7,285 km at DRD). The real DRO geometry is reachable
#   ONLY by DATA-REPLAYING the OEM over the coast (re-anchoring at departure) — declined, to keep the
#   snap-free coast free-propagated. So RPF +60 (353 vs 293) + flyby +16 stay as CR3BP-shape residuals.
#   (The max-Earth "+5,600 km" was a SEPARATE geocentric-vs-altitude reporting bug, fixed at line ~3072.)
# Phase C lever (a) — REAL-APPROACH outbound re-aim (OEM-anchored). The OEM
# diagnosis showed the sim approaches the Moon 92.5 deg off the real flight (-> expensive insertion at
# DRO phase ~0.04); the REAL approach inserts snap-free for ~335 m/s at phase ~0.75 @ the real 9.66 d DRI
# epoch. This re-aims the TLI Lambert at the real lunar-approach point (PHASEC_TLI_AIM_ECI, baked from the
# frame-corrected OEM) and runs the snap-free OPF->DRO at the real phase/epoch.
# Builds on the snap-free machinery (overrides near-polar -> in-plane return). OFF =
# bit-identical to the snap-free/legacy paths.
ENABLE_PHASEC_BPLANE     = True   # DEFAULT ON (validated at scale:
#   91.40% [90.09,92.55] — the −3.5 pt vs the teleport baseline is the decomposed honest
#   insertion-dispersion cost; every burn now matches the as-flown record). OFF = the legacy
#   min-ΔV-Moon-aim + teleport configuration (pre-phase-5).
# Stage-1: launch azimuth co-tuned with the TLI re-aim (offline Phase-C bake) so the min-ΔV
# TLI from THIS parking orbit flies the real lunar approach. 76.77 (was 77.60, the old plane-containment
# value from the earlier plane-containment solve — superseded; the forced PHASEC_TLI_VPOST below is solved at 76.77 and
# REQUIRES this azimuth). NOTE: if ENABLE_PHASEC_BPLANE is ever activated for production, the snap-free
# OPF->DRO machinery (tuned around 77.60) must be re-checked at 76.77.
PHASEC_LAUNCH_AZIMUTH_DEG = 76.77
# Stage-2 target: the REAL Artemis I post-TLI 6-state (frame-corrected OEM). The TLI
# ignition scan injects onto the trans-lunar trajectory through this state -> the real lunar approach.
PHASEC_POSTTLI_T_S       = 7024.32            # GET of the real post-TLI state (s)
PHASEC_POSTTLI_R_M       = (-2764221.766, 8909786.786, 5456146.170)
PHASEC_POSTTLI_V_M       = (-7833.742523, 2344.371466, 2204.966649)
PHASEC_TLI_AIM_ECI       = None               # baked: TLI Lambert aim (ECI m) solved so the min-ΔV scan
#   produces the real post-TLI state (baked offline). None -> natural Moon-aim (legacy/off).
# FORCED-IGNITION bake: the aim-point hook above CANNOT fly the real approach — the min-ΔV
# ignition scan is a sharp valley (see its ~200 m/s jitter note) and re-picks a ~min-earlier ignition that
# the flyby amplifies into 20-30 deg of approach error. Instead force the offline-solved (ignition GET,
# post-TLI ECI velocity) that flies the REAL lunar approach at min-ΔV (~2850 m/s = ΔV-NEUTRAL vs the
# Moon-aim scan). Solved against THIS phase's own finite burn + full-dynamics coast (offline Phase-C bake),
# so the production TLI reproduces the real B-plane. Both non-None + ENABLE_PHASEC_BPLANE -> the scan is
# bypassed; MC dispersions (tli_pointing_rad, tli_dv_bias_ms) apply on top exactly as for the scan.
# Baked offline (az 76.77, solved against the production finite TLI): the
# re-aimed TLI flies the REAL lunar approach — asymptote 87.5°->6.1° off, CA epoch 5.262 d (real 5.257),
# periselene 1,112->253 km (the 253->72 residual is the phase-3 OTC's job), at ΔV 2,837 m/s = ΔV-NEUTRAL
# vs the natural min-ΔV 2,848. Dormant unless ENABLE_PHASEC_BPLANE (flag OFF -> bit-identical min-ΔV scan).
# RE-SOLVED on the TT (ΔT-corrected) ephemeris (offline bake re-run): the bake is
# essentially ΔT-ROBUST (ign +0.011 s, vpost ~0.16 m/s — the solve targets the Moon-independent OEM state).
# Flag-ON approach vs the corrected record: periselene 257 km @ 5.261 d vs real 133 @ 5.257, asymptote
# 6.2° off, TLI ΔV 2837 (neutral). The 257→133 residual = the OTC/OPF chain's job on activation.
PHASEC_TLI_IGN_S         = 5753.15986004518   # forced TLI ignition GET (s); None -> min-ΔV scan
PHASEC_TLI_R_IGN_M       = (6185819.281275214, 2404623.1575740264, 741915.6864720022)
#   ^ the NOMINAL ignition POSITION (300 km alt): the forced ignition is PHASE-ADAPTIVE (v9) — each
#   trial ignites where ITS parking orbit passes nearest this point, not at the nominal wall-clock
#   GET (launch-timing dispersion shifts the orbital phase; firing the baked v_target from the wrong
#   true anomaly was the dominant lunar-impact channel: 6/25 trials at v8).
PHASEC_TLI_VPOST         = (-5459.995225718178, 8079.4782086746245, 4718.102214848516)  # forced post-TLI ECI vel (m/s)
# Phase-5 re-derivation (TT ephemeris): the DRO
# phase + DRI epoch re-solved against the AS-FLOWN record (AAS 23-363: DRI 2022-11-25 21:52:28 = GET
# 831,884 s; the old 834,507 was 44 min off) — the real-arrival ceiling reproduces the real insertion cost
# (OPF 191.9 + DRI 93.9 = 285.8 vs real 178.6+110.6 = 289.3, 1.2%). PHASEC_CA_* = the REAL arrival's
# closest-approach point pushed through the sim's TT dynamics (OEM @ 4.5 d -> CA 146.09 km @ 5.2558 d):
# the Phase-C OTC chain targets THIS POINT AT THIS EPOCH (vector miss, not scalar periselene radius —
# radius-only trimming left the B-plane orientation free and tripled the OPF: 557 vs 192 m/s).
PHASEC_DRO_PHASE         = 0.7917             # baked: real-approach insertion phase (φ sweep, 24-grid best)
PHASEC_DRI_GET_S         = 831884.0           # as-flown DRI epoch (was 9.65864583 d — superseded)
PHASEC_CA_T_S            = 454098.46430773847 # real-arrival CA epoch via sim TT dynamics (GET 5.2558 d)
# Recovery target — AS-PLANNED: the nominal aims the PRE-FLIGHT
# PLAN splash zone "off San Diego", NOT the as-flown Guadalupe point (the ~350 mi southward move was the
# in-mission WEATHER response — it belongs in the dispersion, not the nominal). No precise plan lat/lon
# is published (region-level only), so this is a documented PLAN PROXY: a point in the Pacific off San
# Diego (32.30N, 118.16W), on the same extended-skip approach corridor from the real EI (-27.7S,-120.2W),
# at the D=6,518 km downrange that lands there (inside the AAS 20-649 skip-entry landing zone / range
# window 4,626-7,408 km). rel -0.56 deg unchanged (San Diego is along the same bearing as Guadalupe, just
# farther). Reaching it needs ENABLE_ENTRY_CROSSRANGE (bank reversals null the lateral) + the grid-search
# v_exit (the range sits in the bimodal-skip gap the bisection couldn't hit). The as-flown Guadalupe
# splash (29.0N,118.3W, D~6,314) is then one draw from the fleet (weather-retarget dispersion), NOT baked.
PHASEC_RECOVERY          = {"splash_lat": 32.30, "splash_lon": -118.16, "D": 6518.0, "rel": -0.56}
PHASEC_RPF_SEED_MS       = (208.8840933084797, 184.82247743821358, -4.932886305813071)
#   ^ the REAL RPF vector (gravity-corrected OEM velocity-discontinuity extraction, |279| m/s ≈ sourced
#   292.9; the same extraction reproduces DRD |138.48| EXACTLY) projected into the perilune
#   prograde/radial/normal seed basis — the in-plane 165 m/s warm start could not find the cross-plane
#   Guadalupe family (converged to FPA −15/zone 10,000 km off, or None).
# Joint DDP+RPF solution (6x6 trust-region Newton from the
# sim's own DRO state, real-vector seeds, burns at the REAL epochs, target = the real trans-earth state
# at 21.5 d): DRD 154.3 + RPF 296.5 = 450.7 m/s vs the as-flown 138.5+292.9=431.4 (RPF within 1.2%).
# DRD DOUBLE-COUNT (corrected): the baked DDP |151.4| is NOT +16 above the as-flown DRD via a
# CR3BP-mapping premium (the CR3BP premium is ~0) — it EQUALS the flown COMBINED departure OM-3 (13.0) +
# DRD (138.5) = 151.5 m/s to within 0.1. The sim lumps the real-time OM-3 DRO-maintenance re-opt into the
# single DDP burn, so the nominal departure embeds the in-flight re-optimization. Per the as-planned doctrine
# the plan-clean nominal would fly the PLANNED DRD (design 476.4 ft/s = 145.2 m/s, OM-3 design 0), with the
# re-opt as a DISPERSION. The planned trans-earth CORRIDOR is NOT published (only the flown OEM), so the
# joint solve targeting the FLOWN corridor is an AS-FLOWN STAND-IN (tagged; doctrine's plan-not-recoverable
# rule); the re-opt variability is added as a dispersion (ENABLE_DRD_REOPT_DISPERSION) so it is not baked
# into the nominal only. The sim's generic in-plane
# DDP could NOT present the real RPF's perilune approach (raw real-vector flight: FPA −82/no-EI), so the
# pair is solved jointly and flown as baked burns; the RTC fine-null + extended-skip entry finish to the
# real recovery. Per-trial dispersion: the nominal captures the linear targeting sensitivity
# (J_lin = −J_dv⁻¹·J_s0, the ground's linear correction); trials apply dv = baked + J_lin·Δs0.
PHASEC_RET_DRD_T_S       = 1350371.52         # 15.6293 d — the sim's DDP epoch = the real DRD epoch
PHASEC_RET_RPF_T_S       = 1676998.08         # 19.4097 d — the real RPF window start (= 19.4097×86400; a 400-s bake slip here mis-phased the solved burn ~960 km at perilune and cost three diagnostic rounds — verify baked epoch arithmetic against the solve script's printout)
PHASEC_RET_DDP_DV        = (147.7342670106932, -31.146958694766408, -11.842400955258523)
PHASEC_RET_RPF_DV        = (324.5294706684932, -60.703399730295445, -124.79600178850872)
PHASEC_RET_MATCH_T_S     = 1857600.0          # 21.5 d — the trans-earth match epoch (OEM-derived, baked)
PHASEC_RET_MATCH_R_M     = (282159346.139212, 252115456.35373926, 74052752.36172983)
PHASEC_RET_MATCH_V_MS    = (-188.87800409585768, -164.86579533441414, -242.36949798941768)
PHASEC_RET_S0_REF        = (370373356.4887839, -86849831.07875854, -70794096.80345926,
                            -13.728851449845342, 931.187640808968, 480.9610270061835)
#   ^ the REFERENCE pre-DRD state the joint DDP+RPF vectors were solved FROM. Every caller —
#   including the nominal — applies dv = baked + J_lin·(s0_actual − s0_ref): the burns are exact
#   only for the reference; the current nominal sits ~hundreds of km off it (code drift since the
#   solve), which silently gate-failed the un-corrected baked burns.
# DRD re-optimization dispersion (as-planned). The baked DDP embeds the flown
# in-flight re-opt (OM-3 DRO-maintenance + DRD adjustment: design departure 145.2 m/s → flown 151.5). Since
# the PLANNED corridor is unpublished the nominal DDP is an as-flown stand-in; this adds the re-opt as a
# per-trial DISPERSION on the DDP magnitude (charged to ESM, NOT a separate OM-3 burn — that would
# double-count the OM-3 already lumped in the DDP) so the FLEET spans the plan→flown departure range
# instead of the re-opt being baked into the nominal only. σ ≈ the design↔flown spread. Nominal
# (perturb=None) → 0 (unchanged). Drawn from the EXISTING _od_rng (NOT a new spawn — a new spawn resamples
# every trial's children; see the sample_perturbation note) → flag-OFF bit-identical.
ENABLE_DRD_REOPT_DISPERSION = (os.environ.get("AR1_DRD_REOPT", "1") == "1")   # default ON; AR1_DRD_REOPT=0 -> off (A/B)
DRD_REOPT_SIGMA_MS       = 6.5    # 1σ DDP-magnitude re-opt dispersion (m/s); ≈ the design 145.2 ↔ flown
#   151.5 departure spread (OM-3 13.0 net +6.3 over design). EST-grade (real re-opt was a discrete ops
#   decision; modeled as a per-trial Gaussian departure-magnitude variability). Charged to the ESM ledger.
PHASEC_RET_EI_T_S        = 2197928.617        # the real EI-approach point (the OEM's final state,
PHASEC_RET_EI_R_M        = (-4409583.519599422, -3726289.2525062393, -3030423.206002927)
#   142 km alt, (−27.70, −120.17), GET 25.4390 d): the Phase-C RTC nulls r(t_EI) − THIS vector —
#   a fixed-epoch 3×3 corrector over the clean 2.5-d arc (no flyby; timing MUST be pinned here since
#   Earth rotation sets the splash longitude). The FPA/zone fine-null RTC converged to a local
#   minimum 9,205 km off for the ~145-km/7.5-m/s two-burn family floor; the point-corrector re-plans.
PHASEC_CA_R_ECI_M        = (-335953823.7273121, -163007924.1146904, -60135842.54297061)
PHASEC_OPF_SEED_MS       = None               # baked: OPF burn seed (m/s) for the real-approach insertion (None -> default)
# OD-NAV (the faithful displacement fix). Sensitivity sweep
# found the recovery-zone displacement is driven by the TLI POINTING error (realistic
# ~6 arcsec) propagating via the DRI's per-trial best-match EPOCH (the outbound error shifts the
# insertion epoch -> the reset DRO state/phase -> the return). Return-leg perturbations have ~zero
# effect. So OD-NAV = INSERT ON THE NOMINAL SCHEDULE at the DRI (represents Artemis's outbound MCCs
# delivering to the DRO on time), which corrects the propagation. A calibrated DRI residual (future)
# sets the realistic ~4.7 km landing dispersion. OFF = the open-loop (epoch-floats-with-outbound) DRI.
ENABLE_OD_NAV = True
# OD-NAV DRI TRACKING RESIDUAL — OD-nav above models PERFECT schedule-keeping (snaps to the EXACT
# reference DRO state), so the displacement floor is artificially tight. Real ground-OD has FINITE
# state knowledge: add a small per-trial residual (pos+vel) to the on-schedule inserted state ~ the OD
# covariance at the DRO. Makes the post-DRI dispersion realistic (NOT tighter than the real ~4.7 km
# landing). Drawn UNCONDITIONALLY in sample_perturbation (fixed stream); applied only on the OD-nav
# override path (perturbed trials). OFF reproduces the zero-residual (perfect-schedule) OD-nav.
ENABLE_OD_NAV_RESIDUAL = True
OD_NAV_POS_SIGMA_M  = 300.0     # 1σ OD position-knowledge residual at DRI (~300 m, DSN at lunar dist) EST
OD_NAV_VEL_SIGMA_MS = 0.003     # 1σ OD velocity-knowledge residual (~3 mm/s, realistic DSN)           EST
#   A/B: σ_v=0.01 gave displacement 20→72 km (too loose; 1 cm/s OD is unrealistic). These
#   tighter values keep the zone-spread modest (~30 km, a realistic increase over the open-loop floor);
#   the splash accuracy (~6.7 km ≈ real 4.7) is unaffected by the residual either way.
# CLOSED-LOOP NAVIGATION on launch + TLI. Today the IGM ascent
# and the steered-TLI cutoff guidance act on the TRUE integrated state (perfect knowledge), so the
# injection dispersion collapses to ~42 km — unrealistically tight. Faithful GN&C: guidance is only
# as good as navigation. Modeled at the CALIBRATED-COVARIANCE level (per the OD-nav doc; NOT a
# strapdown IMU sim): the guidance nulls its ESTIMATED miss, so the achieved (true) injection state =
# the guidance target + a per-trial nav error drawn from the OD/INS covariance. Applied at the two
# guided cutoffs in run_mission (insertion + TLI). Error drawn in the state's RIC frame (anisotropic:
# in-track dominant) and rotated to inertial by _clnav_inertial_error. Two regimes: GPS-aided ascent
# (tight) vs IMU-degraded TLI (the dominant downstream term — an in-track velocity error at TLI grows
# to ~100 km at the Moon). Unit draws appended at the END of sample_perturbation (fixed stream) so a
# flag-OFF run is BIT-IDENTICAL to the pre-flag lineage. Does NOT touch the ~1,131 km systematic bias
# (that is the guidance TARGET, not the loop); it sets a realistic injection-DISPERSION floor and
# feeds realistically-larger OTC-1 corrections. All σ are EST-grade (no flown Artemis injection-
# dispersion data); calibrated so the post-TLI dispersion is consistent with a small real OTC-1.
ENABLE_CLOSED_LOOP_NAV = True
# Ascent insertion nav (GPS-aided INS): RIC 1σ [radial, in-track, cross-track] — pos (m) / vel (m/s).
CLNAV_ASC_POS_SIGMA_M  = (50.0, 100.0, 50.0)    # GPS-aided position knowledge at insertion   EST
CLNAV_ASC_VEL_SIGMA_MS = (0.05, 0.15, 0.05)     # GPS-aided velocity knowledge                 EST
# TLI-cutoff nav (IMU-dominated, climbing out of the GPS constellation): in-track vel dominates.
CLNAV_TLI_POS_SIGMA_M  = (200.0, 500.0, 200.0)  # IMU+pre-burn-update position knowledge       EST
CLNAV_TLI_VEL_SIGMA_MS = (0.20, 0.50, 0.20)     # IMU velocity knowledge at TLI cutoff         EST
# v2 — CONTINUOUS within-burn TLI nav. When ON, the TLI
# guidance steers/cuts on a dead-reckoned ESTIMATE that DRIFTS during the burn (initial pre-burn
# knowledge + accelerometer bias integrated over the ~19-min burn), so the injection error EMERGES
# from the cutoff-on-estimate (no discrete offset). Mechanistically faithful; calibrated so the net
# cutoff error ≈ the v1 ~0.5 m/s (success/splash unchanged). When OFF -> v1 discrete-offset TLI nav.
ENABLE_CLNAV_CONTINUOUS_TLI = True
CLNAV_TLI_INIT_POS_SIGMA_M  = (30.0, 60.0, 30.0)    # pre-TLI-burn position knowledge (GPS-aided-ish) EST
CLNAV_TLI_INIT_VEL_SIGMA_MS = (0.03, 0.06, 0.03)    # pre-TLI-burn velocity knowledge                 EST
CLNAV_TLI_ACCEL_BIAS_SIGMA_MS2 = (1.2e-4, 2.4e-4, 1.2e-4)  # IMU accel bias RIC (~14-24 µg, nav-grade); calibrated
#   so the EMERGENT cutoff-on-estimate injection error ≈ v1's ~0.5 m/s (the cutoff-timing sensitivity
#   ~1.7x-amplifies the raw nav error, so these inputs are ~0.6x the naive σ). EST-grade.
# --- OD FILTER — EMERGENT DSN information-matrix covariance.
# The successor to the hand-set nav σ tables (a known limitation). At each DSN-tracked decision epoch
# (TLI-init, OTC-1..6, DRI) the covariance P = (Σ HᵀR⁻¹H)⁻¹ is accumulated over the tracking arc from
# real DSN station geometry (3 complexes ~120° apart, rotating with the Earth) + range/Doppler/ΔDOR
# noise, computed ONCE on the nominal + pinned (chol into _NOMINAL_TARGETS → nominal_targets.json →
# shards). The per-trial injection swaps diag(σ)@unit → chol(P)@unit, REUSING the same spawned-child
# unit draws, so flag OFF is bit-identical.
#
# STM-based LinCov: the covariance is propagated with the STATE-TRANSITION MATRIX Φ over
# each tracking arc — a measurement at time tᵢ informs the EPOCH state through the dynamics: Λ +=
# (Hᵢ Φᵢ)ᵀ R⁻¹ (Hᵢ Φᵢ). This is what pins VELOCITY (range/Doppler over an arc + the dynamics coupling),
# which a Φ≈I batch cannot (finding: batch gave OTC-1 velocity ~900 mm/s vs real ~10-20).
# Three physical terms keep it faithful, not the raw-capability-optimistic sub-metre: (1) EFFECTIVE
# noise at the SYSTEMATIC-error level (delivered OD is systematic-limited, not measurement-limited);
# (2) an INFORMATIVE PRIOR ~ the accumulated-tracking-history knowledge (bounds poorly-observed
# directions — plane-of-sky, transverse velocity on short arcs — to the delivered envelope, the role a
# full sequential filter's carried-forward P plays); (3) a systematic FLOOR (P can't beat the delivered
# accuracy in well-observed directions). Prior + floor clamp P to the delivered envelope; the STM + arc
# geometry MODULATE within it (LOS anisotropy + arc-response = the emergent fidelity content). Verified
# vs the delivered tables: DRI ~230m/2mm/s, OTC-4 in-band, OTC-1 velocity ~17mm/s (bounded).
# OUT of scope: CLNAV_ASC + CLNAV_TLI_INIT (both onboard GPS/EOEKF at LEO, DSN-blind), the TLI
# accel-bias (onboard IMU). DEFAULT ON (AR1_OD_FILTER=0 disables it for a run; spawn-safe: worker processes inherit the env, so the
# nominal build + all workers agree).
ENABLE_OD_FILTER = (os.environ.get("AR1_OD_FILTER", "1") == "1")   # DEFAULT ON: adopted into
#   the definitive production run (outcome-neutral) — geometry-derived DSN nav covariance replaces
#   the calibrated-covariance stand-in (retires a known limitation). AR1_OD_FILTER=0 -> the pre-OD calibrated tables
#   (bit-identical to the prior lineage).
OD_DSN_STATIONS = ((35.426, -116.890, 1000.0),   # Goldstone   (lat°, lon°, alt m) [sourced ~site]
                   (40.431,   -4.248,  830.0),   # Madrid
                   (-35.402, 148.981,  690.0))   # Canberra
# EFFECTIVE (systematic-inclusive) 1σ per observable — calibrated so diag(P) lands on the delivered-OD
# tables, NOT the raw measurement capability (which over-informs sub-metre over multi-day arcs). EST.
OD_DOPPLER_SIGMA_MS = 1.0e-3     # effective 2-way Doppler (systematic-inclusive)          [est, calib]
OD_RANGE_SIGMA_M    = 600.0      # effective ranging (systematic/bias-inclusive)           [est, calib]
OD_DDOR_SIGMA_RAD   = 3.0e-7     # effective Delta-DOR (systematic-inclusive)              [est, calib]
OD_ELEV_MASK_DEG    = 10.0       # DSN elevation mask
OD_ARC_SAMPLES      = 30         # STM/measurement samples per arc (station sweep + dynamics)
OD_PRIOR_POS_M      = 2000.0     # informative prior 1σ pos (accumulated-history knowledge)  [est, calib]
OD_PRIOR_VEL_MS     = 0.020      # informative prior 1σ vel (bounds poorly-observed dirns)   [est, calib]
OD_FLOOR_POS_M      = 200.0      # (legacy global floor — SUPERSEDED by the per-epoch table-derived floor
OD_FLOOR_VEL_MS     = 0.002      #  in _build_od_filter_covariances; kept for reference/fallback)   [est]
OD_Q_POS_M          = 50.0       # process noise 1σ pos per day of measurement age          [est]
OD_Q_VEL_MS         = 0.0005     # process noise 1σ vel per day (older data less informative) [est]
# Per-epoch tracking-arc lengths (s), GROWN from TLI (= accumulated history) for OTC-1..4, reset at OPF
# for OTC-5/6, DRO long-arc for DRI. ΔDOR (the cross-track lever) heavier near the flybys + DRI.
OD_ARC_OTC_S        = (7.0*3600.0, 1.2*86400.0, 4.2*86400.0, 5.0*86400.0, 0.8*86400.0, 2.0*86400.0)
OD_DDOR_EPOCHS      = ("otc2", "otc3", "otc4", "otc5", "dri")   # epochs with ΔDOR (cross-LOS lever)
OD_ARC_DRI_S        = 4.0 * 86400.0    # DRO long arc (best OD of the mission)
# DDP-recovery targeting (abandoned: ESM-budget-limited band-aid + the displacement isn't a
# return-maneuver issue per the sweep). Kept behind a default-off flag; not the fix.
ENABLE_DDP_RECOVERY = False
# Departure-phasing (splashdown option 1) — REFUTED, keep OFF. Hypothesis: the recovery-
# zone displacement is an arrival-TIME scatter, fixable by shifting the DDP epoch (coast the DRO
# ±hours, zero ΔV). FALSIFIED by re-diagnosis: displacement correlates only 0.46 with arrival time
# and the Δlon/Δt slope is +1.7°/h, NOT the −15°/h of pure Earth-rotation — so the scatter is the
# return PLANE / inertial-EI orientation, not timing. Phasing made the 25-trial median WORSE
# (1547→3237 km): it re-times (a weak driver) while perturbing the plane (the real driver). The
# fix is RPF plane-targeting of the recovery's inertial EI, not re-timing. The _np_solve refactor +
# _coast_dro backward-coast are harmless and kept; this flag stays False.
ENABLE_DEPARTURE_PHASING = False
# RPF recovery-targeting (splashdown option 2): for PERTURBED trials, aim the flyby's INERTIAL EI at
# the nominal recovery ZONE (rec=recovery in the min-dV RPF) instead of a free near-polar zone. The
# recovery-zone displacement is the inertial-EI/plane scatter (phasing/option-1 refuted), and the
# flyby is the cheap-leverage point to control it. Relies on the multi-start RPF to converge (this
# fixed-zone target stalled the single Nelder-Mead before). OFF = free zone (production).
# v1 (uncapped) collapsed the displacement p90 14380->5118 at 250 trials BUT over-burned the RPF ->
# rpf_esm_propellant_depleted 10.4% (success 95.5->83.6%). v2 caps it: if the recovery-targeted RPF
# exceeds RPF_RECOVERY_CAP_MS, fall back to the cheap free-zone solve (that trial stays displaced but
# INTACT; the RTC + recovery fleet absorb the residual) — fix the affordable trials, never deplete.
ENABLE_RPF_RECOVERY_TARGET = False
RPF_RECOVERY_CAP_MS = 450.0     # free-zone RPF is ~330; allow ~+120 of recovery pull, else fall back
NEAR_POLAR_DRO_RADIUS_KM = 73000.0          # OEM-fit far-side crossing (real peri/apo 71k/94k)
NEAR_POLAR_DEP_PHASE     = 0.255            # real Artemis DRO departure phase (OEM)
NEAR_POLAR_DEP_GET_S     = 15.6 * 86400.0   # real DDP epoch (~FD15.6; shortens the 6.0 d coast)
_NEAR_POLAR_V_NP = np.array([2547.777690774181, 461.3384845061222, -0.6756553844172686])  # Moon-rel perilune vel, sim frame
# ^ re-derived on the FIRST-REV-TLI (rev-0) nominal: the CHEAP near-polar basin (normal comp ~0,
# RPF ~324 m/s ~ real ~280; the flyby supplies the plane change). The old out-of-plane seed
# [2411.6, 562.2, -522.1] left the rev-0 min-dv stuck in a 555 m/s basin (ESM ~35 kg, perturbed
# trials depleted). Re-seeding moves both revs into the cheap basin. (Same re-derive-on-the-new-
# trajectory pattern as the SRP shift.)
# LEGACY (unused by the flown path): the real Artemis Baja recovery. The flown near-polar return
# does NOT target this — the NOMINAL solves a FREE near-polar zone and its EI DEFINES the recovery
# (captured into _NOMINAL_TARGETS["recovery_zone_nominal"]); perturbed trials re-deliver to THAT
# captured zone. Kept for reference / the AII-AIII anchor path; safe to remove if those stay off.
_NEAR_POLAR_RECOVERY = {"splash_lat": 32.6179, "splash_lon": -114.0047, "D": 6700.0, "rel": 3.37}
# Recovery-pull cap (m/s) on the trans-earth RTC leg. The recovery-zone displacement is almost
# entirely an EI-LONGITUDE (arrival-TIME) scatter (corr 1.00 with |Δlon|; ei_lat std only 1.7°):
# perturbed returns arrive ±4-8 h off-nominal and Earth rotates the EI ground point (hence the
# zone) by ~60-130°. The RTC's prograde component corrects arrival time, but a 50 m/s cap couldn't
# even fix the MEDIAN ~4 h error while sitting on 1000-2400 kg (~190-410 m/s) of unused ESM. Raised
# to use that margin; kept below the worst-trial budget (~190 m/s) so the depletion floor is safe.
RECOVERY_PULL_CAP_MS = 120.0


def _anchored_return_plan(s, t, perturb):
    """Build the return plan from the validated near-polar EI anchor.
    The departure->flyby->trans-earth is abstracted (the EI state is set, same fidelity
    level as the DRI snap); the EI flight-path angle is perturbed per trial by
    entry_fpa_bias_deg (rotate v about the orbit normal). The full flyby optimizer that
    re-solves this per trial is a deferred follow-up."""
    t_ei = _NEAR_POLAR_T_EI
    r_ei = _NEAR_POLAR_EI_R.copy(); v_ei = _NEAR_POLAR_EI_V.copy()
    dfpa = np.deg2rad(perturb.get("entry_fpa_bias_deg", 0.0))
    if dfpa:                                        # rotate v about h_hat -> shift the FPA
        h = np.cross(r_ei, v_ei); h_hat = h / np.linalg.norm(h)
        c, sn = np.cos(dfpa), np.sin(dfpa)
        v_ei = v_ei * c + np.cross(h_hat, v_ei) * sn + h_hat * float(np.dot(h_hat, v_ei)) * (1 - c)
    fpa = float(np.degrees(np.arcsin(np.dot(r_ei, v_ei) /
                                     (np.linalg.norm(r_ei) * np.linalg.norm(v_ei)))))
    return dict(dep_offset_s=0.0, t_dep=t, rd=np.asarray(s[:3]).copy(),
                post_ddp_v=np.asarray(s[3:6]).copy(), ddp_dv=DDP_DV_MS, peri_alt_km=130.0,
                t_peri=_NEAR_POLAR_T_PERI, peri_r=_NEAR_POLAR_PERI_R.copy(),
                peri_v=_NEAR_POLAR_PERI_V.copy(), rpf_vec=np.array([RPF_DV_MS, 0., 0.]),
                rpf_dv=RPF_DV_MS, post_rpf_v=_NEAR_POLAR_PERI_V.copy(),
                rtc_vec=np.zeros(3), rtc_dv=RTC_DV_MS, ei_r=r_ei, ei_v=v_ei, t_ei=t_ei,
                ei_fpa=fpa, ei_vel=float(np.linalg.norm(v_ei)))


def _near_polar_flown_plan(s, t, perturb, recovery):
    """Near-polar return with optional DEPARTURE-PHASING (splashdown option 1).

    Solves the return at the nominal departure (_np_solve). When ENABLE_DEPARTURE_PHASING and
    this is a PERTURBED trial, iterates the DDP epoch (coast the DRO ±hours via _coast_dro, zero
    ΔV) to null the return ARRIVAL-TIME error vs the nominal's captured EI epoch. Shifting the
    whole ~6 d return is ~1:1 in arrival time, so a fixed-point on the departure offset converges
    fast; this collapses the EI-longitude scatter that IS the recovery-zone displacement (the late
    RTC can't — cap-300 == cap-120). Keeps the lowest arrival-time-error valid plan. Returns a
    plan dict or None. NOMINAL (recovery None) and phasing-off paths are bit-identical to before."""
    s = np.asarray(s, float)
    plan = _np_solve(s, t, perturb, recovery)
    if not (globals().get("ENABLE_DEPARTURE_PHASING", False) and recovery is not None):
        return plan
    _nt = globals().get("_NOMINAL_TARGETS")
    t_ei_tgt = _nt.get("t_ei_nominal") if isinstance(_nt, dict) else None
    if plan is None or t_ei_tgt is None:
        return plan
    best, best_err = plan, abs(plan["t_ei"] - t_ei_tgt)
    shift = 0.0
    for _ in range(4):
        dt = plan["t_ei"] - t_ei_tgt
        if abs(dt) < 1800.0:                          # within 30 min (~7.5° lon); RTC nulls the rest
            break
        shift = max(-18 * 3600.0, min(18 * 3600.0, shift - dt))   # depart earlier/later ~1:1; clamp ±18 h
        rd, vd = _coast_dro(s[:3], s[3:6], s[6], t, shift)
        p2 = _np_solve(np.concatenate([rd, vd, [s[6]]]), t + shift, perturb, recovery)
        if p2 is None:
            break
        plan = p2
        err = abs(plan["t_ei"] - t_ei_tgt)
        if err < best_err:
            best, best_err = plan, err
    return best


def _np_solve(s, t, perturb, recovery):
    """Inner near-polar return solve from DRO state `s` at a SINGLE departure epoch `t`
    (wrapped by _near_polar_flown_plan, which adds optional departure-phasing).
    #23 FLOWN near-polar return (ENABLE_NEAR_POLAR_FLOWN): the real plane-change return
    flown end-to-end (replaces the EI anchor). The rephased DRO state s at the real DDP
    epoch t departs to the near-polar flyby: a DDP nails the real near-polar perilune; a
    3-DOF RPF VELOCITY-MATCHES _NEAR_POLAR_V_NP (the ~66 deg plane change -> EI due-north);
    then the FPA (+ recovery, closed-loop) is fine-nulled on the clean trans-earth leg (the
    INBOUND RTC, live). NOMINAL (recovery None): velocity-match RPF + a 1-DOF clean-leg
    prograde FPA trim; the EI DEFINES the recovery (captured by phase_entry). PERTURBED:
    the existing closed-loop 3-DOF _solve_rpf_recovery (velocity-match SEEDED -> near-polar
    basin) -> 3-DOF _solve_rtc to the recovery. Returns a plan dict or None."""
    rd = np.asarray(s[:3], float); vd = np.asarray(s[3:6], float); m0 = float(s[6])
    # OD-nav S1: for PERTURBED trials, re-target the DDP to the nominal flyby B-plane (the perilune
    # position rel. the Moon) — high DRO leverage sets the return plane cheaply so the flyby/RPF/EI
    # match the nominal -> the inertial-plane displacement collapses. Falls back to the 1-DOF
    # perilune-altitude DDP (nominal run, flag off, or B-plane solve fails) -> never a mission failure.
    ddp = None; _ddp_path = "1dof"
    _nt0 = globals().get("_NOMINAL_TARGETS")
    if (recovery is not None and globals().get("ENABLE_DDP_RECOVERY", False)
            and isinstance(_nt0, dict) and "return_rpf_vec" in _nt0):
        ddp = _solve_ddp_recovery(rd, vd, m0, t, np.asarray(_nt0["return_rpf_vec"], float),
                                  recovery, _nt0.get("ddp_seed", [0.0, 0.0, 0.0]))
        if ddp is not None:
            _ddp_path = "recovery"
    if ddp is None:
        ddp = _solve_ddp(rd, vd, m0, t, target_peri_km=130.0)
    if ddp is None:
        _dbg("return_fail", "ddp_none")
        return None
    ddp_dv, post_ddp_v, peri_alt, t_peri, peri_state = ddp
    if recovery is not None:
        _dbg("ddp_path", _ddp_path); _dbg("ddp_dv", float(ddp_dv)); _dbg("peri_alt_km", float(peri_alt))
    rmo, vmo = moon_state(t_peri)
    # velocity-match seed in Moon-rel pro/rad/nrm (carries the plane-change normal)
    vrel = peri_state[3:6] - vmo
    pro = vrel / np.linalg.norm(vrel)
    nrm = np.cross(peri_state[:3] - rmo, vrel); nrm /= np.linalg.norm(nrm)
    rad = np.cross(nrm, pro)
    dvi = (vmo + _NEAR_POLAR_V_NP) - peri_state[3:6]
    seed = np.array([float(np.dot(dvi, pro)), float(np.dot(dvi, rad)), float(np.dot(dvi, nrm))])
    fpa_target = ENTRY_FPA_NOM_DEG + perturb.get("entry_fpa_bias_deg", 0.0)

    # MIN-dV near-polar RPF + a SINGLE RTC. Replaces the velocity-match over-burn (~600 m/s) +
    # 3-RTC chain that exhausted the ESM (level1_2000: ~26% ESM-depletion + steep-FPA overstress).
    # The min-dv solve relaxes the EXACT v_np match to "FPA corridor + predicted splash within
    # GUIDED REACH of the recovery" -> ~real 280-320 m/s. Warm-started from the nominal's captured
    # solution; velocity-match FALLBACK if it fails (so a trial never regresses to ddp_no_earth).
    # NOMINAL solves a FREE near-polar zone (rec=None); its EI DEFINES the captured recovery zone
    # (NOT a fixed Baja point) — perturbed trials then re-deliver to that captured zone.
    was_nominal = recovery is None
    _nt = globals().get("_NOMINAL_TARGETS")
    _hint = _nt.get("return_rpf_vec") if isinstance(_nt, dict) else None
    seed_rpf = np.asarray(_hint, float) if _hint is not None else seed   # warm-start / velocity-match
    # MIN-dV RPF zone target. Default FREE (rec=None: FPA corridor + EI latitude southern) — the
    # proven-robust form; the cheap near-polar return DEFINES the nominal recovery (captured by
    # phase_entry). ENABLE_RPF_RECOVERY_TARGET (splashdown option 2): for PERTURBED trials, target
    # the nominal's recovery ZONE at the RPF so the flyby aims the INERTIAL EI at the recovery —
    # the displacement is the inertial-EI/plane scatter (NOT arrival time; phasing was refuted), and
    # the flyby is the cheap-leverage point to control it. This "fixed-zone" RPF stalled Nelder-Mead
    # historically; the multi-start in _solve_rpf_mindv now makes it converge. OFF = free zone.
    _rpf_rec = (recovery if (not was_nominal and globals().get("ENABLE_RPF_RECOVERY_TARGET", False))
                else None)
    rr = _solve_rpf_mindv(peri_state, t_peri, fpa_target, _rpf_rec, seed_rpf)
    # ESM-AWARE CAP: a recovery-targeted RPF can over-burn (the far trials need a big plane change ->
    # rpf_esm_propellant_depleted, 10.4% in the uncapped 250-run). If it exceeds RPF_RECOVERY_CAP_MS,
    # fall back to the cheap FREE-zone solve — that trial stays displaced but INTACT (the RTC absorbs
    # what it can). Fixes the affordable trials' displacement without ever depleting.
    if _rpf_rec is not None and rr is not None and float(np.linalg.norm(rr[0])) > RPF_RECOVERY_CAP_MS:
        rr_free = _solve_rpf_mindv(peri_state, t_peri, fpa_target, None, seed_rpf)
        if rr_free is not None:
            rr = rr_free
    if rr is not None:
        rpf_vec, post_rpf, e_rpf = rr
        _dbg("rpf_path", "mindv")
    else:                                                   # velocity-match fallback (always closes)
        rpf_vec = seed.copy(); post_rpf = _moon_burn(peri_state, t_peri, rpf_vec)
        e_rpf = _prop_to_ei_full(post_rpf, t_peri)
        _dbg("rpf_path", "velmatch_fallback")
        if e_rpf is None:
            _dbg("return_fail", "rpf_no_ei")
            return None
    _dbg("rpf_dv", float(np.linalg.norm(rpf_vec)))
    e_cur = e_rpf; rtc_last = np.zeros(3); rtc_dv = 0.0
    if not was_nominal:    # perturbed: DECOUPLED RTC chain. A blunt cap on the whole chain starved
        _D = _nt.get("entry_downrange_km", 6700.0) if isinstance(_nt, dict) else 6700.0   # the FPA-null
        _rel = _nt.get("entry_track_offset_deg", 0.0) if isinstance(_nt, dict) else 0.0   # (overstress);
        state = np.asarray(post_rpf, float).copy(); t_state = t_peri   # an unbounded pull to the fixed
        # leg 5 d = bounded recovery-pull (cap RECOVERY_PULL_CAP_MS; an unbounded pull depleted the ESM,
        # 24%). Its PROGRADE component corrects arrival TIME -> EI longitude, which IS the recovery-zone
        # displacement (corr 1.00 with |Δlon|; ei_lat std only 1.7°). The legacy 50 m/s cap couldn't fix
        # even the median ~4 h arrival error; raised to use the ~1000-2400 kg unused ESM margin. leg 2 d =
        # FULL FPA-only null targeting the trial's CURRENT zone (no pull -> cheap) so the entry corridor
        # is always met even when the recovery is out of cheap reach (-> off-zone but intact).
        for _lead_d, _pull in ((5.0, True), (2.0, False)):
            t_rtc = e_cur[1] - _lead_d * 86400.0
            if t_rtc <= t_state + 3600.0:
                continue
            seg = solve_ivp(_return_rhs, (t_state, t_rtc), state, method="RK45",
                            rtol=1e-8, atol=1e0, max_step=3600.0)
            s_rtc = seg.y[:, -1]
            if _pull:
                _rec = recovery; _cap = RECOVERY_PULL_CAP_MS
            else:                                          # FPA-only: hold the CURRENT zone (zero pull)
                _zl, _zo = _gc_dest(e_cur[2], e_cur[3], e_cur[4] + _rel, _D)
                _rec = {"splash_lat": _zl, "splash_lon": _zo, "D": _D, "rel": _rel}; _cap = 1.0e9
            rtc = _solve_rtc(s_rtc, t_rtc, fpa_target, _rec)
            if rtc is None or rtc[1] is None:
                state = s_rtc; t_state = t_rtc; continue
            rtc_vec, e_new = rtc
            _m = float(np.linalg.norm(rtc_vec))
            if _m > _cap:
                rtc_vec = rtc_vec * (_cap / _m)
                e_new = _prop_to_ei_full(_earth_burn(s_rtc, rtc_vec), t_rtc)
                if e_new is None:
                    state = s_rtc; t_state = t_rtc; continue
            state = _earth_burn(s_rtc, rtc_vec); t_state = t_rtc
            rtc_dv += float(np.linalg.norm(rtc_vec)); rtc_last = rtc_vec; e_cur = e_new
    if was_nominal and isinstance(_nt, dict):
        _nt["return_rpf_vec"] = [float(v) for v in rpf_vec]   # warm-start for perturbed trials
        # rev-consistent _NEAR_POLAR_V_NP candidate (post-RPF Moon-rel perilune vel): used to
        # re-derive the hardcoded seed when the upstream trajectory shifts (TLI rev, SRP, ...).
        _nt["v_np_actual"] = [float(x) for x in (np.asarray(post_rpf[3:6], float) - vmo)]
        _nt["t_ei_nominal"] = float(e_cur[1])   # departure-phasing target (the nominal EI epoch)
        # OD-nav S1: warm-start the perturbed DDP-recovery solve from the nominal DDP burn.
        _nt["ddp_seed"] = [float(x) for x in (np.asarray(post_ddp_v, float) - vd)]
    fpa_f, t_ei_f, lat_f, lon_f, az_f, ei_state = e_cur
    if not was_nominal:
        _dbg("rtc_dv", float(rtc_dv)); _dbg("ei_lat", float(lat_f)); _dbg("ei_lon", float(lon_f))
        _dbg("ei_fpa", float(fpa_f)); _dbg("t_ei_d", float(t_ei_f) / 86400.0)
        _dbg("t_dep_d", float(t) / 86400.0)
    return dict(dep_offset_s=0.0, t_dep=t, rd=rd, post_ddp_v=post_ddp_v, ddp_dv=ddp_dv,
                peri_alt_km=peri_alt, t_peri=t_peri, peri_r=peri_state[:3], peri_v=peri_state[3:6],
                rpf_vec=np.asarray(rpf_vec, float), rpf_dv=float(np.linalg.norm(rpf_vec)),
                post_rpf_v=post_rpf[3:6], rtc_vec=np.asarray(rtc_last, float),
                rtc_dv=float(rtc_dv), ei_r=ei_state[:3], ei_v=ei_state[3:6],
                t_ei=t_ei_f, ei_fpa=fpa_f, ei_vel=float(np.linalg.norm(ei_state[3:6])))


def phase_dro_departure(state, t0, perturb=None):
    """DRO Departure (DDP) — flown (ENABLE_FLOWN_RETURN) or pragmatic.

    FLOWN: solve the whole return (so the chain is known to close), apply the
    retrograde-to-Moon DDP that sets up a ~100 km lunar return flyby, charge it to
    the ESM, and hand off the post-DDP DRO state. The departure offset is captured
    from the nominal into _NOMINAL_TARGETS so perturbed trials warm-start (no
    per-trial phase search). PRAGMATIC (flag off): charge a representative DDP to
    the ESM propellant BUDGET; no trajectory (bit-identical to pre-integration).
    Returns success, failure_reason, state ([r,v,m]), t_end, ddp_dv_ms,
    esm_prop_used_kg."""
    perturb = perturb or {}
    result = {"success": True, "failure_reason": None}
    s = np.concatenate([np.asarray(state[:3], float),
                        np.asarray(state[3:6], float), [float(state[6])]])
    t = float(t0)
    isp = OMSE_ISP_S * perturb.get("omse_isp_factor", 1.0)

    if perturb.get("oms_e_fail_ddp"):           # AJ10/OMS-E ignition failure (sourced)
        result.update(success=False, failure_reason="oms_e_ignition_failure",
                      state=s, t_end=t)
        return result

    if not globals().get("ENABLE_FLOWN_RETURN", False):
        dv = DDP_DV_MS + perturb.get("ddp_dv_bias_ms", 0.0)
        m_after = s[6] * np.exp(-dv / (isp * G0))
        if m_after < ORION_TOTAL_KG - ESM_PROP_KG:
            result.update(success=False, failure_reason="esm_propellant_depleted_ddp",
                          state=s, t_end=t)
            return result
        result["ddp_dv_ms"] = float(dv)
        result["esm_prop_used_kg"] = float(s[6] - m_after)
        s[6] = m_after
        result["state"] = s
        result["t_end"] = t
        return result

    # ---- flown DDP: solve the return, warm-starting the departure offset -----
    nt = globals().get("_NOMINAL_TARGETS")
    if nt is None:
        nt = {}
        globals()["_NOMINAL_TARGETS"] = nt
    if (globals().get("ENABLE_NEAR_POLAR_FLOWN", False)
            and not globals().get("ENABLE_SNAPFREE_DRO", False)):
        # #23: FLY the real near-polar plane-change return (resized+rephased DRO -> DDP ->
        # velocity-match RPF -> RTC). The nominal (no recovery yet) defines the recovery;
        # perturbed trials re-deliver to it closed-loop (RPF+RTC) — the inbound MCC is LIVE.
        # (Snap-free bypasses this -> the general in-plane return below, decoupled from near-polar.)
        recovery = None
        if all(k in nt for k in ("recovery_zone_nominal", "entry_downrange_km",
                                 "entry_track_offset_deg")):
            _z = nt["recovery_zone_nominal"]
            recovery = {"splash_lat": _z[0], "splash_lon": _z[1],
                        "D": nt["entry_downrange_km"], "rel": nt["entry_track_offset_deg"]}
        plan = _near_polar_flown_plan(s, t, perturb, recovery)
        off = 0.0; rpf_hint = None
    elif (globals().get("ENABLE_NEAR_POLAR_ANCHOR", False)
          and not globals().get("ENABLE_SNAPFREE_DRO", False)):
        # Anchor the return to the validated near-polar flyby (the real plane-
        # change return); the full per-trial flyby optimizer is a deferred follow-up.
        plan = _anchored_return_plan(s, t, perturb)
        off = 0.0                                    # no offset warm-start (anchored)
        rpf_hint = None; recovery = None
    else:
        off = nt.get("return_dep_offset_s")
        rpf_hint = nt.get("return_rpf_hint")
        # Recovery target = the nominal's PREDICTED SPLASH (zone) + entry-track vector,
        # captured by the nominal's phase_entry. None on the nominal run itself (it runs
        # before its own phase_entry) -> nominal solves FPA-only and DEFINES the zone.
        recovery = None
        if all(k in nt for k in ("recovery_zone_nominal", "entry_downrange_km",
                                 "entry_track_offset_deg")):
            _z = nt["recovery_zone_nominal"]
            recovery = {"splash_lat": _z[0], "splash_lon": _z[1],
                        "D": nt["entry_downrange_km"], "rel": nt["entry_track_offset_deg"]}
        elif (globals().get("ENABLE_PHASEC_BPLANE", False)
              and globals().get("PHASEC_RECOVERY") is not None):
            # Return re-aim: the NOMINAL (no captures yet) targets the REAL recovery — the
            # as-flown Guadalupe splash at the real extended-skip downrange — instead of
            # letting the min-ΔV FPA-only solve define an in-plane (Indian-Ocean) zone.
            recovery = dict(globals()["PHASEC_RECOVERY"])
        plan = None
        if (globals().get("ENABLE_PHASEC_BPLANE", False)
                and globals().get("PHASEC_RET_DDP_DV") is not None):
            # Phase-C: the faithful joint-solved return (real epochs/burns to the real recovery);
            # None (gates/geometry) -> the general solve below, unchanged.
            plan = _phasec_return_plan(s, t, perturb, nt)
        result["return_path"] = "phasec" if plan is not None else None
        if plan is None:
            plan = _solve_flown_return(s[:3], s[3:6], s[6], t, perturb,
                                       dep_offset_s=off, rpf_hint=rpf_hint, recovery=recovery)
            if plan is not None:
                result["return_path"] = "flown_recovery"
    if plan is None and recovery is not None:
        # Closed-loop couldn't hit the recovery within guided reach at this geometry
        # -> FALL BACK to the FPA-corridor return (always closes a valid EI). Real
        # ground-tracked nav never fails to RETURN; the guided entry then flies
        # toward the recovery (splash region may be off if the EI is far, but the
        # vehicle returns + splashes intact). NOT a mission failure (vs the old
        # brittle ddp_no_earth, which non-physically punished solver non-convergence).
        # dep_offset_s=None -> the FPA-only solve SEARCHES offsets 0..8 (like the
        # pre-nav model that always closed), so this fallback essentially never fails.
        # rpf_hint=None -> FULL RPF scan [40,320] (NOT the narrow [hint+-25]): a
        # perturbed trial's RPF can sit outside the nominal's +-25 window, and the
        # narrowed scan was failing to find the EI crossing at all -> spurious
        # ddp_no_earth (6.3% of the 10k). The full scan closes ~all of them.
        plan = _solve_flown_return(s[:3], s[3:6], s[6], t, perturb,
                                   dep_offset_s=None, rpf_hint=None, recovery=None)
        if plan is not None:
            result["return_path"] = "fpa_fallback"
    if plan is None:
        result.update(success=False, failure_reason="no_earth_return_found",
                      state=s, t_end=t)
        return result
    if off is None:                              # nominal: capture departure warm-start
        nt["return_dep_offset_s"] = plan["dep_offset_s"]
        nt["return_rpf_hint"] = plan["rpf_dv"]
        # the recovery zone (splash + entry-track vector) is captured by phase_entry
    globals()["_RETURN_PLAN"] = plan

    # DRD re-optimization dispersion (as-planned): charge a per-trial re-opt ΔV to the
    # ESM ledger + report it in ddp_dv, WITHOUT perturbing the trajectory (which flies plan["post_ddp_v"]
    # from the already-gated, un-biased plan — so no FPA-gate defection). Same mechanism as the existing
    # ddp_dv_bias_ms execution term. Puts the re-opt (design 145.2 ↔ flown 151.5 departure) in the FLEET
    # instead of baked into the nominal only; NOT a separate OM-3 burn (that would double-count the OM-3
    # already lumped in the baked DDP). Nominal (perturb=None) → 0.
    _reopt = (float(perturb.get("drd_reopt_unit", 0.0)) * DRD_REOPT_SIGMA_MS
              if globals().get("ENABLE_DRD_REOPT_DISPERSION", False) else 0.0)
    dv = plan["ddp_dv"] + perturb.get("ddp_dv_bias_ms", 0.0) + _reopt
    m_after = s[6] * np.exp(-dv / (isp * G0))
    if m_after < ORION_TOTAL_KG - ESM_PROP_KG:
        result.update(success=False, failure_reason="esm_propellant_depleted_ddp",
                      state=s, t_end=t)
        return result
    result["ddp_dv_ms"] = float(dv)
    result["esm_prop_used_kg"] = float(s[6] - m_after)
    result["return_dep_offset_d"] = plan["dep_offset_s"] / 86400.0
    result["return_perilune_km"] = float(plan["peri_alt_km"])
    s = np.concatenate([plan["rd"], plan["post_ddp_v"], [m_after]])
    result["state"] = s
    result["t_end"] = plan["t_dep"]
    return result


def phase_return_coast(state, t0, perturb=None, duration=None):
    """Return coast (DRO departure -> RPF). FLOWN: hand off the lunar-perilune state
    from the solved return plan (the real 3-body coast was integrated in the solve).
    PRAGMATIC: advance the timeline by the DDP->RPF reference duration."""
    perturb = perturb or {}
    if globals().get("ENABLE_FLOWN_RETURN", False):
        plan = globals().get("_RETURN_PLAN")
        s = np.concatenate([plan["peri_r"], plan["peri_v"], [float(state[6])]])
        return {"success": True, "failure_reason": None, "state": s,
                "t_end": plan["t_peri"], "return_perilune_km": float(plan["peri_alt_km"])}
    dur = duration if duration is not None else ARTEMIS_PHASE_DUR_S["DRO departure to RPF"]
    s = np.concatenate([np.asarray(state[:3], float),
                        np.asarray(state[3:6], float), [float(state[6])]])
    return {"success": True, "failure_reason": None, "state": s, "t_end": float(t0) + dur}


def phase_return_powered_flyby(state, t0, perturb=None):
    """Return Powered Flyby (RPF). FLOWN: charge the solved prograde RPF (the
    gravity-assist burn at perilune that does the return work) to the ESM and hand
    off the post-flyby state. PRAGMATIC: charge a representative RPF+RTC to the
    propellant budget (the binding return failure mode). Commits Orion to entry."""
    perturb = perturb or {}
    result = {"success": True, "failure_reason": None}
    s = np.concatenate([np.asarray(state[:3], float),
                        np.asarray(state[3:6], float), [float(state[6])]])
    t = float(t0)
    isp = OMSE_ISP_S * perturb.get("omse_isp_factor", 1.0)

    if perturb.get("oms_e_fail_rpf"):           # AJ10/OMS-E ignition failure (sourced)
        result.update(success=False, failure_reason="oms_e_ignition_failure",
                      state=s, t_end=t)
        return result

    if globals().get("ENABLE_FLOWN_RETURN", False):
        plan = globals().get("_RETURN_PLAN")
        # rpf_dv_bias is execution inefficiency (extra propellant); the trajectory
        # error it induces is absorbed by the RTC chain (phase_transearth_coast).
        dv = plan["rpf_dv"] + abs(perturb.get("rpf_dv_bias_ms", 0.0))
        m_after = s[6] * np.exp(-dv / (isp * G0))
        if m_after < ORION_TOTAL_KG - ESM_PROP_KG:
            result.update(success=False, failure_reason="esm_propellant_depleted_rpf",
                          state=s, t_end=t)
            return result
        result["rpf_dv_ms"] = float(dv)
        result["esm_prop_used_kg"] = float(s[6] - m_after)
        result["esm_prop_remaining_kg"] = float(m_after - (ORION_TOTAL_KG - ESM_PROP_KG))
        s = np.concatenate([plan["peri_r"], plan["post_rpf_v"], [m_after]])
        result["state"] = s
        result["t_end"] = plan["t_peri"]
        return result

    dv = RPF_DV_MS + RTC_DV_MS + perturb.get("rpf_dv_bias_ms", 0.0)
    m_after = s[6] * np.exp(-dv / (isp * G0))
    if m_after < ORION_TOTAL_KG - ESM_PROP_KG:
        result.update(success=False, failure_reason="esm_propellant_depleted_rpf",
                      state=s, t_end=t)
        return result
    result["rpf_dv_ms"] = float(dv)
    result["esm_prop_used_kg"] = float(s[6] - m_after)
    result["esm_prop_remaining_kg"] = float(m_after - (ORION_TOTAL_KG - ESM_PROP_KG))
    s[6] = m_after
    result["state"] = s
    result["t_end"] = t
    return result


def phase_transearth_coast(state, t0, perturb=None, duration=None):
    """Trans-earth coast (RPF -> entry interface). FLOWN: charge the Return
    Trajectory Correction (RTC) chain to the ESM and hand off the REAL 122 km
    entry-interface state from the solved return — actual ECI position/velocity at
    the 122 km descending crossing, FPA targeted to the corridor with the per-trial
    dispersion (folded into the RPF solve's FPA target). PRAGMATIC: advance the
    timeline and CONSTRUCT a representative EI state (~122 km, ~10.99 km/s, FPA
    ~-6.5° + dispersion) over a fixed ground point.
    Returns success, state ([r,v,m] at EI), t_end, entry_velocity_ms, entry_fpa_deg."""
    perturb = perturb or {}
    result = {"success": True, "failure_reason": None}
    m = float(state[6])
    isp = OMSE_ISP_S * perturb.get("omse_isp_factor", 1.0)

    if globals().get("ENABLE_FLOWN_RETURN", False):
        plan = globals().get("_RETURN_PLAN")
        # SOLVED RTC (closed-loop fine null to the recovery point) + the modeled
        # mid-course-correction allowance (covers the nominal, which has no solved
        # RTC, and small extra trims). The RPF execution error is charged at the RPF.
        dv = float(plan.get("rtc_dv", 0.0))   # the FLOWN RTC chain is the real cost — no
        #                                       double-count of the RTC_DV_MS stand-in allowance
        m_after = m * np.exp(-dv / (isp * G0))
        if m_after < ORION_TOTAL_KG - ESM_PROP_KG:
            result.update(success=False, failure_reason="esm_propellant_depleted_rtc",
                          state=np.concatenate([plan["ei_r"], plan["ei_v"], [m]]),
                          t_end=plan["t_ei"])
            return result
        result["entry_velocity_ms"] = float(plan["ei_vel"])
        result["entry_fpa_deg"] = float(plan["ei_fpa"])
        result["esm_prop_used_kg"] = float(m - m_after)
        result["state"] = np.concatenate([plan["ei_r"], plan["ei_v"], [m_after]])
        result["t_end"] = plan["t_ei"]
        return result

    dur = duration if duration is not None else ARTEMIS_PHASE_DUR_S["Trans-earth coast"]
    t = float(t0) + dur

    # entry conditions (dispersed FPA drives entry-g / skip depth)
    v_ei = ENTRY_VELOCITY_MS
    fpa = np.deg2rad(ENTRY_FPA_NOM_DEG + perturb.get("entry_fpa_bias_deg", 0.0))

    # representative EI position: ~122 km over a FIXED ground point, heading east.
    # (Fixed — not SPLASH_TARGET-derived — so the nominal splashdown is stable.)
    r_ei = latlon_alt_to_eci(EI_GROUND_LAT, EI_GROUND_LON,
                             ENTRY_INTERFACE_ALT_KM * 1000.0, t)
    up = r_ei / np.linalg.norm(r_ei)
    east = np.cross(np.array([0.0, 0.0, 1.0]), up); east = east / np.linalg.norm(east)
    # velocity: descending (negative FPA) and downrange (east-ish, toward splash)
    v_ei_vec = v_ei * (np.cos(fpa) * east + np.sin(fpa) * up)
    s = np.concatenate([r_ei, v_ei_vec, [m]])

    result["entry_velocity_ms"] = float(v_ei)
    result["entry_fpa_deg"] = float(np.rad2deg(fpa))
    result["state"] = s
    result["t_end"] = t
    return result


def phase_cm_sm_sep(state, t0, perturb=None):
    """CM/ESM separation just before entry: the ESM is jettisoned, leaving the
    Crew Module. Mass -> CM only; state/time carried."""
    perturb = perturb or {}
    s = np.concatenate([np.asarray(state[:3], float),
                        np.asarray(state[3:6], float), [ORION_CM_KG]])
    return {"success": True, "failure_reason": None, "state": s, "t_end": float(t0)}


def _gc_dist_km(lat1, lon1, lat2, lon2):
    """Great-circle distance [km] between two lat/lon points (degrees)."""
    la1, lo1, la2, lo2 = map(np.deg2rad, (lat1, lon1, lat2, lon2))
    d = np.arccos(np.clip(np.sin(la1) * np.sin(la2)
                          + np.cos(la1) * np.cos(la2) * np.cos(lo2 - lo1), -1.0, 1.0))
    return float(R_EARTH * d / 1000.0)


def _gc_bearing(lat1, lon1, lat2, lon2):
    """Initial great-circle bearing [deg, clockwise from N] from point 1 to 2."""
    la1, lo1, la2, lo2 = map(np.deg2rad, (lat1, lon1, lat2, lon2))
    y = np.sin(lo2 - lo1) * np.cos(la2)
    x = np.cos(la1) * np.sin(la2) - np.sin(la1) * np.cos(la2) * np.cos(lo2 - lo1)
    return float(np.rad2deg(np.arctan2(y, x)))


def _crosstrack_km(lat, lon, tlat, tlon, heading_deg):
    """Signed cross-track distance [km] of the target off the current heading:
    + = target lies to the RIGHT of the current ground-track heading, − = left.
    (Apollo lateral-guidance convention; used to command bank-reversal sign.)"""
    d = _gc_dist_km(lat, lon, tlat, tlon) / (R_EARTH / 1000.0)      # angular (rad)
    dth = np.deg2rad(((_gc_bearing(lat, lon, tlat, tlon) - heading_deg + 180.0) % 360.0) - 180.0)
    return float((R_EARTH / 1000.0) * np.arcsin(np.clip(np.sin(d) * np.sin(dth), -1.0, 1.0)))


def _gc_dest(lat, lon, bearing_deg, dist_km):
    """Destination lat/lon [deg] from a start point, initial bearing, distance."""
    la1, lo1 = np.deg2rad(lat), np.deg2rad(lon)
    brg = np.deg2rad(bearing_deg); dr = dist_km * 1000.0 / R_EARTH
    la2 = np.arcsin(np.sin(la1) * np.cos(dr) + np.cos(la1) * np.sin(dr) * np.cos(brg))
    lo2 = lo1 + np.arctan2(np.sin(brg) * np.sin(dr) * np.cos(la1),
                           np.cos(dr) - np.sin(la1) * np.sin(la2))
    return float(np.rad2deg(la2)), float((np.rad2deg(lo2) + 540.0) % 360.0 - 180.0)


# ============================================================
# Guided skip-entry (closed-loop bank predictor-corrector)
# ============================================================
# Validated separately: a predictor-corrector (predict the landing at a held
# bank, bisect the bank magnitude to null range-to-target, hold between ~200 s
# re-solves) flies the skip to a FIXED recovery point across the entry-FPA
# dispersion, collapsing splash_miss from skip-out/thousands of km to ~100-200 km.
# Steep entries still overstress (a real g-limit, not a guidance miss). Cheap
# (~2.5 s/entry: coarse capped-window predicts).
def _entry_rhs(tt, y, bank_deg, sign, cda, ld, peak):
    r = y[:3]; v = y[3:6]; rn = np.linalg.norm(r); alt = rn - R_EARTH
    a_g = gravity_earth_moon(r, tt)
    v_air = np.cross(np.array([0.0, 0.0, OMEGA_E]), r); v_rel = v - v_air
    vr = np.linalg.norm(v_rel); rho = atm_density(alt) * _ENTRY_DENS_SCALE
    if vr > 1.0 and rho > 1e-9:
        q = 0.5 * rho * vr * vr; vhat = v_rel / vr; m = y[6]
        a_drag = -(q * cda / m) * vhat
        up = r / rn
        lift_up = up - np.dot(up, vhat) * vhat; nlu = np.linalg.norm(lift_up)
        if nlu > 1e-6:
            lift_up /= nlu; side = np.cross(vhat, lift_up) * sign
            # SKIP SUPPRESSION (as in the nominal fixed-bank law, which _entry_rhs was
            # missing): if the vehicle is climbing back out of the upper atmosphere it
            # is about to skip OUT and re-enter steeply -> overstress. Override the
            # range bank with full lift-DOWN to stay captured. Triggers only on the
            # post-dip ascent (initial entry is descending, alt_rate<0), so it never
            # fights the normal descent. This — not an in-RHS g-limiter (tried &
            # removed: it deepened skips, 70+ g) — is what keeps steep EIs survivable.
            alt_rate = float(np.dot(v_rel, up))
            b = np.deg2rad(160.0) if (alt > 70_000.0 and alt_rate > 5.0) else np.deg2rad(bank_deg)
            a_lift = (q * cda * ld / m) * (np.cos(b) * lift_up + np.sin(b) * side)
        else:
            a_lift = np.zeros(3)
        a = a_drag + a_lift
        peak[0] = max(peak[0], np.linalg.norm(a) / G0)
    else:
        a = np.zeros(3)
    return np.concatenate([v, a_g + a, [0.0]])


def _entry_splash_ev(tt, y):
    return np.linalg.norm(y[:3]) - R_EARTH
_entry_splash_ev.terminal = True
_entry_splash_ev.direction = -1


def _entry_predict(tt, y, bank_deg, sign, cda, ld, window=3200.0):
    """Coarse landing prediction at a held bank; (lat, lon, landed, peak_g)."""
    pk = [0.0]
    sol = solve_ivp(lambda t, x: _entry_rhs(t, x, bank_deg, sign, cda, ld, pk),
                    (tt, tt + window), y, method="RK45", rtol=1e-6, atol=1e0,
                    max_step=4.0, events=_entry_splash_ev)   # fine enough to catch peak-g
    if len(sol.t_events[0]):
        ys = sol.y_events[0][0]
        la, lo = eci_to_latlon(ys[:3], sol.t_events[0][0])
        return la, lo, True, pk[0]
    return None, None, False, pk[0]   # no land in window = skip-out (range too long)


def _entry_solve_bank(tt, y, tlat, tlon, sign, cda, ld, g_limit=9.0):
    """Bisect bank magnitude [5,88] deg to null range-to-target (more bank=shorter),
    with a g-cap: a bank that would overstress is reduced (more lift, softer). Survival
    is a hard constraint, range secondary, so the vehicle splashes intact off-target
    rather than overstressing to reach an out-of-reach recovery. The skip itself is
    prevented upstream by the lift-down skip-suppression in _entry_rhs, so this stays
    in the well-behaved single-pass regime."""
    rn = eci_to_latlon(y[:3], tt)
    d_tgt = _gc_dist_km(rn[0], rn[1], tlat, tlon)
    b_lo, b_hi = 5.0, 88.0
    for _ in range(11):
        mid = 0.5 * (b_lo + b_hi)
        la, lo, landed, pkg = _entry_predict(tt, y, mid, sign, cda, ld)
        if pkg > g_limit:                 # too steep -> overstress -> reduce bank
            b_hi = mid
        elif (not landed) or (_gc_dist_km(rn[0], rn[1], la, lo) - d_tgt) > 0.0:
            b_lo = mid                    # lands too far / skips -> more bank (shorter)
        else:
            b_hi = mid
    return 0.5 * (b_lo + b_hi)


def _guided_entry_fly(s0, t0, tlat, tlon, cda, ld):
    """Predictor-corrector skip entry to (tlat, tlon): re-solve the bank every
    ~200 s and hold. Returns (state, t_splash, peak_g, landed)."""
    t = float(t0); s = np.asarray(s0, float).copy(); pk = [0.0]; sign = 1.0
    while t < t0 + 7200.0:
        bank = _entry_solve_bank(t, s, tlat, tlon, sign, cda, ld)
        sol = solve_ivp(lambda tt, x: _entry_rhs(tt, x, bank, sign, cda, ld, pk),
                        (t, t + 200.0), s, method="RK45", rtol=1e-7, atol=1e-1,
                        max_step=3.0, events=_entry_splash_ev)
        if len(sol.t_events[0]):
            return sol.y_events[0][0].copy(), float(sol.t_events[0][0]), pk[0], True
        t = float(sol.t[-1]); s = sol.y[:, -1]
    return s, t, pk[0], False


# --- STABLE constant-bank fallback (HARDENS the rare predictor-corrector spike) ---------
# The predictor-corrector above + the in-RHS skip-suppression can spike erratically on the
# steep-FPA / skip-out tail (17-91 g artifacts; e.g. a benign -6.66 deg EI flown to 29 g):
# a low-bank solve lofts the vehicle, the 160 deg lift-down suppression then yanks it back
# steeply. A CONSTANT bank flown with NO suppression is monotonic and spike-free (verified by
# an entry bank-sweep: g rises smoothly with bank/FPA, every case lands), so this is the
# survival-first fallback re-flown only when the primary law spikes >structural-g or skips out.
def _entry_rhs_nosupp(tt, y, bank_deg, sign, cda, ld, peak):
    """Entry RHS with a held bank and NO skip-suppression (the stable-fallback dynamics)."""
    r = y[:3]; v = y[3:6]; rn = np.linalg.norm(r); alt = rn - R_EARTH
    a_g = gravity_earth_moon(r, tt)
    v_air = np.cross(np.array([0.0, 0.0, OMEGA_E]), r); v_rel = v - v_air
    vr = np.linalg.norm(v_rel); rho = atm_density(alt) * _ENTRY_DENS_SCALE
    if vr > 1.0 and rho > 1e-9:
        q = 0.5 * rho * vr * vr; vhat = v_rel / vr; m = y[6]
        a_drag = -(q * cda / m) * vhat
        up = r / rn
        lift_up = up - np.dot(up, vhat) * vhat; nlu = np.linalg.norm(lift_up)
        if nlu > 1e-6:
            lift_up /= nlu; side = np.cross(vhat, lift_up) * sign
            b = np.deg2rad(bank_deg)
            a_lift = (q * cda * ld / m) * (np.cos(b) * lift_up + np.sin(b) * side)
        else:
            a_lift = np.zeros(3)
        a = a_drag + a_lift
        peak[0] = max(peak[0], np.linalg.norm(a) / G0)
    else:
        a = np.zeros(3)
    return np.concatenate([v, a_g + a, [0.0]])


def _fly_const_bank(s0, t0, bank, sign, cda, ld):
    """Fly a constant bank (no suppression) to splash; (state, t, peak_g, landed)."""
    pk = [0.0]
    sol = solve_ivp(lambda tt, x: _entry_rhs_nosupp(tt, x, bank, sign, cda, ld, pk),
                    (t0, t0 + 9000.0), np.asarray(s0, float).copy(), method="RK45",
                    rtol=1e-7, atol=1e-1, max_step=3.0, events=_entry_splash_ev)
    if len(sol.t_events[0]):
        return sol.y_events[0][0].copy(), float(sol.t_events[0][0]), pk[0], True
    return sol.y[:, -1].copy(), float(sol.t[-1]), pk[0], False


def _solve_stable_entry(s0, t0, tlat, tlon, cda, ld, b_floor=30.0):
    """Survival-first fallback: bisect a CONSTANT bank in [floor,88] to null range-to-target
    (no suppression -> monotonic), flooring at the sub-skip threshold so the vehicle never
    lofts into the suppression spike. Returns (state, t, peak_g, landed)."""
    ei = eci_to_latlon(s0[:3], t0)
    d_tgt = _gc_dist_km(ei[0], ei[1], tlat, tlon)
    # crossrange sign: the floor-bank landing side nearer the target
    best = {}
    for sgn in (1.0, -1.0):
        ss, tt, _, ok = _fly_const_bank(s0, t0, b_floor, sgn, cda, ld)
        if ok:
            la, lo = eci_to_latlon(ss[:3], tt); best[sgn] = _gc_dist_km(la, lo, tlat, tlon)
        else:
            best[sgn] = 1e9
    sign = 1.0 if best[1.0] <= best[-1.0] else -1.0
    b_lo, b_hi = b_floor, 88.0
    for _ in range(10):
        mid = 0.5 * (b_lo + b_hi)
        ss, tt, _, ok = _fly_const_bank(s0, t0, mid, sign, cda, ld)
        rng = _gc_dist_km(ei[0], ei[1], *eci_to_latlon(ss[:3], tt)) if ok else 1e9
        if (not ok) or rng > d_tgt:
            b_lo = mid
        else:
            b_hi = mid
    return _fly_const_bank(s0, t0, 0.5 * (b_lo + b_hi), sign, cda, ld)


# --- PredGuid DRAG-TRACKER entry (Orion's shallow skip corridor, ~4-5 g) -----------------
# Validated in entry_harness (predguid2/captest2). Command vertical lift (cos bank) by feedback on
# drag-vs-reference: captures a shallow EI by diving when drag<ref, holds the floor-g, and a CAPPED
# lift-down on climb-out kills the dive-back overshoot (the 7-8 g artifact). cdmax nulls range.
def _predguid_rhs(tt, y, ctrl, cda, ld, peak):
    r = y[:3]; v = y[3:6]; rn = np.linalg.norm(r); alt = rn - R_EARTH
    a_g = gravity_earth_moon(r, tt)
    v_air = np.cross(np.array([0.0, 0.0, OMEGA_E]), r); v_rel = v - v_air
    vr = np.linalg.norm(v_rel); rho = atm_density(alt) * _ENTRY_DENS_SCALE
    up = r / rn
    if vr > 1.0 and rho > 1e-9:
        q = 0.5 * rho * vr * vr; vhat = v_rel / vr; m = y[6]
        D = q * cda / m
        # EXTENDED-SKIP profile v4 (return re-aim): bleed at the standard reference to
        # v_exit, then a SHALLOW ALTITUDE-CAPPED lob — full lift-up below the cap, NEUTRAL vertical
        # lift above it (the lob stays fast, low, short: minutes not hours — the slow-glide form lost
        # its range to Earth rotation over 2.7-h flights, and the uncapped loft exited super-circular
        # and wandered hemispheres). Recapture at the standard reference on drag rebuild. Range knob:
        # ctrl["v_exit"] (continuous). Absent "v_exit" -> single-phase (bit-identical).
        _lob = 0
        if "v_exit" in ctrl:
            ph = ctrl.get("phase", 0)
            if ph == 0 and vr <= ctrl["v_exit"]:
                ctrl["phase"] = ph = 1                     # loft/lob
            if ph == 1:
                _lob = 1 if alt < ctrl.get("alt_cap", 98e3) else 2
                if ctrl.get("lob_peaked") is None and alt >= ctrl.get("alt_cap", 98e3):
                    ctrl["lob_peaked"] = True
                if ctrl.get("lob_peaked") and D > 0.25 * G0:
                    ctrl["phase"] = ph = 2                 # second entry: recapture
        D_ref = ctrl["g_ref"] * G0
        alt_rate = float(np.dot(v_rel, up))
        # UPGRADE 2 (flag-gated): analytical drag-derivative LEAD compensator. dD/dt = D*(-2D/vr -
        # alt_rate/H) is the exact time-derivative of D=0.5*rho*vr^2*cda/m under an exponential atmosphere
        # (rho'/rho = -alt_rate/H) + drag-dominated decel (v' ~ -D). Feed D_pred = D + tau_lead*dD/dt into
        # the regulator so it anticipates the decel peak. Only the COMMAND uses D_pred; the drag/lift
        # physics below still use the real D. Flag OFF -> D_pred = D (bit-identical to the pre-upgrade loop).
        if globals().get("ENABLE_ENTRY_LEAD_COMP", False):
            _H_scale = 7200.0
            _dD_dt = D * (-2.0 * D / vr - alt_rate / _H_scale)
            D_pred = max(0.0, D + ctrl.get("tau_lead", ENTRY_LEAD_TAU_S) * _dD_dt)
        else:
            D_pred = D
        u = ctrl["Kp"] * (D_pred - D_ref) / D_ref - ctrl["Kd"] * alt_rate / 1000.0
        cosb = float(np.clip(u, -ctrl["cdmax"], 1.0))   # cap lift-DOWN to soften the dive-back
        if _lob == 1:
            cosb = ctrl.get("lob_cc", 1.0)               # extended skip: lob lift below the cap (1.0 =
            #   full lift-up; <1 = the smooth DOWNRANGE range-trim knob, Lever A. Absent -> 1.0, bit-ident)
        elif _lob == 2:
            cosb = 0.0                                   # extended skip: neutral vertical through the lob
        if "tlat" in ctrl:
            # CROSSRANGE: command the bank SIGN by cross-track error to the target (bank reversals
            # with a deadband). |bank| stays the range controller (cosb above); the sign steers
            # lateral. Nulls the ~405 km crossrange the single fixed sign left.
            _la_c, _lo_c = eci_to_latlon(r, tt)
            _east = np.cross(np.array([0.0, 0.0, 1.0]), up); _ne = np.linalg.norm(_east)
            if _ne > 1e-9:
                _east /= _ne; _north = np.cross(up, _east)
                _vh = v_rel - np.dot(v_rel, up) * up
                _hdg = float(np.rad2deg(np.arctan2(np.dot(_vh, _east), np.dot(_vh, _north))))
                _xtk = _crosstrack_km(_la_c, _lo_c, ctrl["tlat"], ctrl["tlon"], _hdg)
                _db = ctrl.get("xdead_km", 25.0)
                if _xtk > _db:
                    ctrl["sign"] = 1.0                   # target right -> steer right
                elif _xtk < -_db:
                    ctrl["sign"] = -1.0                  # target left  -> steer left
                # within the deadband: hold the last sign (avoids reversal chatter)
        sinb = np.sqrt(max(0.0, 1.0 - cosb * cosb)) * ctrl["sign"]
        a_drag = -D * vhat
        lift_up = up - np.dot(up, vhat) * vhat; nlu = np.linalg.norm(lift_up)
        if nlu > 1e-6:
            lift_up /= nlu
            a_lift = (q * cda * ld / m) * (cosb * lift_up + sinb * np.cross(vhat, lift_up))
        else:
            a_lift = np.zeros(3)
        a = a_drag + a_lift
        peak[0] = max(peak[0], np.linalg.norm(a) / G0)
    else:
        a = np.zeros(3)
    return np.concatenate([v, a_g + a, [0.0]])


def _predguid_fly(s0, t0, cda, ld, g_ref=PREDGUID_G_REF, Kp=3.0, Kd=6.0, cdmax=0.4, sign=1.0,
                  v_exit=None, tlat=None, tlon=None, lob_cc=None, xdead_km=None):
    """Fly the drag-tracker; (state, t_splash, peak_g, landed, range_km). v_exit (extended skip):
    bleed to this speed, shallow altitude-capped lob, recapture — see _predguid_rhs; None =
    single-phase (unchanged). tlat/tlon (with ENABLE_ENTRY_CROSSRANGE) engage bank-reversal
    crossrange control toward the target; None (or flag off) = the fixed `sign` (bit-identical).
    lob_cc (Lever A): the below-cap lob lift fraction — the smooth DOWNRANGE range-trim knob; None
    -> 1.0 (full lift-up, bit-identical). xdead_km (Lever B): crossrange deadband; None -> the rhs
    default 25.0 (bit-identical)."""
    ctrl = {"g_ref": g_ref, "Kp": Kp, "Kd": Kd, "sign": sign, "cdmax": cdmax}
    if v_exit is not None:
        ctrl["v_exit"] = float(v_exit)
    if lob_cc is not None:
        ctrl["lob_cc"] = float(lob_cc)
    if xdead_km is not None:
        ctrl["xdead_km"] = float(xdead_km)
    if (tlat is not None and tlon is not None
            and globals().get("ENABLE_ENTRY_CROSSRANGE", False)):
        ctrl["tlat"] = float(tlat); ctrl["tlon"] = float(tlon)
    pk = [0.0]
    sol = solve_ivp(lambda tt, x: _predguid_rhs(tt, x, ctrl, cda, ld, pk),
                    (t0, t0 + (12000.0 if v_exit is not None else 9000.0)),
                    np.asarray(s0, float).copy(), method="RK45",
                    rtol=1e-7, atol=1e-1, max_step=3.0, events=_entry_splash_ev)
    if len(sol.t_events[0]):
        s = sol.y_events[0][0]; t = float(sol.t_events[0][0])
        ei = eci_to_latlon(s0[:3], t0); la, lo = eci_to_latlon(s[:3], t)
        return s.copy(), t, pk[0], True, _gc_dist_km(ei[0], ei[1], la, lo)
    return sol.y[:, -1].copy(), float(sol.t[-1]), pk[0], False, -1.0


def _predguid_terminal_trim(s0, t0, tlat, tlon, cda, ld, g_ref, d_tgt):
    """Lever A DOWNRANGE terminal trim (ENABLE_ENTRY_TERMINAL_TRIM). The extended-skip range(v_exit)
    reachable set is gap-riddled (chaotic recapture), so the aim point often falls in a gap the
    open-loop v_exit grid can't reach (~60 km miss). The lob LIFT fraction lob_cc is instead a smooth,
    monotone range knob. Strategy: a coarse v_exit scan (at FULL lob lift) BRACKETS the target energy
    from above (smallest v_exit whose EI->splash range >= d_tgt + margin), then a BISECTION on lob_cc
    (range grows monotonically with lob_cc) nulls the downrange onto d_tgt. Crossrange runs on the
    tightened ENTRY_XRANGE_DEADBAND_KM. Returns (state, t, peak_g, landed, range_km) like _predguid_fly."""
    xdead = globals().get("ENTRY_XRANGE_DEADBAND_KM", 5.0)
    cc_min = globals().get("ENTRY_LOBCC_MIN", 0.35)
    def _fly(ve, cc):
        return _predguid_fly(s0, t0, cda, ld, g_ref=g_ref, cdmax=0.2, v_exit=float(ve),
                             lob_cc=float(cc), tlat=tlat, tlon=tlon, xdead_km=xdead)
    ve_pick = None
    for ve in np.arange(8000.0, 8451.0, 40.0):     # coarse energy bracket at full lob lift
        ss, tt, pkg, l, rng = _fly(ve, 1.0)
        if l and (tt - t0) < 3000.0 and rng >= d_tgt + 30.0:
            ve_pick = float(ve); break
    if ve_pick is None:
        ve_pick = 8450.0                            # d_tgt beyond the full-lift reach -> max energy, best effort
    lo, hi = cc_min, 1.0                            # bisect the smooth lob-lift range knob
    best = None
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        ss, tt, pkg, l, rng = _fly(ve_pick, mid)
        if l and (tt - t0) < 3000.0:
            miss = _gc_dist_km(*eci_to_latlon(ss[:3], tt), tlat, tlon)
            if best is None or miss < best[0]:
                best = (miss, mid)
            if rng > d_tgt:
                hi = mid                            # landed long -> less lob lift
            else:
                lo = mid                            # landed short -> more lob lift
        else:
            hi = mid                                # no clean land -> back off the lift
    cc_final = best[1] if best is not None else 1.0
    return _fly(ve_pick, cc_final)


def _predguid_solve(s0, t0, tlat, tlon, cda, ld, g_ref=PREDGUID_G_REF):
    """Crossrange sign-pick + cdmax range-null (higher cdmax = shorter range). Targets beyond
    ~5,000 km take the EXTENDED-SKIP branch: bisect v_exit (bleed-then-shallow-lob, the real
    Orion long-range profile) with a divergence guard — the cdmax family tops out ~5,000 km."""
    # With ENABLE_ENTRY_CROSSRANGE the bank SIGN is closed-loop (reversals to the target), so the
    # fixed sign becomes just an initial value the reversals refine; the tlat/tlon passed to every
    # _predguid_fly engage the lateral control. Flag OFF -> tlat/tlon ignored, the fixed-sign pick
    # below governs (bit-identical).
    _xr = globals().get("ENABLE_ENTRY_CROSSRANGE", False)
    ei = eci_to_latlon(s0[:3], t0); d_tgt = _gc_dist_km(ei[0], ei[1], tlat, tlon)
    if d_tgt > 5000.0:
        if _xr and globals().get("ENABLE_ENTRY_TERMINAL_TRIM", False):
            # Lever A: coarse v_exit bracket + smooth lob_cc bisection -> downrange ~0 (vs the
            # gap-riddled v_exit grid below, which settles ~60 km off because the aim falls in a gap).
            return _predguid_terminal_trim(s0, t0, tlat, tlon, cda, ld, g_ref, d_tgt)
        if _xr:
            # The extended-skip range(v_exit) is NON-MONOTONIC (bimodal: a small v_exit step can
            # flip which skip recaptures, jumping range ~900 km), so the old bisection stalled in
            # the gap and looked like a range ceiling (the San-Diego class, ~6,500 km). GRID-SEARCH
            # v_exit + local refine, pick the value whose splash is closest to the target (crossrange
            # control nulls the lateral). Verified: nominal 22 km from San Diego, perturbed 35-110 km,
            # peak-g ~4.4. ~25 flies (vs 14 bisection) — entry flies are cheap.
            cand = []
            for ve in np.arange(8000.0, 8451.0, 15.0):
                ss, tt, pkg, l, rng = _predguid_fly(s0, t0, cda, ld, g_ref=g_ref, cdmax=0.2,
                                                    v_exit=float(ve), tlat=tlat, tlon=tlon)
                if l and (tt - t0) < 3000.0:
                    cand.append((_gc_dist_km(*eci_to_latlon(ss[:3], tt), tlat, tlon), float(ve)))
            if cand:
                cand.sort(); ve0 = cand[0][1]
                for ve in np.arange(ve0 - 12.0, ve0 + 12.1, 4.0):   # local refine around the best cell
                    ss, tt, pkg, l, rng = _predguid_fly(s0, t0, cda, ld, g_ref=g_ref, cdmax=0.2,
                                                        v_exit=float(ve), tlat=tlat, tlon=tlon)
                    if l and (tt - t0) < 3000.0:
                        cand.append((_gc_dist_km(*eci_to_latlon(ss[:3], tt), tlat, tlon), float(ve)))
                cand.sort()
                return _predguid_fly(s0, t0, cda, ld, g_ref=g_ref, cdmax=0.2,
                                     v_exit=cand[0][1], tlat=tlat, tlon=tlon)
            return _predguid_fly(s0, t0, cda, ld, g_ref=g_ref, cdmax=0.2, v_exit=8225.0,
                                 tlat=tlat, tlon=tlon)    # nothing landed cleanly -> best-effort
        best = {}                            # legacy bisection (flag OFF): monotonic assumption
        for sgn in (1.0, -1.0):
            ss, tt, _, l, _ = _predguid_fly(s0, t0, cda, ld, g_ref=g_ref, cdmax=0.2, sign=sgn,
                                            v_exit=8150.0, tlat=tlat, tlon=tlon)
            best[sgn] = _gc_dist_km(*eci_to_latlon(ss[:3], tt), tlat, tlon) if l else 1e9
        sign = 1.0 if best[1.0] <= best[-1.0] else -1.0
        lo, hi = 8000.0, 8450.0
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            ss, tt, pkg, l, rng = _predguid_fly(s0, t0, cda, ld, g_ref=g_ref, cdmax=0.2,
                                                sign=sign, v_exit=mid, tlat=tlat, tlon=tlon)
            _la, _ = eci_to_latlon(ss[:3], tt)
            if (not l) or (tt - t0) > 3000.0 or abs(_la) > 60.0 or rng > d_tgt:
                hi = mid                    # diverged / over-range -> bleed more
            else:
                lo = mid
        return _predguid_fly(s0, t0, cda, ld, g_ref=g_ref, cdmax=0.2, sign=sign,
                             v_exit=0.5 * (lo + hi), tlat=tlat, tlon=tlon)
    best = {}
    for sgn in (1.0, -1.0):
        ss, tt, _, l, _ = _predguid_fly(s0, t0, cda, ld, g_ref=g_ref, cdmax=0.3, sign=sgn,
                                        tlat=tlat, tlon=tlon)
        best[sgn] = _gc_dist_km(*eci_to_latlon(ss[:3], tt), tlat, tlon) if l else 1e9
    sign = 1.0 if best[1.0] <= best[-1.0] else -1.0
    lo, hi = 0.0, 1.0
    for _ in range(11):
        mid = 0.5 * (lo + hi)
        ss, tt, pkg, l, rng = _predguid_fly(s0, t0, cda, ld, g_ref=g_ref, cdmax=mid, sign=sign,
                                            tlat=tlat, tlon=tlon)
        if (not l) or rng > d_tgt:        # too far / no-land -> more cap (shorter)
            lo = mid
        else:
            hi = mid
    return _predguid_fly(s0, t0, cda, ld, g_ref=g_ref, cdmax=0.5 * (lo + hi), sign=sign,
                         tlat=tlat, tlon=tlon)


def phase_entry(state, t0, perturb=None):
    """Orion SKIP-ENTRY from the entry interface to Pacific splashdown. Atmospheric
    integration (Earth+J2 gravity, drag, and lift at Orion's L/D with a bank for
    the skip), peak-g tracking, and splashdown point vs a PER-OPPORTUNITY recovery
    zone (Apollo/RTCC practice — the recovery fleet is staged down THIS trial's
    entry ground track, not at a fixed point). Applies ENABLE_HEATSHIELD_FAILURE
    (the Artemis I AVCOAT char-loss tail) and an entry-overstress failure.

    Scope: the entry flies closed-loop PredGuid drag-tracker guidance
    (ENABLE_PREDGUID_ENTRY, default ON); a fixed representative skip is the legacy/flag-off fallback. splash_miss_km = accuracy vs the planned zone;
    recovery_zone_displacement_km = the zone's offset from the nominal (return-timing
    dispersion the recovery fleet must chase). Returns success, failure_reason,
    splash_lat/lon, ei_lat/lon, recovery_zone_lat/lon, splash_miss_km,
    recovery_zone_displacement_km, peak_g, t_splash."""
    perturb = perturb or {}
    global _ENTRY_DENS_SCALE                      # per-trial atmospheric-density scale (serial per worker)
    _ENTRY_DENS_SCALE = float(perturb.get("entry_dens_scale", 1.0))
    result = {"success": True, "failure_reason": None, "entry_dens_scale": _ENTRY_DENS_SCALE}
    s = np.concatenate([np.asarray(state[:3], float),
                        np.asarray(state[3:6], float), [float(state[6])]])
    t = float(t0)

    # Entry-interface ground point + ground-track azimuth (Earth-relative) — the
    # per-opportunity recovery zone is anchored to this track.
    _ei_lat, _ei_lon = eci_to_latlon(s[:3], t)
    _up = s[:3] / np.linalg.norm(s[:3])
    _east = np.cross(np.array([0.0, 0.0, 1.0]), _up); _east /= np.linalg.norm(_east)
    _north = np.cross(_up, _east)
    _v_rel = s[3:6] - np.cross(np.array([0.0, 0.0, OMEGA_E]), s[:3])
    _vh = _v_rel - np.dot(_v_rel, _up) * _up
    _ei_az = float(np.rad2deg(np.arctan2(np.dot(_vh, _east), np.dot(_vh, _north))))

    cd = CM_CD * perturb.get("cd_factor", 1.0)
    ld = CM_LD * perturb.get("ld_factor", 1.0)
    m = s[6]
    peak = {"g": 0.0}
    flags = {"dipped": False}     # skip-suppression state (fixed-bank fallback)
    nt = globals().get("_NOMINAL_TARGETS")
    if nt is None:
        nt = {}
        globals()["_NOMINAL_TARGETS"] = nt
    rec = nt.get("recovery_splash")
    _D = nt.get("entry_downrange_km")
    _rel = nt.get("entry_track_offset_deg")
    if (rec is None or _D is None) and (globals().get("ENABLE_PHASEC_BPLANE", False)
            and globals().get("PHASEC_RECOVERY") is not None):
        # Phase-C nominal (no captures yet): aim the REAL recovery — Guadalupe at the real
        # extended-skip downrange — instead of letting the untargeted tracker define the zone.
        _prec = globals()["PHASEC_RECOVERY"]
        rec = (_prec["splash_lat"], _prec["splash_lon"])
        _D = _prec["D"]; _rel = _prec["rel"]
    guided = (globals().get("ENABLE_SKIP_ENTRY_GUIDANCE", False)
              and rec is not None and _D is not None and _rel is not None)
    # This trial's OWN per-opportunity recovery zone: the nominal downrange _D along
    # THIS EI's ground track. Always reachable by a single guided entry (it IS the
    # nominal's natural range), so the guided entry flies precisely to it.
    _zone = (_gc_dest(_ei_lat, _ei_lon, _ei_az + _rel, _D)
             if (_D is not None and _rel is not None) else None)

    if guided:
        # CLOSED-LOOP guided skip entry to THIS trial's per-opportunity recovery zone
        # (Apollo/RTCC practice). NOT the FIXED nominal splash: off-track (FPA-only
        # fallback) trials land thousands of km from it, and chasing that unreachable
        # point made the guided entry bank full lift-up -> SKIP OUT -> steep re-entry
        # -> 17-35 g overstress (the dominant artifact). The recovery fleet repositions
        # to the zone (recovery_zone_displacement_km); the guidance flies precisely to
        # it (splash_miss_km), so the entry is always survivable.
        _s_ei, _t_ei = s.copy(), t      # preserve the EI state for a possible re-fly
        if globals().get("ENABLE_PREDGUID_ENTRY", False):
            # PredGuid DRAG-TRACKER (shallow corridor, ~5 g): captures + range-nulls to the zone.
            s, t, _pkg, _landed, _ = _predguid_solve(s, t, _zone[0], _zone[1], cd * CM_AREA, ld)
        else:
            s, t, _pkg, _landed = _guided_entry_fly(s, t, _zone[0], _zone[1], cd * CM_AREA, ld)
        # HARDEN: the predictor-corrector + skip-suppression spikes erratically on the rare
        # steep/skip-out tail (17-91 g — non-physical artifacts). When it spikes >structural-g
        # or fails to land, RE-FLY with the stable constant-bank solve (monotonic, spike-free)
        # so the trial gets its TRUE peak-g — rescuing the moderate-steep artifacts and leaving
        # only genuine steep overstress. The well-behaved majority (<=structural g) is untouched.
        if (not _landed) or _pkg > ENTRY_STRUCTURAL_G:
            _s2, _t2, _pkg2, _l2 = _solve_stable_entry(_s_ei, _t_ei, _zone[0], _zone[1],
                                                        cd * CM_AREA, ld)
            if _l2:
                s, t, _pkg, _landed = _s2, _t2, _pkg2, _l2
        peak["g"] = _pkg
        if not _landed:
            result.update(success=False, failure_reason="no_splashdown",
                          state=s, t_end=t, peak_g=float(_pkg))
            return result
    elif globals().get("ENABLE_PREDGUID_ENTRY", False):
        # NOMINAL (zone-defining) PredGuid: untargeted drag-tracker at the default cap -> its natural
        # downrange DEFINES the per-opportunity zone D (captured below), just like the fixed-bank did.
        s, t, _pkg, _landed, _ = _predguid_fly(s, t, cd * CM_AREA, ld, cdmax=PREDGUID_NOM_CDMAX)
        peak["g"] = _pkg
        if not _landed:
            result.update(success=False, failure_reason="no_splashdown",
                          state=s, t_end=t, peak_g=float(_pkg))
            return result
    else:
        # FIXED-bank skip (legacy nominal run — defines the recovery splash — or flag off).
        def rhs(tt, y):
            r = y[:3]; v = y[3:6]; rn = np.linalg.norm(r); alt = rn - R_EARTH
            a_grav = gravity_earth_moon(r, tt)
            v_air = np.cross(np.array([0.0, 0.0, OMEGA_E]), r)
            v_rel = v - v_air; vr = np.linalg.norm(v_rel)
            rho = atm_density(alt) * _ENTRY_DENS_SCALE
            if vr > 1.0 and rho > 1e-9:
                q = 0.5 * rho * vr * vr
                vhat = v_rel / vr
                a_drag = -(q * cd * CM_AREA / m) * vhat
                up = r / rn
                if alt < 85_000.0:
                    flags["dipped"] = True
                alt_rate = float(np.dot(v_rel, up))
                if flags["dipped"] and alt_rate > 5.0 and alt > 75_000.0:
                    bank = np.deg2rad(160.0)            # lift down -> pull back in
                else:
                    bank = np.deg2rad(ENTRY_BANK_DEG)   # lift up
                lift_up = up - np.dot(up, vhat) * vhat
                nlu = np.linalg.norm(lift_up)
                if nlu > 1e-6:
                    lift_up = lift_up / nlu
                    side = np.cross(vhat, lift_up)
                    lift_dir = np.cos(bank) * lift_up + np.sin(bank) * side
                    a_lift = (q * cd * ld * CM_AREA / m) * lift_dir
                else:
                    a_lift = np.zeros(3)
                a_aero = a_drag + a_lift
                peak["g"] = max(peak["g"], np.linalg.norm(a_aero) / G0)
            else:
                a_aero = np.zeros(3)
            return np.concatenate([v, a_grav + a_aero, [0.0]])

        def ev_splash(tt, y):
            return np.linalg.norm(y[:3]) - R_EARTH      # sea level
        ev_splash.terminal = True; ev_splash.direction = -1

        try:
            sol = solve_ivp(rhs, (t, t + 7200.0), s, method="RK45",
                            rtol=1e-8, atol=1e-1, max_step=2.0, events=ev_splash)
        except Exception as e:
            result.update(success=False, failure_reason=f"integration_error: {e}")
            return result
        if len(sol.t_events[0]) > 0:
            s = sol.y_events[0][0].copy(); t = float(sol.t_events[0][0])
        else:
            s = sol.y[:, -1].copy(); t = float(sol.t[-1])
            result.update(success=False, failure_reason="no_splashdown",
                          state=s, t_end=t, peak_g=float(peak["g"]))
            return result

    lat, lon = eci_to_latlon(s[:3], t)
    # EDL parachute descent: below ~drogue deploy the capsule descends under chutes at terminal
    # velocity and DRIFTS with the wind (the entry integration omits this). Offset the splash by the
    # wind drift (~km) and add the descent time. Nominal (no wind perturbation) → zero drift.
    _desc_s = 0.0
    if globals().get("ENABLE_EDL_DESCENT", False):
        _desc_s = ((DROGUE_DEPLOY_ALT_M - MAIN_DEPLOY_ALT_M) / DROGUE_DESCENT_MS
                   + MAIN_DEPLOY_ALT_M / SPLASH_TERMINAL_MS)
        _wind = np.asarray(perturb.get("edl_wind_ms", np.zeros(2)), float)   # [East, North] m/s
        _drift = _wind * _desc_s                                             # [E, N] meters
        _dkm = float(np.hypot(_drift[0], _drift[1]) / 1000.0)
        if _dkm > 1e-3:
            lat, lon = _gc_dest(lat, lon, float(np.rad2deg(np.arctan2(_drift[0], _drift[1]))), _dkm)
        result["edl_descent_s"] = float(_desc_s)
        result["edl_splash_vel_ms"] = SPLASH_TERMINAL_MS
        result["edl_wind_drift_km"] = _dkm
        _dbg("edl_wind_drift_km", _dkm)
    # ENTRY terminal-nav residual: the delivered EI-state OD knowledge error the guidance can't correct
    # -> a splash-position offset (calibrated to the real ~4.7 km accuracy; the OD FILTER later scales it
    # from the emergent EI covariance -> the un-masking hook). Like OD_NAV_RESIDUAL + the wind drift: a
    # drawn offset, not a re-solve. Nominal (no perturb) -> zero. Stacks onto the wind drift.
    if globals().get("ENABLE_ENTRY_NAV_RESIDUAL", False):
        _nu = np.asarray(perturb.get("entry_nav_resid_unit", np.zeros(2)), float)   # [East, North] unit
        _navkm = float(np.hypot(_nu[0], _nu[1]) * ENTRY_NAV_RESIDUAL_KM)
        if _navkm > 1e-3:
            lat, lon = _gc_dest(lat, lon, float(np.rad2deg(np.arctan2(_nu[0], _nu[1]))), _navkm)
        result["entry_nav_resid_km"] = _navkm
        _dbg("entry_nav_resid_km", _navkm)
    result["splash_lat"] = float(lat)
    result["splash_lon"] = float(lon)
    result["peak_g"] = float(peak["g"])
    result["t_splash"] = t + _desc_s
    result["ei_lat"] = float(_ei_lat)
    result["ei_lon"] = float(_ei_lon)

    # --- recovery accuracy ----------------------------------------------------
    # Both guided and fixed-bank fly to THIS trial's per-opportunity recovery zone
    # (down its own EI ground track at the nominal downrange) -> splash_miss = actual
    # vs that zone (guidance accuracy). The nominal run (fixed-bank) DEFINES the zone
    # vector (D, track offset) + the nominal recovery point. recovery_zone_displacement
    # = how far this trial's zone sits from the nominal one (the operational cost the
    # recovery fleet bears for off-nominal/rev-slipped returns).
    D = nt.get("entry_downrange_km")
    rel = nt.get("entry_track_offset_deg")
    if D is None:                       # nominal run: capture the recovery splash + vector
        D = _gc_dist_km(_ei_lat, _ei_lon, lat, lon)
        rel = _gc_bearing(_ei_lat, _ei_lon, lat, lon) - _ei_az
        nt["entry_downrange_km"] = float(D)
        nt["entry_track_offset_deg"] = float(rel)
        nt["recovery_zone_nominal"] = [float(lat), float(lon)]
        nt["recovery_splash"] = [float(lat), float(lon)]
    zlat, zlon = _gc_dest(_ei_lat, _ei_lon, _ei_az + rel, D)
    result["recovery_zone_lat"] = float(zlat)
    result["recovery_zone_lon"] = float(zlon)
    nz = nt.get("recovery_zone_nominal", [lat, lon])
    result["splash_miss_km"] = _gc_dist_km(lat, lon, zlat, zlon)
    result["recovery_zone_displacement_km"] = _gc_dist_km(zlat, zlon, nz[0], nz[1])

    # failure modes
    if peak["g"] > ENTRY_STRUCTURAL_G:
        result.update(success=False, failure_reason="overstress")
        return result
    if perturb.get("heatshield_failed", False):
        result.update(success=False, failure_reason="heatshield_loss")
        return result

    result["state"] = s
    result["t_end"] = t
    return result


# ============================================================
# Phase timeline  (Artemis I reference durations)
# ============================================================
# Segment boundaries marked by _mark() in run_mission, in order.
PHASE_SEGMENTS = [
    ("liftoff",        "insertion",      "Launch to insertion"),
    ("insertion",      "prm",            "Parking orbit to PRM"),
    ("prm",            "tli",            "PRM to TLI"),
    ("tli",            "opf",            "Outbound coast (to OPF)"),
    ("opf",            "dri",            "OPF to DRO insertion"),
    ("dri",            "ddp",            "DRO coast"),
    ("ddp",            "rpf",            "DRO departure to RPF"),
    ("rpf",            "entry_interface","Trans-earth coast"),
    ("entry_interface","splashdown",     "Entry to splashdown"),
]
# Artemis I as-flown reference durations (s), derived from the published per-event
# ground-elapsed times (NASA Artemis I Mission Timeline; OPF burn start 12:44 UTC
# 2022-11-21 per Spaceflight Now). See the mission references for the full GET table.
# Boundary GETs used: liftoff 0; insertion(MECO) 000/00:08:03; TLI ign 000/01:29:27;
# OPF burn 005/05:56:16; DRI 009/15:48:27; DDP 015/15:49:56; RPF 019/10:24:00;
# EI 025/11:15:52; splash 025/11:36:27. (Splash GET totals ~25.48 d; the published
# "25 d 10 h 53 m" headline is ~43 min shorter — a minor inconsistency in NASA's
# own figures, ~0.12%. These per-event GETs are used as the self-consistent set.)
ARTEMIS_PHASE_DUR_S = {
    "Launch to insertion":        483.0,   # liftoff -> core MECO / ICPS sep
    "Parking orbit to PRM":      2693.0,   # MECO (000/00:08:03) -> PRM at apogee (000/00:52:56)
    "PRM to TLI":                2191.0,   # PRM -> TLI ignition (000/01:29:27); sim "tli" mark is at
                                           # TLI BURNOUT, so this segment reads ~burn-length long (the
                                           # pre-existing ignition-vs-burnout quirk, now isolated to this leg)
    "Outbound coast (to OPF)": 448009.0,   # TLI -> outbound powered flyby (FD6)
    "OPF to DRO insertion":    381131.0,   # OPF -> DRI (FD10)
    "DRO coast":               518489.0,   # DRI -> DRO departure (FD16); ~6.0 d
    "DRO departure to RPF":    326044.0,   # DDP -> return powered flyby (FD20)
    "Trans-earth coast":       521512.0,   # RPF -> entry interface (FD26); ~6.0 d
    "Entry to splashdown":       1235.0,   # EI -> splash (~20.6 min; skip entry)
}


def build_phase_timeline(phase_log):
    """Convert raw _mark() events [(event, get_s, wall_s), ...] into a per-phase
    timeline with mission-elapsed duration and compute (wall) time per phase.
    Phases not reached (early failure) are omitted."""
    by_event = {ev: (get, wall) for ev, get, wall in (phase_log or [])}
    out = []
    for a, b, label in PHASE_SEGMENTS:
        if a in by_event and b in by_event:
            ga, wa = by_event[a]
            gb, wb = by_event[b]
            out.append({"phase": label,
                        "get_start_s": round(ga, 2),
                        "duration_s": round(gb - ga, 2),
                        "compute_s": round(wb - wa, 4)})
    return out


# ============================================================
# Single mission
# ============================================================
def _ric_basis(s6):
    """RIC basis of state s6 (r,v): (R=radial, I=in-track ~along v, C=cross-track/orbit-normal)."""
    r = np.asarray(s6[:3], float); v = np.asarray(s6[3:6], float)
    R = r / np.linalg.norm(r)
    C = np.cross(r, v); C = C / np.linalg.norm(C)
    return R, np.cross(C, R), C


def _clnav_inertial_error(s6, err_ric):
    """Closed-loop NAV: rotate a RIC-frame state-estimate error [pr,pi,pc, vr,vi,vc] into an inertial
    6-vec (pos m + vel m/s) error, using the RIC basis of state s6 (r,v)."""
    R, I, C = _ric_basis(s6)
    pos = err_ric[0] * R + err_ric[1] * I + err_ric[2] * C
    vel = err_ric[3] * R + err_ric[4] * I + err_ric[5] * C
    return np.concatenate([pos, vel])


# ---- OD FILTER (rung a): epoch capture (nominal only) + covariance builder + accessor -------------
_OD_CAP = None   # set to a dict during the nominal build to record (epoch_key -> (t, state6))


def _od_capture(key, t, s6):
    """Record a covariance-epoch (time, state) during the nominal build. No-op otherwise (trial path)."""
    if _OD_CAP is not None and key not in _OD_CAP:
        _OD_CAP[key] = (float(t), np.asarray(s6, float)[:6].copy())


def _od_station_fn(t, r_v):
    """Visible DSN stations at time t for a vehicle at r_v (ECI): list of (r_s, v_s) above the mask."""
    vis = []
    for (lat, lon, alt) in OD_DSN_STATIONS:
        r_s, v_s = _odf.station_eci(lat, lon, alt, float(t), R_EARTH, OMEGA_E, _GMST0)
        if _odf.visible(r_v, r_s, OD_ELEV_MASK_DEG):
            vis.append((r_s, v_s))
    return vis


def _build_od_filter_covariances(nt):
    """Build the emergent STM-based LinCov covariances ONCE on the nominal and store their Cholesky
    factors in _NOMINAL_TARGETS (→ nominal_targets.json → shards). Guarded (skip if already built /
    pinned). Uses the epochs captured during the nominal run (_OD_CAP). For each epoch the covariance is
    propagated with the state-transition matrix over the tracking arc (velocity coupling), bounded by an
    informative prior + a systematic floor, and calibrated (effective noise) to the delivered-OD tables.
    RIC sites store L in the RIC frame (err = L@unit, then _clnav_inertial_error rotates); DRI in ECI.
    tli_init is EXCLUDED (LEO parking orbit is DSN-blind → the GPS-derived CLNAV_TLI_INIT stands)."""
    if not globals().get("ENABLE_OD_FILTER", False) or "od_L_dri" in nt:
        return
    caps = dict(_OD_CAP or {})
    sig = {"doppler_ms": OD_DOPPLER_SIGMA_MS, "range_m": OD_RANGE_SIGMA_M, "ddor_rad": OD_DDOR_SIGMA_RAD}
    prior = [OD_PRIOR_POS_M] * 3 + [OD_PRIOR_VEL_MS] * 3
    plan = [("otc0", OD_ARC_OTC_S[0], "ric"), ("otc1", OD_ARC_OTC_S[1], "ric"),
            ("otc2", OD_ARC_OTC_S[2], "ric"), ("otc3", OD_ARC_OTC_S[3], "ric"),
            ("otc4", OD_ARC_OTC_S[4], "ric"), ("otc5", OD_ARC_OTC_S[5], "ric"),
            ("dri",  OD_ARC_DRI_S,     "eci")]

    def _floor_for(key):
        # per-epoch systematic floor = the VALIDATED delivered-OD table's magnitude (geometric mean of
        # its RIC axes) — the tables are calibrated to reproduce the as-flown OTC trim pattern, so the
        # emergent P reproduces that magnitude while the STM + ΔDOR supply the LOS-oriented anisotropy.
        if key.startswith("otc"):
            slot = int(key[3:])
            fp = float(np.exp(np.mean(np.log(OTC_OD_POS_SIGMA_M[slot]))))
            fv = float(np.exp(np.mean(np.log(OTC_OD_VEL_SIGMA_MS[slot]))))
        else:                                        # dri (isotropic delivered residual)
            fp, fv = float(OD_NAV_POS_SIGMA_M), float(OD_NAV_VEL_SIGMA_MS)
        return fp, fv

    for key, arc, frame in plan:
        if key not in caps:
            continue   # not captured this run -> diagonal fallback at the injection site (bit-identical)
        t_ep, s6 = caps[key]
        fp, fv = _floor_for(key)
        P = _odf.accumulate_covariance_stm(gravity_earth_moon, t_ep, s6, arc, OD_ARC_SAMPLES,
                                           _od_station_fn, sig, key in OD_DDOR_EPOCHS, prior,
                                           floor_pos_m=fp, floor_vel_ms=fv,
                                           q_pos_m=OD_Q_POS_M, q_vel_ms=OD_Q_VEL_MS)
        if frame == "ric":
            R, I, C = _ric_basis(s6)
            P = _odf.rotate_eci_to_ric(P, R, I, C)
        nt[f"od_L_{key}"] = _odf.chol_lower(P)


def _od_filter_L(key):
    """Return the pinned Cholesky factor for an epoch, or None (→ diagonal-σ fallback, bit-identical)."""
    if not globals().get("ENABLE_OD_FILTER", False):
        return None
    nt = globals().get("_NOMINAL_TARGETS")
    if not isinstance(nt, dict):
        return None
    L = nt.get(f"od_L_{key}")
    return np.asarray(L, float) if L is not None else None


def run_mission(perturb=None, capture_trajectories=False):
    """Run one complete Artemis I mission; return (results_dict, trajectories).

    Sequence:
      1.  SLS launch: pad -> MECO -> core-stage sep (Orion+ICPS in insertion orbit)
      2.  ICPS Perigee-Raise + Trans-Lunar Injection
      3.  Outbound coast (ICPS jettison, solar arrays) + OTC chain
      4.  Outbound Powered Flyby (OPF) — lunar gravity assist toward the DRO
      5.  DRO Insertion (DRI)
      6.  DRO coast (~6 d; max Earth distance milestone)
      7.  DRO Departure (DDP)
      8.  Return coast (DRO -> Moon)
      9.  Return Powered Flyby (RPF) — gravity assist toward Earth
      10. Trans-Earth coast + RTC chain
      11. CM/ESM separation
      12. Skip-entry & Pacific splashdown

    Failure handling uses strict causal order: at most one failure
    per trial recorded in `mission_failure`, early return on a hard failure.
    """
    perturb = perturb or {}
    results = {}
    trajectories = {}

    # Per-phase timing log (event, mission-elapsed GET s, compute wall s).
    _wall0 = time.time()
    _phase_log = []
    results["_phase_log"] = _phase_log
    def _mark(event, get_s):
        _phase_log.append((event, float(get_s), round(time.time() - _wall0, 4)))
    _mark("liftoff", 0.0)

    # Per-trial DEBUG record: a fresh dict the phase functions write diagnostics into via _dbg()
    # (maneuver dv's, EI state, the OD-nav/return-nav engagement decisions, fallback flags, the
    # captured-targets-present flags). Saved into each trial's JSON as "_debug" so any trial can be
    # inspected post-hoc WITHOUT flipping flags + re-running. Side-channel only — never affects the
    # RNG stream or the trajectory (determinism preserved).
    _dbgd = {}
    globals()["_TRIAL_DBG"] = _dbgd
    results["_debug"] = _dbgd
    _dbg("flags", {f: bool(globals().get(f, False)) for f in
                   ("ENABLE_OD_NAV", "ENABLE_DDP_RECOVERY", "ENABLE_LUNAR_LIBRATION",
                    "ENABLE_DEPARTURE_PHASING", "ENABLE_RPF_RECOVERY_TARGET")})
    _nt_dbg = globals().get("_NOMINAL_TARGETS")
    _dbg("targets_present", sorted(_nt_dbg.keys()) if isinstance(_nt_dbg, dict) else None)

    # ESM catastrophic systems failure (Apollo-13-class; PROB_ESM_CATASTROPHIC).
    # Struck at a uniform timeline fraction; for the uncrewed mission a
    # catastrophic ESM (propulsion/power) failure while the ESM is active
    # (TLI handoff -> CM/SM separation) is mission-ending. Checked at each milepost.
    # Timeline-struck SYSTEMS failures: ESM catastrophic (Apollo-13 analogue),
    # avionics/power radiation (Artemis I latching-limiter), and the foreseeable
    # cislunar modes (MMOD, nav-sensor loss, ESM pressurization, comm-at-burn, RCS,
    # thermal, SPE, DRO station-keeping). Each drawn mode strikes at frac ×
    # MISSION_REF_DURATION and is mission-ending; checked at each milepost with the
    # EARLIEST-struck winning (strict causal order, one failure/trial).
    _sys_events = []   # (struck_time_s, mission_failure_label, result_get_key)
    if perturb.get("esm_failure", False):
        _sys_events.append((float(perturb.get("esm_failure_frac", 0.5)) * MISSION_REF_DURATION_S,
                            "esm_systems_failure", "esm_failure_get_d"))
    if perturb.get("avionics_anomaly", False):
        _sys_events.append((float(perturb.get("avionics_frac", 0.5)) * MISSION_REF_DURATION_S,
                            "avionics_radiation_anomaly", "avionics_failure_get_d"))
    for _nm in ("mmod_strike", "nav_sensor_loss", "esm_pressurization", "comm_loss_at_burn",
                "rcs_failure", "thermal_loss", "solar_particle_event", "dro_stationkeeping"):
        if perturb.get(f"ff_{_nm}", False):
            _sys_events.append((float(perturb.get(f"ff_{_nm}_frac", 0.5)) * MISSION_REF_DURATION_S,
                                _nm, f"ff_{_nm}_get_d"))
    def _esm_check(t_now):
        struck = [e for e in _sys_events if e[0] <= t_now]
        if struck:
            tt, lab, key = min(struck)
            results["full_success"] = False
            results["mission_failure"] = lab
            results[key] = tt / 86400.0
            return True
        return False

    # ---- 1. SLS launch -----------------------------------------------------
    launch = phase_sls_launch(perturb); _dbg_phase("launch", launch)
    results["launch_success"] = launch.get("success")
    results["launch_failure_reason"] = launch.get("failure_reason")
    results["launch_meco_mode"] = launch.get("meco_mode")
    results["launch_azimuth_deg"] = launch.get("launch_azimuth_deg")
    if capture_trajectories and "trajectory_t" in launch:
        trajectories["launch"] = (launch["trajectory_t"], launch["trajectory_y"])
    if not launch.get("success"):
        results["full_success"] = False
        results["mission_failure"] = "launch_" + str(launch.get("failure_reason", "unknown"))
        return results, trajectories
    state, t = launch["state"], launch.get("t_insertion", 0.0)
    # Closed-loop NAV (ascent): the IGM nulled its ESTIMATED insertion miss, so the TRUE insertion
    # state is off by the GPS-aided-INS nav error (TLI then re-targets from this perturbed state).
    if globals().get("ENABLE_CLOSED_LOOP_NAV", False) and perturb.get("asc_nav_unit") is not None:
        _eric = np.asarray(perturb["asc_nav_unit"]) * np.array([*CLNAV_ASC_POS_SIGMA_M, *CLNAV_ASC_VEL_SIGMA_MS])
        _ei = _clnav_inertial_error(state, _eric)
        state = np.asarray(state, float).copy(); state[:6] += _ei
        _dbg("clnav_asc", {"pos_err_m": float(np.linalg.norm(_ei[:3])), "vel_err_ms": float(np.linalg.norm(_ei[3:6]))})
    _mark("insertion", t)

    # ---- 2a. ICPS Perigee-Raise Maneuver (PRM) ----------------------------
    prm = phase_icps_prm(state, t, perturb); _dbg_phase("prm", prm)
    if not prm.get("success"):
        results["full_success"] = False
        # "tli_"-prefixed for failure-LABEL compatibility with pre-split runs (the
        # dashboard keys off these exact strings; the PRM reasons icps_ignition_failure /
        # prm_propellant_depleted reproduce the old "tli_icps_ignition_failure" / "tli_prm_*").
        results["mission_failure"] = "tli_" + str(prm.get("failure_reason", "unknown"))
        return results, trajectories
    state, t = prm["state"], prm["t_end"]
    _mark("prm", t)

    # ---- 2b. Trans-Lunar Injection (TLI) ----------------------------------
    tli = phase_icps_tli(state, t, perturb); _dbg_phase("tli", tli)
    if not tli.get("success"):
        results["full_success"] = False
        results["mission_failure"] = "tli_" + str(tli.get("failure_reason", "unknown"))
        return results, trajectories
    state, t = tli["state"], tli["t_end"]
    # Closed-loop NAV (TLI): the steered burn cut on the ESTIMATED apogee, so the TRUE post-TLI state
    # carries the IMU nav error (the in-track velocity term dominates the downstream path dispersion).
    # v1 = discrete offset HERE; v2 (ENABLE_CLNAV_CONTINUOUS_TLI) produces the error WITHIN the burn
    # (cutoff-on-estimate) -> skip this discrete offset to avoid double-counting.
    if (globals().get("ENABLE_CLOSED_LOOP_NAV", False) and perturb.get("tli_nav_unit") is not None
            and not globals().get("ENABLE_CLNAV_CONTINUOUS_TLI", False)):
        _eric = np.asarray(perturb["tli_nav_unit"]) * np.array([*CLNAV_TLI_POS_SIGMA_M, *CLNAV_TLI_VEL_SIGMA_MS])
        _ei = _clnav_inertial_error(state, _eric)
        state = np.asarray(state, float).copy(); state[:6] += _ei
        _dbg("clnav_tli", {"pos_err_m": float(np.linalg.norm(_ei[:3])), "vel_err_ms": float(np.linalg.norm(_ei[3:6]))})
    _mark("tli", t)
    if _esm_check(t):
        return results, trajectories

    # ---- 3. Outbound coast + OTC ------------------------------------------
    coast = phase_outbound_coast(state, t, perturb); _dbg_phase("outbound_coast", coast)
    # TLI PLAN-ADAPTIVITY (ENABLE_TLI_PLAN_ADAPT, retry-on-artifact-failure): an outbound ARTIFACT
    # death (missed SOI / lunar-impact — the IGM-fallback stale-aim class) triggers ONE retry: re-fly
    # TLI from the same post-PRM state with the min-ΔV Lambert replan (ignition TIME + v1 re-picked on
    # the trial's own orbit), then re-coast with the Phase-C chain (+ its periselene-floor guard).
    # Real ops would have replanned the TLI for the achieved orbit rather than fly a stale plan into
    # a miss. Non-regressive BY CONSTRUCTION (successes never retried); deterministic (same perturb
    # arrays, stateless reads, no RNG); flag-OFF bit-identical (branch not taken).
    if (not coast.get("success", True)
            and globals().get("ENABLE_TLI_PLAN_ADAPT", False)
            and str(coast.get("failure_reason")) in ("missed_lunar_approach",
                                                     "lunar_impact_trajectory")):
        globals()["_TLI_FORCE_REPLAN"] = True
        try:
            tli2 = phase_icps_tli(prm["state"], prm["t_end"], perturb)
        finally:
            globals()["_TLI_FORCE_REPLAN"] = False
        _dbg_phase("tli_retry", tli2)
        if tli2.get("success"):
            state2, t2 = tli2["state"], tli2["t_end"]
            if (globals().get("ENABLE_CLOSED_LOOP_NAV", False) and perturb.get("tli_nav_unit") is not None
                    and not globals().get("ENABLE_CLNAV_CONTINUOUS_TLI", False)):
                _eric2 = np.asarray(perturb["tli_nav_unit"]) * np.array([*CLNAV_TLI_POS_SIGMA_M, *CLNAV_TLI_VEL_SIGMA_MS])
                _ei2 = _clnav_inertial_error(state2, _eric2)
                state2 = np.asarray(state2, float).copy(); state2[:6] += _ei2
            globals()["_TLI_RETRY_COAST"] = True     # peri-floor guard active ONLY on the retry coast
            try:
                coast2 = phase_outbound_coast(state2, t2, perturb)
            finally:
                globals()["_TLI_RETRY_COAST"] = False
            _dbg_phase("outbound_coast_retry", coast2)
            results["tli_replan_retry"] = True
            results["tli_adapt"] = tli2.get("tli_adapt")
            tli, coast = tli2, coast2          # the replanned mission is what flew (honest either way)
            results["post_tli_c3_km2s2"] = tli2.get("post_tli_c3_km2s2")
    if not coast.get("success", True):       # OTC Lambert-fail / never-entered-SOI / coast error
        results["full_success"] = False      # — was SILENTLY IGNORED (and otc_coast_error returns
        results["mission_failure"] = "outbound_" + str(coast.get("failure_reason", "unknown"))
        return results, trajectories          # no "state" -> the unchecked thread would KeyError)
    state, t = coast["state"], coast["t_end"]
    for _k in ("otc1_dv_ms", "otc1_corr_dv_ms", "otc2_dv_ms", "otc3_dv_ms", "otc4_dv_ms",
               "otc_dv_total_ms", "lunar_closest_alt_km"):   # (were debug-JSON-only; now CSV columns)
        if _k in coast:
            results[_k] = coast[_k]

    # ---- 4. Outbound Powered Flyby ----------------------------------------
    opf = phase_outbound_powered_flyby(state, t, perturb); _dbg_phase("opf", opf)
    if not opf.get("success"):
        results["full_success"] = False
        results["mission_failure"] = "opf_" + str(opf.get("failure_reason", "unknown"))
        return results, trajectories
    state, t = opf["state"], opf["t_end"]
    _mark("opf", t)
    if _esm_check(t):
        return results, trajectories

    # ---- 5. DRO Insertion --------------------------------------------------
    dri = phase_dro_insertion(state, t, perturb); _dbg_phase("dri", dri)
    if not dri.get("success"):
        results["full_success"] = False
        results["mission_failure"] = "dri_" + str(dri.get("failure_reason", "unknown"))
        return results, trajectories
    state, t = dri["state"], dri["t_end"]
    _mark("dri", t)
    if _esm_check(t):
        return results, trajectories

    # ---- 6. DRO coast ------------------------------------------------------
    dro = phase_dro_coast(state, t, perturb); _dbg_phase("dro_coast", dro)
    if not dro.get("success"):
        results["full_success"] = False
        results["mission_failure"] = "dro_" + str(dro.get("failure_reason", "unknown"))
        return results, trajectories
    state, t = dro["state"], dro["t_end"]
    results["max_earth_distance_km"] = dro.get("max_earth_distance_km")
    if _esm_check(t):
        return results, trajectories

    # ---- 7. DRO Departure --------------------------------------------------
    ddp = phase_dro_departure(state, t, perturb); _dbg_phase("ddp", ddp)
    if not ddp.get("success"):
        results["full_success"] = False
        results["mission_failure"] = "ddp_" + str(ddp.get("failure_reason", "unknown"))
        return results, trajectories
    state, t = ddp["state"], ddp["t_end"]
    _mark("ddp", t)

    # ---- 8. Return coast ---------------------------------------------------
    rcoast = phase_return_coast(state, t, perturb); _dbg_phase("return_coast", rcoast)
    state, t = rcoast["state"], rcoast["t_end"]

    # ---- 9. Return Powered Flyby ------------------------------------------
    rpf = phase_return_powered_flyby(state, t, perturb); _dbg_phase("rpf", rpf)
    if not rpf.get("success"):
        results["full_success"] = False
        results["mission_failure"] = "rpf_" + str(rpf.get("failure_reason", "unknown"))
        return results, trajectories
    state, t = rpf["state"], rpf["t_end"]
    _mark("rpf", t)
    if _esm_check(t):
        return results, trajectories

    # ---- 10. Trans-Earth coast + RTC --------------------------------------
    te = phase_transearth_coast(state, t, perturb); _dbg_phase("transearth", te)
    if not te.get("success", True):                  # e.g. esm_propellant_depleted_rtc —
        results["full_success"] = False              # previously SILENTLY IGNORED (false success)
        results["mission_failure"] = "transearth_" + str(te.get("failure_reason", "unknown"))
        results["entry_fpa_deg"] = te.get("entry_fpa_deg")
        results["entry_velocity_ms"] = te.get("entry_velocity_ms")
        return results, trajectories
    state, t = te["state"], te["t_end"]
    # Final ESM-active milepost (~25.4 d, just before CM/SM sep): closes the post-RPF
    # coverage gap so systems strikes drawn into the last ~6 d are no longer dropped.
    if globals().get("ENABLE_LATE_SYSTEMS_CHECK", False) and _esm_check(t):
        return results, trajectories
    # Record EI conditions NOW (before entry) so they survive an entry failure
    # (overstress/skip) for diagnostics — the success-path enrichment below is too
    # late for failed trials.
    results["entry_fpa_deg"] = te.get("entry_fpa_deg")
    results["entry_velocity_ms"] = te.get("entry_velocity_ms")

    # ---- 11. CM/ESM separation --------------------------------------------
    sep = phase_cm_sm_sep(state, t, perturb); _dbg_phase("cm_sm_sep", sep)
    if perturb.get("cm_sm_sep_failed"):     # sep-bolt failure (sourced; Artemis I 3/4 eroded)
        results["full_success"] = False
        results["mission_failure"] = "cm_sm_separation_failure"
        return results, trajectories
    state, t = sep["state"], sep["t_end"]
    _mark("entry_interface", t)

    # ---- 12. Skip-entry & splashdown --------------------------------------
    entry = phase_entry(state, t, perturb); _dbg_phase("entry", entry)
    results["splash_lat"] = entry.get("splash_lat")
    results["splash_lon"] = entry.get("splash_lon")
    results["splash_miss_km"] = entry.get("splash_miss_km")
    results["recovery_zone_lat"] = entry.get("recovery_zone_lat")
    results["recovery_zone_lon"] = entry.get("recovery_zone_lon")
    results["recovery_zone_displacement_km"] = entry.get("recovery_zone_displacement_km")
    results["ei_lat"] = entry.get("ei_lat")
    results["ei_lon"] = entry.get("ei_lon")
    results["entry_peak_g"] = entry.get("peak_g")
    results["entry_dens_scale"] = entry.get("entry_dens_scale")   # per-trial atmospheric-density scale
    results["entry_nav_resid_km"] = entry.get("entry_nav_resid_km")   # EI-nav->splash residual (OD hook)
    if not entry.get("success"):
        results["full_success"] = False
        results["mission_failure"] = "entry_" + str(entry.get("failure_reason", "unknown"))
        return results, trajectories
    if perturb.get("parachute_failed"):     # Orion 2-of-3 mains fail to deploy (sourced;
        results["full_success"] = False     # OIG flagged char-loss debris -> chute risk)
        results["mission_failure"] = "parachute_failure"
        return results, trajectories
    _mark("splashdown", entry.get("t_splash", t))

    # Enrich the (scalar) results with key per-trial diagnostics for the dashboard
    # (all phases ran on a successful trial, so these dicts are in scope here).
    results["prm_dv_ms"] = prm.get("prm_dv_ms")        # PRM diagnostics now live in the split `prm` phase
    results["prm_burn_s"] = prm.get("prm_burn_s")
    results["parking_perigee_km"] = prm.get("parking_perigee_km")
    results["parking_apogee_km"] = prm.get("parking_apogee_km")
    results["tli_dv_ms"] = tli.get("tli_dv_ms")
    results["post_tli_c3_km2s2"] = tli.get("post_tli_c3_km2s2")
    results["dri_dv_ms"] = dri.get("dri_dv_ms")
    results["dro_snap_km"] = dri.get("dro_snap_km")
    results["opf_dv_ms"] = opf.get("opf_dv_ms")             # (was debug-JSON-only — reporting-gap fix)
    for _k in ("otc5_dv_ms", "otc6_dv_ms"):                 # post-OPF transit slots (phase-5)
        if _k in dri:
            results[_k] = dri[_k]
    results["ddp_dv_ms"] = ddp.get("ddp_dv_ms")
    results["rpf_dv_ms"] = rpf.get("rpf_dv_ms")
    results["esm_prop_remaining_kg"] = rpf.get("esm_prop_remaining_kg")
    results["entry_fpa_deg"] = te.get("entry_fpa_deg")
    results["mission_duration_d"] = entry.get("t_splash", t) / 86400.0

    # Mission success: SLS delivered Orion AND Orion returned & splashed intact.
    results["full_success"] = True
    results["mission_failure"] = None
    return results, trajectories


# Phase-boundary functions whose OUTPUT state is a dashboard OEM-fidelity checkpoint,
# in flight order (fn name -> stage label). Kept here so the capture + the dashboard agree.
_OEM_BOUNDARY_STAGES = (
    ("phase_icps_tli", "post-TLI"), ("phase_outbound_coast", "outbound"),
    ("phase_outbound_powered_flyby", "OPF"), ("phase_dro_insertion", "DRI"),
    ("phase_dro_coast", "DRO/DDP"), ("phase_dro_departure", "post-DDP"),
    ("phase_return_coast", "return coast"), ("phase_return_powered_flyby", "RPF"),
    ("phase_transearth_coast", "EI"),
)


# Reference markers of the definitive nominal (field, value, tolerance, unit), used by
# check_nominal to detect a wrong-branch re-derivation. Tolerances are wide enough to
# pass any nominal on the intended trajectory branch and narrow enough to catch a branch flip.
_NOMINAL_REF = [
    ("rpf_dv_ms", 352.95, 8.0, "m/s"), ("ddp_dv_ms", 151.45, 8.0, "m/s"),
    ("dri_dv_ms", 93.79, 5.0, "m/s"), ("tli_dv_ms", 2837.44, 15.0, "m/s"),
    ("opf_dv_ms", 186.63, 15.0, "m/s"), ("entry_velocity_ms", 10987.7, 100.0, "m/s"),
    ("entry_fpa_deg", -5.952, 0.30, "deg"), ("ei_lat", -26.22, 2.0, "deg"),
    ("ei_lon", -120.10, 2.0, "deg"), ("splash_lat", 32.318, 2.0, "deg"),
    ("splash_lon", -118.181, 2.0, "deg"), ("mission_duration_d", 25.457, 0.05, "d"),
    ("max_earth_distance_km", 431418.0, 2000.0, "km"), ("lunar_closest_alt_km", 146.08, 20.0, "km"),
]


def check_nominal(results, targets):
    """Physical-plausibility gate on the nominal, run at setup BEFORE any shard burns.
    Catches a nominal that converged onto the wrong trajectory branch — e.g. a marginal
    accept-gate tipped by a different CPU/BLAS numerical environment — which would
    otherwise silently shift the entire fleet. Returns a list of human-readable problem
    strings; an empty list means the nominal is on the intended branch. Reference values
    are the definitive run; see _NOMINAL_REF. (AR1_SKIP_NOMINAL_CHECK bypasses the gate
    for deliberate off-default configurations.)"""
    P = []

    def _num(d, k):
        try:
            return float(d.get(k))
        except (TypeError, ValueError):
            return None

    # structural: the nominal must itself be a clean success on the expected ascent path
    if results.get("full_success") not in (True, "True", 1):
        P.append(f"full_success is not True (got {results.get('full_success')!r})")
    if results.get("launch_success") not in (True, "True", 1):
        P.append("launch_success is not True")
    if results.get("mission_failure") not in (None, "", "None"):
        P.append(f"mission_failure is set: {results.get('mission_failure')!r}")
    if results.get("launch_meco_mode") not in ("peg_lineartangent", "peg_2seg"):
        P.append(f"unexpected launch_meco_mode: {results.get('launch_meco_mode')!r}")

    # trajectory / return-branch markers (the wrong-branch tell lives here)
    for k, ref, tol, unit in _NOMINAL_REF:
        v = _num(results, k)
        if v is None:
            P.append(f"{k}: missing/non-numeric")
        elif abs(v - ref) > tol:
            P.append(f"{k}={v:.5g} off reference {ref:g} by >{tol:g} {unit}")

    # targets: return-branch consistency + (when the OD filter is on) chol-factor presence
    if targets:
        edr = _num(targets, "entry_downrange_km")
        if edr is not None and abs(edr - 6519.7) > 150.0:
            P.append(f"entry_downrange_km={edr:.5g} off reference 6519.7 by >150 km")
        rrh, rdv = _num(targets, "return_rpf_hint"), _num(results, "rpf_dv_ms")
        if rrh is not None and rdv is not None and abs(rrh - rdv) > 1.0:
            P.append(f"return_rpf_hint {rrh:.5g} inconsistent with rpf_dv_ms {rdv:.5g} (>1 m/s)")
        if globals().get("ENABLE_OD_FILTER", False):
            for k in ("od_L_otc0", "od_L_otc1", "od_L_otc2", "od_L_otc3",
                      "od_L_otc4", "od_L_otc5", "od_L_dri"):
                v = targets.get(k)
                # accept either the in-memory numpy factor or the JSON round-trip (nested lists)
                try:
                    arr = np.asarray(v, dtype=float)
                    ok6 = (arr.shape == (6, 6) and bool(np.all(np.isfinite(arr))))
                except (TypeError, ValueError):
                    ok6 = False
                if not ok6:
                    P.append(f"{k}: missing or not a finite 6x6 factor")
    return P


def run_nominal_with_boundaries():
    """Run the nominal mission ONCE (capture_trajectories=True) while recording each
    phase-boundary (label, [x,y,z,vx,vy,vz], t_end) — the checkpoints the dashboard's
    OEM fidelity meter compares to the as-flown ephemeris. Returns (results, trajectories)
    with trajectories["_boundaries"] added, so a cluster run can persist the boundaries in
    nominal_traj.npz and the dashboard reads them WITHOUT re-running the nominal locally
    (and gets the run's ACTUAL nominal, not a scipy/numpy-divergent local re-run).
    Hooks are installed on the module globals (run_mission resolves phase calls at call
    time) and always removed in finally.

    SELF-HEAL: after deriving, the nominal is validated with check_nominal(); if it landed
    off the intended branch (a wrong-branch convergence on this hardware), it is re-derived
    ONCE with the Phase-C branch forced (AR1_FORCE_PHASEC) so the machine reaches a valid
    nominal NATIVELY rather than being pinned to foreign numbers. A still-implausible result
    raises (blocking the run). AR1_SKIP_NOMINAL_CHECK bypasses the gate."""

    def _derive_once():
        boundaries = []
        originals = {}
        label_of = dict(_OEM_BOUNDARY_STAGES)

        def _mk(fnname, orig):
            def _hook(state, t0, perturb=None, *a, **k):
                res = orig(state, t0, perturb, *a, **k)
                try:
                    if res.get("success", True) and res.get("state") is not None:
                        st = np.asarray(res["state"], float)
                        boundaries.append([label_of[fnname],
                                           [float(x) for x in st[:6]],
                                           float(res.get("t_end", t0))])
                except Exception:
                    pass
                return res
            return _hook

        global _OD_CAP
        try:
            for fnname, _ in _OEM_BOUNDARY_STAGES:
                orig = globals().get(fnname)
                if orig is not None:
                    originals[fnname] = orig
                    globals()[fnname] = _mk(fnname, orig)
            if globals().get("ENABLE_OD_FILTER", False):
                _OD_CAP = {}                  # arm epoch capture for the OD-filter covariance build
            res, traj = run_mission(perturb=None, capture_trajectories=True)
            if globals().get("ENABLE_OD_FILTER", False):
                nt = globals().get("_NOMINAL_TARGETS")
                if isinstance(nt, dict):
                    _build_od_filter_covariances(nt)   # emergent per-epoch chol factors -> pinned in nt
        finally:
            _OD_CAP = None
            for fnname, orig in originals.items():
                globals()[fnname] = orig
        traj["_boundaries"] = boundaries
        return res, traj

    res, traj = _derive_once()
    if os.environ.get("AR1_SKIP_NOMINAL_CHECK", "0") == "1":
        return res, traj
    problems = check_nominal(res, globals().get("_NOMINAL_TARGETS") or {})
    if problems and os.environ.get("AR1_FORCE_PHASEC", "0") != "1":
        print(f"  check_nominal: nominal landed OFF the intended branch on this hardware "
              f"({len(problems)} issue(s); first: {problems[0]}).")
        print("  Re-deriving natively with AR1_FORCE_PHASEC=1 (forces the Phase-C return "
              "branch, keeps the hard physics guards)...")
        os.environ["AR1_FORCE_PHASEC"] = "1"
        try:
            res, traj = _derive_once()
            problems = check_nominal(res, globals().get("_NOMINAL_TARGETS") or {})
        finally:
            os.environ.pop("AR1_FORCE_PHASEC", None)
        if problems:
            raise RuntimeError(
                "check_nominal FAILED even with the Phase-C branch forced — the nominal is "
                "implausible on this hardware:\n    " + "\n    ".join(problems) +
                "\n  Last resort: pin a validated nominal into the run directory (see README, "
                "Reproducibility) or set AR1_SKIP_NOMINAL_CHECK=1 for a deliberate off-default run.")
        print("  check_nominal: recovered a valid nominal with the Phase-C branch forced.")
    elif problems:
        raise RuntimeError(
            "check_nominal FAILED (AR1_FORCE_PHASEC already set):\n    " + "\n    ".join(problems))
    return res, traj


# ============================================================
# Monte Carlo perturbation
# ============================================================
def sample_perturbation(rng):
    """Generate one trial's perturbation set (SLS/ICPS/ESM engine dispersions,
    targeting/execution errors, failure-mode draws).

    Draws in a FIXED ORDER and only APPENDS
    new fields, so trial i always maps to the same draws (the determinism the
    parallel/sharded drivers rely on). Magnitudes are estimate-grade (heritage / PRA /
    engineering foresight, tagged inline) — Artemis has no
    empirical dispersion data.
    """
    # --- SLS ascent engine-out draws (heritage-derived) --
    # Artemis has flown once -> no empirical rates. These derive from heritage:
    # SSME ~0.9996 reliability; RSRM post-Challenger record; RL10 upper-stage
    # history. All estimate-grade and refinable.
    p_rs25_failure = 0.0005    # per RS-25 (SSME ~0.9996)       [heritage]
    p_srb_failure  = 0.002     # per SRB (RSRM, post-redesign)  [heritage]
    p_icps_failure = 0.005     # RL10 ignition / TLI            [heritage]
    n_rs25_fail = int(rng.binomial(RS25_COUNT, p_rs25_failure))
    rs25_fail_time = float(rng.uniform(5, CORE_BURN_TIME_S)) if n_rs25_fail else 1e9
    n_srb_fail  = int(rng.binomial(SRB_COUNT, p_srb_failure))
    icps_ignition_fail = bool(rng.uniform(0, 1) < p_icps_failure)

    p = {
        # Ascent failures
        "n_rs25_failures":     n_rs25_fail,
        "rs25_failure_time_s": rs25_fail_time,
        "n_srb_failures":      n_srb_fail,
        "icps_ignition_fail":  icps_ignition_fail,
        # Engine Isp / thrust dispersions (1σ estimates ~0.3–0.5%)
        "srb_thrust_factor":   rng.normal(1, 0.005),
        "rs25_isp_factor":     rng.normal(1, 0.003),
        "rs25_thrust_factor":  rng.normal(1, 0.004),
        "icps_isp_factor":     rng.normal(1, 0.003),
        "icps_thrust_factor":  rng.normal(1, 0.004),
        "omse_isp_factor":     rng.normal(1, 0.003),  # ESM main engine
        "omse_thrust_factor":  rng.normal(1, 0.004),
        # TLI execution (pointing + cutoff velocity bias); estimate-grade magnitudes
        "tli_pointing_rad":    rng.normal(0, 3e-5, 3),
        "tli_dv_bias_ms":      rng.normal(0, 0.3),
        # Powered-flyby / DRO burn execution biases (m/s)
        "opf_dv_bias_ms":      rng.normal(0, 0.3),
        "dri_dv_bias_ms":      rng.normal(0, 0.3),
        "ddp_dv_bias_ms":      rng.normal(0, 0.3),
        "rpf_dv_bias_ms":      rng.normal(0, 0.3),
        # Entry aero dispersions
        "cd_factor":           rng.normal(1, 0.02),
        "ld_factor":           rng.normal(1, 0.04),
    }

    # TIME-EXPOSURE scaling for the time-driven hazards (ENABLE_EXPOSURE_SCALING; cross-mission
    # comparison convention). `_thr(rate)` = the re-anchored occurrence probability at this trial's exposure:
    # OFF -> the sourced per-mission rate (bit-identical); ON -> 1-(1-rate)**(T/T_ref). Artemis flies
    # ~all of its 25.45 d beyond the magnetosphere, so the shielding-weighted exposure ~= the full
    # duration and T_trial == T_ref -> _thr returns the rate exactly (outcome-neutral within Artemis).
    # The scaled modes each consume the SAME single rng.random() as before -> OFF bit-identical.
    _EXPO_ON = globals().get("ENABLE_EXPOSURE_SCALING", False)
    _T_REF_EXPO = MISSION_REF_DURATION_S            # nominal shielding-weighted exposure (~25.45 d)
    _T_TRIAL_EXPO = MISSION_REF_DURATION_S          # fixed profile -> per-trial weighted exposure == T_ref
    def _thr(rate):
        return _exposure_scaled_prob(rate, _T_TRIAL_EXPO, _T_REF_EXPO) if _EXPO_ON else rate

    # --- ESM systems catastrophic failure (Apollo-13-class analogue) --------
    if globals().get("ENABLE_ESM_SYSTEMS_FAILURES", False):
        # SOURCED ("ESM-systems catastrophic"): whole-ESM propulsion/power
        # loss (Apollo-13-class). Apollo SM 1/15 (6.7%, 1960s) × ~3.5x modern improvement (ESM is ATV-derived
        # [ATV 5/5]; Shuttle OMS-E ~0.998; 8 aux + 24 RCS + 4 solar-wing redundancy) -> ~1/50; cross-checked
        # vs the Orion PRA LOC ~1/30-1/75 cislunar. Estimate-grade (no flown Artemis failure data).
        PROB_ESM_CATASTROPHIC = 0.02   # time-driven -> exposure-scaled when ENABLE_EXPOSURE_SCALING
        p["esm_failure"] = bool(rng.random() < _thr(PROB_ESM_CATASTROPHIC))
        p["esm_failure_frac"] = float(rng.random())

    # --- Heat-shield / skip-entry thermal failure tail ----------------------
    if globals().get("ENABLE_HEATSHIELD_FAILURE", False):
        PROB_HEATSHIELD_LOSS = 0.005   # Artemis I saw unexpected char loss,
        p["heatshield_failed"] = bool(rng.random() < PROB_HEATSHIELD_LOSS)  # not a breach

    # Return-burn execution + entry-FPA dispersions (pragmatic return / entry).
    # Appended at the end per the RNG-stream rule. ESTIMATE magnitudes.
    p["rpf_dv_bias_ms"] = float(rng.normal(0.0, 1.0))
    p["entry_fpa_bias_deg"] = float(rng.normal(0.0, 0.1))   # guided-entry corridor

    # --- Extended SOURCED failure modes (phase-3 Apollo-parity) -------------
    # APPENDED at the very end and drawn UNCONDITIONALLY in fixed order (no
    # short-circuit) so the existing RNG stream is untouched and trial i stays
    # deterministic. Sourced rates (NASA / AAS references).
    if globals().get("ENABLE_OMS_E_BURN_FAILURES", False):
        P_OMS_E_BURN = 0.002   # per AJ10/OMS-E burn (~0.998 heritage)
        for b in ("opf", "dri", "ddp", "rpf"):
            p[f"oms_e_fail_{b}"] = bool(rng.random() < P_OMS_E_BURN)
    if globals().get("ENABLE_AVIONICS_FAILURES", False):
        P_AVIONICS = 0.01                 # radiation power-distribution event (Artemis I); time-driven
        P_AVIONICS_UNRECOVERABLE = 0.15   # most reset/recover (latching limiter)
        occ = rng.random() < _thr(P_AVIONICS)             # exposure-scale the radiation-event OCCURRENCE
        unrec = rng.random() < P_AVIONICS_UNRECOVERABLE   # always drawn (fixed stream); NOT time-driven
        p["avionics_anomaly"] = bool(occ and unrec)
        p["avionics_frac"] = float(rng.random())          # timeline fraction struck
    if globals().get("ENABLE_PARACHUTE_FAILURE", False):
        P_PARACHUTE = 0.003    # Orion 2-of-3 mains; char-loss-debris risk (OIG)
        p["parachute_failed"] = bool(rng.random() < P_PARACHUTE)
    if globals().get("ENABLE_SEP_FAILURE", False):
        P_CM_SM_SEP = 0.001    # CM/SM sep-bolt (Artemis I: 3/4 eroded but separated)
        p["cm_sm_sep_failed"] = bool(rng.random() < P_CM_SM_SEP)

    # --- FORESEEABLE cislunar failure modes (NO Artemis data) ---------------
    # Modes that haven't occurred on Artemis but are credible — sourced by
    # analogue / PRA / engineering foresight (all ESTIMATE-grade). Each is a
    # NET mission-ending per-mission probability, struck at a uniform timeline
    # fraction (a deliberate estimate-grade abstraction; phase-specific timing is
    # a refinement). Drawn UNCONDITIONALLY in fixed (insertion) order -> the
    # existing RNG stream and earlier draws are untouched (determinism preserved).
    if globals().get("ENABLE_FORESEEABLE_FAILURES", False):
        FORESEEABLE = {
            "mmod_strike":          0.0020,  # penetrating MMOD: BUMPER/ISS flux × Orion area × 25.5 d
            "nav_sensor_loss":      0.0030,  # star-tracker/IMU loss → nav (GN&C heritage)
            "esm_pressurization":   0.0020,  # ESM leak/pressurization loss (OIG foresight)
            "comm_loss_at_burn":    0.0015,  # DSN outage × critical-burn window (Artemis I 4.5 h)
            "rcs_failure":          0.0015,  # RCS attitude-control loss (heritage)
            "thermal_loss":         0.0015,  # thermal-loop failure over the long coast (heritage)
            "solar_particle_event": 0.0010,  # major SPE damaging avionics (solar-cycle + shielding)
            "dro_stationkeeping":   0.0005,  # DRO station-keeping propellant/control (minor)
        }
        # `comm_loss_at_burn` is per-EVENT (DSN outage × a critical-burn window), NOT time-driven —
        # it is the ONE foreseeable mode NOT exposure-scaled. The rest are time-driven.
        for name, rate in FORESEEABLE.items():
            thr = rate if name == "comm_loss_at_burn" else _thr(rate)
            p[f"ff_{name}"] = bool(rng.random() < thr)
            p[f"ff_{name}_frac"] = float(rng.random())

    # --- OD-nav DRI tracking residual (the ground-OD's finite state knowledge at insertion) ---
    # Drawn UNCONDITIONALLY (fixed stream) so toggling ENABLE_OD_NAV_RESIDUAL doesn't shift the lineage;
    # APPLIED only on the OD-nav override path in phase_dro_insertion. Magnitudes from the OD-nav consts.
    p["dri_od_pos_residual_m"]  = rng.normal(0.0, OD_NAV_POS_SIGMA_M, 3)
    p["dri_od_vel_residual_ms"] = rng.normal(0.0, OD_NAV_VEL_SIGMA_MS, 3)

    # --- EDL parachute-descent wind (drifts the splash point during the chute descent) ---
    p["edl_wind_ms"] = rng.normal(0.0, EDL_WIND_SIGMA_MS, 2)   # [East, North] horizontal wind, m/s

    # --- hypersonic-entry atmospheric DENSITY scale (drawn LAST -> rest of the trial bit-identical) ---
    p["entry_dens_scale"] = (float(np.clip(rng.normal(1.0, ENTRY_DENS_SIGMA), 0.78, 1.22))
                             if ENABLE_ENTRY_ATM_DISP else 1.0)

    # --- Closed-loop NAV state-estimate error on launch + TLI (RIC-frame UNIT draws; scaled by the
    # CLNAV_* σ and rotated to inertial at the guided cutoffs in run_mission, only when
    # ENABLE_CLOSED_LOOP_NAV). Drawn from a SPAWNED CHILD generator — spawn() derives a deterministic
    # per-trial child WITHOUT consuming the parent's bit stream (verified numpy 2.4.6), so every
    # EXISTING draw is untouched and a flag-OFF run is BIT-IDENTICAL to the pre-flag lineage (no lineage
    # shift, unlike appending to the shared stream). Deterministic + sharded≡serial. EST-grade. ---
    _nav_rng = rng.spawn(1)[0]
    p["asc_nav_unit"] = _nav_rng.standard_normal(6)   # ascent insertion: [pr,pi,pc, vr,vi,vc] unit
    p["tli_nav_unit"] = _nav_rng.standard_normal(6)   # TLI: v1 cutoff offset OR v2 pre-burn δx0 unit
    p["tli_nav_ba"]   = _nav_rng.standard_normal(3)   # v2 continuous: IMU accel-bias RIC unit [r,i,c]

    # --- OTC burn-EXECUTION error units (nav slice-1; consumed only when ENABLE_OTC_EXEC_ERRORS).
    # SECOND spawn: the parent bit stream AND the first (_nav_rng) child are both untouched -> every
    # existing lineage is bit-identical with the flag OFF. Row i = OTC-(i+1): (mag, tilt-1, tilt-2). ---
    _exec_rng = rng.spawn(1)[0]
    p["otc_exec_units"] = _exec_rng.standard_normal((6, 3))   # rows 0-3: OTC-1..4 (row-major prefix
    #   identical to the earlier (4,3) draw -> prior lineages bit-identical); rows 4-5: OTC-5/6 (the
    #   post-OPF conditional slots, phase-5)

    # --- OTC ground-OD KNOWLEDGE-error units (nav slice-2; consumed when ENABLE_OTC_OD_ERRORS).
    # THIRD spawn: parent + both prior children untouched. Row i = OTC-(i+1) slot:
    # [pos r,i,c; vel r,i,c] RIC units, scaled by the per-slot OTC_OD_*_SIGMA at the slot state. ---
    _od_rng = rng.spawn(1)[0]
    p["otc_od_units"] = _od_rng.standard_normal((6, 6))       # rows 0-3: OTC-1..4 (prefix-identical);
    #   rows 4-5: OTC-5/6 knowledge errors (post-OPF tracking arc)
    # ENTRY terminal-nav residual unit (consumed only when ENABLE_ENTRY_NAV_RESIDUAL). Drawn from the
    # EXISTING _od_rng AFTER otc_od_units — deliberately NOT a new spawn. A 4th rng.spawn() would advance
    # the GLOBAL spawn counter and RESAMPLE every trial's nav/exec/od child streams (verified: the parent
    # bit stream is untouched, but child #k -> #k+1 for all trials>0), silently shifting the fleet and
    # breaking flag-OFF bit-identity to the prior lineage. Drawing after otc_od_units leaves that draw
    # (and every other child) UNCHANGED, so flag-OFF stays bit-identical and the residual is a PURE
    # terminal overlay (same upstream trajectory; only the splash point shifts). [East, North] unit,
    # scaled by ENTRY_NAV_RESIDUAL_KM; the OD FILTER later scales it from the emergent EI covariance.
    p["entry_nav_resid_unit"] = _od_rng.standard_normal(2)
    # DRD re-optimization dispersion unit (consumed only when ENABLE_DRD_REOPT_DISPERSION). Drawn from the
    # SAME existing _od_rng AFTER entry_nav_resid_unit (again NOT a new spawn — same lineage-safety reason)
    # -> flag-OFF bit-identical. Scaled by DRD_REOPT_SIGMA_MS, applied to the DDP magnitude (the re-opt the
    # baked nominal embeds; put in the dispersion per the as-planned doctrine).
    p["drd_reopt_unit"] = float(_od_rng.standard_normal())

    return p


# ============================================================
# Per-trial debug output
# ============================================================
def _json_safe(v):
    """Recursively convert numpy types to JSON-serializable Python types."""
    if isinstance(v, dict):
        return {k: _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, (np.floating, np.integer)):
        return float(v)
    return v


def save_trial_debug(outdir, trial_idx, results):
    """Write <outdir>/trials/trial_<idx>.json (phase timing + all outcomes)."""
    tdir = os.path.join(outdir, "trials")
    os.makedirs(tdir, exist_ok=True)
    rec = {k: _json_safe(v) for k, v in results.items() if k != "_phase_log"}
    rec["trial"] = trial_idx if isinstance(trial_idx, str) else int(trial_idx)
    rec["phase_timeline"] = build_phase_timeline(results.get("_phase_log"))
    with open(os.path.join(tdir, f"trial_{trial_idx}.json"), "w") as f:
        json.dump(rec, f, indent=2, default=str)


# ============================================================
# Monte Carlo drivers
# ============================================================
# The nominal-target persistence is a single nominal_targets.json. Capture whatever the
# Artemis targeting solvers need from the nominal run into _NOMINAL_TARGETS;
# main()/main_parallel() persist it and inject it into workers. The targeting
# phases above should read globals()["_NOMINAL_TARGETS"].
_NOMINAL_TARGETS = None

# Per-trial debug record (reset each run_mission; saved as the trial JSON's "_debug"). _dbg() writes
# a diagnostic key/value; resolved at call-time so the phase functions above can use it freely.
_TRIAL_DBG = {}
def _dbg(key, value):
    d = globals().get("_TRIAL_DBG")
    if isinstance(d, dict):
        d[key] = value

def _dbg_phase(name, res):
    """Record a phase's scalar result fields into the per-trial _debug under `name` (skips the bulky
    state vector / trajectory arrays). Called for EVERY phase — incl. failed ones, so a trial's
    _debug shows each stage's internals + where/why it failed, with no flag-flip or re-run."""
    if isinstance(res, dict):
        _dbg(name, {k: _json_safe(v) for k, v in res.items()
                    if k not in ("state", "trajectory_t", "trajectory_y", "_phase_log")
                    and not isinstance(v, np.ndarray)})


def main(n=1000, outdir="outputs/artemis1_dev", seed=37, resume=True):
    """Serial Monte Carlo driver. Resumable, checkpoints results.csv every trial,
    captures + persists the nominal targets."""
    os.makedirs(outdir, exist_ok=True)
    csv_path     = os.path.join(outdir, "results.csv")
    targets_path = os.path.join(outdir, "nominal_targets.json")
    json_path    = os.path.join(outdir, "nominal_results.json")
    npz_path     = os.path.join(outdir, "nominal_traj.npz")

    # Load persisted nominal targets on resume.
    globals()["_NOMINAL_TARGETS"] = None
    if os.path.exists(targets_path):
        try:
            with open(targets_path) as f:
                globals()["_NOMINAL_TARGETS"] = json.load(f)
        except Exception:
            pass

    # Resume: load existing results.
    existing_results = []
    start_trial = 0
    if resume and os.path.exists(csv_path):
        try:
            prev = pd.read_csv(csv_path)
            existing_results = prev.to_dict("records")
            start_trial = (int(prev["trial"].max()) + 1
                           if "trial" in prev.columns else len(prev))
            print(f"  Resuming from trial {start_trial} ({len(existing_results)} existing)")
        except Exception:
            pass

    print(f"Running {n} physics-integrated Artemis I missions (start={start_trial})...")
    t_start = time.time()
    rng = np.random.default_rng(seed)
    for _ in range(start_trial):      # advance RNG past completed trials
        sample_perturbation(rng)

    results_list = list(existing_results)
    nominal_traj = nominal_results = None

    # Nominal trajectory (full capture) + target capture.
    if not os.path.exists(json_path):
        print("  Nominal trajectory (full capture)...")
        nominal_results, nominal_traj = run_nominal_with_boundaries()
        print(f"  Nominal full_success: {nominal_results.get('full_success')}")
        save_trial_debug(outdir, "nominal", nominal_results)
        traj_save = {}
        for k, v in nominal_traj.items():
            if isinstance(v, tuple) and len(v) == 2:
                traj_save[k + "_t"] = np.asarray(v[0])
                traj_save[k + "_y"] = np.asarray(v[1])
        _bnd = nominal_traj.get("_boundaries") or []      # OEM-meter phase-boundary checkpoints
        if _bnd:
            traj_save["_boundary_labels"] = np.array([b[0] for b in _bnd])
            traj_save["_boundary_states"] = np.array([b[1] for b in _bnd], dtype=float)
            traj_save["_boundary_t"] = np.array([b[2] for b in _bnd], dtype=float)
        if traj_save:
            np.savez_compressed(npz_path, **traj_save)
        with open(json_path, "w") as f:
            json.dump({k: _json_safe(v) for k, v in nominal_results.items()
                       if k != "_phase_log"}, f, indent=2, default=str)

    # Persist captured nominal targets so resume batches reuse them.
    if globals().get("_NOMINAL_TARGETS") is not None and not os.path.exists(targets_path):
        try:
            with open(targets_path, "w") as f:
                json.dump(_json_safe(globals()["_NOMINAL_TARGETS"]), f)
        except Exception:
            pass

    # MC loop.
    t0_mc = time.time()
    for i in range(start_trial, n):
        if (i - start_trial) % 10 == 0 or i == n - 1:
            done = i - start_trial + 1
            rate = done / max(0.1, time.time() - t0_mc)
            print(f"  Trial {i+1}/{n}  ({rate:.2f}/s, "
                  f"ETA {(n-i-1)/max(0.01, rate):.0f}s)", flush=True)
        t_trial = time.time()
        try:
            perturb = sample_perturbation(rng)
            r, _ = run_mission(perturb=perturb, capture_trajectories=False)
            r["trial"] = i
            r["trial_time_s"] = time.time() - t_trial
            save_trial_debug(outdir, i, r)
            r.pop("_phase_log", None)
            results_list.append(r)
        except Exception as e:
            results_list.append({"trial": i, "error": str(e),
                                 "trial_time_s": time.time() - t_trial})
        pd.DataFrame(results_list).to_csv(csv_path, index=False)

    print(f"Done. {n - start_trial} new trials in {time.time() - t_start:.1f}s")
    df = pd.DataFrame(results_list)
    df.to_csv(csv_path, index=False)
    return df, nominal_traj, nominal_results


# ---------------------------------------------------------------------------
# Parallel Monte Carlo
# ---------------------------------------------------------------------------
def _parallel_worker_init(nominal_targets):
    """Pool initializer: inject the captured nominal targets into each worker."""
    globals()["_NOMINAL_TARGETS"] = nominal_targets


def _parallel_run_trial(args):
    """Top-level worker (picklable — no closures)."""
    trial_idx, perturb = args
    t_trial = time.time()
    try:
        r, _ = run_mission(perturb=perturb, capture_trajectories=False)
        r["trial"] = trial_idx
        r["trial_time_s"] = time.time() - t_trial
        return r
    except Exception as e:
        return {"trial": trial_idx, "error": str(e),
                "trial_time_s": time.time() - t_trial}


def main_parallel(n=1000, outdir="outputs/artemis1_dev", seed=37,
                  resume=True, workers=None, indices=None):
    """Parallel Monte Carlo driver (multiprocessing.Pool). Deterministic
    (all n perturbations pre-generated in trial order; trial i is identical to
    serial), gap-safe resume, and `indices=` sharding hook for the cluster.
    """
    import multiprocessing as mp

    os.makedirs(outdir, exist_ok=True)
    csv_path     = os.path.join(outdir, "results.csv")
    targets_path = os.path.join(outdir, "nominal_targets.json")
    json_path    = os.path.join(outdir, "nominal_results.json")
    npz_path     = os.path.join(outdir, "nominal_traj.npz")

    globals()["_NOMINAL_TARGETS"] = None
    if os.path.exists(targets_path):
        try:
            with open(targets_path) as f:
                globals()["_NOMINAL_TARGETS"] = json.load(f)
        except Exception:
            pass

    # Resume: track the SET of completed trials (imap_unordered leaves holes).
    existing_results = []
    completed_trials = set()
    if resume and os.path.exists(csv_path):
        try:
            prev = pd.read_csv(csv_path)
            existing_results = prev.to_dict("records")
            if "trial" in prev.columns:
                completed_trials = {int(t) for t in prev["trial"].dropna()}
            print(f"  Resuming: {len(completed_trials)} trial(s) already complete")
        except Exception:
            pass

    def _checkpoint(rl):
        _df = pd.DataFrame(rl)
        if "trial" in _df.columns:
            _df = _df.sort_values("trial").reset_index(drop=True)
        _df.to_csv(csv_path, index=False)
        return _df

    print(f"Running {n} physics-integrated Artemis I missions "
          f"({len(completed_trials)} already done)...")
    t_start = time.time()

    # Nominal trajectory + target capture (serial).
    if not os.path.exists(json_path):
        print("  Nominal trajectory (full capture)...")
        nominal_results, nominal_traj = run_nominal_with_boundaries()
        print(f"  Nominal full_success: {nominal_results.get('full_success')}")
        save_trial_debug(outdir, "nominal", nominal_results)
        traj_save = {}
        for k, v in nominal_traj.items():
            if isinstance(v, tuple) and len(v) == 2:
                traj_save[k + "_t"] = np.asarray(v[0])
                traj_save[k + "_y"] = np.asarray(v[1])
        _bnd = nominal_traj.get("_boundaries") or []      # OEM-meter phase-boundary checkpoints
        if _bnd:
            traj_save["_boundary_labels"] = np.array([b[0] for b in _bnd])
            traj_save["_boundary_states"] = np.array([b[1] for b in _bnd], dtype=float)
            traj_save["_boundary_t"] = np.array([b[2] for b in _bnd], dtype=float)
        if traj_save:
            np.savez_compressed(npz_path, **traj_save)
        with open(json_path, "w") as f:
            json.dump({k: _json_safe(v) for k, v in nominal_results.items()
                       if k != "_phase_log"}, f, indent=2, default=str)

    if globals().get("_NOMINAL_TARGETS") is not None and not os.path.exists(targets_path):
        with open(targets_path, "w") as f:
            json.dump(_json_safe(globals()["_NOMINAL_TARGETS"]), f)

    # Pre-generate ALL n perturbations in trial order (determinism), then
    # dispatch only the missing trials (gap-safe + shard-able).
    rng = np.random.default_rng(seed)
    all_perturbs = [sample_perturbation(rng) for _ in range(n)]
    _wanted = set(range(n)) if indices is None else {int(i) for i in indices}
    perturbations = [(i, all_perturbs[i]) for i in range(n)
                     if i in _wanted and i not in completed_trials]

    if not perturbations:
        print("All requested trials already complete.")
        return _checkpoint(existing_results), None, None

    n_workers = workers if workers is not None else max(1, (os.cpu_count() or 2) - 1)
    print(f"  Dispatching {len(perturbations)} trials across {n_workers} workers...")

    results_list = list(existing_results)
    completed = 0
    t0_mc = time.time()
    with mp.Pool(processes=n_workers,
                 initializer=_parallel_worker_init,
                 initargs=(globals()["_NOMINAL_TARGETS"],)) as pool:
        for r in pool.imap_unordered(_parallel_run_trial, perturbations):
            if "trial" in r:
                save_trial_debug(outdir, r["trial"], r)
            r.pop("_phase_log", None)
            results_list.append(r)
            completed += 1
            if completed % 5 == 0 or completed == len(perturbations):
                rate = completed / max(0.1, time.time() - t0_mc)
                print(f"  {completed}/{len(perturbations)} trials done  "
                      f"({rate:.2f}/s, ETA {(len(perturbations)-completed)/max(0.01, rate):.0f}s)",
                      flush=True)
            _checkpoint(results_list)

    print(f"Done. {len(perturbations)} new trials in {time.time() - t_start:.1f}s")
    return _checkpoint(results_list), None, None


if __name__ == "__main__":
    # Smoke test: two trials, exercising the harness + run_mission end-to-end.
    main(n=2)
