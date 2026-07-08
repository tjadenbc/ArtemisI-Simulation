"""Unit tests for the v1.0.1 nominal-guard layer: check_nominal() and the
run_nominal_with_boundaries self-heal (auto-force-retry). Fast — the self-heal
control flow is exercised with a mocked derive so no trajectory is integrated.

Run:  OMP_NUM_THREADS=1 python3 test_check_nominal.py
"""
import json, os, sys
import artemis1


def _good():
    """The definitive nominal (pass case) loaded from outputs/final."""
    nr = json.load(open("outputs/final/nominal_results.json"))
    nt = json.load(open("outputs/final/nominal_targets.json"))
    return nr, nt


def test_pass_on_definitive():
    nr, nt = _good()
    problems = artemis1.check_nominal(nr, nt)
    assert problems == [], f"definitive nominal should PASS, got: {problems}"
    print("PASS  check_nominal accepts the definitive nominal (0 problems)")


def test_fail_on_apollo_style_flip():
    """A gross wrong-branch flip: return burns + geometry + duration all shifted."""
    nr, nt = _good()
    nr["rpf_dv_ms"] = 946.0            # min-energy long-coast branch
    nr["mission_duration_d"] = 26.6    # EI epoch hours late
    nr["ei_lon"] = -95.0               # return geometry longitude shifted tens of deg
    problems = artemis1.check_nominal(nr, nt)
    flagged = " ".join(problems)
    assert any("rpf_dv_ms" in p for p in problems), problems
    assert any("mission_duration_d" in p for p in problems), problems
    assert any("ei_lon" in p for p in problems), problems
    print(f"PASS  check_nominal rejects a gross branch flip ({len(problems)} problems)")


def test_fail_on_subtle_return_flip_with_in_corridor_fpa():
    """The trap: entry-FPA still reads in-corridor, but the return branch moved.
    check_nominal must catch it via the return-branch tells, NOT rely on FPA."""
    nr, nt = _good()
    nr["entry_fpa_deg"] = -6.10        # still inside Orion's corridor -> looks fine
    nr["rpf_dv_ms"] = 300.0            # but the return burn is off-branch
    nt["return_rpf_hint"] = 300.0      # (kept self-consistent so only the branch value is the tell)
    problems = artemis1.check_nominal(nr, nt)
    assert any("rpf_dv_ms" in p for p in problems), problems
    assert not any("entry_fpa_deg" in p for p in problems), "FPA was in-corridor; should not flag it"
    print("PASS  check_nominal catches a return-branch flip even when entry-FPA looks nominal")


def test_fail_on_missing_od_chol():
    nr, nt = _good()
    nt = dict(nt); nt.pop("od_L_dri", None)
    if getattr(artemis1, "ENABLE_OD_FILTER", False):
        problems = artemis1.check_nominal(nr, nt)
        assert any("od_L_dri" in p for p in problems), problems
        print("PASS  check_nominal flags a missing OD Cholesky factor")
    else:
        print("SKIP  ENABLE_OD_FILTER off — od_L check not applicable")


def _mock_selfheal(check_sequence):
    """Run run_nominal_with_boundaries with run_mission + OD build mocked and
    check_nominal driven by `check_sequence` (list of return values per call).
    Returns (result, n_derives, forced_seen). Restores all globals."""
    saved = {k: getattr(artemis1, k) for k in
             ("run_mission", "_build_od_filter_covariances", "check_nominal", "_NOMINAL_TARGETS")}
    saved_env = os.environ.get("AR1_FORCE_PHASEC")
    calls = {"derive": 0}
    forced = {"seen": False}

    def fake_run_mission(perturb=None, capture_trajectories=False):
        calls["derive"] += 1
        if os.environ.get("AR1_FORCE_PHASEC") == "1":
            forced["seen"] = True
        return ({"full_success": True}, {})

    seq = list(check_sequence)
    def fake_check(res, targets):
        return seq.pop(0) if seq else []

    try:
        artemis1.run_mission = fake_run_mission
        artemis1._build_od_filter_covariances = lambda nt: None
        artemis1.check_nominal = fake_check
        artemis1._NOMINAL_TARGETS = {}
        res, traj = artemis1.run_nominal_with_boundaries()
        return res, calls["derive"], forced["seen"]
    finally:
        for k, v in saved.items():
            setattr(artemis1, k, v)
        if saved_env is None:
            os.environ.pop("AR1_FORCE_PHASEC", None)
        else:
            os.environ["AR1_FORCE_PHASEC"] = saved_env


def test_selfheal_clean_no_rederive():
    res, n_derives, forced = _mock_selfheal([[]])          # passes first time
    assert n_derives == 1, n_derives
    assert not forced
    assert "AR1_FORCE_PHASEC" not in os.environ or os.environ["AR1_FORCE_PHASEC"] != "1"
    print("PASS  self-heal: clean nominal -> single derive, no force, env clean")


def test_selfheal_recovers_with_force():
    res, n_derives, forced = _mock_selfheal([["bad-branch"], []])  # fail, then pass under force
    assert n_derives == 2, f"expected a forced re-derive, got {n_derives} derive(s)"
    assert forced, "second derive should have run with AR1_FORCE_PHASEC=1"
    assert os.environ.get("AR1_FORCE_PHASEC") != "1", "AR1_FORCE_PHASEC must be restored/unset after"
    print("PASS  self-heal: bad nominal -> re-derives with forced branch, recovers, env restored")


def test_selfheal_raises_if_still_bad():
    try:
        _mock_selfheal([["bad"], ["still-bad"]])           # fails even under force
    except RuntimeError as e:
        assert "implausible" in str(e)
        assert os.environ.get("AR1_FORCE_PHASEC") != "1"
        print("PASS  self-heal: force still bad -> RuntimeError raised, env restored")
        return
    raise AssertionError("expected RuntimeError when forced re-derive is still implausible")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"\nALL {len(tests)} TESTS PASSED")
