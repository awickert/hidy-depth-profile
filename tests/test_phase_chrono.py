"""
Tests for PhaseChrono and PhaseChronoResult.

Uses the same constant-production helper as test_joint_simulator.py.
Concentrations are chosen to produce two clearly separated age clusters
(~17 ka vs ~10 ka) so that phase assignment tests are unambiguous.
"""
import numpy as np
import pytest

from hidy_depth_profile.settings import ProfileSettings, _DistParam


# ---------------------------------------------------------------------------
# Shared helpers (same pattern as test_joint_simulator.py)
# ---------------------------------------------------------------------------

def _make_settings(concentrations, depths, n_solutions=200):
    s = ProfileSettings()
    s.latitude = 46.4
    s.longitude = -91.7
    s.elevation = 345.0
    s.production_scheme = "constant"
    s.constant_rate = 5.3
    s.production_error = _DistParam("constant", [5.3])
    s.profile_data = {
        "depth":         np.array(depths, dtype=float),
        "thickness":     np.full(len(depths), 5.0),
        "concentration": np.array(concentrations, dtype=float),
        "rel_error":     np.full(len(depths), 0.07),
    }
    s.mc_age                          = _DistParam("uniform", [4000, 25000])
    s.mc_erosion_deposition_rate      = _DistParam("constant", [0.0])
    s.mc_inheritance                  = _DistParam("uniform", [0.0, 20000.0])
    s.mc_neutron_attenuation          = _DistParam("normal",  [160.0, 5.0])
    s.mc_erosion_deposition_threshold = [0.0, 30.0]
    s.density_error                   = _DistParam("uniform", [1.8, 2.3])
    s.muon_percent_error              = 5.0
    s.mc_confidence_mode              = "sigma"
    s.mc_confidence_value             = 2.0
    s.mc_n_solutions                  = n_solutions
    return s


# Old surfaces (~17 ka): high concentrations
_CONC_OLD_A = [90_000.0, 63_000.0, 44_000.0]
_CONC_OLD_B = [88_000.0, 62_000.0, 43_500.0]
# Young surfaces (~10 ka): roughly half the concentrations
_CONC_YNG_A = [47_000.0, 33_000.0, 23_000.0]
_CONC_YNG_B = [45_000.0, 31_500.0, 22_000.0]
_DEPTHS_A = [2.5, 32.5, 62.5]
_DEPTHS_B = [5.0, 35.0, 65.0]


# ---------------------------------------------------------------------------
# _enumerate_valid_assignments
# ---------------------------------------------------------------------------

class TestEnumerateValidAssignments:

    def test_chain_K3_count(self):
        from hidy_depth_profile.phase_chrono import _enumerate_valid_assignments
        # 4-surface chain: 0>1>2>3 with K=3 → non-decreasing sequences of
        # length 4 from {0,1,2}: C(4+3-1, 3-1) = C(6,2) = 15
        va = _enumerate_valid_assignments(4, [(0, 1), (1, 2), (2, 3)], 3)
        assert va.shape == (15, 4)

    def test_all_assignments_non_decreasing(self):
        from hidy_depth_profile.phase_chrono import _enumerate_valid_assignments
        va = _enumerate_valid_assignments(4, [(0, 1), (1, 2), (2, 3)], 3)
        assert np.all(va[:, :-1] <= va[:, 1:])

    def test_no_ordering_gives_all(self):
        # With no ordering constraints every assignment is valid: K^N
        from hidy_depth_profile.phase_chrono import _enumerate_valid_assignments
        va = _enumerate_valid_assignments(3, [], 2)
        assert len(va) == 2 ** 3   # 8

    def test_K1_single_assignment(self):
        from hidy_depth_profile.phase_chrono import _enumerate_valid_assignments
        va = _enumerate_valid_assignments(4, [(0, 1), (2, 3)], 1)
        assert len(va) == 1
        assert np.all(va == 0)


# ---------------------------------------------------------------------------
# PhaseChrono construction
# ---------------------------------------------------------------------------

class TestPhaseChronoValidation:

    def test_unknown_surface_raises(self):
        from hidy_depth_profile.phase_chrono import PhaseChrono
        from hidy_depth_profile.terrace_chrono import OSLSurface
        s = OSLSurface(14_000, 500, seed=1)
        with pytest.raises(ValueError, match="unknown surface"):
            PhaseChrono(
                surfaces={"A": s},
                elevation_ordering=[("A", "TYPO")],
                n_phases=2,
            )

    def test_zero_phases_raises(self):
        from hidy_depth_profile.phase_chrono import PhaseChrono
        from hidy_depth_profile.terrace_chrono import OSLSurface
        s = OSLSurface(14_000, 500, seed=1)
        with pytest.raises(ValueError, match="n_phases"):
            PhaseChrono(surfaces={"A": s}, elevation_ordering=[], n_phases=0)

    def test_sigma_property(self):
        from hidy_depth_profile.phase_chrono import PhaseChrono
        from hidy_depth_profile.terrace_chrono import OSLSurface
        s = OSLSurface(14_000, 500, seed=1)
        pc = PhaseChrono(
            surfaces={"A": s},
            elevation_ordering=[],
            n_phases=2,
            within_phase_sigma_yr=500.0,
        )
        assert pc.within_phase_sigma_yr == 500.0

    def test_default_sigma(self):
        from hidy_depth_profile.phase_chrono import PhaseChrono, _DEFAULT_SIGMA_INTRA
        from hidy_depth_profile.terrace_chrono import OSLSurface
        s = OSLSurface(14_000, 500, seed=1)
        pc = PhaseChrono(surfaces={"A": s}, elevation_ordering=[], n_phases=2)
        assert pc.within_phase_sigma_yr == _DEFAULT_SIGMA_INTRA


# ---------------------------------------------------------------------------
# PhaseChrono.constrain — basic correctness
# ---------------------------------------------------------------------------

class TestPhaseChronoRun:

    @pytest.fixture(scope="class")
    def two_surface_result(self):
        from hidy_depth_profile.phase_chrono import PhaseChrono
        from hidy_depth_profile.simulator import MonteCarloSimulator
        s_old = _make_settings(_CONC_OLD_A, _DEPTHS_A, n_solutions=300)
        s_yng = _make_settings(_CONC_YNG_A, _DEPTHS_A, n_solutions=300)
        r_old = MonteCarloSimulator(s_old).run(seed=1)
        r_yng = MonteCarloSimulator(s_yng).run(seed=2)
        pc = PhaseChrono(
            surfaces={"OLD": r_old, "YNG": r_yng},
            elevation_ordering=[("OLD", "YNG")],
            n_phases=2,
        )
        return pc.constrain(n_draws=500, seed=3)

    def test_n_accepted(self, two_surface_result):
        assert two_surface_result.n_accepted == 500

    def test_ages_shape(self, two_surface_result):
        for name in ("OLD", "YNG"):
            assert two_surface_result.ages[name].shape == (500,)

    def test_phase_ages_shape(self, two_surface_result):
        assert two_surface_result.phase_ages.shape == (500, 2)

    def test_phase_ages_ordered(self, two_surface_result):
        # phase 0 must always be >= phase 1 (phase 0 = oldest)
        assert np.all(
            two_surface_result.phase_ages[:, 0]
            >= two_surface_result.phase_ages[:, 1]
        )

    def test_assignment_probs_sum_to_one(self, two_surface_result):
        for name in ("OLD", "YNG"):
            total = two_surface_result.assignment_probs[name].sum()
            assert abs(total - 1.0) < 1e-10

    def test_n_phases_occupied_range(self, two_surface_result):
        occ = two_surface_result.n_phases_occupied
        assert np.all(occ >= 1)
        assert np.all(occ <= 2)

    def test_n_attempted_ge_n_accepted(self, two_surface_result):
        assert two_surface_result.n_attempted >= two_surface_result.n_accepted


# ---------------------------------------------------------------------------
# Phase assignment correctness
# ---------------------------------------------------------------------------

class TestPhaseAssignment:

    @pytest.fixture(scope="class")
    def four_surface_result(self):
        """Two old + two young surfaces; expect old→phase 0, young→phase 1."""
        from hidy_depth_profile.phase_chrono import PhaseChrono
        from hidy_depth_profile.simulator import MonteCarloSimulator
        r_o1 = MonteCarloSimulator(_make_settings(_CONC_OLD_A, _DEPTHS_A, 300)).run(seed=1)
        r_o2 = MonteCarloSimulator(_make_settings(_CONC_OLD_B, _DEPTHS_B, 300)).run(seed=2)
        r_y1 = MonteCarloSimulator(_make_settings(_CONC_YNG_A, _DEPTHS_A, 300)).run(seed=3)
        r_y2 = MonteCarloSimulator(_make_settings(_CONC_YNG_B, _DEPTHS_B, 300)).run(seed=4)
        pc = PhaseChrono(
            surfaces={"O1": r_o1, "O2": r_o2, "Y1": r_y1, "Y2": r_y2},
            elevation_ordering=[
                ("O1", "Y1"), ("O1", "Y2"),
                ("O2", "Y1"), ("O2", "Y2"),
            ],
            n_phases=2,
        )
        return pc.constrain(n_draws=500, seed=5)

    def test_old_surfaces_prefer_phase_0(self, four_surface_result):
        for name in ("O1", "O2"):
            assert four_surface_result.assignment_probs[name][0] > 0.7

    def test_young_surfaces_prefer_phase_1(self, four_surface_result):
        for name in ("Y1", "Y2"):
            assert four_surface_result.assignment_probs[name][1] > 0.7

    def test_old_phase_age_older_than_young(self, four_surface_result):
        mean_phase0 = four_surface_result.phase_ages[:, 0].mean()
        mean_phase1 = four_surface_result.phase_ages[:, 1].mean()
        assert mean_phase0 > mean_phase1


# ---------------------------------------------------------------------------
# Empty phase allowed
# ---------------------------------------------------------------------------

class TestEmptyPhase:

    def test_extra_phase_often_empty(self):
        """K=3 with only 2 clearly separated clusters → third phase mostly empty."""
        from hidy_depth_profile.phase_chrono import PhaseChrono
        from hidy_depth_profile.simulator import MonteCarloSimulator
        r_old = MonteCarloSimulator(_make_settings(_CONC_OLD_A, _DEPTHS_A, 200)).run(seed=1)
        r_yng = MonteCarloSimulator(_make_settings(_CONC_YNG_A, _DEPTHS_A, 200)).run(seed=2)
        pc = PhaseChrono(
            surfaces={"OLD": r_old, "YNG": r_yng},
            elevation_ordering=[("OLD", "YNG")],
            n_phases=3,
        )
        result = pc.constrain(n_draws=300, seed=6)
        # With only 2 surfaces, at most 2 phases can be occupied per draw
        assert np.all(result.n_phases_occupied <= 2)
        # And with well-separated clusters, both should usually be occupied
        assert np.mean(result.n_phases_occupied == 2) > 0.5


# ---------------------------------------------------------------------------
# OSLSurface compatibility
# ---------------------------------------------------------------------------

class TestOSLSurfaceInput:

    def test_osl_surface_as_input(self):
        from hidy_depth_profile.phase_chrono import PhaseChrono
        from hidy_depth_profile.terrace_chrono import OSLSurface
        from hidy_depth_profile.simulator import MonteCarloSimulator
        utf = OSLSurface(mean_yr=17_000, sigma_yr=900, seed=1)
        r = MonteCarloSimulator(
            _make_settings(_CONC_OLD_A, _DEPTHS_A, 200)
        ).run(seed=2)
        pc = PhaseChrono(
            surfaces={"UTF": utf, "LTF": r},
            elevation_ordering=[("UTF", "LTF")],
            n_phases=2,
        )
        result = pc.constrain(n_draws=200, seed=3)
        assert result.n_accepted == 200
        assert result.phase_ages.shape == (200, 2)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:

    def test_same_seed_same_output(self):
        from hidy_depth_profile.phase_chrono import PhaseChrono
        from hidy_depth_profile.terrace_chrono import OSLSurface
        s1 = OSLSurface(14_000, 800, seed=0)
        s2 = OSLSurface(10_000, 600, seed=0)
        make = lambda: PhaseChrono(
            surfaces={"A": OSLSurface(14_000, 800, seed=0),
                      "B": OSLSurface(10_000, 600, seed=0)},
            elevation_ordering=[("A", "B")],
            n_phases=2,
        ).constrain(n_draws=100, seed=7)
        r1 = make()
        r2 = make()
        np.testing.assert_array_equal(r1.phase_ages, r2.phase_ages)
        np.testing.assert_array_equal(r1.ages["A"], r2.ages["A"])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def test_importable_from_top_level():
    from hidy_depth_profile import PhaseChrono, PhaseChronoResult
    assert PhaseChrono is not None
    assert PhaseChronoResult is not None
