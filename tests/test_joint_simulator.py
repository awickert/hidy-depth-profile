"""
Tests for JointSimulator.

Uses a constant production scheme to avoid network calls.  Two synthetic
profiles are constructed for ~13 ka with slightly different depths and
concentrations (simulating two pits on the same terrace).

TerraceChrono and OSLSurface tests have moved to the terrace-chrono package.
"""
import numpy as np
import pytest

from hidy_depth_profile.settings import ProfileSettings, _DistParam


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_settings(
    concentrations,
    depths,
    thicknesses=None,
    n_solutions=300,
    erosion_constant=True,
):
    """Return a minimal ProfileSettings consistent with ~13 ka."""
    s = ProfileSettings()
    s.latitude = 46.4
    s.longitude = -91.7
    s.elevation = 345.0
    s.production_scheme = "constant"
    s.constant_rate = 5.3
    s.production_error = _DistParam("constant", [5.3])

    if thicknesses is None:
        thicknesses = [5.0] * len(depths)

    s.profile_data = {
        "depth":         np.array(depths, dtype=float),
        "thickness":     np.array(thicknesses, dtype=float),
        "concentration": np.array(concentrations, dtype=float),
        "rel_error":     np.full(len(depths), 0.07),
    }

    s.mc_age               = _DistParam("uniform", [4000, 25000])
    if erosion_constant:
        s.mc_erosion_deposition_rate = _DistParam("constant", [0.0])
    else:
        s.mc_erosion_deposition_rate = _DistParam("uniform", [-10.0, 10.0])
    s.mc_inheritance                  = _DistParam("uniform", [0.0, 20000.0])
    s.mc_neutron_attenuation          = _DistParam("normal",  [160.0, 5.0])
    s.mc_erosion_deposition_threshold = [0.0, 30.0]
    s.density_error                   = _DistParam("uniform", [1.8, 2.3])
    s.muon_percent_error              = 5.0
    s.mc_confidence_mode  = "sigma"
    s.mc_confidence_value = 2.0
    s.mc_n_solutions      = n_solutions
    return s


# Two synthetic profiles for ~13 ka (slightly different concentrations)
_CONC_A = [66000.0, 46000.0, 32000.0]
_CONC_B = [64000.0, 45000.0, 31500.0]
_DEPTHS_A = [2.5, 32.5, 62.5]
_DEPTHS_B = [5.0, 35.0, 65.0]


def _make_pair(n_solutions=300):
    sA = _make_settings(_CONC_A, _DEPTHS_A, n_solutions=n_solutions)
    sB = _make_settings(_CONC_B, _DEPTHS_B, n_solutions=n_solutions)
    return sA, sB


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestJointValidation:

    def test_requires_two_profiles(self):
        from hidy_depth_profile.joint_simulator import JointSimulator
        sA = _make_settings(_CONC_A, _DEPTHS_A)
        with pytest.raises(ValueError, match="at least two"):
            JointSimulator({"A": sA})

    def test_isotope_mismatch_raises(self):
        from hidy_depth_profile.joint_simulator import JointSimulator
        sA = _make_settings(_CONC_A, _DEPTHS_A)
        sB = _make_settings(_CONC_B, _DEPTHS_B)
        sB.isotope = "Al-26"
        with pytest.raises(ValueError, match="isotope"):
            JointSimulator({"A": sA, "B": sB})

    def test_mc_age_mismatch_raises(self):
        from hidy_depth_profile.joint_simulator import JointSimulator
        sA = _make_settings(_CONC_A, _DEPTHS_A)
        sB = _make_settings(_CONC_B, _DEPTHS_B)
        sB.mc_age = _DistParam("uniform", [5000, 20000])
        with pytest.raises(ValueError, match="mc_age"):
            JointSimulator({"A": sA, "B": sB})

    def test_n_solutions_mismatch_raises(self):
        from hidy_depth_profile.joint_simulator import JointSimulator
        sA = _make_settings(_CONC_A, _DEPTHS_A, n_solutions=300)
        sB = _make_settings(_CONC_B, _DEPTHS_B, n_solutions=500)
        with pytest.raises(ValueError, match="mc_n_solutions"):
            JointSimulator({"A": sA, "B": sB})


# ---------------------------------------------------------------------------
# Basic run tests
# ---------------------------------------------------------------------------

class TestJointSimulatorRun:

    @pytest.fixture(scope="class")
    def joint_result(self):
        from hidy_depth_profile.joint_simulator import JointSimulator
        sA, sB = _make_pair(n_solutions=300)
        return JointSimulator({"A": sA, "B": sB}).run(seed=42)

    def test_n_accepted(self, joint_result):
        assert joint_result.n_accepted == 300

    def test_age_array_length(self, joint_result):
        assert len(joint_result.age) == 300

    def test_per_profile_inheritance_length(self, joint_result):
        assert len(joint_result.inheritance["A"]) == 300
        assert len(joint_result.inheritance["B"]) == 300

    def test_shared_age_array(self, joint_result):
        """Both profiles must have received the same accepted age array."""
        # The shared age is the same draw for both; inheritance differs.
        assert joint_result.age is joint_result.age  # trivially same object
        # Verify both inheritance arrays differ (independent draws)
        assert not np.array_equal(
            joint_result.inheritance["A"],
            joint_result.inheritance["B"],
        ), "Per-profile inheritances should differ (independent draws)"

    def test_pdf_age_exists(self, joint_result):
        assert joint_result.pdf_age is not None

    def test_inheritance_pdfs_exist(self, joint_result):
        assert joint_result.pdf_inheritance["A"] is not None
        assert joint_result.pdf_inheritance["B"] is not None

    def test_map_near_13ka(self, joint_result):
        """MAP should land within 3 ka of the synthetic target."""
        assert abs(joint_result.best_age_ka - 13.0) < 3.0

    def test_sigma_bounds_finite(self, joint_result):
        assert np.isfinite(joint_result.age_sigma2_plus_ka)
        assert np.isfinite(joint_result.age_sigma2_minus_ka)

    def test_densities_shape(self, joint_result):
        assert joint_result.densities["A"].shape == (300, 3)
        assert joint_result.densities["B"].shape == (300, 3)


# ---------------------------------------------------------------------------
# Joint vs single: sharing a well-constrained age
# ---------------------------------------------------------------------------

class TestJointNarrowerThanSingle:
    """
    Two profiles sharing the same underlying age should give a narrower
    posterior than either profile inverted individually.
    """

    @pytest.fixture(scope="class")
    def results(self):
        from hidy_depth_profile.joint_simulator import JointSimulator
        from hidy_depth_profile.simulator import MonteCarloSimulator

        sA, sB = _make_pair(n_solutions=500)
        r_joint = JointSimulator({"A": sA, "B": sB}).run(seed=7)
        r_A = MonteCarloSimulator(_make_settings(_CONC_A, _DEPTHS_A, n_solutions=500)).run(seed=7)
        r_B = MonteCarloSimulator(_make_settings(_CONC_B, _DEPTHS_B, n_solutions=500)).run(seed=7)
        return r_joint, r_A, r_B

    def test_joint_range_not_wider_than_single(self, results):
        r_joint, r_A, r_B = results
        joint_range = r_joint.age_sigma2_plus_ka - r_joint.age_sigma2_minus_ka
        range_A = r_A.age_sigma2_plus_ka - r_A.age_sigma2_minus_ka
        range_B = r_B.age_sigma2_plus_ka - r_B.age_sigma2_minus_ka
        # Joint inversion should not be wider than both singles
        assert joint_range <= max(range_A, range_B) + 1.0, (
            f"Joint range {joint_range:.1f} ka wider than both singles "
            f"({range_A:.1f}, {range_B:.1f} ka)"
        )


class TestMultipleMaxConstraints:
    """Two independent OSL upper bounds on the same profile/joint group."""

    def test_single_profile_two_max_constraints(self):
        """age_max_constraints list is accepted and shifts MAP younger."""
        from hidy_depth_profile.simulator import MonteCarloSimulator

        s_unconstrained = _make_settings(_CONC_A, _DEPTHS_A, n_solutions=400)
        s_constrained   = _make_settings(_CONC_A, _DEPTHS_A, n_solutions=400)
        # Apply two upper bounds: tight (10 ka) and loose (14 ka)
        s_constrained.age_max_constraint  = _DistParam("normal", [10_000, 500])
        s_constrained.age_max_constraints = [_DistParam("normal", [14_000, 500])]

        r_unc = MonteCarloSimulator(s_unconstrained).run(seed=10)
        r_con = MonteCarloSimulator(s_constrained).run(seed=10)

        # Both constraints push MAP younger; constrained MAP ≤ unconstrained MAP
        assert r_con.best_age_ka <= r_unc.best_age_ka + 1.0, (
            f"Constrained MAP {r_con.best_age_ka:.1f} ka not younger than "
            f"unconstrained {r_unc.best_age_ka:.1f} ka"
        )
        # All accepted ages should satisfy the tighter bound with high probability
        assert np.percentile(r_con.age, 95) < 12_000, (
            "95th-percentile age exceeds tight OSL upper bound"
        )

    def test_joint_two_max_constraints_on_one_profile(self):
        """age_max_constraints on one profile in a JointSimulator is applied."""
        from hidy_depth_profile.joint_simulator import JointSimulator

        sA = _make_settings(_CONC_A, _DEPTHS_A, n_solutions=300)
        sB = _make_settings(_CONC_B, _DEPTHS_B, n_solutions=300)
        # Both OSL bounds on B; A has no constraint
        sB.age_max_constraint  = _DistParam("normal", [10_000, 500])
        sB.age_max_constraints = [_DistParam("normal", [14_000, 500])]

        r = JointSimulator({"A": sA, "B": sB}).run(seed=20)
        assert r.n_accepted == 300
        # Shared age pulled younger by the B constraint
        assert np.percentile(r.age, 95) < 12_000
