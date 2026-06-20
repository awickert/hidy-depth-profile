"""
Tests for geochronological age constraints (age_max_constraint, age_min_constraint).

Covers:
  - Settings validation (type checking, invalid modes, bad sigma)
  - YAML round-trip
  - Hard (constant) max constraint: no accepted solution older than the bound
  - Hard (constant) min constraint: no accepted solution younger than the bound
  - Soft (normal) max constraint: Bayesian MAP shifts toward younger ages
"""
import numpy as np
import pytest

from hidy_depth_profile.settings import ProfileSettings, _DistParam


# ---------------------------------------------------------------------------
# Shared synthetic profile
# ---------------------------------------------------------------------------

def _make_settings(n_solutions=300):
    """
    Minimal settings with a synthetic 3-sample profile consistent with ~13 ka.

    Uses a constant production scheme to avoid network calls.
    """
    s = ProfileSettings()
    s.latitude = 46.3
    s.longitude = -91.8
    s.elevation = 332.0
    s.production_scheme = "constant"
    s.constant_rate = 5.3
    s.production_error = _DistParam("constant", [5.3])

    # Concentrations computed for ~13 ka, ρ=2.0 g/cm³, Λ=160 g/cm²
    s.profile_data = {
        "depth":         np.array([2.5,     32.5,    62.5]),
        "thickness":     np.array([5.0,      5.0,     5.0]),
        "concentration": np.array([66000.0, 46000.0, 32000.0]),
        "rel_error":     np.array([0.07,     0.07,    0.07]),
    }

    s.mc_age                          = _DistParam("uniform", [4000, 25000])
    s.mc_erosion_deposition_rate      = _DistParam("constant", [0.0])
    s.mc_inheritance                  = _DistParam("uniform",  [0.0, 20000.0])
    s.mc_neutron_attenuation          = _DistParam("normal",   [160.0, 5.0])
    s.mc_erosion_deposition_threshold = [0.0, 30.0]
    s.density_error                   = _DistParam("uniform",  [1.8, 2.3])
    s.muon_percent_error              = 5.0
    s.mc_confidence_mode  = "sigma"
    s.mc_confidence_value = 2.0
    s.mc_n_solutions      = n_solutions
    return s


# ---------------------------------------------------------------------------
# Settings validation
# ---------------------------------------------------------------------------

class TestConstraintSettings:

    def test_max_constant_accepted(self):
        s = _make_settings()
        s.age_max_constraint = _DistParam("constant", [15000.0])
        assert s.age_max_constraint.parameters[0] == 15000.0

    def test_max_normal_accepted(self):
        s = _make_settings()
        s.age_max_constraint = _DistParam("normal", [14000.0, 800.0])
        assert s.age_max_constraint.mode == "normal"

    def test_min_constant_accepted(self):
        s = _make_settings()
        s.age_min_constraint = _DistParam("constant", [8000.0])
        assert s.age_min_constraint.parameters[0] == 8000.0

    def test_min_normal_accepted(self):
        s = _make_settings()
        s.age_min_constraint = _DistParam("normal", [9000.0, 500.0])
        assert s.age_min_constraint.mode == "normal"

    def test_set_none_clears(self):
        s = _make_settings()
        s.age_max_constraint = _DistParam("constant", [15000.0])
        s.age_max_constraint = None
        assert s.age_max_constraint is None

    def test_uniform_mode_rejected(self):
        s = _make_settings()
        with pytest.raises(ValueError, match="constant.*normal"):
            s.age_max_constraint = _DistParam("uniform", [10000.0, 15000.0])

    def test_non_distparam_rejected(self):
        s = _make_settings()
        with pytest.raises(TypeError):
            s.age_max_constraint = 15000.0

    def test_zero_sigma_rejected(self):
        s = _make_settings()
        with pytest.raises(ValueError, match="sigma"):
            s.age_max_constraint = _DistParam("normal", [14000.0, 0.0])

    def test_negative_sigma_rejected(self):
        s = _make_settings()
        with pytest.raises(ValueError, match="sigma"):
            s.age_min_constraint = _DistParam("normal", [9000.0, -100.0])

    def test_impossible_range_raises(self):
        """Clamping both bounds to an empty interval should fail at setup."""
        from hidy_depth_profile.simulator import MonteCarloSimulator
        s = _make_settings()
        s.age_max_constraint = _DistParam("constant", [5000.0])
        s.age_min_constraint = _DistParam("constant", [8000.0])
        with pytest.raises(ValueError, match="no valid draw range"):
            MonteCarloSimulator(s)


# ---------------------------------------------------------------------------
# YAML round-trip
# ---------------------------------------------------------------------------

class TestConstraintYaml:

    def test_roundtrip_both(self, tmp_path):
        from hidy_depth_profile.yaml_io import load_yaml, save_yaml
        s = _make_settings()
        s.age_max_constraint = _DistParam("normal",   [14000.0, 800.0])
        s.age_min_constraint = _DistParam("constant", [8000.0])
        path = str(tmp_path / "c.yaml")
        save_yaml(s, path)
        s2 = load_yaml(path)
        assert s2.age_max_constraint.mode == "normal"
        assert s2.age_max_constraint.parameters == [14000.0, 800.0]
        assert s2.age_min_constraint.mode == "constant"
        assert s2.age_min_constraint.parameters == [8000.0]

    def test_roundtrip_none(self, tmp_path):
        from hidy_depth_profile.yaml_io import load_yaml, save_yaml
        s = _make_settings()
        path = str(tmp_path / "c.yaml")
        save_yaml(s, path)
        s2 = load_yaml(path)
        assert s2.age_max_constraint is None
        assert s2.age_min_constraint is None

    def test_yaml_omits_constraints_block_when_none(self, tmp_path):
        import yaml
        from hidy_depth_profile.yaml_io import save_yaml
        s = _make_settings()
        path = str(tmp_path / "c.yaml")
        save_yaml(s, path)
        with open(path) as fh:
            doc = yaml.safe_load(fh)
        assert "constraints" not in doc


# ---------------------------------------------------------------------------
# Hard (constant) constraints
# ---------------------------------------------------------------------------

class TestHardMaxConstraint:
    """
    Hard upper bound: every accepted solution must be younger than the bound.
    The draw range is clamped at setup, so this also tests the efficiency path.
    """

    @pytest.fixture(scope="class")
    def results(self):
        from hidy_depth_profile.simulator import MonteCarloSimulator
        s = _make_settings(n_solutions=300)
        s.age_max_constraint = _DistParam("constant", [14000.0])
        sim = MonteCarloSimulator(s)
        return sim.run(seed=42)

    def test_no_solution_older_than_bound(self, results):
        assert np.all(results.age <= 14000.0), (
            f"Solutions older than 14 ka found: max = {results.age.max():.0f} yr"
        )

    def test_map_within_bound(self, results):
        assert results.best_age_ka <= 14.0

    def test_n_solutions_reached(self, results):
        assert results.n_accepted == 300


class TestHardMinConstraint:
    """Hard lower bound: every accepted solution must be older than the bound."""

    @pytest.fixture(scope="class")
    def results(self):
        from hidy_depth_profile.simulator import MonteCarloSimulator
        s = _make_settings(n_solutions=300)
        s.age_min_constraint = _DistParam("constant", [10000.0])
        sim = MonteCarloSimulator(s)
        return sim.run(seed=42)

    def test_no_solution_younger_than_bound(self, results):
        assert np.all(results.age >= 10000.0), (
            f"Solutions younger than 10 ka found: min = {results.age.min():.0f} yr"
        )

    def test_n_solutions_reached(self, results):
        assert results.n_accepted == 300


# ---------------------------------------------------------------------------
# Soft (normal) constraint
# ---------------------------------------------------------------------------

class TestAgeEstimateConstraint:
    """
    age_estimate_constraint uses a bilateral Gaussian PDF weight.
    Ages both older and younger than the estimate mean are penalised;
    unlike age_max_constraint, ages older than the mean remain allowed.
    """

    def test_normal_mode_accepted(self):
        s = _make_settings()
        s.age_estimate_constraint = _DistParam("normal", [13000.0, 1000.0])
        assert s.age_estimate_constraint.parameters == [13000.0, 1000.0]

    def test_constant_mode_rejected(self):
        s = _make_settings()
        with pytest.raises(ValueError, match="normal"):
            s.age_estimate_constraint = _DistParam("constant", [13000.0])

    def test_set_none_clears(self):
        s = _make_settings()
        s.age_estimate_constraint = _DistParam("normal", [13000.0, 1000.0])
        s.age_estimate_constraint = None
        assert s.age_estimate_constraint is None

    def test_yaml_roundtrip(self, tmp_path):
        from hidy_depth_profile.yaml_io import load_yaml, save_yaml
        s = _make_settings()
        s.age_estimate_constraint = _DistParam("normal", [13000.0, 1000.0])
        path = str(tmp_path / "est.yaml")
        save_yaml(s, path)
        s2 = load_yaml(path)
        assert s2.age_estimate_constraint.mode == "normal"
        assert s2.age_estimate_constraint.parameters == [13000.0, 1000.0]

    @pytest.fixture(scope="class")
    def results_estimate(self):
        from hidy_depth_profile.simulator import MonteCarloSimulator
        s = _make_settings(n_solutions=500)
        # Estimate at 13 ka matches the synthetic data; MAP should stay near 13 ka
        s.age_estimate_constraint = _DistParam("normal", [13000.0, 1000.0])
        return MonteCarloSimulator(s).run(seed=42)

    def test_map_near_estimate(self, results_estimate):
        """MAP should lie within 2σ of the estimate centre."""
        assert abs(results_estimate.best_age_ka - 13.0) < 2.0

    def test_older_ages_not_excluded(self, results_estimate):
        """Bilateral constraint: accepted solutions older than the estimate mean are allowed."""
        assert np.any(results_estimate.age > 13000.0), (
            "No accepted solution older than the estimate mean — constraint is not bilateral"
        )

    def test_younger_ages_not_excluded(self, results_estimate):
        """Bilateral constraint: accepted solutions younger than the estimate mean are allowed."""
        assert np.any(results_estimate.age < 13000.0)

    @pytest.fixture(scope="class")
    def results_tight_estimate(self):
        """Tight estimate far from unconstrained MAP should pull the result toward the estimate."""
        from hidy_depth_profile.simulator import MonteCarloSimulator
        base = _make_settings(n_solutions=500)
        constrained = _make_settings(n_solutions=500)
        # Tight estimate 3 ka younger than the unconstrained MAP (~13 ka)
        constrained.age_estimate_constraint = _DistParam("normal", [10000.0, 500.0])
        r_base = MonteCarloSimulator(base).run(seed=42)
        r_con  = MonteCarloSimulator(constrained).run(seed=42)
        return r_base, r_con

    def test_tight_estimate_shifts_map(self, results_tight_estimate):
        r_base, r_con = results_tight_estimate
        assert r_con.best_age_ka < r_base.best_age_ka


class TestSoftMaxConstraint:
    """
    A tight normal max constraint centred well below the unconstrained MAP
    should pull the Bayesian MAP toward younger ages.
    """

    @pytest.fixture(scope="class")
    def results_pair(self):
        from hidy_depth_profile.simulator import MonteCarloSimulator
        base = _make_settings(n_solutions=500)
        constrained = _make_settings(n_solutions=500)
        # Constraint mean at 11 ka (2 ka below the ~13 ka MAP), σ = 1 ka
        constrained.age_max_constraint = _DistParam("normal", [11000.0, 1000.0])
        r_base = MonteCarloSimulator(base).run(seed=42)
        r_con  = MonteCarloSimulator(constrained).run(seed=42)
        return r_base, r_con

    def test_constraint_shifts_map_younger(self, results_pair):
        r_base, r_con = results_pair
        assert r_con.best_age_ka < r_base.best_age_ka, (
            f"Expected constrained MAP < unconstrained MAP, "
            f"got {r_con.best_age_ka:.1f} vs {r_base.best_age_ka:.1f} ka"
        )

    def test_constrained_upper_percentile_younger(self, results_pair):
        """75th-percentile of accepted ages should be younger with the constraint."""
        r_base, r_con = results_pair
        assert np.percentile(r_con.age, 75) < np.percentile(r_base.age, 75)
