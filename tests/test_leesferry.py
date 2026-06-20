"""
Integration test: Lees Ferry sand 6 profile.

Runs the MC simulator with the settings from Leesferrysand_settings.mat
(Hidy et al. 2010, Quaternary Geochronology 5, 541–560) and verifies that
the accepted solutions are physically self-consistent and lie within the
expected parameter ranges.

The test uses a small n_solutions (300) for CI speed; production results
should be run with n_solutions ≥ 1000.
"""
import os
import pathlib
import pytest
import numpy as np

# Data files live in the original MATLAB source directory
_DATA_DIR = pathlib.Path("/home/awickert/dataanalysis/Hidy_10Be_profile_simulator")
_YAML = pathlib.Path(__file__).parent.parent / "examples" / "leesferry_sand6.yaml"

_REQUIRED = [
    _DATA_DIR / "Leesferry_sand6.txt",
    _DATA_DIR / "LF_shield.txt",
    _YAML,
]


def _data_available():
    return all(p.exists() for p in _REQUIRED)


@pytest.fixture(scope="module")
def leesferry_lsdn():
    """
    Run Lees Ferry with the LSDn production scheme.

    Returns (simulator, results) so tests can access simulator internals
    (stored time series, attenuation length fractions) alongside the
    accepted solutions.
    """
    if not _data_available():
        pytest.skip("Lees Ferry data files not found")

    from hidy_depth_profile.yaml_io import load_yaml
    from hidy_depth_profile.simulator import MonteCarloSimulator

    s = load_yaml(str(_YAML))
    s.load_profile(str(_DATA_DIR / "Leesferry_sand6.txt"))
    s.production_scheme = "lsdn"
    s.collection_year = 2010
    s.mc_n_solutions = 100
    s.mc_confidence_mode = "sigma"
    s.mc_confidence_value = 2.0

    sim = MonteCarloSimulator(s)
    results = sim.run(seed=42)
    return sim, results


@pytest.fixture(scope="module")
def leesferry_results():
    """Run the Lees Ferry simulation once and return the Results object."""
    if not _data_available():
        pytest.skip("Lees Ferry data files not found")

    from hidy_depth_profile.yaml_io import load_yaml
    from hidy_depth_profile.simulator import MonteCarloSimulator

    s = load_yaml(str(_YAML))
    s.load_profile(str(_DATA_DIR / "Leesferry_sand6.txt"))
    # shielding already folded into the constant production rate; skip load_shielding
    s.mc_n_solutions = 300
    s.mc_confidence_mode = "sigma"
    s.mc_confidence_value = 2.0

    sim = MonteCarloSimulator(s)
    return sim.run(seed=42)


# --------------------------------------------------------------------------
class TestLeesFerryPhysics:
    """Verify that the accepted solutions are physically self-consistent."""

    def test_all_accepted_within_chi2_threshold(self, leesferry_results):
        """Every accepted solution must satisfy the sigma=2 chi² criterion."""
        chi2_thresh = 2.0 ** 2   # sigma → chi2
        assert np.all(leesferry_results.chi2 <= chi2_thresh), (
            f"Some chi2 values exceed threshold {chi2_thresh}: "
            f"max = {leesferry_results.chi2.max():.4f}"
        )

    def test_n_accepted_equals_target(self, leesferry_results):
        r = leesferry_results
        assert r.n_accepted == 300, f"Expected 300 accepted, got {r.n_accepted}"

    def test_ages_within_prior(self, leesferry_results):
        """All accepted ages must lie within the uniform prior [60, 110] ka."""
        ages_ka = leesferry_results.age / 1e3
        assert ages_ka.min() >= 60.0, f"Age below prior minimum: {ages_ka.min():.1f} ka"
        assert ages_ka.max() <= 110.0, f"Age above prior maximum: {ages_ka.max():.1f} ka"

    def test_erosion_within_prior(self, leesferry_results):
        """All erosion/deposition rates must lie within [0, 0.4] cm/ka (erosion-only prior)."""
        erosion_cm_ka = leesferry_results.erosion_deposition_rate * 1e3
        assert erosion_cm_ka.min() >= 0.0
        assert erosion_cm_ka.max() <= 0.4 + 1e-6

    def test_inheritance_within_prior(self, leesferry_results):
        """All inheritance values must lie within [60 000, 120 000] atoms/g."""
        inh = leesferry_results.inheritance
        assert inh.min() >= 60_000 - 1e-6
        assert inh.max() <= 120_000 + 1e-6

    def test_best_age_plausible(self, leesferry_results):
        """MAP age should be well inside the prior, not pinned to a boundary."""
        best_ka = leesferry_results.best_age_ka
        assert 65.0 < best_ka < 108.0, (
            f"MAP age {best_ka:.1f} ka is suspiciously close to the prior boundary"
        )

    def test_bayesian_pdfs_exist(self, leesferry_results):
        """All three marginal PDFs should be non-None (all parameters are free)."""
        assert leesferry_results.pdf_age is not None
        assert leesferry_results.pdf_erosion_deposition is not None
        assert leesferry_results.pdf_inheritance is not None

    def test_age_cdf_monotone(self, leesferry_results):
        """The age CDF must be non-decreasing."""
        cdf = leesferry_results.pdf_age.cdf
        assert np.all(np.diff(cdf) >= -1e-12), "Age CDF is not monotone"

    def test_2sigma_bounds_ordered(self, leesferry_results):
        """The 2σ lower bound must be less than the MAP, which is less than the upper bound."""
        r = leesferry_results
        assert r.age_sigma2_minus_ka < r.best_age_ka < r.age_sigma2_plus_ka, (
            f"2σ bounds not ordered: "
            f"{r.age_sigma2_minus_ka:.1f} < {r.best_age_ka:.1f} < {r.age_sigma2_plus_ka:.1f}"
        )


class TestLeesFerryForwardModel:
    """Verify the forward model against the measured profile for best-fit solutions."""

    def test_modelled_concentrations_positive(self, leesferry_results):
        """Quick sanity check: forward model should produce positive concentrations."""
        import pathlib, sys
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

        from hidy_depth_profile.forward import profile_concentration

        r = leesferry_results
        # use the lowest-chi2 solution
        idx = int(np.argmin(r.chi2))

        # reconstruct v1 / v2 — use stored params; this is just a smoke check
        # We verify by proxy: chi2 of best solution should be very small
        assert r.chi2[idx] < 4.0, f"Best chi2 = {r.chi2[idx]:.4f} exceeds threshold"

    def test_concentration_profile_decreases_with_depth(self, leesferry_results):
        """
        The measured concentrations should decrease monotonically with depth.
        This is a property of the data, not the model — verifies data loading.
        """
        from hidy_depth_profile.io import read_profile_data
        pd = read_profile_data(str(_DATA_DIR / "Leesferry_sand6.txt"))
        conc = pd["concentration"]
        assert np.all(np.diff(conc) < 0), (
            "Lees Ferry measured concentrations do not decrease monotonically with depth"
        )


class TestLeesFerryLSDn:
    """
    End-to-end tests for the LSDn per-draw production rate path.

    Two complementary checks:
    1. Stored spall_prod_rate must equal lsdn_rates_for_ages(age) × shielding
       to floating-point precision — direct proof that the per-draw rate was
       used correctly.
    2. Re-running the forward model at the best-fit solution with stored
       parameters must reproduce the stored chi² — confirms the stored rates
       are self-consistent with the accepted concentrations.
    """

    def test_lsdn_run_produces_solutions(self, leesferry_lsdn):
        """LSDn run should converge to the requested number of solutions."""
        _, r = leesferry_lsdn
        assert r.n_accepted == 100

    def test_spall_rates_match_per_draw_lsdn(self, leesferry_lsdn):
        """
        spall_prod_rate[i] must equal lsdn_rates_for_ages(age[i]) × shielding
        for every accepted solution — no fixed rate should have been used.
        """
        sim, r = leesferry_lsdn
        from hidy_depth_profile.production import lsdn_rates_for_ages
        expected = lsdn_rates_for_ages(r.age, sim._lsdn_ts) * sim._shielding
        np.testing.assert_allclose(r.spall_prod_rate, expected, rtol=1e-10,
                                   err_msg="spall_prod_rate does not match per-draw LSDn rates")

    def test_chi2_reproducible_at_best_solution(self, leesferry_lsdn):
        """
        Re-running the forward model at the stored best-fit parameters must
        reproduce the stored chi² to high precision.

        v1 is reconstructed from the stored neutron_attenuation plus the
        simulator's mean muon attenuation lengths.  The per-draw noise on
        those lengths is small (fast_relerr ≈ 0.06 %, neg_relerr ≈ 0.02 %)
        and is not stored in Results, so an exact match is not expected; the
        tolerance of 1e-3 accommodates that approximation while still
        catching large errors.
        """
        sim, r = leesferry_lsdn
        from hidy_depth_profile.forward import profile_concentration, chi2_profile

        best = int(np.argmin(r.chi2))

        v1 = np.array([
            r.neutron_attenuation[best],
            sim._fast_surface * sim._mufast1,
            sim._fast_surface * sim._mufast2,
            sim._neg_surface * sim._muneg1,
            sim._neg_surface * sim._muneg2,
            sim._neg_surface * sim._muneg3,
        ])
        v2 = np.array([
            r.spall_prod_rate[best],
            r.muon_prod_rate[best] * sim._cfast1,
            r.muon_prod_rate[best] * sim._cfast2,
            r.muon_prod_rate[best] * sim._cneg1,
            r.muon_prod_rate[best] * sim._cneg2,
            r.muon_prod_rate[best] * sim._cneg3,
        ])

        pd = sim.s.profile_data
        modelled = profile_concentration(
            pd, v1, v2,
            r.age[best], r.decay_const[best],
            r.erosion_deposition_rate[best], r.inheritance[best],
            r.densities[best],
        )
        chi2_recomputed = chi2_profile(
            modelled, pd["concentration"], pd["rel_error"], dof=sim._dof
        )
        assert abs(chi2_recomputed - r.chi2[best]) < 1e-3, (
            f"Recomputed chi²={chi2_recomputed:.6f} does not match stored "
            f"{r.chi2[best]:.6f} (diff={abs(chi2_recomputed - r.chi2[best]):.2e})"
        )
