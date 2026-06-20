"""
Tests for the production module: muon model and scaling functions.
"""
import numpy as np
import pytest

from hidy_depth_profile.production import (
    _LZ, _Rv0, muon_production_at_depth,
    lsdn_rates_for_ages, precompute_lsdn_timeseries,
)

# Lees Ferry site for LSDn tests
_LF_LAT, _LF_LON, _LF_ELEV = 36.852, -111.606, 985.0


class TestHelperFunctions:
    def test_Rv0_positive(self):
        """Stopping rate should be positive for z > 0."""
        z = np.array([100.0, 1000.0, 10000.0])
        r = _Rv0(z)
        assert np.all(r > 0)

    def test_LZ_monotonic(self):
        """Attenuation length LZ should increase with depth (range)."""
        z = np.array([100.0, 500.0, 5000.0, 50000.0])
        lz = _LZ(z)
        assert np.all(np.diff(lz) > 0)

    def test_LZ_handles_zero(self):
        """LZ should not crash on z = 0 (clamps to z = 1)."""
        lz = _LZ(np.array([0.0]))
        assert np.isfinite(lz[0])


class TestMuonProduction:
    def test_positive_production(self):
        """Fast and negative muon production must be > 0."""
        massdepths = np.array([0.0, 500.0, 2000.0, 10000.0])
        total, fast, neg = muon_production_at_depth(massdepths, elev=1000.0)
        assert np.all(fast > 0), "Fast muon production should be positive"
        assert np.all(neg > 0), "Negative muon production should be positive"
        assert np.all(total > 0)

    def test_decreasing_with_depth(self):
        """Both pathways should decrease with increasing depth."""
        massdepths = np.linspace(0, 50000, 20)
        _, fast, neg = muon_production_at_depth(massdepths, elev=500.0)
        assert np.all(np.diff(fast) < 0)
        assert np.all(np.diff(neg) < 0)

    def test_higher_elevation_higher_rate(self):
        """Higher elevation → lower atmospheric pressure → higher muon rate."""
        z = np.array([1000.0])
        _, fast_lo, _ = muon_production_at_depth(z, elev=0.0)
        _, fast_hi, _ = muon_production_at_depth(z, elev=3000.0)
        assert fast_hi > fast_lo

    def test_surface_rate_reasonable(self):
        """Surface fast muon rate for Be-10 at sea level should be a few hundredths at/g/yr."""
        z = np.array([0.0])
        _, fast, neg = muon_production_at_depth(z, elev=0.0, isotope='Be-10')
        # Heisinger rates at SLHL are ~0.0 to ~0.1 at/g/yr for each component
        assert 1e-4 < float(fast[0]) < 1.0, f"Unexpected fast muon rate: {fast}"
        assert 1e-4 < float(neg[0]) < 1.0, f"Unexpected neg muon rate: {neg}"

    @pytest.mark.parametrize("isotope", ["Be-10", "Al-26"])
    def test_isotope_modes(self, isotope):
        """Both isotopes should return positive finite values."""
        z = np.array([0.0, 500.0])
        total, fast, neg = muon_production_at_depth(z, elev=500.0, isotope=isotope)
        assert np.all(np.isfinite(total))
        assert np.all(fast > 0)
        assert np.all(neg > 0)


class TestLSDnPerDraw:
    """Tests for precompute_lsdn_timeseries and lsdn_rates_for_ages."""

    @pytest.fixture(scope="class")
    def lsdn_ts(self):
        return precompute_lsdn_timeseries(
            _LF_LAT, _LF_LON, _LF_ELEV, t_max=110_000, collection_year=2010
        )

    def test_timeseries_structure(self, lsdn_ts):
        """Time series should have consistent, adjacent intervals."""
        assert len(lsdn_ts["tmax"]) > 0
        assert np.allclose(lsdn_ts["tmax"][:-1], lsdn_ts["tmin"][1:])
        assert lsdn_ts["tmin"][0] == 0.0
        assert lsdn_ts["P_ref"] > 0

    def test_rates_match_surface_rate(self, lsdn_ts):
        """Per-draw rates must match lsdn_surface_rate at the same ages."""
        from hidy_depth_profile.production import lsdn_surface_rate
        test_ages = np.array([60_000.0, 85_000.0, 110_000.0])
        batch = lsdn_rates_for_ages(test_ages, lsdn_ts)
        single = np.array([
            lsdn_surface_rate(_LF_LAT, _LF_LON, _LF_ELEV, a, 4.086, 2010)
            for a in test_ages
        ])
        np.testing.assert_allclose(batch, single, rtol=1e-10)

    def test_rates_vary_with_age(self, lsdn_ts):
        """LSDn rate should differ between 60 ka and 110 ka (geomagnetic variation)."""
        ages = np.array([60_000.0, 110_000.0])
        rates = lsdn_rates_for_ages(ages, lsdn_ts)
        assert rates[0] != rates[1]
        assert np.all(rates > 0)

    def test_batch_vs_scalar(self, lsdn_ts):
        """Batch evaluation must match scalar loop across a random set of ages."""
        rng = np.random.default_rng(42)
        ages = rng.uniform(60_000, 110_000, 50)
        batch = lsdn_rates_for_ages(ages, lsdn_ts)
        scalar = np.array([
            float(lsdn_rates_for_ages(np.array([a]), lsdn_ts)[0]) for a in ages
        ])
        np.testing.assert_allclose(batch, scalar, rtol=1e-12)
