"""
Tests for the production module: muon model and scaling functions.
"""
import numpy as np
import pytest

from hidy_depth_profile.production import (
    _LZ, _Rv0, muon_production_at_depth,
)


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
