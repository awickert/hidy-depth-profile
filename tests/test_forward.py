"""
Unit tests for the forward model.
"""
import numpy as np
import pytest

from hidy_depth_profile.forward import sample_concentration, chi2_profile


DECAY_BE10 = np.log(2) / 1.387e6   # yr⁻¹


class TestSampleConcentration:
    """Check known limiting cases of the forward model."""

    def _v1_v2(self, spall=5.0, muon_fast=0.05, muon_neg=0.02):
        """Toy v1/v2 arrays with simple values."""
        v1 = np.array([160.0, 1500.0, 3000.0, 1200.0, 2500.0, 4000.0])
        v2 = np.array([spall, muon_fast * 0.6, muon_fast * 0.4,
                       muon_neg * 0.4, muon_neg * 0.35, muon_neg * 0.25])
        return v1, v2

    def test_zero_age_gives_inheritance(self):
        """With t=0 the concentration should equal the inheritance."""
        v1, v2 = self._v1_v2()
        N_inh = 12345.0
        N = sample_concentration(
            depth=10.0, thickness=5.0, density=2.3,
            v1=v1, v2=v2,
            time=0.0, decay_const=DECAY_BE10,
            erosion_deposition_rate=0.0, inheritance=N_inh,
        )
        assert abs(N - N_inh) / N_inh < 1e-8

    def test_positive_concentration(self):
        """Modelled concentration must be positive for typical inputs."""
        v1, v2 = self._v1_v2()
        N = sample_concentration(
            depth=27.5, thickness=10.0, density=2.2,
            v1=v1, v2=v2,
            time=15000.0, decay_const=DECAY_BE10,
            erosion_deposition_rate=0.001, inheritance=5000.0,
        )
        assert N > 0

    def test_deeper_sample_lower_concentration(self):
        """Deeper samples receive less spallation, so N should be lower at depth."""
        v1, v2 = self._v1_v2()
        kwargs = dict(thickness=10.0, density=2.2,
                      v1=v1, v2=v2,
                      time=15000.0, decay_const=DECAY_BE10,
                      erosion_deposition_rate=0.001, inheritance=0.0)
        N_shallow = sample_concentration(depth=10.0, **kwargs)
        N_deep = sample_concentration(depth=200.0, **kwargs)
        assert N_shallow > N_deep

    def test_higher_erosion_lower_concentration(self):
        """Higher erosion (more positive rate) continuously removes material → lower N."""
        v1, v2 = self._v1_v2()
        kwargs = dict(depth=30.0, thickness=10.0, density=2.2,
                      v1=v1, v2=v2,
                      time=15000.0, decay_const=DECAY_BE10,
                      inheritance=0.0)
        N_low_er = sample_concentration(erosion_deposition_rate=0.0001, **kwargs)
        N_high_er = sample_concentration(erosion_deposition_rate=0.01, **kwargs)
        assert N_low_er > N_high_er


class TestChi2:
    def test_perfect_fit(self):
        """chi² = 0 when modelled = measured."""
        measured = np.array([5e5, 4e5, 3e5])
        chi2 = chi2_profile(measured, measured, np.array([0.05, 0.05, 0.05]), dof=3)
        assert abs(chi2) < 1e-12

    def test_nonzero_fit(self):
        """chi² > 0 when modelled ≠ measured."""
        measured = np.array([5e5, 4e5, 3e5])
        modelled = measured * 1.1
        chi2 = chi2_profile(modelled, measured, np.array([0.05, 0.05, 0.05]), dof=3)
        assert chi2 > 0
