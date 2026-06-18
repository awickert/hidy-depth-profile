"""
Bayesian inversion: importance-sampling grid and marginal PDFs.

Translation of be_probfix3.m.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class BayesianPDF:
    """Marginalised 1-D PDF for one parameter."""
    bins: np.ndarray      # bin centres
    pdf: np.ndarray       # normalised probability density
    cdf: np.ndarray       # cumulative probability (0–1)
    map_value: float      # mode (MAP estimate)
    sigma2_plus: float    # 97.7th-percentile value (≈ +2σ)
    sigma2_minus: float   # 2.3rd-percentile value  (≈ -2σ)


def _interp_quantile(bins, cdf, q):
    """Return the bin value at cumulative probability q, or nan if out of range."""
    cdf_u, idx = np.unique(cdf, return_index=True)
    bins_u = bins[idx]
    if q < cdf_u[0] or q > cdf_u[-1]:
        return float("nan")
    return float(np.interp(q, cdf_u, bins_u))


def bayesian_pdf(
    grid_sums: np.ndarray,
    grid_counts: np.ndarray,
    age_bins: np.ndarray,
    erosion_bins: np.ndarray,
    inheritance_bins: np.ndarray,
) -> Tuple[Optional[BayesianPDF], Optional[BayesianPDF], Optional[BayesianPDF]]:
    """
    Compute marginal PDFs and MAP estimates from the 3-D importance-sampling grid.

    Translation of be_probfix3.m.

    Parameters
    ----------
    grid_sums : 3-D array (n_age, n_erosion, n_inheritance), sum of exp(-chi2/2) weights
    grid_counts : 3-D array, draw counts per cell
    age_bins : 1-D array, age axis values (yr) — may be scalar if constant
    erosion_bins : 1-D array, erosion axis values (cm/yr) — may be scalar
    inheritance_bins : 1-D array, inheritance axis values (atoms/g) — may be scalar

    Returns
    -------
    pdf_age, pdf_erosion, pdf_inheritance : BayesianPDF or None if parameter is constant
    """
    # normalise by count: convert sum to mean weight per cell
    grid = grid_sums.copy()
    valid = grid_counts > 0
    grid[valid] /= grid_counts[valid]

    pdf_age = pdf_erosion = pdf_inheritance = None

    # age marginal
    if len(age_bins) > 1:
        step = age_bins[1] - age_bins[0]
        probs = grid.sum(axis=(1, 2))
        probs = probs / (probs.sum() * step)
        cdf = np.cumsum(probs) / probs.sum()
        map_val = float(age_bins[np.argmax(probs)])
        pdf_age = BayesianPDF(
            bins=age_bins,
            pdf=probs,
            cdf=cdf,
            map_value=map_val,
            sigma2_plus=_interp_quantile(age_bins, cdf, 0.977),
            sigma2_minus=_interp_quantile(age_bins, cdf, 0.023),
        )

    # erosion marginal (erosion_bins in cm/yr internally; user sees cm/ka)
    if len(erosion_bins) > 1:
        step = erosion_bins[1] - erosion_bins[0]
        probs = grid.sum(axis=(0, 2))
        probs = probs / (probs.sum() * step)
        cdf = np.cumsum(probs) / probs.sum()
        map_val = float(erosion_bins[np.argmax(probs)])
        pdf_erosion = BayesianPDF(
            bins=erosion_bins,
            pdf=probs,
            cdf=cdf,
            map_value=map_val,
            sigma2_plus=_interp_quantile(erosion_bins, cdf, 0.977),
            sigma2_minus=_interp_quantile(erosion_bins, cdf, 0.023),
        )

    # inheritance marginal
    if len(inheritance_bins) > 1:
        step = inheritance_bins[1] - inheritance_bins[0]
        probs = grid.sum(axis=(0, 1))
        probs = probs / (probs.sum() * step)
        cdf = np.cumsum(probs) / probs.sum()
        map_val = float(inheritance_bins[np.argmax(probs)])
        pdf_inheritance = BayesianPDF(
            bins=inheritance_bins,
            pdf=probs,
            cdf=cdf,
            map_value=map_val,
            sigma2_plus=_interp_quantile(inheritance_bins, cdf, 0.977),
            sigma2_minus=_interp_quantile(inheritance_bins, cdf, 0.023),
        )

    return pdf_age, pdf_erosion, pdf_inheritance


def chi2_threshold(confidence_mode: str, confidence_value: float) -> float:
    """
    Convert a confidence specification to a reduced-chi² acceptance threshold.

    Parameters
    ----------
    confidence_mode : 'sigma' or 'chi2'
        sigma: threshold = confidence_value² (e.g. 2σ → 4.0)
        chi2: threshold = confidence_value directly
    confidence_value : float

    Returns
    -------
    float, reduced-chi² threshold
    """
    if confidence_mode == "sigma":
        return float(confidence_value ** 2)
    if confidence_mode == "chi2":
        return float(confidence_value)
    raise ValueError(f"Unknown confidence_mode: {confidence_mode}")
