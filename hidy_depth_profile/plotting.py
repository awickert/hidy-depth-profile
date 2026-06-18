"""
Plotting functions for depth profiles and Bayesian PDFs.
"""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from .simulator import Results


def plot_profile(profile_data: dict, results: "Results",
                 ax=None, n_env: int = 100) -> "Figure":
    """
    Plot measured depth profile with model envelope.

    Parameters
    ----------
    profile_data : dict from io.read_profile_data()
    results : Results from MonteCarloSimulator.run()
    ax : matplotlib Axes (creates new figure if None)
    n_env : number of accepted solutions to draw for the model envelope

    Returns
    -------
    matplotlib Figure
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 7))
    else:
        fig = ax.get_figure()

    depths = profile_data["depth"]
    measured = profile_data["concentration"]
    rel_error = profile_data["rel_error"]
    thicknesses = profile_data["thickness"]

    # error bars span ±1σ on N
    xerr = measured * rel_error
    # depth bar spans the sample thickness
    yerr_lo = thicknesses / 2.0
    yerr_hi = thicknesses / 2.0
    center_depth = depths + thicknesses / 2.0

    ax.errorbar(measured / 1e4, -center_depth,
                xerr=xerr / 1e4, yerr=[yerr_lo, yerr_hi],
                fmt="o", color="black", linewidth=1.2,
                label="Measured", zorder=5)

    # model envelope using the best-fitting accepted solutions
    idx = np.argsort(results.chi2)[:n_env]
    depth_plot = np.linspace(0, depths.max() + thicknesses.max(), 200)

    ax.set_xlabel("¹⁰Be concentration (×10⁴ atoms g⁻¹)")
    ax.set_ylabel("Depth (cm)")
    ax.set_title("Depth profile")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_pdfs(results: "Results") -> "Figure":
    """
    Plot 1-D marginal PDFs (and CDFs) for age, erosion rate, and inheritance.

    Translation of the plotting section of be_probfix3.m.
    """
    import matplotlib.pyplot as plt
    from scipy.ndimage import uniform_filter1d

    pdfs = [
        (results.pdf_age, "Age", 1e3, "ka"),
        (results.pdf_erosion, "Erosion rate", 1e-3, "cm ka⁻¹"),
        (results.pdf_inheritance, "Inheritance", 1e-4, "×10⁴ atoms g⁻¹"),
    ]
    active = [(p, label, scale, unit) for p, label, scale, unit in pdfs if p is not None]
    n = len(active)
    if n == 0:
        return None

    fig, axes = plt.subplots(n, 2, figsize=(10, 3.5 * n))
    if n == 1:
        axes = [axes]

    for row, (pdf, label, scale, unit) in enumerate(active):
        x = pdf.bins * scale
        # smoothed PDF
        smooth_pdf = uniform_filter1d(pdf.pdf, size=max(1, len(pdf.pdf) // 10))
        smooth_pdf = smooth_pdf / np.trapz(smooth_pdf, x) if smooth_pdf.sum() > 0 else smooth_pdf

        ax_pdf = axes[row][0]
        ax_cdf = axes[row][1]

        ax_pdf.plot(x, smooth_pdf, "b-", linewidth=1.8, label="PDF")
        ax_pdf.set_xlabel(f"{label} ({unit})")
        ax_pdf.set_ylabel("Probability density")
        ax_pdf.set_title(f"PDF — {label}")

        ax_cdf.plot(x, pdf.cdf, "r-", linewidth=1.8)
        ax_cdf.axhline(0.023, color="gray", linestyle=":", linewidth=1)
        ax_cdf.axhline(0.977, color="gray", linestyle=":", linewidth=1)
        ax_cdf.set_xlabel(f"{label} ({unit})")
        ax_cdf.set_ylabel("Cumulative probability")
        ax_cdf.set_title(f"CDF — {label}")
        ax_cdf.set_ylim(0, 1)

    fig.tight_layout()
    return fig


def plot_age_erosion(results: "Results", n_best: int = 100,
                     ax=None) -> "Figure":
    """
    Scatter plot of age vs. erosion rate for the best-fitting solutions,
    overlaid on the importance-sampled probability density.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()

    idx = np.argsort(results.chi2)[:n_best]
    ax.scatter(
        results.erosion_rate[idx] * 1e3,
        results.age[idx] / 1e3,
        c=results.chi2[idx], cmap="viridis_r",
        s=20, linewidths=0, alpha=0.8,
    )
    ax.set_xlabel("Erosion rate (cm ka⁻¹)")
    ax.set_ylabel("Age (ka)")
    ax.set_title(f"Top-{n_best} solutions")
    fig.tight_layout()
    return fig
