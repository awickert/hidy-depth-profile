"""
MonteCarloSimulator and Results.

Translation of be_maincalc.m, extended with vectorised batch evaluation:
- Parameter draws use NumPy RNG batch methods (one call per distribution).
- The forward model evaluates an entire batch of B draws simultaneously.
  Two backends are available, selected automatically at setup:
    * Numba (``pip install hidy-depth-profile[fast]``): JIT-compiled,
      parallelised over all CPU cores via prange. Activates when Numba
      is installed and compatible with the current NumPy.
    * NumPy fallback: (B, 6) broadcasting with a Python loop over depths.
- The signed surface-change rate (positive = erosion, negative = deposition)
  feeds directly into the production integral: simp = λ + rate·ρ/Λ. Negative
  rates represent burial, requiring a physical constraint that total deposition
  does not exceed the depth of the shallowest sample.
- The importance-sampling grid is updated with np.add.at on flat indices.
- Batch size is adapted each cycle to target ~100 accepted solutions per
  batch, staying efficient across both high and low acceptance-rate cases.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ._numba import NUMBA_AVAILABLE, NUMBA_INFO, numba_forward_batch
from .forward import chi2_profile
from .inversion import BayesianPDF, bayesian_pdf, chi2_threshold
from .production import fit_muon_curves, lsdn_surface_rate, stone2000_surface_rate
from .settings import ProfileSettings

_BE10_HALFLIFE = 1.387e6   # yr
_AL26_HALFLIFE = 7.17e5    # yr

_MIN_BATCH = 500
_MAX_BATCH = 200_000
_TARGET_ACCEPTED_PER_BATCH = 100


@dataclass
class Results:
    """
    Accepted MC solutions and Bayesian PDFs from a profile simulation.

    All arrays have length n_accepted.
    """
    chi2: np.ndarray
    age: np.ndarray                        # yr
    erosion_deposition_rate: np.ndarray   # cm/yr (positive=erosion, negative=deposition)
    inheritance: np.ndarray               # atoms/g
    spall_prod_rate: np.ndarray           # atoms/g/yr (drawn)
    muon_prod_rate: np.ndarray            # total muon surface rate (drawn)
    neutron_attenuation: np.ndarray       # g/cm²
    decay_const: np.ndarray              # yr⁻¹
    densities: np.ndarray                 # (n_accepted, n_depths)

    n_accepted: int
    n_iterations: int

    pdf_age: Optional[BayesianPDF]
    pdf_erosion_deposition: Optional[BayesianPDF]
    pdf_inheritance: Optional[BayesianPDF]

    best_age_ka: float = 0.0
    best_erosion_deposition_cm_ka: float = 0.0
    best_inheritance_1e4: float = 0.0

    age_sigma2_plus_ka: float = float("nan")
    age_sigma2_minus_ka: float = float("nan")
    erosion_deposition_sigma2_plus_cm_ka: float = float("nan")
    erosion_deposition_sigma2_minus_cm_ka: float = float("nan")

    def to_csv(self, filename: str):
        """Write chi², age (ka), erosion (cm/ka), inheritance to CSV."""
        header = "chi2,age_ka,erosion_cm_ka,inheritance_atoms_g"
        data = np.column_stack([
            self.chi2,
            self.age / 1e3,
            self.erosion_rate * 1e3,
            self.inheritance,
        ])
        np.savetxt(filename, data, delimiter=",", header=header, comments="")


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def _bin_indices(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """
    Nearest-bin indices for uniformly spaced bins. O(n), no argmin loop.
    Falls back gracefully when bins has only one element.
    """
    if len(bins) == 1:
        return np.zeros(len(values), dtype=np.intp)
    step = (bins[-1] - bins[0]) / (len(bins) - 1)
    idx = np.rint((values - bins[0]) / step).astype(np.intp)
    return np.clip(idx, 0, len(bins) - 1)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class MonteCarloSimulator:
    """
    Run Monte Carlo importance-sampling inversion for a ¹⁰Be depth profile.

    Usage::

        sim = MonteCarloSimulator(settings)
        results = sim.run()
    """

    MAX_TOTAL_ITER = 10_000_000

    def __init__(self, settings: ProfileSettings):
        settings.validate()
        self.s = settings
        self._setup()

    # ------------------------------------------------------------------
    def _setup(self):
        """Pre-compute quantities that stay fixed across all MC draws."""
        s = self.s

        halflife = _BE10_HALFLIFE if s.isotope == "Be-10" else _AL26_HALFLIFE
        self._halflife = halflife

        # muon production curves
        print("Fitting muon production curves …", flush=True)
        muon = fit_muon_curves(s.elevation, s.isotope, s.muon_fit_depth_m)

        fc = muon["fast_coeff"]   # [a1, a2, b1, b2]
        nc = muon["neg_coeff"]    # [a1, a2, a3, b1, b2, b3]
        fbs = muon["fast_surface"]
        nbs = muon["neg_surface"]
        tmu = muon["total_muon"]

        self._mufast1 = fc[2] / fbs
        self._mufast2 = fc[3] / fbs
        self._muneg1 = nc[3] / nbs
        self._muneg2 = nc[4] / nbs
        self._muneg3 = nc[5] / nbs

        self._cfast1 = fc[0] / tmu
        self._cfast2 = fc[1] / tmu
        self._cneg1 = nc[0] / tmu
        self._cneg2 = nc[1] / tmu
        self._cneg3 = nc[2] / tmu

        self._fast_surface = fbs
        self._neg_surface = nbs
        self._total_muon = tmu
        self._fast_relerr = muon["fast_relerr"]
        self._neg_relerr = muon["neg_relerr"]

        # shielding factor (applied to spallation only)
        self._shielding = s.shielding_value

        # spallation surface production rate
        scheme = s.production_scheme
        if scheme == "constant":
            spall_mean = s.constant_rate
        elif scheme == "stone2000":
            raw = stone2000_surface_rate(
                s.latitude, s.longitude, s.elevation,
                s.reference_rate, s.isotope,
            )
            spall_mean = raw * self._shielding
        else:  # lsdn
            raw = lsdn_surface_rate(
                s.latitude, s.longitude, s.elevation,
                s.lsdn_assumed_age_yr, s.reference_rate,
                s.collection_year, s.isotope,
            )
            spall_mean = raw * self._shielding
        self._spall_mean = spall_mean

        if scheme != "constant" and s.production_error.mode == "normal":
            s._production_error.parameters[0] = spall_mean
        elif scheme != "constant" and s.production_error.mode == "uniform":
            half = (s._production_error.parameters[1] - s._production_error.parameters[0]) / 2.0
            s._production_error.parameters = [spall_mean - half, spall_mean + half]

        if s.production_error.mode == "constant":
            print(f"  Spallation rate (constant, pre-computed): {s.production_error.parameters[0]:.4f} at/g/yr",
                  flush=True)
        else:
            print(f"  Spallation rate (computed, shielded): {spall_mean:.4f} at/g/yr", flush=True)

        # profile data
        pd = s.profile_data
        self._depths = pd["depth"]
        self._thicknesses = pd["thickness"]
        self._measured = pd["concentration"]
        self._rel_error = pd["rel_error"]
        self._n_depths = len(self._depths)
        self._dof = max(1, self._n_depths)
        self._z_min = float(np.min(self._depths))  # cm — shallowness constraint for deposition

        if s.density_from_file:
            from .io import cumulative_bulk_density
            self._cbd_mean, self._cbd_std = cumulative_bulk_density(
                s.density_data, self._depths, self._thicknesses,
            )
        else:
            self._cbd_mean = None
            self._cbd_std = None

        # Bayesian grid axes
        resolution = max(2, round(2 * s.mc_n_solutions ** (1 / 3)))
        if s.mc_age.mode != "constant":
            self._age_bins = np.linspace(s.mc_age.parameters[0],
                                          s.mc_age.parameters[1], resolution)
        else:
            self._age_bins = np.array([s.mc_age.parameters[0]])
        if s.mc_erosion_deposition_rate.mode != "constant":
            lo, hi = s.mc_erosion_deposition_rate.parameters
            self._erosion_deposition_bins = np.linspace(lo * 1e-3, hi * 1e-3, resolution)
        else:
            self._erosion_deposition_bins = np.array(
                [s.mc_erosion_deposition_rate.parameters[0] * 1e-3]
            )
        if s.mc_inheritance.mode != "constant":
            self._inh_bins = np.linspace(s.mc_inheritance.parameters[0],
                                          s.mc_inheritance.parameters[1], resolution)
        else:
            self._inh_bins = np.array([s.mc_inheritance.parameters[0]])

        self._chi2_thresh = chi2_threshold(s.mc_confidence_mode, s.mc_confidence_value)

        # flattened grid shape for efficient indexing
        self._grid_shape = (len(self._age_bins), len(self._erosion_deposition_bins), len(self._inh_bins))

        self._use_numba = NUMBA_AVAILABLE
        print(f"  Forward-model backend: {NUMBA_INFO}", flush=True)
        print("Setup complete.", flush=True)

    # ------------------------------------------------------------------
    def _forward_batch(self,
                        ages: np.ndarray,
                        decay_consts: np.ndarray,
                        erosions: np.ndarray,
                        inheritances: np.ndarray,
                        densities: np.ndarray,
                        v1: np.ndarray,
                        v2: np.ndarray) -> np.ndarray:
        """
        Forward model for a batch of B parameter draws → (B, n_depths) atoms/g.

        Dispatches to the Numba JIT kernel when Numba is available; falls back
        to the pure-NumPy implementation otherwise. Both paths are numerically
        identical.
        """
        if self._use_numba:
            return numba_forward_batch(
                ages, decay_consts, erosions, inheritances, densities,
                v1, v2, self._depths, self._thicknesses,
            )
        return self._forward_batch_numpy(
            ages, decay_consts, erosions, inheritances, densities, v1, v2,
        )

    def _forward_batch_numpy(self,
                              ages: np.ndarray,
                              decay_consts: np.ndarray,
                              erosions: np.ndarray,
                              inheritances: np.ndarray,
                              densities: np.ndarray,
                              v1: np.ndarray,
                              v2: np.ndarray) -> np.ndarray:
        """Pure-NumPy vectorised forward model (fallback when Numba is absent)."""
        B = len(ages)
        n = self._n_depths
        modelled = np.empty((B, n))

        inh_term = inheritances * np.exp(-ages * decay_consts)   # (B,)
        t_v = ages[:, None]
        lam_v = decay_consts[:, None]
        er_v = erosions[:, None]

        for k in range(n):
            z = self._depths[k]
            dz = self._thicknesses[k]
            rho_v = densities[:, k:k + 1]  # (B, 1) — broadcasts with (B, 6)

            tmp1 = np.exp(-z * rho_v / v1)
            tmp2 = np.exp(-(z + dz) * rho_v / v1)
            simp = lam_v + er_v * rho_v / v1
            tmp3 = (v1 * v2) * (np.exp(-t_v * simp) - 1.0) / (rho_v * simp)

            modelled[:, k] = np.sum((tmp2 - tmp1) * tmp3, axis=1) / dz + inh_term

        return modelled

    # ------------------------------------------------------------------
    def run(self, seed: Optional[int] = None) -> Results:
        """
        Execute the Monte Carlo simulation and return Results.

        Parameters
        ----------
        seed : int or None — RNG seed (None = OS entropy)
        """
        rng = np.random.default_rng(seed)
        s = self.s
        n_target = s.mc_n_solutions
        edt_lo, edt_hi = s.mc_erosion_deposition_threshold   # cm, signed

        # pre-allocate accepted solution storage
        chi2_acc = np.empty(n_target)
        age_acc = np.empty(n_target)
        erosion_deposition_acc = np.empty(n_target)
        inh_acc = np.empty(n_target)
        spall_acc = np.empty(n_target)
        muon_acc = np.empty(n_target)
        natt_acc = np.empty(n_target)
        decay_acc = np.empty(n_target)
        dens_acc = np.empty((n_target, self._n_depths))

        # flat importance-sampling grid
        grid_sums = np.zeros(self._grid_shape)
        grid_counts = np.zeros(self._grid_shape, dtype=np.intp)
        grid_sums_flat = grid_sums.ravel()
        grid_counts_flat = grid_counts.ravel()

        n_age, n_er, n_inh = self._grid_shape
        stride_age = n_er * n_inh
        stride_er = n_inh

        j = 0          # accepted solutions stored
        counter = 0    # total parameter draws attempted
        batch_size = 10_000
        t0 = time.monotonic()

        print(f"Running {n_target} MC solutions …", flush=True)

        while j < n_target:
            if counter > self.MAX_TOTAL_ITER and j == 0:
                raise RuntimeError(
                    f"No accepted solution after {self.MAX_TOTAL_ITER} iterations. "
                    "Check settings (erosion threshold, chi2 threshold, prior ranges)."
                )

            # ---- draw age + erosion/deposition; apply threshold + shallowness constraints ----
            ages_raw = s.mc_age.draw_batch(rng, batch_size)
            erosions_raw = s.mc_erosion_deposition_rate.draw_batch(rng, batch_size) * 1e-3  # → cm/yr
            total_change = ages_raw * erosions_raw   # signed cm: positive=erosion, negative=deposition
            valid = (total_change >= edt_lo) & (total_change <= edt_hi)
            # Physical constraint: total deposition cannot exceed the depth of the
            # shallowest sample — otherwise that sample was above the surface during
            # part of the integration window.
            valid &= total_change >= -self._z_min
            ages = ages_raw[valid]
            erosions = erosions_raw[valid]
            B = len(ages)
            counter += batch_size

            if B == 0:
                continue

            # ---- draw remaining parameters for valid draws ----
            inheritances = s.mc_inheritance.draw_batch(rng, B)
            prod_rates = s.production_error.draw_batch(rng, B)
            muon_rates = self._total_muon * (
                1.0 + rng.standard_normal(B) * (s.muon_percent_error / 100.0)
            )
            natts = s.mc_neutron_attenuation.draw_batch(rng, B)

            # noisy muon attenuation scale factors
            cfbs = self._fast_surface * (1.0 + rng.standard_normal(B) * self._fast_relerr)
            cnbs = self._neg_surface * (1.0 + rng.standard_normal(B) * self._neg_relerr)

            # decay constant with half-life uncertainty
            decay_consts = np.log(2.0) / (
                self._halflife * (1.0 + rng.standard_normal(B) * s.half_life_sigma)
            )

            # density: (B, n_depths)
            if s.density_from_file:
                densities = rng.standard_normal((B, self._n_depths)) * self._cbd_std + self._cbd_mean
            else:
                rho_scalar = s.density_error.draw_batch(rng, B)   # (B,)
                densities = np.outer(rho_scalar, np.ones(self._n_depths))

            # ---- build v1 (B, 6) and v2 (B, 6) ----
            v1 = np.column_stack([
                natts,
                cfbs * self._mufast1,
                cfbs * self._mufast2,
                cnbs * self._muneg1,
                cnbs * self._muneg2,
                cnbs * self._muneg3,
            ])
            v2 = np.column_stack([
                prod_rates,
                muon_rates * self._cfast1,
                muon_rates * self._cfast2,
                muon_rates * self._cneg1,
                muon_rates * self._cneg2,
                muon_rates * self._cneg3,
            ])

            # ---- evaluate forward model: (B, n_depths) ----
            modelled = self._forward_batch(
                ages, decay_consts, erosions, inheritances, densities, v1, v2
            )

            # ---- chi² for all B draws ----
            residuals = (modelled - self._measured) / (self._measured * self._rel_error)
            chi2_batch = np.sum(residuals ** 2, axis=1) / self._dof   # (B,)

            # ---- update importance-sampling grid (all draws) ----
            chi_weighted = np.exp(-chi2_batch / 2.0)
            xi = _bin_indices(ages, self._age_bins)
            yi = _bin_indices(erosions, self._erosion_deposition_bins)
            zi = _bin_indices(inheritances, self._inh_bins)
            flat_idx = xi * stride_age + yi * stride_er + zi
            np.add.at(grid_sums_flat, flat_idx, chi_weighted)
            np.add.at(grid_counts_flat, flat_idx, 1)

            # ---- collect accepted solutions ----
            accepted = chi2_batch <= self._chi2_thresh
            n_new = accepted.sum()
            if n_new > 0:
                take = min(n_new, n_target - j)
                idx_acc = np.where(accepted)[0][:take]
                chi2_acc[j:j + take] = chi2_batch[idx_acc]
                age_acc[j:j + take] = ages[idx_acc]
                erosion_deposition_acc[j:j + take] = erosions[idx_acc]
                inh_acc[j:j + take] = inheritances[idx_acc]
                spall_acc[j:j + take] = prod_rates[idx_acc]
                muon_acc[j:j + take] = muon_rates[idx_acc]
                natt_acc[j:j + take] = natts[idx_acc]
                decay_acc[j:j + take] = decay_consts[idx_acc]
                dens_acc[j:j + take] = densities[idx_acc]
                j += take

            # ---- adapt batch size for next cycle ----
            acc_rate = n_new / B if B > 0 else 1e-6
            batch_size = int(np.clip(
                _TARGET_ACCEPTED_PER_BATCH / max(acc_rate, 1e-6),
                _MIN_BATCH, _MAX_BATCH,
            ))

            if j > 0 and j % max(1, n_target // 10) == 0:
                elapsed = time.monotonic() - t0
                print(
                    f"  {j}/{n_target} solutions | {counter:,} draws | "
                    f"{elapsed:.1f}s | batch={batch_size:,}",
                    flush=True,
                )

        elapsed = time.monotonic() - t0
        print(f"Done. {j} solutions in {counter:,} draws ({elapsed:.2f}s).", flush=True)

        # ---- Bayesian PDFs ----
        pdf_age, pdf_ed, pdf_inh = bayesian_pdf(
            grid_sums, grid_counts,
            self._age_bins, self._erosion_deposition_bins, self._inh_bins,
        )

        best_age_ka = pdf_age.map_value / 1e3 if pdf_age else age_acc[0] / 1e3
        best_ed = pdf_ed.map_value * 1e3 if pdf_ed else erosion_deposition_acc[0] * 1e3
        best_inh = pdf_inh.map_value / 1e4 if pdf_inh else inh_acc[0] / 1e4

        return Results(
            chi2=chi2_acc[:j],
            age=age_acc[:j],
            erosion_deposition_rate=erosion_deposition_acc[:j],
            inheritance=inh_acc[:j],
            spall_prod_rate=spall_acc[:j],
            muon_prod_rate=muon_acc[:j],
            neutron_attenuation=natt_acc[:j],
            decay_const=decay_acc[:j],
            densities=dens_acc[:j],
            n_accepted=j,
            n_iterations=counter,
            pdf_age=pdf_age,
            pdf_erosion_deposition=pdf_ed,
            pdf_inheritance=pdf_inh,
            best_age_ka=best_age_ka,
            best_erosion_deposition_cm_ka=best_ed,
            best_inheritance_1e4=best_inh,
            age_sigma2_plus_ka=pdf_age.sigma2_plus / 1e3 if pdf_age else float("nan"),
            age_sigma2_minus_ka=pdf_age.sigma2_minus / 1e3 if pdf_age else float("nan"),
            erosion_deposition_sigma2_plus_cm_ka=pdf_ed.sigma2_plus * 1e3 if pdf_ed else float("nan"),
            erosion_deposition_sigma2_minus_cm_ka=pdf_ed.sigma2_minus * 1e3 if pdf_ed else float("nan"),
        )
