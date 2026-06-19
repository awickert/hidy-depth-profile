"""
MonteCarloSimulator and Results.

Translation of be_maincalc.m extended with vectorised MC and multiprocessing.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .forward import chi2_profile, profile_concentration
from .inversion import BayesianPDF, bayesian_pdf, chi2_threshold
from .production import fit_muon_curves, lsdn_surface_rate, stone2000_surface_rate
from .settings import ProfileSettings

_BE10_HALFLIFE = 1.387e6   # yr
_AL26_HALFLIFE = 7.17e5    # yr


@dataclass
class Results:
    """
    Accepted MC solutions and Bayesian PDFs from a profile simulation.

    All arrays are trimmed to n_accepted entries.
    """
    chi2: np.ndarray              # reduced chi² of accepted solutions
    age: np.ndarray               # yr
    erosion_rate: np.ndarray      # cm/yr
    inheritance: np.ndarray       # atoms/g
    spall_prod_rate: np.ndarray   # atoms/g/yr (drawn)
    muon_prod_rate: np.ndarray    # total muon surface rate (drawn)
    neutron_attenuation: np.ndarray  # g/cm²
    decay_const: np.ndarray       # yr⁻¹
    densities: np.ndarray         # (n_accepted, n_depths)

    n_accepted: int
    n_iterations: int

    # Bayesian PDFs (None if parameter was held constant)
    pdf_age: Optional[BayesianPDF]
    pdf_erosion: Optional[BayesianPDF]
    pdf_inheritance: Optional[BayesianPDF]

    # Bayesian MAP estimates (ka / cm/ka / 10⁴ atoms/g for readability)
    best_age_ka: float = 0.0
    best_erosion_cm_ka: float = 0.0
    best_inheritance_1e4: float = 0.0

    # 2σ bounds (same units as above)
    age_sigma2_plus_ka: float = float("nan")
    age_sigma2_minus_ka: float = float("nan")
    erosion_sigma2_plus_cm_ka: float = float("nan")
    erosion_sigma2_minus_cm_ka: float = float("nan")

    def to_csv(self, filename: str):
        """Write chi², age (ka), erosion (cm/ka), inheritance to CSV."""
        header = "chi2,age_ka,erosion_cm_ka,inheritance_atoms_g"
        data = np.column_stack([
            self.chi2,
            self.age / 1e3,
            self.erosion_rate * 1e3,  # cm/yr → cm/ka
            self.inheritance,
        ])
        np.savetxt(filename, data, delimiter=",", header=header, comments="")


class MonteCarloSimulator:
    """
    Run Monte Carlo importance-sampling inversion for a ¹⁰Be depth profile.

    Usage::

        sim = MonteCarloSimulator(settings)
        results = sim.run()
    """

    MAX_ITER = 5_000_000  # hard stop if no convergence

    def __init__(self, settings: ProfileSettings):
        settings.validate()
        self.s = settings
        self._setup()

    # ------------------------------------------------------------------
    def _setup(self):
        """Pre-compute quantities that stay fixed across all MC draws."""
        s = self.s

        # half-life and mean decay constant
        halflife = _BE10_HALFLIFE if s.isotope == "Be-10" else _AL26_HALFLIFE
        self._halflife = halflife
        self._decay_const_mean = np.log(2) / halflife

        # muon production curves
        print("Fitting muon production curves …", flush=True)
        muon = fit_muon_curves(s.elevation, s.isotope, s.muon_fit_depth_m)
        self._muon = muon

        # coefficients for the five muon pathways:
        # v1 (attenuation lengths) = b_i; v2 (prod rates at surface) = a_i
        fc = muon["fast_coeff"]   # [a1, a2, b1, b2]
        nc = muon["neg_coeff"]    # [a1, a2, a3, b1, b2, b3]
        fbs = muon["fast_surface"]  # a1+a2
        nbs = muon["neg_surface"]   # a1+a2+a3
        tmu = muon["total_muon"]    # fbs + nbs

        # normalised attenuation lengths (b_i / surface_sum)
        self._mufast1 = fc[2] / fbs
        self._mufast2 = fc[3] / fbs
        self._muneg1 = nc[3] / nbs
        self._muneg2 = nc[4] / nbs
        self._muneg3 = nc[5] / nbs

        # normalised production-rate fractions
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

        # shielding factor (applied to spallation only; muon terms are unshielded)
        self._shielding = s.shielding_value

        # spallation surface production rate (mean of prior)
        scheme = s.production_scheme
        if scheme == "constant":
            # User provides the final shielded rate; shielding_value should be 1.0
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

        # update the production_error distribution centre for computed schemes
        if scheme != "constant" and s.production_error.mode == "normal":
            s._production_error.parameters[0] = spall_mean
        elif scheme != "constant" and s.production_error.mode == "uniform":
            half = (s._production_error.parameters[1] - s._production_error.parameters[0]) / 2.0
            s._production_error.parameters = [spall_mean - half, spall_mean + half]

        # print the effective rate actually used in the MC draws
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
        self._dof = max(1, self._n_depths)  # normalise by n, like the MATLAB code

        # cumulative bulk density (mean and std per sample) if from file
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
        mc = s
        if mc.mc_age.mode != "constant":
            self._age_bins = np.linspace(mc.mc_age.parameters[0],
                                          mc.mc_age.parameters[1], resolution)
        else:
            self._age_bins = np.array([mc.mc_age.parameters[0]])
        if mc.mc_erosion_rate.mode != "constant":
            lo, hi = mc.mc_erosion_rate.parameters
            self._erosion_bins = np.linspace(lo * 1e-3, hi * 1e-3, resolution)
        else:
            self._erosion_bins = np.array([mc.mc_erosion_rate.parameters[0] * 1e-3])
        if mc.mc_inheritance.mode != "constant":
            self._inh_bins = np.linspace(mc.mc_inheritance.parameters[0],
                                          mc.mc_inheritance.parameters[1], resolution)
        else:
            self._inh_bins = np.array([mc.mc_inheritance.parameters[0]])

        self._chi2_thresh = chi2_threshold(s.mc_confidence_mode, s.mc_confidence_value)

        print("Setup complete.", flush=True)

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
        erosion_thresh = s.mc_total_erosion_threshold  # [min_cm, max_cm]

        # pre-allocate accepted solution arrays
        chi2_arr = np.full(n_target, np.nan)
        age_arr = np.full(n_target, np.nan)
        erosion_arr = np.full(n_target, np.nan)
        inh_arr = np.full(n_target, np.nan)
        spall_arr = np.full(n_target, np.nan)
        muon_arr = np.full(n_target, np.nan)
        natt_arr = np.full(n_target, np.nan)
        decay_arr = np.full(n_target, np.nan)
        dens_arr = np.full((n_target, self._n_depths), np.nan)

        # Bayesian grid
        grid_sums = np.zeros((len(self._age_bins), len(self._erosion_bins), len(self._inh_bins)))
        grid_counts = np.zeros_like(grid_sums, dtype=int)

        j = 0          # accepted solutions found
        counter = 0    # total draws
        t0 = time.monotonic()

        print(f"Running {n_target} MC solutions …", flush=True)

        while j < n_target:
            counter += 1
            if counter > self.MAX_ITER and j == 0:
                raise RuntimeError(
                    f"No accepted solution after {self.MAX_ITER} iterations. "
                    "Check settings (erosion threshold, chi2 threshold, prior ranges)."
                )

            # ----- draw: age + erosion rate (enforce total-erosion constraint) -----
            total_erosion = erosion_thresh[0] - 1.0
            ctime = cerosion = 0.0
            inner = 0
            while total_erosion < erosion_thresh[0] or total_erosion > erosion_thresh[1]:
                ctime = s.mc_age.draw(rng)
                cerosion = s.mc_erosion_rate.draw(rng) * 1e-3   # cm/ka → cm/yr
                total_erosion = ctime * cerosion
                inner += 1
                if inner > 100000:
                    break  # give up; outer loop will discard

            # ----- draw remaining parameters -----
            cprodrate = s.production_error.draw(rng)
            cmuonrate = self._total_muon * (
                1.0 + rng.standard_normal() * s.muon_percent_error / 100.0
            )
            cnatt = s.mc_neutron_attenuation.draw(rng)
            cinh = s.mc_inheritance.draw(rng)

            # noisy muon attenuation lengths
            cfbs = self._fast_surface * (
                1.0 + rng.standard_normal() * self._fast_relerr
            )
            cnbs = self._neg_surface * (
                1.0 + rng.standard_normal() * self._neg_relerr
            )

            # decay constant with half-life uncertainty
            cdecay = np.log(2) / (
                self._halflife * (1.0 + rng.standard_normal() * s.half_life_sigma)
            )

            # density per sample
            if s.density_from_file:
                cdensity = rng.standard_normal(self._n_depths) * self._cbd_std + self._cbd_mean
            else:
                rho_scalar = s.density_error.draw(rng)
                cdensity = np.full(self._n_depths, rho_scalar)

            # ----- build v1 (attenuation lengths) and v2 (production rates) -----
            v1 = np.array([
                cnatt,
                cfbs * self._mufast1,
                cfbs * self._mufast2,
                cnbs * self._muneg1,
                cnbs * self._muneg2,
                cnbs * self._muneg3,
            ])
            v2 = np.array([
                cprodrate,
                cmuonrate * self._cfast1,
                cmuonrate * self._cfast2,
                cmuonrate * self._cneg1,
                cmuonrate * self._cneg2,
                cmuonrate * self._cneg3,
            ])

            # ----- evaluate forward model and chi² -----
            modelled = profile_concentration(
                {"depth": self._depths, "thickness": self._thicknesses},
                v1, v2, ctime, cdecay, cerosion, cinh, cdensity,
            )
            chi2 = chi2_profile(modelled, self._measured, self._rel_error, self._dof)

            # ----- update importance-sampling grid (all draws, before threshold) -----
            chi_weighted = np.exp(-chi2 / 2.0)
            xi = int(np.argmin(np.abs(self._age_bins - ctime)))
            yi = int(np.argmin(np.abs(self._erosion_bins - cerosion)))
            zi = int(np.argmin(np.abs(self._inh_bins - cinh)))
            grid_sums[xi, yi, zi] += chi_weighted
            grid_counts[xi, yi, zi] += 1

            # ----- accept / reject -----
            if chi2 > self._chi2_thresh:
                continue

            # store accepted solution
            chi2_arr[j] = chi2
            age_arr[j] = ctime
            erosion_arr[j] = cerosion
            inh_arr[j] = cinh
            spall_arr[j] = cprodrate
            muon_arr[j] = cmuonrate
            natt_arr[j] = cnatt
            decay_arr[j] = cdecay
            dens_arr[j, :] = cdensity

            j += 1
            if j % max(1, n_target // 20) == 0:
                elapsed = time.monotonic() - t0
                print(
                    f"  {j}/{n_target} solutions | {counter} draws | "
                    f"{elapsed:.1f}s",
                    flush=True,
                )

        elapsed = time.monotonic() - t0
        print(f"Done. {j} solutions in {counter} draws ({elapsed:.1f}s).", flush=True)

        # ----- Bayesian PDFs -----
        pdf_age, pdf_erosion, pdf_inh = bayesian_pdf(
            grid_sums, grid_counts,
            self._age_bins, self._erosion_bins, self._inh_bins,
        )

        best_age_ka = pdf_age.map_value / 1e3 if pdf_age else age_arr[0] / 1e3
        best_erosion = pdf_erosion.map_value * 1e3 if pdf_erosion else erosion_arr[0] * 1e3
        best_inh = pdf_inh.map_value / 1e4 if pdf_inh else inh_arr[0] / 1e4

        return Results(
            chi2=chi2_arr[:j],
            age=age_arr[:j],
            erosion_rate=erosion_arr[:j],
            inheritance=inh_arr[:j],
            spall_prod_rate=spall_arr[:j],
            muon_prod_rate=muon_arr[:j],
            neutron_attenuation=natt_arr[:j],
            decay_const=decay_arr[:j],
            densities=dens_arr[:j, :],
            n_accepted=j,
            n_iterations=counter,
            pdf_age=pdf_age,
            pdf_erosion=pdf_erosion,
            pdf_inheritance=pdf_inh,
            best_age_ka=best_age_ka,
            best_erosion_cm_ka=best_erosion,
            best_inheritance_1e4=best_inh,
            age_sigma2_plus_ka=pdf_age.sigma2_plus / 1e3 if pdf_age else float("nan"),
            age_sigma2_minus_ka=pdf_age.sigma2_minus / 1e3 if pdf_age else float("nan"),
            erosion_sigma2_plus_cm_ka=pdf_erosion.sigma2_plus * 1e3 if pdf_erosion else float("nan"),
            erosion_sigma2_minus_cm_ka=pdf_erosion.sigma2_minus * 1e3 if pdf_erosion else float("nan"),
        )
