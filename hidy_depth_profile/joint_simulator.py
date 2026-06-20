"""
Joint Monte Carlo simulator for ¹⁰Be depth profiles sharing an exposure age.

When two or more profile sites record the same exposure event (e.g. two pits
on the same terrace), the exposure age is shared while site-specific
parameters (inheritance, production rate, density, muon rate) are drawn
independently per profile.  The joint reduced chi² is the sum of raw squared
residuals across all profiles divided by the total number of data points.

Shared parameters (one draw per MC step):
    exposure age, surface-change rate, radioactive decay constant

Per-profile parameters (independent draws per step):
    pre-exposure inheritance, density, neutron-attenuation length,
    spallation production rate, muon production rate
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.special import ndtr as _ndtr

from ._numba import NUMBA_AVAILABLE, numba_forward_batch
from .inversion import BayesianPDF, bayesian_pdf, chi2_threshold
from .production import (fit_muon_curves, lsdn_rates_for_ages,
                         precompute_lsdn_timeseries, stone2000_surface_rate)
from .settings import ProfileSettings, _DistParam
from .simulator import _bin_indices

_BE10_HALFLIFE = 1.387e6
_AL26_HALFLIFE = 7.17e5
_MIN_BATCH = 500
_MAX_BATCH = 200_000
_TARGET_ACCEPTED_PER_BATCH = 100
_MU_EPS = 1e-30


# ---------------------------------------------------------------------------
# Forward model (numpy fallback, profile-specific depths)
# ---------------------------------------------------------------------------

def _forward_batch_numpy_joint(
    ages: np.ndarray,
    decay_consts: np.ndarray,
    erosions: np.ndarray,
    inheritances: np.ndarray,
    densities: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    depths: np.ndarray,
    thicknesses: np.ndarray,
) -> np.ndarray:
    """Pure-NumPy forward model for one profile with explicit depth arrays."""
    B = len(ages)
    n = len(depths)
    modelled = np.empty((B, n))
    inh_term = inheritances * np.exp(-ages * decay_consts)
    t_v = ages[:, None]
    lam_v = decay_consts[:, None]
    er_v = erosions[:, None]
    for k in range(n):
        z = depths[k]
        dz = thicknesses[k]
        rho_v = densities[:, k:k + 1]
        tmp1 = np.exp(-z * rho_v / v1)
        tmp2 = np.exp(-(z + dz) * rho_v / v1)
        simp = lam_v + er_v * rho_v / v1
        tmp3 = (v1 * v2) * (np.exp(-t_v * simp) - 1.0) / (rho_v * simp)
        modelled[:, k] = np.sum((tmp2 - tmp1) * tmp3, axis=1) / dz + inh_term
    return modelled


# ---------------------------------------------------------------------------
# 1-D marginal PDF helper
# ---------------------------------------------------------------------------

def _make_1d_pdf(
    bins: np.ndarray,
    grid_sums: np.ndarray,
    grid_counts: np.ndarray,
) -> Optional[BayesianPDF]:
    """Build a 1-D marginal BayesianPDF from an importance-sampling accumulator."""
    if len(bins) <= 1:
        return None
    grid = grid_sums.copy()
    valid = grid_counts > 0
    grid[valid] /= grid_counts[valid]
    total = grid.sum()
    if total == 0:
        return None
    step = bins[1] - bins[0]
    probs = grid / (total * step)
    cdf = np.cumsum(probs) / probs.sum()
    map_val = float(bins[np.argmax(probs)])

    def _q(q: float) -> float:
        cdf_u, idx = np.unique(cdf, return_index=True)
        b_u = bins[idx]
        if q < cdf_u[0] or q > cdf_u[-1]:
            return float("nan")
        return float(np.interp(q, cdf_u, b_u))

    return BayesianPDF(
        bins=bins, pdf=probs, cdf=cdf, map_value=map_val,
        sigma2_plus=_q(0.977), sigma2_minus=_q(0.023),
    )


# ---------------------------------------------------------------------------
# Per-profile precomputed quantities
# ---------------------------------------------------------------------------

@dataclass
class _ProfileSetup:
    """Pre-computed, per-profile quantities for the joint forward model."""
    name: str
    # Muon scale factors
    mufast1: float
    mufast2: float
    muneg1: float
    muneg2: float
    muneg3: float
    cfast1: float
    cfast2: float
    cneg1: float
    cneg2: float
    cneg3: float
    fast_surface: float
    neg_surface: float
    total_muon: float
    fast_relerr: float
    neg_relerr: float
    shielding: float
    # Production
    lsdn_ts: object          # precomputed time series or None
    spall_mean: float
    production_scheme: str
    production_error: _DistParam   # final, with mean updated for non-constant schemes
    lsdn_production_error_frac: float
    # Profile data
    depths: np.ndarray
    thicknesses: np.ndarray
    measured: np.ndarray
    rel_error: np.ndarray
    n_depths: int
    z_min: float
    # Density
    density_from_file: bool
    cbd_mean: object         # np.ndarray or None
    cbd_std: object          # np.ndarray or None
    density_error: _DistParam
    # Per-profile MC distributions
    mc_inheritance: _DistParam
    mc_neutron_attenuation: _DistParam
    muon_percent_error: float
    # Deposition / erosion threshold (cm total change, profile-specific)
    edt_lo: float
    edt_hi: float
    # Inheritance grid (set by JointSimulator._setup after resolution is known)
    inh_bins: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class JointResults:
    """
    Accepted solutions from a joint MC inversion with a shared exposure age.

    Shared arrays (length n_accepted): age, erosion rate, decay constant, chi².
    Per-profile dicts (keys = profile names): inheritance, densities, etc.
    """
    chi2: np.ndarray
    age: np.ndarray                        # yr
    erosion_deposition_rate: np.ndarray    # cm/yr
    decay_const: np.ndarray               # yr⁻¹

    inheritance: Dict[str, np.ndarray]     # atoms/g
    spall_prod_rate: Dict[str, np.ndarray]
    muon_prod_rate: Dict[str, np.ndarray]
    neutron_attenuation: Dict[str, np.ndarray]
    densities: Dict[str, np.ndarray]       # (n_accepted, n_depths_i)

    n_accepted: int
    n_iterations: int

    pdf_age: Optional[BayesianPDF]
    pdf_erosion_deposition: Optional[BayesianPDF]
    pdf_inheritance: Dict[str, Optional[BayesianPDF]]

    best_age_ka: float = 0.0
    best_erosion_deposition_cm_ka: float = 0.0
    best_inheritance_1e4: Dict[str, float] = field(default_factory=dict)

    age_sigma2_plus_ka: float = float("nan")
    age_sigma2_minus_ka: float = float("nan")
    erosion_deposition_sigma2_plus_cm_ka: float = float("nan")
    erosion_deposition_sigma2_minus_cm_ka: float = float("nan")


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class JointSimulator:
    """
    Joint Monte Carlo inversion for a group of ¹⁰Be profiles sharing an age.

    All profiles must agree on isotope, mc_age, mc_erosion_deposition_rate,
    mc_confidence_mode/value, and mc_n_solutions.  Site-specific settings
    (elevation, production scheme, density, shielding, inheritance) may differ.

    Geochronological constraints (age_max/min/estimate_constraint) are taken
    from any profile that has them set; the first non-None value wins per type.

    Parameters
    ----------
    profiles : dict mapping surface name → ProfileSettings (insertion order
        determines iteration order).
    """

    MAX_TOTAL_ITER = 10_000_000

    def __init__(self, profiles: Dict[str, ProfileSettings]):
        if len(profiles) < 2:
            raise ValueError("JointSimulator requires at least two profiles.")
        self._names: List[str] = list(profiles.keys())
        self._settings: Dict[str, ProfileSettings] = profiles
        for s in profiles.values():
            s.validate()
        self._validate_shared()
        self._setup()

    # ------------------------------------------------------------------
    def _validate_shared(self) -> None:
        first = self._settings[self._names[0]]
        for name in self._names[1:]:
            s = self._settings[name]
            if s.isotope != first.isotope:
                raise ValueError(
                    f"Profile {name!r}: isotope {s.isotope!r} != {first.isotope!r}"
                )
            if (s.mc_age.mode != first.mc_age.mode or
                    list(s.mc_age.parameters) != list(first.mc_age.parameters)):
                raise ValueError(
                    f"Profile {name!r}: mc_age must match across all profiles in a joint group"
                )
            if (s.mc_erosion_deposition_rate.mode != first.mc_erosion_deposition_rate.mode or
                    list(s.mc_erosion_deposition_rate.parameters) !=
                    list(first.mc_erosion_deposition_rate.parameters)):
                raise ValueError(
                    f"Profile {name!r}: mc_erosion_deposition_rate must match across profiles"
                )
            if s.mc_confidence_mode != first.mc_confidence_mode:
                raise ValueError(
                    f"Profile {name!r}: mc_confidence_mode must match"
                )
            if s.mc_confidence_value != first.mc_confidence_value:
                raise ValueError(
                    f"Profile {name!r}: mc_confidence_value must match"
                )
            if s.mc_n_solutions != first.mc_n_solutions:
                raise ValueError(
                    f"Profile {name!r}: mc_n_solutions must match"
                )

    # ------------------------------------------------------------------
    def _setup(self) -> None:
        s0 = self._settings[self._names[0]]
        self._halflife = _BE10_HALFLIFE if s0.isotope == "Be-10" else _AL26_HALFLIFE
        self._half_life_sigma = s0.half_life_sigma
        self._n_solutions = s0.mc_n_solutions
        self._chi2_thresh = chi2_threshold(s0.mc_confidence_mode, s0.mc_confidence_value)

        # Shared erosion/deposition threshold: intersection across all profiles
        self._edt_lo_shared = max(
            float(s.mc_erosion_deposition_threshold[0])
            for s in self._settings.values()
        )
        self._edt_hi_shared = min(
            float(s.mc_erosion_deposition_threshold[1])
            for s in self._settings.values()
        )

        # Geochronological constraints: first non-None per type; all extra max bounds collected
        self._age_max_constraint: Optional[_DistParam] = None
        self._age_max_constraints_extra: list = []   # all age_max_constraints lists merged
        self._age_min_constraint: Optional[_DistParam] = None
        self._age_estimate_constraint: Optional[_DistParam] = None
        for s in self._settings.values():
            if self._age_max_constraint is None:
                self._age_max_constraint = s.age_max_constraint
            elif s.age_max_constraint is not None:
                self._age_max_constraints_extra.append(s.age_max_constraint)
            self._age_max_constraints_extra.extend(s.age_max_constraints)
            if self._age_min_constraint is None:
                self._age_min_constraint = s.age_min_constraint
            if self._age_estimate_constraint is None:
                self._age_estimate_constraint = s.age_estimate_constraint

        # Effective age draw range (clamped by hard constant constraints)
        self._eff_age_lo: Optional[float] = None
        self._eff_age_hi: Optional[float] = None
        if s0.mc_age.mode == "uniform":
            lo = float(s0.mc_age.parameters[0])
            hi = float(s0.mc_age.parameters[1])
            if self._age_max_constraint is not None and self._age_max_constraint.mode == "constant":
                hi = min(hi, float(self._age_max_constraint.parameters[0]))
            for _c in self._age_max_constraints_extra:
                if _c.mode == "constant":
                    hi = min(hi, float(_c.parameters[0]))
            if self._age_min_constraint is not None and self._age_min_constraint.mode == "constant":
                lo = max(lo, float(self._age_min_constraint.parameters[0]))
            if lo >= hi:
                raise ValueError(
                    f"Age constraints leave no valid draw range: [{lo:.0f}, {hi:.0f}] yr"
                )
            self._eff_age_lo, self._eff_age_hi = lo, hi

        if (self._age_max_constraint is not None or
                self._age_max_constraints_extra or
                self._age_min_constraint is not None or
                self._age_estimate_constraint is not None):
            print("  Age constraints (shared):", flush=True)
            if self._age_max_constraint is not None:
                c = self._age_max_constraint
                if c.mode == "constant":
                    print(f"    max age: hard bound at {c.parameters[0]/1e3:.2f} ka", flush=True)
                else:
                    print(
                        f"    max age: {c.parameters[0]/1e3:.2f} ± {c.parameters[1]/1e3:.2f} ka "
                        f"(1σ, one-sided)", flush=True,
                    )
            for _c in self._age_max_constraints_extra:
                if _c.mode == "constant":
                    print(f"    max age (additional): hard bound at {_c.parameters[0]/1e3:.2f} ka", flush=True)
                else:
                    print(
                        f"    max age (additional): {_c.parameters[0]/1e3:.2f} ± {_c.parameters[1]/1e3:.2f} ka "
                        f"(1σ, one-sided)", flush=True,
                    )
            if self._age_min_constraint is not None:
                c = self._age_min_constraint
                if c.mode == "constant":
                    print(f"    min age: hard bound at {c.parameters[0]/1e3:.2f} ka", flush=True)
                else:
                    print(
                        f"    min age: {c.parameters[0]/1e3:.2f} ± {c.parameters[1]/1e3:.2f} ka "
                        f"(1σ, one-sided)", flush=True,
                    )
            if self._age_estimate_constraint is not None:
                c = self._age_estimate_constraint
                print(
                    f"    age estimate: {c.parameters[0]/1e3:.2f} ± {c.parameters[1]/1e3:.2f} ka "
                    f"(1σ, bilateral)", flush=True,
                )
            if self._eff_age_lo is not None:
                print(
                    f"    effective draw range: "
                    f"{self._eff_age_lo/1e3:.2f}–{self._eff_age_hi/1e3:.2f} ka",
                    flush=True,
                )

        # Shared Bayesian grid bins (age, erosion)
        resolution = max(2, round(2 * self._n_solutions ** (1 / 3)))
        if s0.mc_age.mode != "constant":
            age_lo = self._eff_age_lo if self._eff_age_lo is not None else float(s0.mc_age.parameters[0])
            age_hi = self._eff_age_hi if self._eff_age_hi is not None else float(s0.mc_age.parameters[1])
            self._age_bins = np.linspace(age_lo, age_hi, resolution)
        else:
            self._age_bins = np.array([float(s0.mc_age.parameters[0])])

        if s0.mc_erosion_deposition_rate.mode != "constant":
            lo, hi = s0.mc_erosion_deposition_rate.parameters
            self._er_bins = np.linspace(float(lo) * 1e-3, float(hi) * 1e-3, resolution)
        else:
            self._er_bins = np.array([float(s0.mc_erosion_deposition_rate.parameters[0]) * 1e-3])

        # Per-profile setup
        self._profiles: List[_ProfileSetup] = []
        for name, s in self._settings.items():
            ps = self._setup_profile(name, s, resolution)
            self._profiles.append(ps)

        self._total_n_depths = sum(p.n_depths for p in self._profiles)

        # Warn about deposition threshold vs shallowest sample
        for p in self._profiles:
            if self._edt_lo_shared < -p.z_min:
                print(
                    f"  Warning: [{p.name}] edt_lo ({self._edt_lo_shared:.1f} cm) may imply more "
                    f"deposition than the shallowest sample ({p.z_min:.1f} cm). "
                    f"The shallowness constraint will be applied per-profile.",
                    flush=True,
                )

        print("Setup complete.", flush=True)

    # ------------------------------------------------------------------
    def _setup_profile(
        self, name: str, s: ProfileSettings, resolution: int
    ) -> _ProfileSetup:
        """Pre-compute quantities that are fixed across all MC draws for one profile."""
        print(f"  [{name}] Fitting muon production curves …", flush=True)
        muon = fit_muon_curves(s.elevation, s.isotope, s.muon_fit_depth_m)
        fc = muon["fast_coeff"]
        nc = muon["neg_coeff"]
        fbs = muon["fast_surface"]
        nbs = muon["neg_surface"]
        tmu = muon["total_muon"]

        shielding = s.shielding_value
        scheme = s.production_scheme
        lsdn_ts = None

        if scheme == "constant":
            spall_mean = s.constant_rate
        elif scheme == "stone2000":
            raw = stone2000_surface_rate(
                s.latitude, s.longitude, s.elevation, s.reference_rate, s.isotope
            )
            spall_mean = raw * shielding
        else:  # lsdn
            if s.mc_age.mode == "uniform":
                t_max = self._eff_age_hi if self._eff_age_hi is not None else float(s.mc_age.parameters[1])
            elif s.mc_age.mode == "normal":
                t_max = float(s.mc_age.parameters[0]) + 4.0 * float(s.mc_age.parameters[1])
            else:
                t_max = float(s.mc_age.parameters[0])
            print(f"  [{name}] Precomputing LSDn paleomagnetic time series …", flush=True)
            lsdn_ts = precompute_lsdn_timeseries(
                s.latitude, s.longitude, s.elevation,
                t_max, s.collection_year, s.isotope,
            )
            central_age = float(np.mean(s.mc_age.parameters))
            spall_mean = float(lsdn_rates_for_ages(np.array([central_age]), lsdn_ts)[0]) * shielding

        # Final production error distribution (update mean for non-constant schemes)
        prod_err = _DistParam(s.production_error.mode, list(s.production_error.parameters))
        if scheme != "constant":
            if prod_err.mode == "normal":
                prod_err.parameters[0] = spall_mean
            elif prod_err.mode == "uniform":
                half = (prod_err.parameters[1] - prod_err.parameters[0]) / 2.0
                prod_err.parameters = [spall_mean - half, spall_mean + half]

        print(f"  [{name}] Spallation rate: {spall_mean:.4f} at/g/yr", flush=True)

        pd = s.profile_data
        depths = pd["depth"]
        thicknesses = pd["thickness"]
        measured = pd["concentration"]
        rel_error = pd["rel_error"]
        n_depths = len(depths)

        cbd_mean = cbd_std = None
        if s.density_from_file:
            from .io import cumulative_bulk_density
            cbd_mean, cbd_std = cumulative_bulk_density(
                s.density_data, depths, thicknesses
            )

        # Inheritance grid bins
        if s.mc_inheritance.mode != "constant":
            inh_bins = np.linspace(
                float(s.mc_inheritance.parameters[0]),
                float(s.mc_inheritance.parameters[1]),
                resolution,
            )
        else:
            inh_bins = np.array([float(s.mc_inheritance.parameters[0])])

        return _ProfileSetup(
            name=name,
            mufast1=fc[2] / fbs, mufast2=fc[3] / fbs,
            muneg1=nc[3] / nbs, muneg2=nc[4] / nbs, muneg3=nc[5] / nbs,
            cfast1=fc[0] / tmu, cfast2=fc[1] / tmu,
            cneg1=nc[0] / tmu, cneg2=nc[1] / tmu, cneg3=nc[2] / tmu,
            fast_surface=fbs, neg_surface=nbs, total_muon=tmu,
            fast_relerr=muon["fast_relerr"], neg_relerr=muon["neg_relerr"],
            shielding=shielding,
            lsdn_ts=lsdn_ts,
            spall_mean=spall_mean,
            production_scheme=scheme,
            production_error=prod_err,
            lsdn_production_error_frac=s.lsdn_production_error_frac,
            depths=depths, thicknesses=thicknesses,
            measured=measured, rel_error=rel_error,
            n_depths=n_depths, z_min=float(np.min(depths)),
            density_from_file=s.density_from_file,
            cbd_mean=cbd_mean, cbd_std=cbd_std,
            density_error=s.density_error,
            mc_inheritance=s.mc_inheritance,
            mc_neutron_attenuation=s.mc_neutron_attenuation,
            muon_percent_error=s.muon_percent_error,
            edt_lo=float(s.mc_erosion_deposition_threshold[0]),
            edt_hi=float(s.mc_erosion_deposition_threshold[1]),
            inh_bins=inh_bins,
        )

    # ------------------------------------------------------------------
    def run(self, seed: Optional[int] = None) -> JointResults:
        """Execute the joint Monte Carlo simulation and return JointResults."""
        rng = np.random.default_rng(seed)
        s0 = self._settings[self._names[0]]
        n_target = self._n_solutions
        total_n_depths = self._total_n_depths

        # Pre-allocate shared accepted arrays
        chi2_acc = np.empty(n_target)
        age_acc = np.empty(n_target)
        er_acc = np.empty(n_target)
        decay_acc = np.empty(n_target)

        # Pre-allocate per-profile accepted arrays
        inh_acc   = {p.name: np.empty(n_target)               for p in self._profiles}
        spall_acc = {p.name: np.empty(n_target)               for p in self._profiles}
        muon_acc  = {p.name: np.empty(n_target)               for p in self._profiles}
        natt_acc  = {p.name: np.empty(n_target)               for p in self._profiles}
        dens_acc  = {p.name: np.empty((n_target, p.n_depths)) for p in self._profiles}

        # Shared importance-sampling grid (age, erosion, 1)
        # The singleton third dimension lets us reuse bayesian_pdf() unchanged.
        n_age = len(self._age_bins)
        n_er = len(self._er_bins)
        grid_sums   = np.zeros((n_age, n_er, 1))
        grid_counts = np.zeros((n_age, n_er, 1), dtype=np.intp)

        # Per-profile 1-D inheritance grids
        inh_gsums   = {p.name: np.zeros(len(p.inh_bins)) for p in self._profiles}
        inh_gcounts = {p.name: np.zeros(len(p.inh_bins), dtype=np.intp) for p in self._profiles}

        j = 0
        counter = 0
        batch_size = 10_000
        t0 = time.monotonic()

        print(
            f"Running joint MC ({', '.join(self._names)}) for {n_target} solutions …",
            flush=True,
        )

        while j < n_target:
            if counter > self.MAX_TOTAL_ITER and j == 0:
                raise RuntimeError(
                    f"No accepted solution after {self.MAX_TOTAL_ITER} iterations. "
                    "Check settings."
                )

            # ---- draw shared parameters ----
            if self._eff_age_lo is not None:
                ages_raw = rng.uniform(self._eff_age_lo, self._eff_age_hi, batch_size)
            else:
                ages_raw = s0.mc_age.draw_batch(rng, batch_size)
            erosions_raw = s0.mc_erosion_deposition_rate.draw_batch(rng, batch_size) * 1e-3

            total_change = ages_raw * erosions_raw
            valid = (total_change >= self._edt_lo_shared) & (total_change <= self._edt_hi_shared)
            for p in self._profiles:
                valid &= total_change >= -p.z_min

            ages    = ages_raw[valid]
            erosions = erosions_raw[valid]
            B = len(ages)
            counter += batch_size

            if B == 0:
                continue

            decay_consts = np.log(2.0) / (
                self._halflife * (1.0 + rng.standard_normal(B) * self._half_life_sigma)
            )

            # ---- per-profile draws and chi² ----
            raw_chi2_total = np.zeros(B)
            per_draw: Dict[str, dict] = {}

            for p in self._profiles:
                inheritances = p.mc_inheritance.draw_batch(rng, B)

                if p.lsdn_ts is not None:
                    prod_rates = lsdn_rates_for_ages(ages, p.lsdn_ts) * p.shielding
                    if p.lsdn_production_error_frac > 0:
                        prod_rates *= 1.0 + rng.standard_normal(B) * p.lsdn_production_error_frac
                else:
                    prod_rates = p.production_error.draw_batch(rng, B)

                muon_rates = p.total_muon * (
                    1.0 + rng.standard_normal(B) * (p.muon_percent_error / 100.0)
                )
                natts = p.mc_neutron_attenuation.draw_batch(rng, B)
                cfbs = np.maximum(
                    _MU_EPS,
                    p.fast_surface * (1.0 + rng.standard_normal(B) * p.fast_relerr),
                )
                cnbs = np.maximum(
                    _MU_EPS,
                    p.neg_surface * (1.0 + rng.standard_normal(B) * p.neg_relerr),
                )

                if p.density_from_file:
                    densities = rng.standard_normal((B, p.n_depths)) * p.cbd_std + p.cbd_mean
                else:
                    rho = p.density_error.draw_batch(rng, B)
                    densities = np.outer(rho, np.ones(p.n_depths))

                v1 = np.column_stack([
                    natts,
                    cfbs * p.mufast1, cfbs * p.mufast2,
                    cnbs * p.muneg1,  cnbs * p.muneg2,  cnbs * p.muneg3,
                ])
                v2 = np.column_stack([
                    prod_rates,
                    muon_rates * p.cfast1, muon_rates * p.cfast2,
                    muon_rates * p.cneg1,  muon_rates * p.cneg2,  muon_rates * p.cneg3,
                ])

                if NUMBA_AVAILABLE:
                    modelled = numba_forward_batch(
                        ages, decay_consts, erosions, inheritances, densities,
                        v1, v2, p.depths, p.thicknesses,
                    )
                else:
                    modelled = _forward_batch_numpy_joint(
                        ages, decay_consts, erosions, inheritances, densities,
                        v1, v2, p.depths, p.thicknesses,
                    )

                residuals = (modelled - p.measured) / (p.measured * p.rel_error)
                raw_chi2_total += np.sum(residuals ** 2, axis=1)

                per_draw[p.name] = dict(
                    inheritances=inheritances,
                    prod_rates=prod_rates,
                    muon_rates=muon_rates,
                    natts=natts,
                    densities=densities,
                )

            chi2_joint = raw_chi2_total / total_n_depths

            # ---- geochronological constraint weights (on shared age) ----
            max_c = self._age_max_constraint
            max_cs = self._age_max_constraints_extra
            min_c = self._age_min_constraint
            est_c = self._age_estimate_constraint
            has_soft = (
                (max_c is not None and max_c.mode == "normal") or
                any(_c.mode == "normal" for _c in max_cs) or
                (min_c is not None and min_c.mode == "normal") or
                est_c is not None
            )
            has_hard_non_uniform = self._eff_age_lo is None and (
                (max_c is not None and max_c.mode == "constant") or
                any(_c.mode == "constant" for _c in max_cs) or
                (min_c is not None and min_c.mode == "constant")
            )
            if has_soft or has_hard_non_uniform:
                constraint_prob = np.ones(B, dtype=float)
                if max_c is not None:
                    if max_c.mode == "normal":
                        mu, sigma = max_c.parameters
                        constraint_prob *= _ndtr((mu - ages) / sigma)
                    elif self._eff_age_lo is None:
                        constraint_prob *= (ages <= float(max_c.parameters[0])).astype(float)
                for _c in max_cs:
                    if _c.mode == "normal":
                        mu, sigma = _c.parameters
                        constraint_prob *= _ndtr((mu - ages) / sigma)
                    elif self._eff_age_lo is None:
                        constraint_prob *= (ages <= float(_c.parameters[0])).astype(float)
                if min_c is not None:
                    if min_c.mode == "normal":
                        mu, sigma = min_c.parameters
                        constraint_prob *= _ndtr((ages - mu) / sigma)
                    elif self._eff_age_lo is None:
                        constraint_prob *= (ages >= float(min_c.parameters[0])).astype(float)
                if est_c is not None:
                    mu, sigma = est_c.parameters
                    constraint_prob *= np.exp(-0.5 * ((ages - mu) / sigma) ** 2)
            else:
                constraint_prob = None

            # ---- update shared importance-sampling grid ----
            chi_weighted = np.exp(-chi2_joint / 2.0)
            if constraint_prob is not None:
                chi_weighted = chi_weighted * constraint_prob

            xi = _bin_indices(ages, self._age_bins)
            yi = _bin_indices(erosions, self._er_bins)
            np.add.at(grid_sums[:, :, 0],   (xi, yi), chi_weighted)
            np.add.at(grid_counts[:, :, 0], (xi, yi), 1)

            # ---- update per-profile inheritance grids ----
            for p in self._profiles:
                zi = _bin_indices(per_draw[p.name]["inheritances"], p.inh_bins)
                np.add.at(inh_gsums[p.name],   zi, chi_weighted)
                np.add.at(inh_gcounts[p.name], zi, 1)

            # ---- accept/reject ----
            accepted = chi2_joint <= self._chi2_thresh
            if constraint_prob is not None:
                accepted = accepted & (rng.uniform(size=B) < constraint_prob)

            n_new = int(accepted.sum())
            if n_new > 0:
                take = min(n_new, n_target - j)
                idx_acc = np.where(accepted)[0][:take]
                chi2_acc[j:j + take]  = chi2_joint[idx_acc]
                age_acc[j:j + take]   = ages[idx_acc]
                er_acc[j:j + take]    = erosions[idx_acc]
                decay_acc[j:j + take] = decay_consts[idx_acc]
                for p in self._profiles:
                    d = per_draw[p.name]
                    inh_acc[p.name][j:j + take]        = d["inheritances"][idx_acc]
                    spall_acc[p.name][j:j + take]      = d["prod_rates"][idx_acc]
                    muon_acc[p.name][j:j + take]       = d["muon_rates"][idx_acc]
                    natt_acc[p.name][j:j + take]       = d["natts"][idx_acc]
                    dens_acc[p.name][j:j + take]       = d["densities"][idx_acc]
                j += take

            # ---- adapt batch size ----
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
        inh_dummy = np.array([0.0])
        pdf_age, pdf_ed, _ = bayesian_pdf(
            grid_sums, grid_counts, self._age_bins, self._er_bins, inh_dummy
        )
        pdf_inh = {
            p.name: _make_1d_pdf(p.inh_bins, inh_gsums[p.name], inh_gcounts[p.name])
            for p in self._profiles
        }

        best_age_ka = pdf_age.map_value / 1e3 if pdf_age else age_acc[0] / 1e3
        best_ed     = pdf_ed.map_value * 1e3  if pdf_ed  else er_acc[0] * 1e3
        best_inh_1e4 = {
            p.name: (pdf_inh[p.name].map_value / 1e4 if pdf_inh[p.name] else inh_acc[p.name][0] / 1e4)
            for p in self._profiles
        }

        return JointResults(
            chi2=chi2_acc[:j],
            age=age_acc[:j],
            erosion_deposition_rate=er_acc[:j],
            decay_const=decay_acc[:j],
            inheritance={p.name: inh_acc[p.name][:j]   for p in self._profiles},
            spall_prod_rate={p.name: spall_acc[p.name][:j] for p in self._profiles},
            muon_prod_rate={p.name: muon_acc[p.name][:j]   for p in self._profiles},
            neutron_attenuation={p.name: natt_acc[p.name][:j] for p in self._profiles},
            densities={p.name: dens_acc[p.name][:j]     for p in self._profiles},
            n_accepted=j,
            n_iterations=counter,
            pdf_age=pdf_age,
            pdf_erosion_deposition=pdf_ed,
            pdf_inheritance=pdf_inh,
            best_age_ka=best_age_ka,
            best_erosion_deposition_cm_ka=best_ed,
            best_inheritance_1e4=best_inh_1e4,
            age_sigma2_plus_ka=pdf_age.sigma2_plus / 1e3 if pdf_age else float("nan"),
            age_sigma2_minus_ka=pdf_age.sigma2_minus / 1e3 if pdf_age else float("nan"),
            erosion_deposition_sigma2_plus_cm_ka=pdf_ed.sigma2_plus * 1e3 if pdf_ed else float("nan"),
            erosion_deposition_sigma2_minus_cm_ka=pdf_ed.sigma2_minus * 1e3 if pdf_ed else float("nan"),
        )
