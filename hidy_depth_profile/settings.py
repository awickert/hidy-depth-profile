"""
ProfileSettings: validated container for all simulation parameters.

Maps to the YAML schema; every attribute has a property setter that validates
immediately on assignment.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


_VALID_PRODUCTION_SCHEMES = {"stone2000", "lsdn", "constant"}
_VALID_ISOTOPES = {"Be-10", "Al-26"}
_VALID_DIST_MODES = {"constant", "uniform", "normal"}
_VALID_CONFIDENCE_MODES = {"sigma", "chi2"}


@dataclass
class _DistParam:
    """A distribution specification: mode + parameters."""
    mode: str
    parameters: list

    def validate(self, name: str):
        if self.mode not in _VALID_DIST_MODES:
            raise ValueError(f"{name}.mode must be one of {_VALID_DIST_MODES}")
        if self.mode == "constant" and len(self.parameters) != 1:
            raise ValueError(f"{name}: constant mode requires exactly 1 parameter")
        if self.mode in ("uniform", "normal") and len(self.parameters) != 2:
            raise ValueError(f"{name}: {self.mode} mode requires exactly 2 parameters")
        if self.mode == "uniform" and self.parameters[0] > self.parameters[1]:
            raise ValueError(f"{name}: uniform [min, max] must have min ≤ max")

    def draw(self, rng: np.random.Generator) -> float:
        if self.mode == "constant":
            return float(self.parameters[0])
        if self.mode == "uniform":
            lo, hi = self.parameters
            return float(rng.uniform(lo, hi))
        # normal
        mu, sigma = self.parameters
        return float(rng.normal(mu, sigma))

    def draw_batch(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Draw n samples as a 1-D NumPy array."""
        if self.mode == "constant":
            return np.full(n, self.parameters[0])
        if self.mode == "uniform":
            lo, hi = self.parameters
            return rng.uniform(lo, hi, size=n)
        # normal
        mu, sigma = self.parameters
        return rng.normal(mu, sigma, size=n)


class ProfileSettings:
    """
    All parameters needed to run a ¹⁰Be depth-profile MC simulation.

    Construct from a YAML file with ProfileSettings.from_yaml(), or build
    programmatically and call validate() before running.
    """

    # ------------------------------------------------------------------ init
    def __init__(self):
        # site
        self._latitude: float = 0.0
        self._longitude: float = 0.0
        self._elevation: float = 0.0
        self._strike: float = 0.0
        self._dip: float = 0.0
        self._site_name: str = ""

        # data files / loaded data
        self._profile_file: Optional[str] = None
        self._shielding_file: Optional[str] = None
        self._density_file: Optional[str] = None
        self.profile_data: Optional[dict] = None
        self.shielding_data = None  # array or scalar
        self.density_data: Optional[dict] = None

        # production
        self._isotope: str = "Be-10"
        self._production_scheme: str = "stone2000"
        self._reference_rate: float = 4.086
        self._constant_rate: float = 5.5
        self._lsdn_assumed_age_yr: int = 15000
        self._lsdn_iterate: bool = False
        self._lsdn_production_error_frac: float = 0.0
        self._collection_year: int = 2024
        self._production_error: _DistParam = _DistParam("normal", [4.086, 0.2])
        self._half_life_sigma: float = 0.0

        # muons
        self._muon_fit_depth_m: float = 2000.0
        self._muon_percent_error: float = 5.0

        # density
        self._density_from_file: bool = False
        self._density_error: _DistParam = _DistParam("uniform", [2.2, 2.5])

        # Monte Carlo
        self._mc_n_solutions: int = 1000
        self._mc_age: _DistParam = _DistParam("normal", [15000.0, 5000.0])
        self._mc_erosion_deposition_rate: _DistParam = _DistParam("uniform", [0.0, 5.0])
        self._mc_inheritance: _DistParam = _DistParam("uniform", [0.0, 50000.0])
        self._mc_neutron_attenuation: _DistParam = _DistParam("normal", [160.0, 5.0])
        self._mc_erosion_deposition_threshold: list = [0.0, 50.0]
        self._mc_confidence_mode: str = "sigma"
        self._mc_confidence_value: float = 2.0
        self._mc_n_workers: int = 4

        # shielding
        self._shielding_from_file: bool = False
        self._shielding_value: float = 1.0

        # geochronological age constraints
        self._age_max_constraint: Optional[_DistParam] = None
        self._age_min_constraint: Optional[_DistParam] = None

    # ------------------------------------------------------------------ site
    @property
    def site_name(self): return self._site_name
    @site_name.setter
    def site_name(self, v): self._site_name = str(v)

    @property
    def latitude(self): return self._latitude
    @latitude.setter
    def latitude(self, v):
        v = float(v)
        if not -90 <= v <= 90:
            raise ValueError(f"latitude must be -90–90, got {v}")
        self._latitude = v

    @property
    def longitude(self): return self._longitude
    @longitude.setter
    def longitude(self, v):
        v = float(v)
        if not -180 <= v <= 180:
            raise ValueError(f"longitude must be -180–180, got {v}")
        self._longitude = v

    @property
    def elevation(self): return self._elevation
    @elevation.setter
    def elevation(self, v):
        v = float(v)
        if v < -500:
            raise ValueError(f"elevation must be > -500 m, got {v}")
        self._elevation = v

    @property
    def strike(self): return self._strike
    @strike.setter
    def strike(self, v): self._strike = float(v) % 360.0

    @property
    def dip(self): return self._dip
    @dip.setter
    def dip(self, v):
        v = float(v)
        if not 0 <= v <= 90:
            raise ValueError(f"dip must be 0–90, got {v}")
        self._dip = v

    # -------------------------------------------------------------- data files
    @property
    def profile_file(self): return self._profile_file
    @profile_file.setter
    def profile_file(self, v): self._profile_file = str(v) if v is not None else None

    @property
    def shielding_file(self): return self._shielding_file
    @shielding_file.setter
    def shielding_file(self, v): self._shielding_file = str(v) if v is not None else None

    @property
    def density_file(self): return self._density_file
    @density_file.setter
    def density_file(self, v): self._density_file = str(v) if v is not None else None

    def load_profile(self, filename: str):
        from .io import read_profile_data
        self._profile_file = filename
        self.profile_data = read_profile_data(filename)

    def load_shielding(self, filename: str):
        from .io import read_shielding_data
        from .production import shielding_factor
        self._shielding_file = filename
        raw = read_shielding_data(filename)
        self.shielding_data = raw
        self._shielding_from_file = True
        self._shielding_value = shielding_factor(raw, self._strike, self._dip)

    def load_density(self, filename: str):
        from .io import read_density_data
        self._density_file = filename
        self.density_data = read_density_data(filename)
        self._density_from_file = True

    # ------------------------------------------------------------ production
    @property
    def isotope(self): return self._isotope
    @isotope.setter
    def isotope(self, v):
        if v not in _VALID_ISOTOPES:
            raise ValueError(f"isotope must be one of {_VALID_ISOTOPES}")
        self._isotope = v

    @property
    def production_scheme(self): return self._production_scheme
    @production_scheme.setter
    def production_scheme(self, v):
        if v not in _VALID_PRODUCTION_SCHEMES:
            raise ValueError(f"production_scheme must be one of {_VALID_PRODUCTION_SCHEMES}")
        self._production_scheme = v

    @property
    def reference_rate(self): return self._reference_rate
    @reference_rate.setter
    def reference_rate(self, v):
        v = float(v)
        if v <= 0:
            raise ValueError("reference_rate must be positive")
        self._reference_rate = v

    @property
    def constant_rate(self): return self._constant_rate
    @constant_rate.setter
    def constant_rate(self, v):
        v = float(v)
        if v <= 0:
            raise ValueError("constant_rate must be positive")
        self._constant_rate = v

    @property
    def lsdn_assumed_age_yr(self): return self._lsdn_assumed_age_yr
    @lsdn_assumed_age_yr.setter
    def lsdn_assumed_age_yr(self, v):
        v = int(v)
        if v <= 0:
            raise ValueError("lsdn_assumed_age_yr must be positive")
        self._lsdn_assumed_age_yr = v

    @property
    def lsdn_iterate(self): return self._lsdn_iterate
    @lsdn_iterate.setter
    def lsdn_iterate(self, v): self._lsdn_iterate = bool(v)

    @property
    def lsdn_production_error_frac(self): return self._lsdn_production_error_frac
    @lsdn_production_error_frac.setter
    def lsdn_production_error_frac(self, v):
        v = float(v)
        if v < 0:
            raise ValueError("lsdn_production_error_frac must be >= 0")
        self._lsdn_production_error_frac = v

    @property
    def collection_year(self): return self._collection_year
    @collection_year.setter
    def collection_year(self, v): self._collection_year = int(v)

    @property
    def production_error(self): return self._production_error
    @production_error.setter
    def production_error(self, v):
        if not isinstance(v, _DistParam):
            raise TypeError("production_error must be a _DistParam")
        v.validate("production_error")
        self._production_error = v

    @property
    def half_life_sigma(self): return self._half_life_sigma
    @half_life_sigma.setter
    def half_life_sigma(self, v):
        v = float(v)
        if v < 0:
            raise ValueError("half_life_sigma must be ≥ 0")
        self._half_life_sigma = v

    # ----------------------------------------------------------------- muons
    @property
    def muon_fit_depth_m(self): return self._muon_fit_depth_m
    @muon_fit_depth_m.setter
    def muon_fit_depth_m(self, v):
        v = float(v)
        if v <= 0:
            raise ValueError("muon_fit_depth_m must be positive")
        self._muon_fit_depth_m = v

    @property
    def muon_percent_error(self): return self._muon_percent_error
    @muon_percent_error.setter
    def muon_percent_error(self, v):
        v = float(v)
        if v < 0:
            raise ValueError("muon_percent_error must be ≥ 0")
        self._muon_percent_error = v

    # --------------------------------------------------------------- density
    @property
    def density_from_file(self): return self._density_from_file
    @density_from_file.setter
    def density_from_file(self, v): self._density_from_file = bool(v)

    @property
    def density_error(self): return self._density_error
    @density_error.setter
    def density_error(self, v):
        if not isinstance(v, _DistParam):
            raise TypeError("density_error must be a _DistParam")
        v.validate("density_error")
        self._density_error = v

    # -------------------------------------------------------------------- MC
    @property
    def mc_n_solutions(self): return self._mc_n_solutions
    @mc_n_solutions.setter
    def mc_n_solutions(self, v):
        v = int(v)
        if v <= 0:
            raise ValueError("mc_n_solutions must be positive")
        self._mc_n_solutions = v

    @property
    def mc_age(self): return self._mc_age
    @mc_age.setter
    def mc_age(self, v):
        if not isinstance(v, _DistParam):
            raise TypeError("mc_age must be a _DistParam")
        v.validate("mc_age")
        self._mc_age = v

    @property
    def mc_erosion_deposition_rate(self): return self._mc_erosion_deposition_rate
    @mc_erosion_deposition_rate.setter
    def mc_erosion_deposition_rate(self, v):
        if not isinstance(v, _DistParam):
            raise TypeError("mc_erosion_deposition_rate must be a _DistParam")
        v.validate("mc_erosion_deposition_rate")
        self._mc_erosion_deposition_rate = v

    @property
    def mc_inheritance(self): return self._mc_inheritance
    @mc_inheritance.setter
    def mc_inheritance(self, v):
        if not isinstance(v, _DistParam):
            raise TypeError("mc_inheritance must be a _DistParam")
        v.validate("mc_inheritance")
        self._mc_inheritance = v

    @property
    def mc_neutron_attenuation(self): return self._mc_neutron_attenuation
    @mc_neutron_attenuation.setter
    def mc_neutron_attenuation(self, v):
        if not isinstance(v, _DistParam):
            raise TypeError("mc_neutron_attenuation must be a _DistParam")
        v.validate("mc_neutron_attenuation")
        self._mc_neutron_attenuation = v

    @property
    def mc_erosion_deposition_threshold(self): return self._mc_erosion_deposition_threshold
    @mc_erosion_deposition_threshold.setter
    def mc_erosion_deposition_threshold(self, v):
        v = list(v)
        if len(v) != 2 or v[0] > v[1]:
            raise ValueError("mc_erosion_deposition_threshold must be [min, max] with min ≤ max")
        self._mc_erosion_deposition_threshold = v

    @property
    def mc_confidence_mode(self): return self._mc_confidence_mode
    @mc_confidence_mode.setter
    def mc_confidence_mode(self, v):
        if v not in _VALID_CONFIDENCE_MODES:
            raise ValueError(f"mc_confidence_mode must be one of {_VALID_CONFIDENCE_MODES}")
        self._mc_confidence_mode = v

    @property
    def mc_confidence_value(self): return self._mc_confidence_value
    @mc_confidence_value.setter
    def mc_confidence_value(self, v):
        v = float(v)
        if v <= 0:
            raise ValueError("mc_confidence_value must be positive")
        self._mc_confidence_value = v

    @property
    def mc_n_workers(self): return self._mc_n_workers
    @mc_n_workers.setter
    def mc_n_workers(self, v):
        v = int(v)
        if v < 0:
            raise ValueError("mc_n_workers must be ≥ 0")
        self._mc_n_workers = v

    # ------------------------------------------------------------ shielding
    @property
    def shielding_from_file(self): return self._shielding_from_file
    @shielding_from_file.setter
    def shielding_from_file(self, v): self._shielding_from_file = bool(v)

    @property
    def shielding_value(self): return self._shielding_value
    @shielding_value.setter
    def shielding_value(self, v):
        v = float(v)
        if not 0 <= v <= 1:
            raise ValueError("shielding_value must be 0–1")
        self._shielding_value = v

    # -------------------------------------------------- age constraints
    def _validate_constraint(self, name: str, v) -> None:
        if not isinstance(v, _DistParam):
            raise TypeError(f"{name} must be a _DistParam or None")
        if v.mode not in ("constant", "normal"):
            raise ValueError(f"{name}.mode must be 'constant' or 'normal'")
        if v.mode == "constant" and len(v.parameters) != 1:
            raise ValueError(f"{name}: constant mode requires exactly 1 parameter")
        if v.mode == "normal":
            if len(v.parameters) != 2:
                raise ValueError(f"{name}: normal mode requires exactly 2 parameters")
            if v.parameters[1] <= 0:
                raise ValueError(f"{name}: sigma must be > 0")

    @property
    def age_max_constraint(self): return self._age_max_constraint
    @age_max_constraint.setter
    def age_max_constraint(self, v):
        if v is not None:
            self._validate_constraint("age_max_constraint", v)
        self._age_max_constraint = v

    @property
    def age_min_constraint(self): return self._age_min_constraint
    @age_min_constraint.setter
    def age_min_constraint(self, v):
        if v is not None:
            self._validate_constraint("age_min_constraint", v)
        self._age_min_constraint = v

    # ---------------------------------------------------------- serialisation
    def validate(self):
        """Check for required fields and cross-validate settings."""
        if self.profile_data is None and self._profile_file is None:
            raise ValueError("Profile data must be loaded before running.")
        self._production_error.validate("production_error")
        self._density_error.validate("density_error")
        self._mc_age.validate("mc_age")
        self._mc_erosion_deposition_rate.validate("mc_erosion_deposition_rate")
        self._mc_inheritance.validate("mc_inheritance")
        self._mc_neutron_attenuation.validate("mc_neutron_attenuation")
        if self._density_from_file and self.density_data is None:
            raise ValueError("density_from_file=True but no density data loaded.")

    def to_yaml(self, filename: str):
        from .yaml_io import save_yaml
        save_yaml(self, filename)

    @classmethod
    def from_yaml(cls, filename: str) -> "ProfileSettings":
        from .yaml_io import load_yaml
        return load_yaml(filename)
