"""
TerraceChrono: post-hoc ordering constraints for a terrace chronology.

After each surface (or equal-age group) has been inverted — via
MonteCarloSimulator or JointSimulator — TerraceChrono samples age tuples
from the accepted pools and rejects any tuple that violates the specified
stratigraphic ordering constraints.

The underlying factorisation is:

    P(ages | all profiles, ordering) ∝
        ∏_i P(profile_i | age_i) × P(ordering | ages)

where P(ordering | ages) = 1 if the ordering holds, 0 otherwise.  Because
the per-profile likelihoods are already encoded in each surface's accepted
age distribution, sampling from those pools and applying the binary ordering
filter gives correctly constrained marginals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


def _age_array(result) -> np.ndarray:
    """Extract the accepted age array from a Results, JointResults, or OSLSurface."""
    return result.age


class OSLSurface:
    """
    A terrace surface dated by OSL only (no ¹⁰Be depth profile).

    Represents the surface age as a large Gaussian sample so that it can be
    passed directly to TerraceChrono alongside MonteCarloSimulator /
    JointSimulator results.

    Parameters
    ----------
    mean_yr : float
        OSL date mean in years before present.
    sigma_yr : float
        1σ uncertainty in years.
    n_pool : int
        Size of the Gaussian sample pool (default 100 000; large enough that
        resampling by TerraceChrono introduces no meaningful discretisation).
    seed : int or None
        RNG seed for the pool draw.
    """

    def __init__(
        self,
        mean_yr: float,
        sigma_yr: float,
        n_pool: int = 100_000,
        seed=None,
    ):
        self.mean_yr = mean_yr
        self.sigma_yr = sigma_yr
        rng = np.random.default_rng(seed)
        self.age = rng.normal(mean_yr, sigma_yr, n_pool)


@dataclass
class TerraChronoResult:
    """
    Constrained age distributions from a joint terrace ordering inversion.

    ages : dict mapping surface name → 1-D array of constrained ages (yr).
    n_accepted : number of accepted age tuples.
    n_attempted : total tuples drawn before reaching n_accepted.
    acceptance_fraction : n_accepted / n_attempted.
    """
    ages: Dict[str, np.ndarray]
    n_accepted: int
    n_attempted: int
    acceptance_fraction: float

    def summary(self) -> Dict[str, Dict[str, float]]:
        """
        Per-surface MAP (histogram mode), 2σ−, and 2σ+ in ka.

        Returns a dict keyed by surface name, each value a dict with keys
        ``map_ka``, ``sigma2_minus_ka``, ``sigma2_plus_ka``.
        """
        out: Dict[str, Dict[str, float]] = {}
        for name, arr in self.ages.items():
            if len(arr) == 0:
                out[name] = {
                    "map_ka": float("nan"),
                    "sigma2_minus_ka": float("nan"),
                    "sigma2_plus_ka": float("nan"),
                }
                continue
            counts, edges = np.histogram(arr, bins=min(50, len(arr) // 5 + 2))
            centres = 0.5 * (edges[:-1] + edges[1:])
            out[name] = {
                "map_ka":          float(centres[np.argmax(counts)]) / 1e3,
                "sigma2_minus_ka": float(np.percentile(arr, 2.3))  / 1e3,
                "sigma2_plus_ka":  float(np.percentile(arr, 97.7)) / 1e3,
            }
        return out

    def print_summary(self) -> None:
        """Print a formatted summary of constrained ages."""
        print(
            f"TerraceChrono: {self.n_accepted} accepted tuples "
            f"from {self.n_attempted} attempts "
            f"(acceptance {self.acceptance_fraction:.1%})",
            flush=True,
        )
        s = self.summary()
        for name, v in s.items():
            lo = v["map_ka"] - v["sigma2_minus_ka"]
            hi = v["sigma2_plus_ka"] - v["map_ka"]
            print(
                f"  {name}: {v['map_ka']:.1f} ka  "
                f"(+{hi:.1f} / -{lo:.1f} ka, 2σ)",
                flush=True,
            )


class TerraceChrono:
    """
    Joint ordering constraint for a set of surfaces or surface groups.

    Each entry in ``surfaces`` is a single Results or JointResults object
    whose ``.age`` attribute holds the pool of accepted ages.  For a group
    of profiles inverted jointly (JointSimulator), the shared age is already
    a single array, so the group is treated as one entity here.

    Parameters
    ----------
    surfaces : dict mapping name → Results or JointResults
        Surfaces to constrain.  The names must match those used in
        ``ordering_constraints``.
    ordering_constraints : list of (older_name, younger_name) pairs
        Each pair asserts that the named older surface was abandoned before
        the younger one.  Partial orders are supported: not every pair needs
        to be specified.

    Example
    -------
    ::

        # LTF > SC > (RR = D inverted jointly)
        joint_rrd = JointSimulator({"RR": settings_RR, "D": settings_D}).run()
        r_ltf = MonteCarloSimulator(settings_LTF).run()
        r_sc  = MonteCarloSimulator(settings_SC).run()

        tc = TerraceChrono(
            surfaces={"LTF": r_ltf, "SC": r_sc, "RR_D": joint_rrd},
            ordering_constraints=[("LTF", "SC"), ("SC", "RR_D")],
        )
        result = tc.constrain(n_draws=10_000, seed=42)
        result.print_summary()
    """

    def __init__(
        self,
        surfaces: Dict[str, object],
        ordering_constraints: List[Tuple[str, str]],
    ):
        self._surfaces = surfaces
        self._ordering = ordering_constraints
        all_names = set(surfaces.keys())
        for older, younger in ordering_constraints:
            if older not in all_names:
                raise ValueError(
                    f"Ordering constraint references unknown surface: {older!r}"
                )
            if younger not in all_names:
                raise ValueError(
                    f"Ordering constraint references unknown surface: {younger!r}"
                )

    def constrain(
        self,
        n_draws: int = 10_000,
        seed: Optional[int] = None,
    ) -> TerraChronoResult:
        """
        Draw age tuples from the accepted pools and apply ordering constraints.

        Parameters
        ----------
        n_draws : int
            Number of valid (ordering-consistent) tuples to collect.
        seed : int or None
            RNG seed.

        Returns
        -------
        TerraChronoResult
        """
        rng = np.random.default_rng(seed)
        names = list(self._surfaces.keys())
        pools = {name: _age_array(self._surfaces[name]) for name in names}

        ages_out = {name: np.empty(n_draws) for name in names}

        j = 0
        n_attempted = 0
        batch = max(10_000, n_draws * 10)

        while j < n_draws:
            drawn = {name: rng.choice(pools[name], size=batch, replace=True) for name in names}

            valid = np.ones(batch, dtype=bool)
            for older, younger in self._ordering:
                valid &= drawn[older] > drawn[younger]

            n_valid = int(valid.sum())
            n_attempted += batch

            if n_valid > 0:
                take = min(n_valid, n_draws - j)
                idx = np.where(valid)[0][:take]
                for name in names:
                    ages_out[name][j:j + take] = drawn[name][idx]
                j += take

            if n_attempted > 200 * n_draws and j == 0:
                raise RuntimeError(
                    "No valid orderings found after many attempts. "
                    "Check that ordering constraints are consistent with the data."
                )

        return TerraChronoResult(
            ages={name: ages_out[name][:j] for name in names},
            n_accepted=j,
            n_attempted=n_attempted,
            acceptance_fraction=j / n_attempted,
        )
