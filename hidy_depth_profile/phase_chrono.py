"""
PhaseChrono: joint inversion for terrace surfaces with unknown phase ages
and latent surface-to-phase assignments.

K latent phases have unknown ages.  Each surface is assigned to exactly one
phase; surfaces in the same phase share that phase's age (within
within_phase_sigma_yr).  The elevation ordering constrains which assignments
are valid: a geomorphically higher (older) surface must be in the same phase
or an older phase than any lower (younger) surface.

Algorithm
---------
For each batch:
  1. Draw K phase ages from an equal-weight pool of all surface posteriors;
     sort descending (phase 0 = oldest).
  2. For each valid assignment and surface, compute the Gaussian overlap
     weight: E_{age ~ pool_i}[N(age; t_k, sigma_intra)].
  3. Sample one assignment per batch element from the categorical
     distribution over valid assignments (weights = product of per-surface
     overlaps).
  4. Output surface age = phase_age + N(0, sigma_intra) for each surface.

This is importance-weighted sampling: the phase age draw provides the
proposal and the assignment weighting concentrates probability on
configurations consistent with the surface posteriors.  Acceptance fraction
is effectively 100% minus the rare case where all assignments have zero weight
(all phase ages landed far from every surface posterior).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product as _iproduct
from typing import Dict, List, Optional, Tuple

import numpy as np


_DEFAULT_SIGMA_INTRA: float = 300.0   # yr
_N_SUB: int = 2_000                   # pool size used for weight computation


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _enumerate_valid_assignments(
    n_surfaces: int,
    elev_order_idx: List[Tuple[int, int]],
    n_phases: int,
) -> np.ndarray:
    """
    Return every valid phase-assignment tuple as a row of an integer array.

    Shape: (n_valid, n_surfaces).  Phase indices are in {0, …, n_phases-1};
    0 is the oldest phase.  An assignment is valid when phase[higher] ≤
    phase[lower] for every (higher_idx, lower_idx) elevation pair.
    """
    valid = [
        a
        for a in _iproduct(range(n_phases), repeat=n_surfaces)
        if all(a[hi] <= a[lo] for hi, lo in elev_order_idx)
    ]
    return np.array(valid, dtype=np.int32)   # (n_valid, n_surfaces)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PhaseChronoResult:
    """
    Posterior samples from a PhaseChrono inversion.

    ages
        Per-surface age draws in years: dict[name → ndarray(n_draws)].
        Each value is the assigned phase age ± within-phase jitter.
    phase_ages
        Shape (n_draws, K).  Latent phase ages in years, oldest (index 0)
        first.
    assignments
        Per-surface array of sampled phase index (0 = oldest):
        dict[name → ndarray(n_draws, dtype=int)].
    assignment_probs
        Per-surface marginal posterior P(surface → phase k), length K:
        dict[name → ndarray(K)].
    n_phases_occupied
        Per-draw count of distinct phases actually assigned to any surface.
    n_accepted
        Equals n_draws (every draw that passes the zero-weight filter is
        kept).
    n_attempted
        Total batch-element draws made, including zero-weight rejects.
    """

    ages: Dict[str, np.ndarray]
    phase_ages: np.ndarray
    assignments: Dict[str, np.ndarray]
    assignment_probs: Dict[str, np.ndarray]
    n_phases_occupied: np.ndarray
    n_accepted: int
    n_attempted: int

    @property
    def acceptance_fraction(self) -> float:
        return self.n_accepted / self.n_attempted if self.n_attempted else 0.0

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Per-surface MAP and 2σ bounds (2.3 / 97.7 percentiles) in ka."""
        out = {}
        for name, arr in self.ages.items():
            if len(arr) == 0:
                out[name] = {
                    "map_ka": float("nan"),
                    "sigma2_minus_ka": float("nan"),
                    "sigma2_plus_ka": float("nan"),
                }
                continue
            n_bins = min(50, max(2, len(arr) // 5))
            counts, edges = np.histogram(arr, bins=n_bins)
            centres = 0.5 * (edges[:-1] + edges[1:])
            out[name] = {
                "map_ka":          float(centres[np.argmax(counts)]) / 1e3,
                "sigma2_minus_ka": float(np.percentile(arr, 2.3))   / 1e3,
                "sigma2_plus_ka":  float(np.percentile(arr, 97.7))  / 1e3,
            }
        return out

    def phase_summary(self) -> List[Dict[str, float]]:
        """Per-phase MAP, 2σ bounds in ka, and fraction-of-draws occupancy."""
        K = self.phase_ages.shape[1]
        asgn_stack = np.stack(
            [self.assignments[n] for n in self.assignments], axis=1
        )  # (n_draws, N)
        out = []
        for k in range(K):
            arr = self.phase_ages[:, k]
            n_bins = min(50, max(2, len(arr) // 5))
            counts, edges = np.histogram(arr, bins=n_bins)
            centres = 0.5 * (edges[:-1] + edges[1:])
            occ = float(np.mean(np.any(asgn_stack == k, axis=1)))
            out.append({
                "map_ka":          float(centres[np.argmax(counts)]) / 1e3,
                "sigma2_minus_ka": float(np.percentile(arr, 2.3))   / 1e3,
                "sigma2_plus_ka":  float(np.percentile(arr, 97.7))  / 1e3,
                "occupancy":       occ,
            })
        return out

    def print_summary(self) -> None:
        K = self.phase_ages.shape[1]
        print(
            f"PhaseChrono: {self.n_accepted} draws  "
            f"({self.acceptance_fraction:.1%} of {self.n_attempted} attempts)",
            flush=True,
        )
        print(f"  n_phases (upper bound): {K}", flush=True)
        print("  Phase ages, oldest first:", flush=True)
        for k, ps in enumerate(self.phase_summary()):
            map_ka = ps["map_ka"]
            print(
                f"    phase {k}: {map_ka:.1f} ka  "
                f"+{ps['sigma2_plus_ka'] - map_ka:.1f} / "
                f"-{map_ka - ps['sigma2_minus_ka']:.1f} (2σ)  "
                f"[in {ps['occupancy']:.0%} of draws]",
                flush=True,
            )
        print("  Surface posteriors:", flush=True)
        s = self.summary()
        for name in self.ages:
            v = s[name]
            map_ka = v["map_ka"]
            probs = self.assignment_probs[name]
            best = int(np.argmax(probs))
            print(
                f"    {name}: {map_ka:.1f} ka  "
                f"+{v['sigma2_plus_ka'] - map_ka:.1f} / "
                f"-{map_ka - v['sigma2_minus_ka']:.1f} (2σ)  "
                f"→ phase {best} ({probs[best]:.0%})",
                flush=True,
            )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PhaseChrono:
    """
    Joint inversion for K latent incision phases with unknown ages and
    unknown surface-to-phase assignments.

    Parameters
    ----------
    surfaces : dict mapping name → object with .age attribute
        Any surface type used by TerraceChrono is accepted: the result
        of MonteCarloSimulator.run(), JointSimulator.run(), or OSLSurface.
    elevation_ordering : list of (higher_name, lower_name) pairs
        Geomorphic ordering constraint.  A higher surface must be in the same
        phase or an older (lower-index) phase than a lower surface.
    n_phases : int
        Upper bound on the number of distinct incision phases.  Empty phases
        are allowed; the algorithm infers how many are occupied.
    within_phase_sigma_yr : float, optional
        Expected spread (1σ, yr) of surface abandonment ages within a single
        incision episode.  Should be smaller than the between-phase gap and
        smaller than individual dating uncertainty.  Default: 300 yr.
    """

    DEFAULT_SIGMA_INTRA: float = _DEFAULT_SIGMA_INTRA

    def __init__(
        self,
        surfaces: Dict[str, object],
        elevation_ordering: List[Tuple[str, str]],
        n_phases: int,
        within_phase_sigma_yr: float = _DEFAULT_SIGMA_INTRA,
    ) -> None:
        if n_phases < 1:
            raise ValueError("n_phases must be >= 1")
        names = list(surfaces.keys())
        name_set = set(names)
        for hi, lo in elevation_ordering:
            if hi not in name_set:
                raise ValueError(
                    f"elevation_ordering references unknown surface: {hi!r}"
                )
            if lo not in name_set:
                raise ValueError(
                    f"elevation_ordering references unknown surface: {lo!r}"
                )

        self._surfaces = surfaces
        self._names = names
        self._n_phases = n_phases
        self._sigma = within_phase_sigma_yr

        name_to_idx = {n: i for i, n in enumerate(names)}
        self._elev_idx: List[Tuple[int, int]] = [
            (name_to_idx[hi], name_to_idx[lo])
            for hi, lo in elevation_ordering
        ]

        self._valid_asgn = _enumerate_valid_assignments(
            len(names), self._elev_idx, n_phases,
        )   # (n_valid, N)

        if len(self._valid_asgn) == 0:
            raise ValueError(
                "No valid phase assignments exist for the given "
                "elevation_ordering and n_phases.  Check that ordering "
                "constraints are mutually consistent."
            )

    @property
    def within_phase_sigma_yr(self) -> float:
        return self._sigma

    # ------------------------------------------------------------------
    # Core sampling
    # ------------------------------------------------------------------

    def constrain(
        self,
        n_draws: int = 10_000,
        seed: Optional[int] = None,
    ) -> PhaseChronoResult:
        """
        Draw n_draws samples from the joint posterior over phase ages and
        surface-to-phase assignments.

        Every batch element with at least one valid assignment of nonzero
        weight contributes one output sample (no rejection, just
        assignment-weighted categorical sampling).
        """
        rng = np.random.default_rng(seed)
        N = len(self._names)
        K = self._n_phases
        n_valid = len(self._valid_asgn)
        sigma = self._sigma

        # Extract per-surface age pools
        pools = [np.asarray(self._surfaces[n].age) for n in self._names]

        # Subsample to _N_SUB for weight computation; keeps OSLSurface (100 k)
        # and per-surface contribution equal
        pools_sub = [
            rng.choice(p, size=min(len(p), _N_SUB), replace=False)
            for p in pools
        ]

        # Equal-weight combined pool for phase-age proposals
        pool_eq = np.concatenate([
            rng.choice(p, size=min(len(p), _N_SUB), replace=False)
            for p in pools
        ])

        # Output buffers
        ages_out       = {n: np.empty(n_draws)               for n in self._names}
        phase_ages_out = np.empty((n_draws, K))
        asgn_out       = {n: np.empty(n_draws, dtype=np.int32) for n in self._names}
        asgn_count     = np.zeros((N, K))   # accumulator for posterior probs
        occ_out        = np.empty(n_draws, dtype=np.int32)

        j = 0
        n_attempted = 0
        batch = max(20_000, n_draws * 4)
        max_attempts = max(200_000, n_draws * 100)

        while j < n_draws:
            if n_attempted >= max_attempts and j < n_draws // 2:
                raise RuntimeError(
                    f"PhaseChrono: only {j} of {n_draws} draws accepted after "
                    f"{n_attempted} attempts.  Check that surface posteriors "
                    "and n_phases are compatible."
                )

            B = batch

            # -- Phase age proposals: K draws from equal-weight pool, sorted --
            raw = rng.choice(pool_eq, size=(B, K), replace=True)
            phase_ages = np.sort(raw, axis=1)[:, ::-1]   # (B, K), descending

            # -- Gaussian overlap weights: w[b, i, k] --
            # ≈ E_{age ~ pool_i}[ N(age; phase_ages[b,k], sigma) ]
            w = np.empty((B, N, K))
            for i, psub in enumerate(pools_sub):
                for k in range(K):
                    diff = phase_ages[:, k, np.newaxis] - psub[np.newaxis, :]
                    w[:, i, k] = np.mean(
                        np.exp(-0.5 * (diff / sigma) ** 2), axis=1
                    )

            # -- Assignment weights: product over surfaces --
            # asgn_w[b, a] = ∏_i w[b, i, valid_asgn[a, i]]
            asgn_w = np.ones((B, n_valid))
            for i in range(N):
                asgn_w *= w[:, i, self._valid_asgn[:, i]]   # (B, n_valid)

            # -- Skip batch elements with all-zero weights --
            row_sums = asgn_w.sum(axis=1)   # (B,)
            valid_mask = row_sums > 0
            n_attempted += B
            if not valid_mask.any():
                continue

            b_valid = np.where(valid_mask)[0]   # indices into B

            # -- Sample one assignment per valid batch element --
            asgn_prob = asgn_w[b_valid] / row_sums[b_valid, np.newaxis]
            cum = np.cumsum(asgn_prob, axis=1)
            u = rng.uniform(size=(len(b_valid), 1))
            a_idx = np.clip((cum < u).sum(axis=1), 0, n_valid - 1)
            sampled_a = self._valid_asgn[a_idx]   # (n_valid_batch, N)

            # -- Collect up to (n_draws - j) --
            take = min(len(b_valid), n_draws - j)
            bv = b_valid[:take]    # row indices into (B, K) phase_ages
            sa = sampled_a[:take]  # (take, N) assigned phase index per surface

            phase_ages_out[j:j + take] = phase_ages[bv]

            for i, name in enumerate(self._names):
                k_per_draw = sa[:, i]                  # (take,) phase indices
                mu = phase_ages[bv, k_per_draw]        # (take,) phase ages
                ages_out[name][j:j + take] = (
                    mu + rng.normal(0.0, sigma, size=take)
                )
                asgn_out[name][j:j + take] = k_per_draw
                np.add.at(asgn_count[i], k_per_draw, 1)

            # Count distinct occupied phases per draw
            sa_sort = np.sort(sa, axis=1)
            occ_out[j:j + take] = (
                1 + (sa_sort[:, 1:] != sa_sort[:, :-1]).sum(axis=1)
                if N > 1 else np.ones(take, dtype=np.int32)
            )

            j += take

        # Posterior assignment probabilities (marginal per surface)
        asgn_probs_out: Dict[str, np.ndarray] = {}
        for i, name in enumerate(self._names):
            total = asgn_count[i].sum()
            asgn_probs_out[name] = (
                asgn_count[i] / total if total > 0 else np.zeros(K)
            )

        return PhaseChronoResult(
            ages=ages_out,
            phase_ages=phase_ages_out,
            assignments=asgn_out,
            assignment_probs=asgn_probs_out,
            n_phases_occupied=occ_out,
            n_accepted=n_draws,
            n_attempted=n_attempted,
        )
