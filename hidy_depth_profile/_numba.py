"""
Optional Numba JIT-compiled forward-model kernel.

When Numba is importable and compatible with the installed NumPy,
``numba_forward_batch`` is a JIT-compiled, parallelised replacement for
the core accumulation loop in MonteCarloSimulator._forward_batch.

``NUMBA_AVAILABLE`` is True when the kernel was compiled successfully.
``NUMBA_INFO`` carries a human-readable status string printed at setup.

Parallelisation note: with ``parallel=True``, Numba distributes the outer
draw-index loop (``prange(B)``) across all available CPU cores. Each draw is
fully independent, so this is embarrassingly parallel with no synchronisation.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange  # ImportError if not installed or incompatible

    @njit(parallel=True, cache=True)
    def numba_forward_batch(
        ages: np.ndarray,          # (B,) yr
        decay_consts: np.ndarray,  # (B,) yr⁻¹
        erosions: np.ndarray,      # (B,) cm/yr
        inheritances: np.ndarray,  # (B,) atoms/g
        densities: np.ndarray,     # (B, n_depths)  g/cm³
        v1: np.ndarray,            # (B, n_pathways) attenuation lengths g/cm²
        v2: np.ndarray,            # (B, n_pathways) surface rates atoms/g/yr
        depths: np.ndarray,        # (n_depths,) cm   – centre of sample
        thicknesses: np.ndarray,   # (n_depths,) cm   – full thickness
    ) -> np.ndarray:               # (B, n_depths) atoms/g
        """
        Triple loop over draws × depths × pathways, fully parallelised on draws.

        Mathematically identical to the NumPy _forward_batch but avoids
        the Python-level depth loop and large intermediate arrays, enabling
        both multi-core execution and better CPU-cache utilisation.
        """
        B = ages.shape[0]
        n_depths = depths.shape[0]
        n_pathways = v1.shape[1]
        modelled = np.empty((B, n_depths))
        for i in prange(B):
            lam = decay_consts[i]
            age = ages[i]
            er = erosions[i]
            inh_term = inheritances[i] * np.exp(-age * lam)
            for k in range(n_depths):
                z = depths[k]
                dz = thicknesses[k]
                rho = densities[i, k]
                acc = 0.0
                for p in range(n_pathways):
                    vv1 = v1[i, p]
                    vv2 = v2[i, p]
                    tmp1 = np.exp(-z * rho / vv1)
                    tmp2 = np.exp(-(z + dz) * rho / vv1)
                    simp = lam + er * rho / vv1
                    tmp3 = (vv1 * vv2) * (np.exp(-age * simp) - 1.0) / (rho * simp)
                    acc += (tmp2 - tmp1) * tmp3
                modelled[i, k] = acc / dz + inh_term
        return modelled

    NUMBA_AVAILABLE = True
    NUMBA_INFO = "Numba JIT (parallel)"

except ImportError as _e:
    NUMBA_AVAILABLE = False
    NUMBA_INFO = f"NumPy fallback (Numba unavailable: {_e})"

    def numba_forward_batch(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError("Numba is not available; this function should not be called.")
