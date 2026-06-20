"""
hidy_depth_profile
==================
Monte Carlo ¹⁰Be depth-profile simulator.

Python translation of the Hidy et al. MATLAB code, extended with
LSDn scaling via the stoneage package.
"""

from .settings import ProfileSettings
from .simulator import MonteCarloSimulator
from .joint_simulator import JointSimulator, JointResults
from .terrace_chrono import TerraceChrono, TerraChronoResult, OSLSurface

__version__ = "0.1.0"
__all__ = [
    "ProfileSettings",
    "MonteCarloSimulator",
    "JointSimulator",
    "JointResults",
    "TerraceChrono",
    "TerraChronoResult",
    "OSLSurface",
]
