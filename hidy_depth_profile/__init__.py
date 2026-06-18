"""
hidy_depth_profile
==================
Monte Carlo ¹⁰Be depth-profile simulator.

Python translation of the Hidy et al. MATLAB code, extended with
LSDn scaling via the stoneage package.
"""

from .settings import ProfileSettings
from .simulator import MonteCarloSimulator

__version__ = "0.1.0"
__all__ = ["ProfileSettings", "MonteCarloSimulator"]
