"""Validated lid-driven-cavity DSMC teaching solver."""

from .config import SimulationConfig
from .solver import CavitySolver

__all__ = ["CavitySolver", "SimulationConfig"]
__version__ = "0.1.0"
