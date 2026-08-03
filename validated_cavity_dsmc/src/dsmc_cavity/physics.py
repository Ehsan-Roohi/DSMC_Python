"""VHS collision physics shared by every pair-selection algorithm."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

K_B = 1.380649e-23


def vhs_sigma_g_scalar(
    relative_speed: float,
    diameter_ref: float,
    temperature_ref: float,
    viscosity_index: float,
    molecular_mass: float,
) -> float:
    """Return VHS total cross-section times relative speed.

    The identical-particle reduced mass is m/2.  This is the Bird VHS form,
    including Gamma(2.5-omega), rather than a constant hard-sphere diameter.
    """
    g = max(float(relative_speed), 1.0e-30)
    reduced_mass = 0.5 * molecular_mass
    sigma_ref = math.pi * diameter_ref**2 / math.gamma(2.5 - viscosity_index)
    energy_ratio = 2.0 * K_B * temperature_ref / (reduced_mass * g * g)
    return sigma_ref * energy_ratio ** (viscosity_index - 0.5) * g


def vhs_sigma_g(relative_speed: Any, config: Any, xp: Any) -> Any:
    g = xp.maximum(relative_speed, 1.0e-30)
    reduced_mass = 0.5 * config.mass
    sigma_ref = (
        math.pi
        * config.diameter_ref**2
        / math.gamma(2.5 - config.viscosity_index)
    )
    energy_ratio = (
        2.0 * K_B * config.temperature_ref / (reduced_mass * g * g)
    )
    return sigma_ref * energy_ratio ** (config.viscosity_index - 0.5) * g


def total_momentum_energy(velocities: np.ndarray, mass: float) -> tuple[np.ndarray, float]:
    momentum = mass * velocities.sum(axis=0)
    energy = 0.5 * mass * float(np.sum(velocities * velocities))
    return momentum, energy
