from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

KB = 1.380649e-23
MASS_AR = 39.948e-3 / 6.02214076e23


@dataclass(frozen=True)
class VHSParameters:
    """Argon VHS parameters matching the repository's Parallel_TAS kernel."""

    diameter_ref: float = 4.17e-10
    temperature_ref: float = 273.0
    omega: float = 0.81
    mass: float = MASS_AR


def vhs_cross_section(
    relative_speed: float | np.ndarray,
    params: VHSParameters = VHSParameters(),
) -> np.ndarray:
    """Return the VHS total collision cross-section."""

    speed = np.asarray(relative_speed, dtype=float)
    safe_speed = np.maximum(speed, 1.0e-30)
    exponent = params.omega - 0.5
    reference_speed_squared = 2.0 * KB * params.temperature_ref / params.mass
    gamma_value = math.gamma(2.5 - params.omega)
    cross_section = (
        math.pi
        * params.diameter_ref**2
        * (reference_speed_squared / safe_speed**2) ** exponent
        / gamma_value
    )
    return np.where(speed > 0.0, cross_section, 0.0)


def scatter_equal_mass(
    first_velocity: np.ndarray,
    second_velocity: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Isotropically scatter an equal-mass pair in three velocity dimensions."""

    first = np.asarray(first_velocity, dtype=float)
    second = np.asarray(second_velocity, dtype=float)
    center_velocity = 0.5 * (first + second)
    relative_speed = np.linalg.norm(first - second)

    cosine = 2.0 * rng.random() - 1.0
    azimuth = 2.0 * math.pi * rng.random()
    sine = math.sqrt(max(0.0, 1.0 - cosine * cosine))
    scattered_relative_velocity = relative_speed * np.array(
        [sine * math.cos(azimuth), sine * math.sin(azimuth), cosine]
    )
    return (
        center_velocity + 0.5 * scattered_relative_velocity,
        center_velocity - 0.5 * scattered_relative_velocity,
    )


def sbt_collide_cell(
    velocities: np.ndarray,
    fnum: float,
    dt: float,
    cell_volume: float,
    rng: np.random.Generator,
    params: VHSParameters = VHSParameters(),
) -> tuple[np.ndarray, int]:
    """Apply the repository's sequential SBT candidate rule in one cell.

    The probability expression follows ``Parallel_TAS.py`` while the function is
    isolated from problem-specific geometry. Velocities are three-dimensional,
    even when particle positions belong to a two-dimensional cavity.
    """

    updated = np.asarray(velocities, dtype=float).copy()
    particle_count = len(updated)
    accepted_collisions = 0
    if particle_count < 2:
        return updated, accepted_collisions

    order = rng.permutation(particle_count)
    for index in range(particle_count - 1):
        first = order[index]
        remaining_count = particle_count - index - 1
        second = order[index + 1 + rng.integers(0, remaining_count)]
        relative_speed = np.linalg.norm(updated[first] - updated[second])
        if relative_speed <= 0.0:
            continue

        cross_section = float(vhs_cross_section(relative_speed, params))
        probability = min(
            1.0,
            remaining_count
            * fnum
            * dt
            * cross_section
            * relative_speed
            / cell_volume,
        )
        if rng.random() < probability:
            updated[first], updated[second] = scatter_equal_mass(
                updated[first], updated[second], rng
            )
            accepted_collisions += 1

    return updated, accepted_collisions
