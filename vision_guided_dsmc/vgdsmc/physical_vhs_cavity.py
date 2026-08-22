from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .vhs_sbt import KB, MASS_AR, sbt_collide_cell


@dataclass(frozen=True)
class PhysicalCavityConfig:
    nx: int = 8
    ny: int = 8
    particles_per_cell: int = 20
    steps: int = 100
    sample_start: int = 50
    length: float = 1.0e-6
    depth: float = 1.0e-6
    number_density: float = 1.0e25
    dt: float = 5.0e-11
    t_left: float = 300.0
    t_right: float = 240.0
    t_top: float = 270.0
    t_bottom: float = 270.0
    seed: int = 7


def initialize_physical_state(
    cfg: PhysicalCavityConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Initialize 2-D positions and 3-D Argon molecular velocities."""

    rng = np.random.default_rng(cfg.seed)
    particle_count = cfg.nx * cfg.ny * cfg.particles_per_cell
    positions = rng.random((particle_count, 2)) * cfg.length
    thermal_std = np.sqrt(KB * 270.0 / MASS_AR)
    velocities = rng.normal(0.0, thermal_std, (particle_count, 3))
    cell_volume = (
        (cfg.length / cfg.nx) * (cfg.length / cfg.ny) * cfg.depth
    )
    fnum = cfg.number_density * cell_volume / cfg.particles_per_cell
    return positions, velocities, fnum


def _diffuse_velocities(
    velocities: np.ndarray,
    temperature: np.ndarray,
    normal_axis: int,
    sign: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return diffuse-wall velocities without relying on fancy-index views."""

    reflected = velocities.copy()
    sigma = np.sqrt(KB * temperature / MASS_AR)
    tangential_axes = [axis for axis in range(3) if axis != normal_axis]
    reflected[:, tangential_axes] = rng.normal(
        0.0, sigma[:, None], (len(reflected), 2)
    )
    reflected[:, normal_axis] = sign * sigma * np.sqrt(
        -2.0 * np.log(np.maximum(rng.random(len(reflected)), 1.0e-12))
    )
    return reflected


def _apply_physical_walls(
    positions: np.ndarray,
    velocities: np.ndarray,
    cfg: PhysicalCavityConfig,
    rng: np.random.Generator,
) -> None:
    length = cfg.length
    for _ in range(4):
        masks = (
            positions[:, 0] < 0.0,
            positions[:, 0] >= length,
            positions[:, 1] < 0.0,
            positions[:, 1] >= length,
        )
        if not any(np.any(mask) for mask in masks):
            break
        for wall, mask in enumerate(masks):
            if not np.any(mask):
                continue
            ids = np.flatnonzero(mask)
            if wall == 0:
                positions[ids, 0] *= -1.0
                velocities[ids] = _diffuse_velocities(
                    velocities[ids], np.full(len(ids), cfg.t_left), 0, +1.0, rng
                )
            elif wall == 1:
                positions[ids, 0] = 2.0 * length - positions[ids, 0]
                velocities[ids] = _diffuse_velocities(
                    velocities[ids], np.full(len(ids), cfg.t_right), 0, -1.0, rng
                )
            elif wall == 2:
                positions[ids, 1] *= -1.0
                velocities[ids] = _diffuse_velocities(
                    velocities[ids], np.full(len(ids), cfg.t_bottom), 1, +1.0, rng
                )
            else:
                positions[ids, 1] = 2.0 * length - positions[ids, 1]
                velocities[ids] = _diffuse_velocities(
                    velocities[ids], np.full(len(ids), cfg.t_top), 1, -1.0, rng
                )
    positions[:] = np.clip(positions, 0.0, length * (1.0 - 1.0e-12))


def _cell_ids(positions: np.ndarray, cfg: PhysicalCavityConfig) -> np.ndarray:
    ix = np.clip(
        (positions[:, 0] / cfg.length * cfg.nx).astype(np.int64),
        0,
        cfg.nx - 1,
    )
    iy = np.clip(
        (positions[:, 1] / cfg.length * cfg.ny).astype(np.int64),
        0,
        cfg.ny - 1,
    )
    return iy * cfg.nx + ix


def sample_physical_temperature(
    positions: np.ndarray,
    velocities: np.ndarray,
    cfg: PhysicalCavityConfig,
) -> np.ndarray:
    cell = _cell_ids(positions, cfg)
    cell_count = cfg.nx * cfg.ny
    counts = np.bincount(cell, minlength=cell_count).astype(float)
    safe_counts = np.maximum(counts, 1.0)
    mean_velocity = np.stack(
        [
            np.bincount(cell, weights=velocities[:, axis], minlength=cell_count)
            / safe_counts
            for axis in range(3)
        ],
        axis=1,
    )
    mean_square_speed = (
        np.bincount(
            cell,
            weights=np.sum(velocities**2, axis=1),
            minlength=cell_count,
        )
        / safe_counts
    )
    temperature = MASS_AR / (3.0 * KB) * np.maximum(
        mean_square_speed - np.sum(mean_velocity**2, axis=1), 0.0
    )
    return temperature.reshape(cfg.ny, cfg.nx)


def run_physical_vhs_cavity(
    cfg: PhysicalCavityConfig,
) -> tuple[np.ndarray, int]:
    """Run the physical Argon cavity with VHS cross-sections and SBT selection."""

    if not 0 <= cfg.sample_start < cfg.steps:
        raise ValueError("sample_start must satisfy 0 <= sample_start < steps")
    rng = np.random.default_rng(cfg.seed + 1)
    positions, velocities, fnum = initialize_physical_state(cfg)
    cell_volume = (
        (cfg.length / cfg.nx) * (cfg.length / cfg.ny) * cfg.depth
    )
    temperature_sum = np.zeros((cfg.ny, cfg.nx))
    sample_count = 0
    accepted_collisions = 0

    for step in range(cfg.steps):
        positions += velocities[:, :2] * cfg.dt
        _apply_physical_walls(positions, velocities, cfg, rng)
        cell = _cell_ids(positions, cfg)
        for cell_id in range(cfg.nx * cfg.ny):
            ids = np.flatnonzero(cell == cell_id)
            if len(ids) < 2:
                continue
            velocities[ids], accepted = sbt_collide_cell(
                velocities[ids], fnum, cfg.dt, cell_volume, rng
            )
            accepted_collisions += accepted
        if step >= cfg.sample_start:
            temperature_sum += sample_physical_temperature(
                positions, velocities, cfg
            )
            sample_count += 1

    return temperature_sum / sample_count, accepted_collisions
