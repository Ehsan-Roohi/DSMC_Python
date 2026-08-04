from __future__ import annotations

from dataclasses import dataclass
from math import gamma
from typing import Callable
import numpy as np

KB = 1.380649e-23
MASS_AR = 39.948e-3 / 6.02214076e23


@dataclass(frozen=True)
class VHSModel:
    mass: float = MASS_AR
    diameter_ref: float = 4.17e-10
    temperature_ref: float = 273.0
    omega: float = 0.81

    @property
    def gamma_factor(self) -> float:
        return gamma(2.5 - self.omega)

    @property
    def reduced_mass(self) -> float:
        return 0.5 * self.mass

    def cross_section(self, relative_speed: np.ndarray | float) -> np.ndarray | float:
        """Bird VHS total cross section for identical monatomic molecules."""
        g = np.asarray(relative_speed, dtype=np.float64)
        safe = np.maximum(g, 1.0e-12)
        exponent = self.omega - 0.5
        energy_factor = (
            2.0 * KB * self.temperature_ref
            / (self.reduced_mass * safe**2)
        )
        d_sq = (
            self.diameter_ref**2
            * energy_factor**exponent
            / self.gamma_factor
        )
        sigma = np.pi * d_sq
        if np.ndim(relative_speed) == 0:
            return float(sigma)
        return sigma

    def mean_free_path(self, number_density: float, temperature: float) -> float:
        mean_relative_speed = np.sqrt(
            16.0 * KB * temperature / (np.pi * self.mass)
        )
        sigma = self.cross_section(mean_relative_speed)
        return 1.0 / (np.sqrt(2.0) * number_density * sigma)


@dataclass(frozen=True)
class PhysicalCavityConfig:
    nx: int = 10
    ny: int = 10
    particles_per_cell: int = 12
    length: float = 1.0e-6
    depth: float | None = None
    knudsen: float = 0.10
    t_left: float = 330.0
    t_right: float = 270.0
    t_top: float = 300.0
    t_bottom: float = 300.0
    steps: int = 160
    sample_start: int = 80
    target_ppc_subcell: int = 8
    max_subdivisions: int = 4
    dt_safety: float = 0.20
    seed: int = 7
    vhs: VHSModel = VHSModel()
    left_wall_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    right_wall_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    bottom_wall_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    top_wall_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    lid_velocity_x: float = 0.0
    strict_sbt_probability: bool = False
    stratified_initialization: bool = False

    def resolved_wall_velocity(self, wall: str) -> np.ndarray:
        """Return a validated three-component wall velocity in the lab frame.

        The square cavity geometry is fixed, so normal wall motion is not
        supported.  ``lid_velocity_x`` is a convenience alias for the top-wall
        x component used by lid-driven-cavity cases.  An explicitly supplied
        top-wall x component may be used instead, but conflicting values fail
        rather than being silently combined.
        """
        values = {
            "left": self.left_wall_velocity,
            "right": self.right_wall_velocity,
            "bottom": self.bottom_wall_velocity,
            "top": self.top_wall_velocity,
        }
        if wall not in values:
            raise ValueError(f"unknown wall {wall!r}")
        velocity = np.asarray(values[wall], dtype=np.float64)
        if velocity.shape != (3,) or not np.all(np.isfinite(velocity)):
            raise ValueError(f"{wall}_wall_velocity must contain three finite values")
        velocity = velocity.copy()
        if wall == "top" and self.lid_velocity_x != 0.0:
            if velocity[0] != 0.0 and not np.isclose(
                velocity[0], self.lid_velocity_x, rtol=0.0, atol=1.0e-14
            ):
                raise ValueError(
                    "lid_velocity_x conflicts with top_wall_velocity[0]"
                )
            velocity[0] = self.lid_velocity_x
        normal_axis = 0 if wall in {"left", "right"} else 1
        if not np.isclose(velocity[normal_axis], 0.0, rtol=0.0, atol=1.0e-14):
            raise ValueError(
                f"normal motion of the fixed {wall} wall is not supported"
            )
        return velocity

    @property
    def t0(self) -> float:
        return 0.25 * (
            self.t_left + self.t_right + self.t_top + self.t_bottom
        )

    @property
    def domain_depth(self) -> float:
        return self.length if self.depth is None else self.depth

    @property
    def cell_volume(self) -> float:
        return (
            (self.length / self.nx)
            * (self.length / self.ny)
            * self.domain_depth
        )

    @property
    def number_density(self) -> float:
        target_lambda = self.knudsen * self.length
        mean_relative_speed = np.sqrt(
            16.0 * KB * self.t0 / (np.pi * self.vhs.mass)
        )
        sigma = self.vhs.cross_section(mean_relative_speed)
        return 1.0 / (np.sqrt(2.0) * target_lambda * sigma)

    @property
    def real_particles_per_sim_particle(self) -> float:
        real_particles = (
            self.number_density * self.length**2 * self.domain_depth
        )
        simulated_particles = self.nx * self.ny * self.particles_per_cell
        return real_particles / simulated_particles

    @property
    def dt(self) -> float:
        dx = min(self.length / self.nx, self.length / self.ny)
        thermal_speed = np.sqrt(
            2.0
            * KB
            * max(
                self.t_left,
                self.t_right,
                self.t_top,
                self.t_bottom,
            )
            / self.vhs.mass
        )
        transit = dx / max(thermal_speed, 1.0e-30)
        collision = self.vhs.mean_free_path(
            self.number_density,
            self.t0,
        ) / max(thermal_speed, 1.0e-30)
        return self.dt_safety * min(transit, collision)


@dataclass
class PhysicalParticleState:
    pos: np.ndarray  # (N, 2), meters
    vel: np.ndarray  # (N, 3), m/s
    weight: np.ndarray  # relative to base FNUM

    def copy(self) -> "PhysicalParticleState":
        return PhysicalParticleState(
            self.pos.copy(),
            self.vel.copy(),
            self.weight.copy(),
        )


def _thermal_velocities(
    n: int,
    temperature: float,
    model: VHSModel,
    rng: np.random.Generator,
) -> np.ndarray:
    std = np.sqrt(KB * temperature / model.mass)
    return rng.normal(0.0, std, size=(n, 3))


def initialize_physical_state(
    cfg: PhysicalCavityConfig,
) -> PhysicalParticleState:
    rng = np.random.default_rng(cfg.seed)
    n = cfg.nx * cfg.ny * cfg.particles_per_cell
    if cfg.stratified_initialization:
        cell = np.repeat(
            np.arange(cfg.nx * cfg.ny, dtype=np.int64),
            cfg.particles_per_cell,
        )
        ix = cell % cfg.nx
        iy = cell // cfg.nx
        offset = rng.random((n, 2))
        pos = np.column_stack(
            (
                (ix + offset[:, 0]) / cfg.nx,
                (iy + offset[:, 1]) / cfg.ny,
            )
        ) * cfg.length
    else:
        pos = rng.random((n, 2)) * cfg.length
    vel = _thermal_velocities(n, cfg.t0, cfg.vhs, rng)
    return PhysicalParticleState(
        pos=pos,
        vel=vel,
        weight=np.ones(n, dtype=np.float64),
    )


def _diffuse_wall(
    vel: np.ndarray,
    temperature: float,
    normal_axis: int,
    inward_sign: float,
    model: VHSModel,
    rng: np.random.Generator,
    wall_velocity: np.ndarray | tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Sample diffuse reflection in the wall frame and return lab velocities."""
    wall = (
        np.zeros(3, dtype=np.float64)
        if wall_velocity is None
        else np.asarray(wall_velocity, dtype=np.float64)
    )
    if wall.shape != (3,) or not np.all(np.isfinite(wall)):
        raise ValueError("wall_velocity must contain three finite values")
    relative = np.empty(np.asarray(vel).shape, dtype=np.float64)
    std = np.sqrt(KB * temperature / model.mass)
    tangential = [axis for axis in range(3) if axis != normal_axis]
    relative[:, tangential] = rng.normal(
        0.0,
        std,
        size=(len(relative), 2),
    )
    u = np.maximum(rng.random(len(relative)), 1.0e-14)
    relative[:, normal_axis] = (
        inward_sign * std * np.sqrt(-2.0 * np.log(u))
    )
    return relative + wall[None, :]


def apply_diffuse_walls(
    state: PhysicalParticleState,
    cfg: PhysicalCavityConfig,
    rng: np.random.Generator,
    wall_event_handler: Callable[
        [str, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        None,
    ]
    | None = None,
) -> None:
    p, v, length = state.pos, state.vel, cfg.length
    wall_velocity = {
        wall: cfg.resolved_wall_velocity(wall)
        for wall in ("left", "right", "bottom", "top")
    }
    for _ in range(8):
        changed = False
        left = p[:, 0] < 0.0
        if np.any(left):
            ids = np.flatnonzero(left)
            incoming = v[ids].copy()
            if wall_event_handler is not None:
                normal = incoming[:, 0]
                after = np.divide(
                    p[ids, 0],
                    normal,
                    out=np.zeros(len(ids)),
                    where=np.abs(normal) > 1.0e-30,
                )
                impact = np.clip(
                    p[ids, 1] - incoming[:, 1] * after,
                    0.0,
                    length,
                )
                wall_event_handler(
                    "left", impact, incoming, state.weight[ids].copy(),
                    wall_velocity["left"],
                )
            p[ids, 0] *= -1.0
            reflected = _diffuse_wall(
                v[ids], cfg.t_left, 0, +1.0, cfg.vhs, rng,
                wall_velocity["left"],
            )
            v[ids] = reflected
            if wall_event_handler is not None:
                wall_event_handler(
                    "left", impact, reflected.copy(), state.weight[ids].copy(),
                    wall_velocity["left"],
                )
            changed = True
        right = p[:, 0] >= length
        if np.any(right):
            ids = np.flatnonzero(right)
            incoming = v[ids].copy()
            if wall_event_handler is not None:
                normal = incoming[:, 0]
                after = np.divide(
                    p[ids, 0] - length,
                    normal,
                    out=np.zeros(len(ids)),
                    where=np.abs(normal) > 1.0e-30,
                )
                impact = np.clip(
                    p[ids, 1] - incoming[:, 1] * after,
                    0.0,
                    length,
                )
                wall_event_handler(
                    "right", impact, incoming, state.weight[ids].copy(),
                    wall_velocity["right"],
                )
            p[ids, 0] = 2.0 * length - p[ids, 0]
            reflected = _diffuse_wall(
                v[ids], cfg.t_right, 0, -1.0, cfg.vhs, rng,
                wall_velocity["right"],
            )
            v[ids] = reflected
            if wall_event_handler is not None:
                wall_event_handler(
                    "right", impact, reflected.copy(), state.weight[ids].copy(),
                    wall_velocity["right"],
                )
            changed = True
        bottom = p[:, 1] < 0.0
        if np.any(bottom):
            ids = np.flatnonzero(bottom)
            incoming = v[ids].copy()
            if wall_event_handler is not None:
                normal = incoming[:, 1]
                after = np.divide(
                    p[ids, 1],
                    normal,
                    out=np.zeros(len(ids)),
                    where=np.abs(normal) > 1.0e-30,
                )
                impact = np.clip(
                    p[ids, 0] - incoming[:, 0] * after,
                    0.0,
                    length,
                )
                wall_event_handler(
                    "bottom", impact, incoming, state.weight[ids].copy(),
                    wall_velocity["bottom"],
                )
            p[ids, 1] *= -1.0
            reflected = _diffuse_wall(
                v[ids], cfg.t_bottom, 1, +1.0, cfg.vhs, rng,
                wall_velocity["bottom"],
            )
            v[ids] = reflected
            if wall_event_handler is not None:
                wall_event_handler(
                    "bottom", impact, reflected.copy(), state.weight[ids].copy(),
                    wall_velocity["bottom"],
                )
            changed = True
        top = p[:, 1] >= length
        if np.any(top):
            ids = np.flatnonzero(top)
            incoming = v[ids].copy()
            if wall_event_handler is not None:
                normal = incoming[:, 1]
                after = np.divide(
                    p[ids, 1] - length,
                    normal,
                    out=np.zeros(len(ids)),
                    where=np.abs(normal) > 1.0e-30,
                )
                impact = np.clip(
                    p[ids, 0] - incoming[:, 0] * after,
                    0.0,
                    length,
                )
                wall_event_handler(
                    "top", impact, incoming, state.weight[ids].copy(),
                    wall_velocity["top"],
                )
            p[ids, 1] = 2.0 * length - p[ids, 1]
            reflected = _diffuse_wall(
                v[ids], cfg.t_top, 1, -1.0, cfg.vhs, rng,
                wall_velocity["top"],
            )
            v[ids] = reflected
            if wall_event_handler is not None:
                wall_event_handler(
                    "top", impact, reflected.copy(), state.weight[ids].copy(),
                    wall_velocity["top"],
                )
            changed = True
        if not changed:
            break
    p[:] = np.clip(p, 0.0, np.nextafter(length, 0.0))


def _cell_xy(
    pos: np.ndarray,
    cfg: PhysicalCavityConfig,
) -> tuple[np.ndarray, np.ndarray]:
    ix = np.clip(
        (pos[:, 0] / cfg.length * cfg.nx).astype(np.int64),
        0,
        cfg.nx - 1,
    )
    iy = np.clip(
        (pos[:, 1] / cfg.length * cfg.ny).astype(np.int64),
        0,
        cfg.ny - 1,
    )
    return ix, iy
