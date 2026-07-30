from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import numpy as np

KB = 1.380649e-23
MASS_AR = 39.948e-3 / 6.02214076e23


@dataclass(frozen=True)
class DVMReferenceConfig:
    """Deterministic 2-D BGK discrete-velocity thermal cavity.

    Coordinates and velocities are nondimensional internally. Saved temperature
    and velocity fields are converted to kelvin and m/s. This is a deterministic
    reference pilot, not yet a Shakhov implementation.
    """

    nx: int = 16
    ny: int = 16
    nv: int = 12
    velocity_extent: float = 5.0
    knudsen: float = 0.10
    t_left: float = 330.0
    t_right: float = 270.0
    t_top: float = 300.0
    t_bottom: float = 300.0
    max_steps: int = 2500
    cfl: float = 0.35
    tolerance: float = 2.0e-6
    check_interval: int = 50
    minimum_steps: int = 200

    @property
    def reference_temperature(self) -> float:
        return 0.25 * (self.t_left + self.t_right + self.t_top + self.t_bottom)

    @property
    def velocity_scale(self) -> float:
        return float(np.sqrt(KB * self.reference_temperature / MASS_AR))


def _velocity_grid(cfg: DVMReferenceConfig) -> tuple[np.ndarray, np.ndarray, float]:
    dv = 2.0 * cfg.velocity_extent / cfg.nv
    values = -cfg.velocity_extent + dv * (np.arange(cfg.nv) + 0.5)
    vx, vy = np.meshgrid(values, values, indexing="ij")
    return vx, vy, dv


def _maxwellian(
    rho: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    temperature: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    dv: float,
) -> np.ndarray:
    temperature = np.maximum(temperature, 1.0e-10)
    cx = vx[None, None, :, :] - u[:, :, None, None]
    cy = vy[None, None, :, :] - v[:, :, None, None]
    base = np.exp(-(cx * cx + cy * cy) / (2.0 * temperature[:, :, None, None]))
    base /= 2.0 * np.pi * temperature[:, :, None, None]
    discrete_norm = np.sum(base, axis=(-2, -1)) * dv * dv
    return rho[:, :, None, None] * base / discrete_norm[:, :, None, None]


def _macroscopic(
    distribution: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    dv: float,
) -> dict[str, np.ndarray]:
    weight = dv * dv
    rho = np.sum(distribution, axis=(-2, -1)) * weight
    safe_rho = np.maximum(rho, 1.0e-14)
    u = (
        np.sum(distribution * vx[None, None, :, :], axis=(-2, -1))
        * weight
        / safe_rho
    )
    v = (
        np.sum(distribution * vy[None, None, :, :], axis=(-2, -1))
        * weight
        / safe_rho
    )
    cx = vx[None, None, :, :] - u[:, :, None, None]
    cy = vy[None, None, :, :] - v[:, :, None, None]
    peculiar_squared = cx * cx + cy * cy
    temperature = (
        0.5
        * np.sum(distribution * peculiar_squared, axis=(-2, -1))
        * weight
        / safe_rho
    )
    qx = 0.5 * np.sum(distribution * cx * peculiar_squared, axis=(-2, -1)) * weight
    qy = 0.5 * np.sum(distribution * cy * peculiar_squared, axis=(-2, -1)) * weight
    return {"rho": rho, "u": u, "v": v, "T": temperature, "qx": qx, "qy": qy}


def _unit_wall_maxwellian(
    temperature: float,
    vx: np.ndarray,
    vy: np.ndarray,
    dv: float,
) -> np.ndarray:
    base = np.exp(-(vx * vx + vy * vy) / (2.0 * temperature))
    base /= 2.0 * np.pi * temperature
    return base / (np.sum(base) * dv * dv)


def _wall_incoming(
    distribution: np.ndarray,
    cfg: DVMReferenceConfig,
    vx: np.ndarray,
    vy: np.ndarray,
    dv: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scale = cfg.reference_temperature
    wall_temperatures = (
        cfg.t_left / scale,
        cfg.t_right / scale,
        cfg.t_bottom / scale,
        cfg.t_top / scale,
    )
    left_m = _unit_wall_maxwellian(wall_temperatures[0], vx, vy, dv)
    right_m = _unit_wall_maxwellian(wall_temperatures[1], vx, vy, dv)
    bottom_m = _unit_wall_maxwellian(wall_temperatures[2], vx, vy, dv)
    top_m = _unit_wall_maxwellian(wall_temperatures[3], vx, vy, dv)
    positive_x, negative_x = vx > 0.0, vx < 0.0
    positive_y, negative_y = vy > 0.0, vy < 0.0
    weight = dv * dv

    outgoing = np.sum(
        vx[None, :, :] * distribution[:, 0] * negative_x[None, :, :],
        axis=(-2, -1),
    ) * weight
    unit_flux = float(np.sum(vx * left_m * positive_x) * weight)
    left_density = -outgoing / max(unit_flux, 1.0e-14)
    left = left_density[:, None, None] * left_m[None, :, :]

    outgoing = np.sum(
        vx[None, :, :] * distribution[:, -1] * positive_x[None, :, :],
        axis=(-2, -1),
    ) * weight
    unit_flux = float(np.sum((-vx) * right_m * negative_x) * weight)
    right_density = outgoing / max(unit_flux, 1.0e-14)
    right = right_density[:, None, None] * right_m[None, :, :]

    outgoing = np.sum(
        vy[None, :, :] * distribution[0] * negative_y[None, :, :],
        axis=(-2, -1),
    ) * weight
    unit_flux = float(np.sum(vy * bottom_m * positive_y) * weight)
    bottom_density = -outgoing / max(unit_flux, 1.0e-14)
    bottom = bottom_density[:, None, None] * bottom_m[None, :, :]

    outgoing = np.sum(
        vy[None, :, :] * distribution[-1] * positive_y[None, :, :],
        axis=(-2, -1),
    ) * weight
    unit_flux = float(np.sum((-vy) * top_m * negative_y) * weight)
    top_density = outgoing / max(unit_flux, 1.0e-14)
    top = top_density[:, None, None] * top_m[None, :, :]
    return left, right, bottom, top


def solve_dvm_reference(cfg: DVMReferenceConfig) -> dict[str, np.ndarray | float | int]:
    if cfg.nx < 3 or cfg.ny < 3 or cfg.nv < 6:
        raise ValueError("Require nx, ny >= 3 and nv >= 6")
    if cfg.knudsen <= 0.0 or cfg.max_steps <= 0:
        raise ValueError("knudsen and max_steps must be positive")
    if min(cfg.t_left, cfg.t_right, cfg.t_top, cfg.t_bottom) <= 0.0:
        raise ValueError("Wall temperatures must be positive")

    vx, vy, dv = _velocity_grid(cfg)
    rho = np.ones((cfg.ny, cfg.nx))
    u = np.zeros_like(rho)
    v = np.zeros_like(rho)
    temperature = np.ones_like(rho)
    distribution = _maxwellian(rho, u, v, temperature, vx, vy, dv)

    dx, dy = 1.0 / cfg.nx, 1.0 / cfg.ny
    maximum_speed = max(float(np.max(np.abs(vx))), float(np.max(np.abs(vy))))
    relaxation_time = cfg.knudsen
    dt = cfg.cfl * min(dx / maximum_speed, dy / maximum_speed, relaxation_time)
    positive_x = (vx > 0.0)[None, None, :, :]
    positive_y = (vy > 0.0)[None, None, :, :]
    previous_temperature = temperature.copy()
    residual_history: list[float] = []

    for step in range(cfg.max_steps):
        left, right, bottom, top = _wall_incoming(distribution, cfg, vx, vy, dv)
        left_neighbor = np.empty_like(distribution)
        right_neighbor = np.empty_like(distribution)
        bottom_neighbor = np.empty_like(distribution)
        top_neighbor = np.empty_like(distribution)
        left_neighbor[:, 1:] = distribution[:, :-1]
        left_neighbor[:, 0] = left
        right_neighbor[:, :-1] = distribution[:, 1:]
        right_neighbor[:, -1] = right
        bottom_neighbor[1:] = distribution[:-1]
        bottom_neighbor[0] = bottom
        top_neighbor[:-1] = distribution[1:]
        top_neighbor[-1] = top

        dfdx = np.where(
            positive_x,
            (distribution - left_neighbor) / dx,
            (right_neighbor - distribution) / dx,
        )
        dfdy = np.where(
            positive_y,
            (distribution - bottom_neighbor) / dy,
            (top_neighbor - distribution) / dy,
        )
        transported = distribution - dt * (
            vx[None, None, :, :] * dfdx + vy[None, None, :, :] * dfdy
        )
        transported = np.maximum(transported, 1.0e-30)
        fields = _macroscopic(transported, vx, vy, dv)
        equilibrium = _maxwellian(
            fields["rho"], fields["u"], fields["v"], fields["T"], vx, vy, dv
        )
        collision_fraction = min(dt / relaxation_time, 1.0)
        distribution = transported + collision_fraction * (equilibrium - transported)
        distribution = np.maximum(distribution, 1.0e-30)

        if (step + 1) % cfg.check_interval == 0:
            fields = _macroscopic(distribution, vx, vy, dv)
            residual = float(
                np.max(np.abs(fields["T"] - previous_temperature))
                / max(float(np.max(np.abs(fields["T"]))), 1.0e-14)
            )
            residual_history.append(residual)
            previous_temperature = fields["T"].copy()
            if step + 1 >= cfg.minimum_steps and residual < cfg.tolerance:
                break

    fields = _macroscopic(distribution, vx, vy, dv)
    temperature_scale = cfg.reference_temperature
    velocity_scale = cfg.velocity_scale
    density = fields["rho"] / max(float(np.mean(fields["rho"])), 1.0e-14)
    return {
        "T": fields["T"] * temperature_scale,
        "rho": density,
        "u": fields["u"] * velocity_scale,
        "v": fields["v"] * velocity_scale,
        "qx": fields["qx"],
        "qy": fields["qy"],
        "residual_history": np.asarray(residual_history),
        "iterations": step + 1,
        "dt": dt,
    }


def save_dvm_reference(output_path: str | Path, cfg: DVMReferenceConfig) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = solve_dvm_reference(cfg)
    arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
    arrays["iterations"] = np.int64(result["iterations"])
    arrays["dt"] = np.float64(result["dt"])
    np.savez_compressed(output_path, **arrays)
    metadata = {
        "model": "deterministic_2d_bgk_dvm",
        "status": "reference_pilot_not_shakhov",
        "config": asdict(cfg),
        "iterations": int(result["iterations"]),
        "final_residual": (
            float(result["residual_history"][-1])
            if len(result["residual_history"])
            else None
        ),
        "field_contract": ["T", "rho", "u", "v"],
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return output_path
