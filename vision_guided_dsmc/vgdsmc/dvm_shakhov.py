from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import numpy as np

KB = 1.380649e-23
MASS_AR = 39.948e-3 / 6.02214076e23


@dataclass(frozen=True)
class ShakhovReferenceConfig:
    """Deterministic 2-D-space/3-D-velocity Shakhov thermal cavity.

    The solver is dimensionless internally, uses a monatomic Prandtl number of
    2/3 by default, and writes temperature in kelvin and velocity in m/s.
    """

    nx: int = 12
    ny: int = 12
    nv: int = 8
    velocity_extent: float = 5.0
    knudsen: float = 0.10
    prandtl: float = 2.0 / 3.0
    t_left: float = 330.0
    t_right: float = 270.0
    t_top: float = 300.0
    t_bottom: float = 300.0
    max_steps: int = 1800
    cfl: float = 0.30
    tolerance: float = 3.0e-6
    check_interval: int = 50
    minimum_steps: int = 250
    positivity_floor: float = 1.0e-30

    @property
    def reference_temperature(self) -> float:
        return 0.25 * (self.t_left + self.t_right + self.t_top + self.t_bottom)

    @property
    def velocity_scale(self) -> float:
        return float(np.sqrt(KB * self.reference_temperature / MASS_AR))


def _velocity_grid(
    cfg: ShakhovReferenceConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    dv = 2.0 * cfg.velocity_extent / cfg.nv
    values = -cfg.velocity_extent + dv * (np.arange(cfg.nv) + 0.5)
    vx, vy, vz = np.meshgrid(values, values, values, indexing="ij")
    return vx, vy, vz, dv


def _discrete_maxwellian(
    rho: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    temperature: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    dv: float,
) -> np.ndarray:
    temperature = np.maximum(temperature, 1.0e-10)
    cx = vx[None, None] - u[:, :, None, None, None]
    cy = vy[None, None] - v[:, :, None, None, None]
    cz = vz[None, None] - w[:, :, None, None, None]
    c2 = cx * cx + cy * cy + cz * cz
    base = np.exp(-c2 / (2.0 * temperature[:, :, None, None, None]))
    base /= (2.0 * np.pi * temperature[:, :, None, None, None]) ** 1.5
    norm = np.sum(base, axis=(-3, -2, -1)) * dv**3
    return rho[:, :, None, None, None] * base / norm[:, :, None, None, None]


def _macroscopic(
    distribution: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    dv: float,
) -> dict[str, np.ndarray]:
    measure = dv**3
    rho = np.sum(distribution, axis=(-3, -2, -1)) * measure
    safe_rho = np.maximum(rho, 1.0e-14)
    u = np.sum(distribution * vx[None, None], axis=(-3, -2, -1)) * measure / safe_rho
    v = np.sum(distribution * vy[None, None], axis=(-3, -2, -1)) * measure / safe_rho
    w = np.sum(distribution * vz[None, None], axis=(-3, -2, -1)) * measure / safe_rho
    cx = vx[None, None] - u[:, :, None, None, None]
    cy = vy[None, None] - v[:, :, None, None, None]
    cz = vz[None, None] - w[:, :, None, None, None]
    c2 = cx * cx + cy * cy + cz * cz
    temperature = (
        np.sum(distribution * c2, axis=(-3, -2, -1)) * measure
        / (3.0 * safe_rho)
    )
    qx = 0.5 * np.sum(distribution * cx * c2, axis=(-3, -2, -1)) * measure
    qy = 0.5 * np.sum(distribution * cy * c2, axis=(-3, -2, -1)) * measure
    qz = 0.5 * np.sum(distribution * cz * c2, axis=(-3, -2, -1)) * measure
    return {
        "rho": rho,
        "u": u,
        "v": v,
        "w": w,
        "T": temperature,
        "qx": qx,
        "qy": qy,
        "qz": qz,
    }


def _shakhov_equilibrium(
    fields: dict[str, np.ndarray],
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    dv: float,
    prandtl: float,
) -> np.ndarray:
    maxwellian = _discrete_maxwellian(
        fields["rho"], fields["u"], fields["v"], fields["w"], fields["T"],
        vx, vy, vz, dv,
    )
    temperature = np.maximum(fields["T"], 1.0e-10)
    pressure = np.maximum(fields["rho"] * temperature, 1.0e-14)
    cx = vx[None, None] - fields["u"][:, :, None, None, None]
    cy = vy[None, None] - fields["v"][:, :, None, None, None]
    cz = vz[None, None] - fields["w"][:, :, None, None, None]
    c2 = cx * cx + cy * cy + cz * cz
    c_dot_q = (
        cx * fields["qx"][:, :, None, None, None]
        + cy * fields["qy"][:, :, None, None, None]
        + cz * fields["qz"][:, :, None, None, None]
    )
    correction = (
        (1.0 - prandtl)
        * c_dot_q
        / (5.0 * pressure[:, :, None, None, None] * temperature[:, :, None, None, None])
        * (c2 / temperature[:, :, None, None, None] - 5.0)
    )
    # Strongly under-resolved tails can make the polynomial correction negative.
    # Clipping is a positivity safeguard for this reference pilot.
    factor = np.maximum(1.0 + correction, 0.05)
    equilibrium = maxwellian * factor
    density = np.sum(equilibrium, axis=(-3, -2, -1)) * dv**3
    return equilibrium * (
        fields["rho"] / np.maximum(density, 1.0e-14)
    )[:, :, None, None, None]


def _unit_wall_maxwellian(
    temperature: float,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    dv: float,
) -> np.ndarray:
    base = np.exp(-(vx * vx + vy * vy + vz * vz) / (2.0 * temperature))
    base /= (2.0 * np.pi * temperature) ** 1.5
    return base / (np.sum(base) * dv**3)


def _wall_incoming(
    distribution: np.ndarray,
    cfg: ShakhovReferenceConfig,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    dv: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scale = cfg.reference_temperature
    temperatures = (
        cfg.t_left / scale,
        cfg.t_right / scale,
        cfg.t_bottom / scale,
        cfg.t_top / scale,
    )
    wall = [
        _unit_wall_maxwellian(value, vx, vy, vz, dv)
        for value in temperatures
    ]
    positive_x, negative_x = vx > 0.0, vx < 0.0
    positive_y, negative_y = vy > 0.0, vy < 0.0
    measure = dv**3

    outgoing = np.sum(
        vx[None] * distribution[:, 0] * negative_x[None], axis=(-3, -2, -1)
    ) * measure
    incoming_unit = float(np.sum(vx * wall[0] * positive_x) * measure)
    left = (-outgoing / max(incoming_unit, 1.0e-14))[:, None, None, None] * wall[0][None]

    outgoing = np.sum(
        vx[None] * distribution[:, -1] * positive_x[None], axis=(-3, -2, -1)
    ) * measure
    incoming_unit = float(np.sum((-vx) * wall[1] * negative_x) * measure)
    right = (outgoing / max(incoming_unit, 1.0e-14))[:, None, None, None] * wall[1][None]

    outgoing = np.sum(
        vy[None] * distribution[0] * negative_y[None], axis=(-3, -2, -1)
    ) * measure
    incoming_unit = float(np.sum(vy * wall[2] * positive_y) * measure)
    bottom = (-outgoing / max(incoming_unit, 1.0e-14))[:, None, None, None] * wall[2][None]

    outgoing = np.sum(
        vy[None] * distribution[-1] * positive_y[None], axis=(-3, -2, -1)
    ) * measure
    incoming_unit = float(np.sum((-vy) * wall[3] * negative_y) * measure)
    top = (outgoing / max(incoming_unit, 1.0e-14))[:, None, None, None] * wall[3][None]
    return left, right, bottom, top


def solve_shakhov_reference(
    cfg: ShakhovReferenceConfig,
) -> dict[str, np.ndarray | float | int]:
    if cfg.nx < 3 or cfg.ny < 3 or cfg.nv < 6:
        raise ValueError("Require nx, ny >= 3 and nv >= 6")
    if cfg.knudsen <= 0.0 or not 0.0 < cfg.prandtl <= 1.0:
        raise ValueError("Require positive Knudsen number and 0 < Pr <= 1")
    if min(cfg.t_left, cfg.t_right, cfg.t_top, cfg.t_bottom) <= 0.0:
        raise ValueError("Wall temperatures must be positive")

    vx, vy, vz, dv = _velocity_grid(cfg)
    rho = np.ones((cfg.ny, cfg.nx))
    zeros = np.zeros_like(rho)
    temperature = np.ones_like(rho)
    distribution = _discrete_maxwellian(
        rho, zeros, zeros, zeros, temperature, vx, vy, vz, dv
    )
    dx, dy = 1.0 / cfg.nx, 1.0 / cfg.ny
    maximum_speed = max(float(np.max(np.abs(vx))), float(np.max(np.abs(vy))))
    relaxation_time = cfg.knudsen
    dt = cfg.cfl * min(dx / maximum_speed, dy / maximum_speed, relaxation_time)
    positive_x = (vx > 0.0)[None, None]
    positive_y = (vy > 0.0)[None, None]
    previous_temperature = temperature.copy()
    previous_heat_flux = np.zeros((cfg.ny, cfg.nx, 2))
    residual_history: list[float] = []

    for step in range(cfg.max_steps):
        left, right, bottom, top = _wall_incoming(distribution, cfg, vx, vy, vz, dv)
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
            vx[None, None] * dfdx + vy[None, None] * dfdy
        )
        transported = np.maximum(transported, cfg.positivity_floor)
        fields = _macroscopic(transported, vx, vy, vz, dv)
        equilibrium = _shakhov_equilibrium(
            fields, vx, vy, vz, dv, cfg.prandtl
        )
        fraction = min(dt / relaxation_time, 1.0)
        distribution = transported + fraction * (equilibrium - transported)
        distribution = np.maximum(distribution, cfg.positivity_floor)

        if (step + 1) % cfg.check_interval == 0:
            fields = _macroscopic(distribution, vx, vy, vz, dv)
            temperature_residual = float(
                np.max(np.abs(fields["T"] - previous_temperature))
                / max(float(np.max(np.abs(fields["T"]))), 1.0e-14)
            )
            heat_flux = np.stack([fields["qx"], fields["qy"]], axis=-1)
            heat_flux_residual = float(
                np.max(np.abs(heat_flux - previous_heat_flux))
                / max(float(np.max(np.abs(heat_flux))), 1.0e-12)
            )
            residual = max(temperature_residual, heat_flux_residual)
            residual_history.append(residual)
            previous_temperature = fields["T"].copy()
            previous_heat_flux = heat_flux.copy()
            if step + 1 >= cfg.minimum_steps and residual < cfg.tolerance:
                break

    fields = _macroscopic(distribution, vx, vy, vz, dv)
    density = fields["rho"] / max(float(np.mean(fields["rho"])), 1.0e-14)
    velocity_scale = cfg.velocity_scale
    heat_flux_scale = velocity_scale**3
    return {
        "T": fields["T"] * cfg.reference_temperature,
        "rho": density,
        "u": fields["u"] * velocity_scale,
        "v": fields["v"] * velocity_scale,
        "w": fields["w"] * velocity_scale,
        "qx": fields["qx"] * heat_flux_scale,
        "qy": fields["qy"] * heat_flux_scale,
        "qz": fields["qz"] * heat_flux_scale,
        "residual_history": np.asarray(residual_history),
        "iterations": step + 1,
        "dt": dt,
    }


def save_shakhov_reference(
    output_path: str | Path,
    cfg: ShakhovReferenceConfig,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = solve_shakhov_reference(cfg)
    arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
    arrays["iterations"] = np.int64(result["iterations"])
    arrays["dt"] = np.float64(result["dt"])
    np.savez_compressed(output_path, **arrays)
    metadata = {
        "model": "deterministic_2d_space_3d_velocity_shakhov_dvm",
        "status": "reference_pilot",
        "config": asdict(cfg),
        "iterations": int(result["iterations"]),
        "final_residual": (
            float(result["residual_history"][-1])
            if len(result["residual_history"])
            else None
        ),
        "field_contract": ["T", "rho", "u", "v", "qx", "qy"],
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return output_path
