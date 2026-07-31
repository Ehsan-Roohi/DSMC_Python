from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
import json
import numpy as np

from .dvm_shakhov import (
    ShakhovReferenceConfig,
    _macroscopic,
    _velocity_grid,
)


@lru_cache(maxsize=32)
def _temperature_response(
    nv: int,
    velocity_extent: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the discrete temperature response of zero-mean Maxwellians.

    The first array is the measured second-moment temperature on the selected
    velocity quadrature; the second array is the Maxwellian parameter that
    generated it. The map is monotone and is inverted before every equilibrium
    and diffuse-wall reconstruction.
    """
    cfg = ShakhovReferenceConfig(nv=nv, velocity_extent=velocity_extent)
    vx, vy, vz, dv = _velocity_grid(cfg)
    c2 = vx * vx + vy * vy + vz * vz
    parameter = np.geomspace(0.03, 5.0, 1024)
    measured = np.empty_like(parameter)
    measure = dv**3
    for index, theta in enumerate(parameter):
        base = np.exp(-c2 / (2.0 * theta))
        base /= (2.0 * np.pi * theta) ** 1.5
        base /= np.sum(base) * measure
        measured[index] = np.sum(base * c2) * measure / 3.0
    order = np.argsort(measured)
    measured = measured[order]
    parameter = parameter[order]
    keep = np.concatenate(([True], np.diff(measured) > 1.0e-12))
    return measured[keep], parameter[keep]


def _parameter_temperature(
    target_temperature: np.ndarray,
    vx: np.ndarray,
    dv: float,
) -> np.ndarray:
    nv = int(vx.shape[0])
    velocity_extent = float(np.max(np.abs(vx)) + 0.5 * dv)
    measured, parameter = _temperature_response(nv, velocity_extent)
    return np.interp(
        np.asarray(target_temperature, dtype=np.float64),
        measured,
        parameter,
        left=parameter[0],
        right=parameter[-1],
    )


def reconstruct_temperature(
    measured_temperature: np.ndarray,
    cfg: ShakhovReferenceConfig,
) -> np.ndarray:
    """Invert a raw quadrature temperature into its Maxwellian parameter."""
    measured_curve, parameter_curve = _temperature_response(
        cfg.nv,
        cfg.velocity_extent,
    )
    nondimensional = (
        np.asarray(measured_temperature, dtype=np.float64)
        / cfg.reference_temperature
    )
    reconstructed = np.interp(
        nondimensional,
        measured_curve,
        parameter_curve,
        left=parameter_curve[0],
        right=parameter_curve[-1],
    )
    return reconstructed * cfg.reference_temperature


def _discrete_maxwellian(
    rho: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    target_temperature: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    dv: float,
) -> np.ndarray:
    """Build a mass-normalized Maxwellian with calibrated discrete energy."""
    parameter_temperature = np.maximum(
        _parameter_temperature(target_temperature, vx, dv),
        1.0e-10,
    )
    cx = vx[None, None] - u[:, :, None, None, None]
    cy = vy[None, None] - v[:, :, None, None, None]
    cz = vz[None, None] - w[:, :, None, None, None]
    c2 = cx * cx + cy * cy + cz * cz
    theta = parameter_temperature[:, :, None, None, None]
    base = np.exp(-c2 / (2.0 * theta))
    base /= (2.0 * np.pi * theta) ** 1.5
    norm = np.sum(base, axis=(-3, -2, -1)) * dv**3
    return (
        rho[:, :, None, None, None]
        * base
        / np.maximum(norm[:, :, None, None, None], 1.0e-30)
    )


def _shakhov_equilibrium(
    fields: dict[str, np.ndarray],
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    dv: float,
    prandtl: float,
) -> np.ndarray:
    maxwellian = _discrete_maxwellian(
        fields["rho"],
        fields["u"],
        fields["v"],
        fields["w"],
        fields["T"],
        vx,
        vy,
        vz,
        dv,
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
        / (
            5.0
            * pressure[:, :, None, None, None]
            * temperature[:, :, None, None, None]
        )
        * (c2 / temperature[:, :, None, None, None] - 5.0)
    )
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
    one = np.ones((1, 1))
    zero = np.zeros((1, 1))
    return _discrete_maxwellian(
        one,
        zero,
        zero,
        zero,
        np.full((1, 1), temperature),
        vx,
        vy,
        vz,
        dv,
    )[0, 0]


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
        vx[None] * distribution[:, 0] * negative_x[None],
        axis=(-3, -2, -1),
    ) * measure
    incoming_unit = float(np.sum(vx * wall[0] * positive_x) * measure)
    left = (
        -outgoing / max(incoming_unit, 1.0e-14)
    )[:, None, None, None] * wall[0][None]

    outgoing = np.sum(
        vx[None] * distribution[:, -1] * positive_x[None],
        axis=(-3, -2, -1),
    ) * measure
    incoming_unit = float(np.sum((-vx) * wall[1] * negative_x) * measure)
    right = (
        outgoing / max(incoming_unit, 1.0e-14)
    )[:, None, None, None] * wall[1][None]

    outgoing = np.sum(
        vy[None] * distribution[0] * negative_y[None],
        axis=(-3, -2, -1),
    ) * measure
    incoming_unit = float(np.sum(vy * wall[2] * positive_y) * measure)
    bottom = (
        -outgoing / max(incoming_unit, 1.0e-14)
    )[:, None, None, None] * wall[2][None]

    outgoing = np.sum(
        vy[None] * distribution[-1] * positive_y[None],
        axis=(-3, -2, -1),
    ) * measure
    incoming_unit = float(np.sum((-vy) * wall[3] * negative_y) * measure)
    top = (
        outgoing / max(incoming_unit, 1.0e-14)
    )[:, None, None, None] * wall[3][None]
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
    target_temperature = np.ones_like(rho)
    distribution = _discrete_maxwellian(
        rho,
        zeros,
        zeros,
        zeros,
        target_temperature,
        vx,
        vy,
        vz,
        dv,
    )
    dx, dy = 1.0 / cfg.nx, 1.0 / cfg.ny
    maximum_speed = max(
        float(np.max(np.abs(vx))),
        float(np.max(np.abs(vy))),
    )
    relaxation_time = cfg.knudsen
    dt = cfg.cfl * min(
        dx / maximum_speed,
        dy / maximum_speed,
        relaxation_time,
    )
    positive_x = (vx > 0.0)[None, None]
    positive_y = (vy > 0.0)[None, None]
    initial_fields = _macroscopic(distribution, vx, vy, vz, dv)
    previous_temperature = initial_fields["T"].copy()
    previous_heat_flux = np.zeros((cfg.ny, cfg.nx, 2))
    residual_history: list[float] = []

    for step in range(cfg.max_steps):
        left, right, bottom, top = _wall_incoming(
            distribution,
            cfg,
            vx,
            vy,
            vz,
            dv,
        )
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
            vx[None, None] * dfdx
            + vy[None, None] * dfdy
        )
        transported = np.maximum(transported, cfg.positivity_floor)
        fields = _macroscopic(transported, vx, vy, vz, dv)
        equilibrium = _shakhov_equilibrium(
            fields,
            vx,
            vy,
            vz,
            dv,
            cfg.prandtl,
        )
        fraction = min(dt / relaxation_time, 1.0)
        distribution = transported + fraction * (
            equilibrium - transported
        )
        distribution = np.maximum(
            distribution,
            cfg.positivity_floor,
        )

        if (step + 1) % cfg.check_interval == 0:
            fields = _macroscopic(distribution, vx, vy, vz, dv)
            temperature_residual = float(
                np.max(
                    np.abs(fields["T"] - previous_temperature)
                )
                / max(
                    float(np.max(np.abs(fields["T"]))),
                    1.0e-14,
                )
            )
            heat_flux = np.stack(
                [fields["qx"], fields["qy"]],
                axis=-1,
            )
            heat_flux_residual = float(
                np.max(np.abs(heat_flux - previous_heat_flux))
                / max(
                    float(np.max(np.abs(heat_flux))),
                    1.0e-12,
                )
            )
            residual = max(
                temperature_residual,
                heat_flux_residual,
            )
            residual_history.append(residual)
            previous_temperature = fields["T"].copy()
            previous_heat_flux = heat_flux.copy()
            if (
                step + 1 >= cfg.minimum_steps
                and residual < cfg.tolerance
            ):
                break

    fields = _macroscopic(distribution, vx, vy, vz, dv)
    density = fields["rho"] / max(
        float(np.mean(fields["rho"])),
        1.0e-14,
    )
    velocity_scale = cfg.velocity_scale
    heat_flux_scale = velocity_scale**3
    physical_temperature = (
        fields["T"] * cfg.reference_temperature
    )
    return {
        "T": physical_temperature,
        "T_raw_quadrature": physical_temperature.copy(),
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
    arrays = {
        key: value
        for key, value in result.items()
        if isinstance(value, np.ndarray)
    }
    arrays["iterations"] = np.int64(result["iterations"])
    arrays["dt"] = np.float64(result["dt"])
    np.savez_compressed(output_path, **arrays)
    metadata = {
        "model": "deterministic_2d_space_3d_velocity_shakhov_dvm",
        "temperature_calibration": "in_core_inverse_discrete_Maxwellian_second_moment",
        "status": "reference_pilot",
        "config": asdict(cfg),
        "iterations": int(result["iterations"]),
        "final_residual": (
            float(result["residual_history"][-1])
            if len(result["residual_history"])
            else None
        ),
        "field_contract": [
            "T",
            "rho",
            "u",
            "v",
            "qx",
            "qy",
        ],
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return output_path


__all__ = [
    "ShakhovReferenceConfig",
    "reconstruct_temperature",
    "solve_shakhov_reference",
    "save_shakhov_reference",
]
