from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import math
import numpy as np

from .linear_sidewall_validation import (
    LinearSidewallConfig,
    TABLE3_UY_RATIO_0P1,
    TABLE3_Y,
    TABLE6_QAV_RATIO_0P1,
    sidewall_temperature_profile,
)
from .stage32_near_continuum_observable_audit import (
    observable_metrics,
    wall_observable_profiles,
)
from .velocity_quadrature_audit import VelocityQuadrature, spherical_product


STAGE35_KNUDSEN = (0.1, 1.0, 10.0)
STAGE35_GRID = (12, 12)
STAGE35_RATIO = 0.1
STAGE35_QUADRATURE = "spherical_matched_r16_mu12_phi24"
SQRT2 = math.sqrt(2.0)


@dataclass(frozen=True)
class CoordinateSystem:
    """Velocity-coordinate representation of the same physical c0-scaled state.

    ``coordinate_to_c0`` maps stored velocity ordinates to
    c = xi/sqrt(k*T0/m). Thus it is one for solver c0 coordinates and
    sqrt(2) for the paper zeta coordinates.
    """

    name: str
    quadrature: VelocityQuadrature
    coordinate_to_c0: float

    @property
    def distribution_scale_from_c0(self) -> float:
        return self.coordinate_to_c0**3

    @property
    def floor_scale_from_c0(self) -> float:
        return self.distribution_scale_from_c0


def transform_quadrature_from_c0(
    quadrature: VelocityQuadrature,
    coordinate_to_c0: float,
    name: str,
) -> VelocityQuadrature:
    """Represent one physical c0 quadrature in a rescaled velocity coordinate."""
    if coordinate_to_c0 <= 0.0 or not math.isfinite(coordinate_to_c0):
        raise ValueError("coordinate_to_c0 must be finite and positive")
    scale = float(coordinate_to_c0)
    return VelocityQuadrature(
        name=name,
        vx=np.asarray(quadrature.vx, dtype=np.float64) / scale,
        vy=np.asarray(quadrature.vy, dtype=np.float64) / scale,
        vz=np.asarray(quadrature.vz, dtype=np.float64) / scale,
        weight=np.asarray(quadrature.weight, dtype=np.float64) / scale**3,
        family=f"{quadrature.family}_coordinate_transform",
    )


def build_stage35_coordinate_systems() -> tuple[CoordinateSystem, CoordinateSystem]:
    base = spherical_product(16, 12, 24, 5.0, STAGE35_QUADRATURE)
    c0 = CoordinateSystem("c0", base, 1.0)
    zeta = CoordinateSystem(
        "paper_zeta",
        transform_quadrature_from_c0(base, SQRT2, f"{STAGE35_QUADRATURE}_zeta"),
        SQRT2,
    )
    return c0, zeta


def _physical_velocities(
    system: CoordinateSystem,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alpha = system.coordinate_to_c0
    return (
        alpha * system.quadrature.vx,
        alpha * system.quadrature.vy,
        alpha * system.quadrature.vz,
    )


def scaled_discrete_maxwellian(
    rho: np.ndarray,
    u_c0: np.ndarray,
    v_c0: np.ndarray,
    w_c0: np.ndarray,
    temperature: np.ndarray,
    system: CoordinateSystem,
) -> np.ndarray:
    """Maxwellian in a coordinate system, normalized to the same density."""
    rho = np.asarray(rho, dtype=np.float64)
    u_c0 = np.asarray(u_c0, dtype=np.float64)
    v_c0 = np.asarray(v_c0, dtype=np.float64)
    w_c0 = np.asarray(w_c0, dtype=np.float64)
    temperature = np.maximum(np.asarray(temperature, dtype=np.float64), 1.0e-10)
    if not (rho.shape == u_c0.shape == v_c0.shape == w_c0.shape == temperature.shape):
        raise ValueError("all macroscopic fields must have matching shapes")
    vx, vy, vz = _physical_velocities(system)
    cx = vx[None, None, :] - u_c0[..., None]
    cy = vy[None, None, :] - v_c0[..., None]
    cz = vz[None, None, :] - w_c0[..., None]
    c2 = cx * cx + cy * cy + cz * cz
    base = (
        system.distribution_scale_from_c0
        * np.exp(-c2 / (2.0 * temperature[..., None]))
        / (2.0 * math.pi * temperature[..., None]) ** 1.5
    )
    norm = np.sum(
        base * system.quadrature.weight[None, None, :],
        axis=-1,
    )
    return rho[..., None] * base / np.maximum(norm[..., None], 1.0e-300)


def scaled_macroscopic(
    distribution: np.ndarray,
    system: CoordinateSystem,
) -> dict[str, np.ndarray]:
    distribution = np.asarray(distribution, dtype=np.float64)
    if distribution.ndim != 3:
        raise ValueError("distribution must have shape (ny,nx,nq)")
    if distribution.shape[-1] != system.quadrature.point_count:
        raise ValueError("distribution and quadrature point counts must match")
    weight = system.quadrature.weight[None, None, :]
    rho = np.sum(distribution * weight, axis=-1)
    safe_rho = np.maximum(rho, 1.0e-14)
    vx, vy, vz = _physical_velocities(system)
    vx3, vy3, vz3 = vx[None, None, :], vy[None, None, :], vz[None, None, :]
    u = np.sum(distribution * vx3 * weight, axis=-1) / safe_rho
    v = np.sum(distribution * vy3 * weight, axis=-1) / safe_rho
    w = np.sum(distribution * vz3 * weight, axis=-1) / safe_rho
    cx = vx3 - u[..., None]
    cy = vy3 - v[..., None]
    cz = vz3 - w[..., None]
    c2 = cx * cx + cy * cy + cz * cz
    temperature = np.sum(distribution * c2 * weight, axis=-1) / (3.0 * safe_rho)
    qx = 0.5 * np.sum(distribution * cx * c2 * weight, axis=-1)
    qy = 0.5 * np.sum(distribution * cy * c2 * weight, axis=-1)
    qz = 0.5 * np.sum(distribution * cz * c2 * weight, axis=-1)
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


def scaled_shakhov_equilibrium(
    fields: dict[str, np.ndarray],
    system: CoordinateSystem,
    prandtl: float,
) -> np.ndarray:
    if not 0.0 < prandtl <= 1.0:
        raise ValueError("Prandtl number must lie in (0,1]")
    equilibrium = scaled_discrete_maxwellian(
        fields["rho"],
        fields["u"],
        fields["v"],
        fields["w"],
        fields["T"],
        system,
    )
    temperature = np.maximum(fields["T"], 1.0e-10)
    pressure = np.maximum(fields["rho"] * temperature, 1.0e-14)
    vx, vy, vz = _physical_velocities(system)
    cx = vx[None, None, :] - fields["u"][..., None]
    cy = vy[None, None, :] - fields["v"][..., None]
    cz = vz[None, None, :] - fields["w"][..., None]
    c2 = cx * cx + cy * cy + cz * cz
    c_dot_q = (
        cx * fields["qx"][..., None]
        + cy * fields["qy"][..., None]
        + cz * fields["qz"][..., None]
    )
    correction = (
        (1.0 - prandtl)
        * c_dot_q
        / (5.0 * pressure[..., None] * temperature[..., None])
        * (c2 / temperature[..., None] - 5.0)
    )
    equilibrium = equilibrium * np.maximum(1.0 + correction, 0.05)
    density = np.sum(
        equilibrium * system.quadrature.weight[None, None, :],
        axis=-1,
    )
    return equilibrium * (
        fields["rho"] / np.maximum(density, 1.0e-14)
    )[..., None]


def scaled_unit_wall_maxwellian(
    temperature: float,
    system: CoordinateSystem,
) -> np.ndarray:
    if temperature <= 0.0:
        raise ValueError("wall temperature must be positive")
    vx, vy, vz = _physical_velocities(system)
    speed2 = vx * vx + vy * vy + vz * vz
    base = (
        system.distribution_scale_from_c0
        * np.exp(-speed2 / (2.0 * temperature))
        / (2.0 * math.pi * temperature) ** 1.5
    )
    norm = float(np.sum(base * system.quadrature.weight))
    return base / max(norm, 1.0e-300)


def _profile_wall_incoming(
    outgoing_distribution: np.ndarray,
    normal_coordinate_velocity: np.ndarray,
    incoming_mask: np.ndarray,
    outgoing_mask: np.ndarray,
    wall_maxwellians: np.ndarray,
    system: CoordinateSystem,
) -> np.ndarray:
    weight = system.quadrature.weight[None, :]
    outgoing_flux = np.sum(
        normal_coordinate_velocity[None, :]
        * outgoing_distribution
        * outgoing_mask[None, :]
        * weight,
        axis=-1,
    )
    incoming_unit = np.sum(
        normal_coordinate_velocity[None, :]
        * wall_maxwellians
        * incoming_mask[None, :]
        * weight,
        axis=-1,
    )
    scale = -outgoing_flux / np.maximum(incoming_unit, 1.0e-14)
    return scale[:, None] * wall_maxwellians


def scaled_wall_incoming(
    distribution: np.ndarray,
    cfg: LinearSidewallConfig,
    system: CoordinateSystem,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q = system.quadrature
    side_wall = np.stack(
        [
            scaled_unit_wall_maxwellian(float(t), system)
            for t in sidewall_temperature_profile(cfg)
        ]
    )
    bottom_wall = np.repeat(
        scaled_unit_wall_maxwellian(cfg.hot_temperature, system)[None, :],
        cfg.nx,
        axis=0,
    )
    top_wall = np.repeat(
        scaled_unit_wall_maxwellian(cfg.cold_temperature, system)[None, :],
        cfg.nx,
        axis=0,
    )
    px, nx = q.vx > 0.0, q.vx < 0.0
    py, ny = q.vy > 0.0, q.vy < 0.0
    left = _profile_wall_incoming(
        distribution[:, 0], q.vx, px, nx, side_wall, system
    )
    right = _profile_wall_incoming(
        distribution[:, -1], -q.vx, nx, px, side_wall, system
    )
    bottom = _profile_wall_incoming(
        distribution[0], q.vy, py, ny, bottom_wall, system
    )
    top = _profile_wall_incoming(
        distribution[-1], -q.vy, ny, py, top_wall, system
    )
    return left, right, bottom, top


def coordinate_tau_prefactor(kn0: float, system: CoordinateSystem) -> float:
    """Relaxation time in the selected coordinate's streaming equation."""
    if kn0 <= 0.0:
        raise ValueError("kn0 must be positive")
    return (
        system.coordinate_to_c0
        * SQRT2
        * float(kn0)
        / math.sqrt(math.pi)
    )


def scaled_local_relaxation_time(
    density: np.ndarray,
    temperature: np.ndarray,
    cfg: LinearSidewallConfig,
    system: CoordinateSystem,
) -> np.ndarray:
    density = np.maximum(np.asarray(density, dtype=np.float64), 1.0e-12)
    temperature = np.maximum(np.asarray(temperature, dtype=np.float64), 1.0e-10)
    return (
        coordinate_tau_prefactor(cfg.kn0, system)
        * temperature ** (cfg.viscosity_exponent - 1.0)
        / density
    )


def scaled_wall_mass_balance_error(
    interior_outgoing: np.ndarray,
    incoming: np.ndarray,
    normal_coordinate_velocity: np.ndarray,
    incoming_mask: np.ndarray,
    system: CoordinateSystem,
) -> float:
    boundary = np.where(incoming_mask[None, :], incoming, interior_outgoing)
    weighted_flux = (
        boundary
        * normal_coordinate_velocity[None, :]
        * system.quadrature.weight[None, :]
    )
    net = np.sum(weighted_flux, axis=-1)
    scale = np.maximum(np.sum(np.abs(weighted_flux), axis=-1), 1.0e-14)
    return float(np.max(np.abs(net) / scale))


def scaled_left_wall_tangential_velocity(
    distribution: np.ndarray,
    left_incoming: np.ndarray,
    system: CoordinateSystem,
) -> np.ndarray:
    boundary = np.where(
        (system.quadrature.vx > 0.0)[None, :],
        left_incoming,
        distribution[:, 0],
    )
    physical_v_c0 = scaled_macroscopic(boundary[:, None, :], system)["v"][:, 0]
    return np.asarray(physical_v_c0) / SQRT2


def scaled_bottom_heat_flux(
    distribution: np.ndarray,
    bottom_incoming: np.ndarray,
    system: CoordinateSystem,
) -> np.ndarray:
    boundary = np.where(
        (system.quadrature.vy > 0.0)[None, :],
        bottom_incoming,
        distribution[0],
    )
    vx, vy, vz = _physical_velocities(system)
    speed2 = vx * vx + vy * vy + vz * vz
    flux_c0 = 0.5 * np.sum(
        boundary
        * vy[None, :]
        * speed2[None, :]
        * system.quadrature.weight[None, :],
        axis=-1,
    )
    return flux_c0 / SQRT2


def solve_scaled_coordinate_case(
    cfg: LinearSidewallConfig,
    system: CoordinateSystem,
) -> dict[str, object]:
    if cfg.nx < 3 or cfg.ny < 3:
        raise ValueError("nx and ny must be at least three")
    rho = np.ones((cfg.ny, cfg.nx))
    zero = np.zeros_like(rho)
    initial_temperature = np.repeat(
        sidewall_temperature_profile(cfg)[:, None],
        cfg.nx,
        axis=1,
    )
    distribution = scaled_discrete_maxwellian(
        rho, zero, zero, zero, initial_temperature, system
    )
    dx, dy = 1.0 / cfg.nx, 1.0 / cfg.ny
    q = system.quadrature
    maximum_speed = max(
        float(np.max(np.abs(q.vx))),
        float(np.max(np.abs(q.vy))),
    )
    dt = cfg.cfl * min(dx / maximum_speed, dy / maximum_speed)
    positive_x = (q.vx > 0.0)[None, None, :]
    positive_y = (q.vy > 0.0)[None, None, :]
    previous = scaled_macroscopic(distribution, system)
    previous_temperature = previous["T"].copy()
    previous_velocity = np.stack([previous["u"], previous["v"]], axis=-1)
    previous_heat_flux = np.stack([previous["qx"], previous["qy"]], axis=-1)
    residual_history: list[float] = []
    converged = False
    coordinate_floor = cfg.positivity_floor * system.floor_scale_from_c0

    for step in range(cfg.max_steps):
        left, right, bottom, top = scaled_wall_incoming(distribution, cfg, system)
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
        transported = np.maximum(
            distribution
            - dt
            * (
                q.vx[None, None, :] * dfdx
                + q.vy[None, None, :] * dfdy
            ),
            coordinate_floor,
        )
        fields = scaled_macroscopic(transported, system)
        equilibrium = scaled_shakhov_equilibrium(fields, system, cfg.prandtl)
        tau = scaled_local_relaxation_time(
            fields["rho"], fields["T"], cfg, system
        )
        fraction = np.minimum(dt / tau, 1.0)[..., None]
        distribution = np.maximum(
            transported + fraction * (equilibrium - transported),
            coordinate_floor,
        )
        if (step + 1) % cfg.check_interval == 0:
            fields = scaled_macroscopic(distribution, system)
            velocity = np.stack([fields["u"], fields["v"]], axis=-1)
            heat_flux = np.stack([fields["qx"], fields["qy"]], axis=-1)
            change = max(
                float(np.max(np.abs(fields["T"] - previous_temperature))),
                float(np.max(np.abs(velocity - previous_velocity))),
                float(np.max(np.abs(heat_flux - previous_heat_flux))),
            )
            residual_history.append(change)
            previous_temperature = fields["T"].copy()
            previous_velocity = velocity.copy()
            previous_heat_flux = heat_flux.copy()
            if step + 1 >= cfg.minimum_steps and change < cfg.tolerance:
                converged = True
                break

    left, right, bottom, top = scaled_wall_incoming(distribution, cfg, system)
    fields = scaled_macroscopic(distribution, system)
    wall_velocity = scaled_left_wall_tangential_velocity(
        distribution, left, system
    )
    y_centers = (np.arange(cfg.ny, dtype=np.float64) + 0.5) / cfg.ny
    table_velocity = np.interp(TABLE3_Y, y_centers, wall_velocity)
    wall_balance = max(
        scaled_wall_mass_balance_error(
            distribution[:, 0], left, q.vx, q.vx > 0.0, system
        ),
        scaled_wall_mass_balance_error(
            distribution[:, -1], right, -q.vx, q.vx < 0.0, system
        ),
        scaled_wall_mass_balance_error(
            distribution[0], bottom, q.vy, q.vy > 0.0, system
        ),
        scaled_wall_mass_balance_error(
            distribution[-1], top, -q.vy, q.vy < 0.0, system
        ),
    )
    return {
        "distribution": distribution,
        "T": np.asarray(fields["T"]),
        "rho": np.asarray(fields["rho"]) / np.mean(fields["rho"]),
        "u": np.asarray(fields["u"]) / SQRT2,
        "v": np.asarray(fields["v"]) / SQRT2,
        "qx": np.asarray(fields["qx"]) / SQRT2,
        "qy": np.asarray(fields["qy"]) / SQRT2,
        "left_wall_velocity": wall_velocity,
        "table_velocity": table_velocity,
        "bottom_heat_flux": scaled_bottom_heat_flux(
            distribution, bottom, system
        ),
        "residual_history": np.asarray(residual_history),
        "iterations": step + 1,
        "converged": converged,
        "dt": dt,
        "tau_prefactor": coordinate_tau_prefactor(cfg.kn0, system),
        "wall_mass_balance_relative_error": wall_balance,
        "minimum_distribution_c0_equivalent": float(
            np.min(distribution) / system.distribution_scale_from_c0
        ),
    }


def _relative_l1(a: np.ndarray, b: np.ndarray, floor: float = 1.0e-14) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("arrays must have matching shapes")
    return float(np.sum(np.abs(a - b)) / max(float(np.sum(np.abs(a))), floor))


def _relative_max(a: np.ndarray, b: np.ndarray, floor: float = 1.0e-12) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("arrays must have matching shapes")
    scale = max(float(np.max(np.abs(a))), floor)
    return float(np.max(np.abs(a - b)) / scale)


def covariance_metrics(
    c0_result: dict[str, object],
    zeta_result: dict[str, object],
    zeta_system: CoordinateSystem,
) -> dict[str, object]:
    field_names = (
        "T", "rho", "u", "v", "qx", "qy",
        "left_wall_velocity", "table_velocity", "bottom_heat_flux",
    )
    field_errors = {
        name: _relative_max(
            np.asarray(c0_result[name]),
            np.asarray(zeta_result[name]),
        )
        for name in field_names
    }
    c0_distribution = np.asarray(c0_result["distribution"], dtype=np.float64)
    zeta_as_c0 = (
        np.asarray(zeta_result["distribution"], dtype=np.float64)
        / zeta_system.distribution_scale_from_c0
    )
    distribution_relative_l1 = _relative_l1(c0_distribution, zeta_as_c0)
    residual_c0 = np.asarray(c0_result["residual_history"], dtype=np.float64)
    residual_zeta = np.asarray(zeta_result["residual_history"], dtype=np.float64)
    residual_relative_max = (
        _relative_max(residual_c0, residual_zeta)
        if residual_c0.shape == residual_zeta.shape
        else float("inf")
    )
    all_errors = list(field_errors.values()) + [
        distribution_relative_l1,
        residual_relative_max,
    ]
    return {
        "field_relative_max_errors": field_errors,
        "distribution_relative_l1_after_jacobian_transform": distribution_relative_l1,
        "residual_relative_max_error": residual_relative_max,
        "iteration_count_match": int(c0_result["iterations"]) == int(zeta_result["iterations"]),
        "convergence_flag_match": bool(c0_result["converged"]) == bool(zeta_result["converged"]),
        "dt_ratio_zeta_to_c0": float(zeta_result["dt"]) / float(c0_result["dt"]),
        "tau_prefactor_ratio_zeta_to_c0": (
            float(zeta_result["tau_prefactor"]) / float(c0_result["tau_prefactor"])
        ),
        "maximum_covariance_error": max(all_errors),
    }


def validate_stage35_design(
    knudsen_numbers: tuple[float, ...],
    grid: tuple[int, int],
    max_steps: int,
    tolerance: float,
) -> None:
    if tuple(float(value) for value in knudsen_numbers) != STAGE35_KNUDSEN:
        raise ValueError("Stage 35 is fixed to Kn0=(0.1,1,10)")
    if grid != STAGE35_GRID:
        raise ValueError("Stage 35 is fixed to a 12x12 spatial grid")
    if max_steps <= 0 or tolerance <= 0.0:
        raise ValueError("max_steps and tolerance must be positive")


def stage35_decision(rows: list[dict[str, object]]) -> str:
    covariance_passed = all(
        bool(row["c0_converged"])
        and bool(row["zeta_converged"])
        and bool(row["covariance"]["iteration_count_match"])
        and bool(row["covariance"]["convergence_flag_match"])
        and float(row["covariance"]["maximum_covariance_error"]) <= 1.0e-8
        and abs(float(row["covariance"]["dt_ratio_zeta_to_c0"]) - SQRT2) <= 1.0e-12
        and abs(
            float(row["covariance"]["tau_prefactor_ratio_zeta_to_c0"]) - SQRT2
        ) <= 1.0e-12
        for row in rows
    )
    if covariance_passed:
        return (
            "coordinate_covariance_passes_adopt_paper_consistent_mapping_"
            "and_run_high_resolution_stage36"
        )
    return "coordinate_covariance_fails_fix_scale_implementation_before_validation"


def run_stage35(
    output_dir: str | Path,
    *,
    knudsen_numbers: tuple[float, ...] = STAGE35_KNUDSEN,
    grid: tuple[int, int] = STAGE35_GRID,
    max_steps: int = 9000,
    tolerance: float = 3.0e-5,
) -> dict[str, object]:
    validate_stage35_design(knudsen_numbers, grid, max_steps, tolerance)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    c0_system, zeta_system = build_stage35_coordinate_systems()
    rows: list[dict[str, object]] = []
    arrays: dict[str, np.ndarray] = {}

    for kn0 in knudsen_numbers:
        cfg = LinearSidewallConfig(
            nx=grid[0],
            ny=grid[1],
            nv=19,
            velocity_extent=5.0,
            kn0=kn0,
            cold_hot_ratio=STAGE35_RATIO,
            max_steps=max_steps,
            tolerance=tolerance,
            check_interval=100,
            minimum_steps=1200,
        )
        c0_result = solve_scaled_coordinate_case(cfg, c0_system)
        zeta_result = solve_scaled_coordinate_case(cfg, zeta_system)
        covariance = covariance_metrics(c0_result, zeta_result, zeta_system)
        reference_velocity = TABLE3_UY_RATIO_0P1[kn0]
        reference_qav = TABLE6_QAV_RATIO_0P1[kn0]
        c0_profiles = wall_observable_profiles(c0_result, cfg)
        zeta_profiles = wall_observable_profiles(zeta_result, cfg)
        c0_qav = float(np.mean(np.asarray(c0_result["bottom_heat_flux"])))
        zeta_qav = float(np.mean(np.asarray(zeta_result["bottom_heat_flux"])))
        row = {
            "kn0": kn0,
            "c0_iterations": int(c0_result["iterations"]),
            "zeta_iterations": int(zeta_result["iterations"]),
            "c0_converged": bool(c0_result["converged"]),
            "zeta_converged": bool(zeta_result["converged"]),
            "c0_final_change": float(
                np.asarray(c0_result["residual_history"])[-1]
            ),
            "zeta_final_change": float(
                np.asarray(zeta_result["residual_history"])[-1]
            ),
            "c0_predicted_qav": c0_qav,
            "zeta_predicted_qav": zeta_qav,
            "literature_qav": reference_qav,
            "c0_qav_relative_error": abs(c0_qav - reference_qav) / reference_qav,
            "zeta_qav_relative_error": abs(zeta_qav - reference_qav) / reference_qav,
            "c0_observable_metrics": {
                name: observable_metrics(profile, reference_velocity)
                for name, profile in c0_profiles.items()
            },
            "zeta_observable_metrics": {
                name: observable_metrics(profile, reference_velocity)
                for name, profile in zeta_profiles.items()
            },
            "c0_wall_mass_balance_relative_error": float(
                c0_result["wall_mass_balance_relative_error"]
            ),
            "zeta_wall_mass_balance_relative_error": float(
                zeta_result["wall_mass_balance_relative_error"]
            ),
            "covariance": covariance,
        }
        rows.append(row)
        key = str(kn0).replace(".", "p")
        for name in (
            "T", "rho", "u", "v", "qx", "qy",
            "table_velocity", "bottom_heat_flux", "residual_history",
        ):
            arrays[f"c0_{name}_kn{key}"] = np.asarray(c0_result[name])
            arrays[f"zeta_{name}_kn{key}"] = np.asarray(zeta_result[name])

    decision = stage35_decision(rows)
    summary = {
        "stage": 35,
        "description": (
            "Exact coordinate-covariance audit of the reduced spherical Shakhov "
            "solver in c0 and paper-zeta molecular-velocity coordinates"
        ),
        "configuration": {
            "grid": list(grid),
            "kn0_sequence": list(knudsen_numbers),
            "cold_hot_ratio": STAGE35_RATIO,
            "physical_quadrature": STAGE35_QUADRATURE,
            "velocity_point_count": c0_system.quadrature.point_count,
            "c0_coordinate_to_c0": c0_system.coordinate_to_c0,
            "zeta_coordinate_to_c0": zeta_system.coordinate_to_c0,
            "c0_tau_prefactor": "sqrt(2)*Kn0/sqrt(pi)",
            "zeta_tau_prefactor": "2*Kn0/sqrt(pi)",
            "transport": "first-order upwind",
            "collision": "Shakhov",
            "max_steps": max_steps,
            "tolerance": tolerance,
            "physical_parameter_retuning": False,
        },
        "rows": rows,
        "all_cases_covariant_within_1e-8": all(
            float(row["covariance"]["maximum_covariance_error"]) <= 1.0e-8
            for row in rows
        ),
        "decision": decision,
        "interpretation_guard": (
            "The two formulations represent the same physical velocity nodes, "
            "weights, wall conditions, collision model, and Knudsen number. "
            "Only the molecular-velocity coordinate and its exact Jacobian, "
            "streaming speed, relaxation time, distribution scale, and positivity "
            "floor are transformed. A mismatch is an implementation error, not a "
            "physical-parameter result."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    np.savez_compressed(output_dir / "fields_and_profiles.npz", **arrays)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Stage 35 exact c0-versus-zeta coordinate covariance audit"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/stage35_coordinate_covariance",
    )
    parser.add_argument("--max-steps", type=int, default=9000)
    parser.add_argument("--tolerance", type=float, default=3.0e-5)
    args = parser.parse_args()
    print(
        json.dumps(
            run_stage35(
                args.output_dir,
                max_steps=args.max_steps,
                tolerance=args.tolerance,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
