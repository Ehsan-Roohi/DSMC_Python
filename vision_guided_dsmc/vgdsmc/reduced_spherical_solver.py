from __future__ import annotations

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
    _relative_rms,
    local_relaxation_time,
    sidewall_temperature_profile,
)
from .velocity_quadrature_audit import (
    VelocityQuadrature,
    cartesian_midpoint,
    spherical_product,
)


def _check_distribution(distribution: np.ndarray, quadrature: VelocityQuadrature) -> None:
    if distribution.ndim != 3:
        raise ValueError("distribution must have shape (ny,nx,nq)")
    if distribution.shape[-1] != quadrature.point_count:
        raise ValueError("distribution and quadrature point counts must match")


def discrete_maxwellian(
    rho: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    temperature: np.ndarray,
    quadrature: VelocityQuadrature,
) -> np.ndarray:
    """Density-normalized Maxwellian on an arbitrary positive velocity rule."""
    rho = np.asarray(rho, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    temperature = np.maximum(np.asarray(temperature, dtype=np.float64), 1.0e-10)
    if not (rho.shape == u.shape == v.shape == w.shape == temperature.shape):
        raise ValueError("all macroscopic fields must have matching shapes")
    vx = quadrature.vx[None, None, :]
    vy = quadrature.vy[None, None, :]
    vz = quadrature.vz[None, None, :]
    cx = vx - u[..., None]
    cy = vy - v[..., None]
    cz = vz - w[..., None]
    c2 = cx * cx + cy * cy + cz * cz
    base = np.exp(-c2 / (2.0 * temperature[..., None]))
    base /= (2.0 * math.pi * temperature[..., None]) ** 1.5
    norm = np.sum(base * quadrature.weight[None, None, :], axis=-1)
    return rho[..., None] * base / np.maximum(norm[..., None], 1.0e-300)


def macroscopic(
    distribution: np.ndarray,
    quadrature: VelocityQuadrature,
) -> dict[str, np.ndarray]:
    _check_distribution(distribution, quadrature)
    weight = quadrature.weight[None, None, :]
    rho = np.sum(distribution * weight, axis=-1)
    safe_rho = np.maximum(rho, 1.0e-14)
    vx = quadrature.vx[None, None, :]
    vy = quadrature.vy[None, None, :]
    vz = quadrature.vz[None, None, :]
    u = np.sum(distribution * vx * weight, axis=-1) / safe_rho
    v = np.sum(distribution * vy * weight, axis=-1) / safe_rho
    w = np.sum(distribution * vz * weight, axis=-1) / safe_rho
    cx = vx - u[..., None]
    cy = vy - v[..., None]
    cz = vz - w[..., None]
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


def shakhov_equilibrium(
    fields: dict[str, np.ndarray],
    quadrature: VelocityQuadrature,
    prandtl: float,
) -> np.ndarray:
    if not 0.0 < prandtl <= 1.0:
        raise ValueError("Prandtl number must lie in (0,1]")
    equilibrium = discrete_maxwellian(
        fields["rho"], fields["u"], fields["v"], fields["w"], fields["T"], quadrature
    )
    temperature = np.maximum(fields["T"], 1.0e-10)
    pressure = np.maximum(fields["rho"] * temperature, 1.0e-14)
    cx = quadrature.vx[None, None, :] - fields["u"][..., None]
    cy = quadrature.vy[None, None, :] - fields["v"][..., None]
    cz = quadrature.vz[None, None, :] - fields["w"][..., None]
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
        equilibrium * quadrature.weight[None, None, :], axis=-1
    )
    return equilibrium * (
        fields["rho"] / np.maximum(density, 1.0e-14)
    )[..., None]


def unit_wall_maxwellian(
    temperature: float,
    quadrature: VelocityQuadrature,
) -> np.ndarray:
    if temperature <= 0.0:
        raise ValueError("wall temperature must be positive")
    speed2 = quadrature.vx**2 + quadrature.vy**2 + quadrature.vz**2
    base = np.exp(-speed2 / (2.0 * temperature))
    base /= (2.0 * math.pi * temperature) ** 1.5
    return base / max(float(np.sum(base * quadrature.weight)), 1.0e-300)


def _profile_wall_incoming(
    outgoing_distribution: np.ndarray,
    normal_velocity: np.ndarray,
    incoming_mask: np.ndarray,
    outgoing_mask: np.ndarray,
    wall_maxwellians: np.ndarray,
    quadrature: VelocityQuadrature,
) -> np.ndarray:
    weight = quadrature.weight[None, :]
    outgoing_flux = np.sum(
        normal_velocity[None, :]
        * outgoing_distribution
        * outgoing_mask[None, :]
        * weight,
        axis=-1,
    )
    incoming_unit = np.sum(
        normal_velocity[None, :]
        * wall_maxwellians
        * incoming_mask[None, :]
        * weight,
        axis=-1,
    )
    scale = -outgoing_flux / np.maximum(incoming_unit, 1.0e-14)
    return scale[:, None] * wall_maxwellians


def wall_incoming(
    distribution: np.ndarray,
    cfg: LinearSidewallConfig,
    quadrature: VelocityQuadrature,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _check_distribution(distribution, quadrature)
    side_wall = np.stack(
        [unit_wall_maxwellian(float(t), quadrature) for t in sidewall_temperature_profile(cfg)]
    )
    bottom_wall = np.repeat(
        unit_wall_maxwellian(cfg.hot_temperature, quadrature)[None, :], cfg.nx, axis=0
    )
    top_wall = np.repeat(
        unit_wall_maxwellian(cfg.cold_temperature, quadrature)[None, :], cfg.nx, axis=0
    )
    px, nx = quadrature.vx > 0.0, quadrature.vx < 0.0
    py, ny = quadrature.vy > 0.0, quadrature.vy < 0.0
    left = _profile_wall_incoming(
        distribution[:, 0], quadrature.vx, px, nx, side_wall, quadrature
    )
    right = _profile_wall_incoming(
        distribution[:, -1], -quadrature.vx, nx, px, side_wall, quadrature
    )
    bottom = _profile_wall_incoming(
        distribution[0], quadrature.vy, py, ny, bottom_wall, quadrature
    )
    top = _profile_wall_incoming(
        distribution[-1], -quadrature.vy, ny, py, top_wall, quadrature
    )
    return left, right, bottom, top


def wall_mass_balance_error(
    interior_outgoing: np.ndarray,
    incoming: np.ndarray,
    normal_velocity: np.ndarray,
    incoming_mask: np.ndarray,
    quadrature: VelocityQuadrature,
) -> float:
    boundary = np.where(incoming_mask[None, :], incoming, interior_outgoing)
    net = np.sum(
        boundary * normal_velocity[None, :] * quadrature.weight[None, :], axis=-1
    )
    scale = np.maximum(
        np.sum(
            np.abs(boundary * normal_velocity[None, :])
            * quadrature.weight[None, :],
            axis=-1,
        ),
        1.0e-14,
    )
    return float(np.max(np.abs(net) / scale))


def left_wall_tangential_velocity(
    distribution: np.ndarray,
    left_incoming: np.ndarray,
    quadrature: VelocityQuadrature,
) -> np.ndarray:
    boundary = np.where(
        (quadrature.vx > 0.0)[None, :], left_incoming, distribution[:, 0]
    )
    return np.asarray(macroscopic(boundary[:, None, :], quadrature)["v"][:, 0]) / math.sqrt(2.0)


def bottom_heat_flux(
    distribution: np.ndarray,
    bottom_incoming: np.ndarray,
    quadrature: VelocityQuadrature,
) -> np.ndarray:
    boundary = np.where(
        (quadrature.vy > 0.0)[None, :], bottom_incoming, distribution[0]
    )
    speed2 = quadrature.vx**2 + quadrature.vy**2 + quadrature.vz**2
    flux = 0.5 * np.sum(
        boundary
        * quadrature.vy[None, :]
        * speed2[None, :]
        * quadrature.weight[None, :],
        axis=-1,
    )
    return flux / math.sqrt(2.0)


def solve_reduced_case(
    cfg: LinearSidewallConfig,
    quadrature: VelocityQuadrature,
) -> dict[str, object]:
    if cfg.nx < 3 or cfg.ny < 3:
        raise ValueError("nx and ny must be at least three")
    if cfg.kn0 <= 0.0 or not 0.0 < cfg.cold_hot_ratio < 1.0:
        raise ValueError("invalid physical configuration")
    rho = np.ones((cfg.ny, cfg.nx))
    zero = np.zeros_like(rho)
    initial_temperature = np.repeat(
        sidewall_temperature_profile(cfg)[:, None], cfg.nx, axis=1
    )
    distribution = discrete_maxwellian(
        rho, zero, zero, zero, initial_temperature, quadrature
    )
    dx, dy = 1.0 / cfg.nx, 1.0 / cfg.ny
    maximum_speed = max(
        float(np.max(np.abs(quadrature.vx))),
        float(np.max(np.abs(quadrature.vy))),
    )
    dt = cfg.cfl * min(dx / maximum_speed, dy / maximum_speed)
    positive_x = (quadrature.vx > 0.0)[None, None, :]
    positive_y = (quadrature.vy > 0.0)[None, None, :]
    previous = macroscopic(distribution, quadrature)
    previous_T = previous["T"].copy()
    previous_u = np.stack([previous["u"], previous["v"]], axis=-1)
    previous_q = np.stack([previous["qx"], previous["qy"]], axis=-1)
    residual_history: list[float] = []
    converged = False
    left = right = bottom = top = None

    for step in range(cfg.max_steps):
        left, right, bottom, top = wall_incoming(distribution, cfg, quadrature)
        ln = np.empty_like(distribution)
        rn = np.empty_like(distribution)
        bn = np.empty_like(distribution)
        tn = np.empty_like(distribution)
        ln[:, 1:] = distribution[:, :-1]
        ln[:, 0] = left
        rn[:, :-1] = distribution[:, 1:]
        rn[:, -1] = right
        bn[1:] = distribution[:-1]
        bn[0] = bottom
        tn[:-1] = distribution[1:]
        tn[-1] = top
        dfdx = np.where(
            positive_x,
            (distribution - ln) / dx,
            (rn - distribution) / dx,
        )
        dfdy = np.where(
            positive_y,
            (distribution - bn) / dy,
            (tn - distribution) / dy,
        )
        transported = np.maximum(
            distribution
            - dt
            * (
                quadrature.vx[None, None, :] * dfdx
                + quadrature.vy[None, None, :] * dfdy
            ),
            cfg.positivity_floor,
        )
        fields = macroscopic(transported, quadrature)
        equilibrium = shakhov_equilibrium(fields, quadrature, cfg.prandtl)
        tau = local_relaxation_time(fields["rho"], fields["T"], cfg)
        fraction = np.minimum(dt / tau, 1.0)[..., None]
        distribution = np.maximum(
            transported + fraction * (equilibrium - transported),
            cfg.positivity_floor,
        )
        if (step + 1) % cfg.check_interval == 0:
            fields = macroscopic(distribution, quadrature)
            velocity = np.stack([fields["u"], fields["v"]], axis=-1)
            heat_flux = np.stack([fields["qx"], fields["qy"]], axis=-1)
            change = max(
                float(np.max(np.abs(fields["T"] - previous_T))),
                float(np.max(np.abs(velocity - previous_u))),
                float(np.max(np.abs(heat_flux - previous_q))),
            )
            residual_history.append(change)
            previous_T = fields["T"].copy()
            previous_u = velocity.copy()
            previous_q = heat_flux.copy()
            if step + 1 >= cfg.minimum_steps and change < cfg.tolerance:
                converged = True
                break

    # Incoming diffuse-wall states must match the final post-collision distribution.
    # Reusing the states built before the last update creates a false mass-balance error.
    left, right, bottom, top = wall_incoming(distribution, cfg, quadrature)
    assert left is not None and right is not None and bottom is not None and top is not None
    fields = macroscopic(distribution, quadrature)
    wall_velocity = left_wall_tangential_velocity(distribution, left, quadrature)
    y_centers = (np.arange(cfg.ny, dtype=np.float64) + 0.5) / cfg.ny
    table_velocity = np.interp(TABLE3_Y, y_centers, wall_velocity)
    wall_balance = max(
        wall_mass_balance_error(
            distribution[:, 0], left, quadrature.vx, quadrature.vx > 0.0, quadrature
        ),
        wall_mass_balance_error(
            distribution[:, -1], right, -quadrature.vx, quadrature.vx < 0.0, quadrature
        ),
        wall_mass_balance_error(
            distribution[0], bottom, quadrature.vy, quadrature.vy > 0.0, quadrature
        ),
        wall_mass_balance_error(
            distribution[-1], top, -quadrature.vy, quadrature.vy < 0.0, quadrature
        ),
    )
    return {
        "T": np.asarray(fields["T"]),
        "rho": np.asarray(fields["rho"]) / np.mean(fields["rho"]),
        "u": np.asarray(fields["u"]) / math.sqrt(2.0),
        "v": np.asarray(fields["v"]) / math.sqrt(2.0),
        "qx": np.asarray(fields["qx"]) / math.sqrt(2.0),
        "qy": np.asarray(fields["qy"]) / math.sqrt(2.0),
        "left_wall_velocity": wall_velocity,
        "table_velocity": table_velocity,
        "bottom_heat_flux": bottom_heat_flux(distribution, bottom, quadrature),
        "residual_history": np.asarray(residual_history),
        "iterations": step + 1,
        "converged": converged,
        "dt": dt,
        "wall_mass_balance_relative_error": wall_balance,
        "minimum_distribution": float(np.min(distribution)),
    }


def _case_metrics(
    result: dict[str, object],
    cfg: LinearSidewallConfig,
    quadrature: VelocityQuadrature,
) -> dict[str, object]:
    predicted_u = np.asarray(result["table_velocity"], dtype=np.float64)
    reference_u = TABLE3_UY_RATIO_0P1[cfg.kn0]
    predicted_q = float(np.mean(np.asarray(result["bottom_heat_flux"])))
    reference_q = TABLE6_QAV_RATIO_0P1[cfg.kn0]
    residual = np.asarray(result["residual_history"], dtype=np.float64)
    return {
        "scheme": quadrature.name,
        "family": quadrature.family,
        "point_count": quadrature.point_count,
        "iterations": int(result["iterations"]),
        "converged": bool(result["converged"]),
        "final_change": float(residual[-1]) if residual.size else float("nan"),
        "predicted_qav": predicted_q,
        "literature_qav": reference_q,
        "qav_relative_error": abs(predicted_q - reference_q) / reference_q,
        "wall_velocity_relative_rms": _relative_rms(predicted_u, reference_u),
        "wall_velocity_sign_agreement": float(
            np.mean(np.sign(predicted_u) == np.sign(reference_u))
        ),
        "wall_mass_balance_relative_error": float(
            result["wall_mass_balance_relative_error"]
        ),
        "minimum_distribution": float(result["minimum_distribution"]),
        "minimum_temperature": float(np.min(np.asarray(result["T"]))),
        "maximum_temperature": float(np.max(np.asarray(result["T"]))),
        "work_proxy": int(result["iterations"])
        * cfg.nx
        * cfg.ny
        * quadrature.point_count,
    }


def run_stage30(
    output_dir: str | Path,
    base: LinearSidewallConfig,
) -> dict[str, object]:
    if base.kn0 != 1.0 or base.cold_hot_ratio != 0.1:
        raise ValueError("Stage 30 is fixed to Kn0=1 and TC/TH=0.1")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    quadratures = [
        cartesian_midpoint(19, 5.0),
        spherical_product(
            16, 12, 24, 5.0, "spherical_matched_r16_mu12_phi24"
        ),
    ]
    rows: list[dict[str, object]] = []
    arrays: dict[str, np.ndarray] = {}
    for quadrature in quadratures:
        result = solve_reduced_case(base, quadrature)
        rows.append(_case_metrics(result, base, quadrature))
        key = quadrature.name
        for name in (
            "T",
            "rho",
            "u",
            "v",
            "qx",
            "qy",
            "left_wall_velocity",
            "table_velocity",
            "bottom_heat_flux",
            "residual_history",
        ):
            arrays[f"{name}_{key}"] = np.asarray(result[name])
    by_name = {row["scheme"]: row for row in rows}
    cartesian = by_name["cartesian_midpoint_nv19"]
    spherical = by_name["spherical_matched_r16_mu12_phi24"]
    materially_better = bool(
        float(spherical["qav_relative_error"])
        <= 0.85 * float(cartesian["qav_relative_error"])
        and float(spherical["wall_velocity_relative_rms"])
        <= 0.85 * float(cartesian["wall_velocity_relative_rms"])
        and float(spherical["wall_velocity_sign_agreement"])
        >= float(cartesian["wall_velocity_sign_agreement"])
    )
    summary = {
        "stage": 30,
        "description": (
            "Fixed-physics cavity integration of the Stage-29 spherical-product velocity rule"
        ),
        "configuration": {
            "grid": [base.nx, base.ny],
            "kn0": base.kn0,
            "cold_hot_ratio": base.cold_hot_ratio,
            "max_steps": base.max_steps,
            "tolerance": base.tolerance,
            "transport": "first-order upwind",
            "collision": "Shakhov",
            "wall_model": "bottom hot, top cold, side walls linear hot-to-cold",
            "physical_parameter_retuning": False,
        },
        "rows": rows,
        "spherical_qav_error_ratio_to_cartesian": float(
            spherical["qav_relative_error"]
        )
        / max(float(cartesian["qav_relative_error"]), 1.0e-14),
        "spherical_velocity_error_ratio_to_cartesian": float(
            spherical["wall_velocity_relative_rms"]
        )
        / max(float(cartesian["wall_velocity_relative_rms"]), 1.0e-14),
        "spherical_materially_better_on_both_literature_metrics": materially_better,
        "decision": (
            "extend_spherical_solver_to_kn0_0p1_and_10"
            if materially_better
            else "audit_wall_observable_and_sign_convention_before_further_physics_runs"
        ),
        "interpretation_guard": (
            "Stage 30 is a controlled quadrature integration test. It does not retune Knudsen, "
            "collision, wall, or normalization parameters, and negative results remain explicit."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    np.savez_compressed(output_dir / "fields_and_profiles.npz", **arrays)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 30 reduced spherical DVM comparison")
    parser.add_argument("--output-dir", default="outputs/stage30_reduced_spherical")
    parser.add_argument("--nx", type=int, default=12)
    parser.add_argument("--ny", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=4200)
    parser.add_argument("--tolerance", type=float, default=3.0e-5)
    args = parser.parse_args()
    cfg = LinearSidewallConfig(
        nx=args.nx,
        ny=args.ny,
        nv=19,
        velocity_extent=5.0,
        kn0=1.0,
        cold_hot_ratio=0.1,
        max_steps=args.max_steps,
        tolerance=args.tolerance,
        check_interval=100,
        minimum_steps=1200,
    )
    print(json.dumps(run_stage30(args.output_dir, cfg), indent=2))


if __name__ == "__main__":
    main()
