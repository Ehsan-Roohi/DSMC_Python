from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import numpy as np

from .linear_sidewall_validation import LinearSidewallConfig, sidewall_temperature_profile
from .reduced_spherical_solver import (
    bottom_heat_flux,
    discrete_maxwellian,
    left_wall_tangential_velocity,
    macroscopic,
    wall_incoming,
    wall_mass_balance_error,
)
from .stage34_velocity_scale_consistency import local_relaxation_time
from .stage37_low_kn_transport_audit import (
    STAGE37_CFL,
    STAGE37_GRID,
    STAGE37_KNUDSEN,
    STAGE37_OBSERVABLE,
    STAGE37_QUADRATURE,
    STAGE37_RATIO,
    transport_case_metrics,
)
from .stage38_transport_collision_interaction_audit import first_order_transport_reduced
from .velocity_quadrature_audit import VelocityQuadrature, spherical_product


STAGE39_GRID = STAGE37_GRID
STAGE39_CFL = STAGE37_CFL
STAGE39_KNUDSEN = STAGE37_KNUDSEN
STAGE39_RATIO = STAGE37_RATIO
STAGE39_QUADRATURE = STAGE37_QUADRATURE
STAGE39_OBSERVABLE = STAGE37_OBSERVABLE
STAGE39_CLIP_FLOOR = 0.05

# Exact completed Stage 38 endpoint retained for provenance and interpretation.
STAGE38_COMPLETED_ENDPOINT = {
    "workflow_run_id": 30714512141,
    "workflow_job_id": 91407706438,
    "workflow_conclusion": "success",
    "tests_passed": 54,
    "tests_failed": 0,
    "artifact_id": 8823968646,
    "artifact_size_bytes": 15586,
    "artifact_sha256": "b9d17b291c2cefe5ead6b0953de2547296ec617ed2855c825d8c15610b06159d",
    "first_order_strang_explicit": {
        "predicted_qav": 0.0726173364328985,
        "literature_qav": 0.072,
        "qav_relative_error": 0.008574117123590345,
        "wall_velocity_relative_rms": 1.0091453484689894,
        "wall_velocity_sign_agreement": 0.9,
    },
    "muscl_strang_explicit": {
        "predicted_qav": 0.04329230501615015,
        "qav_relative_error": 0.39871798588680346,
        "wall_velocity_relative_rms": 2.7865958679189817,
        "wall_velocity_sign_agreement": 1.0,
    },
    "decision": "no_coupling_rescue_stage39_collision_model_or_benchmark_audit",
}


def validate_stage39_design(
    grid: tuple[int, int],
    cfl: float,
    clip_floor: float,
    max_steps: int,
    tolerance: float,
) -> None:
    if grid != STAGE39_GRID:
        raise ValueError("Stage 39 is fixed to the retained 24x24 low-Kn grid")
    if cfl != STAGE39_CFL:
        raise ValueError("Stage 39 retains CFL=0.2 without retuning")
    if clip_floor != STAGE39_CLIP_FLOOR:
        raise ValueError("Stage 39 audits the retained Shakhov clip floor 0.05")
    if max_steps <= 0 or tolerance <= 0.0:
        raise ValueError("max_steps and tolerance must be positive")


def shakhov_equilibrium_variant(
    fields: dict[str, np.ndarray],
    quadrature: VelocityQuadrature,
    prandtl: float,
    *,
    pointwise_clip: bool,
    clip_floor: float = STAGE39_CLIP_FLOOR,
) -> tuple[np.ndarray, dict[str, float]]:
    """Build clipped or raw Shakhov equilibrium and expose correction diagnostics.

    The raw arm does not clip the Shakhov multiplier pointwise. Positivity is
    instead enforced on the collision update by a cell-wise convex limiter.
    """
    if not 0.0 < prandtl <= 1.0:
        raise ValueError("Prandtl number must lie in (0,1]")
    if clip_floor <= 0.0:
        raise ValueError("clip_floor must be positive")

    maxwellian = discrete_maxwellian(
        fields["rho"],
        fields["u"],
        fields["v"],
        fields["w"],
        fields["T"],
        quadrature,
    )
    temperature = np.maximum(np.asarray(fields["T"], dtype=np.float64), 1.0e-10)
    pressure = np.maximum(np.asarray(fields["rho"], dtype=np.float64) * temperature, 1.0e-14)
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
    multiplier = 1.0 + correction
    applied_multiplier = np.maximum(multiplier, clip_floor) if pointwise_clip else multiplier
    equilibrium = maxwellian * applied_multiplier
    density = np.sum(equilibrium * quadrature.weight[None, None, :], axis=-1)
    if np.any(~np.isfinite(density)) or np.any(density <= 1.0e-14):
        raise FloatingPointError("Shakhov equilibrium has non-positive discrete density")
    equilibrium = equilibrium * (
        fields["rho"] / density
    )[..., None]
    diagnostics = {
        "raw_multiplier_minimum": float(np.min(multiplier)),
        "raw_multiplier_maximum": float(np.max(multiplier)),
        "raw_multiplier_below_zero_fraction": float(np.mean(multiplier < 0.0)),
        "raw_multiplier_below_clip_floor_fraction": float(np.mean(multiplier < clip_floor)),
        "raw_correction_maximum_absolute": float(np.max(np.abs(correction))),
        "equilibrium_minimum_before_collision_update": float(np.min(equilibrium)),
    }
    return equilibrium, diagnostics


def positivity_blend_with_theta(
    old: np.ndarray,
    candidate: np.ndarray,
    floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Cell-wise convex positivity limiter returning the accepted theta field."""
    old = np.asarray(old, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if old.shape != candidate.shape or old.ndim != 3:
        raise ValueError("old and candidate must be matching (ny,nx,nq) arrays")
    if floor <= 0.0:
        raise ValueError("floor must be positive")
    below = candidate < floor
    denominator = np.maximum(old - candidate, 1.0e-300)
    ratio = np.where(below, (old - floor) / denominator, 1.0)
    theta = np.clip(np.min(ratio, axis=-1), 0.0, 1.0)
    limited = old + theta[..., None] * (candidate - old)
    return np.maximum(limited, floor), theta


def collision_substep_variant(
    distribution: np.ndarray,
    cfg: LinearSidewallConfig,
    quadrature: VelocityQuadrature,
    substep: float,
    *,
    pointwise_clip: bool,
    clip_floor: float = STAGE39_CLIP_FLOOR,
) -> tuple[np.ndarray, dict[str, float]]:
    if substep <= 0.0:
        raise ValueError("collision substep must be positive")
    fields = macroscopic(distribution, quadrature)
    equilibrium, diagnostics = shakhov_equilibrium_variant(
        fields,
        quadrature,
        cfg.prandtl,
        pointwise_clip=pointwise_clip,
        clip_floor=clip_floor,
    )
    tau = local_relaxation_time(
        fields["rho"], fields["T"], cfg, mapping="paper_consistent_c0"
    )
    fraction = np.minimum(substep / tau, 1.0)[..., None]
    candidate = distribution + fraction * (equilibrium - distribution)
    limited, theta = positivity_blend_with_theta(
        distribution, candidate, cfg.positivity_floor
    )
    diagnostics = dict(diagnostics)
    diagnostics.update(
        {
            "collision_theta_minimum": float(np.min(theta)),
            "collision_limited_cell_fraction": float(np.mean(theta < 1.0 - 1.0e-14)),
        }
    )
    return limited, diagnostics


def solve_stage39_arm(
    cfg: LinearSidewallConfig,
    quadrature: VelocityQuadrature,
    *,
    pointwise_clip: bool,
    clip_floor: float = STAGE39_CLIP_FLOOR,
) -> dict[str, object]:
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
    previous = macroscopic(distribution, quadrature)
    previous_temperature = previous["T"].copy()
    previous_velocity = np.stack([previous["u"], previous["v"]], axis=-1)
    previous_heat_flux = np.stack([previous["qx"], previous["qy"]], axis=-1)
    residual_history: list[float] = []
    converged = False

    diagnostic_sums = {
        "raw_multiplier_below_zero_fraction": 0.0,
        "raw_multiplier_below_clip_floor_fraction": 0.0,
        "raw_correction_maximum_absolute": 0.0,
        "collision_limited_cell_fraction": 0.0,
    }
    raw_multiplier_minimum = math.inf
    raw_multiplier_maximum = -math.inf
    equilibrium_minimum = math.inf
    collision_theta_minimum = 1.0
    diagnostic_calls = 0

    for step in range(cfg.max_steps):
        half, diag_a = collision_substep_variant(
            distribution,
            cfg,
            quadrature,
            0.5 * dt,
            pointwise_clip=pointwise_clip,
            clip_floor=clip_floor,
        )
        transported = first_order_transport_reduced(
            half, cfg, quadrature, dt, dx, dy
        )
        distribution, diag_b = collision_substep_variant(
            transported,
            cfg,
            quadrature,
            0.5 * dt,
            pointwise_clip=pointwise_clip,
            clip_floor=clip_floor,
        )
        for diag in (diag_a, diag_b):
            diagnostic_calls += 1
            diagnostic_sums["raw_multiplier_below_zero_fraction"] += diag[
                "raw_multiplier_below_zero_fraction"
            ]
            diagnostic_sums["raw_multiplier_below_clip_floor_fraction"] += diag[
                "raw_multiplier_below_clip_floor_fraction"
            ]
            diagnostic_sums["raw_correction_maximum_absolute"] = max(
                diagnostic_sums["raw_correction_maximum_absolute"],
                diag["raw_correction_maximum_absolute"],
            )
            diagnostic_sums["collision_limited_cell_fraction"] += diag[
                "collision_limited_cell_fraction"
            ]
            raw_multiplier_minimum = min(
                raw_multiplier_minimum, diag["raw_multiplier_minimum"]
            )
            raw_multiplier_maximum = max(
                raw_multiplier_maximum, diag["raw_multiplier_maximum"]
            )
            equilibrium_minimum = min(
                equilibrium_minimum,
                diag["equilibrium_minimum_before_collision_update"],
            )
            collision_theta_minimum = min(
                collision_theta_minimum, diag["collision_theta_minimum"]
            )

        if (step + 1) % cfg.check_interval == 0:
            fields = macroscopic(distribution, quadrature)
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

    left, right, bottom, top = wall_incoming(distribution, cfg, quadrature)
    fields = macroscopic(distribution, quadrature)
    wall_velocity = left_wall_tangential_velocity(distribution, left, quadrature)
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
    calls = max(diagnostic_calls, 1)
    diagnostics = {
        "raw_multiplier_minimum": raw_multiplier_minimum,
        "raw_multiplier_maximum": raw_multiplier_maximum,
        "mean_raw_multiplier_below_zero_fraction": diagnostic_sums[
            "raw_multiplier_below_zero_fraction"
        ] / calls,
        "mean_raw_multiplier_below_clip_floor_fraction": diagnostic_sums[
            "raw_multiplier_below_clip_floor_fraction"
        ] / calls,
        "maximum_raw_correction_absolute": diagnostic_sums[
            "raw_correction_maximum_absolute"
        ],
        "minimum_equilibrium_before_collision_update": equilibrium_minimum,
        "minimum_collision_theta": collision_theta_minimum,
        "mean_collision_limited_cell_fraction": diagnostic_sums[
            "collision_limited_cell_fraction"
        ] / calls,
        "diagnostic_collision_calls": diagnostic_calls,
    }
    return {
        "T": np.asarray(fields["T"]),
        "rho": np.asarray(fields["rho"]) / np.mean(fields["rho"]),
        "u": np.asarray(fields["u"]) / math.sqrt(2.0),
        "v": np.asarray(fields["v"]) / math.sqrt(2.0),
        "qx": np.asarray(fields["qx"]) / math.sqrt(2.0),
        "qy": np.asarray(fields["qy"]) / math.sqrt(2.0),
        "left_wall_velocity": wall_velocity,
        "bottom_heat_flux": bottom_heat_flux(distribution, bottom, quadrature),
        "residual_history": np.asarray(residual_history),
        "iterations": step + 1,
        "converged": converged,
        "dt": dt,
        "wall_mass_balance_relative_error": wall_balance,
        "minimum_distribution": float(np.min(distribution)),
        "collision_diagnostics": diagnostics,
    }


def stage39_decision(
    clipped: dict[str, object],
    raw: dict[str, object],
) -> str:
    if not bool(clipped["converged"]) or not bool(raw["converged"]):
        return "stage39_nonconvergence_stage40_numerical_stability_audit"
    clipped_diag = clipped["collision_diagnostics"]
    activation = float(clipped_diag["mean_raw_multiplier_below_clip_floor_fraction"])
    if activation < 1.0e-8:
        return "clipping_inactive_stage40_independent_benchmark"
    q_ratio = float(raw["qav_relative_error"]) / max(
        float(clipped["qav_relative_error"]), 1.0e-14
    )
    v_ratio = float(raw["velocity_metrics"]["relative_rms"]) / max(
        float(clipped["velocity_metrics"]["relative_rms"]), 1.0e-14
    )
    sign_change = float(raw["velocity_metrics"]["sign_agreement"]) - float(
        clipped["velocity_metrics"]["sign_agreement"]
    )
    if q_ratio <= 0.90 and v_ratio <= 0.90 and sign_change >= 0.0:
        return "pointwise_clipping_identified_stage40_high_resolution_confirmation"
    if q_ratio <= 0.90 or v_ratio <= 0.90 or sign_change > 0.0:
        return "clipping_material_but_incomplete_stage40_independent_benchmark"
    return "clipping_not_primary_stage40_independent_benchmark_or_full_boltzmann"


def run_stage39(
    output_dir: str | Path,
    *,
    grid: tuple[int, int] = STAGE39_GRID,
    cfl: float = STAGE39_CFL,
    clip_floor: float = STAGE39_CLIP_FLOOR,
    max_steps: int = 16000,
    tolerance: float = 2.0e-5,
) -> dict[str, object]:
    validate_stage39_design(grid, cfl, clip_floor, max_steps, tolerance)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    quadrature = spherical_product(16, 12, 24, 5.0, STAGE39_QUADRATURE)
    cfg = LinearSidewallConfig(
        nx=grid[0],
        ny=grid[1],
        nv=19,
        velocity_extent=5.0,
        kn0=STAGE39_KNUDSEN,
        cold_hot_ratio=STAGE39_RATIO,
        max_steps=max_steps,
        cfl=cfl,
        tolerance=tolerance,
        check_interval=100,
        minimum_steps=1800,
    )
    clipped_result = solve_stage39_arm(
        cfg, quadrature, pointwise_clip=True, clip_floor=clip_floor
    )
    raw_result = solve_stage39_arm(
        cfg, quadrature, pointwise_clip=False, clip_floor=clip_floor
    )
    clipped_row, clipped_profiles = transport_case_metrics(
        clipped_result, cfg, "clipped_shakhov_strang_first_order", quadrature
    )
    raw_row, raw_profiles = transport_case_metrics(
        raw_result, cfg, "raw_shakhov_collision_limited_strang_first_order", quadrature
    )
    clipped_row["collision_diagnostics"] = clipped_result["collision_diagnostics"]
    raw_row["collision_diagnostics"] = raw_result["collision_diagnostics"]
    comparison = {
        "raw_to_clipped_qav_error_ratio": float(raw_row["qav_relative_error"])
        / max(float(clipped_row["qav_relative_error"]), 1.0e-14),
        "raw_to_clipped_velocity_error_ratio": float(
            raw_row["velocity_metrics"]["relative_rms"]
        ) / max(float(clipped_row["velocity_metrics"]["relative_rms"]), 1.0e-14),
        "raw_minus_clipped_sign_agreement": float(
            raw_row["velocity_metrics"]["sign_agreement"]
        ) - float(clipped_row["velocity_metrics"]["sign_agreement"]),
        "raw_minus_clipped_qav": float(raw_row["predicted_qav"])
        - float(clipped_row["predicted_qav"]),
    }
    decision = stage39_decision(clipped_row, raw_row)
    summary = {
        "stage": 39,
        "description": (
            "Fixed-physics low-Kn audit of pointwise Shakhov multiplier clipping "
            "versus raw Shakhov equilibrium with cell-wise positivity-limited collision updates"
        ),
        "configuration": {
            "kn0": STAGE39_KNUDSEN,
            "cold_hot_ratio": STAGE39_RATIO,
            "grid": list(grid),
            "quadrature": STAGE39_QUADRATURE,
            "velocity_point_count": quadrature.point_count,
            "relaxation_mapping": "sqrt(2)*Kn0/sqrt(pi) in c0 coordinates",
            "transport": "first_order_upwind",
            "coupling": "strang_explicit_half_collision",
            "collision": "Shakhov",
            "collision_positivity_arms": [
                "pointwise_multiplier_clip_0p05",
                "raw_multiplier_cellwise_collision_update_limiter",
            ],
            "wall_observable": STAGE39_OBSERVABLE,
            "cfl_equal_in_both_arms": cfl,
            "clip_floor_audited": clip_floor,
            "max_steps": max_steps,
            "tolerance": tolerance,
            "physical_parameter_retuning": False,
        },
        "retained_stage38_endpoint": STAGE38_COMPLETED_ENDPOINT,
        "rows": [clipped_row, raw_row],
        "comparison": comparison,
        "decision": decision,
        "interpretation_guard": (
            "Knudsen number, wall temperatures, relaxation mapping, Prandtl number, "
            "viscosity law, quadrature, grid, transport, Strang coupling, walls, "
            "observable, CFL and stopping rule are frozen. Only the numerical "
            "positivity treatment of the Shakhov correction changes."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    arrays: dict[str, np.ndarray] = {}
    for token, result, profiles in (
        ("clipped", clipped_result, clipped_profiles),
        ("raw", raw_result, raw_profiles),
    ):
        arrays[f"bottom_heat_flux_{token}"] = np.asarray(result["bottom_heat_flux"])
        arrays[f"residual_history_{token}"] = np.asarray(result["residual_history"])
        arrays[f"T_{token}"] = np.asarray(result["T"])
        arrays[f"rho_{token}"] = np.asarray(result["rho"])
        arrays[f"table_velocity_{token}"] = np.asarray(profiles[STAGE39_OBSERVABLE])
    np.savez_compressed(output_dir / "fields_and_profiles.npz", **arrays)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 39 Shakhov clipping audit")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=16000)
    parser.add_argument("--tolerance", type=float, default=2.0e-5)
    args = parser.parse_args()
    summary = run_stage39(
        args.output_dir,
        max_steps=args.max_steps,
        tolerance=args.tolerance,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
