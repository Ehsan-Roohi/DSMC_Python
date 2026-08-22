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
    shakhov_equilibrium,
    wall_incoming,
    wall_mass_balance_error,
)
from .stage34_velocity_scale_consistency import local_relaxation_time
from .stage37_low_kn_transport_audit import (
    STAGE37_CFL,
    STAGE37_GRID,
    STAGE37_KNUDSEN,
    STAGE37_LIMITER_THETA,
    STAGE37_OBSERVABLE,
    STAGE37_QUADRATURE,
    STAGE37_RATIO,
    positivity_blend_reduced,
    transport_case_metrics,
    transport_ssprk2_reduced,
)
from .velocity_quadrature_audit import VelocityQuadrature, spherical_product


STAGE38_GRID = STAGE37_GRID
STAGE38_CFL = STAGE37_CFL
STAGE38_LIMITER_THETA = STAGE37_LIMITER_THETA
STAGE38_COUPLING = "strang_explicit_half_collision"

# Exact Stage 37 endpoint, including workflow and artifact provenance. These rows
# are retained verbatim and are not silently rerun or replaced in Stage 38.
STAGE37_COMPLETED_ENDPOINT = {
    "workflow_run_id": 30708041761,
    "workflow_job_id": 91390477801,
    "workflow_conclusion": "success",
    "tests_passed": 44,
    "tests_failed": 0,
    "artifact_id": 8822338075,
    "artifact_size_bytes": 14832,
    "artifact_sha256": "2981365dd782c617eba73deb9e54d3ec31909acdc862d6d786934879f6b87152",
    "rows": {
        "first_order_lie_explicit": {
            "scheme": "first_order_lie_explicit",
            "iterations": 4600,
            "converged": True,
            "final_change": 1.6202831369938053e-05,
            "predicted_qav": 0.07473044968217635,
            "literature_qav": 0.072,
            "qav_relative_error": 0.03792291225244935,
            "velocity_observable": STAGE37_OBSERVABLE,
            "velocity_metrics": {
                "relative_rms": 0.8947813193174375,
                "sign_agreement": 0.8,
                "relative_l1": 0.921349764565591,
            },
            "wall_mass_balance_relative_error": 1.9293558552538705e-16,
            "minimum_distribution": 1.0e-30,
            "minimum_temperature": 0.14756557196392397,
            "maximum_temperature": 0.900196703039309,
            "dt": 0.0017034149644037035,
            "work_proxy": 12209356800,
        },
        "muscl_lie_explicit": {
            "scheme": "muscl_lie_explicit",
            "iterations": 5900,
            "converged": True,
            "final_change": 1.9700071770266045e-05,
            "predicted_qav": 0.04324014823991044,
            "literature_qav": 0.072,
            "qav_relative_error": 0.39944238555679945,
            "velocity_observable": STAGE37_OBSERVABLE,
            "velocity_metrics": {
                "relative_rms": 2.7885335044977855,
                "sign_agreement": 1.0,
                "relative_l1": 1.884477062793814,
            },
            "wall_mass_balance_relative_error": 2.415988193870213e-16,
            "minimum_distribution": 1.0e-30,
            "minimum_temperature": 0.1187404588444163,
            "maximum_temperature": 0.8697436824351845,
            "dt": 0.0017034149644037035,
            "work_proxy": 15659827200,
        },
    },
    "comparison": {
        "qav_error_ratio_muscl_to_first_order": 10.533009250390585,
        "velocity_error_ratio_muscl_to_first_order": 3.116441351977432,
        "sign_agreement_change": 0.19999999999999996,
        "qav_change": -0.03149030144226591,
    },
    "decision": "muscl_screen_mixed_stage38_transport_collision_interaction_audit",
}


def validate_stage38_design(
    grid: tuple[int, int],
    cfl: float,
    limiter_theta: float,
    max_steps: int,
    tolerance: float,
) -> None:
    if grid != STAGE38_GRID:
        raise ValueError("Stage 38 is fixed to the Stage 37 24x24 grid")
    if cfl != STAGE38_CFL:
        raise ValueError("Stage 38 retains the Stage 37 CFL without retuning")
    if limiter_theta != STAGE38_LIMITER_THETA:
        raise ValueError("Stage 38 retains limiter theta=1.5 without retuning")
    if max_steps <= 0 or tolerance <= 0.0:
        raise ValueError("max_steps and tolerance must be positive")


def first_order_transport_reduced(
    distribution: np.ndarray,
    cfg: LinearSidewallConfig,
    quadrature: VelocityQuadrature,
    dt: float,
    dx: float,
    dy: float,
) -> np.ndarray:
    """The retained first-order upwind transport operator, without collision."""
    left, right, bottom, top = wall_incoming(distribution, cfg, quadrature)
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
    positive_x = (quadrature.vx > 0.0)[None, None, :]
    positive_y = (quadrature.vy > 0.0)[None, None, :]
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
    candidate = distribution - dt * (
        quadrature.vx[None, None, :] * dfdx
        + quadrature.vy[None, None, :] * dfdy
    )
    return np.maximum(candidate, cfg.positivity_floor)


def explicit_collision_substep(
    distribution: np.ndarray,
    cfg: LinearSidewallConfig,
    quadrature: VelocityQuadrature,
    substep: float,
) -> np.ndarray:
    """Frozen-physics explicit Shakhov collision substep used by Strang splitting."""
    if substep <= 0.0:
        raise ValueError("collision substep must be positive")
    fields = macroscopic(distribution, quadrature)
    equilibrium = shakhov_equilibrium(fields, quadrature, cfg.prandtl)
    tau = local_relaxation_time(
        fields["rho"], fields["T"], cfg, mapping="paper_consistent_c0"
    )
    fraction = np.minimum(substep / tau, 1.0)[..., None]
    candidate = distribution + fraction * (equilibrium - distribution)
    return positivity_blend_reduced(distribution, candidate, cfg.positivity_floor)


def strang_step_reduced(
    distribution: np.ndarray,
    cfg: LinearSidewallConfig,
    quadrature: VelocityQuadrature,
    dt: float,
    dx: float,
    dy: float,
    transport_order: str,
    limiter_theta: float = STAGE38_LIMITER_THETA,
) -> np.ndarray:
    """Half collision, full transport, half collision with no physical retuning."""
    half_collided = explicit_collision_substep(
        distribution, cfg, quadrature, 0.5 * dt
    )
    if transport_order == "first_order_upwind":
        transported = first_order_transport_reduced(
            half_collided, cfg, quadrature, dt, dx, dy
        )
    elif transport_order == "muscl_ssprk2":
        transported = transport_ssprk2_reduced(
            half_collided,
            cfg,
            quadrature,
            dt,
            dx,
            dy,
            limiter_theta,
        )
    else:
        raise ValueError("unknown transport order")
    return explicit_collision_substep(transported, cfg, quadrature, 0.5 * dt)


def solve_strang_reduced_case(
    cfg: LinearSidewallConfig,
    quadrature: VelocityQuadrature,
    *,
    transport_order: str,
    limiter_theta: float = STAGE38_LIMITER_THETA,
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
    previous = macroscopic(distribution, quadrature)
    previous_temperature = previous["T"].copy()
    previous_velocity = np.stack([previous["u"], previous["v"]], axis=-1)
    previous_heat_flux = np.stack([previous["qx"], previous["qy"]], axis=-1)
    residual_history: list[float] = []
    converged = False

    for step in range(cfg.max_steps):
        distribution = strang_step_reduced(
            distribution,
            cfg,
            quadrature,
            dt,
            dx,
            dy,
            transport_order,
            limiter_theta,
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
    }


def _error_ratio(numerator: dict[str, object], denominator: dict[str, object], key: str) -> float:
    return float(numerator[key]) / max(float(denominator[key]), 1.0e-14)


def stage38_decision(
    first_lie: dict[str, object],
    muscl_lie: dict[str, object],
    first_strang: dict[str, object],
    muscl_strang: dict[str, object],
) -> str:
    if not bool(first_strang["converged"]) or not bool(muscl_strang["converged"]):
        return "stage38_nonconvergence_stage39_numerical_stability_audit"

    first_split_q = _error_ratio(first_strang, first_lie, "qav_relative_error")
    first_split_v = _error_ratio(
        first_strang["velocity_metrics"], first_lie["velocity_metrics"], "relative_rms"
    )
    muscl_rescue_q = _error_ratio(muscl_strang, muscl_lie, "qav_relative_error")
    muscl_rescue_v = _error_ratio(
        muscl_strang["velocity_metrics"], muscl_lie["velocity_metrics"], "relative_rms"
    )
    muscl_vs_baseline_q = _error_ratio(
        muscl_strang, first_lie, "qav_relative_error"
    )
    muscl_vs_baseline_v = _error_ratio(
        muscl_strang["velocity_metrics"], first_lie["velocity_metrics"], "relative_rms"
    )
    sign_ok = float(muscl_strang["velocity_metrics"]["sign_agreement"]) >= float(
        first_lie["velocity_metrics"]["sign_agreement"]
    )

    if (
        muscl_rescue_q <= 0.50
        and muscl_rescue_v <= 0.50
        and muscl_vs_baseline_q <= 0.90
        and muscl_vs_baseline_v <= 0.90
        and sign_ok
    ):
        return "coupling_rescues_muscl_stage39_high_resolution_confirmation"
    if first_split_q <= 0.90 and first_split_v <= 0.90 and not (
        muscl_vs_baseline_q <= 0.90 and muscl_vs_baseline_v <= 0.90 and sign_ok
    ):
        return "splitting_dominates_stage39_collision_time_integration_audit"
    if muscl_rescue_q <= 0.90 and muscl_rescue_v <= 0.90 and sign_ok:
        return "partial_muscl_rescue_stage39_limiter_flux_consistency_audit"
    return "no_coupling_rescue_stage39_collision_model_or_benchmark_audit"


def run_stage38(
    output_dir: str | Path,
    *,
    grid: tuple[int, int] = STAGE38_GRID,
    cfl: float = STAGE38_CFL,
    limiter_theta: float = STAGE38_LIMITER_THETA,
    max_steps: int = 16000,
    tolerance: float = 2.0e-5,
) -> dict[str, object]:
    validate_stage38_design(grid, cfl, limiter_theta, max_steps, tolerance)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    quadrature = spherical_product(16, 12, 24, 5.0, STAGE37_QUADRATURE)
    cfg = LinearSidewallConfig(
        nx=grid[0],
        ny=grid[1],
        nv=19,
        velocity_extent=5.0,
        kn0=STAGE37_KNUDSEN,
        cold_hot_ratio=STAGE37_RATIO,
        max_steps=max_steps,
        cfl=cfl,
        tolerance=tolerance,
        check_interval=100,
        minimum_steps=1800,
    )

    first_result = solve_strang_reduced_case(
        cfg, quadrature, transport_order="first_order_upwind"
    )
    muscl_result = solve_strang_reduced_case(
        cfg,
        quadrature,
        transport_order="muscl_ssprk2",
        limiter_theta=limiter_theta,
    )
    first_strang, first_profiles = transport_case_metrics(
        first_result, cfg, "first_order_strang_explicit", quadrature
    )
    muscl_strang, muscl_profiles = transport_case_metrics(
        muscl_result, cfg, "muscl_strang_explicit", quadrature
    )
    first_lie = STAGE37_COMPLETED_ENDPOINT["rows"]["first_order_lie_explicit"]
    muscl_lie = STAGE37_COMPLETED_ENDPOINT["rows"]["muscl_lie_explicit"]
    comparison = {
        "first_order_strang_to_lie_qav_error_ratio": _error_ratio(
            first_strang, first_lie, "qav_relative_error"
        ),
        "first_order_strang_to_lie_velocity_error_ratio": _error_ratio(
            first_strang["velocity_metrics"],
            first_lie["velocity_metrics"],
            "relative_rms",
        ),
        "muscl_strang_to_lie_qav_error_ratio": _error_ratio(
            muscl_strang, muscl_lie, "qav_relative_error"
        ),
        "muscl_strang_to_lie_velocity_error_ratio": _error_ratio(
            muscl_strang["velocity_metrics"],
            muscl_lie["velocity_metrics"],
            "relative_rms",
        ),
        "muscl_strang_to_first_lie_qav_error_ratio": _error_ratio(
            muscl_strang, first_lie, "qav_relative_error"
        ),
        "muscl_strang_to_first_lie_velocity_error_ratio": _error_ratio(
            muscl_strang["velocity_metrics"],
            first_lie["velocity_metrics"],
            "relative_rms",
        ),
        "muscl_strang_sign_change_from_first_lie": (
            float(muscl_strang["velocity_metrics"]["sign_agreement"])
            - float(first_lie["velocity_metrics"]["sign_agreement"])
        ),
    }
    decision = stage38_decision(first_lie, muscl_lie, first_strang, muscl_strang)
    summary = {
        "stage": 38,
        "description": (
            "Fixed-physics 2x2 transport-order/operator-splitting interaction audit "
            "using exact retained Stage 37 Lie-explicit endpoints and new symmetric "
            "half-collision/full-transport/half-collision runs"
        ),
        "configuration": {
            "kn0": STAGE37_KNUDSEN,
            "cold_hot_ratio": STAGE37_RATIO,
            "grid": list(grid),
            "quadrature": STAGE37_QUADRATURE,
            "velocity_point_count": quadrature.point_count,
            "relaxation_mapping": "sqrt(2)*Kn0/sqrt(pi) in c0 coordinates",
            "collision": "Shakhov",
            "transport_orders": ["first_order_upwind", "muscl_ssprk2"],
            "couplings": ["retained_lie_explicit", STAGE38_COUPLING],
            "wall_observable": STAGE37_OBSERVABLE,
            "cfl_equal_in_all_arms": cfl,
            "limiter_theta": limiter_theta,
            "max_steps": max_steps,
            "tolerance": tolerance,
            "physical_parameter_retuning": False,
        },
        "retained_stage37_endpoint": STAGE37_COMPLETED_ENDPOINT,
        "new_rows": [first_strang, muscl_strang],
        "comparison": comparison,
        "decision": decision,
        "interpretation_guard": (
            "Knudsen number, temperatures, corrected relaxation mapping, Shakhov "
            "model, viscosity law, Prandtl number, quadrature, grid, normalization, "
            "walls, observable, CFL, stopping rule and positivity floor are frozen. "
            "Stage 38 changes only operator coupling: retained transport-then-collision "
            "Lie splitting versus explicit half-collision/full-transport/half-collision "
            "symmetric splitting. Negative and mixed outcomes are retained."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    arrays: dict[str, np.ndarray] = {}
    for scheme, result, profiles in (
        ("first_order_strang", first_result, first_profiles),
        ("muscl_strang", muscl_result, muscl_profiles),
    ):
        arrays[f"bottom_heat_flux_{scheme}"] = np.asarray(
            result["bottom_heat_flux"], dtype=np.float64
        )
        arrays[f"residual_history_{scheme}"] = np.asarray(
            result["residual_history"], dtype=np.float64
        )
        arrays[f"T_{scheme}"] = np.asarray(result["T"], dtype=np.float64)
        arrays[f"rho_{scheme}"] = np.asarray(result["rho"], dtype=np.float64)
        for name, profile in profiles.items():
            arrays[f"table_velocity_{name}_{scheme}"] = np.asarray(
                profile, dtype=np.float64
            )
    np.savez_compressed(output_dir / "fields_and_profiles.npz", **arrays)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Stage 38 transport-collision interaction audit"
    )
    parser.add_argument(
        "--output-dir", default="outputs/stage38_transport_collision_interaction_audit"
    )
    parser.add_argument("--max-steps", type=int, default=16000)
    parser.add_argument("--tolerance", type=float, default=2.0e-5)
    args = parser.parse_args()
    print(
        json.dumps(
            run_stage38(
                args.output_dir,
                max_steps=args.max_steps,
                tolerance=args.tolerance,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
