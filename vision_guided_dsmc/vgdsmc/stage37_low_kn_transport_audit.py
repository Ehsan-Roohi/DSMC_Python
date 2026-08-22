from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import numpy as np

from .linear_sidewall_validation import (
    LinearSidewallConfig,
    TABLE3_UY_RATIO_0P1,
    TABLE6_QAV_RATIO_0P1,
    sidewall_temperature_profile,
)
from .reduced_spherical_solver import (
    bottom_heat_flux,
    discrete_maxwellian,
    left_wall_tangential_velocity,
    macroscopic,
    shakhov_equilibrium,
    wall_incoming,
    wall_mass_balance_error,
)
from .second_order_transport import limited_slopes
from .stage32_near_continuum_observable_audit import (
    observable_metrics,
    wall_observable_profiles,
)
from .stage34_velocity_scale_consistency import (
    local_relaxation_time,
    solve_reduced_case_with_mapping,
)
from .velocity_quadrature_audit import VelocityQuadrature, spherical_product


STAGE37_KNUDSEN = 0.1
STAGE37_RATIO = 0.1
STAGE37_GRID = (24, 24)
STAGE37_QUADRATURE = "spherical_matched_r16_mu12_phi24"
STAGE37_OBSERVABLE = "linear_extrapolated_wall"
STAGE37_CFL = 0.20
STAGE37_LIMITER_THETA = 1.5

# Exact Stage 36 endpoints are retained verbatim for context and are not
# silently replaced by the same-CFL Stage 37 transport comparison.
STAGE36_LOW_KN_ENDPOINTS = {
    "24x24": {
        "iterations": 3200,
        "converged": True,
        "final_change": 1.7391815292137035e-05,
        "predicted_qav": 0.07460201631883724,
        "literature_qav": 0.072,
        "qav_relative_error": 0.036139115539406214,
        "wall_velocity_relative_rms": 0.9006380440004526,
        "wall_velocity_sign_agreement": 0.8,
        "wall_mass_balance_relative_error": 1.6713371204475987e-16,
    },
    "36x36": {
        "iterations": 5200,
        "converged": True,
        "final_change": 1.5026217704616068e-05,
        "predicted_qav": 0.0729750813492751,
        "literature_qav": 0.072,
        "qav_relative_error": 0.013542796517709883,
        "wall_velocity_relative_rms": 0.9266806666880097,
        "wall_velocity_sign_agreement": 0.8,
        "wall_mass_balance_relative_error": 2.139677787130385e-16,
    },
    "profile_change_24x24_to_36x36": 0.24659632808808882,
}


def validate_stage37_design(
    grid: tuple[int, int],
    cfl: float,
    limiter_theta: float,
    max_steps: int,
    tolerance: float,
) -> None:
    if grid != STAGE37_GRID:
        raise ValueError("Stage 37 is fixed to the 24x24 Kn0=0.1 transport screen")
    if cfl != STAGE37_CFL:
        raise ValueError("Stage 37 CFL is fixed equally in both transport arms")
    if limiter_theta != STAGE37_LIMITER_THETA:
        raise ValueError("Stage 37 limiter theta is preregistered at 1.5")
    if max_steps <= 0 or tolerance <= 0.0:
        raise ValueError("max_steps and tolerance must be positive")


def positivity_blend_reduced(
    old: np.ndarray,
    candidate: np.ndarray,
    floor: float,
) -> np.ndarray:
    """Cell-wise convex limiting for an arbitrary positive velocity rule."""
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
    return np.maximum(limited, floor)


def muscl_flux_divergence_reduced(
    distribution: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    bottom: np.ndarray,
    top: np.ndarray,
    quadrature: VelocityQuadrature,
    dx: float,
    dy: float,
    theta: float = STAGE37_LIMITER_THETA,
) -> np.ndarray:
    """Conservative MUSCL divergence for distribution shape (ny,nx,nq)."""
    f = np.asarray(distribution, dtype=np.float64)
    if f.ndim != 3 or f.shape[-1] != quadrature.point_count:
        raise ValueError("distribution and quadrature dimensions do not match")
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("cell widths must be positive")
    ny, nx = f.shape[:2]
    sx = limited_slopes(f, axis=1, theta=theta)
    sy = limited_slopes(f, axis=0, theta=theta)
    vx = quadrature.vx[None, None, :]
    vy = quadrature.vy[None, None, :]

    x_divergence = np.empty_like(f)
    if nx > 1:
        left_state = f[:, :-1] + 0.5 * sx[:, :-1]
        right_state = f[:, 1:] - 0.5 * sx[:, 1:]
        interior_flux = vx * np.where(vx >= 0.0, left_state, right_state)
        boundary_left = quadrature.vx[None, :] * np.where(
            quadrature.vx[None, :] >= 0.0, left, f[:, 0]
        )
        boundary_right = quadrature.vx[None, :] * np.where(
            quadrature.vx[None, :] >= 0.0, f[:, -1], right
        )
        x_divergence[:, 0] = (interior_flux[:, 0] - boundary_left) / dx
        if nx > 2:
            x_divergence[:, 1:-1] = (
                interior_flux[:, 1:] - interior_flux[:, :-1]
            ) / dx
        x_divergence[:, -1] = (boundary_right - interior_flux[:, -1]) / dx
    else:
        boundary_left = quadrature.vx[None, :] * np.where(
            quadrature.vx[None, :] >= 0.0, left, f[:, 0]
        )
        boundary_right = quadrature.vx[None, :] * np.where(
            quadrature.vx[None, :] >= 0.0, f[:, 0], right
        )
        x_divergence[:, 0] = (boundary_right - boundary_left) / dx

    y_divergence = np.empty_like(f)
    if ny > 1:
        lower_state = f[:-1] + 0.5 * sy[:-1]
        upper_state = f[1:] - 0.5 * sy[1:]
        interior_flux = vy * np.where(vy >= 0.0, lower_state, upper_state)
        boundary_bottom = quadrature.vy[None, :] * np.where(
            quadrature.vy[None, :] >= 0.0, bottom, f[0]
        )
        boundary_top = quadrature.vy[None, :] * np.where(
            quadrature.vy[None, :] >= 0.0, f[-1], top
        )
        y_divergence[0] = (interior_flux[0] - boundary_bottom) / dy
        if ny > 2:
            y_divergence[1:-1] = (
                interior_flux[1:] - interior_flux[:-1]
            ) / dy
        y_divergence[-1] = (boundary_top - interior_flux[-1]) / dy
    else:
        boundary_bottom = quadrature.vy[None, :] * np.where(
            quadrature.vy[None, :] >= 0.0, bottom, f[0]
        )
        boundary_top = quadrature.vy[None, :] * np.where(
            quadrature.vy[None, :] >= 0.0, f[0], top
        )
        y_divergence[0] = (boundary_top - boundary_bottom) / dy

    return x_divergence + y_divergence


def _transport_euler_reduced(
    distribution: np.ndarray,
    cfg: LinearSidewallConfig,
    quadrature: VelocityQuadrature,
    dt: float,
    dx: float,
    dy: float,
    limiter_theta: float,
) -> np.ndarray:
    left, right, bottom, top = wall_incoming(distribution, cfg, quadrature)
    divergence = muscl_flux_divergence_reduced(
        distribution,
        left,
        right,
        bottom,
        top,
        quadrature,
        dx,
        dy,
        limiter_theta,
    )
    candidate = distribution - dt * divergence
    return positivity_blend_reduced(distribution, candidate, cfg.positivity_floor)


def transport_ssprk2_reduced(
    distribution: np.ndarray,
    cfg: LinearSidewallConfig,
    quadrature: VelocityQuadrature,
    dt: float,
    dx: float,
    dy: float,
    limiter_theta: float = STAGE37_LIMITER_THETA,
) -> np.ndarray:
    """Second-order TVD transport with wall states rebuilt at both RK stages."""
    stage_one = _transport_euler_reduced(
        distribution, cfg, quadrature, dt, dx, dy, limiter_theta
    )
    stage_two = _transport_euler_reduced(
        stage_one, cfg, quadrature, dt, dx, dy, limiter_theta
    )
    return positivity_blend_reduced(
        distribution,
        0.5 * distribution + 0.5 * stage_two,
        cfg.positivity_floor,
    )


def solve_muscl_reduced_case_with_mapping(
    cfg: LinearSidewallConfig,
    quadrature: VelocityQuadrature,
    *,
    mapping: str = "paper_consistent_c0",
    limiter_theta: float = STAGE37_LIMITER_THETA,
) -> dict[str, object]:
    """Solve the reduced Shakhov cavity with MUSCL/SSP-RK2 transport."""
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
        transported = transport_ssprk2_reduced(
            distribution,
            cfg,
            quadrature,
            dt,
            dx,
            dy,
            limiter_theta,
        )
        fields = macroscopic(transported, quadrature)
        equilibrium = shakhov_equilibrium(fields, quadrature, cfg.prandtl)
        tau = local_relaxation_time(
            fields["rho"], fields["T"], cfg, mapping=mapping
        )
        fraction = np.minimum(dt / tau, 1.0)[..., None]
        distribution = positivity_blend_reduced(
            transported,
            transported + fraction * (equilibrium - transported),
            cfg.positivity_floor,
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


def transport_case_metrics(
    result: dict[str, object],
    cfg: LinearSidewallConfig,
    scheme: str,
    quadrature: VelocityQuadrature,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    profiles = wall_observable_profiles(result, cfg)
    reference_velocity = TABLE3_UY_RATIO_0P1[STAGE37_KNUDSEN]
    selected = profiles[STAGE37_OBSERVABLE]
    velocity_metrics = observable_metrics(selected, reference_velocity)
    qav = float(np.mean(np.asarray(result["bottom_heat_flux"], dtype=np.float64)))
    reference_qav = TABLE6_QAV_RATIO_0P1[STAGE37_KNUDSEN]
    residual = np.asarray(result["residual_history"], dtype=np.float64)
    row = {
        "scheme": scheme,
        "iterations": int(result["iterations"]),
        "converged": bool(result["converged"]),
        "final_change": float(residual[-1]) if residual.size else float("nan"),
        "predicted_qav": qav,
        "literature_qav": reference_qav,
        "qav_relative_error": abs(qav - reference_qav) / reference_qav,
        "velocity_observable": STAGE37_OBSERVABLE,
        "velocity_metrics": velocity_metrics,
        "wall_mass_balance_relative_error": float(
            result["wall_mass_balance_relative_error"]
        ),
        "minimum_distribution": float(result["minimum_distribution"]),
        "minimum_temperature": float(np.min(np.asarray(result["T"]))),
        "maximum_temperature": float(np.max(np.asarray(result["T"]))),
        "dt": float(result["dt"]),
        "work_proxy": int(result["iterations"])
        * cfg.nx
        * cfg.ny
        * quadrature.point_count,
    }
    return row, profiles


def stage37_decision(
    first_order: dict[str, object],
    muscl: dict[str, object],
) -> str:
    if not bool(first_order["converged"]) or not bool(muscl["converged"]):
        return "transport_audit_nonconvergence_stage38_numerical_stability"

    first_velocity = first_order["velocity_metrics"]
    muscl_velocity = muscl["velocity_metrics"]
    q_ratio = float(muscl["qav_relative_error"]) / max(
        float(first_order["qav_relative_error"]), 1.0e-14
    )
    v_ratio = float(muscl_velocity["relative_rms"]) / max(
        float(first_velocity["relative_rms"]), 1.0e-14
    )
    sign_change = float(muscl_velocity["sign_agreement"]) - float(
        first_velocity["sign_agreement"]
    )

    if q_ratio <= 0.90 and v_ratio <= 0.90 and sign_change >= 0.0:
        return "muscl_screen_positive_stage38_high_resolution_confirmation"
    if q_ratio <= 0.90 or v_ratio <= 0.90 or sign_change > 0.0:
        return "muscl_screen_mixed_stage38_transport_collision_interaction_audit"
    return "muscl_screen_negative_stage38_collision_model_audit"


def run_stage37(
    output_dir: str | Path,
    *,
    grid: tuple[int, int] = STAGE37_GRID,
    cfl: float = STAGE37_CFL,
    limiter_theta: float = STAGE37_LIMITER_THETA,
    max_steps: int = 16000,
    tolerance: float = 2.0e-5,
) -> dict[str, object]:
    """Run a same-CFL first-order versus MUSCL screen at Kn0=0.1."""
    validate_stage37_design(grid, cfl, limiter_theta, max_steps, tolerance)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    quadrature = spherical_product(
        16, 12, 24, 5.0, STAGE37_QUADRATURE
    )
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

    first_result = solve_reduced_case_with_mapping(
        cfg, quadrature, mapping="paper_consistent_c0"
    )
    muscl_result = solve_muscl_reduced_case_with_mapping(
        cfg,
        quadrature,
        mapping="paper_consistent_c0",
        limiter_theta=limiter_theta,
    )
    first_row, first_profiles = transport_case_metrics(
        first_result, cfg, "first_order_upwind", quadrature
    )
    muscl_row, muscl_profiles = transport_case_metrics(
        muscl_result, cfg, "muscl_ssprk2", quadrature
    )
    comparison = {
        "qav_error_ratio_muscl_to_first_order": (
            float(muscl_row["qav_relative_error"])
            / max(float(first_row["qav_relative_error"]), 1.0e-14)
        ),
        "velocity_error_ratio_muscl_to_first_order": (
            float(muscl_row["velocity_metrics"]["relative_rms"])
            / max(float(first_row["velocity_metrics"]["relative_rms"]), 1.0e-14)
        ),
        "sign_agreement_change": (
            float(muscl_row["velocity_metrics"]["sign_agreement"])
            - float(first_row["velocity_metrics"]["sign_agreement"])
        ),
        "qav_change": (
            float(muscl_row["predicted_qav"])
            - float(first_row["predicted_qav"])
        ),
    }
    decision = stage37_decision(first_row, muscl_row)
    summary = {
        "stage": 37,
        "description": (
            "Same-CFL corrected-mapping spherical-quadrature transport-order "
            "screen for the unresolved Kn0=0.1 wall-velocity discrepancy"
        ),
        "configuration": {
            "kn0": STAGE37_KNUDSEN,
            "cold_hot_ratio": STAGE37_RATIO,
            "grid": list(grid),
            "quadrature": STAGE37_QUADRATURE,
            "velocity_point_count": quadrature.point_count,
            "relaxation_mapping": "sqrt(2)*Kn0/sqrt(pi) in c0 coordinates",
            "collision": "Shakhov",
            "transport_arms": [
                "first-order upwind",
                "MUSCL minmod3 plus SSP-RK2",
            ],
            "wall_observable": STAGE37_OBSERVABLE,
            "cfl_equal_in_both_arms": cfl,
            "limiter_theta": limiter_theta,
            "max_steps": max_steps,
            "tolerance": tolerance,
            "physical_parameter_retuning": False,
        },
        "retained_stage36_low_kn_endpoints": STAGE36_LOW_KN_ENDPOINTS,
        "rows": [first_row, muscl_row],
        "comparison": comparison,
        "decision": decision,
        "interpretation_guard": (
            "Knudsen number, temperatures, corrected relaxation mapping, "
            "Shakhov collision model, viscosity law, Prandtl number, velocity "
            "quadrature, normalization, grid, wall model, wall observable, CFL, "
            "stopping rule, and positivity floor are identical between arms. "
            "Only spatial transport order changes, and negative or mixed results "
            "are retained."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    arrays: dict[str, np.ndarray] = {}
    for scheme, result, profiles in (
        ("first_order", first_result, first_profiles),
        ("muscl", muscl_result, muscl_profiles),
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
        description="Run Stage 37 corrected-mapping reduced MUSCL transport audit"
    )
    parser.add_argument(
        "--output-dir", default="outputs/stage37_low_kn_transport_audit"
    )
    parser.add_argument("--max-steps", type=int, default=16000)
    parser.add_argument("--tolerance", type=float, default=2.0e-5)
    args = parser.parse_args()
    print(
        json.dumps(
            run_stage37(
                args.output_dir,
                max_steps=args.max_steps,
                tolerance=args.tolerance,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
