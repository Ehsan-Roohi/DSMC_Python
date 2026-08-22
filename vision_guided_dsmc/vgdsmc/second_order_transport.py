from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import numpy as np

from .bottom_heated_benchmark import _bottom_boundary_heat_flux
from .dvm_shakhov import ShakhovReferenceConfig, _macroscopic, _velocity_grid
from .dvm_shakhov_corrected import _shakhov_equilibrium
from .linear_sidewall_validation import (
    LinearSidewallConfig,
    TABLE3_UY_RATIO_0P1,
    TABLE3_Y,
    TABLE6_QAV_RATIO_0P1,
    _left_wall_tangential_velocity,
    _relative_rms,
    _wall_incoming,
    local_relaxation_time,
    sidewall_temperature_profile,
)


STAGE27_BASELINE = {
    "predicted_qav": 0.2005499953149862,
    "qav_relative_error": 0.3550675359120691,
    "wall_velocity_relative_rms": 3.9764569069981572,
    "wall_velocity_sign_agreement": 0.0,
    "iterations": 4500,
}


def minmod3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Three-argument minmod limiter."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    same_sign = ((a > 0.0) & (b > 0.0) & (c > 0.0)) | (
        (a < 0.0) & (b < 0.0) & (c < 0.0)
    )
    magnitude = np.minimum(np.minimum(np.abs(a), np.abs(b)), np.abs(c))
    return np.where(same_sign, np.sign(a) * magnitude, 0.0)


def limited_slopes(
    field: np.ndarray,
    axis: int,
    theta: float = 1.5,
) -> np.ndarray:
    """Return monotonized-central slopes with zero boundary slopes."""
    field = np.asarray(field, dtype=np.float64)
    if field.ndim < 2:
        raise ValueError("field must have at least two dimensions")
    if axis not in (0, 1):
        raise ValueError("axis must be zero or one")
    if not 1.0 <= theta <= 2.0:
        raise ValueError("theta must lie in [1,2]")
    slopes = np.zeros_like(field)
    if field.shape[axis] < 3:
        return slopes
    center = [slice(None)] * field.ndim
    lower = [slice(None)] * field.ndim
    upper = [slice(None)] * field.ndim
    center[axis] = slice(1, -1)
    lower[axis] = slice(0, -2)
    upper[axis] = slice(2, None)
    backward = field[tuple(center)] - field[tuple(lower)]
    forward = field[tuple(upper)] - field[tuple(center)]
    centered = 0.5 * (field[tuple(upper)] - field[tuple(lower)])
    slopes[tuple(center)] = minmod3(theta * backward, centered, theta * forward)
    return slopes


def positivity_blend(
    old: np.ndarray,
    candidate: np.ndarray,
    floor: float,
) -> np.ndarray:
    """Cell-wise convex limiting that preserves all velocity ordinates."""
    old = np.asarray(old, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if old.shape != candidate.shape or old.ndim != 5:
        raise ValueError("old and candidate must be matching five-dimensional arrays")
    below = candidate < floor
    denominator = np.maximum(old - candidate, 1.0e-300)
    ratio = np.where(below, (old - floor) / denominator, 1.0)
    theta = np.clip(np.min(ratio, axis=(-3, -2, -1)), 0.0, 1.0)
    limited = old + theta[..., None, None, None] * (candidate - old)
    return np.maximum(limited, floor)


def muscl_flux_divergence(
    distribution: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    bottom: np.ndarray,
    top: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    dx: float,
    dy: float,
    theta: float = 1.5,
) -> np.ndarray:
    """Conservative second-order upwind divergence for Cartesian DVM transport."""
    f = np.asarray(distribution, dtype=np.float64)
    if f.ndim != 5:
        raise ValueError("distribution must have shape (ny,nx,nv,nv,nv)")
    ny, nx = f.shape[:2]
    sx = limited_slopes(f, axis=1, theta=theta)
    sy = limited_slopes(f, axis=0, theta=theta)
    velocity_x = vx[None, None]
    velocity_y = vy[None, None]

    x_divergence = np.empty_like(f)
    if nx > 1:
        left_state = f[:, :-1] + 0.5 * sx[:, :-1]
        right_state = f[:, 1:] - 0.5 * sx[:, 1:]
        interior_flux_x = velocity_x * np.where(
            velocity_x >= 0.0, left_state, right_state
        )
        boundary_left_flux = vx[None] * np.where(
            vx[None] >= 0.0, left, f[:, 0]
        )
        boundary_right_flux = vx[None] * np.where(
            vx[None] >= 0.0, f[:, -1], right
        )
        x_divergence[:, 0] = (interior_flux_x[:, 0] - boundary_left_flux) / dx
        if nx > 2:
            x_divergence[:, 1:-1] = (
                interior_flux_x[:, 1:] - interior_flux_x[:, :-1]
            ) / dx
        x_divergence[:, -1] = (
            boundary_right_flux - interior_flux_x[:, -1]
        ) / dx
    else:
        boundary_left_flux = vx[None] * np.where(vx[None] >= 0.0, left, f[:, 0])
        boundary_right_flux = vx[None] * np.where(vx[None] >= 0.0, f[:, 0], right)
        x_divergence[:, 0] = (boundary_right_flux - boundary_left_flux) / dx

    y_divergence = np.empty_like(f)
    if ny > 1:
        lower_state = f[:-1] + 0.5 * sy[:-1]
        upper_state = f[1:] - 0.5 * sy[1:]
        interior_flux_y = velocity_y * np.where(
            velocity_y >= 0.0, lower_state, upper_state
        )
        boundary_bottom_flux = vy[None] * np.where(
            vy[None] >= 0.0, bottom, f[0]
        )
        boundary_top_flux = vy[None] * np.where(
            vy[None] >= 0.0, f[-1], top
        )
        y_divergence[0] = (interior_flux_y[0] - boundary_bottom_flux) / dy
        if ny > 2:
            y_divergence[1:-1] = (
                interior_flux_y[1:] - interior_flux_y[:-1]
            ) / dy
        y_divergence[-1] = (boundary_top_flux - interior_flux_y[-1]) / dy
    else:
        boundary_bottom_flux = vy[None] * np.where(vy[None] >= 0.0, bottom, f[0])
        boundary_top_flux = vy[None] * np.where(vy[None] >= 0.0, f[0], top)
        y_divergence[0] = (boundary_top_flux - boundary_bottom_flux) / dy

    return x_divergence + y_divergence


def _transport_euler(
    distribution: np.ndarray,
    cfg: LinearSidewallConfig,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    dv: float,
    dt: float,
    dx: float,
    dy: float,
    theta: float,
) -> np.ndarray:
    left, right, bottom, top = _wall_incoming(
        distribution, cfg, vx, vy, vz, dv
    )
    divergence = muscl_flux_divergence(
        distribution, left, right, bottom, top, vx, vy, dx, dy, theta
    )
    candidate = distribution - dt * divergence
    return positivity_blend(distribution, candidate, cfg.positivity_floor)


def transport_ssprk2(
    distribution: np.ndarray,
    cfg: LinearSidewallConfig,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    dv: float,
    dt: float,
    dx: float,
    dy: float,
    theta: float = 1.5,
) -> np.ndarray:
    """Second-order TVD transport with wall states recomputed at each stage."""
    stage_one = _transport_euler(
        distribution, cfg, vx, vy, vz, dv, dt, dx, dy, theta
    )
    stage_two_euler = _transport_euler(
        stage_one, cfg, vx, vy, vz, dv, dt, dx, dy, theta
    )
    return np.maximum(
        0.5 * distribution + 0.5 * stage_two_euler,
        cfg.positivity_floor,
    )


def solve_second_order_case(
    cfg: LinearSidewallConfig,
    limiter_theta: float = 1.5,
) -> dict[str, object]:
    if cfg.nv % 2 == 0:
        raise ValueError("an odd velocity count is required")
    velocity_cfg = ShakhovReferenceConfig(
        nx=cfg.nx,
        ny=cfg.ny,
        nv=cfg.nv,
        velocity_extent=cfg.velocity_extent,
    )
    vx, vy, vz, dv = _velocity_grid(velocity_cfg)
    rho = np.ones((cfg.ny, cfg.nx))
    zero = np.zeros_like(rho)
    initial_temperature = np.repeat(
        sidewall_temperature_profile(cfg)[:, None], cfg.nx, axis=1
    )
    from .dvm_shakhov_corrected import _discrete_maxwellian

    distribution = _discrete_maxwellian(
        rho, zero, zero, zero, initial_temperature, vx, vy, vz, dv
    )
    dx, dy = 1.0 / cfg.nx, 1.0 / cfg.ny
    maximum_speed = max(float(np.max(np.abs(vx))), float(np.max(np.abs(vy))))
    dt = min(cfg.cfl, 0.24) * min(dx / maximum_speed, dy / maximum_speed)
    previous = _macroscopic(distribution, vx, vy, vz, dv)
    previous_T = previous["T"].copy()
    previous_u = np.stack([previous["u"], previous["v"]], axis=-1)
    previous_q = np.stack([previous["qx"], previous["qy"]], axis=-1)
    residual_history: list[float] = []
    converged = False

    for step in range(cfg.max_steps):
        transported = transport_ssprk2(
            distribution, cfg, vx, vy, vz, dv, dt, dx, dy, limiter_theta
        )
        fields = _macroscopic(transported, vx, vy, vz, dv)
        equilibrium = _shakhov_equilibrium(fields, vx, vy, vz, dv, cfg.prandtl)
        tau = local_relaxation_time(fields["rho"], fields["T"], cfg)
        fraction = np.minimum(dt / tau, 1.0)[..., None, None, None]
        distribution = np.maximum(
            transported + fraction * (equilibrium - transported),
            cfg.positivity_floor,
        )
        if (step + 1) % cfg.check_interval == 0:
            fields = _macroscopic(distribution, vx, vy, vz, dv)
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

    left, _, bottom, _ = _wall_incoming(distribution, cfg, vx, vy, vz, dv)
    fields = _macroscopic(distribution, vx, vy, vz, dv)
    wall_velocity = _left_wall_tangential_velocity(
        distribution, left, vx, vy, vz, dv
    )
    y_centers = (np.arange(cfg.ny, dtype=np.float64) + 0.5) / cfg.ny
    table_velocity = np.interp(TABLE3_Y, y_centers, wall_velocity)
    bottom_heat_flux = _bottom_boundary_heat_flux(
        distribution, bottom, vx, vy, vz, dv
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
        "bottom_heat_flux": bottom_heat_flux,
        "residual_history": np.asarray(residual_history),
        "iterations": step + 1,
        "converged": converged,
        "dt": dt,
    }


def run_stage28(
    output_dir: str | Path,
    cfg: LinearSidewallConfig,
    limiter_theta: float = 1.5,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if cfg.kn0 != 1.0 or cfg.cold_hot_ratio != 0.1:
        raise ValueError("Stage 28 is fixed to Kn0=1 and TC/TH=0.1")
    result = solve_second_order_case(cfg, limiter_theta)
    predicted_u = np.asarray(result["table_velocity"], dtype=np.float64)
    reference_u = TABLE3_UY_RATIO_0P1[1.0]
    predicted_q = float(np.mean(result["bottom_heat_flux"]))
    reference_q = TABLE6_QAV_RATIO_0P1[1.0]
    second_order = {
        "iterations": int(result["iterations"]),
        "converged": bool(result["converged"]),
        "final_change": float(np.asarray(result["residual_history"])[-1]),
        "predicted_qav": predicted_q,
        "literature_qav": reference_q,
        "qav_relative_error": abs(predicted_q - reference_q) / reference_q,
        "wall_velocity_relative_rms": _relative_rms(predicted_u, reference_u),
        "wall_velocity_sign_agreement": float(
            np.mean(np.sign(predicted_u) == np.sign(reference_u))
        ),
        "minimum_distribution_proxy": cfg.positivity_floor,
    }
    summary = {
        "stage": 28,
        "description": "Fixed-physics MUSCL SSP-RK2 positivity-preserving transport comparison",
        "configuration": {
            "grid": [cfg.nx, cfg.ny],
            "nv": cfg.nv,
            "kn0": cfg.kn0,
            "cold_hot_ratio": cfg.cold_hot_ratio,
            "limiter": "monotonized-central minmod3",
            "limiter_theta": limiter_theta,
            "time_integrator": "SSP-RK2 transport plus explicit Shakhov relaxation",
            "physics_frozen": True,
        },
        "first_order_baseline": STAGE27_BASELINE,
        "second_order": second_order,
        "fractional_changes": {
            "qav_error": second_order["qav_relative_error"]
            / STAGE27_BASELINE["qav_relative_error"]
            - 1.0,
            "wall_velocity_error": second_order["wall_velocity_relative_rms"]
            / STAGE27_BASELINE["wall_velocity_relative_rms"]
            - 1.0,
            "sign_agreement": second_order["wall_velocity_sign_agreement"]
            - STAGE27_BASELINE["wall_velocity_sign_agreement"],
        },
        "decision": {
            "qav_error_reduced_by_at_least_10pct": bool(
                second_order["qav_relative_error"]
                <= 0.9 * STAGE27_BASELINE["qav_relative_error"]
            ),
            "velocity_error_reduced_by_at_least_10pct": bool(
                second_order["wall_velocity_relative_rms"]
                <= 0.9 * STAGE27_BASELINE["wall_velocity_relative_rms"]
            ),
            "velocity_sign_improved": bool(
                second_order["wall_velocity_sign_agreement"]
                > STAGE27_BASELINE["wall_velocity_sign_agreement"]
            ),
        },
        "interpretation_guard": (
            "No Knudsen, wall, collision, or normalization parameter was retuned. "
            "Failure points to Cartesian velocity quadrature/model details rather than spatial order alone."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        output_dir / "fields_and_profiles.npz",
        **{name: np.asarray(result[name]) for name in (
            "T", "rho", "u", "v", "qx", "qy", "left_wall_velocity",
            "table_velocity", "bottom_heat_flux", "residual_history",
        )},
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 28 second-order transport audit")
    parser.add_argument("--output-dir", default="outputs/stage28_second_order")
    parser.add_argument("--nx", type=int, default=20)
    parser.add_argument("--ny", type=int, default=20)
    parser.add_argument("--nv", type=int, default=17)
    parser.add_argument("--velocity-extent", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=7000)
    parser.add_argument("--tolerance", type=float, default=2.0e-5)
    parser.add_argument("--limiter-theta", type=float, default=1.5)
    args = parser.parse_args()
    cfg = LinearSidewallConfig(
        nx=args.nx,
        ny=args.ny,
        nv=args.nv,
        velocity_extent=args.velocity_extent,
        kn0=1.0,
        cold_hot_ratio=0.1,
        max_steps=args.max_steps,
        tolerance=args.tolerance,
        cfl=0.24,
    )
    print(json.dumps(run_stage28(args.output_dir, cfg, args.limiter_theta), indent=2))


if __name__ == "__main__":
    main()
