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
    sidewall_temperature_profile,
)
from .stage34_velocity_scale_consistency import local_relaxation_time
from .stage41_projected_polar_operator_audit import (
    PolarQuadrature,
    STAGE41_CORRECTION_FLOOR,
    STAGE41_FINE_RULE,
    STAGE41_PRANDTL,
    mapped_polar_quadrature,
    projected_macroscopic,
    projected_maxwellian,
    projected_shakhov_equilibrium,
    wall_mass_balance_error,
)


STAGE41_COMPLETED_ENDPOINT = {
    "workflow_run_id": 30727704751,
    "workflow_job_id": 91442360250,
    "workflow_conclusion": "success",
    "regression_tests_passed": 38,
    "regression_tests_failed": 0,
    "artifact_id": 8827830725,
    "artifact_sha256": "0ed4c1c99a7424f1aaa67037121a1974c72f5dba1571615cde575f16e8776a05",
    "head_sha": "ca0fa6215bbd8078b2ebc0b59f7eb3780ac5c56b",
    "fine_maxwellian_moment_error": 5.417417359154797e-08,
    "fine_shakhov_invariant_error": 4.875995179049702e-08,
    "fine_shakhov_heat_flux_closure_error": 2.1704939264108322e-04,
    "wall_mass_balance_relative_error": 3.5989022049905664e-17,
    "homogeneous_maximum_conserved_moment_change": 1.133500697801099e-08,
    "decision": "projected_polar_operators_pass_stage42_heated_cavity_pilot",
}

STAGE42_GRID = (8, 8)
STAGE42_RULE = STAGE41_FINE_RULE
STAGE42_KNUDSEN = 0.1
STAGE42_RATIO = 0.1
STAGE42_MAX_ITERATIONS = 3000
STAGE42_MINIMUM_ITERATIONS = 500
STAGE42_CHECK_INTERVAL = 25
STAGE42_TOLERANCE = 2.0e-5
STAGE42_TRANSPORT = "steady_first_order_upwind_jacobi"
STAGE42_COLLISION = "projected_shakhov"
STAGE42_RELAXATION_MAPPING = "sqrt(2)*Kn0/sqrt(pi) in c0 coordinates"
STAGE42_SOURCE_RELAXATION = 1.0


def validate_stage42_design(
    grid: tuple[int, int],
    rule: tuple[int, int],
    kn0: float,
    cold_hot_ratio: float,
    max_iterations: int,
    tolerance: float,
    source_relaxation: float,
) -> None:
    if grid != STAGE42_GRID:
        raise ValueError("Stage 42 is fixed to the 8x8 physical pilot grid")
    if rule != STAGE42_RULE:
        raise ValueError("Stage 42 is fixed to the Stage 41 fine 32x96 polar rule")
    if kn0 != STAGE42_KNUDSEN:
        raise ValueError("Stage 42 is fixed to Kn0=0.1")
    if cold_hot_ratio != STAGE42_RATIO:
        raise ValueError("Stage 42 is fixed to Tcold/Thot=0.1")
    if max_iterations != STAGE42_MAX_ITERATIONS:
        raise ValueError("Stage 42 uses a preregistered 3000-iteration pilot horizon")
    if tolerance != STAGE42_TOLERANCE:
        raise ValueError("Stage 42 retains tolerance=2e-5")
    if source_relaxation != STAGE42_SOURCE_RELAXATION:
        raise ValueError("Stage 42 does not tune source-iteration relaxation")


def _profile_diffuse_incoming(
    outgoing_phi: np.ndarray,
    outgoing_psi: np.ndarray,
    normal_velocity: np.ndarray,
    incoming_mask: np.ndarray,
    wall_temperature: np.ndarray,
    quadrature: PolarQuadrature,
) -> tuple[np.ndarray, np.ndarray]:
    outgoing_phi = np.asarray(outgoing_phi, dtype=np.float64)
    outgoing_psi = np.asarray(outgoing_psi, dtype=np.float64)
    wall_temperature = np.asarray(wall_temperature, dtype=np.float64)
    if outgoing_phi.shape != outgoing_psi.shape:
        raise ValueError("outgoing phi and psi must match")
    if outgoing_phi.ndim != 2 or outgoing_phi.shape[-1] != quadrature.point_count:
        raise ValueError("profile distributions must have shape (nprofile,nq)")
    if wall_temperature.shape != (outgoing_phi.shape[0],):
        raise ValueError("wall-temperature profile length must match boundary profile")
    one = np.ones_like(wall_temperature)
    zero = np.zeros_like(wall_temperature)
    wall_phi, wall_psi = projected_maxwellian(
        one, zero, zero, wall_temperature, quadrature
    )
    outgoing_mask = (~incoming_mask) & (np.abs(normal_velocity) > 0.0)
    outgoing_flux = np.sum(
        normal_velocity[None, :]
        * outgoing_phi
        * outgoing_mask[None, :]
        * quadrature.weight[None, :],
        axis=-1,
    )
    incoming_unit_flux = np.sum(
        normal_velocity[None, :]
        * wall_phi
        * incoming_mask[None, :]
        * quadrature.weight[None, :],
        axis=-1,
    )
    denominator = np.where(
        np.abs(incoming_unit_flux) > 1.0e-14,
        incoming_unit_flux,
        np.copysign(1.0e-14, incoming_unit_flux + 1.0e-300),
    )
    scale = -outgoing_flux / denominator
    return scale[:, None] * wall_phi, scale[:, None] * wall_psi


def projected_wall_incoming(
    phi: np.ndarray,
    psi: np.ndarray,
    cfg: LinearSidewallConfig,
    quadrature: PolarQuadrature,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    side_temperature = sidewall_temperature_profile(cfg)
    left_phi, left_psi = _profile_diffuse_incoming(
        phi[:, 0], psi[:, 0], quadrature.vx, quadrature.vx > 0.0,
        side_temperature, quadrature,
    )
    right_phi, right_psi = _profile_diffuse_incoming(
        phi[:, -1], psi[:, -1], -quadrature.vx, quadrature.vx < 0.0,
        side_temperature, quadrature,
    )
    bottom_temperature = np.full(cfg.nx, cfg.hot_temperature)
    bottom_phi, bottom_psi = _profile_diffuse_incoming(
        phi[0], psi[0], quadrature.vy, quadrature.vy > 0.0,
        bottom_temperature, quadrature,
    )
    top_temperature = np.full(cfg.nx, cfg.cold_temperature)
    top_phi, top_psi = _profile_diffuse_incoming(
        phi[-1], psi[-1], -quadrature.vy, quadrature.vy < 0.0,
        top_temperature, quadrature,
    )
    return (
        left_phi, left_psi, right_phi, right_psi,
        bottom_phi, bottom_psi, top_phi, top_psi,
    )


def _upwind_neighbors(
    distribution: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    bottom: np.ndarray,
    top: np.ndarray,
    quadrature: PolarQuadrature,
) -> tuple[np.ndarray, np.ndarray]:
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
    x_upwind = np.where(
        (quadrature.vx > 0.0)[None, None, :],
        left_neighbor,
        right_neighbor,
    )
    y_upwind = np.where(
        (quadrature.vy > 0.0)[None, None, :],
        bottom_neighbor,
        top_neighbor,
    )
    return x_upwind, y_upwind


def steady_source_iteration_step(
    phi: np.ndarray,
    psi: np.ndarray,
    cfg: LinearSidewallConfig,
    quadrature: PolarQuadrature,
    source_relaxation: float = STAGE42_SOURCE_RELAXATION,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if not 0.0 < source_relaxation <= 1.0:
        raise ValueError("source_relaxation must lie in (0,1]")
    (
        left_phi, left_psi, right_phi, right_psi,
        bottom_phi, bottom_psi, top_phi, top_psi,
    ) = projected_wall_incoming(phi, psi, cfg, quadrature)
    fields = projected_macroscopic(phi, psi, quadrature)
    equilibrium_phi, equilibrium_psi, clipping = projected_shakhov_equilibrium(
        fields,
        quadrature,
        prandtl=cfg.prandtl,
        correction_floor=STAGE41_CORRECTION_FLOOR,
    )
    tau = local_relaxation_time(
        fields["rho"], fields["T"], cfg, mapping="paper_consistent_c0"
    )
    collision_frequency = 1.0 / np.maximum(tau, 1.0e-14)
    dx = 1.0 / cfg.nx
    dy = 1.0 / cfg.ny
    ax = np.abs(quadrature.vx) / dx
    ay = np.abs(quadrature.vy) / dy
    phi_x, phi_y = _upwind_neighbors(
        phi, left_phi, right_phi, bottom_phi, top_phi, quadrature
    )
    psi_x, psi_y = _upwind_neighbors(
        psi, left_psi, right_psi, bottom_psi, top_psi, quadrature
    )
    denominator = (
        collision_frequency[..., None]
        + ax[None, None, :]
        + ay[None, None, :]
    )
    candidate_phi = (
        collision_frequency[..., None] * equilibrium_phi
        + ax[None, None, :] * phi_x
        + ay[None, None, :] * phi_y
    ) / denominator
    candidate_psi = (
        collision_frequency[..., None] * equilibrium_psi
        + ax[None, None, :] * psi_x
        + ay[None, None, :] * psi_y
    ) / denominator
    next_phi = phi + source_relaxation * (candidate_phi - phi)
    next_psi = psi + source_relaxation * (candidate_psi - psi)
    next_phi = np.maximum(next_phi, cfg.positivity_floor)
    next_psi = np.maximum(next_psi, cfg.positivity_floor)
    return next_phi, next_psi, clipping


def left_wall_tangential_velocity(
    phi: np.ndarray,
    left_incoming_phi: np.ndarray,
    quadrature: PolarQuadrature,
) -> np.ndarray:
    boundary_phi = np.where(
        (quadrature.vx > 0.0)[None, :], left_incoming_phi, phi[:, 0]
    )
    density = np.sum(boundary_phi * quadrature.weight[None, :], axis=-1)
    velocity = np.sum(
        boundary_phi * quadrature.vy[None, :] * quadrature.weight[None, :],
        axis=-1,
    ) / np.maximum(density, 1.0e-14)
    return velocity / math.sqrt(2.0)


def bottom_wall_heat_flux(
    phi: np.ndarray,
    psi: np.ndarray,
    bottom_incoming_phi: np.ndarray,
    bottom_incoming_psi: np.ndarray,
    quadrature: PolarQuadrature,
) -> np.ndarray:
    incoming = (quadrature.vy > 0.0)[None, :]
    boundary_phi = np.where(incoming, bottom_incoming_phi, phi[0])
    boundary_psi = np.where(incoming, bottom_incoming_psi, psi[0])
    parallel_speed2 = quadrature.vx**2 + quadrature.vy**2
    flux = 0.5 * np.sum(
        quadrature.vy[None, :]
        * (parallel_speed2[None, :] * boundary_phi + boundary_psi)
        * quadrature.weight[None, :],
        axis=-1,
    )
    return flux / math.sqrt(2.0)


def _velocity_metrics(predicted: np.ndarray, literature: np.ndarray) -> dict[str, float]:
    predicted = np.asarray(predicted, dtype=np.float64)
    literature = np.asarray(literature, dtype=np.float64)
    return {
        "relative_rms": float(
            np.linalg.norm(predicted - literature)
            / max(float(np.linalg.norm(literature)), 1.0e-14)
        ),
        "relative_l1": float(
            np.sum(np.abs(predicted - literature))
            / max(float(np.sum(np.abs(literature))), 1.0e-14)
        ),
        "sign_agreement": float(np.mean(np.sign(predicted) == np.sign(literature))),
    }


def _wall_balance(
    phi: np.ndarray,
    incoming: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                    np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    quadrature: PolarQuadrature,
) -> float:
    left_phi, _, right_phi, _, bottom_phi, _, top_phi, _ = incoming
    return max(
        wall_mass_balance_error(
            phi[:, 0], left_phi, quadrature.vx, quadrature.vx > 0.0, quadrature
        ),
        wall_mass_balance_error(
            phi[:, -1], right_phi, -quadrature.vx, quadrature.vx < 0.0, quadrature
        ),
        wall_mass_balance_error(
            phi[0], bottom_phi, quadrature.vy, quadrature.vy > 0.0, quadrature
        ),
        wall_mass_balance_error(
            phi[-1], top_phi, -quadrature.vy, quadrature.vy < 0.0, quadrature
        ),
    )


def solve_stage42_pilot(
    cfg: LinearSidewallConfig,
    quadrature: PolarQuadrature,
    source_relaxation: float = STAGE42_SOURCE_RELAXATION,
) -> dict[str, object]:
    rho = np.ones((cfg.ny, cfg.nx))
    zero = np.zeros_like(rho)
    initial_temperature = np.repeat(
        sidewall_temperature_profile(cfg)[:, None], cfg.nx, axis=1
    )
    phi, psi = projected_maxwellian(
        rho, zero, zero, initial_temperature, quadrature
    )
    previous = projected_macroscopic(phi, psi, quadrature)
    previous_temperature = np.asarray(previous["T"]).copy()
    previous_velocity = np.stack([previous["u"], previous["v"]], axis=-1)
    previous_heat_flux = np.stack([previous["qx"], previous["qy"]], axis=-1)
    residual_history: list[float] = []
    converged = False
    maximum_phi_clipped_fraction = 0.0
    maximum_psi_clipped_fraction = 0.0

    for iteration in range(cfg.max_steps):
        phi, psi, clipping = steady_source_iteration_step(
            phi, psi, cfg, quadrature, source_relaxation
        )
        maximum_phi_clipped_fraction = max(
            maximum_phi_clipped_fraction,
            float(np.max(clipping["phi_clipped_weight_fraction"])),
        )
        maximum_psi_clipped_fraction = max(
            maximum_psi_clipped_fraction,
            float(np.max(clipping["psi_clipped_weight_fraction"])),
        )
        if (iteration + 1) % cfg.check_interval == 0:
            fields = projected_macroscopic(phi, psi, quadrature)
            velocity = np.stack([fields["u"], fields["v"]], axis=-1)
            heat_flux = np.stack([fields["qx"], fields["qy"]], axis=-1)
            change = max(
                float(np.max(np.abs(fields["T"] - previous_temperature))),
                float(np.max(np.abs(velocity - previous_velocity))),
                float(np.max(np.abs(heat_flux - previous_heat_flux))),
            )
            residual_history.append(change)
            previous_temperature = np.asarray(fields["T"]).copy()
            previous_velocity = velocity.copy()
            previous_heat_flux = heat_flux.copy()
            if iteration + 1 >= cfg.minimum_steps and change < cfg.tolerance:
                converged = True
                break

    incoming = projected_wall_incoming(phi, psi, cfg, quadrature)
    left_phi, _, _, _, bottom_phi, bottom_psi, _, _ = incoming
    fields = projected_macroscopic(phi, psi, quadrature)
    wall_velocity = left_wall_tangential_velocity(phi, left_phi, quadrature)
    y_centers = (np.arange(cfg.ny, dtype=np.float64) + 0.5) / cfg.ny
    table_velocity = np.interp(TABLE3_Y, y_centers, wall_velocity)
    bottom_heat_flux = bottom_wall_heat_flux(
        phi, psi, bottom_phi, bottom_psi, quadrature
    )
    predicted_qav = float(np.mean(bottom_heat_flux))
    literature_qav = float(TABLE6_QAV_RATIO_0P1[cfg.kn0])
    velocity_metrics = _velocity_metrics(
        table_velocity, TABLE3_UY_RATIO_0P1[cfg.kn0]
    )
    wall_balance = _wall_balance(phi, incoming, quadrature)
    finite = bool(
        np.isfinite(phi).all()
        and np.isfinite(psi).all()
        and all(np.isfinite(np.asarray(fields[key])).all() for key in fields)
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
        "iterations": iteration + 1,
        "converged": converged,
        "final_change": (
            float(residual_history[-1]) if residual_history else math.inf
        ),
        "predicted_qav": predicted_qav,
        "literature_qav": literature_qav,
        "qav_relative_error": abs(predicted_qav - literature_qav)
        / max(abs(literature_qav), 1.0e-14),
        "velocity_metrics": velocity_metrics,
        "wall_mass_balance_relative_error": wall_balance,
        "minimum_phi": float(np.min(phi)),
        "minimum_psi": float(np.min(psi)),
        "maximum_phi_clipped_weight_fraction": maximum_phi_clipped_fraction,
        "maximum_psi_clipped_weight_fraction": maximum_psi_clipped_fraction,
        "finite": finite,
        "work_proxy": int((iteration + 1) * cfg.nx * cfg.ny * quadrature.point_count),
    }


def stage42_decision(result: dict[str, object]) -> str:
    stable = (
        bool(result["finite"])
        and float(result["minimum_phi"]) > 0.0
        and float(result["minimum_psi"]) > 0.0
        and float(result["wall_mass_balance_relative_error"]) < 1.0e-10
    )
    if not stable:
        return "projected_polar_heated_cavity_blocker"
    if bool(result["converged"]):
        return "projected_polar_pilot_converged_stage43_resolution_sequence"
    return "projected_polar_pilot_stable_nonconverged_stage43_iteration_acceleration"


def run_stage42(
    output_dir: str | Path,
    *,
    grid: tuple[int, int] = STAGE42_GRID,
    rule: tuple[int, int] = STAGE42_RULE,
    kn0: float = STAGE42_KNUDSEN,
    cold_hot_ratio: float = STAGE42_RATIO,
    max_iterations: int = STAGE42_MAX_ITERATIONS,
    tolerance: float = STAGE42_TOLERANCE,
    source_relaxation: float = STAGE42_SOURCE_RELAXATION,
) -> dict[str, object]:
    validate_stage42_design(
        grid, rule, kn0, cold_hot_ratio, max_iterations, tolerance,
        source_relaxation,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = LinearSidewallConfig(
        nx=grid[0],
        ny=grid[1],
        kn0=kn0,
        cold_hot_ratio=cold_hot_ratio,
        viscosity_exponent=0.5,
        prandtl=STAGE41_PRANDTL,
        max_steps=max_iterations,
        cfl=0.2,
        tolerance=tolerance,
        check_interval=STAGE42_CHECK_INTERVAL,
        minimum_steps=STAGE42_MINIMUM_ITERATIONS,
        positivity_floor=1.0e-30,
    )
    quadrature = mapped_polar_quadrature(*rule)
    result = solve_stage42_pilot(cfg, quadrature, source_relaxation)
    decision = stage42_decision(result)
    summary = {
        "stage": 42,
        "description": (
            "First heated-cavity pilot using the Stage 41 projected phi/psi "
            "mapped-polar operators and a positivity-preserving steady first-order "
            "upwind source iteration"
        ),
        "retained_stage41_endpoint": STAGE41_COMPLETED_ENDPOINT,
        "configuration": {
            "kn0": kn0,
            "cold_hot_ratio": cold_hot_ratio,
            "grid": list(grid),
            "polar_rule": {
                "radial_nodes": rule[0],
                "angular_nodes": rule[1],
                "point_count": quadrature.point_count,
            },
            "radial_mapping": "r=s*(1+x)/(1-x)",
            "radial_scale": quadrature.radial_scale,
            "prandtl": cfg.prandtl,
            "shakhov_correction_floor": STAGE41_CORRECTION_FLOOR,
            "transport_iteration": STAGE42_TRANSPORT,
            "collision": STAGE42_COLLISION,
            "relaxation_mapping": STAGE42_RELAXATION_MAPPING,
            "source_relaxation": source_relaxation,
            "max_iterations": max_iterations,
            "minimum_iterations": cfg.minimum_steps,
            "check_interval": cfg.check_interval,
            "tolerance": tolerance,
            "physical_parameter_retuning": False,
            "source_resolution_reproduction": False,
        },
        "result": {
            key: value
            for key, value in result.items()
            if key not in {
                "T", "rho", "u", "v", "qx", "qy", "left_wall_velocity",
                "table_velocity", "bottom_heat_flux", "residual_history",
            }
        },
        "decision": decision,
        "interpretation_guard": (
            "Stage 42 is a fixed low-resolution stability and convergence pilot. "
            "It does not reproduce the source 400x400/32000-vector calculation, "
            "does not validate the Table 3 wall velocity, and does not reactivate "
            "failed learned allocation or MUSCL policies."
        ),
        "scientific_conclusion": (
            "The projected-polar architecture is now exercised in the heated cavity "
            "without physical retuning. Positive and negative pilot outcomes are "
            "retained; closeness to the source heat flux is reported but is not a "
            "validation criterion at this pilot resolution."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        output_dir / "fields_and_profiles.npz",
        T=result["T"],
        rho=result["rho"],
        u=result["u"],
        v=result["v"],
        qx=result["qx"],
        qy=result["qy"],
        left_wall_velocity=result["left_wall_velocity"],
        table_velocity=result["table_velocity"],
        bottom_heat_flux=result["bottom_heat_flux"],
        residual_history=result["residual_history"],
        quadrature_vx=quadrature.vx,
        quadrature_vy=quadrature.vy,
        quadrature_weight=quadrature.weight,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="results/stage42_projected_polar_heated_cavity_pilot",
    )
    args = parser.parse_args()
    print(json.dumps(run_stage42(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
