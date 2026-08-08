from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .linear_sidewall_validation import TABLE3_UY_RATIO_0P1, TABLE3_Y, TABLE6_QAV_RATIO_0P1
from .stage34_velocity_scale_consistency import local_relaxation_time
from .stage41_projected_polar_operator_audit import (
    STAGE41_CORRECTION_FLOOR,
    STAGE41_PRANDTL,
    mapped_polar_quadrature,
    projected_macroscopic,
    projected_shakhov_equilibrium,
)
from .stage42_projected_polar_heated_cavity_pilot import (
    _upwind_neighbors,
    _velocity_metrics,
    _wall_balance,
    bottom_wall_heat_flux,
    left_wall_tangential_velocity,
    projected_wall_incoming,
)
from .stage58_conservative_solver_64x64_confirmation import build_stage58_config

STAGE67_COMPLETED_ENDPOINT = {
    "workflow_run_id": 30991124477,
    "workflow_job_id": 92257254811,
    "artifact_id": 8931272132,
    "artifact_sha256": "5b871e94bdca02bce33846657f8f4dc263d68af9e671d7eb137ca7c83da198d4",
    "summary_sha256": "e04043a1913b2fa9ae57fe1561aa26c70627830d648e91204093c8f1fb57b3d1",
    "distributions_sha256": "d4002a2765137ba517abec2d0483a3e5adcf13f415c53259018556bd14d612d1",
    "residual_maps_sha256": "08722bd5b2036eee1b42b09d37583701ffcc3ef5e4f7d7c68642ea5103f11ced",
    "decision": "stage67_frozen_replay_and_residual_balance_close_stage68_independent_transport_operator_residual_audit",
}
STAGE89_COMPLETED_ENDPOINT = {
    "workflow_run_id": 31272829723,
    "workflow_job_id": 93141582111,
    "artifact_id": 9027714017,
    "artifact_sha256": "6892c6e3c9e06dff3064f674175200c94f603b6e75bdc73658d7e9e2362cf8f2",
    "summary_sha256": "1e04f079a052f506d7c435298213afb3a14ecf21d6436d620e94d84969988358",
    "profiles_sha256": "7dd082781c41fd4e76932e3847c0065a44de1588881f78b6d2a39d54a0fc0fc6",
    "decision": "stage89_local_admissibility_closes_stage90_single_condition_reconstruction_solver_ab_audit",
}

GRID = (64, 64)
KNUDSEN = 10.0
COLD_HOT_RATIO = 0.1
RULE = (40, 96)
RADIAL_SCALE = 2.0
POINT_COUNT = RULE[0] * RULE[1]
LIMITER = "minmod"
BASELINE_BOUNDARY_SLOPE = "zero"
COUNTERFACTUAL_BOUNDARY_SLOPE = "one_sided_first_difference_x_only"
SOURCE_RELAXATION = 1.0
MAX_ITERATIONS = 3000
MINIMUM_ITERATIONS = 500
CHECK_INTERVAL = 25
TOLERANCE = 2.0e-5
MATERIAL_BENCHMARK_IMPROVEMENT = 0.10
NO_DEGRADATION_TOLERANCE = 1.0e-12


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_stage90_design(**overrides: object) -> None:
    frozen = {
        "grid": GRID,
        "kn0": KNUDSEN,
        "cold_hot_ratio": COLD_HOT_RATIO,
        "rule": RULE,
        "radial_scale": RADIAL_SCALE,
        "limiter": LIMITER,
        "baseline_boundary_slope": BASELINE_BOUNDARY_SLOPE,
        "counterfactual_boundary_slope": COUNTERFACTUAL_BOUNDARY_SLOPE,
        "source_relaxation": SOURCE_RELAXATION,
        "max_iterations": MAX_ITERATIONS,
        "minimum_iterations": MINIMUM_ITERATIONS,
        "check_interval": CHECK_INTERVAL,
        "tolerance": TOLERANCE,
        "correction_floor": STAGE41_CORRECTION_FLOOR,
        "material_benchmark_improvement": MATERIAL_BENCHMARK_IMPROVEMENT,
    }
    if any(key not in frozen or frozen[key] != value for key, value in overrides.items()):
        raise ValueError(
            "Stage 90 is a frozen single-condition A/B solver audit at Kn0=10. "
            "Only the x-wall boundary-cell MUSCL slope differs between arms; "
            "physics, collision/source treatment, clipping floor, relaxation, grid, "
            "40x96 radial-scale-2.0 quadrature, wall model, limiter, initialization, "
            "tolerance and stopping rules cannot be retuned."
        )


def _validate_stage67(root: str | Path) -> dict[str, object]:
    root = Path(root)
    expected = {
        "summary.json": STAGE67_COMPLETED_ENDPOINT["summary_sha256"],
        "converged_full_distributions.npz": STAGE67_COMPLETED_ENDPOINT["distributions_sha256"],
        "steady_residual_moment_maps.npz": STAGE67_COMPLETED_ENDPOINT["residual_maps_sha256"],
    }
    for name, checksum in expected.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != checksum:
            raise ValueError(f"Stage-67 checksum mismatch: {name}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("stage") != 67 or summary.get("decision") != STAGE67_COMPLETED_ENDPOINT["decision"]:
        raise ValueError("Stage-67 completed endpoint mismatch")
    return summary


def _validate_stage89(root: str | Path) -> dict[str, object]:
    root = Path(root)
    expected = {
        "summary.json": STAGE89_COMPLETED_ENDPOINT["summary_sha256"],
        "boundary_reconstruction_admissibility_profiles.npz": STAGE89_COMPLETED_ENDPOINT["profiles_sha256"],
    }
    for name, checksum in expected.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != checksum:
            raise ValueError(f"Stage-89 checksum mismatch: {name}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("stage") != 89 or summary.get("decision") != STAGE89_COMPLETED_ENDPOINT["decision"]:
        raise ValueError("Stage-89 completed endpoint mismatch")
    return summary


def minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    same = a * b > 0.0
    return np.where(same, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


def limited_slopes(field: np.ndarray, axis: int, one_sided_boundary: bool = False) -> np.ndarray:
    field = np.asarray(field, dtype=np.float64)
    if field.ndim != 3 or axis not in (0, 1):
        raise ValueError("Stage 90 slopes require (ny,nx,nq) data and axis 0 or 1")
    if field.shape[axis] < 3:
        raise ValueError("Stage 90 requires at least three cells per reconstructed axis")
    slopes = np.zeros_like(field)
    if axis == 1:
        slopes[:, 1:-1] = minmod(field[:, 1:-1] - field[:, :-2], field[:, 2:] - field[:, 1:-1])
        if one_sided_boundary:
            slopes[:, 0] = field[:, 1] - field[:, 0]
            slopes[:, -1] = field[:, -1] - field[:, -2]
    else:
        slopes[1:-1] = minmod(field[1:-1] - field[:-2], field[2:] - field[1:-1])
        if one_sided_boundary:
            slopes[0] = field[1] - field[0]
            slopes[-1] = field[-1] - field[-2]
    return slopes


def muscl_correction_divergence(
    distribution: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    dx: float,
    dy: float,
    one_sided_x_boundary: bool,
) -> np.ndarray:
    """Second-minus-first-order conservative transport divergence.

    Wall fluxes are unchanged. The only A/B difference is whether the two x-boundary
    cell slopes are zero or one-sided first differences. Y-boundary slopes remain zero
    in both arms.
    """
    f = np.asarray(distribution, dtype=np.float64)
    vx = np.asarray(vx, dtype=np.float64)
    vy = np.asarray(vy, dtype=np.float64)
    if f.ndim != 3 or f.shape[-1] != vx.size or vx.shape != vy.shape:
        raise ValueError("distribution and velocity rule shapes are inconsistent")
    sx = limited_slopes(f, 1, one_sided_boundary=one_sided_x_boundary)
    sy = limited_slopes(f, 0, one_sided_boundary=False)
    sign_x = np.where(vx > 0.0, 1.0, np.where(vx < 0.0, -1.0, 0.0))
    sign_y = np.where(vy > 0.0, 1.0, np.where(vy < 0.0, -1.0, 0.0))
    face_x = 0.5 * vx[None, None, :] * np.where(
        sign_x[None, None, :] > 0.0,
        sx[:, :-1],
        np.where(sign_x[None, None, :] < 0.0, -sx[:, 1:], 0.0),
    )
    face_y = 0.5 * vy[None, None, :] * np.where(
        sign_y[None, None, :] > 0.0,
        sy[:-1],
        np.where(sign_y[None, None, :] < 0.0, -sy[1:], 0.0),
    )
    correction = np.zeros_like(f)
    correction[:, :-1] += face_x / dx
    correction[:, 1:] -= face_x / dx
    correction[:-1] += face_y / dy
    correction[1:] -= face_y / dy
    return correction


def steady_muscl_iteration_step(
    phi: np.ndarray,
    psi: np.ndarray,
    cfg,
    quadrature,
    one_sided_x_boundary: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    incoming = projected_wall_incoming(phi, psi, cfg, quadrature)
    left_phi, left_psi, right_phi, right_psi, bottom_phi, bottom_psi, top_phi, top_psi = incoming
    fields = projected_macroscopic(phi, psi, quadrature)
    equilibrium_phi, equilibrium_psi, clipping = projected_shakhov_equilibrium(
        fields, quadrature, prandtl=cfg.prandtl, correction_floor=STAGE41_CORRECTION_FLOOR
    )
    tau = local_relaxation_time(fields["rho"], fields["T"], cfg, mapping="paper_consistent_c0")
    nu = 1.0 / np.maximum(tau, 1.0e-14)
    dx, dy = 1.0 / cfg.nx, 1.0 / cfg.ny
    ax = np.abs(quadrature.vx) / dx
    ay = np.abs(quadrature.vy) / dy
    phi_x, phi_y = _upwind_neighbors(phi, left_phi, right_phi, bottom_phi, top_phi, quadrature)
    psi_x, psi_y = _upwind_neighbors(psi, left_psi, right_psi, bottom_psi, top_psi, quadrature)
    denominator = nu[..., None] + ax[None, None, :] + ay[None, None, :]

    phi_correction = muscl_correction_divergence(
        phi, quadrature.vx, quadrature.vy, dx, dy, one_sided_x_boundary
    )
    candidate_phi = (
        nu[..., None] * equilibrium_phi
        + ax[None, None, :] * phi_x
        + ay[None, None, :] * phi_y
        - phi_correction
    ) / denominator
    del phi_correction, phi_x, phi_y, equilibrium_phi

    psi_correction = muscl_correction_divergence(
        psi, quadrature.vx, quadrature.vy, dx, dy, one_sided_x_boundary
    )
    candidate_psi = (
        nu[..., None] * equilibrium_psi
        + ax[None, None, :] * psi_x
        + ay[None, None, :] * psi_y
        - psi_correction
    ) / denominator
    del psi_correction, psi_x, psi_y, equilibrium_psi

    phi_below = candidate_phi < cfg.positivity_floor
    psi_below = candidate_psi < cfg.positivity_floor
    next_phi = np.maximum(candidate_phi, cfg.positivity_floor)
    next_psi = np.maximum(candidate_psi, cfg.positivity_floor)
    diagnostics = {
        "phi_update_floor_fraction": float(np.mean(phi_below)),
        "psi_update_floor_fraction": float(np.mean(psi_below)),
        "minimum_candidate_phi": float(np.min(candidate_phi)),
        "minimum_candidate_psi": float(np.min(candidate_psi)),
        "maximum_phi_clipped_weight_fraction": float(np.max(clipping["phi_clipped_weight_fraction"])),
        "maximum_psi_clipped_weight_fraction": float(np.max(clipping["psi_clipped_weight_fraction"])),
    }
    return next_phi, next_psi, diagnostics


def _endpoint_metrics(phi: np.ndarray, psi: np.ndarray, cfg, quadrature, residual_history: list[float], iterations: int, converged: bool, diag: dict[str, float]) -> dict[str, object]:
    incoming = projected_wall_incoming(phi, psi, cfg, quadrature)
    left_phi, _, _, _, bottom_phi, bottom_psi, _, _ = incoming
    fields = projected_macroscopic(phi, psi, quadrature)
    wall_velocity = left_wall_tangential_velocity(phi, left_phi, quadrature)
    y_centers = (np.arange(cfg.ny, dtype=np.float64) + 0.5) / cfg.ny
    table_velocity = np.interp(TABLE3_Y, y_centers, wall_velocity)
    bottom_q = bottom_wall_heat_flux(phi, psi, bottom_phi, bottom_psi, quadrature)
    qav = float(np.mean(bottom_q))
    literature_qav = float(TABLE6_QAV_RATIO_0P1[cfg.kn0])
    finite = bool(np.isfinite(phi).all() and np.isfinite(psi).all() and all(np.isfinite(np.asarray(fields[k])).all() for k in fields))
    return {
        "iterations": int(iterations),
        "converged": bool(converged),
        "finite": finite,
        "final_change": float(residual_history[-1]) if residual_history else math.inf,
        "predicted_qav": qav,
        "literature_qav": literature_qav,
        "qav_relative_error": abs(qav - literature_qav) / max(abs(literature_qav), 1.0e-14),
        "velocity_metrics": _velocity_metrics(table_velocity, TABLE3_UY_RATIO_0P1[cfg.kn0]),
        "wall_mass_balance_relative_error": _wall_balance(phi, incoming, quadrature),
        "minimum_phi": float(np.min(phi)),
        "minimum_psi": float(np.min(psi)),
        "maximum_update_phi_floor_fraction": float(diag["maximum_update_phi_floor_fraction"]),
        "maximum_update_psi_floor_fraction": float(diag["maximum_update_psi_floor_fraction"]),
        "minimum_candidate_phi": float(diag["minimum_candidate_phi"]),
        "minimum_candidate_psi": float(diag["minimum_candidate_psi"]),
        "maximum_phi_clipped_weight_fraction": float(diag["maximum_phi_clipped_weight_fraction"]),
        "maximum_psi_clipped_weight_fraction": float(diag["maximum_psi_clipped_weight_fraction"]),
        "table_velocity": np.asarray(table_velocity),
        "bottom_heat_flux": np.asarray(bottom_q),
        "residual_history": np.asarray(residual_history),
        "T": np.asarray(fields["T"]),
        "rho": np.asarray(fields["rho"]) / np.mean(fields["rho"]),
        "u": np.asarray(fields["u"]) / math.sqrt(2.0),
        "v": np.asarray(fields["v"]) / math.sqrt(2.0),
        "qx": np.asarray(fields["qx"]) / math.sqrt(2.0),
        "qy": np.asarray(fields["qy"]) / math.sqrt(2.0),
    }


def solve_arm(initial_phi: np.ndarray, initial_psi: np.ndarray, cfg, quadrature, one_sided_x_boundary: bool) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    phi = np.asarray(initial_phi, dtype=np.float64).copy()
    psi = np.asarray(initial_psi, dtype=np.float64).copy()
    previous = projected_macroscopic(phi, psi, quadrature)
    previous_T = np.asarray(previous["T"]).copy()
    previous_u = np.stack([previous["u"], previous["v"]], axis=-1)
    previous_q = np.stack([previous["qx"], previous["qy"]], axis=-1)
    residual_history: list[float] = []
    converged = False
    diag = {
        "maximum_update_phi_floor_fraction": 0.0,
        "maximum_update_psi_floor_fraction": 0.0,
        "minimum_candidate_phi": math.inf,
        "minimum_candidate_psi": math.inf,
        "maximum_phi_clipped_weight_fraction": 0.0,
        "maximum_psi_clipped_weight_fraction": 0.0,
    }
    iteration = 0
    for iteration in range(1, cfg.max_steps + 1):
        phi, psi, step_diag = steady_muscl_iteration_step(
            phi, psi, cfg, quadrature, one_sided_x_boundary
        )
        diag["maximum_update_phi_floor_fraction"] = max(diag["maximum_update_phi_floor_fraction"], step_diag["phi_update_floor_fraction"])
        diag["maximum_update_psi_floor_fraction"] = max(diag["maximum_update_psi_floor_fraction"], step_diag["psi_update_floor_fraction"])
        diag["minimum_candidate_phi"] = min(diag["minimum_candidate_phi"], step_diag["minimum_candidate_phi"])
        diag["minimum_candidate_psi"] = min(diag["minimum_candidate_psi"], step_diag["minimum_candidate_psi"])
        diag["maximum_phi_clipped_weight_fraction"] = max(diag["maximum_phi_clipped_weight_fraction"], step_diag["maximum_phi_clipped_weight_fraction"])
        diag["maximum_psi_clipped_weight_fraction"] = max(diag["maximum_psi_clipped_weight_fraction"], step_diag["maximum_psi_clipped_weight_fraction"])
        if not np.isfinite(phi).all() or not np.isfinite(psi).all():
            break
        if iteration % cfg.check_interval == 0:
            fields = projected_macroscopic(phi, psi, quadrature)
            velocity = np.stack([fields["u"], fields["v"]], axis=-1)
            heat_flux = np.stack([fields["qx"], fields["qy"]], axis=-1)
            change = max(
                float(np.max(np.abs(fields["T"] - previous_T))),
                float(np.max(np.abs(velocity - previous_u))),
                float(np.max(np.abs(heat_flux - previous_q))),
            )
            residual_history.append(change)
            previous_T = np.asarray(fields["T"]).copy()
            previous_u = velocity.copy()
            previous_q = heat_flux.copy()
            if iteration >= cfg.minimum_steps and change < cfg.tolerance:
                converged = True
                break
    return _endpoint_metrics(phi, psi, cfg, quadrature, residual_history, iteration, converged, diag), phi, psi


def compact_endpoint(result: dict[str, object]) -> dict[str, object]:
    keys = (
        "iterations", "converged", "finite", "final_change", "predicted_qav", "literature_qav",
        "qav_relative_error", "velocity_metrics", "wall_mass_balance_relative_error",
        "minimum_phi", "minimum_psi", "maximum_update_phi_floor_fraction",
        "maximum_update_psi_floor_fraction", "minimum_candidate_phi", "minimum_candidate_psi",
        "maximum_phi_clipped_weight_fraction", "maximum_psi_clipped_weight_fraction",
    )
    return {key: result[key] for key in keys}


def compare_endpoints(baseline: dict[str, object], one_sided: dict[str, object]) -> dict[str, float | bool]:
    q0 = float(baseline["qav_relative_error"])
    q1 = float(one_sided["qav_relative_error"])
    u0 = float(baseline["velocity_metrics"]["relative_rms"])
    u1 = float(one_sided["velocity_metrics"]["relative_rms"])
    q_improve = (q0 - q1) / max(q0, 1.0e-300)
    u_improve = (u0 - u1) / max(u0, 1.0e-300)
    return {
        "qav_error_improvement_fraction": q_improve,
        "table3_velocity_rms_error_improvement_fraction": u_improve,
        "qav_material_improvement": q_improve >= MATERIAL_BENCHMARK_IMPROVEMENT,
        "table3_material_improvement": u_improve >= MATERIAL_BENCHMARK_IMPROVEMENT,
        "qav_not_degraded": q1 <= q0 * (1.0 + NO_DEGRADATION_TOLERANCE),
        "table3_not_degraded": u1 <= u0 * (1.0 + NO_DEGRADATION_TOLERANCE),
        "predicted_qav_relative_change": abs(float(one_sided["predicted_qav"]) - float(baseline["predicted_qav"])) / max(abs(float(baseline["predicted_qav"])), 1.0e-300),
    }


def stage90_decision(baseline: dict[str, object], one_sided: dict[str, object], comparison: dict[str, object]) -> str:
    if not bool(baseline["finite"]) or not bool(one_sided["finite"]):
        return "stage90_nonfinite_solver_blocker_without_retuning"
    if not bool(baseline["converged"]) or not bool(one_sided["converged"]):
        return "stage90_nonconverged_solver_blocker_without_retuning"
    both_material = bool(comparison["qav_material_improvement"]) and bool(comparison["table3_material_improvement"])
    one_material_no_degrade = (
        (bool(comparison["qav_material_improvement"]) or bool(comparison["table3_material_improvement"]))
        and bool(comparison["qav_not_degraded"])
        and bool(comparison["table3_not_degraded"])
    )
    if both_material:
        return "stage90_material_table3_table6_improvement_stage91_independent_confirmation_before_any_extension"
    if one_material_no_degrade:
        return "stage90_partial_material_benchmark_improvement_stage91_independent_confirmation_before_any_extension"
    return "stage90_no_material_benchmark_improvement_close_boundary_reconstruction_route_stage91_fixed_quadrature_model_form_reconciliation_audit"


def _save_arm(path: Path, result: dict[str, object], phi: np.ndarray, psi: np.ndarray) -> None:
    np.savez_compressed(
        path,
        phi=phi,
        psi=psi,
        T=result["T"], rho=result["rho"], u=result["u"], v=result["v"], qx=result["qx"], qy=result["qy"],
        table_velocity=result["table_velocity"], bottom_heat_flux=result["bottom_heat_flux"], residual_history=result["residual_history"],
    )


def run_stage90(stage67_artifact_dir: str | Path, stage89_artifact_dir: str | Path, output_dir: str | Path, **design) -> dict[str, object]:
    validate_stage90_design(**design)
    stage67_summary = _validate_stage67(stage67_artifact_dir)
    stage89_summary = _validate_stage89(stage89_artifact_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = build_stage58_config()
    quadrature = mapped_polar_quadrature(*RULE, radial_scale=RADIAL_SCALE)
    with np.load(Path(stage67_artifact_dir) / "converged_full_distributions.npz") as saved:
        initial_phi = np.asarray(saved["phi"], dtype=np.float64)
        initial_psi = np.asarray(saved["psi"], dtype=np.float64)
        for name, actual, expected in (
            ("vx", np.asarray(saved["vx"]), np.asarray(quadrature.vx)),
            ("vy", np.asarray(saved["vy"]), np.asarray(quadrature.vy)),
            ("weight", np.asarray(saved["weight"]), np.asarray(quadrature.weight)),
        ):
            if not np.array_equal(actual, expected):
                raise ValueError(f"Stage-67 {name} does not exactly match the frozen Stage-90 quadrature")

        baseline, phi0, psi0 = solve_arm(initial_phi, initial_psi, cfg, quadrature, False)
        _save_arm(out / "zero_boundary_slope_endpoint.npz", baseline, phi0, psi0)
        baseline_compact = compact_endpoint(baseline)
        del phi0, psi0, baseline
        gc.collect()

        one_sided, phi1, psi1 = solve_arm(initial_phi, initial_psi, cfg, quadrature, True)
        _save_arm(out / "one_sided_boundary_slope_endpoint.npz", one_sided, phi1, psi1)
        one_sided_compact = compact_endpoint(one_sided)
        del phi1, psi1, one_sided
        gc.collect()

    comparison = compare_endpoints(baseline_compact, one_sided_compact)
    decision = stage90_decision(baseline_compact, one_sided_compact, comparison)
    summary = {
        "stage": 90,
        "description": "Frozen single-condition nonlinear solver A/B audit of zero versus locally admissible one-sided x-wall MUSCL boundary-cell slopes.",
        "retained_stage67_decision": stage67_summary["decision"],
        "retained_stage89_decision": stage89_summary["decision"],
        "configuration": {
            "grid": [64, 64], "kn0": 10.0, "cold_hot_ratio": 0.1,
            "radial_nodes": 40, "angular_nodes": 96, "point_count": 3840, "radial_scale": 2.0,
            "prandtl": STAGE41_PRANDTL, "retained_correction_floor": STAGE41_CORRECTION_FLOOR,
            "limiter": LIMITER, "baseline_boundary_slope": BASELINE_BOUNDARY_SLOPE,
            "counterfactual_boundary_slope": COUNTERFACTUAL_BOUNDARY_SLOPE,
            "source_relaxation": SOURCE_RELAXATION, "max_iterations": MAX_ITERATIONS,
            "minimum_iterations": MINIMUM_ITERATIONS, "check_interval": CHECK_INTERVAL, "tolerance": TOLERANCE,
            "initialization": "exact completed Stage-67 converged phi/psi for both arms",
            "material_benchmark_improvement_guard": MATERIAL_BENCHMARK_IMPROVEMENT,
            "physical_parameter_retuning": False, "collision_parameter_retuning": False,
            "correction_floor_retuning": False, "source_relaxation_retuning": False,
            "transport_parameter_retuning": False, "wall_model_retuning": False,
            "normalization_retuning": False, "limiter_retuning": False,
            "velocity_quadrature_retuning": False, "failed_muscl_endpoint_rehabilitated": False,
            "cross_knudsen_extension_permitted": False, "validation_claim_permitted": False,
        },
        "zero_boundary_slope": baseline_compact,
        "one_sided_boundary_slope": one_sided_compact,
        "paired_comparison": comparison,
        "decision": decision,
        "scientific_conclusion": (
            "Stage 90 changes only the x-wall boundary-cell reconstruction slope and preserves both favorable and unfavorable endpoints. "
            "A converged material improvement is evidence only for this frozen Kn0=10 A/B comparison; it is not validation and does not authorize cross-Knudsen extension without an independent confirmation stage."
        ),
        "negative_result_guard": (
            "The failed Stage-28 MUSCL endpoint remains negative. No failed physical parameter, collision parameter, clipping floor, relaxation, limiter, wall model, normalization or velocity quadrature is retuned, and negative/nonconverged/nonfinite A/B outcomes are retained explicitly."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage67-artifact-dir", required=True)
    parser.add_argument("--stage89-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run_stage90(args.stage67_artifact_dir, args.stage89_artifact_dir, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
