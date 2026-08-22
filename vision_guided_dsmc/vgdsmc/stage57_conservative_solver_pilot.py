from __future__ import annotations

from pathlib import Path
import argparse
import gc
import hashlib
import json
import math
from typing import Mapping

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
    STAGE41_CORRECTION_FLOOR,
    STAGE41_PRANDTL,
    mapped_polar_quadrature,
    projected_macroscopic,
    projected_maxwellian,
    projected_shakhov_equilibrium,
)
from .stage42_projected_polar_heated_cavity_pilot import (
    _upwind_neighbors,
    _velocity_metrics,
    _wall_balance,
    bottom_wall_heat_flux,
    left_wall_tangential_velocity,
    projected_wall_incoming,
    solve_stage42_pilot,
)
from .stage56_conservative_projection_pilot import (
    STAGE56_BOUND_TOLERANCE,
    STAGE56_MAX_ACTIVE_FRACTION_GUARD,
    STAGE56_MAX_ACTIVE_SET_ITERATIONS,
    STAGE56_MAX_CONSERVED_DEFECT,
    STAGE56_MAX_HEAT_FLUX_CLOSURE,
    STAGE56_MAX_RELATIVE_MODIFICATION_GUARD,
    STAGE56_RULE,
    _linear_moments,
    _moment_basis,
    _relative_conserved_defect,
    _retained_clipped_lower_bounds,
    _target_moments,
    _weighted_relative_modification,
    bounded_conservative_projection,
)


STAGE56_COMPLETED_ENDPOINT = {
    "workflow_run_id": 30838743466,
    "workflow_job_id": 91770307142,
    "workflow_conclusion": "success",
    "tests_passed": 113,
    "tests_failed": 0,
    "test_duration_seconds": 0.58,
    "artifact_id": 8870328780,
    "artifact_size_bytes": 180573,
    "artifact_sha256": "eb16133a3288e652b15986fdf9ae6c1738499a030509c06ac9cd785a3ca45946",
    "source_head_sha": "cc1ca1138d8655dbe60e80c13f1e9a04c8420dd9",
    "summary_sha256": "b05b005ef0d28c7f3c4de7f84257db5946d443aea97366b51acd2a6f43a4e3e4",
    "compressed_tail_diagnostics_sha256": "bb2dd4193f580bbf77d77bf3863ec6cd383bca76a95d31509f2332dd3a9a5e86",
    "expanded_tail_diagnostics_sha256": "959074d015f923974d1165bcc238301357468e5f5ebe3f5021e4519d529a2527",
    "decision": (
        "conservative_positive_projection_closes_frozen_fields_"
        "stage57_single_case_solver_pilot"
    ),
}

# Stage 57 is an integration-feasibility experiment, not a resolution study.
# A 16x16 physical pilot is fixed before execution to exercise the conservative
# operator through many nonlinear source iterations at tractable cost. The
# 64x64 confirmation is reserved for Stage 58 only if this pilot passes.
STAGE57_GRID = (16, 16)
STAGE57_KNUDSEN = 10.0
STAGE57_RATIO = 0.1
STAGE57_RULE = STAGE56_RULE
STAGE57_RADIAL_SCALE = 2.0
STAGE57_SOURCE_RELAXATION = 1.0
STAGE57_MAX_ITERATIONS = 3000
STAGE57_MINIMUM_ITERATIONS = 500
STAGE57_CHECK_INTERVAL = 25
STAGE57_TOLERANCE = 2.0e-5
STAGE57_MAX_OBSERVABLE_ERROR_DEGRADATION = 0.10
STAGE57_MIN_SIGN_AGREEMENT_CHANGE = 0.0


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_stage57_design(
    grid=STAGE57_GRID,
    kn0=STAGE57_KNUDSEN,
    cold_hot_ratio=STAGE57_RATIO,
    rule=STAGE57_RULE,
    radial_scale=STAGE57_RADIAL_SCALE,
    source_relaxation=STAGE57_SOURCE_RELAXATION,
    max_iterations=STAGE57_MAX_ITERATIONS,
    minimum_iterations=STAGE57_MINIMUM_ITERATIONS,
    check_interval=STAGE57_CHECK_INTERVAL,
    tolerance=STAGE57_TOLERANCE,
    correction_floor=STAGE41_CORRECTION_FLOOR,
) -> None:
    actual = (
        grid,
        kn0,
        cold_hot_ratio,
        rule,
        radial_scale,
        source_relaxation,
        max_iterations,
        minimum_iterations,
        check_interval,
        tolerance,
        correction_floor,
    )
    expected = (
        STAGE57_GRID,
        STAGE57_KNUDSEN,
        STAGE57_RATIO,
        STAGE57_RULE,
        STAGE57_RADIAL_SCALE,
        STAGE57_SOURCE_RELAXATION,
        STAGE57_MAX_ITERATIONS,
        STAGE57_MINIMUM_ITERATIONS,
        STAGE57_CHECK_INTERVAL,
        STAGE57_TOLERANCE,
        STAGE41_CORRECTION_FLOOR,
    )
    if actual != expected:
        raise ValueError(
            "Stage 57 is frozen to one 16x16 Kn0=10 integration pilot, the "
            "Stage-55 radial 40x96 rule, the Stage-56 expanded-tail mapping, "
            "the retained 0.05 correction floor, and the existing source-iteration "
            "and convergence settings"
        )


def _validate_stage56_artifact(stage56_dir: Path) -> dict[str, object]:
    expected_files = {
        "summary.json": STAGE56_COMPLETED_ENDPOINT["summary_sha256"],
        "compressed_tail_conservative_projection_diagnostics.npz": (
            STAGE56_COMPLETED_ENDPOINT["compressed_tail_diagnostics_sha256"]
        ),
        "expanded_tail_conservative_projection_diagnostics.npz": (
            STAGE56_COMPLETED_ENDPOINT["expanded_tail_diagnostics_sha256"]
        ),
    }
    for filename, expected_sha in expected_files.items():
        path = stage56_dir / filename
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise ValueError(f"Stage 56 artifact checksum mismatch: {filename}")
    summary = json.loads((stage56_dir / "summary.json").read_text())
    if summary.get("stage") != 56:
        raise ValueError("Stage 56 artifact stage mismatch")
    if summary.get("decision") != STAGE56_COMPLETED_ENDPOINT["decision"]:
        raise ValueError("Stage 56 artifact decision mismatch")
    configuration = summary.get("configuration", {})
    if configuration.get("grid") != [64, 64]:
        raise ValueError("Stage 56 artifact grid mismatch")
    if configuration.get("kn0") != STAGE57_KNUDSEN:
        raise ValueError("Stage 56 artifact Knudsen mismatch")
    expanded = summary.get("case_audits", {}).get("expanded_tail", {})
    if not (
        expanded.get("projection_success_fraction") == 1.0
        and float(expanded.get("maximum_conserved_moment_defect", math.inf))
        <= STAGE56_MAX_CONSERVED_DEFECT
        and float(expanded.get("heat_flux_closure_relative_l2", math.inf))
        <= STAGE56_MAX_HEAT_FLUX_CLOSURE
        and float(expanded.get("maximum_floor_violation", math.inf)) == 0.0
        and float(expanded.get("maximum_active_fraction", math.inf))
        <= STAGE56_MAX_ACTIVE_FRACTION_GUARD
        and float(expanded.get("maximum_weighted_relative_modification", math.inf))
        <= STAGE56_MAX_RELATIVE_MODIFICATION_GUARD
    ):
        raise ValueError("Stage 56 expanded-tail feasibility endpoint mismatch")
    return summary


def build_stage57_config() -> LinearSidewallConfig:
    return LinearSidewallConfig(
        nx=STAGE57_GRID[0],
        ny=STAGE57_GRID[1],
        kn0=STAGE57_KNUDSEN,
        cold_hot_ratio=STAGE57_RATIO,
        viscosity_exponent=0.5,
        prandtl=STAGE41_PRANDTL,
        max_steps=STAGE57_MAX_ITERATIONS,
        cfl=0.2,
        tolerance=STAGE57_TOLERANCE,
        check_interval=STAGE57_CHECK_INTERVAL,
        minimum_steps=STAGE57_MINIMUM_ITERATIONS,
        positivity_floor=1.0e-30,
    )


def conservative_projected_shakhov_equilibrium(
    fields: Mapping[str, np.ndarray],
    quadrature,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Apply the Stage-56 bounded projection cell by cell.

    The reference is the retained positivity-clipped projected Shakhov target.
    The projection preserves its exact lower-bound construction while restoring
    density, both momenta, energy, and both Shakhov heat-flux moments.
    """
    phi_reference, psi_reference, clipping = projected_shakhov_equilibrium(
        fields,
        quadrature,
        prandtl=STAGE41_PRANDTL,
        correction_floor=STAGE41_CORRECTION_FLOOR,
    )
    phi_maxwellian, psi_maxwellian = projected_maxwellian(
        fields["rho"],
        fields["u"],
        fields["v"],
        fields["T"],
        quadrature,
    )
    shape = np.asarray(fields["rho"]).shape
    cell_count = int(np.prod(shape))
    phi_reference_flat = np.asarray(phi_reference, dtype=np.float64).reshape(
        cell_count, quadrature.point_count
    )
    psi_reference_flat = np.asarray(psi_reference, dtype=np.float64).reshape(
        cell_count, quadrature.point_count
    )
    phi_out = phi_reference_flat.copy()
    psi_out = psi_reference_flat.copy()
    phi_m = np.asarray(phi_maxwellian, dtype=np.float64).reshape(
        cell_count, quadrature.point_count
    )
    psi_m = np.asarray(psi_maxwellian, dtype=np.float64).reshape(
        cell_count, quadrature.point_count
    )
    flat = {
        key: np.asarray(value, dtype=np.float64).reshape(cell_count)
        for key, value in fields.items()
        if key in {"rho", "u", "v", "T", "qx", "qy"}
    }

    maximum_conserved_defect = 0.0
    heat_scaled_error_squared = 0.0
    minimum_success_fraction = 1.0
    maximum_floor_violation = 0.0
    maximum_active_fraction = 0.0
    maximum_relative_modification = 0.0
    maximum_projection_iterations = 0
    rank_loss_count = 0
    roundoff_floor_clamp_count = 0

    for index in range(cell_count):
        rho = max(float(flat["rho"][index]), 1.0e-14)
        u = float(flat["u"][index])
        v = float(flat["v"][index])
        temperature = max(float(flat["T"][index]), 1.0e-12)
        qx = float(flat["qx"][index])
        qy = float(flat["qy"][index])
        phi_lower, psi_lower = _retained_clipped_lower_bounds(
            rho,
            u,
            v,
            temperature,
            qx,
            qy,
            phi_m[index],
            psi_m[index],
            quadrature,
        )
        phi_basis, psi_basis = _moment_basis(
            quadrature.vx, quadrature.vy, u, v
        )
        target = _target_moments(rho, u, v, temperature, qx, qy)
        phi, psi, projection = bounded_conservative_projection(
            phi_out[index],
            psi_out[index],
            phi_lower,
            psi_lower,
            phi_basis,
            psi_basis,
            quadrature.weight,
            target,
            max_iterations=STAGE56_MAX_ACTIVE_SET_ITERATIONS,
        )
        phi_out[index] = phi
        psi_out[index] = psi
        defect = _linear_moments(
            phi, psi, phi_basis, psi_basis, quadrature.weight
        ) - target
        maximum_conserved_defect = max(
            maximum_conserved_defect,
            _relative_conserved_defect(defect, rho, temperature),
        )
        heat_scale = max(
            float(np.hypot(target[4], target[5])),
            rho * temperature ** 1.5,
            1.0e-14,
        )
        heat_scaled_error_squared += float(
            (defect[4] ** 2 + defect[5] ** 2) / heat_scale**2
        )
        floor_violation = max(
            float(projection["phi_floor_violation"]),
            float(projection["psi_floor_violation"]),
        )
        maximum_floor_violation = max(maximum_floor_violation, floor_violation)
        maximum_active_fraction = max(
            maximum_active_fraction, float(projection["active_fraction"])
        )
        maximum_relative_modification = max(
            maximum_relative_modification,
            _weighted_relative_modification(
                phi,
                psi,
                phi_reference_flat[index],
                psi_reference_flat[index],
                phi_m[index],
                psi_m[index],
                quadrature.weight,
            ),
        )
        maximum_projection_iterations = max(
            maximum_projection_iterations, int(projection["iterations"])
        )
        rank_loss_count += int(projection["linear_system_rank"] != 6)
        roundoff_floor_clamp_count += int(
            projection["roundoff_floor_clamp_count"]
        )
        success = bool(
            projection["converged"]
            and projection["linear_system_rank"] == 6
            and floor_violation
            <= STAGE56_BOUND_TOLERANCE
            * max(
                float(np.max(phi_lower)),
                float(np.max(psi_lower)),
                1.0e-300,
            )
        )
        if not success:
            minimum_success_fraction = 0.0

    diagnostics = {
        "projection_success_fraction": minimum_success_fraction,
        "maximum_conserved_moment_defect": maximum_conserved_defect,
        "heat_flux_closure_relative_l2": math.sqrt(
            heat_scaled_error_squared / max(cell_count, 1)
        ),
        "maximum_floor_violation": maximum_floor_violation,
        "maximum_active_fraction": maximum_active_fraction,
        "maximum_weighted_relative_modification": maximum_relative_modification,
        "maximum_projection_iterations": float(maximum_projection_iterations),
        "rank_loss_count": float(rank_loss_count),
        "roundoff_floor_clamp_count": float(roundoff_floor_clamp_count),
        "maximum_phi_clipped_weight_fraction": float(
            np.max(clipping["phi_clipped_weight_fraction"])
        ),
        "maximum_psi_clipped_weight_fraction": float(
            np.max(clipping["psi_clipped_weight_fraction"])
        ),
    }
    return (
        phi_out.reshape(phi_reference.shape),
        psi_out.reshape(psi_reference.shape),
        diagnostics,
    )


def conservative_source_iteration_step(
    phi: np.ndarray,
    psi: np.ndarray,
    cfg: LinearSidewallConfig,
    quadrature,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    incoming = projected_wall_incoming(phi, psi, cfg, quadrature)
    (
        left_phi,
        left_psi,
        right_phi,
        right_psi,
        bottom_phi,
        bottom_psi,
        top_phi,
        top_psi,
    ) = incoming
    fields = projected_macroscopic(phi, psi, quadrature)
    equilibrium_phi, equilibrium_psi, projection = (
        conservative_projected_shakhov_equilibrium(fields, quadrature)
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
    next_phi = phi + STAGE57_SOURCE_RELAXATION * (candidate_phi - phi)
    next_psi = psi + STAGE57_SOURCE_RELAXATION * (candidate_psi - psi)
    return (
        np.maximum(next_phi, cfg.positivity_floor),
        np.maximum(next_psi, cfg.positivity_floor),
        projection,
    )


def solve_conservative_stage57(
    cfg: LinearSidewallConfig,
    quadrature,
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
    aggregate = {
        "projection_success_fraction": 1.0,
        "maximum_conserved_moment_defect": 0.0,
        "heat_flux_closure_relative_l2": 0.0,
        "maximum_floor_violation": 0.0,
        "maximum_active_fraction": 0.0,
        "maximum_weighted_relative_modification": 0.0,
        "maximum_projection_iterations": 0.0,
        "rank_loss_count": 0.0,
        "roundoff_floor_clamp_count": 0.0,
        "maximum_phi_clipped_weight_fraction": 0.0,
        "maximum_psi_clipped_weight_fraction": 0.0,
    }

    for iteration in range(cfg.max_steps):
        phi, psi, projection = conservative_source_iteration_step(
            phi, psi, cfg, quadrature
        )
        aggregate["projection_success_fraction"] = min(
            aggregate["projection_success_fraction"],
            projection["projection_success_fraction"],
        )
        for key in (
            "maximum_conserved_moment_defect",
            "heat_flux_closure_relative_l2",
            "maximum_floor_violation",
            "maximum_active_fraction",
            "maximum_weighted_relative_modification",
            "maximum_projection_iterations",
            "maximum_phi_clipped_weight_fraction",
            "maximum_psi_clipped_weight_fraction",
        ):
            aggregate[key] = max(aggregate[key], projection[key])
        aggregate["rank_loss_count"] += projection["rank_loss_count"]
        aggregate["roundoff_floor_clamp_count"] += projection[
            "roundoff_floor_clamp_count"
        ]

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
        if (iteration + 1) % 50 == 0:
            gc.collect()

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
        "finite": finite,
        "work_proxy": int(
            (iteration + 1) * cfg.nx * cfg.ny * quadrature.point_count
        ),
        "projection_diagnostics": aggregate,
    }


def _compact_result(result: Mapping[str, object]) -> dict[str, object]:
    return {
        "iterations": int(result["iterations"]),
        "converged": bool(result["converged"]),
        "final_change": float(result["final_change"]),
        "predicted_qav": float(result["predicted_qav"]),
        "literature_qav": float(result["literature_qav"]),
        "qav_relative_error": float(result["qav_relative_error"]),
        "velocity_metrics": {
            key: float(value)
            for key, value in result["velocity_metrics"].items()
        },
        "wall_mass_balance_relative_error": float(
            result["wall_mass_balance_relative_error"]
        ),
        "minimum_phi": float(result["minimum_phi"]),
        "minimum_psi": float(result["minimum_psi"]),
        "finite": bool(result["finite"]),
        "work_proxy": int(result["work_proxy"]),
        "table_velocity": np.asarray(result["table_velocity"]).tolist(),
    }


def compare_arms(
    baseline: Mapping[str, object],
    conservative: Mapping[str, object],
) -> dict[str, object]:
    q0 = float(baseline["qav_relative_error"])
    q1 = float(conservative["qav_relative_error"])
    v0 = float(baseline["velocity_metrics"]["relative_rms"])
    v1 = float(conservative["velocity_metrics"]["relative_rms"])
    sign0 = float(baseline["velocity_metrics"]["sign_agreement"])
    sign1 = float(conservative["velocity_metrics"]["sign_agreement"])
    p0 = np.asarray(baseline["table_velocity"], dtype=np.float64)
    p1 = np.asarray(conservative["table_velocity"], dtype=np.float64)
    return {
        "qav_error_change_fraction": (q1 - q0) / max(q0, 1.0e-300),
        "qav_relative_change": abs(
            float(conservative["predicted_qav"])
            - float(baseline["predicted_qav"])
        )
        / max(abs(float(baseline["predicted_qav"])), 1.0e-300),
        "velocity_rms_error_change_fraction": (v1 - v0)
        / max(v0, 1.0e-300),
        "velocity_profile_change": float(
            np.linalg.norm(p1 - p0) / max(np.linalg.norm(p0), 1.0e-300)
        ),
        "sign_agreement_change": sign1 - sign0,
        "heat_flux_error_improves": q1 < q0,
        "velocity_rms_improves": v1 < v0,
    }


def _stable(result: Mapping[str, object]) -> bool:
    return bool(
        result.get("finite")
        and float(result.get("minimum_phi", 0.0)) > 0.0
        and float(result.get("minimum_psi", 0.0)) > 0.0
        and float(result.get("wall_mass_balance_relative_error", math.inf))
        < 1.0e-10
    )


def stage57_decision(
    baseline: Mapping[str, object],
    conservative: Mapping[str, object],
    comparison: Mapping[str, object],
    projection: Mapping[str, float],
) -> str:
    numeric_values = [
        float(value)
        for row in (baseline, conservative, comparison, projection)
        for value in row.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if not all(math.isfinite(value) for value in numeric_values):
        return "stage57_conservative_solver_nonfinite_blocker_requires_review"
    if not _stable(baseline):
        return "stage57_retained_clipped_baseline_numerical_blocker"
    if not _stable(conservative):
        return "stage57_conservative_solver_numerical_blocker_requires_review"
    projection_pass = bool(
        projection["projection_success_fraction"] == 1.0
        and projection["maximum_conserved_moment_defect"]
        <= STAGE56_MAX_CONSERVED_DEFECT
        and projection["heat_flux_closure_relative_l2"]
        <= STAGE56_MAX_HEAT_FLUX_CLOSURE
        and projection["maximum_floor_violation"] == 0.0
        and projection["maximum_active_fraction"]
        <= STAGE56_MAX_ACTIVE_FRACTION_GUARD
        and projection["maximum_weighted_relative_modification"]
        <= STAGE56_MAX_RELATIVE_MODIFICATION_GUARD
        and projection["rank_loss_count"] == 0.0
    )
    if not projection_pass:
        return "stage57_conservative_projection_in_solver_blocker_requires_review"
    if not bool(baseline["converged"]) or not bool(conservative["converged"]):
        return "stage57_stable_nonconverged_blocker_without_parameter_retuning"
    observable_guard = bool(
        float(comparison["qav_error_change_fraction"])
        <= STAGE57_MAX_OBSERVABLE_ERROR_DEGRADATION
        and float(comparison["velocity_rms_error_change_fraction"])
        <= STAGE57_MAX_OBSERVABLE_ERROR_DEGRADATION
        and float(comparison["sign_agreement_change"])
        >= STAGE57_MIN_SIGN_AGREEMENT_CHANGE
    )
    if not observable_guard:
        return (
            "stage57_conservative_solver_stable_but_observables_degrade_"
            "requires_review_before_full_resolution"
        )
    return (
        "stage57_conservative_solver_pilot_passes_"
        "stage58_frozen_64x64_confirmation"
    )


def run_stage57(
    stage56_artifact_dir: str | Path,
    output_dir: str | Path,
    **design,
) -> dict[str, object]:
    validate_stage57_design(**design)
    stage56_dir = Path(stage56_artifact_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    retained56 = _validate_stage56_artifact(stage56_dir)
    cfg = build_stage57_config()
    quadrature = mapped_polar_quadrature(
        *STAGE57_RULE, radial_scale=STAGE57_RADIAL_SCALE
    )

    baseline_raw = solve_stage42_pilot(
        cfg, quadrature, STAGE57_SOURCE_RELAXATION
    )
    conservative_raw = solve_conservative_stage57(cfg, quadrature)
    baseline = _compact_result(baseline_raw)
    conservative = _compact_result(conservative_raw)
    projection = {
        key: float(value)
        for key, value in conservative_raw["projection_diagnostics"].items()
    }
    comparison = compare_arms(baseline, conservative)
    decision = stage57_decision(
        baseline, conservative, comparison, projection
    )

    np.savez_compressed(
        out / "baseline_clipped_fields_and_profiles.npz",
        **{
            key: np.asarray(baseline_raw[key])
            for key in (
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
            )
        },
    )
    np.savez_compressed(
        out / "conservative_fields_and_profiles.npz",
        **{
            key: np.asarray(conservative_raw[key])
            for key in (
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
            )
        },
    )

    summary = {
        "stage": 57,
        "description": (
            "Paired single-physics-case solver-integration pilot comparing the "
            "retained clipped projected-Shakhov operator with the Stage-56 bounded "
            "conservative positivity projection at Kn0=10"
        ),
        "retained_stage56_endpoint": STAGE56_COMPLETED_ENDPOINT,
        "retained_stage56_decision": retained56["decision"],
        "configuration": {
            "kn0": STAGE57_KNUDSEN,
            "cold_hot_ratio": STAGE57_RATIO,
            "grid": list(STAGE57_GRID),
            "radial_nodes": STAGE57_RULE[0],
            "angular_nodes": STAGE57_RULE[1],
            "point_count": int(np.prod(STAGE57_RULE)),
            "radial_scale": STAGE57_RADIAL_SCALE,
            "radial_scale_selection_basis": (
                "expanded-tail rule with the strongest Stage-55 quadrature closure; "
                "not selected by benchmark fit"
            ),
            "prandtl": STAGE41_PRANDTL,
            "retained_correction_floor": STAGE41_CORRECTION_FLOOR,
            "source_relaxation": STAGE57_SOURCE_RELAXATION,
            "max_iterations": STAGE57_MAX_ITERATIONS,
            "minimum_iterations": STAGE57_MINIMUM_ITERATIONS,
            "check_interval": STAGE57_CHECK_INTERVAL,
            "tolerance": STAGE57_TOLERANCE,
            "maximum_observable_error_degradation": (
                STAGE57_MAX_OBSERVABLE_ERROR_DEGRADATION
            ),
            "physical_parameter_retuning": False,
            "collision_parameter_retuning": False,
            "transport_retuning": False,
            "wall_model_retuning": False,
            "normalization_retuning": False,
            "correction_floor_retuning": False,
            "source_relaxation_retuning": False,
            "grid_is_preregistered_integration_pilot_not_validation_resolution": True,
            "paired_retained_clipped_baseline": True,
        },
        "baseline_clipped": baseline,
        "conservative_projection": conservative,
        "projection_diagnostics": projection,
        "comparison": comparison,
        "decision": decision,
        "interpretation_guard": (
            "This 16x16 run tests whether the conservative positivity treatment can "
            "remain feasible through nonlinear solver iterations. Both improving and "
            "worsening benchmark comparisons are retained. Improvement is not external "
            "validation, and failure is a blocker rather than a reason to retune the "
            "0.05 floor, relaxation, physics, walls, normalization, or stopping rules."
        ),
        "scientific_conclusion": (
            "Only a stable, converged conservative arm with invariant closure, the "
            "retained lower bound, bounded deformation, no rank loss, and no more than "
            "10% relative degradation in either benchmark error may advance to one "
            "frozen 64x64 confirmation. Cross-Knudsen extension remains prohibited."
        ),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage56-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(
        json.dumps(
            run_stage57(args.stage56_artifact_dir, args.output_dir),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
