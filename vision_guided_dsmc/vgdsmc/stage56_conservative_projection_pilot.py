from __future__ import annotations

from pathlib import Path
import argparse
import gc
import json
import math
from typing import Mapping

import numpy as np

from .stage41_projected_polar_operator_audit import (
    STAGE41_CORRECTION_FLOOR,
    STAGE41_PRANDTL,
    mapped_polar_quadrature,
    projected_maxwellian,
    projected_shakhov_equilibrium,
)
from .stage54_projected_collision_moment_audit import (
    STAGE53_COMPLETED_ENDPOINT,
    STAGE54_CASES,
    STAGE54_GRID,
    STAGE54_KNUDSEN,
    STAGE54_RATIO,
    _validate_stage53_artifact,
    restore_internal_fields,
    sha256_file,
)

STAGE55_COMPLETED_ENDPOINT = {
    "workflow_run_id": 30822272403,
    "workflow_job_id": 91714898163,
    "workflow_conclusion": "success",
    "tests_passed": 105,
    "tests_failed": 0,
    "test_duration_seconds": 0.56,
    "artifact_id": 8865015655,
    "artifact_size_bytes": 425843,
    "artifact_sha256": "4a491f39b2c00bf96b01b950bbbea04fb9a792c054717369cb92c0ca3c1dba19",
    "source_head_sha": "6f0d39548121a1c8012dbe9a4f05a9105c43262a",
    "summary_sha256": "c1bca0e32152a44ccc9bd946f98ea105d437ac24b2c752b0f6eb347df4d8d288",
    "compressed_tail_radial_diagnostics_sha256": "0a3b0ad803beaf2b3501e0579246ef58ff6fd372bdde56add820c5055e3b2626",
    "expanded_tail_radial_diagnostics_sha256": "1541bb783eab6e6fb96d0aab6e6330d878f7b5fe6789909b7f66d357bbb9ca84",
    "decision": (
        "radial_quadrature_closes_unclipped_formula_"
        "positivity_clipping_breaks_invariants_"
        "stage56_conservative_projection_pilot"
    ),
}

STAGE56_RULE = (40, 96)
STAGE56_MAX_ACTIVE_SET_ITERATIONS = 12
STAGE56_LINEAR_SOLVE_RCOND = 1.0e-13
STAGE56_BOUND_TOLERANCE = 1.0e-12
STAGE56_MAX_CONSERVED_DEFECT = 1.0e-10
STAGE56_MAX_HEAT_FLUX_CLOSURE = 1.0e-10
STAGE56_MAX_ACTIVE_FRACTION_GUARD = 0.25
STAGE56_MAX_RELATIVE_MODIFICATION_GUARD = 0.30


def validate_stage56_design(
    grid=STAGE54_GRID,
    kn0=STAGE54_KNUDSEN,
    cold_hot_ratio=STAGE54_RATIO,
    cases=STAGE54_CASES,
    rule=STAGE56_RULE,
    correction_floor=STAGE41_CORRECTION_FLOOR,
    prandtl=STAGE41_PRANDTL,
    max_iterations=STAGE56_MAX_ACTIVE_SET_ITERATIONS,
    linear_solve_rcond=STAGE56_LINEAR_SOLVE_RCOND,
) -> None:
    actual = (
        grid,
        kn0,
        cold_hot_ratio,
        cases,
        rule,
        correction_floor,
        prandtl,
        max_iterations,
        linear_solve_rcond,
    )
    expected = (
        STAGE54_GRID,
        STAGE54_KNUDSEN,
        STAGE54_RATIO,
        STAGE54_CASES,
        STAGE56_RULE,
        STAGE41_CORRECTION_FLOOR,
        STAGE41_PRANDTL,
        STAGE56_MAX_ACTIVE_SET_ITERATIONS,
        STAGE56_LINEAR_SOLVE_RCOND,
    )
    if actual != expected:
        raise ValueError(
            "Stage 56 is frozen to the completed Stage 53 fields, the exact Stage 55 "
            "radial-closure endpoint, the retained 0.05 positivity floor, and the "
            "preregistered bounded conservative projection"
        )


def _validate_stage55_artifact(stage55_dir: Path) -> dict[str, object]:
    expected_files = {
        "summary.json": STAGE55_COMPLETED_ENDPOINT["summary_sha256"],
        "compressed_tail_radial_40x96_closure_diagnostics.npz": (
            STAGE55_COMPLETED_ENDPOINT[
                "compressed_tail_radial_diagnostics_sha256"
            ]
        ),
        "expanded_tail_radial_40x96_closure_diagnostics.npz": (
            STAGE55_COMPLETED_ENDPOINT[
                "expanded_tail_radial_diagnostics_sha256"
            ]
        ),
    }
    for filename, expected_sha in expected_files.items():
        path = stage55_dir / filename
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise ValueError(f"Stage 55 artifact checksum mismatch: {filename}")
    summary = json.loads((stage55_dir / "summary.json").read_text())
    if summary.get("stage") != 55:
        raise ValueError("Stage 55 artifact stage mismatch")
    if summary.get("decision") != STAGE55_COMPLETED_ENDPOINT["decision"]:
        raise ValueError("Stage 55 artifact decision mismatch")
    configuration = summary.get("configuration", {})
    if configuration.get("grid") != list(STAGE54_GRID):
        raise ValueError("Stage 55 artifact grid mismatch")
    if configuration.get("kn0") != STAGE54_KNUDSEN:
        raise ValueError("Stage 55 artifact Knudsen mismatch")
    return summary


def _retained_clipped_lower_bounds(
    rho: float,
    u: float,
    v: float,
    temperature: float,
    qx: float,
    qy: float,
    phi_maxwellian: np.ndarray,
    psi_maxwellian: np.ndarray,
    quadrature,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact post-density-renormalization lower bounds of the retained clip."""
    cx = quadrature.vx - u
    cy = quadrature.vy - v
    c_parallel2 = cx * cx + cy * cy
    c_dot_q = cx * qx + cy * qy
    coefficient = (1.0 - STAGE41_PRANDTL) / (
        5.0 * rho * temperature**2
    )
    raw_phi_factor = 1.0 + coefficient * c_dot_q * (
        c_parallel2 / temperature - 4.0
    )
    clipped_phi_density = float(
        np.sum(
            phi_maxwellian
            * np.maximum(raw_phi_factor, STAGE41_CORRECTION_FLOOR)
            * quadrature.weight
        )
    )
    density_scale = rho / max(clipped_phi_density, 1.0e-14)
    return (
        density_scale * STAGE41_CORRECTION_FLOOR * phi_maxwellian,
        density_scale * STAGE41_CORRECTION_FLOOR * psi_maxwellian,
    )


def _moment_basis(
    vx: np.ndarray,
    vy: np.ndarray,
    u: float,
    v: float,
) -> tuple[np.ndarray, np.ndarray]:
    cx = vx - u
    cy = vy - v
    c_parallel2 = cx * cx + cy * cy
    phi_basis = np.vstack(
        [
            np.ones_like(vx),
            vx,
            vy,
            c_parallel2,
            0.5 * cx * c_parallel2,
            0.5 * cy * c_parallel2,
        ]
    )
    psi_basis = np.vstack(
        [
            np.zeros_like(vx),
            np.zeros_like(vx),
            np.zeros_like(vx),
            np.ones_like(vx),
            0.5 * cx,
            0.5 * cy,
        ]
    )
    return phi_basis, psi_basis


def _linear_moments(
    phi: np.ndarray,
    psi: np.ndarray,
    phi_basis: np.ndarray,
    psi_basis: np.ndarray,
    weight: np.ndarray,
) -> np.ndarray:
    return phi_basis @ (weight * phi) + psi_basis @ (weight * psi)


def _target_moments(
    rho: float,
    u: float,
    v: float,
    temperature: float,
    qx: float,
    qy: float,
) -> np.ndarray:
    return np.asarray(
        [
            rho,
            rho * u,
            rho * v,
            3.0 * rho * temperature,
            (1.0 - STAGE41_PRANDTL) * qx,
            (1.0 - STAGE41_PRANDTL) * qy,
        ],
        dtype=np.float64,
    )


def bounded_conservative_projection(
    phi_reference: np.ndarray,
    psi_reference: np.ndarray,
    phi_lower_bound: np.ndarray,
    psi_lower_bound: np.ndarray,
    phi_basis: np.ndarray,
    psi_basis: np.ndarray,
    weight: np.ndarray,
    target: np.ndarray,
    max_iterations: int = STAGE56_MAX_ACTIVE_SET_ITERATIONS,
    linear_solve_rcond: float = STAGE56_LINEAR_SOLVE_RCOND,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Weighted least-change projection with fixed lower-bound active sets."""
    phi_reference = np.asarray(phi_reference, dtype=np.float64)
    psi_reference = np.asarray(psi_reference, dtype=np.float64)
    phi_lower_bound = np.asarray(phi_lower_bound, dtype=np.float64)
    psi_lower_bound = np.asarray(psi_lower_bound, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if not (
        phi_reference.shape
        == psi_reference.shape
        == phi_lower_bound.shape
        == psi_lower_bound.shape
        == weight.shape
    ):
        raise ValueError("projection arrays must have identical one-dimensional shape")
    if phi_basis.shape != (6, weight.size) or psi_basis.shape != (6, weight.size):
        raise ValueError("projection basis must contain six moments")
    if target.shape != (6,):
        raise ValueError("projection target must contain six moments")

    phi_input_tolerance = STAGE56_BOUND_TOLERANCE * max(
        float(np.max(phi_lower_bound)), 1.0e-300
    )
    psi_input_tolerance = STAGE56_BOUND_TOLERANCE * max(
        float(np.max(psi_lower_bound)), 1.0e-300
    )
    if np.any(phi_reference < phi_lower_bound - phi_input_tolerance) or np.any(
        psi_reference < psi_lower_bound - psi_input_tolerance
    ):
        raise ValueError("reference distributions must satisfy retained lower bounds")
    phi_reference = np.maximum(phi_reference, phi_lower_bound)
    psi_reference = np.maximum(psi_reference, psi_lower_bound)

    phi_scale = np.maximum(phi_reference, phi_lower_bound)
    psi_scale = np.maximum(psi_reference, psi_lower_bound)
    active_phi = np.zeros(weight.size, dtype=bool)
    active_psi = np.zeros(weight.size, dtype=bool)
    phi = phi_reference.copy()
    psi = psi_reference.copy()
    converged = False
    rank = 0

    for iteration in range(1, max_iterations + 1):
        free_phi = ~active_phi
        free_psi = ~active_psi
        reference_phi = np.where(active_phi, phi_lower_bound, phi_reference)
        reference_psi = np.where(active_psi, psi_lower_bound, psi_reference)
        residual = target - _linear_moments(
            reference_phi,
            reference_psi,
            phi_basis,
            psi_basis,
            weight,
        )
        gram = (
            (
                phi_basis[:, free_phi]
                * (weight[free_phi] * phi_scale[free_phi] ** 2)
            )
            @ phi_basis[:, free_phi].T
        )
        gram += (
            (
                psi_basis[:, free_psi]
                * (weight[free_psi] * psi_scale[free_psi] ** 2)
            )
            @ psi_basis[:, free_psi].T
        )
        multiplier, _, rank, _ = np.linalg.lstsq(
            gram, residual, rcond=linear_solve_rcond
        )
        phi = reference_phi.copy()
        psi = reference_psi.copy()
        phi[free_phi] += phi_scale[free_phi] ** 2 * (
            phi_basis[:, free_phi].T @ multiplier
        )
        psi[free_psi] += psi_scale[free_psi] ** 2 * (
            psi_basis[:, free_psi].T @ multiplier
        )

        phi_tolerance = STAGE56_BOUND_TOLERANCE * np.maximum(
            np.max(phi_lower_bound), 1.0e-300
        )
        psi_tolerance = STAGE56_BOUND_TOLERANCE * np.maximum(
            np.max(psi_lower_bound), 1.0e-300
        )
        violating_phi = free_phi & (phi < phi_lower_bound - phi_tolerance)
        violating_psi = free_psi & (psi < psi_lower_bound - psi_tolerance)
        if not np.any(violating_phi) and not np.any(violating_psi):
            converged = True
            break
        active_phi |= violating_phi
        active_psi |= violating_psi

    roundoff_phi = phi < phi_lower_bound
    roundoff_psi = psi < psi_lower_bound
    roundoff_floor_clamp_count = int(
        np.sum(roundoff_phi) + np.sum(roundoff_psi)
    )
    phi = np.maximum(phi, phi_lower_bound)
    psi = np.maximum(psi, psi_lower_bound)
    defect = _linear_moments(phi, psi, phi_basis, psi_basis, weight) - target
    diagnostics = {
        "converged": converged,
        "iterations": iteration,
        "linear_system_rank": int(rank),
        "active_phi_count": int(np.sum(active_phi)),
        "active_psi_count": int(np.sum(active_psi)),
        "active_fraction": float(
            (np.sum(active_phi) + np.sum(active_psi)) / (2.0 * weight.size)
        ),
        "roundoff_floor_clamp_count": roundoff_floor_clamp_count,
        "maximum_absolute_moment_defect": float(np.max(np.abs(defect))),
        "phi_floor_violation": float(
            np.max(np.maximum(phi_lower_bound - phi, 0.0))
        ),
        "psi_floor_violation": float(
            np.max(np.maximum(psi_lower_bound - psi, 0.0))
        ),
    }
    return phi, psi, diagnostics


def _relative_conserved_defect(
    defect: np.ndarray,
    rho: float,
    temperature: float,
) -> float:
    scale = np.asarray(
        [
            max(rho, 1.0e-14),
            max(rho * math.sqrt(temperature), 1.0e-14),
            max(rho * math.sqrt(temperature), 1.0e-14),
            max(3.0 * rho * temperature, 1.0e-14),
        ]
    )
    return float(np.max(np.abs(defect[:4]) / scale))


def _weighted_relative_modification(
    phi: np.ndarray,
    psi: np.ndarray,
    phi_reference: np.ndarray,
    psi_reference: np.ndarray,
    phi_maxwellian: np.ndarray,
    psi_maxwellian: np.ndarray,
    weight: np.ndarray,
) -> float:
    numerator = float(
        np.sum(
            weight
            * (
                (phi - phi_reference) ** 2
                / np.maximum(phi_maxwellian, 1.0e-300)
                + (psi - psi_reference) ** 2
                / np.maximum(psi_maxwellian, 1.0e-300)
            )
        )
    )
    denominator = float(
        np.sum(
            weight
            * (
                phi_reference**2 / np.maximum(phi_maxwellian, 1.0e-300)
                + psi_reference**2 / np.maximum(psi_maxwellian, 1.0e-300)
            )
        )
    )
    return float(math.sqrt(numerator / max(denominator, 1.0e-300)))


def audit_case(
    fields: Mapping[str, np.ndarray],
    radial_scale: float,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    quadrature = mapped_polar_quadrature(
        *STAGE56_RULE, radial_scale=radial_scale
    )
    shape = np.asarray(fields["rho"]).shape
    cell_count = int(np.prod(shape))
    local_conserved = np.empty(cell_count, dtype=np.float64)
    local_heat_absolute = np.empty(cell_count, dtype=np.float64)
    active_fraction = np.empty(cell_count, dtype=np.float64)
    iterations = np.empty(cell_count, dtype=np.int64)
    relative_modification = np.empty(cell_count, dtype=np.float64)
    floor_violation = np.empty(cell_count, dtype=np.float64)
    roundoff_floor_clamp_count = np.empty(cell_count, dtype=np.int64)
    successful = np.empty(cell_count, dtype=bool)
    q_difference_squared = 0.0
    q_target_squared = 0.0

    flat = {
        key: np.asarray(value, dtype=np.float64).ravel()
        for key, value in fields.items()
    }
    for index in range(cell_count):
        rho = max(float(flat["rho"][index]), 1.0e-14)
        u = float(flat["u"][index])
        v = float(flat["v"][index])
        temperature = max(float(flat["T"][index]), 1.0e-12)
        qx = float(flat["qx"][index])
        qy = float(flat["qy"][index])
        scalar_fields = {
            "rho": np.asarray(rho),
            "u": np.asarray(u),
            "v": np.asarray(v),
            "T": np.asarray(temperature),
            "qx": np.asarray(qx),
            "qy": np.asarray(qy),
        }
        phi_reference, psi_reference, _ = projected_shakhov_equilibrium(
            scalar_fields,
            quadrature,
            prandtl=STAGE41_PRANDTL,
            correction_floor=STAGE41_CORRECTION_FLOOR,
        )
        phi_maxwellian, psi_maxwellian = projected_maxwellian(
            np.asarray(rho),
            np.asarray(u),
            np.asarray(v),
            np.asarray(temperature),
            quadrature,
        )
        phi_lower_bound, psi_lower_bound = _retained_clipped_lower_bounds(
            rho,
            u,
            v,
            temperature,
            qx,
            qy,
            phi_maxwellian,
            psi_maxwellian,
            quadrature,
        )
        phi_basis, psi_basis = _moment_basis(
            quadrature.vx, quadrature.vy, u, v
        )
        target = _target_moments(rho, u, v, temperature, qx, qy)
        phi, psi, projection = bounded_conservative_projection(
            phi_reference,
            psi_reference,
            phi_lower_bound,
            psi_lower_bound,
            phi_basis,
            psi_basis,
            quadrature.weight,
            target,
        )
        defect = _linear_moments(
            phi, psi, phi_basis, psi_basis, quadrature.weight
        ) - target
        local_conserved[index] = _relative_conserved_defect(
            defect, rho, temperature
        )
        local_heat_absolute[index] = float(np.hypot(defect[4], defect[5]))
        q_difference_squared += float(defect[4] ** 2 + defect[5] ** 2)
        q_target_squared += float(target[4] ** 2 + target[5] ** 2)
        active_fraction[index] = projection["active_fraction"]
        iterations[index] = projection["iterations"]
        relative_modification[index] = _weighted_relative_modification(
            phi,
            psi,
            phi_reference,
            psi_reference,
            phi_maxwellian,
            psi_maxwellian,
            quadrature.weight,
        )
        floor_violation[index] = max(
            projection["phi_floor_violation"],
            projection["psi_floor_violation"],
        )
        roundoff_floor_clamp_count[index] = projection[
            "roundoff_floor_clamp_count"
        ]
        successful[index] = bool(
            projection["converged"]
            and projection["linear_system_rank"] == 6
            and floor_violation[index]
            <= STAGE56_BOUND_TOLERANCE
            * max(
                float(np.max(phi_lower_bound)),
                float(np.max(psi_lower_bound)),
                1.0e-300,
            )
        )
        if (index + 1) % 256 == 0:
            gc.collect()

    local_conserved = local_conserved.reshape(shape)
    local_heat_absolute = local_heat_absolute.reshape(shape)
    active_fraction = active_fraction.reshape(shape)
    iterations = iterations.reshape(shape)
    relative_modification = relative_modification.reshape(shape)
    floor_violation = floor_violation.reshape(shape)
    roundoff_floor_clamp_count = roundoff_floor_clamp_count.reshape(shape)
    successful = successful.reshape(shape)
    summary = {
        "radial_scale": radial_scale,
        "radial_nodes": STAGE56_RULE[0],
        "angular_nodes": STAGE56_RULE[1],
        "point_count": int(np.prod(STAGE56_RULE)),
        "projection_success_fraction": float(np.mean(successful)),
        "maximum_conserved_moment_defect": float(np.max(local_conserved)),
        "rms_conserved_moment_defect": float(
            np.sqrt(np.mean(local_conserved**2))
        ),
        "heat_flux_closure_relative_l2": float(
            math.sqrt(q_difference_squared / max(q_target_squared, 1.0e-28))
        ),
        "maximum_floor_violation": float(np.max(floor_violation)),
        "mean_roundoff_floor_clamp_count": float(
            np.mean(roundoff_floor_clamp_count)
        ),
        "maximum_roundoff_floor_clamp_count": int(
            np.max(roundoff_floor_clamp_count)
        ),
        "mean_active_fraction": float(np.mean(active_fraction)),
        "maximum_active_fraction": float(np.max(active_fraction)),
        "mean_projection_iterations": float(np.mean(iterations)),
        "maximum_projection_iterations": int(np.max(iterations)),
        "mean_weighted_relative_modification": float(
            np.mean(relative_modification)
        ),
        "maximum_weighted_relative_modification": float(
            np.max(relative_modification)
        ),
    }
    arrays = {
        "local_conserved_moment_defect": local_conserved,
        "local_heat_flux_closure_absolute": local_heat_absolute,
        "active_fraction": active_fraction,
        "projection_iterations": iterations,
        "weighted_relative_modification": relative_modification,
        "floor_violation": floor_violation,
        "roundoff_floor_clamp_count": roundoff_floor_clamp_count,
        "projection_success": successful,
    }
    return summary, arrays


def stage56_decision(audits: Mapping[str, Mapping[str, float]]) -> str:
    finite = all(
        math.isfinite(float(value))
        for row in audits.values()
        for value in row.values()
        if isinstance(value, (int, float))
    )
    if not finite:
        return "conservative_projection_nonfinite_blocker_requires_review"
    closure_pass = all(
        row["projection_success_fraction"] == 1.0
        and row["maximum_conserved_moment_defect"]
        <= STAGE56_MAX_CONSERVED_DEFECT
        and row["heat_flux_closure_relative_l2"]
        <= STAGE56_MAX_HEAT_FLUX_CLOSURE
        and row["maximum_floor_violation"] == 0.0
        for row in audits.values()
    )
    if not closure_pass:
        return "conservative_projection_infeasible_blocker_requires_review"
    deformation_guard = all(
        row["maximum_active_fraction"] <= STAGE56_MAX_ACTIVE_FRACTION_GUARD
        and row["maximum_weighted_relative_modification"]
        <= STAGE56_MAX_RELATIVE_MODIFICATION_GUARD
        for row in audits.values()
    )
    if not deformation_guard:
        return (
            "conservative_projection_closes_but_large_deformation_"
            "requires_review_before_solver_rerun"
        )
    return (
        "conservative_positive_projection_closes_frozen_fields_"
        "stage57_single_case_solver_pilot"
    )


def run_stage56(
    stage53_artifact_dir: str | Path,
    stage55_artifact_dir: str | Path,
    output_dir: str | Path,
    **design,
) -> dict[str, object]:
    validate_stage56_design(**design)
    stage53_dir = Path(stage53_artifact_dir)
    stage55_dir = Path(stage55_artifact_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    retained53 = _validate_stage53_artifact(stage53_dir)
    retained55 = _validate_stage55_artifact(stage55_dir)

    audits: dict[str, dict[str, object]] = {}
    for case_name, radial_scale in STAGE54_CASES:
        with np.load(stage53_dir / f"{case_name}_fields_and_profiles.npz") as data:
            fields = restore_internal_fields(data)
        row, arrays = audit_case(fields, radial_scale)
        audits[case_name] = row
        np.savez_compressed(
            out / f"{case_name}_conservative_projection_diagnostics.npz",
            **arrays,
        )
        del fields, arrays
        gc.collect()

    decision = stage56_decision(audits)
    summary = {
        "stage": 56,
        "description": (
            "Bounded conservative-projection pilot on the exact completed Stage 53 "
            "Kn0=10 fields, using the Stage 55 radial quadrature that closed the "
            "algebraically unclipped projected Shakhov formula"
        ),
        "retained_stage53_endpoint": STAGE53_COMPLETED_ENDPOINT,
        "retained_stage53_decision": retained53["decision"],
        "retained_stage55_endpoint": STAGE55_COMPLETED_ENDPOINT,
        "retained_stage55_decision": retained55["decision"],
        "configuration": {
            "kn0": STAGE54_KNUDSEN,
            "cold_hot_ratio": STAGE54_RATIO,
            "grid": list(STAGE54_GRID),
            "mapping_cases": [
                {"name": name, "radial_scale": scale}
                for name, scale in STAGE54_CASES
            ],
            "radial_nodes": STAGE56_RULE[0],
            "angular_nodes": STAGE56_RULE[1],
            "point_count": int(np.prod(STAGE56_RULE)),
            "prandtl": STAGE41_PRANDTL,
            "retained_correction_floor": STAGE41_CORRECTION_FLOOR,
            "maximum_active_set_iterations": STAGE56_MAX_ACTIVE_SET_ITERATIONS,
            "linear_solve_rcond": STAGE56_LINEAR_SOLVE_RCOND,
            "maximum_conserved_defect": STAGE56_MAX_CONSERVED_DEFECT,
            "maximum_heat_flux_closure": STAGE56_MAX_HEAT_FLUX_CLOSURE,
            "maximum_active_fraction_guard": STAGE56_MAX_ACTIVE_FRACTION_GUARD,
            "maximum_relative_modification_guard": (
                STAGE56_MAX_RELATIVE_MODIFICATION_GUARD
            ),
            "physical_parameter_retuning": False,
            "collision_parameter_retuning": False,
            "transport_retuning": False,
            "wall_model_retuning": False,
            "normalization_retuning": False,
            "correction_floor_retuning": False,
            "solver_rerun": False,
            "projection_is_diagnostic_only": True,
            "radial_rule_is_retained_from_stage55_closure_audit": True,
        },
        "case_audits": audits,
        "decision": decision,
        "interpretation_guard": (
            "The conservative projection is evaluated only on frozen converged fields. "
            "It retains the exact 0.05 lower-bound construction and all physical "
            "parameters, does not replace the active collision operator, and does not "
            "rerun the cavity. Moment closure and positivity are operator feasibility "
            "results, not external validation or evidence that the heat-flux "
            "discrepancy is solved."
        ),
        "scientific_conclusion": (
            "A successful pilot isolates a feasible conservative positivity treatment "
            "for one frozen Kn0=10 state. The preregistered next experiment is a single "
            "frozen solver rerun before any cross-Knudsen extension. Failure, rank loss, "
            "large deformation, or loss of the retained floor is a blocker rather than "
            "a reason to retune the failed clipping parameter."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage53-artifact-dir", required=True)
    parser.add_argument("--stage55-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_stage56(
        args.stage53_artifact_dir,
        args.stage55_artifact_dir,
        args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
