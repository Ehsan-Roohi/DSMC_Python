from __future__ import annotations

from pathlib import Path
import argparse
import gc
import json
import math
from typing import Mapping

import numpy as np

from .stage41_projected_polar_operator_audit import (
    STAGE41_PRANDTL,
    mapped_polar_quadrature,
    projected_macroscopic,
)
from .stage54_projected_collision_moment_audit import (
    STAGE53_COMPLETED_ENDPOINT,
    STAGE54_CASES,
    STAGE54_GRID,
    STAGE54_KNUDSEN,
    STAGE54_MAX_UNCLIPPED_CONSERVED_DEFECT,
    STAGE54_MAX_UNCLIPPED_HEAT_FLUX_CLOSURE,
    STAGE54_RATIO,
    _validate_stage53_artifact,
    restore_internal_fields,
    sha256_file,
    unclipped_projected_shakhov_equilibrium,
)

STAGE54_COMPLETED_ENDPOINT = {
    "workflow_run_id": 30814308875,
    "workflow_job_id": 91688181380,
    "workflow_conclusion": "success",
    "tests_passed": 96,
    "tests_failed": 0,
    "test_duration_seconds": 0.53,
    "artifact_id": 8858775690,
    "artifact_size_bytes": 274998,
    "artifact_sha256": "0b8ee8fae6ef5d74eafbb17802ce6f76846f0db95f007ab299e19f4db1726afb",
    "source_head_sha": "0899356cd4ece46ca60417a00ba1c8ed0fdb7a65",
    "summary_sha256": "9382cb1dc259faa5c1b86bc3f9a7629c08811fa530083f71e5a4a090b9820532",
    "compressed_tail_diagnostics_sha256": "20c40eee36d1cbf95fe7f62c01115e9ec2c92c6254417ed7333bf617ac54f6bd",
    "expanded_tail_diagnostics_sha256": "32e98c302be071bf4d6e791f11709f853e3833ba20c8c4112cdccbe2542ac7e3",
    "decision": "projected_collision_formula_or_quadrature_blocker",
}

STAGE55_RULES = (
    ("retained_32x96", (32, 96)),
    ("radial_40x96", (40, 96)),
    ("angular_32x120", (32, 120)),
    ("coupled_40x120", (40, 120)),
)
STAGE55_CHUNK_ROWS = 4
STAGE55_MINIMUM_TREND_REDUCTION = 0.5
STAGE55_MAXIMUM_ALLOWED_WORSENING = 0.1


def validate_stage55_design(
    grid=STAGE54_GRID,
    kn0=STAGE54_KNUDSEN,
    cold_hot_ratio=STAGE54_RATIO,
    cases=STAGE54_CASES,
    rules=STAGE55_RULES,
    chunk_rows=STAGE55_CHUNK_ROWS,
    prandtl=STAGE41_PRANDTL,
) -> None:
    actual = (grid, kn0, cold_hot_ratio, cases, rules, chunk_rows, prandtl)
    expected = (
        STAGE54_GRID,
        STAGE54_KNUDSEN,
        STAGE54_RATIO,
        STAGE54_CASES,
        STAGE55_RULES,
        STAGE55_CHUNK_ROWS,
        STAGE41_PRANDTL,
    )
    if actual != expected:
        raise ValueError(
            "Stage 55 is frozen to the exact Stage 53 states, the completed Stage 54 "
            "endpoint, and the preregistered radial/angular quadrature sequence"
        )


def _validate_stage54_artifact(stage54_dir: Path) -> dict[str, object]:
    expected_files = {
        "summary.json": STAGE54_COMPLETED_ENDPOINT["summary_sha256"],
        "compressed_tail_collision_diagnostics.npz": STAGE54_COMPLETED_ENDPOINT[
            "compressed_tail_diagnostics_sha256"
        ],
        "expanded_tail_collision_diagnostics.npz": STAGE54_COMPLETED_ENDPOINT[
            "expanded_tail_diagnostics_sha256"
        ],
    }
    for filename, expected_sha in expected_files.items():
        path = stage54_dir / filename
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise ValueError(f"Stage 54 artifact checksum mismatch: {filename}")
    summary = json.loads((stage54_dir / "summary.json").read_text())
    if summary.get("stage") != 54:
        raise ValueError("Stage 54 artifact stage mismatch")
    if summary.get("decision") != STAGE54_COMPLETED_ENDPOINT["decision"]:
        raise ValueError("Stage 54 artifact decision mismatch")
    configuration = summary.get("configuration", {})
    if configuration.get("grid") != list(STAGE54_GRID):
        raise ValueError("Stage 54 artifact grid mismatch")
    if configuration.get("kn0") != STAGE54_KNUDSEN:
        raise ValueError("Stage 54 artifact Knudsen mismatch")
    return summary


def _chunk_metrics(
    target: Mapping[str, np.ndarray],
    recovered: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, float, float]:
    rho = np.asarray(target["rho"], dtype=np.float64)
    temperature = np.asarray(target["T"], dtype=np.float64)
    target_mx = rho * np.asarray(target["u"], dtype=np.float64)
    target_my = rho * np.asarray(target["v"], dtype=np.float64)
    recovered_mx = recovered["rho"] * recovered["u"]
    recovered_my = recovered["rho"] * recovered["v"]
    target_internal = 3.0 * rho * temperature
    momentum_scale = np.maximum(rho * np.sqrt(temperature), 1.0e-12)
    energy_scale = np.maximum(target_internal, 1.0e-12)
    components = {
        "density": np.abs(recovered["rho"] - rho) / np.maximum(rho, 1.0e-12),
        "x_momentum": np.abs(recovered_mx - target_mx) / momentum_scale,
        "y_momentum": np.abs(recovered_my - target_my) / momentum_scale,
        "internal_energy": np.abs(
            recovered["total_internal_moment"] - target_internal
        )
        / energy_scale,
    }
    local_conserved = np.maximum.reduce(tuple(components.values()))
    expected_qx = (1.0 - STAGE41_PRANDTL) * target["qx"]
    expected_qy = (1.0 - STAGE41_PRANDTL) * target["qy"]
    q_residual = np.sqrt(
        (recovered["qx"] - expected_qx) ** 2
        + (recovered["qy"] - expected_qy) ** 2
    )
    q_difference_squared = float(
        np.sum(
            (recovered["qx"] - expected_qx) ** 2
            + (recovered["qy"] - expected_qy) ** 2
        )
    )
    q_target_squared = float(np.sum(expected_qx**2 + expected_qy**2))
    return (
        components,
        local_conserved,
        q_residual,
        q_difference_squared,
        q_target_squared,
    )


def audit_rule(
    fields: Mapping[str, np.ndarray],
    radial_scale: float,
    rule: tuple[int, int],
    chunk_rows: int = STAGE55_CHUNK_ROWS,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    quadrature = mapped_polar_quadrature(*rule, radial_scale=radial_scale)
    ny, nx = np.asarray(fields["rho"]).shape
    component_maxima = {
        "density": 0.0,
        "x_momentum": 0.0,
        "y_momentum": 0.0,
        "internal_energy": 0.0,
    }
    local_conserved_blocks: list[np.ndarray] = []
    local_q_blocks: list[np.ndarray] = []
    conserved_square_sum = 0.0
    conserved_count = 0
    q_difference_squared = 0.0
    q_target_squared = 0.0

    for first_row in range(0, ny, chunk_rows):
        last_row = min(first_row + chunk_rows, ny)
        chunk = {
            key: np.asarray(value[first_row:last_row], dtype=np.float64)
            for key, value in fields.items()
        }
        phi, psi = unclipped_projected_shakhov_equilibrium(chunk, quadrature)
        recovered = projected_macroscopic(phi, psi, quadrature)
        components, local_conserved, local_q, q_diff2, q_target2 = _chunk_metrics(
            chunk, recovered
        )
        for key, value in components.items():
            component_maxima[key] = max(
                component_maxima[key], float(np.max(value))
            )
        local_conserved_blocks.append(local_conserved)
        local_q_blocks.append(local_q)
        conserved_square_sum += float(np.sum(local_conserved**2))
        conserved_count += int(local_conserved.size)
        q_difference_squared += q_diff2
        q_target_squared += q_target2
        del chunk, phi, psi, recovered, components
        gc.collect()

    local_conserved = np.concatenate(local_conserved_blocks, axis=0)
    local_q = np.concatenate(local_q_blocks, axis=0)
    summary = {
        "radial_nodes": rule[0],
        "angular_nodes": rule[1],
        "point_count": int(np.prod(rule)),
        "radial_scale": radial_scale,
        "component_maxima": component_maxima,
        "maximum_conserved_moment_defect": float(np.max(local_conserved)),
        "rms_conserved_moment_defect": float(
            math.sqrt(conserved_square_sum / max(conserved_count, 1))
        ),
        "heat_flux_closure_relative_l2": float(
            math.sqrt(q_difference_squared / max(q_target_squared, 1.0e-28))
        ),
    }
    arrays = {
        "local_conserved_moment_defect": local_conserved,
        "local_heat_flux_closure_absolute": local_q,
    }
    return summary, arrays


def _relative_reduction(baseline: float, candidate: float) -> float:
    return float((baseline - candidate) / max(abs(baseline), 1.0e-300))


def stage55_decision(
    audits: Mapping[str, Mapping[str, Mapping[str, float]]]
) -> str:
    coupled_closes = all(
        rows["coupled_40x120"]["maximum_conserved_moment_defect"]
        <= STAGE54_MAX_UNCLIPPED_CONSERVED_DEFECT
        and rows["coupled_40x120"]["heat_flux_closure_relative_l2"]
        <= STAGE54_MAX_UNCLIPPED_HEAT_FLUX_CLOSURE
        for rows in audits.values()
    )
    if coupled_closes:
        return (
            "radial_quadrature_closes_unclipped_formula_"
            "positivity_clipping_breaks_invariants_"
            "stage56_conservative_projection_pilot"
        )

    finite = all(
        math.isfinite(row[metric])
        for rows in audits.values()
        for row in rows.values()
        for metric in (
            "maximum_conserved_moment_defect",
            "heat_flux_closure_relative_l2",
        )
    )
    if not finite:
        return "projected_collision_formula_blocker_requires_review"

    worsening = any(
        rows["coupled_40x120"][metric]
        > (1.0 + STAGE55_MAXIMUM_ALLOWED_WORSENING)
        * rows["retained_32x96"][metric]
        for rows in audits.values()
        for metric in (
            "maximum_conserved_moment_defect",
            "heat_flux_closure_relative_l2",
        )
    )
    if worsening:
        return "projected_collision_formula_blocker_requires_review"

    compressed = audits["compressed_tail"]
    conserved_reduction = _relative_reduction(
        compressed["retained_32x96"]["maximum_conserved_moment_defect"],
        compressed["coupled_40x120"]["maximum_conserved_moment_defect"],
    )
    heat_reduction = _relative_reduction(
        compressed["retained_32x96"]["heat_flux_closure_relative_l2"],
        compressed["coupled_40x120"]["heat_flux_closure_relative_l2"],
    )
    if min(conserved_reduction, heat_reduction) >= STAGE55_MINIMUM_TREND_REDUCTION:
        return (
            "unclipped_formula_quadrature_converging_"
            "stage56_higher_radial_resolution_confirmation"
        )
    return (
        "unclipped_formula_quadrature_unresolved_"
        "stage56_higher_radial_resolution_confirmation"
    )


def run_stage55(
    stage53_artifact_dir: str | Path,
    stage54_artifact_dir: str | Path,
    output_dir: str | Path,
    **design,
) -> dict[str, object]:
    validate_stage55_design(**design)
    stage53_dir = Path(stage53_artifact_dir)
    stage54_dir = Path(stage54_artifact_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    retained53 = _validate_stage53_artifact(stage53_dir)
    retained54 = _validate_stage54_artifact(stage54_dir)

    audits: dict[str, dict[str, dict[str, object]]] = {}
    for case_name, radial_scale in STAGE54_CASES:
        with np.load(stage53_dir / f"{case_name}_fields_and_profiles.npz") as data:
            fields = restore_internal_fields(data)
        case_rows: dict[str, dict[str, object]] = {}
        for rule_name, rule in STAGE55_RULES:
            row, arrays = audit_rule(fields, radial_scale, rule)
            case_rows[rule_name] = row
            np.savez_compressed(
                out / f"{case_name}_{rule_name}_closure_diagnostics.npz",
                **arrays,
            )
            del arrays
            gc.collect()
        baseline = case_rows["retained_32x96"]
        for row in case_rows.values():
            row["conserved_defect_reduction_from_retained"] = _relative_reduction(
                baseline["maximum_conserved_moment_defect"],
                row["maximum_conserved_moment_defect"],
            )
            row["heat_flux_closure_reduction_from_retained"] = _relative_reduction(
                baseline["heat_flux_closure_relative_l2"],
                row["heat_flux_closure_relative_l2"],
            )
        audits[case_name] = case_rows
        del fields
        gc.collect()

    decision = stage55_decision(audits)
    summary = {
        "stage": 55,
        "description": (
            "Orthogonal radial/angular quadrature-closure audit of the algebraically "
            "unclipped projected Shakhov formula on the exact completed Stage 53 "
            "Kn0=10 fields after Stage 54 retained a strict formula-or-quadrature "
            "blocker endpoint"
        ),
        "retained_stage53_endpoint": STAGE53_COMPLETED_ENDPOINT,
        "retained_stage53_decision": retained53["decision"],
        "retained_stage54_endpoint": STAGE54_COMPLETED_ENDPOINT,
        "retained_stage54_decision": retained54["decision"],
        "configuration": {
            "kn0": STAGE54_KNUDSEN,
            "cold_hot_ratio": STAGE54_RATIO,
            "grid": list(STAGE54_GRID),
            "mapping_cases": [
                {"name": name, "radial_scale": scale}
                for name, scale in STAGE54_CASES
            ],
            "quadrature_rules": [
                {
                    "name": name,
                    "radial_nodes": rule[0],
                    "angular_nodes": rule[1],
                    "point_count": int(np.prod(rule)),
                }
                for name, rule in STAGE55_RULES
            ],
            "prandtl": STAGE41_PRANDTL,
            "conserved_defect_threshold": (
                STAGE54_MAX_UNCLIPPED_CONSERVED_DEFECT
            ),
            "heat_flux_closure_threshold": (
                STAGE54_MAX_UNCLIPPED_HEAT_FLUX_CLOSURE
            ),
            "physical_parameter_retuning": False,
            "collision_parameter_retuning": False,
            "transport_retuning": False,
            "wall_model_retuning": False,
            "normalization_retuning": False,
            "active_solver_quadrature_retuning": False,
            "solver_rerun": False,
            "audit_quadrature_varied": True,
            "unclipped_equilibrium_is_diagnostic_only": True,
        },
        "case_audits": audits,
        "decision": decision,
        "interpretation_guard": (
            "Only diagnostic velocity quadrature is varied on frozen fields. The "
            "active cavity solution is not rerun, the positivity-clipped operator is "
            "not replaced, the strict Stage 54 thresholds are unchanged, and every "
            "radial, angular, coupled, mixed, negative, or blocker outcome is retained."
        ),
        "scientific_conclusion": (
            "If radial or coupled refinement brings both mapping arms below the exact "
            "Stage 54 closure thresholds while angular refinement alone does not, the "
            "unclipped projected formula is supported and the retained positivity floor "
            "is isolated as the invariant-breaking mechanism. The next experiment is "
            "then a conservative positivity-preserving projection pilot, not physical "
            "parameter retuning."
        ),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage53-artifact-dir", required=True)
    parser.add_argument("--stage54-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run_stage55(
                args.stage53_artifact_dir,
                args.stage54_artifact_dir,
                args.output_dir,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
