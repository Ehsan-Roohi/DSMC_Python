from __future__ import annotations

from pathlib import Path
import argparse
import gc
import hashlib
import json
import math
from typing import Mapping

import numpy as np

from .stage41_projected_polar_operator_audit import (
    STAGE41_CORRECTION_FLOOR,
    STAGE41_PRANDTL,
    mapped_polar_quadrature,
    projected_macroscopic,
    projected_maxwellian,
    projected_shakhov_equilibrium,
)

STAGE53_COMPLETED_ENDPOINT = {
    "workflow_run_id": 30803098842,
    "workflow_job_id": 91652089093,
    "workflow_conclusion": "success",
    "tests_passed": 87,
    "tests_failed": 0,
    "test_duration_seconds": 0.51,
    "artifact_id": 8854974012,
    "artifact_size_bytes": 266068,
    "artifact_sha256": "35a29adb215b29609084e7aa2e63d6c9f3ba738cbb3e0210926371c4889a0306",
    "source_head_sha": "0250e4fa43d2db74316c9ca2075a6f0b6a08fe6e",
    "summary_sha256": "7ef9603de0b80ebb6bce82c0b0787f42bb3827fc2e2c3a12df8fde1affd3f6ef",
    "compressed_tail_fields_sha256": "7f419c3e371ef64d23facad40aefa437ef07123a0efcfb988ebc45a01a8a3085",
    "expanded_tail_fields_sha256": "4bbe3130476f94a5d80991ccc7938bb3f09dc3dec8ea5fd4dce8c00dabf95c39",
    "decision": (
        "radial_mapping_tail_does_not_explain_cross_kn_heat_flux_"
        "stage54_projected_collision_moment_audit"
    ),
}

STAGE54_GRID = (64, 64)
STAGE54_KNUDSEN = 10.0
STAGE54_RATIO = 0.1
STAGE54_RULE = (32, 96)
STAGE54_CASES = (
    ("compressed_tail", 0.5),
    ("expanded_tail", 2.0),
)
STAGE54_MAX_CURRENT_CONSERVED_DEFECT = 1.0e-3
STAGE54_MAX_CURRENT_HEAT_FLUX_CLOSURE = 1.0e-2
STAGE54_MAX_UNCLIPPED_CONSERVED_DEFECT = 1.0e-5
STAGE54_MAX_UNCLIPPED_HEAT_FLUX_CLOSURE = 1.0e-3


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_stage54_design(
    grid=STAGE54_GRID,
    kn0=STAGE54_KNUDSEN,
    cold_hot_ratio=STAGE54_RATIO,
    rule=STAGE54_RULE,
    cases=STAGE54_CASES,
    correction_floor=STAGE41_CORRECTION_FLOOR,
    prandtl=STAGE41_PRANDTL,
) -> None:
    actual = (
        grid,
        kn0,
        cold_hot_ratio,
        rule,
        cases,
        correction_floor,
        prandtl,
    )
    expected = (
        STAGE54_GRID,
        STAGE54_KNUDSEN,
        STAGE54_RATIO,
        STAGE54_RULE,
        STAGE54_CASES,
        STAGE41_CORRECTION_FLOOR,
        STAGE41_PRANDTL,
    )
    if actual != expected:
        raise ValueError(
            "Stage 54 is frozen to the completed Stage 53 fields and the retained "
            "projected Shakhov collision operator"
        )


def restore_internal_fields(data: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    fields = {
        key: np.asarray(data[key], dtype=np.float64).copy()
        for key in ("rho", "T", "u", "v", "qx", "qy")
    }
    for key in ("u", "v", "qx", "qy"):
        fields[key] *= math.sqrt(2.0)
    return fields


def unclipped_projected_shakhov_equilibrium(
    fields: Mapping[str, np.ndarray],
    quadrature,
    prandtl: float = STAGE41_PRANDTL,
) -> tuple[np.ndarray, np.ndarray]:
    """Algebraic diagnostic only; negative values are retained and never solved."""
    rho = np.maximum(np.asarray(fields["rho"], dtype=np.float64), 1.0e-14)
    u = np.asarray(fields["u"], dtype=np.float64)
    v = np.asarray(fields["v"], dtype=np.float64)
    temperature = np.maximum(
        np.asarray(fields["T"], dtype=np.float64), 1.0e-12
    )
    qx = np.asarray(fields["qx"], dtype=np.float64)
    qy = np.asarray(fields["qy"], dtype=np.float64)
    phi_m, psi_m = projected_maxwellian(
        rho, u, v, temperature, quadrature
    )
    cx = quadrature.vx - u[..., None]
    cy = quadrature.vy - v[..., None]
    c_parallel2 = cx * cx + cy * cy
    c_dot_q = cx * qx[..., None] + cy * qy[..., None]
    coefficient = (1.0 - prandtl) / (
        5.0 * rho[..., None] * temperature[..., None] ** 2
    )
    raw_phi_factor = 1.0 + coefficient * c_dot_q * (
        c_parallel2 / temperature[..., None] - 4.0
    )
    raw_psi_factor = 1.0 + coefficient * c_dot_q * (
        c_parallel2 / temperature[..., None] - 2.0
    )
    with np.errstate(over="ignore", invalid="ignore"):
        phi = phi_m * raw_phi_factor
        psi = psi_m * raw_psi_factor
    density = np.sum(phi * quadrature.weight, axis=-1)
    denominator = np.where(
        np.abs(density) > 1.0e-14,
        density,
        np.copysign(1.0e-14, density + 1.0e-300),
    )
    scale = rho / denominator
    return phi * scale[..., None], psi * scale[..., None]


def _local_and_global_defects(
    fields: Mapping[str, np.ndarray],
    recovered: Mapping[str, np.ndarray],
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    rho = np.asarray(fields["rho"], dtype=np.float64)
    temperature = np.asarray(fields["T"], dtype=np.float64)
    target_mx = rho * np.asarray(fields["u"], dtype=np.float64)
    target_my = rho * np.asarray(fields["v"], dtype=np.float64)
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
    expected_q = np.stack(
        [
            (1.0 - STAGE41_PRANDTL) * fields["qx"],
            (1.0 - STAGE41_PRANDTL) * fields["qy"],
        ],
        axis=-1,
    )
    recovered_q = np.stack([recovered["qx"], recovered["qy"]], axis=-1)
    q_residual = np.linalg.norm(recovered_q - expected_q, axis=-1)
    global_q_scale = max(float(np.linalg.norm(expected_q)), 1.0e-14)
    local_q_scale = max(
        float(np.sqrt(np.mean(np.sum(expected_q**2, axis=-1)))), 1.0e-14
    )
    summary = {
        "component_maxima": {
            key: float(np.max(value)) for key, value in components.items()
        },
        "maximum_conserved_moment_defect": float(np.max(local_conserved)),
        "rms_conserved_moment_defect": float(
            np.sqrt(np.mean(local_conserved**2))
        ),
        "heat_flux_closure_relative_l2": float(
            np.linalg.norm(recovered_q - expected_q) / global_q_scale
        ),
    }
    return summary, local_conserved, q_residual / local_q_scale


def audit_case(
    fields: Mapping[str, np.ndarray],
    radial_scale: float,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    quadrature = mapped_polar_quadrature(
        *STAGE54_RULE, radial_scale=radial_scale
    )
    clipped_phi, clipped_psi, clipping = projected_shakhov_equilibrium(
        fields,
        quadrature,
        prandtl=STAGE41_PRANDTL,
        correction_floor=STAGE41_CORRECTION_FLOOR,
    )
    clipped_moments = projected_macroscopic(
        clipped_phi, clipped_psi, quadrature
    )
    clipped_summary, clipped_local, clipped_q_local = _local_and_global_defects(
        fields, clipped_moments
    )
    del clipped_phi, clipped_psi, clipped_moments
    gc.collect()

    diagnostic_phi, diagnostic_psi = unclipped_projected_shakhov_equilibrium(
        fields, quadrature
    )
    diagnostic_moments = projected_macroscopic(
        diagnostic_phi, diagnostic_psi, quadrature
    )
    diagnostic_summary, diagnostic_local, diagnostic_q_local = (
        _local_and_global_defects(fields, diagnostic_moments)
    )
    del diagnostic_phi, diagnostic_psi, diagnostic_moments
    gc.collect()

    phi_fraction = np.asarray(
        clipping["phi_clipped_weight_fraction"], dtype=np.float64
    )
    psi_fraction = np.asarray(
        clipping["psi_clipped_weight_fraction"], dtype=np.float64
    )
    combined_fraction = np.maximum(phi_fraction, psi_fraction)
    correlation = 0.0
    if np.std(combined_fraction) > 0.0 and np.std(clipped_local) > 0.0:
        correlation = float(
            np.corrcoef(combined_fraction.ravel(), clipped_local.ravel())[0, 1]
        )
    clipped_summary.update(
        {
            "maximum_phi_clipped_weight_fraction": float(np.max(phi_fraction)),
            "maximum_psi_clipped_weight_fraction": float(np.max(psi_fraction)),
            "mean_phi_clipped_weight_fraction": float(np.mean(phi_fraction)),
            "mean_psi_clipped_weight_fraction": float(np.mean(psi_fraction)),
            "fraction_cells_with_any_clipping": float(
                np.mean(combined_fraction > 0.0)
            ),
            "minimum_raw_phi_factor": float(
                np.min(clipping["minimum_raw_phi_factor"])
            ),
            "minimum_raw_psi_factor": float(
                np.min(clipping["minimum_raw_psi_factor"])
            ),
            "clipping_conserved_defect_correlation": correlation,
        }
    )
    summary = {
        "radial_scale": radial_scale,
        "current_clipped_operator": clipped_summary,
        "unclipped_algebraic_diagnostic": diagnostic_summary,
    }
    arrays = {
        "clipped_local_conserved_defect": clipped_local,
        "clipped_local_heat_flux_closure": clipped_q_local,
        "unclipped_local_conserved_defect": diagnostic_local,
        "unclipped_local_heat_flux_closure": diagnostic_q_local,
        "phi_clipped_weight_fraction": phi_fraction,
        "psi_clipped_weight_fraction": psi_fraction,
    }
    return summary, arrays


def stage54_decision(cases: Mapping[str, Mapping[str, object]]) -> str:
    current_bad = any(
        case["current_clipped_operator"]["maximum_conserved_moment_defect"]
        >= STAGE54_MAX_CURRENT_CONSERVED_DEFECT
        or case["current_clipped_operator"]["heat_flux_closure_relative_l2"]
        >= STAGE54_MAX_CURRENT_HEAT_FLUX_CLOSURE
        for case in cases.values()
    )
    unclipped_good = all(
        case["unclipped_algebraic_diagnostic"][
            "maximum_conserved_moment_defect"
        ]
        <= STAGE54_MAX_UNCLIPPED_CONSERVED_DEFECT
        and case["unclipped_algebraic_diagnostic"][
            "heat_flux_closure_relative_l2"
        ]
        <= STAGE54_MAX_UNCLIPPED_HEAT_FLUX_CLOSURE
        for case in cases.values()
    )
    if current_bad and unclipped_good:
        return (
            "positivity_clipping_breaks_collision_invariants_"
            "stage55_conservative_positive_projection_pilot"
        )
    if not unclipped_good:
        return "projected_collision_formula_or_quadrature_blocker"
    return (
        "projected_collision_moments_do_not_explain_cross_kn_heat_flux_"
        "stage55_knudsen_convention_audit"
    )


def _validate_stage53_artifact(stage53_dir: Path) -> dict[str, object]:
    expected_files = {
        "summary.json": STAGE53_COMPLETED_ENDPOINT["summary_sha256"],
        "compressed_tail_fields_and_profiles.npz": STAGE53_COMPLETED_ENDPOINT[
            "compressed_tail_fields_sha256"
        ],
        "expanded_tail_fields_and_profiles.npz": STAGE53_COMPLETED_ENDPOINT[
            "expanded_tail_fields_sha256"
        ],
    }
    for filename, expected_sha in expected_files.items():
        path = stage53_dir / filename
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise ValueError(f"Stage 53 artifact checksum mismatch: {filename}")
    summary = json.loads((stage53_dir / "summary.json").read_text())
    if summary.get("stage") != 53:
        raise ValueError("Stage 53 artifact stage mismatch")
    if summary.get("decision") != STAGE53_COMPLETED_ENDPOINT["decision"]:
        raise ValueError("Stage 53 artifact decision mismatch")
    if summary.get("configuration", {}).get("grid") != list(STAGE54_GRID):
        raise ValueError("Stage 53 artifact grid mismatch")
    if summary.get("configuration", {}).get("kn0") != STAGE54_KNUDSEN:
        raise ValueError("Stage 53 artifact Knudsen mismatch")
    expected_names = [name for name, _ in STAGE54_CASES]
    actual_names = [case.get("name") for case in summary.get("audit_cases", [])]
    if actual_names != expected_names:
        raise ValueError("Stage 53 artifact case ordering mismatch")
    return summary


def run_stage54(
    stage53_artifact_dir: str | Path,
    output_dir: str | Path,
    **design,
) -> dict[str, object]:
    validate_stage54_design(**design)
    stage53_dir = Path(stage53_artifact_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    retained = _validate_stage53_artifact(stage53_dir)

    case_rows: dict[str, dict[str, object]] = {}
    for name, radial_scale in STAGE54_CASES:
        path = stage53_dir / f"{name}_fields_and_profiles.npz"
        with np.load(path) as data:
            fields = restore_internal_fields(data)
        row, arrays = audit_case(fields, radial_scale)
        case_rows[name] = row
        np.savez_compressed(out / f"{name}_collision_diagnostics.npz", **arrays)
        del fields, arrays
        gc.collect()

    decision = stage54_decision(case_rows)
    summary = {
        "stage": 54,
        "description": (
            "Actual-state projected Shakhov collision-moment audit using the exact "
            "completed Stage 53 Kn0=10 fields"
        ),
        "retained_stage53_endpoint": STAGE53_COMPLETED_ENDPOINT,
        "retained_stage53_decision": retained["decision"],
        "configuration": {
            "kn0": STAGE54_KNUDSEN,
            "cold_hot_ratio": STAGE54_RATIO,
            "grid": list(STAGE54_GRID),
            "radial_nodes": STAGE54_RULE[0],
            "angular_nodes": STAGE54_RULE[1],
            "point_count": int(np.prod(STAGE54_RULE)),
            "cases": [
                {"name": name, "radial_scale": scale}
                for name, scale in STAGE54_CASES
            ],
            "prandtl": STAGE41_PRANDTL,
            "retained_correction_floor": STAGE41_CORRECTION_FLOOR,
            "current_conserved_defect_threshold": (
                STAGE54_MAX_CURRENT_CONSERVED_DEFECT
            ),
            "current_heat_flux_closure_threshold": (
                STAGE54_MAX_CURRENT_HEAT_FLUX_CLOSURE
            ),
            "unclipped_conserved_defect_threshold": (
                STAGE54_MAX_UNCLIPPED_CONSERVED_DEFECT
            ),
            "unclipped_heat_flux_closure_threshold": (
                STAGE54_MAX_UNCLIPPED_HEAT_FLUX_CLOSURE
            ),
            "physical_parameter_retuning": False,
            "collision_parameter_retuning": False,
            "transport_retuning": False,
            "wall_model_retuning": False,
            "normalization_retuning": False,
            "quadrature_retuning": False,
            "solver_rerun": False,
            "unclipped_equilibrium_is_diagnostic_only": True,
        },
        "case_audits": case_rows,
        "decision": decision,
        "interpretation_guard": (
            "The retained positive floor is audited exactly as used in the solver. "
            "The unclipped equilibrium is an algebraic diagnostic that may be "
            "negative and is never substituted into the cavity solve. Both mapping "
            "arms are retained, and no failed physical or numerical parameter is "
            "retuned."
        ),
        "scientific_conclusion": (
            "If the clipped operator violates density, momentum, or energy closure "
            "while the unclipped projected formula closes the same actual states, "
            "the next justified experiment is a conservative positivity-preserving "
            "projection pilot. Otherwise the collision operator is not blamed and "
            "the next audit moves to benchmark Knudsen conventions."
        ),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage53-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run_stage54(args.stage53_artifact_dir, args.output_dir),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
