from __future__ import annotations

from pathlib import Path
import argparse
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
    projected_shakhov_equilibrium,
)
from .stage54_projected_collision_moment_audit import (
    _local_and_global_defects,
    restore_internal_fields,
)


STAGE64_COMPLETED_ENDPOINT = {
    "workflow_run_id": 30950873919,
    "workflow_job_id": 92132317102,
    "workflow_conclusion": "success",
    "tests_passed": 35,
    "tests_failed": 0,
    "test_duration_seconds": 0.45,
    "artifact_id": 8913290588,
    "artifact_size_bytes": 9374,
    "artifact_sha256": "6e4f94bc03fd53a45d63b007ab6d427e9587e2435add439ebb901f10cd521e1e",
    "source_head_sha": "8d689fb80cb5eaa7ad2d691563e313fbb15712c1",
    "summary_sha256": "e620f120b9a031215279561bc3a89cafca5b55dcbaa3a9c22c04497bb2951c92",
    "decision": "stage64_conservative_diagnostic_source_step_blocker",
}

STAGE58_COMPLETED_ENDPOINT = {
    "workflow_run_id": 30864287564,
    "workflow_job_id": 91852659651,
    "workflow_conclusion": "success",
    "tests_passed": 72,
    "tests_failed": 0,
    "artifact_id": 8882191150,
    "artifact_size_bytes": 265675,
    "artifact_sha256": "c1e1cb40d35439d44a93bae091f732f35e1790c2c4e3f1180b16d7f8fb54e8f6",
    "source_head_sha": "448ce586344052de1cf5dd0fd86e3ffb4b6a52be",
    "summary_sha256": "b91921a631bd92c2696c7bdd18668afb4994c909bbc7aa82cca1d0afc25ffced",
    "baseline_fields_sha256": "9aa53136e05917236f87fb9279c2ecc4e29d6056ca5ec34ca3bcb4d8f66aa822",
    "decision": "stage58_conservative_confirmation_stable_but_observables_degrade_requires_review_without_retuning",
}

STAGE65_GRID = (64, 64)
STAGE65_KNUDSEN = 10.0
STAGE65_COLD_HOT_RATIO = 0.1
STAGE65_RULE = (40, 96)
STAGE65_RADIAL_SCALE = 2.0
STAGE65_WALL_BAND_LAYERS = 4
STAGE65_CONSERVED_DEFECT_THRESHOLD = 1.0e-3
STAGE65_HEAT_FLUX_DEFECT_THRESHOLD = 1.0e-2
STAGE65_MATERIAL_CLIPPING_THRESHOLD = 1.0e-4
STAGE65_MINIMUM_CLIPPING_DEFECT_CORRELATION = 0.50
STAGE65_WALL_LOCALIZATION_THRESHOLD = 0.60
STAGE65_BROAD_CELL_FRACTION_THRESHOLD = 0.25
STAGE65_TOP_FRACTIONS = (0.01, 0.05, 0.10)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_stage65_design(
    grid: tuple[int, int] = STAGE65_GRID,
    kn0: float = STAGE65_KNUDSEN,
    cold_hot_ratio: float = STAGE65_COLD_HOT_RATIO,
    rule: tuple[int, int] = STAGE65_RULE,
    radial_scale: float = STAGE65_RADIAL_SCALE,
    prandtl: float = STAGE41_PRANDTL,
    correction_floor: float = STAGE41_CORRECTION_FLOOR,
    wall_band_layers: int = STAGE65_WALL_BAND_LAYERS,
) -> None:
    actual = (
        grid,
        kn0,
        cold_hot_ratio,
        rule,
        radial_scale,
        prandtl,
        correction_floor,
        wall_band_layers,
    )
    expected = (
        STAGE65_GRID,
        STAGE65_KNUDSEN,
        STAGE65_COLD_HOT_RATIO,
        STAGE65_RULE,
        STAGE65_RADIAL_SCALE,
        STAGE41_PRANDTL,
        STAGE41_CORRECTION_FLOOR,
        STAGE65_WALL_BAND_LAYERS,
    )
    if actual != expected:
        raise ValueError(
            "Stage 65 is frozen to the exact completed Stage 58 baseline-clipped "
            "64x64 Kn0=10 endpoint, Tc/Th=0.1, the 40x96 mapped-polar rule at "
            "radial scale 2.0, Pr=2/3, correction floor 0.05, and a four-cell "
            "wall band; it is an artifact-only localization audit, not retuning."
        )


def validate_stage64_artifact(root: str | Path) -> dict[str, object]:
    path = Path(root) / "summary.json"
    if not path.is_file() or sha256_file(path) != STAGE64_COMPLETED_ENDPOINT["summary_sha256"]:
        raise ValueError("Stage 64 artifact checksum mismatch")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("stage") != 64:
        raise ValueError("Stage 64 artifact stage mismatch")
    if summary.get("decision") != STAGE64_COMPLETED_ENDPOINT["decision"]:
        raise ValueError("Stage 64 artifact decision mismatch")
    return summary


def validate_stage58_artifact(root: str | Path) -> dict[str, object]:
    root = Path(root)
    expected = {
        "summary.json": STAGE58_COMPLETED_ENDPOINT["summary_sha256"],
        "baseline_clipped_fields_and_profiles.npz": STAGE58_COMPLETED_ENDPOINT[
            "baseline_fields_sha256"
        ],
    }
    for name, checksum in expected.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != checksum:
            raise ValueError(f"Stage 58 artifact checksum mismatch: {name}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("stage") != 58:
        raise ValueError("Stage 58 artifact stage mismatch")
    if summary.get("decision") != STAGE58_COMPLETED_ENDPOINT["decision"]:
        raise ValueError("Stage 58 artifact decision mismatch")
    cfg = summary.get("configuration", {})
    if (
        cfg.get("grid") != list(STAGE65_GRID)
        or cfg.get("kn0") != STAGE65_KNUDSEN
        or cfg.get("cold_hot_ratio") != STAGE65_COLD_HOT_RATIO
        or cfg.get("radial_nodes") != STAGE65_RULE[0]
        or cfg.get("angular_nodes") != STAGE65_RULE[1]
        or cfg.get("radial_scale") != STAGE65_RADIAL_SCALE
        or cfg.get("prandtl") != STAGE41_PRANDTL
        or cfg.get("retained_correction_floor") != STAGE41_CORRECTION_FLOOR
    ):
        raise ValueError("Stage 58 artifact frozen configuration mismatch")
    return summary


def wall_distance_layers(shape: tuple[int, int]) -> np.ndarray:
    if len(shape) != 2 or min(shape) < 2:
        raise ValueError("wall-distance map requires a two-dimensional grid")
    i, j = np.indices(shape)
    return np.minimum.reduce((i, j, shape[0] - 1 - i, shape[1] - 1 - j))


def safe_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    finite = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(finite) < 2:
        return 0.0
    a = a[finite]
    b = b[finite]
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def concentration_metrics(
    field: np.ndarray,
    top_fractions: tuple[float, ...] = STAGE65_TOP_FRACTIONS,
) -> dict[str, float]:
    values = np.asarray(field, dtype=np.float64).ravel()
    if values.size == 0 or np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("concentration field must be finite, nonnegative and nonempty")
    total = float(np.sum(values))
    result: dict[str, float] = {}
    ordered = np.sort(values)[::-1]
    for fraction in top_fractions:
        if not 0.0 < fraction <= 1.0:
            raise ValueError("top fractions must lie in (0,1]")
        count = max(1, int(math.ceil(fraction * values.size)))
        key = f"top_{int(round(100 * fraction))}_percent_share"
        result[key] = float(np.sum(ordered[:count]) / max(total, 1.0e-300))
    return result


def wall_band_metrics(
    field: np.ndarray,
    distance: np.ndarray,
    wall_band_layers: int = STAGE65_WALL_BAND_LAYERS,
) -> dict[str, object]:
    field = np.asarray(field, dtype=np.float64)
    distance = np.asarray(distance, dtype=np.int64)
    if field.shape != distance.shape:
        raise ValueError("field and wall-distance shapes must match")
    if wall_band_layers < 1:
        raise ValueError("wall band must contain at least one layer")
    total = float(np.sum(field))
    band = distance < wall_band_layers
    layer_rows = []
    for layer in range(int(np.max(distance)) + 1):
        mask = distance == layer
        layer_rows.append(
            {
                "layer": layer,
                "cell_count": int(np.count_nonzero(mask)),
                "mean": float(np.mean(field[mask])),
                "maximum": float(np.max(field[mask])),
                "share": float(np.sum(field[mask]) / max(total, 1.0e-300)),
            }
        )
    return {
        "wall_band_layers": wall_band_layers,
        "wall_band_cell_fraction": float(np.mean(band)),
        "wall_band_share": float(np.sum(field[band]) / max(total, 1.0e-300)),
        "interior_share": float(np.sum(field[~band]) / max(total, 1.0e-300)),
        "layers": layer_rows,
    }


def maximum_location(field: np.ndarray) -> dict[str, object]:
    field = np.asarray(field, dtype=np.float64)
    index = np.unravel_index(int(np.nanargmax(field)), field.shape)
    return {
        "index": [int(index[0]), int(index[1])],
        "cell_center": [
            (float(index[0]) + 0.5) / field.shape[0],
            (float(index[1]) + 0.5) / field.shape[1],
        ],
        "value": float(field[index]),
    }


def weighted_centroid(field: np.ndarray) -> list[float]:
    field = np.asarray(field, dtype=np.float64)
    i, j = np.indices(field.shape)
    total = float(np.sum(field))
    if total <= 0.0:
        return [0.5, 0.5]
    return [
        float(np.sum(((i + 0.5) / field.shape[0]) * field) / total),
        float(np.sum(((j + 0.5) / field.shape[1]) * field) / total),
    ]


def _field_summary(field: np.ndarray, distance: np.ndarray) -> dict[str, object]:
    field = np.asarray(field, dtype=np.float64)
    return {
        "minimum": float(np.min(field)),
        "mean": float(np.mean(field)),
        "rms": float(np.sqrt(np.mean(field**2))),
        "maximum": float(np.max(field)),
        "maximum_location": maximum_location(field),
        "weighted_centroid": weighted_centroid(field),
        "concentration": concentration_metrics(field),
        "wall_localization": wall_band_metrics(field, distance),
    }


def stage65_decision(metrics: Mapping[str, object]) -> str:
    if not bool(metrics.get("finite", False)):
        return "stage65_nonfinite_local_activation_map_blocker"
    material_clipping = bool(metrics["material_clipping"])
    material_defect = bool(metrics["material_source_defect"])
    if not material_clipping or not material_defect:
        return "stage65_no_material_full_cavity_clipping_source_defect_stop"
    correlation = float(metrics["maximum_clipping_defect_correlation"])
    if correlation < STAGE65_MINIMUM_CLIPPING_DEFECT_CORRELATION:
        return "stage65_material_defect_weakly_correlated_with_clipping_blocker"
    wall_share = float(metrics["maximum_wall_band_defect_share"])
    broad_fraction = float(metrics["maximum_material_cell_fraction"])
    if wall_share >= STAGE65_WALL_LOCALIZATION_THRESHOLD:
        return (
            "stage65_clipping_source_defects_wall_localized_"
            "stage66_frozen_wall_adjacent_source_observable_audit"
        )
    if broad_fraction >= STAGE65_BROAD_CELL_FRACTION_THRESHOLD:
        return (
            "stage65_clipping_source_defects_broad_"
            "stage66_frozen_collision_source_observable_audit"
        )
    return (
        "stage65_clipping_source_defects_mixed_localization_"
        "stage66_frozen_collision_source_observable_audit"
    )


def audit_fields(fields: Mapping[str, np.ndarray]) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    shape = np.asarray(fields["rho"]).shape
    if shape != STAGE65_GRID:
        raise ValueError("Stage 65 requires exact 64x64 fields")
    quadrature = mapped_polar_quadrature(
        *STAGE65_RULE, radial_scale=STAGE65_RADIAL_SCALE
    )
    phi, psi, clipping = projected_shakhov_equilibrium(
        dict(fields),
        quadrature,
        prandtl=STAGE41_PRANDTL,
        correction_floor=STAGE41_CORRECTION_FLOOR,
    )
    recovered = projected_macroscopic(phi, psi, quadrature)
    defect_summary, local_conserved, local_heat_flux = _local_and_global_defects(
        fields, recovered
    )
    del phi, psi, recovered

    phi_fraction = np.asarray(clipping["phi_clipped_weight_fraction"], dtype=np.float64)
    psi_fraction = np.asarray(clipping["psi_clipped_weight_fraction"], dtype=np.float64)
    combined_clipping = np.maximum(phi_fraction, psi_fraction)
    distance = wall_distance_layers(shape)
    q_strength = np.sqrt(
        np.asarray(fields["qx"]) ** 2 + np.asarray(fields["qy"]) ** 2
    ) / np.maximum(
        np.asarray(fields["rho"]) * np.asarray(fields["T"]) ** 1.5,
        1.0e-14,
    )

    finite_arrays = (
        local_conserved,
        local_heat_flux,
        phi_fraction,
        psi_fraction,
        combined_clipping,
        q_strength,
    )
    finite = all(np.all(np.isfinite(array)) for array in finite_arrays)
    conserved_material = local_conserved >= STAGE65_CONSERVED_DEFECT_THRESHOLD
    heat_material = local_heat_flux >= STAGE65_HEAT_FLUX_DEFECT_THRESHOLD
    clipping_material = combined_clipping >= STAGE65_MATERIAL_CLIPPING_THRESHOLD
    clipping_conserved_correlation = safe_correlation(combined_clipping, local_conserved)
    clipping_heat_correlation = safe_correlation(combined_clipping, local_heat_flux)
    q_clipping_correlation = safe_correlation(q_strength, combined_clipping)

    conserved_wall = wall_band_metrics(local_conserved, distance)
    heat_wall = wall_band_metrics(local_heat_flux, distance)
    metrics = {
        "finite": finite,
        "material_clipping": bool(np.any(clipping_material)),
        "material_source_defect": bool(np.any(conserved_material) or np.any(heat_material)),
        "maximum_clipping_defect_correlation": max(
            clipping_conserved_correlation, clipping_heat_correlation
        ),
        "maximum_wall_band_defect_share": max(
            float(conserved_wall["wall_band_share"]),
            float(heat_wall["wall_band_share"]),
        ),
        "maximum_material_cell_fraction": max(
            float(np.mean(conserved_material)), float(np.mean(heat_material))
        ),
    }
    summary = {
        "grid_shape": list(shape),
        "global_collision_defects": defect_summary,
        "clipping": {
            "fraction_cells_with_any_clipping": float(np.mean(combined_clipping > 0.0)),
            "fraction_cells_with_material_clipping": float(np.mean(clipping_material)),
            "maximum_phi_clipped_weight_fraction": float(np.max(phi_fraction)),
            "maximum_psi_clipped_weight_fraction": float(np.max(psi_fraction)),
            "mean_phi_clipped_weight_fraction": float(np.mean(phi_fraction)),
            "mean_psi_clipped_weight_fraction": float(np.mean(psi_fraction)),
            "minimum_raw_phi_factor": float(np.min(clipping["minimum_raw_phi_factor"])),
            "minimum_raw_psi_factor": float(np.min(clipping["minimum_raw_psi_factor"])),
            "combined_clipping_summary": _field_summary(combined_clipping, distance),
        },
        "local_conserved_defect": {
            **_field_summary(local_conserved, distance),
            "fraction_cells_above_material_threshold": float(np.mean(conserved_material)),
            "material_threshold": STAGE65_CONSERVED_DEFECT_THRESHOLD,
        },
        "local_heat_flux_closure_defect": {
            **_field_summary(local_heat_flux, distance),
            "fraction_cells_above_material_threshold": float(np.mean(heat_material)),
            "material_threshold": STAGE65_HEAT_FLUX_DEFECT_THRESHOLD,
        },
        "heat_flux_strength": _field_summary(q_strength, distance),
        "correlations": {
            "clipping_vs_conserved_defect": clipping_conserved_correlation,
            "clipping_vs_heat_flux_defect": clipping_heat_correlation,
            "heat_flux_strength_vs_clipping": q_clipping_correlation,
        },
        "decision_metrics": metrics,
    }
    arrays = {
        "local_conserved_defect": local_conserved,
        "local_heat_flux_closure_defect": local_heat_flux,
        "phi_clipped_weight_fraction": phi_fraction,
        "psi_clipped_weight_fraction": psi_fraction,
        "combined_clipping_weight_fraction": combined_clipping,
        "nondimensional_heat_flux_strength": q_strength,
        "wall_distance_layers": distance,
        "conserved_material_mask": conserved_material,
        "heat_flux_material_mask": heat_material,
        "clipping_material_mask": clipping_material,
    }
    return summary, arrays


def run_stage65(
    stage64_artifact_dir: str | Path,
    stage58_artifact_dir: str | Path,
    output_dir: str | Path,
    **design,
) -> dict[str, object]:
    validate_stage65_design(**design)
    stage64 = validate_stage64_artifact(stage64_artifact_dir)
    stage58 = validate_stage58_artifact(stage58_artifact_dir)
    baseline_path = Path(stage58_artifact_dir) / "baseline_clipped_fields_and_profiles.npz"
    with np.load(baseline_path) as data:
        fields = restore_internal_fields(data)
    audit, arrays = audit_fields(fields)
    decision = stage65_decision(audit["decision_metrics"])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "local_activation_maps.npz", **arrays)

    summary = {
        "stage": 65,
        "description": (
            "Artifact-only localization audit of retained 0.05 projected-Shakhov "
            "clipping and its collision-source defects on the exact completed "
            "Stage 58 baseline-clipped 64x64 Kn0=10 cavity fields."
        ),
        "retained_stage64_endpoint": STAGE64_COMPLETED_ENDPOINT,
        "retained_stage64_decision": stage64["decision"],
        "retained_stage58_endpoint": STAGE58_COMPLETED_ENDPOINT,
        "retained_stage58_decision": stage58["decision"],
        "configuration": {
            "grid": list(STAGE65_GRID),
            "kn0": STAGE65_KNUDSEN,
            "cold_hot_ratio": STAGE65_COLD_HOT_RATIO,
            "radial_nodes": STAGE65_RULE[0],
            "angular_nodes": STAGE65_RULE[1],
            "point_count": int(np.prod(STAGE65_RULE)),
            "radial_scale": STAGE65_RADIAL_SCALE,
            "prandtl": STAGE41_PRANDTL,
            "retained_correction_floor": STAGE41_CORRECTION_FLOOR,
            "wall_band_layers": STAGE65_WALL_BAND_LAYERS,
            "conserved_defect_threshold": STAGE65_CONSERVED_DEFECT_THRESHOLD,
            "heat_flux_defect_threshold": STAGE65_HEAT_FLUX_DEFECT_THRESHOLD,
            "material_clipping_threshold": STAGE65_MATERIAL_CLIPPING_THRESHOLD,
            "minimum_clipping_defect_correlation": STAGE65_MINIMUM_CLIPPING_DEFECT_CORRELATION,
            "wall_localization_threshold": STAGE65_WALL_LOCALIZATION_THRESHOLD,
            "broad_cell_fraction_threshold": STAGE65_BROAD_CELL_FRACTION_THRESHOLD,
            "solver_rerun": False,
            "physical_parameter_retuning": False,
            "collision_parameter_retuning": False,
            "correction_floor_retuning": False,
            "source_relaxation_retuning": False,
            "normalization_retuning": False,
            "transport_retuning": False,
            "velocity_quadrature_retuning": False,
            "wall_model_retuning": False,
            "conservative_projection_adopted": False,
            "cross_knudsen_extension_permitted": False,
        },
        "audit": audit,
        "decision": decision,
        "interpretation_guard": (
            "Stage 65 evaluates the active clipped collision target on saved converged "
            "macroscopic fields only. It does not reconstruct the full converged "
            "distribution, rerun transport or walls, establish causality for the Table-6 "
            "error, adopt a diagnostic projection, retune the floor, or authorize a "
            "MUSCL or cross-Knudsen extension."
        ),
        "negative_findings": [
            "All material clipping and source defects are retained exactly rather than removed by floor or source-relaxation retuning.",
            "The bounded-conservative diagnostic remains non-adopted after its strict Stage 64 blocker and its previously degraded full-cavity observables.",
            "The failed Stage 28 MUSCL endpoint remains negative and is not extended across Knudsen number."
        ],
        "scientifically_justified_next_scope": (
            "Use the frozen Stage 65 maps to quantify the signed contribution of the "
            "active clipped collision-source defect to heat-flux observables, stratified "
            "by wall distance and local thermodynamic state, without rerunning the solver "
            "or changing any physical, collision, wall, transport, quadrature, floor or "
            "normalization setting."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage64-artifact-dir", required=True)
    parser.add_argument("--stage58-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run_stage65(
                args.stage64_artifact_dir,
                args.stage58_artifact_dir,
                args.output_dir,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
