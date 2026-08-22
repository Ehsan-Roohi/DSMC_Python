from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import stage111_axis_conditioned_asymmetry_audit as s111

STAGE111_RUN_ID = 31590035358
STAGE111_JOB_ID = 94092631513
STAGE111_ARTIFACT_ID = 9149082510
STAGE111_ARTIFACT_SHA256 = "a83c09039e956a47f4d18a86db5f2541fe622b9405bb9b09266c4c2d54f1cd7e"
STAGE111_SUMMARY_SHA256 = "93cff62b97e29e7ad5600b7b0c26b5b39bd3f71517e70c15447c95be239f5a77"
STAGE111_MAPS_SHA256 = "78b8173d81eaf2523791201690d6464295b8aa65413d9ccb3eb2c2215ac85407"
STAGE111_DECISION = "stage111_x_axis_asymmetry_coupled_stage112_axis_specific_spatial_audit"

GRID = s111.GRID
KNUDSEN = s111.KNUDSEN
COLD_HOT_RATIO = s111.COLD_HOT_RATIO
RULE = s111.RULE
RADIAL_SCALE = s111.RADIAL_SCALE
LIMITER = s111.LIMITER
BOUNDARY_SLOPE = s111.BOUNDARY_SLOPE
SOURCE_RELAXATION = s111.SOURCE_RELAXATION
TOLERANCE = s111.TOLERANCE
CORRECTION_FLOOR = s111.CORRECTION_FLOOR
DIAGNOSTIC_STEPS = s111.DIAGNOSTIC_STEPS
WALL_BAND_CELLS = s111.WALL_BAND_CELLS
DOMINANT_RADIAL_SHELL = s111.DOMINANT_RADIAL_SHELL
RADIAL_SHELL_COUNT = s111.RADIAL_SHELL_COUNT
RADIAL_NODES_PER_SHELL = s111.RADIAL_NODES_PER_SHELL
INTERIOR_EXTENT = s111.INTERIOR_EXTENT
MODE_CLOSURE_TOLERANCE = s111.MODE_CLOSURE_TOLERANCE

SPATIAL_BANDS_PER_AXIS = 4
BAND_SIZE = INTERIOR_EXTENT // SPATIAL_BANDS_PER_AXIS
SPATIAL_TILE_COUNT = SPATIAL_BANDS_PER_AXIS**2
SINGLE_TILE_SHARE_GUARD = 0.25
CONTIGUOUS_2X2_SHARE_GUARD = 0.50
OUTER_X_PAIR_SHARE_GUARD = 0.70
OUTER_TO_INNER_RATIO_GUARD = 2.0
LEFT_RIGHT_RELATIVE_IMBALANCE_GUARD = 0.15


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_stage112_design(**overrides: object) -> None:
    frozen = {
        "grid": GRID,
        "kn0": KNUDSEN,
        "cold_hot_ratio": COLD_HOT_RATIO,
        "rule": RULE,
        "radial_scale": RADIAL_SCALE,
        "limiter": LIMITER,
        "boundary_slope": BOUNDARY_SLOPE,
        "source_relaxation": SOURCE_RELAXATION,
        "tolerance": TOLERANCE,
        "correction_floor": CORRECTION_FLOOR,
        "diagnostic_steps": DIAGNOSTIC_STEPS,
        "wall_band_cells": WALL_BAND_CELLS,
        "dominant_radial_shell": DOMINANT_RADIAL_SHELL,
        "radial_shell_count": RADIAL_SHELL_COUNT,
        "radial_nodes_per_shell": RADIAL_NODES_PER_SHELL,
        "spatial_bands_per_axis": SPATIAL_BANDS_PER_AXIS,
        "band_size": BAND_SIZE,
        "single_tile_share_guard": SINGLE_TILE_SHARE_GUARD,
        "contiguous_2x2_share_guard": CONTIGUOUS_2X2_SHARE_GUARD,
        "outer_x_pair_share_guard": OUTER_X_PAIR_SHARE_GUARD,
        "outer_to_inner_ratio_guard": OUTER_TO_INNER_RATIO_GUARD,
        "left_right_relative_imbalance_guard": LEFT_RIGHT_RELATIVE_IMBALANCE_GUARD,
        "mode_closure_tolerance": MODE_CLOSURE_TOLERANCE,
        "stage111_run_id": STAGE111_RUN_ID,
    }
    if any(key not in frozen or frozen[key] != value for key, value in overrides.items()):
        raise ValueError(
            "Stage 112 is frozen to the exact completed Stage-111 x-axis association artifact. "
            "It may only localize the already observed x-axis asymmetry/growth association in a "
            "fixed 4x4 equal-area partition. Physics, collision/source treatment, floors, source "
            "relaxation, transport, wall treatment, limiter, velocity quadrature, normalization, "
            "diagnostic window, decision guards, and failed MUSCL parameters may not be retuned."
        )
    if INTERIOR_EXTENT != 56 or INTERIOR_EXTENT % SPATIAL_BANDS_PER_AXIS != 0 or BAND_SIZE != 14:
        raise ValueError("Stage 112 requires the exact 56x56 Stage-111 interior and fixed 14-cell bands")
    if not (0.5 < OUTER_X_PAIR_SHARE_GUARD < 1.0):
        raise ValueError("Stage-112 outer-pair share guard must be stricter than the uniform 0.5 share")
    if OUTER_TO_INNER_RATIO_GUARD <= 1.0:
        raise ValueError("Stage-112 outer/inner ratio guard must remain above unity")
    if not (0.0 <= LEFT_RIGHT_RELATIVE_IMBALANCE_GUARD < 1.0):
        raise ValueError("Stage-112 left/right imbalance guard must remain in [0,1)")


def _load_stage111(
    root: str | Path,
    record_path: str | Path,
) -> tuple[dict[str, object], dict[str, np.ndarray], dict[str, object]]:
    root = Path(root)
    expected = {
        "summary.json": STAGE111_SUMMARY_SHA256,
        "axis_conditioned_asymmetry_maps.npz": STAGE111_MAPS_SHA256,
    }
    for name, digest in expected.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Stage-111 checksum mismatch: {name}")

    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    record = json.loads(Path(record_path).read_text(encoding="utf-8"))
    if summary.get("stage") != 111 or summary.get("decision") != STAGE111_DECISION or summary.get("finite") is not True:
        raise ValueError("Stage-111 artifact does not authorize Stage 112")
    if float(summary.get("max_parent_closure_relative_l2", np.inf)) > MODE_CLOSURE_TOLERANCE:
        raise ValueError("Stage-111 parent closure failed the unchanged tolerance")

    if record.get("stage") != 111 or record.get("decision") != STAGE111_DECISION:
        raise ValueError("Committed Stage-111 record does not authorize Stage 112")
    if record.get("workflow_status") != "completed" or record.get("workflow_conclusion") != "success":
        raise ValueError("Committed Stage-111 record is not a successful completed workflow")
    if int(record.get("workflow_run_id", -1)) != STAGE111_RUN_ID:
        raise ValueError("Committed Stage-111 record has the wrong workflow run")
    if int(record.get("workflow_job_id", -1)) != STAGE111_JOB_ID:
        raise ValueError("Committed Stage-111 record has the wrong workflow job")
    if int(record.get("artifact_id", -1)) != STAGE111_ARTIFACT_ID:
        raise ValueError("Committed Stage-111 record has the wrong artifact")
    if record.get("artifact_sha256") != STAGE111_ARTIFACT_SHA256:
        raise ValueError("Committed Stage-111 artifact digest mismatch")
    if record.get("summary_sha256") != STAGE111_SUMMARY_SHA256:
        raise ValueError("Committed Stage-111 summary digest mismatch")
    if record.get("axis_conditioned_asymmetry_maps_sha256") != STAGE111_MAPS_SHA256:
        raise ValueError("Committed Stage-111 map digest mismatch")
    tests = record.get("tests", {})
    if not isinstance(tests, dict) or tests.get("failed") != 0 or tests.get("passed") != 194:
        raise ValueError("Committed Stage-111 test record is not the exact successful endpoint")

    cfg = summary.get("configuration", {})
    expected_cfg = {
        "grid": list(GRID),
        "kn0": KNUDSEN,
        "cold_hot_ratio": COLD_HOT_RATIO,
        "rule": list(RULE),
        "radial_scale": RADIAL_SCALE,
        "limiter": LIMITER,
        "boundary_slope": BOUNDARY_SLOPE,
        "source_relaxation": SOURCE_RELAXATION,
        "tolerance": TOLERANCE,
        "correction_floor": CORRECTION_FLOOR,
        "diagnostic_steps": DIAGNOSTIC_STEPS,
        "wall_band_cells": WALL_BAND_CELLS,
        "dominant_radial_shell": DOMINANT_RADIAL_SHELL,
        "radial_shell_count": RADIAL_SHELL_COUNT,
        "radial_nodes_per_shell": RADIAL_NODES_PER_SHELL,
    }
    if not isinstance(cfg, dict) or any(cfg.get(k) != v for k, v in expected_cfg.items()):
        raise ValueError("Stage-111 configuration does not match the frozen Stage-112 design")
    for key, value in cfg.items():
        if key.endswith("_retuning") and value is not False:
            raise ValueError("Stage-111 parent reports forbidden retuning")
    if cfg.get("failed_muscl_endpoint_rehabilitated") is not False:
        raise ValueError("Stage 112 cannot consume a rehabilitated MUSCL endpoint")
    if cfg.get("cross_knudsen_extension_permitted") is not False:
        raise ValueError("Stage 112 cannot consume a cross-Knudsen MUSCL extension")

    needed = {
        "phi_x_same_sign_relative_asymmetry",
        "phi_growth_amplitude",
        "psi_x_same_sign_relative_asymmetry",
        "psi_growth_amplitude",
        "common_x_relative_same_sign_asymmetry",
        "joint_growth_amplitude",
    }
    with np.load(root / "axis_conditioned_asymmetry_maps.npz") as data:
        if not needed.issubset(data.files):
            raise ValueError("Stage-111 map payload is missing required x-axis fields")
        arrays = {name: np.asarray(data[name], dtype=np.float64).copy() for name in needed}
    for name, value in arrays.items():
        if value.shape != (INTERIOR_EXTENT, INTERIOR_EXTENT):
            raise ValueError(f"Stage-111 map {name} has the wrong shape")
        if not np.isfinite(value).all() or np.any(value < 0.0):
            raise ValueError(f"Stage-111 map {name} is invalid")
    return summary, arrays, record


def _tile_index() -> np.ndarray:
    tiles = np.empty((INTERIOR_EXTENT, INTERIOR_EXTENT), dtype=np.int64)
    for row in range(SPATIAL_BANDS_PER_AXIS):
        for col in range(SPATIAL_BANDS_PER_AXIS):
            tiles[
                row * BAND_SIZE : (row + 1) * BAND_SIZE,
                col * BAND_SIZE : (col + 1) * BAND_SIZE,
            ] = row * SPATIAL_BANDS_PER_AXIS + col
    return tiles


def _coupled_density(severity: np.ndarray, growth: np.ndarray) -> np.ndarray:
    s = np.asarray(severity, dtype=np.float64)
    g = np.asarray(growth, dtype=np.float64)
    if s.shape != (INTERIOR_EXTENT, INTERIOR_EXTENT) or g.shape != s.shape:
        raise ValueError("Stage-112 severity and growth maps must be the exact 56x56 interior")
    if not np.isfinite(s).all() or not np.isfinite(g).all() or np.any(s < 0.0) or np.any(g < 0.0):
        raise ValueError("Stage-112 severity and growth maps must be finite and nonnegative")
    return s * g


def _best_contiguous_2x2(tile_share: np.ndarray) -> dict[str, object]:
    share = np.asarray(tile_share, dtype=np.float64)
    if share.shape != (SPATIAL_TILE_COUNT,):
        raise ValueError("Stage-112 tile share vector has the wrong shape")
    best: dict[str, object] | None = None
    for row in range(SPATIAL_BANDS_PER_AXIS - 1):
        for col in range(SPATIAL_BANDS_PER_AXIS - 1):
            tiles = [
                row * SPATIAL_BANDS_PER_AXIS + col,
                row * SPATIAL_BANDS_PER_AXIS + col + 1,
                (row + 1) * SPATIAL_BANDS_PER_AXIS + col,
                (row + 1) * SPATIAL_BANDS_PER_AXIS + col + 1,
            ]
            value = float(np.sum(share[tiles]))
            candidate = {
                "top_left_row": row,
                "top_left_column": col,
                "tiles": tiles,
                "share": value,
            }
            if best is None or value > float(best["share"]):
                best = candidate
    assert best is not None
    return best


def _spatial_metrics(severity: np.ndarray, growth: np.ndarray) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    density = _coupled_density(severity, growth)
    total = float(np.sum(density))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Stage-112 x-axis coupled-density surrogate must have positive finite mass")
    density_share = density / total
    tiles = _tile_index()
    tile_share = np.array(
        [float(np.sum(density_share[tiles == k])) for k in range(SPATIAL_TILE_COUNT)],
        dtype=np.float64,
    )
    x_band_share = np.array(
        [float(np.sum(density_share[:, k * BAND_SIZE : (k + 1) * BAND_SIZE])) for k in range(SPATIAL_BANDS_PER_AXIS)],
        dtype=np.float64,
    )
    y_band_share = np.array(
        [float(np.sum(density_share[k * BAND_SIZE : (k + 1) * BAND_SIZE, :])) for k in range(SPATIAL_BANDS_PER_AXIS)],
        dtype=np.float64,
    )
    outer = float(x_band_share[0] + x_band_share[-1])
    inner = float(x_band_share[1] + x_band_share[2])
    imbalance = float(abs(x_band_share[0] - x_band_share[-1]) / max(outer, 1.0e-300))
    qsev = float(np.quantile(severity, 0.75))
    qgrowth = float(np.quantile(growth, 0.75))
    joint_upper = (severity >= qsev) & (growth >= qgrowth)
    upper_share = float(np.sum(density_share[joint_upper]))
    best2x2 = _best_contiguous_2x2(tile_share)
    metrics: dict[str, object] = {
        "total_x_asymmetry_growth_product": total,
        "maximum_single_tile_share": float(np.max(tile_share)),
        "maximum_single_tile_index": int(np.argmax(tile_share)),
        "effective_tile_count": float(1.0 / max(float(np.sum(tile_share * tile_share)), 1.0e-300)),
        "tile_share": tile_share.tolist(),
        "x_band_share": x_band_share.tolist(),
        "y_band_share": y_band_share.tolist(),
        "outer_x_quarter_pair_share": outer,
        "inner_x_half_share": inner,
        "outer_to_inner_share_ratio": float(outer / max(inner, 1.0e-300)),
        "left_right_outer_band_relative_imbalance": imbalance,
        "best_contiguous_2x2": best2x2,
        "upper_severity_quartile_threshold": qsev,
        "upper_growth_quartile_threshold": qgrowth,
        "joint_upper_quartile_cell_count": int(np.count_nonzero(joint_upper)),
        "joint_upper_quartile_coupled_density_share": upper_share,
    }
    return metrics, density, density_share


def stage112_decision(metrics: dict[str, dict[str, object]], finite: bool) -> str:
    if not finite:
        return "stage112_nonfinite_spatial_localization_blocker_without_retuning"

    labels = ("phi", "psi", "common")
    outer_pair = all(
        float(metrics[label]["outer_x_quarter_pair_share"]) >= OUTER_X_PAIR_SHARE_GUARD
        and float(metrics[label]["outer_to_inner_share_ratio"]) >= OUTER_TO_INNER_RATIO_GUARD
        and float(metrics[label]["left_right_outer_band_relative_imbalance"]) <= LEFT_RIGHT_RELATIVE_IMBALANCE_GUARD
        for label in labels
    )
    if outer_pair:
        return "stage112_symmetric_outer_x_quarter_localization_stage113_x_wall_distance_profile_audit"

    common_tile_candidates: list[tuple[float, int]] = []
    for k in range(SPATIAL_TILE_COUNT):
        score = min(float(metrics[label]["tile_share"][k]) for label in labels)
        if score >= SINGLE_TILE_SHARE_GUARD:
            common_tile_candidates.append((score, k))
    if common_tile_candidates:
        _, k = max(common_tile_candidates)
        return f"stage112_common_single_tile_{k}_stage113_local_coordinate_audit"

    best: dict[str, object] | None = None
    for row in range(SPATIAL_BANDS_PER_AXIS - 1):
        for col in range(SPATIAL_BANDS_PER_AXIS - 1):
            tile_ids = [
                row * SPATIAL_BANDS_PER_AXIS + col,
                row * SPATIAL_BANDS_PER_AXIS + col + 1,
                (row + 1) * SPATIAL_BANDS_PER_AXIS + col,
                (row + 1) * SPATIAL_BANDS_PER_AXIS + col + 1,
            ]
            score = min(
                float(sum(float(metrics[label]["tile_share"][k]) for k in tile_ids))
                for label in labels
            )
            candidate = {"row": row, "col": col, "score": score}
            if best is None or score > float(best["score"]):
                best = candidate
    assert best is not None
    if float(best["score"]) >= CONTIGUOUS_2X2_SHARE_GUARD:
        return (
            "stage112_common_contiguous_2x2_"
            f"{best['row']}_{best['col']}_stage113_local_coordinate_audit"
        )
    return "stage112_x_axis_coupling_spatially_diffuse_stage113_x_gradient_lengthscale_audit"


def run_stage112(
    stage111_artifact_dir: str | Path,
    stage111_record_path: str | Path,
    output_dir: str | Path,
    **design: object,
) -> dict[str, object]:
    validate_stage112_design(**design)
    stage111_summary, maps, record = _load_stage111(stage111_artifact_dir, stage111_record_path)

    pairs = {
        "phi": ("phi_x_same_sign_relative_asymmetry", "phi_growth_amplitude"),
        "psi": ("psi_x_same_sign_relative_asymmetry", "psi_growth_amplitude"),
        "common": ("common_x_relative_same_sign_asymmetry", "joint_growth_amplitude"),
    }
    metrics: dict[str, dict[str, object]] = {}
    output: dict[str, np.ndarray] = {"tile_index": _tile_index()}
    finite = True
    for label, (severity_name, growth_name) in pairs.items():
        try:
            block, density, density_share = _spatial_metrics(maps[severity_name], maps[growth_name])
        except ValueError:
            finite = False
            block = {}
            density = np.full((INTERIOR_EXTENT, INTERIOR_EXTENT), np.nan)
            density_share = density.copy()
        metrics[label] = block
        output[f"{label}_x_asymmetry_growth_product"] = density
        output[f"{label}_x_asymmetry_growth_product_share"] = density_share

    decision = stage112_decision(metrics, finite)
    configuration = {
        "grid": list(GRID),
        "kn0": KNUDSEN,
        "cold_hot_ratio": COLD_HOT_RATIO,
        "rule": list(RULE),
        "radial_scale": RADIAL_SCALE,
        "limiter": LIMITER,
        "boundary_slope": BOUNDARY_SLOPE,
        "source_relaxation": SOURCE_RELAXATION,
        "tolerance": TOLERANCE,
        "correction_floor": CORRECTION_FLOOR,
        "diagnostic_steps": DIAGNOSTIC_STEPS,
        "wall_band_cells": WALL_BAND_CELLS,
        "dominant_radial_shell": DOMINANT_RADIAL_SHELL,
        "radial_shell_count": RADIAL_SHELL_COUNT,
        "radial_nodes_per_shell": RADIAL_NODES_PER_SHELL,
        "interior_extent": INTERIOR_EXTENT,
        "spatial_bands_per_axis": SPATIAL_BANDS_PER_AXIS,
        "band_size": BAND_SIZE,
        "spatial_tile_count": SPATIAL_TILE_COUNT,
        "single_tile_share_guard": SINGLE_TILE_SHARE_GUARD,
        "contiguous_2x2_share_guard": CONTIGUOUS_2X2_SHARE_GUARD,
        "outer_x_pair_share_guard": OUTER_X_PAIR_SHARE_GUARD,
        "outer_to_inner_ratio_guard": OUTER_TO_INNER_RATIO_GUARD,
        "left_right_relative_imbalance_guard": LEFT_RIGHT_RELATIVE_IMBALANCE_GUARD,
        "mode_closure_tolerance": MODE_CLOSURE_TOLERANCE,
        "stage111_run_id": STAGE111_RUN_ID,
        "stage111_job_id": STAGE111_JOB_ID,
        "stage111_artifact_id": STAGE111_ARTIFACT_ID,
        "stage111_artifact_sha256": STAGE111_ARTIFACT_SHA256,
        "full_solver_endpoint_rerun": False,
        "physical_parameter_retuning": False,
        "collision_parameter_retuning": False,
        "correction_floor_retuning": False,
        "positivity_floor_retuning": False,
        "source_relaxation_retuning": False,
        "transport_parameter_retuning": False,
        "wall_model_retuning": False,
        "limiter_retuning": False,
        "velocity_quadrature_retuning": False,
        "normalization_retuning": False,
        "failed_muscl_endpoint_rehabilitated": False,
        "one_sided_boundary_slope_promoted": False,
        "cross_knudsen_extension_permitted": False,
        "validation_claim_permitted": False,
        "solver_endpoint_claim_permitted": False,
    }

    if decision == "stage112_symmetric_outer_x_quarter_localization_stage113_x_wall_distance_profile_audit":
        conclusion = (
            "The Stage-111 x-axis asymmetry/growth association is not concentrated in one 14x14 tile. "
            "Instead, the fixed x-axis product surrogate is strongly and symmetrically concentrated in "
            "the two outer x-quarter bands for phi, psi, and the common map. The next preregistered audit "
            "is a finer frozen wall-distance profile that can distinguish a near-sidewall layer from a "
            "broad outer-quarter effect without changing the solver."
        )
    elif "single_tile" in decision or "contiguous_2x2" in decision:
        conclusion = (
            "The frozen Stage-111 x-axis association has a preregistered compact spatial concentration. "
            "Stage 113 should resolve local coordinates inside that support before any mechanistic claim."
        )
    else:
        conclusion = (
            "The frozen Stage-111 x-axis association is spatially diffuse under the preregistered fixed "
            "partition. Stage 113 should audit its x-direction gradient length scale rather than retune "
            "the limiter or any physical parameter."
        )

    summary: dict[str, object] = {
        "stage": 112,
        "description": "Frozen x-axis-specific spatial localization audit of the Stage-111 asymmetry/growth association.",
        "configuration": configuration,
        "stage111_authorization": {
            "decision": stage111_summary["decision"],
            "source_head": record["source_head"],
            "workflow_run_id": STAGE111_RUN_ID,
            "workflow_job_id": STAGE111_JOB_ID,
            "artifact_id": STAGE111_ARTIFACT_ID,
            "tests_passed": record["tests"]["passed"],
            "tests_failed": record["tests"]["failed"],
        },
        "finite": finite,
        "metrics": metrics,
        "decision": decision,
        "scientific_conclusion": conclusion,
        "negative_result_guard": (
            "Stage 112 is an artifact-only spatial organization audit of the frozen Stage-111 x-axis "
            "association. The asymmetry-times-growth product is a localization surrogate, not a causal "
            "sensitivity, adjoint, solver correction, convergence measure, heat-flux improvement, benchmark "
            "improvement, or validation. Stage 111 remains association rather than causal isolation; "
            "Stage 110 remains confounded by same-sign gradient strength; Stage 99 remains a negative "
            "cross-run reproducibility result; Stage 90 remains nonconverged in both reconstruction arms; "
            "Stage 28 remains a failed MUSCL endpoint; the Stage-89 one-sided boundary slope remains "
            "unpromoted. No failed parameter is retuned and no cross-Knudsen MUSCL extension is permitted."
        ),
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "x_axis_spatial_localization_maps.npz", **output)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage111-artifact-dir", required=True)
    parser.add_argument("--stage111-record-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run_stage112(args.stage111_artifact_dir, args.stage111_record_path, args.output_dir)


if __name__ == "__main__":
    main()
