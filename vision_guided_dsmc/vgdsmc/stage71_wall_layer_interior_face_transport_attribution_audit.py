from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

STAGE69_COMPLETED_ENDPOINT = {
    "workflow_run_id": 31027231271,
    "workflow_job_id": 92378691275,
    "workflow_conclusion": "success",
    "tests_passed": 107,
    "tests_failed": 0,
    "test_duration_seconds": 0.74,
    "artifact_id": 8942482076,
    "artifact_size_bytes": 587209,
    "artifact_sha256": "8040126435a35726fe995360e5c5c3a807f6bfa6ed27d139405a53f1a5775e78",
    "source_head_sha": "45a757bf0c3ed25c27f1aacdb9273d8816f69f42",
    "summary_sha256": "512e8cc0ed5cee547c08968b965a8c45c74352ea7ffe9e21d12d15417512bac8",
    "moment_maps_sha256": "aacceab9597ba7f4607911f1e983fdb4d9f310529ce18c5ddbeb4e08ad3607be",
    "decision": "stage69_monotone_but_wall_dominated_slow_full_scaling_stage70_independent_wall_face_flux_discretization_audit",
}
STAGE70_COMPLETED_ENDPOINT = {
    "workflow_run_id": 31037106991,
    "workflow_job_id": 92411824859,
    "workflow_conclusion": "success",
    "tests_passed": 125,
    "tests_failed": 0,
    "test_duration_seconds": 0.69,
    "artifact_id": 8950151688,
    "artifact_size_bytes": 22482,
    "artifact_sha256": "02fab9c3b577e2faf31d775da73d374a340daf74421fb0f899f7db968b9b5030",
    "source_head_sha": "996b693f76143b9f95c14e3f684f3e78d45c12c4",
    "summary_sha256": "5147c55c535b7b3f8902c90f2ffd7df0f360c47155c119aced26b9e6653f6dbb",
    "profiles_sha256": "1fec5ab239a0eed1333943ba3552d70498e07d269990af3929e1819311666afd",
    "decision": "stage70_wall_face_heat_flux_difference_below_materiality_stage71_wall_layer_interior_face_transport_attribution_audit",
}
GRIDS = (16, 32, 64)
FINE_GRID = 64
KNUDSEN = 10.0
COLD_HOT_RATIO = 0.1
RULE = (40, 96)
RADIAL_SCALE = 2.0
POINT_COUNT = 3840
WALL_BAND_PHYSICAL_FRACTION = 1.0 / 16.0
MATERIAL_WALL_FACE_RATIO = 0.10
WALL_DOMINANCE_FRACTION = 0.50
OUTER_TWO_LAYER_CONCENTRATION = 0.75
SIDE_STRIP_DOMINANCE = 0.50
PARTITION_GUARD = 1.0e-12
ENDPOINT_GUARD = 1.0e-12
REGION_CODES = {"interior": 0, "bottom": 1, "top": 2, "left": 3, "right": 4, "corners": 5}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_stage71_design(
    grids=GRIDS, fine_grid=FINE_GRID, kn0=KNUDSEN,
    cold_hot_ratio=COLD_HOT_RATIO, rule=RULE, radial_scale=RADIAL_SCALE,
    wall_band_physical_fraction=WALL_BAND_PHYSICAL_FRACTION,
    material_wall_face_ratio=MATERIAL_WALL_FACE_RATIO,
    wall_dominance_fraction=WALL_DOMINANCE_FRACTION,
    outer_two_layer_concentration=OUTER_TWO_LAYER_CONCENTRATION,
    side_strip_dominance=SIDE_STRIP_DOMINANCE,
) -> None:
    actual = (grids, fine_grid, kn0, cold_hot_ratio, rule, radial_scale,
              wall_band_physical_fraction, material_wall_face_ratio,
              wall_dominance_fraction, outer_two_layer_concentration,
              side_strip_dominance)
    expected = (GRIDS, FINE_GRID, KNUDSEN, COLD_HOT_RATIO, RULE, RADIAL_SCALE,
                WALL_BAND_PHYSICAL_FRACTION, MATERIAL_WALL_FACE_RATIO,
                WALL_DOMINANCE_FRACTION, OUTER_TWO_LAYER_CONCENTRATION,
                SIDE_STRIP_DOMINANCE)
    if actual != expected:
        raise ValueError("Stage 71 is frozen; no physical or numerical retuning is permitted.")


def _validate_artifact(root, endpoint, files, stage):
    root = Path(root)
    for name, key in files.items():
        if not (root / name).is_file() or sha256_file(root / name) != endpoint[key]:
            raise ValueError(f"Stage {stage} artifact checksum mismatch: {name}")
    summary = json.loads((root / "summary.json").read_text())
    if summary.get("stage") != stage or summary.get("decision") != endpoint["decision"]:
        raise ValueError(f"Stage {stage} completed endpoint mismatch")
    return summary


def _validate_stage69_artifact(root):
    return _validate_artifact(root, STAGE69_COMPLETED_ENDPOINT,
        {"summary.json": "summary_sha256", "grid_transfer_transport_moment_maps.npz": "moment_maps_sha256"}, 69)


def _validate_stage70_artifact(root):
    return _validate_artifact(root, STAGE70_COMPLETED_ENDPOINT,
        {"summary.json": "summary_sha256", "wall_face_flux_profiles.npz": "profiles_sha256"}, 70)


def wall_distance(grid: int) -> np.ndarray:
    yy, xx = np.indices((grid, grid))
    return np.minimum.reduce((yy, grid - 1 - yy, xx, grid - 1 - xx))


def wall_band_layers(grid: int) -> int:
    layers = int(round(grid * WALL_BAND_PHYSICAL_FRACTION))
    if layers < 1 or 2 * layers >= grid:
        raise ValueError("Physical wall band leaves no interior")
    return layers


def attribution_masks(grid: int, layers: int | None = None):
    layers = wall_band_layers(grid) if layers is None else int(layers)
    yy, xx = np.indices((grid, grid))
    bottom = (yy < layers) & (xx >= layers) & (xx < grid - layers)
    top = (yy >= grid - layers) & (xx >= layers) & (xx < grid - layers)
    left = (xx < layers) & (yy >= layers) & (yy < grid - layers)
    right = (xx >= grid - layers) & (yy >= layers) & (yy < grid - layers)
    wall = wall_distance(grid) < layers
    corners = wall & ~(bottom | top | left | right)
    masks = {"bottom": bottom, "top": top, "left": left, "right": right,
             "corners": corners, "interior": ~wall}
    if not np.all(sum(mask.astype(np.int8) for mask in masks.values()) == 1):
        raise ValueError("Attribution partition is not disjoint and complete")
    return masks


def region_code_map(grid: int, masks) -> np.ndarray:
    codes = np.full((grid, grid), -1, dtype=np.int8)
    for name, mask in masks.items():
        codes[mask] = REGION_CODES[name]
    if np.any(codes < 0):
        raise ValueError("Incomplete region coding")
    return codes


def rms(values) -> float:
    values = np.asarray(values, float)
    return float(np.sqrt(np.mean(values * values)))


def signed_statistics(values, mask, global_abs, wall_abs, is_wall):
    selected = np.asarray(values, float)[mask]
    absolute_sum = float(np.sum(np.abs(selected)))
    signed_sum = float(np.sum(selected))
    return {
        "cell_count": int(selected.size),
        "absolute_sum": absolute_sum,
        "absolute_global_share": absolute_sum / max(global_abs, 1e-300),
        "absolute_wall_band_share": absolute_sum / max(wall_abs, 1e-300) if is_wall else 0.0,
        "signed_sum": signed_sum,
        "signed_to_absolute_ratio": signed_sum / max(absolute_sum, 1e-300),
        "mean": float(np.mean(selected)), "rms": rms(selected),
        "maximum_absolute": float(np.max(np.abs(selected))),
        "positive_cell_fraction": float(np.mean(selected > 0)),
        "negative_cell_fraction": float(np.mean(selected < 0)),
    }


def stage71_decision(finite, provenance_consistent, partition_closed,
                     stage69_wall_share_reproduced, stage70_wall_face_submaterial,
                     wall_band_dominant, outer_two_layers_concentrated,
                     side_strips_dominant):
    if not finite: return "stage71_nonfinite_attribution_blocker"
    if not provenance_consistent: return "stage71_completed_endpoint_reproduction_blocker"
    if not partition_closed: return "stage71_region_partition_blocker"
    if not stage69_wall_share_reproduced: return "stage71_stage69_wall_share_reproduction_blocker"
    if not stage70_wall_face_submaterial: return "stage71_material_physical_wall_face_difference_blocker"
    if not wall_band_dominant: return "stage71_wall_band_not_dominant_stop_before_solver_response"
    if outer_two_layers_concentrated and side_strips_dominant:
        return "stage71_near_wall_side_strip_interior_face_dominance_stage72_directional_transport_component_audit"
    if outer_two_layers_concentrated:
        return "stage71_near_wall_interior_face_dominance_stage72_directional_transport_component_audit"
    return "stage71_distributed_wall_layer_defect_stage72_facewise_flux_reconstruction_audit"


def run_stage71(stage69_artifact_dir, stage70_artifact_dir, output_dir, **design):
    validate_stage71_design(**design)
    s69 = _validate_stage69_artifact(stage69_artifact_dir)
    s70 = _validate_stage70_artifact(stage70_artifact_dir)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    required = {f"grid{g}_{c}_{m}" for g in GRIDS
                for c in ("retained_first_order", "independent_second_order", "difference")
                for m in ("mass", "momentum_x", "momentum_y", "energy", "qx", "qy")}
    results, saved = {}, {}
    finite = partition_closed = True
    with np.load(Path(stage69_artifact_dir) / "grid_transfer_transport_moment_maps.npz") as data:
        if set(data.files) != required: raise ValueError("Stage 69 moment-map contract mismatch")
        for grid in GRIDS:
            diff = np.asarray(data[f"grid{grid}_difference_qy"], float)
            retained = np.asarray(data[f"grid{grid}_retained_first_order_qy"], float)
            finite &= bool(np.all(np.isfinite(diff)) and np.all(np.isfinite(retained)))
            layers = wall_band_layers(grid); dist = wall_distance(grid)
            masks = attribution_masks(grid, layers); wall = dist < layers
            global_abs = float(np.sum(abs(diff))); wall_abs = float(np.sum(abs(diff[wall])))
            regions = {name: signed_statistics(diff, mask, global_abs, wall_abs, name != "interior")
                       for name, mask in masks.items()}
            abs_err = abs(sum(r["absolute_sum"] for r in regions.values()) - global_abs)
            signed_err = abs(sum(r["signed_sum"] for r in regions.values()) - float(np.sum(diff)))
            closed = abs_err <= PARTITION_GUARD * max(global_abs, 1.0) and signed_err <= PARTITION_GUARD * max(global_abs, 1.0)
            partition_closed &= closed
            layers_data = [dict(layer=k, distance_interval=[k / grid, (k + 1) / grid],
                **signed_statistics(diff, dist == k, global_abs, wall_abs, True)) for k in range(layers)]
            outer_count = min(2, layers)
            outer_share = float(np.sum(abs(diff[dist < outer_count])) / max(wall_abs, 1e-300))
            vertical = float(regions["left"]["absolute_wall_band_share"] + regions["right"]["absolute_wall_band_share"])
            horizontal = float(regions["bottom"]["absolute_wall_band_share"] + regions["top"]["absolute_wall_band_share"])
            signed_sum = float(np.sum(diff))
            results[str(grid)] = {
                "grid": [grid, grid], "cell_width": 1 / grid, "wall_band_layers": layers,
                "wall_band_physical_fraction": WALL_BAND_PHYSICAL_FRACTION,
                "regions": regions, "layer_profile": layers_data,
                "wall_band_absolute_share": wall_abs / max(global_abs, 1e-300),
                "outer_two_available_layers": outer_count,
                "outer_two_layer_absolute_wall_band_share": outer_share,
                "vertical_side_strip_absolute_wall_band_share": vertical,
                "horizontal_strip_absolute_wall_band_share": horizontal,
                "corner_absolute_wall_band_share": float(regions["corners"]["absolute_wall_band_share"]),
                "global_signed_sum": signed_sum, "global_absolute_sum": global_abs,
                "global_signed_cancellation_ratio": abs(signed_sum) / max(global_abs, 1e-300),
                "difference_rms": rms(diff), "retained_first_order_rms": rms(retained),
                "difference_to_retained_rms_ratio": rms(diff) / max(rms(retained), 1e-300),
                "partition": {"absolute_error": abs_err, "signed_error": signed_err, "within_guard": closed},
            }
            saved[f"grid{grid}_difference_qy"] = diff
            saved[f"grid{grid}_wall_distance_cells"] = dist.astype(np.int16)
            saved[f"grid{grid}_region_code"] = region_code_map(grid, masks)
    fine = results[str(FINE_GRID)]
    s69_share = float(s69["normal_heat_flux_scaling"]["fine_grid_wall_band_absolute_share"])
    share_err = abs(float(fine["wall_band_absolute_share"]) - s69_share)
    s70_ratio = float(s70["wall_face_scaling"]["fine_grid_absolute_relative_qav_difference"])
    flags = {
        "stage69_wall_share_reproduced": share_err <= ENDPOINT_GUARD,
        "stage70_wall_face_submaterial": s70_ratio < MATERIAL_WALL_FACE_RATIO,
        "wall_band_dominant": float(fine["wall_band_absolute_share"]) > WALL_DOMINANCE_FRACTION,
        "outer_two_layers_concentrated": float(fine["outer_two_layer_absolute_wall_band_share"]) >= OUTER_TWO_LAYER_CONCENTRATION,
        "side_strips_dominant": float(fine["vertical_side_strip_absolute_wall_band_share"]) > SIDE_STRIP_DOMINANCE,
    }
    provenance = bool(s69["provenance_consistent"] and s69["restriction_conservative"]
        and s70["provenance_consistent"] and s70["mass_flux_closure"]["within_guard"]
        and s70["qav_reproduction"]["within_guard"]
        and s70["stage68_boundary_face_structure"]["stage68_second_order_boundary_outgoing_equals_retained_cell_center"])
    decision = stage71_decision(finite, provenance, partition_closed, **flags)
    np.savez_compressed(out / "wall_layer_attribution_maps.npz", **saved)
    config = {
        "grids": list(GRIDS), "fine_grid": FINE_GRID, "kn0": KNUDSEN,
        "cold_hot_ratio": COLD_HOT_RATIO, "radial_nodes": RULE[0], "angular_nodes": RULE[1],
        "point_count": POINT_COUNT, "radial_scale": RADIAL_SCALE,
        "wall_band_physical_fraction": WALL_BAND_PHYSICAL_FRACTION,
        "material_wall_face_ratio": MATERIAL_WALL_FACE_RATIO,
        "wall_dominance_fraction": WALL_DOMINANCE_FRACTION,
        "outer_two_layer_concentration": OUTER_TWO_LAYER_CONCENTRATION,
        "side_strip_dominance": SIDE_STRIP_DOMINANCE, "solver_rerun_count": 0,
        "physical_parameter_retuning": False, "collision_parameter_retuning": False,
        "correction_floor_retuning": False, "source_relaxation_retuning": False,
        "transport_parameter_retuning": False, "wall_model_retuning": False,
        "normalization_retuning": False, "velocity_quadrature_retuning": False,
        "bounded_wall_face_arm_adopted": False, "failed_muscl_endpoint_rehabilitated": False,
        "cross_knudsen_extension_permitted": False,
    }
    attribution = {
        "wall_band_absolute_share": float(fine["wall_band_absolute_share"]),
        "stage69_wall_band_absolute_share": s69_share,
        "stage69_wall_share_absolute_error": share_err,
        "outer_two_layer_absolute_wall_band_share": float(fine["outer_two_layer_absolute_wall_band_share"]),
        "vertical_side_strip_absolute_wall_band_share": float(fine["vertical_side_strip_absolute_wall_band_share"]),
        "horizontal_strip_absolute_wall_band_share": float(fine["horizontal_strip_absolute_wall_band_share"]),
        "corner_absolute_wall_band_share": float(fine["corner_absolute_wall_band_share"]),
        "global_signed_cancellation_ratio": float(fine["global_signed_cancellation_ratio"]),
        "stage70_fine_wall_face_qav_difference_ratio": s70_ratio, **flags,
    }
    summary = {
        "stage": 71,
        "description": "Frozen wall-layer attribution of the Stage-69 q_y transport-operator difference, conditioned on Stage 70; no cavity solve.",
        "configuration": config, "retained_stage69_endpoint": STAGE69_COMPLETED_ENDPOINT,
        "retained_stage69_decision": s69["decision"], "retained_stage70_endpoint": STAGE70_COMPLETED_ENDPOINT,
        "retained_stage70_decision": s70["decision"], "grid_results": results,
        "fine_grid_attribution": attribution, "finite": bool(finite),
        "partition_closed": bool(partition_closed), "provenance_consistent": provenance,
        "decision": decision,
        "positive_findings": [
            "Disjoint wall-strip, corner, and interior partitions close on every frozen grid.",
            "The Stage-69 fine-grid wall-band share is independently reproduced.",
            "Stage 70 keeps the physical wall-face q_av difference below the inherited 10% guard."
        ],
        "negative_findings": [
            "Frozen residual localization does not predict a converged q_av response.",
            "A side-strip q_y contribution does not identify whether x- or y-direction transport is responsible.",
            "The bounded wall-face arm remains unadopted and Stage-28 MUSCL remains unrecovered.",
            "No external validation or cross-Knudsen extension is supported."
        ],
        "interpretation_guard": "This is operator-residual localization, not solver sensitivity, causality, or validation.",
        "scientifically_justified_next_scope": "Decompose the exact frozen q_y difference into x- and y-direction interior-face components before any response or solver experiment."
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage69-artifact-dir", required=True)
    p.add_argument("--stage70-artifact-dir", required=True)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()
    print(json.dumps(run_stage71(a.stage69_artifact_dir, a.stage70_artifact_dir, a.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
