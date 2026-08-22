from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

STAGE81_COMPLETED_ENDPOINT = {
    "workflow_run_id": 31189580849,
    "workflow_job_id": 92902686472,
    "workflow_conclusion": "success",
    "tests_passed": 247,
    "tests_failed": 0,
    "artifact_id": 9003488020,
    "artifact_size_bytes": 6806978,
    "artifact_sha256": "0acbac541a41e82254151d0f4b5917fd9ab2559c1a592bbbaeab8c93778a3272",
    "source_head_sha": "27072909ee7d76aa74801d143a05fea5c0b93906",
    "summary_sha256": "0f5c1772622303519f31c4faf447592150de97849667aee2f4b2eb5b1f15a5d4",
    "maps_sha256": "b9d60daed1e4df0d011e9885b332654939911182cdc732f984c2082b56122adb",
    "decision": "stage81_vertical_oblique_sector_distributed_stage82_within_sector_angular_coherence_audit",
}

GRID = (64, 64)
KNUDSEN = 10.0
COLD_HOT_RATIO = 0.1
RADIAL_NODES = 40
ANGULAR_NODES = 96
POINT_COUNT = RADIAL_NODES * ANGULAR_NODES
RADIAL_SCALE = 2.0
LIMITER = "minmod"
DOMINANT_MOMENT = "transverse_kinetic"
DOMINANT_RADIAL_SHELL = 2
DOMINANT_LOCAL_RADIAL_NODE = 1
DOMINANT_GLOBAL_RADIAL_NODE = 21
VERTICAL_OBLIQUE_BINS = (1, 2, 5, 6)
OPPOSITE_SECTOR_PAIRS = ((1, 5), (2, 6))
WEIGHTED_ADJACENT_COHERENCE_GUARD = 0.90
SECTOR_INTERNAL_RETENTION_GUARD = 0.75
CLOSURE_GUARD = 1.0e-10


def validate_stage82_design(
    grid: tuple[int, int] = GRID,
    kn0: float = KNUDSEN,
    cold_hot_ratio: float = COLD_HOT_RATIO,
    radial_nodes: int = RADIAL_NODES,
    angular_nodes: int = ANGULAR_NODES,
    radial_scale: float = RADIAL_SCALE,
    limiter: str = LIMITER,
    dominant_moment: str = DOMINANT_MOMENT,
    dominant_radial_shell: int = DOMINANT_RADIAL_SHELL,
    dominant_local_radial_node: int = DOMINANT_LOCAL_RADIAL_NODE,
    dominant_global_radial_node: int = DOMINANT_GLOBAL_RADIAL_NODE,
    vertical_oblique_bins: tuple[int, ...] = VERTICAL_OBLIQUE_BINS,
    opposite_sector_pairs: tuple[tuple[int, int], ...] = OPPOSITE_SECTOR_PAIRS,
    weighted_adjacent_coherence_guard: float = WEIGHTED_ADJACENT_COHERENCE_GUARD,
    sector_internal_retention_guard: float = SECTOR_INTERNAL_RETENTION_GUARD,
) -> None:
    actual = (
        grid, kn0, cold_hot_ratio, radial_nodes, angular_nodes, radial_scale, limiter,
        dominant_moment, dominant_radial_shell, dominant_local_radial_node,
        dominant_global_radial_node, vertical_oblique_bins, opposite_sector_pairs,
        weighted_adjacent_coherence_guard, sector_internal_retention_guard,
    )
    expected = (
        GRID, KNUDSEN, COLD_HOT_RATIO, RADIAL_NODES, ANGULAR_NODES, RADIAL_SCALE, LIMITER,
        DOMINANT_MOMENT, DOMINANT_RADIAL_SHELL, DOMINANT_LOCAL_RADIAL_NODE,
        DOMINANT_GLOBAL_RADIAL_NODE, VERTICAL_OBLIQUE_BINS, OPPOSITE_SECTOR_PAIRS,
        WEIGHTED_ADJACENT_COHERENCE_GUARD, SECTOR_INTERNAL_RETENTION_GUARD,
    )
    if actual != expected:
        raise ValueError(
            "Stage 82 is frozen to the exact completed Stage-81 node-21 individual-ordinate artifact, "
            "the inherited vertical-oblique bins, and fixed diagnostic guards; no solver or parameter retuning is permitted."
        )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _validate_stage81_artifact(root: str | Path) -> dict[str, object]:
    root = Path(root)
    summary_path = root / "summary.json"
    maps_path = root / "dominant_node_individual_ordinate_attribution_maps.npz"
    if not summary_path.exists() or not maps_path.exists():
        raise FileNotFoundError("Stage 82 requires the exact completed Stage-81 summary and maps artifact")
    if _sha256(summary_path) != STAGE81_COMPLETED_ENDPOINT["summary_sha256"]:
        raise ValueError("Stage-81 summary checksum does not match the registered completed endpoint")
    if _sha256(maps_path) != STAGE81_COMPLETED_ENDPOINT["maps_sha256"]:
        raise ValueError("Stage-81 maps checksum does not match the registered completed endpoint")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("stage") != 81 or summary.get("decision") != STAGE81_COMPLETED_ENDPOINT["decision"]:
        raise ValueError("Stage-81 scientific decision does not match the registered completed endpoint")
    if summary.get("finite") is not True or summary.get("closure_closed") is not True:
        raise ValueError("Stage 82 requires a finite, closure-closed Stage-81 endpoint")
    cfg = summary.get("configuration", {})
    expected = {
        "grid": list(GRID),
        "kn0": KNUDSEN,
        "cold_hot_ratio": COLD_HOT_RATIO,
        "radial_nodes": RADIAL_NODES,
        "angular_nodes": ANGULAR_NODES,
        "point_count": POINT_COUNT,
        "radial_scale": RADIAL_SCALE,
        "limiter": LIMITER,
        "dominant_moment": DOMINANT_MOMENT,
        "dominant_radial_shell": DOMINANT_RADIAL_SHELL,
        "dominant_local_radial_node": DOMINANT_LOCAL_RADIAL_NODE,
        "dominant_global_radial_node": DOMINANT_GLOBAL_RADIAL_NODE,
        "vertical_oblique_bins": list(VERTICAL_OBLIQUE_BINS),
        "solver_rerun_count": 0,
        "failed_muscl_endpoint_rehabilitated": False,
        "cross_knudsen_extension_permitted": False,
    }
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f"Stage-81 frozen configuration mismatch for {key}")
    for key, value in cfg.items():
        if key.endswith("_retuning") and value is not False:
            raise ValueError(f"Stage-81 endpoint unexpectedly retuned {key}")
    return summary


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError("Correlation inputs must have identical shapes")
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 1.0e-300:
        return 1.0 if np.allclose(a, b, rtol=0.0, atol=1.0e-300) else 0.0
    return float(np.dot(x, y) / denom)


def _absolute_magnitude(groups: np.ndarray) -> np.ndarray:
    groups = np.asarray(groups, dtype=np.float64)
    return np.sum(np.abs(groups), axis=tuple(range(1, groups.ndim)))


def _retention_ratio(groups: np.ndarray) -> float:
    groups = np.asarray(groups, dtype=np.float64)
    denominator = float(np.sum(np.abs(groups)))
    if denominator <= 1.0e-300:
        return 0.0
    return float(np.sum(np.abs(np.sum(groups, axis=0))) / denominator)


def _weighted_adjacent_coherence(groups: np.ndarray, ordered_indices: np.ndarray) -> tuple[float, list[float], list[float]]:
    groups = np.asarray(groups, dtype=np.float64)
    ordered_indices = np.asarray(ordered_indices, dtype=np.int64)
    if ordered_indices.size < 2:
        return 1.0, [], []
    magnitudes = _absolute_magnitude(groups[ordered_indices])
    correlations: list[float] = []
    weights: list[float] = []
    for local in range(ordered_indices.size - 1):
        i = int(ordered_indices[local])
        j = int(ordered_indices[local + 1])
        correlations.append(_corr(groups[i], groups[j]))
        weights.append(float(np.sqrt(magnitudes[local] * magnitudes[local + 1])))
    total_weight = float(np.sum(weights))
    weighted = 0.0 if total_weight <= 1.0e-300 else float(np.dot(correlations, weights) / total_weight)
    return weighted, correlations, weights


def _central_half(ordered_indices: np.ndarray) -> np.ndarray:
    ordered_indices = np.asarray(ordered_indices, dtype=np.int64)
    count = max(1, ordered_indices.size // 2)
    start = (ordered_indices.size - count) // 2
    return ordered_indices[start:start + count]


def within_sector_metrics(
    face_groups: np.ndarray,
    cell_groups: np.ndarray,
    ordinate_to_bin: np.ndarray,
    angles_degrees: np.ndarray,
) -> dict[str, object]:
    face_groups = np.asarray(face_groups, dtype=np.float64)
    cell_groups = np.asarray(cell_groups, dtype=np.float64)
    ordinate_to_bin = np.asarray(ordinate_to_bin, dtype=np.int16)
    angles_degrees = np.asarray(angles_degrees, dtype=np.float64)
    if face_groups.shape != (ANGULAR_NODES, GRID[0], GRID[1] - 1):
        raise ValueError("Stage 82 requires exact Stage-81 96x64x63 face maps")
    if cell_groups.shape != (ANGULAR_NODES, GRID[0], GRID[1]):
        raise ValueError("Stage 82 requires exact Stage-81 96x64x64 cell maps")
    if ordinate_to_bin.shape != (ANGULAR_NODES,) or angles_degrees.shape != (ANGULAR_NODES,):
        raise ValueError("Stage 82 requires exact Stage-81 ordinate labels and angles")

    face_abs_all = _absolute_magnitude(face_groups)
    cell_abs_all = _absolute_magnitude(cell_groups)
    total_face = max(float(np.sum(face_abs_all)), 1.0e-300)
    total_cell = max(float(np.sum(cell_abs_all)), 1.0e-300)
    sector_rows: list[dict[str, object]] = []
    sector_face: list[np.ndarray] = []
    sector_cell: list[np.ndarray] = []
    sector_indices: list[np.ndarray] = []

    for angular_bin in VERTICAL_OBLIQUE_BINS:
        indices = np.flatnonzero(ordinate_to_bin == angular_bin)
        indices = indices[np.argsort(angles_degrees[indices], kind="stable")]
        if indices.size == 0:
            raise ValueError(f"Stage-81 retained vertical-oblique bin {angular_bin} is empty")
        sector_indices.append(indices)
        local_face = face_groups[indices]
        local_cell = cell_groups[indices]
        aggregate_face = np.sum(local_face, axis=0)
        aggregate_cell = np.sum(local_cell, axis=0)
        sector_face.append(aggregate_face)
        sector_cell.append(aggregate_cell)
        face_weighted, face_adjacent_corrs, face_pair_weights = _weighted_adjacent_coherence(face_groups, indices)
        cell_weighted, cell_adjacent_corrs, cell_pair_weights = _weighted_adjacent_coherence(cell_groups, indices)
        core = _central_half(indices)
        local_cell_abs = cell_abs_all[indices]
        local_face_abs = face_abs_all[indices]
        cell_core_share = float(np.sum(cell_abs_all[core]) / max(float(np.sum(local_cell_abs)), 1.0e-300))
        face_core_share = float(np.sum(face_abs_all[core]) / max(float(np.sum(local_face_abs)), 1.0e-300))
        cell_weights = local_cell_abs / max(float(np.sum(local_cell_abs)), 1.0e-300)
        angular_centroid = float(np.sum(cell_weights * angles_degrees[indices]))
        angular_std = float(np.sqrt(np.sum(cell_weights * (angles_degrees[indices] - angular_centroid) ** 2)))
        sector_rows.append({
            "angular_bin": int(angular_bin),
            "ordinate_count": int(indices.size),
            "ordinate_indices": indices.tolist(),
            "angle_min_degrees": float(angles_degrees[indices[0]]),
            "angle_max_degrees": float(angles_degrees[indices[-1]]),
            "cell_weighted_angular_centroid_degrees": angular_centroid,
            "cell_weighted_angular_std_degrees": angular_std,
            "node_cell_divergence_absolute_share": float(np.sum(local_cell_abs) / total_cell),
            "node_face_absolute_share": float(np.sum(local_face_abs) / total_face),
            "cell_internal_retention_ratio": _retention_ratio(local_cell),
            "face_internal_retention_ratio": _retention_ratio(local_face),
            "cell_weighted_adjacent_angular_coherence": cell_weighted,
            "face_weighted_adjacent_angular_coherence": face_weighted,
            "cell_adjacent_angular_correlations": [float(v) for v in cell_adjacent_corrs],
            "face_adjacent_angular_correlations": [float(v) for v in face_adjacent_corrs],
            "cell_adjacent_pair_weights": [float(v) for v in cell_pair_weights],
            "face_adjacent_pair_weights": [float(v) for v in face_pair_weights],
            "cell_central_half_individual_magnitude_share": cell_core_share,
            "face_central_half_individual_magnitude_share": face_core_share,
            "aggregate_adjacent_x_face_correlation": _corr(aggregate_face[:, :-1], aggregate_face[:, 1:]),
        })

    sector_face_array = np.stack(sector_face, axis=0)
    sector_cell_array = np.stack(sector_cell, axis=0)
    selected = np.isin(ordinate_to_bin, np.asarray(VERTICAL_OBLIQUE_BINS, dtype=np.int16))
    pair_rows: list[dict[str, object]] = []
    bin_to_local = {int(b): i for i, b in enumerate(VERTICAL_OBLIQUE_BINS)}
    for left_bin, right_bin in OPPOSITE_SECTOR_PAIRS:
        left = bin_to_local[left_bin]
        right = bin_to_local[right_bin]
        face_den = float(np.sum(np.abs(sector_face_array[left])) + np.sum(np.abs(sector_face_array[right])))
        cell_den = float(np.sum(np.abs(sector_cell_array[left])) + np.sum(np.abs(sector_cell_array[right])))
        face_pair = sector_face_array[left] + sector_face_array[right]
        cell_pair = sector_cell_array[left] + sector_cell_array[right]
        pair_rows.append({
            "angular_bin_pair": [left_bin, right_bin],
            "face_retention_ratio": float(np.sum(np.abs(face_pair)) / max(face_den, 1.0e-300)),
            "cell_divergence_retention_ratio": float(np.sum(np.abs(cell_pair)) / max(cell_den, 1.0e-300)),
            "face_map_correlation": _corr(sector_face_array[left], sector_face_array[right]),
            "cell_map_correlation": _corr(sector_cell_array[left], sector_cell_array[right]),
        })

    max_count = max(indices.size for indices in sector_indices)
    padded = np.full((len(sector_indices), max_count), -1, dtype=np.int16)
    counts = np.zeros(len(sector_indices), dtype=np.int16)
    for i, indices in enumerate(sector_indices):
        padded[i, :indices.size] = indices
        counts[i] = indices.size

    return {
        "sectors": sector_rows,
        "opposite_sector_pairs": pair_rows,
        "vertical_oblique_cell_divergence_share_within_node": float(np.sum(cell_abs_all[selected]) / total_cell),
        "vertical_oblique_face_absolute_share_within_node": float(np.sum(face_abs_all[selected]) / total_face),
        "minimum_sector_cell_weighted_adjacent_coherence": float(min(row["cell_weighted_adjacent_angular_coherence"] for row in sector_rows)),
        "minimum_sector_cell_internal_retention_ratio": float(min(row["cell_internal_retention_ratio"] for row in sector_rows)),
        "sector_face_groups": sector_face_array,
        "sector_cell_groups": sector_cell_array,
        "sector_ordinate_indices_padded": padded,
        "sector_ordinate_counts": counts,
    }


def stage82_decision(finite: bool, closure_closed: bool, metrics: dict[str, object]) -> str:
    if not finite:
        return "stage82_nonfinite_within_sector_coherence_blocker"
    if not closure_closed:
        return "stage82_vertical_sector_reconstruction_closure_blocker"
    coherence = float(metrics["minimum_sector_cell_weighted_adjacent_coherence"])
    retention = float(metrics["minimum_sector_cell_internal_retention_ratio"])
    if coherence >= WEIGHTED_ADJACENT_COHERENCE_GUARD and retention >= SECTOR_INTERNAL_RETENTION_GUARD:
        return "stage82_smooth_retained_vertical_oblique_sectors_stage83_opposite_sector_spatial_cancellation_audit"
    if coherence >= WEIGHTED_ADJACENT_COHERENCE_GUARD:
        return "stage82_smooth_but_internally_canceling_sectors_stage83_within_sector_signed_cancellation_localization_audit"
    return "stage82_mixed_within_sector_angular_coherence_stage83_angular_transition_localization_audit"


def run_stage82(stage81_artifact_dir: str | Path, output_dir: str | Path, **design: object) -> dict[str, object]:
    validate_stage82_design(**design)
    retained81 = _validate_stage81_artifact(stage81_artifact_dir)
    root = Path(stage81_artifact_dir)
    with np.load(root / "dominant_node_individual_ordinate_attribution_maps.npz") as data:
        face_groups = np.asarray(data["face_groups"], dtype=np.float64)
        cell_groups = np.asarray(data["cell_divergence_groups"], dtype=np.float64)
        ordinate_to_bin = np.asarray(data["ordinate_to_angular_bin"], dtype=np.int16)
        retained_face_by_bin = np.asarray(data["reconstructed_node_face_by_bin"], dtype=np.float64)
        retained_cell_by_bin = np.asarray(data["reconstructed_node_cell_by_bin"], dtype=np.float64)
    angles = np.asarray(retained81["ordinate_metrics"]["ordinate_angles_degrees"], dtype=np.float64)
    metrics = within_sector_metrics(face_groups, cell_groups, ordinate_to_bin, angles)
    sector_face = np.asarray(metrics.pop("sector_face_groups"), dtype=np.float64)
    sector_cell = np.asarray(metrics.pop("sector_cell_groups"), dtype=np.float64)
    padded = np.asarray(metrics.pop("sector_ordinate_indices_padded"), dtype=np.int16)
    counts = np.asarray(metrics.pop("sector_ordinate_counts"), dtype=np.int16)
    selected_bins = np.asarray(VERTICAL_OBLIQUE_BINS, dtype=np.int16)
    face_ref = retained_face_by_bin[selected_bins]
    cell_ref = retained_cell_by_bin[selected_bins]
    face_delta = sector_face - face_ref
    cell_delta = sector_cell - cell_ref
    face_rel = float(np.linalg.norm(face_delta.ravel()) / max(float(np.linalg.norm(face_ref.ravel())), 1.0e-300))
    cell_rel = float(np.linalg.norm(cell_delta.ravel()) / max(float(np.linalg.norm(cell_ref.ravel())), 1.0e-300))
    max_abs = max(float(np.max(np.abs(face_delta))), float(np.max(np.abs(cell_delta))))
    finite = bool(
        np.all(np.isfinite(sector_face)) and np.all(np.isfinite(sector_cell))
        and np.isfinite(face_rel) and np.isfinite(cell_rel)
    )
    closure_closed = bool(max(face_rel, cell_rel) <= CLOSURE_GUARD)
    decision = stage82_decision(finite, closure_closed, metrics)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "within_sector_angular_coherence_maps.npz",
        vertical_oblique_bins=selected_bins,
        sector_face_groups=sector_face,
        sector_cell_divergence_groups=sector_cell,
        sector_ordinate_indices_padded=padded,
        sector_ordinate_counts=counts,
        ordinate_to_angular_bin=ordinate_to_bin,
        ordinate_angles_degrees=angles,
        retained_stage81_sector_face_groups=face_ref,
        retained_stage81_sector_cell_divergence_groups=cell_ref,
    )
    summary = {
        "stage": 82,
        "description": "Frozen within-sector angular-coherence audit of the Stage-81 vertical-oblique contribution on global radial node 21.",
        "configuration": {
            "grid": list(GRID),
            "kn0": KNUDSEN,
            "cold_hot_ratio": COLD_HOT_RATIO,
            "radial_nodes": RADIAL_NODES,
            "angular_nodes": ANGULAR_NODES,
            "point_count": POINT_COUNT,
            "radial_scale": RADIAL_SCALE,
            "limiter": LIMITER,
            "dominant_moment": DOMINANT_MOMENT,
            "dominant_radial_shell": DOMINANT_RADIAL_SHELL,
            "dominant_local_radial_node": DOMINANT_LOCAL_RADIAL_NODE,
            "dominant_global_radial_node": DOMINANT_GLOBAL_RADIAL_NODE,
            "vertical_oblique_bins": list(VERTICAL_OBLIQUE_BINS),
            "opposite_sector_pairs": [list(pair) for pair in OPPOSITE_SECTOR_PAIRS],
            "weighted_adjacent_coherence_guard": WEIGHTED_ADJACENT_COHERENCE_GUARD,
            "sector_internal_retention_guard": SECTOR_INTERNAL_RETENTION_GUARD,
            "solver_rerun_count": 0,
            "physical_parameter_retuning": False,
            "collision_parameter_retuning": False,
            "correction_floor_retuning": False,
            "source_relaxation_retuning": False,
            "transport_parameter_retuning": False,
            "wall_model_retuning": False,
            "normalization_retuning": False,
            "velocity_quadrature_retuning": False,
            "failed_muscl_endpoint_rehabilitated": False,
            "cross_knudsen_extension_permitted": False,
            "validation_claim_permitted": False,
        },
        "retained_stage81_decision": retained81["decision"],
        "vertical_sector_reconstruction_closure": {
            "maximum_absolute_error": max_abs,
            "face_relative_l2_error": face_rel,
            "cell_relative_l2_error": cell_rel,
            "within_guard": closure_closed,
        },
        "within_sector_metrics": metrics,
        "finite": finite,
        "closure_closed": closure_closed,
        "decision": decision,
        "positive_findings": [
            "The four inherited Stage-81 vertical-oblique sector maps are reconstructed exactly from their already-fixed individual ordinates without rebucketing the angular rule.",
            "Magnitude-weighted adjacent-ordinate coherence and within-sector signed-retention are quantified separately so negligible boundary ordinates cannot dominate the coherence diagnosis.",
        ],
        "negative_findings": [
            "Within-sector angular coherence is a frozen residual-structure diagnostic, not an adjoint sensitivity and not evidence that changing, pruning, moving, or reweighting velocity ordinates would improve q_av.",
            "A high angular correlation does not constitute physical validation, and a low correlation by itself would not establish numerical error.",
            "The failed Stage-28 MUSCL endpoint remains unrecovered and is not extended across Knudsen number.",
            "No physical, collision, correction-floor, source-relaxation, transport, wall, normalization, or velocity-quadrature parameter is retuned, and no cavity solver is rerun.",
        ],
        "scientifically_justified_next_scope": (
            "If every inherited vertical-oblique sector is materially coherent and retains at least 75% of its within-sector cell-divergence magnitude, audit only the spatial cancellation and localization of the two fixed opposite-sector pairs (1,5) and (2,6). "
            "If sectors are coherent but internally cancellation-dominated, localize that signed cancellation instead. Otherwise audit only the angular transition locations; no solver experiment or parameter retuning is authorized by this diagnostic."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage81-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run_stage82(args.stage81_artifact_dir, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
