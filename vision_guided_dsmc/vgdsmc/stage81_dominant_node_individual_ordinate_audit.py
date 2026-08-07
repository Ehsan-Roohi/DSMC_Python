from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from . import stage80_dominant_shell_radial_node_angular_attribution_audit as stage80

STAGE67_COMPLETED_ENDPOINT = stage80.STAGE67_COMPLETED_ENDPOINT
STAGE80_COMPLETED_ENDPOINT = {
    "workflow_run_id": 31174110559,
    "workflow_job_id": 92852196029,
    "workflow_conclusion": "success",
    "tests_passed": 235,
    "tests_failed": 0,
    "artifact_id": 8997800347,
    "artifact_size_bytes": 5326606,
    "artifact_sha256": "3d401bee7ddb8101d37b222fe6e42cc5e2c52563f744d48a2c76b7b5062c3319",
    "source_head_sha": "967de551eaa3b48edd8470dacbb6afd4b63cf5cb",
    "summary_sha256": "7bd0d09e7961c06fec5e797ff6010e0c08e4db50019ca4fb3fc2e208587d2baa",
    "maps_sha256": "3dbc3cdf0098c219ac20d7176d37bf0c6a6902d8b7fadcbf1e27af3a99ab62fb",
    "decision": "stage80_single_radial_node_vertical_oblique_stage81_dominant_node_individual_ordinate_audit",
}

GRID = stage80.GRID
KNUDSEN = stage80.KNUDSEN
COLD_HOT_RATIO = stage80.COLD_HOT_RATIO
RULE = stage80.RULE
RADIAL_SCALE = stage80.RADIAL_SCALE
POINT_COUNT = stage80.POINT_COUNT
CHUNK_SIZE = stage80.CHUNK_SIZE
LIMITER = stage80.LIMITER
DOMINANT_MOMENT = stage80.DOMINANT_MOMENT
DOMINANT_RADIAL_SHELL = stage80.DOMINANT_RADIAL_SHELL
DOMINANT_LOCAL_RADIAL_NODE = 1
DOMINANT_GLOBAL_RADIAL_NODE = 21
ANGULAR_BIN_COUNT = stage80.ANGULAR_BIN_COUNT
ANGULAR_BIN_OFFSET_RADIANS = stage80.ANGULAR_BIN_OFFSET_RADIANS
VERTICAL_OBLIQUE_BINS = stage80.VERTICAL_OBLIQUE_BINS
ORDINATE_COUNT = RULE[1]
NOMINAL_ORDINATES_PER_ANGULAR_BIN = ORDINATE_COUNT // ANGULAR_BIN_COUNT
DOMINANT_ORDINATE_SHARE_GUARD = 0.05
TOP_TWELVE_ORDINATE_CONCENTRATION_GUARD = 0.50
VERTICAL_OBLIQUE_CONCENTRATION_GUARD = stage80.VERTICAL_OBLIQUE_CONCENTRATION_GUARD
CLOSURE_GUARD = stage80.CLOSURE_GUARD


def validate_stage81_design(
    grid: tuple[int, int] = GRID,
    kn0: float = KNUDSEN,
    cold_hot_ratio: float = COLD_HOT_RATIO,
    rule: tuple[int, int] = RULE,
    radial_scale: float = RADIAL_SCALE,
    chunk_size: int = CHUNK_SIZE,
    limiter: str = LIMITER,
    dominant_moment: str = DOMINANT_MOMENT,
    dominant_radial_shell: int = DOMINANT_RADIAL_SHELL,
    dominant_local_radial_node: int = DOMINANT_LOCAL_RADIAL_NODE,
    dominant_global_radial_node: int = DOMINANT_GLOBAL_RADIAL_NODE,
    angular_bin_count: int = ANGULAR_BIN_COUNT,
    angular_bin_offset_radians: float = ANGULAR_BIN_OFFSET_RADIANS,
    vertical_oblique_bins: tuple[int, ...] = VERTICAL_OBLIQUE_BINS,
    dominant_ordinate_share_guard: float = DOMINANT_ORDINATE_SHARE_GUARD,
    top_twelve_ordinate_concentration_guard: float = TOP_TWELVE_ORDINATE_CONCENTRATION_GUARD,
    vertical_oblique_concentration_guard: float = VERTICAL_OBLIQUE_CONCENTRATION_GUARD,
) -> None:
    actual = (
        grid, kn0, cold_hot_ratio, rule, radial_scale, chunk_size, limiter,
        dominant_moment, dominant_radial_shell, dominant_local_radial_node,
        dominant_global_radial_node, angular_bin_count, angular_bin_offset_radians,
        vertical_oblique_bins, dominant_ordinate_share_guard,
        top_twelve_ordinate_concentration_guard, vertical_oblique_concentration_guard,
    )
    expected = (
        GRID, KNUDSEN, COLD_HOT_RATIO, RULE, RADIAL_SCALE, CHUNK_SIZE, LIMITER,
        DOMINANT_MOMENT, DOMINANT_RADIAL_SHELL, DOMINANT_LOCAL_RADIAL_NODE,
        DOMINANT_GLOBAL_RADIAL_NODE, ANGULAR_BIN_COUNT, ANGULAR_BIN_OFFSET_RADIANS,
        VERTICAL_OBLIQUE_BINS, DOMINANT_ORDINATE_SHARE_GUARD,
        TOP_TWELVE_ORDINATE_CONCENTRATION_GUARD, VERTICAL_OBLIQUE_CONCENTRATION_GUARD,
    )
    if actual != expected:
        raise ValueError(
            "Stage 81 is frozen to the exact completed Stage-67 distributions and Stage-80 "
            "global radial node 21 endpoint, the unchanged 40x96 radial-scale-2.0 rule, and "
            "the exact retained Stage-80 angular-bin labels; no solver or parameter retuning is permitted."
        )


def _validate_artifact(
    root: str | Path,
    endpoint: dict[str, object],
    files: dict[str, str],
    stage: int,
) -> dict[str, object]:
    return stage80._validate_artifact(root, endpoint, files, stage)


def dominant_node_ordinate_indices(
    vx: np.ndarray,
    vy: np.ndarray,
    node_labels: np.ndarray,
) -> np.ndarray:
    vx = np.asarray(vx, dtype=np.float64)
    vy = np.asarray(vy, dtype=np.float64)
    node_labels = np.asarray(node_labels, dtype=np.int16)
    if vx.shape != (POINT_COUNT,) or vy.shape != (POINT_COUNT,) or node_labels.shape != (POINT_COUNT,):
        raise ValueError("Stage 81 requires the exact 3840-point velocity rule and node labels")
    selected = np.flatnonzero(node_labels == DOMINANT_GLOBAL_RADIAL_NODE)
    if selected.size != ORDINATE_COUNT:
        raise ValueError("Stage 81 requires exactly 96 ordinates on frozen global radial node 21")
    angles = np.mod(np.arctan2(vy[selected], vx[selected]), 2.0 * math.pi)
    ordered_points = selected[np.argsort(angles, kind="stable")]
    labels = np.full(POINT_COUNT, -1, dtype=np.int16)
    labels[ordered_points] = np.arange(ORDINATE_COUNT, dtype=np.int16)
    return labels


def ordinate_to_angular_bin(
    ordinate_labels: np.ndarray,
    angular_labels: np.ndarray,
) -> np.ndarray:
    ordinate_labels = np.asarray(ordinate_labels, dtype=np.int16)
    angular_labels = np.asarray(angular_labels, dtype=np.int16)
    mapping = np.empty(ORDINATE_COUNT, dtype=np.int16)
    for ordinate in range(ORDINATE_COUNT):
        selected = ordinate_labels == ordinate
        if int(np.sum(selected)) != 1:
            raise ValueError("Each Stage-81 ordinate must identify exactly one frozen velocity point")
        mapping[ordinate] = int(angular_labels[selected][0])
    if np.any(mapping < 0) or np.any(mapping >= ANGULAR_BIN_COUNT):
        raise ValueError("Stage-81 retained angular-bin labels are invalid")
    return mapping


def dominant_node_ordinate_face_maps(
    phi: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    weight: np.ndarray,
    chunk_size: int = CHUNK_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if phi.shape != (GRID[0], GRID[1], POINT_COUNT):
        raise ValueError("Stage 81 requires exact 64x64x3840 phi")
    v = stage80.stage79.macroscopic_v(phi, vy, weight, chunk_size)
    v_mid = 0.5 * (v[:, :-1] + v[:, 1:])
    node_labels = stage80.radial_node_indices(vx, vy)
    angular_labels = stage80.stage79.angular_bin_indices(vx, vy)
    ordinate_labels = dominant_node_ordinate_indices(vx, vy, node_labels)
    groups = np.zeros((ORDINATE_COUNT, GRID[0], GRID[1] - 1), dtype=np.float64)
    dx = 1.0 / GRID[1]
    for start in range(0, weight.size, chunk_size):
        stop = min(start + chunk_size, weight.size)
        sl = slice(start, stop)
        delta_phi = stage80.stage79.interior_x_face_flux_difference_chunk(phi[..., sl], vx[sl])
        B = vy[sl][None, None, :] - v_mid[..., None]
        weighted = delta_phi * (0.5 * B * B * B) * weight[sl][None, None, :] / dx
        chunk_ordinates = ordinate_labels[sl]
        for local_index, ordinate in enumerate(chunk_ordinates):
            if ordinate >= 0:
                groups[int(ordinate)] += weighted[..., local_index]
    return groups, node_labels, angular_labels, ordinate_labels


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    return stage80._corr(a, b)


def attribution_metrics(
    face_groups: np.ndarray,
    cell_groups: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    angular_labels: np.ndarray,
    ordinate_labels: np.ndarray,
) -> dict[str, object]:
    face_abs = np.sum(np.abs(face_groups), axis=(1, 2))
    cell_abs = np.sum(np.abs(cell_groups), axis=(1, 2))
    face_total = max(float(np.sum(face_abs)), 1.0e-300)
    cell_total = max(float(np.sum(cell_abs)), 1.0e-300)
    face_share = face_abs / face_total
    cell_share = cell_abs / cell_total
    bin_by_ordinate = ordinate_to_angular_bin(ordinate_labels, angular_labels)
    angles_degrees = np.zeros(ORDINATE_COUNT, dtype=np.float64)
    rows: list[dict[str, object]] = []
    for ordinate in range(ORDINATE_COUNT):
        point = int(np.flatnonzero(ordinate_labels == ordinate)[0])
        angle = float(np.mod(np.arctan2(vy[point], vx[point]), 2.0 * math.pi))
        angles_degrees[ordinate] = math.degrees(angle)
        rows.append({
            "ordinate": ordinate,
            "angle_degrees": float(angles_degrees[ordinate]),
            "angular_bin": int(bin_by_ordinate[ordinate]),
            "face_absolute_share": float(face_share[ordinate]),
            "cell_divergence_absolute_share": float(cell_share[ordinate]),
            "adjacent_x_face_correlation": _corr(face_groups[ordinate, :, :-1], face_groups[ordinate, :, 1:]),
            "face_to_cell_cancellation_ratio": float(cell_abs[ordinate] / max(2.0 * face_abs[ordinate], 1.0e-300)),
            "signed_cell_sum": float(np.sum(cell_groups[ordinate])),
        })
    dominant = int(np.argmax(cell_share))
    top_twelve = float(np.sum(np.sort(cell_share)[-12:]))
    vertical = float(np.sum(cell_share[np.isin(bin_by_ordinate, np.asarray(VERTICAL_OBLIQUE_BINS))]))
    bin_cell_share = np.asarray([
        np.sum(cell_abs[bin_by_ordinate == angular_bin]) / cell_total
        for angular_bin in range(ANGULAR_BIN_COUNT)
    ], dtype=np.float64)
    bin_face_share = np.asarray([
        np.sum(face_abs[bin_by_ordinate == angular_bin]) / face_total
        for angular_bin in range(ANGULAR_BIN_COUNT)
    ], dtype=np.float64)
    bin_counts = np.bincount(bin_by_ordinate, minlength=ANGULAR_BIN_COUNT).astype(np.int64)
    opposite_pairs: list[dict[str, object]] = []
    paired_face_absolute_sum = 0.0
    paired_cell_absolute_sum = 0.0
    for ordinate in range(ORDINATE_COUNT // 2):
        opposite = ordinate + ORDINATE_COUNT // 2
        pair_face = face_groups[ordinate] + face_groups[opposite]
        pair_cell = cell_groups[ordinate] + cell_groups[opposite]
        pair_face_abs = float(np.sum(np.abs(pair_face)))
        pair_cell_abs = float(np.sum(np.abs(pair_cell)))
        paired_face_absolute_sum += pair_face_abs
        paired_cell_absolute_sum += pair_cell_abs
        opposite_pairs.append({
            "ordinate_pair": [ordinate, opposite],
            "face_retention_ratio": float(pair_face_abs / max(face_abs[ordinate] + face_abs[opposite], 1.0e-300)),
            "cell_divergence_retention_ratio": float(pair_cell_abs / max(cell_abs[ordinate] + cell_abs[opposite], 1.0e-300)),
        })
    return {
        "ordinates": rows,
        "ordinate_angles_degrees": angles_degrees.tolist(),
        "ordinate_to_angular_bin": bin_by_ordinate.tolist(),
        "retained_angular_bin_ordinate_counts": bin_counts.tolist(),
        "nominal_ordinates_per_angular_bin": NOMINAL_ORDINATES_PER_ANGULAR_BIN,
        "dominant_ordinate": dominant,
        "dominant_ordinate_angle_degrees": float(angles_degrees[dominant]),
        "dominant_ordinate_angular_bin": int(bin_by_ordinate[dominant]),
        "dominant_ordinate_cell_divergence_share": float(cell_share[dominant]),
        "dominant_ordinate_face_absolute_share": float(face_share[dominant]),
        "dominant_ordinate_adjacent_x_face_correlation": float(rows[dominant]["adjacent_x_face_correlation"]),
        "dominant_ordinate_face_to_cell_cancellation_ratio": float(rows[dominant]["face_to_cell_cancellation_ratio"]),
        "top_twelve_ordinate_cell_divergence_share": top_twelve,
        "vertical_oblique_cell_divergence_share_within_node": vertical,
        "angular_bin_cell_divergence_shares": bin_cell_share.tolist(),
        "angular_bin_face_absolute_shares": bin_face_share.tolist(),
        "opposite_pairs": opposite_pairs,
        "opposite_pair_face_retention_ratio": float(paired_face_absolute_sum / face_total),
        "opposite_pair_cell_divergence_retention_ratio": float(paired_cell_absolute_sum / cell_total),
        "total_ordinate_face_absolute_sum": float(face_total),
        "total_ordinate_cell_divergence_absolute_sum": float(cell_total),
    }


def stage81_decision(finite: bool, closure_closed: bool, metrics: dict[str, object]) -> str:
    if not finite:
        return "stage81_nonfinite_dominant_node_ordinate_attribution_blocker"
    if not closure_closed:
        return "stage81_dominant_node_ordinate_attribution_closure_blocker"
    dominant = float(metrics["dominant_ordinate_cell_divergence_share"])
    top_twelve = float(metrics["top_twelve_ordinate_cell_divergence_share"])
    vertical = float(metrics["vertical_oblique_cell_divergence_share_within_node"])
    if dominant >= DOMINANT_ORDINATE_SHARE_GUARD:
        return "stage81_single_ordinate_concentration_stage82_dominant_ordinate_spatial_cancellation_audit"
    if top_twelve >= TOP_TWELVE_ORDINATE_CONCENTRATION_GUARD:
        return "stage81_ordinate_cluster_stage82_top_twelve_ordinate_cancellation_coherence_audit"
    if vertical >= VERTICAL_OBLIQUE_CONCENTRATION_GUARD:
        return "stage81_vertical_oblique_sector_distributed_stage82_within_sector_angular_coherence_audit"
    return "stage81_dominant_node_angular_attribution_diffuse_stage82_spatial_localization_only_audit"


def run_stage81(
    stage67_artifact_dir: str | Path,
    stage80_artifact_dir: str | Path,
    output_dir: str | Path,
    **design: object,
) -> dict[str, object]:
    validate_stage81_design(**design)
    retained67 = _validate_artifact(
        stage67_artifact_dir,
        STAGE67_COMPLETED_ENDPOINT,
        {"summary.json": "summary_sha256", "converged_full_distributions.npz": "distributions_sha256"},
        67,
    )
    retained80 = _validate_artifact(
        stage80_artifact_dir,
        STAGE80_COMPLETED_ENDPOINT,
        {"summary.json": "summary_sha256", "dominant_shell_radial_node_angular_attribution_maps.npz": "maps_sha256"},
        80,
    )
    metrics80 = retained80["radial_node_angular_metrics"]
    if int(metrics80["dominant_global_radial_node"]) != DOMINANT_GLOBAL_RADIAL_NODE:
        raise ValueError("Completed Stage-80 dominant global radial node does not match frozen Stage-81 node")
    if int(metrics80["dominant_local_radial_node"]) != DOMINANT_LOCAL_RADIAL_NODE:
        raise ValueError("Completed Stage-80 dominant local radial node does not match frozen Stage-81 node")
    with np.load(Path(stage67_artifact_dir) / "converged_full_distributions.npz") as data:
        phi = np.asarray(data["phi"], dtype=np.float64)
        vx = np.asarray(data["vx"], dtype=np.float64)
        vy = np.asarray(data["vy"], dtype=np.float64)
        weight = np.asarray(data["weight"], dtype=np.float64)
    with np.load(Path(stage80_artifact_dir) / "dominant_shell_radial_node_angular_attribution_maps.npz") as data:
        ref_node_face_by_bin = np.asarray(data["face_groups"][DOMINANT_LOCAL_RADIAL_NODE], dtype=np.float64)
        ref_node_cell_by_bin = np.asarray(data["cell_divergence_groups"][DOMINANT_LOCAL_RADIAL_NODE], dtype=np.float64)
        ref_node_labels = np.asarray(data["global_radial_node_labels"], dtype=np.int16)
        ref_angular_labels = np.asarray(data["angular_bin_labels"], dtype=np.int16)
    face_groups, node_labels, angular_labels, ordinate_labels = dominant_node_ordinate_face_maps(phi, vx, vy, weight)
    cell_groups = np.stack(
        [stage80.stage79.divergence_from_interior_faces(face_groups[ordinate]) for ordinate in range(ORDINATE_COUNT)],
        axis=0,
    )
    bin_by_ordinate = ordinate_to_angular_bin(ordinate_labels, angular_labels)
    reconstructed_face_by_bin = np.stack(
        [np.sum(face_groups[bin_by_ordinate == angular_bin], axis=0) for angular_bin in range(ANGULAR_BIN_COUNT)],
        axis=0,
    )
    reconstructed_cell_by_bin = np.stack(
        [np.sum(cell_groups[bin_by_ordinate == angular_bin], axis=0) for angular_bin in range(ANGULAR_BIN_COUNT)],
        axis=0,
    )
    face_delta = reconstructed_face_by_bin - ref_node_face_by_bin
    cell_delta = reconstructed_cell_by_bin - ref_node_cell_by_bin
    face_rel = float(np.linalg.norm(face_delta.ravel()) / max(float(np.linalg.norm(ref_node_face_by_bin.ravel())), 1.0e-300))
    cell_rel = float(np.linalg.norm(cell_delta.ravel()) / max(float(np.linalg.norm(ref_node_cell_by_bin.ravel())), 1.0e-300))
    closure_max = max(float(np.max(np.abs(face_delta))), float(np.max(np.abs(cell_delta))))
    node_labels_match = bool(np.array_equal(node_labels, ref_node_labels))
    angular_labels_match = bool(np.array_equal(angular_labels, ref_angular_labels))
    metrics = attribution_metrics(face_groups, cell_groups, vx, vy, angular_labels, ordinate_labels)
    finite = bool(
        np.all(np.isfinite(face_groups)) and np.all(np.isfinite(cell_groups))
        and np.all(np.isfinite(reconstructed_face_by_bin)) and np.all(np.isfinite(reconstructed_cell_by_bin))
        and np.isfinite(face_rel) and np.isfinite(cell_rel)
    )
    closure_closed = bool(
        max(face_rel, cell_rel) <= CLOSURE_GUARD and node_labels_match and angular_labels_match
    )
    decision = stage81_decision(finite, closure_closed, metrics)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "dominant_node_individual_ordinate_attribution_maps.npz",
        dominant_moment=np.asarray(DOMINANT_MOMENT),
        dominant_radial_shell=np.asarray(DOMINANT_RADIAL_SHELL),
        dominant_local_radial_node=np.asarray(DOMINANT_LOCAL_RADIAL_NODE),
        dominant_global_radial_node=np.asarray(DOMINANT_GLOBAL_RADIAL_NODE),
        global_radial_node_labels=node_labels,
        angular_bin_labels=angular_labels,
        ordinate_labels=ordinate_labels,
        ordinate_to_angular_bin=bin_by_ordinate,
        face_groups=face_groups,
        cell_divergence_groups=cell_groups,
        reconstructed_node_face_by_bin=reconstructed_face_by_bin,
        reconstructed_node_cell_by_bin=reconstructed_cell_by_bin,
        retained_stage80_node_face_by_bin=ref_node_face_by_bin,
        retained_stage80_node_cell_by_bin=ref_node_cell_by_bin,
    )
    summary = {
        "stage": 81,
        "description": "Frozen individual-angular-ordinate attribution of the exact Stage-80 dominant global radial node 21.",
        "configuration": {
            "grid": list(GRID),
            "kn0": KNUDSEN,
            "cold_hot_ratio": COLD_HOT_RATIO,
            "radial_nodes": RULE[0],
            "angular_nodes": RULE[1],
            "point_count": POINT_COUNT,
            "radial_scale": RADIAL_SCALE,
            "limiter": LIMITER,
            "dominant_moment": DOMINANT_MOMENT,
            "dominant_radial_shell": DOMINANT_RADIAL_SHELL,
            "dominant_local_radial_node": DOMINANT_LOCAL_RADIAL_NODE,
            "dominant_global_radial_node": DOMINANT_GLOBAL_RADIAL_NODE,
            "ordinate_count": ORDINATE_COUNT,
            "angular_bin_count": ANGULAR_BIN_COUNT,
            "angular_bin_offset_radians": ANGULAR_BIN_OFFSET_RADIANS,
            "nominal_ordinates_per_angular_bin": NOMINAL_ORDINATES_PER_ANGULAR_BIN,
            "retained_stage80_angular_bin_labels_preserved_exactly": True,
            "vertical_oblique_bins": list(VERTICAL_OBLIQUE_BINS),
            "dominant_ordinate_share_guard": DOMINANT_ORDINATE_SHARE_GUARD,
            "top_twelve_ordinate_concentration_guard": TOP_TWELVE_ORDINATE_CONCENTRATION_GUARD,
            "vertical_oblique_concentration_guard": VERTICAL_OBLIQUE_CONCENTRATION_GUARD,
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
        },
        "retained_stage67_decision": retained67["decision"],
        "retained_stage80_decision": retained80["decision"],
        "dominant_node_reconstruction_closure": {
            "maximum_absolute_error": closure_max,
            "face_relative_l2_error": face_rel,
            "cell_relative_l2_error": cell_rel,
            "radial_node_labels_exact_match": node_labels_match,
            "angular_bin_labels_exact_match": angular_labels_match,
            "within_guard": closure_closed,
        },
        "ordinate_metrics": metrics,
        "finite": finite,
        "closure_closed": closure_closed,
        "decision": decision,
        "positive_findings": [
            "The exact Stage-80 dominant global radial node 21 is resolved into all 96 already-fixed angular ordinates without changing the velocity rule.",
            "The individual-ordinate contributions are regrouped using the exact retained Stage-80 angular-bin labels and checked against every completed Stage-80 node-21 bin map.",
        ],
        "negative_findings": [
            "Individual-ordinate prominence is a frozen residual-structure diagnostic, not an adjoint sensitivity and not evidence that pruning, moving, or reweighting velocity ordinates would improve q_av.",
            "The exact inherited Stage-80 bin labels are preserved even if floating-point boundary classification makes their ordinate counts differ from the nominal twelve-per-sector geometry; Stage 81 does not rebucket or retune them.",
            "The failed Stage-28 MUSCL endpoint remains unrecovered and is not extended across Knudsen number.",
            "No physical, collision, correction-floor, source-relaxation, transport, wall, normalization, or velocity-quadrature parameter is retuned.",
        ],
        "scientifically_justified_next_scope": (
            "If a single frozen ordinate carries at least 5% of the node-level cell-divergence magnitude, audit only its spatial cancellation/localization. "
            "If the top twelve ordinates carry at least 50%, audit their cancellation/coherence as a fixed cluster. Otherwise, if the inherited vertical-oblique sectors still carry at least 70%, resolve within-sector angular coherence without any solver experiment."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage67-artifact-dir", required=True)
    parser.add_argument("--stage80-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run_stage81(args.stage67_artifact_dir, args.stage80_artifact_dir, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
