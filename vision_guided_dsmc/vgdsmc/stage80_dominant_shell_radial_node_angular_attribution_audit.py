from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import stage79_dominant_moment_radial_angular_gradient_audit as stage79

STAGE67_COMPLETED_ENDPOINT = stage79.STAGE67_COMPLETED_ENDPOINT
STAGE79_COMPLETED_ENDPOINT = {
    "workflow_run_id": 31159895420,
    "workflow_job_id": 92807602344,
    "workflow_conclusion": "success",
    "tests_passed": 224,
    "tests_failed": 0,
    "artifact_id": 8991295336,
    "artifact_size_bytes": 1532270,
    "artifact_sha256": "346bb4a6dd0485c7650ec8f51fba1d063df170f41e72e8ab46537972d98ab13a",
    "source_head_sha": "d043f9dfd0c960fcf430afac9c0a0c26085f8ba4",
    "summary_sha256": "9896aef5ec092e8deb6922fda36065f3dd1f54f452a2d41844d96d0a523930bf",
    "maps_sha256": "b05cb4d125434c7f1c4b6e5bbc2805915d86467d53b310910286e535ae566d5a",
    "decision": "stage79_dominant_radial_shell_vertical_oblique_stage80_dominant_shell_radial_node_angular_attribution_audit",
}

GRID = stage79.GRID
KNUDSEN = stage79.KNUDSEN
COLD_HOT_RATIO = stage79.COLD_HOT_RATIO
RULE = stage79.RULE
RADIAL_SCALE = stage79.RADIAL_SCALE
POINT_COUNT = stage79.POINT_COUNT
CHUNK_SIZE = stage79.CHUNK_SIZE
LIMITER = stage79.LIMITER
DOMINANT_MOMENT = stage79.DOMINANT_MOMENT
DOMINANT_RADIAL_SHELL = 2
RADIAL_NODES_PER_SHELL = stage79.RADIAL_NODES_PER_SHELL
DOMINANT_SHELL_GLOBAL_NODE_START = DOMINANT_RADIAL_SHELL * RADIAL_NODES_PER_SHELL
DOMINANT_SHELL_GLOBAL_NODE_STOP = DOMINANT_SHELL_GLOBAL_NODE_START + RADIAL_NODES_PER_SHELL
ANGULAR_BIN_COUNT = stage79.ANGULAR_BIN_COUNT
ANGULAR_BIN_OFFSET_RADIANS = stage79.ANGULAR_BIN_OFFSET_RADIANS
VERTICAL_OBLIQUE_BINS = stage79.VERTICAL_OBLIQUE_BINS
DOMINANT_RADIAL_NODE_SHARE_GUARD = 0.20
TOP_THREE_RADIAL_NODE_CONCENTRATION_GUARD = 0.60
VERTICAL_OBLIQUE_CONCENTRATION_GUARD = stage79.VERTICAL_OBLIQUE_CONCENTRATION_GUARD
CLOSURE_GUARD = stage79.CLOSURE_GUARD


def validate_stage80_design(
    grid: tuple[int, int] = GRID,
    kn0: float = KNUDSEN,
    cold_hot_ratio: float = COLD_HOT_RATIO,
    rule: tuple[int, int] = RULE,
    radial_scale: float = RADIAL_SCALE,
    chunk_size: int = CHUNK_SIZE,
    limiter: str = LIMITER,
    dominant_moment: str = DOMINANT_MOMENT,
    dominant_radial_shell: int = DOMINANT_RADIAL_SHELL,
    radial_nodes_per_shell: int = RADIAL_NODES_PER_SHELL,
    angular_bin_count: int = ANGULAR_BIN_COUNT,
    angular_bin_offset_radians: float = ANGULAR_BIN_OFFSET_RADIANS,
    vertical_oblique_bins: tuple[int, ...] = VERTICAL_OBLIQUE_BINS,
    dominant_radial_node_share_guard: float = DOMINANT_RADIAL_NODE_SHARE_GUARD,
    top_three_radial_node_concentration_guard: float = TOP_THREE_RADIAL_NODE_CONCENTRATION_GUARD,
    vertical_oblique_concentration_guard: float = VERTICAL_OBLIQUE_CONCENTRATION_GUARD,
) -> None:
    actual = (
        grid, kn0, cold_hot_ratio, rule, radial_scale, chunk_size, limiter,
        dominant_moment, dominant_radial_shell, radial_nodes_per_shell,
        angular_bin_count, angular_bin_offset_radians, vertical_oblique_bins,
        dominant_radial_node_share_guard, top_three_radial_node_concentration_guard,
        vertical_oblique_concentration_guard,
    )
    expected = (
        GRID, KNUDSEN, COLD_HOT_RATIO, RULE, RADIAL_SCALE, CHUNK_SIZE, LIMITER,
        DOMINANT_MOMENT, DOMINANT_RADIAL_SHELL, RADIAL_NODES_PER_SHELL,
        ANGULAR_BIN_COUNT, ANGULAR_BIN_OFFSET_RADIANS, VERTICAL_OBLIQUE_BINS,
        DOMINANT_RADIAL_NODE_SHARE_GUARD, TOP_THREE_RADIAL_NODE_CONCENTRATION_GUARD,
        VERTICAL_OBLIQUE_CONCENTRATION_GUARD,
    )
    if actual != expected:
        raise ValueError(
            "Stage 80 is frozen to the exact completed Stage-67 distributions and Stage-79 "
            "dominant shell-2 endpoint, the unchanged 40x96 radial-scale-2.0 rule, ten fixed "
            "constituent radial nodes, and eight zero-offset angular bins; no solver or parameter retuning is permitted."
        )


def _validate_artifact(root: str | Path, endpoint: dict[str, object], files: dict[str, str], stage: int) -> dict[str, object]:
    root = Path(root)
    for name, checksum_key in files.items():
        path = root / name
        if not path.is_file() or stage79.sha256_file(path) != str(endpoint[checksum_key]):
            raise ValueError(f"Completed Stage-{stage} artifact checksum mismatch: {name}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("stage") != stage or summary.get("decision") != endpoint["decision"]:
        raise ValueError(f"Stage-{stage} completed endpoint mismatch")
    return summary


def radial_node_indices(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    speed = np.hypot(np.asarray(vx, dtype=np.float64), np.asarray(vy, dtype=np.float64))
    if speed.ndim != 1 or speed.size != POINT_COUNT:
        raise ValueError("Stage 80 requires the exact 3840-point velocity rule")
    order = np.argsort(speed, kind="stable")
    labels = np.empty(speed.size, dtype=np.int16)
    labels[order] = np.repeat(np.arange(RULE[0], dtype=np.int16), RULE[1])
    return labels


def radial_shell_indices_from_nodes(node_labels: np.ndarray) -> np.ndarray:
    node_labels = np.asarray(node_labels, dtype=np.int16)
    if node_labels.shape != (POINT_COUNT,):
        raise ValueError("Stage 80 requires exact radial-node labels")
    return (node_labels // RADIAL_NODES_PER_SHELL).astype(np.int8)


def dominant_shell_node_angular_face_maps(
    phi: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    weight: np.ndarray,
    chunk_size: int = CHUNK_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if phi.shape != (GRID[0], GRID[1], POINT_COUNT):
        raise ValueError("Stage 80 requires exact 64x64x3840 phi")
    v = stage79.macroscopic_v(phi, vy, weight, chunk_size)
    v_mid = 0.5 * (v[:, :-1] + v[:, 1:])
    node_labels = radial_node_indices(vx, vy)
    shell_labels = radial_shell_indices_from_nodes(node_labels)
    angular_labels = stage79.angular_bin_indices(vx, vy)
    groups = np.zeros((RADIAL_NODES_PER_SHELL, ANGULAR_BIN_COUNT, GRID[0], GRID[1] - 1), dtype=np.float64)
    dx = 1.0 / GRID[1]
    for start in range(0, weight.size, chunk_size):
        stop = min(start + chunk_size, weight.size)
        sl = slice(start, stop)
        delta_phi = stage79.interior_x_face_flux_difference_chunk(phi[..., sl], vx[sl])
        B = vy[sl][None, None, :] - v_mid[..., None]
        weighted = delta_phi * (0.5 * B * B * B) * weight[sl][None, None, :] / dx
        chunk_nodes = node_labels[sl]
        chunk_bins = angular_labels[sl]
        for local_node in range(RADIAL_NODES_PER_SHELL):
            global_node = DOMINANT_SHELL_GLOBAL_NODE_START + local_node
            for angular_bin in range(ANGULAR_BIN_COUNT):
                selected = (chunk_nodes == global_node) & (chunk_bins == angular_bin)
                if np.any(selected):
                    groups[local_node, angular_bin] += np.sum(weighted[..., selected], axis=-1)
    return groups, node_labels, shell_labels, angular_labels


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.size == 0 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def attribution_metrics(face_groups: np.ndarray, cell_groups: np.ndarray, vx: np.ndarray, vy: np.ndarray, node_labels: np.ndarray) -> dict[str, object]:
    face_abs = np.sum(np.abs(face_groups), axis=(2, 3))
    cell_abs = np.sum(np.abs(cell_groups), axis=(2, 3))
    face_total = max(float(np.sum(face_abs)), 1.0e-300)
    cell_total = max(float(np.sum(cell_abs)), 1.0e-300)
    joint_face_share = face_abs / face_total
    joint_cell_share = cell_abs / cell_total
    radial_face = np.sum(face_abs, axis=1)
    radial_cell = np.sum(cell_abs, axis=1)
    angular_face = np.sum(face_abs, axis=0)
    angular_cell = np.sum(cell_abs, axis=0)
    radial_face_share = radial_face / face_total
    radial_cell_share = radial_cell / cell_total
    angular_face_share = angular_face / face_total
    angular_cell_share = angular_cell / cell_total
    dominant_local_node = int(np.argmax(radial_cell_share))
    dominant_global_node = DOMINANT_SHELL_GLOBAL_NODE_START + dominant_local_node
    dominant_joint = np.unravel_index(int(np.argmax(joint_cell_share)), joint_cell_share.shape)
    speed = np.hypot(vx, vy)
    node_metadata = []
    for local_node in range(RADIAL_NODES_PER_SHELL):
        global_node = DOMINANT_SHELL_GLOBAL_NODE_START + local_node
        selected = node_labels == global_node
        node_face = np.sum(face_groups[local_node], axis=0)
        node_cell = np.sum(cell_groups[local_node], axis=0)
        node_metadata.append({
            "local_node": local_node,
            "global_node": global_node,
            "velocity_point_count": int(np.sum(selected)),
            "minimum_speed": float(np.min(speed[selected])),
            "maximum_speed": float(np.max(speed[selected])),
            "mean_speed": float(np.mean(speed[selected])),
            "face_absolute_share": float(radial_face_share[local_node]),
            "cell_divergence_absolute_share": float(radial_cell_share[local_node]),
            "adjacent_x_face_correlation": _corr(node_face[:, :-1], node_face[:, 1:]),
            "face_to_cell_cancellation_ratio": float(radial_cell[local_node] / max(2.0 * radial_face[local_node], 1.0e-300)),
            "signed_cell_sum": float(np.sum(node_cell)),
        })
    angular_metadata = []
    width_degrees = 360.0 / ANGULAR_BIN_COUNT
    for angular_bin in range(ANGULAR_BIN_COUNT):
        angular_metadata.append({
            "bin": angular_bin,
            "start_degrees": angular_bin * width_degrees,
            "end_degrees": (angular_bin + 1) * width_degrees,
            "face_absolute_share": float(angular_face_share[angular_bin]),
            "cell_divergence_absolute_share": float(angular_cell_share[angular_bin]),
        })
    top_three = float(np.sum(np.sort(radial_cell_share)[-3:]))
    vertical_oblique = float(np.sum(angular_cell_share[list(VERTICAL_OBLIQUE_BINS)]))
    dominant_node_face = np.sum(face_groups[dominant_local_node], axis=0)
    return {
        "joint_face_absolute_share": joint_face_share.tolist(),
        "joint_cell_divergence_absolute_share": joint_cell_share.tolist(),
        "radial_nodes": node_metadata,
        "angular_bins": angular_metadata,
        "dominant_local_radial_node": dominant_local_node,
        "dominant_global_radial_node": dominant_global_node,
        "dominant_radial_node_cell_divergence_share": float(radial_cell_share[dominant_local_node]),
        "dominant_radial_node_face_absolute_share": float(radial_face_share[dominant_local_node]),
        "dominant_radial_node_adjacent_x_face_correlation": _corr(dominant_node_face[:, :-1], dominant_node_face[:, 1:]),
        "top_three_radial_node_cell_divergence_share": top_three,
        "vertical_oblique_cell_divergence_share_within_shell": vertical_oblique,
        "dominant_joint_group": [int(dominant_joint[0]), int(dominant_joint[1])],
        "dominant_joint_group_cell_divergence_share_within_shell": float(joint_cell_share[dominant_joint]),
        "total_group_face_absolute_sum": float(face_total),
        "total_group_cell_divergence_absolute_sum": float(cell_total),
    }


def stage80_decision(finite: bool, closure_closed: bool, metrics: dict[str, object]) -> str:
    if not finite:
        return "stage80_nonfinite_dominant_shell_node_angular_attribution_blocker"
    if not closure_closed:
        return "stage80_dominant_shell_node_angular_attribution_closure_blocker"
    dominant = float(metrics["dominant_radial_node_cell_divergence_share"])
    top_three = float(metrics["top_three_radial_node_cell_divergence_share"])
    vertical = float(metrics["vertical_oblique_cell_divergence_share_within_shell"])
    if dominant >= DOMINANT_RADIAL_NODE_SHARE_GUARD and vertical >= VERTICAL_OBLIQUE_CONCENTRATION_GUARD:
        return "stage80_single_radial_node_vertical_oblique_stage81_dominant_node_individual_ordinate_audit"
    if top_three >= TOP_THREE_RADIAL_NODE_CONCENTRATION_GUARD:
        return "stage80_radial_node_cluster_stage81_top_three_node_cancellation_coherence_audit"
    if vertical >= VERTICAL_OBLIQUE_CONCENTRATION_GUARD:
        return "stage80_vertical_oblique_radially_distributed_stage81_vertical_oblique_node_pair_audit"
    return "stage80_within_shell_attribution_diffuse_stage81_dominant_shell_spatial_localization_audit"


def run_stage80(stage67_artifact_dir: str | Path, stage79_artifact_dir: str | Path, output_dir: str | Path, **design: object) -> dict[str, object]:
    validate_stage80_design(**design)
    retained67 = _validate_artifact(
        stage67_artifact_dir,
        STAGE67_COMPLETED_ENDPOINT,
        {"summary.json": "summary_sha256", "converged_full_distributions.npz": "distributions_sha256"},
        67,
    )
    retained79 = _validate_artifact(
        stage79_artifact_dir,
        STAGE79_COMPLETED_ENDPOINT,
        {"summary.json": "summary_sha256", "dominant_moment_radial_angular_gradient_maps.npz": "maps_sha256"},
        79,
    )
    if int(retained79["radial_angular_metrics"]["dominant_radial_shell"]) != DOMINANT_RADIAL_SHELL:
        raise ValueError("Completed Stage-79 dominant radial shell does not match frozen Stage-80 shell")
    with np.load(Path(stage67_artifact_dir) / "converged_full_distributions.npz") as data:
        phi = np.asarray(data["phi"], dtype=np.float64)
        vx = np.asarray(data["vx"], dtype=np.float64)
        vy = np.asarray(data["vy"], dtype=np.float64)
        weight = np.asarray(data["weight"], dtype=np.float64)
    with np.load(Path(stage79_artifact_dir) / "dominant_moment_radial_angular_gradient_maps.npz") as data:
        ref_shell_face_by_bin = np.asarray(data["face_groups"][DOMINANT_RADIAL_SHELL], dtype=np.float64)
        ref_shell_cell_by_bin = np.asarray(data["cell_divergence_groups"][DOMINANT_RADIAL_SHELL], dtype=np.float64)
        ref_shell_labels = np.asarray(data["radial_shell_labels"], dtype=np.int8)
        ref_angular_labels = np.asarray(data["angular_bin_labels"], dtype=np.int16)
    face_groups, node_labels, shell_labels, angular_labels = dominant_shell_node_angular_face_maps(phi, vx, vy, weight)
    cell_groups = np.stack(
        [[stage79.divergence_from_interior_faces(face_groups[node, angular_bin]) for angular_bin in range(ANGULAR_BIN_COUNT)] for node in range(RADIAL_NODES_PER_SHELL)],
        axis=0,
    )
    reconstructed_face_by_bin = np.sum(face_groups, axis=0)
    reconstructed_cell_by_bin = np.sum(cell_groups, axis=0)
    face_delta = reconstructed_face_by_bin - ref_shell_face_by_bin
    cell_delta = reconstructed_cell_by_bin - ref_shell_cell_by_bin
    face_rel = float(np.linalg.norm(face_delta.ravel()) / max(float(np.linalg.norm(ref_shell_face_by_bin.ravel())), 1.0e-300))
    cell_rel = float(np.linalg.norm(cell_delta.ravel()) / max(float(np.linalg.norm(ref_shell_cell_by_bin.ravel())), 1.0e-300))
    closure_max = max(float(np.max(np.abs(face_delta))), float(np.max(np.abs(cell_delta))))
    label_shell_match = bool(np.array_equal(shell_labels, ref_shell_labels))
    label_angular_match = bool(np.array_equal(angular_labels, ref_angular_labels))
    metrics = attribution_metrics(face_groups, cell_groups, vx, vy, node_labels)
    finite = bool(
        np.all(np.isfinite(face_groups)) and np.all(np.isfinite(cell_groups))
        and np.all(np.isfinite(reconstructed_face_by_bin)) and np.all(np.isfinite(reconstructed_cell_by_bin))
        and np.isfinite(face_rel) and np.isfinite(cell_rel)
    )
    closure_closed = bool(max(face_rel, cell_rel) <= CLOSURE_GUARD and label_shell_match and label_angular_match)
    decision = stage80_decision(finite, closure_closed, metrics)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "dominant_shell_radial_node_angular_attribution_maps.npz",
        dominant_moment=np.asarray(DOMINANT_MOMENT),
        dominant_radial_shell=np.asarray(DOMINANT_RADIAL_SHELL),
        global_radial_node_labels=node_labels,
        radial_shell_labels=shell_labels,
        angular_bin_labels=angular_labels,
        face_groups=face_groups,
        cell_divergence_groups=cell_groups,
        reconstructed_shell_face_by_bin=reconstructed_face_by_bin,
        reconstructed_shell_cell_by_bin=reconstructed_cell_by_bin,
        retained_stage79_shell_face_by_bin=ref_shell_face_by_bin,
        retained_stage79_shell_cell_by_bin=ref_shell_cell_by_bin,
    )
    summary = {
        "stage": 80,
        "description": "Frozen radial-node/angular-sector attribution inside the exact Stage-79 dominant transverse-kinetic radial shell.",
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
            "dominant_shell_global_radial_nodes": list(range(DOMINANT_SHELL_GLOBAL_NODE_START, DOMINANT_SHELL_GLOBAL_NODE_STOP)),
            "radial_nodes_per_shell": RADIAL_NODES_PER_SHELL,
            "angular_bin_count": ANGULAR_BIN_COUNT,
            "angular_bin_offset_radians": ANGULAR_BIN_OFFSET_RADIANS,
            "vertical_oblique_bins": list(VERTICAL_OBLIQUE_BINS),
            "dominant_radial_node_share_guard": DOMINANT_RADIAL_NODE_SHARE_GUARD,
            "top_three_radial_node_concentration_guard": TOP_THREE_RADIAL_NODE_CONCENTRATION_GUARD,
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
        "retained_stage79_decision": retained79["decision"],
        "dominant_shell_reconstruction_closure": {
            "maximum_absolute_error": closure_max,
            "face_relative_l2_error": face_rel,
            "cell_relative_l2_error": cell_rel,
            "radial_shell_labels_exact_match": label_shell_match,
            "angular_bin_labels_exact_match": label_angular_match,
            "within_guard": closure_closed,
        },
        "radial_node_angular_metrics": metrics,
        "finite": finite,
        "closure_closed": closure_closed,
        "decision": decision,
        "positive_findings": [
            "The exact Stage-79 dominant shell is resolved into its ten already-fixed radial nodes and the unchanged eight angular sectors.",
            "The node/angular reconstruction is checked bin-by-bin against the exact completed Stage-79 shell-2 face and cell-divergence maps without rerunning the cavity solver.",
        ],
        "negative_findings": [
            "Within-shell velocity-space concentration is a frozen residual-structure diagnostic, not an adjoint sensitivity and not evidence that modifying quadrature or transport would improve q_av.",
            "The failed Stage-28 MUSCL endpoint remains unrecovered and is not extended across Knudsen number.",
            "No physical, collision, correction-floor, source-relaxation, transport, wall, normalization, or velocity-quadrature parameter is retuned.",
        ],
        "scientifically_justified_next_scope": (
            "If one fixed radial node contributes at least twice the uniform ten-node share and the retained vertical-oblique sectors remain concentrated, resolve only that node into its twelve fixed angular ordinates per 45-degree sector. If concentration is shared among several nodes, audit their cancellation/coherence without any solver experiment."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage67-artifact-dir", required=True)
    parser.add_argument("--stage79-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run_stage80(args.stage67_artifact_dir, args.stage79_artifact_dir, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
