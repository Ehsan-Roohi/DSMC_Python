from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

STAGE84_COMPLETED_ENDPOINT = {
    "workflow_run_id": 31235732726,
    "workflow_job_id": 93047750025,
    "workflow_conclusion": "success",
    "tests_passed": 279,
    "tests_failed": 0,
    "test_duration_seconds": 1.20,
    "artifact_id": 9017015976,
    "artifact_size_bytes": 189090,
    "artifact_sha256": "28559f302646c3a182c5004da6b6db7b68d28f356b45ff5d06026d0cfddb76c2",
    "source_head_sha": "fc285ed33d30e7bb5f2eda7bf821cb58f90e4896",
    "summary_sha256": "18281901612355cc62e9395e78f7f5b4040d17f0fac6b56d483a27ab4726fe4f",
    "maps_sha256": "4b9775d20b964bab0deb0fe6c09e4a085388d5e49977b1cc9df33884969490d2",
    "decision": "stage84_first_interior_negative_lobes_broad_compensation_stage85_near_wall_face_amplitude_suppression_audit",
}

GRID = (64, 64)
KNUDSEN = 10.0
COLD_HOT_RATIO = 0.1
RADIAL_NODES = 40
ANGULAR_NODES = 96
POINT_COUNT = 3840
RADIAL_SCALE = 2.0
LIMITER = "minmod"
DOMINANT_MOMENT = "transverse_kinetic"
DOMINANT_RADIAL_SHELL = 2
DOMINANT_LOCAL_RADIAL_NODE = 1
DOMINANT_GLOBAL_RADIAL_NODE = 21
VERTICAL_OBLIQUE_BINS = (1, 2, 5, 6)
OPPOSITE_SECTOR_PAIRS = ((1, 5), (2, 6))
SIDES = ("left", "right")

# Preregistered diagnostic guards. These do not alter the solver.
WALL_TO_FIRST_L2_RATIO_MAX = 0.10
WALL_TO_FIRST_L1_RATIO_MAX = 0.10
SCALED_SHAPE_RESIDUAL_REL_FIRST_MAX = 0.05
NEAR_WALL_JUMP_SHARE_GUARD = 0.75


def validate_stage85_design(**overrides: object) -> None:
    frozen = {
        "grid": GRID,
        "kn0": KNUDSEN,
        "cold_hot_ratio": COLD_HOT_RATIO,
        "radial_nodes": RADIAL_NODES,
        "angular_nodes": ANGULAR_NODES,
        "radial_scale": RADIAL_SCALE,
        "limiter": LIMITER,
        "vertical_oblique_bins": VERTICAL_OBLIQUE_BINS,
        "opposite_sector_pairs": OPPOSITE_SECTOR_PAIRS,
        "wall_to_first_l2_ratio_max": WALL_TO_FIRST_L2_RATIO_MAX,
        "wall_to_first_l1_ratio_max": WALL_TO_FIRST_L1_RATIO_MAX,
        "scaled_shape_residual_rel_first_max": SCALED_SHAPE_RESIDUAL_REL_FIRST_MAX,
        "near_wall_jump_share_guard": NEAR_WALL_JUMP_SHARE_GUARD,
    }
    for key, value in overrides.items():
        if key not in frozen or value != frozen[key]:
            raise ValueError(
                "Stage 85 is frozen to the exact completed Stage-84 endpoint and preregistered "
                "near-wall face-amplitude guards; no solver or parameter retuning is permitted"
            )


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _validate_stage84_artifact(root: str | Path) -> dict[str, object]:
    root = Path(root)
    summary_path = root / "summary.json"
    maps_path = root / "wall_normal_sign_lobe_geometry_maps.npz"
    if not summary_path.exists() or not maps_path.exists():
        raise FileNotFoundError("Stage 85 requires exact completed Stage-84 summary and maps")
    if _sha256(summary_path) != STAGE84_COMPLETED_ENDPOINT["summary_sha256"]:
        raise ValueError("Stage-84 summary checksum mismatch")
    if _sha256(maps_path) != STAGE84_COMPLETED_ENDPOINT["maps_sha256"]:
        raise ValueError("Stage-84 maps checksum mismatch")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("stage") != 84 or summary.get("decision") != STAGE84_COMPLETED_ENDPOINT["decision"]:
        raise ValueError("Stage-84 completed decision mismatch")
    if not summary.get("finite") or not summary.get("closure_closed"):
        raise ValueError("Stage-84 endpoint is not finite and closure-closed")
    cfg = summary["configuration"]
    expected = {
        "grid": [64, 64],
        "kn0": 10.0,
        "radial_nodes": 40,
        "angular_nodes": 96,
        "point_count": 3840,
        "radial_scale": 2.0,
        "limiter": "minmod",
        "dominant_moment": DOMINANT_MOMENT,
        "dominant_radial_shell": 2,
        "dominant_local_radial_node": 1,
        "dominant_global_radial_node": 21,
        "vertical_oblique_bins": [1, 2, 5, 6],
        "opposite_sector_pairs": [[1, 5], [2, 6]],
        "solver_rerun_count": 0,
        "failed_muscl_endpoint_rehabilitated": False,
        "cross_knudsen_extension_permitted": False,
        "validation_claim_permitted": False,
    }
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f"Stage-84 frozen configuration mismatch for {key}")
    if any(value is not False for key, value in cfg.items() if key.endswith("_retuning")):
        raise ValueError("Stage-84 contains prohibited retuning")
    return summary


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size == 0 or np.std(a) <= 1e-300 or np.std(b) <= 1e-300:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _norm_ratio(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.ravel(a)) / max(float(np.linalg.norm(np.ravel(b))), 1e-300))


def _l1_ratio(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.abs(a)) / max(float(np.sum(np.abs(b))), 1e-300))


def _side_vectors(face: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if side == "left":
        return face[:, 0], face[:, 1], face[:, 2]
    if side == "right":
        return face[:, -1], face[:, -2], face[:, -3]
    raise ValueError("side must be left or right")


def near_wall_face_metrics(
    pair_face_groups: np.ndarray,
    opposite_sector_pairs: np.ndarray,
) -> dict[str, object]:
    face = np.asarray(pair_face_groups, dtype=np.float64)
    pairs = np.asarray(opposite_sector_pairs, dtype=np.int16)
    if face.shape != (2, 64, 63):
        raise ValueError("Stage 85 requires exact Stage-84 pair-face shape (2,64,63)")
    if not np.array_equal(pairs, [[1, 5], [2, 6]]):
        raise ValueError("Stage 85 requires exact inherited opposite-sector pairs")
    if not np.all(np.isfinite(face)):
        raise ValueError("Stage 85 requires finite pair-face maps")

    rows: list[dict[str, object]] = []
    wall_vectors = np.zeros((2, 2, 64), dtype=np.float64)
    first_vectors = np.zeros_like(wall_vectors)
    second_vectors = np.zeros_like(wall_vectors)
    fit_vectors = np.zeros_like(wall_vectors)

    for pair_index, pair in enumerate(pairs.tolist()):
        for side_index, side in enumerate(SIDES):
            wall, first, second = _side_vectors(face[pair_index], side)
            wall_vectors[pair_index, side_index] = wall
            first_vectors[pair_index, side_index] = first
            second_vectors[pair_index, side_index] = second

            first_l2_sq = float(np.dot(first, first))
            scale = 0.0 if first_l2_sq <= 1e-300 else float(np.dot(wall, first) / first_l2_sq)
            fit = scale * first
            fit_vectors[pair_index, side_index] = fit

            wall_first_jump = float(np.linalg.norm(first - wall))
            first_second_jump = float(np.linalg.norm(second - first))
            jump_share = wall_first_jump / max(wall_first_jump + first_second_jump, 1e-300)
            rows.append({
                "angular_bin_pair": pair,
                "side": side,
                "wall_to_first_l2_ratio": _norm_ratio(wall, first),
                "wall_to_first_l1_ratio": _l1_ratio(wall, first),
                "wall_first_raw_y_correlation": _corr(wall, first),
                "wall_as_scaled_first_factor": scale,
                "scaled_shape_residual_rel_to_first": _norm_ratio(wall - fit, first),
                "near_wall_jump_share_of_first_two_transitions": float(jump_share),
                "first_to_second_l2_ratio": _norm_ratio(first, second),
                "first_second_raw_y_correlation": _corr(first, second),
                "wall_face_l2": float(np.linalg.norm(wall)),
                "first_interior_face_l2": float(np.linalg.norm(first)),
                "second_interior_face_l2": float(np.linalg.norm(second)),
            })

    max_l2 = max(float(r["wall_to_first_l2_ratio"]) for r in rows)
    max_l1 = max(float(r["wall_to_first_l1_ratio"]) for r in rows)
    max_fit = max(float(r["scaled_shape_residual_rel_to_first"]) for r in rows)
    min_jump = min(float(r["near_wall_jump_share_of_first_two_transitions"]) for r in rows)
    finite = all(
        np.isfinite(float(value))
        for row in rows
        for key, value in row.items()
        if key not in {"angular_bin_pair", "side"}
    )
    abrupt = (
        finite
        and max_l2 <= WALL_TO_FIRST_L2_RATIO_MAX
        and max_l1 <= WALL_TO_FIRST_L1_RATIO_MAX
        and max_fit <= SCALED_SHAPE_RESIDUAL_REL_FIRST_MAX
        and min_jump >= NEAR_WALL_JUMP_SHARE_GUARD
    )
    if not finite:
        decision = "stage85_nonfinite_near_wall_face_amplitude_blocker"
    elif abrupt:
        decision = "stage85_abrupt_near_wall_face_suppression_stage86_boundary_reconstruction_stencil_activation_audit"
    elif max_l2 <= 0.50 and min_jump >= 0.50:
        decision = "stage85_partial_near_wall_face_suppression_stage86_y_localized_face_transition_audit"
    else:
        decision = "stage85_distributed_or_incoherent_face_variation_stage86_frozen_spatial_profile_audit"

    return {
        "rows": rows,
        "maximum_wall_to_first_l2_ratio": max_l2,
        "maximum_wall_to_first_l1_ratio": max_l1,
        "maximum_scaled_shape_residual_rel_to_first": max_fit,
        "minimum_near_wall_jump_share_of_first_two_transitions": min_jump,
        "minimum_raw_wall_first_y_correlation": min(float(r["wall_first_raw_y_correlation"]) for r in rows),
        "maximum_raw_wall_first_y_correlation": max(float(r["wall_first_raw_y_correlation"]) for r in rows),
        "finite": finite,
        "decision": decision,
        "wall_adjacent_face_vectors": wall_vectors,
        "first_interior_face_vectors": first_vectors,
        "second_interior_face_vectors": second_vectors,
        "scaled_first_fit_to_wall_vectors": fit_vectors,
    }


def build_summary(stage84_artifact_dir: str | Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    validate_stage85_design()
    retained84 = _validate_stage84_artifact(stage84_artifact_dir)
    with np.load(Path(stage84_artifact_dir) / "wall_normal_sign_lobe_geometry_maps.npz") as z:
        pairs = np.asarray(z["opposite_sector_pairs"], dtype=np.int16)
        pair_face_groups = np.asarray(z["pair_face_groups"], dtype=np.float64)
        pair_cell_groups = np.asarray(z["pair_cell_divergence_groups"], dtype=np.float64)

    metrics = near_wall_face_metrics(pair_face_groups, pairs)
    decision = str(metrics["decision"])
    finite = bool(metrics["finite"])
    if decision.endswith("boundary_reconstruction_stencil_activation_audit"):
        next_scope = (
            "Using the exact checksum-verified Stage-85 face vectors together with the frozen transport "
            "reconstruction implementation, audit whether the boundary stencil or limiter activation forces "
            "the second-order correction toward zero at the wall-adjacent x faces. Distinguish an algorithmic "
            "boundary-stencil effect from physical velocity-distribution structure without rerunning the cavity "
            "solver, changing any parameter, rehabilitating Stage 28, or extending across Knudsen number."
        )
    elif decision.endswith("y_localized_face_transition_audit"):
        next_scope = (
            "Localize the frozen wall-to-first-face transition in y before any solver experiment; no retuning, "
            "validation claim, or cross-Knudsen extension is authorized."
        )
    else:
        next_scope = (
            "Audit the frozen x-face spatial profile responsible for the distributed or incoherent transition; "
            "do not rerun or retune the solver."
        )

    metric_public = {k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)}
    cfg = {
        "grid": [64, 64],
        "kn0": 10.0,
        "cold_hot_ratio": 0.1,
        "radial_nodes": 40,
        "angular_nodes": 96,
        "point_count": 3840,
        "radial_scale": 2.0,
        "limiter": "minmod",
        "dominant_moment": DOMINANT_MOMENT,
        "dominant_radial_shell": 2,
        "dominant_local_radial_node": 1,
        "dominant_global_radial_node": 21,
        "vertical_oblique_bins": [1, 2, 5, 6],
        "opposite_sector_pairs": [[1, 5], [2, 6]],
        "wall_to_first_l2_ratio_max": WALL_TO_FIRST_L2_RATIO_MAX,
        "wall_to_first_l1_ratio_max": WALL_TO_FIRST_L1_RATIO_MAX,
        "scaled_shape_residual_rel_first_max": SCALED_SHAPE_RESIDUAL_REL_FIRST_MAX,
        "near_wall_jump_share_guard": NEAR_WALL_JUMP_SHARE_GUARD,
        "raw_pearson_y_correlation_is_diagnostic_only": True,
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
    }
    summary = {
        "stage": 85,
        "description": "Frozen near-wall x-face amplitude-suppression audit on the exact Stage-84 opposite-sector pair maps.",
        "finite": finite,
        "configuration": cfg,
        "retained_stage84_decision": retained84["decision"],
        "near_wall_face_amplitude_metrics": metric_public,
        "decision": decision,
        "scientifically_justified_next_scope": next_scope,
        "positive_findings": [
            "Wall-adjacent, first-interior, and second-interior x-face vectors are compared directly on both sidewalls for both retained opposite-sector pairs.",
            "Amplitude ratios, a scaled-shape residual, raw y correlation, and the fraction of the first two x-face transitions carried by the wall-to-first jump are reported without generating a new solver state.",
        ],
        "negative_findings": [
            "Raw Pearson correlation is diagnostic only when the wall-adjacent vector is nearly extinguished; the preregistered decision uses amplitude ratios, scaled-shape residual, and jump concentration instead.",
            "A frozen near-wall suppression pattern does not establish a wall-model or reconstruction error and is not an adjoint sensitivity or evidence that changing transport would improve q_av.",
            "The failed Stage-28 MUSCL endpoint remains unrecovered and is not extended across Knudsen number.",
            "No physical, collision, correction-floor, source-relaxation, transport, wall, normalization, or velocity-quadrature parameter is retuned, and no cavity solver is rerun.",
        ],
    }
    maps = {
        "opposite_sector_pairs": pairs,
        "pair_face_groups": pair_face_groups,
        "pair_cell_divergence_groups": pair_cell_groups,
        "wall_adjacent_face_vectors": metrics["wall_adjacent_face_vectors"],
        "first_interior_face_vectors": metrics["first_interior_face_vectors"],
        "second_interior_face_vectors": metrics["second_interior_face_vectors"],
        "scaled_first_fit_to_wall_vectors": metrics["scaled_first_fit_to_wall_vectors"],
        "side_names": np.asarray(SIDES),
    }
    return summary, maps


def run(stage84_artifact_dir: str | Path, output_dir: str | Path) -> dict[str, object]:
    summary, maps = build_summary(stage84_artifact_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(out / "near_wall_face_amplitude_suppression_maps.npz", **maps)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage84-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.stage84_artifact_dir, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
