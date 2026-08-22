from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


STAGE67_COMPLETED_ENDPOINT = {
    "workflow_run_id": 30991124477,
    "workflow_job_id": 92257254811,
    "workflow_conclusion": "success",
    "tests_passed": 71,
    "tests_failed": 0,
    "test_duration_seconds": 0.43,
    "artifact_id": 8931272132,
    "artifact_size_bytes": 173096061,
    "artifact_sha256": "5b871e94bdca02bce33846657f8f4dc263d68af9e671d7eb137ca7c83da198d4",
    "source_head_sha": "87e6ca98637754e72482b897492147edfcfcf4d9",
    "summary_sha256": "e04043a1913b2fa9ae57fe1561aa26c70627830d648e91204093c8f1fb57b3d1",
    "distributions_sha256": "d4002a2765137ba517abec2d0483a3e5adcf13f415c53259018556bd14d612d1",
    "residual_maps_sha256": "08722bd5b2036eee1b42b09d37583701ffcc3ef5e4f7d7c68642ea5103f11ced",
    "decision": (
        "stage67_frozen_replay_and_residual_balance_close_"
        "stage68_independent_transport_operator_residual_audit"
    ),
}

STAGE74_COMPLETED_ENDPOINT = {
    "workflow_run_id": 31099020204,
    "workflow_job_id": 92607816279,
    "workflow_conclusion": "success",
    "tests_passed": 178,
    "tests_failed": 0,
    "test_duration_seconds": 0.91,
    "artifact_id": 8973240674,
    "artifact_size_bytes": 1049686,
    "artifact_sha256": "b6f5541571956c9d34d5a2514f3faa9117f1639e1b26948f7c6bb2e31062dc39",
    "source_head_sha": "b6e3d508e7bcb105c223be013726e8f7cef3c50d",
    "summary_sha256": "c207934d1dd9c1fc566d9c76d791b97e47df1936cd21f2414a6ef1e6fb1ef9d7",
    "maps_sha256": "ccd7c542649be94ccf025c30ab4285b8d93b63a075ac62b714fe2609fe210e4f",
    "decision": (
        "stage74_diffuse_speed_or_weak_opposite_sector_cancellation_"
        "stage75_signed_face_location_velocity_moment_audit"
    ),
}

GRID = (64, 64)
KNUDSEN = 10.0
COLD_HOT_RATIO = 0.1
RULE = (40, 96)
RADIAL_SCALE = 2.0
POINT_COUNT = 3840
CHUNK_SIZE = 128
LIMITER = "minmod"
MOMENT_NAMES = ("streamwise_kinetic", "transverse_kinetic", "reduced_internal")
X_ZONE_EDGES = (0.0, 0.25, 0.5, 0.75, 1.0)
Y_BAND_LAYER_EDGES = (0, 2, 4, GRID[0] // 2)
CLOSURE_GUARD = 1.0e-10
FACE_PAIR_CANCELLATION_GUARD = 0.05
SIDE_SIGNED_BALANCE_GUARD = 1.0e-9


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_stage75_design(
    grid: tuple[int, int] = GRID,
    kn0: float = KNUDSEN,
    cold_hot_ratio: float = COLD_HOT_RATIO,
    rule: tuple[int, int] = RULE,
    radial_scale: float = RADIAL_SCALE,
    chunk_size: int = CHUNK_SIZE,
    limiter: str = LIMITER,
    moment_names: tuple[str, ...] = MOMENT_NAMES,
    x_zone_edges: tuple[float, ...] = X_ZONE_EDGES,
    y_band_layer_edges: tuple[int, ...] = Y_BAND_LAYER_EDGES,
    face_pair_cancellation_guard: float = FACE_PAIR_CANCELLATION_GUARD,
) -> None:
    actual = (
        grid, kn0, cold_hot_ratio, rule, radial_scale, chunk_size, limiter,
        moment_names, x_zone_edges, y_band_layer_edges, face_pair_cancellation_guard,
    )
    expected = (
        GRID, KNUDSEN, COLD_HOT_RATIO, RULE, RADIAL_SCALE, CHUNK_SIZE, LIMITER,
        MOMENT_NAMES, X_ZONE_EDGES, Y_BAND_LAYER_EDGES, FACE_PAIR_CANCELLATION_GUARD,
    )
    if actual != expected:
        raise ValueError(
            "Stage 75 is frozen to the exact Stage-67 64x64 distributions, exact "
            "Stage-74 residual map, 40x96 radial-scale-2.0 quadrature, minmod "
            "second-minus-first-order face flux, fixed kinetic/internal moment split, "
            "and preregistered spatial/cancellation guards; no retuning is permitted."
        )


def _validate_artifact(root: str | Path, endpoint: dict[str, object], files: dict[str, str]) -> dict[str, object]:
    root = Path(root)
    for name, checksum_key in files.items():
        path = root / name
        expected = str(endpoint[checksum_key])
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Completed artifact checksum mismatch: {name}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    stage = 67 if endpoint is STAGE67_COMPLETED_ENDPOINT else 74
    if summary.get("stage") != stage or summary.get("decision") != endpoint["decision"]:
        raise ValueError(f"Stage {stage} completed endpoint mismatch")
    return summary


def minmod(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    same_sign = left * right > 0.0
    return np.where(same_sign, np.sign(left) * np.minimum(np.abs(left), np.abs(right)), 0.0)


def limited_slopes_x(distribution: np.ndarray) -> np.ndarray:
    distribution = np.asarray(distribution, dtype=np.float64)
    slope = np.zeros_like(distribution)
    if distribution.shape[1] > 2:
        backward = distribution[:, 1:-1] - distribution[:, :-2]
        forward = distribution[:, 2:] - distribution[:, 1:-1]
        slope[:, 1:-1] = minmod(backward, forward)
    return slope


def macroscopic_velocity(
    phi: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    weight: np.ndarray,
    chunk_size: int = CHUNK_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rho = np.zeros(phi.shape[:2], dtype=np.float64)
    momentum_x = np.zeros_like(rho)
    momentum_y = np.zeros_like(rho)
    for start in range(0, weight.size, chunk_size):
        stop = min(start + chunk_size, weight.size)
        sl = slice(start, stop)
        weighted = phi[..., sl] * weight[sl][None, None, :]
        rho += np.sum(weighted, axis=-1)
        momentum_x += np.sum(weighted * vx[sl][None, None, :], axis=-1)
        momentum_y += np.sum(weighted * vy[sl][None, None, :], axis=-1)
    safe = np.maximum(rho, 1.0e-300)
    return rho, momentum_x / safe, momentum_y / safe


def interior_x_face_flux_difference_chunk(distribution: np.ndarray, vx: np.ndarray) -> np.ndarray:
    distribution = np.asarray(distribution, dtype=np.float64)
    vx = np.asarray(vx, dtype=np.float64)
    slope = limited_slopes_x(distribution)
    delta = np.zeros(
        (distribution.shape[0], distribution.shape[1] - 1, distribution.shape[2]),
        dtype=np.float64,
    )
    positive = vx > 0.0
    negative = vx < 0.0
    if np.any(positive):
        delta[..., positive] = 0.5 * vx[positive][None, None, :] * slope[:, :-1, positive]
    if np.any(negative):
        delta[..., negative] = -0.5 * vx[negative][None, None, :] * slope[:, 1:, negative]
    return delta


def moment_components(
    delta_phi: np.ndarray,
    delta_psi: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    weight: np.ndarray,
    local_u: np.ndarray,
    local_v: np.ndarray,
    sign: float,
    dx: float,
) -> np.ndarray:
    cx = vx[None, None, :] - local_u[..., None]
    cy = vy[None, None, :] - local_v[..., None]
    residual_phi = sign * delta_phi / dx
    residual_psi = sign * delta_psi / dx
    weighted = weight[None, None, :]
    return np.stack(
        [
            0.5 * np.sum(cy * cx * cx * residual_phi * weighted, axis=-1),
            0.5 * np.sum(cy * cy * cy * residual_phi * weighted, axis=-1),
            0.5 * np.sum(cy * residual_psi * weighted, axis=-1),
        ],
        axis=0,
    )


def evaluate_face_location_moment_maps(
    phi: np.ndarray,
    psi: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    weight: np.ndarray,
    chunk_size: int = CHUNK_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if phi.shape != psi.shape or phi.shape != (GRID[0], GRID[1], POINT_COUNT):
        raise ValueError("Stage 75 requires exact 64x64x3840 distributions")
    _, local_u, local_v = macroscopic_velocity(phi, vx, vy, weight, chunk_size)
    left = np.zeros((len(MOMENT_NAMES), GRID[0], GRID[1] - 1), dtype=np.float64)
    right = np.zeros_like(left)
    dx = 1.0 / GRID[1]
    for start in range(0, weight.size, chunk_size):
        stop = min(start + chunk_size, weight.size)
        sl = slice(start, stop)
        delta_phi = interior_x_face_flux_difference_chunk(phi[..., sl], vx[sl])
        delta_psi = interior_x_face_flux_difference_chunk(psi[..., sl], vx[sl])
        left += moment_components(
            delta_phi, delta_psi, vx[sl], vy[sl], weight[sl],
            local_u[:, :-1], local_v[:, :-1], -1.0, dx,
        )
        right += moment_components(
            delta_phi, delta_psi, vx[sl], vy[sl], weight[sl],
            local_u[:, 1:], local_v[:, 1:], 1.0, dx,
        )
    cell = np.zeros((len(MOMENT_NAMES), *GRID), dtype=np.float64)
    cell[:, :, :-1] += left
    cell[:, :, 1:] += right
    return cell, left, right


def fixed_spatial_masks(grid: tuple[int, int] = GRID) -> tuple[list[np.ndarray], list[np.ndarray]]:
    ny, nx = grid
    face_x = np.arange(1, nx, dtype=np.float64) / nx
    x_masks = [
        (face_x >= X_ZONE_EDGES[i]) & (face_x < X_ZONE_EDGES[i + 1])
        for i in range(len(X_ZONE_EDGES) - 1)
    ]
    layer = np.minimum(np.arange(ny), np.arange(ny)[::-1])
    y_masks = [
        (layer >= Y_BAND_LAYER_EDGES[i]) & (layer < Y_BAND_LAYER_EDGES[i + 1])
        for i in range(len(Y_BAND_LAYER_EDGES) - 1)
    ]
    return x_masks, y_masks


def signed_statistics(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    absolute_sum = float(np.sum(np.abs(values)))
    return {
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "rms": float(np.sqrt(np.mean(values * values))),
        "signed_sum": float(np.sum(values)),
        "absolute_sum": absolute_sum,
        "signed_to_absolute_ratio": float(np.sum(values) / max(absolute_sum, 1.0e-300)),
    }


def face_location_moment_metrics(cell: np.ndarray, left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    total = np.sum(cell, axis=0)
    component_abs = np.sum(np.abs(cell), axis=(1, 2))
    face_abs = np.sum(np.abs(left) + np.abs(right), axis=0)
    x_masks, y_masks = fixed_spatial_masks((cell.shape[1], cell.shape[2]))
    x_abs = [float(np.sum(face_abs[:, mask])) for mask in x_masks]
    y_abs = [float(np.sum(face_abs[mask, :])) for mask in y_masks]
    face_pre = float(np.sum(np.abs(left) + np.abs(right)))
    face_post = float(np.sum(np.abs(left + right)))
    left_signed = np.sum(left, axis=(1, 2))
    right_signed = np.sum(right, axis=(1, 2))
    side_balance = float(
        np.max(np.abs(left_signed - right_signed))
        / max(float(np.max(np.abs(left_signed) + np.abs(right_signed))), 1.0e-300)
    )
    return {
        "moment_names": list(MOMENT_NAMES),
        "moment_absolute_shares": (component_abs / max(float(np.sum(component_abs)), 1.0e-300)).tolist(),
        "moment_signed_sums": np.sum(cell, axis=(1, 2)).tolist(),
        "moment_to_total_absolute_cancellation_ratio": float(
            np.sum(np.abs(total)) / max(float(np.sum(component_abs)), 1.0e-300)
        ),
        "face_pair_pre_cancellation_absolute_sum": face_pre,
        "face_pair_post_cancellation_absolute_sum": face_post,
        "face_pair_cancellation_ratio": face_post / max(face_pre, 1.0e-300),
        "left_target_moment_signed_sums": left_signed.tolist(),
        "right_target_moment_signed_sums": right_signed.tolist(),
        "left_right_signed_balance_error": side_balance,
        "x_zone_absolute_shares": (np.asarray(x_abs) / max(sum(x_abs), 1.0e-300)).tolist(),
        "outer_x_quarters_absolute_share": float((x_abs[0] + x_abs[-1]) / max(sum(x_abs), 1.0e-300)),
        "y_band_absolute_shares": (np.asarray(y_abs) / max(sum(y_abs), 1.0e-300)).tolist(),
        "interior_y_band_absolute_share": float(y_abs[-1] / max(sum(y_abs), 1.0e-300)),
        "total_cell_map": signed_statistics(total),
    }


def stage75_decision(
    finite: bool,
    provenance_consistent: bool,
    moment_closure_closed: bool,
    face_pair_cancellation_ratio: float,
    side_signed_balance_error: float,
) -> str:
    if not finite:
        return "stage75_nonfinite_velocity_moment_blocker"
    if not provenance_consistent:
        return "stage75_completed_endpoint_reproduction_blocker"
    if not moment_closure_closed:
        return "stage75_velocity_moment_sum_closure_blocker"
    if (
        face_pair_cancellation_ratio <= FACE_PAIR_CANCELLATION_GUARD
        and side_signed_balance_error <= SIDE_SIGNED_BALANCE_GUARD
    ):
        return (
            "stage75_conservative_face_pair_cancellation_"
            "stage76_local_velocity_frame_jump_audit"
        )
    return (
        "stage75_weak_or_asymmetric_face_pair_cancellation_"
        "stage76_signed_boundary_cell_residual_audit"
    )


def run_stage75(
    stage67_artifact_dir: str | Path,
    stage74_artifact_dir: str | Path,
    output_dir: str | Path,
    **design: object,
) -> dict[str, object]:
    validate_stage75_design(**design)
    retained67 = _validate_artifact(
        stage67_artifact_dir,
        STAGE67_COMPLETED_ENDPOINT,
        {
            "summary.json": "summary_sha256",
            "converged_full_distributions.npz": "distributions_sha256",
            "steady_residual_moment_maps.npz": "residual_maps_sha256",
        },
    )
    retained74 = _validate_artifact(
        stage74_artifact_dir,
        STAGE74_COMPLETED_ENDPOINT,
        {"summary.json": "summary_sha256", "radial_shell_opposite_sector_maps.npz": "maps_sha256"},
    )
    with np.load(Path(stage67_artifact_dir) / "converged_full_distributions.npz") as data:
        phi = np.asarray(data["phi"], dtype=np.float64)
        psi = np.asarray(data["psi"], dtype=np.float64)
        vx = np.asarray(data["vx"], dtype=np.float64)
        vy = np.asarray(data["vy"], dtype=np.float64)
        weight = np.asarray(data["weight"], dtype=np.float64)
    cell, left, right = evaluate_face_location_moment_maps(phi, psi, vx, vy, weight)
    with np.load(Path(stage74_artifact_dir) / "radial_shell_opposite_sector_maps.npz") as data:
        reference = np.asarray(data["grid64_reconstructed_x_qy"], dtype=np.float64)
    reconstructed = np.sum(cell, axis=0)
    delta = reconstructed - reference
    relative_l2 = float(np.linalg.norm(delta.ravel()) / max(float(np.linalg.norm(reference.ravel())), 1.0e-300))
    maximum_absolute = float(np.max(np.abs(delta)))
    moment_closure_closed = bool(relative_l2 <= CLOSURE_GUARD)
    metrics = face_location_moment_metrics(cell, left, right)
    finite = bool(np.all(np.isfinite(cell)) and np.all(np.isfinite(left)) and np.all(np.isfinite(right)))
    provenance_consistent = bool(
        retained67.get("decision") == STAGE67_COMPLETED_ENDPOINT["decision"]
        and retained74.get("decision") == STAGE74_COMPLETED_ENDPOINT["decision"]
        and retained74.get("finite") is True
        and retained74.get("grouped_closure_closed") is True
        and retained74.get("provenance_consistent") is True
    )
    decision = stage75_decision(
        finite,
        provenance_consistent,
        moment_closure_closed,
        float(metrics["face_pair_cancellation_ratio"]),
        float(metrics["left_right_signed_balance_error"]),
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "signed_face_location_velocity_moment_maps.npz",
        moment_cell_maps=cell,
        face_left_target_moment_maps=left,
        face_right_target_moment_maps=right,
        reconstructed_total_cell_map=reconstructed,
        stage74_reference_total_cell_map=reference,
    )
    summary = {
        "stage": 75,
        "description": (
            "Exact frozen decomposition of the Stage-74 x-direction q_y transport-operator "
            "difference into streamwise-kinetic, transverse-kinetic, and reduced-internal "
            "peculiar-velocity moments at every interior face and both adjacent target cells."
        ),
        "configuration": {
            "grid": list(GRID),
            "kn0": KNUDSEN,
            "cold_hot_ratio": COLD_HOT_RATIO,
            "radial_nodes": RULE[0],
            "angular_nodes": RULE[1],
            "point_count": POINT_COUNT,
            "radial_scale": RADIAL_SCALE,
            "chunk_size": CHUNK_SIZE,
            "limiter": LIMITER,
            "moment_names": list(MOMENT_NAMES),
            "x_zone_edges": list(X_ZONE_EDGES),
            "y_band_layer_edges": list(Y_BAND_LAYER_EDGES),
            "face_pair_cancellation_guard": FACE_PAIR_CANCELLATION_GUARD,
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
        "retained_stage67_endpoint": STAGE67_COMPLETED_ENDPOINT,
        "retained_stage67_decision": retained67["decision"],
        "retained_stage74_endpoint": STAGE74_COMPLETED_ENDPOINT,
        "retained_stage74_decision": retained74["decision"],
        "moment_sum_closure": {
            "maximum_absolute_error": maximum_absolute,
            "relative_l2_error": relative_l2,
            "within_guard": moment_closure_closed,
        },
        "fine_grid_face_location_velocity_moment_metrics": metrics,
        "finite": finite,
        "provenance_consistent": provenance_consistent,
        "moment_closure_closed": moment_closure_closed,
        "decision": decision,
        "positive_findings": [
            "The three fixed peculiar-velocity moment components sum to the exact Stage-74 x-direction q_y residual map within the unchanged closure guard.",
            "Every interior face is retained with separate conservative contributions to its left and right target cells.",
            "The analysis uses the exact Stage-67 distributions and performs no cavity solve or operator substitution."
        ],
        "negative_findings": [
            "Large pre-cancellation face contributions are not a sensitivity measure and do not predict the q_av response of a converged modified solver.",
            "The failed Stage-28 MUSCL endpoint remains unrecovered and is not extended across Knudsen number.",
            "No physical, collision, correction-floor, source-relaxation, transport, wall, normalization, or velocity-quadrature parameter was retuned."
        ],
        "interpretation_guard": (
            "This stage is an exact frozen residual decomposition. Face-pair cancellation and "
            "moment shares do not establish causality for the published heat-flux discrepancy."
        ),
        "scientifically_justified_next_scope": (
            "If opposite target-cell contributions cancel conservatively, audit the residual "
            "left by the jump in cell-local peculiar-velocity frames across each face. Otherwise "
            "localize the signed boundary-cell remainder before any solver experiment."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage67-artifact-dir", required=True)
    parser.add_argument("--stage74-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = run_stage75(args.stage67_artifact_dir, args.stage74_artifact_dir, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
