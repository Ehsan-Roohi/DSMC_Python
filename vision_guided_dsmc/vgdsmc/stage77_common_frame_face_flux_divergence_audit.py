from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import stage76_local_velocity_frame_jump_audit as stage76

GRID = (64, 64)
KNUDSEN = 10.0
COLD_HOT_RATIO = 0.1
RULE = (40, 96)
RADIAL_SCALE = 2.0
LIMITER = "minmod"
FACE_TO_CELL_CANCELLATION_GUARD = 0.10
ADJACENT_X_FACE_CORRELATION_GUARD = 0.90
CLOSURE_GUARD = 1.0e-10


def validate_stage77_design(
    grid: tuple[int, int] = GRID,
    kn0: float = KNUDSEN,
    cold_hot_ratio: float = COLD_HOT_RATIO,
    rule: tuple[int, int] = RULE,
    radial_scale: float = RADIAL_SCALE,
    limiter: str = LIMITER,
    face_to_cell_cancellation_guard: float = FACE_TO_CELL_CANCELLATION_GUARD,
    adjacent_x_face_correlation_guard: float = ADJACENT_X_FACE_CORRELATION_GUARD,
) -> None:
    actual = (
        grid,
        kn0,
        cold_hot_ratio,
        rule,
        radial_scale,
        limiter,
        face_to_cell_cancellation_guard,
        adjacent_x_face_correlation_guard,
    )
    expected = (
        GRID,
        KNUDSEN,
        COLD_HOT_RATIO,
        RULE,
        RADIAL_SCALE,
        LIMITER,
        FACE_TO_CELL_CANCELLATION_GUARD,
        ADJACENT_X_FACE_CORRELATION_GUARD,
    )
    if actual != expected:
        raise ValueError(
            "Stage 77 is frozen to the exact Stage-67 distributions, the exact Stage-75 "
            "cell residual, the Stage-76 midpoint common-frame construction, and the "
            "preregistered 0.10 cancellation / 0.90 adjacent-face-correlation guards."
        )


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.size == 0 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def divergence_from_interior_faces(face_flux: np.ndarray) -> np.ndarray:
    face_flux = np.asarray(face_flux, dtype=np.float64)
    if face_flux.shape != (GRID[0], GRID[1] - 1):
        raise ValueError("Stage 77 requires a 64x63 interior x-face map")
    cell = np.zeros(GRID, dtype=np.float64)
    cell[:, :-1] -= face_flux
    cell[:, 1:] += face_flux
    return cell


def face_divergence_metrics(face_flux: np.ndarray, cell_map: np.ndarray) -> dict[str, float]:
    face_flux = np.asarray(face_flux, dtype=np.float64)
    cell_map = np.asarray(cell_map, dtype=np.float64)
    face_abs = float(np.sum(np.abs(face_flux)))
    target_abs = 2.0 * face_abs
    cell_abs = float(np.sum(np.abs(cell_map)))
    x_corr = _corr(face_flux[:, :-1], face_flux[:, 1:])
    y_corr = _corr(face_flux[:-1, :], face_flux[1:, :])
    x_rel = float(
        np.linalg.norm((face_flux[:, 1:] - face_flux[:, :-1]).ravel())
        / max(float(np.linalg.norm(face_flux[:, :-1].ravel())), 1.0e-300)
    )
    y_rel = float(
        np.linalg.norm((face_flux[1:, :] - face_flux[:-1, :]).ravel())
        / max(float(np.linalg.norm(face_flux[:-1, :].ravel())), 1.0e-300)
    )
    return {
        "face_absolute_sum": face_abs,
        "two_target_face_absolute_sum": target_abs,
        "cell_divergence_absolute_sum": cell_abs,
        "face_to_cell_cancellation_ratio": cell_abs / max(target_abs, 1.0e-300),
        "adjacent_x_face_correlation": x_corr,
        "adjacent_y_row_correlation": y_corr,
        "adjacent_x_face_relative_l2_difference": x_rel,
        "adjacent_y_row_relative_l2_difference": y_rel,
        "face_signed_to_absolute_ratio": float(np.sum(face_flux)) / max(face_abs, 1.0e-300),
        "cell_signed_to_absolute_ratio": float(np.sum(cell_map)) / max(cell_abs, 1.0e-300),
    }


def spatial_shares(cell_map: np.ndarray) -> dict[str, float]:
    a = np.abs(np.asarray(cell_map, dtype=np.float64))
    total = max(float(np.sum(a)), 1.0e-300)
    wall1 = (
        np.sum(a[0, :]) + np.sum(a[-1, :]) + np.sum(a[:, 0]) + np.sum(a[:, -1])
        - a[0, 0] - a[0, -1] - a[-1, 0] - a[-1, -1]
    )
    outer2 = (
        np.sum(a[:2, :]) + np.sum(a[-2:, :]) + np.sum(a[:, :2]) + np.sum(a[:, -2:])
        - np.sum(a[:2, :2]) - np.sum(a[:2, -2:])
        - np.sum(a[-2:, :2]) - np.sum(a[-2:, -2:])
    )
    return {
        "outer_one_cell_wall_share": float(wall1 / total),
        "outer_two_cell_wall_share": float(outer2 / total),
        "interior_two_cell_complement_share": float(1.0 - outer2 / total),
    }


def stage77_decision(
    finite: bool,
    closure_closed: bool,
    cancellation_ratio: float,
    adjacent_x_correlation: float,
) -> str:
    if not finite:
        return "stage77_nonfinite_common_frame_flux_blocker"
    if not closure_closed:
        return "stage77_common_frame_divergence_closure_blocker"
    if (
        cancellation_ratio <= FACE_TO_CELL_CANCELLATION_GUARD
        and adjacent_x_correlation >= ADJACENT_X_FACE_CORRELATION_GUARD
    ):
        return "stage77_coherent_face_cancellation_stage78_face_gradient_moment_attribution_audit"
    return "stage77_noncoherent_or_material_divergence_stage78_spatial_face_localization_audit"


def run_stage77(
    stage67_artifact_dir: str | Path,
    stage75_artifact_dir: str | Path,
    output_dir: str | Path,
    **design: object,
) -> dict[str, object]:
    validate_stage77_design(**design)
    retained67 = stage76._validate_artifact(
        stage67_artifact_dir,
        stage76.STAGE67_COMPLETED_ENDPOINT,
        {
            "summary.json": "summary_sha256",
            "converged_full_distributions.npz": "distributions_sha256",
            "steady_residual_moment_maps.npz": "residual_maps_sha256",
        },
        67,
    )
    retained75 = stage76._validate_artifact(
        stage75_artifact_dir,
        stage76.STAGE75_COMPLETED_ENDPOINT,
        {
            "summary.json": "summary_sha256",
            "signed_face_location_velocity_moment_maps.npz": "maps_sha256",
        },
        75,
    )
    with np.load(Path(stage67_artifact_dir) / "converged_full_distributions.npz") as data:
        phi = np.asarray(data["phi"], dtype=np.float64)
        psi = np.asarray(data["psi"], dtype=np.float64)
        vx = np.asarray(data["vx"], dtype=np.float64)
        vy = np.asarray(data["vy"], dtype=np.float64)
        weight = np.asarray(data["weight"], dtype=np.float64)
    frame_maps = stage76.evaluate_frame_jump_maps(phi, psi, vx, vy, weight)
    midpoint_components = frame_maps["midpoint_common_frame_face_components"]
    midpoint_face = np.sum(midpoint_components, axis=0)
    midpoint_cell = divergence_from_interior_faces(midpoint_face)
    with np.load(Path(stage75_artifact_dir) / "signed_face_location_velocity_moment_maps.npz") as data:
        stage75_cell = np.asarray(data["reconstructed_total_cell_map"], dtype=np.float64)
    closure_delta = midpoint_cell - frame_maps["midpoint_common_frame_cell_map"]
    closure_rel = float(
        np.linalg.norm(closure_delta.ravel())
        / max(float(np.linalg.norm(midpoint_cell.ravel())), 1.0e-300)
    )
    closure_max = float(np.max(np.abs(closure_delta)))
    stage75_rel = float(
        np.linalg.norm((midpoint_cell - stage75_cell).ravel())
        / max(float(np.linalg.norm(stage75_cell.ravel())), 1.0e-300)
    )
    metrics = face_divergence_metrics(midpoint_face, midpoint_cell)
    metrics.update(spatial_shares(midpoint_cell))
    metrics["midpoint_vs_stage75_cell_relative_l2_error"] = stage75_rel
    metrics["positive_face_absolute_share"] = float(
        np.sum(np.abs(midpoint_face[midpoint_face > 0.0]))
        / max(float(np.sum(np.abs(midpoint_face))), 1.0e-300)
    )
    metrics["negative_face_absolute_share"] = float(
        np.sum(np.abs(midpoint_face[midpoint_face < 0.0]))
        / max(float(np.sum(np.abs(midpoint_face))), 1.0e-300)
    )
    finite = bool(
        np.all(np.isfinite(midpoint_components))
        and np.all(np.isfinite(midpoint_face))
        and np.all(np.isfinite(midpoint_cell))
        and all(np.isfinite(value) for value in metrics.values())
    )
    closure_closed = bool(closure_rel <= CLOSURE_GUARD)
    decision = stage77_decision(
        finite,
        closure_closed,
        metrics["face_to_cell_cancellation_ratio"],
        metrics["adjacent_x_face_correlation"],
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "common_frame_face_flux_divergence_maps.npz",
        midpoint_common_frame_face_components=midpoint_components,
        midpoint_common_frame_total_face_flux=midpoint_face,
        midpoint_common_frame_cell_divergence=midpoint_cell,
        stage75_reference_cell_map=stage75_cell,
    )
    summary = {
        "stage": 77,
        "description": "Frozen common-frame face-flux divergence and coherence audit after Stage 76 showed negligible local-frame mismatch.",
        "configuration": {
            "grid": list(GRID),
            "kn0": KNUDSEN,
            "cold_hot_ratio": COLD_HOT_RATIO,
            "radial_nodes": RULE[0],
            "angular_nodes": RULE[1],
            "point_count": RULE[0] * RULE[1],
            "radial_scale": RADIAL_SCALE,
            "limiter": LIMITER,
            "face_to_cell_cancellation_guard": FACE_TO_CELL_CANCELLATION_GUARD,
            "adjacent_x_face_correlation_guard": ADJACENT_X_FACE_CORRELATION_GUARD,
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
        "retained_stage75_decision": retained75["decision"],
        "divergence_closure": {
            "maximum_absolute_error": closure_max,
            "relative_l2_error": closure_rel,
            "within_guard": closure_closed,
        },
        "fine_grid_common_frame_metrics": metrics,
        "finite": finite,
        "closure_closed": closure_closed,
        "decision": decision,
        "positive_findings": [
            "The midpoint common-frame face map is constructed directly from the exact Stage-67 distributions with no solver rerun.",
            "The conservative face-to-cell divergence is evaluated with the unchanged Stage-76 construction and exact Stage-75 cell residual as reference.",
        ],
        "negative_findings": [
            "Face-flux coherence and cancellation are residual-structure diagnostics, not adjoint sensitivities and not evidence that a modified solver improves q_av.",
            "The failed Stage-28 MUSCL endpoint remains unrecovered and is not extended across Knudsen number.",
            "No physical or numerical parameter is retuned in this stage.",
        ],
        "scientifically_justified_next_scope": (
            "If the common-frame face field is both strongly cancelling and adjacent-face coherent under the preregistered guards, decompose its discrete face gradient into fixed velocity moments. Otherwise localize the noncoherent face contribution spatially before any solver experiment."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage67-artifact-dir", required=True)
    parser.add_argument("--stage75-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run_stage77(args.stage67_artifact_dir, args.stage75_artifact_dir, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
