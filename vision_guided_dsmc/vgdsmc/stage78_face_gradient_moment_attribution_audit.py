from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

STAGE77_COMPLETED_ENDPOINT = {
    "workflow_run_id": 31136773318,
    "workflow_job_id": 92737757646,
    "workflow_conclusion": "success",
    "tests_passed": 209,
    "tests_failed": 0,
    "test_duration_seconds": 0.93,
    "artifact_id": 8980744966,
    "artifact_size_bytes": 139529,
    "artifact_sha256": "3013020014ba35eb67ab62dd4be71548d8469bcc532cddc5f57b68785b23dfc4",
    "source_head_sha": "13a9b5ba07bfed0eb5246df600cf88abeecbc086",
    "summary_sha256": "2f36cab682389596d1e11d44ea83f5747c4ec67174486f04e9111163768d35a3",
    "maps_sha256": "9bc869d757cb54b6a32e4e670ace9ead02725d2491904d2c1ac922bd6b7a678e",
    "decision": "stage77_coherent_face_cancellation_stage78_face_gradient_moment_attribution_audit",
}

GRID = (64, 64)
KNUDSEN = 10.0
COLD_HOT_RATIO = 0.1
RULE = (40, 96)
RADIAL_SCALE = 2.0
LIMITER = "minmod"
MOMENT_NAMES = ("streamwise_kinetic", "transverse_kinetic", "reduced_internal")
DOMINANT_CELL_DIVERGENCE_SHARE_GUARD = 0.50
COMPONENT_FACE_CORRELATION_GUARD = 0.90
INTERCOMPONENT_CANCELLATION_RATIO_GUARD = 0.75
CLOSURE_GUARD = 1.0e-12


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_stage78_design(
    grid: tuple[int, int] = GRID,
    kn0: float = KNUDSEN,
    cold_hot_ratio: float = COLD_HOT_RATIO,
    rule: tuple[int, int] = RULE,
    radial_scale: float = RADIAL_SCALE,
    limiter: str = LIMITER,
    dominant_cell_divergence_share_guard: float = DOMINANT_CELL_DIVERGENCE_SHARE_GUARD,
    component_face_correlation_guard: float = COMPONENT_FACE_CORRELATION_GUARD,
    intercomponent_cancellation_ratio_guard: float = INTERCOMPONENT_CANCELLATION_RATIO_GUARD,
) -> None:
    actual = (
        grid,
        kn0,
        cold_hot_ratio,
        rule,
        radial_scale,
        limiter,
        dominant_cell_divergence_share_guard,
        component_face_correlation_guard,
        intercomponent_cancellation_ratio_guard,
    )
    expected = (
        GRID,
        KNUDSEN,
        COLD_HOT_RATIO,
        RULE,
        RADIAL_SCALE,
        LIMITER,
        DOMINANT_CELL_DIVERGENCE_SHARE_GUARD,
        COMPONENT_FACE_CORRELATION_GUARD,
        INTERCOMPONENT_CANCELLATION_RATIO_GUARD,
    )
    if actual != expected:
        raise ValueError(
            "Stage 78 is frozen to the exact completed Stage-77 common-frame face map, "
            "its three fixed heat-flux moment components, the 50% dominance guard, the "
            "unchanged 0.90 adjacent-face coherence guard, and a preregistered 0.75 "
            "intercomponent-cancellation ratio guard; no solver or parameter retuning is permitted."
        )


def _validate_stage77_artifact(root: str | Path) -> dict[str, object]:
    root = Path(root)
    files = {
        "summary.json": "summary_sha256",
        "common_frame_face_flux_divergence_maps.npz": "maps_sha256",
    }
    for name, checksum_key in files.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != STAGE77_COMPLETED_ENDPOINT[checksum_key]:
            raise ValueError(f"Completed Stage-77 artifact checksum mismatch: {name}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("stage") != 77 or summary.get("decision") != STAGE77_COMPLETED_ENDPOINT["decision"]:
        raise ValueError("Stage-77 completed endpoint mismatch")
    return summary


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.size == 0 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def divergence_from_interior_faces(face_flux: np.ndarray) -> np.ndarray:
    face_flux = np.asarray(face_flux, dtype=np.float64)
    if face_flux.shape != (GRID[0], GRID[1] - 1):
        raise ValueError("Stage 78 requires a 64x63 interior x-face map")
    cell = np.zeros(GRID, dtype=np.float64)
    cell[:, :-1] -= face_flux
    cell[:, 1:] += face_flux
    return cell


def component_gradient_maps(face_components: np.ndarray) -> np.ndarray:
    face_components = np.asarray(face_components, dtype=np.float64)
    if face_components.shape != (len(MOMENT_NAMES), GRID[0], GRID[1] - 1):
        raise ValueError("Stage 78 requires three 64x63 fixed moment face maps")
    return np.stack([divergence_from_interior_faces(face) for face in face_components], axis=0)


def _spatial_shares(cell_map: np.ndarray) -> dict[str, float]:
    a = np.abs(np.asarray(cell_map, dtype=np.float64))
    total = max(float(np.sum(a)), 1.0e-300)
    outer2 = (
        np.sum(a[:2, :]) + np.sum(a[-2:, :]) + np.sum(a[:, :2]) + np.sum(a[:, -2:])
        - np.sum(a[:2, :2]) - np.sum(a[:2, -2:])
        - np.sum(a[-2:, :2]) - np.sum(a[-2:, -2:])
    )
    return {
        "outer_two_cell_wall_share": float(outer2 / total),
        "interior_two_cell_complement_share": float(1.0 - outer2 / total),
    }


def component_metrics(face_components: np.ndarray, cell_components: np.ndarray) -> dict[str, object]:
    face_components = np.asarray(face_components, dtype=np.float64)
    cell_components = np.asarray(cell_components, dtype=np.float64)
    face_abs = np.sum(np.abs(face_components), axis=(1, 2))
    cell_abs = np.sum(np.abs(cell_components), axis=(1, 2))
    face_abs_total = max(float(np.sum(face_abs)), 1.0e-300)
    cell_abs_total = max(float(np.sum(cell_abs)), 1.0e-300)
    total_face = np.sum(face_components, axis=0)
    total_cell = np.sum(cell_components, axis=0)
    per_component: dict[str, dict[str, float]] = {}
    for index, name in enumerate(MOMENT_NAMES):
        face = face_components[index]
        cell = cell_components[index]
        spatial = _spatial_shares(cell)
        per_component[name] = {
            "face_absolute_sum": float(face_abs[index]),
            "face_absolute_share": float(face_abs[index] / face_abs_total),
            "cell_divergence_absolute_sum": float(cell_abs[index]),
            "cell_divergence_absolute_share": float(cell_abs[index] / cell_abs_total),
            "face_to_cell_cancellation_ratio": float(cell_abs[index] / max(2.0 * face_abs[index], 1.0e-300)),
            "adjacent_x_face_correlation": _corr(face[:, :-1], face[:, 1:]),
            "adjacent_y_row_correlation": _corr(face[:-1, :], face[1:, :]),
            "cell_rms": float(np.sqrt(np.mean(cell * cell))),
            "cell_signed_sum": float(np.sum(cell)),
            **spatial,
        }
    dominant_index = int(np.argmax(cell_abs))
    return {
        "moment_names": list(MOMENT_NAMES),
        "per_component": per_component,
        "dominant_component": MOMENT_NAMES[dominant_index],
        "dominant_component_index": dominant_index,
        "dominant_cell_divergence_absolute_share": float(cell_abs[dominant_index] / cell_abs_total),
        "intercomponent_face_cancellation_ratio": float(np.sum(np.abs(total_face)) / face_abs_total),
        "intercomponent_cell_divergence_cancellation_ratio": float(np.sum(np.abs(total_cell)) / cell_abs_total),
        "total_component_face_absolute_sum": float(face_abs_total),
        "total_component_cell_divergence_absolute_sum": float(cell_abs_total),
    }


def stage78_decision(
    finite: bool,
    closure_closed: bool,
    dominant_share: float,
    dominant_adjacent_x_correlation: float,
    intercomponent_cell_cancellation_ratio: float,
) -> str:
    if not finite:
        return "stage78_nonfinite_face_gradient_moment_blocker"
    if not closure_closed:
        return "stage78_moment_gradient_closure_blocker"
    if dominant_share >= DOMINANT_CELL_DIVERGENCE_SHARE_GUARD:
        if dominant_adjacent_x_correlation >= COMPONENT_FACE_CORRELATION_GUARD:
            return "stage78_dominant_coherent_moment_stage79_dominant_moment_radial_angular_gradient_audit"
        return "stage78_dominant_noncoherent_moment_stage79_dominant_moment_spatial_localization_audit"
    if intercomponent_cell_cancellation_ratio <= INTERCOMPONENT_CANCELLATION_RATIO_GUARD:
        return "stage78_mixed_moment_cancellation_stage79_cross_moment_cancellation_audit"
    return "stage78_mixed_moment_divergence_stage79_joint_moment_spatial_localization_audit"


def run_stage78(stage77_artifact_dir: str | Path, output_dir: str | Path, **design: object) -> dict[str, object]:
    validate_stage78_design(**design)
    retained77 = _validate_stage77_artifact(stage77_artifact_dir)
    with np.load(Path(stage77_artifact_dir) / "common_frame_face_flux_divergence_maps.npz") as maps:
        face_components = np.asarray(maps["midpoint_common_frame_face_components"], dtype=np.float64)
        retained_total_face = np.asarray(maps["midpoint_common_frame_total_face_flux"], dtype=np.float64)
        retained_total_cell = np.asarray(maps["midpoint_common_frame_cell_divergence"], dtype=np.float64)
    cell_components = component_gradient_maps(face_components)
    reconstructed_face = np.sum(face_components, axis=0)
    reconstructed_cell = np.sum(cell_components, axis=0)
    face_delta = reconstructed_face - retained_total_face
    cell_delta = reconstructed_cell - retained_total_cell
    face_rel = float(np.linalg.norm(face_delta.ravel()) / max(float(np.linalg.norm(retained_total_face.ravel())), 1.0e-300))
    cell_rel = float(np.linalg.norm(cell_delta.ravel()) / max(float(np.linalg.norm(retained_total_cell.ravel())), 1.0e-300))
    closure_max = max(float(np.max(np.abs(face_delta))), float(np.max(np.abs(cell_delta))))
    metrics = component_metrics(face_components, cell_components)
    dominant_name = str(metrics["dominant_component"])
    dominant_corr = float(metrics["per_component"][dominant_name]["adjacent_x_face_correlation"])
    finite = bool(
        np.all(np.isfinite(face_components))
        and np.all(np.isfinite(cell_components))
        and np.all(np.isfinite(reconstructed_face))
        and np.all(np.isfinite(reconstructed_cell))
        and np.isfinite(face_rel)
        and np.isfinite(cell_rel)
        and all(np.isfinite(value) for value in (
            metrics["dominant_cell_divergence_absolute_share"],
            metrics["intercomponent_face_cancellation_ratio"],
            metrics["intercomponent_cell_divergence_cancellation_ratio"],
            dominant_corr,
        ))
    )
    closure_closed = bool(max(face_rel, cell_rel) <= CLOSURE_GUARD)
    decision = stage78_decision(
        finite,
        closure_closed,
        float(metrics["dominant_cell_divergence_absolute_share"]),
        dominant_corr,
        float(metrics["intercomponent_cell_divergence_cancellation_ratio"]),
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "face_gradient_moment_attribution_maps.npz",
        moment_names=np.asarray(MOMENT_NAMES),
        face_components=face_components,
        cell_divergence_components=cell_components,
        reconstructed_total_face=reconstructed_face,
        reconstructed_total_cell=reconstructed_cell,
        retained_stage77_total_face=retained_total_face,
        retained_stage77_total_cell=retained_total_cell,
    )
    summary = {
        "stage": 78,
        "description": "Frozen attribution of the coherent Stage-77 common-frame face-gradient into the three fixed heat-flux velocity moments.",
        "configuration": {
            "grid": list(GRID),
            "kn0": KNUDSEN,
            "cold_hot_ratio": COLD_HOT_RATIO,
            "radial_nodes": RULE[0],
            "angular_nodes": RULE[1],
            "point_count": RULE[0] * RULE[1],
            "radial_scale": RADIAL_SCALE,
            "limiter": LIMITER,
            "moment_names": list(MOMENT_NAMES),
            "dominant_cell_divergence_share_guard": DOMINANT_CELL_DIVERGENCE_SHARE_GUARD,
            "component_face_correlation_guard": COMPONENT_FACE_CORRELATION_GUARD,
            "intercomponent_cancellation_ratio_guard": INTERCOMPONENT_CANCELLATION_RATIO_GUARD,
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
        "retained_stage77_decision": retained77["decision"],
        "moment_gradient_closure": {
            "maximum_absolute_error": closure_max,
            "face_relative_l2_error": face_rel,
            "cell_relative_l2_error": cell_rel,
            "within_guard": closure_closed,
        },
        "moment_gradient_metrics": metrics,
        "finite": finite,
        "closure_closed": closure_closed,
        "decision": decision,
        "positive_findings": [
            "The three fixed common-frame heat-flux moment contributions are differentiated conservatively without rerunning the cavity solver.",
            "Their reconstructed face field and cell divergence are compared directly against the exact completed Stage-77 artifact."
        ],
        "negative_findings": [
            "Moment attribution is a residual-structure diagnostic, not an adjoint sensitivity and not evidence that changing a transport discretization improves q_av.",
            "The failed Stage-28 MUSCL endpoint remains unrecovered and is not extended across Knudsen number.",
            "No physical, collision, correction-floor, source-relaxation, transport, wall, normalization, or velocity-quadrature parameter is retuned."
        ],
        "scientifically_justified_next_scope": (
            "If one fixed moment supplies at least half of the componentwise cell-divergence magnitude, follow that moment only: use radial/angular velocity attribution when its adjacent x-face field remains coherent, otherwise spatially localize its noncoherent gradient. If no moment dominates, audit cross-moment cancellation before any solver experiment."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage77-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run_stage78(args.stage77_artifact_dir, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
