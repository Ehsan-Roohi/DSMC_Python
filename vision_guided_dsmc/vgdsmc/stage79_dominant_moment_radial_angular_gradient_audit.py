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
    "artifact_id": 8931272132,
    "artifact_size_bytes": 173096061,
    "artifact_sha256": "5b871e94bdca02bce33846657f8f4dc263d68af9e671d7eb137ca7c83da198d4",
    "source_head_sha": "87e6ca98637754e72482b897492147edfcfcf4d9",
    "summary_sha256": "e04043a1913b2fa9ae57fe1561aa26c70627830d648e91204093c8f1fb57b3d1",
    "distributions_sha256": "d4002a2765137ba517abec2d0483a3e5adcf13f415c53259018556bd14d612d1",
    "decision": "stage67_frozen_replay_and_residual_balance_close_stage68_independent_transport_operator_residual_audit",
}

STAGE78_COMPLETED_ENDPOINT = {
    "workflow_run_id": 31144240478,
    "workflow_job_id": 92760312476,
    "workflow_conclusion": "success",
    "tests_passed": 216,
    "tests_failed": 0,
    "artifact_id": 8986364040,
    "artifact_size_bytes": 231475,
    "artifact_sha256": "b65d7f9a7eee8324e3b06d371857480ef339c8874648aac366b6a900eb698d3c",
    "source_head_sha": "6794782ed71d7d7e1b727a231668989e87fefbed",
    "summary_sha256": "1ee6689d43d8fb25455583a074d30cb4bec908386a89297b4e6a57d88400e880",
    "maps_sha256": "934116951290214c3ac460ecb17387466c28a0bc169071caf53a285cf0ec7b30",
    "decision": "stage78_dominant_coherent_moment_stage79_dominant_moment_radial_angular_gradient_audit",
}

GRID = (64, 64)
KNUDSEN = 10.0
COLD_HOT_RATIO = 0.1
RULE = (40, 96)
RADIAL_SCALE = 2.0
POINT_COUNT = RULE[0] * RULE[1]
CHUNK_SIZE = 128
LIMITER = "minmod"
DOMINANT_MOMENT = "transverse_kinetic"
DOMINANT_MOMENT_INDEX = 1
RADIAL_SHELL_COUNT = 4
RADIAL_NODES_PER_SHELL = RULE[0] // RADIAL_SHELL_COUNT
ANGULAR_BIN_COUNT = 8
ANGULAR_BIN_OFFSET_RADIANS = 0.0
VERTICAL_OBLIQUE_BINS = (1, 2, 5, 6)
TOP_TWO_RADIAL_SHELL_CONCENTRATION_GUARD = 0.65
VERTICAL_OBLIQUE_CONCENTRATION_GUARD = 0.70
DOMINANT_RADIAL_SHELL_SHARE_GUARD = 0.50
CLOSURE_GUARD = 1.0e-10


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_stage79_design(
    grid: tuple[int, int] = GRID,
    kn0: float = KNUDSEN,
    cold_hot_ratio: float = COLD_HOT_RATIO,
    rule: tuple[int, int] = RULE,
    radial_scale: float = RADIAL_SCALE,
    chunk_size: int = CHUNK_SIZE,
    limiter: str = LIMITER,
    dominant_moment: str = DOMINANT_MOMENT,
    radial_shell_count: int = RADIAL_SHELL_COUNT,
    angular_bin_count: int = ANGULAR_BIN_COUNT,
    angular_bin_offset_radians: float = ANGULAR_BIN_OFFSET_RADIANS,
    vertical_oblique_bins: tuple[int, ...] = VERTICAL_OBLIQUE_BINS,
    top_two_radial_shell_concentration_guard: float = TOP_TWO_RADIAL_SHELL_CONCENTRATION_GUARD,
    vertical_oblique_concentration_guard: float = VERTICAL_OBLIQUE_CONCENTRATION_GUARD,
    dominant_radial_shell_share_guard: float = DOMINANT_RADIAL_SHELL_SHARE_GUARD,
) -> None:
    actual = (
        grid, kn0, cold_hot_ratio, rule, radial_scale, chunk_size, limiter,
        dominant_moment, radial_shell_count, angular_bin_count,
        angular_bin_offset_radians, vertical_oblique_bins,
        top_two_radial_shell_concentration_guard,
        vertical_oblique_concentration_guard, dominant_radial_shell_share_guard,
    )
    expected = (
        GRID, KNUDSEN, COLD_HOT_RATIO, RULE, RADIAL_SCALE, CHUNK_SIZE, LIMITER,
        DOMINANT_MOMENT, RADIAL_SHELL_COUNT, ANGULAR_BIN_COUNT,
        ANGULAR_BIN_OFFSET_RADIANS, VERTICAL_OBLIQUE_BINS,
        TOP_TWO_RADIAL_SHELL_CONCENTRATION_GUARD,
        VERTICAL_OBLIQUE_CONCENTRATION_GUARD, DOMINANT_RADIAL_SHELL_SHARE_GUARD,
    )
    if actual != expected:
        raise ValueError(
            "Stage 79 is frozen to the exact completed Stage-67 distributions and Stage-78 "
            "transverse-kinetic endpoint, the unchanged 40x96 radial-scale-2.0 rule, four "
            "equal-radial-node shells, eight zero-offset angular bins, and guards retained "
            "from Stages 73-74/78; no solver or parameter retuning is permitted."
        )
    if RULE[0] % RADIAL_SHELL_COUNT != 0:
        raise ValueError("Frozen radial node count must divide exactly into Stage-79 shells")


def _validate_artifact(
    root: str | Path,
    endpoint: dict[str, object],
    files: dict[str, str],
    stage: int,
) -> dict[str, object]:
    root = Path(root)
    for name, checksum_key in files.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != str(endpoint[checksum_key]):
            raise ValueError(f"Completed Stage-{stage} artifact checksum mismatch: {name}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("stage") != stage or summary.get("decision") != endpoint["decision"]:
        raise ValueError(f"Stage-{stage} completed endpoint mismatch")
    return summary


def minmod(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    same = left * right > 0.0
    return np.where(same, np.sign(left) * np.minimum(np.abs(left), np.abs(right)), 0.0)


def limited_slopes_x(distribution: np.ndarray) -> np.ndarray:
    distribution = np.asarray(distribution, dtype=np.float64)
    slope = np.zeros_like(distribution)
    if distribution.shape[1] > 2:
        slope[:, 1:-1] = minmod(
            distribution[:, 1:-1] - distribution[:, :-2],
            distribution[:, 2:] - distribution[:, 1:-1],
        )
    return slope


def interior_x_face_flux_difference_chunk(distribution: np.ndarray, vx: np.ndarray) -> np.ndarray:
    distribution = np.asarray(distribution, dtype=np.float64)
    vx = np.asarray(vx, dtype=np.float64)
    slope = limited_slopes_x(distribution)
    delta = np.zeros((distribution.shape[0], distribution.shape[1] - 1, distribution.shape[2]), dtype=np.float64)
    positive = vx > 0.0
    negative = vx < 0.0
    if np.any(positive):
        delta[..., positive] = 0.5 * vx[positive][None, None, :] * slope[:, :-1, positive]
    if np.any(negative):
        delta[..., negative] = -0.5 * vx[negative][None, None, :] * slope[:, 1:, negative]
    return delta


def macroscopic_v(phi: np.ndarray, vy: np.ndarray, weight: np.ndarray, chunk_size: int = CHUNK_SIZE) -> np.ndarray:
    rho = np.zeros(phi.shape[:2], dtype=np.float64)
    my = np.zeros_like(rho)
    for start in range(0, weight.size, chunk_size):
        stop = min(start + chunk_size, weight.size)
        sl = slice(start, stop)
        weighted = phi[..., sl] * weight[sl][None, None, :]
        rho += np.sum(weighted, axis=-1)
        my += np.sum(weighted * vy[sl][None, None, :], axis=-1)
    return my / np.maximum(rho, 1.0e-300)


def radial_shell_indices(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    speed = np.hypot(np.asarray(vx, dtype=np.float64), np.asarray(vy, dtype=np.float64))
    if speed.ndim != 1 or speed.size != POINT_COUNT:
        raise ValueError("Stage 79 requires the exact 3840-point velocity rule")
    order = np.argsort(speed, kind="stable")
    shell_size = speed.size // RADIAL_SHELL_COUNT
    labels = np.empty(speed.size, dtype=np.int8)
    labels[order] = np.repeat(np.arange(RADIAL_SHELL_COUNT, dtype=np.int8), shell_size)
    return labels


def angular_bin_indices(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    angle = np.mod(np.arctan2(vy, vx) - ANGULAR_BIN_OFFSET_RADIANS, 2.0 * math.pi)
    width = 2.0 * math.pi / ANGULAR_BIN_COUNT
    return np.floor(angle / width).astype(np.int16) % ANGULAR_BIN_COUNT


def divergence_from_interior_faces(face_flux: np.ndarray) -> np.ndarray:
    face_flux = np.asarray(face_flux, dtype=np.float64)
    if face_flux.shape != (GRID[0], GRID[1] - 1):
        raise ValueError("Stage 79 requires a 64x63 interior x-face map")
    cell = np.zeros(GRID, dtype=np.float64)
    cell[:, :-1] -= face_flux
    cell[:, 1:] += face_flux
    return cell


def dominant_moment_group_face_maps(
    phi: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    weight: np.ndarray,
    chunk_size: int = CHUNK_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if phi.shape != (GRID[0], GRID[1], POINT_COUNT):
        raise ValueError("Stage 79 requires exact 64x64x3840 phi")
    v = macroscopic_v(phi, vy, weight, chunk_size)
    v_mid = 0.5 * (v[:, :-1] + v[:, 1:])
    shells = radial_shell_indices(vx, vy)
    bins = angular_bin_indices(vx, vy)
    groups = np.zeros((RADIAL_SHELL_COUNT, ANGULAR_BIN_COUNT, GRID[0], GRID[1] - 1), dtype=np.float64)
    dx = 1.0 / GRID[1]
    for start in range(0, weight.size, chunk_size):
        stop = min(start + chunk_size, weight.size)
        sl = slice(start, stop)
        delta_phi = interior_x_face_flux_difference_chunk(phi[..., sl], vx[sl])
        B = vy[sl][None, None, :] - v_mid[..., None]
        weighted = delta_phi * (0.5 * B * B * B) * weight[sl][None, None, :] / dx
        chunk_shells = shells[sl]
        chunk_bins = bins[sl]
        for shell in range(RADIAL_SHELL_COUNT):
            for angular_bin in range(ANGULAR_BIN_COUNT):
                selected = (chunk_shells == shell) & (chunk_bins == angular_bin)
                if np.any(selected):
                    groups[shell, angular_bin] += np.sum(weighted[..., selected], axis=-1)
    return groups, shells, bins


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.size == 0 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


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


def attribution_metrics(face_groups: np.ndarray, cell_groups: np.ndarray, vx: np.ndarray, vy: np.ndarray, shell_labels: np.ndarray) -> dict[str, object]:
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
    dominant_shell = int(np.argmax(radial_cell_share))
    dominant_joint = np.unravel_index(int(np.argmax(joint_cell_share)), joint_cell_share.shape)
    dominant_shell_face = np.sum(face_groups[dominant_shell], axis=0)
    dominant_shell_cell = np.sum(cell_groups[dominant_shell], axis=0)
    speed = np.hypot(vx, vy)
    shell_metadata = []
    for shell in range(RADIAL_SHELL_COUNT):
        selected = shell_labels == shell
        shell_metadata.append({
            "shell": shell,
            "velocity_point_count": int(np.sum(selected)),
            "minimum_speed": float(np.min(speed[selected])),
            "maximum_speed": float(np.max(speed[selected])),
            "mean_speed": float(np.mean(speed[selected])),
            "face_absolute_share": float(radial_face_share[shell]),
            "cell_divergence_absolute_share": float(radial_cell_share[shell]),
            "face_to_cell_cancellation_ratio": float(radial_cell[shell] / max(2.0 * radial_face[shell], 1.0e-300)),
        })
    width = 2.0 * math.pi / ANGULAR_BIN_COUNT
    angular_metadata = []
    for angular_bin in range(ANGULAR_BIN_COUNT):
        angular_metadata.append({
            "bin": angular_bin,
            "start_degrees": float(math.degrees(angular_bin * width + ANGULAR_BIN_OFFSET_RADIANS)),
            "end_degrees": float(math.degrees((angular_bin + 1) * width + ANGULAR_BIN_OFFSET_RADIANS)),
            "face_absolute_share": float(angular_face_share[angular_bin]),
            "cell_divergence_absolute_share": float(angular_cell_share[angular_bin]),
        })
    top_two_radial = float(np.sum(np.sort(radial_cell_share)[-2:]))
    vertical_oblique = float(np.sum(angular_cell_share[list(VERTICAL_OBLIQUE_BINS)]))
    return {
        "joint_face_absolute_share": joint_face_share.tolist(),
        "joint_cell_divergence_absolute_share": joint_cell_share.tolist(),
        "radial_shells": shell_metadata,
        "angular_bins": angular_metadata,
        "top_two_radial_shell_cell_divergence_share": top_two_radial,
        "vertical_oblique_cell_divergence_share": vertical_oblique,
        "dominant_radial_shell": dominant_shell,
        "dominant_radial_shell_cell_divergence_share": float(radial_cell_share[dominant_shell]),
        "dominant_radial_shell_adjacent_x_face_correlation": _corr(dominant_shell_face[:, :-1], dominant_shell_face[:, 1:]),
        "dominant_radial_shell_spatial_shares": _spatial_shares(dominant_shell_cell),
        "dominant_joint_group": [int(dominant_joint[0]), int(dominant_joint[1])],
        "dominant_joint_group_cell_divergence_share": float(joint_cell_share[dominant_joint]),
        "total_group_face_absolute_sum": float(face_total),
        "total_group_cell_divergence_absolute_sum": float(cell_total),
    }


def stage79_decision(finite: bool, closure_closed: bool, metrics: dict[str, object]) -> str:
    if not finite:
        return "stage79_nonfinite_dominant_moment_velocity_attribution_blocker"
    if not closure_closed:
        return "stage79_dominant_moment_velocity_attribution_closure_blocker"
    dominant_shell_share = float(metrics["dominant_radial_shell_cell_divergence_share"])
    top_two_radial = float(metrics["top_two_radial_shell_cell_divergence_share"])
    vertical_oblique = float(metrics["vertical_oblique_cell_divergence_share"])
    if dominant_shell_share >= DOMINANT_RADIAL_SHELL_SHARE_GUARD and vertical_oblique >= VERTICAL_OBLIQUE_CONCENTRATION_GUARD:
        return "stage79_dominant_radial_shell_vertical_oblique_stage80_dominant_shell_radial_node_angular_attribution_audit"
    if top_two_radial >= TOP_TWO_RADIAL_SHELL_CONCENTRATION_GUARD:
        return "stage79_radially_concentrated_angularly_mixed_stage80_top_shell_angular_cancellation_audit"
    if vertical_oblique >= VERTICAL_OBLIQUE_CONCENTRATION_GUARD:
        return "stage79_angularly_concentrated_radially_mixed_stage80_vertical_oblique_radial_node_audit"
    return "stage79_velocity_attribution_diffuse_stage80_dominant_moment_spatial_localization_audit"


def run_stage79(stage67_artifact_dir: str | Path, stage78_artifact_dir: str | Path, output_dir: str | Path, **design: object) -> dict[str, object]:
    validate_stage79_design(**design)
    retained67 = _validate_artifact(
        stage67_artifact_dir,
        STAGE67_COMPLETED_ENDPOINT,
        {"summary.json": "summary_sha256", "converged_full_distributions.npz": "distributions_sha256"},
        67,
    )
    retained78 = _validate_artifact(
        stage78_artifact_dir,
        STAGE78_COMPLETED_ENDPOINT,
        {"summary.json": "summary_sha256", "face_gradient_moment_attribution_maps.npz": "maps_sha256"},
        78,
    )
    with np.load(Path(stage67_artifact_dir) / "converged_full_distributions.npz") as data:
        phi = np.asarray(data["phi"], dtype=np.float64)
        vx = np.asarray(data["vx"], dtype=np.float64)
        vy = np.asarray(data["vy"], dtype=np.float64)
        weight = np.asarray(data["weight"], dtype=np.float64)
    with np.load(Path(stage78_artifact_dir) / "face_gradient_moment_attribution_maps.npz") as data:
        ref_face = np.asarray(data["face_components"][DOMINANT_MOMENT_INDEX], dtype=np.float64)
        ref_cell = np.asarray(data["cell_divergence_components"][DOMINANT_MOMENT_INDEX], dtype=np.float64)
    face_groups, shell_labels, angular_labels = dominant_moment_group_face_maps(phi, vx, vy, weight)
    cell_groups = np.stack(
        [[divergence_from_interior_faces(face_groups[shell, angular_bin]) for angular_bin in range(ANGULAR_BIN_COUNT)] for shell in range(RADIAL_SHELL_COUNT)],
        axis=0,
    )
    reconstructed_face = np.sum(face_groups, axis=(0, 1))
    reconstructed_cell = np.sum(cell_groups, axis=(0, 1))
    face_delta = reconstructed_face - ref_face
    cell_delta = reconstructed_cell - ref_cell
    face_rel = float(np.linalg.norm(face_delta.ravel()) / max(float(np.linalg.norm(ref_face.ravel())), 1.0e-300))
    cell_rel = float(np.linalg.norm(cell_delta.ravel()) / max(float(np.linalg.norm(ref_cell.ravel())), 1.0e-300))
    closure_max = max(float(np.max(np.abs(face_delta))), float(np.max(np.abs(cell_delta))))
    metrics = attribution_metrics(face_groups, cell_groups, vx, vy, shell_labels)
    finite = bool(
        np.all(np.isfinite(face_groups)) and np.all(np.isfinite(cell_groups))
        and np.all(np.isfinite(reconstructed_face)) and np.all(np.isfinite(reconstructed_cell))
        and np.isfinite(face_rel) and np.isfinite(cell_rel)
    )
    closure_closed = bool(max(face_rel, cell_rel) <= CLOSURE_GUARD)
    decision = stage79_decision(finite, closure_closed, metrics)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "dominant_moment_radial_angular_gradient_maps.npz",
        dominant_moment=np.asarray(DOMINANT_MOMENT),
        radial_shell_labels=shell_labels,
        angular_bin_labels=angular_labels,
        face_groups=face_groups,
        cell_divergence_groups=cell_groups,
        reconstructed_dominant_face=reconstructed_face,
        reconstructed_dominant_cell=reconstructed_cell,
        retained_stage78_dominant_face=ref_face,
        retained_stage78_dominant_cell=ref_cell,
    )
    summary = {
        "stage": 79,
        "description": "Frozen radial-shell and angular-sector attribution of the Stage-78 dominant transverse-kinetic common-frame face gradient.",
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
            "radial_shell_count": RADIAL_SHELL_COUNT,
            "radial_nodes_per_shell": RADIAL_NODES_PER_SHELL,
            "angular_bin_count": ANGULAR_BIN_COUNT,
            "angular_bin_offset_radians": ANGULAR_BIN_OFFSET_RADIANS,
            "vertical_oblique_bins": list(VERTICAL_OBLIQUE_BINS),
            "top_two_radial_shell_concentration_guard": TOP_TWO_RADIAL_SHELL_CONCENTRATION_GUARD,
            "vertical_oblique_concentration_guard": VERTICAL_OBLIQUE_CONCENTRATION_GUARD,
            "dominant_radial_shell_share_guard": DOMINANT_RADIAL_SHELL_SHARE_GUARD,
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
        "retained_stage78_decision": retained78["decision"],
        "dominant_moment_closure": {
            "maximum_absolute_error": closure_max,
            "face_relative_l2_error": face_rel,
            "cell_relative_l2_error": cell_rel,
            "within_guard": closure_closed,
        },
        "radial_angular_metrics": metrics,
        "finite": finite,
        "closure_closed": closure_closed,
        "decision": decision,
        "positive_findings": [
            "The Stage-78 dominant transverse-kinetic moment is decomposed using the already-fixed Stage-73 angular bins and Stage-74 radial-shell convention.",
            "The radial/angular reconstruction is checked directly against the exact completed Stage-78 face and cell-divergence maps without rerunning the cavity solver.",
        ],
        "negative_findings": [
            "Velocity-space concentration is a frozen residual-structure diagnostic, not an adjoint sensitivity and not evidence that modifying quadrature or transport would improve q_av.",
            "The failed Stage-28 MUSCL endpoint remains unrecovered and is not extended across Knudsen number.",
            "No physical, collision, correction-floor, source-relaxation, transport, wall, normalization, or velocity-quadrature parameter is retuned.",
        ],
        "scientifically_justified_next_scope": (
            "If the dominant transverse-kinetic residual is concentrated in one retained radial shell and the retained vertical-oblique sectors, resolve only that shell into its ten fixed radial nodes and the same eight angular sectors. Otherwise follow the preregistered Stage-79 decision branch without any solver experiment."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage67-artifact-dir", required=True)
    parser.add_argument("--stage78-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run_stage79(args.stage67_artifact_dir, args.stage78_artifact_dir, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
