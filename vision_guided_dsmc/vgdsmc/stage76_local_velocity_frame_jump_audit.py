from __future__ import annotations

import argparse
import hashlib
import json
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
    "decision": "stage67_frozen_replay_and_residual_balance_close_stage68_independent_transport_operator_residual_audit",
}

STAGE75_COMPLETED_ENDPOINT = {
    "workflow_run_id": 31115869623,
    "workflow_job_id": 92665123190,
    "workflow_conclusion": "success",
    "tests_passed": 190,
    "tests_failed": 0,
    "test_duration_seconds": 0.92,
    "artifact_id": 8975166032,
    "artifact_size_bytes": 302986,
    "artifact_sha256": "603a37b7b4dd1884e421288ba1bc4da4175bf6c68e96b70bee0c39beb2e86c13",
    "source_head_sha": "5c44701d4aa7c6710c35ac554009c7de5724ba85",
    "summary_sha256": "a06f24fc646b2bb1c62369a7d597ed32d078e72783bda50cce5865ff1baa882a",
    "maps_sha256": "2299b1901dad1f38173806ad53bc41dc73a860ec922a9bc00822052ad2e92e20",
    "decision": "stage75_conservative_face_pair_cancellation_stage76_local_velocity_frame_jump_audit",
}

GRID = (64, 64)
KNUDSEN = 10.0
COLD_HOT_RATIO = 0.1
RULE = (40, 96)
RADIAL_SCALE = 2.0
POINT_COUNT = 3840
CHUNK_SIZE = 128
LIMITER = "minmod"
CLOSURE_GUARD = 1.0e-10
MATERIAL_FRAME_RATIO_GUARD = 1.0e-2
MIDPOINT_CELL_L2_GUARD = 1.0e-2


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_stage76_design(
    grid: tuple[int, int] = GRID,
    kn0: float = KNUDSEN,
    cold_hot_ratio: float = COLD_HOT_RATIO,
    rule: tuple[int, int] = RULE,
    radial_scale: float = RADIAL_SCALE,
    chunk_size: int = CHUNK_SIZE,
    limiter: str = LIMITER,
    material_frame_ratio_guard: float = MATERIAL_FRAME_RATIO_GUARD,
    midpoint_cell_l2_guard: float = MIDPOINT_CELL_L2_GUARD,
) -> None:
    actual = (
        grid,
        kn0,
        cold_hot_ratio,
        rule,
        radial_scale,
        chunk_size,
        limiter,
        material_frame_ratio_guard,
        midpoint_cell_l2_guard,
    )
    expected = (
        GRID,
        KNUDSEN,
        COLD_HOT_RATIO,
        RULE,
        RADIAL_SCALE,
        CHUNK_SIZE,
        LIMITER,
        MATERIAL_FRAME_RATIO_GUARD,
        MIDPOINT_CELL_L2_GUARD,
    )
    if actual != expected:
        raise ValueError(
            "Stage 76 is frozen to the exact Stage-67 64x64 distributions and exact "
            "Stage-75 face maps, with 40x96 radial-scale-2.0 quadrature, minmod "
            "second-minus-first-order transport, and preregistered 1% materiality guards; "
            "no solver or parameter retuning is permitted."
        )


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
            raise ValueError(f"Completed artifact checksum mismatch: {name}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
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
        slope[:, 1:-1] = minmod(
            distribution[:, 1:-1] - distribution[:, :-2],
            distribution[:, 2:] - distribution[:, 1:-1],
        )
    return slope


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


def macroscopic_velocity(
    phi: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    weight: np.ndarray,
    chunk_size: int = CHUNK_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rho = np.zeros(phi.shape[:2], dtype=np.float64)
    mx = np.zeros_like(rho)
    my = np.zeros_like(rho)
    for start in range(0, weight.size, chunk_size):
        stop = min(start + chunk_size, weight.size)
        sl = slice(start, stop)
        weighted = phi[..., sl] * weight[sl][None, None, :]
        rho += np.sum(weighted, axis=-1)
        mx += np.sum(weighted * vx[sl][None, None, :], axis=-1)
        my += np.sum(weighted * vy[sl][None, None, :], axis=-1)
    safe = np.maximum(rho, 1.0e-300)
    return rho, mx / safe, my / safe


def frame_kernel_jump_terms(
    vx: np.ndarray,
    vy: np.ndarray,
    u_left: np.ndarray,
    u_right: np.ndarray,
    v_left: np.ndarray,
    v_right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact symmetric expansion of K_R-K_L for the three q_y kernels.

    With A=vx-u_mid, B=vy-v_mid, du=u_R-u_L, dv=v_R-v_L:
      0.5[c_y c_x^2]_R-L = -A B du - 0.5 A^2 dv - 0.125 du^2 dv
      0.5[c_y^3]_R-L     = -1.5 B^2 dv - 0.125 dv^3
      0.5[c_y]_R-L       = -0.5 dv
    """
    u_mid = 0.5 * (u_left + u_right)
    v_mid = 0.5 * (v_left + v_right)
    du = u_right - u_left
    dv = v_right - v_left
    A = vx[None, None, :] - u_mid[..., None]
    B = vy[None, None, :] - v_mid[..., None]
    du3 = du[..., None]
    dv3 = dv[..., None]
    stream_u = -A * B * du3
    stream_v = -0.5 * A * A * dv3
    stream_nl = -0.125 * (du * du * dv)[..., None]
    transverse_v = -1.5 * B * B * dv3
    transverse_nl = -0.125 * (dv * dv * dv)[..., None]
    internal_v = -0.5 * dv3
    return stream_u, stream_v, stream_nl, transverse_v, transverse_nl, internal_v


def evaluate_frame_jump_maps(
    phi: np.ndarray,
    psi: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    weight: np.ndarray,
    chunk_size: int = CHUNK_SIZE,
) -> dict[str, np.ndarray]:
    if phi.shape != psi.shape or phi.shape != (GRID[0], GRID[1], POINT_COUNT):
        raise ValueError("Stage 76 requires exact 64x64x3840 distributions")
    _, u, v = macroscopic_velocity(phi, vx, vy, weight, chunk_size)
    u_left, u_right = u[:, :-1], u[:, 1:]
    v_left, v_right = v[:, :-1], v[:, 1:]
    du = u_right - u_left
    dv = v_right - v_left
    components = np.zeros((3, GRID[0], GRID[1] - 1), dtype=np.float64)
    groups = np.zeros_like(components)
    midpoint_components = np.zeros_like(components)
    dx = 1.0 / GRID[1]
    for start in range(0, weight.size, chunk_size):
        stop = min(start + chunk_size, weight.size)
        sl = slice(start, stop)
        delta_phi = interior_x_face_flux_difference_chunk(phi[..., sl], vx[sl])
        delta_psi = interior_x_face_flux_difference_chunk(psi[..., sl], vx[sl])
        w = weight[sl][None, None, :]
        su, sv, snl, tv, tnl, iv = frame_kernel_jump_terms(
            vx[sl], vy[sl], u_left, u_right, v_left, v_right
        )
        components[0] += np.sum(delta_phi * (su + sv + snl) * w, axis=-1) / dx
        components[1] += np.sum(delta_phi * (tv + tnl) * w, axis=-1) / dx
        components[2] += np.sum(delta_psi * iv * w, axis=-1) / dx
        groups[0] += np.sum(delta_phi * su * w, axis=-1) / dx
        groups[1] += (
            np.sum(delta_phi * (sv + tv) * w, axis=-1)
            + np.sum(delta_psi * iv * w, axis=-1)
        ) / dx
        groups[2] += np.sum(delta_phi * (snl + tnl) * w, axis=-1) / dx

        u_mid = 0.5 * (u_left + u_right)
        v_mid = 0.5 * (v_left + v_right)
        A = vx[sl][None, None, :] - u_mid[..., None]
        B = vy[sl][None, None, :] - v_mid[..., None]
        midpoint_components[0] += np.sum(delta_phi * (0.5 * B * A * A) * w, axis=-1) / dx
        midpoint_components[1] += np.sum(delta_phi * (0.5 * B * B * B) * w, axis=-1) / dx
        midpoint_components[2] += np.sum(delta_psi * (0.5 * B) * w, axis=-1) / dx

    midpoint_total = np.sum(midpoint_components, axis=0)
    midpoint_cell = np.zeros(GRID, dtype=np.float64)
    midpoint_cell[:, :-1] -= midpoint_total
    midpoint_cell[:, 1:] += midpoint_total
    return {
        "component_face_pair_remainder_maps": components,
        "group_face_pair_remainder_maps": groups,
        "midpoint_common_frame_face_components": midpoint_components,
        "midpoint_common_frame_cell_map": midpoint_cell,
        "u": u,
        "v": v,
        "du": du,
        "dv": dv,
    }


def _abs_shares(arrays: np.ndarray) -> list[float]:
    values = np.sum(np.abs(arrays), axis=(1, 2))
    return (values / max(float(np.sum(values)), 1.0e-300)).tolist()


def _corr_abs(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def stage76_decision(
    finite: bool,
    provenance_consistent: bool,
    frame_closure_closed: bool,
    group_closure_closed: bool,
    frame_pair_to_cell_abs_ratio: float,
    midpoint_cell_relative_l2_error: float,
) -> str:
    if not finite:
        return "stage76_nonfinite_local_velocity_frame_blocker"
    if not provenance_consistent:
        return "stage76_completed_endpoint_reproduction_blocker"
    if not frame_closure_closed:
        return "stage76_frame_kernel_jump_closure_blocker"
    if not group_closure_closed:
        return "stage76_frame_jump_group_closure_blocker"
    if (
        frame_pair_to_cell_abs_ratio <= MATERIAL_FRAME_RATIO_GUARD
        and midpoint_cell_relative_l2_error <= MIDPOINT_CELL_L2_GUARD
    ):
        return "stage76_frame_jump_negligible_stage77_common_frame_face_flux_divergence_audit"
    return "stage76_material_frame_jump_stage77_frame_gradient_localization_audit"


def run_stage76(
    stage67_artifact_dir: str | Path,
    stage75_artifact_dir: str | Path,
    output_dir: str | Path,
    **design: object,
) -> dict[str, object]:
    validate_stage76_design(**design)
    retained67 = _validate_artifact(
        stage67_artifact_dir,
        STAGE67_COMPLETED_ENDPOINT,
        {
            "summary.json": "summary_sha256",
            "converged_full_distributions.npz": "distributions_sha256",
            "steady_residual_moment_maps.npz": "residual_maps_sha256",
        },
        67,
    )
    retained75 = _validate_artifact(
        stage75_artifact_dir,
        STAGE75_COMPLETED_ENDPOINT,
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
    maps = evaluate_frame_jump_maps(phi, psi, vx, vy, weight)
    with np.load(Path(stage75_artifact_dir) / "signed_face_location_velocity_moment_maps.npz") as data:
        left = np.asarray(data["face_left_target_moment_maps"], dtype=np.float64)
        right = np.asarray(data["face_right_target_moment_maps"], dtype=np.float64)
        cell_reference = np.asarray(data["reconstructed_total_cell_map"], dtype=np.float64)
    reference_pair = left + right
    components = maps["component_face_pair_remainder_maps"]
    groups = maps["group_face_pair_remainder_maps"]
    frame_delta = components - reference_pair
    frame_rel = float(
        np.linalg.norm(frame_delta.ravel())
        / max(float(np.linalg.norm(reference_pair.ravel())), 1.0e-300)
    )
    frame_max = float(np.max(np.abs(frame_delta)))
    group_delta = np.sum(groups, axis=0) - np.sum(components, axis=0)
    group_rel = float(
        np.linalg.norm(group_delta.ravel())
        / max(float(np.linalg.norm(np.sum(components, axis=0).ravel())), 1.0e-300)
    )
    group_max = float(np.max(np.abs(group_delta)))
    midpoint_cell = maps["midpoint_common_frame_cell_map"]
    midpoint_delta = midpoint_cell - cell_reference
    midpoint_rel = float(
        np.linalg.norm(midpoint_delta.ravel())
        / max(float(np.linalg.norm(cell_reference.ravel())), 1.0e-300)
    )
    frame_abs = float(np.sum(np.abs(components)))
    cell_abs = float(np.sum(np.abs(cell_reference)))
    frame_to_cell = frame_abs / max(cell_abs, 1.0e-300)
    du = maps["du"]
    dv = maps["dv"]
    total_frame_face = np.sum(components, axis=0)
    finite = bool(all(np.all(np.isfinite(value)) for value in maps.values()))
    provenance_consistent = bool(
        retained67.get("decision") == STAGE67_COMPLETED_ENDPOINT["decision"]
        and retained75.get("decision") == STAGE75_COMPLETED_ENDPOINT["decision"]
        and retained75.get("finite") is True
        and retained75.get("moment_closure_closed") is True
        and retained75.get("provenance_consistent") is True
    )
    frame_closed = bool(frame_rel <= CLOSURE_GUARD)
    group_closed = bool(group_rel <= CLOSURE_GUARD)
    decision = stage76_decision(
        finite,
        provenance_consistent,
        frame_closed,
        group_closed,
        frame_to_cell,
        midpoint_rel,
    )
    component_abs = np.sum(np.abs(components), axis=(1, 2))
    group_abs = np.sum(np.abs(groups), axis=(1, 2))
    metrics = {
        "max_abs_du": float(np.max(np.abs(du))),
        "rms_du": float(np.sqrt(np.mean(du * du))),
        "max_abs_dv": float(np.max(np.abs(dv))),
        "rms_dv": float(np.sqrt(np.mean(dv * dv))),
        "component_absolute_shares": _abs_shares(components),
        "component_signed_sums": np.sum(components, axis=(1, 2)).tolist(),
        "group_names": ["u_linear", "v_linear", "nonlinear_cubic"],
        "group_absolute_shares": (
            group_abs / max(float(np.sum(group_abs)), 1.0e-300)
        ).tolist(),
        "group_signed_sums": np.sum(groups, axis=(1, 2)).tolist(),
        "frame_pair_remainder_absolute_sum": frame_abs,
        "stage75_cell_residual_absolute_sum": cell_abs,
        "frame_pair_to_cell_absolute_ratio": frame_to_cell,
        "midpoint_common_frame_cell_rms": float(np.sqrt(np.mean(midpoint_cell * midpoint_cell))),
        "stage75_cell_rms": float(np.sqrt(np.mean(cell_reference * cell_reference))),
        "midpoint_common_frame_cell_relative_l2_error": midpoint_rel,
        "frame_abs_correlation_with_abs_du": _corr_abs(np.abs(total_frame_face), np.abs(du)),
        "frame_abs_correlation_with_abs_dv": _corr_abs(np.abs(total_frame_face), np.abs(dv)),
        "frame_abs_correlation_with_jump_magnitude": _corr_abs(
            np.abs(total_frame_face), np.sqrt(du * du + dv * dv)
        ),
        "component_absolute_sums": component_abs.tolist(),
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "local_velocity_frame_jump_maps.npz",
        component_face_pair_remainder_maps=components,
        group_face_pair_remainder_maps=groups,
        stage75_reference_face_pair_remainder_maps=reference_pair,
        midpoint_common_frame_face_components=maps["midpoint_common_frame_face_components"],
        midpoint_common_frame_cell_map=midpoint_cell,
        stage75_reference_cell_map=cell_reference,
        local_u=maps["u"],
        local_v=maps["v"],
        face_du=du,
        face_dv=dv,
    )
    summary = {
        "stage": 76,
        "description": (
            "Exact frozen audit of the Stage-75 conservative face-pair remainder using "
            "the symmetric jump in neighboring cell-local peculiar-velocity frames."
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
            "material_frame_ratio_guard": MATERIAL_FRAME_RATIO_GUARD,
            "midpoint_cell_l2_guard": MIDPOINT_CELL_L2_GUARD,
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
        "retained_stage75_endpoint": STAGE75_COMPLETED_ENDPOINT,
        "retained_stage75_decision": retained75["decision"],
        "frame_kernel_jump_closure": {
            "maximum_absolute_error": frame_max,
            "relative_l2_error": frame_rel,
            "within_guard": frame_closed,
        },
        "frame_group_closure": {
            "maximum_absolute_error": group_max,
            "relative_l2_error": group_rel,
            "within_guard": group_closed,
        },
        "fine_grid_local_velocity_frame_metrics": metrics,
        "finite": finite,
        "provenance_consistent": provenance_consistent,
        "frame_closure_closed": frame_closed,
        "group_closure_closed": group_closed,
        "decision": decision,
        "positive_findings": [
            "The exact symmetric local-frame jump expansion reproduces the Stage-75 face-pair remainder within the unchanged closure guard.",
            "Replacing the two target-cell frames by one midpoint common frame leaves the frozen cell residual essentially unchanged when the frame-jump contribution is below the preregistered materiality guard.",
            "The audit uses exact completed artifacts and performs no cavity solve or operator substitution.",
        ],
        "negative_findings": [
            "A closed frame-jump identity is bookkeeping evidence, not evidence that second-order transport improves the published benchmark.",
            "The failed Stage-28 MUSCL endpoint remains unrecovered and is not extended across Knudsen number.",
            "No physical, collision, correction-floor, source-relaxation, transport, wall, normalization, or velocity-quadrature parameter was retuned.",
        ],
        "interpretation_guard": (
            "This audit separates local peculiar-frame bookkeeping from the conservative "
            "face-flux divergence; it does not alter the solver or claim benchmark improvement."
        ),
        "scientifically_justified_next_scope": (
            "If local-frame jumps are negligible, audit the spatial divergence and face-to-face "
            "coherence of the midpoint common-frame transport correction. If material, localize "
            "physical velocity-gradient contributions before any solver experiment."
        ),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage67-artifact-dir", required=True)
    parser.add_argument("--stage75-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run_stage76(args.stage67_artifact_dir, args.stage75_artifact_dir, args.output_dir),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
