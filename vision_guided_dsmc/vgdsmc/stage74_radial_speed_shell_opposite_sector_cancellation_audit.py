from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import stage73_velocity_sign_angular_bin_interior_face_flux_attribution_audit as stage73


STAGE73_COMPLETED_ENDPOINT = {
    "workflow_run_id": 31088628167,
    "workflow_job_id": 92573989499,
    "workflow_conclusion": "success",
    "tests_passed": 166,
    "tests_failed": 0,
    "test_duration_seconds": 0.98,
    "artifact_id": 8966181833,
    "artifact_size_bytes": 471060,
    "artifact_sha256": "125b1333eec2b617496b5e58c35ba0824e37f7507fab44e23faf2af8713285d9",
    "source_head_sha": "8efb02c77216b339d3fe9b1ff45a2e344157aded",
    "summary_sha256": "5cb64245b39b2ea0d33c7004f399c2d93c1954195719fff986378c4dc7ec5b9f",
    "velocity_group_maps_sha256": "bcb64f8edee745f6945e673b2f52a6cc6fac98dbdda95600238902064a7dfcd1",
    "decision": (
        "stage73_balanced_vx_sign_vertical_oblique_sector_concentration_"
        "stage74_radial_speed_shell_and_opposite_sector_cancellation_audit"
    ),
}

GRIDS = stage73.GRIDS
FINE_GRID = stage73.FINE_GRID
KNUDSEN = stage73.KNUDSEN
COLD_HOT_RATIO = stage73.COLD_HOT_RATIO
RULE = stage73.RULE
RADIAL_SCALE = stage73.RADIAL_SCALE
POINT_COUNT = stage73.POINT_COUNT
CHUNK_SIZE = stage73.CHUNK_SIZE
LIMITER = stage73.LIMITER
RESTRICTION = stage73.RESTRICTION
WALL_BAND_PHYSICAL_FRACTION = stage73.WALL_BAND_PHYSICAL_FRACTION
ANGULAR_BIN_COUNT = stage73.ANGULAR_BIN_COUNT
VERTICAL_OBLIQUE_BINS = stage73.VERTICAL_OBLIQUE_BINS
RADIAL_SHELL_COUNT = 4
RADIAL_NODES_PER_SHELL = RULE[0] // RADIAL_SHELL_COUNT
OPPOSITE_BIN_PAIRS = ((0, 4), (1, 5), (2, 6), (3, 7))
VERTICAL_OBLIQUE_OPPOSITE_PAIRS = ((1, 5), (2, 6))
TOP_TWO_SHELL_CONCENTRATION_GUARD = 0.65
OPPOSITE_PAIR_CANCELLATION_GUARD = 0.50
GROUP_CLOSURE_GUARD = 1.0e-10
ENDPOINT_GUARD = 1.0e-12


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_stage74_design(
    grids: tuple[int, ...] = GRIDS,
    fine_grid: int = FINE_GRID,
    kn0: float = KNUDSEN,
    cold_hot_ratio: float = COLD_HOT_RATIO,
    rule: tuple[int, int] = RULE,
    radial_scale: float = RADIAL_SCALE,
    chunk_size: int = CHUNK_SIZE,
    limiter: str = LIMITER,
    restriction: str = RESTRICTION,
    wall_band_physical_fraction: float = WALL_BAND_PHYSICAL_FRACTION,
    angular_bin_count: int = ANGULAR_BIN_COUNT,
    radial_shell_count: int = RADIAL_SHELL_COUNT,
    opposite_bin_pairs: tuple[tuple[int, int], ...] = OPPOSITE_BIN_PAIRS,
    vertical_oblique_opposite_pairs: tuple[tuple[int, int], ...] = VERTICAL_OBLIQUE_OPPOSITE_PAIRS,
    top_two_shell_concentration_guard: float = TOP_TWO_SHELL_CONCENTRATION_GUARD,
    opposite_pair_cancellation_guard: float = OPPOSITE_PAIR_CANCELLATION_GUARD,
) -> None:
    actual = (
        grids,
        fine_grid,
        kn0,
        cold_hot_ratio,
        rule,
        radial_scale,
        chunk_size,
        limiter,
        restriction,
        wall_band_physical_fraction,
        angular_bin_count,
        radial_shell_count,
        opposite_bin_pairs,
        vertical_oblique_opposite_pairs,
        top_two_shell_concentration_guard,
        opposite_pair_cancellation_guard,
    )
    expected = (
        GRIDS,
        FINE_GRID,
        KNUDSEN,
        COLD_HOT_RATIO,
        RULE,
        RADIAL_SCALE,
        CHUNK_SIZE,
        LIMITER,
        RESTRICTION,
        WALL_BAND_PHYSICAL_FRACTION,
        ANGULAR_BIN_COUNT,
        RADIAL_SHELL_COUNT,
        OPPOSITE_BIN_PAIRS,
        VERTICAL_OBLIQUE_OPPOSITE_PAIRS,
        TOP_TWO_SHELL_CONCENTRATION_GUARD,
        OPPOSITE_PAIR_CANCELLATION_GUARD,
    )
    if actual != expected:
        raise ValueError(
            "Stage 74 is frozen to the exact completed Stage-67 distributions, "
            "the exact completed Stage-73 velocity grouping, conservative "
            "64->32->16 restriction, four equal-radial-node shells, fixed opposite "
            "angular pairs, and preregistered diagnostic guards; no retuning is permitted."
        )
    if RULE[0] % RADIAL_SHELL_COUNT != 0:
        raise ValueError("Radial node count must divide exactly into frozen shells")


def _validate_stage73_artifact(root: str | Path) -> dict[str, object]:
    root = Path(root)
    expected = {
        "summary.json": STAGE73_COMPLETED_ENDPOINT["summary_sha256"],
        "velocity_group_interior_face_maps.npz": STAGE73_COMPLETED_ENDPOINT[
            "velocity_group_maps_sha256"
        ],
    }
    for name, checksum in expected.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != checksum:
            raise ValueError(f"Stage 73 artifact checksum mismatch: {name}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if (
        summary.get("stage") != 73
        or summary.get("decision") != STAGE73_COMPLETED_ENDPOINT["decision"]
    ):
        raise ValueError("Stage 73 completed endpoint mismatch")
    return summary


def radial_shell_indices(
    vx: np.ndarray,
    vy: np.ndarray,
    shell_count: int = RADIAL_SHELL_COUNT,
) -> np.ndarray:
    """Assign equal numbers of frozen radial nodes to ordered speed shells.

    The exact mapped-polar rule has the same number of angular points at every
    radial node. Stable speed sorting therefore partitions the 40 frozen radial
    nodes into four groups of ten without introducing a fitted speed threshold.
    """
    vx = np.asarray(vx, dtype=np.float64)
    vy = np.asarray(vy, dtype=np.float64)
    if vx.shape != vy.shape or vx.ndim != 1:
        raise ValueError("Velocity coordinates must be same-length one-dimensional arrays")
    if shell_count <= 0 or vx.size % shell_count != 0:
        raise ValueError("Velocity point count must divide exactly into radial shells")
    order = np.argsort(np.hypot(vx, vy), kind="stable")
    shell_size = vx.size // shell_count
    labels = np.empty(vx.size, dtype=np.int8)
    labels[order] = np.repeat(np.arange(shell_count, dtype=np.int8), shell_size)
    return labels


def radial_shell_metadata(
    vx: np.ndarray,
    vy: np.ndarray,
    shell_indices: np.ndarray,
) -> list[dict[str, float | int]]:
    speed = np.hypot(np.asarray(vx, dtype=np.float64), np.asarray(vy, dtype=np.float64))
    shell_indices = np.asarray(shell_indices)
    metadata: list[dict[str, float | int]] = []
    previous_max = -math.inf
    for shell in range(RADIAL_SHELL_COUNT):
        selected = shell_indices == shell
        if not np.any(selected):
            raise ValueError("Empty radial shell")
        minimum = float(np.min(speed[selected]))
        maximum = float(np.max(speed[selected]))
        if minimum + 1.0e-14 < previous_max:
            raise ValueError("Radial shell speeds are not ordered")
        metadata.append(
            {
                "shell": shell,
                "velocity_point_count": int(np.sum(selected)),
                "minimum_speed": minimum,
                "maximum_speed": maximum,
                "mean_speed": float(np.mean(speed[selected])),
            }
        )
        previous_max = maximum
    return metadata


def evaluate_radial_shell_angular_qy_maps(
    phi: np.ndarray,
    psi: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    weight: np.ndarray,
    chunk_size: int = CHUNK_SIZE,
) -> np.ndarray:
    """Attribute the frozen x-direction q_y residual by speed shell and angle."""
    phi = np.asarray(phi, dtype=np.float64)
    psi = np.asarray(psi, dtype=np.float64)
    vx = np.asarray(vx, dtype=np.float64)
    vy = np.asarray(vy, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    if phi.shape != psi.shape or phi.ndim != 3:
        raise ValueError("phi and psi must be matching ny x nx x nq arrays")
    if phi.shape[-1] != weight.size or vx.size != weight.size or vy.size != weight.size:
        raise ValueError("Velocity quadrature shape mismatch")
    _, local_u, local_v = stage73.stage68.macroscopic_velocity(
        phi, vx, vy, weight, chunk_size
    )
    shell_indices = radial_shell_indices(vx, vy)
    angular_indices = stage73.angular_bin_indices(vx, vy)
    groups = np.zeros(
        (RADIAL_SHELL_COUNT, ANGULAR_BIN_COUNT, phi.shape[0], phi.shape[1]),
        dtype=np.float64,
    )
    dx = 1.0 / phi.shape[1]
    for start in range(0, weight.size, chunk_size):
        stop = min(start + chunk_size, weight.size)
        sl = slice(start, stop)
        delta_phi = stage73.interior_x_face_flux_difference_chunk(phi[..., sl], vx[sl])
        delta_psi = stage73.interior_x_face_flux_difference_chunk(psi[..., sl], vx[sl])
        chunk_shells = shell_indices[sl]
        chunk_bins = angular_indices[sl]
        for shell in range(RADIAL_SHELL_COUNT):
            for angular_bin in range(ANGULAR_BIN_COUNT):
                selected = (chunk_shells == shell) & (chunk_bins == angular_bin)
                if not np.any(selected):
                    continue
                target = groups[shell, angular_bin]
                stage73._accumulate_target_cell_qy(
                    target[:, :-1],
                    delta_phi[..., selected],
                    delta_psi[..., selected],
                    vx[sl][selected],
                    vy[sl][selected],
                    weight[sl][selected],
                    local_u[:, :-1],
                    local_v[:, :-1],
                    -1.0,
                    dx,
                )
                stage73._accumulate_target_cell_qy(
                    target[:, 1:],
                    delta_phi[..., selected],
                    delta_psi[..., selected],
                    vx[sl][selected],
                    vy[sl][selected],
                    weight[sl][selected],
                    local_u[:, 1:],
                    local_v[:, 1:],
                    1.0,
                    dx,
                )
    return groups


def grouped_closure(
    groups: np.ndarray,
    stage73_bin_maps: np.ndarray,
) -> dict[str, float | bool]:
    groups = np.asarray(groups, dtype=np.float64)
    stage73_bin_maps = np.asarray(stage73_bin_maps, dtype=np.float64)
    reconstructed_bins = np.sum(groups, axis=0)
    reconstructed_total = np.sum(reconstructed_bins, axis=0)
    reference_total = np.sum(stage73_bin_maps, axis=0)
    total_delta = reconstructed_total - reference_total
    total_relative = float(
        np.linalg.norm(total_delta.ravel())
        / max(float(np.linalg.norm(reference_total.ravel())), 1.0e-300)
    )
    bin_relatives = []
    maximum_absolute_error = float(np.max(np.abs(total_delta)))
    for angular_bin in range(ANGULAR_BIN_COUNT):
        delta = reconstructed_bins[angular_bin] - stage73_bin_maps[angular_bin]
        maximum_absolute_error = max(maximum_absolute_error, float(np.max(np.abs(delta))))
        bin_relatives.append(
            float(
                np.linalg.norm(delta.ravel())
                / max(float(np.linalg.norm(stage73_bin_maps[angular_bin].ravel())), 1.0e-300)
            )
        )
    maximum_bin_relative = max(bin_relatives)
    return {
        "maximum_absolute_error": maximum_absolute_error,
        "total_relative_l2_error": total_relative,
        "maximum_bin_relative_l2_error": maximum_bin_relative,
        "within_guard": bool(
            total_relative <= GROUP_CLOSURE_GUARD
            and maximum_bin_relative <= GROUP_CLOSURE_GUARD
        ),
    }


def _absolute_sum(values: np.ndarray) -> float:
    return float(np.sum(np.abs(np.asarray(values, dtype=np.float64))))


def opposite_pair_metrics(bin_maps: np.ndarray) -> dict[str, object]:
    bin_maps = np.asarray(bin_maps, dtype=np.float64)
    denominator = sum(_absolute_sum(bin_maps[b]) for b in range(ANGULAR_BIN_COUNT))
    pairs: dict[str, dict[str, float | list[int]]] = {}
    for first, second in OPPOSITE_BIN_PAIRS:
        first_abs = _absolute_sum(bin_maps[first])
        second_abs = _absolute_sum(bin_maps[second])
        pair_map = bin_maps[first] + bin_maps[second]
        pre = first_abs + second_abs
        pairs[f"{first}_{second}"] = {
            "bins": [first, second],
            "pre_cancellation_absolute_sum": pre,
            "absolute_share": pre / max(denominator, 1.0e-300),
            "post_cancellation_absolute_sum": _absolute_sum(pair_map),
            "cancellation_ratio": _absolute_sum(pair_map) / max(pre, 1.0e-300),
            "signed_sum_first": float(np.sum(bin_maps[first])),
            "signed_sum_second": float(np.sum(bin_maps[second])),
            "signed_sum_pair": float(np.sum(pair_map)),
        }
    vertical_pre = sum(
        pairs[f"{a}_{b}"]["pre_cancellation_absolute_sum"]
        for a, b in VERTICAL_OBLIQUE_OPPOSITE_PAIRS
    )
    vertical_post_map = sum(
        (bin_maps[a] + bin_maps[b] for a, b in VERTICAL_OBLIQUE_OPPOSITE_PAIRS),
        np.zeros_like(bin_maps[0]),
    )
    return {
        "pairs": pairs,
        "vertical_oblique_pairs": [list(pair) for pair in VERTICAL_OBLIQUE_OPPOSITE_PAIRS],
        "vertical_oblique_pre_cancellation_absolute_sum": float(vertical_pre),
        "vertical_oblique_absolute_share": float(vertical_pre / max(denominator, 1.0e-300)),
        "vertical_oblique_post_cancellation_absolute_sum": _absolute_sum(vertical_post_map),
        "vertical_oblique_cancellation_ratio": _absolute_sum(vertical_post_map)
        / max(float(vertical_pre), 1.0e-300),
    }


def shell_angular_attribution(groups: np.ndarray) -> dict[str, object]:
    groups = np.asarray(groups, dtype=np.float64)
    shell_maps = np.sum(groups, axis=1)
    bin_maps = np.sum(groups, axis=0)
    shell_denominator = sum(_absolute_sum(shell_maps[s]) for s in range(RADIAL_SHELL_COUNT))
    shell_shares = [
        _absolute_sum(shell_maps[s]) / max(shell_denominator, 1.0e-300)
        for s in range(RADIAL_SHELL_COUNT)
    ]
    ranked_shells = sorted(
        range(RADIAL_SHELL_COUNT), key=lambda shell: shell_shares[shell], reverse=True
    )
    shell_pair_cancellation: dict[str, dict[str, float]] = {}
    for shell in range(RADIAL_SHELL_COUNT):
        metrics = opposite_pair_metrics(groups[shell])
        shell_pair_cancellation[str(shell)] = {
            "vertical_oblique_absolute_share": float(
                metrics["vertical_oblique_absolute_share"]
            ),
            "vertical_oblique_cancellation_ratio": float(
                metrics["vertical_oblique_cancellation_ratio"]
            ),
        }
    return {
        "total": stage73.stage72.signed_statistics(np.sum(shell_maps, axis=0)),
        "shell_absolute_shares": shell_shares,
        "shell_signed_sums": [float(np.sum(shell_maps[s])) for s in range(RADIAL_SHELL_COUNT)],
        "ranked_shells": ranked_shells,
        "dominant_shell": ranked_shells[0],
        "top_two_shell_absolute_share": float(
            shell_shares[ranked_shells[0]] + shell_shares[ranked_shells[1]]
        ),
        "opposite_pair_metrics": opposite_pair_metrics(bin_maps),
        "shell_vertical_oblique_pair_metrics": shell_pair_cancellation,
    }


def stage74_decision(
    finite: bool,
    provenance_consistent: bool,
    grouped_closure_closed: bool,
    radial_concentrated: bool,
    vertical_pair_cancellation_strong: bool,
) -> str:
    if not finite:
        return "stage74_nonfinite_radial_shell_blocker"
    if not provenance_consistent:
        return "stage74_completed_endpoint_reproduction_blocker"
    if not grouped_closure_closed:
        return "stage74_radial_shell_group_sum_closure_blocker"
    if radial_concentrated and vertical_pair_cancellation_strong:
        return (
            "stage74_radial_shell_concentration_and_opposite_sector_cancellation_"
            "stage75_facewise_shell_pair_wall_distance_localization_audit"
        )
    return (
        "stage74_diffuse_speed_or_weak_opposite_sector_cancellation_"
        "stage75_signed_face_location_velocity_moment_audit"
    )


def run_stage74(
    stage67_artifact_dir: str | Path,
    stage73_artifact_dir: str | Path,
    output_dir: str | Path,
    **design: object,
) -> dict[str, object]:
    validate_stage74_design(**design)
    stage67_artifact_dir = Path(stage67_artifact_dir)
    stage73_artifact_dir = Path(stage73_artifact_dir)
    retained67 = stage73.stage68._validate_stage67_artifact(stage67_artifact_dir)
    retained73 = _validate_stage73_artifact(stage73_artifact_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with np.load(stage67_artifact_dir / "converged_full_distributions.npz") as data:
        if set(data.files) != {"phi", "psi", "vx", "vy", "weight"}:
            raise ValueError("Stage 67 full-distribution contract mismatch")
        fine_phi = np.asarray(data["phi"], dtype=np.float64)
        fine_psi = np.asarray(data["psi"], dtype=np.float64)
        vx = np.asarray(data["vx"], dtype=np.float64)
        vy = np.asarray(data["vy"], dtype=np.float64)
        weight = np.asarray(data["weight"], dtype=np.float64)
    expected_shape = (FINE_GRID, FINE_GRID, POINT_COUNT)
    if fine_phi.shape != expected_shape or fine_psi.shape != expected_shape:
        raise ValueError("Stage 74 requires exact 64x64x3840 Stage-67 distributions")

    shell_indices = radial_shell_indices(vx, vy)
    shell_metadata = radial_shell_metadata(vx, vy, shell_indices)
    expected_shell_points = POINT_COUNT // RADIAL_SHELL_COUNT
    shell_contract_consistent = all(
        row["velocity_point_count"] == expected_shell_points for row in shell_metadata
    )

    grid_results: dict[str, dict[str, object]] = {}
    saved: dict[str, np.ndarray] = {}
    all_finite = True
    all_closure = True
    with np.load(stage73_artifact_dir / "velocity_group_interior_face_maps.npz") as reference:
        required = {
            f"grid{grid}_bin{angular_bin}_{sign_name}_qy"
            for grid in GRIDS
            for angular_bin in range(ANGULAR_BIN_COUNT)
            for sign_name in stage73.SIGN_NAMES
        }
        if not required.issubset(set(reference.files)):
            raise ValueError("Stage 73 velocity-group map contract mismatch")
        for grid in GRIDS:
            phi = fine_phi if grid == FINE_GRID else stage73.stage69.restrict_cell_average(fine_phi, grid)
            psi = fine_psi if grid == FINE_GRID else stage73.stage69.restrict_cell_average(fine_psi, grid)
            groups = evaluate_radial_shell_angular_qy_maps(
                phi, psi, vx, vy, weight, CHUNK_SIZE
            )
            stage73_bins = np.stack(
                [
                    np.asarray(reference[f"grid{grid}_bin{angular_bin}_vx_negative_qy"])
                    + np.asarray(reference[f"grid{grid}_bin{angular_bin}_vx_positive_qy"])
                    for angular_bin in range(ANGULAR_BIN_COUNT)
                ],
                axis=0,
            )
            closure = grouped_closure(groups, stage73_bins)
            attribution = shell_angular_attribution(groups)
            grid_results[str(grid)] = {
                "grid": [grid, grid],
                "cell_width": 1.0 / grid,
                "grouped_closure": closure,
                "attribution": attribution,
            }
            all_finite &= bool(np.all(np.isfinite(groups)))
            all_closure &= bool(closure["within_guard"])
            for shell in range(RADIAL_SHELL_COUNT):
                for angular_bin in range(ANGULAR_BIN_COUNT):
                    saved[f"grid{grid}_shell{shell}_bin{angular_bin}_qy"] = groups[
                        shell, angular_bin
                    ]
            saved[f"grid{grid}_reconstructed_x_qy"] = np.sum(groups, axis=(0, 1))
            saved[f"grid{grid}_stage73_reference_x_qy"] = np.sum(stage73_bins, axis=0)
            if grid != FINE_GRID:
                del phi, psi, groups

    fine = grid_results[str(FINE_GRID)]["attribution"]
    radial_concentrated = bool(
        fine["top_two_shell_absolute_share"] >= TOP_TWO_SHELL_CONCENTRATION_GUARD
    )
    vertical_ratio = float(
        fine["opposite_pair_metrics"]["vertical_oblique_cancellation_ratio"]
    )
    vertical_pair_cancellation_strong = bool(
        vertical_ratio <= OPPOSITE_PAIR_CANCELLATION_GUARD
    )
    stage73_fine = retained73["fine_grid_velocity_group_attribution"]
    stage73_vertical_share_error = abs(
        float(stage73_fine["vertical_oblique_absolute_share"])
        - 0.7456486709250579
    )
    provenance_consistent = bool(
        retained73.get("finite") is True
        and retained73.get("grouped_closure_closed") is True
        and retained73.get("provenance_consistent") is True
        and retained67.get("decision") == stage73.stage68.STAGE67_COMPLETED_ENDPOINT["decision"]
        and retained73.get("decision") == STAGE73_COMPLETED_ENDPOINT["decision"]
        and stage73_vertical_share_error <= ENDPOINT_GUARD
        and shell_contract_consistent
    )
    finite = bool(
        all_finite
        and all(math.isfinite(float(value)) for value in fine["shell_absolute_shares"])
    )
    decision = stage74_decision(
        finite,
        provenance_consistent,
        all_closure,
        radial_concentrated,
        vertical_pair_cancellation_strong,
    )
    np.savez_compressed(out / "radial_shell_opposite_sector_maps.npz", **saved)

    summary = {
        "stage": 74,
        "description": (
            "Exact frozen attribution of the Stage-73 x-direction normal heat-flux "
            "transport-operator difference to four equal-radial-node speed shells "
            "and fixed opposite angular-sector pairs on the unchanged 16x16, 32x32, "
            "and 64x64 cell-average sequence; no cavity solve is performed."
        ),
        "configuration": {
            "grids": list(GRIDS),
            "fine_grid": FINE_GRID,
            "kn0": KNUDSEN,
            "cold_hot_ratio": COLD_HOT_RATIO,
            "radial_nodes": RULE[0],
            "angular_nodes": RULE[1],
            "point_count": POINT_COUNT,
            "radial_scale": RADIAL_SCALE,
            "chunk_size": CHUNK_SIZE,
            "limiter": LIMITER,
            "restriction": RESTRICTION,
            "wall_band_physical_fraction": WALL_BAND_PHYSICAL_FRACTION,
            "angular_bin_count": ANGULAR_BIN_COUNT,
            "vertical_oblique_bins": list(VERTICAL_OBLIQUE_BINS),
            "radial_shell_count": RADIAL_SHELL_COUNT,
            "radial_nodes_per_shell": RADIAL_NODES_PER_SHELL,
            "radial_shell_metadata": shell_metadata,
            "opposite_bin_pairs": [list(pair) for pair in OPPOSITE_BIN_PAIRS],
            "vertical_oblique_opposite_pairs": [
                list(pair) for pair in VERTICAL_OBLIQUE_OPPOSITE_PAIRS
            ],
            "top_two_shell_concentration_guard": TOP_TWO_SHELL_CONCENTRATION_GUARD,
            "opposite_pair_cancellation_guard": OPPOSITE_PAIR_CANCELLATION_GUARD,
            "solver_rerun_count": 0,
            "physical_parameter_retuning": False,
            "collision_parameter_retuning": False,
            "correction_floor_retuning": False,
            "source_relaxation_retuning": False,
            "transport_parameter_retuning": False,
            "wall_model_retuning": False,
            "normalization_retuning": False,
            "velocity_quadrature_retuning": False,
            "bounded_wall_face_arm_adopted": False,
            "failed_muscl_endpoint_rehabilitated": False,
            "cross_knudsen_extension_permitted": False,
        },
        "retained_stage67_endpoint": stage73.stage68.STAGE67_COMPLETED_ENDPOINT,
        "retained_stage67_decision": retained67["decision"],
        "retained_stage73_endpoint": STAGE73_COMPLETED_ENDPOINT,
        "retained_stage73_decision": retained73["decision"],
        "grid_results": grid_results,
        "fine_grid_radial_shell_opposite_sector_attribution": {
            **fine,
            "radial_concentrated": radial_concentrated,
            "vertical_pair_cancellation_strong": vertical_pair_cancellation_strong,
            "stage73_vertical_oblique_absolute_share_error": stage73_vertical_share_error,
        },
        "finite": finite,
        "grouped_closure_closed": bool(all_closure),
        "radial_shell_contract_consistent": shell_contract_consistent,
        "provenance_consistent": provenance_consistent,
        "decision": decision,
        "positive_findings": [
            "Every radial-shell/angular-sector contribution is generated at an interior x face and the complete grouped sum is required to reproduce each exact Stage-73 angular-bin map.",
            "The four speed shells contain equal numbers of frozen radial nodes and introduce no fitted speed threshold.",
            "The exact Stage-67 distributions, quadrature weights, local peculiar-velocity moments, and conservative 64->32->16 restriction are retained."
        ],
        "negative_findings": [
            "Radial-shell or opposite-sector residual magnitude is not an adjoint sensitivity and does not predict a converged q_av response.",
            "Strong opposite-sector cancellation can make pre-cancellation absolute shares noncausal.",
            "The failed Stage-28 MUSCL endpoint remains unrecovered and is not extended across Knudsen number.",
            "No external validation, parameter retuning, operator adoption, or solver response claim is supported."
        ],
        "interpretation_guard": (
            "This stage resolves the exact frozen interior-face residual difference in "
            "radial-speed and opposite-angle groups only. It does not solve a modified "
            "system, validate MUSCL transport, or establish that any velocity group "
            "causes the published heat-flux discrepancy."
        ),
        "scientifically_justified_next_scope": (
            "If a small set of speed shells dominates while opposite vertical-oblique "
            "sectors cancel strongly, localize those shell-pair contributions by face "
            "position and wall distance. Otherwise retain the diffuse result and audit "
            "signed face-location and velocity-moment cancellation before any solver experiment."
        ),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage67-artifact-dir", required=True)
    parser.add_argument("--stage73-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = run_stage74(
        args.stage67_artifact_dir,
        args.stage73_artifact_dir,
        args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
