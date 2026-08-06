from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import stage68_independent_transport_operator_residual_audit as stage68
from . import stage69_frozen_grid_transfer_residual_scaling_audit as stage69
from . import stage71_wall_layer_interior_face_transport_attribution_audit as stage71
from . import stage72_directional_transport_component_audit as stage72


STAGE72_COMPLETED_ENDPOINT = {
    "workflow_run_id": 31074807690,
    "workflow_job_id": 92530390942,
    "workflow_conclusion": "success",
    "tests_passed": 154,
    "tests_failed": 0,
    "test_duration_seconds": 0.60,
    "artifact_id": 8958793397,
    "artifact_size_bytes": 236007,
    "artifact_sha256": "b8f71d66e4cc5abf1ee99c2bcc8157b8b75695b10f37e6a76f9f1a2b995c4def",
    "source_head_sha": "1086520d4985428bdcaafb8bc442f8934de83bd1",
    "summary_sha256": "879bbc210b361f1f045a66d49c95c223b97189396c4336726b953022c9addcb7",
    "component_maps_sha256": "65153d7fcacd81cc39f14915414b64bee9991e92d9b73db3b6fc81fe3d00f35d",
    "decision": (
        "stage72_x_direction_near_wall_side_strip_dominance_"
        "stage73_velocity_sign_angular_bin_interior_face_flux_attribution_audit"
    ),
}

GRIDS = stage69.STAGE69_GRIDS
FINE_GRID = stage69.STAGE69_FINE_GRID
KNUDSEN = stage69.STAGE69_KNUDSEN
COLD_HOT_RATIO = stage69.STAGE69_COLD_HOT_RATIO
RULE = stage69.STAGE69_RULE
RADIAL_SCALE = stage69.STAGE69_RADIAL_SCALE
POINT_COUNT = stage69.STAGE69_POINT_COUNT
CHUNK_SIZE = stage69.STAGE69_CHUNK_SIZE
LIMITER = stage69.STAGE69_LIMITER
RESTRICTION = stage69.STAGE69_RESTRICTION
WALL_BAND_PHYSICAL_FRACTION = stage69.STAGE69_WALL_BAND_PHYSICAL_FRACTION
ANGULAR_BIN_COUNT = 8
ANGULAR_BIN_OFFSET_RADIANS = 0.0
VERTICAL_OBLIQUE_BINS = (1, 2, 5, 6)
FINE_VERTICAL_OBLIQUE_CONCENTRATION = 0.70
COARSE_VERTICAL_OBLIQUE_FLOOR = 0.65
SIGN_BALANCE_TOLERANCE = 0.05
GROUP_CLOSURE_GUARD = 1.0e-10
ENDPOINT_GUARD = 1.0e-12
SIGN_NAMES = ("vx_negative", "vx_positive")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_stage73_design(
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
    angular_bin_offset_radians: float = ANGULAR_BIN_OFFSET_RADIANS,
    vertical_oblique_bins: tuple[int, ...] = VERTICAL_OBLIQUE_BINS,
    fine_vertical_oblique_concentration: float = FINE_VERTICAL_OBLIQUE_CONCENTRATION,
    coarse_vertical_oblique_floor: float = COARSE_VERTICAL_OBLIQUE_FLOOR,
    sign_balance_tolerance: float = SIGN_BALANCE_TOLERANCE,
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
        angular_bin_offset_radians,
        vertical_oblique_bins,
        fine_vertical_oblique_concentration,
        coarse_vertical_oblique_floor,
        sign_balance_tolerance,
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
        ANGULAR_BIN_OFFSET_RADIANS,
        VERTICAL_OBLIQUE_BINS,
        FINE_VERTICAL_OBLIQUE_CONCENTRATION,
        COARSE_VERTICAL_OBLIQUE_FLOOR,
        SIGN_BALANCE_TOLERANCE,
    )
    if actual != expected:
        raise ValueError(
            "Stage 73 is frozen to the exact completed Stage-67 distributions, "
            "the exact completed Stage-72 x-direction endpoint, conservative "
            "64->32->16 cell-average restriction, the 40x96 radial-scale-2.0 "
            "quadrature, eight zero-offset angular bins, fixed vertical-oblique "
            "sectors, and preregistered concentration/balance guards; no retuning "
            "is permitted."
        )


def _validate_stage72_artifact(root: str | Path) -> dict[str, object]:
    root = Path(root)
    expected = {
        "summary.json": STAGE72_COMPLETED_ENDPOINT["summary_sha256"],
        "directional_transport_component_maps.npz": STAGE72_COMPLETED_ENDPOINT[
            "component_maps_sha256"
        ],
    }
    for name, checksum in expected.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != checksum:
            raise ValueError(f"Stage 72 artifact checksum mismatch: {name}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if (
        summary.get("stage") != 72
        or summary.get("decision") != STAGE72_COMPLETED_ENDPOINT["decision"]
    ):
        raise ValueError("Stage 72 completed endpoint mismatch")
    return summary


def angular_bin_indices(
    vx: np.ndarray,
    vy: np.ndarray,
    bin_count: int = ANGULAR_BIN_COUNT,
    offset_radians: float = ANGULAR_BIN_OFFSET_RADIANS,
) -> np.ndarray:
    vx = np.asarray(vx, dtype=np.float64)
    vy = np.asarray(vy, dtype=np.float64)
    if vx.shape != vy.shape or vx.ndim != 1:
        raise ValueError("Velocity coordinates must be same-length one-dimensional arrays")
    if bin_count <= 0:
        raise ValueError("Angular bin count must be positive")
    width = 2.0 * math.pi / bin_count
    angle = np.mod(np.arctan2(vy, vx) - offset_radians, 2.0 * math.pi)
    bins = np.floor(angle / width).astype(np.int16)
    return np.minimum(bins, bin_count - 1)


def velocity_sign_indices(vx: np.ndarray) -> np.ndarray:
    vx = np.asarray(vx, dtype=np.float64)
    if np.any(vx == 0.0):
        raise ValueError("Stage 73 requires the exact nonzero-vx mapped-polar rule")
    return (vx > 0.0).astype(np.int8)


def interior_x_face_flux_difference_chunk(
    distribution: np.ndarray,
    vx: np.ndarray,
) -> np.ndarray:
    """Second-minus-first-order flux at every interior x face."""
    distribution = np.asarray(distribution, dtype=np.float64)
    vx = np.asarray(vx, dtype=np.float64)
    if distribution.ndim != 3 or distribution.shape[-1] != vx.size:
        raise ValueError("Distribution/velocity shape mismatch")
    slope = stage68.limited_slopes_x(distribution)
    delta = np.zeros(
        (distribution.shape[0], distribution.shape[1] - 1, distribution.shape[2]),
        dtype=np.float64,
    )
    positive = vx > 0.0
    negative = vx < 0.0
    if np.any(positive):
        delta[..., positive] = (
            0.5
            * vx[positive][None, None, :]
            * slope[:, :-1, positive]
        )
    if np.any(negative):
        delta[..., negative] = (
            -0.5
            * vx[negative][None, None, :]
            * slope[:, 1:, negative]
        )
    return delta


def _accumulate_target_cell_qy(
    target: np.ndarray,
    delta_phi: np.ndarray,
    delta_psi: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    weight: np.ndarray,
    local_u: np.ndarray,
    local_v: np.ndarray,
    sign: float,
    dx: float,
) -> None:
    cx = vx[None, None, :] - local_u[..., None]
    cy = vy[None, None, :] - local_v[..., None]
    residual_phi = sign * delta_phi / dx
    residual_psi = sign * delta_psi / dx
    target += 0.5 * np.sum(
        cy
        * ((cx * cx + cy * cy) * residual_phi + residual_psi)
        * weight[None, None, :],
        axis=-1,
    )


def evaluate_velocity_group_x_qy_maps(
    phi: np.ndarray,
    psi: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    weight: np.ndarray,
    chunk_size: int = CHUNK_SIZE,
    angular_bin_count: int = ANGULAR_BIN_COUNT,
) -> np.ndarray:
    """Attribute the frozen x-direction q_y residual to sign/angular groups.

    The returned shape is ``(angular_bin_count, 2, ny, nx)``. Axis 1 is
    ``vx_negative, vx_positive``. Every contribution is generated at an
    interior x face, then applied with opposite conservative signs to its two
    adjacent target cells using the same cell-local peculiar-velocity moment as
    Stage 72.
    """
    phi = np.asarray(phi, dtype=np.float64)
    psi = np.asarray(psi, dtype=np.float64)
    vx = np.asarray(vx, dtype=np.float64)
    vy = np.asarray(vy, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    if phi.shape != psi.shape or phi.ndim != 3:
        raise ValueError("phi and psi must be matching ny x nx x nq arrays")
    if phi.shape[-1] != weight.size or vx.size != weight.size or vy.size != weight.size:
        raise ValueError("Velocity quadrature shape mismatch")
    _, local_u, local_v = stage68.macroscopic_velocity(
        phi, vx, vy, weight, chunk_size
    )
    groups = np.zeros(
        (angular_bin_count, 2, phi.shape[0], phi.shape[1]), dtype=np.float64
    )
    all_bins = angular_bin_indices(vx, vy, angular_bin_count)
    all_signs = velocity_sign_indices(vx)
    dx = 1.0 / phi.shape[1]
    for start in range(0, weight.size, chunk_size):
        stop = min(start + chunk_size, weight.size)
        sl = slice(start, stop)
        delta_phi = interior_x_face_flux_difference_chunk(phi[..., sl], vx[sl])
        delta_psi = interior_x_face_flux_difference_chunk(psi[..., sl], vx[sl])
        chunk_bins = all_bins[sl]
        chunk_signs = all_signs[sl]
        for angular_bin in range(angular_bin_count):
            for sign_index in range(2):
                selected = (chunk_bins == angular_bin) & (chunk_signs == sign_index)
                if not np.any(selected):
                    continue
                target = groups[angular_bin, sign_index]
                _accumulate_target_cell_qy(
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
                _accumulate_target_cell_qy(
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
    group_maps: np.ndarray,
    reference_x_component: np.ndarray,
) -> dict[str, float | bool]:
    reconstructed = np.sum(np.asarray(group_maps, dtype=np.float64), axis=(0, 1))
    reference = np.asarray(reference_x_component, dtype=np.float64)
    delta = reconstructed - reference
    relative_l2_error = float(
        np.linalg.norm(delta.ravel())
        / max(float(np.linalg.norm(reference.ravel())), 1.0e-300)
    )
    return {
        "maximum_absolute_error": float(np.max(np.abs(delta))),
        "relative_l2_error": relative_l2_error,
        "within_guard": bool(relative_l2_error <= GROUP_CLOSURE_GUARD),
    }


def _absolute_sum(values: np.ndarray) -> float:
    return float(np.sum(np.abs(np.asarray(values, dtype=np.float64))))


def _wall_metrics(values: np.ndarray, grid: int) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    distance = stage71.wall_distance(grid)
    layers = stage71.wall_band_layers(grid)
    wall = distance < layers
    masks = stage71.attribution_masks(grid, layers)
    vertical = masks["left"] | masks["right"]
    outer_two = distance < min(2, layers)
    global_abs = _absolute_sum(values)
    wall_abs = float(np.sum(np.abs(values[wall])))
    return {
        "wall_band_layers": layers,
        "wall_band_absolute_share": wall_abs / max(global_abs, 1.0e-300),
        "vertical_side_strip_absolute_wall_share": float(
            np.sum(np.abs(values[vertical])) / max(wall_abs, 1.0e-300)
        ),
        "outer_two_layer_absolute_wall_share": float(
            np.sum(np.abs(values[outer_two])) / max(wall_abs, 1.0e-300)
        ),
    }


def velocity_group_attribution(
    group_maps: np.ndarray,
    grid: int,
) -> dict[str, object]:
    group_maps = np.asarray(group_maps, dtype=np.float64)
    if group_maps.shape[:2] != (ANGULAR_BIN_COUNT, 2):
        raise ValueError("Stage 73 group-map contract mismatch")
    angular_maps = np.sum(group_maps, axis=1)
    sign_maps = np.sum(group_maps, axis=0)
    total = np.sum(angular_maps, axis=0)
    group_denominator = sum(_absolute_sum(group_maps[b, s]) for b in range(ANGULAR_BIN_COUNT) for s in range(2))
    angular_denominator = sum(_absolute_sum(angular_maps[b]) for b in range(ANGULAR_BIN_COUNT))
    sign_denominator = sum(_absolute_sum(sign_maps[s]) for s in range(2))
    angular_shares = [
        _absolute_sum(angular_maps[b]) / max(angular_denominator, 1.0e-300)
        for b in range(ANGULAR_BIN_COUNT)
    ]
    sign_shares = [
        _absolute_sum(sign_maps[s]) / max(sign_denominator, 1.0e-300)
        for s in range(2)
    ]
    vertical_oblique_share = float(sum(angular_shares[b] for b in VERTICAL_OBLIQUE_BINS))
    ranked_bins = sorted(
        range(ANGULAR_BIN_COUNT), key=lambda b: angular_shares[b], reverse=True
    )
    return {
        "total": stage72.signed_statistics(total),
        "group_pre_cancellation_absolute_sum": group_denominator,
        "group_to_total_cancellation_ratio": _absolute_sum(total) / max(group_denominator, 1.0e-300),
        "angular_bin_absolute_shares": angular_shares,
        "angular_bin_signed_sums": [float(np.sum(angular_maps[b])) for b in range(ANGULAR_BIN_COUNT)],
        "angular_bin_velocity_point_counts": None,
        "sign_absolute_shares": {
            SIGN_NAMES[0]: sign_shares[0],
            SIGN_NAMES[1]: sign_shares[1],
        },
        "sign_signed_sums": {
            SIGN_NAMES[0]: float(np.sum(sign_maps[0])),
            SIGN_NAMES[1]: float(np.sum(sign_maps[1])),
        },
        "sign_balance_error": abs(sign_shares[1] - sign_shares[0]),
        "vertical_oblique_bins": list(VERTICAL_OBLIQUE_BINS),
        "vertical_oblique_absolute_share": vertical_oblique_share,
        "ranked_angular_bins": ranked_bins,
        "top_two_angular_bin_absolute_share": float(
            angular_shares[ranked_bins[0]] + angular_shares[ranked_bins[1]]
        ),
        "wall_localization": _wall_metrics(total, grid),
    }


def stage73_decision(
    finite: bool,
    provenance_consistent: bool,
    grouped_closure_closed: bool,
    sign_balanced: bool,
    fine_vertical_oblique_concentrated: bool,
    coarse_vertical_oblique_supported: bool,
) -> str:
    if not finite:
        return "stage73_nonfinite_velocity_group_blocker"
    if not provenance_consistent:
        return "stage73_completed_endpoint_reproduction_blocker"
    if not grouped_closure_closed:
        return "stage73_velocity_group_sum_closure_blocker"
    if (
        sign_balanced
        and fine_vertical_oblique_concentrated
        and coarse_vertical_oblique_supported
    ):
        return (
            "stage73_balanced_vx_sign_vertical_oblique_sector_concentration_"
            "stage74_radial_speed_shell_and_opposite_sector_cancellation_audit"
        )
    return (
        "stage73_diffuse_or_grid_sensitive_angular_attribution_"
        "stage74_face_location_and_group_cancellation_audit"
    )


def run_stage73(
    stage67_artifact_dir: str | Path,
    stage72_artifact_dir: str | Path,
    output_dir: str | Path,
    **design: object,
) -> dict[str, object]:
    validate_stage73_design(**design)
    stage67_artifact_dir = Path(stage67_artifact_dir)
    stage72_artifact_dir = Path(stage72_artifact_dir)
    retained67 = stage68._validate_stage67_artifact(stage67_artifact_dir)
    retained72 = _validate_stage72_artifact(stage72_artifact_dir)
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
        raise ValueError("Stage 73 requires exact 64x64x3840 Stage-67 distributions")

    bin_indices = angular_bin_indices(vx, vy)
    sign_indices = velocity_sign_indices(vx)
    point_counts = [int(np.sum(bin_indices == b)) for b in range(ANGULAR_BIN_COUNT)]
    sign_point_counts = {
        SIGN_NAMES[0]: int(np.sum(sign_indices == 0)),
        SIGN_NAMES[1]: int(np.sum(sign_indices == 1)),
    }

    required_stage72_maps = {
        f"grid{grid}_{name}"
        for grid in GRIDS
        for name in (
            "difference_x_qy",
            "difference_y_qy",
            "difference_total_qy",
            "reference_difference_qy",
            "wall_distance_cells",
            "region_code",
        )
    }
    grid_results: dict[str, dict[str, object]] = {}
    saved: dict[str, np.ndarray] = {}
    all_finite = True
    all_closure = True
    region_contract_consistent = True
    with np.load(stage72_artifact_dir / "directional_transport_component_maps.npz") as reference:
        available = set(reference.files)
        if not required_stage72_maps.issubset(available):
            raise ValueError("Stage 72 directional-map contract mismatch")
        for grid in GRIDS:
            phi = fine_phi if grid == FINE_GRID else stage69.restrict_cell_average(fine_phi, grid)
            psi = fine_psi if grid == FINE_GRID else stage69.restrict_cell_average(fine_psi, grid)
            groups = evaluate_velocity_group_x_qy_maps(
                phi, psi, vx, vy, weight, CHUNK_SIZE, ANGULAR_BIN_COUNT
            )
            expected_x = np.asarray(reference[f"grid{grid}_difference_x_qy"], dtype=np.float64)
            closure = grouped_closure(groups, expected_x)
            attribution = velocity_group_attribution(groups, grid)
            attribution["angular_bin_velocity_point_counts"] = point_counts
            grid_results[str(grid)] = {
                "grid": [grid, grid],
                "cell_width": 1.0 / grid,
                "grouped_closure": closure,
                "attribution": attribution,
            }
            all_finite &= bool(np.all(np.isfinite(groups)))
            all_closure &= bool(closure["within_guard"])
            expected_distance = stage71.wall_distance(grid).astype(np.int16)
            expected_codes = stage71.region_code_map(grid, stage71.attribution_masks(grid))
            region_contract_consistent &= bool(
                np.array_equal(reference[f"grid{grid}_wall_distance_cells"], expected_distance)
                and np.array_equal(reference[f"grid{grid}_region_code"], expected_codes)
            )
            for angular_bin in range(ANGULAR_BIN_COUNT):
                for sign_index, sign_name in enumerate(SIGN_NAMES):
                    saved[f"grid{grid}_bin{angular_bin}_{sign_name}_qy"] = groups[
                        angular_bin, sign_index
                    ]
            saved[f"grid{grid}_reconstructed_x_qy"] = np.sum(groups, axis=(0, 1))
            saved[f"grid{grid}_reference_x_qy"] = expected_x
            saved[f"grid{grid}_wall_distance_cells"] = expected_distance
            saved[f"grid{grid}_region_code"] = expected_codes
            if grid != FINE_GRID:
                del phi, psi, groups

    fine = grid_results[str(FINE_GRID)]["attribution"]
    sign_balanced = bool(fine["sign_balance_error"] <= SIGN_BALANCE_TOLERANCE)
    fine_concentrated = bool(
        fine["vertical_oblique_absolute_share"] >= FINE_VERTICAL_OBLIQUE_CONCENTRATION
    )
    coarse_supported = bool(
        all(
            grid_results[str(grid)]["attribution"]["vertical_oblique_absolute_share"]
            >= COARSE_VERTICAL_OBLIQUE_FLOOR
            for grid in GRIDS[:-1]
        )
    )
    stage72_fine = retained72["fine_grid_directional_attribution"]
    x_share_error = abs(
        float(stage72_fine["x_component_absolute_share"])
        - 0.6077477460898555
    )
    provenance_consistent = bool(
        retained72.get("finite") is True
        and retained72.get("directional_closure_closed") is True
        and retained72.get("provenance_consistent") is True
        and retained72.get("region_contract_consistent") is True
        and retained67.get("decision") == stage68.STAGE67_COMPLETED_ENDPOINT["decision"]
        and region_contract_consistent
        and x_share_error <= ENDPOINT_GUARD
    )
    finite = bool(all_finite and all(math.isfinite(float(v)) for v in fine["angular_bin_absolute_shares"]))
    decision = stage73_decision(
        finite,
        provenance_consistent,
        all_closure,
        sign_balanced,
        fine_concentrated,
        coarse_supported,
    )
    np.savez_compressed(out / "velocity_group_interior_face_maps.npz", **saved)

    summary = {
        "stage": 73,
        "description": (
            "Exact frozen attribution of the dominant Stage-72 x-direction normal "
            "heat-flux transport-operator difference to vx sign and eight fixed "
            "angular sectors at interior x faces on the unchanged 16x16, 32x32, "
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
            "angular_bin_offset_radians": ANGULAR_BIN_OFFSET_RADIANS,
            "angular_bin_width_radians": 2.0 * math.pi / ANGULAR_BIN_COUNT,
            "vertical_oblique_bins": list(VERTICAL_OBLIQUE_BINS),
            "fine_vertical_oblique_concentration": FINE_VERTICAL_OBLIQUE_CONCENTRATION,
            "coarse_vertical_oblique_floor": COARSE_VERTICAL_OBLIQUE_FLOOR,
            "sign_balance_tolerance": SIGN_BALANCE_TOLERANCE,
            "angular_bin_velocity_point_counts": point_counts,
            "sign_velocity_point_counts": sign_point_counts,
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
        "retained_stage67_endpoint": stage68.STAGE67_COMPLETED_ENDPOINT,
        "retained_stage67_decision": retained67["decision"],
        "retained_stage72_endpoint": STAGE72_COMPLETED_ENDPOINT,
        "retained_stage72_decision": retained72["decision"],
        "grid_results": grid_results,
        "fine_grid_velocity_group_attribution": {
            **fine,
            "sign_balanced": sign_balanced,
            "fine_vertical_oblique_concentrated": fine_concentrated,
            "coarse_vertical_oblique_supported": coarse_supported,
            "stage72_x_component_absolute_share_error": x_share_error,
        },
        "finite": finite,
        "grouped_closure_closed": bool(all_closure),
        "region_contract_consistent": region_contract_consistent,
        "provenance_consistent": provenance_consistent,
        "decision": decision,
        "positive_findings": [
            "Every velocity-sign/angular-bin contribution is generated at an interior x face and the complete grouped sum reproduces the exact Stage-72 x-component map.",
            "The exact Stage-67 distributions, quadrature weights, local peculiar-velocity moments, and conservative 64->32->16 restriction are retained.",
            "Physical walls, collision terms, clipping, relaxation, normalization, and stopping criteria are unchanged."
        ],
        "negative_findings": [
            "Velocity-group residual magnitude is not an adjoint sensitivity and does not predict a converged q_av response.",
            "Cancellation within and between angular/sign groups can make absolute-share rankings noncausal.",
            "The failed Stage-28 MUSCL endpoint remains unrecovered and is not extended across Knudsen number.",
            "No external validation, parameter retuning, or solver response claim is supported."
        ],
        "interpretation_guard": (
            "This stage attributes the exact frozen interior-face residual difference "
            "in velocity space only. It does not solve a modified system, validate "
            "MUSCL transport, or establish that any velocity group causes the "
            "published heat-flux discrepancy."
        ),
        "scientifically_justified_next_scope": (
            "If vertical-oblique sectors are concentrated and vx signs remain "
            "balanced across the frozen grid sequence, resolve their radial-speed "
            "shells and opposite-sector cancellation. Otherwise resolve face "
            "location and diffuse group cancellation before any solver response experiment."
        ),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage67-artifact-dir", required=True)
    parser.add_argument("--stage72-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = run_stage73(
        args.stage67_artifact_dir,
        args.stage72_artifact_dir,
        args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
