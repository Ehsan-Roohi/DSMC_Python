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


STAGE71_COMPLETED_ENDPOINT = {
    "workflow_run_id": 31057376562,
    "workflow_job_id": 92477704992,
    "workflow_conclusion": "success",
    "tests_passed": 143,
    "tests_failed": 0,
    "test_duration_seconds": 0.80,
    "artifact_id": 8953709936,
    "artifact_size_bytes": 36993,
    "artifact_sha256": "05e6cdd83553613155b3d3f1e6ef0518297ac07ee78fb7ac581694d72c422b9f",
    "source_head_sha": "60616567f680c937f3042488734f5bf3c31a6ea3",
    "summary_sha256": "fc3ea9b78778f1ca2b0d811a0482a00eb87bc58d89e1f7e6b68d584327290923",
    "attribution_maps_sha256": "8f05a9a3d7135003fcd7659deabbd17740c8408e99c902bff4be7ab9d5db8b71",
    "decision": "stage71_near_wall_side_strip_interior_face_dominance_stage72_directional_transport_component_audit",
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
COMPONENT_DOMINANCE_FRACTION = 0.60
SIDE_STRIP_DOMINANCE_FRACTION = 0.50
OUTER_TWO_LAYER_CONCENTRATION = 0.75
DIRECTIONAL_CLOSURE_GUARD = 1.0e-10
ENDPOINT_GUARD = 1.0e-12


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_stage72_design(
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
    component_dominance_fraction: float = COMPONENT_DOMINANCE_FRACTION,
    side_strip_dominance_fraction: float = SIDE_STRIP_DOMINANCE_FRACTION,
    outer_two_layer_concentration: float = OUTER_TWO_LAYER_CONCENTRATION,
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
        component_dominance_fraction,
        side_strip_dominance_fraction,
        outer_two_layer_concentration,
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
        COMPONENT_DOMINANCE_FRACTION,
        SIDE_STRIP_DOMINANCE_FRACTION,
        OUTER_TWO_LAYER_CONCENTRATION,
    )
    if actual != expected:
        raise ValueError(
            "Stage 72 is frozen to the exact completed Stage-67 distributions, "
            "the completed Stage-71 attribution endpoint, conservative 64->32->16 "
            "cell-average restriction, the 40x96 radial-scale-2.0 quadrature, "
            "minmod reconstruction, and preregistered directional attribution "
            "guards; no retuning is permitted."
        )


def _validate_stage71_artifact(root: str | Path) -> dict[str, object]:
    root = Path(root)
    expected = {
        "summary.json": STAGE71_COMPLETED_ENDPOINT["summary_sha256"],
        "wall_layer_attribution_maps.npz": STAGE71_COMPLETED_ENDPOINT[
            "attribution_maps_sha256"
        ],
    }
    for name, checksum in expected.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != checksum:
            raise ValueError(f"Stage 71 artifact checksum mismatch: {name}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if (
        summary.get("stage") != 71
        or summary.get("decision") != STAGE71_COMPLETED_ENDPOINT["decision"]
    ):
        raise ValueError("Stage 71 completed endpoint mismatch")
    return summary


def first_order_directional_chunk(
    distribution: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    bottom: np.ndarray,
    top: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact retained first-order x- and y-transport residuals."""
    distribution = np.asarray(distribution, dtype=np.float64)
    residual_x = np.zeros_like(distribution)
    residual_y = np.zeros_like(distribution)
    for k, (vx_k, vy_k) in enumerate(zip(vx, vy, strict=True)):
        ax = abs(float(vx_k)) / dx
        ay = abs(float(vy_k)) / dy
        if vx_k > 0.0:
            residual_x[:, 1:, k] += ax * (
                distribution[:, :-1, k] - distribution[:, 1:, k]
            )
            residual_x[:, 0, k] += ax * (
                left[:, k] - distribution[:, 0, k]
            )
        elif vx_k < 0.0:
            residual_x[:, :-1, k] += ax * (
                distribution[:, 1:, k] - distribution[:, :-1, k]
            )
            residual_x[:, -1, k] += ax * (
                right[:, k] - distribution[:, -1, k]
            )
        if vy_k > 0.0:
            residual_y[1:, :, k] += ay * (
                distribution[:-1, :, k] - distribution[1:, :, k]
            )
            residual_y[0, :, k] += ay * (
                bottom[:, k] - distribution[0, :, k]
            )
        elif vy_k < 0.0:
            residual_y[:-1, :, k] += ay * (
                distribution[1:, :, k] - distribution[:-1, :, k]
            )
            residual_y[-1, :, k] += ay * (
                top[:, k] - distribution[-1, :, k]
            )
    return residual_x, residual_y


def second_order_directional_chunk(
    distribution: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    bottom: np.ndarray,
    top: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return x/y pieces of the frozen conservative minmod MUSCL residual."""
    distribution = np.asarray(distribution, dtype=np.float64)
    ny, nx, _ = distribution.shape
    residual_x = np.zeros_like(distribution)
    residual_y = np.zeros_like(distribution)
    slope_x = stage68.limited_slopes_x(distribution)
    slope_y = stage68.limited_slopes_y(distribution)
    for k, (vx_k, vy_k) in enumerate(zip(vx, vy, strict=True)):
        if vx_k != 0.0:
            faces_x = np.empty((ny, nx + 1), dtype=np.float64)
            if vx_k > 0.0:
                faces_x[:, 0] = vx_k * left[:, k]
                faces_x[:, 1:] = vx_k * (
                    distribution[:, :, k] + 0.5 * slope_x[:, :, k]
                )
            else:
                faces_x[:, :-1] = vx_k * (
                    distribution[:, :, k] - 0.5 * slope_x[:, :, k]
                )
                faces_x[:, -1] = vx_k * right[:, k]
            residual_x[:, :, k] -= (
                faces_x[:, 1:] - faces_x[:, :-1]
            ) / dx
        if vy_k != 0.0:
            faces_y = np.empty((ny + 1, nx), dtype=np.float64)
            if vy_k > 0.0:
                faces_y[0, :] = vy_k * bottom[:, k]
                faces_y[1:, :] = vy_k * (
                    distribution[:, :, k] + 0.5 * slope_y[:, :, k]
                )
            else:
                faces_y[:-1, :] = vy_k * (
                    distribution[:, :, k] - 0.5 * slope_y[:, :, k]
                )
                faces_y[-1, :] = vy_k * top[:, k]
            residual_y[:, :, k] -= (
                faces_y[1:, :] - faces_y[:-1, :]
            ) / dy
    return residual_x, residual_y


def _accumulate_qy(
    output: np.ndarray,
    residual_phi: np.ndarray,
    residual_psi: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    weight: np.ndarray,
    local_u: np.ndarray,
    local_v: np.ndarray,
) -> None:
    vx3 = vx[None, None, :]
    vy3 = vy[None, None, :]
    w3 = weight[None, None, :]
    cx = vx3 - local_u[..., None]
    cy = vy3 - local_v[..., None]
    frozen_energy = (cx * cx + cy * cy) * residual_phi + residual_psi
    output += 0.5 * np.sum(cy * frozen_energy * w3, axis=-1)


def evaluate_directional_qy_components(
    phi: np.ndarray,
    psi: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    weight: np.ndarray,
    incoming: tuple[np.ndarray, ...],
    chunk_size: int = CHUNK_SIZE,
) -> dict[str, np.ndarray]:
    """Evaluate retained/independent q_y transport moments by spatial direction."""
    _, local_u, local_v = stage68.macroscopic_velocity(
        phi, vx, vy, weight, chunk_size
    )
    shape = phi.shape[:2]
    output = {
        name: np.zeros(shape, dtype=np.float64)
        for name in (
            "retained_first_order_x_qy",
            "retained_first_order_y_qy",
            "independent_second_order_x_qy",
            "independent_second_order_y_qy",
        )
    }
    (
        left_phi,
        left_psi,
        right_phi,
        right_psi,
        bottom_phi,
        bottom_psi,
        top_phi,
        top_psi,
    ) = incoming
    dx = 1.0 / phi.shape[1]
    dy = 1.0 / phi.shape[0]
    for start in range(0, weight.size, chunk_size):
        stop = min(start + chunk_size, weight.size)
        sl = slice(start, stop)
        p = phi[..., sl]
        s = psi[..., sl]
        args_phi = (
            p,
            left_phi[..., sl],
            right_phi[..., sl],
            bottom_phi[..., sl],
            top_phi[..., sl],
            vx[sl],
            vy[sl],
            dx,
            dy,
        )
        args_psi = (
            s,
            left_psi[..., sl],
            right_psi[..., sl],
            bottom_psi[..., sl],
            top_psi[..., sl],
            vx[sl],
            vy[sl],
            dx,
            dy,
        )
        first_x_phi, first_y_phi = first_order_directional_chunk(*args_phi)
        first_x_psi, first_y_psi = first_order_directional_chunk(*args_psi)
        second_x_phi, second_y_phi = second_order_directional_chunk(*args_phi)
        second_x_psi, second_y_psi = second_order_directional_chunk(*args_psi)
        for name, rphi, rpsi in (
            ("retained_first_order_x_qy", first_x_phi, first_x_psi),
            ("retained_first_order_y_qy", first_y_phi, first_y_psi),
            ("independent_second_order_x_qy", second_x_phi, second_x_psi),
            ("independent_second_order_y_qy", second_y_phi, second_y_psi),
        ):
            _accumulate_qy(
                output[name],
                rphi,
                rpsi,
                vx[sl],
                vy[sl],
                weight[sl],
                local_u,
                local_v,
            )
    output["difference_x_qy"] = (
        output["independent_second_order_x_qy"]
        - output["retained_first_order_x_qy"]
    )
    output["difference_y_qy"] = (
        output["independent_second_order_y_qy"]
        - output["retained_first_order_y_qy"]
    )
    output["difference_total_qy"] = (
        output["difference_x_qy"] + output["difference_y_qy"]
    )
    return output


def rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(values * values)))


def signed_statistics(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    absolute_sum = float(np.sum(np.abs(values)))
    signed_sum = float(np.sum(values))
    return {
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
        "rms": rms(values),
        "absolute_sum": absolute_sum,
        "signed_sum": signed_sum,
        "signed_to_absolute_ratio": signed_sum / max(absolute_sum, 1.0e-300),
        "positive_cell_fraction": float(np.mean(values > 0.0)),
        "negative_cell_fraction": float(np.mean(values < 0.0)),
    }


def directional_closure(
    difference_x: np.ndarray,
    difference_y: np.ndarray,
    reference_total: np.ndarray,
) -> dict[str, float | bool]:
    reconstructed = np.asarray(difference_x) + np.asarray(difference_y)
    reference_total = np.asarray(reference_total)
    delta = reconstructed - reference_total
    maximum_absolute_error = float(np.max(np.abs(delta)))
    relative_l2_error = float(
        np.linalg.norm(delta.ravel())
        / max(float(np.linalg.norm(reference_total.ravel())), 1.0e-300)
    )
    return {
        "maximum_absolute_error": maximum_absolute_error,
        "relative_l2_error": relative_l2_error,
        "within_guard": bool(relative_l2_error <= DIRECTIONAL_CLOSURE_GUARD),
    }


def _absolute_share(values: np.ndarray, mask: np.ndarray) -> float:
    absolute = np.abs(np.asarray(values, dtype=np.float64))
    return float(np.sum(absolute[mask]) / max(float(np.sum(absolute)), 1.0e-300))


def _wall_relative_share(
    values: np.ndarray,
    region_mask: np.ndarray,
    wall_mask: np.ndarray,
) -> float:
    absolute = np.abs(np.asarray(values, dtype=np.float64))
    return float(
        np.sum(absolute[region_mask])
        / max(float(np.sum(absolute[wall_mask])), 1.0e-300)
    )


def component_attribution(
    difference_x: np.ndarray,
    difference_y: np.ndarray,
    grid: int,
) -> dict[str, object]:
    difference_x = np.asarray(difference_x, dtype=np.float64)
    difference_y = np.asarray(difference_y, dtype=np.float64)
    total = difference_x + difference_y
    distance = stage71.wall_distance(grid)
    layers = stage71.wall_band_layers(grid)
    masks = stage71.attribution_masks(grid, layers)
    wall = distance < layers
    outer_two = distance < min(2, layers)
    x_abs = float(np.sum(np.abs(difference_x)))
    y_abs = float(np.sum(np.abs(difference_y)))
    denominator = max(x_abs + y_abs, 1.0e-300)
    vertical = masks["left"] | masks["right"]
    horizontal = masks["bottom"] | masks["top"]
    correlation = 0.0
    if np.std(difference_x) > 0.0 and np.std(difference_y) > 0.0:
        correlation = float(np.corrcoef(difference_x.ravel(), difference_y.ravel())[0, 1])
    return {
        "x_component": signed_statistics(difference_x),
        "y_component": signed_statistics(difference_y),
        "total": signed_statistics(total),
        "x_component_absolute_share": x_abs / denominator,
        "y_component_absolute_share": y_abs / denominator,
        "component_cancellation_ratio": float(
            np.sum(np.abs(total)) / denominator
        ),
        "x_y_correlation": correlation,
        "x_wall_band_absolute_share": _absolute_share(difference_x, wall),
        "y_wall_band_absolute_share": _absolute_share(difference_y, wall),
        "x_outer_two_layer_absolute_wall_share": _wall_relative_share(
            difference_x, outer_two, wall
        ),
        "y_outer_two_layer_absolute_wall_share": _wall_relative_share(
            difference_y, outer_two, wall
        ),
        "x_vertical_side_strip_absolute_wall_share": _wall_relative_share(
            difference_x, vertical, wall
        ),
        "y_vertical_side_strip_absolute_wall_share": _wall_relative_share(
            difference_y, vertical, wall
        ),
        "x_horizontal_strip_absolute_wall_share": _wall_relative_share(
            difference_x, horizontal, wall
        ),
        "y_horizontal_strip_absolute_wall_share": _wall_relative_share(
            difference_y, horizontal, wall
        ),
        "wall_band_layers": layers,
    }


def stage72_decision(
    finite: bool,
    provenance_consistent: bool,
    directional_closure_closed: bool,
    x_component_dominant: bool,
    y_component_dominant: bool,
    dominant_outer_two_layers_concentrated: bool,
    dominant_oriented_strips: bool,
) -> str:
    if not finite:
        return "stage72_nonfinite_directional_component_blocker"
    if not provenance_consistent:
        return "stage72_completed_endpoint_reproduction_blocker"
    if not directional_closure_closed:
        return "stage72_directional_sum_closure_blocker"
    if (
        x_component_dominant
        and dominant_outer_two_layers_concentrated
        and dominant_oriented_strips
    ):
        return (
            "stage72_x_direction_near_wall_side_strip_dominance_"
            "stage73_velocity_sign_angular_bin_interior_face_flux_attribution_audit"
        )
    if (
        y_component_dominant
        and dominant_outer_two_layers_concentrated
        and dominant_oriented_strips
    ):
        return (
            "stage72_y_direction_near_wall_horizontal_strip_dominance_"
            "stage73_velocity_sign_angular_bin_interior_face_flux_attribution_audit"
        )
    return (
        "stage72_mixed_direction_or_cancellation_"
        "stage73_facewise_directional_flux_cancellation_audit"
    )


def run_stage72(
    stage67_artifact_dir: str | Path,
    stage71_artifact_dir: str | Path,
    output_dir: str | Path,
    **design: object,
) -> dict[str, object]:
    validate_stage72_design(**design)
    stage67_artifact_dir = Path(stage67_artifact_dir)
    stage71_artifact_dir = Path(stage71_artifact_dir)
    retained67 = stage68._validate_stage67_artifact(stage67_artifact_dir)
    retained71 = _validate_stage71_artifact(stage71_artifact_dir)
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
        raise ValueError("Stage 72 requires exact 64x64x3840 Stage-67 distributions")

    required_stage71_maps = {
        f"grid{grid}_{name}"
        for grid in GRIDS
        for name in ("difference_qy", "wall_distance_cells", "region_code")
    }
    grid_results: dict[str, dict[str, object]] = {}
    saved: dict[str, np.ndarray] = {}
    all_finite = True
    all_closure = True
    region_contract_consistent = True
    with np.load(stage71_artifact_dir / "wall_layer_attribution_maps.npz") as reference:
        if set(reference.files) != required_stage71_maps:
            raise ValueError("Stage 71 attribution-map contract mismatch")
        for grid in GRIDS:
            phi = (
                fine_phi
                if grid == FINE_GRID
                else stage69.restrict_cell_average(fine_phi, grid)
            )
            psi = (
                fine_psi
                if grid == FINE_GRID
                else stage69.restrict_cell_average(fine_psi, grid)
            )
            incoming = stage68.reconstruct_wall_incoming(
                phi, psi, vx, vy, weight, COLD_HOT_RATIO
            )
            components = evaluate_directional_qy_components(
                phi, psi, vx, vy, weight, incoming, CHUNK_SIZE
            )
            expected_total = np.asarray(
                reference[f"grid{grid}_difference_qy"], dtype=np.float64
            )
            closure = directional_closure(
                components["difference_x_qy"],
                components["difference_y_qy"],
                expected_total,
            )
            all_closure &= bool(closure["within_guard"])
            all_finite &= bool(
                all(np.all(np.isfinite(values)) for values in components.values())
            )
            expected_distance = stage71.wall_distance(grid).astype(np.int16)
            expected_codes = stage71.region_code_map(
                grid, stage71.attribution_masks(grid)
            )
            region_contract_consistent &= bool(
                np.array_equal(
                    np.asarray(reference[f"grid{grid}_wall_distance_cells"]),
                    expected_distance,
                )
                and np.array_equal(
                    np.asarray(reference[f"grid{grid}_region_code"]),
                    expected_codes,
                )
            )
            attribution = component_attribution(
                components["difference_x_qy"],
                components["difference_y_qy"],
                grid,
            )
            grid_results[str(grid)] = {
                "grid": [grid, grid],
                "cell_width": 1.0 / grid,
                "directional_closure": closure,
                "attribution": attribution,
            }
            for name, values in components.items():
                saved[f"grid{grid}_{name}"] = values
            saved[f"grid{grid}_reference_difference_qy"] = expected_total
            saved[f"grid{grid}_wall_distance_cells"] = expected_distance
            saved[f"grid{grid}_region_code"] = expected_codes
            if grid != FINE_GRID:
                del phi, psi, components

    fine = grid_results[str(FINE_GRID)]["attribution"]
    x_dominant = bool(
        fine["x_component_absolute_share"] >= COMPONENT_DOMINANCE_FRACTION
    )
    y_dominant = bool(
        fine["y_component_absolute_share"] >= COMPONENT_DOMINANCE_FRACTION
    )
    if x_dominant:
        outer_concentrated = bool(
            fine["x_outer_two_layer_absolute_wall_share"]
            >= OUTER_TWO_LAYER_CONCENTRATION
        )
        oriented_strips = bool(
            fine["x_vertical_side_strip_absolute_wall_share"]
            >= SIDE_STRIP_DOMINANCE_FRACTION
        )
        dominant_direction = "x"
    elif y_dominant:
        outer_concentrated = bool(
            fine["y_outer_two_layer_absolute_wall_share"]
            >= OUTER_TWO_LAYER_CONCENTRATION
        )
        oriented_strips = bool(
            fine["y_horizontal_strip_absolute_wall_share"]
            >= SIDE_STRIP_DOMINANCE_FRACTION
        )
        dominant_direction = "y"
    else:
        outer_concentrated = False
        oriented_strips = False
        dominant_direction = "mixed"

    stage71_fine = retained71["fine_grid_attribution"]
    fine_total = np.asarray(saved[f"grid{FINE_GRID}_difference_total_qy"])
    fine_wall_share = stage68.wall_band_absolute_share(
        fine_total, stage71.wall_band_layers(FINE_GRID)
    )
    stage71_wall_share_error = abs(
        fine_wall_share - float(stage71_fine["wall_band_absolute_share"])
    )
    provenance_consistent = bool(
        retained71["finite"]
        and retained71["partition_closed"]
        and retained71["provenance_consistent"]
        and region_contract_consistent
        and stage71_wall_share_error <= ENDPOINT_GUARD
        and retained67["finite"]
    )
    finite = bool(all_finite and math.isfinite(fine_wall_share))
    decision = stage72_decision(
        finite,
        provenance_consistent,
        all_closure,
        x_dominant,
        y_dominant,
        outer_concentrated,
        oriented_strips,
    )
    np.savez_compressed(out / "directional_transport_component_maps.npz", **saved)

    summary = {
        "stage": 72,
        "description": (
            "Exact frozen decomposition of the Stage-69/71 normal heat-flux "
            "transport-operator difference into x- and y-direction control-volume "
            "face contributions on the unchanged 16x16, 32x32, and 64x64 "
            "cell-average grid sequence; no cavity solve is performed."
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
            "component_dominance_fraction": COMPONENT_DOMINANCE_FRACTION,
            "side_strip_dominance_fraction": SIDE_STRIP_DOMINANCE_FRACTION,
            "outer_two_layer_concentration": OUTER_TWO_LAYER_CONCENTRATION,
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
        "retained_stage71_endpoint": STAGE71_COMPLETED_ENDPOINT,
        "retained_stage71_decision": retained71["decision"],
        "grid_results": grid_results,
        "fine_grid_directional_attribution": {
            **fine,
            "dominant_direction": dominant_direction,
            "x_component_dominant": x_dominant,
            "y_component_dominant": y_dominant,
            "dominant_outer_two_layers_concentrated": outer_concentrated,
            "dominant_oriented_strips": oriented_strips,
            "reproduced_stage71_wall_band_absolute_share": fine_wall_share,
            "stage71_wall_band_absolute_share_error": stage71_wall_share_error,
        },
        "finite": finite,
        "directional_closure_closed": bool(all_closure),
        "region_contract_consistent": region_contract_consistent,
        "provenance_consistent": provenance_consistent,
        "decision": decision,
        "positive_findings": [
            "The x- and y-direction residual components sum back to every completed Stage-71 q_y map within the frozen closure guard.",
            "The decomposition uses the exact Stage-67 distributions and the unchanged conservative 64->32->16 restriction.",
            "Physical wall faces, collision terms, quadrature, clipping, relaxation, normalization, and stopping criteria are unchanged.",
        ],
        "negative_findings": [
            "Directional residual magnitude is not an adjoint sensitivity and does not predict a converged q_av change.",
            "Opposite-signed x/y components may cancel, so component magnitude alone does not establish causality.",
            "The bounded wall-face arm remains unadopted and the failed Stage-28 MUSCL endpoint remains unrecovered.",
            "No external validation, parameter retuning, or cross-Knudsen extension is supported.",
        ],
        "interpretation_guard": (
            "This stage attributes the exact frozen discrete operator difference by "
            "spatial direction only. It does not solve a modified system, establish "
            "observable sensitivity, validate either operator, or rehabilitate a "
            "failed higher-order endpoint."
        ),
        "scientifically_justified_next_scope": (
            "Condition the dominant directional component, or the mixed-direction "
            "cancellation if no direction dominates, on velocity sign and frozen "
            "angular bins at interior faces before considering any solver response experiment."
        ),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage67-artifact-dir", required=True)
    parser.add_argument("--stage71-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = run_stage72(
        args.stage67_artifact_dir,
        args.stage71_artifact_dir,
        args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
