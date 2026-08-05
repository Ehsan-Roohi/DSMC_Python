from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import math
from typing import Mapping

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

STAGE68_GRID = (64, 64)
STAGE68_KNUDSEN = 10.0
STAGE68_COLD_HOT_RATIO = 0.1
STAGE68_RULE = (40, 96)
STAGE68_RADIAL_SCALE = 2.0
STAGE68_POINT_COUNT = 3840
STAGE68_CHUNK_SIZE = 128
STAGE68_LIMITER = "minmod"
STAGE68_SECOND_ORDER_OPERATOR = "conservative_muscl_piecewise_linear_frozen_field"
STAGE68_MATERIAL_HEAT_FLUX_RATIO = 0.10
STAGE68_RETAINED_OPERATOR_GUARD = 1.0e-10
STAGE68_WALL_BAND_LAYERS = 4


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_stage68_design(
    grid: tuple[int, int] = STAGE68_GRID,
    kn0: float = STAGE68_KNUDSEN,
    cold_hot_ratio: float = STAGE68_COLD_HOT_RATIO,
    rule: tuple[int, int] = STAGE68_RULE,
    radial_scale: float = STAGE68_RADIAL_SCALE,
    chunk_size: int = STAGE68_CHUNK_SIZE,
    limiter: str = STAGE68_LIMITER,
    material_heat_flux_ratio: float = STAGE68_MATERIAL_HEAT_FLUX_RATIO,
) -> None:
    actual = (
        grid, kn0, cold_hot_ratio, rule, radial_scale, chunk_size, limiter,
        material_heat_flux_ratio,
    )
    expected = (
        STAGE68_GRID, STAGE68_KNUDSEN, STAGE68_COLD_HOT_RATIO, STAGE68_RULE,
        STAGE68_RADIAL_SCALE, STAGE68_CHUNK_SIZE, STAGE68_LIMITER,
        STAGE68_MATERIAL_HEAT_FLUX_RATIO,
    )
    if actual != expected:
        raise ValueError(
            "Stage 68 is frozen to the completed Stage-67 64x64 Kn0=10 fields, "
            "the 40x96 radial-scale-2.0 quadrature, minmod reconstruction, and "
            "the preregistered 10% materiality threshold; no tuning is permitted."
        )


def _validate_stage67_artifact(root: str | Path) -> dict[str, object]:
    root = Path(root)
    expected = {
        "summary.json": STAGE67_COMPLETED_ENDPOINT["summary_sha256"],
        "converged_full_distributions.npz":
            STAGE67_COMPLETED_ENDPOINT["distributions_sha256"],
        "steady_residual_moment_maps.npz":
            STAGE67_COMPLETED_ENDPOINT["residual_maps_sha256"],
    }
    for name, checksum in expected.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != checksum:
            raise ValueError(f"Stage 67 artifact checksum mismatch: {name}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if (
        summary.get("stage") != 67
        or summary.get("decision") != STAGE67_COMPLETED_ENDPOINT["decision"]
    ):
        raise ValueError("Stage 67 artifact endpoint mismatch")
    return summary


def minmod(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    same_sign = left * right > 0.0
    return np.where(
        same_sign,
        np.sign(left) * np.minimum(np.abs(left), np.abs(right)),
        0.0,
    )


def limited_slopes_x(distribution: np.ndarray) -> np.ndarray:
    distribution = np.asarray(distribution, dtype=np.float64)
    slope = np.zeros_like(distribution)
    if distribution.shape[1] > 2:
        backward = distribution[:, 1:-1] - distribution[:, :-2]
        forward = distribution[:, 2:] - distribution[:, 1:-1]
        slope[:, 1:-1] = minmod(backward, forward)
    return slope


def limited_slopes_y(distribution: np.ndarray) -> np.ndarray:
    distribution = np.asarray(distribution, dtype=np.float64)
    slope = np.zeros_like(distribution)
    if distribution.shape[0] > 2:
        backward = distribution[1:-1] - distribution[:-2]
        forward = distribution[2:] - distribution[1:-1]
        slope[1:-1] = minmod(backward, forward)
    return slope


def projected_unit_wall_maxwellian(
    wall_temperature: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    wall_temperature = np.asarray(wall_temperature, dtype=np.float64)
    temperature = np.maximum(wall_temperature, 1.0e-12)
    speed2 = vx[None, :] ** 2 + vy[None, :] ** 2
    raw_phi = np.exp(-speed2 / (2.0 * temperature[:, None]))
    raw_phi /= 2.0 * math.pi * temperature[:, None]
    discrete_mass = np.sum(raw_phi * weight[None, :], axis=-1)
    phi = raw_phi / np.maximum(discrete_mass[:, None], 1.0e-300)
    psi = temperature[:, None] * phi
    return phi, psi


def profile_diffuse_incoming(
    outgoing_phi: np.ndarray,
    outgoing_psi: np.ndarray,
    normal_velocity: np.ndarray,
    incoming_mask: np.ndarray,
    wall_temperature: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    outgoing_phi = np.asarray(outgoing_phi, dtype=np.float64)
    outgoing_psi = np.asarray(outgoing_psi, dtype=np.float64)
    wall_phi, wall_psi = projected_unit_wall_maxwellian(
        wall_temperature, vx, vy, weight
    )
    outgoing_mask = (~incoming_mask) & (np.abs(normal_velocity) > 0.0)
    outgoing_flux = np.sum(
        normal_velocity[None, :]
        * outgoing_phi
        * outgoing_mask[None, :]
        * weight[None, :],
        axis=-1,
    )
    incoming_unit_flux = np.sum(
        normal_velocity[None, :]
        * wall_phi
        * incoming_mask[None, :]
        * weight[None, :],
        axis=-1,
    )
    denominator = np.where(
        np.abs(incoming_unit_flux) > 1.0e-14,
        incoming_unit_flux,
        np.copysign(1.0e-14, incoming_unit_flux + 1.0e-300),
    )
    scale = -outgoing_flux / denominator
    return scale[:, None] * wall_phi, scale[:, None] * wall_psi


def reconstruct_wall_incoming(
    phi: np.ndarray,
    psi: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    weight: np.ndarray,
    cold_hot_ratio: float = STAGE68_COLD_HOT_RATIO,
) -> tuple[np.ndarray, ...]:
    ny, nx, _ = phi.shape
    y = (np.arange(ny, dtype=np.float64) + 0.5) / ny
    side_temperature = 1.0 - (1.0 - cold_hot_ratio) * y
    left_phi, left_psi = profile_diffuse_incoming(
        phi[:, 0], psi[:, 0], vx, vx > 0.0,
        side_temperature, vx, vy, weight,
    )
    right_phi, right_psi = profile_diffuse_incoming(
        phi[:, -1], psi[:, -1], -vx, vx < 0.0,
        side_temperature, vx, vy, weight,
    )
    bottom_temperature = np.ones(nx, dtype=np.float64)
    bottom_phi, bottom_psi = profile_diffuse_incoming(
        phi[0], psi[0], vy, vy > 0.0,
        bottom_temperature, vx, vy, weight,
    )
    top_temperature = np.full(nx, cold_hot_ratio, dtype=np.float64)
    top_phi, top_psi = profile_diffuse_incoming(
        phi[-1], psi[-1], -vy, vy < 0.0,
        top_temperature, vx, vy, weight,
    )
    return (
        left_phi, left_psi, right_phi, right_psi,
        bottom_phi, bottom_psi, top_phi, top_psi,
    )


def first_order_transport_chunk(
    distribution: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    bottom: np.ndarray,
    top: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    distribution = np.asarray(distribution, dtype=np.float64)
    residual = np.zeros_like(distribution)
    for k, (vx_k, vy_k) in enumerate(zip(vx, vy, strict=True)):
        ax = abs(float(vx_k)) / dx
        ay = abs(float(vy_k)) / dy
        if vx_k > 0.0:
            residual[:, 1:, k] += ax * (
                distribution[:, :-1, k] - distribution[:, 1:, k]
            )
            residual[:, 0, k] += ax * (left[:, k] - distribution[:, 0, k])
        elif vx_k < 0.0:
            residual[:, :-1, k] += ax * (
                distribution[:, 1:, k] - distribution[:, :-1, k]
            )
            residual[:, -1, k] += ax * (right[:, k] - distribution[:, -1, k])
        if vy_k > 0.0:
            residual[1:, :, k] += ay * (
                distribution[:-1, :, k] - distribution[1:, :, k]
            )
            residual[0, :, k] += ay * (bottom[:, k] - distribution[0, :, k])
        elif vy_k < 0.0:
            residual[:-1, :, k] += ay * (
                distribution[1:, :, k] - distribution[:-1, :, k]
            )
            residual[-1, :, k] += ay * (top[:, k] - distribution[-1, :, k])
    return residual


def second_order_transport_chunk(
    distribution: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    bottom: np.ndarray,
    top: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Frozen-field conservative MUSCL residual with minmod-limited cell slopes."""
    distribution = np.asarray(distribution, dtype=np.float64)
    ny, nx, nq = distribution.shape
    residual = np.zeros_like(distribution)
    slope_x = limited_slopes_x(distribution)
    slope_y = limited_slopes_y(distribution)
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
            residual[:, :, k] -= (faces_x[:, 1:] - faces_x[:, :-1]) / dx
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
            residual[:, :, k] -= (faces_y[1:, :] - faces_y[:-1, :]) / dy
    return residual


def macroscopic_velocity(
    phi: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    weight: np.ndarray,
    chunk_size: int = STAGE68_CHUNK_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rho = np.zeros(phi.shape[:2], dtype=np.float64)
    mx = np.zeros_like(rho)
    my = np.zeros_like(rho)
    for start in range(0, weight.size, chunk_size):
        stop = min(start + chunk_size, weight.size)
        p = phi[..., start:stop]
        w = weight[start:stop][None, None, :]
        rho += np.sum(p * w, axis=-1)
        mx += np.sum(p * vx[start:stop][None, None, :] * w, axis=-1)
        my += np.sum(p * vy[start:stop][None, None, :] * w, axis=-1)
    safe = np.maximum(rho, 1.0e-14)
    return rho, mx / safe, my / safe


def _empty_moment_maps(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    return {
        name: np.zeros(shape, dtype=np.float64)
        for name in ("mass", "momentum_x", "momentum_y", "energy", "qx", "qy")
    }


def accumulate_moments(
    output: dict[str, np.ndarray],
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
    absolute_energy = (vx3 * vx3 + vy3 * vy3) * residual_phi + residual_psi
    frozen_energy = (cx * cx + cy * cy) * residual_phi + residual_psi
    output["mass"] += np.sum(residual_phi * w3, axis=-1)
    output["momentum_x"] += np.sum(vx3 * residual_phi * w3, axis=-1)
    output["momentum_y"] += np.sum(vy3 * residual_phi * w3, axis=-1)
    output["energy"] += 0.5 * np.sum(absolute_energy * w3, axis=-1)
    output["qx"] += 0.5 * np.sum(cx * frozen_energy * w3, axis=-1)
    output["qy"] += 0.5 * np.sum(cy * frozen_energy * w3, axis=-1)


def evaluate_transport_operators(
    phi: np.ndarray,
    psi: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    weight: np.ndarray,
    incoming: tuple[np.ndarray, ...],
    chunk_size: int = STAGE68_CHUNK_SIZE,
) -> dict[str, dict[str, np.ndarray]]:
    _, local_u, local_v = macroscopic_velocity(phi, vx, vy, weight, chunk_size)
    maps = {
        name: _empty_moment_maps(phi.shape[:2])
        for name in ("retained_first_order", "independent_second_order", "difference")
    }
    (
        left_phi, left_psi, right_phi, right_psi,
        bottom_phi, bottom_psi, top_phi, top_psi,
    ) = incoming
    dx = 1.0 / phi.shape[1]
    dy = 1.0 / phi.shape[0]
    for start in range(0, weight.size, chunk_size):
        stop = min(start + chunk_size, weight.size)
        sl = slice(start, stop)
        p = phi[..., sl]
        s = psi[..., sl]
        args_phi = (
            p, left_phi[..., sl], right_phi[..., sl], bottom_phi[..., sl],
            top_phi[..., sl], vx[sl], vy[sl], dx, dy,
        )
        args_psi = (
            s, left_psi[..., sl], right_psi[..., sl], bottom_psi[..., sl],
            top_psi[..., sl], vx[sl], vy[sl], dx, dy,
        )
        first_phi = first_order_transport_chunk(*args_phi)
        first_psi = first_order_transport_chunk(*args_psi)
        second_phi = second_order_transport_chunk(*args_phi)
        second_psi = second_order_transport_chunk(*args_psi)
        pairs = {
            "retained_first_order": (first_phi, first_psi),
            "independent_second_order": (second_phi, second_psi),
            "difference": (second_phi - first_phi, second_psi - first_psi),
        }
        for name, (rphi, rpsi) in pairs.items():
            accumulate_moments(
                maps[name], rphi, rpsi, vx[sl], vy[sl], weight[sl],
                local_u, local_v,
            )
    return maps


def _rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(values * values)))


def signed_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
        "mean_absolute": float(np.mean(np.abs(values))),
        "rms": _rms(values),
        "negative_fraction": float(np.mean(values < -1.0e-14)),
        "positive_fraction": float(np.mean(values > 1.0e-14)),
        "near_zero_fraction": float(np.mean(np.abs(values) <= 1.0e-14)),
    }


def retained_operator_consistency(
    maps: Mapping[str, Mapping[str, np.ndarray]],
    stage67_residual_maps_path: str | Path,
) -> dict[str, object]:
    per_moment: dict[str, dict[str, float]] = {}
    with np.load(stage67_residual_maps_path) as retained:
        for moment, actual in maps["retained_first_order"].items():
            expected = (
                np.asarray(retained[f"interior_transport_{moment}"], dtype=np.float64)
                + np.asarray(retained[f"diffuse_wall_{moment}"], dtype=np.float64)
            )
            delta = np.asarray(actual, dtype=np.float64) - expected
            per_moment[moment] = {
                "maximum_absolute_error": float(np.max(np.abs(delta))),
                "relative_l2_error": float(
                    np.linalg.norm(delta.ravel())
                    / max(float(np.linalg.norm(expected.ravel())), 1.0e-300)
                ),
            }
    max_abs = max(row["maximum_absolute_error"] for row in per_moment.values())
    max_rel = max(row["relative_l2_error"] for row in per_moment.values())
    return {
        "per_moment": per_moment,
        "maximum_absolute_error": max_abs,
        "maximum_relative_l2_error": max_rel,
        "within_guard": bool(max_rel <= STAGE68_RETAINED_OPERATOR_GUARD),
    }


def wall_band_absolute_share(values: np.ndarray, layers: int) -> float:
    values = np.asarray(values, dtype=np.float64)
    ny, nx = values.shape
    yy, xx = np.indices((ny, nx))
    distance = np.minimum.reduce((yy, xx, ny - 1 - yy, nx - 1 - xx))
    absolute = np.abs(values)
    total = float(np.sum(absolute))
    if total <= 1.0e-300:
        return 0.0
    return float(np.sum(absolute[distance < layers]) / total)


def stage68_decision(
    finite: bool,
    retained_consistency: bool,
    normal_heat_flux_operator_difference_ratio: float,
) -> str:
    if not finite:
        return "stage68_nonfinite_transport_operator_blocker"
    if not retained_consistency:
        return "stage68_retained_transport_reconstruction_blocker"
    if normal_heat_flux_operator_difference_ratio >= STAGE68_MATERIAL_HEAT_FLUX_RATIO:
        return (
            "stage68_material_higher_order_transport_residual_without_observable_"
            "causality_stage69_frozen_grid_transfer_residual_scaling_audit"
        )
    return (
        "stage68_higher_order_transport_residual_not_material_"
        "stage69_wall_flux_discretization_audit"
    )


def run_stage68(
    stage67_artifact_dir: str | Path,
    output_dir: str | Path,
    **design: object,
) -> dict[str, object]:
    validate_stage68_design(**design)
    stage67_artifact_dir = Path(stage67_artifact_dir)
    retained67 = _validate_stage67_artifact(stage67_artifact_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with np.load(stage67_artifact_dir / "converged_full_distributions.npz") as data:
        if set(data.files) != {"phi", "psi", "vx", "vy", "weight"}:
            raise ValueError("Stage 67 full-distribution contract mismatch")
        phi = np.asarray(data["phi"], dtype=np.float64)
        psi = np.asarray(data["psi"], dtype=np.float64)
        vx = np.asarray(data["vx"], dtype=np.float64)
        vy = np.asarray(data["vy"], dtype=np.float64)
        weight = np.asarray(data["weight"], dtype=np.float64)

    if phi.shape != (64, 64, 3840) or psi.shape != phi.shape:
        raise ValueError("Stage 68 requires exact 64x64x3840 frozen distributions")
    incoming = reconstruct_wall_incoming(phi, psi, vx, vy, weight)
    maps = evaluate_transport_operators(phi, psi, vx, vy, weight, incoming)
    consistency = retained_operator_consistency(
        maps, stage67_artifact_dir / "steady_residual_moment_maps.npz"
    )

    summaries = {
        component: {moment: signed_summary(values) for moment, values in moments.items()}
        for component, moments in maps.items()
    }
    ratios = {
        moment: (
            summaries["difference"][moment]["rms"]
            / max(summaries["retained_first_order"][moment]["rms"], 1.0e-300)
        )
        for moment in maps["difference"]
    }
    correlations: dict[str, float] = {}
    with np.load(stage67_artifact_dir / "steady_residual_moment_maps.npz") as retained:
        collision_qy = np.asarray(retained["collision_qy"], dtype=np.float64).ravel()
        difference_qy = maps["difference"]["qy"].ravel()
        if np.std(collision_qy) > 0.0 and np.std(difference_qy) > 0.0:
            correlations["difference_qy_vs_collision_qy"] = float(
                np.corrcoef(difference_qy, collision_qy)[0, 1]
            )
        else:
            correlations["difference_qy_vs_collision_qy"] = 0.0

    finite = bool(
        all(np.all(np.isfinite(values)) for component in maps.values()
            for values in component.values())
        and all(math.isfinite(value) for value in ratios.values())
    )
    decision = stage68_decision(
        finite,
        bool(consistency["within_guard"]),
        float(ratios["qy"]),
    )
    np.savez_compressed(
        out / "transport_operator_moment_maps.npz",
        **{
            f"{component}_{moment}": values
            for component, moments in maps.items()
            for moment, values in moments.items()
        },
    )
    summary = {
        "stage": 68,
        "description": (
            "Independent frozen-field comparison of the retained first-order "
            "upwind transport operator and a conservative minmod-limited "
            "second-order control-volume operator on the exact completed "
            "Stage-67 phi/psi distributions."
        ),
        "configuration": {
            "grid": list(STAGE68_GRID),
            "kn0": STAGE68_KNUDSEN,
            "cold_hot_ratio": STAGE68_COLD_HOT_RATIO,
            "radial_nodes": STAGE68_RULE[0],
            "angular_nodes": STAGE68_RULE[1],
            "point_count": STAGE68_POINT_COUNT,
            "radial_scale": STAGE68_RADIAL_SCALE,
            "chunk_size": STAGE68_CHUNK_SIZE,
            "limiter": STAGE68_LIMITER,
            "second_order_operator": STAGE68_SECOND_ORDER_OPERATOR,
            "material_heat_flux_ratio": STAGE68_MATERIAL_HEAT_FLUX_RATIO,
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
        "retained_operator_consistency": consistency,
        "operator_summaries": summaries,
        "operator_difference_rms_ratios": ratios,
        "normal_heat_flux_operator_difference_ratio": float(ratios["qy"]),
        "difference_qy_wall_band_absolute_share": wall_band_absolute_share(
            maps["difference"]["qy"], STAGE68_WALL_BAND_LAYERS
        ),
        "correlations": correlations,
        "finite": finite,
        "decision": decision,
        "interpretation_guard": (
            "The second-order-minus-first-order residual is a frozen operator "
            "defect, not an adjoint sensitivity, converged observable change, "
            "or evidence that Table 3 or Table 6 improves. The failed Stage-28 "
            "MUSCL endpoint remains negative and is not rehabilitated."
        ),
        "negative_findings": [
            "No cavity solve is rerun and no physical or numerical parameter is retuned.",
            "A material residual difference does not establish the sign or magnitude of a converged heat-flux response.",
            "Cross-Knudsen extension remains prohibited regardless of this frozen-field diagnostic.",
        ],
        "scientifically_justified_next_scope": (
            "If the higher-order heat-flux residual is material, test its grid-transfer "
            "scaling on frozen restricted fields before any solver experiment. If it is "
            "not material, audit the wall-face flux discretization independently."
        ),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage67-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = run_stage68(args.stage67_artifact_dir, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
