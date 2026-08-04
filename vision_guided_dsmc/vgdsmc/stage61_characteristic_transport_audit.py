from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np

from .linear_sidewall_validation import LinearSidewallConfig, sidewall_temperature_profile
from .stage41_projected_polar_operator_audit import (
    PolarQuadrature,
    mapped_polar_quadrature,
    projected_macroscopic,
    projected_maxwellian,
)
from .stage42_projected_polar_heated_cavity_pilot import bottom_wall_heat_flux


STAGE60_COMPLETED_ENDPOINT = {
    "workflow_run_id": 30897895198,
    "workflow_job_id": 91955054751,
    "workflow_conclusion": "success",
    "tests_passed": 12,
    "tests_failed": 0,
    "artifact_id": 8891954081,
    "artifact_size_bytes": 2802,
    "artifact_sha256": "1f40ad0d84fb6388eaa8ecb6299861efaaec041dd16b062ce1d5f1ce803aad31",
    "source_head_sha": "12e4ed729c05ea6602aa80f89d8d649600198e1b",
    "summary_sha256": "38201cc72f824c27b588bdc3c2b7a82973d2a0de886e7b07ebcd02f4af1790a3",
    "decision": (
        "stage60_transport_and_diffuse_wall_equations_close_"
        "discrepancy_not_explained_characteristic_audit_next"
    ),
}

STAGE61_GRIDS = ((8, 8), (16, 16), (32, 32))
STAGE61_RULE = (40, 96)
STAGE61_RADIAL_SCALE = 2.0
STAGE61_KNUDSEN_SCOPE = 10.0
STAGE61_COLD_HOT_RATIO = 0.1
STAGE61_EIGENVALUE_TOLERANCE = 1.0e-10
STAGE61_WALL_BALANCE_TOLERANCE = 1.0e-10
STAGE61_DISCRETE_RESIDUAL_TOLERANCE = 1.0e-12
STAGE61_MATERIAL_ERROR_THRESHOLD = 0.10
STAGE61_CORNER_TIE_TOLERANCE = 1.0e-12

WALL_LEFT = 0
WALL_RIGHT = 1
WALL_BOTTOM = 2
WALL_TOP = 3


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_stage60_artifact(root: str | Path) -> dict[str, object]:
    root = Path(root)
    summary_path = root / "summary.json"
    if not summary_path.is_file():
        raise ValueError("Stage 60 summary is missing")
    if sha256_file(summary_path) != STAGE60_COMPLETED_ENDPOINT["summary_sha256"]:
        raise ValueError("Stage 60 summary checksum mismatch")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("stage") != 60:
        raise ValueError("Stage 60 artifact stage mismatch")
    if summary.get("decision") != STAGE60_COMPLETED_ENDPOINT["decision"]:
        raise ValueError("Stage 60 artifact decision mismatch")
    return summary


def validate_stage61_design(
    grids: tuple[tuple[int, int], ...],
    rule: tuple[int, int],
    radial_scale: float,
    kn0_scope: float,
    cold_hot_ratio: float,
    material_threshold: float,
) -> None:
    if grids != STAGE61_GRIDS:
        raise ValueError("Stage 61 uses the preregistered 8/16/32 spatial sequence")
    if rule != STAGE61_RULE:
        raise ValueError("Stage 61 retains the Stage 58 40x96 velocity rule")
    if radial_scale != STAGE61_RADIAL_SCALE:
        raise ValueError("Stage 61 retains radial mapping scale 2.0")
    if kn0_scope != STAGE61_KNUDSEN_SCOPE:
        raise ValueError("Stage 61 remains scoped to the Kn0=10 investigation")
    if cold_hot_ratio != STAGE61_COLD_HOT_RATIO:
        raise ValueError("Stage 61 retains Tcold/Thot=0.1")
    if material_threshold != STAGE61_MATERIAL_ERROR_THRESHOLD:
        raise ValueError("Stage 61 retains the preregistered 10% material threshold")


def _wall_offsets(nx: int, ny: int) -> tuple[int, int, int, int, int]:
    return 0, ny, 2 * ny, 2 * ny + nx, 2 * ny + 2 * nx


def _wall_metadata(cfg: LinearSidewallConfig) -> dict[str, np.ndarray]:
    left, right, bottom, top, total = _wall_offsets(cfg.nx, cfg.ny)
    wall = np.empty(total, dtype=np.int8)
    local = np.empty(total, dtype=np.int32)
    x = np.empty(total)
    y = np.empty(total)
    temperature = np.empty(total)
    face_length = np.empty(total)
    yc = (np.arange(cfg.ny, dtype=np.float64) + 0.5) / cfg.ny
    xc = (np.arange(cfg.nx, dtype=np.float64) + 0.5) / cfg.nx
    side_temperature = sidewall_temperature_profile(cfg)

    wall[left:right] = WALL_LEFT
    local[left:right] = np.arange(cfg.ny)
    x[left:right] = 0.0
    y[left:right] = yc
    temperature[left:right] = side_temperature
    face_length[left:right] = 1.0 / cfg.ny

    wall[right:bottom] = WALL_RIGHT
    local[right:bottom] = np.arange(cfg.ny)
    x[right:bottom] = 1.0
    y[right:bottom] = yc
    temperature[right:bottom] = side_temperature
    face_length[right:bottom] = 1.0 / cfg.ny

    wall[bottom:top] = WALL_BOTTOM
    local[bottom:top] = np.arange(cfg.nx)
    x[bottom:top] = xc
    y[bottom:top] = 0.0
    temperature[bottom:top] = cfg.hot_temperature
    face_length[bottom:top] = 1.0 / cfg.nx

    wall[top:total] = WALL_TOP
    local[top:total] = np.arange(cfg.nx)
    x[top:total] = xc
    y[top:total] = 1.0
    temperature[top:total] = cfg.cold_temperature
    face_length[top:total] = 1.0 / cfg.nx
    return {
        "wall": wall,
        "local": local,
        "x": x,
        "y": y,
        "temperature": temperature,
        "face_length": face_length,
    }


def _face_id_from_hit(
    wall: np.ndarray, coordinate: np.ndarray, nx: int, ny: int,
) -> np.ndarray:
    left, right, bottom, top, _ = _wall_offsets(nx, ny)
    wall_b, coordinate_b = np.broadcast_arrays(wall, coordinate)
    result = np.empty(wall_b.shape, dtype=np.int32)
    vertical = np.clip((coordinate_b * ny).astype(np.int64), 0, ny - 1)
    horizontal = np.clip((coordinate_b * nx).astype(np.int64), 0, nx - 1)
    result[wall_b == WALL_LEFT] = left + vertical[wall_b == WALL_LEFT]
    result[wall_b == WALL_RIGHT] = right + vertical[wall_b == WALL_RIGHT]
    result[wall_b == WALL_BOTTOM] = bottom + horizontal[wall_b == WALL_BOTTOM]
    result[wall_b == WALL_TOP] = top + horizontal[wall_b == WALL_TOP]
    return result


def trace_back_to_wall_faces(
    x: np.ndarray | float,
    y: np.ndarray | float,
    vx: np.ndarray,
    vy: np.ndarray,
    nx: int,
    ny: int,
    tie_tolerance: float = STAGE61_CORNER_TIE_TOLERANCE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trace x-t*v to the emitting wall, splitting exact corner hits equally."""
    x, y, vx, vy = np.broadcast_arrays(
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        np.asarray(vx, dtype=np.float64),
        np.asarray(vy, dtype=np.float64),
    )
    bx = -vx
    by = -vy
    eps = 1.0e-14
    tx = np.full(x.shape, np.inf)
    ty = np.full(x.shape, np.inf)

    leftward = bx < 0.0
    candidate = np.where(leftward, -x / np.where(leftward, bx, -1.0), np.inf)
    valid = leftward & (candidate > eps)
    tx[valid] = candidate[valid]
    rightward = bx > 0.0
    candidate = np.where(rightward, (1.0 - x) / np.where(rightward, bx, 1.0), np.inf)
    valid = rightward & (candidate > eps)
    tx[valid] = candidate[valid]

    downward = by < 0.0
    candidate = np.where(downward, -y / np.where(downward, by, -1.0), np.inf)
    valid = downward & (candidate > eps)
    ty[valid] = candidate[valid]
    upward = by > 0.0
    candidate = np.where(upward, (1.0 - y) / np.where(upward, by, 1.0), np.inf)
    valid = upward & (candidate > eps)
    ty[valid] = candidate[valid]

    if np.any(~np.isfinite(np.minimum(tx, ty))):
        raise ValueError("every nonzero characteristic must reach a wall")
    scale = np.maximum(np.minimum(tx, ty), 1.0)
    tie = np.abs(tx - ty) <= tie_tolerance * scale
    choose_y = ty < tx
    x_wall = np.where(bx < 0.0, WALL_LEFT, WALL_RIGHT).astype(np.int8)
    y_wall = np.where(by < 0.0, WALL_BOTTOM, WALL_TOP).astype(np.int8)
    y_hit_x = np.clip(y + by * tx, 0.0, 1.0)
    x_hit_y = np.clip(x + bx * ty, 0.0, 1.0)
    x_face = _face_id_from_hit(x_wall, y_hit_x, nx, ny)
    y_face = _face_id_from_hit(y_wall, x_hit_y, nx, ny)
    source_a = np.where(choose_y, y_face, x_face)
    source_b = source_a.copy()
    blend_b = np.zeros(x.shape, dtype=np.float64)
    source_a[tie] = x_face[tie]
    source_b[tie] = y_face[tie]
    blend_b[tie] = 0.5
    return source_a.astype(np.int32), source_b.astype(np.int32), blend_b


def _wall_inward_normal_velocity(
    wall: np.ndarray, quadrature: PolarQuadrature,
) -> np.ndarray:
    normal = np.empty(wall.shape + (quadrature.point_count,), dtype=np.float64)
    normal[wall == WALL_LEFT] = quadrature.vx
    normal[wall == WALL_RIGHT] = -quadrature.vx
    normal[wall == WALL_BOTTOM] = quadrature.vy
    normal[wall == WALL_TOP] = -quadrature.vy
    return normal


def build_characteristic_wall_operator(
    cfg: LinearSidewallConfig, quadrature: PolarQuadrature,
) -> dict[str, object]:
    metadata = _wall_metadata(cfg)
    wall = metadata["wall"]
    total = int(wall.size)
    one = np.ones(total)
    zero = np.zeros(total)
    unit_phi, unit_psi = projected_maxwellian(
        one, zero, zero, metadata["temperature"], quadrature
    )
    normal = _wall_inward_normal_velocity(wall, quadrature)
    incoming = normal > 0.0
    outgoing = normal < 0.0
    incoming_unit_flux = np.sum(
        normal * unit_phi * incoming * quadrature.weight[None, :], axis=1
    )
    if np.any(incoming_unit_flux <= 0.0):
        raise ValueError("incoming unit wall flux must be positive")

    transfer = np.zeros((total, total), dtype=np.float64)
    source_a = np.empty((total, quadrature.point_count), dtype=np.int32)
    source_b = np.empty_like(source_a)
    blend_b = np.empty((total, quadrature.point_count), dtype=np.float64)
    corner_count = 0
    for face in range(total):
        source_a[face].fill(face)
        source_b[face].fill(face)
        blend_b[face].fill(0.0)
        q = np.flatnonzero(outgoing[face])
        a, b, blend = trace_back_to_wall_faces(
            metadata["x"][face], metadata["y"][face],
            quadrature.vx[q], quadrature.vy[q], cfg.nx, cfg.ny,
        )
        source_a[face, q] = a
        source_b[face, q] = b
        blend_b[face, q] = blend
        corner_count += int(np.count_nonzero(blend > 0.0))
        coefficient = -normal[face, q] * quadrature.weight[q] / incoming_unit_flux[face]
        wa = 1.0 - blend
        wb = blend
        np.add.at(transfer[face], a, coefficient * wa * unit_phi[a, q])
        tied = wb > 0.0
        if np.any(tied):
            qt = q[tied]
            np.add.at(
                transfer[face], b[tied],
                coefficient[tied] * wb[tied] * unit_phi[b[tied], qt],
            )

    eigenvalues, eigenvectors = np.linalg.eig(transfer)
    index = int(np.argmax(np.abs(eigenvalues)))
    eigenvalue = eigenvalues[index]
    if abs(float(np.imag(eigenvalue))) > STAGE61_EIGENVALUE_TOLERANCE:
        raise ValueError("dominant characteristic wall eigenvalue is not real")
    alpha = np.real(eigenvectors[:, index])
    if np.sum(alpha * metadata["face_length"]) < 0.0:
        alpha = -alpha
    if np.min(alpha) <= 0.0:
        alpha = np.abs(alpha)
    normalization = np.sum(alpha * metadata["face_length"]) / np.sum(metadata["face_length"])
    alpha /= max(float(normalization), 1.0e-300)
    eigenvalue_real = float(np.real(eigenvalue))
    eigen_residual = float(
        np.linalg.norm(transfer @ alpha - eigenvalue_real * alpha)
        / max(np.linalg.norm(alpha), 1.0e-300)
    )
    return {
        "metadata": metadata,
        "unit_phi": unit_phi,
        "unit_psi": unit_psi,
        "normal_velocity": normal,
        "incoming_mask": incoming,
        "outgoing_mask": outgoing,
        "incoming_unit_flux": incoming_unit_flux,
        "transfer": transfer,
        "alpha": alpha,
        "dominant_eigenvalue": eigenvalue_real,
        "dominant_eigenvalue_defect": abs(eigenvalue_real - 1.0),
        "eigen_residual": eigen_residual,
        "source_a": source_a,
        "source_b": source_b,
        "blend_b": blend_b,
        "corner_tie_fraction": corner_count / float(total * quadrature.point_count),
    }


def characteristic_boundary_distributions(
    operator: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    unit_phi = np.asarray(operator["unit_phi"])
    unit_psi = np.asarray(operator["unit_psi"])
    alpha = np.asarray(operator["alpha"])
    incoming = np.asarray(operator["incoming_mask"])
    a = np.asarray(operator["source_a"])
    b = np.asarray(operator["source_b"])
    blend = np.asarray(operator["blend_b"])
    q = np.arange(unit_phi.shape[1])[None, :]
    emitted_phi = alpha[:, None] * unit_phi
    emitted_psi = alpha[:, None] * unit_psi
    outgoing_phi = (
        (1.0 - blend) * alpha[a] * unit_phi[a, q]
        + blend * alpha[b] * unit_phi[b, q]
    )
    outgoing_psi = (
        (1.0 - blend) * alpha[a] * unit_psi[a, q]
        + blend * alpha[b] * unit_psi[b, q]
    )
    return np.where(incoming, emitted_phi, outgoing_phi), np.where(incoming, emitted_psi, outgoing_psi)


def _characteristic_cell_distributions(
    cfg: LinearSidewallConfig,
    quadrature: PolarQuadrature,
    operator: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    unit_phi = np.asarray(operator["unit_phi"])
    unit_psi = np.asarray(operator["unit_psi"])
    alpha = np.asarray(operator["alpha"])
    x = (np.arange(cfg.nx, dtype=np.float64) + 0.5) / cfg.nx
    y = (np.arange(cfg.ny, dtype=np.float64) + 0.5) / cfg.ny
    phi = np.empty((cfg.ny, cfg.nx, quadrature.point_count))
    psi = np.empty_like(phi)
    q = np.arange(quadrature.point_count)[None, :]
    for i, yi in enumerate(y):
        a, b, blend = trace_back_to_wall_faces(
            x[:, None], yi, quadrature.vx[None, :], quadrature.vy[None, :],
            cfg.nx, cfg.ny,
        )
        phi[i] = (
            (1.0 - blend) * alpha[a] * unit_phi[a, q]
            + blend * alpha[b] * unit_phi[b, q]
        )
        psi[i] = (
            (1.0 - blend) * alpha[a] * unit_psi[a, q]
            + blend * alpha[b] * unit_psi[b, q]
        )
    return phi, psi


def _inflow_profiles(
    cfg: LinearSidewallConfig, operator: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left, right, bottom, top, total = _wall_offsets(cfg.nx, cfg.ny)
    alpha = np.asarray(operator["alpha"])
    emitted_phi = alpha[:, None] * np.asarray(operator["unit_phi"])
    emitted_psi = alpha[:, None] * np.asarray(operator["unit_psi"])
    return (
        emitted_phi[left:right], emitted_psi[left:right],
        emitted_phi[right:bottom], emitted_psi[right:bottom],
        emitted_phi[bottom:top], emitted_psi[bottom:top],
        emitted_phi[top:total], emitted_psi[top:total],
    )


def solve_first_order_upwind_with_fixed_inflow(
    cfg: LinearSidewallConfig,
    quadrature: PolarQuadrature,
    inflow: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                  np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    (
        left_phi, left_psi, right_phi, right_psi,
        bottom_phi, bottom_psi, top_phi, top_psi,
    ) = inflow
    phi = np.empty((cfg.ny, cfg.nx, quadrature.point_count))
    psi = np.empty_like(phi)
    ax_all = np.abs(quadrature.vx) * cfg.nx
    ay_all = np.abs(quadrature.vy) * cfg.ny
    for sx in (1, -1):
        mask_x = quadrature.vx > 0.0 if sx == 1 else quadrature.vx <= 0.0
        j_values = range(cfg.nx) if sx == 1 else range(cfg.nx - 1, -1, -1)
        for sy in (1, -1):
            mask_y = quadrature.vy > 0.0 if sy == 1 else quadrature.vy <= 0.0
            q = np.flatnonzero(mask_x & mask_y)
            if q.size == 0:
                continue
            i_values = range(cfg.ny) if sy == 1 else range(cfg.ny - 1, -1, -1)
            ax = ax_all[q]
            ay = ay_all[q]
            denominator = ax + ay
            for i in i_values:
                for j in j_values:
                    if sx == 1:
                        x_phi = left_phi[i, q] if j == 0 else phi[i, j - 1, q]
                        x_psi = left_psi[i, q] if j == 0 else psi[i, j - 1, q]
                    else:
                        x_phi = right_phi[i, q] if j == cfg.nx - 1 else phi[i, j + 1, q]
                        x_psi = right_psi[i, q] if j == cfg.nx - 1 else psi[i, j + 1, q]
                    if sy == 1:
                        y_phi = bottom_phi[j, q] if i == 0 else phi[i - 1, j, q]
                        y_psi = bottom_psi[j, q] if i == 0 else psi[i - 1, j, q]
                    else:
                        y_phi = top_phi[j, q] if i == cfg.ny - 1 else phi[i + 1, j, q]
                        y_psi = top_psi[j, q] if i == cfg.ny - 1 else psi[i + 1, j, q]
                    phi[i, j, q] = (ax * x_phi + ay * y_phi) / denominator
                    psi[i, j, q] = (ax * x_psi + ay * y_psi) / denominator
    return phi, psi


def first_order_residual_relative_error(
    distribution: np.ndarray,
    cfg: LinearSidewallConfig,
    quadrature: PolarQuadrature,
    left: np.ndarray,
    right: np.ndarray,
    bottom: np.ndarray,
    top: np.ndarray,
) -> float:
    ax = np.abs(quadrature.vx) * cfg.nx
    ay = np.abs(quadrature.vy) * cfg.ny
    ln = np.empty_like(distribution)
    rn = np.empty_like(distribution)
    bn = np.empty_like(distribution)
    tn = np.empty_like(distribution)
    ln[:, 1:] = distribution[:, :-1]
    ln[:, 0] = left
    rn[:, :-1] = distribution[:, 1:]
    rn[:, -1] = right
    bn[1:] = distribution[:-1]
    bn[0] = bottom
    tn[:-1] = distribution[1:]
    tn[-1] = top
    x_up = np.where((quadrature.vx > 0.0)[None, None, :], ln, rn)
    y_up = np.where((quadrature.vy > 0.0)[None, None, :], bn, tn)
    residual = (
        (ax + ay)[None, None, :] * distribution
        - ax[None, None, :] * x_up
        - ay[None, None, :] * y_up
    )
    scale = (
        (ax + ay)[None, None, :] * np.abs(distribution)
        + ax[None, None, :] * np.abs(x_up)
        + ay[None, None, :] * np.abs(y_up)
    )
    return float(np.sum(np.abs(residual)) / max(float(np.sum(scale)), 1.0e-300))


def _weighted_distribution_relative_l1(
    candidate_phi: np.ndarray,
    candidate_psi: np.ndarray,
    reference_phi: np.ndarray,
    reference_psi: np.ndarray,
    quadrature: PolarQuadrature,
) -> float:
    weight = quadrature.weight[None, None, :]
    numerator = np.sum(
        (np.abs(candidate_phi - reference_phi) + np.abs(candidate_psi - reference_psi)) * weight
    )
    denominator = np.sum((np.abs(reference_phi) + np.abs(reference_psi)) * weight)
    return float(numerator / max(float(denominator), 1.0e-300))


def _relative_rms(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(candidate) - np.asarray(reference))
        / max(float(np.linalg.norm(reference)), 1.0e-300)
    )


def _wall_mass_balance(
    boundary_phi: np.ndarray,
    normal_velocity: np.ndarray,
    quadrature: PolarQuadrature,
) -> float:
    net = np.sum(normal_velocity * boundary_phi * quadrature.weight[None, :], axis=1)
    scale = np.sum(
        np.abs(normal_velocity * boundary_phi) * quadrature.weight[None, :], axis=1
    )
    return float(np.max(np.abs(net) / np.maximum(scale, 1.0e-300)))


def evaluate_grid(
    grid: tuple[int, int], quadrature: PolarQuadrature,
) -> dict[str, object]:
    cfg = LinearSidewallConfig(
        nx=grid[0], ny=grid[1], kn0=STAGE61_KNUDSEN_SCOPE,
        cold_hot_ratio=STAGE61_COLD_HOT_RATIO,
    )
    operator = build_characteristic_wall_operator(cfg, quadrature)
    exact_boundary_phi, exact_boundary_psi = characteristic_boundary_distributions(operator)
    exact_phi, exact_psi = _characteristic_cell_distributions(cfg, quadrature, operator)
    inflow = _inflow_profiles(cfg, operator)
    upwind_phi, upwind_psi = solve_first_order_upwind_with_fixed_inflow(cfg, quadrature, inflow)
    exact_fields = projected_macroscopic(exact_phi, exact_psi, quadrature)
    upwind_fields = projected_macroscopic(upwind_phi, upwind_psi, quadrature)

    left, right, bottom, top, total = _wall_offsets(cfg.nx, cfg.ny)
    normal = np.asarray(operator["normal_velocity"])
    exact_wall_balance = _wall_mass_balance(exact_boundary_phi, normal, quadrature)
    (
        left_phi, left_psi, right_phi, right_psi,
        bottom_phi, bottom_psi, top_phi, top_psi,
    ) = inflow
    discrete_boundary_phi = np.empty_like(exact_boundary_phi)
    discrete_boundary_phi[left:right] = np.where(
        (quadrature.vx > 0.0)[None, :], left_phi, upwind_phi[:, 0]
    )
    discrete_boundary_phi[right:bottom] = np.where(
        (quadrature.vx < 0.0)[None, :], right_phi, upwind_phi[:, -1]
    )
    discrete_boundary_phi[bottom:top] = np.where(
        (quadrature.vy > 0.0)[None, :], bottom_phi, upwind_phi[0]
    )
    discrete_boundary_phi[top:total] = np.where(
        (quadrature.vy < 0.0)[None, :], top_phi, upwind_phi[-1]
    )
    discrete_wall_balance = _wall_mass_balance(discrete_boundary_phi, normal, quadrature)

    exact_bottom_phi = exact_boundary_phi[bottom:top]
    exact_bottom_psi = exact_boundary_psi[bottom:top]
    speed2 = quadrature.vx**2 + quadrature.vy**2
    exact_bottom_q = 0.5 * np.sum(
        quadrature.vy[None, :]
        * (speed2[None, :] * exact_bottom_phi + exact_bottom_psi)
        * quadrature.weight[None, :], axis=1,
    ) / math.sqrt(2.0)
    upwind_bottom_q = bottom_wall_heat_flux(
        upwind_phi, upwind_psi, bottom_phi, bottom_psi, quadrature
    )
    exact_qav = float(np.mean(exact_bottom_q))
    upwind_qav = float(np.mean(upwind_bottom_q))
    heat_flux_reference = np.stack([exact_fields["qx"], exact_fields["qy"]], axis=-1)
    heat_flux_candidate = np.stack([upwind_fields["qx"], upwind_fields["qy"]], axis=-1)
    velocity_reference = np.stack([exact_fields["u"], exact_fields["v"]], axis=-1)
    velocity_candidate = np.stack([upwind_fields["u"], upwind_fields["v"]], axis=-1)

    return {
        "grid": list(grid),
        "wall_faces": int(total),
        "wall_operator_dominant_eigenvalue": float(operator["dominant_eigenvalue"]),
        "wall_operator_eigenvalue_defect": float(operator["dominant_eigenvalue_defect"]),
        "wall_operator_eigen_residual": float(operator["eigen_residual"]),
        "corner_tie_fraction": float(operator["corner_tie_fraction"]),
        "exact_maximum_wall_mass_balance_error": exact_wall_balance,
        "fixed_inflow_upwind_maximum_wall_mass_balance_error": discrete_wall_balance,
        "phi_discrete_residual_relative_error": first_order_residual_relative_error(
            upwind_phi, cfg, quadrature, left_phi, right_phi, bottom_phi, top_phi
        ),
        "psi_discrete_residual_relative_error": first_order_residual_relative_error(
            upwind_psi, cfg, quadrature, left_psi, right_psi, bottom_psi, top_psi
        ),
        "distribution_weighted_relative_l1": _weighted_distribution_relative_l1(
            upwind_phi, upwind_psi, exact_phi, exact_psi, quadrature
        ),
        "temperature_relative_rms": _relative_rms(upwind_fields["T"], exact_fields["T"]),
        "velocity_relative_rms": _relative_rms(velocity_candidate, velocity_reference),
        "heat_flux_relative_rms": _relative_rms(heat_flux_candidate, heat_flux_reference),
        "exact_bottom_heat_flux_average": exact_qav,
        "upwind_bottom_heat_flux_average": upwind_qav,
        "bottom_heat_flux_average_relative_error": abs(upwind_qav - exact_qav)
        / max(abs(exact_qav), 1.0e-300),
        "finite": bool(
            np.all(np.isfinite(exact_phi))
            and np.all(np.isfinite(exact_psi))
            and np.all(np.isfinite(upwind_phi))
            and np.all(np.isfinite(upwind_psi))
        ),
    }


def _strictly_nonincreasing(values: list[float], slack: float = 1.0e-12) -> bool:
    return all(values[i + 1] <= values[i] + slack for i in range(len(values) - 1))


def evaluate_stage61() -> dict[str, object]:
    validate_stage61_design(
        STAGE61_GRIDS, STAGE61_RULE, STAGE61_RADIAL_SCALE,
        STAGE61_KNUDSEN_SCOPE, STAGE61_COLD_HOT_RATIO,
        STAGE61_MATERIAL_ERROR_THRESHOLD,
    )
    quadrature = mapped_polar_quadrature(
        STAGE61_RULE[0], STAGE61_RULE[1], STAGE61_RADIAL_SCALE
    )
    rows = [evaluate_grid(grid, quadrature) for grid in STAGE61_GRIDS]
    finite_pass = all(bool(row["finite"]) for row in rows)
    wall_operator_pass = all(
        float(row["wall_operator_eigenvalue_defect"]) <= STAGE61_EIGENVALUE_TOLERANCE
        and float(row["wall_operator_eigen_residual"]) <= STAGE61_EIGENVALUE_TOLERANCE
        and float(row["exact_maximum_wall_mass_balance_error"]) <= STAGE61_WALL_BALANCE_TOLERANCE
        for row in rows
    )
    residual_pass = all(
        max(
            float(row["phi_discrete_residual_relative_error"]),
            float(row["psi_discrete_residual_relative_error"]),
        ) <= STAGE61_DISCRETE_RESIDUAL_TOLERANCE
        for row in rows
    )
    metrics = (
        "distribution_weighted_relative_l1",
        "temperature_relative_rms",
        "heat_flux_relative_rms",
        "bottom_heat_flux_average_relative_error",
    )
    monotonic = {
        metric: _strictly_nonincreasing([float(row[metric]) for row in rows])
        for metric in metrics
    }
    monotonic_pass = all(monotonic.values())
    finest = rows[-1]
    material_bias = max(
        float(finest["heat_flux_relative_rms"]),
        float(finest["bottom_heat_flux_average_relative_error"]),
    ) >= STAGE61_MATERIAL_ERROR_THRESHOLD

    if not finite_pass:
        decision = "stage61_nonfinite_characteristic_or_upwind_blocker"
        next_scope = "Review the independent characteristic tracing and fixed-inflow upwind solve before any physical interpretation."
    elif not wall_operator_pass:
        decision = "stage61_characteristic_wall_fixed_point_blocker"
        next_scope = "Review wall-face ray reciprocity, corner treatment and diffuse-wall eigenproblem before any further stage."
    elif not residual_pass:
        decision = "stage61_first_order_upwind_algebraic_residual_blocker"
        next_scope = "Review the causal upwind sweep before using the transport-diffusion comparison."
    elif not monotonic_pass:
        decision = "stage61_transport_error_not_monotonic_decision_blocker"
        next_scope = "Inspect face discretization and corner-ray sensitivity; do not infer a convergent transport-diffusion trend."
    elif material_bias:
        decision = "stage61_material_collision_off_transport_diffusion_stage62_64x64_confirmation"
        next_scope = (
            "Run one preregistered 64x64 characteristic-versus-first-order confirmation with the same "
            "40x96 quadrature and fixed inflow; do not retune the transport operator or extend across Knudsen number."
        )
    else:
        decision = "stage61_collision_off_transport_diffusion_below_material_threshold_finite_kn_audit_next"
        next_scope = (
            "Audit the finite-Kn relaxation-frequency and collision-transport normalization at Kn0=10, "
            "because collision-off first-order diffusion at 32x32 is below the 10% material threshold."
        )

    return {
        "stage": 61,
        "description": (
            "Independent face-consistent characteristic solution of the collision-off non-isothermal "
            "diffuse cavity, compared with the algebraically exact first-order upwind solution under "
            "identical prescribed inflow on an 8/16/32 spatial sequence."
        ),
        "retained_stage60_endpoint": STAGE60_COMPLETED_ENDPOINT,
        "configuration": {
            "spatial_grids": [list(grid) for grid in STAGE61_GRIDS],
            "velocity_rule": list(STAGE61_RULE),
            "radial_scale": STAGE61_RADIAL_SCALE,
            "kn0_investigation_scope": STAGE61_KNUDSEN_SCOPE,
            "collision_operator": "disabled_diagnostic_limit",
            "cold_hot_ratio": STAGE61_COLD_HOT_RATIO,
            "wall_temperature_discretization": "same_piecewise_constant_face_profile_in_both_arms",
            "wall_density_solution": "dominant_eigenvector_of_characteristic_diffuse_transfer_operator",
            "corner_characteristic_convention": "equal_split_between_the_two_walls_for_exact_corner_hits",
            "upwind_boundary_condition": "fixed_to_characteristic_wall_emission_to_isolate_interior_transport",
            "physical_parameter_retuning": False,
            "collision_parameter_retuning": False,
            "transport_retuning": False,
            "wall_model_retuning": False,
            "normalization_retuning": False,
            "velocity_quadrature_retuning": False,
            "cross_knudsen_extension_permitted": False,
            "external_validation_claim_permitted": False,
        },
        "thresholds": {
            "wall_eigenvalue_and_residual": STAGE61_EIGENVALUE_TOLERANCE,
            "exact_wall_mass_balance": STAGE61_WALL_BALANCE_TOLERANCE,
            "upwind_algebraic_residual": STAGE61_DISCRETE_RESIDUAL_TOLERANCE,
            "material_transport_error": STAGE61_MATERIAL_ERROR_THRESHOLD,
        },
        "rows": rows,
        "checks": {
            "finite_pass": bool(finite_pass),
            "characteristic_wall_fixed_point_pass": bool(wall_operator_pass),
            "first_order_upwind_algebraic_residual_pass": bool(residual_pass),
            "spatial_error_monotonic_pass": bool(monotonic_pass),
        },
        "monotonic_error_metrics": monotonic,
        "finest_grid_material_transport_bias": bool(material_bias),
        "decision": decision,
        "positive_findings": [
            "The characteristic arm is independent of the interior finite-volume transport stencil and uses only straight-line ray tracing plus the same frozen velocity quadrature.",
            "The paired upwind arm is solved to its algebraic steady state under exactly the same wall-face inflow, isolating interior first-order transport diffusion from collision and wall-density feedback.",
            "All grid levels and all pass/fail thresholds were fixed before execution; no failed physical or numerical parameter is retuned.",
        ],
        "negative_findings": [
            "This is a collision-off diagnostic limit, not a Kn0=10 cavity solution and not external validation.",
            "The wall temperature is face-discretized and exact corner rays use a documented equal-split convention; both are retained limitations of the comparison.",
            "Because the upwind arm holds the characteristic wall emission fixed, any diffuse-wall density feedback induced by the discrete outgoing field is measured as wall imbalance but is not iterated away in this stage.",
            "A transport-diffusion discrepancy cannot by itself prove the cause of the greater-than-25% finite-Kn heat-flux error confirmed in Stage 59.",
        ],
        "interpretation_guard": (
            "Stage 61 can quantify a collision-off first-order transport bias and its spatial trend. "
            "It cannot validate the finite-Kn solver, justify parameter retuning, adopt the conservative "
            "projection, or authorize cross-Knudsen extension."
        ),
        "scientifically_justified_next_scope": next_scope,
    }


def run_stage61(stage60_artifact_dir: str | Path, output_dir: str | Path) -> dict[str, object]:
    validate_stage60_artifact(stage60_artifact_dir)
    summary = evaluate_stage61()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage60-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(
        run_stage61(args.stage60_artifact_dir, args.output_dir),
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
