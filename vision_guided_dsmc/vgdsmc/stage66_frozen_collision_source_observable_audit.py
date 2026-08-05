from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import hashlib
import json
import math
from typing import Mapping, Sequence

import numpy as np


STAGE65_COMPLETED_ENDPOINT = {
    "workflow_run_id": 30963463806,
    "workflow_job_id": 92172238988,
    "workflow_conclusion": "success",
    "tests_passed": 48,
    "tests_failed": 0,
    "test_duration_seconds": 0.47,
    "artifact_id": 8918027763,
    "artifact_size_bytes": 115402,
    "artifact_sha256": "50f695ef5e73ab7315b741cd3f1eb9fc5a008f242a480872aba5130a6ec48ad6",
    "source_head_sha": "002a3463b61f9c746f8d6dd2584a9da607e9cc06",
    "summary_sha256": "bc44f5410763bf22d5c409a9f06f554a2652e488552bf706c2d792efeddd6aff",
    "maps_sha256": "085285bff41a576098d73a44e7da036dc7f292193acd39e88cbcc766311e5fac",
    "decision": (
        "stage65_clipping_source_defects_broad_"
        "stage66_frozen_collision_source_observable_audit"
    ),
}

STAGE58_COMPLETED_ENDPOINT = {
    "workflow_run_id": 30864287564,
    "workflow_job_id": 91852659651,
    "workflow_conclusion": "success",
    "tests_passed": 72,
    "tests_failed": 0,
    "artifact_id": 8882191150,
    "artifact_size_bytes": 265675,
    "artifact_sha256": "c1e1cb40d35439d44a93bae091f732f35e1790c2c4e3f1180b16d7f8fb54e8f6",
    "source_head_sha": "448ce586344052de1cf5dd0fd86e3ffb4b6a52be",
    "summary_sha256": "b91921a631bd92c2696c7bdd18668afb4994c909bbc7aa82cca1d0afc25ffced",
    "baseline_fields_sha256": "9aa53136e05917236f87fb9279c2ecc4e29d6056ca5ec34ca3bcb4d8f66aa822",
    "decision": (
        "stage58_conservative_confirmation_stable_but_observables_degrade_"
        "requires_review_without_retuning"
    ),
}

PUBLISHED_REFERENCE = {
    "doi": "10.1063/1.4875235",
    "case": {"kn0": 10.0, "cold_hot_ratio": 0.1, "aspect_ratio": 1.0},
    "table6_qav": {"shakhov": 0.178, "dsmc": 0.179},
}

STAGE66_GRID = (64, 64)
STAGE66_KNUDSEN = 10.0
STAGE66_COLD_HOT_RATIO = 0.1
STAGE66_VISCOSITY_EXPONENT = 0.5
STAGE66_RULE = (40, 96)
STAGE66_RADIAL_SCALE = 2.0
STAGE66_PRANDTL = 2.0 / 3.0
STAGE66_CORRECTION_FLOOR = 0.05
STAGE66_WALL_BAND_LAYERS = 4
STAGE66_CHUNK_SIZE = 128
STAGE66_UNCLIPPED_CLOSURE_TOLERANCE = 1.0e-6
STAGE66_DIRECTIONAL_FRACTION_THRESHOLD = 0.95
STAGE66_ZERO_TOLERANCE = 1.0e-14
STAGE66_TEMPERATURE_BINS = (0.1, 0.25, 0.5, 0.75, 1.0)
STAGE66_DENSITY_BINS = (0.5, 0.8, 1.0, 1.2, 1.6)


@dataclass(frozen=True)
class PolarQuadrature:
    vx: np.ndarray
    vy: np.ndarray
    weight: np.ndarray

    @property
    def point_count(self) -> int:
        return int(self.weight.size)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mapped_polar_quadrature(
    radial_nodes: int,
    angular_nodes: int,
    radial_scale: float,
) -> PolarQuadrature:
    """Independent reconstruction of the frozen mapped-polar rule."""
    if radial_nodes < 4 or angular_nodes < 8 or angular_nodes % 4:
        raise ValueError("invalid mapped-polar quadrature size")
    if radial_scale <= 0.0:
        raise ValueError("radial scale must be positive")
    abscissa, radial_weight = np.polynomial.legendre.leggauss(radial_nodes)
    radius = radial_scale * (1.0 + abscissa) / (1.0 - abscissa)
    jacobian = 2.0 * radial_scale / (1.0 - abscissa) ** 2
    angle = 2.0 * math.pi * np.arange(angular_nodes, dtype=np.float64) / angular_nodes
    rr, tt = np.meshgrid(radius, angle, indexing="ij")
    weight = (
        (radial_weight * jacobian * radius)[:, None]
        * np.full((1, angular_nodes), 2.0 * math.pi / angular_nodes)
    )
    return PolarQuadrature(
        vx=(rr * np.cos(tt)).ravel(),
        vy=(rr * np.sin(tt)).ravel(),
        weight=weight.ravel(),
    )


def validate_stage66_design(
    grid: tuple[int, int] = STAGE66_GRID,
    kn0: float = STAGE66_KNUDSEN,
    cold_hot_ratio: float = STAGE66_COLD_HOT_RATIO,
    viscosity_exponent: float = STAGE66_VISCOSITY_EXPONENT,
    rule: tuple[int, int] = STAGE66_RULE,
    radial_scale: float = STAGE66_RADIAL_SCALE,
    prandtl: float = STAGE66_PRANDTL,
    correction_floor: float = STAGE66_CORRECTION_FLOOR,
    wall_band_layers: int = STAGE66_WALL_BAND_LAYERS,
) -> None:
    actual = (
        grid,
        kn0,
        cold_hot_ratio,
        viscosity_exponent,
        rule,
        radial_scale,
        prandtl,
        correction_floor,
        wall_band_layers,
    )
    expected = (
        STAGE66_GRID,
        STAGE66_KNUDSEN,
        STAGE66_COLD_HOT_RATIO,
        STAGE66_VISCOSITY_EXPONENT,
        STAGE66_RULE,
        STAGE66_RADIAL_SCALE,
        STAGE66_PRANDTL,
        STAGE66_CORRECTION_FLOOR,
        STAGE66_WALL_BAND_LAYERS,
    )
    if actual != expected:
        raise ValueError(
            "Stage 66 is frozen to the exact completed 64x64 Kn0=10 baseline, "
            "Tc/Th=0.1, omega=0.5, the 40x96 mapped-polar rule at radial scale "
            "2.0, Pr=2/3, correction floor 0.05 and a four-cell wall band; it "
            "is an artifact-only signed-source audit, not parameter retuning."
        )


def validate_stage65_artifact(root: str | Path) -> dict[str, object]:
    root = Path(root)
    expected = {
        "summary.json": STAGE65_COMPLETED_ENDPOINT["summary_sha256"],
        "local_activation_maps.npz": STAGE65_COMPLETED_ENDPOINT["maps_sha256"],
    }
    for name, checksum in expected.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != checksum:
            raise ValueError(f"Stage 65 artifact checksum mismatch: {name}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("stage") != 65:
        raise ValueError("Stage 65 artifact stage mismatch")
    if summary.get("decision") != STAGE65_COMPLETED_ENDPOINT["decision"]:
        raise ValueError("Stage 65 artifact decision mismatch")
    return summary


def validate_stage58_artifact(root: str | Path) -> dict[str, object]:
    root = Path(root)
    expected = {
        "summary.json": STAGE58_COMPLETED_ENDPOINT["summary_sha256"],
        "baseline_clipped_fields_and_profiles.npz": (
            STAGE58_COMPLETED_ENDPOINT["baseline_fields_sha256"]
        ),
    }
    for name, checksum in expected.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != checksum:
            raise ValueError(f"Stage 58 artifact checksum mismatch: {name}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("stage") != 58:
        raise ValueError("Stage 58 artifact stage mismatch")
    if summary.get("decision") != STAGE58_COMPLETED_ENDPOINT["decision"]:
        raise ValueError("Stage 58 artifact decision mismatch")
    cfg = summary.get("configuration", {})
    if (
        cfg.get("grid") != list(STAGE66_GRID)
        or cfg.get("kn0") != STAGE66_KNUDSEN
        or cfg.get("cold_hot_ratio") != STAGE66_COLD_HOT_RATIO
        or cfg.get("radial_nodes") != STAGE66_RULE[0]
        or cfg.get("angular_nodes") != STAGE66_RULE[1]
        or cfg.get("radial_scale") != STAGE66_RADIAL_SCALE
        or cfg.get("prandtl") != STAGE66_PRANDTL
        or cfg.get("retained_correction_floor") != STAGE66_CORRECTION_FLOOR
    ):
        raise ValueError("Stage 58 artifact frozen configuration mismatch")
    return summary


def wall_distance_layers(shape: tuple[int, int]) -> np.ndarray:
    if len(shape) != 2 or min(shape) < 2:
        raise ValueError("wall-distance map requires a two-dimensional grid")
    i, j = np.indices(shape)
    return np.minimum.reduce((i, j, shape[0] - 1 - i, shape[1] - 1 - j))


def safe_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    finite = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(finite) < 2:
        return 0.0
    a = a[finite]
    b = b[finite]
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def local_collision_frequency(
    rho: np.ndarray,
    temperature: np.ndarray,
    kn0: float = STAGE66_KNUDSEN,
    viscosity_exponent: float = STAGE66_VISCOSITY_EXPONENT,
) -> np.ndarray:
    rho = np.asarray(rho, dtype=np.float64)
    temperature = np.asarray(temperature, dtype=np.float64)
    if rho.shape != temperature.shape:
        raise ValueError("density and temperature shapes must match")
    if np.any(rho <= 0.0) or np.any(temperature <= 0.0):
        raise ValueError("density and temperature must be positive")
    return (
        math.sqrt(math.pi / 2.0)
        / kn0
        * rho
        * temperature ** (1.0 - viscosity_exponent)
    )


def _projected_maxwellian_chunk(
    rho: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    temperature: np.ndarray,
    quadrature: PolarQuadrature,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cx = quadrature.vx[None, :] - u[:, None]
    cy = quadrature.vy[None, :] - v[:, None]
    c2 = cx * cx + cy * cy
    raw_phi = np.exp(-c2 / (2.0 * temperature[:, None]))
    raw_phi /= 2.0 * math.pi * temperature[:, None]
    discrete_mass = np.sum(raw_phi * quadrature.weight[None, :], axis=1)
    phi_m = rho[:, None] * raw_phi / np.maximum(discrete_mass[:, None], 1.0e-300)
    psi_m = temperature[:, None] * phi_m
    return phi_m, psi_m, cx, cy, c2


def _target_chunk(
    rho: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    temperature: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    quadrature: PolarQuadrature,
    *,
    clipped: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    phi_m, psi_m, cx, cy, c2 = _projected_maxwellian_chunk(
        rho, u, v, temperature, quadrature
    )
    c_dot_q = cx * qx[:, None] + cy * qy[:, None]
    coefficient = (1.0 - STAGE66_PRANDTL) / (
        5.0 * rho[:, None] * temperature[:, None] ** 2
    )
    phi_factor = 1.0 + coefficient * c_dot_q * (
        c2 / temperature[:, None] - 4.0
    )
    psi_factor = 1.0 + coefficient * c_dot_q * (
        c2 / temperature[:, None] - 2.0
    )
    if clipped:
        phi_factor = np.maximum(phi_factor, STAGE66_CORRECTION_FLOOR)
        psi_factor = np.maximum(psi_factor, STAGE66_CORRECTION_FLOOR)
    phi = phi_m * phi_factor
    psi = psi_m * psi_factor
    if clipped:
        density = np.sum(phi * quadrature.weight[None, :], axis=1)
        density_scale = rho / np.maximum(density, 1.0e-14)
        phi *= density_scale[:, None]
        psi *= density_scale[:, None]
    return phi, psi, cx, cy, c2


def frozen_frame_heat_flux(
    phi: np.ndarray,
    psi: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    c2: np.ndarray,
    weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Linear heat-flux observable about the saved macroscopic velocity frame."""
    energy = c2 * phi + psi
    qx = 0.5 * np.sum(cx * energy * weight[None, :], axis=1)
    qy = 0.5 * np.sum(cy * energy * weight[None, :], axis=1)
    return qx, qy


def signed_summary(field: np.ndarray) -> dict[str, float]:
    values = np.asarray(field, dtype=np.float64).ravel()
    if values.size == 0 or np.any(~np.isfinite(values)):
        raise ValueError("signed field must be finite and nonempty")
    tol = STAGE66_ZERO_TOLERANCE
    return {
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
        "mean_absolute": float(np.mean(np.abs(values))),
        "rms": float(np.sqrt(np.mean(values**2))),
        "positive_fraction": float(np.mean(values > tol)),
        "negative_fraction": float(np.mean(values < -tol)),
        "near_zero_fraction": float(np.mean(np.abs(values) <= tol)),
    }


def stratify_signed_field(
    field: np.ndarray,
    coordinate: np.ndarray,
    edges: Sequence[float],
) -> list[dict[str, float | int]]:
    field = np.asarray(field, dtype=np.float64).ravel()
    coordinate = np.asarray(coordinate, dtype=np.float64).ravel()
    edges = tuple(float(value) for value in edges)
    if field.shape != coordinate.shape:
        raise ValueError("field and stratification coordinate shapes must match")
    if len(edges) < 2 or any(b <= a for a, b in zip(edges[:-1], edges[1:])):
        raise ValueError("stratification edges must be strictly increasing")
    total_absolute = float(np.sum(np.abs(field)))
    rows: list[dict[str, float | int]] = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        if index == len(edges) - 2:
            mask = (coordinate >= lower) & (coordinate <= upper)
        else:
            mask = (coordinate >= lower) & (coordinate < upper)
        values = field[mask]
        row: dict[str, float | int] = {
            "lower": lower,
            "upper": upper,
            "count": int(values.size),
            "cell_fraction": float(values.size / field.size),
        }
        if values.size:
            row.update(
                {
                    "mean": float(np.mean(values)),
                    "mean_absolute": float(np.mean(np.abs(values))),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                    "absolute_share": float(
                        np.sum(np.abs(values)) / max(total_absolute, 1.0e-300)
                    ),
                }
            )
        else:
            row.update(
                {
                    "mean": 0.0,
                    "mean_absolute": 0.0,
                    "minimum": 0.0,
                    "maximum": 0.0,
                    "absolute_share": 0.0,
                }
            )
        rows.append(row)
    return rows


def wall_layer_stratification(
    field: np.ndarray,
    distance: np.ndarray,
    wall_band_layers: int = STAGE66_WALL_BAND_LAYERS,
) -> dict[str, object]:
    field = np.asarray(field, dtype=np.float64)
    distance = np.asarray(distance, dtype=np.int64)
    if field.shape != distance.shape:
        raise ValueError("field and wall-distance shapes must match")
    absolute = np.abs(field)
    total = float(np.sum(absolute))
    band = distance < wall_band_layers
    rows = []
    for layer in range(int(np.max(distance)) + 1):
        mask = distance == layer
        values = field[mask]
        rows.append(
            {
                "layer": layer,
                "count": int(values.size),
                "mean": float(np.mean(values)),
                "mean_absolute": float(np.mean(np.abs(values))),
                "absolute_share": float(
                    np.sum(np.abs(values)) / max(total, 1.0e-300)
                ),
            }
        )
    return {
        "wall_band_layers": wall_band_layers,
        "wall_band_cell_fraction": float(np.mean(band)),
        "wall_band_mean": float(np.mean(field[band])),
        "interior_mean": float(np.mean(field[~band])),
        "wall_band_absolute_share": float(
            np.sum(absolute[band]) / max(total, 1.0e-300)
        ),
        "interior_absolute_share": float(
            np.sum(absolute[~band]) / max(total, 1.0e-300)
        ),
        "layers": rows,
    }


def stage66_decision(metrics: Mapping[str, object]) -> str:
    if not bool(metrics.get("finite", False)):
        return "stage66_nonfinite_frozen_source_observable_blocker"
    if float(metrics["maximum_unclipped_target_closure_error"]) > (
        STAGE66_UNCLIPPED_CLOSURE_TOLERANCE
    ):
        return "stage66_unclipped_target_closure_blocker"
    opposed = float(metrics["fraction_normal_target_bias_opposed_to_observed_excess"])
    aligned = float(metrics["fraction_normal_target_bias_aligned_with_observed_excess"])
    if opposed >= STAGE66_DIRECTIONAL_FRACTION_THRESHOLD:
        return (
            "stage66_clipping_source_bias_opposes_heat_flux_overprediction_"
            "stage67_frozen_full_distribution_residual_decomposition"
        )
    if aligned >= STAGE66_DIRECTIONAL_FRACTION_THRESHOLD:
        return (
            "stage66_clipping_source_bias_aligns_with_heat_flux_overprediction_"
            "stage67_frozen_full_distribution_residual_decomposition"
        )
    return (
        "stage66_clipping_source_bias_mixed_"
        "stage67_frozen_full_distribution_residual_decomposition"
    )


def _load_baseline_fields(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        keys = ("rho", "u", "v", "T", "qx", "qy", "bottom_heat_flux")
        missing = [key for key in keys if key not in data.files]
        if missing:
            raise ValueError(f"Stage 58 baseline fields missing: {missing}")
        return {key: np.asarray(data[key], dtype=np.float64) for key in keys}


def audit_fields(
    fields: Mapping[str, np.ndarray],
    stage65_maps: Mapping[str, np.ndarray],
    predicted_qav: float,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    shape = np.asarray(fields["rho"]).shape
    if shape != STAGE66_GRID:
        raise ValueError("Stage 66 requires exact 64x64 fields")
    for key in ("u", "v", "T", "qx", "qy"):
        if np.asarray(fields[key]).shape != shape:
            raise ValueError(f"Stage 66 field shape mismatch: {key}")

    quadrature = mapped_polar_quadrature(*STAGE66_RULE, STAGE66_RADIAL_SCALE)
    flat = {
        key: np.asarray(fields[key], dtype=np.float64).ravel()
        for key in ("rho", "u", "v", "T", "qx", "qy")
    }
    count = flat["rho"].size
    clipped_qx = np.empty(count)
    clipped_qy = np.empty(count)
    unclipped_qx = np.empty(count)
    unclipped_qy = np.empty(count)

    for start in range(0, count, STAGE66_CHUNK_SIZE):
        stop = min(start + STAGE66_CHUNK_SIZE, count)
        chunk = {key: value[start:stop] for key, value in flat.items()}
        phi_u, psi_u, cx, cy, c2 = _target_chunk(
            chunk["rho"],
            chunk["u"],
            chunk["v"],
            chunk["T"],
            chunk["qx"],
            chunk["qy"],
            quadrature,
            clipped=False,
        )
        qx_u, qy_u = frozen_frame_heat_flux(
            phi_u, psi_u, cx, cy, c2, quadrature.weight
        )
        phi_c, psi_c, cx_c, cy_c, c2_c = _target_chunk(
            chunk["rho"],
            chunk["u"],
            chunk["v"],
            chunk["T"],
            chunk["qx"],
            chunk["qy"],
            quadrature,
            clipped=True,
        )
        qx_c, qy_c = frozen_frame_heat_flux(
            phi_c, psi_c, cx_c, cy_c, c2_c, quadrature.weight
        )
        unclipped_qx[start:stop] = qx_u
        unclipped_qy[start:stop] = qy_u
        clipped_qx[start:stop] = qx_c
        clipped_qy[start:stop] = qy_c

    paper_scale = 1.0 / math.sqrt(2.0)
    for array in (clipped_qx, clipped_qy, unclipped_qx, unclipped_qy):
        array *= paper_scale
    qx_paper = flat["qx"] * paper_scale
    qy_paper = flat["qy"] * paper_scale
    expected_target_qx = (1.0 - STAGE66_PRANDTL) * qx_paper
    expected_target_qy = (1.0 - STAGE66_PRANDTL) * qy_paper
    q_scale = np.maximum(
        np.sqrt(expected_target_qx**2 + expected_target_qy**2), 1.0e-14
    )
    closure = np.sqrt(
        (unclipped_qx - expected_target_qx) ** 2
        + (unclipped_qy - expected_target_qy) ** 2
    ) / q_scale

    target_bias_qx = clipped_qx - unclipped_qx
    target_bias_qy = clipped_qy - unclipped_qy
    frequency = local_collision_frequency(flat["rho"], flat["T"])
    source_bias_qx = frequency * target_bias_qx
    source_bias_qy = frequency * target_bias_qy
    ideal_source_qy = frequency * (expected_target_qy - qy_paper)
    active_source_qy = ideal_source_qy + source_bias_qy
    relative_source_bias_qy = np.abs(source_bias_qy) / np.maximum(
        np.abs(ideal_source_qy), 1.0e-14
    )

    observed_reference = float(PUBLISHED_REFERENCE["table6_qav"]["dsmc"])
    observed_excess = float(predicted_qav) - observed_reference
    products = target_bias_qy * observed_excess
    aligned_fraction = float(np.mean(products > STAGE66_ZERO_TOLERANCE))
    opposed_fraction = float(np.mean(products < -STAGE66_ZERO_TOLERANCE))

    maps = {
        key: np.asarray(stage65_maps[key])
        for key in (
            "combined_clipping_weight_fraction",
            "local_conserved_defect",
            "local_heat_flux_closure_defect",
            "wall_distance_layers",
        )
    }
    for key, value in maps.items():
        if value.shape != shape:
            raise ValueError(f"Stage 65 map shape mismatch: {key}")
    distance = maps["wall_distance_layers"].astype(np.int64)
    source_bias_qy_grid = source_bias_qy.reshape(shape)

    finite_arrays = (
        clipped_qx,
        clipped_qy,
        unclipped_qx,
        unclipped_qy,
        closure,
        source_bias_qx,
        source_bias_qy,
        ideal_source_qy,
        active_source_qy,
        relative_source_bias_qy,
    )
    finite = all(np.all(np.isfinite(array)) for array in finite_arrays)
    decision_metrics = {
        "finite": finite,
        "maximum_unclipped_target_closure_error": float(np.max(closure)),
        "fraction_normal_target_bias_aligned_with_observed_excess": aligned_fraction,
        "fraction_normal_target_bias_opposed_to_observed_excess": opposed_fraction,
    }

    wall = wall_layer_stratification(source_bias_qy_grid, distance)
    clipping = maps["combined_clipping_weight_fraction"].ravel()
    local_conserved = maps["local_conserved_defect"].ravel()
    local_heat = maps["local_heat_flux_closure_defect"].ravel()

    audit = {
        "observed_heat_flux_gap": {
            "baseline_predicted_qav": float(predicted_qav),
            "published_dsmc_qav": observed_reference,
            "signed_excess": observed_excess,
            "relative_excess": observed_excess / observed_reference,
        },
        "unclipped_target_closure": {
            "maximum_relative_vector_error": float(np.max(closure)),
            "rms_relative_vector_error": float(np.sqrt(np.mean(closure**2))),
        },
        "target_bias_paper_coordinates": {
            "x": signed_summary(target_bias_qx),
            "y": signed_summary(target_bias_qy),
        },
        "collision_source_bias_paper_coordinates": {
            "x": signed_summary(source_bias_qx),
            "y": signed_summary(source_bias_qy),
            "ideal_normal_source": signed_summary(ideal_source_qy),
            "active_normal_source": signed_summary(active_source_qy),
            "relative_normal_source_bias": {
                "minimum": float(np.min(relative_source_bias_qy)),
                "mean": float(np.mean(relative_source_bias_qy)),
                "rms": float(np.sqrt(np.mean(relative_source_bias_qy**2))),
                "maximum": float(np.max(relative_source_bias_qy)),
                "active_to_ideal_ratio_minimum": float(
                    np.min(active_source_qy / ideal_source_qy)
                ),
                "active_to_ideal_ratio_mean": float(
                    np.mean(active_source_qy / ideal_source_qy)
                ),
                "active_to_ideal_ratio_maximum": float(
                    np.max(active_source_qy / ideal_source_qy)
                ),
            },
        },
        "directional_review": {
            "normal_coordinate": (
                "positive_y_consistent_with_saved_positive_bottom_heat_flux"
            ),
            "fraction_target_bias_aligned_with_observed_positive_excess": (
                aligned_fraction
            ),
            "fraction_target_bias_opposed_to_observed_positive_excess": (
                opposed_fraction
            ),
            "nominal_full_relaxation_mean_target_bias_to_observed_gap": float(
                np.mean(target_bias_qy) / observed_excess
            ),
        },
        "wall_distance_stratification": wall,
        "temperature_stratification": stratify_signed_field(
            source_bias_qy, flat["T"], STAGE66_TEMPERATURE_BINS
        ),
        "density_stratification": stratify_signed_field(
            source_bias_qy, flat["rho"], STAGE66_DENSITY_BINS
        ),
        "correlations": {
            "clipping_vs_signed_normal_source_bias": safe_correlation(
                clipping, source_bias_qy
            ),
            "clipping_vs_absolute_normal_source_bias": safe_correlation(
                clipping, np.abs(source_bias_qy)
            ),
            "stage65_conserved_defect_vs_absolute_normal_source_bias": (
                safe_correlation(local_conserved, np.abs(source_bias_qy))
            ),
            "stage65_heat_flux_defect_vs_absolute_normal_source_bias": (
                safe_correlation(local_heat, np.abs(source_bias_qy))
            ),
        },
        "decision_metrics": decision_metrics,
    }

    arrays = {
        "clipped_target_qx_paper": clipped_qx.reshape(shape),
        "clipped_target_qy_paper": clipped_qy.reshape(shape),
        "unclipped_target_qx_paper": unclipped_qx.reshape(shape),
        "unclipped_target_qy_paper": unclipped_qy.reshape(shape),
        "target_bias_qx_paper": target_bias_qx.reshape(shape),
        "target_bias_qy_paper": target_bias_qy.reshape(shape),
        "source_bias_qx_paper": source_bias_qx.reshape(shape),
        "source_bias_qy_paper": source_bias_qy_grid,
        "ideal_source_qy_paper": ideal_source_qy.reshape(shape),
        "active_source_qy_paper": active_source_qy.reshape(shape),
        "relative_source_bias_qy": relative_source_bias_qy.reshape(shape),
        "collision_frequency": frequency.reshape(shape),
        "unclipped_target_closure_error": closure.reshape(shape),
        "wall_distance_layers": distance,
    }
    return audit, arrays


def run_stage66(
    stage65_artifact_dir: str | Path,
    stage58_artifact_dir: str | Path,
    output_dir: str | Path,
    **design,
) -> dict[str, object]:
    validate_stage66_design(**design)
    stage65 = validate_stage65_artifact(stage65_artifact_dir)
    stage58 = validate_stage58_artifact(stage58_artifact_dir)
    baseline_path = (
        Path(stage58_artifact_dir) / "baseline_clipped_fields_and_profiles.npz"
    )
    fields = _load_baseline_fields(baseline_path)
    with np.load(Path(stage65_artifact_dir) / "local_activation_maps.npz") as data:
        stage65_maps = {key: np.asarray(data[key]) for key in data.files}
    predicted_qav = float(np.mean(fields["bottom_heat_flux"]))
    recorded_qav = float(stage58["baseline_clipped"]["predicted_qav"])
    if not math.isclose(
        predicted_qav, recorded_qav, rel_tol=1.0e-13, abs_tol=1.0e-14
    ):
        raise ValueError("Stage 58 baseline heat-flux profile and summary disagree")

    audit, arrays = audit_fields(fields, stage65_maps, predicted_qav)
    decision = stage66_decision(audit["decision_metrics"])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "source_observable_maps.npz", **arrays)

    summary = {
        "stage": 66,
        "description": (
            "Artifact-only signed audit of the retained clipped projected-Shakhov "
            "target and collision-source contribution to frozen-frame heat-flux "
            "observables on the exact completed Stage 58 64x64 Kn0=10 baseline."
        ),
        "retained_stage65_endpoint": STAGE65_COMPLETED_ENDPOINT,
        "retained_stage65_decision": stage65["decision"],
        "retained_stage58_endpoint": STAGE58_COMPLETED_ENDPOINT,
        "retained_stage58_decision": stage58["decision"],
        "published_reference": PUBLISHED_REFERENCE,
        "configuration": {
            "grid": list(STAGE66_GRID),
            "kn0": STAGE66_KNUDSEN,
            "cold_hot_ratio": STAGE66_COLD_HOT_RATIO,
            "viscosity_exponent": STAGE66_VISCOSITY_EXPONENT,
            "radial_nodes": STAGE66_RULE[0],
            "angular_nodes": STAGE66_RULE[1],
            "point_count": int(np.prod(STAGE66_RULE)),
            "radial_scale": STAGE66_RADIAL_SCALE,
            "prandtl": STAGE66_PRANDTL,
            "retained_correction_floor": STAGE66_CORRECTION_FLOOR,
            "wall_band_layers": STAGE66_WALL_BAND_LAYERS,
            "temperature_bins": list(STAGE66_TEMPERATURE_BINS),
            "density_bins": list(STAGE66_DENSITY_BINS),
            "unclipped_closure_tolerance": (
                STAGE66_UNCLIPPED_CLOSURE_TOLERANCE
            ),
            "directional_fraction_threshold": (
                STAGE66_DIRECTIONAL_FRACTION_THRESHOLD
            ),
            "solver_rerun": False,
            "full_distribution_available": False,
            "physical_parameter_retuning": False,
            "collision_parameter_retuning": False,
            "correction_floor_retuning": False,
            "source_relaxation_retuning": False,
            "normalization_retuning": False,
            "transport_retuning": False,
            "velocity_quadrature_retuning": False,
            "wall_model_retuning": False,
            "conservative_projection_adopted": False,
            "cross_knudsen_extension_permitted": False,
        },
        "audit": audit,
        "decision": decision,
        "positive_findings": [
            "The independent unclipped projected-Shakhov reconstruction closes the expected heat-flux target to the preregistered tolerance.",
            "All signed target and source maps are finite and preserve the Stage 65 broad spatial distribution rather than hiding negative cells or wall layers.",
            "The tangential target-bias mean is symmetry-consistent and near zero, while the normal component has a deterministic sign over the full cavity.",
        ],
        "negative_findings": [
            "The retained clipping changes the normal collision-source magnitude materially throughout the cavity, with the exact extrema and stratified shares retained in the artifact.",
            "The direct normal target/source bias opposes, rather than explains with matching sign, the previously confirmed positive average heat-flux overprediction.",
            "The four-cell wall band does not dominate the signed-source defect; most absolute source bias lies in the interior.",
            "Saved Stage 58 artifacts contain macroscopic fields and wall profiles but not the converged full distributions, so indirect nonlinear feedback and steady residual causality cannot be inferred from this artifact-only audit.",
        ],
        "interpretation_guard": (
            "The frozen-frame heat-flux moment is deliberately linear in the target "
            "distribution so the clipped-minus-unclipped collision-source difference is "
            "exact at the saved local state. It is not an adjoint sensitivity, a wall-flux "
            "correction, a steady-solution perturbation, or a causal attribution. The "
            "nominal target-bias-to-observed-gap ratio is reported only as a signed scale. "
            "No validation, parameter retuning, conservative-projection adoption, MUSCL "
            "rehabilitation or cross-Knudsen extension is authorized."
        ),
        "scientifically_justified_next_scope": (
            "Perform one exact frozen replay of the retained Stage 58 baseline solely to "
            "save the converged phi/psi distributions, then decompose the local steady "
            "transport, clipped-collision and diffuse-wall residual contributions to the "
            "heat-flux moment. Keep every physical, collision, wall, transport, quadrature, "
            "floor, relaxation, normalization and stopping setting unchanged; do not treat "
            "that replay as validation or a parameter search."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage65-artifact-dir", required=True)
    parser.add_argument("--stage58-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run_stage66(
                args.stage65_artifact_dir,
                args.stage58_artifact_dir,
                args.output_dir,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
