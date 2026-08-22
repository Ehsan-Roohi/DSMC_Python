from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import math
from typing import Mapping

import numpy as np

from . import stage68_independent_transport_operator_residual_audit as stage68


STAGE68_COMPLETED_ENDPOINT = {
    "workflow_run_id": 31010112390,
    "workflow_job_id": 92319799599,
    "workflow_conclusion": "success",
    "tests_passed": 87,
    "tests_failed": 0,
    "test_duration_seconds": 0.74,
    "artifact_id": 8938442470,
    "artifact_size_bytes": 446487,
    "artifact_sha256": "d38fcf90e998341a8ed1ad443ac3de8a1597a91a89c454e8430491ff75751ea6",
    "source_head_sha": "1703bbb0d15d9d34fe972f565ec7da9a07693489",
    "summary_sha256": "67e8ab90c900d57f39ac1fe5f0835ca0346684da26964fb786bfb3f1c10bd06d",
    "moment_maps_sha256": "0e8ba77f6c6cd1ca98bc7d22c0e4b41d81e7ba99c0fcca922809804f79d3ad7d",
    "decision": (
        "stage68_material_higher_order_transport_residual_without_observable_"
        "causality_stage69_frozen_grid_transfer_residual_scaling_audit"
    ),
}

STAGE69_GRIDS = (16, 32, 64)
STAGE69_FINE_GRID = 64
STAGE69_KNUDSEN = 10.0
STAGE69_COLD_HOT_RATIO = 0.1
STAGE69_RULE = (40, 96)
STAGE69_RADIAL_SCALE = 2.0
STAGE69_POINT_COUNT = 3840
STAGE69_CHUNK_SIZE = 128
STAGE69_LIMITER = "minmod"
STAGE69_RESTRICTION = "conservative_cell_average_from_exact_stage67_64x64_distribution"
STAGE69_WALL_BAND_PHYSICAL_FRACTION = 1.0 / 16.0
STAGE69_MATERIAL_HEAT_FLUX_RATIO = 0.10
STAGE69_RESTRICTION_GUARD = 1.0e-13
STAGE69_ENDPOINT_GUARD = 1.0e-10
STAGE69_MONOTONIC_RELATIVE_TOLERANCE = 1.0e-12


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_stage69_design(
    grids: tuple[int, ...] = STAGE69_GRIDS,
    fine_grid: int = STAGE69_FINE_GRID,
    kn0: float = STAGE69_KNUDSEN,
    cold_hot_ratio: float = STAGE69_COLD_HOT_RATIO,
    rule: tuple[int, int] = STAGE69_RULE,
    radial_scale: float = STAGE69_RADIAL_SCALE,
    chunk_size: int = STAGE69_CHUNK_SIZE,
    limiter: str = STAGE69_LIMITER,
    restriction: str = STAGE69_RESTRICTION,
    wall_band_physical_fraction: float = STAGE69_WALL_BAND_PHYSICAL_FRACTION,
    material_heat_flux_ratio: float = STAGE69_MATERIAL_HEAT_FLUX_RATIO,
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
        material_heat_flux_ratio,
    )
    expected = (
        STAGE69_GRIDS,
        STAGE69_FINE_GRID,
        STAGE69_KNUDSEN,
        STAGE69_COLD_HOT_RATIO,
        STAGE69_RULE,
        STAGE69_RADIAL_SCALE,
        STAGE69_CHUNK_SIZE,
        STAGE69_LIMITER,
        STAGE69_RESTRICTION,
        STAGE69_WALL_BAND_PHYSICAL_FRACTION,
        STAGE69_MATERIAL_HEAT_FLUX_RATIO,
    )
    if actual != expected:
        raise ValueError(
            "Stage 69 is frozen to conservative 64->32->16 cell-average restriction "
            "of the exact Stage-67 64x64 distributions, the completed Stage-68 "
            "operators, Kn0=10, the 40x96 radial-scale-2.0 quadrature, minmod, "
            "the physical 1/16 wall band, and the inherited 10% materiality guard; "
            "no retuning is permitted."
        )


def _validate_stage67_artifact(root: str | Path) -> dict[str, object]:
    return stage68._validate_stage67_artifact(root)


def _validate_stage68_artifact(root: str | Path) -> dict[str, object]:
    root = Path(root)
    expected = {
        "summary.json": STAGE68_COMPLETED_ENDPOINT["summary_sha256"],
        "transport_operator_moment_maps.npz":
            STAGE68_COMPLETED_ENDPOINT["moment_maps_sha256"],
    }
    for name, checksum in expected.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != checksum:
            raise ValueError(f"Stage 68 artifact checksum mismatch: {name}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if (
        summary.get("stage") != 68
        or summary.get("decision") != STAGE68_COMPLETED_ENDPOINT["decision"]
    ):
        raise ValueError("Stage 68 artifact endpoint mismatch")
    return summary


def restrict_cell_average(distribution: np.ndarray, target_grid: int) -> np.ndarray:
    distribution = np.asarray(distribution, dtype=np.float64)
    if distribution.ndim != 3 or distribution.shape[0] != distribution.shape[1]:
        raise ValueError("Restriction requires a square ny x nx x nq distribution")
    source_grid = distribution.shape[0]
    if target_grid <= 0 or source_grid % target_grid != 0:
        raise ValueError("Target grid must exactly divide the source grid")
    factor = source_grid // target_grid
    return distribution.reshape(
        target_grid, factor, target_grid, factor, distribution.shape[2]
    ).mean(axis=(1, 3))


def restriction_conservation(
    fine: np.ndarray,
    restricted: np.ndarray,
) -> dict[str, float | bool]:
    fine_mean = np.mean(np.asarray(fine, dtype=np.float64), axis=(0, 1))
    restricted_mean = np.mean(
        np.asarray(restricted, dtype=np.float64), axis=(0, 1)
    )
    delta = restricted_mean - fine_mean
    maximum_absolute_error = float(np.max(np.abs(delta)))
    relative_l2_error = float(
        np.linalg.norm(delta)
        / max(float(np.linalg.norm(fine_mean)), 1.0e-300)
    )
    return {
        "maximum_absolute_error": maximum_absolute_error,
        "relative_l2_error": relative_l2_error,
        "within_guard": bool(relative_l2_error <= STAGE69_RESTRICTION_GUARD),
    }


def physical_interior_mask(
    grid: int,
    wall_band_physical_fraction: float = STAGE69_WALL_BAND_PHYSICAL_FRACTION,
) -> tuple[np.ndarray, int]:
    layers = int(round(grid * wall_band_physical_fraction))
    if layers < 1 or 2 * layers >= grid:
        raise ValueError("Physical wall band leaves no interior cells")
    mask = np.ones((grid, grid), dtype=bool)
    mask[:layers, :] = False
    mask[-layers:, :] = False
    mask[:, :layers] = False
    mask[:, -layers:] = False
    return mask, layers


def rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(values * values)))


def observed_order(coarse_error: float, fine_error: float) -> float:
    if coarse_error <= 0.0 or fine_error <= 0.0:
        return math.nan
    return float(math.log(coarse_error / fine_error) / math.log(2.0))


def monotonically_decreases_with_refinement(
    errors: list[float] | tuple[float, ...],
    relative_tolerance: float = STAGE69_MONOTONIC_RELATIVE_TOLERANCE,
) -> bool:
    return all(
        fine < coarse * (1.0 + relative_tolerance)
        and not math.isclose(fine, coarse, rel_tol=relative_tolerance, abs_tol=0.0)
        for coarse, fine in zip(errors[:-1], errors[1:], strict=True)
    )


def compare_endpoint_maps(
    calculated: Mapping[str, Mapping[str, np.ndarray]],
    stage68_maps_path: str | Path,
) -> dict[str, object]:
    per_map: dict[str, dict[str, float]] = {}
    with np.load(stage68_maps_path) as expected:
        required = {
            f"{component}_{moment}"
            for component, moments in calculated.items()
            for moment in moments
        }
        if set(expected.files) != required:
            raise ValueError("Stage 68 moment-map contract mismatch")
        for component, moments in calculated.items():
            for moment, actual in moments.items():
                name = f"{component}_{moment}"
                reference = np.asarray(expected[name], dtype=np.float64)
                delta = np.asarray(actual, dtype=np.float64) - reference
                per_map[name] = {
                    "maximum_absolute_error": float(np.max(np.abs(delta))),
                    "relative_l2_error": float(
                        np.linalg.norm(delta.ravel())
                        / max(float(np.linalg.norm(reference.ravel())), 1.0e-300)
                    ),
                }
    maximum_absolute_error = max(
        row["maximum_absolute_error"] for row in per_map.values()
    )
    maximum_relative_l2_error = max(
        row["relative_l2_error"] for row in per_map.values()
    )
    return {
        "per_map": per_map,
        "maximum_absolute_error": maximum_absolute_error,
        "maximum_relative_l2_error": maximum_relative_l2_error,
        "within_guard": bool(maximum_relative_l2_error <= STAGE69_ENDPOINT_GUARD),
    }


def stage69_decision(
    finite: bool,
    provenance_consistent: bool,
    restriction_conservative: bool,
    full_monotonic: bool,
    interior_monotonic: bool,
    fine_wall_absolute_share: float,
    fine_pair_full_order: float,
    fine_pair_interior_order: float,
    fine_normal_heat_flux_ratio: float,
) -> str:
    if not finite:
        return "stage69_nonfinite_grid_transfer_blocker"
    if not provenance_consistent:
        return "stage69_completed_endpoint_reproduction_blocker"
    if not restriction_conservative:
        return "stage69_conservative_restriction_blocker"
    if not full_monotonic or not interior_monotonic:
        return "stage69_nonmonotone_frozen_grid_transfer_blocker"
    if (
        fine_wall_absolute_share > 0.5
        and fine_pair_full_order < fine_pair_interior_order
    ):
        return (
            "stage69_monotone_but_wall_dominated_slow_full_scaling_"
            "stage70_independent_wall_face_flux_discretization_audit"
        )
    if fine_normal_heat_flux_ratio >= STAGE69_MATERIAL_HEAT_FLUX_RATIO:
        return (
            "stage69_monotone_material_nonwall_residual_"
            "stage70_frozen_linearized_response_audit"
        )
    return (
        "stage69_transport_difference_scales_below_materiality_"
        "stage70_independent_wall_face_flux_discretization_audit"
    )


def run_stage69(
    stage67_artifact_dir: str | Path,
    stage68_artifact_dir: str | Path,
    output_dir: str | Path,
    **design: object,
) -> dict[str, object]:
    validate_stage69_design(**design)
    stage67_artifact_dir = Path(stage67_artifact_dir)
    stage68_artifact_dir = Path(stage68_artifact_dir)
    retained67 = _validate_stage67_artifact(stage67_artifact_dir)
    retained68 = _validate_stage68_artifact(stage68_artifact_dir)
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

    expected_shape = (STAGE69_FINE_GRID, STAGE69_FINE_GRID, STAGE69_POINT_COUNT)
    if fine_phi.shape != expected_shape or fine_psi.shape != expected_shape:
        raise ValueError("Stage 69 requires exact 64x64x3840 Stage-67 distributions")

    grid_results: dict[str, dict[str, object]] = {}
    restriction_checks: dict[str, dict[str, object]] = {}
    saved_maps: dict[str, np.ndarray] = {}
    endpoint_consistency: dict[str, object] | None = None

    for grid in STAGE69_GRIDS:
        phi = (
            fine_phi if grid == STAGE69_FINE_GRID
            else restrict_cell_average(fine_phi, grid)
        )
        psi = (
            fine_psi if grid == STAGE69_FINE_GRID
            else restrict_cell_average(fine_psi, grid)
        )
        phi_check = restriction_conservation(fine_phi, phi)
        psi_check = restriction_conservation(fine_psi, psi)
        restriction_checks[str(grid)] = {
            "factor": STAGE69_FINE_GRID // grid,
            "phi": phi_check,
            "psi": psi_check,
            "finite": bool(np.all(np.isfinite(phi)) and np.all(np.isfinite(psi))),
            "minimum_phi": float(np.min(phi)),
            "minimum_psi": float(np.min(psi)),
        }

        incoming = stage68.reconstruct_wall_incoming(phi, psi, vx, vy, weight)
        maps = stage68.evaluate_transport_operators(
            phi, psi, vx, vy, weight, incoming, STAGE69_CHUNK_SIZE
        )
        for component, moments in maps.items():
            for moment, values in moments.items():
                saved_maps[f"grid{grid}_{component}_{moment}"] = values

        if grid == STAGE69_FINE_GRID:
            endpoint_consistency = compare_endpoint_maps(
                maps,
                stage68_artifact_dir / "transport_operator_moment_maps.npz",
            )

        summaries = {
            component: {
                moment: stage68.signed_summary(values)
                for moment, values in moments.items()
            }
            for component, moments in maps.items()
        }
        ratios = {
            moment: (
                summaries["difference"][moment]["rms"]
                / max(
                    summaries["retained_first_order"][moment]["rms"],
                    1.0e-300,
                )
            )
            for moment in maps["difference"]
        }
        interior_mask, wall_layers = physical_interior_mask(grid)
        difference_qy = maps["difference"]["qy"]
        retained_qy = maps["retained_first_order"]["qy"]
        grid_results[str(grid)] = {
            "grid": [grid, grid],
            "cell_width": 1.0 / grid,
            "wall_band_layers": wall_layers,
            "wall_band_physical_fraction": STAGE69_WALL_BAND_PHYSICAL_FRACTION,
            "operator_summaries": summaries,
            "operator_difference_rms_ratios": ratios,
            "normal_heat_flux": {
                "full_difference_rms": rms(difference_qy),
                "interior_difference_rms": rms(difference_qy[interior_mask]),
                "full_retained_first_order_rms": rms(retained_qy),
                "interior_retained_first_order_rms": rms(
                    retained_qy[interior_mask]
                ),
                "full_difference_ratio": float(ratios["qy"]),
                "interior_difference_ratio": float(
                    rms(difference_qy[interior_mask])
                    / max(rms(retained_qy[interior_mask]), 1.0e-300)
                ),
                "difference_signed_mean": float(np.mean(difference_qy)),
                "wall_band_absolute_share": stage68.wall_band_absolute_share(
                    difference_qy, wall_layers
                ),
            },
        }

    assert endpoint_consistency is not None
    full_errors = [
        float(grid_results[str(grid)]["normal_heat_flux"]["full_difference_rms"])
        for grid in STAGE69_GRIDS
    ]
    interior_errors = [
        float(
            grid_results[str(grid)]["normal_heat_flux"]["interior_difference_rms"]
        )
        for grid in STAGE69_GRIDS
    ]
    full_orders = {
        "16_to_32": observed_order(full_errors[0], full_errors[1]),
        "32_to_64": observed_order(full_errors[1], full_errors[2]),
    }
    interior_orders = {
        "16_to_32": observed_order(interior_errors[0], interior_errors[1]),
        "32_to_64": observed_order(interior_errors[1], interior_errors[2]),
    }
    full_monotonic = monotonically_decreases_with_refinement(full_errors)
    interior_monotonic = monotonically_decreases_with_refinement(interior_errors)
    restriction_conservative = bool(
        all(
            row[variable]["within_guard"]
            for row in restriction_checks.values()
            for variable in ("phi", "psi")
        )
    )
    finite = bool(
        all(row["finite"] for row in restriction_checks.values())
        and all(
            np.all(np.isfinite(values)) for values in saved_maps.values()
        )
        and all(math.isfinite(value) for value in full_errors + interior_errors)
        and all(math.isfinite(value) for value in full_orders.values())
        and all(math.isfinite(value) for value in interior_orders.values())
    )
    fine_result = grid_results[str(STAGE69_FINE_GRID)]["normal_heat_flux"]
    stage68_ratio_consistent = math.isclose(
        float(fine_result["full_difference_ratio"]),
        float(retained68["normal_heat_flux_operator_difference_ratio"]),
        rel_tol=STAGE69_ENDPOINT_GUARD,
        abs_tol=STAGE69_ENDPOINT_GUARD,
    )
    stage68_wall_share_consistent = math.isclose(
        float(fine_result["wall_band_absolute_share"]),
        float(retained68["difference_qy_wall_band_absolute_share"]),
        rel_tol=STAGE69_ENDPOINT_GUARD,
        abs_tol=STAGE69_ENDPOINT_GUARD,
    )
    provenance_consistent = bool(
        endpoint_consistency["within_guard"]
        and stage68_ratio_consistent
        and stage68_wall_share_consistent
    )
    decision = stage69_decision(
        finite,
        provenance_consistent,
        restriction_conservative,
        full_monotonic,
        interior_monotonic,
        float(fine_result["wall_band_absolute_share"]),
        float(full_orders["32_to_64"]),
        float(interior_orders["32_to_64"]),
        float(fine_result["full_difference_ratio"]),
    )

    np.savez_compressed(out / "grid_transfer_transport_moment_maps.npz", **saved_maps)
    summary = {
        "stage": 69,
        "description": (
            "Frozen conservative grid-transfer audit of the Stage-68 first-order "
            "and independent second-order transport-operator difference on exact "
            "64x64 Stage-67 distributions restricted by cell averaging to 32x32 "
            "and 16x16; no cavity solve is performed."
        ),
        "configuration": {
            "grids": list(STAGE69_GRIDS),
            "fine_grid": STAGE69_FINE_GRID,
            "kn0": STAGE69_KNUDSEN,
            "cold_hot_ratio": STAGE69_COLD_HOT_RATIO,
            "radial_nodes": STAGE69_RULE[0],
            "angular_nodes": STAGE69_RULE[1],
            "point_count": STAGE69_POINT_COUNT,
            "radial_scale": STAGE69_RADIAL_SCALE,
            "chunk_size": STAGE69_CHUNK_SIZE,
            "limiter": STAGE69_LIMITER,
            "restriction": STAGE69_RESTRICTION,
            "wall_band_physical_fraction": STAGE69_WALL_BAND_PHYSICAL_FRACTION,
            "material_heat_flux_ratio": STAGE69_MATERIAL_HEAT_FLUX_RATIO,
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
        "retained_stage67_endpoint": stage68.STAGE67_COMPLETED_ENDPOINT,
        "retained_stage67_decision": retained67["decision"],
        "retained_stage68_endpoint": STAGE68_COMPLETED_ENDPOINT,
        "retained_stage68_decision": retained68["decision"],
        "restriction_checks": restriction_checks,
        "endpoint_consistency": {
            **endpoint_consistency,
            "stage68_ratio_consistent": stage68_ratio_consistent,
            "stage68_wall_share_consistent": stage68_wall_share_consistent,
        },
        "grid_results": grid_results,
        "normal_heat_flux_scaling": {
            "full_difference_rms_sequence": full_errors,
            "interior_difference_rms_sequence": interior_errors,
            "full_observed_orders": full_orders,
            "interior_observed_orders": interior_orders,
            "full_monotonic_decrease": full_monotonic,
            "interior_monotonic_decrease": interior_monotonic,
            "fine_grid_wall_band_absolute_share": float(
                fine_result["wall_band_absolute_share"]
            ),
            "fine_grid_difference_ratio": float(
                fine_result["full_difference_ratio"]
            ),
        },
        "finite": finite,
        "restriction_conservative": restriction_conservative,
        "provenance_consistent": provenance_consistent,
        "decision": decision,
        "positive_findings": [
            "Cell-average restriction preserves every velocity-bin global spatial mean within the frozen 1e-13 relative guard.",
            "The normal heat-flux operator-difference RMS is evaluated on a preregistered 16x16, 32x32, 64x64 sequence without solving or tuning.",
            "The exact 64x64 endpoint is independently reproduced against the completed Stage-68 artifact.",
        ],
        "negative_findings": [
            "A decreasing frozen residual sequence is not a converged-solution or observable-sensitivity result.",
            "The failed Stage-28 MUSCL solver endpoint remains negative and is not rehabilitated.",
            "The 64x64 normal heat-flux operator difference remains material relative to the retained first-order residual.",
            "Cross-Knudsen extension and external-validation claims remain prohibited.",
        ],
        "interpretation_guard": (
            "Restriction changes only spatial representation of the exact frozen "
            "Stage-67 distributions. Observed residual orders diagnose operator "
            "localization and scaling; they do not predict the sign or magnitude "
            "of a converged heat-flux change and do not validate either operator."
        ),
        "scientifically_justified_next_scope": (
            "If the fine-grid defect is wall dominated and full-domain scaling is "
            "slower than interior scaling, independently audit diffuse-wall face "
            "flux discretization before any response or solver experiment. Only a "
            "material non-wall defect with monotone scaling may proceed to a frozen "
            "linearized response audit."
        ),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage67-artifact-dir", required=True)
    parser.add_argument("--stage68-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = run_stage69(
        args.stage67_artifact_dir,
        args.stage68_artifact_dir,
        args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
