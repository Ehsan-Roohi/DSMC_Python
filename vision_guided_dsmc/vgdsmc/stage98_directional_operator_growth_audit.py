from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np

from .stage34_velocity_scale_consistency import local_relaxation_time
from .stage41_projected_polar_operator_audit import (
    STAGE41_CORRECTION_FLOOR,
    mapped_polar_quadrature,
    projected_macroscopic,
)
from .stage58_conservative_solver_64x64_confirmation import build_stage58_config
from .stage90_single_condition_reconstruction_solver_ab_audit import (
    COLD_HOT_RATIO,
    GRID,
    KNUDSEN,
    LIMITER,
    RADIAL_SCALE,
    RULE,
    SOURCE_RELAXATION,
    TOLERANCE,
    _validate_stage67,
    limited_slopes,
    muscl_correction_divergence,
    steady_muscl_iteration_step,
)

STAGE67_RUN_ID = 30991124477
STAGE97_RUN_ID = 31351436518
STAGE97_ARTIFACT_ID = 9051881261
STAGE97_ARTIFACT_SHA256 = "b90d32d172d0ac34c23c878acc4215bec4c0511a7f4d4d2f38b20984903af643"
STAGE97_DECISION = "stage97_interior_dominant_redistribution_stage98_directional_operator_growth_audit"
DIAGNOSTIC_STEPS = 25
BOUNDARY_SLOPE = "zero"
DIRECTIONAL_DOMINANCE_SHARE = 2.0 / 3.0
MATERIAL_DIRECTIONAL_GROWTH_RATIO = 2.0
MATERIAL_CANCELLATION_RATIO = 0.50
DECOMPOSITION_CLOSURE_TOLERANCE = 1.0e-12
PARENT_MAP_REPLAY_TOLERANCE = 1.0e-12


def validate_stage98_design(**overrides: object) -> None:
    frozen = {
        "grid": GRID,
        "kn0": KNUDSEN,
        "cold_hot_ratio": COLD_HOT_RATIO,
        "rule": RULE,
        "radial_scale": RADIAL_SCALE,
        "limiter": LIMITER,
        "boundary_slope": BOUNDARY_SLOPE,
        "source_relaxation": SOURCE_RELAXATION,
        "tolerance": TOLERANCE,
        "correction_floor": STAGE41_CORRECTION_FLOOR,
        "diagnostic_steps": DIAGNOSTIC_STEPS,
        "directional_dominance_share": DIRECTIONAL_DOMINANCE_SHARE,
        "material_directional_growth_ratio": MATERIAL_DIRECTIONAL_GROWTH_RATIO,
        "material_cancellation_ratio": MATERIAL_CANCELLATION_RATIO,
        "decomposition_closure_tolerance": DECOMPOSITION_CLOSURE_TOLERANCE,
        "parent_map_replay_tolerance": PARENT_MAP_REPLAY_TOLERANCE,
        "stage67_run_id": STAGE67_RUN_ID,
        "stage97_run_id": STAGE97_RUN_ID,
    }
    if any(key not in frozen or frozen[key] != value for key, value in overrides.items()):
        raise ValueError(
            "Stage 98 is the fixed directional-operator growth audit authorized by Stage 97. "
            "It replays exactly the retained 25-step zero-boundary-slope diagnostic and may not "
            "retune physics, collision/source treatment, clipping or positivity floor, source "
            "relaxation, transport parameters, wall model, limiter, quadrature, tolerance, "
            "diagnostic window, or any failed solver parameter."
        )


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(denominator, 1.0e-300))


def _relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    return _safe_ratio(
        float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))),
        float(np.linalg.norm(np.asarray(b, dtype=np.float64))),
    )


def muscl_correction_components(
    distribution: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Split the retained zero-boundary-slope MUSCL correction into x and y divergences."""
    f = np.asarray(distribution, dtype=np.float64)
    vx = np.asarray(vx, dtype=np.float64)
    vy = np.asarray(vy, dtype=np.float64)
    if f.ndim != 3 or f.shape[-1] != vx.size or vx.shape != vy.shape:
        raise ValueError("distribution and velocity rule shapes are inconsistent")

    sx = limited_slopes(f, 1, one_sided_boundary=False)
    sy = limited_slopes(f, 0, one_sided_boundary=False)
    sign_x = np.where(vx > 0.0, 1.0, np.where(vx < 0.0, -1.0, 0.0))
    sign_y = np.where(vy > 0.0, 1.0, np.where(vy < 0.0, -1.0, 0.0))
    face_x = 0.5 * vx[None, None, :] * np.where(
        sign_x[None, None, :] > 0.0,
        sx[:, :-1],
        np.where(sign_x[None, None, :] < 0.0, -sx[:, 1:], 0.0),
    )
    face_y = 0.5 * vy[None, None, :] * np.where(
        sign_y[None, None, :] > 0.0,
        sy[:-1],
        np.where(sign_y[None, None, :] < 0.0, -sy[1:], 0.0),
    )
    corr_x = np.zeros_like(f)
    corr_y = np.zeros_like(f)
    corr_x[:, :-1] += face_x / dx
    corr_x[:, 1:] -= face_x / dx
    corr_y[:-1] += face_y / dy
    corr_y[1:] -= face_y / dy
    return corr_x, corr_y


def _directional_metrics(
    distribution: np.ndarray,
    denominator: np.ndarray,
    quadrature,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    dx = 1.0 / GRID[0]
    dy = 1.0 / GRID[1]
    corr_x, corr_y = muscl_correction_components(
        distribution, quadrature.vx, quadrature.vy, dx, dy
    )
    retained = muscl_correction_divergence(
        distribution, quadrature.vx, quadrature.vy, dx, dy, False
    )
    closure = _relative_l2(corr_x + corr_y, retained)

    term_x = corr_x / denominator
    term_y = corr_y / denominator
    term_net = term_x + term_y
    weight = quadrature.weight[None, None, :]

    x_abs = float(np.sum(np.abs(term_x) * weight))
    y_abs = float(np.sum(np.abs(term_y) * weight))
    net_abs = float(np.sum(np.abs(term_net) * weight))
    x_l2 = float(np.linalg.norm(term_x))
    y_l2 = float(np.linalg.norm(term_y))
    net_l2 = float(np.linalg.norm(term_net))
    directional_total = x_abs + y_abs

    maps = {
        "x_signed_m0": np.sum(term_x * weight, axis=-1),
        "y_signed_m0": np.sum(term_y * weight, axis=-1),
        "net_signed_m0": np.sum(term_net * weight, axis=-1),
        "x_abs_m0": np.sum(np.abs(term_x) * weight, axis=-1),
        "y_abs_m0": np.sum(np.abs(term_y) * weight, axis=-1),
        "net_abs_m0": np.sum(np.abs(term_net) * weight, axis=-1),
    }
    metrics = {
        "x_weighted_abs": x_abs,
        "y_weighted_abs": y_abs,
        "net_weighted_abs": net_abs,
        "x_l2": x_l2,
        "y_l2": y_l2,
        "net_l2": net_l2,
        "x_directional_abs_share": _safe_ratio(x_abs, directional_total),
        "y_directional_abs_share": _safe_ratio(y_abs, directional_total),
        "weighted_abs_cancellation_ratio": _safe_ratio(net_abs, directional_total),
        "l2_cancellation_ratio": _safe_ratio(net_l2, x_l2 + y_l2),
        "decomposition_closure_relative_l2": closure,
    }
    return metrics, {key: np.asarray(value, dtype=np.float64) for key, value in maps.items()}


def _metric_history_summary(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    return {
        "first": float(a[0]),
        "final": float(a[-1]),
        "minimum": float(np.min(a)),
        "maximum": float(np.max(a)),
        "final_to_first_ratio": _safe_ratio(float(a[-1]), float(a[0])),
        "maximum_to_first_ratio": _safe_ratio(float(np.max(a)), float(a[0])),
    }


def stage98_decision(
    summary_by_distribution: dict[str, dict[str, dict[str, float]]],
    maximum_decomposition_closure_relative_l2: float,
    maximum_parent_map_replay_relative_l2: float,
) -> str:
    if (
        maximum_decomposition_closure_relative_l2 > DECOMPOSITION_CLOSURE_TOLERANCE
        or maximum_parent_map_replay_relative_l2 > PARENT_MAP_REPLAY_TOLERANCE
    ):
        return "stage98_decomposition_or_replay_mismatch_blocker_without_retuning"

    final_x_share = min(
        summary_by_distribution[d]["x_directional_abs_share"]["final"] for d in ("phi", "psi")
    )
    final_y_share = min(
        summary_by_distribution[d]["y_directional_abs_share"]["final"] for d in ("phi", "psi")
    )
    x_growth = min(
        summary_by_distribution[d]["x_weighted_abs"]["final_to_first_ratio"] for d in ("phi", "psi")
    )
    y_growth = min(
        summary_by_distribution[d]["y_weighted_abs"]["final_to_first_ratio"] for d in ("phi", "psi")
    )
    minimum_cancellation = min(
        summary_by_distribution[d]["weighted_abs_cancellation_ratio"]["minimum"]
        for d in ("phi", "psi")
    )

    if final_x_share >= DIRECTIONAL_DOMINANCE_SHARE and x_growth >= MATERIAL_DIRECTIONAL_GROWTH_RATIO:
        return "stage98_x_dominant_growing_operator_stage99_x_signed_lobe_localization_audit"
    if final_y_share >= DIRECTIONAL_DOMINANCE_SHARE and y_growth >= MATERIAL_DIRECTIONAL_GROWTH_RATIO:
        return "stage98_y_dominant_growing_operator_stage99_y_signed_lobe_localization_audit"
    if minimum_cancellation <= MATERIAL_CANCELLATION_RATIO:
        return "stage98_material_cross_axis_cancellation_stage99_signed_cancellation_localization_audit"
    return "stage98_mixed_directional_growth_stage99_interior_velocity_sector_audit"


def _load_stage97(root: str | Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    root = Path(root)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("stage") != 97 or summary.get("decision") != STAGE97_DECISION:
        raise ValueError("Stage-97 completed endpoint does not authorize Stage 98")
    with np.load(root / "spatial_localization_maps.npz") as data:
        maps = {
            name: np.asarray(data[name], dtype=np.float64)
            for name in ("first_phi", "final_phi", "first_psi", "final_psi")
        }
    return summary, maps


def run_stage98(
    stage67_artifact_dir: str | Path,
    stage97_artifact_dir: str | Path,
    output_dir: str | Path,
    **design: object,
) -> dict[str, object]:
    validate_stage98_design(**design)
    _validate_stage67(stage67_artifact_dir)
    stage97_summary, parent_maps = _load_stage97(stage97_artifact_dir)

    cfg = build_stage58_config()
    quadrature = mapped_polar_quadrature(*RULE, radial_scale=RADIAL_SCALE)
    with np.load(Path(stage67_artifact_dir) / "converged_full_distributions.npz") as saved:
        phi = np.asarray(saved["phi"], dtype=np.float64).copy()
        psi = np.asarray(saved["psi"], dtype=np.float64).copy()
        for name, actual, expected in (
            ("vx", np.asarray(saved["vx"]), np.asarray(quadrature.vx)),
            ("vy", np.asarray(saved["vy"]), np.asarray(quadrature.vy)),
            ("weight", np.asarray(saved["weight"]), np.asarray(quadrature.weight)),
        ):
            if not np.array_equal(actual, expected):
                raise ValueError(f"Stage-67 {name} does not exactly match the frozen Stage-98 quadrature")

    metric_names = (
        "x_weighted_abs",
        "y_weighted_abs",
        "net_weighted_abs",
        "x_l2",
        "y_l2",
        "net_l2",
        "x_directional_abs_share",
        "y_directional_abs_share",
        "weighted_abs_cancellation_ratio",
        "l2_cancellation_ratio",
        "decomposition_closure_relative_l2",
    )
    histories: dict[str, list[float]] = {
        f"{distribution}_{metric}": []
        for distribution in ("phi", "psi")
        for metric in metric_names
    }
    saved_maps: dict[str, np.ndarray] = {}
    maximum_parent_replay = 0.0

    for step in range(1, DIAGNOSTIC_STEPS + 1):
        fields = projected_macroscopic(phi, psi, quadrature)
        tau = local_relaxation_time(fields["rho"], fields["T"], cfg, mapping="paper_consistent_c0")
        nu = 1.0 / np.maximum(tau, 1.0e-14)
        dx, dy = 1.0 / cfg.nx, 1.0 / cfg.ny
        ax = np.abs(quadrature.vx) / dx
        ay = np.abs(quadrature.vy) / dy
        denominator = nu[..., None] + ax[None, None, :] + ay[None, None, :]

        for distribution, field in (("phi", phi), ("psi", psi)):
            metrics, maps = _directional_metrics(field, denominator, quadrature)
            for metric in metric_names:
                histories[f"{distribution}_{metric}"].append(float(metrics[metric]))
            when = "first" if step == 1 else "final" if step == DIAGNOSTIC_STEPS else None
            if when is not None:
                for map_name, value in maps.items():
                    saved_maps[f"{when}_{distribution}_{map_name}"] = value.copy()
                replay_error = _relative_l2(maps["net_abs_m0"], parent_maps[f"{when}_{distribution}"])
                saved_maps[f"{when}_{distribution}_parent_net_abs_m0"] = parent_maps[f"{when}_{distribution}"].copy()
                maximum_parent_replay = max(maximum_parent_replay, replay_error)
            del maps
            gc.collect()

        phi, psi, _ = steady_muscl_iteration_step(
            phi, psi, cfg, quadrature, one_sided_x_boundary=False
        )
        if not (np.isfinite(phi).all() and np.isfinite(psi).all()):
            raise ValueError("Stage-98 exact fixed replay became nonfinite before 25 steps")
        gc.collect()

    summary_by_distribution: dict[str, dict[str, dict[str, float]]] = {}
    for distribution in ("phi", "psi"):
        summary_by_distribution[distribution] = {
            metric: _metric_history_summary(histories[f"{distribution}_{metric}"])
            for metric in metric_names
        }

    max_closure = max(
        max(histories[f"{distribution}_decomposition_closure_relative_l2"])
        for distribution in ("phi", "psi")
    )
    decision = stage98_decision(summary_by_distribution, max_closure, maximum_parent_replay)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    arrays = {
        name: np.asarray(values, dtype=np.float64) for name, values in histories.items()
    }
    arrays.update(saved_maps)
    np.savez_compressed(out / "directional_operator_growth_histories.npz", **arrays)

    result: dict[str, object] = {
        "stage": 98,
        "description": (
            "Fixed 25-step replay of the retained Stage-96 zero-boundary-slope MUSCL diagnostic, "
            "decomposing the second-minus-first-order transport correction into signed x and y "
            "operator contributions without changing the solver design."
        ),
        "retained_stage97_decision": stage97_summary["decision"],
        "configuration": {
            "grid": list(GRID),
            "kn0": KNUDSEN,
            "cold_hot_ratio": COLD_HOT_RATIO,
            "radial_nodes": RULE[0],
            "angular_nodes": RULE[1],
            "point_count": RULE[0] * RULE[1],
            "radial_scale": RADIAL_SCALE,
            "limiter": LIMITER,
            "boundary_slope": BOUNDARY_SLOPE,
            "source_relaxation": SOURCE_RELAXATION,
            "tolerance": TOLERANCE,
            "positivity_floor": cfg.positivity_floor,
            "correction_floor": STAGE41_CORRECTION_FLOOR,
            "diagnostic_steps": DIAGNOSTIC_STEPS,
            "directional_dominance_share": DIRECTIONAL_DOMINANCE_SHARE,
            "material_directional_growth_ratio": MATERIAL_DIRECTIONAL_GROWTH_RATIO,
            "material_cancellation_ratio": MATERIAL_CANCELLATION_RATIO,
            "decomposition_closure_tolerance": DECOMPOSITION_CLOSURE_TOLERANCE,
            "parent_map_replay_tolerance": PARENT_MAP_REPLAY_TOLERANCE,
            "stage67_run_id": STAGE67_RUN_ID,
            "stage97_run_id": STAGE97_RUN_ID,
            "stage97_artifact_id": STAGE97_ARTIFACT_ID,
            "stage97_artifact_sha256": STAGE97_ARTIFACT_SHA256,
            "fixed_diagnostic_replay": True,
            "full_solver_endpoint_rerun": False,
            "solver_endpoint_search": False,
            "physical_parameter_retuning": False,
            "collision_parameter_retuning": False,
            "correction_floor_retuning": False,
            "positivity_floor_retuning": False,
            "source_relaxation_retuning": False,
            "transport_parameter_retuning": False,
            "wall_model_retuning": False,
            "normalization_retuning": False,
            "limiter_retuning": False,
            "velocity_quadrature_retuning": False,
            "failed_muscl_endpoint_rehabilitated": False,
            "one_sided_boundary_slope_promoted": False,
            "cross_knudsen_extension_permitted": False,
            "validation_claim_permitted": False,
            "solver_endpoint_claim_permitted": False,
        },
        "directional_growth": summary_by_distribution,
        "maximum_decomposition_closure_relative_l2": float(max_closure),
        "maximum_parent_map_replay_relative_l2": float(maximum_parent_replay),
        "decision": decision,
        "scientific_conclusion": (
            "Stage 98 measures whether the Stage-97 interior redistribution is preferentially carried "
            "by the retained x or y MUSCL correction and whether x/y cancellation is material. "
            "It is a frozen diagnostic replay, not a solver repair. Directional growth, dominance, "
            "or cancellation may justify further localization but cannot by itself establish causality, "
            "stability, accuracy, or benchmark validity."
        ),
        "negative_result_guard": (
            "Stage 90 remains nonconverged in both reconstruction arms, Stage 28 remains a failed MUSCL "
            "endpoint, and the Stage-89 one-sided boundary slope is not promoted. No failed physical or "
            "numerical parameter is retuned, no cross-Knudsen extension is allowed, and no stable-solver, "
            "accuracy, benchmark, or validation claim is authorized."
        ),
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage67-artifact-dir", required=True)
    parser.add_argument("--stage97-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run_stage98(args.stage67_artifact_dir, args.stage97_artifact_dir, args.output_dir),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
