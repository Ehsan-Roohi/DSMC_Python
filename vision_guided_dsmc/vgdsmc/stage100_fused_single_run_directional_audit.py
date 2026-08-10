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
    steady_muscl_iteration_step,
)
from .stage96_muscl_correction_growth_audit import correction_diagnostic
from .stage98_directional_operator_growth_audit import (
    DECOMPOSITION_CLOSURE_TOLERANCE,
    DIAGNOSTIC_STEPS,
    DIRECTIONAL_DOMINANCE_SHARE,
    MATERIAL_CANCELLATION_RATIO,
    MATERIAL_DIRECTIONAL_GROWTH_RATIO,
    _directional_metrics,
    _metric_history_summary,
)

STAGE67_RUN_ID = 30991124477
STAGE99_RUN_ID = 31378863028
STAGE99_JOB_ID = 93424248816
STAGE99_ARTIFACT_ID = 9062898563
STAGE99_ARTIFACT_SHA256 = "f76a778b9317c3921e3a19dc808b2eabbd4d5a760513bd2999a55a27e278d10c"
STAGE99_SUMMARY_SHA256 = "392faa3b817e06bc463461106c690a9f4fa93b2421064d82b25fae8294f27e17"
STAGE99_DECISION = "stage99_cross_run_iterative_replay_drift_stage100_fused_single_run_directional_audit"
BOUNDARY_SLOPE = "zero"
SAME_RUN_PARENT_MAP_TOLERANCE = 1.0e-12


def validate_stage100_design(**overrides: object) -> None:
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
        "same_run_parent_map_tolerance": SAME_RUN_PARENT_MAP_TOLERANCE,
        "stage67_run_id": STAGE67_RUN_ID,
        "stage99_run_id": STAGE99_RUN_ID,
    }
    if any(key not in frozen or frozen[key] != value for key, value in overrides.items()):
        raise ValueError(
            "Stage 100 is the frozen fused single-run audit authorized by Stage 99. It may not "
            "retune physics, collision/source treatment, clipping or positivity floors, source "
            "relaxation, transport parameters, wall model, limiter, quadrature, normalization, "
            "tolerance, diagnostic window, or any failed solver parameter."
        )


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(abs(denominator), 1.0e-300))


def _relative_l2(actual: np.ndarray, reference: np.ndarray) -> float:
    a = np.asarray(actual, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    return _safe_ratio(float(np.linalg.norm(a - b)), float(np.linalg.norm(b)))


def _load_and_validate_stage99(root: str | Path) -> dict[str, object]:
    root = Path(root)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("stage") != 99 or summary.get("decision") != STAGE99_DECISION:
        raise ValueError("Stage-99 artifact does not authorize the fused Stage-100 audit")
    cfg = summary.get("configuration", {})
    if not isinstance(cfg, dict) or cfg.get("artifact_only") is not True:
        raise ValueError("Stage-99 provenance contract is incomplete")
    if float(summary.get("final_replay_max_relative_l2", 0.0)) <= SAME_RUN_PARENT_MAP_TOLERANCE:
        raise ValueError("Stage-99 artifact does not contain the retained cross-run replay blocker")
    return summary


def stage100_decision(
    summary_by_distribution: dict[str, dict[str, dict[str, float]]],
    maximum_decomposition_closure_relative_l2: float,
    maximum_same_run_parent_map_relative_l2: float,
    finite: bool,
) -> str:
    if not finite:
        return "stage100_nonfinite_fused_replay_blocker_without_retuning"
    if (
        maximum_decomposition_closure_relative_l2 > DECOMPOSITION_CLOSURE_TOLERANCE
        or maximum_same_run_parent_map_relative_l2 > SAME_RUN_PARENT_MAP_TOLERANCE
    ):
        return "stage100_same_run_decomposition_or_parent_mismatch_blocker_without_retuning"

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
        return "stage100_x_dominant_growing_operator_stage101_x_signed_lobe_localization_audit"
    if final_y_share >= DIRECTIONAL_DOMINANCE_SHARE and y_growth >= MATERIAL_DIRECTIONAL_GROWTH_RATIO:
        return "stage100_y_dominant_growing_operator_stage101_y_signed_lobe_localization_audit"
    if minimum_cancellation <= MATERIAL_CANCELLATION_RATIO:
        return "stage100_material_cross_axis_cancellation_stage101_signed_cancellation_localization_audit"
    return "stage100_mixed_directional_growth_stage101_interior_velocity_sector_audit"


def run_stage100(
    stage67_artifact_dir: str | Path,
    stage99_artifact_dir: str | Path,
    output_dir: str | Path,
    **design: object,
) -> dict[str, object]:
    validate_stage100_design(**design)
    _validate_stage67(stage67_artifact_dir)
    stage99_summary = _load_and_validate_stage99(stage99_artifact_dir)

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
                raise ValueError(f"Stage-67 {name} does not exactly match the frozen Stage-100 quadrature")

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
        "same_run_parent_map_relative_l2",
    )
    histories: dict[str, list[float]] = {
        f"{distribution}_{metric}": []
        for distribution in ("phi", "psi")
        for metric in metric_names
    }
    saved_maps: dict[str, np.ndarray] = {}
    finite = True

    for step in range(1, DIAGNOSTIC_STEPS + 1):
        fields = projected_macroscopic(phi, psi, quadrature)
        tau = local_relaxation_time(fields["rho"], fields["T"], cfg, mapping="paper_consistent_c0")
        nu = 1.0 / np.maximum(tau, 1.0e-14)
        dx, dy = 1.0 / cfg.nx, 1.0 / cfg.ny
        ax = np.abs(quadrature.vx) / dx
        ay = np.abs(quadrature.vy) / dy
        denominator = nu[..., None] + ax[None, None, :] + ay[None, None, :]

        _, parent_maps = correction_diagnostic(phi, psi, cfg, quadrature)
        for distribution, field in (("phi", phi), ("psi", psi)):
            metrics, maps = _directional_metrics(field, denominator, quadrature)
            parent = np.asarray(parent_maps[f"{distribution}_cell_correction_m0"], dtype=np.float64)
            same_run_parent_error = _relative_l2(maps["net_abs_m0"], parent)
            for metric in metric_names:
                value = same_run_parent_error if metric == "same_run_parent_map_relative_l2" else metrics[metric]
                histories[f"{distribution}_{metric}"].append(float(value))

            when = "first" if step == 1 else "final" if step == DIAGNOSTIC_STEPS else None
            if when is not None:
                saved_maps[f"{when}_{distribution}_parent_net_abs_m0"] = parent.copy()
                for map_name, value in maps.items():
                    saved_maps[f"{when}_{distribution}_{map_name}"] = np.asarray(value, dtype=np.float64).copy()
            del maps
        del parent_maps
        gc.collect()

        phi, psi, _ = steady_muscl_iteration_step(
            phi, psi, cfg, quadrature, one_sided_x_boundary=False
        )
        finite = bool(np.isfinite(phi).all() and np.isfinite(psi).all())
        if not finite:
            break
        gc.collect()

    summary_by_distribution: dict[str, dict[str, dict[str, float]]] = {}
    for distribution in ("phi", "psi"):
        summary_by_distribution[distribution] = {
            metric: _metric_history_summary(histories[f"{distribution}_{metric}"])
            for metric in metric_names
            if histories[f"{distribution}_{metric}"]
        }

    max_closure = max(
        max(histories[f"{distribution}_decomposition_closure_relative_l2"] or [float("inf")])
        for distribution in ("phi", "psi")
    )
    max_same_run_parent = max(
        max(histories[f"{distribution}_same_run_parent_map_relative_l2"] or [float("inf")])
        for distribution in ("phi", "psi")
    )
    decision = stage100_decision(
        summary_by_distribution,
        max_closure,
        max_same_run_parent,
        finite,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.asarray(values, dtype=np.float64) for name, values in histories.items()}
    arrays.update(saved_maps)
    np.savez_compressed(out / "fused_directional_histories.npz", **arrays)

    result: dict[str, object] = {
        "stage": 100,
        "description": (
            "Fused single-run replay of the retained 25-step zero-boundary-slope MUSCL diagnostic. "
            "The Stage-96 parent correction map and Stage-98 x/y directional decomposition are "
            "computed from the same in-memory state at every step so cross-run iterative drift "
            "cannot contaminate the attribution gate."
        ),
        "configuration": {
            "grid": list(GRID),
            "kn0": KNUDSEN,
            "cold_hot_ratio": COLD_HOT_RATIO,
            "rule": list(RULE),
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
            "same_run_parent_map_tolerance": SAME_RUN_PARENT_MAP_TOLERANCE,
            "stage67_run_id": STAGE67_RUN_ID,
            "stage99_run_id": STAGE99_RUN_ID,
            "stage99_job_id": STAGE99_JOB_ID,
            "stage99_artifact_id": STAGE99_ARTIFACT_ID,
            "stage99_artifact_sha256": STAGE99_ARTIFACT_SHA256,
            "fused_single_run": True,
            "full_solver_endpoint_rerun": False,
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
            "cross_knudsen_extension_permitted": False,
            "validation_claim_permitted": False,
            "solver_endpoint_claim_permitted": False,
        },
        "stage99_authorization": {
            "decision": stage99_summary["decision"],
            "cross_run_final_replay_max_relative_l2": stage99_summary["final_replay_max_relative_l2"],
            "cross_run_final_replay_to_strict_tolerance_ratio": stage99_summary["final_replay_to_strict_tolerance_ratio"],
        },
        "executed_steps": min(
            len(histories["phi_x_weighted_abs"]),
            len(histories["psi_x_weighted_abs"]),
        ),
        "finite": finite,
        "maximum_decomposition_closure_relative_l2": float(max_closure),
        "maximum_same_run_parent_map_relative_l2": float(max_same_run_parent),
        "directional_summary": summary_by_distribution,
        "decision": decision,
        "scientific_conclusion": (
            "Stage 100 preserves the Stage-99 1e-12 reproducibility criterion while removing only "
            "cross-run state drift from the attribution experiment. Passing same-run closure permits "
            "the frozen x/y growth and cancellation metrics to be interpreted diagnostically; failing "
            "that closure remains a blocker. In neither case does this establish benchmark improvement, "
            "nonlinear MUSCL stability, or physical validation."
        ),
        "negative_result_guard": (
            "Stage 98 remains a negative cross-run replay result. Stage 90 remains nonconverged in both "
            "reconstruction arms, Stage 28 remains a failed MUSCL endpoint, and the Stage-89 one-sided "
            "boundary slope is not promoted. No failed parameter is retuned, no cross-Knudsen MUSCL "
            "extension is permitted, and no accuracy, stability, benchmark, or validation claim is authorized."
        ),
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage67-artifact-dir", required=True)
    parser.add_argument("--stage99-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run_stage100(
        args.stage67_artifact_dir,
        args.stage99_artifact_dir,
        args.output_dir,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
