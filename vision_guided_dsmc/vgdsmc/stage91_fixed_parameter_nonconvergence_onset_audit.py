from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .stage41_projected_polar_operator_audit import mapped_polar_quadrature, projected_macroscopic
from .stage58_conservative_solver_64x64_confirmation import build_stage58_config
from .stage90_single_condition_reconstruction_solver_ab_audit import (
    COLD_HOT_RATIO,
    GRID,
    KNUDSEN,
    LIMITER,
    RADIAL_SCALE,
    RULE,
    SOURCE_RELAXATION,
    STAGE67_COMPLETED_ENDPOINT,
    TOLERANCE,
    _validate_stage67,
    steady_muscl_iteration_step,
)

STAGE90_OBSERVED = {
    "workflow_run_id": 31279256923,
    "workflow_job_id": 93157808959,
    "decision": "stage90_nonconverged_solver_blocker_without_retuning",
    "baseline_final_change": 8.503439760878251,
    "one_sided_final_change": 8.420175955403389,
    "baseline_qav_relative_error": 56.20037875011162,
    "one_sided_qav_relative_error": 56.45298899946829,
    "table3_velocity_rms_error_improvement_fraction": 0.06583340024214912,
    "qav_error_improvement_fraction": -0.004494814002586582,
}
DIAGNOSTIC_STEPS = 25


def validate_stage91_design(**overrides: object) -> None:
    frozen = {
        "grid": GRID,
        "kn0": KNUDSEN,
        "cold_hot_ratio": COLD_HOT_RATIO,
        "rule": RULE,
        "radial_scale": RADIAL_SCALE,
        "limiter": LIMITER,
        "source_relaxation": SOURCE_RELAXATION,
        "tolerance": TOLERANCE,
        "diagnostic_steps": DIAGNOSTIC_STEPS,
    }
    if any(key not in frozen or frozen[key] != value for key, value in overrides.items()):
        raise ValueError(
            "Stage 91 is a fixed-parameter onset diagnostic of the Stage-90 nonconvergence. "
            "It may not retune physics, source relaxation, limiter, quadrature, wall model, "
            "positivity floor, tolerance, or the 25-step diagnostic window."
        )


def _macro_change(previous: dict[str, np.ndarray], current: dict[str, np.ndarray]) -> float:
    velocity_previous = np.stack([previous["u"], previous["v"]], axis=-1)
    velocity_current = np.stack([current["u"], current["v"]], axis=-1)
    heat_previous = np.stack([previous["qx"], previous["qy"]], axis=-1)
    heat_current = np.stack([current["qx"], current["qy"]], axis=-1)
    return max(
        float(np.max(np.abs(current["T"] - previous["T"]))),
        float(np.max(np.abs(velocity_current - velocity_previous))),
        float(np.max(np.abs(heat_current - heat_previous))),
    )


def _first_positive(values: list[float]) -> int | None:
    for index, value in enumerate(values, start=1):
        if value > 0.0:
            return index
    return None


def run_onset_arm(
    initial_phi: np.ndarray,
    initial_psi: np.ndarray,
    cfg,
    quadrature,
    *,
    one_sided_x_boundary: bool,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    phi = np.asarray(initial_phi, dtype=np.float64).copy()
    psi = np.asarray(initial_psi, dtype=np.float64).copy()
    previous = projected_macroscopic(phi, psi, quadrature)

    macro_change: list[float] = []
    phi_floor_fraction: list[float] = []
    psi_floor_fraction: list[float] = []
    min_candidate_phi: list[float] = []
    min_candidate_psi: list[float] = []
    max_phi_equilibrium_clip: list[float] = []
    max_psi_equilibrium_clip: list[float] = []
    finite = True

    for _ in range(DIAGNOSTIC_STEPS):
        phi, psi, diag = steady_muscl_iteration_step(
            phi,
            psi,
            cfg,
            quadrature,
            one_sided_x_boundary,
        )
        finite = bool(np.isfinite(phi).all() and np.isfinite(psi).all())
        phi_floor_fraction.append(float(diag["phi_update_floor_fraction"]))
        psi_floor_fraction.append(float(diag["psi_update_floor_fraction"]))
        min_candidate_phi.append(float(diag["minimum_candidate_phi"]))
        min_candidate_psi.append(float(diag["minimum_candidate_psi"]))
        max_phi_equilibrium_clip.append(float(diag["maximum_phi_clipped_weight_fraction"]))
        max_psi_equilibrium_clip.append(float(diag["maximum_psi_clipped_weight_fraction"]))
        if not finite:
            macro_change.append(math.inf)
            break
        current = projected_macroscopic(phi, psi, quadrature)
        macro_change.append(_macro_change(previous, current))
        previous = current

    executed = len(macro_change)
    first_phi_floor = _first_positive(phi_floor_fraction)
    first_psi_floor = _first_positive(psi_floor_fraction)
    first_change = float(macro_change[0]) if macro_change else math.inf
    final_change = float(macro_change[-1]) if macro_change else math.inf
    finite_changes = np.asarray([v for v in macro_change if np.isfinite(v)], dtype=np.float64)
    summary = {
        "finite": finite,
        "executed_steps": executed,
        "first_phi_floor_activation_step": first_phi_floor,
        "first_psi_floor_activation_step": first_psi_floor,
        "maximum_phi_floor_fraction": float(max(phi_floor_fraction, default=0.0)),
        "maximum_psi_floor_fraction": float(max(psi_floor_fraction, default=0.0)),
        "minimum_candidate_phi": float(min(min_candidate_phi, default=math.inf)),
        "minimum_candidate_psi": float(min(min_candidate_psi, default=math.inf)),
        "maximum_phi_equilibrium_clipped_weight_fraction": float(max(max_phi_equilibrium_clip, default=0.0)),
        "maximum_psi_equilibrium_clipped_weight_fraction": float(max(max_psi_equilibrium_clip, default=0.0)),
        "first_step_macro_change": first_change,
        "final_step_macro_change": final_change,
        "maximum_macro_change": float(np.max(finite_changes)) if finite_changes.size else math.inf,
        "final_to_first_macro_change_ratio": final_change / max(first_change, 1.0e-300) if np.isfinite(final_change) else math.inf,
    }
    history = {
        "macro_change": np.asarray(macro_change, dtype=np.float64),
        "phi_floor_fraction": np.asarray(phi_floor_fraction, dtype=np.float64),
        "psi_floor_fraction": np.asarray(psi_floor_fraction, dtype=np.float64),
        "minimum_candidate_phi": np.asarray(min_candidate_phi, dtype=np.float64),
        "minimum_candidate_psi": np.asarray(min_candidate_psi, dtype=np.float64),
        "maximum_phi_equilibrium_clipped_weight_fraction": np.asarray(max_phi_equilibrium_clip, dtype=np.float64),
        "maximum_psi_equilibrium_clipped_weight_fraction": np.asarray(max_psi_equilibrium_clip, dtype=np.float64),
    }
    return summary, history


def stage91_decision(baseline: dict[str, object], one_sided: dict[str, object]) -> str:
    if not bool(baseline["finite"]) or not bool(one_sided["finite"]):
        return "stage91_nonfinite_onset_blocker_without_retuning"
    baseline_floor = baseline["first_phi_floor_activation_step"] is not None or baseline["first_psi_floor_activation_step"] is not None
    one_sided_floor = one_sided["first_phi_floor_activation_step"] is not None or one_sided["first_psi_floor_activation_step"] is not None
    if baseline_floor and one_sided_floor:
        return "stage91_both_arms_activate_positivity_floor_within_fixed_onset_window_stage92_candidate_update_localization_audit"
    if one_sided_floor and not baseline_floor:
        return "stage91_boundary_specific_floor_onset_signal_stage92_candidate_update_localization_audit"
    if baseline_floor and not one_sided_floor:
        return "stage91_baseline_floor_onset_precedes_boundary_counterfactual_interpretation_stage92_candidate_update_localization_audit"
    return "stage91_no_floor_onset_within_fixed_window_blocker_without_retuning"


def run_stage91(stage67_artifact_dir: str | Path, output_dir: str | Path, **design: object) -> dict[str, object]:
    validate_stage91_design(**design)
    stage67_summary = _validate_stage67(stage67_artifact_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = build_stage58_config()
    quadrature = mapped_polar_quadrature(*RULE, radial_scale=RADIAL_SCALE)
    with np.load(Path(stage67_artifact_dir) / "converged_full_distributions.npz") as saved:
        initial_phi = np.asarray(saved["phi"], dtype=np.float64)
        initial_psi = np.asarray(saved["psi"], dtype=np.float64)
        for name, actual, expected in (
            ("vx", np.asarray(saved["vx"]), np.asarray(quadrature.vx)),
            ("vy", np.asarray(saved["vy"]), np.asarray(quadrature.vy)),
            ("weight", np.asarray(saved["weight"]), np.asarray(quadrature.weight)),
        ):
            if not np.array_equal(actual, expected):
                raise ValueError(f"Stage-67 {name} does not exactly match the frozen Stage-91 quadrature")

        baseline, baseline_history = run_onset_arm(
            initial_phi,
            initial_psi,
            cfg,
            quadrature,
            one_sided_x_boundary=False,
        )
        one_sided, one_sided_history = run_onset_arm(
            initial_phi,
            initial_psi,
            cfg,
            quadrature,
            one_sided_x_boundary=True,
        )

    np.savez_compressed(
        out / "onset_histories.npz",
        baseline_macro_change=baseline_history["macro_change"],
        baseline_phi_floor_fraction=baseline_history["phi_floor_fraction"],
        baseline_psi_floor_fraction=baseline_history["psi_floor_fraction"],
        baseline_minimum_candidate_phi=baseline_history["minimum_candidate_phi"],
        baseline_minimum_candidate_psi=baseline_history["minimum_candidate_psi"],
        one_sided_macro_change=one_sided_history["macro_change"],
        one_sided_phi_floor_fraction=one_sided_history["phi_floor_fraction"],
        one_sided_psi_floor_fraction=one_sided_history["psi_floor_fraction"],
        one_sided_minimum_candidate_phi=one_sided_history["minimum_candidate_phi"],
        one_sided_minimum_candidate_psi=one_sided_history["minimum_candidate_psi"],
    )

    decision = stage91_decision(baseline, one_sided)
    summary = {
        "stage": 91,
        "description": "Fixed-parameter 25-step onset audit following the Stage-90 nonlinear nonconvergence blocker.",
        "retained_stage67_decision": stage67_summary["decision"],
        "retained_stage90_observation": STAGE90_OBSERVED,
        "configuration": {
            "grid": list(GRID),
            "kn0": KNUDSEN,
            "cold_hot_ratio": COLD_HOT_RATIO,
            "radial_nodes": RULE[0],
            "angular_nodes": RULE[1],
            "point_count": RULE[0] * RULE[1],
            "radial_scale": RADIAL_SCALE,
            "limiter": LIMITER,
            "source_relaxation": SOURCE_RELAXATION,
            "tolerance": TOLERANCE,
            "diagnostic_steps": DIAGNOSTIC_STEPS,
            "initialization": "exact completed Stage-67 converged phi/psi for both arms",
            "physical_parameter_retuning": False,
            "collision_parameter_retuning": False,
            "correction_floor_retuning": False,
            "source_relaxation_retuning": False,
            "transport_parameter_retuning": False,
            "wall_model_retuning": False,
            "normalization_retuning": False,
            "limiter_retuning": False,
            "velocity_quadrature_retuning": False,
            "failed_muscl_endpoint_rehabilitated": False,
            "cross_knudsen_extension_permitted": False,
            "validation_claim_permitted": False,
        },
        "zero_boundary_slope": baseline,
        "one_sided_boundary_slope": one_sided,
        "paired_onset_comparison": {
            "final_macro_change_ratio_one_sided_over_baseline": float(one_sided["final_step_macro_change"]) / max(float(baseline["final_step_macro_change"]), 1.0e-300),
            "maximum_phi_floor_fraction_difference_one_sided_minus_baseline": float(one_sided["maximum_phi_floor_fraction"]) - float(baseline["maximum_phi_floor_fraction"]),
            "maximum_psi_floor_fraction_difference_one_sided_minus_baseline": float(one_sided["maximum_psi_floor_fraction"]) - float(baseline["maximum_psi_floor_fraction"]),
        },
        "decision": decision,
        "scientific_conclusion": (
            "Stage 91 does not seek a converged endpoint. It localizes the earliest fixed-configuration onset of candidate negativity, positivity-floor activation, and macroscopic update growth in the two Stage-90 arms. "
            "Any observed floor activation is a numerical symptom to localize, not evidence that the positivity floor, wall model, limiter, transport parameters, or physical model should be retuned."
        ),
        "negative_result_guard": (
            "Stage 90 remains nonconverged in both arms and the Stage-28 MUSCL endpoint remains negative. No parameter is retuned, no cross-Knudsen extension is allowed, and no benchmark or validation claim is authorized by this onset diagnostic."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage67-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run_stage91(args.stage67_artifact_dir, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
