from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import numpy as np

from .stage34_velocity_scale_consistency import local_relaxation_time
from .stage41_projected_polar_operator_audit import (
    STAGE41_CORRECTION_FLOOR,
    mapped_polar_quadrature,
    projected_macroscopic,
    projected_shakhov_equilibrium,
)
from .stage42_projected_polar_heated_cavity_pilot import _upwind_neighbors, projected_wall_incoming
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
    muscl_correction_divergence,
    steady_muscl_iteration_step,
)

DIAGNOSTIC_STEPS = 25
BOUNDARY_SLOPE = "zero"
MATERIAL_CORRECTION_RATIO = 0.10
MATERIAL_GROWTH_RATIO = 2.0


def validate_stage96_design(**overrides: object) -> None:
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
        "material_correction_ratio": MATERIAL_CORRECTION_RATIO,
        "material_growth_ratio": MATERIAL_GROWTH_RATIO,
    }
    if any(key not in frozen or frozen[key] != value for key, value in overrides.items()):
        raise ValueError(
            "Stage 96 is a fixed-design diagnostic of the retained zero-boundary-slope "
            "MUSCL correction at the Stage-90 condition. It may not retune physics, "
            "collision/source treatment, clipping floor, positivity floor, source "
            "relaxation, transport parameters, wall model, limiter, quadrature, "
            "tolerance, or the 25-step diagnostic window."
        )


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(denominator, 1.0e-300))


def _relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    return _safe_ratio(float(np.linalg.norm(np.asarray(a) - np.asarray(b))), float(np.linalg.norm(np.asarray(b))))


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.count_nonzero(mask)) < 3:
        return None
    x = x[mask]
    y = y[mask]
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx <= 1.0e-300 or sy <= 1.0e-300:
        return None
    return float(np.corrcoef(x, y)[0, 1])


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


def _component_stats(
    field: np.ndarray,
    equilibrium: np.ndarray,
    upwind_x: np.ndarray,
    upwind_y: np.ndarray,
    correction_divergence: np.ndarray,
    nu: np.ndarray,
    ax: np.ndarray,
    ay: np.ndarray,
    denominator: np.ndarray,
    weight: np.ndarray,
) -> dict[str, float]:
    f = np.asarray(field, dtype=np.float64)
    eq = np.asarray(equilibrium, dtype=np.float64)

    transport = (
        ax[None, None, :] * (f - upwind_x)
        + ay[None, None, :] * (f - upwind_y)
    ) / denominator
    correction = np.asarray(correction_divergence, dtype=np.float64) / denominator
    first_order_residual = transport + nu[..., None] * (f - eq) / denominator
    full_muscl_residual = first_order_residual + correction

    transport_l2 = float(np.linalg.norm(transport))
    correction_l2 = float(np.linalg.norm(correction))
    first_order_l2 = float(np.linalg.norm(first_order_residual))
    full_l2 = float(np.linalg.norm(full_muscl_residual))

    w = np.asarray(weight, dtype=np.float64)[None, None, :]
    transport_weighted = float(np.sum(np.abs(transport) * w))
    correction_weighted = float(np.sum(np.abs(correction) * w))
    first_order_weighted = float(np.sum(np.abs(first_order_residual) * w))
    full_weighted = float(np.sum(np.abs(full_muscl_residual) * w))

    stats = {
        "transport_l2": transport_l2,
        "correction_l2": correction_l2,
        "first_order_residual_l2": first_order_l2,
        "full_muscl_residual_l2": full_l2,
        "correction_to_transport_l2_ratio": _safe_ratio(correction_l2, transport_l2),
        "correction_to_first_order_residual_l2_ratio": _safe_ratio(correction_l2, first_order_l2),
        "correction_to_transport_weighted_abs_ratio": _safe_ratio(correction_weighted, transport_weighted),
        "correction_to_first_order_residual_weighted_abs_ratio": _safe_ratio(correction_weighted, first_order_weighted),
        "full_to_first_order_residual_l2_ratio": _safe_ratio(full_l2, first_order_l2),
        "full_to_first_order_residual_weighted_abs_ratio": _safe_ratio(full_weighted, first_order_weighted),
        "correction_first_order_cancellation_ratio": _safe_ratio(
            full_l2, first_order_l2 + correction_l2
        ),
    }
    del transport, correction, first_order_residual, full_muscl_residual
    return stats


def correction_diagnostic(
    phi: np.ndarray,
    psi: np.ndarray,
    cfg,
    quadrature,
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
    incoming = projected_wall_incoming(phi, psi, cfg, quadrature)
    left_phi, left_psi, right_phi, right_psi, bottom_phi, bottom_psi, top_phi, top_psi = incoming
    fields = projected_macroscopic(phi, psi, quadrature)
    equilibrium_phi, equilibrium_psi, _ = projected_shakhov_equilibrium(
        fields,
        quadrature,
        prandtl=cfg.prandtl,
        correction_floor=STAGE41_CORRECTION_FLOOR,
    )
    tau = local_relaxation_time(fields["rho"], fields["T"], cfg, mapping="paper_consistent_c0")
    nu = 1.0 / np.maximum(tau, 1.0e-14)
    dx, dy = 1.0 / cfg.nx, 1.0 / cfg.ny
    ax = np.abs(quadrature.vx) / dx
    ay = np.abs(quadrature.vy) / dy
    denominator = nu[..., None] + ax[None, None, :] + ay[None, None, :]

    phi_x, phi_y = _upwind_neighbors(
        phi, left_phi, right_phi, bottom_phi, top_phi, quadrature
    )
    phi_correction = muscl_correction_divergence(
        phi, quadrature.vx, quadrature.vy, dx, dy, False
    )
    phi_stats = _component_stats(
        phi,
        equilibrium_phi,
        phi_x,
        phi_y,
        phi_correction,
        nu,
        ax,
        ay,
        denominator,
        quadrature.weight,
    )
    phi_cell_correction_m0 = np.sum(
        np.abs(phi_correction / denominator) * quadrature.weight[None, None, :],
        axis=-1,
    )
    del phi_x, phi_y, phi_correction, equilibrium_phi
    gc.collect()

    psi_x, psi_y = _upwind_neighbors(
        psi, left_psi, right_psi, bottom_psi, top_psi, quadrature
    )
    psi_correction = muscl_correction_divergence(
        psi, quadrature.vx, quadrature.vy, dx, dy, False
    )
    psi_stats = _component_stats(
        psi,
        equilibrium_psi,
        psi_x,
        psi_y,
        psi_correction,
        nu,
        ax,
        ay,
        denominator,
        quadrature.weight,
    )
    psi_cell_correction_m0 = np.sum(
        np.abs(psi_correction / denominator) * quadrature.weight[None, None, :],
        axis=-1,
    )
    del psi_x, psi_y, psi_correction, equilibrium_psi
    gc.collect()

    return (
        {"phi": phi_stats, "psi": psi_stats},
        {
            "phi_cell_correction_m0": np.asarray(phi_cell_correction_m0, dtype=np.float64),
            "psi_cell_correction_m0": np.asarray(psi_cell_correction_m0, dtype=np.float64),
        },
    )


def _history_summary(history: np.ndarray) -> dict[str, float]:
    values = np.asarray(history, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "first": math.inf,
            "final": math.inf,
            "maximum": math.inf,
            "final_to_first_ratio": math.inf,
            "maximum_to_first_ratio": math.inf,
        }
    first = float(finite[0])
    final = float(finite[-1])
    maximum = float(np.max(finite))
    return {
        "first": first,
        "final": final,
        "maximum": maximum,
        "final_to_first_ratio": _safe_ratio(final, first),
        "maximum_to_first_ratio": _safe_ratio(maximum, first),
    }


def stage96_decision(
    *,
    finite: bool,
    maximum_correction_to_transport_ratio: float,
    maximum_growth_ratio: float,
) -> str:
    if not finite:
        return "stage96_nonfinite_fixed_window_blocker_without_retuning"
    if maximum_correction_to_transport_ratio >= MATERIAL_CORRECTION_RATIO:
        if maximum_growth_ratio >= MATERIAL_GROWTH_RATIO:
            return "stage96_material_and_growing_muscl_correction_stage97_spatial_localization_audit"
        return "stage96_material_persistent_muscl_correction_stage97_spatial_localization_audit"
    return "stage96_muscl_correction_submaterial_in_fixed_window_blocker_without_retuning"


def run_stage96(
    stage67_artifact_dir: str | Path,
    output_dir: str | Path,
    **design: object,
) -> dict[str, object]:
    validate_stage96_design(**design)
    stage67_summary = _validate_stage67(stage67_artifact_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

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
                raise ValueError(
                    f"Stage-67 {name} does not exactly match the frozen Stage-96 quadrature"
                )

    previous = projected_macroscopic(phi, psi, quadrature)
    macro_change: list[float] = []
    phi_floor_fraction: list[float] = []
    psi_floor_fraction: list[float] = []
    minimum_candidate_phi: list[float] = []
    minimum_candidate_psi: list[float] = []
    histories: dict[str, list[float]] = {}
    first_maps: dict[str, np.ndarray] | None = None
    final_maps: dict[str, np.ndarray] | None = None
    finite = True

    scalar_keys = (
        "correction_to_transport_l2_ratio",
        "correction_to_first_order_residual_l2_ratio",
        "correction_to_transport_weighted_abs_ratio",
        "correction_to_first_order_residual_weighted_abs_ratio",
        "full_to_first_order_residual_l2_ratio",
        "correction_first_order_cancellation_ratio",
    )
    for distribution in ("phi", "psi"):
        for key in scalar_keys:
            histories[f"{distribution}_{key}"] = []

    for step in range(1, DIAGNOSTIC_STEPS + 1):
        stats, maps = correction_diagnostic(phi, psi, cfg, quadrature)
        if step == 1:
            first_maps = {name: value.copy() for name, value in maps.items()}
        if step == DIAGNOSTIC_STEPS:
            final_maps = {name: value.copy() for name, value in maps.items()}
        for distribution in ("phi", "psi"):
            for key in scalar_keys:
                histories[f"{distribution}_{key}"].append(float(stats[distribution][key]))

        phi, psi, diag = steady_muscl_iteration_step(
            phi,
            psi,
            cfg,
            quadrature,
            one_sided_x_boundary=False,
        )
        finite = bool(np.isfinite(phi).all() and np.isfinite(psi).all())
        phi_floor_fraction.append(float(diag["phi_update_floor_fraction"]))
        psi_floor_fraction.append(float(diag["psi_update_floor_fraction"]))
        minimum_candidate_phi.append(float(diag["minimum_candidate_phi"]))
        minimum_candidate_psi.append(float(diag["minimum_candidate_psi"]))
        if not finite:
            macro_change.append(math.inf)
            break
        current = projected_macroscopic(phi, psi, quadrature)
        macro_change.append(_macro_change(previous, current))
        previous = current
        gc.collect()

    executed_steps = len(macro_change)
    if final_maps is None:
        final_maps = first_maps

    arrays = {
        "macro_change": np.asarray(macro_change, dtype=np.float64),
        "phi_floor_fraction": np.asarray(phi_floor_fraction, dtype=np.float64),
        "psi_floor_fraction": np.asarray(psi_floor_fraction, dtype=np.float64),
        "minimum_candidate_phi": np.asarray(minimum_candidate_phi, dtype=np.float64),
        "minimum_candidate_psi": np.asarray(minimum_candidate_psi, dtype=np.float64),
    }
    arrays.update(
        {name: np.asarray(values, dtype=np.float64) for name, values in histories.items()}
    )
    if first_maps is not None:
        arrays.update({f"first_{name}": value for name, value in first_maps.items()})
    if final_maps is not None:
        arrays.update({f"final_{name}": value for name, value in final_maps.items()})
    np.savez_compressed(out / "muscl_correction_growth_histories.npz", **arrays)

    trace_summary: dict[str, dict[str, float | None]] = {}
    maximum_correction_to_transport_ratio = 0.0
    maximum_growth_ratio = 0.0
    for distribution in ("phi", "psi"):
        l2_key = f"{distribution}_correction_to_transport_l2_ratio"
        weighted_key = f"{distribution}_correction_to_transport_weighted_abs_ratio"
        l2_summary = _history_summary(arrays[l2_key])
        weighted_summary = _history_summary(arrays[weighted_key])
        maximum_correction_to_transport_ratio = max(
            maximum_correction_to_transport_ratio,
            float(l2_summary["maximum"]),
            float(weighted_summary["maximum"]),
        )
        maximum_growth_ratio = max(
            maximum_growth_ratio,
            float(l2_summary["maximum_to_first_ratio"]),
            float(weighted_summary["maximum_to_first_ratio"]),
        )
        trace_summary[distribution] = {
            "correction_to_transport_l2_first": l2_summary["first"],
            "correction_to_transport_l2_final": l2_summary["final"],
            "correction_to_transport_l2_maximum": l2_summary["maximum"],
            "correction_to_transport_l2_maximum_to_first_ratio": l2_summary[
                "maximum_to_first_ratio"
            ],
            "correction_to_transport_weighted_abs_first": weighted_summary["first"],
            "correction_to_transport_weighted_abs_final": weighted_summary["final"],
            "correction_to_transport_weighted_abs_maximum": weighted_summary["maximum"],
            "correction_to_transport_weighted_abs_maximum_to_first_ratio": weighted_summary[
                "maximum_to_first_ratio"
            ],
            "pearson_l2_ratio_vs_macro_change": _pearson(
                arrays[l2_key], arrays["macro_change"][: arrays[l2_key].size]
            ),
            "pearson_weighted_ratio_vs_macro_change": _pearson(
                arrays[weighted_key],
                arrays["macro_change"][: arrays[weighted_key].size],
            ),
        }

    decision = stage96_decision(
        finite=finite,
        maximum_correction_to_transport_ratio=maximum_correction_to_transport_ratio,
        maximum_growth_ratio=maximum_growth_ratio,
    )
    summary: dict[str, object] = {
        "stage": 96,
        "description": (
            "Fixed 25-step retained-baseline MUSCL correction growth audit following "
            "the Stage-95 finding that first-update positivity-floor propagation is negligible."
        ),
        "retained_stage67_decision": stage67_summary["decision"],
        "retained_stage95_decision": (
            "stage95_floor_propagation_negligible_stage96_muscl_correction_growth_audit"
        ),
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
            "material_correction_ratio": MATERIAL_CORRECTION_RATIO,
            "material_growth_ratio": MATERIAL_GROWTH_RATIO,
            "initialization": "exact completed Stage-67 converged phi/psi",
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
        "finite": finite,
        "executed_steps": executed_steps,
        "macro_change": _history_summary(arrays["macro_change"]),
        "maximum_phi_floor_fraction": float(max(phi_floor_fraction, default=0.0)),
        "maximum_psi_floor_fraction": float(max(psi_floor_fraction, default=0.0)),
        "minimum_candidate_phi": float(min(minimum_candidate_phi, default=math.inf)),
        "minimum_candidate_psi": float(min(minimum_candidate_psi, default=math.inf)),
        "correction_growth": trace_summary,
        "maximum_correction_to_transport_ratio": maximum_correction_to_transport_ratio,
        "maximum_correction_growth_ratio": maximum_growth_ratio,
        "decision": decision,
        "scientific_conclusion": (
            "Stage 96 measures the size and evolution of the retained MUSCL correction "
            "relative to the fixed first-order transport residual scale during the same "
            "25-step diagnostic window used after Stage 90. A material or growing "
            "correction is a localization signal only; correlation with macroscopic "
            "change does not prove causality or justify changing the limiter, boundary "
            "slope, positivity floor, relaxation, wall model, collision model, or physics."
        ),
        "negative_result_guard": (
            "Stage 90 remains nonconverged in both reconstruction arms, Stage 28 remains "
            "a failed MUSCL endpoint, and the Stage-89 one-sided boundary slope is not "
            "promoted. No failed parameter is retuned, no cross-Knudsen extension is "
            "allowed, and no stable-solver, accuracy, benchmark, or validation claim is authorized."
        ),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage67-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run_stage96(args.stage67_artifact_dir, args.output_dir),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
