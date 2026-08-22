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
from .stage97_muscl_correction_spatial_localization_audit import (
    WALL_BAND_CELLS,
    _region_masks,
)
from .stage98_directional_operator_growth_audit import (
    DECOMPOSITION_CLOSURE_TOLERANCE,
    DIAGNOSTIC_STEPS,
    MATERIAL_DIRECTIONAL_GROWTH_RATIO,
    muscl_correction_components,
)
from .stage100_fused_single_run_directional_audit import (
    SAME_RUN_PARENT_MAP_TOLERANCE,
)

STAGE67_RUN_ID = 30991124477
STAGE100_RUN_ID = 31423728782
STAGE100_JOB_ID = 93570374447
STAGE100_ARTIFACT_ID = 9078152795
STAGE100_ARTIFACT_SHA256 = "5ac099facd9356682b897c6b94c887ee90ff136fcf4112fad4e4bbdba7bdf2f3"
STAGE100_SUMMARY_SHA256 = "91d23d59a0d09a1cf34f11cdc2bf9594883aac6b65cf39f3b79b7d308425c85a"
STAGE100_DECISION = "stage100_mixed_directional_growth_stage101_interior_velocity_sector_audit"
BOUNDARY_SLOPE = "zero"
N_ANGULAR_SECTORS = 8
SECTOR_WIDTH = 2.0 * np.pi / N_ANGULAR_SECTORS
SECTOR_CENTER_OFFSET = 0.5 * SECTOR_WIDTH
SECTOR_LABELS = (
    "+x",
    "+x+y",
    "+y",
    "-x+y",
    "-x",
    "-x-y",
    "-y",
    "+x-y",
)
SECTOR_DOMINANCE_SHARE = 2.0 / N_ANGULAR_SECTORS
MATERIAL_SECTOR_GROWTH_RATIO = MATERIAL_DIRECTIONAL_GROWTH_RATIO
SECTOR_PARENT_CLOSURE_TOLERANCE = SAME_RUN_PARENT_MAP_TOLERANCE


def validate_stage101_design(**overrides: object) -> None:
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
        "wall_band_cells": WALL_BAND_CELLS,
        "n_angular_sectors": N_ANGULAR_SECTORS,
        "sector_dominance_share": SECTOR_DOMINANCE_SHARE,
        "material_sector_growth_ratio": MATERIAL_SECTOR_GROWTH_RATIO,
        "sector_parent_closure_tolerance": SECTOR_PARENT_CLOSURE_TOLERANCE,
        "stage67_run_id": STAGE67_RUN_ID,
        "stage100_run_id": STAGE100_RUN_ID,
    }
    if any(key not in frozen or frozen[key] != value for key, value in overrides.items()):
        raise ValueError(
            "Stage 101 is the frozen interior velocity-sector audit authorized by Stage 100. "
            "It may not retune physics, collision/source treatment, clipping or positivity floors, "
            "source relaxation, transport parameters, wall model, limiter, quadrature, normalization, "
            "tolerance, the four-cell interior definition, the 25-step window, or any failed solver parameter."
        )


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(abs(denominator), 1.0e-300))


def velocity_sector_index(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    vx = np.asarray(vx, dtype=np.float64)
    vy = np.asarray(vy, dtype=np.float64)
    if vx.shape != vy.shape or vx.ndim != 1:
        raise ValueError("vx and vy must be one-dimensional arrays with identical shape")
    angle = np.mod(np.arctan2(vy, vx), 2.0 * np.pi)
    shifted = np.mod(angle + SECTOR_CENTER_OFFSET, 2.0 * np.pi)
    index = np.floor(shifted / SECTOR_WIDTH).astype(np.int64)
    if np.any(index < 0) or np.any(index >= N_ANGULAR_SECTORS):
        raise ValueError("velocity-sector partition failed")
    return index


def _load_and_validate_stage100(root: str | Path) -> dict[str, object]:
    summary = json.loads((Path(root) / "summary.json").read_text(encoding="utf-8"))
    if summary.get("stage") != 100 or summary.get("decision") != STAGE100_DECISION:
        raise ValueError("Stage-100 artifact does not authorize the Stage-101 velocity-sector audit")
    cfg = summary.get("configuration", {})
    if not isinstance(cfg, dict):
        raise ValueError("Stage-100 configuration is missing")
    frozen_checks = {
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
    }
    if any(cfg.get(key) != value for key, value in frozen_checks.items()):
        raise ValueError("Stage-100 artifact does not match the frozen Stage-101 parent design")
    if summary.get("finite") is not True or summary.get("executed_steps") != DIAGNOSTIC_STEPS:
        raise ValueError("Stage-100 fused audit did not complete the frozen diagnostic window")
    if float(summary.get("maximum_decomposition_closure_relative_l2", np.inf)) > DECOMPOSITION_CLOSURE_TOLERANCE:
        raise ValueError("Stage-100 decomposition closure failed")
    if float(summary.get("maximum_same_run_parent_map_relative_l2", np.inf)) > SAME_RUN_PARENT_MAP_TOLERANCE:
        raise ValueError("Stage-100 same-run parent-map closure failed")
    return summary


def _sector_metrics_from_term(
    term_net: np.ndarray,
    weight: np.ndarray,
    sector_index: np.ndarray,
    interior: np.ndarray,
    parent_abs_map: np.ndarray,
) -> dict[str, object]:
    term = np.asarray(term_net, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    sectors = np.asarray(sector_index, dtype=np.int64)
    mask = np.asarray(interior, dtype=bool)
    parent = np.asarray(parent_abs_map, dtype=np.float64)
    if term.ndim != 3 or term.shape[:2] != mask.shape or parent.shape != mask.shape:
        raise ValueError("Stage-101 spatial shapes are inconsistent")
    if term.shape[-1] != w.size or sectors.shape != w.shape:
        raise ValueError("Stage-101 velocity shapes are inconsistent")
    if not np.isfinite(term).all() or not np.isfinite(parent).all():
        raise ValueError("Stage-101 term or parent map is nonfinite")

    interior_term = term[mask]
    sector_abs = np.zeros(N_ANGULAR_SECTORS, dtype=np.float64)
    sector_signed = np.zeros(N_ANGULAR_SECTORS, dtype=np.float64)
    for k in range(N_ANGULAR_SECTORS):
        smask = sectors == k
        if not np.any(smask):
            raise ValueError(f"velocity sector {k} is empty")
        block = interior_term[:, smask]
        wk = w[smask][None, :]
        sector_abs[k] = float(np.sum(np.abs(block) * wk))
        sector_signed[k] = float(np.sum(block * wk))

    sector_total = float(np.sum(sector_abs))
    parent_total = float(np.sum(parent[mask]))
    closure = _safe_ratio(abs(sector_total - parent_total), parent_total)
    shares = sector_abs / max(sector_total, 1.0e-300)
    signed_to_abs = np.abs(sector_signed) / np.maximum(sector_abs, 1.0e-300)
    return {
        "sector_weighted_abs": sector_abs,
        "sector_weighted_signed": sector_signed,
        "sector_abs_share": shares,
        "sector_signed_to_abs_ratio": signed_to_abs,
        "interior_parent_weighted_abs": parent_total,
        "sector_parent_closure_relative": closure,
    }


def _history_summary(values: np.ndarray) -> dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    return {
        "first": float(a[0]),
        "final": float(a[-1]),
        "minimum": float(np.min(a)),
        "maximum": float(np.max(a)),
        "final_to_first_ratio": _safe_ratio(float(a[-1]), float(a[0])),
        "maximum_to_first_ratio": _safe_ratio(float(np.max(a)), float(a[0])),
    }


def _sector_summary(histories: dict[str, np.ndarray], distribution: str) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for k, label in enumerate(SECTOR_LABELS):
        abs_hist = histories[f"{distribution}_sector_weighted_abs"][:, k]
        share_hist = histories[f"{distribution}_sector_abs_share"][:, k]
        signed_ratio_hist = histories[f"{distribution}_sector_signed_to_abs_ratio"][:, k]
        signed_hist = histories[f"{distribution}_sector_weighted_signed"][:, k]
        out.append(
            {
                "index": k,
                "label": label,
                "weighted_abs": _history_summary(abs_hist),
                "abs_share": _history_summary(share_hist),
                "weighted_signed": _history_summary(signed_hist),
                "signed_to_abs_ratio": _history_summary(signed_ratio_hist),
            }
        )
    return out


def stage101_decision(
    sector_summary: dict[str, list[dict[str, object]]],
    maximum_sector_parent_closure_relative: float,
    finite: bool,
) -> str:
    if not finite:
        return "stage101_nonfinite_replay_blocker_without_retuning"
    if maximum_sector_parent_closure_relative > SECTOR_PARENT_CLOSURE_TOLERANCE:
        return "stage101_sector_parent_closure_blocker_without_retuning"

    candidates: list[tuple[float, int]] = []
    for k in range(N_ANGULAR_SECTORS):
        final_share = min(
            float(sector_summary[d][k]["abs_share"]["final"]) for d in ("phi", "psi")
        )
        growth = min(
            float(sector_summary[d][k]["weighted_abs"]["final_to_first_ratio"])
            for d in ("phi", "psi")
        )
        if final_share >= SECTOR_DOMINANCE_SHARE and growth >= MATERIAL_SECTOR_GROWTH_RATIO:
            candidates.append((final_share, k))
    if candidates:
        _, k = max(candidates)
        return f"stage101_common_dominant_sector_{k}_stage102_sector_radial_shell_audit"
    return "stage101_diffuse_velocity_sector_growth_stage102_radial_speed_shell_audit"


def run_stage101(
    stage67_artifact_dir: str | Path,
    stage100_artifact_dir: str | Path,
    output_dir: str | Path,
    **design: object,
) -> dict[str, object]:
    validate_stage101_design(**design)
    _validate_stage67(stage67_artifact_dir)
    stage100_summary = _load_and_validate_stage100(stage100_artifact_dir)

    cfg = build_stage58_config()
    quadrature = mapped_polar_quadrature(*RULE, radial_scale=RADIAL_SCALE)
    sectors = velocity_sector_index(quadrature.vx, quadrature.vy)
    if set(np.unique(sectors).tolist()) != set(range(N_ANGULAR_SECTORS)):
        raise ValueError("Stage-101 frozen velocity rule does not populate all eight sectors")
    interior = _region_masks((cfg.ny, cfg.nx), wall_band_cells=WALL_BAND_CELLS)["interior"]

    with np.load(Path(stage67_artifact_dir) / "converged_full_distributions.npz") as saved:
        phi = np.asarray(saved["phi"], dtype=np.float64).copy()
        psi = np.asarray(saved["psi"], dtype=np.float64).copy()
        for name, actual, expected in (
            ("vx", np.asarray(saved["vx"]), np.asarray(quadrature.vx)),
            ("vy", np.asarray(saved["vy"]), np.asarray(quadrature.vy)),
            ("weight", np.asarray(saved["weight"]), np.asarray(quadrature.weight)),
        ):
            if not np.array_equal(actual, expected):
                raise ValueError(f"Stage-67 {name} does not exactly match the frozen Stage-101 quadrature")

    history_lists: dict[str, list[np.ndarray | float]] = {}
    for distribution in ("phi", "psi"):
        history_lists[f"{distribution}_sector_weighted_abs"] = []
        history_lists[f"{distribution}_sector_weighted_signed"] = []
        history_lists[f"{distribution}_sector_abs_share"] = []
        history_lists[f"{distribution}_sector_signed_to_abs_ratio"] = []
        history_lists[f"{distribution}_interior_parent_weighted_abs"] = []
        history_lists[f"{distribution}_sector_parent_closure_relative"] = []

    finite = True
    for _step in range(1, DIAGNOSTIC_STEPS + 1):
        fields = projected_macroscopic(phi, psi, quadrature)
        tau = local_relaxation_time(fields["rho"], fields["T"], cfg, mapping="paper_consistent_c0")
        nu = 1.0 / np.maximum(tau, 1.0e-14)
        dx, dy = 1.0 / cfg.nx, 1.0 / cfg.ny
        ax = np.abs(quadrature.vx) / dx
        ay = np.abs(quadrature.vy) / dy
        denominator = nu[..., None] + ax[None, None, :] + ay[None, None, :]

        _, parent_maps = correction_diagnostic(phi, psi, cfg, quadrature)
        for distribution, field in (("phi", phi), ("psi", psi)):
            corr_x, corr_y = muscl_correction_components(
                field, quadrature.vx, quadrature.vy, dx, dy
            )
            corr_x += corr_y
            del corr_y
            corr_x /= denominator
            metrics = _sector_metrics_from_term(
                corr_x,
                quadrature.weight,
                sectors,
                interior,
                np.asarray(parent_maps[f"{distribution}_cell_correction_m0"], dtype=np.float64),
            )
            del corr_x
            for key, value in metrics.items():
                history_lists[f"{distribution}_{key}"].append(
                    np.asarray(value, dtype=np.float64).copy()
                    if isinstance(value, np.ndarray)
                    else float(value)
                )
        del parent_maps
        gc.collect()

        phi, psi, _ = steady_muscl_iteration_step(
            phi, psi, cfg, quadrature, one_sided_x_boundary=False
        )
        finite = bool(np.isfinite(phi).all() and np.isfinite(psi).all())
        if not finite:
            break
        gc.collect()

    histories = {
        key: np.asarray(values, dtype=np.float64) for key, values in history_lists.items()
    }
    sector_summary = {
        distribution: _sector_summary(histories, distribution)
        for distribution in ("phi", "psi")
    }
    max_closure = max(
        float(np.max(histories[f"{distribution}_sector_parent_closure_relative"]))
        for distribution in ("phi", "psi")
    )
    decision = stage101_decision(sector_summary, max_closure, finite)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "interior_velocity_sector_histories.npz",
        **histories,
        velocity_sector_index=sectors,
        interior_mask=interior,
        vx=np.asarray(quadrature.vx, dtype=np.float64),
        vy=np.asarray(quadrature.vy, dtype=np.float64),
        weight=np.asarray(quadrature.weight, dtype=np.float64),
    )

    result: dict[str, object] = {
        "stage": 101,
        "description": (
            "Frozen 25-step interior velocity-sector localization of the retained zero-boundary-slope "
            "MUSCL correction. The exact Stage-97 four-cell interior is partitioned into eight fixed "
            "45-degree velocity sectors centered on the coordinate axes and diagonals."
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
            "wall_band_cells": WALL_BAND_CELLS,
            "n_angular_sectors": N_ANGULAR_SECTORS,
            "sector_labels": list(SECTOR_LABELS),
            "sector_width_degrees": 360.0 / N_ANGULAR_SECTORS,
            "sector_dominance_share": SECTOR_DOMINANCE_SHARE,
            "material_sector_growth_ratio": MATERIAL_SECTOR_GROWTH_RATIO,
            "sector_parent_closure_tolerance": SECTOR_PARENT_CLOSURE_TOLERANCE,
            "stage67_run_id": STAGE67_RUN_ID,
            "stage100_run_id": STAGE100_RUN_ID,
            "stage100_job_id": STAGE100_JOB_ID,
            "stage100_artifact_id": STAGE100_ARTIFACT_ID,
            "stage100_artifact_sha256": STAGE100_ARTIFACT_SHA256,
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
            "one_sided_boundary_slope_promoted": False,
            "cross_knudsen_extension_permitted": False,
            "validation_claim_permitted": False,
            "solver_endpoint_claim_permitted": False,
        },
        "stage100_authorization": {
            "decision": stage100_summary["decision"],
            "maximum_decomposition_closure_relative_l2": stage100_summary[
                "maximum_decomposition_closure_relative_l2"
            ],
            "maximum_same_run_parent_map_relative_l2": stage100_summary[
                "maximum_same_run_parent_map_relative_l2"
            ],
        },
        "executed_steps": int(histories["phi_sector_weighted_abs"].shape[0]),
        "finite": finite,
        "maximum_sector_parent_closure_relative": float(max_closure),
        "sector_summary": sector_summary,
        "decision": decision,
        "scientific_conclusion": (
            "Stage 101 only localizes which fixed velocity-angle sectors carry the already-observed "
            "interior MUSCL-correction growth. Sector concentration or growth is diagnostic evidence, "
            "not proof of a causal instability mechanism, solver accuracy, or physical validation."
        ),
        "negative_result_guard": (
            "Stage 100 supports same-run directional attribution but does not establish nonlinear MUSCL "
            "stability. Stage 99 remains a negative cross-run reproducibility result, Stage 98 remains a "
            "negative cross-run replay result, Stage 90 remains nonconverged in both reconstruction arms, "
            "Stage 28 remains a failed MUSCL endpoint, and the Stage-89 one-sided boundary slope is not "
            "promoted. No failed parameter is retuned, no cross-Knudsen MUSCL extension is permitted, and "
            "no accuracy, stability, benchmark, or validation claim is authorized."
        ),
    }
    (out / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage67-artifact-dir", required=True)
    parser.add_argument("--stage100-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run_stage101(
                args.stage67_artifact_dir,
                args.stage100_artifact_dir,
                args.output_dir,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
