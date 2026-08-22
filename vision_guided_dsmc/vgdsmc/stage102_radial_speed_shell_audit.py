from __future__ import annotations

import argparse
import gc
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from .stage34_velocity_scale_consistency import local_relaxation_time
from .stage41_projected_polar_operator_audit import (
    STAGE41_CORRECTION_FLOOR,
    mapped_polar_quadrature,
    projected_macroscopic,
)
from .stage58_conservative_solver_64x64_confirmation import build_stage58_config
from .stage79_dominant_moment_radial_angular_gradient_audit import (
    DOMINANT_RADIAL_SHELL_SHARE_GUARD,
    RADIAL_NODES_PER_SHELL,
    RADIAL_SHELL_COUNT,
    TOP_TWO_RADIAL_SHELL_CONCENTRATION_GUARD,
    radial_shell_indices,
)
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
    DIAGNOSTIC_STEPS,
    MATERIAL_DIRECTIONAL_GROWTH_RATIO,
    muscl_correction_components,
)
from .stage101_interior_velocity_sector_audit import (
    BOUNDARY_SLOPE,
    SECTOR_PARENT_CLOSURE_TOLERANCE,
)

STAGE67_RUN_ID = 30991124477
STAGE100_RUN_ID = 31423728782
STAGE101_RUN_ID = 31430526392
STAGE101_JOB_ID = 93592574615
STAGE101_ARTIFACT_ID = 9079226660
STAGE101_ARTIFACT_SHA256 = "3dde191c119623999aa50d2f543d8961990d234bfc550a76d42845ec3c5fa72c"
STAGE101_SUMMARY_SHA256 = "339e33253536621d5570718d0774e70d6518e1c8635b8726c968738826acb0cf"
STAGE101_HISTORIES_SHA256 = "4e701cc74596702abda43590aa4c63fda63734dfd3e7600a81fd3d343c16917e"
STAGE101_DECISION = "stage101_diffuse_velocity_sector_growth_stage102_radial_speed_shell_audit"
MATERIAL_SHELL_GROWTH_RATIO = MATERIAL_DIRECTIONAL_GROWTH_RATIO
SHELL_PARENT_CLOSURE_TOLERANCE = SECTOR_PARENT_CLOSURE_TOLERANCE


def validate_stage102_design(**overrides: object) -> None:
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
        "radial_shell_count": RADIAL_SHELL_COUNT,
        "radial_nodes_per_shell": RADIAL_NODES_PER_SHELL,
        "dominant_radial_shell_share_guard": DOMINANT_RADIAL_SHELL_SHARE_GUARD,
        "top_two_radial_shell_concentration_guard": TOP_TWO_RADIAL_SHELL_CONCENTRATION_GUARD,
        "material_shell_growth_ratio": MATERIAL_SHELL_GROWTH_RATIO,
        "shell_parent_closure_tolerance": SHELL_PARENT_CLOSURE_TOLERANCE,
        "stage67_run_id": STAGE67_RUN_ID,
        "stage100_run_id": STAGE100_RUN_ID,
        "stage101_run_id": STAGE101_RUN_ID,
    }
    if any(key not in frozen or frozen[key] != value for key, value in overrides.items()):
        raise ValueError(
            "Stage 102 is frozen to the exact completed Stage-67/100/101 lineage and four "
            "equal-radial-node shells on the unchanged 40x96 mapped-polar rule. It may not "
            "retune physics, collision/source treatment, clipping or positivity floors, source "
            "relaxation, transport parameters, wall model, limiter, quadrature, normalization, "
            "tolerance, the four-cell interior, the 25-step window, or any failed solver parameter."
        )
    if RULE[0] % RADIAL_SHELL_COUNT != 0:
        raise ValueError("Frozen radial node count must divide exactly into Stage-102 shells")


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(abs(denominator), 1.0e-300))


def _load_and_validate_stage101(root: str | Path) -> dict[str, object]:
    summary = json.loads((Path(root) / "summary.json").read_text(encoding="utf-8"))
    if summary.get("stage") != 101 or summary.get("decision") != STAGE101_DECISION:
        raise ValueError("Stage-101 artifact does not authorize the Stage-102 radial-shell audit")
    cfg = summary.get("configuration", {})
    if not isinstance(cfg, dict):
        raise ValueError("Stage-101 configuration is missing")
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
        "wall_band_cells": WALL_BAND_CELLS,
    }
    if any(cfg.get(key) != value for key, value in frozen_checks.items()):
        raise ValueError("Stage-101 artifact does not match the frozen Stage-102 parent design")
    if summary.get("finite") is not True or summary.get("executed_steps") != DIAGNOSTIC_STEPS:
        raise ValueError("Stage-101 audit did not complete the frozen diagnostic window")
    if float(summary.get("maximum_sector_parent_closure_relative", np.inf)) > SHELL_PARENT_CLOSURE_TOLERANCE:
        raise ValueError("Stage-101 parent closure failed")
    return summary


def _shell_metrics_from_term(
    term_net: np.ndarray,
    weight: np.ndarray,
    shell_index: np.ndarray,
    interior: np.ndarray,
    parent_abs_map: np.ndarray,
) -> dict[str, object]:
    term = np.asarray(term_net, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    shells = np.asarray(shell_index, dtype=np.int64)
    mask = np.asarray(interior, dtype=bool)
    parent = np.asarray(parent_abs_map, dtype=np.float64)
    if term.ndim != 3 or term.shape[:2] != mask.shape or parent.shape != mask.shape:
        raise ValueError("Stage-102 spatial shapes are inconsistent")
    if term.shape[-1] != w.size or shells.shape != w.shape:
        raise ValueError("Stage-102 velocity shapes are inconsistent")
    if not np.isfinite(term).all() or not np.isfinite(parent).all():
        raise ValueError("Stage-102 term or parent map is nonfinite")

    interior_term = term[mask]
    shell_abs = np.zeros(RADIAL_SHELL_COUNT, dtype=np.float64)
    shell_signed = np.zeros(RADIAL_SHELL_COUNT, dtype=np.float64)
    for k in range(RADIAL_SHELL_COUNT):
        smask = shells == k
        if int(np.sum(smask)) != RADIAL_NODES_PER_SHELL * RULE[1]:
            raise ValueError(f"radial shell {k} does not contain the frozen number of velocity points")
        block = interior_term[:, smask]
        wk = w[smask][None, :]
        shell_abs[k] = float(np.sum(np.abs(block) * wk))
        shell_signed[k] = float(np.sum(block * wk))

    shell_total = float(np.sum(shell_abs))
    parent_total = float(np.sum(parent[mask]))
    closure = _safe_ratio(abs(shell_total - parent_total), parent_total)
    return {
        "shell_weighted_abs": shell_abs,
        "shell_weighted_signed": shell_signed,
        "shell_abs_share": shell_abs / max(shell_total, 1.0e-300),
        "shell_signed_to_abs_ratio": np.abs(shell_signed) / np.maximum(shell_abs, 1.0e-300),
        "interior_parent_weighted_abs": parent_total,
        "shell_parent_closure_relative": closure,
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


def _shell_summary(
    histories: dict[str, np.ndarray], distribution: str, shell_index: np.ndarray, speed: np.ndarray
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for k in range(RADIAL_SHELL_COUNT):
        selected = shell_index == k
        out.append(
            {
                "index": k,
                "velocity_point_count": int(np.sum(selected)),
                "minimum_speed": float(np.min(speed[selected])),
                "maximum_speed": float(np.max(speed[selected])),
                "mean_speed": float(np.mean(speed[selected])),
                "weighted_abs": _history_summary(histories[f"{distribution}_shell_weighted_abs"][:, k]),
                "abs_share": _history_summary(histories[f"{distribution}_shell_abs_share"][:, k]),
                "weighted_signed": _history_summary(histories[f"{distribution}_shell_weighted_signed"][:, k]),
                "signed_to_abs_ratio": _history_summary(histories[f"{distribution}_shell_signed_to_abs_ratio"][:, k]),
            }
        )
    return out


def _best_common_pair(histories: dict[str, np.ndarray]) -> dict[str, object]:
    best: dict[str, object] | None = None
    for i, j in combinations(range(RADIAL_SHELL_COUNT), 2):
        per_distribution: dict[str, dict[str, float]] = {}
        for distribution in ("phi", "psi"):
            abs_hist = histories[f"{distribution}_shell_weighted_abs"][:, [i, j]].sum(axis=1)
            share_hist = histories[f"{distribution}_shell_abs_share"][:, [i, j]].sum(axis=1)
            per_distribution[distribution] = {
                "final_share": float(share_hist[-1]),
                "weighted_abs_growth": _safe_ratio(float(abs_hist[-1]), float(abs_hist[0])),
            }
        score = min(per_distribution[d]["final_share"] for d in ("phi", "psi"))
        candidate = {"shells": [i, j], "score": float(score), "distributions": per_distribution}
        if best is None or float(candidate["score"]) > float(best["score"]):
            best = candidate
    assert best is not None
    return best


def stage102_decision(
    shell_summary: dict[str, list[dict[str, object]]],
    histories: dict[str, np.ndarray],
    maximum_shell_parent_closure_relative: float,
    finite: bool,
) -> str:
    if not finite:
        return "stage102_nonfinite_replay_blocker_without_retuning"
    if maximum_shell_parent_closure_relative > SHELL_PARENT_CLOSURE_TOLERANCE:
        return "stage102_shell_parent_closure_blocker_without_retuning"

    candidates: list[tuple[float, int]] = []
    for k in range(RADIAL_SHELL_COUNT):
        final_share = min(float(shell_summary[d][k]["abs_share"]["final"]) for d in ("phi", "psi"))
        growth = min(float(shell_summary[d][k]["weighted_abs"]["final_to_first_ratio"]) for d in ("phi", "psi"))
        if final_share >= DOMINANT_RADIAL_SHELL_SHARE_GUARD and growth >= MATERIAL_SHELL_GROWTH_RATIO:
            candidates.append((final_share, k))
    if candidates:
        _, k = max(candidates)
        return f"stage102_common_dominant_radial_shell_{k}_stage103_shell_spatial_localization_audit"

    pair = _best_common_pair(histories)
    pair_share = min(
        float(pair["distributions"][d]["final_share"]) for d in ("phi", "psi")
    )
    pair_growth = min(
        float(pair["distributions"][d]["weighted_abs_growth"]) for d in ("phi", "psi")
    )
    if pair_share >= TOP_TWO_RADIAL_SHELL_CONCENTRATION_GUARD and pair_growth >= MATERIAL_SHELL_GROWTH_RATIO:
        i, j = pair["shells"]
        return f"stage102_common_top_two_radial_shells_{i}_{j}_stage103_shell_pair_spatial_localization_audit"

    return "stage102_diffuse_radial_growth_stage103_interior_gradient_scale_audit"


def run_stage102(
    stage67_artifact_dir: str | Path,
    stage101_artifact_dir: str | Path,
    output_dir: str | Path,
    **design: object,
) -> dict[str, object]:
    validate_stage102_design(**design)
    _validate_stage67(stage67_artifact_dir)
    stage101_summary = _load_and_validate_stage101(stage101_artifact_dir)

    cfg = build_stage58_config()
    quadrature = mapped_polar_quadrature(*RULE, radial_scale=RADIAL_SCALE)
    shells = radial_shell_indices(quadrature.vx, quadrature.vy).astype(np.int64)
    speed = np.hypot(quadrature.vx, quadrature.vy)
    if set(np.unique(shells).tolist()) != set(range(RADIAL_SHELL_COUNT)):
        raise ValueError("Stage-102 frozen velocity rule does not populate all radial shells")
    for k in range(RADIAL_SHELL_COUNT):
        if int(np.sum(shells == k)) != RADIAL_NODES_PER_SHELL * RULE[1]:
            raise ValueError("Stage-102 radial shell sizes are not frozen")
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
                raise ValueError(f"Stage-67 {name} does not exactly match the frozen Stage-102 quadrature")

    history_lists: dict[str, list[np.ndarray | float]] = {}
    for distribution in ("phi", "psi"):
        for key in (
            "shell_weighted_abs",
            "shell_weighted_signed",
            "shell_abs_share",
            "shell_signed_to_abs_ratio",
            "interior_parent_weighted_abs",
            "shell_parent_closure_relative",
        ):
            history_lists[f"{distribution}_{key}"] = []

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
            corr_x, corr_y = muscl_correction_components(field, quadrature.vx, quadrature.vy, dx, dy)
            corr_x += corr_y
            del corr_y
            corr_x /= denominator
            metrics = _shell_metrics_from_term(
                corr_x,
                quadrature.weight,
                shells,
                interior,
                np.asarray(parent_maps[f"{distribution}_cell_correction_m0"], dtype=np.float64),
            )
            del corr_x
            for key, value in metrics.items():
                history_lists[f"{distribution}_{key}"].append(
                    np.asarray(value, dtype=np.float64).copy() if isinstance(value, np.ndarray) else float(value)
                )
        del parent_maps
        gc.collect()

        phi, psi, _ = steady_muscl_iteration_step(phi, psi, cfg, quadrature, one_sided_x_boundary=False)
        finite = bool(np.isfinite(phi).all() and np.isfinite(psi).all())
        if not finite:
            break
        gc.collect()

    histories = {key: np.asarray(values, dtype=np.float64) for key, values in history_lists.items()}
    shell_summary = {
        distribution: _shell_summary(histories, distribution, shells, speed)
        for distribution in ("phi", "psi")
    }
    max_closure = max(
        float(np.max(histories[f"{distribution}_shell_parent_closure_relative"]))
        for distribution in ("phi", "psi")
    )
    best_pair = _best_common_pair(histories)
    decision = stage102_decision(shell_summary, histories, max_closure, finite)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "interior_radial_speed_shell_histories.npz",
        **histories,
        radial_shell_index=shells,
        interior_mask=interior,
        vx=np.asarray(quadrature.vx, dtype=np.float64),
        vy=np.asarray(quadrature.vy, dtype=np.float64),
        speed=np.asarray(speed, dtype=np.float64),
        weight=np.asarray(quadrature.weight, dtype=np.float64),
    )

    result: dict[str, object] = {
        "stage": 102,
        "description": (
            "Frozen 25-step radial-speed localization of the already-observed interior MUSCL-correction growth. "
            "The unchanged 40 radial nodes are partitioned into four fixed equal-radial-node shells, 10 nodes "
            "per shell and 960 velocity points per shell, without changing the solver or any failed parameter."
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
            "radial_shell_count": RADIAL_SHELL_COUNT,
            "radial_nodes_per_shell": RADIAL_NODES_PER_SHELL,
            "dominant_radial_shell_share_guard": DOMINANT_RADIAL_SHELL_SHARE_GUARD,
            "top_two_radial_shell_concentration_guard": TOP_TWO_RADIAL_SHELL_CONCENTRATION_GUARD,
            "material_shell_growth_ratio": MATERIAL_SHELL_GROWTH_RATIO,
            "shell_parent_closure_tolerance": SHELL_PARENT_CLOSURE_TOLERANCE,
            "stage67_run_id": STAGE67_RUN_ID,
            "stage100_run_id": STAGE100_RUN_ID,
            "stage101_run_id": STAGE101_RUN_ID,
            "stage101_job_id": STAGE101_JOB_ID,
            "stage101_artifact_id": STAGE101_ARTIFACT_ID,
            "stage101_artifact_sha256": STAGE101_ARTIFACT_SHA256,
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
        "stage101_authorization": {
            "decision": stage101_summary["decision"],
            "maximum_sector_parent_closure_relative": stage101_summary["maximum_sector_parent_closure_relative"],
        },
        "executed_steps": int(histories["phi_shell_weighted_abs"].shape[0]),
        "finite": finite,
        "maximum_shell_parent_closure_relative": float(max_closure),
        "shell_summary": shell_summary,
        "best_common_shell_pair": best_pair,
        "decision": decision,
        "scientific_conclusion": (
            "Stage 102 only tests whether the angularly diffuse interior correction growth is concentrated by "
            "speed magnitude in a fixed radial shell or shell pair. Any concentration is diagnostic localization, "
            "not proof of a causal instability mechanism, solver accuracy, or physical validation."
        ),
        "negative_result_guard": (
            "Stage 101 found diffuse velocity-angle growth and did not establish causality. Stage 100 supports "
            "same-run directional attribution but not nonlinear MUSCL stability. Stage 99 remains a negative "
            "cross-run reproducibility result, Stage 98 remains a negative cross-run replay result, Stage 90 "
            "remains nonconverged in both reconstruction arms, Stage 28 remains a failed MUSCL endpoint, and "
            "the Stage-89 one-sided boundary slope is not promoted. No failed parameter is retuned, no cross-"
            "Knudsen MUSCL extension is permitted, and no accuracy, stability, benchmark, or validation claim "
            "is authorized."
        ),
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage67-artifact-dir", required=True)
    parser.add_argument("--stage101-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run_stage102(args.stage67_artifact_dir, args.stage101_artifact_dir, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
