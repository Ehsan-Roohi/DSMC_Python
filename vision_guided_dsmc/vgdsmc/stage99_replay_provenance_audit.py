from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

STAGE96_RUN_ID = 31338220298
STAGE96_ARTIFACT_ID = 9048777718
STAGE96_ARTIFACT_SHA256 = "33505969b2e43263d9a876d27c6594600fb4eb34858194f31dc89f33668d3b1f"
STAGE97_RUN_ID = 31351436518
STAGE97_ARTIFACT_ID = 9051881261
STAGE97_ARTIFACT_SHA256 = "b90d32d172d0ac34c23c878acc4215bec4c0511a7f4d4d2f38b20984903af643"
STAGE98_PUSH_RUN_ID = 31360755010
STAGE98_PUSH_JOB_ID = 93369020668
STAGE98_PUSH_ARTIFACT_ID = 9058218680
STAGE98_PUSH_ARTIFACT_SHA256 = "9ac24cb7de61f96c25023abeaf296aece1dd946a63c86bb84a5a21539edf0c19"
STAGE98_PR_RUN_ID = 31360757869
STAGE98_PR_JOB_ID = 93369028989
STAGE98_PR_ARTIFACT_ID = 9058393518
STAGE98_PR_ARTIFACT_SHA256 = "9f617c9f26825087ba29ae944081a6a2ab8918d15722286636e900d511be08ae"
STAGE98_SUMMARY_SHA256 = "abf70769ee9b2a206373c98e94d77a814c7368a12f57e166545770e9ece7179e"
STAGE98_HISTORIES_SHA256 = "042a224491e951ff93daffba9d29b7ee9bfdd9bb8f9b5b02037f55c0f441e051"
STAGE98_DECISION = "stage98_decomposition_or_replay_mismatch_blocker_without_retuning"
STAGE97_DECISION = "stage97_interior_dominant_redistribution_stage98_directional_operator_growth_audit"
STAGE96_DECISION = "stage96_material_persistent_muscl_correction_stage97_spatial_localization_audit"
REPLAY_TOLERANCE = 1.0e-12
WALL_BAND_CELLS = 4
GRID = (64, 64)


def validate_stage99_design(**overrides: object) -> None:
    frozen = {
        "grid": GRID,
        "replay_tolerance": REPLAY_TOLERANCE,
        "wall_band_cells": WALL_BAND_CELLS,
        "stage96_run_id": STAGE96_RUN_ID,
        "stage97_run_id": STAGE97_RUN_ID,
        "stage98_push_run_id": STAGE98_PUSH_RUN_ID,
        "stage98_pr_run_id": STAGE98_PR_RUN_ID,
    }
    if any(key not in frozen or frozen[key] != value for key, value in overrides.items()):
        raise ValueError(
            "Stage 99 is an artifact-only provenance audit. It may not retune the strict replay "
            "tolerance, physics, collision/source treatment, clipping or positivity floors, "
            "transport, wall model, limiter, quadrature, normalization, or any failed solver parameter."
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(abs(denominator), 1.0e-300))


def _relative_l2(actual: np.ndarray, reference: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    return _safe_ratio(float(np.linalg.norm(actual - reference)), float(np.linalg.norm(reference)))


def _wall_band_mask(shape: tuple[int, int], cells: int = WALL_BAND_CELLS) -> np.ndarray:
    ny, nx = shape
    mask = np.zeros((ny, nx), dtype=bool)
    mask[:cells, :] = True
    mask[-cells:, :] = True
    mask[:, :cells] = True
    mask[:, -cells:] = True
    return mask


def drift_metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    actual = np.asarray(actual, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if actual.shape != reference.shape or actual.ndim != 2:
        raise ValueError("replay maps must be matching two-dimensional arrays")
    diff = actual - reference
    abs_diff = np.abs(diff)
    total_abs = float(np.sum(abs_diff))
    wall = _wall_band_mask(actual.shape)
    flat = abs_diff.ravel()
    count = max(1, int(np.ceil(0.10 * flat.size)))
    top_sum = float(np.sum(np.partition(flat, flat.size - count)[-count:]))
    max_index = np.unravel_index(int(np.argmax(abs_diff)), abs_diff.shape)
    return {
        "relative_l2": _relative_l2(actual, reference),
        "maximum_absolute": float(np.max(abs_diff)),
        "absolute_difference_sum": total_abs,
        "signed_difference_sum": float(np.sum(diff)),
        "reference_sum": float(np.sum(reference)),
        "actual_sum": float(np.sum(actual)),
        "relative_sum_difference": _safe_ratio(float(np.sum(actual) - np.sum(reference)), float(np.sum(reference))),
        "wall_band_absolute_difference_share": _safe_ratio(float(np.sum(abs_diff[wall])), total_abs),
        "interior_absolute_difference_share": _safe_ratio(float(np.sum(abs_diff[~wall])), total_abs),
        "top_decile_absolute_difference_share": _safe_ratio(top_sum, total_abs),
        "maximum_cell_yx": [int(max_index[0]), int(max_index[1])],
    }


def stage99_decision(
    event_artifacts_identical: bool,
    archive_handoff_max_relative_l2: float,
    first_replay_max_relative_l2: float,
    final_replay_max_relative_l2: float,
) -> str:
    if not event_artifacts_identical:
        return "stage99_push_pr_event_context_mismatch_blocker_without_retuning"
    if archive_handoff_max_relative_l2 > 0.0:
        return "stage99_stage96_stage97_archive_handoff_mismatch_blocker_without_retuning"
    if first_replay_max_relative_l2 <= REPLAY_TOLERANCE and final_replay_max_relative_l2 > REPLAY_TOLERANCE:
        return "stage99_cross_run_iterative_replay_drift_stage100_fused_single_run_directional_audit"
    if final_replay_max_relative_l2 <= REPLAY_TOLERANCE:
        return "stage99_replay_within_strict_tolerance_stage100_interior_velocity_sector_audit"
    return "stage99_replay_mismatch_not_isolated_blocker_without_retuning"


def _load_json(root: Path) -> dict[str, object]:
    return json.loads((root / "summary.json").read_text(encoding="utf-8"))


def _validate_inputs(stage96: Path, stage97: Path, stage98_push: Path, stage98_pr: Path) -> None:
    s96, s97, s98a, s98b = map(_load_json, (stage96, stage97, stage98_push, stage98_pr))
    if s96.get("stage") != 96 or s96.get("decision") != STAGE96_DECISION:
        raise ValueError("Stage-96 artifact is not the retained completed endpoint")
    if s97.get("stage") != 97 or s97.get("decision") != STAGE97_DECISION:
        raise ValueError("Stage-97 artifact is not the retained completed endpoint")
    for summary in (s98a, s98b):
        if summary.get("stage") != 98 or summary.get("decision") != STAGE98_DECISION:
            raise ValueError("Stage-98 artifact does not contain the observed strict replay blocker")


def run_stage99(
    stage96_artifact_dir: str | Path,
    stage97_artifact_dir: str | Path,
    stage98_push_artifact_dir: str | Path,
    stage98_pr_artifact_dir: str | Path,
    output_dir: str | Path,
    **design: object,
) -> dict[str, object]:
    validate_stage99_design(**design)
    p96, p97, p98push, p98pr = map(
        Path,
        (stage96_artifact_dir, stage97_artifact_dir, stage98_push_artifact_dir, stage98_pr_artifact_dir),
    )
    _validate_inputs(p96, p97, p98push, p98pr)

    push_summary_sha = _sha256(p98push / "summary.json")
    pr_summary_sha = _sha256(p98pr / "summary.json")
    push_hist_sha = _sha256(p98push / "directional_operator_growth_histories.npz")
    pr_hist_sha = _sha256(p98pr / "directional_operator_growth_histories.npz")
    event_identical = (
        push_summary_sha == pr_summary_sha == STAGE98_SUMMARY_SHA256
        and push_hist_sha == pr_hist_sha == STAGE98_HISTORIES_SHA256
    )

    with np.load(p96 / "muscl_correction_growth_histories.npz") as d96, \
         np.load(p97 / "spatial_localization_maps.npz") as d97, \
         np.load(p98push / "directional_operator_growth_histories.npz") as d98:
        archive_pairs = {
            "first_phi": (np.asarray(d97["first_phi"]), np.asarray(d96["first_phi_cell_correction_m0"])),
            "final_phi": (np.asarray(d97["final_phi"]), np.asarray(d96["final_phi_cell_correction_m0"])),
            "first_psi": (np.asarray(d97["first_psi"]), np.asarray(d96["first_psi_cell_correction_m0"])),
            "final_psi": (np.asarray(d97["final_psi"]), np.asarray(d96["final_psi_cell_correction_m0"])),
        }
        archive = {
            name: {
                "array_equal": bool(np.array_equal(actual, reference)),
                "relative_l2": _relative_l2(actual, reference),
                "maximum_absolute": float(np.max(np.abs(actual - reference))),
            }
            for name, (actual, reference) in archive_pairs.items()
        }

        replay: dict[str, dict[str, dict[str, object]]] = {"phi": {}, "psi": {}}
        for distribution in ("phi", "psi"):
            for when in ("first", "final"):
                replay[distribution][when] = drift_metrics(
                    np.asarray(d98[f"{when}_{distribution}_net_abs_m0"]),
                    np.asarray(d97[f"{when}_{distribution}"]),
                )

    archive_max = max(float(v["relative_l2"]) for v in archive.values())
    first_max = max(float(replay[d]["first"]["relative_l2"]) for d in ("phi", "psi"))
    final_max = max(float(replay[d]["final"]["relative_l2"]) for d in ("phi", "psi"))
    decision = stage99_decision(event_identical, archive_max, first_max, final_max)

    result: dict[str, object] = {
        "stage": 99,
        "description": (
            "Artifact-only provenance audit of the Stage-98 strict parent-map replay blocker. "
            "It compares exact Stage-96 archived correction maps, the Stage-97 handoff, and both "
            "push- and pull-request-triggered Stage-98 artifacts without rerunning the cavity solver."
        ),
        "configuration": {
            "grid": list(GRID),
            "replay_tolerance": REPLAY_TOLERANCE,
            "wall_band_cells": WALL_BAND_CELLS,
            "stage96_run_id": STAGE96_RUN_ID,
            "stage96_artifact_id": STAGE96_ARTIFACT_ID,
            "stage96_artifact_sha256": STAGE96_ARTIFACT_SHA256,
            "stage97_run_id": STAGE97_RUN_ID,
            "stage97_artifact_id": STAGE97_ARTIFACT_ID,
            "stage97_artifact_sha256": STAGE97_ARTIFACT_SHA256,
            "stage98_push_run_id": STAGE98_PUSH_RUN_ID,
            "stage98_push_job_id": STAGE98_PUSH_JOB_ID,
            "stage98_push_artifact_id": STAGE98_PUSH_ARTIFACT_ID,
            "stage98_push_artifact_sha256": STAGE98_PUSH_ARTIFACT_SHA256,
            "stage98_pr_run_id": STAGE98_PR_RUN_ID,
            "stage98_pr_job_id": STAGE98_PR_JOB_ID,
            "stage98_pr_artifact_id": STAGE98_PR_ARTIFACT_ID,
            "stage98_pr_artifact_sha256": STAGE98_PR_ARTIFACT_SHA256,
            "artifact_only": True,
            "solver_rerun": False,
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
        "stage98_event_reproducibility": {
            "push_and_pr_artifacts_bitwise_identical": event_identical,
            "push_summary_sha256": push_summary_sha,
            "pr_summary_sha256": pr_summary_sha,
            "push_histories_sha256": push_hist_sha,
            "pr_histories_sha256": pr_hist_sha,
        },
        "stage96_to_stage97_archive_handoff": archive,
        "stage98_parent_replay_drift": replay,
        "archive_handoff_max_relative_l2": archive_max,
        "first_replay_max_relative_l2": first_max,
        "final_replay_max_relative_l2": final_max,
        "final_replay_to_strict_tolerance_ratio": _safe_ratio(final_max, REPLAY_TOLERANCE),
        "decision": decision,
        "scientific_conclusion": (
            "This audit distinguishes archived-data integrity from cross-run iterative replay. "
            "A bitwise-stable Stage-96-to-Stage-97 handoff with bitwise-identical push/PR Stage-98 "
            "artifacts, combined with a first-step replay that closes at roundoff but a later replay "
            "that exceeds the unchanged 1e-12 gate, identifies accumulated cross-run floating-point "
            "replay drift rather than event-context divergence or archive corruption. The negative "
            "Stage-98 blocker is retained; the next justified audit must fuse parent-map and "
            "directional diagnostics in one run rather than loosen the reproducibility criterion."
        ),
        "negative_result_guard": (
            "Stage 98 remains blocked by its preregistered 1e-12 parent-map replay gate. Stage 90 "
            "remains nonconverged in both reconstruction arms, Stage 28 remains a failed MUSCL endpoint, "
            "and the Stage-89 one-sided boundary slope is not promoted. No failed physical or numerical "
            "parameter is retuned, no cross-Knudsen extension is allowed, and no accuracy, benchmark, "
            "stability, or validation claim is authorized."
        ),
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage96-artifact-dir", required=True)
    parser.add_argument("--stage97-artifact-dir", required=True)
    parser.add_argument("--stage98-push-artifact-dir", required=True)
    parser.add_argument("--stage98-pr-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run_stage99(
        args.stage96_artifact_dir,
        args.stage97_artifact_dir,
        args.stage98_push_artifact_dir,
        args.stage98_pr_artifact_dir,
        args.output_dir,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
