from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import stage110_same_sign_slope_asymmetry_audit as s110
from . import stage114_wall_distance_conditioned_velocity_quadrature_audit as s114
from . import stage119_exact_directional_moment_kernel_audit as s119

STAGE124_RUN_ID = 31898392223
STAGE124_JOB_ID = 95045137550
STAGE124_ARTIFACT_ID = 9253134830
STAGE124_ARTIFACT_SHA256 = "49da2c8c9b19ceec1eae9b1b0781e51b6c7679f7a448578d33af3e51dadbf09f"
STAGE124_SUMMARY_SHA256 = "2ba40389d7851127fdc823741284fef6d7c07e9857f378fac8f1d45da1245660"
STAGE124_PAYLOAD_SHA256 = "272b3def41e2fa97a929712bcd264b8ca283e296af1cb99ebf414f5ca2164e1f"
STAGE124_SOURCE_HEAD = "8430bf16cb423b6f615eb08dd87867cc1307a9c6"
STAGE124_DECISION = (
    "stage124_strong_within_band_cancellation_with_radial_node_remainder_"
    "stage125_dominant_node_spatial_sign_audit"
)

STAGE129_RUN_ID = 31954505181
STAGE129_JOB_ID = 95182938788
STAGE129_ARTIFACT_ID = 9268079819
STAGE129_ARTIFACT_SHA256 = "39b87f20ba1333b7d08641b7b147521e2bb83169907a79d7c39c2ae15293cce0"
STAGE129_SUMMARY_SHA256 = "20ac64af784f26355f24e130eecfbf528b9223ea530a8f2794d76c1a10b7eaf8"
STAGE129_PAYLOAD_SHA256 = "94919a8e7059351dbc2e8bca534a678f0a4ffbbbcefbe26bd31e9e20fa721d27"
STAGE129_SOURCE_HEAD = "ac413512364842acd5e145283691ea764f8048c3"
STAGE129_DECISION = "stage129_material_sign_continuous_transition_stage130_fixed_sector_continuity_audit"

GRID = (56, 56)
DEPTH_COUNT = 28
WITNESS_NODE = 9
PAIR_SECTORS = (5, 6)
PARENT_CLOSURE_TOLERANCE = 1.0e-12
SINGLE_SECTOR_CARRIER_MIN = 0.75
COHERENT_PAIR_DELTA_RATIO_MIN = 0.90
CROSS_WALL_DELTA_COSINE_MIN = 0.95

SINGLE = "stage130_common_single_sector_carrier_stage131_fixed_sector_spatial_profile_audit"
PAIR = "stage130_coherent_pair_carrier_stage131_pair_interaction_cancellation_audit"
MIXED = "stage130_wall_specific_sector_carriage_stage131_cross_wall_sector_interaction_audit"
NONFINITE = "stage130_nonfinite_fixed_sector_blocker_without_retuning"
CLOSURE_BLOCKER = "stage130_parent_or_sector_reconstruction_closure_blocker_without_retuning"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_stage130_design(**overrides: object) -> None:
    frozen = {
        "grid": GRID,
        "depth_count": DEPTH_COUNT,
        "witness_node": WITNESS_NODE,
        "pair_sectors": PAIR_SECTORS,
        "parent_closure_tolerance": PARENT_CLOSURE_TOLERANCE,
        "single_sector_carrier_min": SINGLE_SECTOR_CARRIER_MIN,
        "coherent_pair_delta_ratio_min": COHERENT_PAIR_DELTA_RATIO_MIN,
        "cross_wall_delta_cosine_min": CROSS_WALL_DELTA_COSINE_MIN,
        "stage67_run_id": s110.STAGE67_RUN_ID,
        "stage111_run_id": s114.STAGE111_RUN_ID,
        "stage124_run_id": STAGE124_RUN_ID,
        "stage129_run_id": STAGE129_RUN_ID,
        "kn0": s110.KNUDSEN,
        "cold_hot_ratio": s110.COLD_HOT_RATIO,
        "rule": s110.RULE,
        "radial_scale": s110.RADIAL_SCALE,
        "limiter": s110.LIMITER,
        "boundary_slope": s110.BOUNDARY_SLOPE,
        "source_relaxation": s110.SOURCE_RELAXATION,
        "tolerance": s110.TOLERANCE,
        "correction_floor": s110.CORRECTION_FLOOR,
        "diagnostic_steps": s110.DIAGNOSTIC_STEPS,
    }
    if any(key not in frozen or frozen[key] != value for key, value in overrides.items()):
        raise ValueError(
            "Stage 130 is fixed to radial node 9, sectors 5+6, the exact Stage-67/111 kinetic snapshot, "
            "the Stage-124 amplitude-matched residual definition, and the Stage-129 crossing depths. "
            "It may not retune physics, collision/source treatment, walls, reconstruction, transport, limiter, "
            "floors, normalization, source relaxation, velocity quadrature, failed MUSCL parameters, or guards."
        )
    if GRID != (56, 56) or s110.GRID != (64, 64) or s110.RULE != (40, 96):
        raise ValueError("Stage 130 requires the exact retained 64x64 / 56x56 interior and 40x96 velocity design")
    if WITNESS_NODE != 9 or PAIR_SECTORS != (5, 6):
        raise ValueError("Stage 130 requires the preregistered node-9, sectors-5+6 witness")


def _verify_record(record: dict[str, object], *, stage: int) -> None:
    if stage == 124:
        checks = (
            record.get("stage") == 124,
            record.get("source_head") == STAGE124_SOURCE_HEAD,
            record.get("workflow_status") == "completed",
            record.get("workflow_conclusion") == "success",
            record.get("workflow_run_id") == STAGE124_RUN_ID,
            record.get("workflow_job_id") == STAGE124_JOB_ID,
            record.get("artifact_id") == STAGE124_ARTIFACT_ID,
            record.get("artifact_sha256") == STAGE124_ARTIFACT_SHA256,
            record.get("summary_sha256") == STAGE124_SUMMARY_SHA256,
            record.get("within_band_cancellation_sha256") == STAGE124_PAYLOAD_SHA256,
            record.get("decision") == STAGE124_DECISION,
            record.get("tests", {}).get("passed") == 6,
            record.get("tests", {}).get("failed") == 0,
        )
    elif stage == 129:
        checks = (
            record.get("stage") == 129,
            record.get("source_head") == STAGE129_SOURCE_HEAD,
            record.get("workflow_status") == "completed",
            record.get("workflow_conclusion") == "success",
            record.get("workflow_run_id") == STAGE129_RUN_ID,
            record.get("workflow_job_id") == STAGE129_JOB_ID,
            record.get("artifact_id") == STAGE129_ARTIFACT_ID,
            record.get("artifact_sha256") == STAGE129_ARTIFACT_SHA256,
            record.get("summary_sha256") == STAGE129_SUMMARY_SHA256,
            record.get("transition_strength_sha256") == STAGE129_PAYLOAD_SHA256,
            record.get("decision") == STAGE129_DECISION,
            record.get("tests", {}).get("passed") == 17,
            record.get("tests", {}).get("failed") == 0,
        )
    else:
        raise ValueError("Unsupported Stage-130 parent record")
    if not all(checks):
        raise ValueError(f"Committed Stage-{stage} provenance does not authorize Stage 130")


def _load_stage124(root: str | Path, record_path: str | Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    root = Path(root)
    expected = {"summary.json": STAGE124_SUMMARY_SHA256, "within_band_cancellation.npz": STAGE124_PAYLOAD_SHA256}
    for name, digest in expected.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Stage-124 checksum mismatch: {name}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    record = json.loads(Path(record_path).read_text(encoding="utf-8"))
    _verify_record(record, stage=124)
    if summary.get("stage") != 124 or summary.get("decision") != STAGE124_DECISION or summary.get("finite") is not True:
        raise ValueError("Stage-124 artifact does not authorize Stage 130")
    with np.load(root / "within_band_cancellation.npz") as data:
        needed = {"amplitude_matched_residual", "amplitude_scale", "band_index", "node_net_residual", "loo_templates"}
        if not needed.issubset(data.files):
            raise ValueError("Stage-124 payload is incomplete")
        arrays = {name: np.asarray(data[name]).copy() for name in needed}
    if arrays["amplitude_matched_residual"].shape != (*GRID, 10):
        raise ValueError("Stage-124 amplitude-matched residual shape mismatch")
    if arrays["amplitude_scale"].shape != GRID or arrays["band_index"].shape != GRID:
        raise ValueError("Stage-124 spatial metadata shape mismatch")
    if arrays["node_net_residual"].shape != (3, 10) or arrays["loo_templates"].shape != (3, 10):
        raise ValueError("Stage-124 radial metadata shape mismatch")
    for key in ("amplitude_matched_residual", "amplitude_scale", "node_net_residual", "loo_templates"):
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"Stage-124 {key} contains nonfinite values")
    return summary, arrays


def _load_stage129(root: str | Path, record_path: str | Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    root = Path(root)
    expected = {"summary.json": STAGE129_SUMMARY_SHA256, "transition_strength.npz": STAGE129_PAYLOAD_SHA256}
    for name, digest in expected.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Stage-129 checksum mismatch: {name}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    record = json.loads(Path(record_path).read_text(encoding="utf-8"))
    _verify_record(record, stage=129)
    if summary.get("stage") != 129 or summary.get("decision") != STAGE129_DECISION or summary.get("finite") is not True:
        raise ValueError("Stage-129 artifact does not authorize Stage 130")
    with np.load(root / "transition_strength.npz") as data:
        needed = {"depth", "wall0_asymmetry", "wall1_asymmetry", "wall0_same_sign_l1", "wall1_same_sign_l1", "wall0_net_sign", "wall1_net_sign"}
        if not needed.issubset(data.files):
            raise ValueError("Stage-129 payload is incomplete")
        arrays = {name: np.asarray(data[name]).copy() for name in needed}
    for name, value in arrays.items():
        if value.shape != (DEPTH_COUNT,) or not np.isfinite(value).all():
            raise ValueError(f"Stage-129 profile {name} is invalid")
    return summary, arrays


def _build_sector_residual(
    distributions_path: Path,
    maps: dict[str, np.ndarray],
    stage124: dict[str, np.ndarray],
) -> tuple[np.ndarray, float]:
    with np.load(distributions_path) as data:
        vx = np.asarray(data["vx"], dtype=np.float64)
        vy = np.asarray(data["vy"], dtype=np.float64)
        weight = np.asarray(data["weight"], dtype=np.float64)
        shell = s110._radial_shell_indices(vx, vy) == s110.DOMINANT_RADIAL_SHELL
        svx, svy, sw = vx[shell], vy[shell], weight[shell]
        sector = s114.angular_sector_indices(svx, svy)
        node = s119.radial_node_indices_within_shell(svx, svy)
        for sector_id in PAIR_SECTORS:
            if int(np.count_nonzero((node == WITNESS_NODE) & (sector == sector_id))) != 12:
                raise ValueError("Each Stage-130 node-9 sector must contain exactly twelve angular ordinates")
        common_kernel = np.abs(svx)
        per_distribution: dict[str, np.ndarray] = {}
        for name in ("phi", "psi"):
            change = s119._x_same_sign_change_pointwise(np.asarray(data[name], dtype=np.float64)[..., shell])
            growth = np.asarray(maps[f"{name}_growth_amplitude"], dtype=np.float64)
            density = change * sw[None, None, :] * common_kernel[None, None, :] * growth[..., None]
            per_distribution[name] = np.stack(
                [
                    np.sum(density[..., (node == WITNESS_NODE) & (sector == sector_id)], axis=-1)
                    for sector_id in PAIR_SECTORS
                ],
                axis=-1,
            )
    band = np.asarray(stage124["band_index"], dtype=np.int64)
    scale = np.asarray(stage124["amplitude_scale"], dtype=np.float64)
    template = np.asarray(stage124["loo_templates"], dtype=np.float64)[band, WITNESS_NODE]
    sector_residual = per_distribution["phi"] - scale[..., None] * per_distribution["psi"] * template[..., None]
    parent = np.asarray(stage124["amplitude_matched_residual"], dtype=np.float64)[..., WITNESS_NODE]
    diff = sector_residual.sum(axis=-1) - parent
    closure = float(np.linalg.norm(diff) / max(float(np.linalg.norm(parent)), 1.0e-300))
    return sector_residual, closure


def _column_for_depth(wall_index: int, depth: int) -> int:
    if wall_index not in (0, 1) or not 1 <= depth <= DEPTH_COUNT:
        raise ValueError("Invalid Stage-130 wall/depth index")
    return depth - 1 if wall_index == 0 else GRID[1] - depth


def parent_conditioned_sector_profiles(
    parent_residual: np.ndarray,
    sector_residual: np.ndarray,
    band_index: np.ndarray,
    node_net_residual: np.ndarray,
    wall_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parent = np.asarray(parent_residual, dtype=np.float64)
    sectors = np.asarray(sector_residual, dtype=np.float64)
    band = np.asarray(band_index, dtype=np.int64)
    node_net = np.asarray(node_net_residual, dtype=np.float64)
    if parent.shape != GRID or sectors.shape != (*GRID, 2) or band.shape != GRID or node_net.shape != (3, 10):
        raise ValueError("Invalid Stage-130 parent-conditioned sector payload")
    pair_asymmetry = np.zeros(DEPTH_COUNT, dtype=np.float64)
    contribution = np.zeros((DEPTH_COUNT, 2), dtype=np.float64)
    support = np.zeros(DEPTH_COUNT, dtype=np.float64)
    for depth in range(1, DEPTH_COUNT + 1):
        j = _column_for_depth(wall_index, depth)
        unique = np.unique(band[:, j])
        if unique.size != 1 or int(unique[0]) not in (0, 1, 2):
            raise ValueError("Stage-130 wall-distance line crosses inconsistent parent bands")
        b = int(unique[0])
        sign = 1 if float(node_net[b, WITNESS_NODE]) > 0.0 else -1 if float(node_net[b, WITNESS_NODE]) < 0.0 else 0
        if sign == 0:
            raise ValueError("Stage-130 node-9 parent net sign is zero")
        values = parent[:, j]
        same = values * sign > 0.0
        weights = np.abs(values)
        total = float(np.sum(weights[same]))
        if total <= 0.0:
            raise ValueError("Stage-130 parent same-sign support is empty")
        lo = float(np.sum(weights[: GRID[0] // 2][same[: GRID[0] // 2]]))
        hi = float(np.sum(weights[GRID[0] // 2 :][same[GRID[0] // 2 :]]))
        pair_asymmetry[depth - 1] = (hi - lo) / total
        support[depth - 1] = total
        for sector_index in range(2):
            signed_sector = sectors[:, j, sector_index] * sign
            slo = float(np.sum(signed_sector[: GRID[0] // 2][same[: GRID[0] // 2]]))
            shi = float(np.sum(signed_sector[GRID[0] // 2 :][same[GRID[0] // 2 :]]))
            contribution[depth - 1, sector_index] = (shi - slo) / total
    return pair_asymmetry, contribution, support


def bracket_depths(crossing: float) -> tuple[int, int]:
    if not np.isfinite(crossing) or not 1.0 <= crossing <= float(DEPTH_COUNT):
        raise ValueError("Invalid Stage-130 crossing depth")
    before = int(math.floor(crossing))
    after = int(math.ceil(crossing))
    if before == after:
        if after < DEPTH_COUNT:
            after += 1
        elif before > 1:
            before -= 1
        else:
            raise ValueError("Stage-130 crossing cannot be bracketed")
    return before, after


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Invalid Stage-130 cosine payload")
    return float(np.dot(x, y) / max(float(np.linalg.norm(x) * np.linalg.norm(y)), 1.0e-300))


def classify_sector_carriage(
    deltas: np.ndarray,
    pair_deltas: np.ndarray,
    *,
    finite: bool = True,
    closure: float = 0.0,
) -> str:
    d = np.asarray(deltas, dtype=np.float64)
    p = np.asarray(pair_deltas, dtype=np.float64)
    if d.shape != (2, 2) or p.shape != (2,):
        raise ValueError("Stage-130 classification requires two walls by two sectors")
    if not finite or not np.isfinite(d).all() or not np.isfinite(p).all():
        return NONFINITE
    if closure > PARENT_CLOSURE_TOLERANCE:
        return CLOSURE_BLOCKER
    sums = np.sum(np.abs(d), axis=1)
    carrier = np.abs(d) / np.maximum(sums[:, None], 1.0e-300)
    dominant = np.argmax(carrier, axis=1)
    dominant_matches = np.array([d[w, dominant[w]] * p[w] > 0.0 for w in range(2)], dtype=bool)
    if dominant[0] == dominant[1] and np.min(np.max(carrier, axis=1)) >= SINGLE_SECTOR_CARRIER_MIN and bool(np.all(dominant_matches)):
        return SINGLE
    coherence = np.abs(np.sum(d, axis=1)) / np.maximum(sums, 1.0e-300)
    all_delta_match = bool(np.all(d * p[:, None] > 0.0))
    if (
        np.min(coherence) >= COHERENT_PAIR_DELTA_RATIO_MIN
        and all_delta_match
        and _cosine(d[0], d[1]) >= CROSS_WALL_DELTA_COSINE_MIN
    ):
        return PAIR
    return MIXED


def run(
    stage67_dir: str | Path,
    stage111_dir: str | Path,
    stage124_dir: str | Path,
    stage124_record_path: str | Path,
    stage129_dir: str | Path,
    stage129_record_path: str | Path,
    output_dir: str | Path,
    **design: object,
) -> dict[str, object]:
    validate_stage130_design(**design)
    _, distributions = s110._load_stage67(stage67_dir)
    _, maps = s114._load_stage111(stage111_dir)
    _, parent124 = _load_stage124(stage124_dir, stage124_record_path)
    summary129, parent129 = _load_stage129(stage129_dir, stage129_record_path)

    sector_residual, sector_reconstruction_closure = _build_sector_residual(distributions, maps, parent124)
    parent_residual = np.asarray(parent124["amplitude_matched_residual"], dtype=np.float64)[..., WITNESS_NODE]
    profiles: list[np.ndarray] = []
    contributions: list[np.ndarray] = []
    supports: list[np.ndarray] = []
    profile_closure = 0.0
    support_closure = 0.0
    for wall in range(2):
        p, c, s = parent_conditioned_sector_profiles(
            parent_residual,
            sector_residual,
            parent124["band_index"],
            parent124["node_net_residual"],
            wall,
        )
        profiles.append(p)
        contributions.append(c)
        supports.append(s)
        profile_closure = max(profile_closure, float(np.max(np.abs(p - parent129[f"wall{wall}_asymmetry"]))))
        support_closure = max(
            support_closure,
            float(np.linalg.norm(s - parent129[f"wall{wall}_same_sign_l1"]) / max(float(np.linalg.norm(parent129[f"wall{wall}_same_sign_l1"])), 1.0e-300)),
        )
    contributions_array = np.asarray(contributions, dtype=np.float64)
    profiles_array = np.asarray(profiles, dtype=np.float64)
    supports_array = np.asarray(supports, dtype=np.float64)
    additive_closure = float(np.max(np.abs(np.sum(contributions_array, axis=-1) - profiles_array)))
    maximum_closure = max(sector_reconstruction_closure, profile_closure, support_closure, additive_closure)

    wall_names = ("axis1_low", "axis1_high")
    wall_metrics: dict[str, dict[str, object]] = {}
    deltas = np.zeros((2, 2), dtype=np.float64)
    pair_deltas = np.zeros(2, dtype=np.float64)
    for wall, wall_name in enumerate(wall_names):
        crossing = float(summary129["metrics"][wall_name]["crossing_depth_cells"])
        before, after = bracket_depths(crossing)
        before_contribution = contributions_array[wall, before - 1]
        after_contribution = contributions_array[wall, after - 1]
        delta = after_contribution - before_contribution
        pair_before = float(profiles_array[wall, before - 1])
        pair_after = float(profiles_array[wall, after - 1])
        pair_delta = pair_after - pair_before
        abs_sum = float(np.sum(np.abs(delta)))
        carrier = np.abs(delta) / max(abs_sum, 1.0e-300)
        coherence = abs(float(np.sum(delta))) / max(abs_sum, 1.0e-300)
        deltas[wall] = delta
        pair_deltas[wall] = pair_delta
        wall_metrics[wall_name] = {
            "crossing_depth_cells": crossing,
            "before_depth_cells": before,
            "after_depth_cells": after,
            "parent_asymmetry_before": pair_before,
            "parent_asymmetry_after": pair_after,
            "parent_transition_delta": pair_delta,
            "sector5_contribution_before": float(before_contribution[0]),
            "sector5_contribution_after": float(after_contribution[0]),
            "sector5_transition_delta": float(delta[0]),
            "sector5_absolute_delta_carrier_fraction": float(carrier[0]),
            "sector5_delta_sign_matches_parent": bool(delta[0] * pair_delta > 0.0),
            "sector6_contribution_before": float(before_contribution[1]),
            "sector6_contribution_after": float(after_contribution[1]),
            "sector6_transition_delta": float(delta[1]),
            "sector6_absolute_delta_carrier_fraction": float(carrier[1]),
            "sector6_delta_sign_matches_parent": bool(delta[1] * pair_delta > 0.0),
            "dominant_transition_sector": int(PAIR_SECTORS[int(np.argmax(carrier))]),
            "dominant_sector_carrier_fraction": float(np.max(carrier)),
            "pair_transition_coherence_ratio": float(coherence),
            "transition_delta_closure_abs": abs(float(np.sum(delta)) - pair_delta),
        }

    cross_wall_delta_cosine = _cosine(deltas[0], deltas[1])
    finite = bool(
        np.isfinite(sector_residual).all()
        and np.isfinite(contributions_array).all()
        and np.isfinite(deltas).all()
        and np.isfinite(pair_deltas).all()
        and np.isfinite(maximum_closure)
        and np.isfinite(cross_wall_delta_cosine)
    )
    decision = classify_sector_carriage(deltas, pair_deltas, finite=finite, closure=maximum_closure)

    if decision == SINGLE:
        scientific_conclusion = (
            "The Stage-129 bilateral node-9 transition is carried predominantly by the same one of sectors 5 or 6 on both opposite walls under an exact additive, parent-conditioned decomposition. This is a fixed diagnostic localization result only and does not establish limiter causality, MUSCL stability, endpoint convergence, q_av improvement, benchmark accuracy, or validation."
        )
    elif decision == PAIR:
        scientific_conclusion = (
            "The Stage-129 bilateral node-9 transition is carried coherently by sectors 5 and 6 together on both opposite walls, with aligned sector-transition vectors and little within-pair cancellation. This is a fixed diagnostic localization result only and does not establish limiter causality, MUSCL stability, endpoint convergence, q_av improvement, benchmark accuracy, or validation."
        )
    elif decision == MIXED:
        scientific_conclusion = (
            "The Stage-129 bilateral node-9 transition does not reduce to one common sector or one cross-wall coherent sector pair: the fixed sector-transition carriage differs between the two opposite walls and/or contains material within-pair cancellation. The angular support is therefore wall-specific at this resolution. This negative sufficiency result does not establish limiter causality, MUSCL stability, endpoint convergence, q_av improvement, benchmark accuracy, or validation."
        )
    else:
        scientific_conclusion = (
            "Stage 130 is blocked by nonfinite data or failure to reconstruct the exact Stage-124/129 parent quantities. No angular interpretation or parameter change is justified."
        )

    summary = {
        "stage": 130,
        "finite": finite,
        "configuration": {
            "grid": list(s110.GRID),
            "interior_grid": list(GRID),
            "kn0": s110.KNUDSEN,
            "cold_hot_ratio": s110.COLD_HOT_RATIO,
            "rule": list(s110.RULE),
            "radial_scale": s110.RADIAL_SCALE,
            "limiter": s110.LIMITER,
            "boundary_slope": s110.BOUNDARY_SLOPE,
            "source_relaxation": s110.SOURCE_RELAXATION,
            "tolerance": s110.TOLERANCE,
            "correction_floor": s110.CORRECTION_FLOOR,
            "diagnostic_steps": s110.DIAGNOSTIC_STEPS,
            "witness_node": WITNESS_NODE,
            "pair_sectors": list(PAIR_SECTORS),
            "common_kernel": "abs(c_x)",
            "decomposition": "sector contributions to the exact Stage-124 node-9 amplitude-matched residual, conditioned on the Stage-129 parent same-sign mask and normalized by the parent same-sign L1 support",
            "single_sector_carrier_min": SINGLE_SECTOR_CARRIER_MIN,
            "coherent_pair_delta_ratio_min": COHERENT_PAIR_DELTA_RATIO_MIN,
            "cross_wall_delta_cosine_min": CROSS_WALL_DELTA_COSINE_MIN,
            "parent_closure_tolerance": PARENT_CLOSURE_TOLERANCE,
            "solver_rerun": False,
            "solver_endpoint_advanced": False,
            "physical_parameter_retuning": False,
            "wall_retuning": False,
            "collision_source_retuning": False,
            "reconstruction_retuning": False,
            "transport_retuning": False,
            "limiter_retuning": False,
            "floor_retuning": False,
            "normalization_retuning": False,
            "source_relaxation_retuning": False,
            "velocity_grid_retuning": False,
            "cross_knudsen_extension_permitted": False,
            "benchmark_or_validation_claim_permitted": False,
        },
        "parents": {
            "stage67_run_id": s110.STAGE67_RUN_ID,
            "stage111_run_id": s114.STAGE111_RUN_ID,
            "stage124_run_id": STAGE124_RUN_ID,
            "stage124_artifact_id": STAGE124_ARTIFACT_ID,
            "stage129_run_id": STAGE129_RUN_ID,
            "stage129_artifact_id": STAGE129_ARTIFACT_ID,
        },
        "metrics": wall_metrics,
        "aggregate": {
            "sector_residual_parent_relative_l2_closure": sector_reconstruction_closure,
            "maximum_stage129_parent_asymmetry_absolute_closure": profile_closure,
            "maximum_stage129_support_relative_l2_closure": support_closure,
            "maximum_additive_sector_asymmetry_absolute_closure": additive_closure,
            "maximum_parent_closure": maximum_closure,
            "cross_wall_sector_transition_delta_cosine": cross_wall_delta_cosine,
            "minimum_pair_transition_coherence_ratio": float(min(float(wall_metrics[name]["pair_transition_coherence_ratio"]) for name in wall_names)),
            "minimum_dominant_sector_carrier_fraction": float(min(float(wall_metrics[name]["dominant_sector_carrier_fraction"]) for name in wall_names)),
            "same_dominant_transition_sector_on_both_walls": bool(wall_metrics[wall_names[0]]["dominant_transition_sector"] == wall_metrics[wall_names[1]]["dominant_transition_sector"]),
        },
        "decision": decision,
        "scientific_conclusion": scientific_conclusion,
        "negative_result_guard": (
            "Stage 28 remains a failed MUSCL endpoint and Stage 90 remains nonconverged. No physical, collision/source, floor, wall, reconstruction, transport, limiter, normalization, source-relaxation, or velocity-quadrature parameter is retuned; no solver endpoint or cross-Knudsen extension is advanced."
        ),
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(
        output / "fixed_sector_continuity.npz",
        depth=np.arange(1, DEPTH_COUNT + 1, dtype=np.float64),
        wall0_parent_asymmetry=profiles_array[0],
        wall1_parent_asymmetry=profiles_array[1],
        wall0_sector_contributions=contributions_array[0],
        wall1_sector_contributions=contributions_array[1],
        wall0_parent_same_sign_l1=supports_array[0],
        wall1_parent_same_sign_l1=supports_array[1],
        sector_residual_node9=sector_residual,
        parent_residual_node9=parent_residual,
        sector_transition_deltas=deltas,
        parent_transition_deltas=pair_deltas,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 130 fixed sector-continuity audit")
    parser.add_argument("--stage67-dir", required=True)
    parser.add_argument("--stage111-dir", required=True)
    parser.add_argument("--stage124-dir", required=True)
    parser.add_argument("--stage124-record", required=True)
    parser.add_argument("--stage129-dir", required=True)
    parser.add_argument("--stage129-record", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = run(
        args.stage67_dir,
        args.stage111_dir,
        args.stage124_dir,
        args.stage124_record,
        args.stage129_dir,
        args.stage129_record,
        args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
