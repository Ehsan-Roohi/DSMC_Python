from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

STAGE127_RUN_ID = 31935671977
STAGE127_JOB_ID = 95137068274
STAGE127_ARTIFACT_ID = 9261546150
STAGE127_ARTIFACT_SHA256 = "3f6c18a98afad27367c0371857c1f4eff64aa1ae9c6391adcc6a16f477abe989"
STAGE127_SUMMARY_SHA256 = "5c25b03efaee7e96add6e5ca46ba4b75d3e776b014c7a43ad374ed53b74a6f42"
STAGE127_PAYLOAD_SHA256 = "cf031274bd6b06deef6898883c0b74f1a410a6d7d974bb1289af9ea19bef9c8e"
STAGE127_SOURCE_HEAD = "c65950ac48c57ab067d404d1cc49c5defcb5a141"
STAGE127_COMPLETION_COMMIT = "f82b311eaab551d0420dd7eb8507539ecbfe02bf"
STAGE127_DECISION = "stage127_bilateral_single_transition_zone_stage128_radial_node_continuity_audit"

STAGE124_RUN_ID = 31898392223
STAGE124_JOB_ID = 95045137550
STAGE124_ARTIFACT_ID = 9253134830
STAGE124_ARTIFACT_SHA256 = "49da2c8c9b19ceec1eae9b1b0781e51b6c7679f7a448578d33af3e51dadbf09f"
STAGE124_SUMMARY_SHA256 = "2ba40389d7851127fdc823741284fef6d7c07e9857f378fac8f1d45da1245660"
STAGE124_PAYLOAD_SHA256 = "272b3def41e2fa97a929712bcd264b8ca283e296af1cb99ebf414f5ca2164e1f"
STAGE124_SOURCE_HEAD = "8430bf16cb423b6f615eb08dd87867cc1307a9c6"
STAGE124_DECISION = "stage124_strong_within_band_cancellation_with_radial_node_remainder_stage125_dominant_node_spatial_sign_audit"

GRID = (56, 56)
DEPTH_COUNT = 28
RADIAL_NODES = 10
BANDS = ("near_1_4", "mid_5_14", "inner_15_28")
WALLS = ("axis1_low", "axis1_high")
TRANSITION_BAND_CODES = (1, 2)

PARENT_PROFILE_CLOSURE_TOLERANCE = 1.0e-12
CROSS_WALL_PROFILE_COSINE_MIN = 0.95
DEPTH_SIDE_AGREEMENT_MIN = 0.90
MAX_PARENT_CROSSING_OFFSET_CELLS = 2.0
EXPECTED_CROSSINGS_PER_WALL = 1

SIGN_CONTINUOUS_REPRODUCTION = (
    "stage128_fixed_node_sign_continuous_transition_reproduced_"
    "stage129_transition_strength_audit"
)
SIGN_CHANGING_REPRODUCTION = (
    "stage128_fixed_node_transition_reproduced_only_with_band_sign_change_"
    "stage129_sign_definition_sensitivity_audit"
)
NO_FIXED_NODE_REPRODUCTION = (
    "stage128_transition_not_reproduced_at_adjacent_fixed_nodes_"
    "stage129_band_node_selection_origin_audit"
)
NONFINITE = "stage128_nonfinite_radial_node_profile_blocker_without_retuning"
CLOSURE_BLOCKER = "stage128_parent_profile_closure_blocker_without_retuning"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_stage128_design(**overrides: object) -> None:
    frozen = {
        "grid": GRID,
        "depth_count": DEPTH_COUNT,
        "radial_nodes": RADIAL_NODES,
        "transition_band_codes": TRANSITION_BAND_CODES,
        "parent_profile_closure_tolerance": PARENT_PROFILE_CLOSURE_TOLERANCE,
        "cross_wall_profile_cosine_min": CROSS_WALL_PROFILE_COSINE_MIN,
        "depth_side_agreement_min": DEPTH_SIDE_AGREEMENT_MIN,
        "max_parent_crossing_offset_cells": MAX_PARENT_CROSSING_OFFSET_CELLS,
        "expected_crossings_per_wall": EXPECTED_CROSSINGS_PER_WALL,
        "stage127_run_id": STAGE127_RUN_ID,
        "stage127_job_id": STAGE127_JOB_ID,
        "stage127_artifact_id": STAGE127_ARTIFACT_ID,
        "stage124_run_id": STAGE124_RUN_ID,
        "stage124_job_id": STAGE124_JOB_ID,
        "stage124_artifact_id": STAGE124_ARTIFACT_ID,
    }
    if any(key not in frozen or frozen[key] != value for key, value in overrides.items()):
        raise ValueError(
            "Stage 128 is fixed to the exact completed Stage-127/124 artifacts, the two "
            "transition-adjacent parent radial nodes, and inherited transition guards; it may "
            "not retune physics, collision/source treatment, walls, reconstruction, transport, "
            "limiter, floors, normalization, source relaxation, velocity quadrature, failed "
            "MUSCL parameters, or diagnostic thresholds"
        )


def _verify_record(record: dict[str, object], *, stage: int, source_head: str, run_id: int,
                   job_id: int, artifact_id: int, artifact_sha: str, decision: str,
                   tests_passed: int) -> None:
    checks = (
        record.get("stage") == stage,
        record.get("source_head") == source_head,
        record.get("workflow_status") == "completed",
        record.get("workflow_conclusion") == "success",
        record.get("workflow_run_id") == run_id,
        record.get("workflow_job_id") == job_id,
        record.get("artifact_id") == artifact_id,
        record.get("artifact_sha256") == artifact_sha,
        record.get("decision") == decision,
        record.get("tests", {}).get("passed") == tests_passed,
        record.get("tests", {}).get("failed") == 0,
    )
    if not all(checks):
        raise ValueError(f"Committed Stage-{stage} provenance does not authorize Stage 128")


def _load_inputs(stage127_dir: str | Path, stage127_record_path: str | Path,
                 stage124_dir: str | Path, stage124_record_path: str | Path):
    s127 = Path(stage127_dir)
    s124 = Path(stage124_dir)
    expected127 = {
        "summary.json": STAGE127_SUMMARY_SHA256,
        "wall_normal_side_switch.npz": STAGE127_PAYLOAD_SHA256,
    }
    expected124 = {
        "summary.json": STAGE124_SUMMARY_SHA256,
        "within_band_cancellation.npz": STAGE124_PAYLOAD_SHA256,
    }
    for root, expected, stage in ((s127, expected127, 127), (s124, expected124, 124)):
        for name, digest in expected.items():
            path = root / name
            if not path.is_file() or sha256_file(path) != digest:
                raise ValueError(f"Stage-{stage} checksum mismatch: {name}")

    summary127 = json.loads((s127 / "summary.json").read_text(encoding="utf-8"))
    summary124 = json.loads((s124 / "summary.json").read_text(encoding="utf-8"))
    record127 = json.loads(Path(stage127_record_path).read_text(encoding="utf-8"))
    record124 = json.loads(Path(stage124_record_path).read_text(encoding="utf-8"))
    if summary127.get("stage") != 127 or summary127.get("decision") != STAGE127_DECISION or summary127.get("finite") is not True:
        raise ValueError("Stage-127 artifact does not authorize Stage 128")
    if summary124.get("stage") != 124 or summary124.get("decision") != STAGE124_DECISION or summary124.get("finite") is not True:
        raise ValueError("Stage-124 artifact does not authorize Stage 128")
    _verify_record(record127, stage=127, source_head=STAGE127_SOURCE_HEAD,
                   run_id=STAGE127_RUN_ID, job_id=STAGE127_JOB_ID,
                   artifact_id=STAGE127_ARTIFACT_ID, artifact_sha=STAGE127_ARTIFACT_SHA256,
                   decision=STAGE127_DECISION, tests_passed=17)
    _verify_record(record124, stage=124, source_head=STAGE124_SOURCE_HEAD,
                   run_id=STAGE124_RUN_ID, job_id=STAGE124_JOB_ID,
                   artifact_id=STAGE124_ARTIFACT_ID, artifact_sha=STAGE124_ARTIFACT_SHA256,
                   decision=STAGE124_DECISION, tests_passed=6)
    if record127.get("summary_sha256") != STAGE127_SUMMARY_SHA256 or record127.get("wall_normal_side_switch_sha256") != STAGE127_PAYLOAD_SHA256:
        raise ValueError("Stage-127 committed payload hashes do not authorize Stage 128")
    if record124.get("summary_sha256") != STAGE124_SUMMARY_SHA256 or record124.get("within_band_cancellation_sha256") != STAGE124_PAYLOAD_SHA256:
        raise ValueError("Stage-124 committed payload hashes do not authorize Stage 128")

    with np.load(s127 / "wall_normal_side_switch.npz") as data:
        needed127 = {
            "depth", "wall0_tangential_asymmetry", "wall1_tangential_asymmetry",
            "parent_dominant_nodes", "parent_net_signs", "parent_band_tangential_side_codes",
        }
        if not needed127.issubset(data.files):
            raise ValueError("Stage-127 payload is incomplete")
        a127 = {name: np.asarray(data[name]).copy() for name in needed127}
    with np.load(s124 / "within_band_cancellation.npz") as data:
        needed124 = {"amplitude_matched_residual", "band_index", "node_net_residual", "node_abs_residual"}
        if not needed124.issubset(data.files):
            raise ValueError("Stage-124 payload is incomplete")
        a124 = {name: np.asarray(data[name]).copy() for name in needed124}

    if a124["amplitude_matched_residual"].shape != GRID + (RADIAL_NODES,):
        raise ValueError("Stage-124 residual tensor shape mismatch")
    if a124["band_index"].shape != GRID or a124["node_net_residual"].shape != (3, RADIAL_NODES):
        raise ValueError("Stage-124 radial-node metadata shape mismatch")
    if a127["parent_dominant_nodes"].shape != (3,) or a127["parent_net_signs"].shape != (3,):
        raise ValueError("Stage-127 parent radial-node metadata shape mismatch")
    return summary127, a127, a124


def _column_for_depth(wall_index: int, depth: int) -> int:
    if wall_index not in (0, 1) or not 1 <= depth <= DEPTH_COUNT:
        raise ValueError("Invalid Stage-128 wall/depth index")
    return depth - 1 if wall_index == 0 else GRID[1] - depth


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    den = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / max(den, 1.0e-300))


def crossing_depths(asymmetry: np.ndarray) -> np.ndarray:
    a = np.asarray(asymmetry, dtype=np.float64)
    if a.shape != (DEPTH_COUNT,) or not np.isfinite(a).all():
        raise ValueError("Invalid Stage-128 asymmetry profile")
    crossings: list[float] = []
    for k in range(DEPTH_COUNT - 1):
        left = float(a[k])
        right = float(a[k + 1])
        if left == 0.0:
            crossings.append(float(k + 1))
        elif left * right < 0.0:
            crossings.append(float(k + 1) + abs(left) / (abs(left) + abs(right)))
    if float(a[-1]) == 0.0:
        crossings.append(float(DEPTH_COUNT))
    if not crossings:
        return np.empty(0, dtype=np.float64)
    out = np.asarray(crossings, dtype=np.float64)
    keep = np.ones(out.size, dtype=bool)
    if out.size > 1:
        keep[1:] = np.diff(out) > 1.0e-12
    return out[keep]


def fixed_node_profile(residual: np.ndarray, band_index: np.ndarray, node_net: np.ndarray,
                       node: int, wall_index: int) -> dict[str, np.ndarray]:
    r = np.asarray(residual, dtype=np.float64)
    bands = np.asarray(band_index, dtype=np.int8)
    net = np.asarray(node_net, dtype=np.float64)
    if r.shape != GRID + (RADIAL_NODES,) or bands.shape != GRID or net.shape != (3, RADIAL_NODES):
        raise ValueError("Invalid Stage-128 fixed-node payload")
    if not 0 <= node < RADIAL_NODES or wall_index not in (0, 1):
        raise ValueError("Invalid Stage-128 fixed node/wall")

    low = np.zeros(DEPTH_COUNT)
    high = np.zeros(DEPTH_COUNT)
    same_l1 = np.zeros(DEPTH_COUNT)
    band_code = np.zeros(DEPTH_COUNT, dtype=np.int8)
    net_sign = np.zeros(DEPTH_COUNT, dtype=np.int8)
    for depth in range(1, DEPTH_COUNT + 1):
        j = _column_for_depth(wall_index, depth)
        unique = np.unique(bands[:, j])
        if unique.size != 1 or int(unique[0]) not in (0, 1, 2):
            raise ValueError("Stage-128 wall-depth line crosses inconsistent bands")
        b = int(unique[0])
        s = 1 if float(net[b, node]) > 0.0 else -1 if float(net[b, node]) < 0.0 else 0
        if s == 0:
            raise ValueError("Stage-128 fixed radial node has zero band-net sign")
        values = r[:, j, node]
        same = values * s > 0.0
        weights = np.abs(values)
        total = float(np.sum(weights[same]))
        if total <= 0.0:
            raise ValueError("Stage-128 fixed-node same-sign support is empty")
        low[depth - 1] = float(np.sum(weights[: GRID[0] // 2][same[: GRID[0] // 2]]) / total)
        high[depth - 1] = float(np.sum(weights[GRID[0] // 2 :][same[GRID[0] // 2 :]]) / total)
        same_l1[depth - 1] = total
        band_code[depth - 1] = b
        net_sign[depth - 1] = s
    asym = high - low
    return {
        "axis0_low_fraction": low,
        "axis0_high_fraction": high,
        "same_sign_l1": same_l1,
        "band_code": band_code,
        "net_sign": net_sign,
        "tangential_asymmetry": asym,
        "dominant_side_code": (asym >= 0.0).astype(np.int8),
    }


def piecewise_parent_profile(residual: np.ndarray, band_index: np.ndarray,
                             parent_nodes: np.ndarray, parent_signs: np.ndarray,
                             wall_index: int) -> np.ndarray:
    r = np.asarray(residual, dtype=np.float64)
    bands = np.asarray(band_index, dtype=np.int8)
    nodes = np.asarray(parent_nodes, dtype=np.int16)
    signs = np.asarray(parent_signs, dtype=np.int8)
    out = np.zeros(DEPTH_COUNT, dtype=np.float64)
    for depth in range(1, DEPTH_COUNT + 1):
        j = _column_for_depth(wall_index, depth)
        b = int(np.unique(bands[:, j])[0])
        node = int(nodes[b])
        s = int(signs[b])
        values = r[:, j, node]
        same = values * s > 0.0
        weights = np.abs(values)
        total = float(np.sum(weights[same]))
        lo = float(np.sum(weights[:28][same[:28]]) / total)
        hi = float(np.sum(weights[28:][same[28:]]) / total)
        out[depth - 1] = hi - lo
    return out


def node_transition_metrics(profile0: dict[str, np.ndarray], profile1: dict[str, np.ndarray],
                            parent_crossings: np.ndarray, node_net: np.ndarray, node: int) -> dict[str, object]:
    a0 = profile0["tangential_asymmetry"]
    a1 = profile1["tangential_asymmetry"]
    c0 = crossing_depths(a0)
    c1 = crossing_depths(a1)
    counts = [int(c0.size), int(c1.size)]
    offsets = [float("inf"), float("inf")]
    if c0.size == 1:
        offsets[0] = abs(float(c0[0]) - float(parent_crossings[0]))
    if c1.size == 1:
        offsets[1] = abs(float(c1[0]) - float(parent_crossings[1]))
    cosine = _cosine(a0, a1)
    agreement = float(np.mean(profile0["dominant_side_code"] == profile1["dominant_side_code"]))
    signs = np.sign(np.asarray(node_net)[list(TRANSITION_BAND_CODES), node]).astype(np.int8)
    sign_continuous = bool(signs[0] == signs[1] and signs[0] != 0)
    reproduced = bool(
        all(c == EXPECTED_CROSSINGS_PER_WALL for c in counts)
        and max(offsets) <= MAX_PARENT_CROSSING_OFFSET_CELLS
        and cosine >= CROSS_WALL_PROFILE_COSINE_MIN
        and agreement >= DEPTH_SIDE_AGREEMENT_MIN
    )
    return {
        "crossing_counts": counts,
        "wall_crossing_depths_cells": [float(c0[0]) if c0.size == 1 else None,
                                        float(c1[0]) if c1.size == 1 else None],
        "parent_crossing_offsets_cells": offsets,
        "maximum_parent_crossing_offset_cells": max(offsets),
        "cross_wall_profile_cosine": cosine,
        "cross_wall_depth_side_agreement_fraction": agreement,
        "mid_inner_net_signs": signs.tolist(),
        "mid_inner_net_sign_continuous": sign_continuous,
        "transition_reproduced": reproduced,
        "minimum_same_sign_l1": float(min(np.min(profile0["same_sign_l1"]), np.min(profile1["same_sign_l1"]))),
    }


def stage128_decision(*, finite: bool, parent_profile_closure: float,
                      reproduced: list[bool], sign_continuous: list[bool]) -> str:
    if not finite:
        return NONFINITE
    if parent_profile_closure > PARENT_PROFILE_CLOSURE_TOLERANCE:
        return CLOSURE_BLOCKER
    if any(r and s for r, s in zip(reproduced, sign_continuous)):
        return SIGN_CONTINUOUS_REPRODUCTION
    if any(reproduced):
        return SIGN_CHANGING_REPRODUCTION
    return NO_FIXED_NODE_REPRODUCTION


def run(stage127_dir: str | Path, stage127_record_path: str | Path,
        stage124_dir: str | Path, stage124_record_path: str | Path,
        output_dir: str | Path, **design: object) -> dict[str, object]:
    validate_stage128_design(**design)
    summary127, a127, a124 = _load_inputs(stage127_dir, stage127_record_path, stage124_dir, stage124_record_path)

    parent_profiles = [np.asarray(a127[f"wall{w}_tangential_asymmetry"], dtype=np.float64) for w in range(2)]
    recomputed = [piecewise_parent_profile(a124["amplitude_matched_residual"], a124["band_index"],
                                           a127["parent_dominant_nodes"], a127["parent_net_signs"], w)
                  for w in range(2)]
    parent_profile_closure = float(max(
        np.linalg.norm(recomputed[w] - parent_profiles[w]) / max(np.linalg.norm(parent_profiles[w]), 1.0e-300)
        for w in range(2)
    ))
    parent_crossings = np.asarray(summary127["aggregate"]["wall_crossing_depths_cells"], dtype=np.float64)
    transition_nodes = [int(a127["parent_dominant_nodes"][b]) for b in TRANSITION_BAND_CODES]
    if transition_nodes[0] == transition_nodes[1]:
        transition_nodes = [transition_nodes[0]]

    metrics: dict[str, object] = {}
    profiles_to_save: dict[str, np.ndarray] = {}
    reproduced: list[bool] = []
    continuous: list[bool] = []
    for node in transition_nodes:
        profiles = [fixed_node_profile(a124["amplitude_matched_residual"], a124["band_index"],
                                       a124["node_net_residual"], node, w) for w in range(2)]
        m = node_transition_metrics(profiles[0], profiles[1], parent_crossings,
                                    a124["node_net_residual"], node)
        metrics[f"radial_node_{node}"] = m
        reproduced.append(bool(m["transition_reproduced"]))
        continuous.append(bool(m["mid_inner_net_sign_continuous"]))
        for w, p in enumerate(profiles):
            profiles_to_save[f"node{node}_wall{w}_asymmetry"] = p["tangential_asymmetry"]
            profiles_to_save[f"node{node}_wall{w}_same_sign_l1"] = p["same_sign_l1"]
            profiles_to_save[f"node{node}_wall{w}_net_sign"] = p["net_sign"]

    finite = bool(np.isfinite(parent_profile_closure) and np.isfinite(parent_crossings).all()
                  and np.isfinite(a124["amplitude_matched_residual"]).all())
    decision = stage128_decision(finite=finite, parent_profile_closure=parent_profile_closure,
                                 reproduced=reproduced, sign_continuous=continuous)
    sign_continuous_witnesses = [node for node, r, s in zip(transition_nodes, reproduced, continuous) if r and s]
    result = {
        "stage": 128,
        "finite": finite,
        "configuration": {
            "artifact_only": True,
            "grid": list(GRID),
            "radial_nodes": RADIAL_NODES,
            "transition_band_codes": list(TRANSITION_BAND_CODES),
            "transition_adjacent_parent_nodes": transition_nodes,
            "parent_profile_closure_tolerance": PARENT_PROFILE_CLOSURE_TOLERANCE,
            "cross_wall_profile_cosine_min": CROSS_WALL_PROFILE_COSINE_MIN,
            "depth_side_agreement_min": DEPTH_SIDE_AGREEMENT_MIN,
            "max_parent_crossing_offset_cells": MAX_PARENT_CROSSING_OFFSET_CELLS,
            "expected_crossings_per_wall": EXPECTED_CROSSINGS_PER_WALL,
            "solver_rerun": False,
            "solver_endpoint_advanced": False,
            "model_retuning": False,
            "wall_retuning": False,
            "source_retuning": False,
            "reconstruction_retuning": False,
            "transport_retuning": False,
            "limiter_retuning": False,
            "floor_retuning": False,
            "normalization_retuning": False,
            "source_relaxation_retuning": False,
            "velocity_grid_retuning": False,
            "cross_knudsen_extension_permitted": False,
            "benchmark_or_validation_claim_permitted": False
        },
        "parent_stage127": {
            "run_id": STAGE127_RUN_ID,
            "job_id": STAGE127_JOB_ID,
            "artifact_id": STAGE127_ARTIFACT_ID,
            "source_head": STAGE127_SOURCE_HEAD,
            "completion_commit": STAGE127_COMPLETION_COMMIT,
            "decision": STAGE127_DECISION
        },
        "supporting_stage124": {
            "run_id": STAGE124_RUN_ID,
            "job_id": STAGE124_JOB_ID,
            "artifact_id": STAGE124_ARTIFACT_ID,
            "source_head": STAGE124_SOURCE_HEAD,
            "decision": STAGE124_DECISION
        },
        "aggregate": {
            "maximum_parent_profile_closure_rel_l2": parent_profile_closure,
            "parent_wall_crossing_depths_cells": parent_crossings.tolist(),
            "transition_adjacent_parent_nodes": transition_nodes,
            "reproducing_fixed_node_count": int(sum(reproduced)),
            "sign_continuous_reproducing_fixed_nodes": sign_continuous_witnesses
        },
        "metrics": metrics,
        "decision": decision,
        "scientific_conclusion": (
            "The Stage-127 wall-normal side switch is tested without changing radial node across the "
            "mid-to-inner transition. A sign-continuous fixed-node witness means the bilateral switch "
            "cannot be attributed solely to the parent change in dominant radial node or to a change "
            "in the sign used for same-sign conditioning. This remains an artifact-level attribution "
            "test and does not establish limiter causality, MUSCL stability, endpoint convergence, "
            "q_av improvement, benchmark accuracy, or validation."
            if sign_continuous_witnesses else
            "The Stage-127 transition is not reproduced by any transition-adjacent fixed radial node "
            "with continuous band-net sign under the inherited guards. The band-level switch therefore "
            "cannot yet be separated from radial-node selection/sign conditioning. No solver or "
            "validation claim is made."
        ),
        "negative_result_guard": (
            "Stage 28 remains a failed MUSCL endpoint and Stage 90 remains nonconverged. No physical, "
            "collision/source, floor, wall, reconstruction, transport, limiter, normalization, "
            "source-relaxation, or velocity-quadrature parameter is retuned; no solver endpoint or "
            "cross-Knudsen extension is advanced."
        )
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(
        out / "radial_node_continuity.npz",
        depth=np.arange(1, DEPTH_COUNT + 1, dtype=np.int16),
        parent_wall0_asymmetry=parent_profiles[0],
        parent_wall1_asymmetry=parent_profiles[1],
        recomputed_parent_wall0_asymmetry=recomputed[0],
        recomputed_parent_wall1_asymmetry=recomputed[1],
        transition_nodes=np.asarray(transition_nodes, dtype=np.int16),
        **profiles_to_save
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-128 fixed radial-node continuity audit")
    parser.add_argument("--stage127-dir", required=True)
    parser.add_argument("--stage127-record", required=True)
    parser.add_argument("--stage124-dir", required=True)
    parser.add_argument("--stage124-record", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.stage127_dir, args.stage127_record, args.stage124_dir,
                         args.stage124_record, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
