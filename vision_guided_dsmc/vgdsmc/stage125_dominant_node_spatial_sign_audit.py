from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

STAGE124_RUN_ID = 31898392223
STAGE124_JOB_ID = 95045137550
STAGE124_ARTIFACT_ID = 9253134830
STAGE124_ARTIFACT_SHA256 = "49da2c8c9b19ceec1eae9b1b0781e51b6c7679f7a448578d33af3e51dadbf09f"
STAGE124_SUMMARY_SHA256 = "2ba40389d7851127fdc823741284fef6d7c07e9857f378fac8f1d45da1245660"
STAGE124_PAYLOAD_SHA256 = "272b3def41e2fa97a929712bcd264b8ca283e296af1cb99ebf414f5ca2164e1f"
STAGE124_SOURCE_HEAD = "8430bf16cb423b6f615eb08dd87867cc1307a9c6"
STAGE124_COMPLETION_COMMIT = "dc3645fb54415df53b68a25adb130069dbc4141c"
STAGE124_DECISION = (
    "stage124_strong_within_band_cancellation_with_radial_node_remainder_"
    "stage125_dominant_node_spatial_sign_audit"
)

GRID = (56, 56)
RADIAL_NODES = 10
BANDS = ("near_1_4", "mid_5_14", "inner_15_28")
HALFSPACES = ("axis0_low", "axis0_high", "axis1_low", "axis1_high")

# Fixed artifact-only guards. These thresholds classify topology; they do not tune the solver.
PARENT_ARRAY_CLOSURE_TOLERANCE = 1.0e-12
NET_SIGN_L1_FRACTION_MIN = 0.75
HALFSPACE_LOCALIZATION_MIN = 0.75
CONNECTED_COMPONENT_L1_FRACTION_MIN = 0.50

LOCALIZED = (
    "stage125_persistent_same_sign_halfspace_localization_"
    "stage126_wall_side_asymmetry_audit"
)
CONNECTED = (
    "stage125_persistent_same_sign_connected_topology_"
    "stage126_component_geometry_audit"
)
DIFFUSE = (
    "stage125_persistent_same_sign_diffuse_"
    "stage126_coordinate_covariance_audit"
)
WEAK = (
    "stage125_weak_same_sign_persistence_"
    "stage126_node_amplitude_covariance_audit"
)
NONFINITE = "stage125_nonfinite_spatial_sign_blocker_without_retuning"
CLOSURE_BLOCKER = "stage125_parent_array_closure_blocker_without_retuning"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_stage125_design(**overrides: object) -> None:
    frozen = {
        "grid": GRID,
        "radial_nodes": RADIAL_NODES,
        "bands": BANDS,
        "parent_array_closure_tolerance": PARENT_ARRAY_CLOSURE_TOLERANCE,
        "net_sign_l1_fraction_min": NET_SIGN_L1_FRACTION_MIN,
        "halfspace_localization_min": HALFSPACE_LOCALIZATION_MIN,
        "connected_component_l1_fraction_min": CONNECTED_COMPONENT_L1_FRACTION_MIN,
        "stage124_run_id": STAGE124_RUN_ID,
        "stage124_job_id": STAGE124_JOB_ID,
        "stage124_artifact_id": STAGE124_ARTIFACT_ID,
    }
    if any(key not in frozen or frozen[key] != value for key, value in overrides.items()):
        raise ValueError(
            "Stage 125 is fixed to the exact completed Stage-124 artifact and preregistered "
            "spatial-sign guards; it may not retune physics, wall/collision/source treatment, "
            "reconstruction, transport, limiter, floors, normalization, source relaxation, "
            "velocity quadrature, failed MUSCL parameters, or diagnostic thresholds"
        )


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(x - y) / max(float(np.linalg.norm(y)), 1.0e-300))


def _load_stage124(root: str | Path, record_path: str | Path):
    root = Path(root)
    expected = {
        "summary.json": STAGE124_SUMMARY_SHA256,
        "within_band_cancellation.npz": STAGE124_PAYLOAD_SHA256,
    }
    for name, digest in expected.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Stage-124 checksum mismatch: {name}")

    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    record = json.loads(Path(record_path).read_text(encoding="utf-8"))
    if summary.get("stage") != 124 or summary.get("decision") != STAGE124_DECISION or summary.get("finite") is not True:
        raise ValueError("Stage-124 artifact does not authorize Stage 125")
    checks = (
        record.get("stage") == 124,
        record.get("decision") == STAGE124_DECISION,
        record.get("source_head") == STAGE124_SOURCE_HEAD,
        record.get("workflow_status") == "completed",
        record.get("workflow_conclusion") == "success",
        record.get("workflow_run_id") == STAGE124_RUN_ID,
        record.get("workflow_job_id") == STAGE124_JOB_ID,
        record.get("artifact_id") == STAGE124_ARTIFACT_ID,
        record.get("artifact_sha256") == STAGE124_ARTIFACT_SHA256,
        record.get("summary_sha256") == STAGE124_SUMMARY_SHA256,
        record.get("within_band_cancellation_sha256") == STAGE124_PAYLOAD_SHA256,
        record.get("tests", {}).get("passed") == 6,
        record.get("tests", {}).get("failed") == 0,
    )
    if not all(checks):
        raise ValueError("Committed Stage-124 provenance does not authorize Stage 125")

    with np.load(root / "within_band_cancellation.npz") as data:
        needed = {
            "amplitude_matched_residual",
            "band_index",
            "pass_mask",
            "node_net_residual",
            "node_abs_residual",
            "node_uncancelled_fraction",
        }
        if not needed.issubset(data.files):
            raise ValueError("Stage-124 payload is incomplete")
        arrays = {name: np.asarray(data[name]).copy() for name in needed}

    residual = np.asarray(arrays["amplitude_matched_residual"], dtype=np.float64)
    band_index = np.asarray(arrays["band_index"], dtype=np.int8)
    passed = np.asarray(arrays["pass_mask"], dtype=bool)
    node_net = np.asarray(arrays["node_net_residual"], dtype=np.float64)
    node_abs = np.asarray(arrays["node_abs_residual"], dtype=np.float64)
    node_unc = np.asarray(arrays["node_uncancelled_fraction"], dtype=np.float64)

    if residual.shape != (*GRID, RADIAL_NODES):
        raise ValueError("Stage-124 residual-field shape mismatch")
    if band_index.shape != GRID or passed.shape != GRID:
        raise ValueError("Stage-124 spatial-map shape mismatch")
    if node_net.shape != (3, RADIAL_NODES) or node_abs.shape != node_net.shape or node_unc.shape != node_net.shape:
        raise ValueError("Stage-124 radial-node summary shape mismatch")
    if not np.isfinite(residual).all() or not np.isfinite(node_net).all() or not np.isfinite(node_abs).all() or not np.isfinite(node_unc).all():
        raise ValueError("Stage-124 spatial-sign inputs are nonfinite")
    if np.any(node_abs < 0.0) or np.any(node_unc < 0.0):
        raise ValueError("Stage-124 radial-node magnitudes are invalid")
    return summary, record, {
        "residual": residual,
        "band_index": band_index,
        "passed": passed,
        "node_net": node_net,
        "node_abs": node_abs,
        "node_unc": node_unc,
    }


def recompute_parent_arrays(residual: np.ndarray, band_index: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = np.asarray(residual, dtype=np.float64)
    bands = np.asarray(band_index)
    if r.shape != (*GRID, RADIAL_NODES) or bands.shape != GRID:
        raise ValueError("Invalid Stage-125 parent payload")
    net = np.zeros((3, RADIAL_NODES), dtype=np.float64)
    absolute = np.zeros_like(net)
    for i in range(3):
        values = r[bands == i]
        if values.size == 0:
            raise ValueError("Stage-125 band is empty")
        net[i] = np.sum(values, axis=0)
        absolute[i] = np.sum(np.abs(values), axis=0)
    unc = np.divide(np.abs(net), absolute, out=np.zeros_like(net), where=absolute > 0.0)
    return net, absolute, unc


def _largest_component_l1_fraction(mask: np.ndarray, weights: np.ndarray) -> tuple[float, int, int]:
    m = np.asarray(mask, dtype=bool)
    w = np.asarray(weights, dtype=np.float64)
    if m.shape != GRID or w.shape != GRID or not np.isfinite(w).all() or np.any(w < 0.0):
        raise ValueError("Invalid Stage-125 component payload")
    total = float(np.sum(w[m]))
    if total <= 0.0:
        return 0.0, 0, 0
    seen = np.zeros(GRID, dtype=bool)
    best_weight = 0.0
    best_cells = 0
    component_count = 0
    for i, j in zip(*np.where(m)):
        if seen[i, j]:
            continue
        component_count += 1
        stack = [(int(i), int(j))]
        seen[i, j] = True
        comp_weight = 0.0
        comp_cells = 0
        while stack:
            x, y = stack.pop()
            comp_weight += float(w[x, y])
            comp_cells += 1
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                xx, yy = x + dx, y + dy
                if 0 <= xx < GRID[0] and 0 <= yy < GRID[1] and m[xx, yy] and not seen[xx, yy]:
                    seen[xx, yy] = True
                    stack.append((xx, yy))
        if comp_weight > best_weight:
            best_weight = comp_weight
            best_cells = comp_cells
    return float(best_weight / total), int(best_cells), int(component_count)


def dominant_node_spatial_metrics(residual_node: np.ndarray, band_mask: np.ndarray) -> dict[str, object]:
    x = np.asarray(residual_node, dtype=np.float64)
    band = np.asarray(band_mask, dtype=bool)
    if x.shape != GRID or band.shape != GRID or not np.isfinite(x).all() or not np.any(band):
        raise ValueError("Invalid Stage-125 dominant-node field")
    values = x[band]
    total_l1 = float(np.sum(np.abs(values)))
    net = float(np.sum(values))
    if total_l1 <= 0.0:
        return {
            "net_residual": 0.0,
            "net_sign": 0,
            "node_uncancelled_fraction": 0.0,
            "net_sign_l1_fraction": 0.0,
            "opposite_sign_l1_fraction": 0.0,
            "net_sign_cell_fraction": 0.0,
            "largest_net_sign_component_l1_fraction": 0.0,
            "largest_net_sign_component_cell_count": 0,
            "net_sign_component_count": 0,
            "dominant_halfspace": HALFSPACES[0],
            "dominant_halfspace_l1_fraction": 0.0,
            "effective_net_sign_support_fraction": 0.0,
            "total_l1": 0.0,
        }

    net_sign = 1 if net > 0.0 else -1 if net < 0.0 else 0
    same = band & ((x * net_sign) > 0.0) if net_sign else np.zeros(GRID, dtype=bool)
    opposite = band & ((x * net_sign) < 0.0) if net_sign else np.zeros(GRID, dtype=bool)
    same_l1 = float(np.sum(np.abs(x[same])))
    opposite_l1 = float(np.sum(np.abs(x[opposite])))
    weight_field = np.abs(x)
    largest_share, largest_cells, component_count = _largest_component_l1_fraction(same, weight_field)

    split0 = GRID[0] // 2
    split1 = GRID[1] // 2
    indices = np.indices(GRID)
    half_masks = (
        indices[0] < split0,
        indices[0] >= split0,
        indices[1] < split1,
        indices[1] >= split1,
    )
    if same_l1 > 0.0:
        half_shares = [float(np.sum(weight_field[same & hm]) / same_l1) for hm in half_masks]
        best_half = int(np.argmax(half_shares))
        same_weights = weight_field[same]
        effective = float((np.sum(same_weights) ** 2) / max(float(np.sum(same_weights ** 2)), 1.0e-300))
    else:
        half_shares = [0.0] * 4
        best_half = 0
        effective = 0.0

    return {
        "net_residual": net,
        "net_sign": net_sign,
        "node_uncancelled_fraction": float(abs(net) / total_l1),
        "net_sign_l1_fraction": float(same_l1 / total_l1),
        "opposite_sign_l1_fraction": float(opposite_l1 / total_l1),
        "net_sign_cell_fraction": float(np.count_nonzero(same) / np.count_nonzero(band)),
        "largest_net_sign_component_l1_fraction": largest_share,
        "largest_net_sign_component_cell_count": largest_cells,
        "net_sign_component_count": component_count,
        "dominant_halfspace": HALFSPACES[best_half],
        "dominant_halfspace_l1_fraction": half_shares[best_half],
        "halfspace_l1_fractions": {name: share for name, share in zip(HALFSPACES, half_shares)},
        "effective_net_sign_support_fraction": float(effective / np.count_nonzero(band)),
        "total_l1": total_l1,
    }


def stage125_decision(
    *,
    finite: bool,
    parent_array_closure: float,
    net_sign_l1_fractions: list[float],
    dominant_halfspace_l1_fractions: list[float],
    largest_component_l1_fractions: list[float],
) -> str:
    if not finite:
        return NONFINITE
    if parent_array_closure > PARENT_ARRAY_CLOSURE_TOLERANCE:
        return CLOSURE_BLOCKER
    if min(net_sign_l1_fractions) < NET_SIGN_L1_FRACTION_MIN:
        return WEAK
    if min(dominant_halfspace_l1_fractions) >= HALFSPACE_LOCALIZATION_MIN:
        return LOCALIZED
    if min(largest_component_l1_fractions) >= CONNECTED_COMPONENT_L1_FRACTION_MIN:
        return CONNECTED
    return DIFFUSE


def run(stage124_dir: str | Path, stage124_record_path: str | Path, output_dir: str | Path, **design: object) -> dict[str, object]:
    validate_stage125_design(**design)
    parent_summary, _, a = _load_stage124(stage124_dir, stage124_record_path)

    net, absolute, unc = recompute_parent_arrays(a["residual"], a["band_index"])
    parent_array_closure = max(
        _rel_l2(net, a["node_net"]),
        _rel_l2(absolute, a["node_abs"]),
        _rel_l2(unc, a["node_unc"]),
    )

    dominant_nodes = np.zeros(3, dtype=np.int16)
    net_signs = np.zeros(3, dtype=np.int8)
    dominant_field = np.zeros(GRID, dtype=np.float64)
    metrics: dict[str, dict[str, object]] = {}
    net_sign_l1_fractions: list[float] = []
    halfspace_fractions: list[float] = []
    component_fractions: list[float] = []

    for i, band in enumerate(BANDS):
        parent_metric = parent_summary["metrics"][band]
        node = int(parent_metric["dominant_uncancelled_radial_node"])
        if node != int(np.argmax(a["node_unc"][i])):
            raise ValueError("Stage-124 dominant-node provenance mismatch")
        mask = a["band_index"] == i
        node_field = a["residual"][:, :, node]
        m = dominant_node_spatial_metrics(node_field, mask)
        if not np.isclose(float(m["node_uncancelled_fraction"]), float(parent_metric["maximum_node_uncancelled_fraction"]), rtol=2e-13, atol=2e-15):
            raise ValueError("Stage-124 dominant-node metric does not reproduce")
        dominant_nodes[i] = node
        net_signs[i] = int(m["net_sign"])
        dominant_field[mask] = node_field[mask]
        metrics[band] = {"dominant_radial_node": node, **m}
        net_sign_l1_fractions.append(float(m["net_sign_l1_fraction"]))
        halfspace_fractions.append(float(m["dominant_halfspace_l1_fraction"]))
        component_fractions.append(float(m["largest_net_sign_component_l1_fraction"]))

    finite = bool(
        np.isfinite(parent_array_closure)
        and np.isfinite(dominant_field).all()
        and all(np.isfinite(x) for x in net_sign_l1_fractions + halfspace_fractions + component_fractions)
    )
    decision = stage125_decision(
        finite=finite,
        parent_array_closure=parent_array_closure,
        net_sign_l1_fractions=net_sign_l1_fractions,
        dominant_halfspace_l1_fractions=halfspace_fractions,
        largest_component_l1_fractions=component_fractions,
    )

    if decision == LOCALIZED:
        scientific_conclusion = (
            "At each Stage-124 band-dominant radial node, at least three quarters of the residual L1 magnitude carries the band-net sign, "
            "and that same-sign magnitude is itself concentrated in one fixed coordinate halfspace. The persistent radial-node remainder is "
            "therefore spatially side-localized rather than a uniform band-wide mechanism. A fixed wall-side/asymmetry audit is justified. "
            "This is artifact attribution only and does not establish limiter causality, MUSCL stability, endpoint convergence, q_av "
            "improvement, benchmark accuracy, or validation."
        )
    elif decision == CONNECTED:
        scientific_conclusion = (
            "The Stage-124 dominant-node remainders retain the band-net sign and form connected same-sign structures, but not a common "
            "coordinate-half localization. A fixed component-geometry audit is justified; no solver parameter is changed."
        )
    elif decision == DIFFUSE:
        scientific_conclusion = (
            "The Stage-124 dominant-node remainders retain the band-net sign but are spatially diffuse under the preregistered halfspace "
            "and connected-component guards. A fixed coordinate-covariance audit is justified; no solver parameter is changed."
        )
    elif decision == WEAK:
        scientific_conclusion = (
            "The Stage-124 dominant-node same-sign remainder is not persistent enough across all bands for a spatial-topology claim. A "
            "fixed node-amplitude covariance audit is justified; no solver parameter is changed."
        )
    else:
        scientific_conclusion = (
            "Stage 125 is blocked by nonfinite data or failure to reproduce the exact Stage-124 radial-node arrays. No spatial "
            "interpretation or parameter change is justified."
        )

    aggregate = {
        "minimum_net_sign_l1_fraction": float(min(net_sign_l1_fractions)),
        "minimum_dominant_halfspace_l1_fraction": float(min(halfspace_fractions)),
        "minimum_largest_component_l1_fraction": float(min(component_fractions)),
        "maximum_parent_array_closure_rel_l2": float(parent_array_closure),
    }
    summary = {
        "stage": 125,
        "parent_stage124": {
            "run_id": STAGE124_RUN_ID,
            "job_id": STAGE124_JOB_ID,
            "artifact_id": STAGE124_ARTIFACT_ID,
            "source_head": STAGE124_SOURCE_HEAD,
            "completion_commit": STAGE124_COMPLETION_COMMIT,
            "decision": parent_summary["decision"],
        },
        "configuration": {
            "grid": list(GRID),
            "radial_nodes": RADIAL_NODES,
            "bands": list(BANDS),
            "net_sign_l1_fraction_min": NET_SIGN_L1_FRACTION_MIN,
            "halfspace_localization_min": HALFSPACE_LOCALIZATION_MIN,
            "connected_component_l1_fraction_min": CONNECTED_COMPONENT_L1_FRACTION_MIN,
            "parent_array_closure_tolerance": PARENT_ARRAY_CLOSURE_TOLERANCE,
            "connectivity": "4-neighbor",
            "halfspaces": list(HALFSPACES),
            "artifact_only": True,
            "solver_rerun": False,
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
            "solver_endpoint_advanced": False,
            "cross_knudsen_extension_permitted": False,
            "benchmark_or_validation_claim_permitted": False,
        },
        "metrics": metrics,
        "aggregate": aggregate,
        "finite": finite,
        "decision": decision,
        "scientific_conclusion": scientific_conclusion,
        "negative_result_guard": (
            "Stage 28 remains a failed MUSCL endpoint and Stage 90 remains nonconverged. No physical, collision/source, floor, wall, "
            "reconstruction, transport, limiter, normalization, source-relaxation, or velocity-quadrature parameter is retuned; "
            "no solver endpoint or cross-Knudsen extension is advanced."
        ),
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(
        output / "dominant_node_spatial_sign.npz",
        dominant_node_residual=dominant_field,
        band_index=a["band_index"],
        pass_mask=a["passed"],
        dominant_nodes=dominant_nodes,
        net_signs=net_signs,
        parent_node_net=net,
        parent_node_abs=absolute,
        parent_node_uncancelled=unc,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 125 fixed dominant-node spatial-sign audit")
    parser.add_argument("--stage124-dir", required=True)
    parser.add_argument("--stage124-record", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = run(args.stage124_dir, args.stage124_record, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
