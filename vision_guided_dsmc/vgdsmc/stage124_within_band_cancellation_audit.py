from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

STAGE123_RUN_ID = 31897076543
STAGE123_JOB_ID = 95041915435
STAGE123_ARTIFACT_ID = 9250136297
STAGE123_ARTIFACT_SHA256 = "299e02034eec88fe9c362d4dc3dd4775f31fc7ffea4e873ec19bb5c91f5df065"
STAGE123_SUMMARY_SHA256 = "cb172427f16dd2ce772be839a9227ffa36ed9e2868e496e27dece97687e0a6fe"
STAGE123_PROFILES_SHA256 = "6536a7ac5bcdf5d872b189f3c5f78414b50b6a85034d1f4004d4a7b4c723f1b7"
STAGE123_SOURCE_HEAD = "cf1dd2b8bdf10a2de003f1b7acaa062332ba6d39"
STAGE123_COMPLETION_COMMIT = "c738cb7c8fd0c2121d460874f4ff6a0ecfb097cf"
STAGE123_DECISION = "stage123_band_ratio_aggregation_only_stage124_within_band_cancellation_audit"

GRID = (56, 56)
RADIAL_NODES = 10
BANDS = ("near_1_4", "mid_5_14", "inner_15_28")

# Preregistered artifact-only guards. Stage 124 changes no solver parameter.
PARENT_PROFILE_CLOSURE_TOLERANCE = 1.0e-12
CELL_MASS_CLOSURE_TOLERANCE = 1.0e-12
STRONG_CANCELLATION_FRACTION_MIN = 0.60
PERSISTENT_NODE_UNCANCELLED_MAX = 0.75

STRONG_WITH_NODE_REMAINDER = (
    "stage124_strong_within_band_cancellation_with_radial_node_remainder_"
    "stage125_dominant_node_spatial_sign_audit"
)
STRONG_BROAD = "stage124_strong_within_band_cancellation_stage125_spatial_sign_topology_audit"
WEAK = "stage124_weak_within_band_cancellation_stage125_amplitude_weighting_covariance_audit"
NONFINITE = "stage124_nonfinite_within_band_cancellation_blocker_without_retuning"
CLOSURE_BLOCKER = "stage124_parent_or_cell_mass_closure_blocker_without_retuning"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_stage124_design(**overrides: object) -> None:
    frozen = {
        "grid": GRID,
        "radial_nodes": RADIAL_NODES,
        "bands": BANDS,
        "parent_profile_closure_tolerance": PARENT_PROFILE_CLOSURE_TOLERANCE,
        "cell_mass_closure_tolerance": CELL_MASS_CLOSURE_TOLERANCE,
        "strong_cancellation_fraction_min": STRONG_CANCELLATION_FRACTION_MIN,
        "persistent_node_uncancelled_max": PERSISTENT_NODE_UNCANCELLED_MAX,
        "stage123_run_id": STAGE123_RUN_ID,
        "stage123_job_id": STAGE123_JOB_ID,
        "stage123_artifact_id": STAGE123_ARTIFACT_ID,
    }
    if any(key not in frozen or frozen[key] != value for key, value in overrides.items()):
        raise ValueError(
            "Stage 124 is fixed to the exact completed Stage-123 artifact and preregistered "
            "cancellation guards; it may not retune physics, wall/collision/source treatment, "
            "reconstruction, transport, floors, normalization, source relaxation, velocity "
            "quadrature, failed MUSCL parameters, or the diagnostic decision thresholds"
        )


def _normalize(profile: np.ndarray) -> np.ndarray:
    x = np.asarray(profile, dtype=np.float64)
    total = float(np.sum(x))
    if x.shape != (RADIAL_NODES,) or not np.isfinite(x).all() or np.any(x < 0.0) or total <= 0.0:
        raise ValueError("Invalid Stage-124 radial profile")
    return x / total


def _load_stage123(root: str | Path, record_path: str | Path):
    root = Path(root)
    expected = {
        "summary.json": STAGE123_SUMMARY_SHA256,
        "cellwise_ratio_persistence.npz": STAGE123_PROFILES_SHA256,
    }
    for name, digest in expected.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Stage-123 checksum mismatch: {name}")

    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    record = json.loads(Path(record_path).read_text(encoding="utf-8"))
    if summary.get("stage") != 123 or summary.get("decision") != STAGE123_DECISION or summary.get("finite") is not True:
        raise ValueError("Stage-123 artifact does not authorize Stage 124")
    checks = (
        record.get("stage") == 123,
        record.get("decision") == STAGE123_DECISION,
        record.get("source_head") == STAGE123_SOURCE_HEAD,
        record.get("workflow_status") == "completed",
        record.get("workflow_conclusion") == "success",
        record.get("workflow_run_id") == STAGE123_RUN_ID,
        record.get("workflow_job_id") == STAGE123_JOB_ID,
        record.get("artifact_id") == STAGE123_ARTIFACT_ID,
        record.get("artifact_sha256") == STAGE123_ARTIFACT_SHA256,
        record.get("summary_sha256") == STAGE123_SUMMARY_SHA256,
        record.get("cellwise_ratio_persistence_sha256") == STAGE123_PROFILES_SHA256,
        record.get("tests", {}).get("passed") == 6,
        record.get("tests", {}).get("failed") == 0,
    )
    if not all(checks):
        raise ValueError("Committed Stage-123 provenance does not authorize Stage 124")

    with np.load(root / "cellwise_ratio_persistence.npz") as data:
        needed = {
            "phi_common_cell_profiles",
            "psi_common_cell_profiles",
            "valid_mask",
            "pass_mask",
            "band_index",
            "loo_templates",
            "parent_phi_common",
            "parent_psi_common",
        }
        if not needed.issubset(data.files):
            raise ValueError("Stage-123 payload is incomplete")
        arrays = {name: np.asarray(data[name]).copy() for name in needed}

    phi = np.asarray(arrays["phi_common_cell_profiles"], dtype=np.float64)
    psi = np.asarray(arrays["psi_common_cell_profiles"], dtype=np.float64)
    valid = np.asarray(arrays["valid_mask"], dtype=bool)
    passed = np.asarray(arrays["pass_mask"], dtype=bool)
    band_index = np.asarray(arrays["band_index"], dtype=np.int8)
    templates = np.asarray(arrays["loo_templates"], dtype=np.float64)
    parent_phi = np.asarray(arrays["parent_phi_common"], dtype=np.float64)
    parent_psi = np.asarray(arrays["parent_psi_common"], dtype=np.float64)

    if phi.shape != (*GRID, RADIAL_NODES) or psi.shape != phi.shape:
        raise ValueError("Stage-123 cellwise profile shape mismatch")
    if valid.shape != GRID or passed.shape != GRID or band_index.shape != GRID:
        raise ValueError("Stage-123 diagnostic-map shape mismatch")
    if templates.shape != (3, RADIAL_NODES) or parent_phi.shape != templates.shape or parent_psi.shape != templates.shape:
        raise ValueError("Stage-123 band-profile shape mismatch")
    if not np.isfinite(phi).all() or not np.isfinite(psi).all() or np.any(phi < 0.0) or np.any(psi < 0.0):
        raise ValueError("Stage-123 cellwise profiles are nonfinite or negative")
    if not np.isfinite(templates).all() or np.any(templates <= 0.0):
        raise ValueError("Stage-123 templates are nonfinite or nonpositive")
    if not np.isfinite(parent_phi).all() or not np.isfinite(parent_psi).all():
        raise ValueError("Stage-123 parent profiles are nonfinite")
    if not bool(np.all(valid)):
        raise ValueError("Stage 124 requires the completed Stage-123 all-valid-cell endpoint")

    arrays = {
        "phi": phi,
        "psi": psi,
        "valid": valid,
        "passed": passed,
        "band_index": band_index,
        "templates": templates,
        "parent_phi": parent_phi,
        "parent_psi": parent_psi,
    }
    return summary, record, arrays


def amplitude_matched_residual(
    phi_cells: np.ndarray,
    psi_cells: np.ndarray,
    template: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    phi = np.asarray(phi_cells, dtype=np.float64)
    psi = np.asarray(psi_cells, dtype=np.float64)
    t = np.asarray(template, dtype=np.float64)
    if phi.ndim != 2 or phi.shape[1] != RADIAL_NODES or psi.shape != phi.shape:
        raise ValueError("Stage-124 cells must be N x 10")
    if t.shape != (RADIAL_NODES,) or not np.isfinite(t).all() or np.any(t <= 0.0):
        raise ValueError("Invalid Stage-124 template")
    if not np.isfinite(phi).all() or not np.isfinite(psi).all() or np.any(phi < 0.0) or np.any(psi < 0.0):
        raise ValueError("Stage-124 cell profiles must be finite and nonnegative")

    phi_total = np.sum(phi, axis=1)
    base = psi * t[None, :]
    base_total = np.sum(base, axis=1)
    if np.any(phi_total <= 0.0) or np.any(base_total <= 0.0):
        raise ValueError("Stage-124 cannot amplitude-match empty cell profiles")
    scale = phi_total / base_total
    prediction = base * scale[:, None]
    residual = phi - prediction
    mass_error = np.abs(np.sum(residual, axis=1)) / np.maximum(phi_total, 1.0e-300)
    max_mass_error = float(np.max(mass_error))
    return residual, scale, max_mass_error


def cancellation_metrics(residual: np.ndarray, passed: np.ndarray) -> dict[str, object]:
    r = np.asarray(residual, dtype=np.float64)
    p = np.asarray(passed, dtype=bool)
    if r.ndim != 2 or r.shape[1] != RADIAL_NODES or p.shape != (r.shape[0],):
        raise ValueError("Invalid Stage-124 cancellation payload")
    if not np.isfinite(r).all():
        raise ValueError("Nonfinite Stage-124 residual")

    cell_l1 = np.sum(np.abs(r), axis=1)
    total_abs = float(np.sum(cell_l1))
    if total_abs <= 0.0:
        node_net = np.zeros(RADIAL_NODES, dtype=np.float64)
        node_abs = np.zeros(RADIAL_NODES, dtype=np.float64)
        node_uncancelled = np.zeros(RADIAL_NODES, dtype=np.float64)
        uncancelled = 0.0
        cancellation = 1.0
    else:
        node_net = np.sum(r, axis=0)
        node_abs = np.sum(np.abs(r), axis=0)
        aggregate_abs = float(np.sum(np.abs(node_net)))
        uncancelled = aggregate_abs / total_abs
        cancellation = 1.0 - uncancelled
        node_uncancelled = np.divide(
            np.abs(node_net),
            node_abs,
            out=np.zeros_like(node_abs),
            where=node_abs > 0.0,
        )

    dominant_node = int(np.argmax(node_uncancelled))
    residual_total = max(float(np.sum(cell_l1)), 1.0e-300)
    return {
        "total_cellwise_residual_l1": total_abs,
        "aggregate_residual_l1": float(np.sum(np.abs(node_net))),
        "uncancelled_fraction": float(uncancelled),
        "cancellation_fraction": float(cancellation),
        "median_node_uncancelled_fraction": float(np.median(node_uncancelled)),
        "p90_node_uncancelled_fraction": float(np.quantile(node_uncancelled, 0.90)),
        "maximum_node_uncancelled_fraction": float(np.max(node_uncancelled)),
        "dominant_uncancelled_radial_node": dominant_node,
        "dominant_node_net_residual_share": float(
            np.abs(node_net[dominant_node]) / max(float(np.sum(np.abs(node_net))), 1.0e-300)
        ),
        "passing_cell_residual_share": float(np.sum(cell_l1[p]) / residual_total),
        "failing_cell_residual_share": float(np.sum(cell_l1[~p]) / residual_total),
        "node_net_residual": node_net,
        "node_abs_residual": node_abs,
        "node_uncancelled_fraction": node_uncancelled,
        "cell_residual_l1": cell_l1,
    }


def stage124_decision(
    *,
    finite: bool,
    parent_closure: float,
    cell_mass_closure: float,
    cancellation_fractions: list[float],
    maximum_node_uncancelled_fraction: float,
) -> str:
    if not finite:
        return NONFINITE
    if parent_closure > PARENT_PROFILE_CLOSURE_TOLERANCE or cell_mass_closure > CELL_MASS_CLOSURE_TOLERANCE:
        return CLOSURE_BLOCKER
    if min(cancellation_fractions) >= STRONG_CANCELLATION_FRACTION_MIN:
        if maximum_node_uncancelled_fraction > PERSISTENT_NODE_UNCANCELLED_MAX:
            return STRONG_WITH_NODE_REMAINDER
        return STRONG_BROAD
    return WEAK


def run(stage123_dir: str | Path, stage123_record_path: str | Path, output_dir: str | Path, **design: object) -> dict[str, object]:
    validate_stage124_design(**design)
    parent_summary, _, a = _load_stage123(stage123_dir, stage123_record_path)

    residual_field = np.zeros_like(a["phi"], dtype=np.float64)
    scale_field = np.ones(GRID, dtype=np.float64)
    node_net_all = np.zeros((3, RADIAL_NODES), dtype=np.float64)
    node_abs_all = np.zeros((3, RADIAL_NODES), dtype=np.float64)
    node_unc_all = np.zeros((3, RADIAL_NODES), dtype=np.float64)
    cell_l1_field = np.zeros(GRID, dtype=np.float64)

    metrics: dict[str, dict[str, float | int]] = {}
    cancellation_fractions: list[float] = []
    parent_closure = 0.0
    cell_mass_closure = 0.0

    for i, band in enumerate(BANDS):
        mask = a["band_index"] == i
        if not np.any(mask):
            raise ValueError(f"Stage-124 band {band} is empty")
        phi_cells = a["phi"][mask]
        psi_cells = a["psi"][mask]
        residual, scale, mass_error = amplitude_matched_residual(phi_cells, psi_cells, a["templates"][i])
        cm = cancellation_metrics(residual, a["passed"][mask])
        cancellation_fractions.append(float(cm["cancellation_fraction"]))
        cell_mass_closure = max(cell_mass_closure, mass_error)

        phi_agg = _normalize(np.sum(phi_cells, axis=0))
        psi_agg = _normalize(np.sum(psi_cells, axis=0))
        phi_closure = float(
            np.linalg.norm(phi_agg - a["parent_phi"][i]) /
            max(float(np.linalg.norm(a["parent_phi"][i])), 1.0e-300)
        )
        psi_closure = float(
            np.linalg.norm(psi_agg - a["parent_psi"][i]) /
            max(float(np.linalg.norm(a["parent_psi"][i])), 1.0e-300)
        )
        band_parent_closure = max(phi_closure, psi_closure)
        parent_closure = max(parent_closure, band_parent_closure)

        residual_field[mask] = residual
        scale_field[mask] = scale
        cell_l1_field[mask] = np.asarray(cm["cell_residual_l1"], dtype=np.float64)
        node_net_all[i] = np.asarray(cm["node_net_residual"], dtype=np.float64)
        node_abs_all[i] = np.asarray(cm["node_abs_residual"], dtype=np.float64)
        node_unc_all[i] = np.asarray(cm["node_uncancelled_fraction"], dtype=np.float64)

        metrics[band] = {
            "cell_count": int(np.count_nonzero(mask)),
            "cancellation_fraction": float(cm["cancellation_fraction"]),
            "uncancelled_fraction": float(cm["uncancelled_fraction"]),
            "total_cellwise_residual_l1": float(cm["total_cellwise_residual_l1"]),
            "aggregate_residual_l1": float(cm["aggregate_residual_l1"]),
            "median_node_uncancelled_fraction": float(cm["median_node_uncancelled_fraction"]),
            "p90_node_uncancelled_fraction": float(cm["p90_node_uncancelled_fraction"]),
            "maximum_node_uncancelled_fraction": float(cm["maximum_node_uncancelled_fraction"]),
            "dominant_uncancelled_radial_node": int(cm["dominant_uncancelled_radial_node"]),
            "dominant_node_net_residual_share": float(cm["dominant_node_net_residual_share"]),
            "passing_cell_residual_share": float(cm["passing_cell_residual_share"]),
            "failing_cell_residual_share": float(cm["failing_cell_residual_share"]),
            "maximum_cell_mass_closure_relative": mass_error,
            "phi_parent_profile_closure_rel_l2": phi_closure,
            "psi_parent_profile_closure_rel_l2": psi_closure,
        }

    finite = bool(
        np.isfinite(residual_field).all()
        and np.isfinite(scale_field).all()
        and np.isfinite(node_unc_all).all()
        and np.isfinite(parent_closure)
        and np.isfinite(cell_mass_closure)
    )
    maximum_node_uncancelled = float(np.max(node_unc_all))
    decision = stage124_decision(
        finite=finite,
        parent_closure=parent_closure,
        cell_mass_closure=cell_mass_closure,
        cancellation_fractions=cancellation_fractions,
        maximum_node_uncancelled_fraction=maximum_node_uncancelled,
    )

    if decision == STRONG_WITH_NODE_REMAINDER:
        scientific_conclusion = (
            "Most Stage-123 cellwise template-shape residual magnitude cancels when cells are aggregated within every fixed wall-distance "
            "band, but at least one radial node retains a large same-sign remainder. The stable Stage-122 band ratio is therefore "
            "substantially aggregation/cancellation-driven rather than a universal cellwise mechanism, while the persistent radial-node "
            "remainder justifies one fixed spatial-sign audit. This is an artifact decomposition only and does not establish limiter "
            "causality, MUSCL stability, endpoint convergence, q_av improvement, benchmark accuracy, or validation."
        )
    elif decision == STRONG_BROAD:
        scientific_conclusion = (
            "Most Stage-123 cellwise template-shape residual magnitude cancels within every fixed wall-distance band without a single "
            "strongly persistent radial-node remainder. A fixed spatial sign-topology audit is justified before any mechanistic claim. "
            "No solver parameter is changed."
        )
    elif decision == WEAK:
        scientific_conclusion = (
            "Within-band sign cancellation is not strong enough to explain the stable Stage-122 aggregate ratio. A fixed amplitude-"
            "weighting covariance audit is justified; no solver parameter is changed."
        )
    else:
        scientific_conclusion = (
            "Stage 124 is blocked by nonfinite data or failure of exact parent/cell-mass closure. No mechanistic interpretation or "
            "parameter change is justified."
        )

    aggregate = {
        "minimum_band_cancellation_fraction": float(min(cancellation_fractions)),
        "maximum_band_uncancelled_fraction": float(max(1.0 - c for c in cancellation_fractions)),
        "maximum_node_uncancelled_fraction": maximum_node_uncancelled,
        "maximum_parent_profile_closure_rel_l2": float(parent_closure),
        "maximum_cell_mass_closure_relative": float(cell_mass_closure),
    }
    configuration = {
        "grid": list(GRID),
        "radial_nodes": RADIAL_NODES,
        "bands": list(BANDS),
        "prediction": "Stage-123 leave-one-band-out template applied cellwise, then amplitude-matched only to each cell phi total",
        "residual": "phi_raw - alpha_cell * psi_raw * template",
        "strong_cancellation_fraction_min": STRONG_CANCELLATION_FRACTION_MIN,
        "persistent_node_uncancelled_max": PERSISTENT_NODE_UNCANCELLED_MAX,
        "parent_profile_closure_tolerance": PARENT_PROFILE_CLOSURE_TOLERANCE,
        "cell_mass_closure_tolerance": CELL_MASS_CLOSURE_TOLERANCE,
        "artifact_only": True,
        "solver_rerun": False,
        "model_retuning": False,
        "wall_retuning": False,
        "source_retuning": False,
        "reconstruction_retuning": False,
        "transport_retuning": False,
        "floor_retuning": False,
        "normalization_retuning": False,
        "velocity_grid_retuning": False,
        "solver_endpoint_advanced": False,
        "cross_knudsen_extension_permitted": False,
        "benchmark_or_validation_claim_permitted": False,
    }
    summary = {
        "stage": 124,
        "parent_stage123": {
            "run_id": STAGE123_RUN_ID,
            "job_id": STAGE123_JOB_ID,
            "artifact_id": STAGE123_ARTIFACT_ID,
            "source_head": STAGE123_SOURCE_HEAD,
            "completion_commit": STAGE123_COMPLETION_COMMIT,
            "decision": parent_summary["decision"],
        },
        "configuration": configuration,
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
        output / "within_band_cancellation.npz",
        amplitude_matched_residual=residual_field,
        amplitude_scale=scale_field,
        cell_residual_l1=cell_l1_field,
        band_index=a["band_index"],
        pass_mask=a["passed"],
        node_net_residual=node_net_all,
        node_abs_residual=node_abs_all,
        node_uncancelled_fraction=node_unc_all,
        loo_templates=a["templates"],
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 124 fixed within-band aggregation/cancellation audit")
    parser.add_argument("--stage123-dir", required=True)
    parser.add_argument("--stage123-record", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = run(args.stage123_dir, args.stage123_record, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
