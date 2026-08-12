from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "make_mv7_jcp_publication_suite.py"
SPEC = importlib.util.spec_from_file_location("mv7_publication", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
publication = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publication)


def synthetic_summary():
    curves = {}
    noninferiority = {}
    equivalence = {"by_method_budget": {}, "raw_loglog_fit": {"slope": -0.5}}
    for method_index, method in enumerate(publication.METHODS):
        curves[method] = {}
        noninferiority[method] = {}
        equivalence["by_method_budget"][method] = {}
        for budget in publication.BUDGETS:
            value = (0.20 / np.sqrt(budget)) * (1.0 - 0.04 * method_index)
            curves[method][str(budget)] = {
                "mean_composite_nrmse": float(value),
                "training_seed_sd": 0.002 if method in publication.ARCHITECTURES else None,
            }
            ratio = 1.0 if method == "raw" and budget == 10 else 0.75 + 0.08 * method_index
            upper = ratio + 0.05
            noninferiority[method][str(budget)] = {
                "geometric_mean_ratio_to_raw10": ratio,
                "one_sided_95_upper_ratio_to_raw10": upper,
                "noninferior": upper <= 1.10,
            }
            equivalence["by_method_budget"][method][str(budget)] = {
                "empirical_equivalent_budget_over_consumed_budget": max(0.9, 20.0 / budget - method_index)
            }
    return {
        "status": "complete_MV7_JCP_full_budget_matrix",
        "curves": curves,
        "noninferiority_to_raw_budget_10": noninferiority,
        "raw_scaling_and_effective_variance_reduction": equivalence,
        "bias_floor_diagnostics": {
            architecture: {"B10_over_B5_error": 0.95 + 0.08 * index}
            for index, architecture in enumerate(publication.ARCHITECTURES)
        },
        "cost_accounting": {
            "reference_wall_seconds_per_block_including_amortized_burn_in": 1200.0,
            "by_architecture_budget": {
                architecture: {"1": {"training_wall_seconds_mean": 100.0}}
                for architecture in publication.ARCHITECTURES
            },
        },
        "checks": {"all_tasks": True},
    }


def test_metric_figures_render_as_pdf_and_high_resolution_png(tmp_path):
    publication.configure_matplotlib()
    summary = synthetic_summary()
    names = []
    names.extend(publication.sampling_efficiency_figure(tmp_path, summary))
    names.extend(publication.noninferiority_figure(tmp_path, summary))
    names.extend(publication.bias_floor_figure(tmp_path, summary))
    assert len(names) == 6
    assert all((tmp_path / name).is_file() and (tmp_path / name).stat().st_size > 1000 for name in names)


def test_fno_diagnostic_uses_absolute_boundary_and_interior_mse(tmp_path):
    mv7_root = tmp_path / "mv7"
    mv6_root = tmp_path / "mv6"
    rng = np.random.default_rng(20260811)
    for architecture in ("mambairv2_tiny_adapted", "fno_residual_small"):
        for budget in (1, 10):
            for seed in publication.TRAINING_SEEDS:
                root = mv6_root if budget == 1 else mv7_root
                directory = (
                    root / "tasks" / architecture / f"training_seed_{seed}"
                    if budget == 1
                    else root / "tasks" / f"budget_{budget}" / architecture / f"training_seed_{seed}"
                )
                directory.mkdir(parents=True)
                target = rng.normal(size=(4, 2, 16, 16)).astype(np.float32)
                scale = 0.10 if architecture == "fno_residual_small" else 0.04
                prediction = target + scale * rng.normal(size=target.shape).astype(np.float32)
                np.savez_compressed(
                    directory / "predictions.npz",
                    identity_condition=np.asarray(["kn0p1_u400"] * 4),
                    architecture_prediction=prediction,
                    target=target,
                )
    publication.configure_matplotlib()
    names, rows = publication.fno_diagnostic_figure(tmp_path, mv7_root, mv6_root)
    assert len(rows) == 4
    assert all(row["boundary_band_mse"] > 0 and row["interior_mse"] > 0 for row in rows)
    assert all((tmp_path / name).is_file() for name in names)


def test_verified_summary_rejects_incomplete_or_unverified_inputs(tmp_path):
    summary = synthetic_summary()
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (tmp_path / "verification.json").write_text(json.dumps({"decision": "verified"}), encoding="utf-8")
    loaded, verification = publication.load_verified_summary(tmp_path)
    assert loaded["status"] == "complete_MV7_JCP_full_budget_matrix"
    assert verification["decision"] == "verified"
    summary["checks"]["all_tasks"] = False
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    try:
        publication.load_verified_summary(tmp_path)
    except ValueError as exc:
        assert "checks" in str(exc)
    else:
        raise AssertionError("false MV7 check should be rejected")
