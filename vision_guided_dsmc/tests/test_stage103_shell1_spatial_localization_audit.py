import json

import numpy as np
import pytest

from vgdsmc import stage103_shell1_spatial_localization_audit as stage103


def test_stage103_design_is_frozen():
    stage103.validate_stage103_design()
    with pytest.raises(ValueError):
        stage103.validate_stage103_design(dominant_radial_shell=0)
    with pytest.raises(ValueError):
        stage103.validate_stage103_design(interior_tile_count_per_axis=8)
    with pytest.raises(ValueError):
        stage103.validate_stage103_design(source_relaxation=0.5)


def test_interior_tiles_are_equal_area_and_exactly_cover_frozen_interior():
    tiles = stage103._interior_tile_index(stage103.GRID)
    assert set(np.unique(tiles).tolist()) == set([-1] + list(range(stage103.INTERIOR_TILE_COUNT)))
    assert int(np.sum(tiles >= 0)) == (stage103.GRID[0] - 2 * stage103.WALL_BAND_CELLS) ** 2
    for k in range(stage103.INTERIOR_TILE_COUNT):
        assert int(np.sum(tiles == k)) == stage103.INTERIOR_TILE_SIZE**2


def test_tile_metrics_close_against_shell_parent_magnitude():
    rng = np.random.default_rng(20260810)
    nv = 12
    term = rng.normal(size=(stage103.GRID[0], stage103.GRID[1], nv))
    weight = 0.5 + rng.random(nv)
    tiles = stage103._interior_tile_index(stage103.GRID)
    metrics = stage103._tile_metrics_from_shell_term(term, weight, tiles)
    direct = np.sum(np.abs(term[tiles >= 0]) * weight[None, :])
    assert np.isclose(metrics["interior_shell_weighted_abs"], direct)
    assert np.isclose(np.sum(metrics["tile_abs_share"]), 1.0)
    assert np.all(metrics["tile_signed_to_abs_ratio"] <= 1.0 + 1.0e-14)


def _make_histories(final_phi, final_psi, growth_phi=None, growth_psi=None):
    if growth_phi is None:
        growth_phi = np.full(stage103.INTERIOR_TILE_COUNT, 2.5)
    if growth_psi is None:
        growth_psi = np.full(stage103.INTERIOR_TILE_COUNT, 2.5)
    out = {}
    for distribution, final, growth in (("phi", np.asarray(final_phi, dtype=float), np.asarray(growth_phi, dtype=float)), ("psi", np.asarray(final_psi, dtype=float), np.asarray(growth_psi, dtype=float))):
        first_share = np.full(stage103.INTERIOR_TILE_COUNT, 1.0 / stage103.INTERIOR_TILE_COUNT)
        out[f"{distribution}_tile_abs_share"] = np.vstack([first_share, final])
        out[f"{distribution}_tile_weighted_abs"] = np.vstack([np.ones(stage103.INTERIOR_TILE_COUNT), growth])
        out[f"{distribution}_tile_weighted_signed"] = np.zeros((2, stage103.INTERIOR_TILE_COUNT))
        out[f"{distribution}_tile_signed_to_abs_ratio"] = np.zeros((2, stage103.INTERIOR_TILE_COUNT))
    return out


def _tile_summary_from_histories(histories):
    return {d: stage103._tile_summary(histories, d) for d in ("phi", "psi")}


def test_best_common_contiguous_2x2_finds_expected_block():
    phi = np.full(stage103.INTERIOR_TILE_COUNT, 0.5 / 12.0)
    psi = phi.copy()
    block = [5, 6, 9, 10]
    phi[block] = 0.125
    psi[block] = 0.125
    histories = _make_histories(phi, psi)
    best = stage103._best_common_contiguous_2x2(histories)
    assert best["top_left_row"] == 1
    assert best["top_left_column"] == 1
    assert best["tiles"] == block
    assert np.isclose(best["score"], 0.5)


def test_stage103_decision_accepts_common_localized_single_tile():
    phi = np.full(stage103.INTERIOR_TILE_COUNT, 0.75 / 15.0)
    psi = phi.copy()
    phi[10] = 0.25
    psi[10] = 0.25
    histories = _make_histories(phi, psi)
    summary = _tile_summary_from_histories(histories)
    assert stage103.stage103_decision(summary, histories, 0.0, True) == "stage103_common_localized_tile_10_stage104_local_spatial_gradient_audit"


def test_stage103_decision_routes_common_contiguous_block():
    phi = np.full(stage103.INTERIOR_TILE_COUNT, 0.5 / 12.0)
    psi = phi.copy()
    block = [0, 1, 4, 5]
    phi[block] = 0.125
    psi[block] = 0.125
    histories = _make_histories(phi, psi)
    summary = _tile_summary_from_histories(histories)
    assert stage103.stage103_decision(summary, histories, 0.0, True) == "stage103_common_localized_contiguous_2x2_0_0_stage104_local_spatial_gradient_audit"


def test_stage103_decision_routes_spatially_diffuse_growth():
    uniform = np.full(stage103.INTERIOR_TILE_COUNT, 1.0 / stage103.INTERIOR_TILE_COUNT)
    histories = _make_histories(uniform, uniform)
    summary = _tile_summary_from_histories(histories)
    assert stage103.stage103_decision(summary, histories, 0.0, True) == "stage103_spatially_diffuse_shell1_growth_stage104_interior_gradient_scale_audit"


def test_stage103_decision_preserves_nonfinite_and_closure_blockers():
    uniform = np.full(stage103.INTERIOR_TILE_COUNT, 1.0 / stage103.INTERIOR_TILE_COUNT)
    histories = _make_histories(uniform, uniform)
    summary = _tile_summary_from_histories(histories)
    assert stage103.stage103_decision(summary, histories, 0.0, False) == "stage103_nonfinite_replay_blocker_without_retuning"
    assert stage103.stage103_decision(summary, histories, 2.0e-12, True) == "stage103_stage102_shell_history_closure_blocker_without_retuning"


def test_stage102_authorization_requires_exact_dominant_shell_endpoint(tmp_path):
    shell_records = []
    for k in range(stage103.RADIAL_SHELL_COUNT):
        shell_records.append({"index": k, "abs_share": {"final": 0.70 if k == stage103.DOMINANT_RADIAL_SHELL else 0.10}, "weighted_abs": {"final_to_first_ratio": 3.0 if k == stage103.DOMINANT_RADIAL_SHELL else 1.0}})
    summary = {"stage": 102, "decision": stage103.STAGE102_DECISION, "configuration": {"grid": list(stage103.GRID), "kn0": stage103.KNUDSEN, "cold_hot_ratio": stage103.COLD_HOT_RATIO, "rule": list(stage103.RULE), "radial_scale": stage103.RADIAL_SCALE, "limiter": stage103.LIMITER, "boundary_slope": stage103.BOUNDARY_SLOPE, "source_relaxation": stage103.SOURCE_RELAXATION, "tolerance": stage103.TOLERANCE, "correction_floor": stage103.STAGE41_CORRECTION_FLOOR, "diagnostic_steps": stage103.DIAGNOSTIC_STEPS, "wall_band_cells": stage103.WALL_BAND_CELLS, "radial_shell_count": stage103.RADIAL_SHELL_COUNT, "radial_nodes_per_shell": stage103.RADIAL_NODES_PER_SHELL}, "finite": True, "executed_steps": stage103.DIAGNOSTIC_STEPS, "maximum_shell_parent_closure_relative": 1.0e-16, "shell_summary": {"phi": shell_records, "psi": shell_records}}
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    histories = np.ones((stage103.DIAGNOSTIC_STEPS, stage103.RADIAL_SHELL_COUNT))
    np.savez_compressed(tmp_path / "interior_radial_speed_shell_histories.npz", phi_shell_weighted_abs=histories, psi_shell_weighted_abs=histories)
    loaded, loaded_histories = stage103._load_and_validate_stage102(tmp_path)
    assert loaded["decision"] == stage103.STAGE102_DECISION
    assert loaded_histories["phi_shell_weighted_abs"].shape == histories.shape
    summary["decision"] = "wrong"
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError):
        stage103._load_and_validate_stage102(tmp_path)
