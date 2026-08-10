import json

import numpy as np
import pytest

from vgdsmc import stage102_radial_speed_shell_audit as stage102


def test_stage102_design_is_frozen():
    stage102.validate_stage102_design()
    with pytest.raises(ValueError):
        stage102.validate_stage102_design(radial_shell_count=5)
    with pytest.raises(ValueError):
        stage102.validate_stage102_design(wall_band_cells=5)
    with pytest.raises(ValueError):
        stage102.validate_stage102_design(source_relaxation=0.5)


def test_radial_shell_partition_is_equal_node_count_and_speed_ordered():
    quadrature = stage102.mapped_polar_quadrature(
        *stage102.RULE, radial_scale=stage102.RADIAL_SCALE
    )
    shells = stage102.radial_shell_indices(quadrature.vx, quadrature.vy)
    speed = np.hypot(quadrature.vx, quadrature.vy)
    assert set(np.unique(shells).tolist()) == set(range(stage102.RADIAL_SHELL_COUNT))
    expected = stage102.RADIAL_NODES_PER_SHELL * stage102.RULE[1]
    for k in range(stage102.RADIAL_SHELL_COUNT):
        selected = shells == k
        assert int(np.sum(selected)) == expected
        if k + 1 < stage102.RADIAL_SHELL_COUNT:
            assert np.max(speed[selected]) <= np.min(speed[shells == k + 1])


def test_shell_metrics_close_against_parent_map():
    rng = np.random.default_rng(20260810)
    ny, nx = 4, 5
    nv = stage102.RULE[0] * stage102.RULE[1]
    term = rng.normal(size=(ny, nx, nv))
    weight = 0.5 + rng.random(nv)
    shells = np.repeat(
        np.arange(stage102.RADIAL_SHELL_COUNT, dtype=np.int64),
        stage102.RADIAL_NODES_PER_SHELL * stage102.RULE[1],
    )
    interior = np.ones((ny, nx), dtype=bool)
    parent = np.sum(np.abs(term) * weight[None, None, :], axis=-1)
    metrics = stage102._shell_metrics_from_term(term, weight, shells, interior, parent)
    assert metrics["shell_parent_closure_relative"] <= 1.0e-14
    assert np.isclose(np.sum(metrics["shell_abs_share"]), 1.0)
    assert np.all(metrics["shell_signed_to_abs_ratio"] <= 1.0 + 1.0e-14)


def _record(final_share: float, growth: float, index: int) -> dict[str, object]:
    return {
        "index": index,
        "weighted_abs": {
            "first": 1.0,
            "final": growth,
            "minimum": 1.0,
            "maximum": growth,
            "final_to_first_ratio": growth,
            "maximum_to_first_ratio": growth,
        },
        "abs_share": {
            "first": 0.25,
            "final": final_share,
            "minimum": min(0.25, final_share),
            "maximum": max(0.25, final_share),
            "final_to_first_ratio": final_share / 0.25,
            "maximum_to_first_ratio": max(0.25, final_share) / 0.25,
        },
    }


def _histories(final_shares, growths):
    out = {}
    for distribution in ("phi", "psi"):
        out[f"{distribution}_shell_abs_share"] = np.vstack(
            [np.full(4, 0.25), np.asarray(final_shares[distribution], dtype=float)]
        )
        first_abs = np.ones(4)
        final_abs = np.asarray(growths[distribution], dtype=float)
        out[f"{distribution}_shell_weighted_abs"] = np.vstack([first_abs, final_abs])
    return out


def test_stage102_decision_preserves_blockers_and_accepts_common_single_shell():
    summary = {
        "phi": [_record(0.15, 2.5, i) for i in range(4)],
        "psi": [_record(0.15, 2.5, i) for i in range(4)],
    }
    summary["phi"][2] = _record(0.55, 2.2, 2)
    summary["psi"][2] = _record(0.53, 2.3, 2)
    histories = _histories(
        {"phi": [0.15, 0.15, 0.55, 0.15], "psi": [0.16, 0.15, 0.53, 0.16]},
        {"phi": [2.2, 2.3, 2.2, 2.4], "psi": [2.1, 2.3, 2.3, 2.4]},
    )
    assert stage102.stage102_decision(summary, histories, 0.0, True) == (
        "stage102_common_dominant_radial_shell_2_stage103_shell_spatial_localization_audit"
    )
    assert stage102.stage102_decision(summary, histories, 2.0e-12, True) == (
        "stage102_shell_parent_closure_blocker_without_retuning"
    )
    assert stage102.stage102_decision(summary, histories, 0.0, False) == (
        "stage102_nonfinite_replay_blocker_without_retuning"
    )


def test_stage102_decision_routes_common_top_two_shells_without_retuning():
    final = {"phi": [0.35, 0.34, 0.16, 0.15], "psi": [0.36, 0.33, 0.16, 0.15]}
    growth = {"phi": [2.4, 2.5, 2.3, 2.2], "psi": [2.3, 2.4, 2.2, 2.1]}
    summary = {
        d: [_record(final[d][i], growth[d][i], i) for i in range(4)]
        for d in ("phi", "psi")
    }
    histories = _histories(final, growth)
    decision = stage102.stage102_decision(summary, histories, 0.0, True)
    assert decision == (
        "stage102_common_top_two_radial_shells_0_1_stage103_shell_pair_spatial_localization_audit"
    )


def test_stage101_authorization_rejects_wrong_endpoint(tmp_path):
    base = {
        "stage": 101,
        "decision": stage102.STAGE101_DECISION,
        "configuration": {
            "grid": list(stage102.GRID),
            "kn0": stage102.KNUDSEN,
            "cold_hot_ratio": stage102.COLD_HOT_RATIO,
            "rule": list(stage102.RULE),
            "radial_scale": stage102.RADIAL_SCALE,
            "limiter": stage102.LIMITER,
            "boundary_slope": stage102.BOUNDARY_SLOPE,
            "source_relaxation": stage102.SOURCE_RELAXATION,
            "tolerance": stage102.TOLERANCE,
            "correction_floor": stage102.STAGE41_CORRECTION_FLOOR,
            "diagnostic_steps": stage102.DIAGNOSTIC_STEPS,
            "wall_band_cells": stage102.WALL_BAND_CELLS,
        },
        "finite": True,
        "executed_steps": stage102.DIAGNOSTIC_STEPS,
        "maximum_sector_parent_closure_relative": 1.0e-16,
    }
    (tmp_path / "summary.json").write_text(json.dumps(base), encoding="utf-8")
    loaded = stage102._load_and_validate_stage101(tmp_path)
    assert loaded["decision"] == stage102.STAGE101_DECISION

    bad = dict(base)
    bad["decision"] = "wrong"
    (tmp_path / "summary.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError):
        stage102._load_and_validate_stage101(tmp_path)
