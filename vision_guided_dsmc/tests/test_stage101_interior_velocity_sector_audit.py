import json

import numpy as np
import pytest

from vgdsmc import stage101_interior_velocity_sector_audit as stage101


def test_stage101_design_is_frozen():
    stage101.validate_stage101_design()
    with pytest.raises(ValueError):
        stage101.validate_stage101_design(n_angular_sectors=16)
    with pytest.raises(ValueError):
        stage101.validate_stage101_design(wall_band_cells=5)
    with pytest.raises(ValueError):
        stage101.validate_stage101_design(source_relaxation=0.5)


def test_velocity_sector_partition_centers_axes_and_diagonals():
    angle = np.arange(8, dtype=np.float64) * (np.pi / 4.0)
    vx = np.cos(angle)
    vy = np.sin(angle)
    index = stage101.velocity_sector_index(vx, vy)
    assert np.array_equal(index, np.arange(8))
    assert set(index.tolist()) == set(range(8))


def test_sector_metrics_close_against_independent_parent_map():
    rng = np.random.default_rng(20260810)
    ny, nx, nv = 6, 7, 8
    term = rng.normal(size=(ny, nx, nv))
    weight = 0.5 + rng.random(nv)
    sectors = np.arange(nv, dtype=np.int64)
    interior = np.ones((ny, nx), dtype=bool)
    interior[[0, -1], :] = False
    interior[:, [0, -1]] = False
    parent = np.sum(np.abs(term) * weight[None, None, :], axis=-1)

    metrics = stage101._sector_metrics_from_term(
        term, weight, sectors, interior, parent
    )
    assert metrics["sector_parent_closure_relative"] <= 1.0e-14
    assert np.isclose(np.sum(metrics["sector_abs_share"]), 1.0)
    assert np.all(metrics["sector_signed_to_abs_ratio"] <= 1.0 + 1.0e-14)


def _sector_record(final_share: float, growth: float) -> dict[str, object]:
    return {
        "index": 0,
        "label": "+x",
        "weighted_abs": {
            "first": 1.0,
            "final": growth,
            "minimum": 1.0,
            "maximum": growth,
            "final_to_first_ratio": growth,
            "maximum_to_first_ratio": growth,
        },
        "abs_share": {
            "first": 0.125,
            "final": final_share,
            "minimum": 0.125,
            "maximum": final_share,
            "final_to_first_ratio": final_share / 0.125,
            "maximum_to_first_ratio": final_share / 0.125,
        },
        "weighted_signed": {
            "first": 0.0,
            "final": 0.0,
            "minimum": 0.0,
            "maximum": 0.0,
            "final_to_first_ratio": 0.0,
            "maximum_to_first_ratio": 0.0,
        },
        "signed_to_abs_ratio": {
            "first": 0.0,
            "final": 0.0,
            "minimum": 0.0,
            "maximum": 0.0,
            "final_to_first_ratio": 0.0,
            "maximum_to_first_ratio": 0.0,
        },
    }


def test_stage101_decision_preserves_blockers_and_requires_common_sector():
    diffuse = [_sector_record(0.125, 2.5) for _ in range(8)]
    summary = {"phi": [dict(x, index=i) for i, x in enumerate(diffuse)],
               "psi": [dict(x, index=i) for i, x in enumerate(diffuse)]}
    assert stage101.stage101_decision(summary, 0.0, True) == (
        "stage101_diffuse_velocity_sector_growth_stage102_radial_speed_shell_audit"
    )
    assert stage101.stage101_decision(summary, 2.0e-12, True) == (
        "stage101_sector_parent_closure_blocker_without_retuning"
    )
    assert stage101.stage101_decision(summary, 0.0, False) == (
        "stage101_nonfinite_replay_blocker_without_retuning"
    )

    for d in ("phi", "psi"):
        summary[d][3] = _sector_record(0.30, 2.2)
        summary[d][3]["index"] = 3
    assert stage101.stage101_decision(summary, 0.0, True) == (
        "stage101_common_dominant_sector_3_stage102_sector_radial_shell_audit"
    )


def test_stage100_authorization_rejects_wrong_or_incomplete_endpoint(tmp_path):
    root = tmp_path
    base = {
        "stage": 100,
        "decision": stage101.STAGE100_DECISION,
        "configuration": {
            "grid": list(stage101.GRID),
            "kn0": stage101.KNUDSEN,
            "cold_hot_ratio": stage101.COLD_HOT_RATIO,
            "rule": list(stage101.RULE),
            "radial_scale": stage101.RADIAL_SCALE,
            "limiter": stage101.LIMITER,
            "boundary_slope": stage101.BOUNDARY_SLOPE,
            "source_relaxation": stage101.SOURCE_RELAXATION,
            "tolerance": stage101.TOLERANCE,
            "correction_floor": stage101.STAGE41_CORRECTION_FLOOR,
            "diagnostic_steps": stage101.DIAGNOSTIC_STEPS,
        },
        "finite": True,
        "executed_steps": stage101.DIAGNOSTIC_STEPS,
        "maximum_decomposition_closure_relative_l2": 1.0e-16,
        "maximum_same_run_parent_map_relative_l2": 1.0e-16,
    }
    (root / "summary.json").write_text(json.dumps(base), encoding="utf-8")
    loaded = stage101._load_and_validate_stage100(root)
    assert loaded["decision"] == stage101.STAGE100_DECISION

    bad = dict(base)
    bad["decision"] = "wrong"
    (root / "summary.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError):
        stage101._load_and_validate_stage100(root)

    bad = dict(base)
    bad["maximum_same_run_parent_map_relative_l2"] = 1.0e-8
    (root / "summary.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError):
        stage101._load_and_validate_stage100(root)
