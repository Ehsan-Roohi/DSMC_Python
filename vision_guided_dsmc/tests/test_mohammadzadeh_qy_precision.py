from __future__ import annotations

import json

import numpy as np
import pytest

from vgdsmc.mohammadzadeh_qy_precision import (
    STAGE,
    stage_configuration,
    verify_lock,
)
from vgdsmc.mohammadzadeh_qy_ensemble import (
    STUDENT_T_95_DF7,
    _student_t_statistics,
)
from vgdsmc.mohammadzadeh_spatial_refinement import _block_index, _sample_steps


def test_m3_lock_and_all_preregistered_seeds_are_valid():
    report = verify_lock()
    assert report["status"] == "M3_lock_verified_without_running_trajectories"
    assert report["seeds"] == list(range(91901, 91909))
    assert report["grid"] == 100
    assert report["particles_per_seed"] == 320_000


def test_m3_uses_five_times_the_original_R100_production_window():
    cfg, protocol, specification, _, _ = stage_configuration(STAGE, 91901)
    assert cfg.steps == 106_250
    assert cfg.sample_start == 12_500
    assert cfg.steps - cfg.sample_start == 5 * (31_250 - 12_500)
    assert specification["checkpoint_interval_steps"] == 1000
    assert protocol["runtime_contract"]["sample_count"] == 3000


def test_m3_sample_schedule_is_exact_unique_and_spans_all_blocks():
    cfg, protocol, _, _, _ = stage_configuration(STAGE, 91901)
    schedule = _sample_steps(cfg, protocol["runtime_contract"]["sample_count"])
    assert len(schedule) == len(np.unique(schedule)) == 3000
    assert schedule[0] == cfg.sample_start
    assert schedule[-1] == cfg.steps - 1
    blocks = [_block_index(int(step), cfg, 10) for step in schedule]
    assert sorted(set(blocks)) == list(range(10))
    assert all(blocks.count(index) == 300 for index in range(10))


def test_m3_precision_prediction_is_predeclared_not_outcome_dependent():
    _, protocol, _, _, _ = stage_configuration(STAGE, 91901)
    design = protocol["precision_design"]
    expected = 0.7324286272718353 / np.sqrt((8 * 5) / (2 * 1))
    assert design["predicted_qy_profile_rse_from_locked_R100"] == pytest.approx(expected)
    assert protocol["statistical_gates"]["normalized_qy_profile_global_rse_max"] == 0.2


def test_m3_rejects_nonpreregistered_seed():
    with pytest.raises(ValueError, match="not preregistered"):
        stage_configuration(STAGE, 91909)


def test_m3_uses_eight_seed_student_t_intervals():
    values = np.arange(8 * 3, dtype=float).reshape(8, 3)
    statistics = _student_t_statistics(values)
    expected_se = np.std(values, axis=0, ddof=1) / np.sqrt(8.0)
    assert statistics["replicate_count"] == 8
    assert statistics["degrees_of_freedom"] == 7
    assert statistics["ci95_critical_value"] == STUDENT_T_95_DF7
    assert np.allclose(statistics["standard_error"], expected_se)
    assert np.allclose(
        statistics["ci95_high"] - statistics["mean"],
        STUDENT_T_95_DF7 * expected_se,
    )
