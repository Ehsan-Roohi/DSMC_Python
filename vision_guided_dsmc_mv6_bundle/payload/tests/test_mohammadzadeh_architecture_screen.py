import json

import numpy as np
import pytest

from vgdsmc import mohammadzadeh_architecture_screen as screen


def test_task_matrix_is_four_architectures_by_three_seeds():
    tasks = [screen.task_from_index(index) for index in range(12)]
    assert len(tasks) == 12
    assert {name for name, _ in tasks} == set(screen.ARCHITECTURES)
    assert {seed for _, seed in tasks} == set(screen.TRAINING_SEEDS)
    assert all(tasks.count((name, seed)) == 1 for name in screen.ARCHITECTURES for seed in screen.TRAINING_SEEDS)


def test_protocol_locks_budget_one_and_no_automatic_full_matrix():
    protocol = json.loads(screen.protocol_path().read_text(encoding="utf-8"))
    assert protocol["comparison_contract"]["budget_blocks"] == 1
    assert protocol["source_contract"]["reuse_mv5_confirmatory_references_without_new_DSMC"]
    assert not protocol["promotion_rule"]["automatic_full_budget_matrix_submission"]
    assert protocol["source_contract"]["confirmatory_conditions"] == [
        "kn0p075_u150",
        "kn0p075_u300",
        "kn0p1_u200",
        "kn0p1_u400",
    ]


def test_fixed_physical_scaling_never_uses_degenerate_condition_std():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(4, 7, 6, 6)).astype(np.float32)
    y = rng.normal(size=(4, 2, 6, 6)).astype(np.float32)
    x[:, 5] = np.log10(0.05)
    x[:, 6] = 1.0
    scaling = screen.fixed_physical_scaling(x, y)
    assert np.isclose(scaling["input_mean"][0, 5, 0, 0], screen.CONDITION_CENTERS[0])
    assert np.isclose(scaling["input_mean"][0, 6, 0, 0], screen.CONDITION_CENTERS[1])
    assert np.isclose(scaling["input_std"][0, 5, 0, 0], screen.CONDITION_SCALES[0])
    assert np.isclose(scaling["input_std"][0, 6, 0, 0], screen.CONDITION_SCALES[1])
    assert scaling["input_std"][0, 5, 0, 0] > 0.1


def test_parameter_counts_match_lock_and_parity_when_torch_available():
    pytest.importorskip("torch")
    protocol = screen.locked_protocol()
    report = screen.parameter_report(7)
    assert report["trainable_parameters"] == protocol["comparison_contract"][
        "parameter_counts_for_seven_input_channels"
    ]
    assert report["pass"]
    assert report["maximum_to_minimum_ratio"] <= 1.10


@pytest.mark.parametrize("architecture", screen.ARCHITECTURES)
def test_architecture_output_shape(architecture):
    torch = pytest.importorskip("torch")
    model = screen.build_architecture(architecture, in_channels=7)
    value = model(torch.zeros(2, 7, 20, 20))
    assert value.shape == (2, 2, 20, 20)
    assert torch.isfinite(value).all()

def test_metric_tree_comparison_tolerates_only_cpu_reduction_roundoff():
    left = {
        "raw_composite_nrmse": 0.125,
        "per_field": {
            "T": {"raw_nrmse": 0.25, "ratio": float("nan")},
            "u": {"raw_nrmse": 0.5, "ratio": 1.0},
        },
        "label": "raw",
    }
    roundoff = {
        "raw_composite_nrmse": 0.12500000001,
        "per_field": {
            "T": {"raw_nrmse": 0.25000000002, "ratio": float("nan")},
            "u": {"raw_nrmse": 0.50000000003, "ratio": 1.0},
        },
        "label": "raw",
    }
    changed = {
        **roundoff,
        "raw_composite_nrmse": 0.126,
    }

    assert screen._metrics_equivalent(left, roundoff)
    assert not screen._metrics_equivalent(left, changed)

