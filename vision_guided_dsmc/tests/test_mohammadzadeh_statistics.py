import numpy as np

from vgdsmc.mohammadzadeh_statistics import (
    audit_seed_sets,
    block_means,
    ensemble_field_statistics,
    gate_ready_summary,
    half_stationarity,
    stationarity_by_field,
)


def test_equal_nonoverlapping_block_means_are_exact_and_strict():
    samples = np.arange(24.0).reshape(8, 3)
    expected = np.stack(
        [samples[0:2].mean(axis=0), samples[2:4].mean(axis=0),
         samples[4:6].mean(axis=0), samples[6:8].mean(axis=0)]
    )
    assert np.array_equal(block_means(samples, 4), expected)
    with np.testing.assert_raises_regex(ValueError, "exactly divisible"):
        block_means(samples[:7], 4)


def test_half_stationarity_uses_block_variance_for_difference_se():
    blocks = np.array([[1.0, 4.0], [3.0, 4.0], [2.0, 4.0], [4.0, 4.0]])
    summary = half_stationarity(blocks)
    assert np.array_equal(summary["first_half_mean"], np.array([2.0, 4.0]))
    assert np.array_equal(summary["second_half_mean"], np.array([3.0, 4.0]))
    assert np.allclose(summary["drift_standard_error"], np.array([np.sqrt(2.0), 0.0]))
    assert np.allclose(summary["drift_z_score"], np.array([1.0 / np.sqrt(2.0), 0.0]))
    assert np.isclose(summary["global_relative_drift"], np.sqrt(0.5) / np.sqrt(11.125))


def test_ensemble_field_mean_ci_and_rse_are_exact():
    runs = np.array(
        [
            [[8.0, 0.0], [4.0, 5.0]],
            [[10.0, 0.0], [4.0, 7.0]],
            [[12.0, 0.0], [4.0, 9.0]],
            [[14.0, 0.0], [4.0, 11.0]],
        ]
    )
    result = ensemble_field_statistics(runs)
    expected_mean = runs.mean(axis=0)
    expected_se = runs.std(axis=0, ddof=1) / 2.0
    assert np.array_equal(result["mean"], expected_mean)
    assert np.allclose(result["standard_error"], expected_se)
    assert np.allclose(result["ci95_low"], expected_mean - 1.96 * expected_se)
    assert np.allclose(result["ci95_high"], expected_mean + 1.96 * expected_se)
    assert result["relative_standard_error"][0, 1] == 0.0
    expected_global = np.sqrt(np.mean(expected_se**2)) / np.sqrt(np.mean(expected_mean**2))
    assert np.isclose(result["global_relative_standard_error"], expected_global)


def test_seed_audit_finds_duplicates_and_cross_group_overlap():
    failed = audit_seed_sets(
        {"development": [10, 10, 11], "confirmatory": [11, 20]}
    )
    assert failed["duplicates_within_groups"] == {
        "development": [10],
        "confirmatory": [],
    }
    assert failed["overlaps_between_groups"] == {"development__confirmatory": [11]}
    assert not failed["seed_gate_passed"]

    passed = audit_seed_sets(
        {"development": [10, 11], "validation": [20], "confirmatory": [30, 31]}
    )
    assert passed["within_group_unique"]
    assert passed["pairwise_disjoint"]
    assert passed["seed_gate_passed"]


def test_gate_summary_uses_explicit_thresholds_without_retuning():
    histories = {
        "T": np.tile(np.array([299.0, 301.0, 299.0, 301.0, 299.0, 301.0, 299.0, 301.0]), (2, 1)).T,
        "qy": np.tile(np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]), (2, 1)).T,
    }
    stationarity = stationarity_by_field(histories, block_count=8)
    ensemble = {
        "T": ensemble_field_statistics(np.array([[299.0, 301.0], [301.0, 299.0]])),
        "qy": ensemble_field_statistics(np.array([[-1.1, 1.1], [-0.9, 0.9]])),
    }
    seeds = audit_seed_sets({"development": [1, 2], "confirmatory": [3, 4]})
    summary = gate_ready_summary(
        stationarity,
        ensemble,
        seeds,
        stationarity_z_limit=1.96,
        rse_limits={"T": 0.01, "qy": 0.20},
    )
    assert summary["thresholds"] == {
        "stationarity_z_limit": 1.96,
        "rse_limits": {"T": 0.01, "qy": 0.20},
    }
    assert summary["checks"] == {
        "T_stationary": True,
        "T_rse": True,
        "qy_stationary": True,
        "qy_rse": True,
        "seeds_unique_and_disjoint": True,
    }
    assert summary["all_passed"]
