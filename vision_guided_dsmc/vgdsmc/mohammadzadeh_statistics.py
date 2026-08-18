from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any

import numpy as np


NORMAL_95_CRITICAL_VALUE = 1.96


def _finite_array(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        raise ValueError(f"{name} must have a leading replicate axis")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _safe_relative(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    absolute_numerator = np.abs(numerator)
    absolute_denominator = np.abs(denominator)
    return np.divide(
        absolute_numerator,
        absolute_denominator,
        out=np.where(absolute_numerator == 0.0, 0.0, np.inf),
        where=absolute_denominator > 0.0,
    )


def _global_relative(numerator: np.ndarray, denominator: np.ndarray) -> float:
    numerator_rms = float(np.sqrt(np.mean(np.asarray(numerator) ** 2)))
    denominator_rms = float(np.sqrt(np.mean(np.asarray(denominator) ** 2)))
    if denominator_rms > 0.0:
        return numerator_rms / denominator_rms
    return 0.0 if numerator_rms == 0.0 else float("inf")


def block_means(samples: np.ndarray, block_count: int) -> np.ndarray:
    """Return equal, contiguous, nonoverlapping means along axis zero.

    The sample count must be exactly divisible by ``block_count``.  Refusing
    to trim or redistribute samples makes the block definition reproducible
    and prevents an outcome-dependent choice of time window.
    """
    values = _finite_array(samples, "samples")
    if isinstance(block_count, bool) or not isinstance(
        block_count, (int, np.integer)
    ):
        raise TypeError("block_count must be an integer")
    if block_count < 2:
        raise ValueError("block_count must be at least two")
    if len(values) < block_count or len(values) % block_count != 0:
        raise ValueError(
            "sample count must be at least block_count and exactly divisible by it"
        )
    block_length = len(values) // block_count
    return values.reshape(
        (block_count, block_length) + values.shape[1:]
    ).mean(axis=1)


def block_means_by_field(
    samples_by_field: Mapping[str, np.ndarray],
    block_count: int,
) -> dict[str, np.ndarray]:
    """Apply ``block_means`` to a consistently sampled field dictionary."""
    if not samples_by_field:
        raise ValueError("samples_by_field must not be empty")
    result = {
        str(name): block_means(values, block_count)
        for name, values in samples_by_field.items()
    }
    sample_counts = {
        np.asarray(values).shape[0] for values in samples_by_field.values()
    }
    if len(sample_counts) != 1:
        raise ValueError("all fields must contain the same number of samples")
    return result


def half_stationarity(block_values: np.ndarray) -> dict[str, Any]:
    """Compare first- and second-half block means with an independent SE.

    The standard error of the difference is formed from sample variances of
    the block means in the two halves.  No pass/fail threshold is embedded;
    the returned scalar maxima can be evaluated against a preregistered gate.
    """
    blocks = _finite_array(block_values, "block_values")
    if len(blocks) < 4 or len(blocks) % 2 != 0:
        raise ValueError("stationarity requires an even number of at least four blocks")
    half_count = len(blocks) // 2
    first = blocks[:half_count]
    second = blocks[half_count:]
    first_mean = np.mean(first, axis=0)
    second_mean = np.mean(second, axis=0)
    difference = second_mean - first_mean
    standard_error = np.sqrt(
        np.var(first, axis=0, ddof=1) / half_count
        + np.var(second, axis=0, ddof=1) / half_count
    )
    z_score = np.zeros_like(difference, dtype=np.float64)
    np.divide(
        difference,
        standard_error,
        out=z_score,
        where=standard_error > 0.0,
    )
    zero_error_drift = (standard_error == 0.0) & (difference != 0.0)
    z_score[zero_error_drift] = np.copysign(
        np.inf,
        difference[zero_error_drift],
    )
    midpoint = 0.5 * (first_mean + second_mean)
    relative_drift = _safe_relative(difference, midpoint)
    return {
        "block_count": int(len(blocks)),
        "blocks_per_half": int(half_count),
        "first_half_mean": first_mean,
        "second_half_mean": second_mean,
        "drift": difference,
        "drift_standard_error": standard_error,
        "drift_z_score": z_score,
        "relative_drift": relative_drift,
        "max_abs_drift_z_score": float(np.max(np.abs(z_score))),
        "global_relative_drift": _global_relative(difference, midpoint),
    }


def stationarity_by_field(
    samples_by_field: Mapping[str, np.ndarray],
    block_count: int,
) -> dict[str, dict[str, Any]]:
    """Build block-based half-stationarity summaries for multiple fields."""
    blocks = block_means_by_field(samples_by_field, block_count)
    return {name: half_stationarity(values) for name, values in blocks.items()}


def ensemble_field_statistics(
    field_replicates: np.ndarray,
) -> dict[str, Any]:
    """Mean, normal-approximation 95% CI, and RSE of a field ensemble."""
    values = _finite_array(field_replicates, "field_replicates")
    if len(values) < 2:
        raise ValueError("at least two independent field replicates are required")
    mean = np.mean(values, axis=0)
    standard_deviation = np.std(values, axis=0, ddof=1)
    standard_error = standard_deviation / np.sqrt(len(values))
    margin = NORMAL_95_CRITICAL_VALUE * standard_error
    relative_standard_error = _safe_relative(standard_error, mean)
    return {
        "replicate_count": int(len(values)),
        "mean": mean,
        "standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
        "relative_standard_error": relative_standard_error,
        "global_relative_standard_error": _global_relative(standard_error, mean),
        "ci95_critical_value": NORMAL_95_CRITICAL_VALUE,
    }


def ensemble_statistics_by_field(
    field_replicates: Mapping[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    """Apply independent-seed ensemble statistics to multiple field arrays."""
    if not field_replicates:
        raise ValueError("field_replicates must not be empty")
    replicate_counts = {
        np.asarray(values).shape[0] for values in field_replicates.values()
    }
    if len(replicate_counts) != 1:
        raise ValueError("all fields must contain the same number of replicates")
    return {
        str(name): ensemble_field_statistics(values)
        for name, values in field_replicates.items()
    }


def _validated_seeds(seeds: Sequence[int], group: str) -> list[int]:
    result: list[int] = []
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError(f"all seeds in {group!r} must be integers")
        result.append(int(seed))
    if not result:
        raise ValueError(f"seed group {group!r} must not be empty")
    return result


def audit_seed_sets(seed_sets: Mapping[str, Sequence[int]]) -> dict[str, Any]:
    """Audit within-group uniqueness and pairwise disjointness of seed sets."""
    if not seed_sets:
        raise ValueError("seed_sets must not be empty")
    validated = {
        str(group): _validated_seeds(seeds, str(group))
        for group, seeds in seed_sets.items()
    }
    duplicates: dict[str, list[int]] = {}
    for group, seeds in validated.items():
        unique, counts = np.unique(np.asarray(seeds, dtype=np.int64), return_counts=True)
        duplicates[group] = [int(seed) for seed in unique[counts > 1]]

    overlaps: dict[str, list[int]] = {}
    for left, right in combinations(validated, 2):
        overlap = sorted(set(validated[left]) & set(validated[right]))
        overlaps[f"{left}__{right}"] = overlap
    within_group_unique = all(not values for values in duplicates.values())
    pairwise_disjoint = all(not values for values in overlaps.values())
    return {
        "group_sizes": {group: len(seeds) for group, seeds in validated.items()},
        "unique_group_sizes": {
            group: len(set(seeds)) for group, seeds in validated.items()
        },
        "duplicates_within_groups": duplicates,
        "overlaps_between_groups": overlaps,
        "within_group_unique": within_group_unique,
        "pairwise_disjoint": pairwise_disjoint,
        "seed_gate_passed": within_group_unique and pairwise_disjoint,
    }


def gate_ready_summary(
    stationarity: Mapping[str, Mapping[str, Any]],
    ensemble: Mapping[str, Mapping[str, Any]],
    seed_audit: Mapping[str, Any],
    *,
    stationarity_z_limit: float,
    rse_limits: Mapping[str, float],
) -> dict[str, Any]:
    """Evaluate statistics against caller-supplied, preregistered thresholds.

    Thresholds have no data-dependent defaults and are copied into the result,
    making later changes visible in the artifact rather than silently
    retuning a gate after outcomes have been inspected.
    """
    if not np.isfinite(stationarity_z_limit) or stationarity_z_limit <= 0.0:
        raise ValueError("stationarity_z_limit must be finite and positive")
    stationarity_fields = set(stationarity)
    ensemble_fields = set(ensemble)
    limit_fields = set(rse_limits)
    if stationarity_fields != ensemble_fields or ensemble_fields != limit_fields:
        raise ValueError(
            "stationarity, ensemble, and rse_limits must have identical field keys"
        )
    if not all(np.isfinite(limit) and limit > 0.0 for limit in rse_limits.values()):
        raise ValueError("all RSE limits must be finite and positive")

    metrics: dict[str, dict[str, float]] = {}
    checks: dict[str, bool] = {}
    for name in sorted(ensemble_fields):
        maximum_z = float(stationarity[name]["max_abs_drift_z_score"])
        global_rse = float(ensemble[name]["global_relative_standard_error"])
        metrics[name] = {
            "max_abs_drift_z_score": maximum_z,
            "global_relative_standard_error": global_rse,
        }
        checks[f"{name}_stationary"] = maximum_z <= stationarity_z_limit
        checks[f"{name}_rse"] = global_rse <= float(rse_limits[name])
    checks["seeds_unique_and_disjoint"] = bool(seed_audit.get("seed_gate_passed", False))
    return {
        "thresholds": {
            "stationarity_z_limit": float(stationarity_z_limit),
            "rse_limits": {name: float(rse_limits[name]) for name in sorted(rse_limits)},
        },
        "metrics": metrics,
        "checks": checks,
        "all_passed": all(checks.values()),
    }
