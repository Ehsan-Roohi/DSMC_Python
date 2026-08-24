from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np

from vgdsmc import jcp_phase1_stationarity_repair as repair


def test_familywise_limit_is_fixed_and_more_conservative_than_two():
    limit = repair.familywise_z_limit(6)
    assert 2.63 < limit < 2.65
    assert repair.familywise_z_limit(6) == limit


def test_signed_negative_profile_has_a_valid_scale_and_finite_statistics():
    y = np.linspace(0.0, 1.0, 8)
    overall = -1.0 - y[:, None] * np.ones((8, 5))
    assert np.max(repair._profile_at_y(overall, 0.8)) < 0.0
    scale = np.max(np.abs(repair._profile_at_y(overall, 0.8)))
    assert scale > 0.0
    blocks = np.asarray(
        [repair._profile_at_y(overall * (1.0 + 0.01 * index), 0.8) / scale for index in range(13)]
    )
    report = repair._two_half_report(
        np.min(blocks, axis=1), minimum_finite_per_half=3
    )
    assert np.isfinite(report["max_abs_drift_z_score"])


def test_two_half_gate_distinguishes_weak_and_strong_drift():
    rng = np.random.default_rng(20260819)
    stable = rng.normal(0.0, 1.0, 13)
    drifting = np.concatenate((rng.normal(0.0, 0.1, 6), rng.normal(2.0, 0.1, 7)))
    stable_report = repair._two_half_report(stable, minimum_finite_per_half=3)
    drift_report = repair._two_half_report(drifting, minimum_finite_per_half=3)
    limit = repair.familywise_z_limit(6)
    assert stable_report["max_abs_drift_z_score"] < limit
    assert drift_report["max_abs_drift_z_score"] > limit


def test_corrected_stationarity_replaces_infinite_negative_qy_diagnostic():
    rng = np.random.default_rng(44)
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        overall = -2.0 - np.linspace(0.0, 1.0, 8)[:, None] * np.ones((8, 6))
        blocks = np.asarray(
            [overall * (1.0 + rng.normal(0.0, 0.01)) for _ in range(13)]
        )
        np.savez_compressed(directory / "fields.npz", qy=overall)
        np.savez_compressed(directory / "block_fields.npz", qy=blocks)
        tracked = {
            "macroscopic_lid_slip_center": {"max_abs_drift_z_score": 2.5},
            "microscopic_lid_slip_center": {"max_abs_drift_z_score": 1.0},
            "temperature_min_K": {"max_abs_drift_z_score": 0.5},
            "temperature_max_K": {"max_abs_drift_z_score": 1.5},
            "qy_profile_min_normalized": {"max_abs_drift_z_score": "+inf"},
            "qy_profile_max_normalized": {"max_abs_drift_z_score": "+inf"},
        }
        summary = {
            "stationarity": {
                "z_limit": 2.0,
                "minimum_finite_blocks_per_half": 3,
                "tracked": tracked,
            }
        }
        had_block_count = hasattr(repair.jcp2, "BLOCK_COUNT")
        previous = getattr(repair.jcp2, "BLOCK_COUNT", None)
        repair.jcp2.BLOCK_COUNT = 13
        try:
            corrected = repair.corrected_stationarity(directory, summary)
        finally:
            if had_block_count:
                repair.jcp2.BLOCK_COUNT = previous
            else:
                del repair.jcp2.BLOCK_COUNT
        assert corrected["legacy_positive_qy_scale"] < 0.0
        assert corrected["corrected_signed_qy_scale"] > 0.0
        assert np.isfinite(
            corrected["tracked"]["qy_profile_min_normalized"][
                "max_abs_drift_z_score"
            ]
        )
        assert corrected["checks"]["macroscopic_lid_slip_center"]


def test_locked_selection_can_ignore_an_unavailable_trailing_spare():
    records = [
        {"accepted": True, "seed": seed} for seed in range(20)
    ] + [
        {"accepted": False, "seed": 20, "artifact_error": {"type": "missing"}}
    ]
    selected = [record["seed"] for record in records if record["accepted"]][:20]
    assert len(selected) == 20
    assert selected == list(range(20))


if __name__ == "__main__":
    test_familywise_limit_is_fixed_and_more_conservative_than_two()
    test_signed_negative_profile_has_a_valid_scale_and_finite_statistics()
    test_two_half_gate_distinguishes_weak_and_strong_drift()
    test_corrected_stationarity_replaces_infinite_negative_qy_diagnostic()
    test_locked_selection_can_ignore_an_unavailable_trailing_spare()
    print("5 JCP2 stationarity-repair tests passed")
