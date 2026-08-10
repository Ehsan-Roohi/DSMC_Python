import json

import numpy as np
import pytest

from vgdsmc import stage99_replay_provenance_audit as stage99


def test_frozen_design_defaults():
    stage99.validate_stage99_design()


def test_frozen_design_rejects_tolerance_or_unknown_changes():
    with pytest.raises(ValueError):
        stage99.validate_stage99_design(replay_tolerance=1e-9)
    with pytest.raises(ValueError):
        stage99.validate_stage99_design(kn0=0.1)


def test_relative_l2_identity_and_perturbation():
    a = np.arange(16.0).reshape(4, 4)
    assert stage99._relative_l2(a, a) == 0.0
    assert stage99._relative_l2(a + 1e-8, a) > 0.0


def test_drift_metrics_localize_and_close_shares():
    ref = np.ones((64, 64))
    actual = ref.copy()
    actual[0, 0] += 1e-6
    m = stage99.drift_metrics(actual, ref)
    assert m["relative_l2"] > 0.0
    assert m["maximum_cell_yx"] == [0, 0]
    assert m["wall_band_absolute_difference_share"] == pytest.approx(1.0)
    assert m["wall_band_absolute_difference_share"] + m["interior_absolute_difference_share"] == pytest.approx(1.0)


def test_decision_event_context_mismatch_blocks():
    assert stage99.stage99_decision(False, 0.0, 0.0, 0.0).startswith(
        "stage99_push_pr_event_context_mismatch"
    )


def test_decision_archive_handoff_mismatch_blocks():
    assert stage99.stage99_decision(True, 1e-16, 0.0, 0.0).startswith(
        "stage99_stage96_stage97_archive_handoff_mismatch"
    )


def test_decision_cross_run_drift_routes_fused_single_run():
    assert stage99.stage99_decision(True, 0.0, 1e-16, 1e-10) == (
        "stage99_cross_run_iterative_replay_drift_stage100_fused_single_run_directional_audit"
    )


def test_decision_clean_replay_routes_velocity_sector():
    assert stage99.stage99_decision(True, 0.0, 1e-16, 1e-13) == (
        "stage99_replay_within_strict_tolerance_stage100_interior_velocity_sector_audit"
    )


def test_wall_band_mask_has_expected_boundary_only():
    mask = stage99._wall_band_mask((64, 64))
    assert mask.shape == (64, 64)
    assert mask[0, 32]
    assert mask[32, 0]
    assert not mask[32, 32]


def test_constants_preserve_negative_stage98_endpoint():
    assert stage99.REPLAY_TOLERANCE == 1e-12
    assert stage99.STAGE98_DECISION == "stage98_decomposition_or_replay_mismatch_blocker_without_retuning"
    assert stage99.STAGE98_PUSH_RUN_ID == 31360755010
    assert stage99.STAGE98_PR_RUN_ID == 31360757869


def test_validate_inputs_rejects_wrong_stage(tmp_path):
    roots = []
    for i in range(4):
        root = tmp_path / str(i)
        root.mkdir()
        (root / "summary.json").write_text(json.dumps({"stage": -1, "decision": "bad"}))
        roots.append(root)
    with pytest.raises(ValueError):
        stage99._validate_inputs(*roots)
