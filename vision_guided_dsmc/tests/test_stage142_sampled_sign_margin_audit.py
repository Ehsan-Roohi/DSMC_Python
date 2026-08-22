import json
from pathlib import Path

import numpy as np
import pytest

from vgdsmc.stage142_sampled_sign_margin_audit import (
    EXPECTED_STAGE141_ARTIFACT_ID,
    EXPECTED_STAGE141_ARTIFACT_SHA256,
    EXPECTED_STAGE141_DECISION,
    EXPECTED_STAGE141_JOB_ID,
    EXPECTED_STAGE141_PAYLOAD_SHA256,
    EXPECTED_STAGE141_RUN_ID,
    EXPECTED_STAGE141_SOURCE_HEAD,
    EXPECTED_STAGE141_SUMMARY_SHA256,
    MIXED_SUPPORT,
    NONFINITE,
    PARENT_RECORD_BLOCKER,
    PERSISTENT_STRONG_MARGIN,
    PERSISTENT_WEAK_MARGIN,
    classify_sampled_sign_margin,
    positive_run_length,
    run_stage142,
    validate_stage142_design,
)


def test_validate_design_accepts_frozen_configuration():
    validate_stage142_design()


def test_validate_design_rejects_retuning():
    with pytest.raises(ValueError):
        validate_stage142_design(velocity_grid_retuning=True)


def test_positive_run_length_stops_at_first_nonpositive():
    assert positive_run_length(np.array([1.0, 2.0, 0.0, 4.0])) == 2


def test_classifier_routes_nonfinite_and_parent_blocker():
    kwargs = dict(
        positive_run_length_samples=5,
        later_positive_sign_coherence=1.0,
        upper_to_later_positive_median_ratio=0.2,
        deletion_sign_retention_fraction=1.0,
    )
    assert classify_sampled_sign_margin(**kwargs, finite=False) == NONFINITE
    assert classify_sampled_sign_margin(**kwargs, parent_record_ok=False) == PARENT_RECORD_BLOCKER


def test_classifier_routes_persistent_weak_margin():
    assert classify_sampled_sign_margin(
        positive_run_length_samples=5,
        later_positive_sign_coherence=1.0,
        upper_to_later_positive_median_ratio=0.2,
        deletion_sign_retention_fraction=1.0,
    ) == PERSISTENT_WEAK_MARGIN


def test_classifier_routes_persistent_strong_margin():
    assert classify_sampled_sign_margin(
        positive_run_length_samples=5,
        later_positive_sign_coherence=1.0,
        upper_to_later_positive_median_ratio=0.4,
        deletion_sign_retention_fraction=1.0,
    ) == PERSISTENT_STRONG_MARGIN


def test_classifier_routes_mixed_support_when_deletion_fails():
    assert classify_sampled_sign_margin(
        positive_run_length_samples=5,
        later_positive_sign_coherence=1.0,
        upper_to_later_positive_median_ratio=0.2,
        deletion_sign_retention_fraction=2.0 / 3.0,
    ) == MIXED_SUPPORT


def test_exact_stage141_sampled_sign_margin_route(tmp_path: Path):
    parent = tmp_path / "stage141"
    parent.mkdir()
    summary = {
        "stage": 141,
        "finite": True,
        "decision": EXPECTED_STAGE141_DECISION,
        "configuration": {
            "grid": [64, 64],
            "interior_grid": [56, 56],
            "kn0": 10.0,
            "cold_hot_ratio": 0.1,
            "rule": [40, 96],
            "radial_scale": 2.0,
            "limiter": "minmod",
            "boundary_slope": "zero",
            "source_relaxation": 1.0,
            "correction_floor": 0.05,
            "physical_parameter_retuning": False,
            "velocity_grid_retuning": False,
        },
    }
    (parent / "summary.json").write_text(json.dumps(summary))
    depth = np.array([5.961196168207854, 6.961196168207854, 7.961196168207854, 8.961196168207854, 9.961196168207854, 10.961196168207854, 11.961196168207854])
    complement = np.array([-0.01503602, -0.024704146040515018, 0.0036969451288478283, 0.017300897584414998, 0.00916962404949162, 0.017917405508898476, 0.01657221223619726])
    np.savez_compressed(
        parent / "node_leverage.npz",
        right_depth=depth,
        complement_signed=complement,
        crossing_bracket_indices=np.array([1, 2], dtype=int),
    )
    record = {
        "stage": 141,
        "source_head": EXPECTED_STAGE141_SOURCE_HEAD,
        "workflow_run_id": EXPECTED_STAGE141_RUN_ID,
        "workflow_job_id": EXPECTED_STAGE141_JOB_ID,
        "artifact_id": EXPECTED_STAGE141_ARTIFACT_ID,
        "artifact_sha256": EXPECTED_STAGE141_ARTIFACT_SHA256,
        "summary_sha256": EXPECTED_STAGE141_SUMMARY_SHA256,
        "node_leverage_sha256": EXPECTED_STAGE141_PAYLOAD_SHA256,
        "decision": EXPECTED_STAGE141_DECISION,
        "workflow_status": "completed",
        "workflow_conclusion": "success",
    }
    record_path = tmp_path / "record.json"
    record_path.write_text(json.dumps(record))
    out = tmp_path / "out"
    result = run_stage142(parent, record_path, out)
    assert result["decision"] == PERSISTENT_WEAK_MARGIN
    assert result["metrics"]["positive_run_length_samples"] == 5
    assert result["metrics"]["later_positive_sign_coherence"] == 1.0
    assert result["metrics"]["deletion_sign_retention_fraction"] == 1.0
    assert result["metrics"]["upper_to_later_positive_median_ratio"] == pytest.approx(0.21828200294725736)
    assert (out / "sampled_sign_margin.npz").exists()
