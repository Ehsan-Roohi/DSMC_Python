from __future__ import annotations

import numpy as np

from vgdsmc.mohammadzadeh_vision_mv4 import (
    RESIDUAL_CAP_SIGMA,
    bounded_residual_candidate,
    coordinate_support_report,
    project_physical_fields,
    select_fallback,
)


def _condition(identifier: str, knudsen: float, speed: float) -> dict[str, float | str]:
    return {
        "id": identifier,
        "knudsen": knudsen,
        "lid_speed_m_per_s": speed,
    }


def test_bounded_residual_cannot_exceed_locked_sigma_cap() -> None:
    raw = np.zeros((2, 2, 3, 3), dtype=np.float32)
    unbounded = np.full_like(raw, 1.0e9)
    residual_std = np.asarray([[[[2.0]], [[5.0]]]], dtype=np.float32)
    corrected = bounded_residual_candidate(raw, unbounded, residual_std)
    normalized = (corrected - raw) / residual_std
    assert np.max(np.abs(normalized)) <= RESIDUAL_CAP_SIGMA + 1.0e-6


def test_coordinate_support_accepts_interpolation_without_target() -> None:
    training = (
        _condition("a", 0.05, 100.0),
        _condition("b", 0.05, 400.0),
        _condition("c", 0.10, 100.0),
    )
    report = coordinate_support_report(_condition("held", 0.05, 200.0), training)
    assert report["trusted_interpolation"]
    assert report["action"] == "bounded_vision"
    assert "target" not in report["rule"] or report["rule"].endswith("no_test_target")


def test_coordinate_support_rejects_knudsen_extrapolation() -> None:
    training = (
        _condition("a", 0.05, 100.0),
        _condition("b", 0.05, 200.0),
        _condition("c", 0.05, 400.0),
    )
    report = coordinate_support_report(_condition("held", 0.10, 100.0), training)
    assert not report["trusted_interpolation"]
    assert report["action"] == "raw_identity_fallback"
    assert not report["coordinates"]["log10_Kn"]["inside"]


def test_coordinate_support_rejects_speed_extrapolation() -> None:
    training = (
        _condition("a", 0.05, 100.0),
        _condition("b", 0.05, 200.0),
        _condition("c", 0.10, 100.0),
    )
    report = coordinate_support_report(_condition("held", 0.05, 400.0), training)
    assert not report["trusted_interpolation"]
    assert not report["coordinates"]["U_lid_over_100"]["inside"]


def test_physical_projection_replaces_nonfinite_and_bounds_fields() -> None:
    raw = np.zeros((1, 2, 2, 2), dtype=np.float32)
    raw[:, 0] = 300.0
    candidate = raw.copy()
    candidate[0, 0, 0, 0] = np.nan
    candidate[0, 0, 0, 1] = -20.0
    candidate[0, 0, 1, 0] = 1.0e7
    candidate[0, 1, 0, 0] = -1.0e7
    candidate[0, 1, 0, 1] = 1.0e7
    projected, report = project_physical_fields(candidate, raw, 100.0)
    assert np.all(np.isfinite(projected))
    assert projected[:, 0].min() >= 1.0
    assert projected[:, 0].max() <= 2000.0
    assert np.max(np.abs(projected[:, 1])) <= 200.0
    assert report["nonfinite_replaced_count"] == 1
    assert report["projected_value_count"] >= 4


def test_fallback_selection_uses_validation_scores() -> None:
    raw = np.ones((2, 2, 4, 4), dtype=np.float32)
    target = np.zeros_like(raw)
    labels = np.asarray(["c", "c"])
    specs = {"c": _condition("c", 0.05, 100.0)}
    method, records = select_fallback(raw, target, labels, specs, 1, 1)
    assert method in {"raw", "gaussian_like", "tsvd_pod_type"}
    assert {item["method"] for item in records} == {
        "raw",
        "gaussian_like",
        "tsvd_pod_type",
    }
    assert all("validation_condition_mean_composite_nrmse" in item for item in records)
