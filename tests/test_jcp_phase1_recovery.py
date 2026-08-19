from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np

from vgdsmc import jcp_phase1_recovery as recovery


def _with_fake_seed_contract(verify):
    original_bank = recovery.jcp2.load_seed_bank
    original_seeds = recovery.jcp2.group_seeds
    original_verify = recovery.jcp2._verify_artifacts
    recovery.jcp2.load_seed_bank = lambda: {
        "evaluation_primary": [101, 102],
        "evaluation_spares": [103, 104],
        "reference_primary": [201],
        "reference_spares": [202],
    }
    recovery.jcp2.group_seeds = lambda group: (
        (101, 102, 103, 104) if group == "evaluation" else (201, 202)
    )
    recovery.jcp2._verify_artifacts = verify
    return original_bank, original_seeds, original_verify


def _restore(contract):
    (
        recovery.jcp2.load_seed_bank,
        recovery.jcp2.group_seeds,
        recovery.jcp2._verify_artifacts,
    ) = contract


def test_missing_primary_is_replaced_by_first_passing_spare():
    def verify(directory: Path):
        seed = int(directory.name.removeprefix("seed_"))
        if seed == 101:
            raise FileNotFoundError(directory / "artifact_manifest.json")
        return {"seed": seed, "mechanical_checks": {"complete": True}}

    contract = _with_fake_seed_contract(verify)
    previous_audit_root = os.environ.pop("JCP2_SELECTION_AUDIT_ROOT", None)
    try:
        audit = recovery.qc_audit(Path("/unused"), "evaluation", 2)
    finally:
        _restore(contract)
        if previous_audit_root is not None:
            os.environ["JCP2_SELECTION_AUDIT_ROOT"] = previous_audit_root

    assert [record["seed"] for record in audit["accepted"]] == [102, 103]
    assert [record["role"] for record in audit["accepted"]] == [
        "primary",
        "spare",
    ]
    assert audit["rejected_before_selection_completed"][0]["seed"] == 101
    assert audit["selection_complete"]


def test_failed_check_and_seed_mismatch_are_rejected_in_locked_order():
    def verify(directory: Path):
        seed = int(directory.name.removeprefix("seed_"))
        if seed == 101:
            return {"seed": seed, "mechanical_checks": {"stationary": False}}
        if seed == 102:
            return {"seed": 999, "mechanical_checks": {"complete": True}}
        return {"seed": seed, "mechanical_checks": {"complete": True}}

    contract = _with_fake_seed_contract(verify)
    previous_audit_root = os.environ.pop("JCP2_SELECTION_AUDIT_ROOT", None)
    try:
        audit = recovery.qc_audit(Path("/unused"), "evaluation", 2)
    finally:
        _restore(contract)
        if previous_audit_root is not None:
            os.environ["JCP2_SELECTION_AUDIT_ROOT"] = previous_audit_root

    assert [record["seed"] for record in audit["accepted"]] == [103, 104]
    assert [record["seed"] for record in audit["rejected_before_selection_completed"]] == [
        101,
        102,
    ]


def test_shifted_completion_evaluation_is_reference_free():
    class Config:
        knudsen = 0.085
        lid_velocity_x = 350.0
        nx = 3
        ny = 2

    fields = {
        "rho": np.ones((2, 3)),
        "u": np.zeros((2, 3)),
        "v": np.zeros((2, 3)),
        "T": np.full((2, 3), 300.0),
        "qx": np.arange(6, dtype=float).reshape(2, 3),
        "qy": -np.arange(6, dtype=float).reshape(2, 3),
    }
    report = recovery.shifted_condition_evaluation(fields, Config())
    assert report["scope"] == recovery.RECOVERY_EVALUATION_SCOPE
    assert report["external_validation_claim"] is False
    assert report["legacy_kn0p05_u100_reference_applied"] is False
    assert report["all_completion_diagnostics_pass"]


def test_shifted_completion_rejects_the_legacy_condition():
    class Config:
        knudsen = 0.05
        lid_velocity_x = 100.0
        nx = 1
        ny = 1

    fields = {name: np.ones((1, 1)) for name in recovery.CORE_COMPLETION_FIELDS}
    try:
        recovery.shifted_condition_evaluation(fields, Config())
    except ValueError as error:
        assert "locked JCP2 S2" in str(error)
    else:
        raise AssertionError("legacy condition was accepted by JCP2 recovery")


def test_reference_free_heat_flux_topology_counts_sign_changes():
    class Config:
        nx = 4
        ny = 4

    fields = {
        "qx": np.asarray(
            [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, -1.0, 1.0],
             [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, -1.0, 1.0]]
        ),
        "qy": np.asarray(
            [[1.0, -1.0, 1.0, -1.0], [1.0, 1.0, 1.0, 1.0],
             [1.0, -1.0, 1.0, -1.0], [-1.0, -1.0, 1.0, 1.0]]
        ),
    }
    topology = recovery.shifted_heat_flux_topology(
        fields,
        Config(),
        {"scope": recovery.RECOVERY_EVALUATION_SCOPE},
    )
    assert topology["scope"] == "reference_free_sign_topology"
    assert topology["qx_vertical_centerline"] == 3
    assert topology["qy_horizontal_centerline"] == 3
    assert topology["qy_near_y_over_l_0p8"] == 1


def test_partial_final_artifacts_are_preserved_before_resume():
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        (directory / "summary.json").write_text("partial", encoding="utf-8")
        (directory / "fields.npz").write_bytes(b"partial")
        backup = recovery._preserve_partial_final_artifacts(directory)
        assert backup == directory / "pre_checkpoint_recovery_partial"
        assert (backup / "summary.json").read_text(encoding="utf-8") == "partial"
        assert (backup / "fields.npz").read_bytes() == b"partial"
        assert not (directory / "summary.json").exists()
        assert not (directory / "fields.npz").exists()


if __name__ == "__main__":
    test_missing_primary_is_replaced_by_first_passing_spare()
    test_failed_check_and_seed_mismatch_are_rejected_in_locked_order()
    test_shifted_completion_evaluation_is_reference_free()
    test_shifted_completion_rejects_the_legacy_condition()
    test_reference_free_heat_flux_topology_counts_sign_changes()
    test_partial_final_artifacts_are_preserved_before_resume()
    print("6 JCP2 recovery tests passed")
