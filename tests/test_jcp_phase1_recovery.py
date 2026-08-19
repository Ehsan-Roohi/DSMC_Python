from __future__ import annotations

import os
from pathlib import Path

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


if __name__ == "__main__":
    test_missing_primary_is_replaced_by_first_passing_spare()
    test_failed_check_and_seed_mismatch_are_rejected_in_locked_order()
    print("2 JCP2 recovery tests passed")
