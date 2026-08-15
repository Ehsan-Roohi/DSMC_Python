#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import tempfile

from vgdsmc import mohammadzadeh_mv17b_a1_mechanical_recovery as a1


def test_only_incomplete_indices_are_recovered() -> None:
    assert a1.RECOVERY_INDICES == (1, 2, 3, 4, 9, 10, 11)
    assert len(a1.EXPECTED_CASES) == 12
    assert sum(row[-1] for row in a1.EXPECTED_CASES) == 5


def test_all_original_seeds_are_unchanged_and_unique() -> None:
    assert [row[4] for row in a1.EXPECTED_CASES] == list(range(171701, 171713))


def test_original_locked_window_is_unchanged() -> None:
    assert a1.LOCKED_NOUT == tuple(range(100, 117))


def test_amendment_preserves_scientific_contract() -> None:
    path = Path(__file__).resolve().parents[1] / "reference_data" / "mohammadzadeh_2012" / a1.AMENDMENT
    value = json.loads(path.read_text(encoding="utf-8"))
    contract = value["unchanged_scientific_contract"]
    assert contract["observation_B3_nout"] == [100, 108, 116]
    assert contract["independent_reference_B10_nout"] == [101, 102, 103, 104, 105, 109, 110, 111, 112, 113]
    assert contract["endpoint_or_gate_change"] is False
    assert contract["seed_replacement"] is False


def test_runner_fails_closed_and_checks_overlap() -> None:
    runner = (Path(__file__).resolve().parents[1] / "scripts" / "unity_mohammadzadeh_mv17b_a1_recover_array.sbatch").read_text(encoding="utf-8")
    assert "cmp -s" in runner
    assert "OVERLAP_MISMATCH" in runner
    assert "CONTROL[5]=4" in runner
    assert "original files are never overwritten" in runner.lower()
    assert "1,2,3,4,9,10,11" in runner


def test_submitter_does_not_resubmit_complete_cases() -> None:
    submitter = (Path(__file__).resolve().parents[1] / "scripts" / "submit_mohammadzadeh_mv17b_a1_unity.sh").read_text(encoding="utf-8")
    assert "--array='1-4,9-11%4'" in submitter
    assert "reruns=7" in submitter
    assert "reused=5" in submitter


def test_postprocessor_adds_recovery_provenance() -> None:
    source = Path(a1.__file__).read_text(encoding="utf-8")
    assert "def augment_package" in source
    post = (Path(__file__).resolve().parents[1] / "scripts" / "unity_mohammadzadeh_mv17b_a1_post.sbatch").read_text(encoding="utf-8")
    assert "augment-package" in post
    assert "MV17B_A1_MECHANICAL_RECOVERY=true" in post


def test_lock_is_outcome_blind() -> None:
    source = Path(a1.__file__).read_text(encoding="utf-8")
    start = source.index("def lock_recovery")
    end = source.index("\ndef verify_ready", start)
    locked = source[start:end]
    for forbidden in ("target_qy", "selected_qy", "nrmse", "prediction_fields"):
        assert forbidden not in locked


def test_mechanical_lock_accepts_only_the_observed_incomplete_set() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "frozen").mkdir()
        (root / "frozen" / "FROZEN_MODEL_PASS").write_text("pass\n")
        (root / "frozen" / "mv17b_frozen_cylinder_model.npz").write_bytes(b"model")
        (root / "frozen" / "mv17b_fresh_cylinder_confirmation_protocol.json").write_text("{}\n")
        (root / "SUBMISSION.env").write_text("MV17B=frozen\n")
        (root / a1.CODE_MANIFEST).write_text("code=frozen\n")
        for _, case_id, pair_id, role, seed, complete in a1.EXPECTED_CASES:
            case = root / "campaign" / "cases" / case_id
            moments = case / "results" / "moments"
            moments.mkdir(parents=True)
            (case / "CASE_METADATA.json").write_text(json.dumps({
                "case_id": case_id, "pair_id": pair_id, "role": role, "seed": seed,
            }))
            last = 116 if complete else 115
            status = "LOCKED_WINDOW_COMPLETE" if complete else "SOLVER_EXITED_BEFORE_LOCKED_WINDOW"
            (case / "results" / "RUN_STATUS.env").write_text(
                f"STATUS={status}\nLAST_NOUT={last}\nLAST_TUD=11.0\n"
            )
            for nout in range(100, last + 1):
                (moments / f"MV11_MOMENTS_NOUT{nout:04d}.DAT").write_text(f"{nout}\n")
        amendment = Path(__file__).resolve().parents[1] / "reference_data" / "mohammadzadeh_2012" / a1.AMENDMENT
        lock = a1.lock_recovery(root, amendment)
        assert lock["recovery_indices"] == list(a1.RECOVERY_INDICES)
        assert lock["q_y_values_accessed"] is False
        assert lock["scientific_contract_changed"] is False


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"MV17B_A1_MECHANICAL_RECOVERY_TESTS_PASS count={len(tests)}")


if __name__ == "__main__":
    main()
