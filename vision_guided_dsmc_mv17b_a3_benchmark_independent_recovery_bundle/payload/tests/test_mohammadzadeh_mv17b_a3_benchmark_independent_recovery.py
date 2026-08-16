#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

from vgdsmc import mohammadzadeh_mv17b_a3_benchmark_independent_recovery as a3


ROOT = Path(__file__).resolve().parents[1]


def test_only_originally_incomplete_indices_are_recovered() -> None:
    assert a3.RECOVERY_INDICES == (1, 2, 3, 4, 9, 10, 11)
    assert len(a3.EXPECTED_CASES) == 12
    assert sum(row[-1] for row in a3.EXPECTED_CASES) == 5


def test_all_original_seeds_are_unchanged_and_unique() -> None:
    assert [row[4] for row in a3.EXPECTED_CASES] == list(range(171701, 171713))


def test_original_locked_window_is_unchanged() -> None:
    assert a3.LOCKED_NOUT == tuple(range(100, 117))


def test_amendment_preserves_scientific_contract() -> None:
    path = ROOT / "reference_data" / "mohammadzadeh_2012" / a3.AMENDMENT
    value = json.loads(path.read_text(encoding="utf-8"))
    recovery = value["recovery_contract"]
    contract = value["unchanged_scientific_contract"]
    assert recovery["fresh_IRUN_remains_3"] is True
    assert recovery["source_replacement_count"] == 2
    assert recovery["fixed_endpoint_nout"] == 116
    assert recovery["benchmark_values_used_for_termination"] is False
    assert recovery["physical_parameters_changed"] is False
    assert contract["observation_B3_nout"] == [100, 108, 116]
    assert contract["independent_reference_B10_nout"] == [
        101, 102, 103, 104, 105, 109, 110, 111, 112, 113
    ]
    assert contract["endpoint_or_gate_change"] is False
    assert contract["seed_replacement"] is False


def test_source_patch_makes_endpoint_benchmark_independent() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "source.F90"
        output = root / "output.F90"
        report = root / "report.json"
        source.write_text(
            "IF (IRUN == 3) WRITE(*,*) 'fresh'\n"
            "WRITE(*,*) 'send 3 to start a new run'\n"
            "CONV_PAR=0\n"
            "DO WHILE (TIME < TLIM)\n"
            "IF (CONV_PAR.EQ.1) GO TO 2512\n"
            "CALL STEP\n"
            "END DO\n"
            "IF (SUM_heat_ER.LE.Heat_er) CONV_PAR=1\n",
            encoding="utf-8",
        )
        value = a3.patch_locked_window_source(source, output, report)
        patched = output.read_text(encoding="utf-8")
        assert a3.ORIGINAL_LOOP not in patched
        assert patched.count(a3.FIXED_LOOP) == 1
        assert patched.count(a3.LOOP_PATCH_MARKER) == 2
        assert patched.count(a3.STOP_PATCH_MARKER) == 2
        assert a3.BENCHMARK_STOP not in patched
        assert value["replacement_count"] == 2
        assert value["target_dependent_benchmark_stop_disabled"] is True
        assert value["physical_parameters_changed"] is False


def test_source_patch_rejects_nonunique_anchor() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "source.F90"
        source.write_text(
            "IF (IRUN == 3) CONTINUE\n"
            "send 3 to start a new run\n"
            f"{a3.ORIGINAL_LOOP}\n{a3.ORIGINAL_LOOP}\n",
            encoding="utf-8",
        )
        try:
            a3.patch_locked_window_source(source, root / "out.F90", root / "report.json")
        except ValueError as error:
            assert "exactly one" in str(error)
        else:
            raise AssertionError("nonunique source anchor was accepted")


def test_source_patch_rejects_missing_benchmark_stop_anchor() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "source.F90"
        source.write_text(
            "IF (IRUN == 3) CONTINUE\n"
            "send 3 to start a new run\n"
            "CONV_PAR=0\n"
            f"{a3.ORIGINAL_LOOP}\n"
            f"{a3.EARLY_EXIT}\n",
            encoding="utf-8",
        )
        try:
            a3.patch_locked_window_source(source, root / "out.F90", root / "report.json")
        except ValueError as error:
            assert "benchmark stop setter" in str(error)
        else:
            raise AssertionError("missing benchmark stop anchor was accepted")


def test_runner_keeps_fresh_irun3_and_fails_closed() -> None:
    runner = (ROOT / "scripts" / "unity_mohammadzadeh_mv17b_a3_recover_array.sbatch").read_text(
        encoding="utf-8"
    )
    assert 'test "${CONTROL[5]}" = "3"' in runner
    assert "CONTROL[5]=4" not in runner
    assert "ds2v_bird_mv17b_a3" in runner
    assert "cmp -s" in runner
    assert "MV17B_A3_OVERLAP_MISMATCH" in runner
    assert "original files are never overwritten" in runner.lower()
    assert "1,2,3,4,9,10,11" in runner
    assert 'if [[ -s "${STAGED_MOMENTS}/MV11_MOMENTS_NOUT0116.DAT" ]]' not in runner
    assert 'RECOVERY_DIR="${CAMPAIGN}/recovery_a3/${CASE_ID}"' in runner


def test_prepare_builds_only_patched_source() -> None:
    prepare = (ROOT / "scripts" / "unity_mohammadzadeh_mv17b_a3_prepare.sbatch").read_text(
        encoding="utf-8"
    )
    assert "patch-source" in prepare
    assert "DS2V_BIRD_MV17B_A3.F90" in prepare
    assert "DO WHILE (NOUT < 116)" in prepare
    assert "MV17B_A3_BENCHMARK_STOP_DISABLED_BEGIN" in prepare
    assert "IF (SUM_heat_ER.LE.Heat_er) CONV_PAR=1" in prepare
    assert 'sed -n \'6p\'' in prepare
    assert "-fcheck=bounds" in prepare
    assert "patch-source +" not in prepare
    assert "gfortran +" not in prepare


def test_submitter_uses_prepare_dependency_and_seven_cases() -> None:
    submitter = (ROOT / "scripts" / "submit_mohammadzadeh_mv17b_a3_unity.sh").read_text(
        encoding="utf-8"
    )
    assert "--array='1-4,9-11%4'" in submitter
    assert 'afterok:${PREP_JOB}' in submitter
    assert 'afterok:${RECOVERY_JOB}' in submitter
    assert "reruns=7" in submitter
    assert "reused=5" in submitter
    assert "LAST_MOHAMMADZADEH_MV17B_A2_LOCKED_WINDOW_RECOVERY_JOB.env" in submitter


def test_postprocessor_adds_a1_and_a2_provenance() -> None:
    source = Path(a3.__file__).read_text(encoding="utf-8")
    assert "def augment_package" in source
    assert "A1_failure_evidence" in source
    assert "A2_failure_evidence" in source
    post = (ROOT / "scripts" / "unity_mohammadzadeh_mv17b_a3_post.sbatch").read_text(
        encoding="utf-8"
    )
    assert "augment-package" in post
    assert "MV17B_A3_BENCHMARK_INDEPENDENT_RECOVERY=true" in post
    assert "MV17B_A1_FAILURE_PRESERVED=true" in post
    assert "MV17B_A2_FAILURE_PRESERVED=true" in post


def test_lock_is_outcome_blind() -> None:
    source = Path(a3.__file__).read_text(encoding="utf-8")
    start = source.index("def lock_recovery")
    end = source.index("\ndef verify_ready", start)
    locked = source[start:end]
    for forbidden in ("target_qy", "selected_qy", "nrmse", "prediction_fields"):
        assert forbidden not in locked


def test_lock_requires_exact_a1_and_a2_failures_and_unchanged_cases() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "frozen").mkdir()
        (root / "frozen" / "FROZEN_MODEL_PASS").write_text("pass\n")
        (root / "frozen" / "mv17b_frozen_cylinder_model.npz").write_bytes(b"model")
        (root / "frozen" / "mv17b_fresh_cylinder_confirmation_protocol.json").write_text("{}\n")
        (root / "SUBMISSION.env").write_text("MV17B=frozen\n")
        (root / a3.CODE_MANIFEST).write_text("code=frozen\n")
        a1_lock = root / a3.A1_LOCK_FILE
        a1_lock.write_text("{}\n")
        digest = hashlib.sha256(a1_lock.read_bytes()).hexdigest()
        (root / a3.A1_LOCK_MANIFEST).write_text(f"{digest}  {a1_lock.name}\n")
        (root / a3.A1_SUBMISSION).write_text("MV17B_A1_JOB_IDS=failed\n")
        a2_lock = root / a3.A2_LOCK_FILE
        a2_lock.write_text("{}\n")
        a2_digest = hashlib.sha256(a2_lock.read_bytes()).hexdigest()
        (root / a3.A2_LOCK_MANIFEST).write_text(f"{a2_digest}  {a2_lock.name}\n")
        (root / a3.A2_SUBMISSION).write_text("MV17B_A2_JOB_IDS=failed\n")
        build = root / "campaign" / "build"
        build.mkdir(parents=True)
        (build / a3.A2_SOURCE_PATCH_REPORT).write_text(json.dumps({
            "status": "MV17B_A2_source_patch_verified"
        }))
        (build / "DS2V_BIRD_MV17B_A2.F90").write_text(
            "DO WHILE ((TIME < TLIM) .OR. (NOUT < 116))\n"
            f"{a3.BENCHMARK_STOP}\n"
        )
        for _, case_id, pair_id, role, seed, complete in a3.EXPECTED_CASES:
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
            if not complete:
                recovery = root / "campaign" / "recovery" / case_id
                recovery.mkdir(parents=True)
                (recovery / "run.log").write_text(
                    "At line 1663 of file DS2V_BIRD_MV17B.F90\n"
                    "Fortran runtime error: End of file\n"
                )
                a2_recovery = root / "campaign" / "recovery_a2" / case_id
                (a2_recovery / "moments").mkdir(parents=True)
                (a2_recovery / "run.log").write_text("A2 exited before NOUT 116\n")
                for nout in range(100, last + 1):
                    (a2_recovery / "moments" / f"MV11_MOMENTS_NOUT{nout:04d}.DAT").write_text(
                        f"{nout}\n"
                    )
        amendment = ROOT / "reference_data" / "mohammadzadeh_2012" / a3.AMENDMENT
        lock = a3.lock_recovery(root, amendment)
        assert lock["recovery_indices"] == list(a3.RECOVERY_INDICES)
        assert lock["A1_failure_evidence"]["failed_case_count"] == 7
        assert lock["A2_failure_evidence"]["failed_case_count"] == 7
        assert lock["fresh_IRUN"] == 3
        assert lock["target_dependent_benchmark_stop_disabled"] is True
        assert lock["q_y_values_accessed"] is False
        assert lock["scientific_contract_changed"] is False


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"MV17B_A3_BENCHMARK_INDEPENDENT_RECOVERY_TESTS_PASS count={len(tests)}")


if __name__ == "__main__":
    main()
