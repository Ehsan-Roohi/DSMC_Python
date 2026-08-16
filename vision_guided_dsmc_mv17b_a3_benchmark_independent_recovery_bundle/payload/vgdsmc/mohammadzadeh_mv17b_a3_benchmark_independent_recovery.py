from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any
import zipfile


STAGE = "MV17B_A3_Mohammadzadeh_benchmark_independent_recovery"
AMENDMENT = "mv17b_a3_benchmark_independent_recovery_amendment.json"
LOCK_FILE = "MV17B_A3_BENCHMARK_INDEPENDENT_RECOVERY_LOCK.json"
LOCK_MANIFEST = "MV17B_A3_BENCHMARK_INDEPENDENT_RECOVERY_LOCK.sha256"
AMENDMENT_COPY = "MV17B_A3_BENCHMARK_INDEPENDENT_RECOVERY_AMENDMENT.json"
RECOVERY_PROVENANCE = "MV17B_A3_RECOVERY_PROVENANCE.json"
CODE_MANIFEST = "MV17B_A3_CODE_SHA256SUMS.txt"
SOURCE_PATCH_REPORT = "MV17B_A3_SOURCE_PATCH_REPORT.json"
A1_LOCK_FILE = "MV17B_A1_MECHANICAL_RECOVERY_LOCK.json"
A1_LOCK_MANIFEST = "MV17B_A1_MECHANICAL_RECOVERY_LOCK.sha256"
A1_SUBMISSION = "MV17B_A1_SUBMISSION.env"
A2_LOCK_FILE = "MV17B_A2_LOCKED_WINDOW_RECOVERY_LOCK.json"
A2_LOCK_MANIFEST = "MV17B_A2_LOCKED_WINDOW_RECOVERY_LOCK.sha256"
A2_SUBMISSION = "MV17B_A2_SUBMISSION.env"
A2_SOURCE_PATCH_REPORT = "MV17B_A2_SOURCE_PATCH_REPORT.json"
ORIGINAL_LOOP = "DO WHILE (TIME < TLIM)"
FIXED_LOOP = "DO WHILE (NOUT < 116)"
BENCHMARK_STOP = "IF (SUM_heat_ER.LE.Heat_er) CONV_PAR=1"
EARLY_EXIT = "IF (CONV_PAR.EQ.1) GO TO 2512"
LOOP_PATCH_MARKER = "MV17B_A3_FIXED_NOUT_LOOP"
STOP_PATCH_MARKER = "MV17B_A3_BENCHMARK_STOP_DISABLED"
EXPECTED_A1_FATAL = "Fortran runtime error: End of file"
EXPECTED_CASES = (
    (0, "pair_01_observation", "pair_01", "observation", 171701, True),
    (1, "pair_01_reference", "pair_01", "reference", 171702, False),
    (2, "pair_02_observation", "pair_02", "observation", 171703, False),
    (3, "pair_02_reference", "pair_02", "reference", 171704, False),
    (4, "pair_03_observation", "pair_03", "observation", 171705, False),
    (5, "pair_03_reference", "pair_03", "reference", 171706, True),
    (6, "pair_04_observation", "pair_04", "observation", 171707, True),
    (7, "pair_04_reference", "pair_04", "reference", 171708, True),
    (8, "pair_05_observation", "pair_05", "observation", 171709, True),
    (9, "pair_05_reference", "pair_05", "reference", 171710, False),
    (10, "pair_06_observation", "pair_06", "observation", 171711, False),
    (11, "pair_06_reference", "pair_06", "reference", 171712, False),
)
RECOVERY_INDICES = tuple(row[0] for row in EXPECTED_CASES if not row[-1])
LOCKED_NOUT = tuple(range(100, 117))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _moment_nout(path: Path) -> int:
    return int(path.stem.rsplit("NOUT", 1)[1])


def _active_conv_par_one_lines(text: str) -> list[str]:
    return [
        line.strip()
        for line in text.splitlines()
        if not line.lstrip().startswith("!")
        and re.search(r"\bCONV_PAR\s*=\s*1\b", line, flags=re.IGNORECASE)
    ]


def patch_locked_window_source(source: Path, output: Path, report: Path) -> dict[str, Any]:
    """Make acquisition end at the locked NOUT, independent of benchmark data."""

    source = Path(source).resolve()
    output = Path(output).resolve()
    report = Path(report).resolve()
    text = source.read_text(encoding="utf-8", errors="strict")
    if LOOP_PATCH_MARKER in text or STOP_PATCH_MARKER in text:
        raise ValueError("source was already patched for MV17B-A3")
    if text.count(ORIGINAL_LOOP) != 1:
        raise ValueError(
            f"expected exactly one locked-loop anchor, found {text.count(ORIGINAL_LOOP)}"
        )
    if "IF (IRUN == 3)" not in text or "send 3 to start a new run" not in text:
        raise ValueError("fresh IRUN=3 contract is absent from source")
    if text.count(BENCHMARK_STOP) != 1:
        raise ValueError(
            "expected exactly one active benchmark stop setter, "
            f"found {text.count(BENCHMARK_STOP)}"
        )
    active_setters = _active_conv_par_one_lines(text)
    if active_setters != [BENCHMARK_STOP]:
        raise ValueError(f"unexpected active CONV_PAR=1 setters: {active_setters}")
    if text.count(EARLY_EXIT) != 1 or "CONV_PAR=0" not in text:
        raise ValueError("CONV_PAR initialization or main-loop early-exit anchor is absent")
    loop_replacement = (
        f"! {LOOP_PATCH_MARKER}_BEGIN\n"
        f"{FIXED_LOOP}\n"
        f"! {LOOP_PATCH_MARKER}_END"
    )
    stop_replacement = (
        f"! {STOP_PATCH_MARKER}_BEGIN\n"
        "! Legacy benchmark-dependent CONV_PAR assignment is disabled.\n"
        "! Fixed acquisition ends only after the locked NOUT=116 block is written.\n"
        f"! {STOP_PATCH_MARKER}_END"
    )
    patched = text.replace(ORIGINAL_LOOP, loop_replacement, 1)
    patched = patched.replace(BENCHMARK_STOP, stop_replacement, 1)
    if patched.count(FIXED_LOOP) != 1:
        raise ValueError("fixed-NOUT loop patch was not unique")
    if patched.count(LOOP_PATCH_MARKER) != 2 or patched.count(STOP_PATCH_MARKER) != 2:
        raise ValueError("MV17B-A3 source markers are not unique")
    if BENCHMARK_STOP in patched or _active_conv_par_one_lines(patched):
        raise ValueError("active benchmark-dependent stop setter remains in patched source")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(patched, encoding="utf-8")
    value = {
        "stage": STAGE,
        "status": "MV17B_A3_source_patch_verified",
        "source": str(source),
        "output": str(output),
        "source_sha256": _sha256(source),
        "output_sha256": _sha256(output),
        "replacement_count": 2,
        "original_loop": ORIGINAL_LOOP,
        "patched_loop": FIXED_LOOP,
        "disabled_benchmark_stop": BENCHMARK_STOP,
        "retained_inert_early_exit": EARLY_EXIT,
        "locked_last_nout": 116,
        "fresh_IRUN": 3,
        "physical_parameters_changed": False,
        "collision_model_changed": False,
        "sampling_schedule_changed": False,
        "fixed_endpoint_acquisition": True,
        "target_dependent_benchmark_stop_disabled": True,
        "benchmark_values_not_used_for_termination": True,
    }
    _atomic_json(report, value)
    return value


def mechanical_inventory(output_root: Path) -> list[dict[str, Any]]:
    root = Path(output_root).resolve()
    rows: list[dict[str, Any]] = []
    for index, case_id, pair_id, role, seed, originally_complete in EXPECTED_CASES:
        case = root / "campaign" / "cases" / case_id
        metadata = _json(case / "CASE_METADATA.json")
        status = _env(case / "results" / "RUN_STATUS.env")
        if metadata.get("case_id") != case_id or metadata.get("pair_id") != pair_id:
            raise ValueError(f"case identity changed: {case_id}")
        if metadata.get("role") != role or int(metadata.get("seed", -1)) != seed:
            raise ValueError(f"case seed or role changed: {case_id}")
        moments = sorted(
            (case / "results" / "moments").glob("MV11_MOMENTS_NOUT*.DAT"),
            key=_moment_nout,
        )
        nout = [_moment_nout(path) for path in moments]
        if not nout or nout != list(range(min(nout), max(nout) + 1)):
            raise ValueError(f"stored NOUT sequence is not contiguous: {case_id}")
        rows.append(
            {
                "array_index": index,
                "case_id": case_id,
                "pair_id": pair_id,
                "role": role,
                "seed": seed,
                "status": status.get("STATUS"),
                "last_nout": int(status.get("LAST_NOUT", "0")),
                "last_tU_over_D": float(status.get("LAST_TUD", "nan")),
                "stored_first_nout": min(nout),
                "stored_last_nout": max(nout),
                "stored_block_count": len(nout),
                "originally_complete": originally_complete,
                "requires_recovery": index in RECOVERY_INDICES,
            }
        )
    return rows


def a1_failure_evidence(output_root: Path) -> dict[str, Any]:
    root = Path(output_root).resolve()
    lock = root / A1_LOCK_FILE
    manifest = root / A1_LOCK_MANIFEST
    submission = root / A1_SUBMISSION
    if not lock.is_file() or not manifest.is_file() or not submission.is_file():
        raise FileNotFoundError("MV17B-A1 lock or submission evidence is absent")
    expected = manifest.read_text(encoding="utf-8").split()[0]
    if _sha256(lock) != expected:
        raise ValueError("MV17B-A1 lock hash changed")
    logs: dict[str, dict[str, Any]] = {}
    for index, case_id, _, _, _, complete in EXPECTED_CASES:
        if complete:
            continue
        result = root / "campaign" / "cases" / case_id / "results"
        if (result / "MV17B_A1_RECOVERY_PASS").exists():
            raise ValueError(f"A1 unexpectedly modified original result: {case_id}")
        log = root / "campaign" / "recovery" / case_id / "run.log"
        if not log.is_file():
            raise FileNotFoundError(log)
        content = log.read_text(encoding="utf-8", errors="replace")
        if EXPECTED_A1_FATAL not in content or "line 1663" not in content:
            raise ValueError(f"unexpected A1 failure mode: {case_id}")
        logs[case_id] = {
            "array_index": index,
            "sha256": _sha256(log),
            "size_bytes": log.stat().st_size,
            "failure": "invalid_IRUN_4_reprompt_then_stdin_EOF_at_source_line_1663",
        }
    return {
        "A1_lock_sha256": _sha256(lock),
        "A1_submission_sha256": _sha256(submission),
        "failed_case_count": len(logs),
        "failed_cases": logs,
        "A1_original_results_modified": False,
    }


def a2_failure_evidence(output_root: Path) -> dict[str, Any]:
    """Verify that A2 retained the active benchmark stop and recovered no case."""

    root = Path(output_root).resolve()
    lock = root / A2_LOCK_FILE
    manifest = root / A2_LOCK_MANIFEST
    submission = root / A2_SUBMISSION
    patch_report = root / "campaign" / "build" / A2_SOURCE_PATCH_REPORT
    patched_source = root / "campaign" / "build" / "DS2V_BIRD_MV17B_A2.F90"
    for required in (lock, manifest, submission, patch_report, patched_source):
        if not required.is_file():
            raise FileNotFoundError(required)
    expected = manifest.read_text(encoding="utf-8").split()[0]
    if _sha256(lock) != expected:
        raise ValueError("MV17B-A2 lock hash changed")
    report = _json(patch_report)
    if report.get("status") != "MV17B_A2_source_patch_verified":
        raise ValueError("MV17B-A2 source patch report is invalid")
    source_text = patched_source.read_text(encoding="utf-8", errors="strict")
    if source_text.count("DO WHILE ((TIME < TLIM) .OR. (NOUT < 116))") != 1:
        raise ValueError("MV17B-A2 extended loop is absent")
    if source_text.count(BENCHMARK_STOP) != 1:
        raise ValueError("MV17B-A2 did not retain the active benchmark stop setter")
    if _active_conv_par_one_lines(source_text) != [BENCHMARK_STOP]:
        raise ValueError("MV17B-A2 CONV_PAR setter inventory changed")

    failed: dict[str, dict[str, Any]] = {}
    for index, case_id, _, _, _, complete in EXPECTED_CASES:
        if complete:
            continue
        original = root / "campaign" / "cases" / case_id / "results"
        if (original / "MV17B_A2_RECOVERY_PASS").exists():
            raise ValueError(f"A2 unexpectedly modified original result: {case_id}")
        recovery = root / "campaign" / "recovery_a2" / case_id
        log = recovery / "run.log"
        moments = sorted((recovery / "moments").glob("MV11_MOMENTS_NOUT*.DAT"), key=_moment_nout)
        if not log.is_file() or not moments:
            raise FileNotFoundError(f"MV17B-A2 failure evidence is incomplete: {case_id}")
        nout = [_moment_nout(path) for path in moments]
        if 116 in nout or max(nout) > 115:
            raise ValueError(f"MV17B-A2 unexpectedly reached the locked endpoint: {case_id}")
        failed[case_id] = {
            "array_index": index,
            "run_log_sha256": _sha256(log),
            "run_log_size_bytes": log.stat().st_size,
            "staged_first_nout": min(nout),
            "staged_last_nout": max(nout),
            "staged_block_count": len(nout),
            "locked_nout_116_absent": True,
        }
    return {
        "A2_lock_sha256": _sha256(lock),
        "A2_submission_sha256": _sha256(submission),
        "A2_source_patch_report_sha256": _sha256(patch_report),
        "A2_patched_source_sha256": _sha256(patched_source),
        "failed_case_count": len(failed),
        "failed_cases": failed,
        "root_cause": "active_legacy_benchmark_setter_raised_CONV_PAR_before_NOUT_116",
        "A2_original_results_modified": False,
    }


def lock_recovery(output_root: Path, amendment_path: Path) -> dict[str, Any]:
    root = Path(output_root).resolve()
    amendment_path = Path(amendment_path).resolve()
    if (root / "analysis").exists():
        raise ValueError("MV17B analysis already exists; outcome-blind recovery is forbidden")
    for required in (
        root / "frozen" / "FROZEN_MODEL_PASS",
        root / "frozen" / "mv17b_frozen_cylinder_model.npz",
        root / "frozen" / "mv17b_fresh_cylinder_confirmation_protocol.json",
        root / "SUBMISSION.env",
        root / CODE_MANIFEST,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    amendment = _json(amendment_path)
    if amendment.get("status") != "locked_after_A2_mechanical_failure_before_any_MV17B_prediction_or_target_construction":
        raise ValueError("MV17B-A3 amendment is not locked")
    inventory = mechanical_inventory(root)
    found = tuple(row["array_index"] for row in inventory if row["last_nout"] < 116)
    if found != RECOVERY_INDICES:
        raise ValueError(f"mechanical recovery set changed: {found}")
    for row in inventory:
        expected_status = (
            "LOCKED_WINDOW_COMPLETE" if row["originally_complete"]
            else "SOLVER_EXITED_BEFORE_LOCKED_WINDOW"
        )
        if row["status"] != expected_status:
            raise ValueError(f"unexpected pre-A3 status: {row['case_id']}")
    a1_evidence = a1_failure_evidence(root)
    a2_evidence = a2_failure_evidence(root)
    copied_amendment = root / AMENDMENT_COPY
    if copied_amendment.exists():
        if copied_amendment.read_bytes() != amendment_path.read_bytes():
            raise FileExistsError("existing A3 amendment differs")
    else:
        copied_amendment.write_bytes(amendment_path.read_bytes())
    value = {
        "stage": STAGE,
        "status": "MV17B_A3_locked_before_recovery_and_before_prediction",
        "scientific_classification": amendment["scientific_classification"],
        "output_root": str(root),
        "amendment_sha256": _sha256(copied_amendment),
        "original_submission_sha256": _sha256(root / "SUBMISSION.env"),
        "recovery_code_manifest_sha256": _sha256(root / CODE_MANIFEST),
        "original_protocol_sha256": _sha256(
            root / "frozen" / "mv17b_fresh_cylinder_confirmation_protocol.json"
        ),
        "frozen_model_sha256": _sha256(
            root / "frozen" / "mv17b_frozen_cylinder_model.npz"
        ),
        "A1_failure_evidence": a1_evidence,
        "A2_failure_evidence": a2_evidence,
        "recovery_indices": list(RECOVERY_INDICES),
        "mechanical_inventory": inventory,
        "fresh_IRUN": 3,
        "fixed_endpoint_acquisition": True,
        "target_dependent_benchmark_stop_disabled": True,
        "source_replacement_count": 2,
        "q_y_values_accessed": False,
        "model_predictions_constructed": False,
        "reference_targets_constructed": False,
        "seed_replacement": False,
        "scientific_contract_changed": False,
    }
    lock = root / LOCK_FILE
    if lock.exists():
        if _json(lock) != value:
            raise FileExistsError("existing MV17B-A3 lock differs")
    else:
        _atomic_json(lock, value)
    (root / LOCK_MANIFEST).write_text(f"{_sha256(lock)}  {lock.name}\n", encoding="utf-8")
    return value


def verify_ready(output_root: Path) -> dict[str, Any]:
    root = Path(output_root).resolve()
    if (root / "analysis").exists():
        raise ValueError("analysis directory existed before A3 verification")
    lock = root / LOCK_FILE
    manifest = root / LOCK_MANIFEST
    if not lock.is_file() or not manifest.is_file():
        raise FileNotFoundError("MV17B-A3 lock is absent")
    if _sha256(lock) != manifest.read_text(encoding="utf-8").split()[0]:
        raise ValueError("MV17B-A3 lock hash changed")
    rows = mechanical_inventory(root)
    for row in rows:
        case = root / "campaign" / "cases" / row["case_id"]
        status = _env(case / "results" / "RUN_STATUS.env")
        if status.get("STATUS") != "LOCKED_WINDOW_COMPLETE":
            raise ValueError(f"trajectory remains incomplete: {row['case_id']}")
        for nout in LOCKED_NOUT:
            path = case / "results" / "moments" / f"MV11_MOMENTS_NOUT{nout:04d}.DAT"
            if not path.is_file():
                raise FileNotFoundError(path)
        if row["array_index"] in RECOVERY_INDICES:
            if not (case / "results" / "MV17B_A3_RECOVERY_PASS").is_file():
                raise FileNotFoundError(f"A3 recovery attestation absent: {row['case_id']}")
    recovery_files: list[Path] = [
        root / AMENDMENT_COPY,
        root / LOCK_FILE,
        root / LOCK_MANIFEST,
        root / "MV17B_A3_SUBMISSION.env",
        root / CODE_MANIFEST,
        root / A1_LOCK_FILE,
        root / A1_LOCK_MANIFEST,
        root / A1_SUBMISSION,
        root / A2_LOCK_FILE,
        root / A2_LOCK_MANIFEST,
        root / A2_SUBMISSION,
        root / "campaign" / "build" / A2_SOURCE_PATCH_REPORT,
        root / "campaign" / "build" / "DS2V_BIRD_MV17B_A2.F90",
        root / "campaign" / "build" / SOURCE_PATCH_REPORT,
        root / "campaign" / "build" / "DS2V_BIRD_MV17B_A3.F90",
        root / "campaign" / "build" / "ds2v_bird_mv17b_a3",
        root / "campaign" / "build" / "MV17B_A3_BUILD_CONFIGURATION.txt",
    ]
    for row in rows:
        if row["array_index"] not in RECOVERY_INDICES:
            continue
        case = root / "campaign" / "cases" / row["case_id"]
        recovery = root / "campaign" / "recovery_a3" / row["case_id"]
        recovery_files.extend(
            (
                case / "results" / "RUN_STATUS.pre_mv17b_a3.env",
                case / "results" / "MV17B_A3_RECOVERY_METADATA.json",
                case / "results" / "MV17B_A3_RECOVERY_PASS",
                recovery / "MV17B_A3_RECOVERY_METADATA.json",
                recovery / "INPUT_EXECUTABLE_SHA256.txt",
                recovery / "RNG_SEED_USED.txt",
                recovery / "run.log",
            )
        )
    missing = [str(path) for path in recovery_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"A3 recovery provenance is incomplete: {missing}")
    provenance = {
        "stage": STAGE,
        "status": "MV17B_A3_recovery_provenance_verified_before_analysis",
        "files": {
            str(path.relative_to(root)): {
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in recovery_files
        },
        "A1_failure_evidence": a1_failure_evidence(root),
        "A2_failure_evidence": a2_failure_evidence(root),
        "recovered_array_indices": list(RECOVERY_INDICES),
        "same_seeds": True,
        "fresh_IRUN": 3,
        "fixed_endpoint_acquisition": True,
        "target_dependent_benchmark_stop_disabled": True,
        "overlap_byte_identity_required": True,
        "scientific_contract_changed": False,
    }
    _atomic_json(root / RECOVERY_PROVENANCE, provenance)
    return {
        "stage": STAGE,
        "status": "MV17B_A3_all_original_locked_blocks_ready",
        "trajectory_count": len(rows),
        "recovered_trajectory_count": len(RECOVERY_INDICES),
        "complete_trajectory_count_reused": len(rows) - len(RECOVERY_INDICES),
        "original_scientific_contract_preserved": True,
        "DSMC_reruns": len(RECOVERY_INDICES),
        "seed_replacement": False,
        "neural_retraining": False,
        "fresh_parameter_selection": False,
    }


def augment_package(output_root: Path, archive: Path) -> dict[str, Any]:
    root = Path(output_root).resolve()
    archive = Path(archive).resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    provenance_path = root / RECOVERY_PROVENANCE
    provenance = _json(provenance_path)
    if provenance.get("status") != "MV17B_A3_recovery_provenance_verified_before_analysis":
        raise ValueError("A3 recovery provenance was not verified")
    paths = [root / relative for relative in provenance["files"]]
    paths.extend(
        root / "campaign" / "recovery" / case_id / "run.log"
        for _, case_id, _, _, _, complete in EXPECTED_CASES
        if not complete
    )
    paths.extend(
        root / "campaign" / "recovery_a2" / case_id / "run.log"
        for _, case_id, _, _, _, complete in EXPECTED_CASES
        if not complete
    )
    paths.append(provenance_path)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    with zipfile.ZipFile(archive, "a", zipfile.ZIP_DEFLATED, compresslevel=9) as stream:
        existing = set(stream.namelist())
        for path in paths:
            relative = str(path.relative_to(root))
            if relative in existing:
                archived_digest = hashlib.sha256(stream.read(relative)).hexdigest()
                if archived_digest != _sha256(path):
                    raise ValueError(f"A3 provenance archive collision: {relative}")
                continue
            stream.write(path, arcname=relative)
    digest = _sha256(archive)
    archive.with_suffix(archive.suffix + ".sha256").write_text(
        f"{digest}  {archive.name}\n", encoding="utf-8"
    )
    return_path = root / "analysis" / "return.json"
    returned = _json(return_path)
    returned.update(
        {
            "archive_sha256": digest,
            "MV17B_A3_benchmark_independent_recovery": True,
            "A1_failure_preserved": True,
            "A2_failure_preserved": True,
            "fixed_endpoint_nout": 116,
            "target_dependent_benchmark_stop_disabled": True,
            "recovered_same_seed_trajectories": len(RECOVERY_INDICES),
            "complete_trajectories_reused": len(EXPECTED_CASES) - len(RECOVERY_INDICES),
            "scientific_contract_changed": False,
        }
    )
    _atomic_json(return_path, returned)
    return {
        "stage": STAGE,
        "status": "MV17B_A3_provenance_appended_to_result_archive",
        "archive": str(archive),
        "archive_sha256": digest,
        "provenance_file_count": len(paths),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    lock = commands.add_parser("lock")
    lock.add_argument("--output-root", type=Path, required=True)
    lock.add_argument("--amendment", type=Path, required=True)
    patch = commands.add_parser("patch-source")
    patch.add_argument("--source", type=Path, required=True)
    patch.add_argument("--output", type=Path, required=True)
    patch.add_argument("--report", type=Path, required=True)
    ready = commands.add_parser("verify-ready")
    ready.add_argument("--output-root", type=Path, required=True)
    augment = commands.add_parser("augment-package")
    augment.add_argument("--output-root", type=Path, required=True)
    augment.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "lock":
        value = lock_recovery(args.output_root, args.amendment)
    elif args.command == "patch-source":
        value = patch_locked_window_source(args.source, args.output, args.report)
    elif args.command == "verify-ready":
        value = verify_ready(args.output_root)
    else:
        value = augment_package(args.output_root, args.archive)
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
