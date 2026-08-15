"""Evaluate frozen MV15C B3 q_y from completed references after a QC warning.

MV15C generated all eight preregistered DSMC trajectories, but its Slurm
chain stopped before prediction because two trajectories missed a stochastic
temperature-extremum stationarity z gate.  The trajectories themselves are
complete and pass their mechanical, provenance, and finite-field checks.

This module implements an outcome-blind amendment: before any MV15C model
prediction or cross-seed target exists, it freezes the original QC failures,
uses every completed trajectory without seed replacement, leaves the B3
predictor and all q_y gates unchanged, and reports the q_y result separately
from the original reference-QC outcome.  It never upgrades this analysis to
the unamended preregistered confirmation.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import zipfile

try:
    import numpy as np
except ImportError:  # pragma: no cover - Unity's selected runtime has NumPy.
    np = None  # type: ignore[assignment]


STAGE = "MV15C_A1_Mohammadzadeh_qy_evaluation"
STATUS = "outcome_blind_QC_amendment_before_predictions_or_targets"
AMENDMENT_FILE = "mv15c_a1_qy_evaluation_amendment.json"
QC_JSON = "mv15c_a1_reference_qc.json"
QC_CSV = "mv15c_a1_reference_qc.csv"
LOCK_JSON = "mv15c_a1_amendment_lock.json"
LOCK_MANIFEST = "mv15c_a1_amendment_lock_manifest.json"
PREDICTION_ATTESTATION = "mv15c_a1_prediction_attestation.json"
PREDICTION_MANIFEST = "mv15c_a1_prediction_attestation_manifest.json"
ORIGINAL_QY_SUMMARY = "mv15c_original_qy_summary_before_a1_annotation.json"
ARTIFACT_MANIFEST = "mv15c_a1_artifact_manifest.json"
VERIFICATION_FILE = "mv15c_a1_verification.json"
RETURN_FILE = "mv15c_a1_return.json"
RESULT_POINTER = "LAST_MOHAMMADZADEH_MV15C_A1_QY_RESULT.env"

FRESH_TASKS = (
    ("kn0p1_u400", 151501),
    ("kn0p1_u400", 151502),
    ("kn0p1_u400", 151503),
    ("kn0p1_u400", 151504),
    ("kn0p08_u350", 151511),
    ("kn0p08_u350", 151512),
    ("kn0p08_u350", 151513),
    ("kn0p08_u350", 151514),
)

PREDICTION_OUTCOME_FILES = (
    "PREDICTION_LOCK_PASS",
    "locked_fresh_predictions.npz",
    "prediction_summary.json",
    "prediction_manifest.json",
    "fresh_source_audit.csv",
    "summary.json",
    "mv15c_fresh_qy_metrics.csv",
)


def _mv15c_module():
    from . import mohammadzadeh_mv15c_fresh_b3_confirmation as mv15c

    return mv15c


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if np is not None and isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _json_dumps(value: Any) -> str:
    return json.dumps(
        value,
        indent=2,
        sort_keys=True,
        allow_nan=False,
        default=_json_default,
    )


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_json_dumps(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def _verify_manifest(root: Path, name: str) -> dict[str, Any]:
    root = Path(root)
    path = root / name
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for relative, record in manifest["files"].items():
        candidate = root / relative
        if (
            not candidate.is_file()
            or candidate.stat().st_size != int(record["size_bytes"])
            or _sha256(candidate) != record["sha256"]
        ):
            raise ValueError(f"MV15C-A1 recursive verification failed: {candidate}")
    return manifest


def amendment_path() -> Path:
    path = (
        Path(__file__).resolve().parents[1]
        / "reference_data"
        / "mohammadzadeh_2012"
        / AMENDMENT_FILE
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def amendment_contract() -> dict[str, Any]:
    value = json.loads(amendment_path().read_text(encoding="utf-8"))
    if value.get("stage") != STAGE or value.get("status") != STATUS:
        raise ValueError("MV15C-A1 amendment is absent or unlocked")
    return value


def verify_contract() -> dict[str, Any]:
    mv15c = _mv15c_module()
    original = mv15c.verify_lock()
    tasks = tuple((str(condition), int(seed)) for condition, seed in mv15c.fresh_tasks())
    if tasks != FRESH_TASKS:
        raise ValueError("MV15C-A1 task matrix differs from locked MV15C")
    amendment = amendment_contract()
    return {
        "stage": STAGE,
        "status": "MV15C_A1_contract_verified",
        "amendment_sha256": _sha256(amendment_path()),
        "original_MV15C_protocol_sha256": original["protocol_sha256"],
        "trajectory_count": len(tasks),
        "DSMC_rerun_required": False,
        "B3_predictor_changed": False,
        "q_y_gates_changed": False,
        "seed_replacement_allowed": False,
        "classification": amendment["scientific_classification"],
    }


def _is_heat_flux_key(key: str) -> bool:
    lowered = key.lower()
    return "qy" in lowered or "heat_flux" in lowered


def _reference_record(output: Path, condition: str, seed: int) -> dict[str, Any]:
    mv15c = _mv15c_module()
    directory = output / "references" / condition / f"seed_{seed}"
    summary_path = directory / "summary.json"
    manifest_path = directory / "artifact_manifest.json"
    if not summary_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"incomplete MV15C reference directory: {directory}")
    _verify_manifest(directory, "artifact_manifest.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != mv15c.COMPLETE_REFERENCE_STATUS:
        raise ValueError(f"MV15C reference did not finish: {directory}")

    mechanical = dict(summary.get("mechanical_checks", {}))
    mechanics_without_stationarity = {
        key: bool(value)
        for key, value in mechanical.items()
        if key != "stationarity_pass"
    }
    if not mechanics_without_stationarity or not all(mechanics_without_stationarity.values()):
        raise ValueError(f"MV15C reference mechanics/provenance failed: {directory}")

    stationarity = dict(summary.get("stationarity", {}))
    checks = {key: bool(value) for key, value in stationarity.get("checks", {}).items()}
    non_qy_checks = {key: value for key, value in checks.items() if not _is_heat_flux_key(key)}
    qy_checks = {key: value for key, value in checks.items() if _is_heat_flux_key(key)}
    tracked = stationarity.get("tracked", {})
    temperature_rows: dict[str, dict[str, Any]] = {}
    for key, value in tracked.items():
        if str(key).startswith("temperature_"):
            temperature_rows[str(key)] = {
                name: value.get(name)
                for name in (
                    "first_half_mean",
                    "second_half_mean",
                    "drift",
                    "drift_standard_error",
                    "drift_z_score",
                    "relative_drift",
                    "max_abs_drift_z_score",
                )
            }
    relative_drifts = [
        abs(float(value["relative_drift"]))
        for value in temperature_rows.values()
        if value.get("relative_drift") is not None
    ]
    original_decision = str(summary.get("decision", ""))
    original_gate_pass = (
        original_decision == "accept_MV15C_fresh_reference_for_cross_seed_qy_analysis"
    )
    return {
        "condition": condition,
        "seed": int(seed),
        "directory": str(directory),
        "status": summary["status"],
        "original_decision": original_decision,
        "original_reference_gate_pass": original_gate_pass,
        "mechanics_and_provenance_pass": True,
        "mechanical_checks_without_stationarity": mechanics_without_stationarity,
        "original_non_qy_stationarity_checks": non_qy_checks,
        "original_qy_stationarity_diagnostics_not_used_for_inclusion": qy_checks,
        "original_failed_non_qy_checks": sorted(
            key for key, value in non_qy_checks.items() if not value
        ),
        "temperature_diagnostics": temperature_rows,
        "maximum_absolute_relative_temperature_drift": (
            max(relative_drifts) if relative_drifts else None
        ),
        "summary_sha256": _sha256(summary_path),
        "artifact_manifest_sha256": _sha256(manifest_path),
    }


def collect_reference_qc(output_root: Path) -> dict[str, Any]:
    output = Path(output_root).resolve()
    records = [
        _reference_record(output, condition, seed)
        for condition, seed in FRESH_TASKS
    ]
    held = [record for record in records if not record["original_reference_gate_pass"]]
    if len(records) != 8 or not all(record["mechanics_and_provenance_pass"] for record in records):
        raise ValueError("MV15C-A1 requires all eight complete mechanical/provenance-passing references")
    return {
        "stage": STAGE,
        "status": "complete_reference_QC_audit_before_q_y_prediction",
        "trajectory_count": len(records),
        "complete_mechanical_provenance_count": sum(
            bool(record["mechanics_and_provenance_pass"]) for record in records
        ),
        "original_reference_gate_pass_count": len(records) - len(held),
        "original_reference_gate_hold_count": len(held),
        "original_reference_gate_all_pass": not held,
        "held_condition_seed_pairs": [
            {"condition": record["condition"], "seed": record["seed"]}
            for record in held
        ],
        "records": records,
    }


def _write_qc_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "condition",
                "seed",
                "original_decision",
                "original_reference_gate_pass",
                "mechanics_and_provenance_pass",
                "failed_non_qy_checks",
                "maximum_absolute_relative_temperature_drift",
                "summary_sha256",
            )
        )
        for record in records:
            writer.writerow(
                (
                    record["condition"],
                    record["seed"],
                    record["original_decision"],
                    int(bool(record["original_reference_gate_pass"])),
                    int(bool(record["mechanics_and_provenance_pass"])),
                    ";".join(record["original_failed_non_qy_checks"]),
                    record["maximum_absolute_relative_temperature_drift"],
                    record["summary_sha256"],
                )
            )


def _prediction_outputs_present(output: Path) -> list[str]:
    return [name for name in PREDICTION_OUTCOME_FILES if (output / name).exists()]


def prepare_amendment(output_root: Path) -> dict[str, Any]:
    output = Path(output_root).resolve()
    if not output.is_dir():
        raise FileNotFoundError(output)
    if (output / LOCK_MANIFEST).is_file():
        return verify_amendment_lock(output)

    present = _prediction_outputs_present(output)
    if present:
        raise RuntimeError(
            "MV15C-A1 amendment must precede every q_y prediction/target; found "
            + ", ".join(present)
        )
    for required in ("submission_lock.json", "source_lock_manifest.json"):
        if not (output / required).is_file():
            raise FileNotFoundError(output / required)
    _verify_manifest(output, "source_lock_manifest.json")
    verify_contract()
    qc = collect_reference_qc(output)
    if qc["original_reference_gate_all_pass"]:
        raise ValueError("MV15C-A1 is unnecessary because the original reference QC passed")

    copied_amendment = output / AMENDMENT_FILE
    copied_amendment.write_bytes(amendment_path().read_bytes())
    _atomic_json(output / QC_JSON, qc)
    _write_qc_csv(output / QC_CSV, qc["records"])
    lock = {
        "stage": STAGE,
        "status": "MV15C_A1_locked_before_model_predictions_and_cross_seed_targets",
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_classification": amendment_contract()["scientific_classification"],
        "output_root": str(output),
        "amendment_sha256": _sha256(copied_amendment),
        "original_submission_lock_sha256": _sha256(output / "submission_lock.json"),
        "original_source_lock_manifest_sha256": _sha256(output / "source_lock_manifest.json"),
        "reference_QC_sha256": _sha256(output / QC_JSON),
        "reference_QC_csv_sha256": _sha256(output / QC_CSV),
        "reference_summary_hashes": {
            f"{record['condition']}/seed_{record['seed']}": {
                "summary_sha256": record["summary_sha256"],
                "artifact_manifest_sha256": record["artifact_manifest_sha256"],
            }
            for record in qc["records"]
        },
        "original_reference_gate_all_pass": qc["original_reference_gate_all_pass"],
        "original_reference_gate_hold_count": qc["original_reference_gate_hold_count"],
        "all_eight_complete_mechanical_provenance_references_included": True,
        "prediction_outcome_files_present_before_amendment": [],
        "model_predictions_seen_before_amendment": False,
        "cross_seed_targets_seen_before_amendment": False,
        "B3_predictor_or_weights_changed": False,
        "q_y_acceptance_gates_changed": False,
        "fresh_seed_replacement": False,
        "DSMC_rerun": False,
    }
    _atomic_json(output / LOCK_JSON, lock)
    lock_files = (AMENDMENT_FILE, QC_JSON, QC_CSV, LOCK_JSON)
    _atomic_json(
        output / LOCK_MANIFEST,
        {
            "stage": STAGE,
            "files": {
                name: {
                    "sha256": _sha256(output / name),
                    "size_bytes": (output / name).stat().st_size,
                }
                for name in lock_files
            },
        },
    )
    return verify_amendment_lock(output)


def verify_amendment_lock(output_root: Path) -> dict[str, Any]:
    output = Path(output_root).resolve()
    _verify_manifest(output, LOCK_MANIFEST)
    lock = json.loads((output / LOCK_JSON).read_text(encoding="utf-8"))
    if lock.get("status") != "MV15C_A1_locked_before_model_predictions_and_cross_seed_targets":
        raise ValueError("MV15C-A1 lock status is invalid")
    if _sha256(amendment_path()) != lock["amendment_sha256"]:
        raise ValueError("MV15C-A1 installed amendment differs from the frozen output copy")
    for key, hashes in lock["reference_summary_hashes"].items():
        directory = output / "references" / key
        if (
            _sha256(directory / "summary.json") != hashes["summary_sha256"]
            or _sha256(directory / "artifact_manifest.json")
            != hashes["artifact_manifest_sha256"]
        ):
            raise ValueError(f"MV15C reference changed after A1 lock: {directory}")
        _verify_manifest(directory, "artifact_manifest.json")
    return lock


def run_prediction(output_root: Path, *, batch_size: int) -> dict[str, Any]:
    output = Path(output_root).resolve()
    lock = verify_amendment_lock(output)
    present = _prediction_outputs_present(output)
    if present:
        raise RuntimeError("refusing to overwrite an existing MV15C prediction: " + ", ".join(present))
    mv15c = _mv15c_module()
    original = mv15c.run_prediction_stage(output, batch_size=int(batch_size))
    mv15c._verify_manifest(output, "prediction_manifest.json")
    attestation = {
        "stage": STAGE,
        "status": "MV15C_A1_q_y_predictions_locked_before_cross_seed_targets",
        "amendment_lock_manifest_sha256": _sha256(output / LOCK_MANIFEST),
        "original_prediction_manifest_sha256": _sha256(output / "prediction_manifest.json"),
        "original_reference_gate_all_pass": lock["original_reference_gate_all_pass"],
        "all_eight_complete_references_used": True,
        "Raw_B10_used_by_prediction": False,
        "cross_seed_targets_constructed": False,
        "B3_predictor_or_weights_changed": False,
        "q_y_gates_changed": False,
        "original_prediction_status": original["status"],
    }
    _atomic_json(output / PREDICTION_ATTESTATION, attestation)
    _atomic_json(
        output / PREDICTION_MANIFEST,
        {
            "stage": STAGE,
            "files": {
                name: {
                    "sha256": _sha256(output / name),
                    "size_bytes": (output / name).stat().st_size,
                }
                for name in (
                    LOCK_MANIFEST,
                    "prediction_manifest.json",
                    PREDICTION_ATTESTATION,
                )
            },
        },
    )
    return attestation


def run_post(output_root: Path) -> dict[str, Any]:
    output = Path(output_root).resolve()
    lock = verify_amendment_lock(output)
    _verify_manifest(output, PREDICTION_MANIFEST)
    mv15c = _mv15c_module()
    original = mv15c.run_post(output)
    _atomic_json(output / ORIGINAL_QY_SUMMARY, original)
    qy_all_pass = bool(original["all_gates_pass"])
    decision = (
        "MV15C_A1_fresh_q_y_supports_frozen_B3_DCIR_QY_with_original_temperature_QC_warning"
        if qy_all_pass
        else "MV15C_A1_fresh_q_y_does_not_support_frozen_B3_DCIR_QY_no_retuning"
    )
    qc = json.loads((output / QC_JSON).read_text(encoding="utf-8"))
    amended = dict(original)
    amended.update(
        {
            "stage": STAGE,
            "status": "complete_MV15C_A1_q_y_evaluation",
            "decision": decision,
            "scientific_classification": amendment_contract()["scientific_classification"],
            "unamended_MV15C_confirmatory_status": (
                "inconclusive_reference_QC_gate_not_met_before_q_y_evaluation"
            ),
            "original_MV15C_q_y_decision_if_evaluated": original["decision"],
            "original_reference_gate_all_pass": lock["original_reference_gate_all_pass"],
            "original_reference_gate_hold_count": lock["original_reference_gate_hold_count"],
            "original_reference_QC": qc,
            "reference_temperature_QC_warning": True,
            "all_q_y_gates_pass": qy_all_pass,
            "all_gates_pass": False,
            "q_y_gates": original["gates"],
            "global_all_gates_pass_is_false_because_original_reference_QC_failed": True,
            "B3_predictor_or_weights_changed_after_fresh_data": False,
            "q_y_gates_changed_after_fresh_data": False,
            "fresh_q_y_outcomes_used_for_tuning": False,
            "DSMC_rerun_performed": False,
            "amendment_lock_manifest_sha256": _sha256(output / LOCK_MANIFEST),
            "prediction_attestation_manifest_sha256": _sha256(
                output / PREDICTION_MANIFEST
            ),
            "original_q_y_summary_sha256": _sha256(output / ORIGINAL_QY_SUMMARY),
        }
    )
    _atomic_json(output / "summary.json", amended)
    return amended


def package_results(output_root: Path, return_directory: Path) -> dict[str, Any]:
    output = Path(output_root).resolve()
    returned = Path(return_directory).resolve()
    _verify_manifest(output, LOCK_MANIFEST)
    _verify_manifest(output, PREDICTION_MANIFEST)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    if summary.get("stage") != STAGE:
        raise ValueError("MV15C-A1 summary is absent")
    mv15c = _mv15c_module()
    names = [
        "submission_lock.json",
        "source_lock_manifest.json",
        mv15c.PROTOCOL_FILE,
        AMENDMENT_FILE,
        QC_JSON,
        QC_CSV,
        LOCK_JSON,
        LOCK_MANIFEST,
        "prediction_summary.json",
        "prediction_manifest.json",
        "fresh_source_audit.csv",
        PREDICTION_ATTESTATION,
        PREDICTION_MANIFEST,
        ORIGINAL_QY_SUMMARY,
        "summary.json",
        "mv15c_fresh_qy_metrics.csv",
    ]
    names.extend(str(name) for name in summary.get("figures", []))
    accounting = output / "mv15c_a1_slurm_accounting.psv"
    if accounting.is_file():
        names.append(accounting.name)
    names = list(dict.fromkeys(names))
    files = [output / name for name in names]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = {
        "stage": STAGE,
        "status": "complete_MV15C_A1_compact_return_manifest",
        "files": {
            path.name: {
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in files
        },
    }
    _atomic_json(output / ARTIFACT_MANIFEST, manifest)
    _verify_manifest(output, ARTIFACT_MANIFEST)
    verification = {
        "stage": STAGE,
        "status": "complete_MV15C_A1_recursive_return_verification",
        "decision": "verified",
        "verified_file_count": len(manifest["files"]),
        "manifest_sha256": _sha256(output / ARTIFACT_MANIFEST),
        "q_y_decision": summary["decision"],
        "all_q_y_gates_pass": bool(summary["all_q_y_gates_pass"]),
        "unamended_MV15C_confirmatory_status": summary[
            "unamended_MV15C_confirmatory_status"
        ],
    }
    _atomic_json(output / VERIFICATION_FILE, verification)
    files.extend((output / ARTIFACT_MANIFEST, output / VERIFICATION_FILE))
    returned.mkdir(parents=True, exist_ok=True)
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = returned / f"MV15C_A1_QY_EVALUATION_BUNDLE_{tag}.zip"
    if archive.exists():
        raise FileExistsError(f"refusing to overwrite MV15C-A1 archive: {archive}")
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as stream:
        for path in files:
            stream.write(path, arcname=path.name)
    if archive.stat().st_size > 450 * 1024 * 1024:
        raise RuntimeError("MV15C-A1 return archive exceeds 450 MiB")
    result = {
        "stage": STAGE,
        "archive": str(archive),
        "archive_sha256": _sha256(archive),
        "decision": summary["decision"],
        "all_q_y_gates_pass": bool(summary["all_q_y_gates_pass"]),
        "original_reference_gate_all_pass": bool(
            summary["original_reference_gate_all_pass"]
        ),
    }
    _atomic_json(output / RETURN_FILE, result)
    (returned / RESULT_POINTER).write_text(
        "\n".join(
            (
                f"MV15C_A1_OUTPUT_ROOT={output}",
                f"MV15C_A1_RESULT_ARCHIVE={archive}",
                f"MV15C_A1_RESULT_ARCHIVE_SHA256={result['archive_sha256']}",
                f"MV15C_A1_DECISION={result['decision']}",
                f"MV15C_A1_ALL_QY_GATES_PASS={int(result['all_q_y_gates_pass'])}",
                "",
            )
        ),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("verify-amendment")
    prepare = subparsers.add_parser("prepare-amendment")
    prepare.add_argument("--output-root", type=Path, required=True)
    predict = subparsers.add_parser("predict")
    predict.add_argument("--output-root", type=Path, required=True)
    predict.add_argument("--batch-size", type=int, default=8)
    post = subparsers.add_parser("post")
    post.add_argument("--output-root", type=Path, required=True)
    package = subparsers.add_parser("package")
    package.add_argument("--output-root", type=Path, required=True)
    package.add_argument("--return-directory", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "verify-amendment":
        result = verify_contract()
    elif args.command == "prepare-amendment":
        result = prepare_amendment(args.output_root)
    elif args.command == "predict":
        result = run_prediction(args.output_root, batch_size=args.batch_size)
    elif args.command == "post":
        result = run_post(args.output_root)
    else:
        result = package_results(args.output_root, args.return_directory)
    print(_json_dumps(result))


if __name__ == "__main__":
    main()
