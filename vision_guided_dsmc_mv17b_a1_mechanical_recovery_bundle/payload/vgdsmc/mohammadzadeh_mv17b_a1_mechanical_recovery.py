from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
import zipfile


STAGE = "MV17B_A1_Mohammadzadeh_mechanical_recovery"
AMENDMENT = "mv17b_a1_mechanical_recovery_amendment.json"
LOCK_FILE = "MV17B_A1_MECHANICAL_RECOVERY_LOCK.json"
LOCK_MANIFEST = "MV17B_A1_MECHANICAL_RECOVERY_LOCK.sha256"
AMENDMENT_COPY = "MV17B_A1_MECHANICAL_RECOVERY_AMENDMENT.json"
RECOVERY_PROVENANCE = "MV17B_A1_RECOVERY_PROVENANCE.json"
CODE_MANIFEST = "MV17B_A1_CODE_SHA256SUMS.txt"
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


def mechanical_inventory(output_root: Path) -> list[dict[str, Any]]:
    root = Path(output_root).resolve()
    campaign = root / "campaign"
    rows: list[dict[str, Any]] = []
    for index, case_id, pair_id, role, seed, originally_complete in EXPECTED_CASES:
        case = campaign / "cases" / case_id
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


def lock_recovery(output_root: Path, amendment_path: Path) -> dict[str, Any]:
    root = Path(output_root).resolve()
    amendment_path = Path(amendment_path).resolve()
    if (root / "analysis").exists():
        raise ValueError("MV17B analysis already exists; outcome-blind recovery is forbidden")
    if not (root / "frozen" / "FROZEN_MODEL_PASS").is_file():
        raise FileNotFoundError(root / "frozen" / "FROZEN_MODEL_PASS")
    if not (root / "SUBMISSION.env").is_file():
        raise FileNotFoundError(root / "SUBMISSION.env")
    if not (root / CODE_MANIFEST).is_file():
        raise FileNotFoundError(root / CODE_MANIFEST)
    amendment = _json(amendment_path)
    if amendment.get("status") != "locked_before_recovery_and_before_any_MV17B_prediction_or_target_construction":
        raise ValueError("mechanical recovery amendment is not locked")
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
            raise ValueError(f"unexpected pre-recovery status: {row['case_id']}")
    model = root / "frozen" / "mv17b_frozen_cylinder_model.npz"
    protocol = root / "frozen" / "mv17b_fresh_cylinder_confirmation_protocol.json"
    copied_amendment = root / AMENDMENT_COPY
    if copied_amendment.exists():
        if copied_amendment.read_bytes() != amendment_path.read_bytes():
            raise FileExistsError("existing recovery amendment differs")
    else:
        copied_amendment.write_bytes(amendment_path.read_bytes())
    value = {
        "stage": STAGE,
        "status": "MV17B_A1_locked_before_recovery_and_before_prediction",
        "scientific_classification": amendment["scientific_classification"],
        "output_root": str(root),
        "amendment_sha256": _sha256(copied_amendment),
        "original_submission_sha256": _sha256(root / "SUBMISSION.env"),
        "recovery_code_manifest_sha256": _sha256(root / CODE_MANIFEST),
        "original_protocol_sha256": _sha256(protocol),
        "frozen_model_sha256": _sha256(model),
        "recovery_indices": list(RECOVERY_INDICES),
        "mechanical_inventory": inventory,
        "q_y_values_accessed": False,
        "model_predictions_constructed": False,
        "reference_targets_constructed": False,
        "seed_replacement": False,
        "scientific_contract_changed": False,
    }
    lock = root / LOCK_FILE
    if lock.exists():
        if _json(lock) != value:
            raise FileExistsError("existing MV17B-A1 lock differs")
    else:
        _atomic_json(lock, value)
    (root / LOCK_MANIFEST).write_text(f"{_sha256(lock)}  {lock.name}\n", encoding="utf-8")
    return value


def verify_ready(output_root: Path) -> dict[str, Any]:
    root = Path(output_root).resolve()
    if (root / "analysis").exists():
        raise ValueError("analysis directory existed before recovery verification")
    lock = root / LOCK_FILE
    manifest = root / LOCK_MANIFEST
    if not lock.is_file() or not manifest.is_file():
        raise FileNotFoundError("MV17B-A1 recovery lock is absent")
    expected = manifest.read_text(encoding="utf-8").split()[0]
    if _sha256(lock) != expected:
        raise ValueError("MV17B-A1 recovery lock hash changed")
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
            if not (case / "results" / "MV17B_A1_RECOVERY_PASS").is_file():
                raise FileNotFoundError(f"recovery attestation absent: {row['case_id']}")
    recovery_files: list[Path] = [
        root / AMENDMENT_COPY,
        root / LOCK_FILE,
        root / LOCK_MANIFEST,
        root / "MV17B_A1_SUBMISSION.env",
        root / CODE_MANIFEST,
    ]
    for row in rows:
        if row["array_index"] not in RECOVERY_INDICES:
            continue
        case = root / "campaign" / "cases" / row["case_id"]
        recovery = root / "campaign" / "recovery" / row["case_id"]
        recovery_files.extend(
            (
                case / "results" / "RUN_STATUS.pre_mv17b_a1.env",
                case / "results" / "MV17B_A1_RECOVERY_METADATA.json",
                case / "results" / "MV17B_A1_RECOVERY_PASS",
                recovery / "MV17B_A1_RECOVERY_METADATA.json",
                recovery / "INPUT_EXECUTABLE_SHA256.txt",
                recovery / "RNG_SEED_USED.txt",
            )
        )
    missing = [str(path) for path in recovery_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"recovery provenance is incomplete: {missing}")
    provenance = {
        "stage": STAGE,
        "status": "MV17B_A1_recovery_provenance_verified_before_analysis",
        "files": {
            str(path.relative_to(root)): {
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in recovery_files
        },
        "recovered_array_indices": list(RECOVERY_INDICES),
        "same_seeds": True,
        "overlap_byte_identity_required": True,
        "scientific_contract_changed": False,
    }
    _atomic_json(root / RECOVERY_PROVENANCE, provenance)
    return {
        "stage": STAGE,
        "status": "MV17B_A1_all_original_locked_blocks_ready",
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
    """Append a separately hashed acquisition-recovery provenance layer."""

    root = Path(output_root).resolve()
    archive = Path(archive).resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    provenance_path = root / RECOVERY_PROVENANCE
    provenance = _json(provenance_path)
    if provenance.get("status") != "MV17B_A1_recovery_provenance_verified_before_analysis":
        raise ValueError("recovery provenance was not verified")
    paths = [root / relative for relative in provenance["files"]]
    paths.append(provenance_path)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    with zipfile.ZipFile(archive, "a", zipfile.ZIP_DEFLATED, compresslevel=9) as stream:
        existing = set(stream.namelist())
        for path in paths:
            relative = str(path.relative_to(root))
            if relative in existing:
                raise ValueError(f"recovery provenance archive collision: {relative}")
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
            "MV17B_A1_mechanical_recovery": True,
            "recovered_same_seed_trajectories": len(RECOVERY_INDICES),
            "complete_trajectories_reused": len(EXPECTED_CASES) - len(RECOVERY_INDICES),
            "scientific_contract_changed": False,
        }
    )
    _atomic_json(return_path, returned)
    return {
        "stage": STAGE,
        "status": "MV17B_A1_provenance_appended_to_result_archive",
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
    ready = commands.add_parser("verify-ready")
    ready.add_argument("--output-root", type=Path, required=True)
    augment = commands.add_parser("augment-package")
    augment.add_argument("--output-root", type=Path, required=True)
    augment.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "lock":
        value = lock_recovery(args.output_root, args.amendment)
    elif args.command == "verify-ready":
        value = verify_ready(args.output_root)
    else:
        value = augment_package(args.output_root, args.archive)
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
