#!/usr/bin/env python3
"""Select and package the first eight QC-pass Mach-12 candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


SEEDS = tuple(range(26082701, 26082713))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    accepted, rejected, summaries = [], [], []
    for order, seed in enumerate(SEEDS):
        unit = args.work / "units" / f"seed_{seed}"
        summary_path = unit / "JCP7_M12_EVALUATION_SUMMARY.json"
        archive_path = unit / f"JCP7_M12_EVALUATION_seed_{seed}.zip"
        checksum_path = archive_path.with_suffix(".zip.sha256")
        if not summary_path.is_file():
            rejected.append({"order": order, "seed": seed, "reason": "missing_summary"})
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summaries.append(summary_path)
        if summary.get("status") != "qc_pass":
            rejected.append({"order": order, "seed": seed, "reason": "qc_reject", "stationarity": summary.get("stationarity")})
            continue
        if not archive_path.is_file() or not checksum_path.is_file():
            rejected.append({"order": order, "seed": seed, "reason": "missing_archive"})
            continue
        expected = checksum_path.read_text(encoding="utf-8").split()[0]
        actual = sha256(archive_path)
        if expected != actual:
            rejected.append({"order": order, "seed": seed, "reason": "checksum_mismatch"})
            continue
        accepted.append({"order": order, "seed": seed, "archive": archive_path, "checksum": checksum_path, "sha256": actual})
    selected = accepted[:8]
    if len(selected) < 8:
        raise ValueError(f"only {len(selected)} QC-pass evaluation units; rejected={rejected}")
    audit = {
        "stage": "JCP7_M12_prospective_evaluation",
        "classification": "prospective_evaluation_before_reference",
        "status": "eight_qc_pass_units_locked",
        "selection_rule": "first eight QC-pass candidates in locked order",
        "selected_seeds": [item["seed"] for item in selected],
        "selected_units": [{k: item[k] for k in ("order", "seed", "sha256")} for item in selected],
        "rejected_before_selection_completed": [item for item in rejected if item["order"] <= selected[-1]["order"]],
        "reference_artifacts_read": False,
    }
    audit_path = args.work / "JCP7_M12_EVALUATION_AUDIT.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with zipfile.ZipFile(args.archive, "w", compression=zipfile.ZIP_STORED) as outer:
        outer.write(args.protocol, arcname=args.protocol.name)
        outer.write(audit_path, arcname=audit_path.name)
        for summary_path in summaries:
            outer.write(summary_path, arcname=f"summaries/{summary_path.parent.name}/{summary_path.name}")
        for item in selected:
            outer.write(item["archive"], arcname=f"units/seed_{item['seed']}/{item['archive'].name}")
            outer.write(item["checksum"], arcname=f"units/seed_{item['seed']}/{item['checksum'].name}")
    args.archive.with_suffix(".zip.sha256").write_text(
        f"{sha256(args.archive)}  {args.archive.name}\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
