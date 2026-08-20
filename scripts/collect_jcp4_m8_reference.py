#!/usr/bin/env python3
"""Collect four verified Mach-8 development-reference units."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


SEEDS = (26082401, 26082402, 26082403, 26082404)


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

    units = []
    members = [args.protocol]
    for seed in SEEDS:
        unit = args.work / "units" / f"seed_{seed}"
        summary_path = unit / "JCP4_M8_REFERENCE_SUMMARY.json"
        archive_path = unit / f"JCP4_M8_REFERENCE_seed_{seed}.zip"
        checksum_path = archive_path.with_suffix(".zip.sha256")
        for path in (summary_path, archive_path, checksum_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "mechanical_reference_unit_pass":
            raise ValueError(f"seed {seed} did not pass")
        if summary.get("retained_blocks") != 40:
            raise ValueError(f"seed {seed} retained-block mismatch")
        expected = checksum_path.read_text(encoding="utf-8").split()[0]
        actual = sha256(archive_path)
        if expected != actual:
            raise ValueError(f"seed {seed} archive checksum mismatch")
        units.append(
            {
                "seed": seed,
                "retained_blocks": 40,
                "archive": archive_path.name,
                "archive_sha256": actual,
            }
        )
        members.extend((summary_path, archive_path, checksum_path))

    audit = {
        "stage": "JCP4_M8_development_reference",
        "classification": "development_only_not_prospective_evidence",
        "status": "mechanical_campaign_pass",
        "seed_count": len(units),
        "total_retained_blocks": sum(x["retained_blocks"] for x in units),
        "units": units,
        "next_gate": "train_and_freeze_on_M8_and_existing_M10_before_M12_evaluation",
    }
    audit_path = args.work / "JCP4_M8_REFERENCE_AUDIT.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    members.append(audit_path)
    with zipfile.ZipFile(args.archive, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in members:
            archive.write(path, arcname=path.relative_to(args.work) if path.is_relative_to(args.work) else path.name)
    checksum = sha256(args.archive)
    args.archive.with_suffix(args.archive.suffix + ".sha256").write_text(
        f"{checksum}  {args.archive.name}\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
