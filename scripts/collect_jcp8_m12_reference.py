#!/usr/bin/env python3
"""Collect four independent Mach-12 reference units without reading predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


SEEDS = (26082801, 26082802, 26082803, 26082804)


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
    units, members = [], [args.protocol]
    for seed in SEEDS:
        unit = args.work / "units" / f"seed_{seed}"
        summary = unit / "JCP8_M12_REFERENCE_SUMMARY.json"
        archive = unit / f"JCP8_M12_REFERENCE_seed_{seed}.zip"
        checksum = archive.with_suffix(".zip.sha256")
        for path in (summary, archive, checksum):
            if not path.is_file():
                raise FileNotFoundError(path)
        data = json.loads(summary.read_text(encoding="utf-8"))
        if data.get("status") != "mechanical_reference_unit_pass" or data.get("retained_blocks") != 40:
            raise ValueError(f"invalid reference seed {seed}")
        expected = checksum.read_text(encoding="utf-8").split()[0]
        actual = sha256(archive)
        if expected != actual:
            raise ValueError(f"checksum mismatch seed {seed}")
        units.append({"seed": seed, "retained_blocks": 40, "archive_sha256": actual, "stationarity_sensitivity": data["stationarity_sensitivity"]})
        members.extend((summary, archive, checksum))
    audit = {
        "stage": "JCP8_M12_independent_reference",
        "classification": "prospective_reference_after_prediction_lock",
        "status": "four_reference_units_mechanically_locked",
        "seed_count": 4,
        "total_retained_blocks": 160,
        "all_seeds_included": True,
        "prediction_artifacts_read": False,
        "units": units,
    }
    audit_path = args.work / "JCP8_M12_REFERENCE_AUDIT.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    members.append(audit_path)
    with zipfile.ZipFile(args.archive, "w", compression=zipfile.ZIP_STORED) as outer:
        for path in members:
            arcname = path.relative_to(args.work) if path.is_relative_to(args.work) else path.name
            outer.write(path, arcname=arcname)
    args.archive.with_suffix(".zip.sha256").write_text(f"{sha256(args.archive)}  {args.archive.name}\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
