#!/usr/bin/env python3
"""Inventory locked Mach-10 DS2V development assets before model freezing."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import re
import zipfile


TEXT_SUFFIXES = {".dat", ".txt", ".json", ".csv", ".tsv", ".env", ".out", ".err", ".f90", ".md"}
SNAPSHOT_LIMIT = 2 * 1024 * 1024
HASH_LIMIT = 64 * 1024 * 1024
NOUT_RE = re.compile(r"NOUT(\d+)", re.IGNORECASE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify(path: Path) -> str:
    name = path.name.upper()
    if name.startswith(("JCP3_MOMENTS_NOUT", "MV11_MOMENTS_NOUT")):
        return "additive_moment_block"
    if name.startswith("JCP3_WALL_NOUT"):
        return "paired_wall_block"
    if name == "HEAT FLUX ERROR.DAT" and NOUT_RE.fullmatch(path.parent.name):
        return "paired_wall_block"
    if name == "DS2VD.DAT":
        return "locked_input"
    if name == "HEAT-BENCH.TXT":
        return "heat_benchmark_input"
    if name.startswith("DS2") and path.suffix.lower() in {".dat", ".txt"}:
        return "native_ds2v_output_or_input"
    if path.suffix.lower() in {".f90", ".f", ".for"}:
        return "source"
    if path.suffix.lower() in {".json", ".yaml", ".yml", ".env"}:
        return "metadata"
    if path.suffix.lower() in {".out", ".err", ".log"}:
        return "log"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m10-root", type=Path, required=True)
    parser.add_argument("--jcp4-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.m10_root.is_dir():
        raise FileNotFoundError(args.m10_root)
    if not args.jcp4_archive.is_file():
        raise FileNotFoundError(args.jcp4_archive)
    campaigns = sorted(path for path in args.m10_root.glob("MV11_DS2V_CYLINDER_*") if path.is_dir())
    if not campaigns:
        raise ValueError(f"no MV11 Mach-10 campaigns under {args.m10_root}")

    rows = []
    snapshots: list[tuple[Path, str]] = []
    campaign_summary = []
    for campaign in campaigns:
        counts = Counter()
        paired_by_seed: dict[str, dict[str, set[int]]] = {}
        for path in sorted(p for p in campaign.rglob("*") if p.is_file()):
            rel = path.relative_to(args.m10_root)
            size = path.stat().st_size
            kind = classify(path)
            counts[kind] += 1
            digest = sha256(path) if size <= HASH_LIMIT else None
            rows.append({
                "campaign": campaign.name,
                "relative_path": str(rel),
                "size_bytes": size,
                "classification": kind,
                "sha256": digest,
            })
            parts = path.relative_to(campaign).parts
            seed = next((part for part in parts if part.startswith("seed_")), None)
            if seed and kind in {"additive_moment_block", "paired_wall_block"}:
                match = NOUT_RE.search(path.name) or NOUT_RE.search(path.parent.name)
                if match:
                    entry = paired_by_seed.setdefault(seed, {"moments": set(), "walls": set()})
                    entry["moments" if kind == "additive_moment_block" else "walls"].add(int(match.group(1)))
            if size <= SNAPSHOT_LIMIT and path.suffix.lower() in TEXT_SUFFIXES and kind in {
                "locked_input", "heat_benchmark_input", "source", "metadata", "log"
            }:
                snapshots.append((path, f"snapshots/{rel}"))
        paired_units = []
        for key, value in sorted(paired_by_seed.items()):
            paired_units.append({
                "directory": f"cases/{key}",
                "moments": len(value["moments"]),
                "walls": len(value["walls"]),
                "paired": len(value["moments"] & value["walls"]),
            })
        campaign_summary.append({
            "campaign": campaign.name,
            "file_count": int(sum(counts.values())),
            "class_counts": dict(sorted(counts.items())),
            "additive_units": paired_units,
        })

    ready_units = [
        unit
        for campaign in campaign_summary
        for unit in campaign["additive_units"]
        if unit["paired"] >= 10
    ]
    decision = (
        "m10_additive_blocks_ready_for_dataset_lock"
        if len(ready_units) >= 4
        else "m10_instrumented_development_rerun_required"
    )
    report = {
        "stage": "JCP5_M10_development_asset_audit",
        "classification": "development_only_no_new_trajectory",
        "status": "audit_complete",
        "m10_root": str(args.m10_root),
        "jcp4_archive": str(args.jcp4_archive),
        "jcp4_archive_sha256": sha256(args.jcp4_archive),
        "campaign_count": len(campaigns),
        "campaigns": campaign_summary,
        "ready_additive_unit_count": len(ready_units),
        "decision": decision,
        "rule": "at least four independent Mach-10 units with ten or more paired additive-moment/wall blocks",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report_name = "JCP5_M10_AUDIT.json"
    csv_name = "JCP5_M10_FILE_INVENTORY.csv"
    report_bytes = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    csv_path = args.output.parent / csv_name
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr(report_name, report_bytes)
        archive.write(csv_path, arcname=csv_name)
        seen = set()
        for path, arcname in snapshots:
            if arcname not in seen:
                archive.write(path, arcname=arcname)
                seen.add(arcname)
    csv_path.unlink()
    digest = sha256(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
