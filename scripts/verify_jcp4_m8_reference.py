#!/usr/bin/env python3
"""Verify and package one Mach-8 development-reference trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
import zipfile

import numpy as np


NOUT_RE = re.compile(r"NOUT(\d+)\.DAT$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def indexed(case: Path, prefix: str) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in case.glob(f"{prefix}NOUT*.DAT"):
        match = NOUT_RE.search(path.name)
        if match:
            result[int(match.group(1))] = path
    return result


def verify_block(moment_path: Path, wall_path: Path) -> dict[str, float | int]:
    moment = np.loadtxt(moment_path, comments="#")
    wall = np.loadtxt(wall_path, skiprows=2)
    moment = np.atleast_2d(moment)
    wall = np.atleast_2d(wall)
    if moment.shape[0] < 1000 or moment.shape[1] != 18:
        raise ValueError(f"invalid moment shape {moment.shape} in {moment_path}")
    if wall.shape[0] < 20 or wall.shape[1] < 16:
        raise ValueError(f"invalid wall shape {wall.shape} in {wall_path}")
    if not np.isfinite(moment).all() or not np.isfinite(wall).all():
        raise ValueError(f"non-finite data in NOUT pair {moment_path.name}")
    m0 = moment[:, 5]
    if np.any(m0 <= 0.0):
        raise ValueError(f"non-positive m0 in {moment_path}")
    total_m0 = float(m0.sum())
    qwall = wall[:, 15]
    nout = int(NOUT_RE.search(moment_path.name).group(1))
    return {
        "nout": nout,
        "populated_cells": int(moment.shape[0]),
        "wall_elements": int(wall.shape[0]),
        "m0_sum": total_m0,
        "global_u_m_per_s": float(moment[:, 6].sum() / total_m0),
        "global_v_m_per_s": float(moment[:, 7].sum() / total_m0),
        "energy_per_weight_J": float(moment[:, 15].sum() / total_m0),
        "wall_heat_flux_mean_W_m2": float(qwall.mean()),
        "wall_heat_flux_max_abs_W_m2": float(np.max(np.abs(qwall))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--patch-report", type=Path, required=True)
    parser.add_argument("--minimum-total-blocks", type=int, default=60)
    parser.add_argument("--retained-blocks", type=int, default=40)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()

    moments = indexed(args.case, "JCP3_MOMENTS_")
    walls = indexed(args.case, "JCP3_WALL_")
    paired = sorted(set(moments) & set(walls))
    if len(paired) < args.minimum_total_blocks:
        raise ValueError(
            f"only {len(paired)} paired blocks; {args.minimum_total_blocks} required"
        )
    retained = paired[-args.retained_blocks :]
    diagnostics = [verify_block(moments[n], walls[n]) for n in retained]
    summary = {
        "stage": "JCP4_M8_development_reference",
        "classification": "development_only_not_prospective_evidence",
        "status": "mechanical_reference_unit_pass",
        "seed": args.seed,
        "paired_completed_blocks": len(paired),
        "retained_blocks": len(retained),
        "retained_nout": retained,
        "patch_report_sha256": sha256(args.patch_report),
        "block_diagnostics": diagnostics,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    members = [args.patch_report, args.summary]
    for optional in ("RNG_SEED_USED.txt", "DS2VD.TXT"):
        path = args.case / optional
        if path.is_file():
            members.append(path)
    members.extend(moments[n] for n in retained)
    members.extend(walls[n] for n in retained)
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        args.archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in members:
            archive.write(path, arcname=path.name)
    checksum = sha256(args.archive)
    args.archive.with_suffix(args.archive.suffix + ".sha256").write_text(
        f"{checksum}  {args.archive.name}\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
