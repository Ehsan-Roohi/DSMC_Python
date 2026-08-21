#!/usr/bin/env python3
"""Verify and package one prospective Mach-12 evaluation candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import zipfile

import numpy as np


NOUT_RE = re.compile(r"NOUT(\d+)\.DAT$")
DRIFT_FIELDS = ("global_u_m_per_s", "energy_per_weight_J", "wall_heat_flux_mean_W_m2")


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
    moment = np.atleast_2d(np.loadtxt(moment_path, comments="#"))
    wall = np.atleast_2d(np.loadtxt(wall_path, skiprows=2))
    if moment.shape[0] < 1000 or moment.shape[1] != 18:
        raise ValueError(f"invalid moment shape {moment.shape}")
    if wall.shape[0] < 20 or wall.shape[1] < 16:
        raise ValueError(f"invalid wall shape {wall.shape}")
    if not np.isfinite(moment).all() or not np.isfinite(wall).all():
        raise ValueError("non-finite evaluation output")
    m0 = moment[:, 5]
    if np.any(m0 <= 0.0):
        raise ValueError("non-positive additive mass")
    total = float(m0.sum())
    return {
        "nout": int(NOUT_RE.search(moment_path.name).group(1)),
        "populated_cells": int(moment.shape[0]),
        "wall_elements": int(wall.shape[0]),
        "m0_sum": total,
        "global_u_m_per_s": float(moment[:, 6].sum() / total),
        "global_v_m_per_s": float(moment[:, 7].sum() / total),
        "energy_per_weight_J": float(moment[:, 15].sum() / total),
        "wall_heat_flux_mean_W_m2": float(wall[:, 15].mean()),
        "wall_heat_flux_max_abs_W_m2": float(np.max(np.abs(wall[:, 15]))),
    }


def drift_z(values: np.ndarray) -> float:
    first, last = values[:7], values[7:]
    se = math.sqrt(float(np.var(first, ddof=1) / 7.0 + np.var(last, ddof=1) / 7.0))
    difference = abs(float(np.mean(first) - np.mean(last)))
    if se == 0.0:
        return 0.0 if difference == 0.0 else 1.0e300
    return difference / se


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--patch-report", type=Path, required=True)
    parser.add_argument("--minimum-total-blocks", type=int, default=40)
    parser.add_argument("--retained-blocks", type=int, default=14)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()

    patch = json.loads(args.patch_report.read_text(encoding="utf-8"))
    if patch.get("changed_token_count") != 1 or not math.isclose(
        float(patch.get("new_speed_m_per_s", 0.0)), 3160.92, rel_tol=0.0, abs_tol=1.0e-8
    ):
        raise ValueError("Mach-12 patch report is invalid")

    moments = indexed(args.case, "JCP3_MOMENTS_")
    walls = indexed(args.case, "JCP3_WALL_")
    paired = sorted(set(moments) & set(walls))
    if len(paired) < args.minimum_total_blocks:
        raise ValueError(f"only {len(paired)} paired blocks")
    retained = paired[-args.retained_blocks:]
    if any(b != a + 1 for a, b in zip(retained[:-1], retained[1:], strict=True)):
        raise ValueError("retained NOUT sequence is not consecutive")
    diagnostics = [verify_block(moments[n], walls[n]) for n in retained]
    if any(float(row["wall_heat_flux_max_abs_W_m2"]) <= 0.0 for row in diagnostics):
        raise ValueError("zero direct wall heat-flux tally")
    drift = {
        name: drift_z(np.asarray([float(row[name]) for row in diagnostics]))
        for name in DRIFT_FIELDS
    }
    stationarity_pass = all(np.isfinite(list(drift.values()))) and max(drift.values()) <= 3.5
    summary = {
        "stage": "JCP7_M12_prospective_evaluation",
        "classification": "prospective_evaluation_before_reference",
        "status": "qc_pass" if stationarity_pass else "qc_reject",
        "seed": args.seed,
        "paired_completed_blocks": len(paired),
        "retained_blocks": len(retained),
        "retained_nout": retained,
        "block_roles": {"B3": retained[:3], "guard": retained[3], "Raw_B10": retained[4:]},
        "stationarity": {"maximum_absolute_drift_z": 3.5, "drift_z": drift, "pass": stationarity_pass},
        "patch_report_sha256": sha256(args.patch_report),
        "block_diagnostics": diagnostics,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    members = [args.patch_report, args.summary]
    for optional in ("RNG_SEED_USED.txt", "DS2VD.TXT"):
        path = args.case / optional
        if path.is_file():
            members.append(path)
    members.extend(moments[n] for n in retained)
    members.extend(walls[n] for n in retained)
    with zipfile.ZipFile(args.archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in members:
            archive.write(path, arcname=path.name)
    args.archive.with_suffix(".zip.sha256").write_text(
        f"{sha256(args.archive)}  {args.archive.name}\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
