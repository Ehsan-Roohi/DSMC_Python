#!/usr/bin/env python3
"""Mechanical validation and compact packaging for the JCP3 Mach-12 pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import zipfile

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latest(root: Path, pattern: str) -> Path:
    paths = sorted(Path(root).glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no {pattern} in {root}")
    return paths[-1]


def read_numeric(path: Path, skiprows: int) -> np.ndarray:
    data = np.loadtxt(path, comments="#", skiprows=skiprows)
    if data.ndim == 1:
        data = data[None, :]
    if not np.all(np.isfinite(data)):
        raise ValueError(f"non-finite numeric output: {path}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--patch-report", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    case = args.case.resolve()
    moments = latest(case, "JCP3_MOMENTS_NOUT*.DAT")
    wall = latest(case, "JCP3_WALL_NOUT*.DAT")
    summary_text = (case / "DS2VD.TXT").read_text(encoding="utf-8", errors="replace")
    speed_matches = [float(x.replace("D", "E")) for x in re.findall(r"velocity component in the x direction is\s+([-+0-9.EeDd]+)", summary_text)]
    if not speed_matches or not math.isclose(speed_matches[-1], 3160.92, rel_tol=2e-5):
        raise ValueError(f"Mach-12 freestream speed not verified: {speed_matches}")
    moment_data = read_numeric(moments, 0)
    if moment_data.shape[1] != 18 or moment_data.shape[0] < 1000:
        raise ValueError(f"unexpected moment output shape {moment_data.shape}")
    if np.count_nonzero(moment_data[:, 5]) < 1000:
        raise ValueError("insufficient populated kinetic-moment cells")
    wall_lines = [line for line in wall.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip() and not line.lstrip().startswith(("VARIABLES", "ZONE"))]
    wall_values = []
    for line in wall_lines:
        try:
            values = [float(token.replace("D", "E")) for token in line.replace(",", " ").split()]
        except ValueError:
            continue
        if len(values) >= 16:
            wall_values.append(values)
    if len(wall_values) < 20:
        raise ValueError(f"too few wall-tally rows: {len(wall_values)}")
    wall_array = np.asarray(wall_values, dtype=float)
    if not np.all(np.isfinite(wall_array)) or np.max(np.abs(wall_array[:, 15])) <= 0.0:
        raise ValueError("wall heat-flux tally is absent or zero")
    report = {
        "stage": "JCP3_M12_cylinder_pilot",
        "status": "mechanical_pilot_pass",
        "classification": "preflight_only_not_publication_evidence",
        "freestream_speed_m_per_s": speed_matches[-1],
        "moment_file": moments.name,
        "moment_cells": int(moment_data.shape[0]),
        "moment_columns": int(moment_data.shape[1]),
        "all_properties_reconstructable": [
            "rho", "u", "v", "T", "Pxx", "Pxy", "Pyy", "qx", "qy"
        ],
        "wall_file": wall.name,
        "wall_tally_rows": int(len(wall_values)),
        "wall_heat_flux_max_abs_W_m2": float(np.max(np.abs(wall_array[:, 15]))),
        "patch_report_sha256": sha256(args.patch_report),
    }
    report_path = case / "JCP3_PILOT_SUMMARY.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    files = [args.patch_report, report_path, moments, wall, case / "DS2VD.TXT", case / "RNG_SEED_USED.txt"]
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=path.name)
    checksum = args.archive.with_suffix(args.archive.suffix + ".sha256")
    checksum.write_text(f"{sha256(args.archive)}  {args.archive.name}\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"UPLOAD={args.archive} {checksum}")


if __name__ == "__main__":
    main()
