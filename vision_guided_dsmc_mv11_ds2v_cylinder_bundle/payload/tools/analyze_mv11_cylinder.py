#!/usr/bin/env python3
"""Reconstruct stress and heat flux from MV11 additive DS2V moments."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
from pathlib import Path

import numpy as np


ARGON_MASS = 6.63e-26
META_RE = re.compile(r"([A-Z0-9_]+)=([^\s]+)")


def parse_moment_file(path: Path) -> tuple[dict[str, float], np.ndarray]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 4 or "MV11_ADDITIVE_KINETIC_MOMENTS_VERSION=1" not in lines[0]:
        raise ValueError(f"invalid MV11 moment header: {path}")
    metadata: dict[str, float] = {}
    for key, raw in META_RE.findall(lines[1]):
        metadata[key] = float(raw.replace("D", "E"))
    required = {"NOUT", "TIME", "FNUM", "BLOCK_SAMPLES"}
    missing = required - metadata.keys()
    if missing:
        raise ValueError(f"missing metadata {sorted(missing)} in {path}")
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.shape[1] != 18:
        raise ValueError(f"expected 18 columns in {path}, found {data.shape[1]}")
    if not np.isfinite(data).all():
        raise ValueError(f"nonfinite raw moment in {path}")
    return metadata, data


def reconstruct(
    metadata: dict[str, float], data: np.ndarray, molecular_mass: float
) -> dict[str, np.ndarray]:
    cell = data[:, 0].astype(np.int64)
    species = data[:, 1].astype(np.int64)
    x, y, area = data[:, 2], data[:, 3], data[:, 4]
    raw = data[:, 5:]
    m0 = raw[:, 0]
    if np.any(m0 <= 0.0) or np.any(area <= 0.0):
        raise ValueError("nonpositive count or cell area")
    block_samples = metadata["BLOCK_SAMPLES"]
    if block_samples <= 0.0:
        raise ValueError("nonpositive block sample count")

    mean = raw / m0[:, None]
    ux, uy, uz = mean[:, 1], mean[:, 2], mean[:, 3]
    vv_xx, vv_yy, vv_zz, vv_xy = (
        mean[:, 4],
        mean[:, 5],
        mean[:, 6],
        mean[:, 7],
    )
    vv_xz, vv_yz = mean[:, 8], mean[:, 9]
    mean_energy, mean_evx, mean_evy = mean[:, 10], mean[:, 11], mean[:, 12]

    number_density = metadata["FNUM"] * m0 / (area * block_samples)
    density = molecular_mass * number_density
    cxx = vv_xx - ux * ux
    cyy = vv_yy - uy * uy
    czz = vv_zz - uz * uz
    cxy = vv_xy - ux * uy
    pxx = density * cxx
    pyy = density * cyy
    pzz = density * czz
    pxy = density * cxy

    speed2 = ux * ux + uy * uy + uz * uz
    qx_per_particle = (
        mean_evx
        - ux * mean_energy
        - molecular_mass * (ux * vv_xx + uy * vv_xy + uz * vv_xz)
        + molecular_mass * ux * speed2
    )
    qy_per_particle = (
        mean_evy
        - uy * mean_energy
        - molecular_mass * (ux * vv_xy + uy * vv_yy + uz * vv_yz)
        + molecular_mass * uy * speed2
    )
    qx = number_density * qx_per_particle
    qy = number_density * qy_per_particle

    result = {
        "cell": cell,
        "species": species,
        "x_m": x,
        "y_m": y,
        "area_m2": area,
        "m0": m0,
        "number_density_m3": number_density,
        "density_kg_m3": density,
        "u_m_s": ux,
        "v_m_s": uy,
        "w_m_s": uz,
        "Pxx_Pa": pxx,
        "Pyy_Pa": pyy,
        "Pzz_Pa": pzz,
        "Pxy_Pa": pxy,
        "Pxx_minus_Pyy_Pa": pxx - pyy,
        "qx_W_m2": qx,
        "qy_W_m2": qy,
    }
    for name, values in result.items():
        if not np.isfinite(values).all():
            raise ValueError(f"nonfinite reconstructed field: {name}")
    return result


def write_csv(path: Path, fields: dict[str, np.ndarray]) -> None:
    names = list(fields)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(names)
        for index in range(len(fields[names[0]])):
            writer.writerow([fields[name][index] for name in names])


def scalar_metrics(fields: dict[str, np.ndarray]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in ("Pxy_Pa", "Pxx_minus_Pyy_Pa", "qx_W_m2", "qy_W_m2"):
        values = fields[name]
        out[f"{name}_rms"] = float(np.sqrt(np.mean(values * values)))
        out[f"{name}_max_abs"] = float(np.max(np.abs(values)))
    out["number_density_min"] = float(np.min(fields["number_density_m3"]))
    out["number_density_max"] = float(np.max(fields["number_density_m3"]))
    out["cell_count"] = int(len(fields["cell"]))
    return out


def paired_metrics(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict[str, float]:
    key_a = {(int(c), int(s)): i for i, (c, s) in enumerate(zip(a["cell"], a["species"]))}
    key_b = {(int(c), int(s)): i for i, (c, s) in enumerate(zip(b["cell"], b["species"]))}
    common = sorted(key_a.keys() & key_b.keys())
    if not common:
        raise ValueError("seed cases have no common cells")
    result: dict[str, float] = {"common_cells": len(common)}
    ia = np.array([key_a[key] for key in common])
    ib = np.array([key_b[key] for key in common])
    for name in ("Pxy_Pa", "Pxx_minus_Pyy_Pa", "qx_W_m2", "qy_W_m2"):
        va, vb = a[name][ia], b[name][ib]
        da, db = va - va.mean(), vb - vb.mean()
        denom = float(np.sqrt(np.dot(da, da) * np.dot(db, db)))
        corr = float(np.dot(da, db) / denom) if denom > 0.0 else None
        scale = float(np.sqrt(0.5 * (np.mean(va * va) + np.mean(vb * vb))))
        nrmse = float(np.sqrt(np.mean((va - vb) ** 2)) / max(scale, 1e-300))
        result[f"{name}_correlation"] = corr
        result[f"{name}_nrmse"] = nrmse
    return result


def analyze_campaign(campaign_root: Path, output_dir: Path, molecular_mass: float) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases: dict[str, dict] = {}
    fields_by_case: dict[str, dict[str, np.ndarray]] = {}
    block_metric_rows: list[dict] = []
    for case_dir in sorted((campaign_root / "cases").glob("seed_*")):
        moment_files = sorted((case_dir / "results" / "moments").glob("MV11_MOMENTS_NOUT*.DAT"))
        if not moment_files:
            continue
        case_id = case_dir.name
        case_meta_path = case_dir / "CASE_METADATA.json"
        case_meta = json.loads(case_meta_path.read_text()) if case_meta_path.is_file() else {}
        for moment_file in moment_files:
            block_metadata, block_raw = parse_moment_file(moment_file)
            block_fields = reconstruct(block_metadata, block_raw, molecular_mass)
            block_metric_rows.append(
                {
                    "case_id": case_id,
                    "seed": case_meta.get("seed"),
                    "nout": int(block_metadata["NOUT"]),
                    "time_s": block_metadata["TIME"],
                    "block_samples": int(block_metadata["BLOCK_SAMPLES"]),
                    **scalar_metrics(block_fields),
                }
            )
        latest = moment_files[-1]
        metadata, raw = parse_moment_file(latest)
        fields = reconstruct(metadata, raw, molecular_mass)
        fields_by_case[case_id] = fields
        write_csv(output_dir / f"fields_{case_id}.csv", fields)
        np.savez_compressed(output_dir / f"fields_{case_id}.npz", **fields)
        cases[case_id] = {
            "latest_moment_file": str(latest.relative_to(campaign_root)),
            "nout": int(metadata["NOUT"]),
            "time_s": metadata["TIME"],
            "block_samples": int(metadata["BLOCK_SAMPLES"]),
            "seed": case_meta.get("seed"),
            "completed_moment_blocks": len(moment_files),
            "metrics": scalar_metrics(fields),
        }

    if block_metric_rows:
        block_names = list(block_metric_rows[0])
        with (output_dir / "all_block_metrics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=block_names)
            writer.writeheader()
            writer.writerows(block_metric_rows)

    pairs = {}
    for first, second in itertools.combinations(sorted(fields_by_case), 2):
        pairs[f"{first}__{second}"] = paired_metrics(
            fields_by_case[first], fields_by_case[second]
        )
    seeds = [entry.get("seed") for entry in cases.values()]
    summary = {
        "analysis_version": 1,
        "campaign_root": str(campaign_root),
        "molecular_mass_kg": molecular_mass,
        "case_count": len(cases),
        "distinct_seed_count": len(set(seed for seed in seeds if seed is not None)),
        "cases": cases,
        "pairwise": pairs,
        "analysis_pass": len(cases) == 4 and len(set(seeds)) == 4,
    }
    (output_dir / "mv11_cylinder_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--molecular-mass", type=float, default=ARGON_MASS)
    args = parser.parse_args()
    summary = analyze_campaign(
        args.campaign_root.resolve(), args.output_dir.resolve(), args.molecular_mass
    )
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
