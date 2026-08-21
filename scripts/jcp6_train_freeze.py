#!/usr/bin/env python3
"""Build the M8/M10 development dataset, train, validate, and freeze PNET-C.

This stage reads development data only.  It converts additive particle moments
to the complete nine-field hierarchy, trains a bounded B/noise-conditioned
random-feature neural restorer, applies target-free polar-DCT empirical-Bayes
fusion, and emits a hash-locked model before any prospective Mach-12 data exist.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable
import zipfile

import numpy as np


FIELDS = ("n", "u", "v", "T", "Pxx", "Pxy", "Pyy", "qx", "qy")
M8_SEEDS = (26082401, 26082402, 26082403, 26082404)
M10_SEEDS = (20260813, 32452843, 49979687, 67867967)
BUDGETS = (1, 2, 3, 5, 10)
MASS = 6.62999997e-26
KB = 1.380649e-23
EPS = np.finfo(np.float64).tiny
NOUT_RE = re.compile(r"NOUT(\d+)")
HEADER_RE = re.compile(
    r"FNUM=\s*([+\-0-9.EeDd]+).*BLOCK_SAMPLES=(\d+)", re.IGNORECASE
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_moment_bytes(data: bytes) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    header = data.splitlines()[:3]
    text = b"\n".join(header).decode("ascii", errors="strict")
    match = HEADER_RE.search(text)
    if not match:
        raise ValueError("moment header lacks FNUM/BLOCK_SAMPLES")
    fnum = float(match.group(1).replace("D", "E").replace("d", "e"))
    block_samples = int(match.group(2))
    raw = np.loadtxt(io.BytesIO(data), comments="#")
    raw = np.atleast_2d(raw)
    if raw.shape[0] < 1000 or raw.shape[1] != 18 or not np.isfinite(raw).all():
        raise ValueError(f"invalid additive-moment array {raw.shape}")
    if np.any(raw[:, 5] <= 0.0) or block_samples <= 0 or fnum <= 0.0:
        raise ValueError("invalid additive-moment weights")

    m0 = raw[:, 5]
    m1x, m1y, m1z = raw[:, 6], raw[:, 7], raw[:, 8]
    m2xx, m2yy, m2zz = raw[:, 9], raw[:, 10], raw[:, 11]
    m2xy, m2xz, m2yz = raw[:, 12], raw[:, 13], raw[:, 14]
    energy, energy_vx, energy_vy = raw[:, 15], raw[:, 16], raw[:, 17]
    u, v, w = m1x / m0, m1y / m0, m1z / m0
    factor = fnum / (raw[:, 4] * float(block_samples))
    n = factor * m0
    cxx = m2xx - m1x * u
    cyy = m2yy - m1y * v
    czz = m2zz - m1z * w
    cxy = m2xy - m1x * v
    pxx, pyy, pzz, pxy = (
        factor * MASS * cxx,
        factor * MASS * cyy,
        factor * MASS * czz,
        factor * MASS * cxy,
    )
    temperature = (pxx + pyy + pzz) / (3.0 * n * KB)
    speed2 = u * u + v * v + w * w
    qx_sum = (
        energy_vx
        - u * energy
        - MASS * (u * m2xx + v * m2xy + w * m2xz)
        + MASS * u * m0 * speed2
    )
    qy_sum = (
        energy_vy
        - v * energy
        - MASS * (u * m2xy + v * m2yy + w * m2yz)
        + MASS * v * m0 * speed2
    )
    fields = np.column_stack(
        (n, u, v, temperature, pxx, pxy, pyy, factor * qx_sum, factor * qy_sum)
    )
    if not np.isfinite(fields).all() or np.any(fields[:, 0] <= 0.0) or np.any(fields[:, 3] <= 0.0):
        raise ValueError("nonphysical reconstructed field")
    coords = raw[:, (0, 2, 3, 4)].copy()
    meta = {"fnum": fnum, "block_samples": block_samples, "cells": int(raw.shape[0])}
    return coords, fields.astype(np.float32), meta


def parse_wall_bytes(data: bytes) -> np.ndarray:
    wall = np.loadtxt(io.BytesIO(data), skiprows=2)
    wall = np.atleast_2d(wall)
    if wall.shape[0] < 20 or wall.shape[1] < 16 or not np.isfinite(wall).all():
        raise ValueError(f"invalid wall tally {wall.shape}")
    return wall[:, 15].astype(np.float64)


def assert_coords(reference: np.ndarray | None, current: np.ndarray) -> np.ndarray:
    if reference is None:
        return current
    if reference.shape != current.shape or not np.allclose(reference, current, rtol=0.0, atol=2e-8):
        raise ValueError("cell coordinates changed across development blocks")
    return reference


def load_m8(archive_path: Path) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    all_units, manifest = [], []
    coords_ref: np.ndarray | None = None
    with zipfile.ZipFile(archive_path) as outer:
        for seed in M8_SEEDS:
            nested_name = f"units/seed_{seed}/JCP4_M8_REFERENCE_seed_{seed}.zip"
            nested_bytes = outer.read(nested_name)
            with zipfile.ZipFile(io.BytesIO(nested_bytes)) as nested:
                names = nested.namelist()
                moments = {int(NOUT_RE.search(n).group(1)): n for n in names if n.startswith("JCP3_MOMENTS_NOUT")}
                walls = {int(NOUT_RE.search(n).group(1)): n for n in names if n.startswith("JCP3_WALL_NOUT")}
                paired = sorted(set(moments) & set(walls))
                if len(paired) != 40:
                    raise ValueError(f"M8 seed {seed} has {len(paired)} retained pairs")
                unit = []
                entries = []
                for nout in paired:
                    moment_bytes = nested.read(moments[nout])
                    wall_bytes = nested.read(walls[nout])
                    coords, fields, meta = parse_moment_bytes(moment_bytes)
                    coords_ref = assert_coords(coords_ref, coords)
                    qwall = parse_wall_bytes(wall_bytes)
                    unit.append(fields)
                    entries.append({
                        "nout": nout,
                        "moment_sha256": sha256_bytes(moment_bytes),
                        "wall_sha256": sha256_bytes(wall_bytes),
                        "wall_q_mean": float(np.mean(qwall)),
                        "wall_q_max_abs": float(np.max(np.abs(qwall))),
                        **meta,
                    })
                all_units.append(np.asarray(unit, dtype=np.float32))
                manifest.append({"mach": 8.0, "seed": seed, "blocks": entries, "nested_zip_sha256": sha256_bytes(nested_bytes)})
    assert coords_ref is not None
    return coords_ref, np.asarray(all_units, dtype=np.float32), manifest


def load_m10(root: Path, coords_ref: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    all_units, manifest = [], []
    campaign = root / "MV11_DS2V_CYLINDER_20260813_170355"
    for seed in M10_SEEDS:
        case = campaign / "cases" / f"seed_{seed}" / "results"
        moments = {}
        for path in (case / "moments").glob("MV11_MOMENTS_NOUT*.DAT"):
            moments[int(NOUT_RE.search(path.name).group(1))] = path
        walls = {}
        for path in (case / "surface").glob("NOUT*/HEAT FLUX ERROR.dat"):
            match = NOUT_RE.search(path.parent.name)
            if match:
                walls[int(match.group(1))] = path
        paired = sorted(set(moments) & set(walls))
        if len(paired) < 40:
            raise ValueError(f"M10 seed {seed} has only {len(paired)} pairs")
        retained = paired[-40:]
        unit, entries = [], []
        for nout in retained:
            moment_bytes = moments[nout].read_bytes()
            wall_bytes = walls[nout].read_bytes()
            coords, fields, meta = parse_moment_bytes(moment_bytes)
            assert_coords(coords_ref, coords)
            qwall = parse_wall_bytes(wall_bytes)
            unit.append(fields)
            entries.append({
                "nout": nout,
                "moment_path": str(moments[nout].relative_to(root)),
                "wall_path": str(walls[nout].relative_to(root)),
                "moment_sha256": sha256_bytes(moment_bytes),
                "wall_sha256": sha256_bytes(wall_bytes),
                "wall_q_mean": float(np.mean(qwall)),
                "wall_q_max_abs": float(np.max(np.abs(qwall))),
                **meta,
            })
        all_units.append(np.asarray(unit, dtype=np.float32))
        manifest.append({"mach": 10.0, "seed": seed, "blocks": entries})
    return np.asarray(all_units, dtype=np.float32), manifest


def normalisation(blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    references = np.mean(blocks, axis=1, dtype=np.float64)
    flat = references.reshape(-1, references.shape[-1])
    center = np.median(flat, axis=0)
    scale = np.quantile(np.abs(flat - center), 0.90, axis=0)
    floor = np.maximum(np.abs(center) * 1e-8, 1e-12)
    scale = np.maximum(scale, floor)
    return center, scale


def draw_indices(seed: int, budget: int, draw: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 1009 * budget + 9176 * draw)
    return np.sort(rng.choice(40, size=budget, replace=False))


def samples_for_units(
    blocks: np.ndarray,
    machs: np.ndarray,
    seeds: np.ndarray,
    unit_indices: Iterable[int],
    center: np.ndarray,
    scale: np.ndarray,
) -> list[dict[str, Any]]:
    samples = []
    unit_means = np.mean(blocks, axis=1, dtype=np.float64)
    for unit in unit_indices:
        peers = np.flatnonzero((machs == machs[unit]) & (np.arange(len(machs)) != unit))
        target = np.mean(unit_means[peers], axis=0)
        scatter = np.std(blocks[unit].astype(np.float64), axis=0, ddof=1)
        for budget in BUDGETS:
            for draw in range(4):
                chosen = draw_indices(int(seeds[unit]), budget, draw)
                raw = np.mean(blocks[unit, chosen], axis=0, dtype=np.float64)
                samples.append({
                    "unit": int(unit), "mach": float(machs[unit]), "seed": int(seeds[unit]),
                    "budget": budget, "draw": draw, "indices": chosen.tolist(),
                    "raw": ((raw - center) / scale).astype(np.float32),
                    "noise": (scatter / math.sqrt(budget) / scale).astype(np.float32),
                    "target": ((target - center) / scale).astype(np.float32),
                })
    return samples


def base_features(raw: np.ndarray, noise: np.ndarray, coords: np.ndarray, mach: float, budget: int) -> np.ndarray:
    x = coords[:, 1]
    y = coords[:, 2]
    # Locked DS2V domain: x in [-0.2,0.65], y in [0,0.4].  Fixed scaling is
    # essential because training uses cell subsamples while prediction uses the
    # complete mesh.
    xy = np.column_stack(((x - 0.225) / 0.425, (y - 0.2) / 0.2))
    condition = np.column_stack((
        np.full(len(x), (mach - 9.0) / 2.0),
        np.full(len(x), math.log(float(budget)) / math.log(10.0)),
    ))
    return np.column_stack((raw, noise, xy, condition)).astype(np.float64)


def fit_network(samples: list[dict[str, Any]], coords: np.ndarray, *, hidden: int, ridge: float, seed: int, max_cells: int = 4096) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    input_dim = 2 * len(FIELDS) + 4
    weights = rng.normal(0.0, 1.0 / math.sqrt(input_dim), size=(input_dim, hidden))
    bias = rng.uniform(-0.5, 0.5, size=hidden)
    gram = np.zeros((hidden + 1, hidden + 1), dtype=np.float64)
    cross = np.zeros((hidden + 1, len(FIELDS)), dtype=np.float64)
    for index, sample in enumerate(samples):
        chooser = np.random.default_rng(seed + 7919 * (index + 1))
        cells = chooser.choice(len(coords), size=min(max_cells, len(coords)), replace=False)
        features = base_features(sample["raw"][cells], sample["noise"][cells], coords[cells], sample["mach"], sample["budget"])
        hidden_value = np.tanh(features @ weights + bias)
        design = np.column_stack((hidden_value, np.ones(len(cells))))
        residual = sample["target"][cells] - sample["raw"][cells]
        gram += design.T @ design
        cross += design.T @ residual
    penalty = ridge * np.eye(hidden + 1)
    penalty[-1, -1] = 0.0
    beta = np.linalg.solve(gram + penalty, cross)
    return {"weights": weights, "bias": bias, "beta": beta}


def predict(network: dict[str, np.ndarray], sample: dict[str, Any], coords: np.ndarray) -> np.ndarray:
    features = base_features(sample["raw"], sample["noise"], coords, sample["mach"], sample["budget"])
    hidden = np.tanh(features @ network["weights"] + network["bias"])
    correction = np.column_stack((hidden, np.ones(len(hidden)))) @ network["beta"]
    bound = 4.0 / math.sqrt(float(sample["budget"]))
    return sample["raw"].astype(np.float64) + np.clip(correction, -bound, bound)


def dct_matrix(size: int) -> np.ndarray:
    k = np.arange(size)[:, None]
    n = np.arange(size)[None, :]
    matrix = np.sqrt(2.0 / size) * np.cos(np.pi * (n + 0.5) * k / size)
    matrix[0] /= math.sqrt(2.0)
    return matrix


def polar_mapping(coords: np.ndarray, nr: int = 64, nt: int = 64) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    x, y = coords[:, 1] - 0.1524, coords[:, 2]
    radius, theta = np.hypot(x, y), np.arctan2(y, x)
    rmin, rmax = float(radius.min()), float(radius.max())
    ir = np.clip(np.rint((radius - rmin) / (rmax - rmin) * (nr - 1)), 0, nr - 1).astype(int)
    it = np.clip(np.rint(theta / np.pi * (nt - 1)), 0, nt - 1).astype(int)
    cell_to_grid = ir * nt + it
    rg = np.linspace(rmin, rmax, nr)
    tg = np.linspace(0.0, np.pi, nt)
    points = np.column_stack((radius / (rmax - rmin), theta / np.pi))
    grid = np.column_stack((np.repeat(rg, nt) / (rmax - rmin), np.tile(tg, nr) / np.pi))
    grid_to_cell = np.empty(len(grid), dtype=np.int64)
    for start in range(0, len(grid), 128):
        delta = grid[start:start + 128, None, :] - points[None, :, :]
        grid_to_cell[start:start + 128] = np.argmin(np.sum(delta * delta, axis=2), axis=1)
    return grid_to_cell, cell_to_grid, (nr, nt)


def polar_eb(observation: np.ndarray, prior: np.ndarray, raw_blocks: np.ndarray, mapping: tuple[np.ndarray, np.ndarray, tuple[int, int]], width: int = 8) -> np.ndarray:
    grid_to_cell, cell_to_grid, shape = mapping
    cy, cx = dct_matrix(shape[0]), dct_matrix(shape[1])
    output = np.empty_like(observation, dtype=np.float64)
    for field in range(observation.shape[1]):
        obs_grid = observation[grid_to_cell, field].reshape(shape)
        prior_grid = prior[grid_to_cell, field].reshape(shape)
        residual = cy @ (obs_grid - prior_grid) @ cx.T
        block_coeff = []
        for block in raw_blocks:
            bg = block[grid_to_cell, field].reshape(shape)
            block_coeff.append(cy @ bg @ cx.T)
        noise = np.var(np.asarray(block_coeff), axis=0, ddof=1) / float(len(raw_blocks))
        gain = np.empty(shape, dtype=np.float64)
        for r0 in range(0, shape[0], width):
            for t0 in range(0, shape[1], width):
                rs, ts = slice(r0, min(r0 + width, shape[0])), slice(t0, min(t0 + width, shape[1]))
                rp = float(np.mean(residual[rs, ts] ** 2))
                npow = float(np.mean(noise[rs, ts]))
                gain[rs, ts] = 0.0 if rp <= EPS else np.clip((rp - npow) / rp, 0.0, 1.0)
        fused_grid = cy.T @ (cy @ prior_grid @ cx.T + gain * residual) @ cx
        output[:, field] = fused_grid.ravel()[cell_to_grid]
    return output


def nrmse(value: np.ndarray, target: np.ndarray) -> np.ndarray:
    numerator = np.sqrt(np.mean((value - target) ** 2, axis=0))
    denominator = np.sqrt(np.mean(target ** 2, axis=0))
    return numerator / np.maximum(denominator, EPS)


def validate(
    network: dict[str, np.ndarray],
    samples: list[dict[str, Any]],
    blocks: np.ndarray,
    coords: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mapping = polar_mapping(coords)
    rows = []
    for sample in samples:
        if sample["budget"] not in (3, 10) or sample["draw"] != 0:
            continue
        prediction = predict(network, sample, coords)
        chosen = np.asarray(sample["indices"], dtype=int)
        fused = polar_eb(sample["raw"], prediction, ((blocks[sample["unit"], chosen] - 0.0)), mapping)
        for method, value in (("raw", sample["raw"]), ("pnet", prediction), ("pnet_polar_eb", fused)):
            physical = value * scale + center
            target_physical = sample["target"] * scale + center
            errors = nrmse(physical, target_physical)
            for field, error in zip(FIELDS, errors, strict=True):
                rows.append({"unit": sample["unit"], "mach": sample["mach"], "seed": sample["seed"], "budget": sample["budget"], "method": method, "field": field, "nrmse": float(error)})
            dx, dy = coords[:, 1] - 0.1524, coords[:, 2]
            radius = np.hypot(dx, dy)
            near = radius <= 0.20
            qn = (physical[:, 7] * dx + physical[:, 8] * dy) / radius
            qn_target = (target_physical[:, 7] * dx + target_physical[:, 8] * dy) / radius
            qn_error = float(nrmse(qn[near, None], qn_target[near, None])[0])
            rows.append({"unit": sample["unit"], "mach": sample["mach"], "seed": sample["seed"], "budget": sample["budget"], "method": method, "field": "qn_near_wall", "nrmse": qn_error})
    def ratio_for(fields: set[str]) -> float:
        pnet_eb = np.asarray([r["nrmse"] for r in rows if r["method"] == "pnet_polar_eb" and r["budget"] == 3 and r["field"] in fields])
        raw10 = np.asarray([r["nrmse"] for r in rows if r["method"] == "raw" and r["budget"] == 10 and r["field"] in fields])
        return float(np.exp(np.mean(np.log(np.maximum(pnet_eb, EPS) / np.maximum(raw10, EPS)))))
    ratio_all = ratio_for(set(FIELDS))
    ratio_qy = ratio_for({"qy"})
    ratio_qn = ratio_for({"qn_near_wall"})
    summary = {
        "validation_unit_count": len(set((r["mach"], r["seed"]) for r in rows)),
        "field_count": len(FIELDS) + 1,
        "all_nine_fields_pnet_polar_eb_B3_to_raw_B10_geometric_nrmse_ratio": ratio_all,
        "global_qy_pnet_polar_eb_B3_to_raw_B10_geometric_nrmse_ratio": ratio_qy,
        "near_wall_qn_pnet_polar_eb_B3_to_raw_B10_geometric_nrmse_ratio": ratio_qn,
        "primary_development_gate_both_heat_flux_ratios_below_0p95": bool(ratio_qy < 0.95 and ratio_qn < 0.95),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jcp4", type=Path, required=True)
    parser.add_argument("--m10-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    coords, m8, manifest8 = load_m8(args.jcp4)
    m10, manifest10 = load_m10(args.m10_root, coords)
    blocks = np.concatenate((m8, m10), axis=0)
    machs = np.asarray([8.0] * 4 + [10.0] * 4)
    seeds = np.asarray(M8_SEEDS + M10_SEEDS)
    center, scale = normalisation(blocks)
    train_units = (0, 1, 2, 4, 5, 6)
    validation_units = (3, 7)
    train_samples = samples_for_units(blocks, machs, seeds, train_units, center, scale)
    validation_samples = samples_for_units(blocks, machs, seeds, validation_units, center, scale)
    provisional = fit_network(train_samples, coords, hidden=96, ridge=1e-3, seed=26082601)
    metrics, validation = validate(
        provisional, validation_samples, (blocks - center) / scale, coords, center, scale
    )
    all_samples = samples_for_units(blocks, machs, seeds, range(8), center, scale)
    final = fit_network(all_samples, coords, hidden=96, ridge=1e-3, seed=26082601)

    model_path = output / "JCP6_MODEL.npz"
    np.savez_compressed(
        model_path,
        weights=final["weights"], bias=final["bias"], beta=final["beta"],
        field_center=center, field_scale=scale, coordinates=coords,
        fields=np.asarray(FIELDS), budgets=np.asarray(BUDGETS),
    )
    manifest_path = output / "JCP6_DEVELOPMENT_MANIFEST.json"
    json_write(manifest_path, {"m8": manifest8, "m10": manifest10})
    metrics_path = output / "JCP6_VALIDATION_METRICS.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader(); writer.writerows(metrics)
    lock = {
        "stage": "JCP6_cylinder_development_model_lock",
        "classification": "development_only_model_frozen_before_M12",
        "status": "model_lock_complete",
        "jcp4_sha256": sha256(args.jcp4),
        "protocol_sha256": sha256(args.protocol),
        "development_manifest_sha256": sha256(manifest_path),
        "model_sha256": sha256(model_path),
        "fields": list(FIELDS),
        "development_units": 8,
        "development_blocks": int(blocks.shape[0] * blocks.shape[1]),
        "validation": validation,
        "next_stage": "M12_prospective_evaluation_only_after_this_lock",
    }
    lock_path = output / "JCP6_MODEL_LOCK.json"
    json_write(lock_path, lock)
    archive_path = output / "JCP6_MODEL_LOCK.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in (args.protocol, model_path, manifest_path, metrics_path, lock_path):
            archive.write(path, arcname=path.name)
    archive_path.with_suffix(".zip.sha256").write_text(
        f"{sha256(archive_path)}  {archive_path.name}\n", encoding="utf-8"
    )
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
