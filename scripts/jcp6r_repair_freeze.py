#!/usr/bin/env python3
"""Repair the failed JCP6 development model without opening Mach-12 data.

The failed JCP6 artifact is retained as an immutable audit input.  This stage
uses leave-one-seed-out development validation at Mach 8 and 10, with B3/B10
observations disjoint from a B30 reference.  It freezes a cell-registered,
Mach-conditioned prior and a data-consistent adaptive empirical-Bayes rule.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any
import zipfile

import numpy as np

from jcp6_train_freeze import (
    EPS,
    FIELDS,
    M8_SEEDS,
    M10_SEEDS,
    json_write,
    load_m8,
    load_m10,
    nrmse,
    sha256,
)


EXPECTED_FAILED_JCP6_SHA256 = "1f3881fc4b59716922a581304891f5d7b721423b2301d518210b499b54483ac1"
METHOD = "registered_prior_adaptive_eb"
ZONE_NAMES = ("near_wall", "wake", "outer")


def read_failed_lock(archive: Path) -> dict[str, Any]:
    if sha256(archive) != EXPECTED_FAILED_JCP6_SHA256:
        raise ValueError("failed JCP6 archive checksum mismatch")
    with zipfile.ZipFile(archive) as handle:
        lock = json.loads(handle.read("JCP6_MODEL_LOCK.json"))
    validation = lock.get("validation", {})
    if lock.get("status") != "model_lock_complete":
        raise ValueError("input JCP6 lock is incomplete")
    if validation.get("primary_development_gate_both_heat_flux_ratios_below_0p95") is not False:
        raise ValueError("JCP6R is only valid for the recorded failed development gate")
    return lock


def zones_for(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dx, dy = coords[:, 1] - 0.1524, coords[:, 2]
    radius = np.hypot(dx, dy)
    if np.any(radius <= 0.0):
        raise ValueError("cylinder-relative radius contains zero")
    zones = np.full(len(coords), 2, dtype=np.int8)
    zones[radius <= 0.20] = 0
    zones[(radius > 0.20) & (dx >= 0.0)] = 1
    return zones, dx / radius, dy / radius, radius


def field_fuse(
    observation: np.ndarray,
    prior: np.ndarray,
    block_variance: np.ndarray,
    zones: np.ndarray,
    budget: int,
) -> tuple[np.ndarray, list[float]]:
    output = prior.copy()
    gains: list[float] = []
    for zone in range(len(ZONE_NAMES)):
        mask = zones == zone
        residual = observation[mask] - prior[mask]
        residual_power = float(np.mean(residual * residual))
        noise_power = float(np.mean(block_variance[mask])) / float(budget)
        gain = 0.0 if residual_power <= EPS else float(
            np.clip((residual_power - noise_power) / residual_power, 0.0, 1.0)
        )
        output[mask] = prior[mask] + gain * residual
        gains.append(gain)
    return output, gains


def fuse_candidate(
    observation: np.ndarray,
    prior: np.ndarray,
    block_variance: np.ndarray,
    zones: np.ndarray,
    ex: np.ndarray,
    ey: np.ndarray,
    budget: int,
) -> tuple[np.ndarray, dict[str, list[float]]]:
    estimate = np.empty_like(observation, dtype=np.float64)
    gain_record: dict[str, list[float]] = {}
    for field in range(7):
        estimate[:, field], gains = field_fuse(
            observation[:, field], prior[:, field], block_variance[:, field], zones, budget
        )
        gain_record[FIELDS[field]] = gains

    obs_qn = observation[:, 7] * ex + observation[:, 8] * ey
    obs_qt = -observation[:, 7] * ey + observation[:, 8] * ex
    prior_qn = prior[:, 7] * ex + prior[:, 8] * ey
    prior_qt = -prior[:, 7] * ey + prior[:, 8] * ex
    # Rotate the diagonal qx/qy block-variance approximation.  This is a
    # preregistered conservative approximation because cross-covariance was
    # not accumulated in the archived development campaign.
    var_qn = block_variance[:, 7] * ex**2 + block_variance[:, 8] * ey**2
    var_qt = block_variance[:, 7] * ey**2 + block_variance[:, 8] * ex**2
    qn, gain_record["qn"] = field_fuse(obs_qn, prior_qn, var_qn, zones, budget)
    qt, gain_record["qt"] = field_fuse(obs_qt, prior_qt, var_qt, zones, budget)
    estimate[:, 7] = qn * ex - qt * ey
    estimate[:, 8] = qn * ey + qt * ex
    return estimate, gain_record


def split_indices(seed: int, draw: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 104729 * (draw + 1))
    order = rng.permutation(40)
    b10 = np.sort(order[:10])
    b3 = np.sort(b10[:3])
    reference = np.sort(order[10:])
    if set(b10) & set(reference) or not set(b3).issubset(set(b10)):
        raise AssertionError("invalid development partition")
    return b3, b10, reference


def geometric(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if np.any(array <= 0.0) or not np.isfinite(array).all():
        raise ValueError("geometric mean inputs must be finite and positive")
    return float(np.exp(np.mean(np.log(array))))


def validate(
    blocks: np.ndarray,
    machs: np.ndarray,
    seeds: np.ndarray,
    coords: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    zones, ex, ey, radius = zones_for(coords)
    near = radius <= 0.20
    rows: list[dict[str, Any]] = []
    for unit in range(len(blocks)):
        peers = np.flatnonzero((machs == machs[unit]) & (np.arange(len(blocks)) != unit))
        if len(peers) != 3:
            raise ValueError("each validation unit requires three condition peers")
        prior = np.mean(blocks[peers], axis=(0, 1), dtype=np.float64)
        peer_variances = np.var(blocks[peers].astype(np.float64), axis=1, ddof=1)
        block_variance = np.mean(peer_variances, axis=0)
        for draw in range(4):
            b3, b10, reference_indices = split_indices(int(seeds[unit]), draw)
            raw3 = np.mean(blocks[unit, b3], axis=0, dtype=np.float64)
            raw10 = np.mean(blocks[unit, b10], axis=0, dtype=np.float64)
            target = np.mean(blocks[unit, reference_indices], axis=0, dtype=np.float64)
            candidate, gains = fuse_candidate(raw3, prior, block_variance, zones, ex, ey, 3)
            for method, value in (("raw_B3", raw3), ("raw_B10", raw10), ("condition_prior", prior), (METHOD, candidate)):
                errors = nrmse(value, target)
                for field, error in zip(FIELDS, errors, strict=True):
                    rows.append({
                        "unit": unit, "mach": float(machs[unit]), "seed": int(seeds[unit]),
                        "draw": draw, "method": method, "field": field, "nrmse": float(error),
                    })
                qn = value[:, 7] * ex + value[:, 8] * ey
                qn_target = target[:, 7] * ex + target[:, 8] * ey
                rows.append({
                    "unit": unit, "mach": float(machs[unit]), "seed": int(seeds[unit]),
                    "draw": draw, "method": method, "field": "qn_near_wall",
                    "nrmse": float(nrmse(qn[near, None], qn_target[near, None])[0]),
                })
            for component, component_gains in gains.items():
                for zone, gain in zip(ZONE_NAMES, component_gains, strict=True):
                    rows.append({
                        "unit": unit, "mach": float(machs[unit]), "seed": int(seeds[unit]),
                        "draw": draw, "method": METHOD, "field": f"gain_{component}_{zone}",
                        "nrmse": float(gain),
                    })

    endpoint_ratios: dict[str, float] = {}
    improved_counts: dict[str, int] = {}
    for field in ("qy", "qn_near_wall"):
        ratios = []
        unit_ratios = []
        for unit in range(len(blocks)):
            candidate = [r["nrmse"] for r in rows if r["unit"] == unit and r["field"] == field and r["method"] == METHOD]
            raw10 = [r["nrmse"] for r in rows if r["unit"] == unit and r["field"] == field and r["method"] == "raw_B10"]
            ratio = geometric(candidate) / geometric(raw10)
            unit_ratios.append(ratio)
            ratios.extend(c / b for c, b in zip(candidate, raw10, strict=True))
        endpoint_ratios[field] = geometric(ratios)
        improved_counts[field] = sum(ratio < 1.0 for ratio in unit_ratios)

    all_ratios = []
    for field in FIELDS:
        candidate = [r["nrmse"] for r in rows if r["field"] == field and r["method"] == METHOD]
        raw10 = [r["nrmse"] for r in rows if r["field"] == field and r["method"] == "raw_B10"]
        all_ratios.extend(c / b for c, b in zip(candidate, raw10, strict=True))
    gate = (
        endpoint_ratios["qy"] < 0.95
        and endpoint_ratios["qn_near_wall"] < 0.95
        and improved_counts["qy"] >= 6
        and improved_counts["qn_near_wall"] >= 6
    )
    summary = {
        "validation_units": int(len(blocks)),
        "draws_per_unit": 4,
        "all_nine_fields_candidate_B3_to_raw_B10_geometric_nrmse_ratio": geometric(all_ratios),
        "global_qy_candidate_B3_to_raw_B10_geometric_nrmse_ratio": endpoint_ratios["qy"],
        "near_wall_qn_candidate_B3_to_raw_B10_geometric_nrmse_ratio": endpoint_ratios["qn_near_wall"],
        "global_qy_units_improved": improved_counts["qy"],
        "near_wall_qn_units_improved": improved_counts["qn_near_wall"],
        "primary_development_gate_pass": bool(gate),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jcp4", type=Path, required=True)
    parser.add_argument("--m10-root", type=Path, required=True)
    parser.add_argument("--failed-jcp6", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    failed_lock = read_failed_lock(args.failed_jcp6)
    coords, m8, _ = load_m8(args.jcp4)
    m10, _ = load_m10(args.m10_root, coords)
    blocks = np.concatenate((m8, m10), axis=0)
    machs = np.asarray([8.0] * 4 + [10.0] * 4)
    seeds = np.asarray(M8_SEEDS + M10_SEEDS)
    rows, validation = validate(blocks, machs, seeds, coords)

    zones, _, _, _ = zones_for(coords)
    prior_m8 = np.mean(m8, axis=(0, 1), dtype=np.float64)
    prior_m10 = np.mean(m10, axis=(0, 1), dtype=np.float64)
    variance_m8 = np.mean(np.var(m8.astype(np.float64), axis=1, ddof=1), axis=0)
    variance_m10 = np.mean(np.var(m10.astype(np.float64), axis=1, ddof=1), axis=0)
    model_path = args.output / "JCP6R_MODEL.npz"
    np.savez_compressed(
        model_path,
        coordinates=coords,
        fields=np.asarray(FIELDS),
        zones=zones,
        prior_m8=prior_m8.astype(np.float32),
        prior_m10=prior_m10.astype(np.float32),
        # Density is O(1e20), so its variance can exceed the float32 range.
        # Preserve variance ledgers in float64 for prospective fusion.
        block_variance_m8=variance_m8.astype(np.float64),
        block_variance_m10=variance_m10.astype(np.float64),
    )
    with np.load(model_path, allow_pickle=False) as frozen:
        for name in frozen.files:
            value = frozen[name]
            if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                raise ValueError(f"nonfinite frozen-model array {name}")

    metrics_path = args.output / "JCP6R_VALIDATION_METRICS.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lock = {
        "stage": "JCP6R_pre_M12_development_repair",
        "classification": "development_only_repair_model_frozen_before_M12",
        "status": "repair_model_lock_complete_gate_pass" if validation["primary_development_gate_pass"] else "repair_model_lock_complete_gate_fail",
        "failed_jcp6_archive_sha256": sha256(args.failed_jcp6),
        "failed_jcp6_model_sha256": failed_lock["model_sha256"],
        "jcp4_sha256": sha256(args.jcp4),
        "protocol_sha256": sha256(args.protocol),
        "model_sha256": sha256(model_path),
        "model_all_numeric_arrays_finite": True,
        "metrics_sha256": sha256(metrics_path),
        "fields": list(FIELDS),
        "development_units": 8,
        "development_blocks": 320,
        "validation": validation,
        "next_stage": "M12_prospective_evaluation" if validation["primary_development_gate_pass"] else "stop_cylinder_redesign_no_M12",
    }
    lock_path = args.output / "JCP6R_MODEL_LOCK.json"
    json_write(lock_path, lock)
    archive_path = args.output / "JCP6R_MODEL_LOCK.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in (args.protocol, model_path, metrics_path, lock_path):
            archive.write(path, arcname=path.name)
    archive_path.with_suffix(".zip.sha256").write_text(
        f"{sha256(archive_path)}  {archive_path.name}\n", encoding="utf-8"
    )
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
