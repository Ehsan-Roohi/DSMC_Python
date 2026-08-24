#!/usr/bin/env python3
"""Score the frozen JCP7 predictions against the independent JCP8 reference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from jcp6_train_freeze import FIELDS, EPS, nrmse, parse_moment_bytes, parse_wall_bytes


EXPECTED_PREDICTION_SHA256 = "54db6c0be71764df87f9912090821d4676625ea7ccd8da1f4c069e7edd2ac0d8"
EXPECTED_MODEL_SHA256 = "bcb57b4585f9be949c8c859cf2d5036a1570499794cf402599f162119390fd20"
REFERENCE_SEEDS = (26082801, 26082802, 26082803, 26082804)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def geometric(values: list[float]) -> float:
    values_array = np.maximum(np.asarray(values, dtype=np.float64), EPS)
    return float(np.exp(np.mean(np.log(values_array))))


def bootstrap_ratio_ci(seed_ratios: list[float], seed: int) -> list[float]:
    values = np.log(np.maximum(np.asarray(seed_ratios, dtype=np.float64), EPS))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(20000, len(values)))
    boot = np.exp(np.mean(values[indices], axis=1))
    return [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]


def load_reference(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray], dict]:
    blocks, walls, seed_means, coords_ref = [], [], [], None
    with zipfile.ZipFile(path) as outer:
        audit = json.loads(outer.read("JCP8_M12_REFERENCE_AUDIT.json"))
        if audit.get("prediction_artifacts_read") is not False or audit.get("total_retained_blocks") != 160:
            raise ValueError("invalid independent-reference audit")
        for seed in REFERENCE_SEEDS:
            nested_name = f"units/seed_{seed}/JCP8_M12_REFERENCE_seed_{seed}.zip"
            nested_bytes = outer.read(nested_name)
            expected = next(item["archive_sha256"] for item in audit["units"] if item["seed"] == seed)
            if sha256_bytes(nested_bytes) != expected:
                raise ValueError(f"reference nested hash mismatch seed {seed}")
            unit = []
            with zipfile.ZipFile(io.BytesIO(nested_bytes)) as nested:
                summary = json.loads(nested.read("JCP8_M12_REFERENCE_SUMMARY.json"))
                for nout in summary["retained_nout"]:
                    coords, fields, _ = parse_moment_bytes(nested.read(f"JCP3_MOMENTS_NOUT{nout:04d}.DAT"))
                    if coords_ref is None:
                        coords_ref = coords
                    elif not np.allclose(coords_ref, coords, rtol=0.0, atol=2e-8):
                        raise ValueError("reference coordinates changed")
                    unit.append(fields.astype(np.float64))
                    walls.append(parse_wall_bytes(nested.read(f"JCP3_WALL_NOUT{nout:04d}.DAT")))
            blocks.extend(unit)
            seed_means.append(np.mean(np.asarray(unit), axis=0))
    assert coords_ref is not None
    block_array, wall_array = np.asarray(blocks), np.asarray(walls)
    if block_array.shape[0] != 160 or not np.isfinite(block_array).all() or not np.isfinite(wall_array).all():
        raise ValueError("invalid pooled reference")
    return coords_ref, np.mean(block_array, axis=0), np.mean(wall_array, axis=0), seed_means, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if sha256(args.prediction) != EXPECTED_PREDICTION_SHA256:
        raise ValueError("prediction-lock checksum mismatch")
    if sha256(args.model_lock) != EXPECTED_MODEL_SHA256:
        raise ValueError("model-lock checksum mismatch")
    coords, target, wall_target, reference_seed_means, reference_audit = load_reference(args.reference)
    dx, dy = coords[:, 1] - 0.1524, coords[:, 2]
    radius = np.hypot(dx, dy)
    near, ex, ey = radius <= 0.20, dx / radius, dy / radius
    qn_target = target[:, 7] * ex + target[:, 8] * ey

    rows, seed_records = [], []
    with zipfile.ZipFile(args.prediction) as prediction_zip:
        lock = json.loads(prediction_zip.read("JCP7_M12_PREDICTION_LOCK.json"))
        if lock.get("reference_artifacts_read") is not False or lock.get("prediction_count") != 12:
            raise ValueError("invalid prediction lock")
        for seed in lock["selected_seeds"]:
            with np.load(io.BytesIO(prediction_zip.read(f"JCP7_PREDICTION_seed_{seed}.npz")), allow_pickle=False) as frozen:
                arrays = {name: frozen[name].astype(np.float64) for name in ("candidate", "raw_B3", "raw_B10")}
                wall3, wall10 = frozen["wall_raw_B3"].astype(np.float64), frozen["wall_raw_B10"].astype(np.float64)
            method_errors = {}
            for method, value in arrays.items():
                errors = nrmse(value, target)
                method_errors[method] = dict(zip(FIELDS, map(float, errors), strict=True))
                for field, error in zip(FIELDS, errors, strict=True):
                    rows.append({"seed": seed, "method": method, "field": field, "nrmse": float(error)})
                qn = value[:, 7] * ex + value[:, 8] * ey
                qn_error = float(nrmse(qn[near, None], qn_target[near, None])[0])
                method_errors[method]["qn_near_wall"] = qn_error
                rows.append({"seed": seed, "method": method, "field": "qn_near_wall", "nrmse": qn_error})
            wall3_error = float(nrmse(wall3[:, None], wall_target[:, None])[0])
            wall10_error = float(nrmse(wall10[:, None], wall_target[:, None])[0])
            rows.extend((
                {"seed": seed, "method": "raw_B3", "field": "direct_wall_heat_flux", "nrmse": wall3_error},
                {"seed": seed, "method": "raw_B10", "field": "direct_wall_heat_flux", "nrmse": wall10_error},
            ))
            seed_records.append({
                "seed": seed,
                "all_nine_ratio": geometric([method_errors["candidate"][f] / max(method_errors["raw_B10"][f], EPS) for f in FIELDS]),
                "qy_ratio": method_errors["candidate"]["qy"] / max(method_errors["raw_B10"]["qy"], EPS),
                "qn_ratio": method_errors["candidate"]["qn_near_wall"] / max(method_errors["raw_B10"]["qn_near_wall"], EPS),
                "direct_wall_raw_B3_to_B10_ratio": wall3_error / max(wall10_error, EPS),
            })

    metrics_path = args.output / "JCP8_M12_SCORE_METRICS.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("seed", "method", "field", "nrmse"))
        writer.writeheader(); writer.writerows(rows)
    endpoint = {}
    for key in ("all_nine_ratio", "qy_ratio", "qn_ratio"):
        values = [record[key] for record in seed_records]
        endpoint[key] = {"geometric_mean": geometric(values), "paired_seed_bootstrap_95pct_CI": bootstrap_ratio_ci(values, 26082899), "improved_seed_count": sum(value < 1.0 for value in values)}
    wall_ratios = [record["direct_wall_raw_B3_to_B10_ratio"] for record in seed_records]
    reference_seed_array = np.asarray(reference_seed_means)
    reference_repeatability = nrmse(reference_seed_array, target[None, :, :])
    gate = bool(endpoint["qy_ratio"]["geometric_mean"] < 0.95 and endpoint["qn_ratio"]["geometric_mean"] < 0.95 and endpoint["qy_ratio"]["improved_seed_count"] >= 8 and endpoint["qn_ratio"]["improved_seed_count"] >= 8)
    summary = {
        "stage": "JCP8_M12_independent_prospective_score",
        "classification": "prospective_score_after_prediction_and_reference_locks",
        "status": "score_complete_gate_pass" if gate else "score_complete_gate_fail",
        "prediction_archive_sha256": sha256(args.prediction),
        "reference_archive_sha256": sha256(args.reference),
        "model_archive_sha256": sha256(args.model_lock),
        "protocol_sha256": sha256(args.protocol),
        "prediction_seed_count": len(seed_records),
        "reference_seed_count": 4,
        "reference_block_count": 160,
        "endpoints": endpoint,
        "direct_wall_heat_flux": {
            "candidate_not_defined": True,
            "reason": "frozen model restores volume fields, not the native DS2V surface tally",
            "Raw_B3_to_Raw_B10_geometric_NRMSE_ratio": geometric(wall_ratios),
            "paired_seed_bootstrap_95pct_CI": bootstrap_ratio_ci(wall_ratios, 26082901)
        },
        "reference_seed_repeatability_mean_NRMSE_by_field": dict(zip(FIELDS, map(float, np.mean(reference_repeatability, axis=0)), strict=True)),
        "primary_prospective_gate_pass": gate,
        "seed_level_ratios": seed_records,
        "reference_audit_prediction_artifacts_read": reference_audit["prediction_artifacts_read"],
    }
    summary_path = args.output / "JCP8_M12_SCORE_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    target_path = args.output / "JCP8_M12_REFERENCE_TARGET.npz"
    np.savez_compressed(target_path, coordinates=coords, fields=np.asarray(FIELDS), reference_mean=target.astype(np.float32), wall_heat_flux_mean=wall_target.astype(np.float64))
    archive_path = args.output / "JCP8_M12_SCORE.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in (args.protocol, summary_path, metrics_path, target_path):
            archive.write(path, arcname=path.name)
    archive_path.with_suffix(".zip.sha256").write_text(f"{sha256(archive_path)}  {archive_path.name}\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
