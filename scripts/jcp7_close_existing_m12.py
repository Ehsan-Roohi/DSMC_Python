#!/usr/bin/env python3
"""Close JCP7 from the twelve completed runs without additional DSMC.

The original stationarity screen is retained as a sensitivity diagnostic.  To
avoid selecting trajectories after seeing that diagnostic, every mechanically
complete seed is included.  B3 and Raw-B10 are disjoint and time-balanced
within the same locked fourteen-block window, so a slow temporal drift cannot
systematically favor either estimator.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from jcp6_train_freeze import parse_moment_bytes
from jcp6r_repair_freeze import fuse_candidate, zones_for
from jcp7_lock_m12_predictions import model_from_archive, wall_q


SEEDS = tuple(range(26082701, 26082713))
B3_INDICES = (0, 6, 13)
GUARD_INDEX = 7
B10_INDICES = (1, 2, 3, 4, 5, 8, 9, 10, 11, 12)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_roles() -> None:
    roles = (*B3_INDICES, GUARD_INDEX, *B10_INDICES)
    if len(roles) != 14 or set(roles) != set(range(14)):
        raise RuntimeError("time-balanced roles must partition fourteen blocks")
    if len(set(B3_INDICES) & set(B10_INDICES)) != 0:
        raise RuntimeError("B3 and Raw-B10 must be disjoint")
    if abs(float(np.mean(B3_INDICES)) - float(np.mean(B10_INDICES))) > 0.25:
        raise RuntimeError("B3 and Raw-B10 are not time-balanced")


def load_units(work: Path) -> tuple[list[dict], list[dict]]:
    units, stationarity = [], []
    for order, seed in enumerate(SEEDS):
        unit = work / "units" / f"seed_{seed}"
        summary_path = unit / "JCP7_M12_EVALUATION_SUMMARY.json"
        archive_path = unit / f"JCP7_M12_EVALUATION_seed_{seed}.zip"
        checksum_path = archive_path.with_suffix(".zip.sha256")
        if not all(path.is_file() for path in (summary_path, archive_path, checksum_path)):
            raise FileNotFoundError(f"incomplete completed-run artifact for seed {seed}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if int(summary.get("seed", -1)) != seed:
            raise ValueError(f"summary seed mismatch for {seed}")
        nouts = list(map(int, summary.get("retained_nout", [])))
        if int(summary.get("paired_completed_blocks", 0)) < 40 or len(nouts) != 14:
            raise ValueError(f"mechanically incomplete seed {seed}")
        if any(b != a + 1 for a, b in zip(nouts[:-1], nouts[1:])):
            raise ValueError(f"nonconsecutive retained blocks for seed {seed}")
        expected = checksum_path.read_text(encoding="utf-8").split()[0]
        actual = sha256(archive_path)
        if expected != actual:
            raise ValueError(f"unit checksum mismatch for seed {seed}")
        with zipfile.ZipFile(archive_path) as nested:
            names = set(nested.namelist())
            for nout in nouts:
                for name in (f"JCP3_MOMENTS_NOUT{nout:04d}.DAT", f"JCP3_WALL_NOUT{nout:04d}.DAT"):
                    if name not in names:
                        raise ValueError(f"missing {name} for seed {seed}")
        passed = bool(summary.get("stationarity", {}).get("pass", False))
        stationarity.append({
            "order": order,
            "seed": seed,
            "original_stationarity_pass": passed,
            "original_drift_z": summary.get("stationarity", {}).get("drift_z", {}),
        })
        units.append({
            "order": order,
            "seed": seed,
            "summary": summary_path,
            "archive": archive_path,
            "checksum": checksum_path,
            "sha256": actual,
            "nouts": nouts,
        })
    return units, stationarity


def build_evaluation(work: Path, amendment: Path, units: list[dict], stationarity: list[dict]) -> Path:
    pass_count = sum(item["original_stationarity_pass"] for item in stationarity)
    audit = {
        "stage": "JCP7_M12_pre_reference_closeout",
        "classification": "pre_reference_protocol_amendment",
        "status": "twelve_mechanically_complete_units_locked",
        "reason": "avoid post-diagnostic seed selection and remove temporal-position bias without additional simulation",
        "all_completed_seeds_included": True,
        "selected_seeds": list(SEEDS),
        "unit_count": len(units),
        "original_stationarity_pass_count": pass_count,
        "original_stationarity_reject_count": len(units) - pass_count,
        "stationarity_use": "reported sensitivity diagnostic; not an exclusion rule",
        "balanced_roles": {
            "B3_indices": list(B3_INDICES),
            "guard_index": GUARD_INDEX,
            "Raw_B10_indices": list(B10_INDICES),
            "B3_mean_index": float(np.mean(B3_INDICES)),
            "Raw_B10_mean_index": float(np.mean(B10_INDICES)),
        },
        "units": [{"order": u["order"], "seed": u["seed"], "sha256": u["sha256"]} for u in units],
        "stationarity_sensitivity": stationarity,
        "reference_artifacts_read": False,
    }
    audit_path = work / "JCP7_M12_EVALUATION_AUDIT.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    archive_path = work / "JCP7_M12_EVALUATION.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as outer:
        outer.write(amendment, arcname=amendment.name)
        outer.write(audit_path, arcname=audit_path.name)
        for unit in units:
            seed = unit["seed"]
            outer.write(unit["summary"], arcname=f"summaries/seed_{seed}/{unit['summary'].name}")
            outer.write(unit["archive"], arcname=f"units/seed_{seed}/{unit['archive'].name}")
            outer.write(unit["checksum"], arcname=f"units/seed_{seed}/{unit['checksum'].name}")
    archive_path.with_suffix(".zip.sha256").write_text(
        f"{sha256(archive_path)}  {archive_path.name}\n", encoding="utf-8"
    )
    return archive_path


def lock_predictions(work: Path, amendment: Path, model_path: Path, units: list[dict], evaluation: Path) -> Path:
    model, model_lock = model_from_archive(model_path)
    coords = model["coordinates"].astype(np.float64)
    prior = 2.0 * model["prior_m10"].astype(np.float64) - model["prior_m8"].astype(np.float64)
    variance = np.maximum(model["block_variance_m8"], model["block_variance_m10"]).astype(np.float64)
    zones, ex, ey, _ = zones_for(coords)
    if not np.array_equal(zones, model["zones"]):
        raise ValueError("locked spatial-zone mismatch")

    predictions, manifest = [], []
    for unit in units:
        seed, nouts = unit["seed"], unit["nouts"]
        blocks, walls, coords_seen = [], [], None
        with zipfile.ZipFile(unit["archive"]) as nested:
            for nout in nouts:
                block_coords, fields, _ = parse_moment_bytes(nested.read(f"JCP3_MOMENTS_NOUT{nout:04d}.DAT"))
                if coords_seen is None:
                    coords_seen = block_coords
                elif not np.allclose(coords_seen, block_coords, rtol=0.0, atol=2e-8):
                    raise ValueError(f"coordinates changed within seed {seed}")
                blocks.append(fields.astype(np.float64))
                walls.append(wall_q(nested.read(f"JCP3_WALL_NOUT{nout:04d}.DAT")))
        if coords_seen is None or not np.allclose(coords, coords_seen, rtol=0.0, atol=2e-8):
            raise ValueError(f"coordinates differ from locked model for seed {seed}")
        block_array, wall_array = np.asarray(blocks), np.asarray(walls)
        raw_b3 = np.mean(block_array[list(B3_INDICES)], axis=0)
        raw_b10 = np.mean(block_array[list(B10_INDICES)], axis=0)
        candidate, gains = fuse_candidate(raw_b3, prior, variance, zones, ex, ey, 3)
        for field in (0, 3, 4, 6):
            candidate[:, field] = np.maximum(candidate[:, field], 0.05 * raw_b3[:, field])
        if not np.isfinite(candidate).all():
            raise ValueError(f"nonfinite prediction for seed {seed}")
        prediction = work / f"JCP7_PREDICTION_seed_{seed}.npz"
        np.savez_compressed(
            prediction,
            candidate=candidate.astype(np.float32),
            raw_B3=raw_b3.astype(np.float32),
            raw_B10=raw_b10.astype(np.float32),
            wall_raw_B3=np.mean(wall_array[list(B3_INDICES)], axis=0).astype(np.float64),
            wall_raw_B10=np.mean(wall_array[list(B10_INDICES)], axis=0).astype(np.float64),
            retained_nout=np.asarray(nouts),
            B3_indices=np.asarray(B3_INDICES),
            guard_index=np.asarray([GUARD_INDEX]),
            Raw_B10_indices=np.asarray(B10_INDICES),
        )
        predictions.append(prediction)
        manifest.append({
            "seed": seed,
            "evaluation_unit_sha256": unit["sha256"],
            "prediction_sha256": sha256(prediction),
            "B3_nout": [nouts[i] for i in B3_INDICES],
            "guard_nout": nouts[GUARD_INDEX],
            "Raw_B10_nout": [nouts[i] for i in B10_INDICES],
            "gains": gains,
        })

    manifest_path = work / "JCP7_PREDICTION_MANIFEST.json"
    manifest_path.write_text(json.dumps({"units": manifest}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lock = {
        "stage": "JCP7_M12_prospective_prediction_lock",
        "classification": "predictions_frozen_after_pre_reference_amendment",
        "status": "prediction_lock_complete",
        "evaluation_archive_sha256": sha256(evaluation),
        "model_lock_archive_sha256": sha256(model_path),
        "model_sha256": model_lock["model_sha256"],
        "amendment_sha256": sha256(amendment),
        "manifest_sha256": sha256(manifest_path),
        "selected_seeds": list(SEEDS),
        "prediction_count": len(predictions),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reference_artifacts_read": False,
        "next_stage": "submit independent Mach-12 reference only after verifying this lock",
    }
    lock_path = work / "JCP7_M12_PREDICTION_LOCK.json"
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    archive_path = work / "JCP7_M12_PREDICTION_LOCK.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in (amendment, manifest_path, lock_path, *predictions):
            archive.write(path, arcname=path.name)
    archive_path.with_suffix(".zip.sha256").write_text(
        f"{sha256(archive_path)}  {archive_path.name}\n", encoding="utf-8"
    )
    return archive_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    args = parser.parse_args()
    validate_roles()
    units, stationarity = load_units(args.work)
    evaluation = build_evaluation(args.work, args.amendment, units, stationarity)
    prediction = lock_predictions(args.work, args.amendment, args.model_lock, units, evaluation)
    for path in (evaluation, prediction):
        expected = path.with_suffix(".zip.sha256").read_text(encoding="utf-8").split()[0]
        if sha256(path) != expected:
            raise RuntimeError(f"final checksum mismatch: {path.name}")
    print(json.dumps({
        "status": "JCP7_closed_without_additional_DSMC",
        "evaluation_units": len(units),
        "original_stationarity_pass": sum(x["original_stationarity_pass"] for x in stationarity),
        "evaluation_archive": str(evaluation),
        "prediction_archive": str(prediction),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
