#!/usr/bin/env python3
"""Apply the frozen JCP6R model to B3 and lock M12 predictions."""

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


EXPECTED_MODEL_ARCHIVE_SHA256 = "bcb57b4585f9be949c8c859cf2d5036a1570499794cf402599f162119390fd20"
EXPECTED_MODEL_SHA256 = "e79cfd8660612d48490d8799a4fbe5886eccb85809dffde676499ed94708dac0"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wall_q(data: bytes) -> np.ndarray:
    table = np.atleast_2d(np.loadtxt(io.BytesIO(data), skiprows=2))
    if table.shape[0] < 20 or table.shape[1] < 16 or not np.isfinite(table).all():
        raise ValueError(f"invalid M12 wall table {table.shape}")
    return table[:, 15].astype(np.float64)


def model_from_archive(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    if sha256(path) != EXPECTED_MODEL_ARCHIVE_SHA256:
        raise ValueError("JCP6R archive checksum mismatch")
    with zipfile.ZipFile(path) as archive:
        lock = json.loads(archive.read("JCP6R_MODEL_LOCK.json"))
        model_bytes = archive.read("JCP6R_MODEL.npz")
    if lock.get("status") != "repair_model_lock_complete_gate_pass":
        raise ValueError("JCP6R gate did not pass")
    if sha256_bytes(model_bytes) != EXPECTED_MODEL_SHA256 or lock.get("model_sha256") != EXPECTED_MODEL_SHA256:
        raise ValueError("JCP6R model hash mismatch")
    with np.load(io.BytesIO(model_bytes), allow_pickle=False) as frozen:
        model = {name: frozen[name].copy() for name in frozen.files}
    for name, value in model.items():
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError(f"nonfinite locked model array {name}")
    return model, lock


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    model, model_lock = model_from_archive(args.model_lock)
    coords = model["coordinates"].astype(np.float64)
    prior = 2.0 * model["prior_m10"].astype(np.float64) - model["prior_m8"].astype(np.float64)
    variance = np.maximum(model["block_variance_m8"], model["block_variance_m10"]).astype(np.float64)
    zones, ex, ey, _ = zones_for(coords)
    if not np.array_equal(zones, model["zones"]):
        raise ValueError("locked spatial-zone mismatch")

    predictions, manifest = [], []
    with zipfile.ZipFile(args.evaluation) as outer:
        audit = json.loads(outer.read("JCP7_M12_EVALUATION_AUDIT.json"))
        seeds = audit["selected_seeds"]
        if len(seeds) != 8 or audit.get("reference_artifacts_read") is not False:
            raise ValueError("invalid prospective evaluation audit")
        for seed in seeds:
            nested_name = f"units/seed_{seed}/JCP7_M12_EVALUATION_seed_{seed}.zip"
            nested_bytes = outer.read(nested_name)
            with zipfile.ZipFile(io.BytesIO(nested_bytes)) as nested:
                summary = json.loads(nested.read("JCP7_M12_EVALUATION_SUMMARY.json"))
                nouts = summary["retained_nout"]
                if len(nouts) != 14 or summary.get("status") != "qc_pass":
                    raise ValueError(f"invalid selected seed {seed}")
                blocks, wall_blocks = [], []
                coords_seen = None
                for nout in nouts:
                    moment_bytes = nested.read(f"JCP3_MOMENTS_NOUT{nout:04d}.DAT")
                    wall_bytes = nested.read(f"JCP3_WALL_NOUT{nout:04d}.DAT")
                    block_coords, fields, _ = parse_moment_bytes(moment_bytes)
                    if coords_seen is None:
                        coords_seen = block_coords
                    elif not np.allclose(coords_seen, block_coords, rtol=0.0, atol=2e-8):
                        raise ValueError("coordinates changed within M12 unit")
                    blocks.append(fields.astype(np.float64))
                    wall_blocks.append(wall_q(wall_bytes))
                if coords_seen is None or not np.allclose(coords, coords_seen, rtol=0.0, atol=2e-8):
                    raise ValueError("M12 coordinates differ from locked model")
                blocks_array = np.asarray(blocks)
                walls_array = np.asarray(wall_blocks)
                raw_b3 = np.mean(blocks_array[:3], axis=0)
                raw_b10 = np.mean(blocks_array[4:], axis=0)
                candidate, gains = fuse_candidate(raw_b3, prior, variance, zones, ex, ey, 3)
                for field in (0, 3, 4, 6):
                    candidate[:, field] = np.maximum(candidate[:, field], 0.05 * raw_b3[:, field])
                if not np.isfinite(candidate).all():
                    raise ValueError("nonfinite prospective prediction")
                prediction_path = args.output / f"JCP7_PREDICTION_seed_{seed}.npz"
                np.savez_compressed(
                    prediction_path,
                    candidate=candidate.astype(np.float32),
                    raw_B3=raw_b3.astype(np.float32),
                    raw_B10=raw_b10.astype(np.float32),
                    wall_raw_B3=np.mean(walls_array[:3], axis=0).astype(np.float64),
                    wall_raw_B10=np.mean(walls_array[4:], axis=0).astype(np.float64),
                    retained_nout=np.asarray(nouts),
                )
                predictions.append(prediction_path)
                manifest.append({
                    "seed": seed,
                    "evaluation_unit_sha256": sha256_bytes(nested_bytes),
                    "prediction_sha256": sha256(prediction_path),
                    "B3_nout": nouts[:3],
                    "guard_nout": nouts[3],
                    "Raw_B10_nout": nouts[4:],
                    "gains": gains,
                })

    manifest_path = args.output / "JCP7_PREDICTION_MANIFEST.json"
    manifest_path.write_text(json.dumps({"units": manifest}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lock = {
        "stage": "JCP7_M12_prospective_prediction_lock",
        "classification": "prospective_predictions_frozen_before_reference",
        "status": "prediction_lock_complete",
        "evaluation_archive_sha256": sha256(args.evaluation),
        "model_lock_archive_sha256": sha256(args.model_lock),
        "model_sha256": model_lock["model_sha256"],
        "protocol_sha256": sha256(args.protocol),
        "manifest_sha256": sha256(manifest_path),
        "selected_seeds": [item["seed"] for item in manifest],
        "prediction_count": len(predictions),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reference_artifacts_read": False,
        "next_stage": "submit_independent_M12_reference_only_after_verifying_this_lock",
    }
    lock_path = args.output / "JCP7_M12_PREDICTION_LOCK.json"
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    archive_path = args.output / "JCP7_M12_PREDICTION_LOCK.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in (args.protocol, manifest_path, lock_path, *predictions):
            archive.write(path, arcname=path.name)
    archive_path.with_suffix(".zip.sha256").write_text(
        f"{sha256(archive_path)}  {archive_path.name}\n", encoding="utf-8"
    )
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
