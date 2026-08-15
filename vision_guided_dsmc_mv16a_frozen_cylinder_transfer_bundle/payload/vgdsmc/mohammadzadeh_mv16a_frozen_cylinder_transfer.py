"""Frozen cavity-to-cylinder transfer audit for the MV15C-A1 q_y estimator.

MV16A does not run DSMC and does not train or tune a model.  It reuses the four
completed MV11 Bird/DS2V Mach-10 cylinder trajectories.  A late, common,
outcome-blind block split is locked before any cylinder neural prediction:

* B3 input blocks: NOUT 100, 108, 116;
* B10 reference blocks: NOUT 101--105 and 109--113;
* unused guard/QC blocks: NOUT 106, 107, 114, 115.

Additive moments are summed before centralisation.  The unstructured cylinder
cells are deterministically rasterised to the exact frozen MV15B weight-map
shape.  The literal MV9 conditioning semantics are retained (log10 Kn and
physical speed / 100); consequently the large Mach-10 speed is explicitly
reported as out of the cavity training range, never silently clipped.

The Mamba checkpoints, TSVD rank, and DCIR-QY weight map are inherited from the
successful MV15C-A1 result and remain immutable.  Targets are leave-one-seed-
out means of the other three disjoint Raw-B10 cylinder fields and are built
only after recursively locked predictions exist.  Because MV11 stopped near
tU/D=11.5 rather than the originally requested 30, this is a transparent
retrospective frozen-transfer audit, not an unamended preregistered
confirmation.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np


STAGE = "MV16A_Mohammadzadeh_frozen_cylinder_transfer"
STATUS = "locked_retrospective_transfer_before_cylinder_model_predictions"
PROTOCOL_FILE = "mv16a_frozen_cylinder_transfer_protocol.json"
RESULT_POINTER = "LAST_MOHAMMADZADEH_MV16A_CYLINDER_RESULT.env"
SEEDS = (20260813, 32452843, 49979687, 67867967)
B3_NOUT = (100, 108, 116)
B10_NOUT = (101, 102, 103, 104, 105, 109, 110, 111, 112, 113)
QC_NOUT = (106, 107, 114, 115)
COMMON_LATE_NOUT = tuple(range(100, 117))
QY_INDEX = 3
ARGON_MASS = 6.63e-26
KB = 1.380649e-23
N_INF = 4.247e20
T_INF = 200.0
U_INF = 2634.1
KNUDSEN = 0.1
WALL_TEMPERATURE = 500.0
DOMAIN = (-0.2, 0.65, 0.0, 0.4)
CYLINDER_RADIUS = 0.1524
EPS = 1.0e-12
META_RE = re.compile(r"([A-Z0-9_]+)=\s*([^\s]+)")
MOMENT_RE = re.compile(r"^MV11_MOMENTS_NOUT(\d+)\.DAT$")


def _mv9_module():
    from . import mohammadzadeh_mv9_heat_flux as mv9

    return mv9


def _mv14_module():
    from . import mohammadzadeh_mv14_kinetic_conservation_cavity as mv14

    return mv14


def _mv15b_module():
    from . import mohammadzadeh_mv15b_data_consistent_budget as mv15b

    return mv15b


def _mv15c_module():
    from . import mohammadzadeh_mv15c_fresh_b3_confirmation as mv15c

    return mv15c


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _json_dumps(value: Any) -> str:
    return json.dumps(
        value,
        indent=2,
        sort_keys=True,
        allow_nan=False,
        default=_json_default,
    )


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_json_dumps(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_manifest(root: Path, name: str, files: Sequence[Path]) -> dict[str, Any]:
    root = Path(root)
    value = {
        "stage": STAGE,
        "files": {
            str(Path(path).relative_to(root)): {
                "sha256": _sha256(path),
                "size_bytes": Path(path).stat().st_size,
            }
            for path in files
        },
    }
    _atomic_json(root / name, value)
    return value


def _verify_manifest(root: Path, name: str) -> dict[str, Any]:
    root = Path(root)
    manifest = json.loads((root / name).read_text(encoding="utf-8"))
    for relative, record in manifest["files"].items():
        path = root / relative
        if (
            not path.is_file()
            or path.stat().st_size != int(record["size_bytes"])
            or _sha256(path) != record["sha256"]
        ):
            raise ValueError(f"MV16A recursive verification failed: {path}")
    return manifest


def protocol_path() -> Path:
    path = (
        Path(__file__).resolve().parents[1]
        / "reference_data"
        / "mohammadzadeh_2012"
        / PROTOCOL_FILE
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def locked_protocol() -> dict[str, Any]:
    value = json.loads(protocol_path().read_text(encoding="utf-8"))
    if value.get("stage") != STAGE or value.get("status") != STATUS:
        raise ValueError("MV16A protocol is absent or unlocked")
    split = value["late_block_split"]
    if (
        tuple(int(v) for v in split["B3_nout"]) != B3_NOUT
        or tuple(int(v) for v in split["B10_nout"]) != B10_NOUT
        or tuple(int(v) for v in split["unused_QC_nout"]) != QC_NOUT
        or set(B3_NOUT) & set(B10_NOUT)
        or set(B3_NOUT) & set(QC_NOUT)
        or set(B10_NOUT) & set(QC_NOUT)
        or set(B3_NOUT + B10_NOUT + QC_NOUT) != set(COMMON_LATE_NOUT)
    ):
        raise ValueError("MV16A implementation differs from its locked split")
    if tuple(int(v) for v in value["cylinder_sources"]["seeds"]) != SEEDS:
        raise ValueError("MV16A seed set differs from the locked protocol")
    return value


def verify_contract() -> dict[str, Any]:
    protocol = locked_protocol()
    return {
        "stage": STAGE,
        "status": "MV16A_contract_verified",
        "protocol_sha256": _sha256(protocol_path()),
        "seed_count": len(SEEDS),
        "B3_nout": list(B3_NOUT),
        "B10_nout": list(B10_NOUT),
        "unused_QC_nout": list(QC_NOUT),
        "DSMC_rerun": False,
        "neural_retraining": False,
        "cylinder_parameter_selection": False,
        "classification": protocol["scientific_classification"],
    }


def _moment_path(campaign_root: Path, seed: int, nout: int) -> Path:
    return (
        Path(campaign_root)
        / "cases"
        / f"seed_{seed}"
        / "results"
        / "moments"
        / f"MV11_MOMENTS_NOUT{nout:04d}.DAT"
    )


def parse_moment_file(path: Path) -> tuple[dict[str, float], np.ndarray]:
    path = Path(path)
    match = MOMENT_RE.match(path.name)
    if match is None:
        raise ValueError(f"invalid MV11 moment filename: {path.name}")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 4 or "MV11_ADDITIVE_KINETIC_MOMENTS_VERSION=1" not in lines[0]:
        raise ValueError(f"invalid MV11 moment header: {path}")
    metadata: dict[str, float] = {}
    for key, raw in META_RE.findall(lines[1]):
        metadata[key] = float(raw.replace("D", "E").replace("d", "e"))
    required = {"NOUT", "TIME", "FNUM", "BLOCK_SAMPLES"}
    missing = required - metadata.keys()
    if missing:
        raise ValueError(f"missing metadata {sorted(missing)} in {path}")
    if int(metadata["NOUT"]) != int(match.group(1)):
        raise ValueError(f"filename/header NOUT mismatch: {path}")
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.shape[1] != 18 or not np.isfinite(data).all():
        raise ValueError(f"invalid raw moment matrix in {path}: {data.shape}")
    order = np.lexsort((data[:, 1], data[:, 0]))
    return metadata, np.asarray(data[order], dtype=np.float64)


def aggregate_moment_files(paths: Sequence[Path]) -> tuple[dict[str, Any], np.ndarray]:
    if not paths:
        raise ValueError("cannot aggregate an empty moment-file list")
    parsed = [parse_moment_file(Path(path)) for path in paths]
    metadata0, data0 = parsed[0]
    key0 = data0[:, :2].astype(np.int64)
    geometry0 = data0[:, 2:5]
    fnum = float(metadata0["FNUM"])
    raw_sum = np.zeros_like(data0[:, 5:], dtype=np.float64)
    total_samples = 0.0
    nouts: list[int] = []
    times: list[float] = []
    for path, (metadata, data) in zip(paths, parsed):
        if data.shape != data0.shape:
            raise ValueError(f"adaptive cell shape changed inside locked late split: {path}")
        if not np.array_equal(data[:, :2].astype(np.int64), key0):
            raise ValueError(f"cell/species identity changed inside locked late split: {path}")
        if not np.allclose(data[:, 2:5], geometry0, rtol=0.0, atol=2.0e-10):
            raise ValueError(f"cell geometry changed inside locked late split: {path}")
        if not math.isclose(float(metadata["FNUM"]), fnum, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"FNUM changed inside locked late split: {path}")
        samples = float(metadata["BLOCK_SAMPLES"])
        if samples <= 0.0:
            raise ValueError(f"nonpositive block sample count: {path}")
        raw_sum += data[:, 5:]
        total_samples += samples
        nouts.append(int(metadata["NOUT"]))
        times.append(float(metadata["TIME"]))
    combined = np.column_stack((data0[:, :5], raw_sum))
    return {
        "FNUM": fnum,
        "BLOCK_SAMPLES": total_samples,
        "NOUTS": nouts,
        "TIMES": times,
    }, combined


def reconstruct_fields(metadata: Mapping[str, Any], data: np.ndarray) -> dict[str, np.ndarray]:
    data = np.asarray(data, dtype=np.float64)
    x, y, area = data[:, 2], data[:, 3], data[:, 4]
    raw = data[:, 5:]
    m0 = raw[:, 0]
    samples = float(metadata["BLOCK_SAMPLES"])
    if samples <= 0.0 or np.any(m0 <= 0.0) or np.any(area <= 0.0):
        raise ValueError("nonpositive samples, count, or cell area")
    mean = raw / m0[:, None]
    ux, uy, uz = mean[:, 1], mean[:, 2], mean[:, 3]
    vv_xx, vv_yy, vv_zz = mean[:, 4], mean[:, 5], mean[:, 6]
    vv_xy, vv_xz, vv_yz = mean[:, 7], mean[:, 8], mean[:, 9]
    mean_energy, mean_evx, mean_evy = mean[:, 10], mean[:, 11], mean[:, 12]
    number_density = float(metadata["FNUM"]) * m0 / (area * samples)
    density = ARGON_MASS * number_density
    cxx, cyy, czz = vv_xx - ux * ux, vv_yy - uy * uy, vv_zz - uz * uz
    pxx, pyy, pzz = density * cxx, density * cyy, density * czz
    pxy = density * (vv_xy - ux * uy)
    speed2 = ux * ux + uy * uy + uz * uz
    qx_per_particle = (
        mean_evx
        - ux * mean_energy
        - ARGON_MASS * (ux * vv_xx + uy * vv_xy + uz * vv_xz)
        + ARGON_MASS * ux * speed2
    )
    qy_per_particle = (
        mean_evy
        - uy * mean_energy
        - ARGON_MASS * (ux * vv_xy + uy * vv_yy + uz * vv_yz)
        + ARGON_MASS * uy * speed2
    )
    temperature = ARGON_MASS * np.maximum(cxx + cyy + czz, 0.0) / (3.0 * KB)
    p_ref = N_INF * KB * T_INF
    q_ref = p_ref * math.sqrt(KB * T_INF / ARGON_MASS)
    outputs = np.stack(
        (
            pxy / p_ref,
            (pxx - pyy) / p_ref,
            number_density * qx_per_particle / q_ref,
            number_density * qy_per_particle / q_ref,
        ),
        axis=1,
    )
    auxiliary = np.stack(
        (
            number_density / N_INF,
            ux / U_INF,
            uy / U_INF,
            temperature / T_INF,
        ),
        axis=1,
    )
    if not np.isfinite(outputs).all() or not np.isfinite(auxiliary).all():
        raise ValueError("nonfinite reconstructed MV16A fields")
    return {
        "cell": data[:, 0].astype(np.int64),
        "species": data[:, 1].astype(np.int64),
        "x_m": x,
        "y_m": y,
        "area_m2": area,
        "outputs": outputs,
        "auxiliary": auxiliary,
        "p_ref_Pa": np.asarray(p_ref),
        "q_ref_W_m2": np.asarray(q_ref),
    }


def raster_grid(shape: Sequence[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ny, nx = int(shape[0]), int(shape[1])
    if ny < 8 or nx < 8:
        raise ValueError("frozen model raster is unexpectedly small")
    xmin, xmax, ymin, ymax = DOMAIN
    x = np.linspace(xmin, xmax, nx, dtype=np.float64)
    y = np.linspace(ymin, ymax, ny, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)
    fluid = (xx * xx + yy * yy) >= CYLINDER_RADIUS**2
    return xx, yy, fluid


def rasterize_fields(
    fields: Mapping[str, np.ndarray], shape: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from scipy.interpolate import griddata

    xx, yy, fluid = raster_grid(shape)
    points = np.column_stack((fields["x_m"], fields["y_m"]))
    channels = np.column_stack((fields["outputs"], fields["auxiliary"]))
    raster = np.empty((channels.shape[1], *xx.shape), dtype=np.float64)
    coverages: list[float] = []
    for index in range(channels.shape[1]):
        linear = griddata(points, channels[:, index], (xx, yy), method="linear")
        valid = np.isfinite(linear) & fluid
        coverages.append(float(np.count_nonzero(valid) / max(np.count_nonzero(fluid), 1)))
        if not np.all(np.isfinite(linear[fluid])):
            nearest = griddata(points, channels[:, index], (xx, yy), method="nearest")
            linear = np.where(np.isfinite(linear), linear, nearest)
        raster[index] = linear
    raster[:4, ~fluid] = 0.0
    raster[4, ~fluid] = 0.0
    raster[5:7, ~fluid] = 0.0
    raster[7, ~fluid] = WALL_TEMPERATURE / T_INF
    if not np.isfinite(raster).all():
        raise ValueError("nonfinite raster after deterministic fill")
    condition = np.stack(
        (
            np.full(xx.shape, np.log10(KNUDSEN), dtype=np.float64),
            np.full(xx.shape, U_INF / 100.0, dtype=np.float64),
        )
    )
    image = np.concatenate((raster, condition), axis=0).astype(np.float32)
    return image, fluid, {
        "shape": [int(xx.shape[0]), int(xx.shape[1])],
        "minimum_linear_fluid_coverage": min(coverages),
        "fluid_pixel_count": int(np.count_nonzero(fluid)),
        "solid_pixel_count": int(np.count_nonzero(~fluid)),
        "literal_log10_Kn_channel": float(np.log10(KNUDSEN)),
        "literal_speed_over_100_channel": U_INF / 100.0,
        "speed_condition_is_outside_cavity_training_support": True,
        "condition_clipping_or_reinterpretation": False,
    }


def _frozen_sources(mv15c_output_root: Path) -> tuple[Path, Path, np.ndarray, int]:
    root = Path(mv15c_output_root).resolve()
    lock = json.loads((root / "submission_lock.json").read_text(encoding="utf-8"))
    mv9_root = Path(lock["mv9_output_root"]).resolve()
    mv15b_root = Path(lock["mv15b_output_root"]).resolve()
    mv15c = _mv15c_module()
    _, _, weight = mv15c._validate_mv15b_outcome(mv15b_root)
    assembly = json.loads((mv9_root / "assembly_summary.json").read_text(encoding="utf-8"))
    rank = int(assembly["classical_selection_development_only"]["tsvd_rank"])
    return mv9_root, mv15b_root, np.asarray(weight, dtype=np.float64), rank


def prepare_lock(
    campaign_root: Path, mv15c_output_root: Path, output_root: Path
) -> dict[str, Any]:
    campaign = Path(campaign_root).resolve()
    mv15c_root = Path(mv15c_output_root).resolve()
    output = Path(output_root).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite MV16A output: {output}")
    verify_contract()
    cylinder_summary = json.loads(
        (campaign / "analysis" / "mv11_cylinder_summary.json").read_text(encoding="utf-8")
    )
    if int(cylinder_summary.get("case_count", 0)) != 4:
        raise ValueError("MV16A requires all four MV11 cylinder seeds")
    a1_summary = json.loads((mv15c_root / "summary.json").read_text(encoding="utf-8"))
    if not bool(a1_summary.get("all_q_y_gates_pass")):
        raise ValueError("MV16A requires the successful frozen MV15C-A1 q_y result")
    mv9_root, mv15b_root, weight, rank = _frozen_sources(mv15c_root)
    required_paths: list[Path] = []
    for seed in SEEDS:
        for nout in COMMON_LATE_NOUT:
            path = _moment_path(campaign, seed, nout)
            if not path.is_file():
                raise FileNotFoundError(path)
            required_paths.append(path)
    output.mkdir(parents=True)
    copied_protocol = output / PROTOCOL_FILE
    copied_protocol.write_bytes(protocol_path().read_bytes())
    source_records = {
        str(path.relative_to(campaign)): {
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in required_paths
    }
    _atomic_json(
        output / "cylinder_source_manifest.json",
        {"stage": STAGE, "campaign_root": str(campaign), "files": source_records},
    )
    lock = {
        "stage": STAGE,
        "status": "MV16A_locked_before_any_cylinder_neural_prediction",
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_classification": locked_protocol()["scientific_classification"],
        "campaign_root": str(campaign),
        "mv15c_output_root": str(mv15c_root),
        "mv9_output_root": str(mv9_root),
        "mv15b_output_root": str(mv15b_root),
        "protocol_sha256": _sha256(copied_protocol),
        "cylinder_source_manifest_sha256": _sha256(output / "cylinder_source_manifest.json"),
        "mv15c_a1_summary_sha256": _sha256(mv15c_root / "summary.json"),
        "frozen_weight_map_sha256": hashlib.sha256(weight.tobytes()).hexdigest(),
        "frozen_weight_shape": list(weight.shape),
        "frozen_tsvd_rank": rank,
        "B3_nout": list(B3_NOUT),
        "B10_nout": list(B10_NOUT),
        "unused_QC_nout": list(QC_NOUT),
        "target_construction_before_prediction": False,
        "cylinder_model_outcomes_seen_before_lock": False,
        "DSMC_rerun": False,
        "neural_retraining": False,
        "cylinder_parameter_selection": False,
        "original_MV11_tUD30_gate_pass": bool(cylinder_summary.get("analysis_pass")),
        "original_MV11_tUD30_gate_warning_preserved": True,
    }
    _atomic_json(output / "submission_lock.json", lock)
    _write_manifest(
        output,
        "source_lock_manifest.json",
        [copied_protocol, output / "cylinder_source_manifest.json", output / "submission_lock.json"],
    )
    return lock


def verify_source_lock(output_root: Path) -> dict[str, Any]:
    output = Path(output_root).resolve()
    _verify_manifest(output, "source_lock_manifest.json")
    lock = json.loads((output / "submission_lock.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "cylinder_source_manifest.json").read_text(encoding="utf-8"))
    campaign = Path(lock["campaign_root"])
    for relative, record in manifest["files"].items():
        path = campaign / relative
        if (
            not path.is_file()
            or path.stat().st_size != int(record["size_bytes"])
            or _sha256(path) != record["sha256"]
        ):
            raise ValueError(f"locked MV11 source changed: {path}")
    if lock["status"] != "MV16A_locked_before_any_cylinder_neural_prediction":
        raise ValueError("invalid MV16A submission lock")
    return lock


def run_prediction(output_root: Path, *, batch_size: int) -> dict[str, Any]:
    output = Path(output_root).resolve()
    lock = verify_source_lock(output)
    if (output / "locked_cylinder_predictions.npz").exists():
        raise FileExistsError("refusing to overwrite existing MV16A predictions")
    campaign = Path(lock["campaign_root"])
    mv9_root, _, weight, rank = _frozen_sources(Path(lock["mv15c_output_root"]))
    images, masks, raster_records = [], [], []
    for seed in SEEDS:
        paths = [_moment_path(campaign, seed, nout) for nout in B3_NOUT]
        metadata, additive = aggregate_moment_files(paths)
        fields = reconstruct_fields(metadata, additive)
        image, mask, raster = rasterize_fields(fields, weight.shape)
        if raster["minimum_linear_fluid_coverage"] < 0.90:
            raise ValueError(f"insufficient cylinder raster coverage for seed {seed}: {raster}")
        images.append(image)
        masks.append(mask)
        raster_records.append({"seed": seed, "B3_nout": list(B3_NOUT), **raster})
    image_array = np.asarray(images, dtype=np.float32)
    mask_array = np.asarray(masks, dtype=bool)
    if image_array.shape[-2:] != weight.shape:
        raise ValueError("MV16A raster differs from frozen MV15B weight shape")
    mv14, mv15b, mv9 = _mv14_module(), _mv15b_module(), _mv9_module()
    vision = mv14._predict_mamba_validation(
        mv9_root, image_array, batch_size=int(batch_size)
    )[:, QY_INDEX].astype(np.float64)
    raw_b3 = image_array[:, QY_INDEX].astype(np.float64)
    selected = mv15b.data_consistent_residual(raw_b3, vision, weight)
    dc_weight = np.zeros_like(weight)
    dc_weight[0, 0] = 1.0
    dc_only = mv15b.data_consistent_residual(raw_b3, vision, dc_weight)
    tsvd = mv9._project_modules()["tsvd"]
    tsvd_qy = np.asarray(tsvd(image_array[:, :4], rank)[:, QY_INDEX], dtype=np.float64)
    permutation = np.roll(np.arange(len(SEEDS), dtype=np.int64), 1)
    permuted = mv15b.data_consistent_residual(raw_b3[permutation], vision, weight)
    np.savez_compressed(
        output / "locked_cylinder_predictions.npz",
        seeds=np.asarray(SEEDS, dtype=np.int64),
        B3_nout=np.tile(np.asarray(B3_NOUT, dtype=np.int64), (len(SEEDS), 1)),
        raw_b3_qy=raw_b3,
        vision_b3_qy=vision,
        selected_b3_qy=selected,
        dc_only_b3_qy=dc_only,
        tsvd_b3_qy=tsvd_qy,
        permuted_b3_qy=permuted,
        permutation=permutation,
        fluid_mask=mask_array,
        frozen_weight_map=weight,
    )
    _atomic_json(
        output / "raster_audit.json",
        {"stage": STAGE, "records": raster_records},
    )
    summary = {
        "stage": STAGE,
        "status": "MV16A_cylinder_predictions_locked_before_B10_target_construction",
        "seed_count": len(SEEDS),
        "B3_nout": list(B3_NOUT),
        "B10_target_constructed": False,
        "Raw_B10_used_by_prediction": False,
        "parameter_selection_on_cylinder": False,
        "neural_retraining": False,
        "literal_condition_semantics": {
            "log10_Kn": float(np.log10(KNUDSEN)),
            "speed_m_per_s_over_100": U_INF / 100.0,
            "speed_outside_cavity_training_support": True,
            "clipped": False,
        },
        "frozen_weight_map_sha256": hashlib.sha256(weight.tobytes()).hexdigest(),
        "frozen_tsvd_rank": rank,
    }
    _atomic_json(output / "prediction_summary.json", summary)
    (output / "PREDICTION_LOCK_PASS").write_text(
        "MV16A cylinder predictions locked before Raw-B10 target construction\n",
        encoding="utf-8",
    )
    _write_manifest(
        output,
        "prediction_manifest.json",
        [
            output / "source_lock_manifest.json",
            output / "locked_cylinder_predictions.npz",
            output / "raster_audit.json",
            output / "prediction_summary.json",
            output / "PREDICTION_LOCK_PASS",
        ],
    )
    return summary


def leave_one_seed_out(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] != 4:
        raise ValueError("MV16A leave-one-out target requires exactly four seeds")
    return np.stack(
        [np.mean(np.delete(values, index, axis=0), axis=0, dtype=np.float64) for index in range(4)]
    )


def masked_nrmse(candidate: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    candidate = np.asarray(candidate, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    difference = candidate[mask] - target[mask]
    scale = math.sqrt(float(np.mean(target[mask] ** 2)))
    return math.sqrt(float(np.mean(difference**2))) / max(scale, EPS)


def _per_seed_metrics(
    methods: Mapping[str, np.ndarray], target: np.ndarray, masks: np.ndarray
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    per_seed: dict[str, dict[str, float]] = {}
    means: dict[str, float] = {}
    for name, values in methods.items():
        per_seed[name] = {
            str(seed): masked_nrmse(values[index], target[index], masks[index])
            for index, seed in enumerate(SEEDS)
        }
        means[name] = float(np.mean(list(per_seed[name].values())))
    return per_seed, means


def _write_metrics_csv(
    path: Path,
    per_seed: Mapping[str, Mapping[str, float]],
    raw_b10: Mapping[str, float],
) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("seed", "method", "qy_nrmse", "ratio_to_Raw_B10"))
        for method, records in per_seed.items():
            for seed, value in records.items():
                writer.writerow((seed, method, value, value / max(raw_b10[seed], EPS)))


def _plot_six_panel(
    output: Path,
    seed: int,
    methods: Mapping[str, np.ndarray],
    reference: np.ndarray,
    mask: np.ndarray,
    q_ref: float,
    field_limit: float,
    error_limit_percent: float,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ("reference", "raw_b3", "vision_b3", "selected_b3", "tsvd_b3", "raw_b10")
    titles = (
        "Reference",
        "Raw DSMC\n$B=3$",
        "MambaIRv2\n$B=3$",
        "DCIR-QY\n$B=3$",
        "TSVD/POD\n$B=3$",
        "Raw DSMC\n$B=10$",
    )
    arrays = {"reference": reference, **methods}
    xmin, xmax, ymin, ymax = DOMAIN
    extent = (xmin / (2 * CYLINDER_RADIUS), xmax / (2 * CYLINDER_RADIUS), ymin / (2 * CYLINDER_RADIUS), ymax / (2 * CYLINDER_RADIUS))
    fig, axes = plt.subplots(2, 6, figsize=(18.0, 6.9), sharex=True, sharey=True, constrained_layout=True)
    top_image = None
    error_image = None
    reference_scale = max(math.sqrt(float(np.mean(reference[mask] ** 2))), EPS)
    for column, (name, title) in enumerate(zip(names, titles)):
        field = np.where(mask, arrays[name] * q_ref, np.nan)
        top_image = axes[0, column].imshow(
            field,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-field_limit,
            vmax=field_limit,
            interpolation="nearest",
            aspect="auto",
        )
        axes[0, column].set_title(title)
        if name == "reference":
            axes[1, column].set_facecolor("0.94")
            axes[1, column].text(0.5, 0.5, "Reference", ha="center", va="center", color="0.45", transform=axes[1, column].transAxes)
        else:
            error = 100.0 * (arrays[name] - reference) / reference_scale
            error = np.where(mask, error, np.nan)
            error_image = axes[1, column].imshow(
                error,
                origin="lower",
                extent=extent,
                cmap="RdBu_r",
                vmin=-error_limit_percent,
                vmax=error_limit_percent,
                interpolation="nearest",
                aspect="auto",
            )
        for row in range(2):
            axes[row, column].set_xlim(extent[0], extent[1])
            axes[row, column].set_ylim(extent[2], extent[3])
            axes[row, column].set_aspect("equal")
            axes[row, column].tick_params(labelsize=8)
        axes[1, column].set_xlabel("$x/D$")
    axes[0, 0].set_ylabel("$y/D$")
    axes[1, 0].set_ylabel("$y/D$")
    assert top_image is not None and error_image is not None
    fig.colorbar(top_image, ax=axes[0, :], shrink=0.83, pad=0.01, label=r"$q_y$ [W m$^{-2}$]")
    fig.colorbar(error_image, ax=axes[1, :], shrink=0.83, pad=0.01, label=r"$100(q_y-q_{y,ref})/\mathrm{RMS}(q_{y,ref})$ [%]")
    fig.suptitle(rf"Frozen cavity-to-cylinder transfer, Bird/DS2V Mach 10, seed {seed}")
    paths: list[str] = []
    for suffix in ("png", "pdf"):
        path = output / f"mv16a_cylinder_qy_six_panel_seed_{seed}.{suffix}"
        fig.savefig(path, dpi=500 if suffix == "png" else None, bbox_inches="tight")
        paths.append(path.name)
    plt.close(fig)
    return paths


def run_post(output_root: Path) -> dict[str, Any]:
    output = Path(output_root).resolve()
    _verify_manifest(output, "prediction_manifest.json")
    lock = verify_source_lock(output)
    campaign = Path(lock["campaign_root"])
    with np.load(output / "locked_cylinder_predictions.npz", allow_pickle=False) as source:
        predicted = {name: np.asarray(source[name]) for name in source.files}
    masks = predicted["fluid_mask"].astype(bool)
    raw_b10, q_refs = [], []
    for seed in SEEDS:
        paths = [_moment_path(campaign, seed, nout) for nout in B10_NOUT]
        metadata, additive = aggregate_moment_files(paths)
        fields = reconstruct_fields(metadata, additive)
        image, mask, _ = rasterize_fields(fields, predicted["frozen_weight_map"].shape)
        if not np.array_equal(mask, masks[0]):
            raise ValueError("B3/B10 cylinder raster masks differ")
        raw_b10.append(image[QY_INDEX].astype(np.float64))
        q_refs.append(float(fields["q_ref_W_m2"]))
    raw_b10_array = np.asarray(raw_b10, dtype=np.float64)
    target = leave_one_seed_out(raw_b10_array)
    methods = {
        "raw_b3": predicted["raw_b3_qy"],
        "vision_b3": predicted["vision_b3_qy"],
        "selected_b3": predicted["selected_b3_qy"],
        "dc_only_b3": predicted["dc_only_b3_qy"],
        "tsvd_b3": predicted["tsvd_b3_qy"],
        "permuted_b3": predicted["permuted_b3_qy"],
        "raw_b10": raw_b10_array,
    }
    per_seed, means = _per_seed_metrics(methods, target, masks)
    ratios = {name: value / max(means["raw_b10"], EPS) for name, value in means.items()}
    per_seed_ratios = {
        name: {
            seed: value / max(per_seed["raw_b10"][seed], EPS)
            for seed, value in records.items()
        }
        for name, records in per_seed.items()
    }
    full_dc_error = float(
        np.max(
            np.abs(
                np.mean(methods["selected_b3"], axis=(-2, -1))
                - np.mean(methods["raw_b3"], axis=(-2, -1))
            )
        )
    )
    contract = _mv15c_module().locked_protocol()["acceptance_gates"]
    gates = {
        "mean_no_worse_than_Raw_B10": ratios["selected_b3"] <= float(contract["maximum_each_condition_mean_ratio_to_Raw_B10"]),
        "every_seed_within_Raw_B10_cap": all(value <= float(contract["maximum_each_seed_ratio_to_Raw_B10"]) for value in per_seed_ratios["selected_b3"].values()),
        "selected_beats_Mamba_B3": means["selected_b3"] < means["vision_b3"],
        "selected_beats_TSVD_B3": means["selected_b3"] < means["tsvd_b3"],
        "selected_beats_Raw_B3": means["selected_b3"] < means["raw_b3"],
        "permuted_observation_degrades": means["permuted_b3"] >= (1.0 + float(contract["minimum_permutation_degradation_fraction"])) * means["selected_b3"],
        "DC_preserved_to_MV15C_tolerance": full_dc_error <= float(contract["maximum_DC_absolute_error"]),
        "all_four_locked_MV11_seeds_present": tuple(int(v) for v in predicted["seeds"]) == SEEDS,
        "prediction_locked_before_B10_target": True,
        "no_cylinder_parameter_selection_or_retraining": True,
        "no_DSMC_rerun": True,
    }
    _write_metrics_csv(output / "mv16a_cylinder_qy_metrics.csv", per_seed, per_seed["raw_b10"])
    top_limit = max(float(np.nanmax(np.abs(value[:, masks[0]]))) for value in (*methods.values(), target)) * float(q_refs[0])
    error_values = []
    for value in (methods["raw_b3"], methods["vision_b3"], methods["selected_b3"], methods["tsvd_b3"], methods["raw_b10"]):
        for index in range(len(SEEDS)):
            scale = max(math.sqrt(float(np.mean(target[index][masks[index]] ** 2))), EPS)
            error_values.append(np.abs(100.0 * (value[index][masks[index]] - target[index][masks[index]]) / scale))
    error_limit = max(float(np.quantile(np.concatenate(error_values), 0.995)), 1.0)
    figures: list[str] = []
    for index, seed in enumerate(SEEDS):
        figures.extend(
            _plot_six_panel(
                output,
                seed,
                {name: value[index] for name, value in methods.items() if name in {"raw_b3", "vision_b3", "selected_b3", "tsvd_b3", "raw_b10"}},
                target[index],
                masks[index],
                q_refs[index],
                top_limit,
                error_limit,
            )
        )
    all_pass = all(bool(value) for value in gates.values())
    summary = {
        "stage": STAGE,
        "status": "complete_MV16A_frozen_cylinder_transfer_audit",
        "decision": (
            "MV16A_cylinder_supports_frozen_cavity_B3_DCIR_QY_transfer"
            if all_pass
            else "MV16A_cylinder_does_not_support_frozen_cavity_B3_DCIR_QY_transfer_no_retuning"
        ),
        "scientific_classification": locked_protocol()["scientific_classification"],
        "original_MV11_tUD30_gate_pass": lock["original_MV11_tUD30_gate_pass"],
        "original_MV11_tUD30_gate_warning_preserved": True,
        "seed_qy_nrmse": per_seed,
        "mean_seed_qy_nrmse": means,
        "mean_ratios_to_Raw_B10": ratios,
        "per_seed_ratios_to_Raw_B10": per_seed_ratios,
        "maximum_full_raster_DC_absolute_error": full_dc_error,
        "gates": gates,
        "all_transfer_gates_pass": all_pass,
        "B3_nout": list(B3_NOUT),
        "B10_nout": list(B10_NOUT),
        "unused_QC_nout": list(QC_NOUT),
        "literal_speed_condition_outside_cavity_training_support": True,
        "cylinder_outcomes_used_for_tuning": False,
        "neural_retraining": False,
        "DSMC_rerun": False,
        "figures": figures,
    }
    _atomic_json(output / "summary.json", summary)
    np.savez_compressed(
        output / "mv16a_targets_and_fields.npz",
        seeds=np.asarray(SEEDS),
        target_qy=target,
        raw_b10_qy=raw_b10_array,
        fluid_mask=masks,
        q_ref_W_m2=np.asarray(q_refs),
    )
    return summary


def package_results(output_root: Path, return_directory: Path) -> dict[str, Any]:
    output = Path(output_root).resolve()
    returned = Path(return_directory).resolve()
    _verify_manifest(output, "prediction_manifest.json")
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    names = [
        PROTOCOL_FILE,
        "submission_lock.json",
        "source_lock_manifest.json",
        "cylinder_source_manifest.json",
        "raster_audit.json",
        "prediction_summary.json",
        "prediction_manifest.json",
        "locked_cylinder_predictions.npz",
        "summary.json",
        "mv16a_cylinder_qy_metrics.csv",
        "mv16a_targets_and_fields.npz",
    ] + [str(name) for name in summary["figures"]]
    accounting = output / "mv16a_slurm_accounting.psv"
    if accounting.is_file():
        names.append(accounting.name)
    files = [output / name for name in names]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    _write_manifest(output, "artifact_manifest.json", files)
    _verify_manifest(output, "artifact_manifest.json")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    returned.mkdir(parents=True, exist_ok=True)
    archive = returned / f"MV16A_FROZEN_CYLINDER_TRANSFER_BUNDLE_{timestamp}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in (*files, output / "artifact_manifest.json"):
            bundle.write(path, arcname=path.name)
    digest = _sha256(archive)
    verification = {
        "stage": STAGE,
        "archive": str(archive),
        "archive_sha256": digest,
        "decision": summary["decision"],
        "all_transfer_gates_pass": summary["all_transfer_gates_pass"],
    }
    _atomic_json(output / "return.json", verification)
    pointer = returned / RESULT_POINTER
    temporary = pointer.with_suffix(pointer.suffix + ".tmp")
    temporary.write_text(
        "\n".join(
            (
                f"MV16A_OUTPUT_ROOT={output}",
                f"MV16A_RESULT_ARCHIVE={archive}",
                f"MV16A_RESULT_ARCHIVE_SHA256={digest}",
                f"MV16A_ALL_TRANSFER_GATES_PASS={str(bool(summary['all_transfer_gates_pass'])).lower()}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(pointer)
    return verification


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify")
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--campaign-root", type=Path, required=True)
    prepare.add_argument("--mv15c-output-root", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    predict = sub.add_parser("predict")
    predict.add_argument("--output-root", type=Path, required=True)
    predict.add_argument("--batch-size", type=int, default=4)
    post = sub.add_parser("post")
    post.add_argument("--output-root", type=Path, required=True)
    package = sub.add_parser("package")
    package.add_argument("--output-root", type=Path, required=True)
    package.add_argument("--return-directory", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "verify":
        value = verify_contract()
    elif args.command == "prepare":
        value = prepare_lock(args.campaign_root, args.mv15c_output_root, args.output_root)
    elif args.command == "predict":
        value = run_prediction(args.output_root, batch_size=args.batch_size)
    elif args.command == "post":
        value = run_post(args.output_root)
    else:
        value = package_results(args.output_root, args.return_directory)
    print(_json_dumps(value))


if __name__ == "__main__":
    main()
