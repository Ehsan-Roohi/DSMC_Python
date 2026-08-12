"""MV8 exploratory recovery of non-equilibrium DSMC moment fields.

The stage reads immutable additive accumulators from completed DSMC
checkpoints.  It never derives stress or heat flux from an MV7 T/u prediction
and never launches a DSMC trajectory.  A development-only information gate is
evaluated before any model is allowed to train.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np


STAGE = "MV8_Mohammadzadeh_kinetic_moment_feasibility_pilot"
STATUS = "exploratory_lock_before_any_MV8_kinetic_model_outcome"
PROTOCOL_FILE = "mv8_kinetic_moment_feasibility_protocol.json"
OUTPUT_FIELDS = (
    "tau_xy_over_p_ref",
    "normal_stress_difference_over_p_ref",
    "qx_over_q_ref",
    "qy_over_q_ref",
)
AUXILIARY_FIELDS = (
    "rho_over_rho_ref",
    "u_over_U_lid",
    "v_over_U_lid",
    "T_over_T0",
)
CONDITION_FIELDS = ("log10_Kn", "U_lid_over_100")
ARCHITECTURES = ("nafnet_small", "mambairv2_tiny_adapted")
TRAINING_SEEDS = (2608091, 2608092, 2608093)
RESIDUAL_CAP_SIGMA = 4.0
EPSILON = np.finfo(np.float64).tiny
ADDITIVE_ARRAY_FIELDS = (
    "simulated_count",
    "m0",
    "m1",
    "m2",
    "energy",
    "energy_velocity",
)

DISPLAY_NAMES = {
    "raw_b1": "Raw DSMC, B=1",
    "gaussian_b1": "Gaussian, B=1",
    "tsvd_b1": "TSVD/POD, B=1",
    "nafnet_small": "NAFNet-Small, B=1",
    "mambairv2_tiny_adapted": "MambaIRv2, B=1",
    "raw_b10": "Raw DSMC, B=10",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _project_modules():
    from . import mohammadzadeh_architecture_screen as mv6
    from . import mohammadzadeh_mv5_reference as mv5ref
    from . import mohammadzadeh_vision_mv3 as mv3
    from . import mohammadzadeh_vision_mv5 as mv5
    from .mohammadzadeh_vision_mv2 import gaussian_like, tsvd
    from .ntc_checkpoint import load_ntc_checkpoint
    from .vhs_model import KB, PhysicalCavityConfig, VHSModel

    return {
        "mv3": mv3,
        "mv5": mv5,
        "mv5ref": mv5ref,
        "mv6": mv6,
        "gaussian_like": gaussian_like,
        "tsvd": tsvd,
        "load_ntc_checkpoint": load_ntc_checkpoint,
        "KB": KB,
        "PhysicalCavityConfig": PhysicalCavityConfig,
        "VHSModel": VHSModel,
    }


def protocol_path() -> Path:
    modules = _project_modules()
    return modules["mv5ref"].protocol_path().parent / PROTOCOL_FILE


def locked_protocol() -> dict[str, Any]:
    modules = _project_modules()
    path = protocol_path()
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("stage") != STAGE or value.get("status") != STATUS:
        raise ValueError("MV8 kinetic-moment protocol is absent or unlocked")
    execution = value["execution_matrix"]
    moment = value["moment_contract"]
    if (
        tuple(execution["architectures"]) != ARCHITECTURES
        or tuple(execution["training_initialization_seeds"]) != TRAINING_SEEDS
        or int(execution["model_tasks"]) != len(ARCHITECTURES) * len(TRAINING_SEEDS)
        or tuple(moment["outputs"]) != OUTPUT_FIELDS
        or tuple(moment["auxiliary_inputs"][:4]) != AUXILIARY_FIELDS
        or tuple(moment["auxiliary_inputs"][4:]) != CONDITION_FIELDS
        or not bool(moment["local_percent_error_forbidden"])
    ):
        raise ValueError("MV8 code differs from the locked protocol")
    source = value["source_contract"]
    root = path.parent
    checks = {
        "mv5_protocol_sha256": modules["mv5ref"].protocol_path(),
        "mv6_protocol_sha256": modules["mv6"].protocol_path(),
        "mv7_protocol_sha256": root / "mv7_jcp_budget_matrix_analysis_plan.json",
    }
    for key, source_path in checks.items():
        if _sha256(source_path) != source[key]:
            raise ValueError(f"MV8 source ancestry hash mismatch: {key}")
    return value


def verify_lock() -> dict[str, Any]:
    protocol = locked_protocol()
    return {
        "stage": STAGE,
        "status": "MV8_kinetic_moment_lock_verified_without_reading_model_outcomes",
        "protocol_sha256": _sha256(protocol_path()),
        "model_tasks": protocol["execution_matrix"]["model_tasks"],
        "outputs": list(OUTPUT_FIELDS),
        "primary_condition": protocol["execution_matrix"]["primary_condition"],
    }


def task_from_index(index: int) -> tuple[str, int]:
    total = len(ARCHITECTURES) * len(TRAINING_SEEDS)
    if not 0 <= index < total:
        raise ValueError(f"MV8 model task index must be in [0,{total - 1}]")
    return (
        ARCHITECTURES[index // len(TRAINING_SEEDS)],
        TRAINING_SEEDS[index % len(TRAINING_SEEDS)],
    )


def _config_from_summary(summary: Mapping[str, Any]):
    modules = _project_modules()
    raw = dict(summary["config"])
    vhs = modules["VHSModel"](**dict(raw.pop("vhs")))
    tuple_fields = (
        "left_wall_velocity",
        "right_wall_velocity",
        "bottom_wall_velocity",
        "top_wall_velocity",
    )
    for name in tuple_fields:
        if name in raw:
            raw[name] = tuple(float(item) for item in raw[name])
    return modules["PhysicalCavityConfig"](**raw, vhs=vhs)


def _verify_source_artifacts(directory: Path) -> dict[str, Any]:
    directory = Path(directory)
    manifest = json.loads(
        (directory / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    required = ("checkpoint.npz", "fields.npz", "block_fields.npz", "summary.json")
    for name in required:
        record = manifest.get("files", {}).get(name)
        path = directory / name
        if (
            not isinstance(record, Mapping)
            or not path.is_file()
            or path.stat().st_size != int(record["size_bytes"])
            or _sha256(path) != record["sha256"]
        ):
            raise ValueError(f"MV8 source artifact verification failed: {path}")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if "config" not in summary or not str(summary.get("status", "")).startswith("complete_"):
        raise ValueError(f"MV8 source summary is not complete: {directory}")
    return summary


def _payload_from_accumulator(accumulator: Any) -> dict[str, Any]:
    return {
        "samples": int(accumulator.samples),
        "simulated_count": np.asarray(accumulator.simulated_count),
        "m0": np.asarray(accumulator.m0),
        "m1": np.asarray(accumulator.m1),
        "m2": np.asarray(accumulator.m2),
        "energy": np.asarray(accumulator.energy),
        "energy_velocity": np.asarray(accumulator.energy_velocity),
    }


def merge_moment_payloads(payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not payloads:
        raise ValueError("cannot merge an empty moment-payload sequence")
    result = {
        "samples": sum(int(payload["samples"]) for payload in payloads),
    }
    for name in ADDITIVE_ARRAY_FIELDS:
        arrays = [np.asarray(payload[name], dtype=np.float64) for payload in payloads]
        if any(array.shape != arrays[0].shape for array in arrays[1:]):
            raise ValueError(f"moment payload shape mismatch for {name}")
        result[name] = np.sum(arrays, axis=0, dtype=np.float64)
    return result


def additive_payload_agreement(
    merged: Mapping[str, Any], full: Mapping[str, Any]
) -> dict[str, Any]:
    """Audit two additive payloads without dividing by locally vanishing entries.

    The full accumulator and the sum of independently accumulated blocks contain
    the same samples, but their floating-point addition orders differ.  A local
    elementwise relative test is therefore invalid for momentum components that
    cross zero.  The fixed global L-infinity scale below remains sensitive to a
    structural block mismatch while tolerating addition-order roundoff.
    """
    components: dict[str, dict[str, float]] = {}
    for name in ADDITIVE_ARRAY_FIELDS:
        first = np.asarray(merged[name], dtype=np.float64)
        second = np.asarray(full[name], dtype=np.float64)
        if first.shape != second.shape:
            raise ValueError(f"block/full additive-moment shape mismatch: {name}")
        if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
            raise ValueError(f"block/full additive-moment non-finite value: {name}")
        difference = first - second
        scale = max(
            float(np.max(np.abs(first))),
            float(np.max(np.abs(second))),
            EPSILON,
        )
        scaled_difference = difference / scale
        scaled_reference = second / scale
        reference_l2 = max(float(np.linalg.norm(scaled_reference.ravel())), EPSILON)
        components[name] = {
            "absolute_linf": float(np.max(np.abs(difference))),
            "fixed_scale": scale,
            "relative_linf": float(np.max(np.abs(difference)) / scale),
            "relative_l2": float(
                np.linalg.norm(scaled_difference.ravel()) / reference_l2
            ),
        }
    return {
        "sample_count_match": int(merged["samples"]) == int(full["samples"]),
        "merged_samples": int(merged["samples"]),
        "full_samples": int(full["samples"]),
        "maximum_relative_linf": max(
            record["relative_linf"] for record in components.values()
        ),
        "components": components,
    }


def moment_fields(payload: Mapping[str, Any], cfg: Any, kb: float) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Finalize stress, heat flux, and auxiliary fields from additive moments."""
    samples = int(payload["samples"])
    if samples <= 0:
        raise ValueError("moment payload has no samples")
    ncell = int(cfg.nx) * int(cfg.ny)
    m0 = np.asarray(payload["m0"], dtype=np.float64)
    m1 = np.asarray(payload["m1"], dtype=np.float64)
    m2 = np.asarray(payload["m2"], dtype=np.float64)
    energy = np.asarray(payload["energy"], dtype=np.float64)
    energy_velocity = np.asarray(payload["energy_velocity"], dtype=np.float64)
    if (
        m0.shape != (ncell,)
        or m1.shape != (ncell, 3)
        or m2.shape != (ncell, 3, 3)
        or energy.shape != (ncell,)
        or energy_velocity.shape != (ncell, 3)
        or np.any(m0 <= 0.0)
    ):
        raise ValueError("moment payload has invalid shapes or empty cells")

    velocity = m1 / m0[:, None]
    second = m2 / m0[:, None, None]
    covariance = second - np.einsum("ni,nj->nij", velocity, velocity)
    covariance = 0.5 * (covariance + np.swapaxes(covariance, 1, 2))
    eigenvalues = np.linalg.eigvalsh(covariance)
    covariance_scale = np.maximum(np.trace(covariance, axis1=1, axis2=2) / 3.0, EPSILON)
    minimum_eigenvalue_ratio = float(np.min(eigenvalues / covariance_scale[:, None]))

    number_density = m0 / float(samples) / float(cfg.cell_volume)
    pressure = float(cfg.vhs.mass) * number_density[:, None, None] * covariance
    mean_speed2 = energy / m0
    mean_energy_velocity = energy_velocity / m0[:, None]
    mean_velocity2 = np.sum(velocity**2, axis=1)
    second_times_mean = np.einsum("nij,nj->ni", second, velocity)
    central_energy_velocity = (
        mean_energy_velocity
        - velocity * mean_speed2[:, None]
        - 2.0 * second_times_mean
        + 2.0 * velocity * mean_velocity2[:, None]
    )
    heat_flux = (
        0.5
        * float(cfg.vhs.mass)
        * number_density[:, None]
        * central_energy_velocity
    )
    temperature = (
        float(cfg.vhs.mass)
        * np.maximum(np.trace(covariance, axis1=1, axis2=2), 0.0)
        / (3.0 * float(kb))
    )

    p_ref = float(cfg.number_density) * float(kb) * float(cfg.t0)
    thermal_speed = math.sqrt(float(kb) * float(cfg.t0) / float(cfg.vhs.mass))
    q_ref = p_ref * thermal_speed
    speed_ref = max(abs(float(cfg.lid_velocity_x)), EPSILON)
    shape = (int(cfg.ny), int(cfg.nx))
    outputs = np.stack(
        (
            pressure[:, 0, 1] / p_ref,
            (pressure[:, 0, 0] - pressure[:, 1, 1]) / p_ref,
            heat_flux[:, 0] / q_ref,
            heat_flux[:, 1] / q_ref,
        )
    ).reshape((len(OUTPUT_FIELDS), *shape))
    auxiliary = np.stack(
        (
            number_density / float(cfg.number_density),
            velocity[:, 0] / speed_ref,
            velocity[:, 1] / speed_ref,
            temperature / float(cfg.t0),
        )
    ).reshape((len(AUXILIARY_FIELDS), *shape))
    if not np.all(np.isfinite(outputs)) or not np.all(np.isfinite(auxiliary)):
        raise ValueError("non-finite finalized MV8 moment field")
    return outputs.astype(np.float32), auxiliary.astype(np.float32), {
        "p_ref_Pa": p_ref,
        "q_ref_W_m2": q_ref,
        "T0_K": float(cfg.t0),
        "minimum_covariance_eigenvalue_over_isotropic_scale": minimum_eigenvalue_ratio,
    }


def _relative_array_difference(first: np.ndarray, second: np.ndarray) -> float:
    scale = max(float(np.max(np.abs(second))), 1.0)
    return float(np.max(np.abs(first - second)) / scale)


def load_moment_source(directory: Path) -> dict[str, Any]:
    modules = _project_modules()
    directory = Path(directory)
    summary = _verify_source_artifacts(directory)
    cfg = _config_from_summary(summary)
    checkpoint = modules["load_ntc_checkpoint"](directory / "checkpoint.npz", cfg)
    full_payload = _payload_from_accumulator(checkpoint.moments)
    block_root = checkpoint.block_accumulators
    if not isinstance(block_root, Mapping):
        raise ValueError(f"checkpoint has no block accumulators: {directory}")
    block_mapping = block_root.get("block_moments")
    if not isinstance(block_mapping, Mapping) or len(block_mapping) != 10:
        raise ValueError(f"MV8 requires ten additive checkpoint blocks: {directory}")
    block_payloads = [dict(block_mapping[key]) for key in sorted(block_mapping)]
    merged = merge_moment_payloads(block_payloads)
    additive_agreement = additive_payload_agreement(merged, full_payload)

    full, full_aux, full_diagnostics = moment_fields(
        full_payload, cfg, modules["KB"]
    )
    block_outputs, block_auxiliary, diagnostics = [], [], []
    for payload in block_payloads:
        output, auxiliary, record = moment_fields(payload, cfg, modules["KB"])
        block_outputs.append(output)
        block_auxiliary.append(auxiliary)
        diagnostics.append(record)

    with np.load(directory / "fields.npz", allow_pickle=False) as stored:
        q_full = np.stack((np.asarray(stored["qx"]), np.asarray(stored["qy"])))
    with np.load(directory / "block_fields.npz", allow_pickle=False) as stored:
        q_blocks = np.stack(
            (np.asarray(stored["qx"]), np.asarray(stored["qy"])), axis=1
        )
    scale = np.asarray(
        [full_diagnostics["q_ref_W_m2"], full_diagnostics["q_ref_W_m2"]]
    )[:, None, None]
    reconstructed_q_full = full[2:] * scale
    reconstructed_q_blocks = (
        np.stack(block_outputs)[:, 2:]
        * scale[None]
    )
    q_relative_difference = max(
        _relative_array_difference(reconstructed_q_full, q_full),
        _relative_array_difference(reconstructed_q_blocks, q_blocks),
    )
    minimum_eigenvalue_ratio = min(
        full_diagnostics["minimum_covariance_eigenvalue_over_isotropic_scale"],
        *(record["minimum_covariance_eigenvalue_over_isotropic_scale"] for record in diagnostics),
    )
    return {
        "directory": directory,
        "summary_status": summary["status"],
        "condition_id": str(summary.get("condition_id", "")),
        "seed": int(summary.get("seed", summary["config"]["seed"])),
        "full": full,
        "full_auxiliary": full_aux,
        "blocks": np.stack(block_outputs),
        "block_auxiliary": np.stack(block_auxiliary),
        "scales": np.asarray(
            [
                full_diagnostics["p_ref_Pa"],
                full_diagnostics["p_ref_Pa"],
                full_diagnostics["q_ref_W_m2"],
                full_diagnostics["q_ref_W_m2"],
            ],
            dtype=np.float64,
        ),
        "q_reconstruction_relative_difference": q_relative_difference,
        "minimum_covariance_eigenvalue_ratio": minimum_eigenvalue_ratio,
        "block_full_additive_agreement": additive_agreement,
    }


def _conditioned_image(
    output: np.ndarray,
    auxiliary: np.ndarray,
    condition: Mapping[str, Any],
) -> np.ndarray:
    ny, nx = output.shape[-2:]
    condition_channels = np.stack(
        (
            np.full((ny, nx), np.log10(float(condition["knudsen"]))),
            np.full((ny, nx), float(condition["lid_speed_m_per_s"]) / 100.0),
        )
    ).astype(np.float32)
    return np.concatenate((output, auxiliary, condition_channels)).astype(np.float32)


def _leave_one_out(values: Mapping[int, np.ndarray]) -> dict[int, np.ndarray]:
    seeds = tuple(values)
    if len(seeds) < 3:
        raise ValueError("MV8 leave-one-out target requires at least three seeds")
    return {
        seed: np.mean(
            [np.asarray(values[other], dtype=np.float64) for other in seeds if other != seed],
            axis=0,
        ).astype(np.float32)
        for seed in seeds
    }


def _build_b1_split(
    sources: Mapping[str, Mapping[int, Mapping[str, Any]]],
    targets: Mapping[str, Mapping[int, np.ndarray]],
    selection: Mapping[str, Sequence[int]],
    specs: Mapping[str, Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, y, condition_ids, identities, scales = [], [], [], [], []
    for condition_id, seeds in selection.items():
        for seed in seeds:
            source = sources[condition_id][int(seed)]
            for block_index, (output, auxiliary) in enumerate(
                zip(source["blocks"], source["block_auxiliary"])
            ):
                x.append(_conditioned_image(output, auxiliary, specs[condition_id]))
                y.append(targets[condition_id][int(seed)])
                condition_ids.append(condition_id)
                identities.append((int(seed), block_index, 1))
                scales.append(source["scales"])
    return (
        np.stack(x),
        np.stack(y),
        np.asarray(condition_ids, dtype="U32"),
        np.asarray(identities, dtype=np.int64),
        np.stack(scales),
    )


def _build_b10_split(
    sources: Mapping[str, Mapping[int, Mapping[str, Any]]],
    targets: Mapping[str, Mapping[int, np.ndarray]],
    selection: Mapping[str, Sequence[int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw, target, condition_ids, identities, scales = [], [], [], [], []
    for condition_id, seeds in selection.items():
        for seed in seeds:
            source = sources[condition_id][int(seed)]
            raw.append(source["full"])
            target.append(targets[condition_id][int(seed)])
            condition_ids.append(condition_id)
            identities.append((int(seed), 0, 10))
            scales.append(source["scales"])
    return (
        np.stack(raw),
        np.stack(target),
        np.asarray(condition_ids, dtype="U32"),
        np.asarray(identities, dtype=np.int64),
        np.stack(scales),
    )


def field_metrics(candidate: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    candidate = np.asarray(candidate, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if candidate.shape != target.shape or candidate.ndim != 4:
        raise ValueError("MV8 metric arrays must be matching NCHW tensors")
    per_field: dict[str, float] = {}
    for index, name in enumerate(OUTPUT_FIELDS):
        error = np.sqrt(np.mean((candidate[:, index] - target[:, index]) ** 2))
        reference = np.sqrt(np.mean(target[:, index] ** 2))
        per_field[name] = float(error / max(reference, 1.0e-12))
    return {
        "per_field_nrmse": per_field,
        "composite_nrmse": float(np.mean(list(per_field.values()))),
    }


def _select_classical(
    validation_raw: np.ndarray,
    validation_target: np.ndarray,
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    modules = _project_modules()
    records: dict[str, list[dict[str, Any]]] = {"gaussian": [], "tsvd": []}
    gaussian_candidates = []
    for passes in protocol["selection_contract"]["gaussian_pass_candidates"]:
        value = modules["gaussian_like"](validation_raw, int(passes))
        metric = field_metrics(value, validation_target)
        records["gaussian"].append({"passes": int(passes), **metric})
        gaussian_candidates.append((metric["composite_nrmse"], int(passes), value))
    tsvd_candidates = []
    for rank in protocol["selection_contract"]["tsvd_rank_candidates"]:
        value = modules["tsvd"](validation_raw, int(rank))
        metric = field_metrics(value, validation_target)
        records["tsvd"].append({"rank": int(rank), **metric})
        tsvd_candidates.append((metric["composite_nrmse"], int(rank), value))
    gaussian_best = min(gaussian_candidates, key=lambda item: (item[0], item[1]))
    tsvd_best = min(tsvd_candidates, key=lambda item: (item[0], item[1]))
    return (
        {
            "gaussian_passes": gaussian_best[1],
            "tsvd_rank": tsvd_best[1],
            "records": records,
        },
        gaussian_best[2].astype(np.float32),
        tsvd_best[2].astype(np.float32),
    )


def run_assembly(
    existing_m3_root: Path,
    mv3_root: Path,
    reference_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    modules = _project_modules()
    protocol = locked_protocol()
    mv3_protocol = modules["mv3"].locked_protocol()
    mv5_protocol = modules["mv5"].locked_protocol()
    development_specs = modules["mv3"]._condition_map(mv3_protocol)
    confirmatory_specs = modules["mv5ref"].condition_map(mv5_protocol)
    train_split = {
        key: tuple(int(seed) for seed in values)
        for key, values in mv5_protocol["development_seed_split"]["train"].items()
    }
    validation_split = {
        key: tuple(int(seed) for seed in values)
        for key, values in mv5_protocol["development_seed_split"]["validation"].items()
    }
    test_split = {
        key: tuple(int(seed) for seed in value["evaluation_seeds"])
        for key, value in confirmatory_specs.items()
    }

    development: dict[str, dict[int, dict[str, Any]]] = {}
    confirmatory: dict[str, dict[int, dict[str, Any]]] = {}
    audit_rows: list[dict[str, Any]] = []
    print("STAGE=MV8_additive_moment_source_audit", flush=True)
    for condition_id, condition in development_specs.items():
        development[condition_id] = {}
        seeds = tuple(train_split[condition_id]) + tuple(validation_split[condition_id])
        for seed in seeds:
            directory = modules["mv3"]._source_directory(
                condition, seed, Path(existing_m3_root), Path(mv3_root)
            )
            source = load_moment_source(directory)
            development[condition_id][seed] = source
            audit_rows.append(
                {
                    "role": "development",
                    "condition_id": condition_id,
                    "seed": seed,
                    "directory": str(directory),
                    "summary_status": source["summary_status"],
                    "q_reconstruction_relative_difference": source[
                        "q_reconstruction_relative_difference"
                    ],
                    "minimum_covariance_eigenvalue_ratio": source[
                        "minimum_covariance_eigenvalue_ratio"
                    ],
                    "block_full_sample_count_match": source[
                        "block_full_additive_agreement"
                    ]["sample_count_match"],
                    "block_full_maximum_relative_linf": source[
                        "block_full_additive_agreement"
                    ]["maximum_relative_linf"],
                    "block_full_component_relative_linf_json": json.dumps(
                        {
                            name: record["relative_linf"]
                            for name, record in source[
                                "block_full_additive_agreement"
                            ]["components"].items()
                        },
                        sort_keys=True,
                    ),
                }
            )
    for condition_id, condition in confirmatory_specs.items():
        confirmatory[condition_id] = {}
        for seed in test_split[condition_id]:
            directory = (
                Path(reference_root) / "references" / condition_id / f"seed_{seed}"
            )
            source = load_moment_source(directory)
            confirmatory[condition_id][seed] = source
            audit_rows.append(
                {
                    "role": "confirmatory",
                    "condition_id": condition_id,
                    "seed": seed,
                    "directory": str(directory),
                    "summary_status": source["summary_status"],
                    "q_reconstruction_relative_difference": source[
                        "q_reconstruction_relative_difference"
                    ],
                    "minimum_covariance_eigenvalue_ratio": source[
                        "minimum_covariance_eigenvalue_ratio"
                    ],
                    "block_full_sample_count_match": source[
                        "block_full_additive_agreement"
                    ]["sample_count_match"],
                    "block_full_maximum_relative_linf": source[
                        "block_full_additive_agreement"
                    ]["maximum_relative_linf"],
                    "block_full_component_relative_linf_json": json.dumps(
                        {
                            name: record["relative_linf"]
                            for name, record in source[
                                "block_full_additive_agreement"
                            ]["components"].items()
                        },
                        sort_keys=True,
                    ),
                }
            )

    development_full = {
        condition: {seed: source["full"] for seed, source in values.items()}
        for condition, values in development.items()
    }
    confirmatory_full = {
        condition: {seed: source["full"] for seed, source in values.items()}
        for condition, values in confirmatory.items()
    }
    development_targets: dict[str, dict[int, np.ndarray]] = {}
    for condition_id, train_seeds in train_split.items():
        training_values = {seed: development_full[condition_id][seed] for seed in train_seeds}
        development_targets[condition_id] = _leave_one_out(training_values)
        training_mean = np.mean(list(training_values.values()), axis=0).astype(np.float32)
        for seed in validation_split[condition_id]:
            development_targets[condition_id][seed] = training_mean.copy()
    confirmatory_targets = {
        condition: _leave_one_out(values)
        for condition, values in confirmatory_full.items()
    }

    train = _build_b1_split(
        development, development_targets, train_split, development_specs
    )
    validation = _build_b1_split(
        development, development_targets, validation_split, development_specs
    )
    test = _build_b1_split(
        confirmatory, confirmatory_targets, test_split, confirmatory_specs
    )
    validation_b10 = _build_b10_split(
        development, development_targets, validation_split
    )
    test_b10 = _build_b10_split(confirmatory, confirmatory_targets, test_split)

    validation_raw = validation[0][:, : len(OUTPUT_FIELDS)]
    classical, _, _ = _select_classical(validation_raw, validation[1], protocol)
    test_raw = test[0][:, : len(OUTPUT_FIELDS)]
    test_gaussian = modules["gaussian_like"](
        test_raw, int(classical["gaussian_passes"])
    ).astype(np.float32)
    test_tsvd = modules["tsvd"](
        test_raw, int(classical["tsvd_rank"])
    ).astype(np.float32)

    raw_b1_metric = field_metrics(validation_raw, validation[1])
    raw_b10_metric = field_metrics(validation_b10[0], validation_b10[1])
    improved_fields = sum(
        raw_b10_metric["per_field_nrmse"][name]
        < raw_b1_metric["per_field_nrmse"][name]
        for name in OUTPUT_FIELDS
    )
    gates = protocol["pre_model_feasibility_gates"]
    maximum_q_difference = max(
        float(row["q_reconstruction_relative_difference"]) for row in audit_rows
    )
    minimum_eigenvalue_ratio = min(
        float(row["minimum_covariance_eigenvalue_ratio"]) for row in audit_rows
    )
    maximum_additive_difference = max(
        float(row["block_full_maximum_relative_linf"]) for row in audit_rows
    )
    additive_tolerance = float(
        gates["block_full_additive_moment_fixed_scale_relative_linf_tolerance"]
    )
    checks = {
        "all_source_checkpoint_artifacts_hash_verified": True,
        "all_block_sample_counts_sum_to_full_sample_count": all(
            bool(row["block_full_sample_count_match"]) for row in audit_rows
        ),
        "block_sums_match_full_additive_accumulators_with_fixed_scale_tolerance": (
            maximum_additive_difference <= additive_tolerance
        ),
        "all_reconstructed_moment_fields_finite": all(
            np.all(np.isfinite(array))
            for array in (train[0], train[1], validation[0], validation[1], test[0], test[1])
        ),
        "pressure_covariance_positive_semidefinite": minimum_eigenvalue_ratio
        >= -float(gates["pressure_covariance_positive_semidefinite_tolerance"]),
        "stored_and_reconstructed_heat_flux_match": maximum_q_difference
        <= float(gates["stored_and_reconstructed_heat_flux_relative_tolerance"]),
        "development_validation_raw_B10_composite_better_than_raw_B1": raw_b10_metric[
            "composite_nrmse"
        ]
        < raw_b1_metric["composite_nrmse"],
        "minimum_individual_fields_improved_by_raw_B10": improved_fields
        >= int(gates["minimum_individual_fields_improved_by_raw_B10"]),
        "confirmatory_results_not_used_for_gate": True,
    }
    decision = (
        "proceed_to_MV8_B1_kinetic_models"
        if all(checks.values())
        else "hold_MV8_models_return_information_audit_only"
    )

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_root / "dataset.npz",
        train_x=train[0],
        train_y=train[1],
        train_condition=train[2],
        train_identity=train[3],
        validation_x=validation[0],
        validation_y=validation[1],
        validation_condition=validation[2],
        validation_identity=validation[3],
        validation_raw10=validation_b10[0],
        validation_target10=validation_b10[1],
        test_x=test[0],
        test_y=test[1],
        test_condition=test[2],
        test_identity=test[3],
        test_scale=test[4],
        test_gaussian=test_gaussian,
        test_tsvd=test_tsvd,
        test_raw10=test_b10[0],
        test_target10=test_b10[1],
        test_condition10=test_b10[2],
        test_identity10=test_b10[3],
        test_scale10=test_b10[4],
    )
    audit_path = output_root / "source_moment_audit.csv"
    with audit_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)
    summary = {
        "stage": STAGE,
        "status": "complete_MV8_additive_moment_assembly_and_information_gate",
        "protocol_sha256": _sha256(protocol_path()),
        "source_count": len(audit_rows),
        "sample_counts": {
            "train_B1": len(train[0]),
            "validation_B1": len(validation[0]),
            "confirmatory_B1": len(test[0]),
            "validation_B10": len(validation_b10[0]),
            "confirmatory_B10": len(test_b10[0]),
        },
        "maximum_q_reconstruction_relative_difference": maximum_q_difference,
        "maximum_block_full_additive_moment_fixed_scale_relative_linf": (
            maximum_additive_difference
        ),
        "block_full_additive_moment_fixed_scale_relative_linf_tolerance": (
            additive_tolerance
        ),
        "minimum_covariance_eigenvalue_ratio": minimum_eigenvalue_ratio,
        "development_validation_information_test": {
            "raw_B1": raw_b1_metric,
            "raw_B10": raw_b10_metric,
            "individual_fields_improved": improved_fields,
        },
        "classical_selection_development_only": classical,
        "checks": checks,
        "decision": decision,
    }
    _atomic_json(output_root / "assembly_summary.json", summary)
    _atomic_json(
        output_root / "assembly_manifest.json",
        {
            "stage": STAGE,
            "files": {
                name: {
                    "sha256": _sha256(output_root / name),
                    "size_bytes": (output_root / name).stat().st_size,
                }
                for name in ("dataset.npz", "source_moment_audit.csv", "assembly_summary.json")
            },
        },
    )
    print(f"MV8_INFORMATION_GATE={decision}", flush=True)
    return summary


def _fit_scaling(train_x: np.ndarray, train_y: np.ndarray) -> dict[str, np.ndarray]:
    input_mean = train_x.mean(axis=(0, 2, 3), keepdims=True)
    input_std = np.maximum(train_x.std(axis=(0, 2, 3), keepdims=True), 1.0e-6)
    condition_centers = (-1.1505149978319906, 2.5)
    condition_scales = (0.3010299956639812, 3.0)
    for offset, (center, scale) in enumerate(zip(condition_centers, condition_scales)):
        index = train_x.shape[1] - len(CONDITION_FIELDS) + offset
        input_mean[0, index, 0, 0] = center
        input_std[0, index, 0, 0] = scale
    residual = train_y - train_x[:, : len(OUTPUT_FIELDS)]
    residual_std = np.maximum(
        residual.std(axis=(0, 2, 3), keepdims=True), 1.0e-4
    )
    return {
        "input_mean": input_mean.astype(np.float32),
        "input_std": input_std.astype(np.float32),
        "residual_std": residual_std.astype(np.float32),
    }


def _train_model(
    architecture: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    scaling: Mapping[str, np.ndarray],
    *,
    seed: int,
    epochs: int,
    batch_size: int,
) -> tuple[Any, dict[str, Any]]:
    modules = _project_modules()
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tx = torch.from_numpy(
        ((train_x - scaling["input_mean"]) / scaling["input_std"]).astype(np.float32)
    )
    ty = torch.from_numpy(
        ((train_y - train_x[:, : len(OUTPUT_FIELDS)]) / scaling["residual_std"]).astype(np.float32)
    )
    vx = torch.from_numpy(
        ((validation_x - scaling["input_mean"]) / scaling["input_std"]).astype(np.float32)
    )
    vy = torch.from_numpy(
        ((validation_y - validation_x[:, : len(OUTPUT_FIELDS)]) / scaling["residual_std"]).astype(np.float32)
    )
    loader = DataLoader(
        TensorDataset(tx, ty),
        batch_size=min(batch_size, len(tx)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    model = modules["mv6"].build_architecture(
        architecture, int(train_x.shape[1]), out_channels=len(OUTPUT_FIELDS)
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-5)

    def loss_function(prediction, target):
        pixel = torch.mean((prediction - target) ** 2)
        grad_x = torch.mean(
            ((prediction[..., 1:] - prediction[..., :-1]) - (target[..., 1:] - target[..., :-1])) ** 2
        )
        grad_y = torch.mean(
            ((prediction[..., 1:, :] - prediction[..., :-1, :]) - (target[..., 1:, :] - target[..., :-1, :])) ** 2
        )
        return pixel + 0.10 * (grad_x + grad_y)

    best_state: dict[str, Any] | None = None
    best_value, best_epoch, stale = float("inf"), 0, 0
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            latent = model(xb)
            bounded = RESIDUAL_CAP_SIGMA * torch.tanh(latent / RESIDUAL_CAP_SIGMA)
            loss = loss_function(bounded, yb)
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu()) * len(xb)
        model.eval()
        with torch.no_grad():
            latent = model(vx.to(device))
            bounded = RESIDUAL_CAP_SIGMA * torch.tanh(latent / RESIDUAL_CAP_SIGMA)
            validation = float(loss_function(bounded, vy.to(device)).cpu())
        history.append(
            {"epoch": epoch, "train_loss": running / len(tx), "validation_loss": validation}
        )
        if validation < best_value - 1.0e-7:
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            best_value, best_epoch, stale = validation, epoch, 0
        else:
            stale += 1
        if stale >= 25:
            break
    if best_state is None:
        raise RuntimeError("MV8 training produced no checkpoint")
    model.load_state_dict(best_state)
    model.to("cpu").eval()
    return model, {
        "architecture": architecture,
        "training_seed": seed,
        "device": str(device),
        "seconds": time.perf_counter() - started,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_loss": best_value,
        "history": history,
    }


def _predict(model: Any, x: np.ndarray, scaling: Mapping[str, np.ndarray], batch_size: int) -> tuple[np.ndarray, float]:
    import torch

    normalized = ((x - scaling["input_mean"]) / scaling["input_std"]).astype(np.float32)
    values, maximum = [], 0.0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            latent = model(torch.from_numpy(normalized[start : start + batch_size]))
            bounded = RESIDUAL_CAP_SIGMA * torch.tanh(latent / RESIDUAL_CAP_SIGMA)
            maximum = max(maximum, float(torch.max(torch.abs(bounded))))
            values.append(
                x[start : start + batch_size, : len(OUTPUT_FIELDS)]
                + bounded.numpy() * scaling["residual_std"]
            )
    return np.concatenate(values).astype(np.float32), maximum


def _task_directory(root: Path, architecture: str, seed: int) -> Path:
    return Path(root) / "tasks" / architecture / f"training_seed_{seed}"


def run_model_task(
    output_root: Path,
    *,
    architecture: str,
    training_seed: int,
    epochs: int,
    batch_size: int,
) -> dict[str, Any]:
    protocol = locked_protocol()
    if architecture not in ARCHITECTURES or training_seed not in TRAINING_SEEDS:
        raise ValueError("MV8 model task is outside the locked matrix")
    output_root = Path(output_root)
    assembly = json.loads((output_root / "assembly_summary.json").read_text(encoding="utf-8"))
    directory = _task_directory(output_root, architecture, training_seed)
    directory.mkdir(parents=True, exist_ok=False)
    if assembly["decision"] != "proceed_to_MV8_B1_kinetic_models":
        skipped = {
            "stage": STAGE,
            "status": "skipped_MV8_model_after_failed_information_gate",
            "architecture": architecture,
            "training_seed": training_seed,
            "assembly_decision": assembly["decision"],
        }
        _atomic_json(directory / "skipped.json", skipped)
        _atomic_json(
            directory / "artifact_manifest.json",
            {
                "stage": STAGE,
                "files": {
                    "skipped.json": {
                        "sha256": _sha256(directory / "skipped.json"),
                        "size_bytes": (directory / "skipped.json").stat().st_size,
                    }
                },
            },
        )
        return skipped

    with np.load(output_root / "dataset.npz", allow_pickle=False) as data:
        train_x = np.asarray(data["train_x"])
        train_y = np.asarray(data["train_y"])
        validation_x = np.asarray(data["validation_x"])
        validation_y = np.asarray(data["validation_y"])
        test_x = np.asarray(data["test_x"])
        test_y = np.asarray(data["test_y"])
        test_condition = np.asarray(data["test_condition"])
        test_identity = np.asarray(data["test_identity"])
    scaling = _fit_scaling(train_x, train_y)
    model, training = _train_model(
        architecture,
        train_x,
        train_y,
        validation_x,
        validation_y,
        scaling,
        seed=training_seed,
        epochs=epochs,
        batch_size=batch_size,
    )
    validation_ungated, validation_bound = _predict(
        model, validation_x, scaling, batch_size
    )
    test_ungated, test_bound = _predict(model, test_x, scaling, batch_size)
    validation_raw = validation_x[:, : len(OUTPUT_FIELDS)]
    test_raw = test_x[:, : len(OUTPUT_FIELDS)]
    alpha_records = []
    for alpha in protocol["selection_contract"]["residual_alpha_candidates"]:
        value = validation_raw + float(alpha) * (validation_ungated - validation_raw)
        alpha_records.append({"alpha": float(alpha), **field_metrics(value, validation_y)})
    selected = min(alpha_records, key=lambda item: (item["composite_nrmse"], item["alpha"]))
    alpha = float(selected["alpha"])
    prediction = test_raw + alpha * (test_ungated - test_raw)

    import torch

    torch.save(
        {
            "stage": STAGE,
            "architecture": architecture,
            "training_seed": training_seed,
            "state_dict": model.state_dict(),
            "scaling": {key: np.asarray(value) for key, value in scaling.items()},
            "residual_alpha": alpha,
            "input_fields": (*OUTPUT_FIELDS, *AUXILIARY_FIELDS, *CONDITION_FIELDS),
            "output_fields": OUTPUT_FIELDS,
        },
        directory / "model.pt",
    )
    np.savez_compressed(
        directory / "predictions.npz",
        identity_condition=test_condition,
        identity_numeric=test_identity,
        raw=test_raw,
        target=test_y,
        architecture_prediction=prediction,
    )
    checks = {
        "assembly_information_gate_passed": True,
        "confirmatory_not_used_for_training_or_selection": True,
        "finite_prediction": bool(np.all(np.isfinite(prediction))),
        "bounded_residual": max(validation_bound, test_bound) <= RESIDUAL_CAP_SIGMA + 1e-6,
    }
    summary = {
        "stage": STAGE,
        "status": "complete_MV8_B1_kinetic_model_task",
        "architecture": architecture,
        "training_seed": training_seed,
        "protocol_sha256": _sha256(protocol_path()),
        "training": training,
        "residual_alpha_selection_development_only": {
            "selected": alpha,
            "candidates": alpha_records,
        },
        "confirmatory_metrics": field_metrics(prediction, test_y),
        "checks": checks,
        "decision": "accept_MV8_model_task" if all(checks.values()) else "hold_MV8_model_task",
    }
    _atomic_json(directory / "summary.json", summary)
    _atomic_json(
        directory / "artifact_manifest.json",
        {
            "stage": STAGE,
            "files": {
                name: {
                    "sha256": _sha256(directory / name),
                    "size_bytes": (directory / name).stat().st_size,
                }
                for name in ("model.pt", "predictions.npz", "summary.json")
            },
        },
    )
    return summary


def _verify_task(directory: Path, architecture: str, seed: int) -> tuple[dict[str, Any], np.ndarray]:
    manifest = json.loads((directory / "artifact_manifest.json").read_text(encoding="utf-8"))
    for name, record in manifest["files"].items():
        path = directory / name
        if (
            not path.is_file()
            or path.stat().st_size != int(record["size_bytes"])
            or _sha256(path) != record["sha256"]
        ):
            raise ValueError(f"MV8 task artifact failed verification: {path}")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if (
        summary.get("status") != "complete_MV8_B1_kinetic_model_task"
        or summary.get("decision") != "accept_MV8_model_task"
        or summary.get("architecture") != architecture
        or int(summary.get("training_seed", -1)) != seed
    ):
        raise ValueError(f"MV8 task summary contract failed: {directory}")
    with np.load(directory / "predictions.npz", allow_pickle=False) as data:
        prediction = np.asarray(data["architecture_prediction"]).copy()
    return summary, prediction


def _per_seed_metrics(
    method: np.ndarray,
    target: np.ndarray,
    conditions: np.ndarray,
    identities: np.ndarray,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for condition in np.unique(conditions):
        result[str(condition)] = {}
        for seed in np.unique(identities[conditions == condition, 0]):
            mask = (conditions == condition) & (identities[:, 0] == seed)
            result[str(condition)][str(int(seed))] = field_metrics(method[mask], target[mask])
    return result


def _physical_figure(
    output: Path,
    field_index: int,
    methods: Mapping[str, np.ndarray],
    reference: np.ndarray,
    scale: float,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    meta = (
        ("tau_xy", r"$P_{xy}$", "Pa", r"$100\,\Delta P_{xy}/p_{ref}$ [%]"),
        ("normal_stress_difference", r"$P_{xx}-P_{yy}$", "Pa", r"$100\,\Delta(P_{xx}-P_{yy})/p_{ref}$ [%]"),
        ("qx", r"$q_x$", r"W m$^{-2}$", r"$100\,\Delta q_x/q_{ref}$ [%]"),
        ("qy", r"$q_y$", r"W m$^{-2}$", r"$100\,\Delta q_y/q_{ref}$ [%]"),
    )[field_index]
    key, symbol, unit, error_label = meta
    columns = ("reference", "raw_b1", "gaussian_b1", "tsvd_b1", "nafnet_small", "mambairv2_tiny_adapted", "raw_b10")
    titles = (
        "Reference",
        "Raw DSMC\nB=1",
        "Gaussian\nB=1",
        "TSVD/POD\nB=1",
        "NAFNet-Small\nB=1",
        "MambaIRv2\nB=1",
        "Raw DSMC\nB=10",
    )
    normalized = {"reference": reference, **methods}
    physical = {name: normalized[name] * scale for name in columns}
    errors = {
        name: 100.0 * (normalized[name] - reference)
        for name in columns
        if name != "reference"
    }
    physical_values = np.concatenate(
        [np.abs(value).ravel() for value in physical.values()]
    )
    physical_limit = max(float(np.quantile(physical_values, 0.995)), 1e-12)
    physical_clipped = int(
        sum(np.count_nonzero(np.abs(value) > physical_limit) for value in physical.values())
    )
    error_values = np.concatenate([np.abs(value).ravel() for value in errors.values()])
    error_limit = max(float(np.quantile(error_values, 0.995)), 1e-4)
    clipped = int(sum(np.count_nonzero(np.abs(value) > error_limit) for value in errors.values()))

    fig, axes = plt.subplots(2, len(columns), figsize=(15.4, 6.1), constrained_layout=True)
    physical_norm = TwoSlopeNorm(vmin=-physical_limit, vcenter=0.0, vmax=physical_limit)
    error_norm = TwoSlopeNorm(vmin=-error_limit, vcenter=0.0, vmax=error_limit)
    physical_artist = error_artist = None
    levels = np.linspace(-physical_limit, physical_limit, 41)
    error_levels = np.linspace(-error_limit, error_limit, 41)
    for column, (name, title) in enumerate(zip(columns, titles)):
        physical_artist = axes[0, column].contourf(
            physical[name], levels=levels, cmap="RdBu_r", norm=physical_norm, extend="both"
        )
        axes[0, column].set_title(title, pad=7, fontsize=10.5)
        error_field = np.zeros_like(reference) if name == "reference" else errors[name]
        error_artist = axes[1, column].contourf(
            error_field, levels=error_levels, cmap="RdBu_r", norm=error_norm, extend="both"
        )
        for row in range(2):
            axis = axes[row, column]
            axis.set_aspect("equal")
            axis.set_xlim(0, reference.shape[1] - 1)
            axis.set_ylim(0, reference.shape[0] - 1)
            axis.set_xticks([0, (reference.shape[1] - 1) / 2, reference.shape[1] - 1])
            axis.set_yticks([0, (reference.shape[0] - 1) / 2, reference.shape[0] - 1])
            axis.set_xticklabels(["0", "0.5", "1"] if row == 1 else [])
            axis.set_yticklabels(["0", "0.5", "1"] if column == 0 else [])
            if row == 1:
                axis.set_xlabel(r"$x/L$")
            if column == 0:
                axis.set_ylabel(r"$y/L$")
    assert physical_artist is not None and error_artist is not None
    first = fig.colorbar(physical_artist, ax=axes[0, :], shrink=0.88, pad=0.012)
    first.set_label(f"{symbol} [{unit}]")
    second = fig.colorbar(error_artist, ax=axes[1, :], shrink=0.88, pad=0.012)
    second.set_label(error_label)
    png = output / f"mv8_{key}_B1_vs_B10_physical_contours.png"
    pdf = output / f"mv8_{key}_B1_vs_B10_physical_contours.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "field": OUTPUT_FIELDS[field_index],
        "png": png.name,
        "pdf": pdf.name,
        "physical_limit": physical_limit,
        "physical_values_clipped_by_robust_display_limit": physical_clipped,
        "error_percent_limit": error_limit,
        "error_values_clipped_by_robust_display_limit": clipped,
    }


def _create_archive(output_root: Path, names: Sequence[Path]) -> Path:
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = Path.home() / f"MOHAMMADZADEH_MV8_KINETIC_MOMENT_PILOT_{tag}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as stream:
        for path in names:
            stream.write(path, arcname=str(path.relative_to(output_root)))
    if archive.stat().st_size > 450 * 1024 * 1024:
        raise RuntimeError("MV8 return archive exceeds the 450 MiB upload limit")
    return archive


def run_post(output_root: Path) -> dict[str, Any]:
    protocol = locked_protocol()
    output_root = Path(output_root)
    assembly = json.loads((output_root / "assembly_summary.json").read_text(encoding="utf-8"))
    generated: list[Path] = []
    if assembly["decision"] != "proceed_to_MV8_B1_kinetic_models":
        skipped = 0
        for architecture in ARCHITECTURES:
            for seed in TRAINING_SEEDS:
                path = _task_directory(output_root, architecture, seed) / "skipped.json"
                if path.is_file():
                    skipped += 1
        if skipped != len(ARCHITECTURES) * len(TRAINING_SEEDS):
            raise ValueError("MV8 audit-only path is missing guarded task records")
        summary = {
            "stage": STAGE,
            "status": "complete_MV8_information_audit_without_model_training",
            "assembly_decision": assembly["decision"],
            "skipped_model_tasks": skipped,
            "decision": "MV8_kinetic_moment_pilot_not_authorized_by_information_gate",
        }
        _atomic_json(output_root / "summary.json", summary)
        generated.append(output_root / "summary.json")
    else:
        with np.load(output_root / "dataset.npz", allow_pickle=False) as data:
            test_x = np.asarray(data["test_x"])
            test_y = np.asarray(data["test_y"])
            conditions = np.asarray(data["test_condition"])
            identities = np.asarray(data["test_identity"])
            scales = np.asarray(data["test_scale"])
            gaussian = np.asarray(data["test_gaussian"])
            tsvd_value = np.asarray(data["test_tsvd"])
            raw10 = np.asarray(data["test_raw10"])
            target10 = np.asarray(data["test_target10"])
            conditions10 = np.asarray(data["test_condition10"])
            identities10 = np.asarray(data["test_identity10"])
            scales10 = np.asarray(data["test_scale10"])
        raw = test_x[:, : len(OUTPUT_FIELDS)]
        predictions: dict[str, list[np.ndarray]] = {name: [] for name in ARCHITECTURES}
        task_summaries = []
        for architecture in ARCHITECTURES:
            for seed in TRAINING_SEEDS:
                summary, prediction = _verify_task(
                    _task_directory(output_root, architecture, seed), architecture, seed
                )
                if prediction.shape != raw.shape:
                    raise ValueError("MV8 task prediction shape differs from dataset")
                predictions[architecture].append(prediction)
                task_summaries.append(summary)
        ensembles = {
            architecture: np.mean(values, axis=0).astype(np.float32)
            for architecture, values in predictions.items()
        }
        methods_b1 = {
            "raw_b1": raw,
            "gaussian_b1": gaussian,
            "tsvd_b1": tsvd_value,
            **ensembles,
        }
        per_seed = {
            method: _per_seed_metrics(value, test_y, conditions, identities)
            for method, value in methods_b1.items()
        }
        per_seed["raw_b10"] = _per_seed_metrics(
            raw10, target10, conditions10, identities10
        )
        primary = str(protocol["execution_matrix"]["primary_condition"])
        primary_records: dict[str, Any] = {}
        for method, records in per_seed.items():
            values = list(records[primary].values())
            primary_records[method] = {
                "mean_composite_nrmse": float(np.mean([item["composite_nrmse"] for item in values])),
                "mean_per_field_nrmse": {
                    field: float(np.mean([item["per_field_nrmse"][field] for item in values]))
                    for field in OUTPUT_FIELDS
                },
            }
        raw10_primary = primary_records["raw_b10"]
        success = {}
        for architecture in ARCHITECTURES:
            current = primary_records[architecture]
            ratio = current["mean_composite_nrmse"] / max(
                raw10_primary["mean_composite_nrmse"], 1e-12
            )
            improved = sum(
                current["mean_per_field_nrmse"][field]
                <= raw10_primary["mean_per_field_nrmse"][field]
                for field in OUTPUT_FIELDS
            )
            success[architecture] = {
                "composite_ratio_to_raw_B10": ratio,
                "fields_no_worse_than_raw_B10": improved,
                "passes_pilot_rule": bool(ratio <= 1.10 and improved >= 3),
            }

        representative_seed = int(protocol["execution_matrix"]["representative_contour_seed"])
        representative_block = int(protocol["execution_matrix"]["representative_contour_block"])
        mask = (
            (conditions == primary)
            & (identities[:, 0] == representative_seed)
            & (identities[:, 1] == representative_block)
        )
        mask10 = (conditions10 == primary) & (identities10[:, 0] == representative_seed)
        if np.count_nonzero(mask) != 1 or np.count_nonzero(mask10) != 1:
            raise ValueError("locked MV8 representative contour identity is absent")
        index = int(np.flatnonzero(mask)[0])
        index10 = int(np.flatnonzero(mask10)[0])
        reference = test_y[index]
        if not np.array_equal(reference, target10[index10]):
            raise ValueError("MV8 representative B1/B10 cross-fit targets differ")
        figure_methods = {
            "raw_b1": raw[index],
            "gaussian_b1": gaussian[index],
            "tsvd_b1": tsvd_value[index],
            "nafnet_small": ensembles["nafnet_small"][index],
            "mambairv2_tiny_adapted": ensembles["mambairv2_tiny_adapted"][index],
            "raw_b10": raw10[index10],
        }
        if not np.allclose(scales[index], scales10[index10], rtol=1e-12, atol=0.0):
            raise ValueError("MV8 representative B1/B10 physical scales differ")
        figure_records = []
        for field_index in range(len(OUTPUT_FIELDS)):
            record = _physical_figure(
                output_root,
                field_index,
                {name: value[field_index] for name, value in figure_methods.items()},
                reference[field_index],
                float(scales[index, field_index]),
            )
            figure_records.append(record)
            generated.extend((output_root / record["png"], output_root / record["pdf"]))

        metrics_path = output_root / "mv8_primary_condition_metrics.csv"
        with metrics_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("method", "composite_nrmse", *OUTPUT_FIELDS))
            for method, record in primary_records.items():
                writer.writerow(
                    (
                        method,
                        record["mean_composite_nrmse"],
                        *(record["mean_per_field_nrmse"][field] for field in OUTPUT_FIELDS),
                    )
                )
        generated.append(metrics_path)
        summary = {
            "stage": STAGE,
            "status": "complete_MV8_kinetic_moment_feasibility_pilot",
            "protocol_sha256": _sha256(protocol_path()),
            "primary_condition": primary,
            "representative_contour": {
                "evaluation_seed": representative_seed,
                "block": representative_block,
            },
            "primary_condition_metrics": primary_records,
            "pilot_success_by_architecture": success,
            "per_evaluation_seed_metrics": per_seed,
            "figure_records": figure_records,
            "model_task_count": len(task_summaries),
            "decision": (
                "MV8_feasibility_pass_proceed_to_separately_locked_full_confirmatory_stage"
                if any(value["passes_pilot_rule"] for value in success.values())
                else "MV8_feasibility_does_not_support_one_block_kinetic_moment_claim"
            ),
        }
        _atomic_json(output_root / "summary.json", summary)
        generated.append(output_root / "summary.json")

    shutil.copy2(protocol_path(), output_root / PROTOCOL_FILE)
    generated.extend(
        (
            output_root / PROTOCOL_FILE,
            output_root / "assembly_summary.json",
            output_root / "source_moment_audit.csv",
        )
    )
    accounting = output_root / "slurm_accounting.psv"
    if accounting.is_file():
        generated.append(accounting)
    generated = sorted({path.resolve() for path in generated if path.is_file()})
    manifest = {
        "stage": STAGE,
        "status": "complete_MV8_return_artifact_manifest",
        "files": {
            str(path.relative_to(output_root.resolve())): {
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in generated
        },
    }
    _atomic_json(output_root / "artifact_manifest.json", manifest)
    generated.append((output_root / "artifact_manifest.json").resolve())
    for name, record in manifest["files"].items():
        path = output_root / name
        if _sha256(path) != record["sha256"] or path.stat().st_size != record["size_bytes"]:
            raise ValueError(f"MV8 recursive return verification failed: {path}")
    verification = {
        "stage": STAGE,
        "status": "complete_MV8_recursive_return_verification",
        "decision": "verified",
        "verified_file_count": len(manifest["files"]),
        "manifest_sha256": _sha256(output_root / "artifact_manifest.json"),
    }
    _atomic_json(output_root / "verification.json", verification)
    generated.append((output_root / "verification.json").resolve())
    archive = _create_archive(output_root.resolve(), generated)
    print(f"MV8_OUTPUT_ROOT={output_root}")
    print(f"ARCHIVE={archive}")
    print(f"ARCHIVE_SIZE_MIB={archive.stat().st_size / 1024**2:.2f}")
    print(f"ARCHIVE_SHA256={_sha256(archive)}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("verify-lock", "assemble", "model", "post"), required=True)
    parser.add_argument("--existing-m3-root", type=Path)
    parser.add_argument("--mv3-root", type=Path)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=6)
    args = parser.parse_args()
    if args.mode == "verify-lock":
        result = verify_lock()
    elif args.mode == "assemble":
        if None in (args.existing_m3_root, args.mv3_root, args.reference_root, args.output_root):
            parser.error("assemble requires all source roots and --output-root")
        result = run_assembly(
            args.existing_m3_root,
            args.mv3_root,
            args.reference_root,
            args.output_root,
        )
    elif args.mode == "model":
        if args.output_root is None or args.task_index is None:
            parser.error("model requires --output-root and --task-index")
        architecture, seed = task_from_index(args.task_index)
        result = run_model_task(
            args.output_root,
            architecture=architecture,
            training_seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    else:
        if args.output_root is None:
            parser.error("post requires --output-root")
        result = run_post(args.output_root)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
