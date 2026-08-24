"""Prospective JCP cavity shift experiment with a disjoint 260-block reference.

The module has three deliberately separated interfaces:

``run``
    Generates one preregistered DSMC trajectory and never evaluates a model.
``predict``
    Reads only the observation trajectories, applies the estimator frozen by
    JCP1, and writes a hashed prediction file.  No reference path is accepted.
``score``
    Verifies the prediction hash before opening the independently generated
    reference trajectories, then reports observed and reference-deconvolved
    errors for all eight moment fields.

This separation makes the prospective claim mechanically auditable.  The
fourth raw speed moment is recorded for direct temperature-noise diagnostics;
heat-flux noise remains block based because its particle-level variance needs
sixth-order raw moments.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np

from . import mohammadzadeh_spatial_refinement as engine
from .jcp_phase0 import (
    DEFAULT_BIN_WIDTH,
    EPS,
    FIELD_NAMES,
    KINETIC_OFFSET,
    OUTPUT_FIELDS,
    dct2,
    development_priors,
    eb_gain,
    fuse,
    geometric_mean,
    mode_bins,
    nrmse,
    out_of_support_statistic,
    pnet_cross_block_gain,
    pool_power,
)
from .mohammadzadeh_production import _atomic_write_json, _strict_json_ready
from .mohammadzadeh_validation import mohammadzadeh_config, reference_directory


STAGE = "JCP2_prospective_cavity_S2"
PROTOCOL_FILE = "jcp2_cavity_protocol_v1.json"
SEED_FILE = "jcp2_cavity_seed_bank_v1.json"
LOCK_FILE = "jcp2_cavity_lock_v1.json"
LOCK_STATUS = "locked_before_any_JCP2_trajectory"
OBSERVATION_BLOCKS = (0, 1, 2)
COMPARATOR_BLOCKS = tuple(range(3, 13))
BLOCK_COUNT = 13
LOW_ORDER = tuple(range(4))
HIGH_ORDER = tuple(range(4, 8))
SOURCE_FILES = (
    "vgdsmc/jcp_phase1_cavity.py",
    "vgdsmc/jcp_phase0.py",
    "vgdsmc/moment_sampling.py",
    "vgdsmc/mohammadzadeh_production.py",
    "vgdsmc/mohammadzadeh_spatial_refinement.py",
    "vgdsmc/ntc_checkpoint.py",
    "scripts/unity_jcp2_task.sbatch",
    "scripts/unity_jcp2_predict.sbatch",
    "scripts/unity_jcp2_score.sbatch",
    "scripts/run_jcp2_unity.sh",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def protocol_path(reference_dir: Path | None = None) -> Path:
    root = reference_directory() if reference_dir is None else Path(reference_dir)
    return root / PROTOCOL_FILE


def load_protocol(reference_dir: Path | None = None) -> dict[str, Any]:
    protocol = _json(protocol_path(reference_dir))
    if protocol.get("stage") != STAGE or protocol.get("status") != LOCK_STATUS:
        raise ValueError("JCP2 protocol is not locked")
    active = [item for item in protocol.get("conditions", []) if item.get("active_in_this_phase")]
    if len(active) != 1 or active[0].get("id") != "S2_kn0p085_u350":
        raise ValueError("JCP2 must have exactly one active S2 condition")
    contract = protocol.get("trajectory_contract", {})
    if (
        int(contract.get("nonoverlapping_sampling_blocks", -1)) != BLOCK_COUNT
        or tuple(contract.get("observation_blocks", ())) != OBSERVATION_BLOCKS
        or tuple(contract.get("raw_B10_comparator_blocks", ())) != COMPARATOR_BLOCKS
    ):
        raise ValueError("JCP2 block partition differs from the frozen contract")
    return protocol


def load_seed_bank(reference_dir: Path | None = None) -> dict[str, Any]:
    root = reference_directory() if reference_dir is None else Path(reference_dir)
    seeds = _json(root / SEED_FILE)
    if seeds.get("status") != LOCK_STATUS:
        raise ValueError("JCP2 seed bank is not locked")
    groups = {
        name: [int(value) for value in seeds[name]]
        for name in (
            "evaluation_primary",
            "evaluation_spares",
            "reference_primary",
            "reference_spares",
        )
    }
    flat = [seed for values in groups.values() for seed in values]
    if len(flat) != len(set(flat)):
        raise ValueError("JCP2 seed groups overlap")
    if tuple(map(len, groups.values())) != (8, 4, 20, 5):
        raise ValueError("JCP2 seed-group sizes differ from the protocol")
    return seeds


def group_seeds(group: str, reference_dir: Path | None = None) -> tuple[int, ...]:
    seeds = load_seed_bank(reference_dir)
    if group == "evaluation":
        names = ("evaluation_primary", "evaluation_spares")
    elif group == "reference":
        names = ("reference_primary", "reference_spares")
    else:
        raise ValueError("JCP2 group must be evaluation or reference")
    return tuple(int(seed) for name in names for seed in seeds[name])


def task_from_index(group: str, index: int) -> int:
    tasks = group_seeds(group)
    if not 0 <= int(index) < len(tasks):
        raise ValueError("JCP2 task index is outside the preregistered array")
    return tasks[int(index)]


def expected_lock_hashes(reference_dir: Path | None = None) -> dict[str, str]:
    ref = reference_directory() if reference_dir is None else Path(reference_dir)
    root = Path(__file__).resolve().parents[1]
    hashes = {
        "protocol_sha256": _sha256(ref / PROTOCOL_FILE),
        "seed_bank_sha256": _sha256(ref / SEED_FILE),
    }
    hashes.update({f"source::{name}": _sha256(root / name) for name in SOURCE_FILES})
    return hashes


def verify_lock(reference_dir: Path | None = None) -> dict[str, Any]:
    ref = reference_directory() if reference_dir is None else Path(reference_dir)
    protocol = load_protocol(ref)
    load_seed_bank(ref)
    lock = _json(ref / LOCK_FILE)
    expected = expected_lock_hashes(ref)
    if lock.get("status") != LOCK_STATUS or lock.get("hashes") != expected:
        raise ValueError("JCP2 lock does not match the protocol, seeds, and source bundle")
    contract = protocol["trajectory_contract"]
    return {
        "status": "JCP2_lock_verified_without_running_a_trajectory",
        "condition": "S2_kn0p085_u350",
        "evaluation_tasks": len(group_seeds("evaluation", ref)),
        "reference_tasks": len(group_seeds("reference", ref)),
        "primary_evaluation_units": protocol["qc_contract"]["evaluation_units_required"],
        "primary_reference_blocks": (
            protocol["qc_contract"]["reference_units_required"]
            * contract["nonoverlapping_sampling_blocks"]
        ),
        "lock_hashes": expected,
    }


def _active_condition(protocol: Mapping[str, Any]) -> dict[str, Any]:
    return next(dict(item) for item in protocol["conditions"] if item["active_in_this_phase"])


def stage_configuration(
    stage: str,
    seed: int,
    *,
    reference_dir: Path | None = None,
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    prefix = f"{STAGE}::"
    if not stage.startswith(prefix):
        raise ValueError(f"JCP2 accepts only {prefix}<group>")
    group = stage[len(prefix) :]
    if int(seed) not in group_seeds(group, reference_dir):
        raise ValueError("seed is not preregistered for this JCP2 group")
    ref = reference_directory() if reference_dir is None else Path(reference_dir)
    protocol = load_protocol(ref)
    lock = _json(ref / LOCK_FILE)
    expected = expected_lock_hashes(ref)
    if lock.get("status") != LOCK_STATUS or lock.get("hashes") != expected:
        raise ValueError("JCP2 source lock is absent or stale")
    condition = _active_condition(protocol)
    contract = protocol["trajectory_contract"]
    base = mohammadzadeh_config(
        grid=int(contract["grid"]),
        particles_per_cell=int(contract["particles_per_cell"]),
        steps=int(contract["steps"]),
        sample_start=int(contract["sample_start"]),
        seed=int(seed),
        dt_safety=float(contract["dt_safety"]),
    )
    cfg = replace(
        base,
        knudsen=float(condition["knudsen"]),
        lid_velocity_x=float(condition["lid_speed_m_per_s"]),
    )
    engine_protocol = dict(protocol)
    engine_protocol["runtime_contract"] = {
        "nonoverlapping_sampling_blocks": int(contract["nonoverlapping_sampling_blocks"]),
        "sample_count": int(contract["sample_count"]),
        "maximum_events_per_particle_per_step": int(
            contract["maximum_events_per_particle_per_step"]
        ),
    }
    specification = {
        "name": stage,
        "condition_id": condition["id"],
        "group": group,
        "grid": cfg.nx,
        "steps": cfg.steps,
        "sample_start": cfg.sample_start,
        "checkpoint_interval_steps": int(contract["checkpoint_interval_steps"]),
    }
    m1r = _json(ref / "m1r_repair_protocol.json")
    hashes = {**expected, "jcp2_lock_sha256": _sha256(ref / LOCK_FILE)}
    return cfg, engine_protocol, specification, m1r, hashes


def _rewrite_run_summary(output_dir: Path, summary: dict[str, Any], group: str) -> dict[str, Any]:
    summary = dict(summary)
    if summary.get("status") != "complete_M2_spatial_development_seed":
        return summary
    checks = summary.get("mechanical_checks", {})
    summary.update(
        {
            "stage": STAGE,
            "condition_id": "S2_kn0p085_u350",
            "group": group,
            "status": "complete_JCP2_cavity_seed",
            "scientific_scope": "prospective_full_hierarchy_rho_u_v_T_Pxy_Pxx_minus_Pyy_qx_qy",
            "fourth_raw_speed_moment_recorded": True,
            "heat_flux_particle_variance_claim": False,
            "decision": (
                "accept_JCP2_seed_for_preregistered_QC_selection"
                if checks and all(bool(value) for value in checks.values())
                else "hold_JCP2_seed_for_preregistered_spare_replacement"
            ),
        }
    )
    _atomic_write_json(output_dir / "summary.json", summary)
    manifest_path = output_dir / "artifact_manifest.json"
    manifest = _json(manifest_path)
    manifest.update({"stage": STAGE, "group": group, "status": summary["status"]})
    manifest["files"]["summary.json"] = {
        "sha256": _sha256(output_dir / "summary.json"),
        "size_bytes": (output_dir / "summary.json").stat().st_size,
    }
    _atomic_write_json(manifest_path, manifest)
    return summary


def run_seed(
    *,
    group: str,
    seed: int,
    output_root: Path,
    resume: bool = True,
    stop_after_step: int | None = None,
) -> dict[str, Any]:
    original = engine.stage_configuration
    engine.stage_configuration = stage_configuration
    directory = Path(output_root) / group / f"seed_{int(seed)}"
    try:
        result = engine.run_refinement_seed(
            stage=f"{STAGE}::{group}",
            seed=int(seed),
            output_dir=directory,
            resume=resume,
            stop_after_step=stop_after_step,
            progress=lambda step, total: print(
                json.dumps({"step": step, "total": total}), flush=True
            ),
        )
    finally:
        engine.stage_configuration = original
    return _strict_json_ready(_rewrite_run_summary(directory, result, group))


def _verify_artifacts(directory: Path) -> dict[str, Any]:
    directory = Path(directory)
    manifest = _json(directory / "artifact_manifest.json")
    for name, record in manifest.get("files", {}).items():
        path = directory / name
        if (
            not path.is_file()
            or path.stat().st_size != int(record["size_bytes"])
            or _sha256(path) != record["sha256"]
        ):
            raise ValueError(f"JCP2 artifact verification failed: {path}")
    summary = _json(directory / "summary.json")
    if summary.get("status") != "complete_JCP2_cavity_seed":
        raise ValueError(f"JCP2 run has an invalid status: {directory}")
    return summary


def _qc_selected(output_root: Path, group: str, required: int) -> list[Path]:
    accepted: list[Path] = []
    for seed in group_seeds(group):
        directory = Path(output_root) / group / f"seed_{seed}"
        summary = _verify_artifacts(directory)
        checks = summary.get("mechanical_checks", {})
        if checks and all(bool(value) for value in checks.values()):
            accepted.append(directory)
        if len(accepted) == int(required):
            return accepted
    raise ValueError(
        f"JCP2 has only {len(accepted)} passing {group} seeds; {required} are required"
    )


def _project_modules() -> tuple[Any, Any, Any, Any]:
    from . import mohammadzadeh_mv9_heat_flux as mv9
    from . import mohammadzadeh_mv14_kinetic_conservation_cavity as mv14
    from .ntc_checkpoint import load_ntc_checkpoint
    from .vhs_model import KB

    return mv9, mv14, load_ntc_checkpoint, KB


MOMENT_KEYS = (
    "simulated_count",
    "m0",
    "m1",
    "m2",
    "energy",
    "energy_velocity",
    "speed4",
)


def _block_payloads(directory: Path) -> tuple[Any, list[dict[str, Any]]]:
    mv9, _, load_checkpoint, _ = _project_modules()
    summary = _verify_artifacts(directory)
    cfg = mv9._config_from_summary(summary)
    checkpoint = load_checkpoint(Path(directory) / "checkpoint.npz", cfg)
    root = checkpoint.block_accumulators
    mapping = root.get("block_moments") if isinstance(root, Mapping) else None
    if not isinstance(mapping, Mapping) or len(mapping) != BLOCK_COUNT:
        raise ValueError(f"JCP2 expected {BLOCK_COUNT} additive blocks: {directory}")
    payloads = [dict(mapping[key]) for key in sorted(mapping)]
    for payload in payloads:
        if "speed4" not in payload:
            raise ValueError(f"JCP2 fourth raw speed moment is absent: {directory}")
    return cfg, payloads


def merge_payloads(payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not payloads:
        raise ValueError("cannot merge an empty additive-moment collection")
    result: dict[str, Any] = {"samples": sum(int(item["samples"]) for item in payloads)}
    for key in MOMENT_KEYS:
        arrays = [np.asarray(item[key], dtype=np.float64) for item in payloads]
        if any(value.shape != arrays[0].shape for value in arrays[1:]):
            raise ValueError(f"additive moment {key} has inconsistent shapes")
        result[key] = np.sum(arrays, axis=0, dtype=np.float64)
    return result


def _fields(payload: Mapping[str, Any], cfg: Any) -> np.ndarray:
    mv9, _, _, kb = _project_modules()
    outputs, auxiliary, _ = mv9.moment_fields(payload, cfg, kb, output_dtype=None)
    return np.concatenate((auxiliary, outputs), axis=0)


def _single_unit_noise(blocks: np.ndarray, *, budget: int, width: int) -> np.ndarray:
    blocks = np.asarray(blocks, dtype=np.float64)
    if blocks.ndim != 3 or len(blocks) < 2:
        raise ValueError("single-unit noise needs (block,ny,nx)")
    coefficient_variance = np.var(dct2(blocks), axis=0, ddof=1) / float(budget)
    return pool_power(coefficient_variance, width)


def _temperature_delta_variance(payload: Mapping[str, Any], cfg: Any, kb: float) -> np.ndarray:
    """Equal-simulated-particle delta variance using the recorded fourth moment."""

    m0 = np.asarray(payload["m0"], dtype=np.float64)
    m1 = np.asarray(payload["m1"], dtype=np.float64)
    m2 = np.asarray(payload["m2"], dtype=np.float64)
    energy = np.asarray(payload["energy"], dtype=np.float64)
    energy_velocity = np.asarray(payload["energy_velocity"], dtype=np.float64)
    speed4 = np.asarray(payload["speed4"], dtype=np.float64)
    count = np.maximum(np.asarray(payload["simulated_count"], dtype=np.float64), 1.0)
    velocity = m1 / m0[:, None]
    second = m2 / m0[:, None, None]
    e_speed2 = energy / m0
    e_speed2_velocity = energy_velocity / m0[:, None]
    e_speed4 = speed4 / m0
    a = 2.0 * velocity
    c = 2.0 * np.sum(velocity**2, axis=1) - e_speed2
    influence2 = (
        e_speed4
        + np.einsum("ni,nij,nj->n", a, second, a)
        + c**2
        - 2.0 * np.sum(a * e_speed2_velocity, axis=1)
        + 2.0 * c * e_speed2
        - 2.0 * c * np.sum(a * velocity, axis=1)
    )
    influence2 = np.maximum(influence2, 0.0)
    factor = float(cfg.vhs.mass) / (3.0 * float(kb) * float(cfg.t0))
    return (factor**2 * influence2 / count).reshape((cfg.ny, cfg.nx))


def _development_prior_inputs(mv9_root: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with np.load(Path(mv9_root) / "dataset.npz", allow_pickle=False) as data:
        train_x = np.asarray(data["train_x"], dtype=np.float64)
        train_y = np.asarray(data["train_y"], dtype=np.float64)
        train_condition = np.asarray(data["train_condition"]).astype(str)
    samples = np.concatenate((train_x[:, KINETIC_OFFSET : 2 * KINETIC_OFFSET], train_y), axis=1)
    fields, features, names = [], [], []
    for condition in sorted(np.unique(train_condition)):
        mask = train_condition == condition
        names.append(str(condition))
        fields.append(np.mean(samples[mask], axis=0))
        features.append(
            (
                float(np.mean(train_x[mask, -2, 0, 0])),
                float(np.mean(train_x[mask, -1, 0, 0])),
            )
        )
    return np.asarray(fields), np.asarray(features), names


def predict(
    *,
    run_root: Path,
    mv9_root: Path,
    mv15c_root: Path,
    output_dir: Path,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Create and hash predictions without accepting or opening a reference path."""

    output = Path(output_dir)
    lock_path = output / "prediction_lock.json"
    prediction_path = output / "predictions.npz"
    if lock_path.is_file() and prediction_path.is_file():
        lock = _json(lock_path)
        if _sha256(prediction_path) != lock.get("prediction_sha256"):
            raise ValueError("existing JCP2 prediction lock is stale")
        return lock
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite partial predictions: {output}")
    output.mkdir(parents=True, exist_ok=True)
    selected = _qc_selected(run_root, "evaluation", 8)
    configs, payload_sets, block_fields = [], [], []
    for directory in selected:
        cfg, payloads = _block_payloads(directory)
        configs.append(cfg)
        payload_sets.append(payloads)
        block_fields.append(np.asarray([_fields(payload, cfg) for payload in payloads]))
    blocks = np.asarray(block_fields, dtype=np.float64)
    if len({(cfg.nx, cfg.ny, cfg.knudsen, cfg.lid_velocity_x) for cfg in configs}) != 1:
        raise ValueError("JCP2 evaluation configurations are inconsistent")
    cfg = configs[0]
    raw_b3 = np.asarray(
        [_fields(merge_payloads([payloads[i] for i in OBSERVATION_BLOCKS]), cfg) for payloads in payload_sets]
    )
    raw_b10 = np.asarray(
        [_fields(merge_payloads([payloads[i] for i in COMPARATOR_BLOCKS]), cfg) for payloads in payload_sets]
    )
    raw_b2 = np.asarray(
        [
            [
                _fields(
                    merge_payloads(
                        [payloads[i] for i in OBSERVATION_BLOCKS if i != held]
                    ),
                    cfg,
                )
                for held in OBSERVATION_BLOCKS
            ]
            for payloads in payload_sets
        ]
    )

    condition = _active_condition(load_protocol())
    development_fields, development_features, development_names = _development_prior_inputs(mv9_root)
    target_features = np.tile(
        np.asarray([[math.log10(float(condition["knudsen"])), float(condition["lid_speed_m_per_s"]) / 100.0]]),
        (len(selected), 1),
    )
    pnn, pnns, prior_audit = development_priors(
        development_fields, development_features, target_features
    )

    mv9, mv14, _, kb = _project_modules()
    b3_images = np.asarray(
        [
            mv9._conditioned_image(
                value[list(HIGH_ORDER)], value[list(LOW_ORDER)], condition
            )
            for value in raw_b3
        ],
        dtype=np.float32,
    )
    b2_images = np.asarray(
        [
            mv9._conditioned_image(
                raw_b2[unit, held, list(HIGH_ORDER)],
                raw_b2[unit, held, list(LOW_ORDER)],
                condition,
            )
            for unit in range(len(selected))
            for held in range(len(OBSERVATION_BLOCKS))
        ],
        dtype=np.float32,
    )
    pnet_b3 = np.asarray(
        mv14._predict_mamba_validation(Path(mv9_root), b3_images, batch_size=int(batch_size)),
        dtype=np.float64,
    )
    pnet_b2 = np.asarray(
        mv14._predict_mamba_validation(Path(mv9_root), b2_images, batch_size=int(batch_size)),
        dtype=np.float64,
    ).reshape(len(selected), len(OBSERVATION_BLOCKS), len(HIGH_ORDER), cfg.ny, cfg.nx)
    with np.load(Path(mv15c_root) / "locked_fresh_predictions.npz", allow_pickle=False) as data:
        frozen_weight = np.asarray(data["frozen_weight_map"], dtype=np.float64)

    shape = raw_b3.shape
    method_names = (
        "raw_b3",
        "raw_b10",
        "p0_eb",
        "pnn_prior_only",
        "pnns_prior_only",
        "pnn_eb",
        "pnns_eb",
        "pnet_alone",
        "pnet_cross_block",
        "pnet_frozen_gain",
        "promoted_full_hierarchy",
    )
    methods = {name: np.full(shape, np.nan, dtype=np.float64) for name in method_names}
    methods["raw_b3"][:] = raw_b3
    methods["raw_b10"][:] = raw_b10
    methods["pnn_prior_only"][:] = pnn
    methods["pnns_prior_only"][:] = pnns
    noise = np.empty(shape, dtype=np.float64)
    gain = np.full(shape, np.nan, dtype=np.float64)
    detector = np.empty(shape[:2], dtype=np.float64)
    temperature_delta = np.empty((len(selected), cfg.ny, cfg.nx), dtype=np.float64)
    for unit, payloads in enumerate(payload_sets):
        observation_payload = merge_payloads([payloads[i] for i in OBSERVATION_BLOCKS])
        temperature_delta[unit] = _temperature_delta_variance(observation_payload, cfg, kb)
        for field in range(len(FIELD_NAMES)):
            noise[unit, field] = _single_unit_noise(
                blocks[unit, OBSERVATION_BLOCKS, field],
                budget=len(OBSERVATION_BLOCKS),
                width=DEFAULT_BIN_WIDTH,
            )
            p0_gain = eb_gain(raw_b3[unit, field], np.zeros_like(raw_b3[unit, field]), noise[unit, field])
            methods["p0_eb"][unit, field] = fuse(
                raw_b3[unit, field], np.zeros_like(raw_b3[unit, field]), p0_gain
            )
            for prior_name, prior in (("pnn", pnn), ("pnns", pnns)):
                current_gain = eb_gain(raw_b3[unit, field], prior[unit, field], noise[unit, field])
                methods[f"{prior_name}_eb"][unit, field] = fuse(
                    raw_b3[unit, field], prior[unit, field], current_gain
                )
                if field in LOW_ORDER and prior_name == "pnns":
                    gain[unit, field] = current_gain
                    methods["promoted_full_hierarchy"][unit, field] = methods["pnns_eb"][unit, field]
            detector[unit, field] = out_of_support_statistic(
                raw_b3[unit, field],
                pnns[unit, field] if field in LOW_ORDER else pnet_b3[unit, field - KINETIC_OFFSET],
                noise[unit, field],
            )
        for output_index, field in enumerate(HIGH_ORDER):
            methods["pnet_alone"][unit, field] = pnet_b3[unit, output_index]
            cross_gain = pnet_cross_block_gain(
                blocks[unit, OBSERVATION_BLOCKS, field],
                pnet_b2[unit, :, output_index],
            )
            gain[unit, field] = cross_gain
            methods["pnet_cross_block"][unit, field] = fuse(
                raw_b3[unit, field], pnet_b3[unit, output_index], cross_gain
            )
            methods["pnet_frozen_gain"][unit, field] = fuse(
                raw_b3[unit, field], pnet_b3[unit, output_index], frozen_weight
            )
            methods["promoted_full_hierarchy"][unit, field] = methods["pnet_cross_block"][unit, field]

    seeds = np.asarray([int(_json(path / "summary.json")["seed"]) for path in selected], dtype=np.int64)
    _atomic_npz(
        prediction_path,
        seeds=seeds,
        field_names=np.asarray(FIELD_NAMES),
        condition=np.asarray([condition["id"]]),
        noise_power=noise.astype(np.float32),
        gain=gain.astype(np.float32),
        detector=detector,
        temperature_delta_variance=temperature_delta,
        **{f"method_{name}": value.astype(np.float32) for name, value in methods.items()},
    )
    lock = {
        "stage": STAGE,
        "status": "predictions_locked_before_reference_interface",
        "classification": "fully_prospective",
        "condition": condition["id"],
        "selected_evaluation_seeds": seeds.tolist(),
        "selection_used_only_preregistered_mechanical_and_stationarity_QC": True,
        "reference_path_accepted_by_prediction_interface": False,
        "reference_opened": False,
        "field_names": list(FIELD_NAMES),
        "promoted_rule": load_protocol()["phase0_basis"]["selection_rule"],
        "development_condition_names": development_names,
        "development_condition_features": development_features.tolist(),
        "target_features": target_features[0].tolist(),
        "prior_audit": prior_audit,
        "prediction_sha256": _sha256(prediction_path),
        "prediction_size_bytes": prediction_path.stat().st_size,
        "protocol_sha256": _sha256(protocol_path()),
        "source_lock_sha256": _sha256(reference_directory() / LOCK_FILE),
    }
    _atomic_write_json(lock_path, lock)
    return lock


def _effective_blocks(reference_blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Field-wise effective block counts from within-seed spatial correlations."""

    values = np.asarray(reference_blocks, dtype=np.float64)
    if values.ndim != 5 or values.shape[1] != BLOCK_COUNT:
        raise ValueError("reference blocks need (seed,13,field,ny,nx)")
    centered = values - np.mean(values, axis=(0, 1), keepdims=True)
    fields = values.shape[2]
    rho = np.zeros((fields, BLOCK_COUNT), dtype=np.float64)
    rho[:, 0] = 1.0
    denominator = np.sum(centered**2, axis=(0, 1, 3, 4))
    for lag in range(1, BLOCK_COUNT):
        numerator = np.sum(
            centered[:, :-lag] * centered[:, lag:], axis=(0, 1, 3, 4)
        )
        normalization = (BLOCK_COUNT - lag) / BLOCK_COUNT
        rho[:, lag] = numerator / np.maximum(denominator * normalization, EPS)
    total = values.shape[0] * values.shape[1]
    effective = np.empty(fields, dtype=np.float64)
    for field in range(fields):
        positive = []
        for lag in range(1, BLOCK_COUNT):
            if rho[field, lag] <= 0.0:
                break
            positive.append((1.0 - lag / BLOCK_COUNT) * rho[field, lag])
        tau = max(1.0 + 2.0 * sum(positive), 1.0)
        effective[field] = float(np.clip(total / tau, values.shape[0], total))
    return effective, rho


def _sign_pvalue(wins: int, units: int) -> float:
    return float(
        sum(math.comb(units, index) for index in range(int(wins), int(units) + 1))
        / (2.0**units)
    )


def _bootstrap_geometric_ci(values: Sequence[float], *, seed: int = 20260817) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(10000, len(array)))
    samples = np.exp(np.mean(np.log(array[indices]), axis=1))
    return tuple(float(value) for value in np.quantile(samples, (0.025, 0.975)))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score(
    *,
    run_root: Path,
    prediction_dir: Path,
    output_dir: Path,
    archive: Path,
) -> dict[str, Any]:
    prediction_dir = Path(prediction_dir)
    prediction_path = prediction_dir / "predictions.npz"
    prediction_lock = _json(prediction_dir / "prediction_lock.json")
    if prediction_lock.get("status") != "predictions_locked_before_reference_interface":
        raise ValueError("JCP2 prediction lock has an invalid status")
    if _sha256(prediction_path) != prediction_lock.get("prediction_sha256"):
        raise ValueError("JCP2 predictions changed after locking")
    references = _qc_selected(run_root, "reference", 20)
    reference_payloads: list[dict[str, Any]] = []
    reference_fields: list[np.ndarray] = []
    configurations = []
    for directory in references:
        cfg, payloads = _block_payloads(directory)
        configurations.append(cfg)
        reference_payloads.extend(payloads)
        reference_fields.append(np.asarray([_fields(payload, cfg) for payload in payloads]))
    cfg = configurations[0]
    if len(reference_payloads) != 260:
        raise ValueError("JCP2 reference does not contain exactly 260 blocks")
    reference = _fields(merge_payloads(reference_payloads), cfg)
    reference_blocks = np.asarray(reference_fields, dtype=np.float64)
    effective, autocorrelation = _effective_blocks(reference_blocks)
    flat_reference_blocks = reference_blocks.reshape(-1, *reference_blocks.shape[2:])
    block_variance = np.var(flat_reference_blocks, axis=0, ddof=1)
    reference_mean_variance = block_variance / effective[:, None, None]

    with np.load(prediction_path, allow_pickle=False) as data:
        seeds = np.asarray(data["seeds"], dtype=np.int64)
        if seeds.tolist() != prediction_lock["selected_evaluation_seeds"]:
            raise ValueError("prediction seed list differs from its lock")
        methods = {
            name[len("method_") :]: np.asarray(data[name], dtype=np.float64)
            for name in data.files
            if name.startswith("method_")
        }
        noise_power = np.asarray(data["noise_power"], dtype=np.float64)
        gain = np.asarray(data["gain"], dtype=np.float64)
        detector = np.asarray(data["detector"], dtype=np.float64)
        temperature_delta = np.asarray(data["temperature_delta_variance"], dtype=np.float64)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite JCP2 score output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    reference_noise_nrmse2 = np.sum(reference_mean_variance, axis=(1, 2)) / np.maximum(
        np.sum(reference**2, axis=(1, 2)), EPS
    )
    errors: dict[str, np.ndarray] = {
        name: np.full((len(seeds), len(FIELD_NAMES)), np.nan, dtype=np.float64)
        for name in methods
    }
    observed_errors = {name: value.copy() for name, value in errors.items()}
    rows = []
    for unit, seed in enumerate(seeds):
        for field, field_name in enumerate(FIELD_NAMES):
            baseline_observed = nrmse(methods["raw_b10"][unit, field], reference[field])
            baseline_true = math.sqrt(max(baseline_observed**2 - reference_noise_nrmse2[field], 0.0))
            for name, values in methods.items():
                candidate = values[unit, field]
                if not np.all(np.isfinite(candidate)):
                    continue
                observed = nrmse(candidate, reference[field])
                deconvolved = math.sqrt(max(observed**2 - reference_noise_nrmse2[field], 0.0))
                observed_errors[name][unit, field] = observed
                errors[name][unit, field] = deconvolved
                rows.append(
                    {
                        "condition": "S2_kn0p085_u350",
                        "seed": int(seed),
                        "field": field_name,
                        "method": name,
                        "nrmse_observed": observed,
                        "nrmse_reference_deconvolved": deconvolved,
                        "ratio_to_raw_b10_deconvolved": deconvolved / max(baseline_true, EPS),
                    }
                )

    qy = FIELD_NAMES.index("qy")
    comparators = ("pnet_alone", "p0_eb", "pnet_frozen_gain")
    qy_tests = {}
    for comparator in comparators:
        wins = int(np.sum(errors["pnet_cross_block"][:, qy] < errors[comparator][:, qy]))
        ratios = errors["pnet_cross_block"][:, qy] / np.maximum(errors[comparator][:, qy], EPS)
        ci = _bootstrap_geometric_ci(ratios)
        qy_tests[comparator] = {
            "wins": wins,
            "units": len(seeds),
            "one_sided_exact_sign_p": _sign_pvalue(wins, len(seeds)),
            "geometric_error_ratio": geometric_mean(ratios),
            "bootstrap_95_percent_CI": list(ci),
        }
    envelope = np.minimum.reduce([errors[name][:, qy] for name in comparators])
    envelope_margin = errors["pnet_cross_block"][:, qy] < 0.95 * envelope
    promoted_ratios = errors["promoted_full_hierarchy"] / np.maximum(errors["raw_b10"], EPS)
    finite_promoted = np.isfinite(promoted_ratios)

    observed_temperature_noise = np.mean(
        np.var(
            np.asarray(
                [
                    [_fields(payload, cfg)[FIELD_NAMES.index("T")] for payload in _block_payloads(path)[1]][:3]
                    for path in _qc_selected(run_root, "evaluation", 8)
                ]
            ),
            axis=1,
            ddof=1,
        )
        / 3.0,
        axis=0,
    )
    predicted_temperature_noise = np.mean(temperature_delta, axis=0)
    positive = (observed_temperature_noise > EPS) & (predicted_temperature_noise > EPS)
    temperature_ratio = predicted_temperature_noise[positive] / observed_temperature_noise[positive]

    gates = {
        "prediction_hash_verified_before_reference_open": True,
        "eight_evaluation_units_after_preregistered_QC": len(seeds) == 8,
        "reference_has_260_disjoint_blocks": len(reference_payloads) == 260,
        "reference_minimum_effective_blocks_at_least_150": bool(np.min(effective) >= 150.0),
        "raw_B3_and_raw_B10_are_disjoint": set(OBSERVATION_BLOCKS).isdisjoint(COMPARATOR_BLOCKS),
        "all_evaluation_and_reference_seeds_disjoint": not bool(
            set(seeds.tolist())
            & {int(_json(path / "summary.json")["seed"]) for path in references}
        ),
        "all_eight_fields_reported_for_promoted_method": bool(
            np.all(np.isfinite(errors["promoted_full_hierarchy"]))
        ),
        "promoted_below_raw_B10_in_all_field_seed_units": bool(
            np.all(promoted_ratios[finite_promoted] < 1.0)
        ),
        "G1_qy_below_0p95_of_pnet_alone_p0_and_frozen_envelope_in_all_units": bool(
            np.all(envelope_margin)
        ),
        "G1_qy_exact_sign_p_at_most_0p05_for_each_comparator": bool(
            all(value["one_sided_exact_sign_p"] <= 0.05 for value in qy_tests.values())
        ),
    }
    summary = {
        "stage": STAGE,
        "classification": "fully_prospective_blind_interface",
        "condition": "S2_kn0p085_u350",
        "fields": list(FIELD_NAMES),
        "selected_evaluation_seeds": seeds.tolist(),
        "selected_reference_seeds": [int(_json(path / "summary.json")["seed"]) for path in references],
        "reference_blocks": len(reference_payloads),
        "reference_effective_blocks_by_field": {
            field: float(value) for field, value in zip(FIELD_NAMES, effective, strict=True)
        },
        "reference_noise_nrmse_squared_by_field": {
            field: float(value)
            for field, value in zip(FIELD_NAMES, reference_noise_nrmse2, strict=True)
        },
        "qy_primary_tests": qy_tests,
        "qy_envelope_margin_wins": int(np.sum(envelope_margin)),
        "full_hierarchy_promoted_units_below_raw_B10": int(
            np.sum(promoted_ratios[finite_promoted] < 1.0)
        ),
        "full_hierarchy_promoted_units_total": int(np.sum(finite_promoted)),
        "temperature_particle_delta_noise_diagnostic": {
            "uses_recorded_fourth_raw_speed_moment": True,
            "persistence_calibration_fitted_on_prospective_reference": False,
            "median_predicted_to_three_block_scatter_ratio": float(np.median(temperature_ratio)),
            "cell_fraction_within_15_percent_before_persistence_calibration": float(
                np.mean((temperature_ratio >= 0.85) & (temperature_ratio <= 1.15))
            ),
        },
        "heat_flux_noise_policy": "within-unit between-block scatter; sixth-order raw moments were not stored, so no direct particle-level q variance is claimed",
        "gates": gates,
        "all_G1_gates_pass": bool(all(gates.values())),
    }
    _write_csv(output / "metrics.csv", rows)
    _atomic_write_json(output / "summary.json", summary)
    _atomic_npz(
        output / "reference_stats.npz",
        field_names=np.asarray(FIELD_NAMES),
        reference=reference.astype(np.float32),
        block_variance=block_variance.astype(np.float32),
        reference_mean_variance=reference_mean_variance.astype(np.float32),
        effective_blocks=effective,
        autocorrelation=autocorrelation,
        noise_power=noise_power.astype(np.float32),
        gain=gain.astype(np.float32),
        detector=detector,
        temperature_delta_variance=temperature_delta,
        **{f"deconvolved_error_{name}": value for name, value in errors.items()},
    )
    shutil.copy2(prediction_dir / "prediction_lock.json", output / "prediction_lock.json")
    manifest = {
        "stage": STAGE,
        "files": {
            path.name: {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
            for path in sorted(output.iterdir())
            if path.is_file()
        },
    }
    _atomic_write_json(output / "manifest.json", manifest)
    archive = Path(archive)
    if archive.exists():
        raise FileExistsError(f"refusing to overwrite {archive}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as stream:
        for name in ("summary.json", "metrics.csv", "reference_stats.npz", "prediction_lock.json", "manifest.json"):
            stream.write(output / name, arcname=name)
        stream.write(prediction_path, arcname="predictions.npz")
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    checksum.write_text(f"{_sha256(archive)}  {archive.name}\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify_parser = subparsers.add_parser("verify-lock")
    verify_parser.add_argument("--reference-dir", type=Path)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--group", choices=("evaluation", "reference"), required=True)
    run_parser.add_argument("--task-index", type=int)
    run_parser.add_argument("--seed", type=int)
    run_parser.add_argument("--output-root", type=Path, required=True)
    run_parser.add_argument("--no-resume", action="store_true")
    run_parser.add_argument("--stop-after-step", type=int)
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--run-root", type=Path, required=True)
    predict_parser.add_argument("--mv9-root", type=Path, required=True)
    predict_parser.add_argument("--mv15c-root", type=Path, required=True)
    predict_parser.add_argument("--output-dir", type=Path, required=True)
    predict_parser.add_argument("--batch-size", type=int, default=8)
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--run-root", type=Path, required=True)
    score_parser.add_argument("--prediction-dir", type=Path, required=True)
    score_parser.add_argument("--output-dir", type=Path, required=True)
    score_parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "verify-lock":
        result = verify_lock(args.reference_dir)
    elif args.command == "run":
        seed = args.seed if args.seed is not None else task_from_index(args.group, args.task_index)
        result = run_seed(
            group=args.group,
            seed=seed,
            output_root=args.output_root,
            resume=not args.no_resume,
            stop_after_step=args.stop_after_step,
        )
    elif args.command == "predict":
        result = predict(
            run_root=args.run_root,
            mv9_root=args.mv9_root,
            mv15c_root=args.mv15c_root,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )
    else:
        result = score(
            run_root=args.run_root,
            prediction_dir=args.prediction_dir,
            output_dir=args.output_dir,
            archive=args.archive,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
