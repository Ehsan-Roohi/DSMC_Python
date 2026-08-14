"""Prospectively locked fresh-DSMC confirmation of MV15B DCIR-QY at B3.

MV15C runs eight new trajectories: four new seeds at the failed
``kn0p1_u400`` corner and four seeds at the entirely new ``kn0p08_u350``
condition.  Only additive blocks 0, 1, and 2 from a seed enter its prediction.
The reference for that seed is constructed after prediction locking from the
exact B10 fields of the other three seeds at the same condition.

The MV15B B3 weight map, Mamba ensemble, TSVD rank, metrics, and acceptance
gates are immutable.  No fresh field may select or tune any parameter.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np


STAGE = "MV15C_Mohammadzadeh_fresh_B3_confirmation"
STATUS = "locked_before_any_MV15C_fresh_trajectory_or_outcome"
COMPLETE_REFERENCE_STATUS = "complete_MV15C_fresh_reference_seed"
PROTOCOL_FILE = "mv15c_fresh_b3_confirmation_protocol.json"
PRIMARY_CONDITION = "kn0p1_u400"
NEW_CONDITION = "kn0p08_u350"
FRESH_SEEDS = {
    PRIMARY_CONDITION: (151501, 151502, 151503, 151504),
    NEW_CONDITION: (151511, 151512, 151513, 151514),
}
BUDGET = 3
INPUT_BLOCKS = (0, 1, 2)
QY_INDEX = 3
EPS = 1.0e-12


def _mv9_module():
    from . import mohammadzadeh_mv9_heat_flux as mv9

    return mv9


def _mv14_module():
    from . import mohammadzadeh_mv14_kinetic_conservation_cavity as mv14

    return mv14


def _mv15a_module():
    from . import mohammadzadeh_mv15a_spectral_information_audit as mv15a

    return mv15a


def _mv15b_module():
    from . import mohammadzadeh_mv15b_data_consistent_budget as mv15b

    return mv15b


def _mv3ref_module():
    from . import mohammadzadeh_mv3_reference as mv3ref

    return mv3ref


def _mv3_module():
    from . import mohammadzadeh_vision_mv3 as mv3

    return mv3


def _engine_module():
    from . import mohammadzadeh_spatial_refinement as engine

    return engine


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(
        f"Object of type {value.__class__.__name__} is not JSON serializable"
    )


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
            raise ValueError(f"MV15C recursive verification failed: {path}")
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
    protocol = json.loads(protocol_path().read_text(encoding="utf-8"))
    if protocol.get("stage") != STAGE or protocol.get("status") != STATUS:
        raise ValueError("MV15C protocol is absent or unlocked")
    conditions = condition_map(protocol)
    observed = {
        key: tuple(int(seed) for seed in value["seeds"])
        for key, value in conditions.items()
    }
    frozen = protocol["frozen_B3_contract"]
    if (
        observed != FRESH_SEEDS
        or int(frozen["input_budget"]) != BUDGET
        or tuple(int(value) for value in frozen["input_block_indices"])
        != INPUT_BLOCKS
        or float(frozen["mode_reliability_threshold"]) != 0.97
        or float(frozen["trusted_mode_strength"]) != 0.25
    ):
        raise ValueError("MV15C implementation differs from its locked matrix")
    source = protocol["source_contract"]
    mv9, mv14, mv15a, mv15b = (
        _mv9_module(),
        _mv14_module(),
        _mv15a_module(),
        _mv15b_module(),
    )
    checks = {
        "mv9_module_sha256": Path(mv9.__file__),
        "mv14_module_sha256": Path(mv14.__file__),
        "mv15a_module_sha256": Path(mv15a.__file__),
        "mv15b_module_sha256": Path(mv15b.__file__),
        "mv15b_protocol_sha256": mv15b.protocol_path(),
    }
    for key, path in checks.items():
        if _sha256(path) != source[key]:
            raise ValueError(f"MV15C immutable source mismatch: {key}")
    return protocol


def condition_map(
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    value = (
        json.loads(protocol_path().read_text(encoding="utf-8"))
        if protocol is None
        else protocol
    )
    return {str(item["id"]): dict(item) for item in value["fresh_conditions"]}


def fresh_tasks() -> list[tuple[str, int]]:
    return [
        (condition, int(seed))
        for condition, specification in condition_map().items()
        for seed in specification["seeds"]
    ]


def task_from_index(index: int) -> tuple[str, int]:
    tasks = fresh_tasks()
    if not 0 <= int(index) < len(tasks):
        raise ValueError("MV15C reference index is outside the locked array")
    return tasks[int(index)]


def verify_lock() -> dict[str, Any]:
    protocol = locked_protocol()
    configured = [
        stage_configuration(f"{STAGE}::{condition}", seed)
        for condition, seed in fresh_tasks()
    ]
    return {
        "stage": STAGE,
        "status": "MV15C_lock_verified_before_fresh_trajectories",
        "protocol_sha256": _sha256(protocol_path()),
        "fresh_tasks": [
            {"condition": condition, "seed": seed}
            for condition, seed in fresh_tasks()
        ],
        "input_budget": BUDGET,
        "input_blocks": list(INPUT_BLOCKS),
        "grid": configured[0][0].nx,
        "steps": configured[0][0].steps,
        "trajectory_count": len(configured),
        "parameter_tuning_on_fresh_data": False,
        "neural_network_retraining": False,
    }


def stage_configuration(
    stage: str,
    seed: int,
    *,
    reference_dir: Path | None = None,
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    """Inherit the exact verified MV3 trajectory engine under the MV15C lock."""

    del reference_dir
    if not stage.startswith(f"{STAGE}::"):
        raise ValueError(f"MV15C runner accepts only {STAGE}::<condition>")
    condition_id = stage.split("::", 1)[1]
    protocol = locked_protocol()
    conditions = condition_map(protocol)
    if condition_id not in conditions:
        raise ValueError(f"unknown MV15C condition {condition_id!r}")
    condition = conditions[condition_id]
    if int(seed) not in [int(value) for value in condition["seeds"]]:
        raise ValueError("MV15C seed is not preregistered")
    mv3ref, mv3 = _mv3ref_module(), _mv3_module()
    mv3_protocol = mv3.locked_protocol()
    if (
        _sha256(mv3.protocol_path())
        != protocol["source_contract"]["mv3_protocol_sha256"]
    ):
        raise ValueError("MV3 trajectory protocol differs from the MV15C lock")
    contract = mv3_protocol["reference_contract"]
    base = mv3ref.mohammadzadeh_config(
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
    engine_protocol = dict(mv3_protocol)
    engine_protocol["stage"] = STAGE
    engine_protocol["runtime_contract"] = {
        "nonoverlapping_sampling_blocks": int(
            contract["nonoverlapping_sampling_blocks"]
        ),
        "sample_count": int(contract["sample_count"]),
        "maximum_events_per_particle_per_step": int(
            contract["maximum_events_per_particle_per_step"]
        ),
    }
    specification = {
        "name": stage,
        "condition_id": condition_id,
        "grid": cfg.nx,
        "steps": cfg.steps,
        "sample_start": cfg.sample_start,
        "checkpoint_interval_steps": int(contract["checkpoint_interval_steps"]),
    }
    repair = json.loads(
        (mv3.reference_directory() / "m1r_repair_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    hashes = {
        "mv15c_protocol_sha256": _sha256(protocol_path()),
        "mv3_protocol_sha256": _sha256(mv3.protocol_path()),
    }
    return cfg, engine_protocol, specification, repair, hashes


def run_reference_seed(
    *,
    condition_id: str,
    seed: int,
    output_dir: Path,
    resume: bool = True,
    stop_after_step: int | None = None,
    progress: Any = None,
) -> dict[str, Any]:
    mv3ref, mv3 = _mv3ref_module(), _mv3_module()
    original_configuration, original_stage = mv3ref.stage_configuration, mv3ref.STAGE
    mv3ref.stage_configuration, mv3ref.STAGE = stage_configuration, STAGE
    try:
        result = mv3ref.run_reference_seed(
            condition_id=condition_id,
            seed=int(seed),
            output_dir=Path(output_dir),
            resume=resume,
            stop_after_step=stop_after_step,
            progress=progress,
        )
    finally:
        mv3ref.stage_configuration, mv3ref.STAGE = (
            original_configuration,
            original_stage,
        )
    if result.get("status") != "complete_MV3_reference_seed":
        return result
    path = Path(output_dir)
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    mechanical = [
        bool(value)
        for key, value in summary.get("mechanical_checks", {}).items()
        if key != "stationarity_pass"
    ]
    macroscopic_stationarity = [
        bool(value)
        for key, value in summary.get("stationarity", {}).get("checks", {}).items()
        if not mv3._is_heat_flux_stationarity_key(str(key))
    ]
    accepted = (
        bool(mechanical)
        and all(mechanical)
        and bool(macroscopic_stationarity)
        and all(macroscopic_stationarity)
    )
    summary.update(
        {
            "stage": STAGE,
            "condition_id": condition_id,
            "seed": int(seed),
            "status": COMPLETE_REFERENCE_STATUS,
            "decision": (
                "accept_MV15C_fresh_reference_for_cross_seed_qy_analysis"
                if accepted
                else "hold_MV15C_fresh_reference"
            ),
            "single_seed_heat_flux_stationarity_used_for_selection": False,
            "mv15c_protocol_sha256": _sha256(protocol_path()),
        }
    )
    _atomic_json(path / "summary.json", summary)
    manifest_path = path / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "stage": STAGE,
            "condition_id": condition_id,
            "seed": int(seed),
            "status": COMPLETE_REFERENCE_STATUS,
        }
    )
    manifest["files"]["summary.json"] = {
        "sha256": _sha256(path / "summary.json"),
        "size_bytes": (path / "summary.json").stat().st_size,
    }
    _atomic_json(manifest_path, manifest)
    if not accepted:
        raise RuntimeError(f"MV15C trajectory gate failed: {condition_id}/{seed}")
    return summary


def _validate_mv15b_outcome(
    mv15b_output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    root = Path(mv15b_output_root).resolve()
    _verify_manifest(root, "prediction_manifest.json")
    _verify_manifest(root, "artifact_manifest.json")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    selection = json.loads(
        (root / "selection_summary.json").read_text(encoding="utf-8")
    )
    required = locked_protocol()["required_MV15B_outcome"]
    budget = summary["budget_results"]["3"]
    selected = budget["selected_development_configuration"]
    checks = {
        "decision": summary.get("decision") == required["decision"],
        "recommended_budget": summary.get(
            "recommended_budget_for_separately_locked_fresh_confirmation"
        )
        == int(required["recommended_budget"]),
        "primary_condition": summary.get("primary_condition")
        == required["primary_condition"],
        "B3_all_gates_pass": bool(budget.get("all_gates_pass"))
        == bool(required["B3_all_gates_pass"]),
        "B3_primary_ratio": bool(
            np.isclose(
                float(budget["ratios_to_Raw_B10"]["selected"][PRIMARY_CONDITION]),
                float(required["B3_primary_ratio_to_Raw_B10"]),
                rtol=0.0,
                atol=1.0e-12,
            )
        ),
        "B3_threshold": float(selected["mode_reliability_threshold"])
        == float(required["B3_threshold"]),
        "B3_strength": float(selected["trusted_mode_strength"])
        == float(required["B3_strength"]),
        "B3_trusted_modes": int(selected["trusted_non_DC_mode_count"])
        == int(required["B3_trusted_non_DC_modes"]),
        "legacy_not_confirmation": summary.get("old_evaluation_seeds_are_confirmation")
        is False,
    }
    if not all(checks.values()):
        raise ValueError(f"MV15C required MV15B outcome mismatch: {checks}")
    with np.load(root / "locked_predictions.npz", allow_pickle=False) as arrays:
        weight = np.asarray(arrays["b3_selected_weight_map"], dtype=np.float64)
    frozen = locked_protocol()["frozen_B3_contract"]
    if (
        weight.ndim != 2
        or weight[0, 0] != 1.0
        or np.count_nonzero(weight) - 1
        != int(frozen["expected_trusted_non_DC_mode_count"])
        or not np.isclose(
            np.max(weight.reshape(-1)[1:]),
            float(frozen["expected_maximum_non_DC_weight"]),
            rtol=0.0,
            atol=1.0e-12,
        )
    ):
        raise ValueError("MV15C B3 weight map differs from the locked MV15B map")
    return summary, selection, weight


def prepare_submission_lock(
    mv9_output_root: Path,
    mv15b_output_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    protocol = locked_protocol()
    mv9_root = Path(mv9_output_root).resolve()
    mv15b_root = Path(mv15b_output_root).resolve()
    output = Path(output_root).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite MV15C output: {output}")
    _verify_manifest(mv9_root, "assembly_manifest.json")
    _verify_manifest(mv9_root, "artifact_manifest.json")
    mv15b_summary, _, weight = _validate_mv15b_outcome(mv15b_root)
    output.mkdir(parents=True)
    (output / PROTOCOL_FILE).write_bytes(protocol_path().read_bytes())
    lock = {
        "stage": STAGE,
        "status": "MV15C_submission_locked_before_any_fresh_trajectory",
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": _sha256(protocol_path()),
        "mv9_output_root": str(mv9_root),
        "mv15b_output_root": str(mv15b_root),
        "mv9_artifact_manifest_sha256": _sha256(
            mv9_root / "artifact_manifest.json"
        ),
        "mv15b_artifact_manifest_sha256": _sha256(
            mv15b_root / "artifact_manifest.json"
        ),
        "mv15b_prediction_manifest_sha256": _sha256(
            mv15b_root / "prediction_manifest.json"
        ),
        "mv15b_summary_sha256": _sha256(mv15b_root / "summary.json"),
        "mv15b_locked_predictions_sha256": _sha256(
            mv15b_root / "locked_predictions.npz"
        ),
        "mv15b_decision": mv15b_summary["decision"],
        "frozen_weight_map_sha256": hashlib.sha256(weight.tobytes()).hexdigest(),
        "fresh_tasks": [
            {"condition": condition, "seed": seed}
            for condition, seed in fresh_tasks()
        ],
        "fresh_data_read_before_lock": False,
    }
    _atomic_json(output / "submission_lock.json", lock)
    _atomic_json(
        output / "source_lock_manifest.json",
        {
            "stage": STAGE,
            "files": {
                name: {
                    "sha256": _sha256(output / name),
                    "size_bytes": (output / name).stat().st_size,
                }
                for name in (PROTOCOL_FILE, "submission_lock.json")
            },
        },
    )
    return lock


def build_b3_image(
    source: Mapping[str, Any], condition: Mapping[str, Any]
) -> np.ndarray:
    mv9 = _mv9_module()
    blocks = np.asarray(source["blocks"])
    auxiliary = np.asarray(source["block_auxiliary"])
    if blocks.shape[0] != 10 or auxiliary.shape[0] != 10:
        raise ValueError("MV15C requires exactly ten additive source blocks")
    output_b3 = np.mean(blocks[list(INPUT_BLOCKS)], axis=0, dtype=np.float64)
    auxiliary_b3 = np.mean(
        auxiliary[list(INPUT_BLOCKS)], axis=0, dtype=np.float64
    )
    return mv9._conditioned_image(output_b3, auxiliary_b3, condition)


def _write_source_audit(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_prediction_stage(
    output_root: Path,
    *,
    batch_size: int,
) -> dict[str, Any]:
    """Predict from B3 and lock all arms before constructing fresh targets."""

    output = Path(output_root).resolve()
    _verify_manifest(output, "source_lock_manifest.json")
    lock = json.loads((output / "submission_lock.json").read_text(encoding="utf-8"))
    mv9_root = Path(lock["mv9_output_root"]).resolve()
    mv15b_root = Path(lock["mv15b_output_root"]).resolve()
    _, _, weight = _validate_mv15b_outcome(mv15b_root)
    mv9, mv14, mv15b = _mv9_module(), _mv14_module(), _mv15b_module()
    conditions_spec = condition_map(locked_protocol())
    images, conditions, seeds, raw_b10, scales, audit_rows = [], [], [], [], [], []
    for condition, seed in fresh_tasks():
        directory = output / "references" / condition / f"seed_{seed}"
        source = mv9.load_moment_source(directory)
        if source["summary_status"] != COMPLETE_REFERENCE_STATUS:
            raise ValueError(f"MV15C fresh source is incomplete: {directory}")
        additive = source["block_full_additive_agreement"]
        if (
            not bool(additive["sample_count_match"])
            or float(additive["maximum_relative_linf"]) > 1.0e-9
            or float(source["q_reconstruction_relative_difference"]) > 1.0e-10
            or float(source["minimum_covariance_eigenvalue_ratio"]) < -1.0e-10
        ):
            raise ValueError(f"MV15C fresh moment provenance failed: {directory}")
        images.append(build_b3_image(source, conditions_spec[condition]))
        conditions.append(condition)
        seeds.append(int(seed))
        raw_b10.append(np.asarray(source["full"])[QY_INDEX])
        scales.append(float(np.asarray(source["scales"])[QY_INDEX]))
        audit_rows.append(
            {
                "condition": condition,
                "seed": int(seed),
                "directory": str(directory),
                "summary_status": source["summary_status"],
                "B3_input_blocks": "0,1,2",
                "q_reconstruction_relative_difference": source[
                    "q_reconstruction_relative_difference"
                ],
                "block_full_maximum_relative_linf": additive[
                    "maximum_relative_linf"
                ],
                "minimum_covariance_eigenvalue_ratio": source[
                    "minimum_covariance_eigenvalue_ratio"
                ],
            }
        )
    images_array = np.asarray(images, dtype=np.float32)
    condition_array = np.asarray(conditions, dtype="U32")
    seed_array = np.asarray(seeds, dtype=np.int64)
    raw_b10_array = np.asarray(raw_b10, dtype=np.float64)
    if images_array.shape[-2:] != weight.shape:
        raise ValueError("fresh DSMC grid differs from the frozen B3 weight map")
    vision = mv14._predict_mamba_validation(
        mv9_root, images_array, batch_size=batch_size
    )[:, QY_INDEX].astype(np.float64)
    raw_b3 = images_array[:, QY_INDEX].astype(np.float64)
    selected = mv15b.data_consistent_residual(raw_b3, vision, weight)
    dc_weight = np.zeros_like(weight)
    dc_weight[0, 0] = 1.0
    dc_only = mv15b.data_consistent_residual(raw_b3, vision, dc_weight)
    assembly = json.loads(
        (mv9_root / "assembly_summary.json").read_text(encoding="utf-8")
    )
    tsvd_rank = int(assembly["classical_selection_development_only"]["tsvd_rank"])
    tsvd = mv9._project_modules()["tsvd"]
    tsvd_qy = np.asarray(
        tsvd(images_array[:, :4], tsvd_rank)[:, QY_INDEX], dtype=np.float64
    )
    permutation = _mv15a_module().cross_condition_permutation(condition_array)
    permuted = mv15b.data_consistent_residual(
        raw_b3[permutation], vision, weight
    )
    prediction_file = output / "locked_fresh_predictions.npz"
    np.savez_compressed(
        prediction_file,
        conditions=condition_array,
        seeds=seed_array,
        B3_members=np.tile(np.asarray(INPUT_BLOCKS, dtype=np.int64), (len(seeds), 1)),
        raw_b3_qy=raw_b3,
        vision_b3_qy=vision,
        dc_only_b3_qy=dc_only,
        selected_b3_qy=selected,
        tsvd_b3_qy=tsvd_qy,
        permuted_b3_qy=permuted,
        permutation=permutation,
        raw_b10_qy=raw_b10_array,
        q_ref_scales=np.asarray(scales, dtype=np.float64),
        frozen_weight_map=weight,
    )
    _write_source_audit(output / "fresh_source_audit.csv", audit_rows)
    summary = {
        "stage": STAGE,
        "status": "MV15C_fresh_predictions_locked_before_target_construction",
        "protocol_sha256": _sha256(protocol_path()),
        "fresh_source_count": len(audit_rows),
        "conditions": list(conditions_spec),
        "seeds": {key: list(value) for key, value in FRESH_SEEDS.items()},
        "B3_input_blocks": list(INPUT_BLOCKS),
        "Raw_B10_used_by_prediction": False,
        "fresh_cross_seed_targets_constructed": False,
        "parameter_selection_on_fresh_data": False,
        "tsvd_rank_from_MV9_development": tsvd_rank,
        "frozen_weight_map_sha256": hashlib.sha256(weight.tobytes()).hexdigest(),
    }
    _atomic_json(output / "prediction_summary.json", summary)
    (output / "PREDICTION_LOCK_PASS").write_text(
        "MV15C predictions locked before fresh target construction\n",
        encoding="utf-8",
    )
    locked_names = (
        "locked_fresh_predictions.npz",
        "fresh_source_audit.csv",
        "prediction_summary.json",
        "PREDICTION_LOCK_PASS",
        PROTOCOL_FILE,
        "submission_lock.json",
    )
    _atomic_json(
        output / "prediction_manifest.json",
        {
            "stage": STAGE,
            "files": {
                name: {
                    "sha256": _sha256(output / name),
                    "size_bytes": (output / name).stat().st_size,
                }
                for name in locked_names
            },
        },
    )
    return summary


def leave_one_seed_out_targets(
    raw_b10: np.ndarray,
    conditions: np.ndarray,
    seeds: np.ndarray,
) -> np.ndarray:
    raw_b10 = np.asarray(raw_b10, dtype=np.float64)
    conditions = np.asarray(conditions)
    seeds = np.asarray(seeds)
    targets = np.empty_like(raw_b10)
    for index, (condition, seed) in enumerate(zip(conditions, seeds)):
        peers = (conditions == condition) & (seeds != seed)
        if np.count_nonzero(peers) != 3:
            raise ValueError("MV15C target requires exactly three other fresh seeds")
        targets[index] = np.mean(raw_b10[peers], axis=0, dtype=np.float64)
    return targets


def _nrmse(candidate: np.ndarray, target: np.ndarray) -> float:
    candidate = np.asarray(candidate, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return float(
        np.sqrt(np.mean((candidate - target) ** 2))
        / max(np.sqrt(np.mean(target**2)), EPS)
    )


def _metrics(
    methods: Mapping[str, np.ndarray],
    target: np.ndarray,
    conditions: np.ndarray,
    seeds: np.ndarray,
) -> tuple[
    dict[str, dict[str, dict[str, float]]],
    dict[str, dict[str, float]],
]:
    per_seed: dict[str, dict[str, dict[str, float]]] = {}
    means: dict[str, dict[str, float]] = {}
    for name, values in methods.items():
        per_seed[name], means[name] = {}, {}
        for condition in np.unique(conditions):
            per_seed[name][str(condition)] = {}
            for seed in seeds[conditions == condition]:
                mask = (conditions == condition) & (seeds == seed)
                per_seed[name][str(condition)][str(int(seed))] = _nrmse(
                    values[mask], target[mask]
                )
            means[name][str(condition)] = float(
                np.mean(list(per_seed[name][str(condition)].values()))
            )
    return per_seed, means


def confirmation_gates(
    *,
    means: Mapping[str, Mapping[str, float]],
    selected_seed_ratios: Mapping[str, Mapping[str, float]],
    dc_error: float,
    conditions: np.ndarray,
    seeds: np.ndarray,
    contract: Mapping[str, Any],
) -> dict[str, bool]:
    """Evaluate only preregistered gates; never select a fresh-data parameter."""

    expected_pairs = {
        (condition, int(seed))
        for condition, values in FRESH_SEEDS.items()
        for seed in values
    }
    observed_pairs = {
        (str(condition), int(seed))
        for condition, seed in zip(np.asarray(conditions), np.asarray(seeds))
    }
    ratios = {
        name: {
            condition: float(value)
            / max(float(means["raw_b10"][condition]), EPS)
            for condition, value in by_condition.items()
        }
        for name, by_condition in means.items()
    }
    return {
        "each_condition_mean_no_worse_than_Raw_B10": all(
            ratios["selected_b3"][condition]
            <= float(contract["maximum_each_condition_mean_ratio_to_Raw_B10"])
            for condition in FRESH_SEEDS
        ),
        "every_seed_within_Raw_B10_cap": all(
            float(value) <= float(contract["maximum_each_seed_ratio_to_Raw_B10"])
            for by_seed in selected_seed_ratios.values()
            for value in by_seed.values()
        ),
        "selected_beats_Mamba_B3_each_condition": all(
            float(means["selected_b3"][condition])
            < float(means["vision_b3"][condition])
            for condition in FRESH_SEEDS
        ),
        "selected_beats_TSVD_B3_each_condition": all(
            float(means["selected_b3"][condition])
            < float(means["tsvd_b3"][condition])
            for condition in FRESH_SEEDS
        ),
        "selected_beats_Raw_B3_each_condition": all(
            float(means["selected_b3"][condition])
            < float(means["raw_b3"][condition])
            for condition in FRESH_SEEDS
        ),
        "permuted_observation_degrades_each_condition": all(
            float(means["permuted_b3"][condition])
            >= (
                1.0 + float(contract["minimum_permutation_degradation_fraction"])
            )
            * float(means["selected_b3"][condition])
            for condition in FRESH_SEEDS
        ),
        "DC_preserved_to_tolerance": float(dc_error)
        <= float(contract["maximum_DC_absolute_error"]),
        "all_eight_fresh_trajectories_present": len(observed_pairs) == 8
        and observed_pairs == expected_pairs,
        "prediction_locked_before_target_construction": True,
        "no_fresh_parameter_selection": True,
    }


def _write_metrics_csv(
    path: Path,
    per_seed: Mapping[str, Mapping[str, Mapping[str, float]]],
    raw_b10: Mapping[str, Mapping[str, float]],
) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("condition", "seed", "method", "qy_nrmse", "ratio_to_Raw_B10"))
        for method, by_condition in per_seed.items():
            for condition, by_seed in by_condition.items():
                for seed, value in by_seed.items():
                    writer.writerow(
                        (
                            condition,
                            seed,
                            method,
                            value,
                            value / max(raw_b10[condition][seed], EPS),
                        )
                    )


def _plot_confirmation_ratios(
    output: Path,
    ratios: Mapping[str, Mapping[str, float]],
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = ("raw_b3", "vision_b3", "dc_only_b3", "selected_b3", "tsvd_b3")
    labels = ("Raw B3", "Mamba B3", "DC-only", "MV15C B3", "TSVD B3")
    conditions = (PRIMARY_CONDITION, NEW_CONDITION)
    x = np.arange(len(methods))
    width = 0.36
    fig, axis = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    for offset, condition in enumerate(conditions):
        axis.bar(
            x + (offset - 0.5) * width,
            [ratios[name][condition] for name in methods],
            width,
            label=condition,
        )
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axis.set(
        ylabel=r"mean $q_y$ NRMSE / paired Raw B10",
        xticks=x,
        xticklabels=labels,
        title="Prospectively locked fresh DSMC confirmation",
    )
    axis.tick_params(axis="x", rotation=18)
    axis.legend(frameon=False)
    paths = []
    for suffix in ("png", "pdf"):
        path = output / f"mv15c_fresh_qy_confirmation_ratios.{suffix}"
        fig.savefig(path, dpi=600 if suffix == "png" else None, bbox_inches="tight")
        paths.append(path.name)
    plt.close(fig)
    return paths


def _plot_condition_contour(
    output: Path,
    condition: str,
    seed: int,
    fields: Mapping[str, np.ndarray],
    target: np.ndarray,
    scale: float,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ("raw_b3", "vision_b3", "selected_b3", "tsvd_b3", "raw_b10")
    titles = ("Raw B3", "Mamba B3", "MV15C DCIR-QY B3", "TSVD B3", "Raw B10")
    arrays = [np.asarray(fields[name]) * scale for name in order]
    reference = np.asarray(target) * scale
    limit = max(float(np.max(np.abs(value))) for value in (*arrays, reference))
    levels = np.linspace(-limit, limit, 41)
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.0), constrained_layout=True)
    contour = None
    for axis, value, title in zip(
        axes.flat, (*arrays, reference), (*titles, "Independent cross-seed reference")
    ):
        contour = axis.contourf(value, levels=levels, cmap="coolwarm", extend="both")
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_aspect("equal")
    assert contour is not None
    fig.colorbar(contour, ax=axes, shrink=0.82, label=r"$q_y$ (W m$^{-2}$)")
    fig.suptitle(f"{condition}, fresh seed {seed}")
    paths = []
    for suffix in ("png", "pdf"):
        path = output / f"mv15c_fresh_qy_{condition}_seed_{seed}.{suffix}"
        fig.savefig(path, dpi=600 if suffix == "png" else None, bbox_inches="tight")
        paths.append(path.name)
    plt.close(fig)
    return paths


def run_post(output_root: Path) -> dict[str, Any]:
    """Construct fresh references only after recursive prediction locking."""

    output = Path(output_root).resolve()
    _verify_manifest(output, "prediction_manifest.json")
    prediction_summary = json.loads(
        (output / "prediction_summary.json").read_text(encoding="utf-8")
    )
    if prediction_summary.get("fresh_cross_seed_targets_constructed") is not False:
        raise ValueError("MV15C prediction/target separation failed")
    with np.load(output / "locked_fresh_predictions.npz", allow_pickle=False) as data:
        locked = {key: np.asarray(data[key]) for key in data.files}
    conditions, seeds = locked["conditions"], locked["seeds"]
    target = leave_one_seed_out_targets(locked["raw_b10_qy"], conditions, seeds)
    methods = {
        "raw_b3": locked["raw_b3_qy"],
        "vision_b3": locked["vision_b3_qy"],
        "dc_only_b3": locked["dc_only_b3_qy"],
        "selected_b3": locked["selected_b3_qy"],
        "tsvd_b3": locked["tsvd_b3_qy"],
        "permuted_b3": locked["permuted_b3_qy"],
        "raw_b10": locked["raw_b10_qy"],
    }
    per_seed, means = _metrics(methods, target, conditions, seeds)
    ratios = {
        name: {
            condition: value / max(means["raw_b10"][condition], EPS)
            for condition, value in by_condition.items()
        }
        for name, by_condition in means.items()
    }
    selected_seed_ratios = {
        condition: {
            seed: value / max(per_seed["raw_b10"][condition][seed], EPS)
            for seed, value in by_seed.items()
        }
        for condition, by_seed in per_seed["selected_b3"].items()
    }
    dc_error = float(
        np.max(
            np.abs(
                np.mean(methods["selected_b3"], axis=(-2, -1))
                - np.mean(methods["raw_b3"], axis=(-2, -1))
            )
        )
    )
    gates = confirmation_gates(
        means=means,
        selected_seed_ratios=selected_seed_ratios,
        dc_error=dc_error,
        conditions=conditions,
        seeds=seeds,
        contract=locked_protocol()["acceptance_gates"],
    )
    decision = (
        "MV15C_fresh_DSMC_confirms_B3_DCIR_QY"
        if all(gates.values())
        else "MV15C_fresh_DSMC_does_not_confirm_B3_DCIR_QY_no_retuning"
    )
    _write_metrics_csv(
        output / "mv15c_fresh_qy_metrics.csv", per_seed, per_seed["raw_b10"]
    )
    figures = _plot_confirmation_ratios(output, ratios)
    for condition in (PRIMARY_CONDITION, NEW_CONDITION):
        seed = FRESH_SEEDS[condition][0]
        index = int(np.flatnonzero((conditions == condition) & (seeds == seed))[0])
        figures.extend(
            _plot_condition_contour(
                output,
                condition,
                seed,
                {name: values[index] for name, values in methods.items()},
                target[index],
                float(locked["q_ref_scales"][index]),
            )
        )
    decomposition = {
        condition: {
            name: _mv15a_module().exact_affine_error_decomposition(
                values[conditions == condition], target[conditions == condition]
            )
            for name, values in methods.items()
        }
        for condition in FRESH_SEEDS
    }
    summary = {
        "stage": STAGE,
        "status": "complete_MV15C_prospectively_locked_fresh_confirmation",
        "protocol_sha256": _sha256(protocol_path()),
        "decision": decision,
        "fresh_conditions": list(FRESH_SEEDS),
        "mean_seed_qy_nrmse": means,
        "condition_mean_ratios_to_Raw_B10": ratios,
        "selected_per_seed_ratios_to_Raw_B10": selected_seed_ratios,
        "exact_error_decomposition": decomposition,
        "maximum_DC_absolute_error": dc_error,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "prediction_hash_locked_before_fresh_target_construction": True,
        "fresh_outcomes_used_for_tuning": False,
        "figures": figures,
    }
    _atomic_json(output / "summary.json", summary)
    return summary


def package_results(output_root: Path, return_directory: Path) -> dict[str, Any]:
    output = Path(output_root).resolve()
    returned = Path(return_directory).resolve()
    _verify_manifest(output, "prediction_manifest.json")
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    names = [
        "submission_lock.json",
        "source_lock_manifest.json",
        "prediction_summary.json",
        "prediction_manifest.json",
        "fresh_source_audit.csv",
        "summary.json",
        "mv15c_fresh_qy_metrics.csv",
        PROTOCOL_FILE,
        "mv15c_fresh_qy_confirmation_ratios.png",
        "mv15c_fresh_qy_confirmation_ratios.pdf",
    ]
    for condition in (PRIMARY_CONDITION, NEW_CONDITION):
        seed = FRESH_SEEDS[condition][0]
        names.extend(
            (
                f"mv15c_fresh_qy_{condition}_seed_{seed}.png",
                f"mv15c_fresh_qy_{condition}_seed_{seed}.pdf",
            )
        )
    accounting = output / "slurm_accounting.psv"
    if accounting.is_file():
        names.append(accounting.name)
    files = [output / name for name in names]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    artifact_manifest = {
        "stage": STAGE,
        "status": "complete_MV15C_compact_return_manifest",
        "files": {
            path.name: {
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in files
        },
    }
    _atomic_json(output / "artifact_manifest.json", artifact_manifest)
    _verify_manifest(output, "artifact_manifest.json")
    verification = {
        "stage": STAGE,
        "status": "complete_MV15C_recursive_return_verification",
        "decision": "verified",
        "verified_file_count": len(artifact_manifest["files"]),
        "manifest_sha256": _sha256(output / "artifact_manifest.json"),
    }
    _atomic_json(output / "verification.json", verification)
    files.extend((output / "artifact_manifest.json", output / "verification.json"))
    returned.mkdir(parents=True, exist_ok=True)
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = returned / f"MV15C_FRESH_B3_CONFIRMATION_BUNDLE_{tag}.zip"
    if archive.exists():
        raise FileExistsError(f"refusing to overwrite MV15C archive: {archive}")
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as stream:
        for path in files:
            stream.write(path, arcname=path.name)
    if archive.stat().st_size > 450 * 1024 * 1024:
        raise RuntimeError("MV15C return archive exceeds 450 MiB")
    result = {
        "stage": STAGE,
        "archive": str(archive),
        "archive_sha256": _sha256(archive),
        "decision": summary["decision"],
        "all_gates_pass": summary["all_gates_pass"],
    }
    _atomic_json(output / "return.json", result)
    pointer = returned / "LAST_MOHAMMADZADEH_MV15C_FRESH_B3_RESULT.env"
    pointer.write_text(
        "\n".join(
            (
                f"MV15C_OUTPUT_ROOT={output}",
                f"MV15C_RESULT_ARCHIVE={archive}",
                f"MV15C_RESULT_ARCHIVE_SHA256={result['archive_sha256']}",
                f"MV15C_DECISION={result['decision']}",
                f"MV15C_ALL_GATES_PASS={int(bool(result['all_gates_pass']))}",
                "",
            )
        ),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("verify-lock")
    prepare = subparsers.add_parser("prepare-lock")
    prepare.add_argument("--mv9-output-root", type=Path, required=True)
    prepare.add_argument("--mv15b-output-root", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    reference = subparsers.add_parser("run-reference")
    reference.add_argument("--task-index", type=int, required=True)
    reference.add_argument("--output-root", type=Path, required=True)
    reference.add_argument("--no-resume", action="store_true")
    reference.add_argument("--stop-after-step", type=int)
    predict = subparsers.add_parser("predict")
    predict.add_argument("--output-root", type=Path, required=True)
    predict.add_argument("--batch-size", type=int, default=8)
    post = subparsers.add_parser("post")
    post.add_argument("--output-root", type=Path, required=True)
    package = subparsers.add_parser("package")
    package.add_argument("--output-root", type=Path, required=True)
    package.add_argument("--return-directory", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "verify-lock":
        result = verify_lock()
    elif args.command == "prepare-lock":
        result = prepare_submission_lock(
            args.mv9_output_root, args.mv15b_output_root, args.output_root
        )
    elif args.command == "run-reference":
        condition, seed = task_from_index(args.task_index)
        directory = args.output_root / "references" / condition / f"seed_{seed}"
        result = run_reference_seed(
            condition_id=condition,
            seed=seed,
            output_dir=directory,
            resume=not args.no_resume,
            stop_after_step=args.stop_after_step,
            progress=lambda step, total: print(
                _json_dumps({"step": step, "total": total}), flush=True
            ),
        )
    elif args.command == "predict":
        result = run_prediction_stage(args.output_root, batch_size=args.batch_size)
    elif args.command == "post":
        result = run_post(args.output_root)
    else:
        result = package_results(args.output_root, args.return_directory)
    print(_json_dumps(result))


if __name__ == "__main__":
    main()
