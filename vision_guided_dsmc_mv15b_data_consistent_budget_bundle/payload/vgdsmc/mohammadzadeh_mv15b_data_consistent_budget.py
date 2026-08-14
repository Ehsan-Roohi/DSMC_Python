"""MV15B data-consistent q_y reconstruction and disjoint budget ladder.

MV15B launches no DSMC trajectory and trains no new neural network.  It uses
the ten already verified B1 blocks in every MV9 seed to construct strictly
disjoint B1/B2/B3/B5 inputs.  A two-dimensional, target-free spectral
reliability map determines which measured modes may correct the locked Mamba
prediction.  The DCT DC coefficient is always copied from the indexed DSMC
observation, so extrapolation cannot invent the spatial mean of q_y.

Development labels select only a reliability threshold and correction
strength for each budget.  Legacy targets are not loaded until every budget
prediction and control has been recursively SHA256 locked.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np


STAGE = "MV15B_Mohammadzadeh_data_consistent_budget_ladder"
STATUS = "locked_after_MV15A_failure_before_any_MV15B_legacy_outcome"
PROTOCOL_FILE = "mv15b_data_consistent_budget_protocol.json"
QY_INDEX = 3
BUDGETS = (1, 2, 3, 5)
TRUST_THRESHOLDS = (0.50, 0.75, 0.90, 0.97)
TRUST_STRENGTHS = (0.25, 0.50, 0.75, 1.00)
PRIMARY_CONDITION = "kn0p1_u400"
REPRESENTATIVE_SEED = 94302
REPRESENTATIVE_GROUP = 0
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    """Convert NumPy scalars at the JSON boundary without hiding arrays."""

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
    temporary.write_text(
        _json_dumps(value) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


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


def _ancestry_protocol_path(module: Any, filename: str) -> Path:
    path = (
        Path(module.__file__).resolve().parents[1]
        / "reference_data"
        / "mohammadzadeh_2012"
        / filename
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def locked_protocol() -> dict[str, Any]:
    protocol = json.loads(protocol_path().read_text(encoding="utf-8"))
    if protocol.get("stage") != STAGE or protocol.get("status") != STATUS:
        raise ValueError("MV15B protocol is absent or unlocked")
    matrix = protocol["selection_contract"]
    if (
        tuple(int(value) for value in matrix["budgets"]) != BUDGETS
        or tuple(float(value) for value in matrix["mode_reliability_thresholds"])
        != TRUST_THRESHOLDS
        or tuple(float(value) for value in matrix["trusted_mode_strengths"])
        != TRUST_STRENGTHS
        or protocol["analysis_contract"]["primary_condition"]
        != PRIMARY_CONDITION
    ):
        raise ValueError("MV15B implementation differs from its locked matrix")
    mv15a = _mv15a_module()
    mv15a.locked_protocol()
    sources = {
        "mv15a_module_sha256": Path(mv15a.__file__),
        "mv15a_protocol_sha256": _ancestry_protocol_path(
            mv15a, "mv15a_spectral_information_audit_protocol.json"
        ),
    }
    for key, path in sources.items():
        if _sha256(path) != protocol["source_contract"][key]:
            raise ValueError(f"MV15B immutable ancestry mismatch: {key}")
    return protocol


def verify_lock() -> dict[str, Any]:
    protocol = locked_protocol()
    return {
        "stage": STAGE,
        "status": "MV15B_lock_verified_without_any_MV15B_legacy_outcome",
        "protocol_sha256": _sha256(protocol_path()),
        "budgets": list(BUDGETS),
        "DSMC_rerun": False,
        "neural_network_retraining": False,
        "legacy_targets_loaded_by_prediction_stage": False,
        "exact_DCT_DC_preservation": True,
        "mode_trust_is_two_dimensional_not_radial": True,
    }


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
            raise ValueError(f"MV15B recursive artifact verification failed: {path}")
    return manifest


def verify_data_contract(mv9_output_root: Path, mv15a_output_root: Path) -> dict[str, Any]:
    protocol = locked_protocol()
    mv9_root = Path(mv9_output_root).resolve()
    mv15a_root = Path(mv15a_output_root).resolve()
    _verify_manifest(mv9_root, "assembly_manifest.json")
    _verify_manifest(mv9_root, "artifact_manifest.json")
    _verify_manifest(mv15a_root, "prediction_manifest.json")
    _verify_manifest(mv15a_root, "artifact_manifest.json")
    mv15a_summary = json.loads((mv15a_root / "summary.json").read_text(encoding="utf-8"))
    mv15a_selection = json.loads(
        (mv15a_root / "selection_summary.json").read_text(encoding="utf-8")
    )
    expected = protocol["source_contract"]
    primary = mv15a_summary.get("primary_qy_ratios_to_Raw_B10", {})
    selected = mv15a_summary.get("selected_spectral_fusion", {})
    checks = {
        "MV15A_failure_outcome_explicitly_required": mv15a_summary.get("decision")
        == expected["required_MV15A_decision"],
        "MV15A_primary_condition_matches": mv15a_summary.get("primary_condition")
        == PRIMARY_CONDITION,
        "MV15A_primary_spectral_ratio_matches": bool(
            np.isclose(
                float(primary.get("spectral_fusion", math.nan)),
                float(expected["required_MV15A_primary_spectral_ratio"]),
                rtol=0.0,
                atol=1.0e-12,
            )
        ),
        "MV15A_primary_TSVD_ratio_matches": bool(
            np.isclose(
                float(primary.get("tsvd_b1", math.nan)),
                float(expected["required_MV15A_primary_TSVD_ratio"]),
                rtol=0.0,
                atol=1.0e-12,
            )
        ),
        "MV15A_raw_weight_collapse_matches": bool(
            np.isclose(
                float(selected.get("maximum_weight", math.nan)),
                float(expected["required_MV15A_maximum_raw_weight"]),
                rtol=0.0,
                atol=1.0e-12,
            )
        ),
        "MV9_dataset_present": (mv9_root / "dataset.npz").is_file(),
        "MV15A_predictions_present": (mv15a_root / "locked_predictions.npz").is_file(),
        "MV15A_uses_same_MV9_root": Path(
            mv15a_selection["mv9_output_root"]
        ).resolve()
        == mv9_root,
        "legacy_outcomes_are_development_history_not_confirmation": True,
    }
    if not all(checks.values()):
        raise ValueError(f"MV15B data contract failed: {checks}")
    return {
        "stage": STAGE,
        "status": "MV15B_data_contract_verified",
        "mv9_output_root": str(mv9_root),
        "mv15a_output_root": str(mv15a_root),
        "checks": checks,
    }


def aggregate_disjoint(
    images: np.ndarray,
    conditions: np.ndarray,
    identities: np.ndarray,
    budget: int,
    targets: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Average non-overlapping consecutive B1 blocks within each seed.

    B3 deliberately uses blocks 0--8 and records block 9 as unused.  No block
    can appear in two samples of the same budget.
    """

    images = np.asarray(images)
    conditions = np.asarray(conditions)
    identities = np.asarray(identities)
    targets_array = None if targets is None else np.asarray(targets)
    budget = int(budget)
    if budget < 1 or budget > 10 or budget not in (*BUDGETS, 10):
        raise ValueError(f"unsupported MV15B budget: {budget}")
    if images.ndim != 4 or len(images) != len(conditions) or identities.shape[0] != len(images):
        raise ValueError("MV15B budget inputs have incompatible shapes")
    if targets_array is not None and len(targets_array) != len(images):
        raise ValueError("MV15B target count differs from input count")
    outputs, output_conditions, output_identities, members = [], [], [], []
    output_targets = []
    used: set[tuple[str, int, int]] = set()
    for condition in sorted(str(value) for value in np.unique(conditions)):
        condition_mask = conditions == condition
        for seed_value in sorted(int(value) for value in np.unique(identities[condition_mask, 0])):
            indices = np.flatnonzero(condition_mask & (identities[:, 0] == seed_value))
            indices = indices[np.argsort(identities[indices, 1])]
            blocks = identities[indices, 1].astype(int)
            if len(indices) != 10 or not np.array_equal(blocks, np.arange(10)):
                raise ValueError(f"MV15B requires blocks 0..9 for {condition}/{seed_value}")
            group_count = len(indices) // budget
            for group in range(group_count):
                chosen = indices[group * budget : (group + 1) * budget]
                block_members = identities[chosen, 1].astype(int)
                for block in block_members:
                    key = (condition, seed_value, int(block))
                    if key in used:
                        raise AssertionError(f"MV15B reused a B1 block: {key}")
                    used.add(key)
                condition_channels = images[chosen, -2:]
                if not np.all(condition_channels == condition_channels[0]):
                    raise ValueError("MV15B condition channels changed inside a seed")
                outputs.append(np.mean(images[chosen], axis=0, dtype=np.float64))
                output_conditions.append(condition)
                output_identities.append((seed_value, group, budget))
                padded = np.full(10, -1, dtype=np.int64)
                padded[:budget] = block_members
                members.append(padded)
                if targets_array is not None:
                    reference = targets_array[chosen]
                    if not np.allclose(reference, reference[0], rtol=0.0, atol=1.0e-7):
                        raise ValueError("MV15B targets change across blocks of one seed")
                    output_targets.append(reference[0])
    result = {
        "images": np.asarray(outputs, dtype=np.float32),
        "conditions": np.asarray(output_conditions, dtype="U32"),
        "identities": np.asarray(output_identities, dtype=np.int64),
        "members": np.asarray(members, dtype=np.int64),
    }
    if targets_array is not None:
        result["targets"] = np.asarray(output_targets, dtype=np.float32)
    return result


def budget_reliability(signal: np.ndarray, noise: np.ndarray, budget: int) -> np.ndarray:
    signal = np.maximum(np.asarray(signal, dtype=np.float64), 0.0)
    noise = np.maximum(np.asarray(noise, dtype=np.float64), 0.0) / float(budget)
    if signal.shape != noise.shape:
        raise ValueError("MV15B signal/noise spectra have different shapes")
    return signal / np.maximum(signal + noise, EPS)


def trust_weight_map(
    reliability: np.ndarray, threshold: float, strength: float
) -> np.ndarray:
    reliability = np.asarray(reliability, dtype=np.float64)
    weight = np.where(reliability >= float(threshold), float(strength) * reliability, 0.0)
    weight = np.clip(weight, 0.0, 1.0)
    weight[0, 0] = 1.0
    return weight


def data_consistent_residual(
    raw_qy: np.ndarray, vision_qy: np.ndarray, weight_map: np.ndarray
) -> np.ndarray:
    mv15a = _mv15a_module()
    raw_qy = np.asarray(raw_qy, dtype=np.float64)
    vision_qy = np.asarray(vision_qy, dtype=np.float64)
    weight_map = np.asarray(weight_map, dtype=np.float64)
    if raw_qy.shape != vision_qy.shape or raw_qy.shape[-2:] != weight_map.shape:
        raise ValueError("MV15B data-consistency shapes are incompatible")
    raw_coefficients = mv15a._dct2(raw_qy)
    vision_coefficients = mv15a._dct2(vision_qy)
    prediction = mv15a._idct2(
        vision_coefficients + weight_map * (raw_coefficients - vision_coefficients)
    )
    prediction += np.mean(raw_qy, axis=(-2, -1), keepdims=True) - np.mean(
        prediction, axis=(-2, -1), keepdims=True
    )
    return np.asarray(prediction, dtype=np.float64)


def _nrmse(candidate: np.ndarray, target: np.ndarray) -> float:
    candidate = np.asarray(candidate, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return float(
        np.sqrt(np.mean((candidate - target) ** 2))
        / max(np.sqrt(np.mean(target**2)), EPS)
    )


def select_data_consistency(
    raw_qy: np.ndarray,
    vision_qy: np.ndarray,
    target_qy: np.ndarray,
    conditions: np.ndarray,
    reliability: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = [(1.1, 0.0)] + [
        (threshold, strength)
        for threshold in TRUST_THRESHOLDS
        for strength in TRUST_STRENGTHS
    ]
    records: list[dict[str, Any]] = []
    for threshold, strength in candidates:
        weights = trust_weight_map(reliability, threshold, strength)
        prediction = data_consistent_residual(raw_qy, vision_qy, weights)
        by_condition = {
            str(condition): _nrmse(
                prediction[conditions == condition], target_qy[conditions == condition]
            )
            for condition in np.unique(conditions)
        }
        records.append(
            {
                "mode_reliability_threshold": float(threshold),
                "trusted_mode_strength": float(strength),
                "trusted_non_DC_mode_count": int(np.count_nonzero(weights) - 1),
                "development_qy_nrmse_by_condition": by_condition,
                "condition_balanced_mean_development_qy_nrmse": float(
                    np.mean(list(by_condition.values()))
                ),
            }
        )
    selected = min(
        records,
        key=lambda row: (
            row["condition_balanced_mean_development_qy_nrmse"],
            row["trusted_non_DC_mode_count"],
            row["trusted_mode_strength"],
            -row["mode_reliability_threshold"],
        ),
    )
    selected = dict(selected)
    weights = trust_weight_map(
        reliability,
        selected["mode_reliability_threshold"],
        selected["trusted_mode_strength"],
    )
    selected["maximum_non_DC_weight"] = float(np.max(weights.reshape(-1)[1:]))
    selected["DC_weight"] = float(weights[0, 0])
    return selected, records


def _write_development_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    conditions = sorted(
        {
            condition
            for row in rows
            for condition in row["development_qy_nrmse_by_condition"]
        }
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "budget",
                "threshold",
                "strength",
                "trusted_non_DC_modes",
                "condition_balanced_mean_qy_nrmse",
                *conditions,
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row["budget"],
                    row["mode_reliability_threshold"],
                    row["trusted_mode_strength"],
                    row["trusted_non_DC_mode_count"],
                    row["condition_balanced_mean_development_qy_nrmse"],
                    *(
                        row["development_qy_nrmse_by_condition"].get(condition, "")
                        for condition in conditions
                    ),
                )
            )


def run_prediction_stage(
    mv9_output_root: Path,
    mv15a_output_root: Path,
    output_root: Path,
    *,
    batch_size: int,
) -> dict[str, Any]:
    """Select on development arrays and hash-lock every legacy prediction."""

    data_contract = verify_data_contract(mv9_output_root, mv15a_output_root)
    mv9_root = Path(mv9_output_root).resolve()
    mv15a_root = Path(mv15a_output_root).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite MV15B output: {output_root}")
    with np.load(mv9_root / "dataset.npz", allow_pickle=False) as data:
        train_x = np.asarray(data["train_x"])
        train_conditions = np.asarray(data["train_condition"])
        train_identities = np.asarray(data["train_identity"])
        validation_x = np.asarray(data["validation_x"])
        validation_y = np.asarray(data["validation_y"])
        validation_conditions = np.asarray(data["validation_condition"])
        validation_identities = np.asarray(data["validation_identity"])
        test_x = np.asarray(data["test_x"])
        test_conditions = np.asarray(data["test_condition"])
        test_identities = np.asarray(data["test_identity"])
    mv15a = _mv15a_module()
    mv14 = _mv14_module()
    mv9 = _mv9_module()
    audit = mv15a.cross_spectral_information(
        train_x[:, QY_INDEX], train_conditions, train_identities
    )
    with np.load(mv15a_root / "locked_predictions.npz", allow_pickle=False) as previous:
        if not np.allclose(
            audit["global_signal_mode"], previous["global_signal_mode"], rtol=1.0e-12, atol=1.0e-12
        ) or not np.allclose(
            audit["global_noise_mode"], previous["global_noise_mode"], rtol=1.0e-12, atol=1.0e-12
        ):
            raise ValueError("MV15B spectral ancestry differs from MV15A")
    assembly = json.loads((mv9_root / "assembly_summary.json").read_text(encoding="utf-8"))
    tsvd_rank = int(assembly["classical_selection_development_only"]["tsvd_rank"])
    tsvd = mv9._project_modules()["tsvd"]
    locked_arrays: dict[str, np.ndarray] = {
        "global_signal_mode": np.asarray(audit["global_signal_mode"], dtype=np.float64),
        "global_noise_mode_B1": np.asarray(audit["global_noise_mode"], dtype=np.float64),
    }
    selections: dict[str, Any] = {}
    all_records: list[dict[str, Any]] = []
    for budget in BUDGETS:
        train = aggregate_disjoint(
            train_x, train_conditions, train_identities, budget
        )
        validation = aggregate_disjoint(
            validation_x,
            validation_conditions,
            validation_identities,
            budget,
            validation_y,
        )
        test = aggregate_disjoint(test_x, test_conditions, test_identities, budget)
        validation_vision = mv14._predict_mamba_validation(
            mv9_root, validation["images"], batch_size=batch_size
        )
        test_vision = mv14._predict_mamba_validation(
            mv9_root, test["images"], batch_size=batch_size
        )
        budget_audit = mv15a.cross_spectral_information(
            train["images"][:, QY_INDEX],
            train["conditions"],
            train["identities"],
        )
        reliability = np.asarray(
            budget_audit["global_reliability_mode"], dtype=np.float64
        )
        selected, records = select_data_consistency(
            validation["images"][:, QY_INDEX],
            validation_vision[:, QY_INDEX],
            validation["targets"][:, QY_INDEX],
            validation["conditions"],
            reliability,
        )
        for row in records:
            all_records.append({"budget": budget, **row})
        selected_weights = trust_weight_map(
            reliability,
            selected["mode_reliability_threshold"],
            selected["trusted_mode_strength"],
        )
        dc_weights = np.zeros_like(selected_weights)
        dc_weights[0, 0] = 1.0
        raw_qy = test["images"][:, QY_INDEX]
        vision_qy = test_vision[:, QY_INDEX]
        selected_qy = data_consistent_residual(raw_qy, vision_qy, selected_weights)
        dc_only_qy = data_consistent_residual(raw_qy, vision_qy, dc_weights)
        permutation = mv15a.cross_condition_permutation(test["conditions"])
        permuted_qy = data_consistent_residual(
            raw_qy[permutation], vision_qy, selected_weights
        )
        condition_only_qy = mv15a.parametric_condition_only(
            validation["images"],
            validation["targets"][:, QY_INDEX],
            validation["conditions"],
            test["images"],
        )
        tsvd_qy = np.asarray(
            tsvd(test["images"][:, :4], tsvd_rank)[:, QY_INDEX], dtype=np.float64
        )
        prefix = f"b{budget}_"
        locked_arrays.update(
            {
                prefix + "conditions": test["conditions"],
                prefix + "identities": test["identities"],
                prefix + "members": test["members"],
                prefix + "raw_qy": np.asarray(raw_qy, dtype=np.float64),
                prefix + "vision_qy": np.asarray(vision_qy, dtype=np.float64),
                prefix + "dc_only_qy": dc_only_qy,
                prefix + "selected_qy": selected_qy,
                prefix + "tsvd_qy": tsvd_qy,
                prefix + "condition_only_qy": np.asarray(condition_only_qy, dtype=np.float64),
                prefix + "permuted_qy": permuted_qy,
                prefix + "permutation": permutation,
                prefix + "reliability_mode": reliability,
                prefix + "signal_mode": np.asarray(
                    budget_audit["global_signal_mode"], dtype=np.float64
                ),
                prefix + "noise_mode": np.asarray(
                    budget_audit["global_noise_mode"], dtype=np.float64
                ),
                prefix + "selected_weight_map": selected_weights,
            }
        )
        selections[str(budget)] = {
            **selected,
            "sample_count": int(len(test["images"])),
            "groups_per_seed": int(10 // budget),
            "used_blocks_per_seed": int((10 // budget) * budget),
            "unused_blocks_per_seed": int(10 - (10 // budget) * budget),
            "empirical_noise_power_ratio_to_B1": float(
                np.sum(budget_audit["global_noise_mode"])
                / max(np.sum(audit["global_noise_mode"]), EPS)
            ),
            "ideal_independent_noise_power_ratio_to_B1": 1.0 / float(budget),
        }
    b10_average = aggregate_disjoint(test_x, test_conditions, test_identities, 10)
    locked_arrays.update(
        {
            "field_average_b10_conditions": b10_average["conditions"],
            "field_average_b10_identities": b10_average["identities"],
            "field_average_b10_qy": np.asarray(
                b10_average["images"][:, QY_INDEX], dtype=np.float64
            ),
        }
    )
    output_root.mkdir(parents=True)
    np.savez_compressed(output_root / "locked_predictions.npz", **locked_arrays)
    _write_development_csv(output_root / "mv15b_development_selection.csv", all_records)
    selection_summary = {
        "stage": STAGE,
        "status": "complete_MV15B_development_selection_and_legacy_prediction_lock",
        "protocol_sha256": _sha256(protocol_path()),
        "mv9_output_root": str(mv9_root),
        "mv15a_output_root": str(mv15a_root),
        "data_contract": data_contract,
        "selected_by_budget": selections,
        "tsvd_rank_from_MV9_development": tsvd_rank,
        "legacy_test_targets_loaded": False,
        "legacy_conditions_predicted_without_labels": sorted(
            str(value) for value in np.unique(test_conditions)
        ),
        "disjoint_budget_rule": "consecutive_nonoverlapping_B1_groups_with_B3_block9_unused",
        "exact_DC_preservation": True,
        "decision": "lock_MV15B_predictions_before_legacy_diagnostic",
    }
    _atomic_json(output_root / "selection_summary.json", selection_summary)
    (output_root / PROTOCOL_FILE).write_bytes(protocol_path().read_bytes())
    locked_files = (
        "locked_predictions.npz",
        "selection_summary.json",
        "mv15b_development_selection.csv",
        PROTOCOL_FILE,
    )
    _atomic_json(
        output_root / "prediction_manifest.json",
        {
            "stage": STAGE,
            "files": {
                name: {
                    "sha256": _sha256(output_root / name),
                    "size_bytes": (output_root / name).stat().st_size,
                }
                for name in locked_files
            },
        },
    )
    return selection_summary


def _per_seed_qy(
    candidate: np.ndarray,
    target: np.ndarray,
    conditions: np.ndarray,
    identities: np.ndarray,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for condition in np.unique(conditions):
        condition_key = str(condition)
        result[condition_key] = {}
        condition_mask = conditions == condition
        for seed in np.unique(identities[condition_mask, 0]):
            mask = condition_mask & (identities[:, 0] == seed)
            result[condition_key][str(int(seed))] = _nrmse(
                candidate[mask], target[mask]
            )
    return result


def _mean_seed_metric(records: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    return {
        condition: float(np.mean(list(per_seed.values())))
        for condition, per_seed in records.items()
    }


def _raw10_metrics(
    raw10: np.ndarray,
    target10: np.ndarray,
    conditions10: np.ndarray,
    identities10: np.ndarray,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    per_seed = _per_seed_qy(
        raw10[:, QY_INDEX], target10[:, QY_INDEX], conditions10, identities10
    )
    return per_seed, _mean_seed_metric(per_seed)


def _plot_budget_curves(
    output_root: Path, budget_results: Mapping[str, Any]
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    budgets = np.asarray(BUDGETS, dtype=int)
    methods = ("raw", "vision", "dc_only", "selected", "tsvd")
    labels = ("Raw Bn", "Mamba Bn", "DC-only", "MV15B DCIR-QY", "TSVD Bn")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    for method, label in zip(methods, labels):
        axes[0].plot(
            budgets,
            [budget_results[str(b)]["ratios_to_Raw_B10"][method][PRIMARY_CONDITION] for b in budgets],
            "o-",
            label=label,
        )
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0].set(
        xlabel="number of disjoint B1 blocks",
        ylabel=r"$q_y$ NRMSE / Raw B10",
        title=f"Primary condition: {PRIMARY_CONDITION}",
        xticks=budgets,
    )
    axes[0].legend(frameon=False, fontsize=8)
    conditions = sorted(
        budget_results[str(BUDGETS[0])]["ratios_to_Raw_B10"]["selected"]
    )
    for condition in conditions:
        axes[1].plot(
            budgets,
            [budget_results[str(b)]["ratios_to_Raw_B10"]["selected"][condition] for b in budgets],
            "o-",
            label=condition,
        )
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1].set(
        xlabel="number of disjoint B1 blocks",
        ylabel=r"MV15B $q_y$ NRMSE / Raw B10",
        title="Condition-resolved budget ladder",
        xticks=budgets,
    )
    axes[1].legend(frameon=False, fontsize=8)
    paths = []
    for suffix in ("png", "pdf"):
        path = output_root / f"mv15b_qy_disjoint_budget_ladder.{suffix}"
        fig.savefig(path, dpi=600 if suffix == "png" else None, bbox_inches="tight")
        paths.append(path.name)
    plt.close(fig)
    return paths


def _plot_trust_map(
    output_root: Path,
    reliability: np.ndarray,
    weights: np.ndarray,
    budget: int,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), constrained_layout=True)
    image0 = axes[0].imshow(reliability, origin="lower", vmin=0.0, vmax=1.0, cmap="viridis")
    axes[0].set(title=f"Target-free reliability, B={budget}", xlabel="DCT x mode", ylabel="DCT y mode")
    fig.colorbar(image0, ax=axes[0], shrink=0.82)
    image1 = axes[1].imshow(weights, origin="lower", vmin=0.0, vmax=1.0, cmap="magma")
    axes[1].set(title="Locked 2-D data-consistency weights", xlabel="DCT x mode", ylabel="DCT y mode")
    fig.colorbar(image1, ax=axes[1], shrink=0.82)
    paths = []
    for suffix in ("png", "pdf"):
        path = output_root / f"mv15b_qy_modewise_trust_B{budget}.{suffix}"
        fig.savefig(path, dpi=600 if suffix == "png" else None, bbox_inches="tight")
        paths.append(path.name)
    plt.close(fig)
    return paths


def _plot_contours(
    output_root: Path,
    fields: Mapping[str, np.ndarray],
    target: np.ndarray,
    scale: float,
    budget: int,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ("raw", "vision", "dc_only", "selected", "tsvd", "raw10")
    titles = (f"Raw B{budget}", f"Mamba B{budget}", "DC-only", "MV15B DCIR-QY", f"TSVD B{budget}", "Raw B10")
    arrays = [np.asarray(fields[name]) * scale for name in names]
    reference = np.asarray(target) * scale
    limit = max(float(np.max(np.abs(value))) for value in (*arrays, reference))
    levels = np.linspace(-limit, limit, 41)
    fig, axes = plt.subplots(2, 4, figsize=(14.0, 7.0), constrained_layout=True)
    contour = None
    for axis, value, title in zip(axes.flat, (*arrays, reference), (*titles, "Cross-seed reference")):
        contour = axis.contourf(value, levels=levels, cmap="coolwarm", extend="both")
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_aspect("equal")
    axes.flat[-1].axis("off")
    assert contour is not None
    fig.colorbar(contour, ax=axes, shrink=0.8, label=r"$q_y$ (W m$^{-2}$)")
    paths = []
    for suffix in ("png", "pdf"):
        path = output_root / f"mv15b_qy_physical_contours_B{budget}.{suffix}"
        fig.savefig(path, dpi=600 if suffix == "png" else None, bbox_inches="tight")
        paths.append(path.name)
    plt.close(fig)
    return paths


def run_legacy_post(output_root: Path) -> dict[str, Any]:
    """Read legacy labels only after recursive prediction-lock verification."""

    protocol = locked_protocol()
    output_root = Path(output_root).resolve()
    _verify_manifest(output_root, "prediction_manifest.json")
    selection = json.loads((output_root / "selection_summary.json").read_text(encoding="utf-8"))
    if selection.get("legacy_test_targets_loaded") is not False:
        raise ValueError("MV15B prediction/label separation failed")
    mv9_root = Path(selection["mv9_output_root"]).resolve()
    with np.load(output_root / "locked_predictions.npz", allow_pickle=False) as data:
        locked = {key: np.asarray(data[key]) for key in data.files}
    with np.load(mv9_root / "dataset.npz", allow_pickle=False) as data:
        test_x = np.asarray(data["test_x"])
        test_y = np.asarray(data["test_y"])
        test_conditions = np.asarray(data["test_condition"])
        test_identities = np.asarray(data["test_identity"])
        test_scales = np.asarray(data["test_scale"])
        raw10 = np.asarray(data["test_raw10"])
        target10 = np.asarray(data["test_target10"])
        conditions10 = np.asarray(data["test_condition10"])
        identities10 = np.asarray(data["test_identity10"])
    raw10_per_seed, raw10_aggregate = _raw10_metrics(
        raw10, target10, conditions10, identities10
    )
    budget_results: dict[str, Any] = {}
    metrics_rows: list[tuple[Any, ...]] = []
    for budget in BUDGETS:
        aggregate = aggregate_disjoint(
            test_x, test_conditions, test_identities, budget, test_y
        )
        prefix = f"b{budget}_"
        if not np.array_equal(aggregate["conditions"], locked[prefix + "conditions"]) or not np.array_equal(
            aggregate["identities"], locked[prefix + "identities"]
        ):
            raise ValueError(f"MV15B locked identity mismatch for B{budget}")
        target_qy = aggregate["targets"][:, QY_INDEX]
        methods = {
            "raw": locked[prefix + "raw_qy"],
            "vision": locked[prefix + "vision_qy"],
            "dc_only": locked[prefix + "dc_only_qy"],
            "selected": locked[prefix + "selected_qy"],
            "tsvd": locked[prefix + "tsvd_qy"],
            "condition_only": locked[prefix + "condition_only_qy"],
            "permuted": locked[prefix + "permuted_qy"],
        }
        per_seed = {
            name: _per_seed_qy(
                values, target_qy, aggregate["conditions"], aggregate["identities"]
            )
            for name, values in methods.items()
        }
        aggregate_metrics = {
            name: _mean_seed_metric(values) for name, values in per_seed.items()
        }
        ratios = {
            name: {
                condition: value / max(raw10_aggregate[condition], EPS)
                for condition, value in records.items()
            }
            for name, records in aggregate_metrics.items()
        }
        primary_seed_ratios = {
            seed: value / max(raw10_per_seed[PRIMARY_CONDITION][seed], EPS)
            for seed, value in per_seed["selected"][PRIMARY_CONDITION].items()
        }
        dc_error = float(
            np.max(
                np.abs(
                    np.mean(methods["selected"], axis=(-2, -1))
                    - np.mean(methods["raw"], axis=(-2, -1))
                )
            )
        )
        budget_gates = {
            "primary_no_worse_than_Raw_B10": ratios["selected"][PRIMARY_CONDITION]
            <= float(protocol["analysis_contract"]["maximum_primary_ratio_to_Raw_B10"]),
            "every_primary_seed_within_cap": max(primary_seed_ratios.values())
            <= float(protocol["analysis_contract"]["maximum_primary_seed_ratio_to_Raw_B10"]),
            "no_condition_mean_worse_than_Raw_B10": max(ratios["selected"].values())
            <= float(protocol["analysis_contract"]["maximum_each_condition_ratio_to_Raw_B10"]),
            "beats_same_budget_vision_primary": aggregate_metrics["selected"][PRIMARY_CONDITION]
            < aggregate_metrics["vision"][PRIMARY_CONDITION],
            "beats_same_budget_TSVD_primary": aggregate_metrics["selected"][PRIMARY_CONDITION]
            < aggregate_metrics["tsvd"][PRIMARY_CONDITION],
            "beats_same_budget_raw_primary": aggregate_metrics["selected"][PRIMARY_CONDITION]
            < aggregate_metrics["raw"][PRIMARY_CONDITION],
            "permuted_observation_degrades": aggregate_metrics["permuted"][PRIMARY_CONDITION]
            >= (
                1.0
                + float(protocol["analysis_contract"]["permuted_minimum_degradation_fraction"])
            )
            * aggregate_metrics["selected"][PRIMARY_CONDITION],
            "DC_preserved_to_tolerance": dc_error
            <= float(protocol["analysis_contract"]["maximum_DC_absolute_error"]),
        }
        for method, condition_records in aggregate_metrics.items():
            for condition, value in condition_records.items():
                metrics_rows.append(
                    (budget, condition, method, value, ratios[method][condition])
                )
        decomposition = {
            method: _mv15a_module().exact_affine_error_decomposition(
                values[aggregate["conditions"] == PRIMARY_CONDITION],
                target_qy[aggregate["conditions"] == PRIMARY_CONDITION],
            )
            for method, values in methods.items()
        }
        budget_results[str(budget)] = {
            "selected_development_configuration": selection["selected_by_budget"][str(budget)],
            "mean_seed_qy_nrmse": aggregate_metrics,
            "ratios_to_Raw_B10": ratios,
            "primary_selected_per_seed_ratios_to_Raw_B10": primary_seed_ratios,
            "primary_exact_error_decomposition": decomposition,
            "maximum_DC_absolute_error": dc_error,
            "gates": budget_gates,
            "all_gates_pass": all(budget_gates.values()),
        }
    with (output_root / "mv15b_legacy_budget_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(("budget", "condition", "method", "mean_seed_qy_nrmse", "ratio_to_Raw_B10"))
        writer.writerows(metrics_rows)
    eligible = [budget for budget in BUDGETS if budget_results[str(budget)]["all_gates_pass"]]
    recommended_budget = min(eligible) if eligible else None
    diagnostic_budget = (
        recommended_budget
        if recommended_budget is not None
        else min(
            BUDGETS,
            key=lambda budget: budget_results[str(budget)]["ratios_to_Raw_B10"]["selected"][PRIMARY_CONDITION],
        )
    )
    diagnostic = aggregate_disjoint(
        test_x, test_conditions, test_identities, diagnostic_budget, test_y
    )
    prefix = f"b{diagnostic_budget}_"
    representative = (
        (diagnostic["conditions"] == PRIMARY_CONDITION)
        & (diagnostic["identities"][:, 0] == REPRESENTATIVE_SEED)
        & (diagnostic["identities"][:, 1] == REPRESENTATIVE_GROUP)
    )
    raw10_mask = (conditions10 == PRIMARY_CONDITION) & (identities10[:, 0] == REPRESENTATIVE_SEED)
    if np.count_nonzero(representative) != 1 or np.count_nonzero(raw10_mask) != 1:
        raise ValueError("MV15B representative identity is absent")
    index = int(np.flatnonzero(representative)[0])
    index10 = int(np.flatnonzero(raw10_mask)[0])
    source_indices = np.flatnonzero(
        (test_conditions == PRIMARY_CONDITION)
        & (test_identities[:, 0] == REPRESENTATIVE_SEED)
    )
    scale = float(test_scales[source_indices[0], QY_INDEX])
    figures = []
    figures.extend(_plot_budget_curves(output_root, budget_results))
    figures.extend(
        _plot_trust_map(
            output_root,
            locked[prefix + "reliability_mode"],
            locked[prefix + "selected_weight_map"],
            diagnostic_budget,
        )
    )
    figures.extend(
        _plot_contours(
            output_root,
            {
                "raw": locked[prefix + "raw_qy"][index],
                "vision": locked[prefix + "vision_qy"][index],
                "dc_only": locked[prefix + "dc_only_qy"][index],
                "selected": locked[prefix + "selected_qy"][index],
                "tsvd": locked[prefix + "tsvd_qy"][index],
                "raw10": raw10[index10, QY_INDEX],
            },
            diagnostic["targets"][index, QY_INDEX],
            scale,
            diagnostic_budget,
        )
    )
    integrity = {
        "prediction_hash_locked_before_legacy_label_access": True,
        "all_budget_groups_are_disjoint": True,
        "B3_unused_block_explicitly_recorded": True,
        "legacy_results_are_not_confirmation": True,
        "fresh_seed_and_fresh_condition_still_required": True,
    }
    supports_fresh = recommended_budget is not None and all(integrity.values())
    summary = {
        "stage": STAGE,
        "status": "complete_MV15B_post_lock_legacy_budget_diagnostic",
        "protocol_sha256": _sha256(protocol_path()),
        "primary_condition": PRIMARY_CONDITION,
        "budget_results": budget_results,
        "raw_B10_mean_seed_qy_nrmse": raw10_aggregate,
        "recommended_budget_for_separately_locked_fresh_confirmation": recommended_budget,
        "diagnostic_best_budget": diagnostic_budget,
        "integrity_gates": integrity,
        "figures": figures,
        "same_condition_mean_is_the_reference_construction_and_cannot_be_a_claim": True,
        "old_evaluation_seeds_are_confirmation": False,
        "decision": (
            "MV15B_legacy_diagnostic_authorizes_separately_locked_fresh_seed_and_condition_confirmation"
            if supports_fresh
            else "MV15B_legacy_diagnostic_does_not_authorize_fresh_DSMC"
        ),
    }
    _atomic_json(output_root / "summary.json", summary)
    return summary


def package_results(output_root: Path, return_directory: Path) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    return_directory = Path(return_directory).resolve()
    _verify_manifest(output_root, "prediction_manifest.json")
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    names = [
        "selection_summary.json",
        "prediction_manifest.json",
        "summary.json",
        PROTOCOL_FILE,
        "mv15b_development_selection.csv",
        "mv15b_legacy_budget_metrics.csv",
        "mv15b_qy_disjoint_budget_ladder.png",
        "mv15b_qy_disjoint_budget_ladder.pdf",
        f"mv15b_qy_modewise_trust_B{summary['diagnostic_best_budget']}.png",
        f"mv15b_qy_modewise_trust_B{summary['diagnostic_best_budget']}.pdf",
        f"mv15b_qy_physical_contours_B{summary['diagnostic_best_budget']}.png",
        f"mv15b_qy_physical_contours_B{summary['diagnostic_best_budget']}.pdf",
    ]
    accounting = output_root / "slurm_accounting.psv"
    if accounting.is_file():
        names.append(accounting.name)
    generated = [output_root / name for name in names]
    for path in generated:
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = {
        "stage": STAGE,
        "status": "complete_MV15B_return_artifact_manifest",
        "files": {
            path.name: {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
            for path in generated
        },
    }
    _atomic_json(output_root / "artifact_manifest.json", manifest)
    _verify_manifest(output_root, "artifact_manifest.json")
    verification = {
        "stage": STAGE,
        "status": "complete_MV15B_recursive_return_verification",
        "decision": "verified",
        "verified_file_count": len(manifest["files"]),
        "manifest_sha256": _sha256(output_root / "artifact_manifest.json"),
    }
    _atomic_json(output_root / "verification.json", verification)
    generated.extend((output_root / "artifact_manifest.json", output_root / "verification.json"))
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return_directory.mkdir(parents=True, exist_ok=True)
    archive = return_directory / f"MV15B_DATA_CONSISTENT_BUDGET_BUNDLE_{tag}.zip"
    if archive.exists():
        raise FileExistsError(f"refusing to overwrite MV15B archive: {archive}")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as stream:
        for path in generated:
            stream.write(path, arcname=path.name)
    if archive.stat().st_size > 450 * 1024 * 1024:
        raise RuntimeError("MV15B return archive exceeds 450 MiB")
    result = {
        "stage": STAGE,
        "archive": str(archive),
        "archive_sha256": _sha256(archive),
        "decision": summary["decision"],
        "recommended_budget": summary[
            "recommended_budget_for_separately_locked_fresh_confirmation"
        ],
        "diagnostic_best_budget": summary["diagnostic_best_budget"],
    }
    _atomic_json(output_root / "return.json", result)
    pointer = return_directory / "LAST_MOHAMMADZADEH_MV15B_DATA_CONSISTENT_BUDGET_RESULT.env"
    recommended = "none" if result["recommended_budget"] is None else str(result["recommended_budget"])
    pointer.write_text(
        "\n".join(
            (
                f"MV15B_OUTPUT_ROOT={output_root}",
                f"MV15B_RESULT_ARCHIVE={archive}",
                f"MV15B_RESULT_ARCHIVE_SHA256={result['archive_sha256']}",
                f"MV15B_DECISION={result['decision']}",
                f"MV15B_RECOMMENDED_BUDGET={recommended}",
                f"MV15B_DIAGNOSTIC_BEST_BUDGET={result['diagnostic_best_budget']}",
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
    verify = subparsers.add_parser("verify-data")
    verify.add_argument("--mv9-output-root", type=Path, required=True)
    verify.add_argument("--mv15a-output-root", type=Path, required=True)
    predict = subparsers.add_parser("predict")
    predict.add_argument("--mv9-output-root", type=Path, required=True)
    predict.add_argument("--mv15a-output-root", type=Path, required=True)
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
    elif args.command == "verify-data":
        result = verify_data_contract(args.mv9_output_root, args.mv15a_output_root)
    elif args.command == "predict":
        result = run_prediction_stage(
            args.mv9_output_root,
            args.mv15a_output_root,
            args.output_root,
            batch_size=args.batch_size,
        )
    elif args.command == "post":
        result = run_legacy_post(args.output_root)
    else:
        result = package_results(args.output_root, args.return_directory)
    print(_json_dumps(result))


if __name__ == "__main__":
    main()
