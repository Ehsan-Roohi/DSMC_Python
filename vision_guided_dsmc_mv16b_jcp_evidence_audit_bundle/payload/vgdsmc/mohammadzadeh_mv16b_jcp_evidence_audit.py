"""Post-confirmation JCP evidence audit for cavity q_y and cylinder transfer.

MV16B is deliberately an analysis-only stage.  It launches neither DSMC nor
training and it never changes the frozen MV15B/MV15C estimator.  The cavity
part quantifies attribution, reference-noise sensitivity, stationarity,
spectral localisation, and paired small-sample uncertainty.  The cylinder
part replaces the rectangular masked DCT used by the engineering MV16A screen
with an area-weighted basis evaluated only at native DS2V fluid-cell centres.

The stage is retrospective.  Its outputs can support a JCP evidence chain but
must not be described as a new preregistered confirmation.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np


STAGE = "MV16B_Mohammadzadeh_JCP_evidence_audit"
STATUS = "locked_analysis_only_after_MV15C_A1_and_MV16A"
PROTOCOL_FILE = "mv16b_jcp_evidence_audit_protocol.json"
RESULT_POINTER = "LAST_MOHAMMADZADEH_MV16B_JCP_EVIDENCE_RESULT.env"
QX_INDEX = 2
QY_INDEX = 3
EPS = 1.0e-12
CAVITY_CONDITIONS = ("kn0p08_u350", "kn0p1_u400")
CYLINDER_SEEDS = (20260813, 32452843, 49979687, 67867967)
SPECTRAL_BANDS = (("low", 0.0, 0.10), ("mid", 0.10, 0.35), ("high", 0.35, math.inf))


def _mv9_module():
    from . import mohammadzadeh_mv9_heat_flux as mv9

    return mv9


def _mv14_module():
    from . import mohammadzadeh_mv14_kinetic_conservation_cavity as mv14

    return mv14


def _mv15c_module():
    from . import mohammadzadeh_mv15c_fresh_b3_confirmation as mv15c

    return mv15c


def _mv16a_module():
    from . import mohammadzadeh_mv16a_frozen_cylinder_transfer as mv16a

    return mv16a


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=_json_default)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_json_dumps(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def _verify_manifest(root: Path, name: str) -> dict[str, Any]:
    root = Path(root)
    path = root / name
    value = json.loads(path.read_text(encoding="utf-8"))
    for relative, record in value["files"].items():
        candidate = root / relative
        if (
            not candidate.is_file()
            or candidate.stat().st_size != int(record["size_bytes"])
            or _sha256(candidate) != record["sha256"]
        ):
            raise ValueError(f"recursive artifact verification failed: {candidate}")
    return value


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


def protocol_path() -> Path:
    value = (
        Path(__file__).resolve().parents[1]
        / "reference_data"
        / "mohammadzadeh_2012"
        / PROTOCOL_FILE
    )
    if not value.is_file():
        raise FileNotFoundError(value)
    return value


def locked_protocol() -> dict[str, Any]:
    value = json.loads(protocol_path().read_text(encoding="utf-8"))
    if value.get("stage") != STAGE or value.get("status") != STATUS:
        raise ValueError("MV16B protocol is absent or unlocked")
    if tuple(value["cavity_contract"]["conditions"]) != CAVITY_CONDITIONS:
        raise ValueError("MV16B cavity condition contract changed")
    if tuple(int(v) for v in value["cylinder_contract"]["seeds"]) != CYLINDER_SEEDS:
        raise ValueError("MV16B cylinder seed contract changed")
    return value


def verify_contract() -> dict[str, Any]:
    value = locked_protocol()
    return {
        "stage": STAGE,
        "status": "MV16B_contract_verified",
        "protocol_sha256": _sha256(protocol_path()),
        "DSMC_rerun": False,
        "neural_training": False,
        "fresh_parameter_selection": False,
        "classification": value["scientific_classification"],
        "cavity_conditions": list(CAVITY_CONDITIONS),
        "cylinder_seeds": list(CYLINDER_SEEDS),
    }


def _dct2(array: np.ndarray) -> np.ndarray:
    from scipy.fft import dctn

    return dctn(np.asarray(array, dtype=np.float64), axes=(-2, -1), type=2, norm="ortho")


def _idct2(array: np.ndarray) -> np.ndarray:
    from scipy.fft import idctn

    return idctn(np.asarray(array, dtype=np.float64), axes=(-2, -1), type=2, norm="ortho")


def data_consistent_residual(
    raw: np.ndarray, prior: np.ndarray, weight: np.ndarray
) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    prior = np.asarray(prior, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    if raw.shape != prior.shape or raw.shape[-2:] != weight.shape:
        raise ValueError("incompatible data-consistency shapes")
    result = _idct2(_dct2(prior) + weight * (_dct2(raw) - _dct2(prior)))
    result += np.mean(raw, axis=(-2, -1), keepdims=True) - np.mean(
        result, axis=(-2, -1), keepdims=True
    )
    return result


def continuous_wiener_gain(signal: np.ndarray, noise_b1: np.ndarray, budget: int) -> np.ndarray:
    signal = np.maximum(np.asarray(signal, dtype=np.float64), 0.0)
    noise = np.maximum(np.asarray(noise_b1, dtype=np.float64), 0.0) / float(budget)
    if signal.shape != noise.shape:
        raise ValueError("signal/noise spectral shapes differ")
    gain = signal / np.maximum(signal + noise, EPS)
    gain[0, 0] = 1.0
    return np.clip(gain, 0.0, 1.0)


def pure_wiener(raw: np.ndarray, gain: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    result = _idct2(_dct2(raw) * np.asarray(gain, dtype=np.float64))
    result += np.mean(raw, axis=(-2, -1), keepdims=True) - np.mean(
        result, axis=(-2, -1), keepdims=True
    )
    return result


_CONDITION_RE = re.compile(r"^kn([0-9]+p[0-9]+)_u([0-9]+)$")


def condition_coordinates(condition: str) -> tuple[float, float]:
    match = _CONDITION_RE.match(str(condition))
    if match is None:
        raise ValueError(f"invalid condition identifier: {condition}")
    kn = float(match.group(1).replace("p", "."))
    speed = float(match.group(2))
    return math.log10(kn), speed / 100.0


def _normalise_strings(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value) for value in values],
        dtype="U64",
    )


def development_condition_prior(
    validation_y: np.ndarray,
    validation_conditions: np.ndarray,
    requested_conditions: Sequence[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a frozen per-pixel condition surface using development labels only."""

    values = np.asarray(validation_y, dtype=np.float64)
    labels = _normalise_strings(validation_conditions)
    if values.ndim != 4 or values.shape[1] <= QY_INDEX or len(values) != len(labels):
        raise ValueError("invalid MV9 development arrays")
    unique = sorted(np.unique(labels).tolist())
    condition_means = np.stack(
        [np.mean(values[labels == condition, QY_INDEX], axis=0, dtype=np.float64) for condition in unique]
    )
    coordinates = np.asarray([condition_coordinates(condition) for condition in unique])
    full = np.column_stack(
        (
            np.ones(len(unique)),
            coordinates[:, 0],
            coordinates[:, 1],
            coordinates[:, 0] * coordinates[:, 1],
        )
    )
    feature_names = ["intercept", "log10_Kn", "U_over_100", "interaction"]
    if np.linalg.matrix_rank(full) < full.shape[1]:
        full = full[:, :3]
        feature_names = feature_names[:3]
    if np.linalg.matrix_rank(full) < full.shape[1]:
        raise ValueError("development conditions cannot identify the frozen prior surface")
    coefficients = np.linalg.lstsq(full, condition_means.reshape(len(unique), -1), rcond=None)[0]
    requested = []
    for condition in requested_conditions:
        log_kn, speed = condition_coordinates(condition)
        row = np.asarray([1.0, log_kn, speed, log_kn * speed])[: full.shape[1]]
        requested.append((row @ coefficients).reshape(condition_means.shape[1:]))
    fitted = (full @ coefficients).reshape(condition_means.shape)
    residual = float(np.sqrt(np.mean((fitted - condition_means) ** 2)))
    return np.asarray(requested), {
        "development_conditions": unique,
        "feature_names": feature_names,
        "design_rank": int(np.linalg.matrix_rank(full)),
        "condition_surface_RMSE": residual,
        "fresh_labels_used": False,
    }


def nrmse(candidate: np.ndarray, target: np.ndarray, weights: np.ndarray | None = None) -> float:
    candidate = np.asarray(candidate, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    difference2 = (candidate - target) ** 2
    target2 = target**2
    if weights is None:
        numerator = float(np.mean(difference2))
        denominator = float(np.mean(target2))
    else:
        weight = np.asarray(weights, dtype=np.float64)
        weight = np.broadcast_to(weight, candidate.shape)
        denominator_weight = max(float(np.sum(weight)), EPS)
        numerator = float(np.sum(weight * difference2) / denominator_weight)
        denominator = float(np.sum(weight * target2) / denominator_weight)
    return math.sqrt(numerator) / max(math.sqrt(denominator), EPS)


def leave_one_seed_out(
    fields: np.ndarray, conditions: np.ndarray | None = None
) -> np.ndarray:
    fields = np.asarray(fields, dtype=np.float64)
    if conditions is None:
        if fields.shape[0] != 4:
            raise ValueError("unconditioned leave-one-out requires four seeds")
        return np.stack([np.mean(np.delete(fields, i, axis=0), axis=0) for i in range(4)])
    labels = _normalise_strings(conditions)
    output = np.empty_like(fields)
    for index, label in enumerate(labels):
        peers = (labels == label) & (np.arange(len(labels)) != index)
        if np.count_nonzero(peers) != 3:
            raise ValueError(f"condition {label} does not have exactly three peers")
        output[index] = np.mean(fields[peers], axis=0, dtype=np.float64)
    return output


def grouped_metrics(
    methods: Mapping[str, np.ndarray], target: np.ndarray, conditions: np.ndarray
) -> tuple[dict[str, dict[str, list[float]]], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    labels = _normalise_strings(conditions)
    per_seed: dict[str, dict[str, list[float]]] = {}
    means: dict[str, dict[str, float]] = {}
    ratios: dict[str, dict[str, float]] = {}
    for name, array in methods.items():
        per_seed[name], means[name] = {}, {}
        for condition in sorted(np.unique(labels)):
            indices = np.flatnonzero(labels == condition)
            records = [nrmse(array[index], target[index]) for index in indices]
            per_seed[name][condition] = records
            means[name][condition] = float(np.mean(records))
    for name in methods:
        ratios[name] = {
            condition: means[name][condition] / max(means["raw_b10"][condition], EPS)
            for condition in means[name]
        }
    return per_seed, means, ratios


def reference_noise_corrected_ratio(observed_ratio: float) -> float:
    """Equal-variance correction for a candidate compared with a 3-seed LOO target."""

    return math.sqrt(max((float(observed_ratio) ** 2 - 0.25) / 0.75, 0.0))


def paired_log_ratio_statistics(method: Sequence[float], baseline: Sequence[float]) -> dict[str, Any]:
    from scipy.stats import t

    method_array = np.asarray(method, dtype=np.float64)
    baseline_array = np.asarray(baseline, dtype=np.float64)
    if method_array.shape != baseline_array.shape or method_array.ndim != 1 or len(method_array) < 2:
        raise ValueError("paired log-ratio inputs are incompatible")
    logs = np.log(np.maximum(method_array, EPS) / np.maximum(baseline_array, EPS))
    mean = float(np.mean(logs))
    standard_error = float(np.std(logs, ddof=1) / math.sqrt(len(logs)))
    critical = float(t.ppf(0.975, len(logs) - 1))
    samples = []
    for indices in itertools.product(range(len(logs)), repeat=len(logs)):
        samples.append(float(np.mean(logs[list(indices)])))
    low, high = np.quantile(samples, [0.025, 0.975])
    improvements = int(np.count_nonzero(logs < 0.0))
    from math import comb

    one_sided = sum(comb(len(logs), k) for k in range(improvements, len(logs) + 1)) / 2 ** len(logs)
    two_sided = min(1.0, 2.0 * one_sided)
    return {
        "n_pairs": len(logs),
        "per_pair_log_ratios": logs.tolist(),
        "geometric_mean_ratio": math.exp(mean),
        "t95_geometric_CI": [math.exp(mean - critical * standard_error), math.exp(mean + critical * standard_error)],
        "exact_bootstrap95_geometric_CI": [math.exp(float(low)), math.exp(float(high))],
        "improved_pair_count": improvements,
        "exact_sign_test_one_sided_p": float(one_sided),
        "exact_sign_test_two_sided_p": float(two_sided),
        "minimum_attainable_one_sided_p_at_n": 1.0 / 2 ** len(logs),
    }


def dersimonian_laird(condition_effects: Sequence[tuple[float, float]]) -> dict[str, float]:
    effects = np.asarray([item[0] for item in condition_effects], dtype=np.float64)
    variances = np.maximum(np.asarray([item[1] for item in condition_effects]), EPS)
    fixed_weight = 1.0 / variances
    fixed = float(np.sum(fixed_weight * effects) / np.sum(fixed_weight))
    q = float(np.sum(fixed_weight * (effects - fixed) ** 2))
    c = float(np.sum(fixed_weight) - np.sum(fixed_weight**2) / np.sum(fixed_weight))
    tau2 = max((q - (len(effects) - 1)) / max(c, EPS), 0.0)
    random_weight = 1.0 / (variances + tau2)
    mean = float(np.sum(random_weight * effects) / np.sum(random_weight))
    standard_error = math.sqrt(1.0 / float(np.sum(random_weight)))
    return {
        "log_effect": mean,
        "geometric_mean_ratio": math.exp(mean),
        "tau_squared": tau2,
        "normal95_geometric_CI_low": math.exp(mean - 1.96 * standard_error),
        "normal95_geometric_CI_high": math.exp(mean + 1.96 * standard_error),
    }


def spectral_band_metrics(candidate: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = _dct2(np.asarray(candidate) - np.asarray(target))
    reference = _dct2(target)
    ny, nx = error.shape[-2:]
    yy, xx = np.meshgrid(np.arange(ny) / max(ny - 1, 1), np.arange(nx) / max(nx - 1, 1), indexing="ij")
    radius = np.sqrt(xx * xx + yy * yy) / math.sqrt(2.0)
    result = {}
    for name, lower, upper in SPECTRAL_BANDS:
        mask = (radius >= lower) & (radius < upper)
        numerator = float(np.mean(np.abs(error[..., mask]) ** 2))
        denominator = float(np.mean(np.abs(reference[..., mask]) ** 2))
        result[name] = math.sqrt(numerator) / max(math.sqrt(denominator), EPS)
    return result


def _write_rows(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _cavity_sources(output: Path, conditions: np.ndarray, seeds: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    mv9 = _mv9_module()
    blocks, records = [], []
    for condition, seed in zip(_normalise_strings(conditions), seeds):
        directory = output / "references" / str(condition) / f"seed_{int(seed)}"
        source = mv9.load_moment_source(directory)
        value = np.asarray(source["blocks"], dtype=np.float64)
        if value.shape[0] != 10 or value.shape[1] <= QY_INDEX:
            raise ValueError(f"invalid fresh source blocks: {directory}")
        blocks.append(value[:, QY_INDEX])
        rms = np.sqrt(np.mean(value[:, QY_INDEX] ** 2, axis=(-2, -1)))
        slope = float(np.polyfit(np.arange(10), rms, 1)[0] / max(float(np.mean(rms)), EPS))
        first, second = np.mean(value[:5, QY_INDEX], axis=0), np.mean(value[5:, QY_INDEX], axis=0)
        records.append(
            {
                "condition": str(condition),
                "seed": int(seed),
                "relative_qy_RMS_slope_per_block": slope,
                "first_to_second_half_field_shift_NRMSE": nrmse(first, second),
                "first_half_qy_RMS": math.sqrt(float(np.mean(first**2))),
                "second_half_qy_RMS": math.sqrt(float(np.mean(second**2))),
            }
        )
    return np.asarray(blocks), records


def analyze_cavity(mv15c_root: Path, output: Path) -> dict[str, Any]:
    root = Path(mv15c_root).resolve()
    _verify_manifest(root, "prediction_manifest.json")
    with np.load(root / "locked_fresh_predictions.npz", allow_pickle=False) as source:
        locked = {name: np.asarray(source[name]) for name in source.files}
    conditions = _normalise_strings(locked["conditions"])
    seeds = locked["seeds"].astype(np.int64)
    if set(conditions.tolist()) != set(CAVITY_CONDITIONS) or len(seeds) != 8:
        raise ValueError("MV16B requires the eight MV15C-A1 fresh trajectories")
    submission = json.loads((root / "submission_lock.json").read_text(encoding="utf-8"))
    mv9_root = Path(submission["mv9_output_root"])
    mv15b_root = Path(submission["mv15b_output_root"])
    with np.load(mv9_root / "dataset.npz", allow_pickle=False) as data:
        development_prior, prior_audit = development_condition_prior(
            np.asarray(data["validation_y"]),
            np.asarray(data["validation_condition"]),
            conditions,
        )
    with np.load(mv15b_root / "locked_predictions.npz", allow_pickle=False) as data:
        signal = np.asarray(data["global_signal_mode"], dtype=np.float64)
        noise = np.asarray(data["global_noise_mode_B1"], dtype=np.float64)
    frozen_weight = np.asarray(locked["frozen_weight_map"], dtype=np.float64)
    raw_b3 = np.asarray(locked["raw_b3_qy"], dtype=np.float64)
    vision = np.asarray(locked["vision_b3_qy"], dtype=np.float64)
    gain = continuous_wiener_gain(signal, noise, 3)
    methods = {
        "raw_b3": raw_b3,
        "vision_b3": vision,
        "selected_b3": np.asarray(locked["selected_b3_qy"], dtype=np.float64),
        "dc_only_b3": np.asarray(locked["dc_only_b3_qy"], dtype=np.float64),
        "tsvd_b3": np.asarray(locked["tsvd_b3_qy"], dtype=np.float64),
        "permuted_b3": np.asarray(locked["permuted_b3_qy"], dtype=np.float64),
        "raw_b10": np.asarray(locked["raw_b10_qy"], dtype=np.float64),
        "development_prior_only": development_prior,
        "development_prior_plus_frozen_DCIR": data_consistent_residual(raw_b3, development_prior, frozen_weight),
        "pure_continuous_Wiener_B3": pure_wiener(raw_b3, gain),
        "Mamba_plus_continuous_Wiener_residual": data_consistent_residual(raw_b3, vision, gain),
    }
    target = leave_one_seed_out(methods["raw_b10"], conditions)
    per_seed, means, ratios = grouped_metrics(methods, target, conditions)
    corrected = {
        name: {condition: reference_noise_corrected_ratio(value) for condition, value in record.items()}
        for name, record in ratios.items()
    }
    blocks, stationarity = _cavity_sources(root, conditions, seeds)
    half_results = {}
    for half_name, selection in (("blocks_0_4", slice(0, 5)), ("blocks_5_9", slice(5, 10))):
        half_raw = np.mean(blocks[:, selection], axis=1, dtype=np.float64)
        half_target = leave_one_seed_out(half_raw, conditions)
        _, half_means, half_ratios = grouped_metrics({**methods, "raw_b10": methods["raw_b10"]}, half_target, conditions)
        half_results[half_name] = {"mean_nrmse": half_means, "ratios_to_Raw_B10": half_ratios}
    budget_rows = []
    for budget in (1, 2, 5):
        for index, (condition, seed) in enumerate(zip(conditions, seeds)):
            group_count = 10 // budget
            own_full = np.mean(blocks[index], axis=0, dtype=np.float64)
            errors = [
                nrmse(np.mean(blocks[index, group * budget : (group + 1) * budget], axis=0), own_full)
                for group in range(group_count)
            ]
            budget_rows.append(
                {
                    "condition": str(condition),
                    "seed": int(seed),
                    "budget": budget,
                    "mean_within_trajectory_NRMSE_to_B10": float(np.mean(errors)),
                    "sqrt_B_scaled_NRMSE": float(math.sqrt(budget) * np.mean(errors)),
                }
            )
    _write_rows(
        output / "cavity_stationarity.csv",
        list(stationarity[0]),
        stationarity,
    )
    _write_rows(
        output / "cavity_budget_scaling.csv",
        list(budget_rows[0]),
        budget_rows,
    )
    paired = {}
    random_effect_inputs = []
    for condition in CAVITY_CONDITIONS:
        selected_values = per_seed["selected_b3"][condition]
        baseline_values = per_seed["raw_b10"][condition]
        paired[condition] = paired_log_ratio_statistics(selected_values, baseline_values)
        logs = np.log(np.maximum(selected_values, EPS) / np.maximum(baseline_values, EPS))
        random_effect_inputs.append((float(np.mean(logs)), float(np.var(logs, ddof=1) / len(logs))))
    random_effect = dersimonian_laird(random_effect_inputs)
    top_start = int(round(0.75 * target.shape[-2]))
    top_ratios = {}
    for name, array in methods.items():
        top_ratios[name] = {}
        for condition in CAVITY_CONDITIONS:
            indices = np.flatnonzero(conditions == condition)
            candidate_error = np.mean([nrmse(array[i, top_start:], target[i, top_start:]) for i in indices])
            raw10_error = np.mean([nrmse(methods["raw_b10"][i, top_start:], target[i, top_start:]) for i in indices])
            top_ratios[name][condition] = float(candidate_error / max(raw10_error, EPS))
    spectral = {}
    for name, array in methods.items():
        spectral[name] = {}
        for condition in CAVITY_CONDITIONS:
            indices = np.flatnonzero(conditions == condition)
            records = [spectral_band_metrics(array[i], target[i]) for i in indices]
            spectral[name][condition] = {
                band: float(np.mean([record[band] for record in records])) for band, _, _ in SPECTRAL_BANDS
            }
    figures = _plot_cavity(output, methods, target, conditions, seeds, locked["q_ref_scales"])
    summary = {
        "classification": "retrospective_post_confirmation_attribution_and_uncertainty_audit",
        "conditions": list(CAVITY_CONDITIONS),
        "mean_qy_nrmse": means,
        "ratios_to_Raw_B10": ratios,
        "reference_noise_corrected_ratios": corrected,
        "paired_selected_vs_Raw_B10": paired,
        "two_condition_random_effects_selected_vs_Raw_B10": random_effect,
        "top_quarter_ratios_to_Raw_B10": top_ratios,
        "spectral_band_nrmse": spectral,
        "reference_half_sensitivity": half_results,
        "development_prior_audit": prior_audit,
        "continuous_Wiener_gain": {
            "minimum": float(np.min(gain)),
            "maximum_non_DC": float(np.max(gain.reshape(-1)[1:])),
            "mean_non_DC": float(np.mean(gain.reshape(-1)[1:])),
            "DC": float(gain[0, 0]),
        },
        "frozen_binary_weight_nonzero_count": int(np.count_nonzero(frozen_weight)),
        "fresh_outcomes_used_for_parameter_selection": False,
        "figures": figures,
    }
    _atomic_json(output / "cavity_summary.json", summary)
    np.savez_compressed(
        output / "cavity_audit_fields.npz",
        conditions=conditions,
        seeds=seeds,
        target_qy=target,
        continuous_wiener_gain=gain,
        **{f"method_{name}": value for name, value in methods.items()},
    )
    return summary


def _plot_cavity(
    output: Path,
    methods: Mapping[str, np.ndarray],
    target: np.ndarray,
    conditions: np.ndarray,
    seeds: np.ndarray,
    scales: np.ndarray,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = (
        "reference",
        "raw_b3",
        "vision_b3",
        "selected_b3",
        "development_prior_plus_frozen_DCIR",
        "Mamba_plus_continuous_Wiener_residual",
    )
    titles = ("Reference", "Raw DSMC B=3", "MambaIRv2 B=3", "Frozen DCIR-QY B=3", "Dev-prior + same DCIR", "Continuous Wiener residual")
    paths = []
    for condition in CAVITY_CONDITIONS:
        index = int(np.flatnonzero(conditions == condition)[0])
        arrays = {"reference": target[index], **{name: value[index] for name, value in methods.items()}}
        physical = [arrays[name] * float(scales[index]) for name in order]
        limit = max(float(np.max(np.abs(value))) for value in physical)
        scale = max(math.sqrt(float(np.mean(target[index] ** 2))), EPS)
        errors = [100.0 * (arrays[name] - target[index]) / scale for name in order[1:]]
        error_limit = max(float(np.quantile(np.abs(np.concatenate([v.ravel() for v in errors])), 0.995)), 1.0)
        fig, axes = plt.subplots(2, 6, figsize=(18.5, 6.8), sharex=True, sharey=True, constrained_layout=True)
        field_image = error_image = None
        for column, (name, title, field) in enumerate(zip(order, titles, physical)):
            field_image = axes[0, column].imshow(field, origin="lower", extent=(0, 1, 0, 1), cmap="RdBu_r", vmin=-limit, vmax=limit)
            axes[0, column].set_title(title, fontsize=10)
            if column == 0:
                axes[1, column].set_facecolor("0.94")
                axes[1, column].text(0.5, 0.5, "Reference", ha="center", va="center", color="0.45")
            else:
                error_image = axes[1, column].imshow(errors[column - 1], origin="lower", extent=(0, 1, 0, 1), cmap="RdBu_r", vmin=-error_limit, vmax=error_limit)
            axes[1, column].set_xlabel("$x/L$")
            for row in range(2):
                axes[row, column].set_aspect("equal")
        axes[0, 0].set_ylabel("$y/L$")
        axes[1, 0].set_ylabel("$y/L$")
        assert field_image is not None and error_image is not None
        fig.colorbar(field_image, ax=axes[0, :], shrink=0.82, label=r"$q_y$ [W m$^{-2}$]")
        fig.colorbar(error_image, ax=axes[1, :], shrink=0.82, label=r"$100\Delta q_y/\mathrm{RMS}(q_{y,ref})$ [%]")
        fig.suptitle(f"{condition}, seed {int(seeds[index])}: frozen post-confirmation attribution")
        for suffix in ("png", "pdf"):
            path = output / f"mv16b_cavity_qy_six_panel_{condition}.{suffix}"
            fig.savefig(path, dpi=500 if suffix == "png" else None, bbox_inches="tight")
            paths.append(path.name)
        plt.close(fig)
    return paths


def _interpolate_raster_to_cells(raster: np.ndarray, x: np.ndarray, y: np.ndarray, mv16a: Any) -> np.ndarray:
    from scipy.interpolate import RegularGridInterpolator

    raster = np.asarray(raster, dtype=np.float64)
    xmin, xmax, ymin, ymax = mv16a.DOMAIN
    grid_x = np.linspace(xmin, xmax, raster.shape[-1])
    grid_y = np.linspace(ymin, ymax, raster.shape[-2])
    interpolator = RegularGridInterpolator((grid_y, grid_x), raster, method="linear", bounds_error=False, fill_value=None)
    result = interpolator(np.column_stack((y, x)))
    if not np.isfinite(result).all():
        raise ValueError("nonfinite raster-to-cell interpolation")
    return np.asarray(result, dtype=np.float64)


def _map_peer_to_points(
    peer_x: np.ndarray,
    peer_y: np.ndarray,
    peer_values: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
) -> tuple[np.ndarray, float]:
    if (
        peer_values.shape[0] == target_x.shape[0]
        and np.allclose(peer_x, target_x, rtol=0.0, atol=2e-10)
        and np.allclose(peer_y, target_y, rtol=0.0, atol=2e-10)
    ):
        return np.asarray(peer_values, dtype=np.float64), 1.0
    from scipy.interpolate import griddata

    points = np.column_stack((peer_x, peer_y))
    requested = np.column_stack((target_x, target_y))
    mapped = griddata(points, peer_values, requested, method="linear")
    coverage = float(np.mean(np.isfinite(mapped)))
    if not np.all(np.isfinite(mapped)):
        nearest = griddata(points, peer_values, requested, method="nearest")
        mapped = np.where(np.isfinite(mapped), mapped, nearest)
    if not np.isfinite(mapped).all():
        raise ValueError("peer-to-native-grid interpolation failed")
    return np.asarray(mapped), coverage


def masked_mode_matrix(
    x: np.ndarray,
    y: np.ndarray,
    area: np.ndarray,
    frozen_weight: np.ndarray,
    domain: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Evaluate frozen rectangular modes only on native fluid cells."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    area = np.asarray(area, dtype=np.float64)
    weight = np.asarray(frozen_weight, dtype=np.float64)
    if len(x) != len(y) or len(x) != len(area) or np.any(area <= 0.0):
        raise ValueError("invalid native cell quadrature")
    indices = np.argwhere(weight > 0.0)
    indices = indices[np.lexsort((indices[:, 1], indices[:, 0]))]
    if not np.array_equal(indices[0], (0, 0)):
        raise ValueError("frozen mode set does not begin with DC")
    xmin, xmax, ymin, ymax = map(float, domain)
    xn = (x - xmin) / (xmax - xmin)
    yn = (y - ymin) / (ymax - ymin)
    columns = [np.cos(np.pi * int(kx) * xn) * np.cos(np.pi * int(ky) * yn) for ky, kx in indices]
    matrix = np.column_stack(columns)
    scaled = np.sqrt(area / np.sum(area))[:, None] * matrix
    singular = np.linalg.svd(scaled, compute_uv=False)
    condition = float(singular[0] / max(singular[-1], EPS))
    rank = int(np.linalg.matrix_rank(scaled, tol=singular[0] * 1e-12))
    if rank != matrix.shape[1]:
        raise ValueError(f"masked frozen basis is rank deficient: {rank}/{matrix.shape[1]}")
    return matrix, weight[indices[:, 0], indices[:, 1]], {
        "native_cell_count": len(x),
        "frozen_mode_count": len(indices),
        "weighted_design_condition_number": condition,
        "weighted_design_rank": rank,
        "minimum_singular_value": float(singular[-1]),
        "solid_cells_in_operator": 0,
    }


def native_data_consistent_residual(
    raw: np.ndarray,
    prior: np.ndarray,
    area: np.ndarray,
    matrix: np.ndarray,
    gains: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    raw = np.asarray(raw, dtype=np.float64)
    prior = np.asarray(prior, dtype=np.float64)
    area = np.asarray(area, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    gains = np.asarray(gains, dtype=np.float64)
    if raw.shape != prior.shape or matrix.shape != (len(raw), len(gains)):
        raise ValueError("native residual shapes differ")
    scaled_matrix = np.sqrt(area)[:, None] * matrix
    scaled_residual = np.sqrt(area) * (raw - prior)
    coefficients = np.linalg.lstsq(scaled_matrix, scaled_residual, rcond=1e-12)[0]
    result = prior + matrix @ (gains * coefficients)
    weighted_mean = lambda values: float(np.sum(area * values) / np.sum(area))
    result += weighted_mean(raw) - weighted_mean(result)
    return result, {
        "weighted_DC_absolute_error": abs(weighted_mean(result) - weighted_mean(raw)),
        "raw_weighted_mean": weighted_mean(raw),
        "prior_weighted_mean": weighted_mean(prior),
        "selected_weighted_mean": weighted_mean(result),
        "weighted_residual_projection_fraction": float(
            np.linalg.norm(scaled_matrix @ coefficients) / max(np.linalg.norm(scaled_residual), EPS)
        ),
    }


def _normal_heat_flux(qx: np.ndarray, qy: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    radius = np.sqrt(x * x + y * y)
    return qx * x / np.maximum(radius, EPS) + qy * y / np.maximum(radius, EPS)


def _weighted_field_metric(candidate: np.ndarray, target: np.ndarray, area: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        return nrmse(candidate, target, area)
    return nrmse(np.asarray(candidate)[mask], np.asarray(target)[mask], np.asarray(area)[mask])


def analyze_cylinder(mv16a_root: Path, campaign_root: Path, output: Path, batch_size: int) -> dict[str, Any]:
    root = Path(mv16a_root).resolve()
    campaign = Path(campaign_root).resolve()
    _verify_manifest(root, "prediction_manifest.json")
    mv16a, mv14, mv9 = _mv16a_module(), _mv14_module(), _mv9_module()
    # The prediction manifest locks the source-lock manifest as a file; this
    # second call recursively verifies every MV11 moment file referenced by it.
    mv16a.verify_source_lock(root)
    lock = json.loads((root / "submission_lock.json").read_text(encoding="utf-8"))
    with np.load(root / "locked_cylinder_predictions.npz", allow_pickle=False) as source:
        v1 = {name: np.asarray(source[name]) for name in source.files}
    if tuple(v1["seeds"].astype(int)) != CYLINDER_SEEDS:
        raise ValueError("MV16A cylinder seed identity changed")
    weight = np.asarray(v1["frozen_weight_map"], dtype=np.float64)
    b3_fields, b10_fields, b10_on_b3, images = [], [], [], []
    basis_audits, dc_audits, b10_mapping_coverages = [], [], []
    for seed in CYLINDER_SEEDS:
        metadata3, additive3 = mv16a.aggregate_moment_files(
            [mv16a._moment_path(campaign, seed, nout) for nout in mv16a.B3_NOUT]
        )
        fields3 = mv16a.reconstruct_fields(metadata3, additive3)
        image, _, raster_audit = mv16a.rasterize_fields(fields3, weight.shape)
        metadata10, additive10 = mv16a.aggregate_moment_files(
            [mv16a._moment_path(campaign, seed, nout) for nout in mv16a.B10_NOUT]
        )
        fields10 = mv16a.reconstruct_fields(metadata10, additive10)
        mapped_b10_qx, coverage_x = _map_peer_to_points(
            fields10["x_m"],
            fields10["y_m"],
            fields10["outputs"][:, QX_INDEX],
            fields3["x_m"],
            fields3["y_m"],
        )
        mapped_b10_qy, coverage_y = _map_peer_to_points(
            fields10["x_m"],
            fields10["y_m"],
            fields10["outputs"][:, QY_INDEX],
            fields3["x_m"],
            fields3["y_m"],
        )
        b3_fields.append(fields3)
        b10_fields.append(fields10)
        b10_on_b3.append((mapped_b10_qx, mapped_b10_qy))
        b10_mapping_coverages.extend((coverage_x, coverage_y))
        images.append(image)
        basis_audits.append({"seed": seed, "raster": raster_audit})
    image_array = np.asarray(images, dtype=np.float32)
    neural = mv14._predict_mamba_validation(Path(lock["mv9_output_root"]), image_array, batch_size=int(batch_size)).astype(np.float64)
    tsvd = mv9._project_modules()["tsvd"]
    tsvd_rank = int(lock["frozen_tsvd_rank"])
    tsvd_output = np.asarray(tsvd(image_array[:, :4], tsvd_rank), dtype=np.float64)
    native_methods: dict[str, list[np.ndarray]] = {
        name: []
        for name in ("raw_b3_qx", "raw_b3_qy", "vision_qx", "vision_qy", "selected_qx", "selected_qy", "tsvd_qx", "tsvd_qy", "permuted_qx", "permuted_qy", "raw_b10_qx", "raw_b10_qy")
    }
    for index, (seed, fields3, fields10) in enumerate(zip(CYLINDER_SEEDS, b3_fields, b10_fields)):
        x, y, area = fields3["x_m"], fields3["y_m"], fields3["area_m2"]
        matrix, gains, basis = masked_mode_matrix(x, y, area, weight, mv16a.DOMAIN)
        if basis["weighted_design_condition_number"] > 1.0e10:
            raise ValueError(f"masked basis is numerically unsafe for seed {seed}: {basis}")
        raw_qx, raw_qy = fields3["outputs"][:, QX_INDEX], fields3["outputs"][:, QY_INDEX]
        vision_qx = _interpolate_raster_to_cells(neural[index, QX_INDEX], x, y, mv16a)
        vision_qy = _interpolate_raster_to_cells(neural[index, QY_INDEX], x, y, mv16a)
        tsvd_qx = _interpolate_raster_to_cells(tsvd_output[index, QX_INDEX], x, y, mv16a)
        tsvd_qy = _interpolate_raster_to_cells(tsvd_output[index, QY_INDEX], x, y, mv16a)
        selected_qx, audit_x = native_data_consistent_residual(raw_qx, vision_qx, area, matrix, gains)
        selected_qy, audit_y = native_data_consistent_residual(raw_qy, vision_qy, area, matrix, gains)
        source_index = (index - 1) % len(CYLINDER_SEEDS)
        peer3 = b3_fields[source_index]
        perm_qx, _ = _map_peer_to_points(peer3["x_m"], peer3["y_m"], peer3["outputs"][:, QX_INDEX], x, y)
        perm_qy, _ = _map_peer_to_points(peer3["x_m"], peer3["y_m"], peer3["outputs"][:, QY_INDEX], x, y)
        permuted_qx, audit_px = native_data_consistent_residual(perm_qx, vision_qx, area, matrix, gains)
        permuted_qy, audit_py = native_data_consistent_residual(perm_qy, vision_qy, area, matrix, gains)
        values = {
            "raw_b3_qx": raw_qx,
            "raw_b3_qy": raw_qy,
            "vision_qx": vision_qx,
            "vision_qy": vision_qy,
            "selected_qx": selected_qx,
            "selected_qy": selected_qy,
            "tsvd_qx": tsvd_qx,
            "tsvd_qy": tsvd_qy,
            "permuted_qx": permuted_qx,
            "permuted_qy": permuted_qy,
            "raw_b10_qx": b10_on_b3[index][0],
            "raw_b10_qy": b10_on_b3[index][1],
        }
        for name, value in values.items():
            native_methods[name].append(np.asarray(value, dtype=np.float64))
        basis_audits[index].update({"basis": basis, "selected_qx": audit_x, "selected_qy": audit_y, "permuted_qx": audit_px, "permuted_qy": audit_py})
        dc_audits.extend((audit_x["weighted_DC_absolute_error"], audit_y["weighted_DC_absolute_error"]))
    targets_qx, targets_qy = [], []
    interpolation_coverages = list(b10_mapping_coverages)
    for index, fields in enumerate(b3_fields):
        peer_qx, peer_qy = [], []
        for peer_index in range(len(CYLINDER_SEEDS)):
            if peer_index == index:
                continue
            mapped_qx, coverage_x = _map_peer_to_points(
                b3_fields[peer_index]["x_m"], b3_fields[peer_index]["y_m"], native_methods["raw_b10_qx"][peer_index], fields["x_m"], fields["y_m"]
            )
            mapped_qy, coverage_y = _map_peer_to_points(
                b3_fields[peer_index]["x_m"], b3_fields[peer_index]["y_m"], native_methods["raw_b10_qy"][peer_index], fields["x_m"], fields["y_m"]
            )
            peer_qx.append(mapped_qx)
            peer_qy.append(mapped_qy)
            interpolation_coverages.extend((coverage_x, coverage_y))
        targets_qx.append(np.mean(peer_qx, axis=0))
        targets_qy.append(np.mean(peer_qy, axis=0))
    method_roots = ("raw_b3", "vision", "selected", "tsvd", "permuted", "raw_b10")
    per_seed_qy = {name: [] for name in method_roots}
    per_seed_qn = {name: [] for name in method_roots}
    near_wall_counts = []
    for index, fields in enumerate(b3_fields):
        x, y, area = fields["x_m"], fields["y_m"], fields["area_m2"]
        radius = np.sqrt(x * x + y * y)
        near_wall = (radius >= mv16a.CYLINDER_RADIUS) & (radius - mv16a.CYLINDER_RADIUS <= 0.05 * (2.0 * mv16a.CYLINDER_RADIUS))
        near_wall_counts.append(int(np.count_nonzero(near_wall)))
        if np.count_nonzero(near_wall) < 20:
            raise ValueError(f"too few native near-wall cells for seed {CYLINDER_SEEDS[index]}")
        target_qn = _normal_heat_flux(targets_qx[index], targets_qy[index], x, y)
        for name in method_roots:
            qx = native_methods[f"{name}_qx"][index]
            qy = native_methods[f"{name}_qy"][index]
            per_seed_qy[name].append(_weighted_field_metric(qy, targets_qy[index], area))
            qn = _normal_heat_flux(qx, qy, x, y)
            per_seed_qn[name].append(_weighted_field_metric(qn, target_qn, area, near_wall))
    mean_qy = {name: float(np.mean(values)) for name, values in per_seed_qy.items()}
    mean_qn = {name: float(np.mean(values)) for name, values in per_seed_qn.items()}
    qy_ratios = {name: value / max(mean_qy["raw_b10"], EPS) for name, value in mean_qy.items()}
    qn_ratios = {name: value / max(mean_qn["raw_b10"], EPS) for name, value in mean_qn.items()}
    stats_qy = paired_log_ratio_statistics(per_seed_qy["selected"], per_seed_qy["raw_b10"])
    stats_qn = paired_log_ratio_statistics(per_seed_qn["selected"], per_seed_qn["raw_b10"])
    raw_p = [stats_qy["exact_sign_test_two_sided_p"], stats_qn["exact_sign_test_two_sided_p"]]
    order = np.argsort(raw_p)
    holm = [0.0, 0.0]
    running = 0.0
    for rank, endpoint in enumerate(order):
        running = max(running, (2 - rank) * raw_p[endpoint])
        holm[endpoint] = min(1.0, running)
    stats_qy["Holm_two_endpoint_adjusted_p"] = holm[0]
    stats_qn["Holm_two_endpoint_adjusted_p"] = holm[1]
    figures = _plot_cylinder(output, b3_fields, native_methods, targets_qx, targets_qy, mv16a)
    native_field_files = []
    for index, (seed, fields) in enumerate(zip(CYLINDER_SEEDS, b3_fields)):
        x, y = fields["x_m"], fields["y_m"]
        radius = np.sqrt(x * x + y * y)
        near_wall = (radius >= mv16a.CYLINDER_RADIUS) & (
            radius - mv16a.CYLINDER_RADIUS
            <= 0.05 * (2.0 * mv16a.CYLINDER_RADIUS)
        )
        path = output / f"cylinder_native_fields_seed_{seed}.npz"
        np.savez_compressed(
            path,
            seed=np.asarray(seed),
            x_m=x,
            y_m=y,
            area_m2=fields["area_m2"],
            near_wall_mask=near_wall,
            target_qx=targets_qx[index],
            target_qy=targets_qy[index],
            **{name: values[index] for name, values in native_methods.items()},
        )
        native_field_files.append(path.name)
    gates = {
        "no_DSMC_rerun": True,
        "no_neural_training_or_cylinder_tuning": True,
        "all_four_predeclared_seeds_present": len(b3_fields) == 4,
        "native_area_weighted_DC_preserved": max(dc_audits) <= 1.0e-10,
        "minimum_peer_linear_interpolation_coverage_at_least_95_percent": min(interpolation_coverages) >= 0.95,
        "selected_global_qy_effect_size_better_than_Raw_B10": qy_ratios["selected"] < 1.0,
        "selected_near_wall_qn_effect_size_better_than_Raw_B10": qn_ratios["selected"] < 1.0,
        "permuted_observation_degrades_global_qy": mean_qy["permuted"] > mean_qy["selected"],
        "original_tUD30_warning_preserved": True,
    }
    summary = {
        "classification": "retrospective_mask_aware_cross_geometry_transfer_audit_not_new_confirmation",
        "original_MV11_tUD30_gate_pass": bool(lock.get("original_MV11_tUD30_gate_pass", False)),
        "original_MV11_tUD30_warning_preserved": True,
        "mask_aware_operator": "native_DS2V_fluid_cells_area_weighted_frozen_mode_span",
        "per_seed_global_qy_nrmse": per_seed_qy,
        "mean_global_qy_nrmse": mean_qy,
        "global_qy_ratios_to_Raw_B10": qy_ratios,
        "per_seed_near_wall_qn_nrmse": per_seed_qn,
        "mean_near_wall_qn_nrmse": mean_qn,
        "near_wall_qn_ratios_to_Raw_B10": qn_ratios,
        "paired_statistics": {"global_qy": stats_qy, "near_wall_qn": stats_qn},
        "near_wall_cell_counts": near_wall_counts,
        "minimum_peer_linear_interpolation_coverage": min(interpolation_coverages),
        "maximum_native_DC_absolute_error": max(dc_audits),
        "basis_audit": basis_audits,
        "gates": gates,
        "all_engineering_gates_pass": all(gates.values()),
        "formal_small_n_superiority_claim_authorized": False,
        "formal_claim_reason": "With four independent seeds, the minimum one-sided exact sign-test p-value is 0.0625.",
        "DSMC_rerun": False,
        "neural_training": False,
        "cylinder_parameter_selection": False,
        "native_field_files": native_field_files,
        "figures": figures,
    }
    _atomic_json(output / "cylinder_summary.json", summary)
    _atomic_json(output / "cylinder_basis_audit.json", {"records": basis_audits})
    rows = []
    for endpoint, values in (("global_qy", per_seed_qy), ("near_wall_qn", per_seed_qn)):
        for method, records in values.items():
            for seed, value in zip(CYLINDER_SEEDS, records):
                baseline = (per_seed_qy if endpoint == "global_qy" else per_seed_qn)["raw_b10"][CYLINDER_SEEDS.index(seed)]
                rows.append({"endpoint": endpoint, "seed": seed, "method": method, "nrmse": value, "ratio_to_Raw_B10": value / max(baseline, EPS)})
    _write_rows(output / "cylinder_native_metrics.csv", list(rows[0]), rows)
    return summary


def _plot_cylinder(
    output: Path,
    fields: Sequence[Mapping[str, np.ndarray]],
    methods: Mapping[str, Sequence[np.ndarray]],
    target_qx: Sequence[np.ndarray],
    target_qy: Sequence[np.ndarray],
    mv16a: Any,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    index = 0
    x, y = fields[index]["x_m"], fields[index]["y_m"]
    order = ("reference", "raw_b3", "vision", "selected", "tsvd", "raw_b10")
    titles = ("LOSO Reference", "Raw DSMC B=3", "MambaIRv2 B=3", "Mask-aware DCIR B=3", "TSVD/POD B=3", "Raw DSMC B=10")
    arrays = {"reference": target_qy[index], **{name: methods[f"{name}_qy"][index] for name in order[1:]}}
    triangulation = mtri.Triangulation(x, y)
    triangles = triangulation.triangles
    cx = np.mean(x[triangles], axis=1)
    cy = np.mean(y[triangles], axis=1)
    edge = np.max(
        np.stack(
            (
                np.hypot(x[triangles[:, 0]] - x[triangles[:, 1]], y[triangles[:, 0]] - y[triangles[:, 1]]),
                np.hypot(x[triangles[:, 1]] - x[triangles[:, 2]], y[triangles[:, 1]] - y[triangles[:, 2]]),
                np.hypot(x[triangles[:, 2]] - x[triangles[:, 0]], y[triangles[:, 2]] - y[triangles[:, 0]]),
            )
        ),
        axis=0,
    )
    typical = float(np.median(np.sqrt(fields[index]["area_m2"])))
    triangulation.set_mask((cx * cx + cy * cy < mv16a.CYLINDER_RADIUS**2) | (edge > 6.0 * typical))
    limit = max(float(np.max(np.abs(value))) for value in arrays.values())
    reference_scale = max(math.sqrt(float(np.mean(target_qy[index] ** 2))), EPS)
    error_limit = max(float(np.quantile(np.abs(np.concatenate([(arrays[name] - target_qy[index]) / reference_scale for name in order[1:]])), 0.995)), 0.01)
    levels = np.linspace(-limit, limit, 41)
    error_levels = np.linspace(-error_limit, error_limit, 41)
    fig, axes = plt.subplots(2, 6, figsize=(19.0, 7.0), sharex=True, sharey=True, constrained_layout=True)
    field_contour = error_contour = None
    for column, (name, title) in enumerate(zip(order, titles)):
        field_contour = axes[0, column].tricontourf(triangulation, arrays[name], levels=levels, cmap="RdBu_r", extend="both")
        axes[0, column].set_title(title, fontsize=10)
        if name == "reference":
            axes[1, column].set_facecolor("0.94")
            axes[1, column].text(0.5, 0.5, "Reference", transform=axes[1, column].transAxes, ha="center", va="center", color="0.45")
        else:
            error_contour = axes[1, column].tricontourf(triangulation, (arrays[name] - target_qy[index]) / reference_scale, levels=error_levels, cmap="RdBu_r", extend="both")
            axes[1, column].set_xlabel("$x$ [m]")
        for row in range(2):
            axes[row, column].set_aspect("equal")
            axes[row, column].set_xlim(mv16a.DOMAIN[0], mv16a.DOMAIN[1])
            axes[row, column].set_ylim(mv16a.DOMAIN[2], mv16a.DOMAIN[3])
    axes[0, 0].set_ylabel("$y$ [m]")
    axes[1, 0].set_ylabel("$y$ [m]")
    assert field_contour is not None and error_contour is not None
    fig.colorbar(field_contour, ax=axes[0, :], shrink=0.82, label=r"normalised $q_y$")
    fig.colorbar(error_contour, ax=axes[1, :], shrink=0.82, label=r"$\Delta q_y/\mathrm{RMS}(q_{y,ref})$")
    fig.suptitle(f"Native-cell mask-aware cylinder audit, seed {CYLINDER_SEEDS[index]}")
    paths = []
    for suffix in ("png", "pdf"):
        path = output / f"mv16b_cylinder_native_qy_six_panel_seed_{CYLINDER_SEEDS[index]}.{suffix}"
        fig.savefig(path, dpi=500 if suffix == "png" else None, bbox_inches="tight")
        paths.append(path.name)
    plt.close(fig)

    radius = np.sqrt(x * x + y * y)
    wall = (radius >= mv16a.CYLINDER_RADIUS) & (radius - mv16a.CYLINDER_RADIUS <= 0.05 * 2.0 * mv16a.CYLINDER_RADIUS)
    theta = np.arctan2(y[wall], x[wall]) * 180.0 / np.pi
    fig, axis = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    for name, label in zip(order, titles):
        if name == "reference":
            qx, qy = target_qx[index], target_qy[index]
        else:
            qx, qy = methods[f"{name}_qx"][index], methods[f"{name}_qy"][index]
        qn = _normal_heat_flux(qx, qy, x, y)[wall]
        bins = np.linspace(0.0, 180.0, 37)
        centers = 0.5 * (bins[:-1] + bins[1:])
        curve = np.asarray([np.mean(qn[(theta >= bins[i]) & (theta < bins[i + 1])]) if np.any((theta >= bins[i]) & (theta < bins[i + 1])) else np.nan for i in range(len(centers))])
        axis.plot(centers, curve, label=label, linewidth=1.5)
    axis.set(xlabel=r"cylinder angle $\theta$ [deg]", ylabel=r"near-wall normal heat flux $q_n$ (normalised)", title="Near-wall kinetic heat-flux transfer (cell moments, not wall tally)")
    axis.legend(frameon=False, ncol=2, fontsize=8)
    for suffix in ("png", "pdf"):
        path = output / f"mv16b_cylinder_near_wall_qn_seed_{CYLINDER_SEEDS[index]}.{suffix}"
        fig.savefig(path, dpi=500 if suffix == "png" else None, bbox_inches="tight")
        paths.append(path.name)
    plt.close(fig)
    return paths


def run_audit(
    mv15c_root: Path,
    mv16a_root: Path,
    campaign_root: Path,
    output_root: Path,
    *,
    batch_size: int,
) -> dict[str, Any]:
    output = Path(output_root).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite MV16B output: {output}")
    verify_contract()
    cavity_root = Path(mv15c_root).resolve()
    cylinder_root = Path(mv16a_root).resolve()
    campaign = Path(campaign_root).resolve()
    _verify_manifest(cavity_root, "prediction_manifest.json")
    _verify_manifest(cylinder_root, "prediction_manifest.json")
    output.mkdir(parents=True)
    copied_protocol = output / PROTOCOL_FILE
    copied_protocol.write_bytes(protocol_path().read_bytes())
    lock = {
        "stage": STAGE,
        "status": "MV16B_sources_locked_before_new_attribution_and_mask_aware_computation",
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": _sha256(copied_protocol),
        "mv15c_output_root": str(cavity_root),
        "mv16a_output_root": str(cylinder_root),
        "mv11_campaign_root": str(campaign),
        "mv15c_prediction_manifest_sha256": _sha256(cavity_root / "prediction_manifest.json"),
        "mv15c_predictions_sha256": _sha256(cavity_root / "locked_fresh_predictions.npz"),
        "mv16a_prediction_manifest_sha256": _sha256(cylinder_root / "prediction_manifest.json"),
        "mv16a_predictions_sha256": _sha256(cylinder_root / "locked_cylinder_predictions.npz"),
        "DSMC_rerun": False,
        "neural_training": False,
        "fresh_parameter_selection": False,
        "retrospective_classification_preserved": True,
    }
    _atomic_json(output / "source_lock.json", lock)
    _write_manifest(output, "source_lock_manifest.json", [copied_protocol, output / "source_lock.json"])
    cavity = analyze_cavity(cavity_root, output)
    cylinder = analyze_cylinder(cylinder_root, campaign, output, int(batch_size))
    decision = (
        "MV16B_effect_size_evidence_supports_JCP_draft_with_explicit_small_n_and_retrospective_limits"
        if cylinder["gates"]["selected_global_qy_effect_size_better_than_Raw_B10"]
        and cylinder["gates"]["selected_near_wall_qn_effect_size_better_than_Raw_B10"]
        else "MV16B_cylinder_effect_size_does_not_support_cross_geometry_claim"
    )
    summary = {
        "stage": STAGE,
        "status": "complete_MV16B_JCP_evidence_audit",
        "decision": decision,
        "scientific_classification": locked_protocol()["scientific_classification"],
        "cavity": cavity,
        "cylinder": cylinder,
        "DSMC_rerun": False,
        "neural_training": False,
        "new_claim_is_confirmatory": False,
        "recommended_manuscript_language": "Report paired effect sizes and uncertainty; call cylinder evidence retrospective frozen transfer, preserve the original tU/D=30 warning, and do not claim p<0.05 superiority from four seeds.",
    }
    _atomic_json(output / "summary.json", summary)
    return summary


def package_results(output_root: Path, return_directory: Path) -> dict[str, Any]:
    output = Path(output_root).resolve()
    returned = Path(return_directory).resolve()
    _verify_manifest(output, "source_lock_manifest.json")
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    names = [
        PROTOCOL_FILE,
        "source_lock.json",
        "source_lock_manifest.json",
        "summary.json",
        "cavity_summary.json",
        "cavity_stationarity.csv",
        "cavity_budget_scaling.csv",
        "cavity_audit_fields.npz",
        "cylinder_summary.json",
        "cylinder_basis_audit.json",
        "cylinder_native_metrics.csv",
    ]
    names.extend(summary["cavity"]["figures"])
    names.extend(summary["cylinder"]["figures"])
    names.extend(summary["cylinder"]["native_field_files"])
    accounting = output / "mv16b_slurm_accounting.psv"
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
    archive = returned / f"MV16B_JCP_EVIDENCE_AUDIT_BUNDLE_{timestamp}.zip"
    if archive.exists():
        raise FileExistsError(archive)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as stream:
        for path in (*files, output / "artifact_manifest.json"):
            stream.write(path, arcname=path.name)
    digest = _sha256(archive)
    result = {
        "stage": STAGE,
        "archive": str(archive),
        "archive_sha256": digest,
        "decision": summary["decision"],
        "DSMC_rerun": False,
        "neural_training": False,
    }
    _atomic_json(output / "return.json", result)
    pointer = returned / RESULT_POINTER
    temporary = pointer.with_suffix(pointer.suffix + ".tmp")
    temporary.write_text(
        "\n".join(
            (
                f"MV16B_OUTPUT_ROOT={output}",
                f"MV16B_RESULT_ARCHIVE={archive}",
                f"MV16B_RESULT_ARCHIVE_SHA256={digest}",
                f"MV16B_DECISION={summary['decision']}",
                "MV16B_DSMC_RERUN=false",
                "MV16B_NEURAL_TRAINING=false",
                "",
            )
        ),
        encoding="utf-8",
    )
    temporary.replace(pointer)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify")
    run = sub.add_parser("run")
    run.add_argument("--mv15c-root", type=Path, required=True)
    run.add_argument("--mv16a-root", type=Path, required=True)
    run.add_argument("--campaign-root", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--batch-size", type=int, default=4)
    package = sub.add_parser("package")
    package.add_argument("--output-root", type=Path, required=True)
    package.add_argument("--return-directory", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "verify":
        value = verify_contract()
    elif args.command == "run":
        value = run_audit(
            args.mv15c_root,
            args.mv16a_root,
            args.campaign_root,
            args.output_root,
            batch_size=args.batch_size,
        )
    else:
        value = package_results(args.output_root, args.return_directory)
    print(_json_dumps(value))


if __name__ == "__main__":
    main()
