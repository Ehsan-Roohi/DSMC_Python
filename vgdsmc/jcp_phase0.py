"""JCP Phase-0 full-hierarchy adaptive-fusion audit.

This development-only stage launches no DSMC trajectory and performs no
parameter selection on prospective data.  It reconstructs the eight fields

    rho, u, v, T, Pxy, Pxx-Pyy, qx, qy

from the exact additive accumulators of the existing MV15C trajectories.  The
same target-free blockwise empirical-Bayes (EB) kernel is applied to every
field.  Heat-flux noise is estimated from between-block scatter and from the
overlapping B3/B10 identity; a direct particle-level heat-flux variance is not
claimed because the stored third moment would require sixth-order raw moments
for that calculation.
"""

from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np


FIELD_NAMES = (
    "rho",
    "u",
    "v",
    "T",
    "Pxy",
    "Pxx_minus_Pyy",
    "qx",
    "qy",
)
OUTPUT_FIELDS = FIELD_NAMES[4:]
KINETIC_OFFSET = 4
INPUT_BLOCKS = (0, 1, 2)
LARGE_BUDGET = 10
DEFAULT_BIN_WIDTH = 10
MIN_MODES_PER_BIN = 64
EPS = np.finfo(np.float64).tiny


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialise {type(value).__name__}")


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=32)
def _dct_matrix(size: int) -> np.ndarray:
    size = int(size)
    if size < 1:
        raise ValueError("DCT axis must be nonempty")
    modes = np.arange(size, dtype=np.float64)[:, None]
    points = np.arange(size, dtype=np.float64)[None, :]
    matrix = np.sqrt(2.0 / size) * np.cos(
        np.pi * (points + 0.5) * modes / size
    )
    matrix[0] /= math.sqrt(2.0)
    matrix.setflags(write=False)
    return matrix


def dct2(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim < 2:
        raise ValueError("DCT input must have at least two axes")
    try:
        from scipy.fft import dctn
    except ModuleNotFoundError:
        cy = _dct_matrix(value.shape[-2])
        cx = _dct_matrix(value.shape[-1])
        return np.matmul(np.matmul(cy, value), cx.T)
    return dctn(value, axes=(-2, -1), norm="ortho")


def idct2(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim < 2:
        raise ValueError("inverse-DCT input must have at least two axes")
    try:
        from scipy.fft import idctn
    except ModuleNotFoundError:
        cy = _dct_matrix(value.shape[-2])
        cx = _dct_matrix(value.shape[-1])
        return np.matmul(np.matmul(cy.T, value), cx)
    return idctn(value, axes=(-2, -1), norm="ortho")


def axis_slices(size: int, width: int, minimum_width: int = 8) -> list[slice]:
    size, width, minimum_width = int(size), int(width), int(minimum_width)
    if size < minimum_width or width < minimum_width:
        raise ValueError("each transform bin axis must support at least eight modes")
    edges = list(range(0, size, width)) + [size]
    if len(edges) >= 3 and edges[-1] - edges[-2] < minimum_width:
        edges.pop(-2)
    return [slice(left, right) for left, right in zip(edges[:-1], edges[1:], strict=True)]


def mode_bins(shape: Sequence[int], width: int) -> list[tuple[slice, slice]]:
    ys = axis_slices(int(shape[-2]), int(width))
    xs = axis_slices(int(shape[-1]), int(width))
    bins = [(sy, sx) for sy in ys for sx in xs]
    if min((sy.stop - sy.start) * (sx.stop - sx.start) for sy, sx in bins) < MIN_MODES_PER_BIN:
        raise AssertionError("transform bin has fewer than 64 modes")
    return bins


def pool_power(power: np.ndarray, width: int) -> np.ndarray:
    power = np.asarray(power, dtype=np.float64)
    if power.ndim != 2:
        raise ValueError("power map must be two-dimensional")
    result = np.empty_like(power)
    for sy, sx in mode_bins(power.shape, width):
        result[sy, sx] = float(np.mean(power[sy, sx]))
    return result


def overlapping_noise_power(
    raw_b: np.ndarray,
    raw_large: np.ndarray,
    peer_indices: Sequence[int],
    *,
    budget: int = 3,
    large_budget: int = LARGE_BUDGET,
    width: int = DEFAULT_BIN_WIDTH,
) -> np.ndarray:
    """Estimate transform noise at budget B from overlapping B/B_large means."""

    raw_b = np.asarray(raw_b, dtype=np.float64)
    raw_large = np.asarray(raw_large, dtype=np.float64)
    peers = np.asarray(peer_indices, dtype=np.int64)
    budget, large_budget = int(budget), int(large_budget)
    if raw_b.shape != raw_large.shape or raw_b.ndim != 3:
        raise ValueError("overlap arrays must have shape (unit,ny,nx)")
    if peers.size < 2 or not (0 < budget < large_budget):
        raise ValueError("overlap estimate needs two peers and 0 < B < B_large")
    difference = dct2(raw_b[peers] - raw_large[peers])
    # Var(X_B-X_L)=sigma_1^2(1/B-1/L); requested Var(X_B)=sigma_1^2/B.
    factor = float(large_budget) / float(large_budget - budget)
    return pool_power(factor * np.mean(difference**2, axis=0), width)


def direct_block_noise_power(
    raw_blocks: np.ndarray,
    peer_indices: Sequence[int],
    *,
    budget: int = 3,
    width: int = DEFAULT_BIN_WIDTH,
) -> np.ndarray:
    """Estimate B-block coefficient noise from independent B1 block scatter."""

    blocks = np.asarray(raw_blocks, dtype=np.float64)
    peers = np.asarray(peer_indices, dtype=np.int64)
    if blocks.ndim != 4 or peers.size < 2 or blocks.shape[1] < 2:
        raise ValueError("block noise needs shape (unit,block,ny,nx) and two peers")
    coefficients = dct2(blocks[peers])
    per_unit = np.var(coefficients, axis=1, ddof=1) / float(budget)
    return pool_power(np.mean(per_unit, axis=0), width)


def eb_gain(
    observation: np.ndarray,
    prior: np.ndarray,
    noise_power: np.ndarray,
    *,
    width: int = DEFAULT_BIN_WIDTH,
) -> np.ndarray:
    """Target-free block EB gain clip((R-N)/R,0,1), including the DC bin."""

    observation = np.asarray(observation, dtype=np.float64)
    prior = np.asarray(prior, dtype=np.float64)
    noise_power = np.asarray(noise_power, dtype=np.float64)
    if observation.shape != prior.shape or observation.shape != noise_power.shape:
        raise ValueError("EB observation, prior, and noise maps must match")
    residual = dct2(observation - prior)
    gain = np.empty_like(residual)
    for sy, sx in mode_bins(residual.shape, width):
        residual_power = float(np.mean(residual[sy, sx] ** 2))
        noise = float(np.mean(noise_power[sy, sx]))
        value = 0.0 if residual_power <= EPS else (residual_power - noise) / residual_power
        gain[sy, sx] = float(np.clip(value, 0.0, 1.0))
    return gain


def fuse(observation: np.ndarray, prior: np.ndarray, gain: np.ndarray) -> np.ndarray:
    observation = np.asarray(observation, dtype=np.float64)
    prior = np.asarray(prior, dtype=np.float64)
    gain = np.asarray(gain, dtype=np.float64)
    if observation.shape != prior.shape or observation.shape != gain.shape:
        raise ValueError("fusion arrays must have identical shapes")
    return idct2(dct2(prior) + gain * (dct2(observation) - dct2(prior)))


def pnet_cross_block_gain(
    raw_blocks: np.ndarray,
    leave_one_out_priors: np.ndarray,
    *,
    full_budget: int = 3,
    width: int = DEFAULT_BIN_WIDTH,
) -> np.ndarray:
    """Noise2Noise regression gain for a prior dependent on the observation.

    Each prior j is produced from the mean of all blocks except j.  Regression
    first estimates H at budget B-1 and then rescales the signal/noise odds to
    the full budget B, as required by the pre-registration protocol.
    """

    blocks = np.asarray(raw_blocks, dtype=np.float64)
    priors = np.asarray(leave_one_out_priors, dtype=np.float64)
    full_budget = int(full_budget)
    if blocks.shape != priors.shape or blocks.ndim != 3 or blocks.shape[0] != full_budget:
        raise ValueError("cross-block arrays must have shape (B,ny,nx)")
    predictors, responses = [], []
    for held_out in range(full_budget):
        retained = np.mean(np.delete(blocks, held_out, axis=0), axis=0)
        predictors.append(dct2(retained - priors[held_out]))
        responses.append(dct2(blocks[held_out] - priors[held_out]))
    x = np.asarray(predictors)
    y = np.asarray(responses)
    gain = np.empty_like(x[0])
    odds_factor = float(full_budget) / float(full_budget - 1)
    for sy, sx in mode_bins(gain.shape, width):
        denominator = float(np.mean(x[:, sy, sx] ** 2))
        h_previous = 0.0 if denominator <= EPS else float(
            np.clip(np.mean(x[:, sy, sx] * y[:, sy, sx]) / denominator, 0.0, 1.0)
        )
        previous_odds = h_previous / max(1.0 - h_previous, EPS)
        full_odds = odds_factor * previous_odds
        gain[sy, sx] = 1.0 if h_previous >= 1.0 else full_odds / (1.0 + full_odds)
    return np.clip(gain, 0.0, 1.0)


def out_of_support_statistic(
    observation: np.ndarray,
    prior: np.ndarray,
    noise_power: np.ndarray,
    *,
    width: int = DEFAULT_BIN_WIDTH,
) -> float:
    residual = dct2(np.asarray(observation) - np.asarray(prior))
    noise_power = np.asarray(noise_power, dtype=np.float64)
    numerator = 0.0
    denominator = 0.0
    for sy, sx in mode_bins(residual.shape, width):
        count = (sy.stop - sy.start) * (sx.stop - sx.start)
        residual_power = float(np.mean(residual[sy, sx] ** 2))
        noise = float(np.mean(noise_power[sy, sx]))
        numerator += count * max(residual_power - noise, 0.0)
        denominator += count * noise
    return numerator / max(denominator, EPS)


def nrmse(candidate: np.ndarray, target: np.ndarray) -> float:
    candidate = np.asarray(candidate, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return math.sqrt(float(np.sum((candidate - target) ** 2))) / max(
        math.sqrt(float(np.sum(target**2))), EPS
    )


def geometric_mean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or np.any(array <= 0.0):
        raise ValueError("geometric mean requires positive values")
    return float(np.exp(np.mean(np.log(array))))


def leave_one_seed_out(values: np.ndarray, conditions: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    conditions = np.asarray(conditions).astype(str)
    result = np.empty_like(values)
    for index, condition in enumerate(conditions):
        peers = np.flatnonzero((conditions == condition) & (np.arange(len(values)) != index))
        if peers.size < 2:
            raise ValueError(f"condition {condition} has fewer than two reference peers")
        result[index] = np.mean(values[peers], axis=0)
    return result


def development_priors(
    condition_fields: np.ndarray,
    condition_features: np.ndarray,
    target_features: np.ndarray,
    *,
    k: int = 4,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Construct P-NN and development-only physically scaled P-NNs priors."""

    fields = np.asarray(condition_fields, dtype=np.float64)
    features = np.asarray(condition_features, dtype=np.float64)
    targets = np.asarray(target_features, dtype=np.float64)
    if fields.ndim != 4 or features.shape != (len(fields), 2) or targets.ndim != 2:
        raise ValueError("development prior inputs have incompatible shapes")
    design = np.column_stack((np.ones(len(features)), features))
    target_design = np.column_stack((np.ones(len(targets)), targets))
    dc = np.mean(fields, axis=(-2, -1))
    centered = fields - dc[:, :, None, None]
    amplitude = np.sqrt(np.mean(centered**2, axis=(-2, -1)))
    beta_dc = np.linalg.lstsq(design, dc, rcond=None)[0]
    beta_log_amplitude = np.linalg.lstsq(
        design,
        np.log(np.maximum(amplitude, 1.0e-12)),
        rcond=None,
    )[0]
    scale = np.maximum(np.ptp(features, axis=0), 1.0e-6)
    pnn, pnns, audits = [], [], []
    for target_index, target in enumerate(targets):
        distance = np.sqrt(np.sum(((features - target) / scale) ** 2, axis=1))
        eligible = np.flatnonzero(distance > 1.0e-12)
        if eligible.size < 1:
            raise ValueError("target exclusion removed every development condition")
        selected = eligible[np.argsort(distance[eligible])[: min(int(k), eligible.size)]]
        weights = 1.0 / np.maximum(distance[selected], 1.0e-6) ** 2
        weights /= np.sum(weights)
        base = np.tensordot(weights, fields[selected], axes=(0, 0))
        base_dc = np.mean(base, axis=(-2, -1))
        base_centered = base - base_dc[:, None, None]
        base_amplitude = np.sqrt(np.mean(base_centered**2, axis=(-2, -1)))
        predicted_dc = target_design[target_index] @ beta_dc
        predicted_log_amplitude = target_design[target_index] @ beta_log_amplitude
        lower = np.min(np.log(np.maximum(amplitude, 1.0e-12)), axis=0) - math.log(4.0)
        upper = np.max(np.log(np.maximum(amplitude, 1.0e-12)), axis=0) + math.log(4.0)
        predicted_amplitude = np.exp(np.clip(predicted_log_amplitude, lower, upper))
        ratio = np.divide(
            predicted_amplitude,
            base_amplitude,
            out=np.ones_like(predicted_amplitude),
            where=base_amplitude > 1.0e-12,
        )
        scaled = predicted_dc[:, None, None] + ratio[:, None, None] * base_centered
        pnn.append(base)
        pnns.append(scaled)
        audits.append(
            {
                "target_features": target.tolist(),
                "selected_development_indices": selected.tolist(),
                "weights": weights.tolist(),
                "target_condition_excluded": bool(np.all(distance[selected] > 1.0e-12)),
            }
        )
    return np.asarray(pnn), np.asarray(pnns), audits


def _load_exact_trajectory(mv9: Any, directory: Path) -> dict[str, Any]:
    summary = mv9._verify_source_artifacts(directory)
    cfg = mv9._config_from_summary(summary)
    modules = mv9._project_modules()
    checkpoint = modules["load_ntc_checkpoint"](Path(directory) / "checkpoint.npz", cfg)
    root = checkpoint.block_accumulators
    mapping = root.get("block_moments") if isinstance(root, Mapping) else None
    if not isinstance(mapping, Mapping) or len(mapping) != LARGE_BUDGET:
        raise ValueError(f"expected ten additive blocks: {directory}")
    payloads = [dict(mapping[key]) for key in sorted(mapping)]

    def finalise(indices: Sequence[int]) -> np.ndarray:
        payload = mv9.merge_moment_payloads([payloads[int(index)] for index in indices])
        outputs, auxiliary, _ = mv9.moment_fields(
            payload,
            cfg,
            modules["KB"],
            output_dtype=None,
        )
        return np.concatenate((auxiliary, outputs), axis=0)

    blocks = np.asarray([finalise((index,)) for index in range(LARGE_BUDGET)])
    b3 = finalise(INPUT_BLOCKS)
    b10 = finalise(range(LARGE_BUDGET))
    b2_leave_one_out = np.asarray(
        [finalise(tuple(index for index in INPUT_BLOCKS if index != held)) for held in INPUT_BLOCKS]
    )
    return {
        "condition": str(summary.get("condition_id", "")),
        "seed": int(summary.get("seed", summary["config"]["seed"])),
        "blocks": blocks,
        "b3": b3,
        "b10": b10,
        "b2_leave_one_out": b2_leave_one_out,
        "additive_block_count": len(payloads),
    }


def load_unity_data(
    repo_root: Path,
    mv15c_root: Path,
    *,
    batch_size: int,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    mv15c_root = Path(mv15c_root).resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from vgdsmc import mohammadzadeh_mv9_heat_flux as mv9
    from vgdsmc import mohammadzadeh_mv14_kinetic_conservation_cavity as mv14
    from vgdsmc import mohammadzadeh_mv15c_fresh_b3_confirmation as mv15c

    submission = json.loads((mv15c_root / "submission_lock.json").read_text(encoding="utf-8"))
    mv9_root = Path(submission["mv9_output_root"]).resolve()
    specs = mv15c.condition_map(mv15c.locked_protocol())
    trajectories = []
    b3_images, b2_images = [], []
    conditions, seeds, target_features = [], [], []
    for condition, seed in mv15c.fresh_tasks():
        directory = mv15c_root / "references" / condition / f"seed_{seed}"
        item = _load_exact_trajectory(mv9, directory)
        if item["condition"] and item["condition"] != condition:
            raise ValueError(f"condition mismatch in {directory}")
        trajectories.append(item)
        spec = specs[condition]
        conditions.append(condition)
        seeds.append(int(seed))
        target_features.append((math.log10(float(spec["knudsen"])), float(spec["lid_speed_m_per_s"]) / 100.0))
        output_b3, auxiliary_b3 = item["b3"][KINETIC_OFFSET:], item["b3"][:KINETIC_OFFSET]
        b3_images.append(mv9._conditioned_image(output_b3, auxiliary_b3, spec))
        for b2 in item["b2_leave_one_out"]:
            b2_images.append(mv9._conditioned_image(b2[KINETIC_OFFSET:], b2[:KINETIC_OFFSET], spec))
    b3_images_array = np.asarray(b3_images, dtype=np.float32)
    pnet_b3 = mv14._predict_mamba_validation(mv9_root, b3_images_array, batch_size=int(batch_size))
    pnet_b2_flat = mv14._predict_mamba_validation(
        mv9_root,
        np.asarray(b2_images, dtype=np.float32),
        batch_size=int(batch_size),
    )
    pnet_b2 = pnet_b2_flat.reshape(len(trajectories), len(INPUT_BLOCKS), len(OUTPUT_FIELDS), *pnet_b2_flat.shape[-2:])

    with np.load(mv9_root / "dataset.npz", allow_pickle=False) as data:
        train_x = np.asarray(data["train_x"], dtype=np.float64)
        train_y = np.asarray(data["train_y"], dtype=np.float64)
        train_condition = np.asarray(data["train_condition"]).astype(str)
    development_samples = np.concatenate((train_x[:, KINETIC_OFFSET:2 * KINETIC_OFFSET], train_y), axis=1)
    development_fields, development_features, development_names = [], [], []
    for condition in sorted(np.unique(train_condition)):
        mask = train_condition == condition
        development_names.append(str(condition))
        development_fields.append(np.mean(development_samples[mask], axis=0))
        development_features.append(
            (
                float(np.mean(train_x[mask, -2, 0, 0])),
                float(np.mean(train_x[mask, -1, 0, 0])),
            )
        )
    pnn, pnns, prior_audit = development_priors(
        np.asarray(development_fields),
        np.asarray(development_features),
        np.asarray(target_features),
    )

    with np.load(mv15c_root / "locked_fresh_predictions.npz", allow_pickle=False) as data:
        frozen_weight = np.asarray(data["frozen_weight_map"], dtype=np.float64)
    return {
        "conditions": np.asarray(conditions, dtype="U64"),
        "seeds": np.asarray(seeds, dtype=np.int64),
        "blocks": np.asarray([item["blocks"] for item in trajectories], dtype=np.float64),
        "raw_b3": np.asarray([item["b3"] for item in trajectories], dtype=np.float64),
        "raw_b10": np.asarray([item["b10"] for item in trajectories], dtype=np.float64),
        "pnn": pnn,
        "pnns": pnns,
        "pnet_b3": np.asarray(pnet_b3, dtype=np.float64),
        "pnet_b2": np.asarray(pnet_b2, dtype=np.float64),
        "frozen_weight": frozen_weight,
        "development_condition_names": development_names,
        "development_condition_features": np.asarray(development_features),
        "target_features": np.asarray(target_features),
        "prior_audit": prior_audit,
        "mv9_root": str(mv9_root),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze_hierarchy(data: Mapping[str, Any], output_dir: Path, *, width: int) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    conditions = np.asarray(data["conditions"]).astype(str)
    seeds = np.asarray(data["seeds"], dtype=np.int64)
    blocks = np.asarray(data["blocks"], dtype=np.float64)
    raw_b3 = np.asarray(data["raw_b3"], dtype=np.float64)
    raw_b10 = np.asarray(data["raw_b10"], dtype=np.float64)
    pnn = np.asarray(data["pnn"], dtype=np.float64)
    pnns = np.asarray(data["pnns"], dtype=np.float64)
    target = np.stack(
        [leave_one_seed_out(raw_b10[:, field], conditions) for field in range(len(FIELD_NAMES))],
        axis=1,
    )
    methods = {
        "raw_b3": raw_b3.copy(),
        "raw_b10": raw_b10.copy(),
        "pnn_prior_only": pnn.copy(),
        "pnns_prior_only": pnns.copy(),
        "p0_eb": np.empty_like(raw_b3),
        "pnn_eb": np.empty_like(raw_b3),
        "pnns_eb": np.empty_like(raw_b3),
        "pnet_alone": np.full_like(raw_b3, np.nan),
        "pnet_cross_block": np.full_like(raw_b3, np.nan),
        "pnet_frozen_gain": np.full_like(raw_b3, np.nan),
    }
    noise_overlap = np.empty_like(raw_b3)
    noise_direct = np.empty_like(raw_b3)
    detector_rows, ledger_rows = [], []
    pnet_b3 = np.asarray(data["pnet_b3"], dtype=np.float64)
    pnet_b2 = np.asarray(data["pnet_b2"], dtype=np.float64)
    frozen_weight = np.asarray(data["frozen_weight"], dtype=np.float64)
    for unit, condition in enumerate(conditions):
        peers = np.flatnonzero((conditions == condition) & (np.arange(len(conditions)) != unit))
        for field, field_name in enumerate(FIELD_NAMES):
            overlap = overlapping_noise_power(raw_b3[:, field], raw_b10[:, field], peers, width=width)
            direct = direct_block_noise_power(blocks[:, :, field], peers, width=width)
            noise_overlap[unit, field] = overlap
            noise_direct[unit, field] = direct
            for name, prior in (("p0", np.zeros_like(raw_b3[unit, field])), ("pnn", pnn[unit, field]), ("pnns", pnns[unit, field])):
                gain = eb_gain(raw_b3[unit, field], prior, overlap, width=width)
                methods[f"{name}_eb"][unit, field] = fuse(raw_b3[unit, field], prior, gain)
                detector_rows.append(
                    {
                        "condition": condition,
                        "seed": int(seeds[unit]),
                        "field": field_name,
                        "prior": name,
                        "D": out_of_support_statistic(raw_b3[unit, field], prior, overlap, width=width),
                        "mean_gain": float(np.mean(gain)),
                        "minimum_gain": float(np.min(gain)),
                        "maximum_gain": float(np.max(gain)),
                    }
                )
            overlap_total = float(np.sum(overlap))
            direct_total = float(np.sum(direct))
            ledger_rows.append(
                {
                    "condition": condition,
                    "seed": int(seeds[unit]),
                    "field": field_name,
                    "overlap_noise_power": overlap_total,
                    "direct_block_noise_power": direct_total,
                    "overlap_to_direct_ratio": overlap_total / max(direct_total, EPS),
                }
            )
        for output_index, field_name in enumerate(OUTPUT_FIELDS):
            field = KINETIC_OFFSET + output_index
            methods["pnet_alone"][unit, field] = pnet_b3[unit, output_index]
            cross_gain = pnet_cross_block_gain(
                blocks[unit, list(INPUT_BLOCKS), field],
                pnet_b2[unit, :, output_index],
                width=width,
            )
            methods["pnet_cross_block"][unit, field] = fuse(
                raw_b3[unit, field], pnet_b3[unit, output_index], cross_gain
            )
            methods["pnet_frozen_gain"][unit, field] = fuse(
                raw_b3[unit, field], pnet_b3[unit, output_index], frozen_weight
            )
            detector_rows.append(
                {
                    "condition": condition,
                    "seed": int(seeds[unit]),
                    "field": field_name,
                    "prior": "pnet_cross_block",
                    "D": out_of_support_statistic(
                        raw_b3[unit, field],
                        pnet_b3[unit, output_index],
                        noise_overlap[unit, field],
                        width=width,
                    ),
                    "mean_gain": float(np.mean(cross_gain)),
                    "minimum_gain": float(np.min(cross_gain)),
                    "maximum_gain": float(np.max(cross_gain)),
                }
            )

    metric_rows = []
    for unit, (condition, seed) in enumerate(zip(conditions, seeds, strict=True)):
        for field, field_name in enumerate(FIELD_NAMES):
            baseline = nrmse(raw_b10[unit, field], target[unit, field])
            for method_name, array in methods.items():
                candidate = array[unit, field]
                if not np.all(np.isfinite(candidate)):
                    continue
                error = nrmse(candidate, target[unit, field])
                metric_rows.append(
                    {
                        "condition": condition,
                        "seed": int(seed),
                        "field": field_name,
                        "method": method_name,
                        "nrmse": error,
                        "ratio_to_raw_b10": error / max(baseline, EPS),
                    }
                )

    grouped: dict[str, Any] = {}
    for condition in sorted(set(conditions)):
        grouped[condition] = {}
        for field_name in FIELD_NAMES:
            selected_rows = [
                row for row in metric_rows
                if row["condition"] == condition and row["field"] == field_name
            ]
            grouped[condition][field_name] = {}
            for method in sorted({str(row["method"]) for row in selected_rows}):
                values = [float(row["ratio_to_raw_b10"]) for row in selected_rows if row["method"] == method]
                grouped[condition][field_name][method] = {
                    "geometric_ratio_to_raw_b10": geometric_mean(values),
                    "unit_count_below_raw_b10": int(sum(value < 1.0 for value in values)),
                    "n_units": len(values),
                }

    expected_raw_ratio = math.sqrt((1.0 / 3.0 + 1.0 / 30.0) / (1.0 / 10.0 + 1.0 / 30.0))
    raw_ratios = []
    for condition in grouped.values():
        for field in condition.values():
            raw_ratios.append(field["raw_b3"]["geometric_ratio_to_raw_b10"])
    raw_ledger_relative_error = float(np.mean(np.abs(np.asarray(raw_ratios) / expected_raw_ratio - 1.0)))
    noise_ratios = np.asarray([float(row["overlap_to_direct_ratio"]) for row in ledger_rows])

    qy_rows = [row for row in metric_rows if row["field"] == "qy"]
    prior_condition_ratios = {}
    for condition in sorted(set(conditions)):
        values = [
            float(row["ratio_to_raw_b10"])
            for row in qy_rows
            if row["condition"] == condition and row["method"] == "pnn_prior_only"
        ]
        prior_condition_ratios[condition] = geometric_mean(values)
    moderate = min(prior_condition_ratios, key=lambda key: abs(math.log(prior_condition_ratios[key])))
    unit_gates = []
    for seed in seeds[conditions == moderate]:
        rows = {
            str(row["method"]): float(row["nrmse"])
            for row in qy_rows
            if row["condition"] == moderate and int(row["seed"]) == int(seed)
        }
        envelope = min(rows["pnn_prior_only"], rows["p0_eb"])
        comparator = min(envelope, rows["pnet_frozen_gain"])
        unit_gates.append(rows["pnet_cross_block"] < 0.95 * comparator)
    pnet_gate_fraction = float(np.mean(unit_gates))
    gates = {
        "raw_block_ledger_mean_relative_error_within_15_percent": raw_ledger_relative_error <= 0.15,
        "overlap_vs_direct_noise_mean_ratio_within_15_percent": abs(float(np.mean(noise_ratios)) - 1.0) <= 0.15,
        "pnet_cross_block_below_0p95_of_frozen_and_prior_p0_envelope_in_75_percent_of_moderate_qy_units": pnet_gate_fraction >= 0.75,
        "all_eight_fields_reported": set(row["field"] for row in metric_rows) == set(FIELD_NAMES),
        "prospective_data_used": False,
        "new_DSMC_trajectory_launched": False,
    }
    summary = {
        "stage": "JCP1_phase0_full_hierarchy_adaptive_fusion",
        "classification": "retrospective_development_only",
        "fields": list(FIELD_NAMES),
        "heat_flux_noise_policy": "between-block scatter and overlapping B3/B10 identity; no direct particle-level variance claim without sixth-order raw moments",
        "bin_width": int(width),
        "minimum_modes_per_bin": MIN_MODES_PER_BIN,
        "conditions": sorted(set(conditions)),
        "units": len(seeds),
        "raw_block_model": {
            "expected_RawB3_to_RawB10_ratio_against_30_block_LOSO_reference": expected_raw_ratio,
            "mean_relative_error_across_condition_field_cells": raw_ledger_relative_error,
        },
        "noise_identity": {
            "formula": "Var(X_B-X_10)=sigma_1^2(1/B-1/10)",
            "mean_overlap_to_direct_block_noise_ratio": float(np.mean(noise_ratios)),
            "median_overlap_to_direct_block_noise_ratio": float(np.median(noise_ratios)),
        },
        "moderate_shift_condition_selected_development_only": moderate,
        "moderate_qy_pnet_gate_fraction": pnet_gate_fraction,
        "gates": gates,
        "all_gates_pass": bool(all(gates.values())),
        "grouped_metrics": grouped,
        "development_condition_names": list(data["development_condition_names"]),
        "development_condition_features": np.asarray(data["development_condition_features"]),
        "target_features": np.asarray(data["target_features"]),
        "prior_audit": data["prior_audit"],
        "mv9_root": data["mv9_root"],
    }
    _write_csv(output / "metrics.csv", metric_rows)
    _write_csv(output / "noise_ledger.csv", ledger_rows)
    _write_csv(output / "detector.csv", detector_rows)
    _atomic_json(output / "summary.json", summary)
    np.savez_compressed(
        output / "fields.npz",
        conditions=conditions,
        seeds=seeds,
        field_names=np.asarray(FIELD_NAMES),
        target=target,
        noise_overlap=noise_overlap,
        noise_direct=noise_direct,
        **{f"method_{name}": array for name, array in methods.items()},
    )
    manifest = {
        "stage": summary["stage"],
        "files": {
            path.name: {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
            for path in sorted(output.iterdir())
            if path.is_file()
        },
    }
    _atomic_json(output / "manifest.json", manifest)
    return summary


def package(output_dir: Path, archive: Path) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    archive = Path(archive).resolve()
    if archive.exists():
        raise FileExistsError(f"refusing to overwrite {archive}")
    required = ("summary.json", "metrics.csv", "noise_ledger.csv", "detector.csv", "fields.npz", "manifest.json")
    for name in required:
        if not (output / name).is_file():
            raise FileNotFoundError(output / name)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as stream:
        for name in required:
            stream.write(output / name, arcname=name)
    digest = _sha256(archive)
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    checksum.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    return {"archive": str(archive), "sha256": digest, "checksum": str(checksum)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--repo-root", type=Path, required=True)
    run.add_argument("--mv15c-root", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--archive", type=Path, required=True)
    run.add_argument("--bin-width", type=int, default=DEFAULT_BIN_WIDTH)
    run.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.command == "run":
        data = load_unity_data(args.repo_root, args.mv15c_root, batch_size=args.batch_size)
        summary = analyze_hierarchy(data, args.output_dir, width=args.bin_width)
        result = package(args.output_dir, args.archive)
        print(_json_dumps({"summary": summary, "package": result}))


if __name__ == "__main__":
    main()
