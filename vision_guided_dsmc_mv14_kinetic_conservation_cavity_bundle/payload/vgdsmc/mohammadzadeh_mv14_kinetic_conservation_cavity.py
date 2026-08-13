"""MV14 kinetic-conservation reconstruction for the DSMC cavity.

The one-block DSMC third moment is retained as an observation.  Exact
collision-invariant moment balances of the Boltzmann equation are used only as
weak statistical constraints.  No Fourier, Newtonian-stress, or wall-flux
closure is imposed.  Development labels select hyperparameters; legacy test
labels remain inaccessible until predictions are hash locked.
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


STAGE = "MV14_Mohammadzadeh_kinetic_conservation_cavity"
STATUS = "locked_after_MV12_failure_before_any_MV14_outcome"
PROTOCOL_FILE = "mv14_kinetic_conservation_cavity_protocol.json"
OUTPUT_FIELDS = (
    "Pxy_over_p_ref",
    "Pxx_minus_Pyy_over_p_ref",
    "qx_over_q_ref",
    "qy_over_q_ref",
)
QX_INDEX = 2
QY_INDEX = 3
MACRO_SMOOTHING = (0, 1, 2)
SPAN_SETS = {
    "local": (1, 2),
    "multiscale": (1, 2, 4, 8),
    "coarse": (4, 8, 16),
}
LAMBDA_STRENGTHS = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
MAX_DEVELOPMENT_WEAK_RESIDUAL_RATIO = 0.85
EPS = 1.0e-12


def _mv9_module():
    from . import mohammadzadeh_mv9_heat_flux as mv9

    return mv9


def _mv12_module():
    from . import mohammadzadeh_mv12_sage_qy as mv12

    return mv12


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


def _mv9_protocol_path() -> Path:
    mv9 = _mv9_module()
    try:
        return mv9.protocol_path()
    except (ImportError, ModuleNotFoundError):
        candidate = (
            Path(mv9.__file__).resolve().parents[1]
            / "reference_data"
            / "mohammadzadeh_2012"
            / "mv9_heat_flux_noise2noise_protocol.json"
        )
        if not candidate.is_file():
            raise
        return candidate


def _mv12_protocol_path() -> Path:
    mv12 = _mv12_module()
    candidate = (
        Path(mv12.__file__).resolve().parents[1]
        / "reference_data"
        / "mohammadzadeh_2012"
        / "mv12_sage_qy_protocol.json"
    )
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


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
    path = protocol_path()
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("stage") != STAGE or protocol.get("status") != STATUS:
        raise ValueError("MV14 protocol is absent or unlocked")
    selection = protocol["selection_contract"]
    if (
        tuple(int(value) for value in selection["macro_smoothing_passes"])
        != MACRO_SMOOTHING
        or {
            str(key): tuple(int(value) for value in values)
            for key, values in selection["weak_vertical_span_sets"].items()
        }
        != SPAN_SETS
        or tuple(float(value) for value in selection["lambda_strengths"])
        != LAMBDA_STRENGTHS
        or float(selection["maximum_development_weak_residual_ratio"])
        != MAX_DEVELOPMENT_WEAK_RESIDUAL_RATIO
    ):
        raise ValueError("MV14 code differs from the locked selection matrix")
    mv9 = _mv9_module()
    mv12 = _mv12_module()
    ancestry = {
        "mv9_module_sha256": Path(mv9.__file__),
        "mv9_protocol_sha256": _mv9_protocol_path(),
        "mv12_module_sha256": Path(mv12.__file__),
        "mv12_protocol_sha256": _mv12_protocol_path(),
    }
    for key, source in ancestry.items():
        if not source.is_file() or _sha256(source) != protocol["source_contract"][key]:
            raise ValueError(f"MV14 immutable ancestry mismatch: {key}")
    return protocol


def verify_lock() -> dict[str, Any]:
    protocol = locked_protocol()
    return {
        "stage": STAGE,
        "status": "MV14_lock_verified_without_any_MV14_outcome",
        "protocol_sha256": _sha256(protocol_path()),
        "method": protocol["method_name"],
        "Fourier_law_used": False,
        "Navier_Stokes_closure_used": False,
        "wall_heat_flux_imposed": False,
        "legacy_targets_loaded_by_prediction_stage": False,
        "DSMC_rerun": False,
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
            raise ValueError(f"MV14 recursive artifact verification failed: {path}")
    return manifest


def verify_data_contract(mv9_output_root: Path, mv12_output_root: Path) -> dict[str, Any]:
    protocol = locked_protocol()
    mv9_root = Path(mv9_output_root).resolve()
    mv12_root = Path(mv12_output_root).resolve()
    mv9_summary = json.loads((mv9_root / "summary.json").read_text(encoding="utf-8"))
    mv12_summary = json.loads((mv12_root / "summary.json").read_text(encoding="utf-8"))
    _verify_manifest(mv9_root, "assembly_manifest.json")
    _verify_manifest(mv9_root, "artifact_manifest.json")
    _verify_manifest(mv12_root, "artifact_manifest.json")
    mv9 = _mv9_module()
    checks = {
        "MV9_failure_outcome_explicitly_required": mv9_summary.get("decision")
        == protocol["source_contract"]["required_MV9_decision"],
        "MV12_failure_outcome_explicitly_required": mv12_summary.get("decision")
        == protocol["source_contract"]["required_MV12_decision"],
        "MV9_dataset_present": (mv9_root / "dataset.npz").is_file(),
        "MV9_raw_source_index_present": (mv9_root / "source_moment_audit.csv").is_file(),
        "MV9_Mamba_models_present": all(
            (mv9._task_directory(mv9_root, "mambairv2_tiny_adapted", seed) / "model.pt").is_file()
            for seed in mv9.TRAINING_SEEDS
        ),
        "legacy_outcomes_are_not_reclassified_as_confirmation": True,
    }
    if not all(checks.values()):
        raise ValueError(f"MV14 data contract failed: {checks}")
    return {
        "stage": STAGE,
        "status": "MV14_data_contract_verified",
        "mv9_output_root": str(mv9_root),
        "mv12_output_root": str(mv12_root),
        "checks": checks,
    }


def smooth2d(value: np.ndarray, passes: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).copy()
    for _ in range(int(passes)):
        padded = np.pad(result, ((0, 0), (1, 1)), mode="edge")
        result = (padded[:, :-2] + 2.0 * padded[:, 1:-1] + padded[:, 2:]) / 4.0
        padded = np.pad(result, ((1, 1), (0, 0)), mode="edge")
        result = (padded[:-2] + 2.0 * padded[1:-1] + padded[2:]) / 4.0
    return result


def _gradient(value: np.ndarray, *, axis: int) -> np.ndarray:
    spacing = 1.0 / max(value.shape[axis] - 1, 1)
    edge_order = 2 if value.shape[axis] >= 3 else 1
    return np.gradient(value, spacing, axis=axis, edge_order=edge_order)


def kinetic_fields_from_payload(payload: Mapping[str, Any], cfg: Any, kb: float) -> dict[str, np.ndarray | float]:
    """Finalize one additive DSMC block without a constitutive closure."""

    samples = int(payload["samples"])
    ncell = int(cfg.nx) * int(cfg.ny)
    m0 = np.asarray(payload["m0"], dtype=np.float64)
    m1 = np.asarray(payload["m1"], dtype=np.float64)
    m2 = np.asarray(payload["m2"], dtype=np.float64)
    energy = np.asarray(payload["energy"], dtype=np.float64)
    energy_velocity = np.asarray(payload["energy_velocity"], dtype=np.float64)
    if (
        samples <= 0
        or m0.shape != (ncell,)
        or m1.shape != (ncell, 3)
        or m2.shape != (ncell, 3, 3)
        or energy.shape != (ncell,)
        or energy_velocity.shape != (ncell, 3)
        or np.any(m0 <= 0.0)
    ):
        raise ValueError("invalid MV14 additive moment payload")

    velocity = m1 / m0[:, None]
    second = m2 / m0[:, None, None]
    covariance = second - np.einsum("ni,nj->nij", velocity, velocity)
    covariance = 0.5 * (covariance + np.swapaxes(covariance, 1, 2))
    number_density = m0 / float(samples) / float(cfg.cell_volume)
    pressure = float(cfg.vhs.mass) * number_density[:, None, None] * covariance
    mean_speed2 = energy / m0
    mean_energy_velocity = energy_velocity / m0[:, None]
    speed2 = np.sum(velocity**2, axis=1)
    central_energy_velocity = (
        mean_energy_velocity
        - velocity * mean_speed2[:, None]
        - 2.0 * np.einsum("nij,nj->ni", second, velocity)
        + 2.0 * velocity * speed2[:, None]
    )
    heat_flux = 0.5 * float(cfg.vhs.mass) * number_density[:, None] * central_energy_velocity
    temperature = (
        float(cfg.vhs.mass)
        * np.maximum(np.trace(covariance, axis1=1, axis2=2), 0.0)
        / (3.0 * float(kb))
    )

    p_ref = float(cfg.number_density) * float(kb) * float(cfg.t0)
    thermal_speed = math.sqrt(float(kb) * float(cfg.t0) / float(cfg.vhs.mass))
    q_ref = p_ref * thermal_speed
    speed_ref = max(abs(float(cfg.lid_velocity_x)), EPS)
    shape = (int(cfg.ny), int(cfg.nx))
    rho = (number_density / float(cfg.number_density)).reshape(shape)
    temp = (temperature / float(cfg.t0)).reshape(shape)
    result: dict[str, np.ndarray | float] = {
        "rho": rho,
        "u": (velocity[:, 0] / speed_ref).reshape(shape),
        "v": (velocity[:, 1] / speed_ref).reshape(shape),
        "w": (velocity[:, 2] / speed_ref).reshape(shape),
        "temperature": temp,
        "pxx": (pressure[:, 0, 0] / p_ref).reshape(shape),
        "pxy": (pressure[:, 0, 1] / p_ref).reshape(shape),
        "pxz": (pressure[:, 0, 2] / p_ref).reshape(shape),
        "pyy": (pressure[:, 1, 1] / p_ref).reshape(shape),
        "pyz": (pressure[:, 1, 2] / p_ref).reshape(shape),
        "pzz": (pressure[:, 2, 2] / p_ref).reshape(shape),
        "qx": (heat_flux[:, 0] / q_ref).reshape(shape),
        "qy": (heat_flux[:, 1] / q_ref).reshape(shape),
        "m0": m0.reshape(shape),
        "beta": speed_ref / thermal_speed,
        "q_ref": q_ref,
    }
    variance = rho**2 * np.maximum(temp, 1.0e-6) ** 3 / np.maximum(result["m0"], 1.0)
    median = max(float(np.median(variance)), EPS)
    result["qy_variance_proxy"] = np.clip(variance / median, 0.1, 10.0)
    if any(
        isinstance(value, np.ndarray) and not np.all(np.isfinite(value))
        for value in result.values()
    ):
        raise ValueError("nonfinite MV14 kinetic field")
    return result


def _source_index(mv9_root: Path) -> dict[tuple[str, str, int], Path]:
    result: dict[tuple[str, str, int], Path] = {}
    with (Path(mv9_root) / "source_moment_audit.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        for row in csv.DictReader(stream):
            key = (str(row["role"]), str(row["condition_id"]), int(row["seed"]))
            result[key] = Path(row["directory"])
    if not result:
        raise ValueError("empty MV14 raw source index")
    return result


def load_kinetic_dataset(
    mv9_root: Path,
    conditions: np.ndarray,
    identities: np.ndarray,
    images: np.ndarray,
    *,
    role: str,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Load exactly the indexed B=1 block for every prediction sample."""

    mv9 = _mv9_module()
    modules = mv9._project_modules()
    index = _source_index(mv9_root)
    cache: dict[tuple[str, int], tuple[Any, list[Mapping[str, Any]]]] = {}
    rows: dict[str, list[np.ndarray | float]] = {
        key: []
        for key in (
            "rho",
            "u",
            "v",
            "w",
            "temperature",
            "pxx",
            "pxy",
            "pxz",
            "pyy",
            "pyz",
            "pzz",
            "qx",
            "qy",
            "m0",
            "qy_variance_proxy",
            "beta",
            "q_ref",
        )
    }
    maximum_consistency = 0.0
    for condition, identity, image in zip(conditions, identities, images):
        condition_id = str(condition)
        seed, block = int(identity[0]), int(identity[1])
        source_key = (role, condition_id, seed)
        if source_key not in index:
            raise ValueError(f"MV14 source absent: {source_key}")
        cache_key = (condition_id, seed)
        if cache_key not in cache:
            directory = index[source_key]
            summary = mv9._verify_source_artifacts(directory)
            cfg = mv9._config_from_summary(summary)
            checkpoint = modules["load_ntc_checkpoint"](directory / "checkpoint.npz", cfg)
            block_root = checkpoint.block_accumulators
            mapping = block_root.get("block_moments") if isinstance(block_root, Mapping) else None
            if not isinstance(mapping, Mapping) or len(mapping) != 10:
                raise ValueError(f"MV14 requires ten raw blocks: {directory}")
            payloads = [dict(mapping[key]) for key in sorted(mapping)]
            cache[cache_key] = (cfg, payloads)
        cfg, payloads = cache[cache_key]
        if block < 0 or block >= len(payloads):
            raise ValueError(f"MV14 block index out of range: {block}")
        fields = kinetic_fields_from_payload(payloads[block], cfg, modules["KB"])
        reconstructed = np.stack(
            (
                fields["pxy"],
                np.asarray(fields["pxx"]) - np.asarray(fields["pyy"]),
                fields["qx"],
                fields["qy"],
                fields["rho"],
                fields["u"],
                fields["v"],
                fields["temperature"],
            )
        )
        stored = np.asarray(image[:8], dtype=np.float64)
        scale = max(float(np.max(np.abs(stored))), 1.0)
        maximum_consistency = max(
            maximum_consistency,
            float(np.max(np.abs(reconstructed - stored)) / scale),
        )
        for key in rows:
            rows[key].append(fields[key])
    if maximum_consistency > 2.0e-6:
        raise ValueError(
            f"MV14 B1 identity/raw-moment mismatch: {maximum_consistency:.3e}"
        )
    arrays = {key: np.asarray(values, dtype=np.float64) for key, values in rows.items()}
    return arrays, {
        "source_count": float(len(cache)),
        "maximum_B1_raw_moment_consistency_relative_linf": maximum_consistency,
        "only_indexed_B1_block_used_per_prediction": 1.0,
    }


def exact_energy_rhs(
    fields: Mapping[str, np.ndarray | float],
    qx: np.ndarray,
    *,
    macro_smoothing_passes: int,
) -> np.ndarray:
    """Return d(q_y)/dy from the exact steady Boltzmann energy moment."""

    rho = smooth2d(np.asarray(fields["rho"]), macro_smoothing_passes)
    u = smooth2d(np.asarray(fields["u"]), macro_smoothing_passes)
    v = smooth2d(np.asarray(fields["v"]), macro_smoothing_passes)
    temperature = smooth2d(
        np.asarray(fields["temperature"]), macro_smoothing_passes
    )
    pxx = smooth2d(np.asarray(fields["pxx"]), macro_smoothing_passes)
    pxy = smooth2d(np.asarray(fields["pxy"]), macro_smoothing_passes)
    pxz = smooth2d(np.asarray(fields["pxz"]), macro_smoothing_passes)
    pyy = smooth2d(np.asarray(fields["pyy"]), macro_smoothing_passes)
    pyz = smooth2d(np.asarray(fields["pyz"]), macro_smoothing_passes)
    ux, uy = _gradient(u, axis=1), _gradient(u, axis=0)
    vx, vy = _gradient(v, axis=1), _gradient(v, axis=0)
    w = smooth2d(np.asarray(fields["w"]), macro_smoothing_passes)
    wx, wy = _gradient(w, axis=1), _gradient(w, axis=0)
    tx, ty = _gradient(temperature, axis=1), _gradient(temperature, axis=0)
    pressure_work = (
        pxx * ux + pxy * uy + pxy * vx + pyy * vy + pxz * wx + pyz * wy
    )
    internal_advection = 1.5 * rho * (u * tx + v * ty)
    divergence_q = -float(fields["beta"]) * (internal_advection + pressure_work)
    rhs = divergence_q - _gradient(np.asarray(qx, dtype=np.float64), axis=1)
    if not np.all(np.isfinite(rhs)):
        raise ValueError("nonfinite MV14 exact energy right-hand side")
    return rhs


def _weak_operator(ny: int, span_name: str) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    if span_name not in SPAN_SETS:
        raise ValueError(f"unknown MV14 span set: {span_name}")
    pairs: list[tuple[int, int]] = []
    weights: list[float] = []
    for span in SPAN_SETS[span_name]:
        if span >= ny:
            continue
        for start in range(0, ny - span):
            pairs.append((start, start + span))
            weights.append(1.0 / float(span))
    if not pairs:
        raise ValueError("MV14 weak operator has no admissible intervals")
    operator = np.zeros((len(pairs), ny), dtype=np.float64)
    for row, (start, stop) in enumerate(pairs):
        operator[row, start] = -1.0
        operator[row, stop] = 1.0
    return operator, np.asarray(weights), pairs


def _weak_targets(rhs: np.ndarray, pairs: Sequence[tuple[int, int]]) -> np.ndarray:
    rhs = np.asarray(rhs, dtype=np.float64)
    dy = 1.0 / max(rhs.shape[0] - 1, 1)
    targets = []
    for start, stop in pairs:
        interior = np.sum(rhs[start + 1 : stop], axis=0) if stop > start + 1 else 0.0
        targets.append(dy * (0.5 * rhs[start] + interior + 0.5 * rhs[stop]))
    return np.stack(targets)


def weak_gls_project_qy(
    qy_observation: np.ndarray,
    rhs: np.ndarray,
    variance_proxy: np.ndarray,
    *,
    span_name: str,
    lambda_strength: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Softly reconcile direct kinetic q_y with integrated energy balances.

    The observation term anchors the additive constant and every wall-adjacent
    value.  Consequently, this routine imposes neither a wall heat flux nor a
    continuum constitutive relation.
    """

    q0 = np.asarray(qy_observation, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    variance = np.asarray(variance_proxy, dtype=np.float64)
    if q0.shape != rhs.shape or q0.shape != variance.shape:
        raise ValueError("MV14 weak projection fields have different shapes")
    ny, nx = q0.shape
    operator, interval_weights, pairs = _weak_operator(ny, span_name)
    targets = _weak_targets(rhs, pairs)
    weighted_operator = interval_weights[:, None] * operator
    physics_matrix = operator.T @ weighted_operator
    projected = np.empty_like(q0)
    for column in range(nx):
        precision = 1.0 / np.clip(variance[:, column], 0.1, 10.0)
        matrix = np.diag(precision) + float(lambda_strength) * physics_matrix
        right = precision * q0[:, column] + float(lambda_strength) * (
            operator.T @ (interval_weights * targets[:, column])
        )
        projected[:, column] = np.linalg.solve(matrix, right)
    base_defect = operator @ q0 - targets
    projected_defect = operator @ projected - targets
    weighted_base = interval_weights[:, None] * base_defect**2
    weighted_projected = interval_weights[:, None] * projected_defect**2
    correction = (projected - q0) / np.sqrt(np.clip(variance, 0.1, 10.0))
    diagnostics = {
        "base_weak_residual_rms": float(np.sqrt(np.mean(weighted_base))),
        "projected_weak_residual_rms": float(np.sqrt(np.mean(weighted_projected))),
        "weak_residual_ratio": float(
            np.sqrt(np.mean(weighted_projected))
            / max(np.sqrt(np.mean(weighted_base)), EPS)
        ),
        "standardized_observation_correction_rms": float(
            np.sqrt(np.mean(correction**2))
        ),
        "bottom_boundary_correction_rms": float(
            np.sqrt(np.mean((projected[0] - q0[0]) ** 2))
        ),
        "top_boundary_correction_rms": float(
            np.sqrt(np.mean((projected[-1] - q0[-1]) ** 2))
        ),
    }
    return projected, diagnostics


def _sample_fields(dataset: Mapping[str, np.ndarray], index: int) -> dict[str, np.ndarray | float]:
    return {
        key: (float(value[index]) if np.asarray(value[index]).ndim == 0 else value[index])
        for key, value in dataset.items()
    }


def project_dataset(
    kinetic: Mapping[str, np.ndarray],
    qx: np.ndarray,
    qy: np.ndarray,
    *,
    macro_smoothing_passes: int,
    span_name: str,
    lambda_strength: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    values: list[np.ndarray] = []
    diagnostics: dict[str, list[float]] = {
        "base_weak_residual_rms": [],
        "projected_weak_residual_rms": [],
        "weak_residual_ratio": [],
        "standardized_observation_correction_rms": [],
        "bottom_boundary_correction_rms": [],
        "top_boundary_correction_rms": [],
    }
    for index in range(len(qy)):
        fields = _sample_fields(kinetic, index)
        rhs = exact_energy_rhs(
            fields,
            qx[index],
            macro_smoothing_passes=macro_smoothing_passes,
        )
        value, record = weak_gls_project_qy(
            qy[index],
            rhs,
            np.asarray(fields["qy_variance_proxy"]),
            span_name=span_name,
            lambda_strength=lambda_strength,
        )
        values.append(value.astype(np.float32))
        for key in diagnostics:
            diagnostics[key].append(record[key])
    return np.stack(values), {
        key: np.asarray(values, dtype=np.float64) for key, values in diagnostics.items()
    }


def _component_nrmse(candidate: np.ndarray, target: np.ndarray) -> float:
    candidate = np.asarray(candidate, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return float(
        np.sqrt(np.mean((candidate - target) ** 2))
        / max(np.sqrt(np.mean(target**2)), EPS)
    )


def _condition_score(
    prediction: np.ndarray, target: np.ndarray, conditions: np.ndarray
) -> tuple[float, dict[str, float]]:
    values = {
        str(condition): _component_nrmse(
            prediction[conditions == condition], target[conditions == condition]
        )
        for condition in np.unique(conditions)
    }
    return float(np.mean(list(values.values()))), values


def _predict_mamba_validation(
    mv9_root: Path, validation_x: np.ndarray, *, batch_size: int
) -> np.ndarray:
    mv9 = _mv9_module()
    modules = mv9._project_modules()
    import torch

    predictions = []
    for seed in mv9.TRAINING_SEEDS:
        directory = mv9._task_directory(mv9_root, "mambairv2_tiny_adapted", seed)
        _verify_manifest(directory, "artifact_manifest.json")
        checkpoint = torch.load(directory / "model.pt", map_location="cpu", weights_only=False)
        model = modules["mv6"].build_architecture(
            "mambairv2_tiny_adapted",
            int(validation_x.shape[1]),
            out_channels=len(OUTPUT_FIELDS),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        scaling = {key: np.asarray(value) for key, value in checkpoint["scaling"].items()}
        ungated, _ = mv9._predict(model, validation_x, scaling, batch_size)
        raw = validation_x[:, : len(OUTPUT_FIELDS)]
        alpha = float(checkpoint["residual_alpha"])
        predictions.append(raw + alpha * (ungated - raw))
    return np.mean(predictions, axis=0).astype(np.float32)


def _predict_mamba_test(mv9_root: Path) -> np.ndarray:
    mv9 = _mv9_module()
    predictions = []
    for seed in mv9.TRAINING_SEEDS:
        _, prediction = mv9._verify_task(
            mv9._task_directory(mv9_root, "mambairv2_tiny_adapted", seed),
            "mambairv2_tiny_adapted",
            seed,
        )
        predictions.append(prediction)
    return np.mean(predictions, axis=0).astype(np.float32)


def select_arm(
    arm_name: str,
    kinetic: Mapping[str, np.ndarray],
    qx: np.ndarray,
    qy: np.ndarray,
    target_qy: np.ndarray,
    conditions: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    for macro_passes in MACRO_SMOOTHING:
        for span_name in SPAN_SETS:
            for strength in LAMBDA_STRENGTHS:
                prediction, diagnostics = project_dataset(
                    kinetic,
                    qx,
                    qy,
                    macro_smoothing_passes=macro_passes,
                    span_name=span_name,
                    lambda_strength=strength,
                )
                score, by_condition = _condition_score(
                    prediction, target_qy, conditions
                )
                weak_ratio = float(np.sqrt(np.mean(diagnostics["projected_weak_residual_rms"] ** 2)) / max(np.sqrt(np.mean(diagnostics["base_weak_residual_rms"] ** 2)), EPS))
                records.append(
                    {
                        "arm": arm_name,
                        "macro_smoothing_passes": macro_passes,
                        "span_name": span_name,
                        "lambda_strength": strength,
                        "mean_condition_qy_nrmse": score,
                        "condition_qy_nrmse": by_condition,
                        "development_weak_residual_ratio": weak_ratio,
                        "development_weak_residual_gate_passed": weak_ratio
                        <= MAX_DEVELOPMENT_WEAK_RESIDUAL_RATIO,
                    }
                )
    eligible = [row for row in records if row["development_weak_residual_gate_passed"]]
    pool = eligible if eligible else records
    selected = min(
        pool,
        key=lambda row: (
            row["mean_condition_qy_nrmse"],
            row["lambda_strength"],
            row["macro_smoothing_passes"],
            row["span_name"],
        ),
    )
    selected = dict(selected)
    selected["any_development_weak_feasible_candidate"] = bool(eligible)
    leaders = sorted(
        records,
        key=lambda row: (
            not row["development_weak_residual_gate_passed"],
            row["mean_condition_qy_nrmse"],
        ),
    )[:20]
    return selected, leaders


def _apply_selected(
    kinetic: Mapping[str, np.ndarray],
    qx: np.ndarray,
    qy: np.ndarray,
    selected: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    return project_dataset(
        kinetic,
        qx,
        qy,
        macro_smoothing_passes=int(selected["macro_smoothing_passes"]),
        span_name=str(selected["span_name"]),
        lambda_strength=float(selected["lambda_strength"]),
    )


def run_prediction_stage(
    mv9_output_root: Path,
    mv12_output_root: Path,
    output_root: Path,
    *,
    batch_size: int,
) -> dict[str, Any]:
    verify_data_contract(mv9_output_root, mv12_output_root)
    mv9_root = Path(mv9_output_root).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite MV14 output: {output_root}")
    # Prediction intentionally reads development labels but never indexes a
    # legacy test target.  The latter is isolated in run_legacy_post.
    with np.load(mv9_root / "dataset.npz", allow_pickle=False) as data:
        validation_x = np.asarray(data["validation_x"])
        validation_y = np.asarray(data["validation_y"])
        validation_conditions = np.asarray(data["validation_condition"])
        validation_identities = np.asarray(data["validation_identity"])
        test_x = np.asarray(data["test_x"])
        test_conditions = np.asarray(data["test_condition"])
        test_identities = np.asarray(data["test_identity"])
    validation_kinetic, validation_audit = load_kinetic_dataset(
        mv9_root,
        validation_conditions,
        validation_identities,
        validation_x,
        role="development",
    )
    test_kinetic, test_audit = load_kinetic_dataset(
        mv9_root,
        test_conditions,
        test_identities,
        test_x,
        role="confirmatory",
    )
    validation_vision = _predict_mamba_validation(
        mv9_root, validation_x, batch_size=batch_size
    )
    test_vision = _predict_mamba_test(mv9_root)
    validation_raw = validation_x[:, : len(OUTPUT_FIELDS)]
    test_raw = test_x[:, : len(OUTPUT_FIELDS)]
    target_qy = validation_y[:, QY_INDEX]

    physics_selected, physics_leaders = select_arm(
        "physics_only",
        validation_kinetic,
        validation_raw[:, QX_INDEX],
        validation_raw[:, QY_INDEX],
        target_qy,
        validation_conditions,
    )
    hybrid_selected, hybrid_leaders = select_arm(
        "vision_plus_kinetic_physics",
        validation_kinetic,
        validation_vision[:, QX_INDEX],
        validation_vision[:, QY_INDEX],
        target_qy,
        validation_conditions,
    )
    physics_qy, physics_diagnostics = _apply_selected(
        test_kinetic,
        test_raw[:, QX_INDEX],
        test_raw[:, QY_INDEX],
        physics_selected,
    )
    hybrid_qy, hybrid_diagnostics = _apply_selected(
        test_kinetic,
        test_vision[:, QX_INDEX],
        test_vision[:, QY_INDEX],
        hybrid_selected,
    )

    output_root.mkdir(parents=True)
    np.savez_compressed(
        output_root / "locked_predictions.npz",
        identity_condition=test_conditions,
        identity_numeric=test_identities,
        raw_b1_qy=test_raw[:, QY_INDEX],
        vision_only_fields=test_vision,
        vision_only_qx=test_vision[:, QX_INDEX],
        vision_only_qy=test_vision[:, QY_INDEX],
        physics_only_qy=physics_qy,
        hybrid_qy=hybrid_qy,
        physics_base_weak_residual_rms=physics_diagnostics["base_weak_residual_rms"],
        physics_projected_weak_residual_rms=physics_diagnostics["projected_weak_residual_rms"],
        hybrid_base_weak_residual_rms=hybrid_diagnostics["base_weak_residual_rms"],
        hybrid_projected_weak_residual_rms=hybrid_diagnostics["projected_weak_residual_rms"],
    )
    selection_summary = {
        "stage": STAGE,
        "status": "complete_MV14_development_selection_and_legacy_prediction",
        "protocol_sha256": _sha256(protocol_path()),
        "mv9_output_root": str(mv9_root),
        "mv12_output_root": str(Path(mv12_output_root).resolve()),
        "physics_only_selected": physics_selected,
        "hybrid_selected": hybrid_selected,
        "physics_only_top_20": physics_leaders,
        "hybrid_top_20": hybrid_leaders,
        "validation_raw_moment_audit": validation_audit,
        "test_raw_moment_audit": test_audit,
        "ablation_arms_locked": [
            "raw_B1",
            "vision_only",
            "physics_only",
            "vision_plus_kinetic_physics",
            "raw_B10",
        ],
        "legacy_test_targets_loaded": False,
        "legacy_prediction_count": int(len(test_x)),
        "decision": "lock_MV14_predictions_before_legacy_ablation",
    }
    _atomic_json(output_root / "selection_summary.json", selection_summary)
    (output_root / PROTOCOL_FILE).write_bytes(protocol_path().read_bytes())
    _atomic_json(
        output_root / "prediction_manifest.json",
        {
            "stage": STAGE,
            "files": {
                name: {
                    "sha256": _sha256(output_root / name),
                    "size_bytes": (output_root / name).stat().st_size,
                }
                for name in (
                    "locked_predictions.npz",
                    "selection_summary.json",
                    PROTOCOL_FILE,
                )
            },
        },
    )
    return selection_summary


def _per_seed_metrics(
    candidate: np.ndarray,
    target: np.ndarray,
    conditions: np.ndarray,
    identities: np.ndarray,
) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for condition in np.unique(conditions):
        condition_key = str(condition)
        result[condition_key] = {}
        mask_condition = conditions == condition
        for seed in np.unique(identities[mask_condition, 0]):
            mask = mask_condition & (identities[:, 0] == seed)
            field_values = {
                name: _component_nrmse(candidate[mask, index], target[mask, index])
                for index, name in enumerate(OUTPUT_FIELDS)
            }
            result[condition_key][str(int(seed))] = {
                "per_field_nrmse": field_values,
                "composite_nrmse": float(np.mean(list(field_values.values()))),
                "heat_flux_composite_nrmse": float(
                    np.mean((field_values[OUTPUT_FIELDS[QX_INDEX]], field_values[OUTPUT_FIELDS[QY_INDEX]]))
                ),
            }
    return result


def _aggregate_per_seed(records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    fields = {
        name: float(
            np.mean([record["per_field_nrmse"][name] for record in records.values()])
        )
        for name in OUTPUT_FIELDS
    }
    return {
        "seed_count": len(records),
        "mean_per_field_nrmse": fields,
        "mean_composite_nrmse": float(
            np.mean([record["composite_nrmse"] for record in records.values()])
        ),
        "mean_heat_flux_composite_nrmse": float(
            np.mean(
                [record["heat_flux_composite_nrmse"] for record in records.values()]
            )
        ),
    }


def _shape_diagnostics(candidate: np.ndarray, target: np.ndarray) -> dict[str, float]:
    candidate = np.asarray(candidate, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    first = candidate.ravel()
    second = target.ravel()
    first_centered = first - np.mean(first)
    second_centered = second - np.mean(second)
    correlation = float(
        np.dot(first_centered, second_centered)
        / max(np.linalg.norm(first_centered) * np.linalg.norm(second_centered), EPS)
    )
    amplitude = float(np.dot(first, second) / max(np.dot(second, second), EPS))
    top = max(1, target.shape[-2] // 4)
    top_candidate = candidate[..., -top:, :].ravel()
    top_target = target[..., -top:, :].ravel()
    return {
        "centered_pattern_correlation": correlation,
        "least_squares_amplitude_ratio": amplitude,
        "top_quarter_least_squares_amplitude_ratio": float(
            np.dot(top_candidate, top_target)
            / max(np.dot(top_target, top_target), EPS)
        ),
        "mean_bias_over_target_rms": float(
            np.mean(candidate - target) / max(np.sqrt(np.mean(target**2)), EPS)
        ),
    }


def _qy_figure(
    output_root: Path,
    methods: Mapping[str, np.ndarray],
    reference: np.ndarray,
    q_scale: float,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    plt.rcParams.update(
        {"font.family": "serif", "font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42}
    )
    columns = (
        "reference",
        "raw_b1",
        "vision_only",
        "physics_only",
        "hybrid",
        "raw_b10",
    )
    titles = (
        "Reference",
        "Raw DSMC\nB=1",
        "Vision only\nB=1",
        "Kinetic physics only\nB=1",
        "Vision + kinetic physics\nB=1",
        "Raw DSMC\nB=10",
    )
    normalized = {"reference": reference, **methods}
    physical = {name: normalized[name] * q_scale for name in columns}
    errors = {
        name: 100.0 * (normalized[name] - reference)
        for name in columns
        if name != "reference"
    }
    physical_limit = max(
        float(np.quantile(np.concatenate([np.abs(value).ravel() for value in physical.values()]), 0.995)),
        1.0e-12,
    )
    error_limit = max(
        float(np.quantile(np.concatenate([np.abs(value).ravel() for value in errors.values()]), 0.995)),
        1.0e-4,
    )
    physical_norm = TwoSlopeNorm(vmin=-physical_limit, vcenter=0.0, vmax=physical_limit)
    error_norm = TwoSlopeNorm(vmin=-error_limit, vcenter=0.0, vmax=error_limit)
    fig, axes = plt.subplots(2, len(columns), figsize=(14.8, 5.4), constrained_layout=True)
    top_artist = bottom_artist = None
    for column, (name, title) in enumerate(zip(columns, titles)):
        top_artist = axes[0, column].contourf(
            physical[name],
            levels=np.linspace(-physical_limit, physical_limit, 41),
            cmap="RdBu_r",
            norm=physical_norm,
            extend="both",
        )
        error = np.zeros_like(reference) if name == "reference" else errors[name]
        bottom_artist = axes[1, column].contourf(
            error,
            levels=np.linspace(-error_limit, error_limit, 41),
            cmap="RdBu_r",
            norm=error_norm,
            extend="both",
        )
        axes[0, column].set_title(title)
        for row in range(2):
            axis = axes[row, column]
            axis.set_aspect("equal")
            axis.set_xticks((0, (reference.shape[1] - 1) / 2, reference.shape[1] - 1))
            axis.set_yticks((0, (reference.shape[0] - 1) / 2, reference.shape[0] - 1))
            axis.set_xticklabels(
                ("0", "0.5", "1") if row == 1 else ("", "", "")
            )
            axis.set_yticklabels(
                ("0", "0.5", "1") if column == 0 else ("", "", "")
            )
            if row == 1:
                axis.set_xlabel(r"$x/L$")
            if column == 0:
                axis.set_ylabel(r"$y/L$")
    assert top_artist is not None and bottom_artist is not None
    first = fig.colorbar(top_artist, ax=axes[0, :], shrink=0.88, pad=0.012)
    first.set_label(r"$q_y$ [W m$^{-2}$]")
    second = fig.colorbar(bottom_artist, ax=axes[1, :], shrink=0.88, pad=0.012)
    second.set_label(r"$100\,\Delta q_y/q_{ref}$ [%]")
    png = output_root / "mv14_qy_five_arm_physical_contours.png"
    pdf = output_root / "mv14_qy_five_arm_physical_contours.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "png": png.name,
        "pdf": pdf.name,
        "physical_limit": physical_limit,
        "error_percent_limit": error_limit,
    }


def _full_cavity_figure(
    output_root: Path,
    methods: Mapping[str, np.ndarray],
    reference: np.ndarray,
    scales: np.ndarray,
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    columns = ("reference", "raw_b1", "vision_only", "physics_only", "hybrid", "raw_b10")
    titles = ("Reference", "Raw B=1", "Vision only", "Physics only", "Hybrid", "Raw B=10")
    row_titles = (r"$P_{xy}$ [Pa]", r"$P_{xx}-P_{yy}$ [Pa]", r"$q_x$ [W m$^{-2}$]", r"$q_y$ [W m$^{-2}$]")
    all_methods = {"reference": reference, **methods}
    fig, axes = plt.subplots(4, len(columns), figsize=(14.5, 9.2), constrained_layout=True)
    for row in range(4):
        physical = {name: all_methods[name][row] * float(scales[row]) for name in columns}
        limit = max(
            float(np.quantile(np.concatenate([np.abs(value).ravel() for value in physical.values()]), 0.995)),
            1.0e-12,
        )
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        artist = None
        for column, (name, title) in enumerate(zip(columns, titles)):
            artist = axes[row, column].contourf(
                physical[name], levels=np.linspace(-limit, limit, 41), cmap="RdBu_r", norm=norm, extend="both"
            )
            axes[row, column].set_aspect("equal")
            axes[row, column].set_xticks(())
            axes[row, column].set_yticks(())
            if row == 0:
                axes[row, column].set_title(title)
            if column == 0:
                axes[row, column].set_ylabel(row_titles[row])
        assert artist is not None
        fig.colorbar(artist, ax=axes[row, :], shrink=0.76, pad=0.01)
    png = output_root / "mv14_full_cavity_five_arm_fields.png"
    pdf = output_root / "mv14_full_cavity_five_arm_fields.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"png": png.name, "pdf": pdf.name}


def run_legacy_post(output_root: Path) -> dict[str, Any]:
    """Evaluate hash-locked five-arm predictions on legacy labels."""

    protocol = locked_protocol()
    output_root = Path(output_root).resolve()
    _verify_manifest(output_root, "prediction_manifest.json")
    selection = json.loads((output_root / "selection_summary.json").read_text(encoding="utf-8"))
    if selection.get("legacy_test_targets_loaded") is not False:
        raise ValueError("MV14 prediction did not preserve label separation")
    mv9_root = Path(selection["mv9_output_root"]).resolve()
    with np.load(output_root / "locked_predictions.npz", allow_pickle=False) as data:
        locked_conditions = np.asarray(data["identity_condition"])
        locked_identities = np.asarray(data["identity_numeric"])
        raw_qy = np.asarray(data["raw_b1_qy"])
        vision_fields = np.asarray(data["vision_only_fields"])
        vision_qx = np.asarray(data["vision_only_qx"])
        vision_qy = np.asarray(data["vision_only_qy"])
        physics_qy = np.asarray(data["physics_only_qy"])
        hybrid_qy = np.asarray(data["hybrid_qy"])
        physics_base_residual = np.asarray(data["physics_base_weak_residual_rms"])
        physics_projected_residual = np.asarray(data["physics_projected_weak_residual_rms"])
        hybrid_base_residual = np.asarray(data["hybrid_base_weak_residual_rms"])
        hybrid_projected_residual = np.asarray(data["hybrid_projected_weak_residual_rms"])
    with np.load(mv9_root / "dataset.npz", allow_pickle=False) as data:
        test_x = np.asarray(data["test_x"])
        test_y = np.asarray(data["test_y"])
        conditions = np.asarray(data["test_condition"])
        identities = np.asarray(data["test_identity"])
        scales = np.asarray(data["test_scale"])
        raw10 = np.asarray(data["test_raw10"])
        target10 = np.asarray(data["test_target10"])
        conditions10 = np.asarray(data["test_condition10"])
        identities10 = np.asarray(data["test_identity10"])
        scales10 = np.asarray(data["test_scale10"])
    if not np.array_equal(conditions, locked_conditions) or not np.array_equal(identities, locked_identities):
        raise ValueError("MV14 locked prediction identities differ from legacy data")
    raw = test_x[:, : len(OUTPUT_FIELDS)]
    if not np.array_equal(raw[:, QY_INDEX], raw_qy):
        raise ValueError("MV14 raw B1 qy changed after prediction lock")
    if (
        vision_fields.shape != raw.shape
        or not np.array_equal(vision_fields[:, QX_INDEX], vision_qx)
        or not np.array_equal(vision_fields[:, QY_INDEX], vision_qy)
    ):
        raise ValueError("MV14 vision-only field lock is inconsistent")
    vision = vision_fields.copy()
    physics = raw.copy()
    physics[:, QY_INDEX] = physics_qy
    hybrid = vision.copy()
    hybrid[:, QY_INDEX] = hybrid_qy
    methods = {
        "raw_b1": raw,
        "vision_only": vision,
        "physics_only": physics,
        "hybrid": hybrid,
    }
    per_seed = {
        name: _per_seed_metrics(value, test_y, conditions, identities)
        for name, value in methods.items()
    }
    per_seed["raw_b10"] = _per_seed_metrics(raw10, target10, conditions10, identities10)
    aggregates = {
        method: {
            condition: _aggregate_per_seed(seed_records)
            for condition, seed_records in condition_records.items()
        }
        for method, condition_records in per_seed.items()
    }
    primary = str(protocol["analysis_contract"]["primary_condition"])
    qy_name = OUTPUT_FIELDS[QY_INDEX]
    qy_ratios = {
        method: aggregates[method][primary]["mean_per_field_nrmse"][qy_name]
        / max(aggregates["raw_b10"][primary]["mean_per_field_nrmse"][qy_name], EPS)
        for method in methods
    }
    hybrid_vs_components = {
        "hybrid_qy_nrmse_ratio_to_vision_only": aggregates["hybrid"][primary]["mean_per_field_nrmse"][qy_name]
        / max(aggregates["vision_only"][primary]["mean_per_field_nrmse"][qy_name], EPS),
        "hybrid_qy_nrmse_ratio_to_physics_only": aggregates["hybrid"][primary]["mean_per_field_nrmse"][qy_name]
        / max(aggregates["physics_only"][primary]["mean_per_field_nrmse"][qy_name], EPS),
    }
    condition_ratios = {
        condition: aggregates["hybrid"][condition]["mean_per_field_nrmse"][qy_name]
        / max(aggregates["raw_b10"][condition]["mean_per_field_nrmse"][qy_name], EPS)
        for condition in aggregates["hybrid"]
    }
    primary_seed_ratios = {
        seed: record["per_field_nrmse"][qy_name]
        / max(per_seed["raw_b10"][primary][seed]["per_field_nrmse"][qy_name], EPS)
        for seed, record in per_seed["hybrid"][primary].items()
    }
    physics_residual_ratio = float(np.sqrt(np.mean(physics_projected_residual**2)) / max(np.sqrt(np.mean(physics_base_residual**2)), EPS))
    hybrid_residual_ratio = float(np.sqrt(np.mean(hybrid_projected_residual**2)) / max(np.sqrt(np.mean(hybrid_base_residual**2)), EPS))
    contract = protocol["analysis_contract"]
    gates = {
        "hybrid_primary_qy_no_worse_than_Raw_B10": qy_ratios["hybrid"] <= float(contract["maximum_primary_mean_qy_ratio_to_raw_B10"]),
        "hybrid_beats_vision_only": hybrid_vs_components["hybrid_qy_nrmse_ratio_to_vision_only"] < 1.0,
        "hybrid_beats_physics_only": hybrid_vs_components["hybrid_qy_nrmse_ratio_to_physics_only"] < 1.0,
        "every_primary_seed_qy_within_cap": max(primary_seed_ratios.values()) <= float(contract["maximum_individual_seed_qy_ratio_to_raw_B10"]),
        "no_condition_mean_qy_worse_than_Raw_B10": max(condition_ratios.values()) <= float(contract["maximum_each_condition_mean_qy_ratio_to_raw_B10"]),
        "physics_only_weak_balance_reduced": physics_residual_ratio <= float(contract["maximum_weak_residual_ratio"]),
        "hybrid_weak_balance_reduced": hybrid_residual_ratio <= float(contract["maximum_weak_residual_ratio"]),
        "prediction_hash_locked_before_legacy_label_access": True,
        "legacy_diagnostic_not_reclassified_as_confirmation": True,
    }

    metrics_path = output_root / "mv14_five_arm_legacy_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("condition", "method", "mean_composite_nrmse", "mean_heat_flux_composite_nrmse", *OUTPUT_FIELDS))
        for method in sorted(aggregates):
            for condition in sorted(aggregates[method]):
                record = aggregates[method][condition]
                writer.writerow((condition, method, record["mean_composite_nrmse"], record["mean_heat_flux_composite_nrmse"], *(record["mean_per_field_nrmse"][field] for field in OUTPUT_FIELDS)))

    representative_seed = int(contract["representative_seed"])
    representative_block = int(contract["representative_block"])
    mask = (conditions == primary) & (identities[:, 0] == representative_seed) & (identities[:, 1] == representative_block)
    mask10 = (conditions10 == primary) & (identities10[:, 0] == representative_seed)
    if np.count_nonzero(mask) != 1 or np.count_nonzero(mask10) != 1:
        raise ValueError("MV14 representative identity is absent")
    index, index10 = int(np.flatnonzero(mask)[0]), int(np.flatnonzero(mask10)[0])
    if not np.array_equal(test_y[index], target10[index10]):
        raise ValueError("MV14 representative B1/B10 targets differ")
    if not np.allclose(scales[index], scales10[index10], rtol=1.0e-12, atol=0.0):
        raise ValueError("MV14 representative B1/B10 scales differ")
    qy_figure = _qy_figure(
        output_root,
        {
            "raw_b1": raw[index, QY_INDEX],
            "vision_only": vision[index, QY_INDEX],
            "physics_only": physics[index, QY_INDEX],
            "hybrid": hybrid[index, QY_INDEX],
            "raw_b10": raw10[index10, QY_INDEX],
        },
        test_y[index, QY_INDEX],
        float(scales[index, QY_INDEX]),
    )
    full_figure = _full_cavity_figure(
        output_root,
        {
            "raw_b1": raw[index],
            "vision_only": vision[index],
            "physics_only": physics[index],
            "hybrid": hybrid[index],
            "raw_b10": raw10[index10],
        },
        test_y[index],
        scales[index],
    )
    primary_mask = conditions == primary
    shape = {
        name: _shape_diagnostics(value[primary_mask, QY_INDEX], test_y[primary_mask, QY_INDEX])
        for name, value in methods.items()
    }
    supports_fresh = all(gates.values())
    summary = {
        "stage": STAGE,
        "status": "complete_MV14_post_lock_five_arm_legacy_diagnostic",
        "protocol_sha256": _sha256(protocol_path()),
        "scientific_classification": protocol["scientific_role"]["classification"],
        "primary_condition": primary,
        "primary_qy_ratios_to_raw_B10": qy_ratios,
        "hybrid_vs_single_component_ablation": hybrid_vs_components,
        "primary_hybrid_per_seed_qy_ratios_to_raw_B10": primary_seed_ratios,
        "all_condition_hybrid_qy_ratios_to_raw_B10": condition_ratios,
        "physics_only_weak_residual_ratio": physics_residual_ratio,
        "hybrid_weak_residual_ratio": hybrid_residual_ratio,
        "primary_qy_shape_diagnostics": shape,
        "legacy_diagnostic_aggregates": aggregates,
        "gates_for_authorizing_fresh_seed_confirmation": gates,
        "qy_figure_record": qy_figure,
        "full_cavity_figure_record": full_figure,
        "old_evaluation_seeds_are_confirmation": False,
        "fresh_unobserved_seed_confirmation_still_required": True,
        "machine_vision_contribution_demonstrated": bool(
            gates["hybrid_beats_vision_only"] and gates["hybrid_beats_physics_only"]
        ),
        "decision": (
            "MV14_hybrid_legacy_diagnostic_supports_fresh_seed_confirmation"
            if supports_fresh
            else "MV14_hybrid_legacy_diagnostic_does_not_support_fresh_seed_confirmation"
        ),
    }
    _atomic_json(output_root / "summary.json", summary)
    return summary


def package_results(output_root: Path, return_directory: Path) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    return_directory = Path(return_directory).resolve()
    _verify_manifest(output_root, "prediction_manifest.json")
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    generated = [
        output_root / "selection_summary.json",
        output_root / "prediction_manifest.json",
        output_root / "summary.json",
        output_root / PROTOCOL_FILE,
        output_root / "mv14_five_arm_legacy_metrics.csv",
        output_root / "mv14_qy_five_arm_physical_contours.png",
        output_root / "mv14_qy_five_arm_physical_contours.pdf",
        output_root / "mv14_full_cavity_five_arm_fields.png",
        output_root / "mv14_full_cavity_five_arm_fields.pdf",
    ]
    accounting = output_root / "slurm_accounting.psv"
    if accounting.is_file():
        generated.append(accounting)
    manifest = {
        "stage": STAGE,
        "status": "complete_MV14_return_artifact_manifest",
        "files": {
            str(path.relative_to(output_root)): {
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in generated
        },
    }
    _atomic_json(output_root / "artifact_manifest.json", manifest)
    _verify_manifest(output_root, "artifact_manifest.json")
    verification = {
        "stage": STAGE,
        "status": "complete_MV14_recursive_return_verification",
        "decision": "verified",
        "verified_file_count": len(manifest["files"]),
        "manifest_sha256": _sha256(output_root / "artifact_manifest.json"),
    }
    _atomic_json(output_root / "verification.json", verification)
    generated.extend((output_root / "artifact_manifest.json", output_root / "verification.json"))
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return_directory.mkdir(parents=True, exist_ok=True)
    archive = return_directory / f"MV14_KINETIC_CONSERVATION_CAVITY_ANALYSIS_BUNDLE_{tag}.zip"
    if archive.exists():
        raise FileExistsError(f"refusing to overwrite MV14 archive: {archive}")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as stream:
        for path in generated:
            stream.write(path, arcname=str(path.relative_to(output_root)))
    if archive.stat().st_size > 450 * 1024 * 1024:
        raise RuntimeError("MV14 return archive exceeds 450 MiB")
    result = {
        "stage": STAGE,
        "archive": str(archive),
        "archive_sha256": _sha256(archive),
        "decision": summary["decision"],
        "primary_hybrid_qy_ratio_to_raw_B10": summary["primary_qy_ratios_to_raw_B10"]["hybrid"],
    }
    _atomic_json(output_root / "return.json", result)
    pointer = return_directory / "LAST_MOHAMMADZADEH_MV14_KINETIC_CAVITY_RESULT.env"
    pointer.write_text(
        "\n".join(
            (
                f"MV14_OUTPUT_ROOT={output_root}",
                f"MV14_RESULT_ARCHIVE={archive}",
                f"MV14_RESULT_ARCHIVE_SHA256={result['archive_sha256']}",
                f"MV14_DECISION={result['decision']}",
                f"MV14_PRIMARY_HYBRID_QY_RATIO_TO_RAW_B10={result['primary_hybrid_qy_ratio_to_raw_B10']}",
                "",
            )
        ),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify-lock")
    verify.set_defaults(action="verify")
    verify_data = subparsers.add_parser("verify-data")
    verify_data.add_argument("--mv9-output-root", required=True, type=Path)
    verify_data.add_argument("--mv12-output-root", required=True, type=Path)
    verify_data.set_defaults(action="verify_data")
    predict = subparsers.add_parser("predict")
    predict.add_argument("--mv9-output-root", required=True, type=Path)
    predict.add_argument("--mv12-output-root", required=True, type=Path)
    predict.add_argument("--output-root", required=True, type=Path)
    predict.add_argument("--batch-size", type=int, default=8)
    predict.set_defaults(action="predict")
    post = subparsers.add_parser("post")
    post.add_argument("--output-root", required=True, type=Path)
    post.set_defaults(action="post")
    package = subparsers.add_parser("package")
    package.add_argument("--output-root", required=True, type=Path)
    package.add_argument("--return-directory", required=True, type=Path)
    package.set_defaults(action="package")
    args = parser.parse_args()
    if args.action == "verify":
        result = verify_lock()
    elif args.action == "verify_data":
        result = verify_data_contract(args.mv9_output_root, args.mv12_output_root)
    elif args.action == "predict":
        result = run_prediction_stage(
            args.mv9_output_root,
            args.mv12_output_root,
            args.output_root,
            batch_size=args.batch_size,
        )
    elif args.action == "post":
        result = run_legacy_post(args.output_root)
    else:
        result = package_results(args.output_root, args.return_directory)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
