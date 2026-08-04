"""Aggregate and verify the locked eight-seed M3 heat-flux precision stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from .mohammadzadeh_qy_precision import LOCK_FILE, STAGE, stage_configuration
from .mohammadzadeh_validation import (
    _profile_at_x,
    _profile_at_y,
    evaluate_mohammadzadeh_fields,
)


SEEDS = tuple(range(91901, 91909))
STUDENT_T_95_DF7 = 2.3646242510102993
REQUIRED_SEED_FILES = (
    "checkpoint.npz",
    "fields.npz",
    "block_fields.npz",
    "summary.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(_json_ready(value), stream, allow_nan=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_save_figure(figure: Any, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=path.suffix, dir=path.parent)
    os.close(handle)
    try:
        figure.savefig(temporary, dpi=220, metadata={"Title": title})
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _strict_levels(low: float, high: float, count: int = 21) -> np.ndarray:
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("contour limits must be finite")
    if high <= low:
        pad = max(abs(low), 1.0) * 1.0e-12
        low, high = low - pad, high + pad
    return np.linspace(low, high, count)


def _student_t_statistics(replicates: np.ndarray) -> dict[str, Any]:
    """Return the preregistered eight-seed mean, SE, and 95% t interval."""
    values = np.asarray(replicates, dtype=np.float64)
    if values.ndim < 1 or values.shape[0] != len(SEEDS):
        raise ValueError("M3 statistics require exactly eight independent seeds")
    if not np.all(np.isfinite(values)):
        raise ValueError("M3 ensemble replicates must be finite")
    mean = np.mean(values, axis=0)
    sample_standard_deviation = np.std(values, axis=0, ddof=1)
    standard_error = sample_standard_deviation / np.sqrt(float(len(SEEDS)))
    half_width = STUDENT_T_95_DF7 * standard_error
    denominator = max(float(np.linalg.norm(mean.ravel())), 1.0e-300)
    return {
        "replicate_count": len(SEEDS),
        "degrees_of_freedom": len(SEEDS) - 1,
        "ci95_critical_value": STUDENT_T_95_DF7,
        "mean": mean,
        "sample_standard_deviation": sample_standard_deviation,
        "standard_error": standard_error,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "global_relative_standard_error": float(
            np.linalg.norm(standard_error.ravel()) / denominator
        ),
    }


def _load_seed(directory: Path, seed: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (directory / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    if summary.get("seed") != seed or summary.get("stage") != STAGE:
        raise ValueError(f"M3 identity mismatch for seed {seed}")
    if summary.get("status") != "complete_M3_qy_precision_seed":
        raise ValueError(f"M3 seed {seed} is incomplete")
    if manifest.get("status") != summary["status"]:
        raise ValueError(f"M3 manifest status mismatch for seed {seed}")
    for name in REQUIRED_SEED_FILES:
        path = directory / name
        record = manifest.get("files", {}).get(name, {})
        if (
            record.get("sha256") != _sha256(path)
            or record.get("size_bytes") != path.stat().st_size
        ):
            raise ValueError(f"M3 artifact mismatch for seed {seed}/{name}")
    expected_lock = stage_configuration(STAGE, seed)[4]
    if summary.get("lock_hashes") != expected_lock:
        raise ValueError(f"M3 lock hashes mismatch for seed {seed}")
    with np.load(directory / "fields.npz", allow_pickle=False) as archive:
        fields = {name: np.asarray(archive[name]).copy() for name in archive.files}
    return fields, summary


def _profile_statistics(
    replicates: Mapping[str, np.ndarray], cfg: Any
) -> tuple[dict[str, dict[str, Any]], float]:
    raw_qy = np.stack([_profile_at_y(field, 0.8) for field in replicates["qy"]])
    qy_scale = float(np.max(np.mean(raw_qy, axis=0)))
    if not np.isfinite(qy_scale) or qy_scale <= 0.0:
        raise ValueError("M3 qy normalization scale must be positive")
    values = {
        "macroscopic_lid_slip": 1.0
        - replicates["u"][:, -1, :] / cfg.lid_velocity_x,
        "microscopic_lid_slip": replicates[
            "microscopic_lid_slip_over_uwall"
        ],
        "macroscopic_lid_temperature_K": replicates["T"][:, -1, :],
        "microscopic_lid_temperature_K": replicates["microscopic_lid_T"],
        "vertical_temperature_x08_K": np.stack(
            [_profile_at_x(field, 0.8) for field in replicates["T"]]
        ),
        "normalized_qy_y08": raw_qy / qy_scale,
    }
    return {
        name: _student_t_statistics(values)
        for name, values in values.items()
    }, qy_scale


def _compute(input_root: Path) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    loaded = [_load_seed(input_root / f"seed_{seed}", seed) for seed in SEEDS]
    names = sorted(set.intersection(*(set(item[0]) for item in loaded)))
    replicates = {
        name: np.stack([item[0][name] for item in loaded]) for name in names
    }
    field_statistics = {
        name: _student_t_statistics(values)
        for name, values in replicates.items()
    }
    means = {name: stats["mean"] for name, stats in field_statistics.items()}
    cfg, protocol, _, _, lock_hashes = stage_configuration(STAGE, SEEDS[0])
    profiles, qy_scale = _profile_statistics(replicates, cfg)
    evaluation = evaluate_mohammadzadeh_fields(means, cfg)
    metrics = evaluation["metrics"]
    mechanics = [
        all(
            value
            for key, value in item[1]["mechanical_checks"].items()
            if key != "stationarity_pass"
        )
        for item in loaded
    ]
    stationarity = [
        bool(item[1]["mechanical_checks"]["stationarity_pass"])
        for item in loaded
    ]
    gates = protocol["statistical_gates"]
    checks = {
        "all_seed_mechanics": all(mechanics),
        "all_seed_stationarity": all(stationarity),
        "normalized_qy_profile_global_rse": profiles["normalized_qy_y08"][
            "global_relative_standard_error"
        ]
        <= float(gates["normalized_qy_profile_global_rse_max"]),
        "qy_nrmse": metrics["fig9_qy_nrmse_over_reference_rms"]
        <= float(gates["qy_nrmse_max"]),
        "qy_sign_agreement": metrics["fig9_qy_sign_agreement"]
        >= float(gates["qy_sign_agreement_min"]),
    }
    qy_resolved = all(checks.values())
    summary = {
        "stage": "M3_QY100_precision_ensemble",
        "status": "complete_locked_eight_seed_precision_aggregation",
        "external_validation_claim": False,
        "seeds": list(SEEDS),
        "mechanics_pass_by_seed": mechanics,
        "stationarity_pass_by_seed": stationarity,
        "lock_file": LOCK_FILE,
        "lock_hashes": lock_hashes,
        "evaluation": {
            key: value for key, value in evaluation.items()
            if key != "comparison_arrays"
        },
        "profile_global_rse": {
            name: stats["global_relative_standard_error"]
            for name, stats in profiles.items()
        },
        "field_global_rse": {
            name: field_statistics[name]["global_relative_standard_error"]
            for name in ("T", "rho", "u", "v", "qy")
        },
        "normalized_qy_scale": qy_scale,
        "preregistered_checks": checks,
        "all_preregistered_qy_precision_checks_pass": qy_resolved,
        "decision": (
            "qy_precision_resolved_reassess_remaining_temperature_error_and_R200_need"
            if qy_resolved
            else "hold_R200_and_diagnose_remaining_qy_precision_or_model_mismatch"
        ),
        "visualization_contract": {
            "bar_charts": False,
            "smoothing": False,
            "line_profiles": "article_DSMC_vs_M3_R100_mean_with_95pct_CI",
            "contours": "unsmoothed_M3_R100_temperature_and_qy",
        },
    }
    arrays: dict[str, np.ndarray] = {
        "seeds": np.asarray(SEEDS, dtype=np.int64),
        "x_centers": (np.arange(100) + 0.5) / 100,
        "y_centers": (np.arange(100) + 0.5) / 100,
        "normalized_qy_scale": np.asarray(qy_scale),
    }
    for name, stats in field_statistics.items():
        for statistic in ("mean", "standard_error", "ci95_low", "ci95_high"):
            arrays[f"field__{name}__{statistic}"] = np.asarray(stats[statistic])
    for name, stats in profiles.items():
        for statistic in ("mean", "standard_error", "ci95_low", "ci95_high"):
            arrays[f"profile__{name}__{statistic}"] = np.asarray(stats[statistic])
    return summary, arrays, evaluation["comparison_arrays"]


def _draw_profiles(arrays: Mapping[str, np.ndarray], reference: Mapping[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    x = np.asarray(arrays["x_centers"])
    y = np.asarray(arrays["y_centers"])
    figure, axes = plt.subplots(2, 3, figsize=(14.4, 8.4), constrained_layout=True)

    def band(axis: Any, coordinate: np.ndarray, prefix: str, *, vertical: bool = False) -> None:
        low = arrays[f"profile__{prefix}__ci95_low"]
        high = arrays[f"profile__{prefix}__ci95_high"]
        if vertical:
            axis.fill_betweenx(coordinate, low, high, color="tab:blue", alpha=0.16, linewidth=0)
        else:
            axis.fill_between(coordinate, low, high, color="tab:blue", alpha=0.16, linewidth=0)

    panels = (
        ("macroscopic_lid_slip", "Macroscopic lid slip", r"$u_{slip}/U_{wall}$", "reference_slip_x", "reference_slip"),
        ("microscopic_lid_slip", "Microscopic lid slip", r"$u_{slip}/U_{wall}$", "reference_microscopic_slip_x", "reference_microscopic_slip"),
        ("macroscopic_lid_temperature_K", "Macroscopic lid temperature", "T (K)", "reference_lid_temperature_x", "reference_lid_temperature_K"),
        ("microscopic_lid_temperature_K", "Microscopic lid temperature", "T (K)", "reference_microscopic_lid_temperature_x", "reference_microscopic_lid_temperature_K"),
    )
    for axis, panel in zip(axes.ravel()[:4], panels):
        key, title, ylabel, ref_x, ref_y = panel
        axis.plot(reference[ref_x], reference[ref_y], "o", color="0.1", ms=2.8, label="Article DSMC")
        axis.plot(x, arrays[f"profile__{key}__mean"], color="tab:blue", lw=1.8, label="M3 100×100 mean")
        band(axis, x, key)
        axis.set(title=title, xlabel="x/L", ylabel=ylabel, xlim=(0, 1))
    key = "vertical_temperature_x08_K"
    axes[1, 1].plot(reference["reference_vertical_temperature_K"], reference["reference_vertical_temperature_y"], color="0.1", lw=1.7, label="Article DSMC")
    axes[1, 1].plot(arrays[f"profile__{key}__mean"], y, color="tab:blue", lw=1.8, label="M3 100×100 mean")
    band(axes[1, 1], y, key, vertical=True)
    axes[1, 1].set(title="Temperature at x/L = 0.8", xlabel="T (K)", ylabel="y/L", ylim=(0, 1))
    key = "normalized_qy_y08"
    axes[1, 2].plot(reference["reference_qy_x"], reference["reference_qy_normalized"], color="0.1", lw=1.7, label="Article DSMC")
    axes[1, 2].plot(x, arrays[f"profile__{key}__mean"], color="tab:blue", lw=1.8, label="M3 100×100 mean")
    band(axes[1, 2], x, key)
    axes[1, 2].axhline(0, color="0.35", lw=0.7)
    axes[1, 2].set(title="Vertical heat flux at y/L = 0.8", xlabel="x/L", ylabel=r"$q_y/q_0$", xlim=(0, 1))
    for axis in axes.ravel():
        axis.grid(alpha=0.16, linewidth=0.5)
        axis.legend(fontsize=7, loc="best")
    title = "Mohammadzadeh Kn=0.05: unsmoothed M3 R100 precision ensemble"
    figure.suptitle(title)
    _atomic_save_figure(figure, output, title)
    plt.close(figure)


def _draw_contours(arrays: Mapping[str, np.ndarray], output: Path) -> None:
    import matplotlib.pyplot as plt

    x = np.asarray(arrays["x_centers"])
    y = np.asarray(arrays["y_centers"])
    xx, yy = np.meshgrid(x, y)
    temperature = np.asarray(arrays["field__T__mean"])
    qy = np.asarray(arrays["field__qy__mean"])
    tlevels = _strict_levels(float(temperature.min()), float(temperature.max()))
    qlim = float(np.max(np.abs(qy)))
    qlevels = _strict_levels(-qlim, qlim)
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.5), constrained_layout=True)
    tplot = axes[0].contourf(xx, yy, temperature, levels=tlevels, cmap="coolwarm", extend="both")
    qplot = axes[1].contourf(xx, yy, qy, levels=qlevels, cmap="RdBu_r", extend="both")
    axes[0].set_title("M3 R100 ensemble mean T")
    axes[1].set_title(r"M3 R100 ensemble mean $q_y$")
    for axis in axes:
        axis.set(xlabel="x/L", ylabel="y/L", aspect="equal")
    figure.colorbar(tplot, ax=axes[0], label="T (K)")
    figure.colorbar(qplot, ax=axes[1], label=r"$q_y$ (W m$^{-2}$)")
    title = "Unsmoothed M3 R100 precision contours"
    figure.suptitle(title)
    _atomic_save_figure(figure, output, title)
    plt.close(figure)


def aggregate(input_root: Path, output_dir: Path) -> dict[str, Any]:
    summary, arrays, reference = _compute(input_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_save_npz(output_dir / "ensemble_fields.npz", arrays)
    _atomic_write_json(output_dir / "summary.json", summary)
    _draw_profiles(arrays, reference, output_dir / "line_profiles_m3.png")
    _draw_contours(arrays, output_dir / "contours_m3.png")
    names = ("summary.json", "ensemble_fields.npz", "line_profiles_m3.png", "contours_m3.png")
    manifest = {
        "stage": summary["stage"],
        "files": {
            name: {
                "sha256": _sha256(output_dir / name),
                "size_bytes": (output_dir / name).stat().st_size,
            }
            for name in names
        },
    }
    _atomic_write_json(output_dir / "artifact_manifest.json", manifest)
    return summary


def verify(input_root: Path, output_dir: Path) -> dict[str, Any]:
    recorded = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    recomputed, _, _ = _compute(input_root)
    if recorded != recomputed:
        raise ValueError("M3 ensemble summary differs from independent reconstruction")
    manifest = json.loads(
        (output_dir / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    expected_names = {
        "summary.json", "ensemble_fields.npz", "line_profiles_m3.png", "contours_m3.png"
    }
    if set(manifest.get("files", {})) != expected_names:
        raise ValueError("M3 ensemble manifest has missing or extra files")
    for name in expected_names:
        path = output_dir / name
        record = manifest["files"][name]
        if record.get("sha256") != _sha256(path) or record.get("size_bytes") != path.stat().st_size:
            raise ValueError(f"M3 ensemble artifact mismatch for {name}")
    return {
        "status": "complete_M3_artifacts_independently_verified",
        "seed_count": len(SEEDS),
        "all_preregistered_qy_precision_checks_pass": recorded[
            "all_preregistered_qy_precision_checks_pass"
        ],
        "decision": recorded["decision"],
        "summary_sha256": _sha256(output_dir / "summary.json"),
        "artifact_manifest_sha256": _sha256(output_dir / "artifact_manifest.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    result = (
        verify(args.input_root, args.output_dir)
        if args.verify_only
        else aggregate(args.input_root, args.output_dir)
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
