#!/usr/bin/env python3
"""Create audit-only physical moment figures from an assembled MV8 dataset.

This script is deliberately independent of the MV8 model gate.  It visualizes
only quantities that exist before neural training: the cross-fit reference,
Raw DSMC at B=1, the development-selected Gaussian and TSVD/POD baselines, and
paired Raw DSMC at B=10.  It never labels a classical field as a neural result.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tarfile
from typing import Any, Mapping, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter, ScalarFormatter


OUTPUT_FIELDS = (
    "tau_xy_over_p_ref",
    "normal_stress_difference_over_p_ref",
    "qx_over_q_ref",
    "qy_over_q_ref",
)
FIELD_SPECS = (
    ("tau_xy", r"$P_{xy}$", "Pa", r"$100\,\Delta P_{xy}/p_{\mathrm{ref}}$ [\%]"),
    (
        "normal_stress_difference",
        r"$P_{xx}-P_{yy}$",
        "Pa",
        r"$100\,\Delta(P_{xx}-P_{yy})/p_{\mathrm{ref}}$ [\%]",
    ),
    ("qx", r"$q_x$", r"W m$^{-2}$", r"$100\,\Delta q_x/q_{\mathrm{ref}}$ [\%]"),
    ("qy", r"$q_y$", r"W m$^{-2}$", r"$100\,\Delta q_y/q_{\mathrm{ref}}$ [\%]"),
)
METHOD_SPECS = (
    ("reference", "Reference"),
    ("raw_b1", "Raw DSMC\n$B=1$"),
    ("gaussian_b1", "Gaussian\n$B=1$"),
    ("tsvd_b1", "TSVD/POD\n$B=1$"),
    ("raw_b10", "Raw DSMC\n$B=10$"),
)
REQUIRED_DATASET_KEYS = (
    "test_x",
    "test_y",
    "test_condition",
    "test_identity",
    "test_scale",
    "test_gaussian",
    "test_tsvd",
    "test_raw10",
    "test_target10",
    "test_condition10",
    "test_identity10",
    "test_scale10",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def git_head(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ("git", "-C", str(repo_root), "rev-parse", "HEAD"),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unavailable"


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10.5,
            "axes.labelsize": 12.0,
            "axes.titlesize": 11.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.linewidth": 0.9,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def nice_ceiling(value: float, minimum: float) -> float:
    value = max(float(value), minimum)
    exponent = 10.0 ** math.floor(math.log10(value))
    scaled = value / exponent
    for multiplier in (1.0, 2.0, 2.5, 5.0, 10.0):
        if scaled <= multiplier + 1.0e-12:
            return float(multiplier * exponent)
    return float(10.0 * exponent)


def save_figure(figure: plt.Figure, stem: Path) -> tuple[Path, Path]:
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    figure.savefig(pdf, bbox_inches="tight", pad_inches=0.035, facecolor="white")
    figure.savefig(
        png,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.035,
        facecolor="white",
    )
    plt.close(figure)
    return pdf, png


def load_protocol(mv8_root: Path, repo_root: Path) -> dict[str, Any]:
    candidates = (
        mv8_root / "mv8_kinetic_moment_feasibility_protocol.json",
        repo_root
        / "reference_data"
        / "mohammadzadeh_2012"
        / "mv8_kinetic_moment_feasibility_protocol.json",
    )
    for path in candidates:
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            outputs = tuple(value.get("moment_contract", {}).get("outputs", ()))
            if outputs != OUTPUT_FIELDS:
                raise ValueError(f"MV8 protocol output contract differs in {path}")
            return value
    raise FileNotFoundError("MV8 kinetic-moment protocol is absent")


def load_inputs(mv8_root: Path, repo_root: Path) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    assembly_path = mv8_root / "assembly_summary.json"
    dataset_path = mv8_root / "dataset.npz"
    if not assembly_path.is_file() or not dataset_path.is_file():
        raise FileNotFoundError("MV8 assembly_summary.json or dataset.npz is absent")
    assembly = json.loads(assembly_path.read_text(encoding="utf-8"))
    if assembly.get("status") != "complete_MV8_additive_moment_assembly_and_information_gate":
        raise ValueError("MV8 assembly is not complete")
    if assembly.get("decision") not in {
        "hold_MV8_models_return_information_audit_only",
        "proceed_to_MV8_B1_kinetic_models",
    }:
        raise ValueError("MV8 assembly decision is not recognized")
    protocol = load_protocol(mv8_root, repo_root)
    with np.load(dataset_path, allow_pickle=False) as source:
        missing = set(REQUIRED_DATASET_KEYS) - set(source.files)
        if missing:
            raise ValueError(f"MV8 dataset is missing {sorted(missing)}")
        data = {name: np.asarray(source[name]).copy() for name in REQUIRED_DATASET_KEYS}
    raw = data["test_x"][:, : len(OUTPUT_FIELDS)]
    for name, value in {
        "raw_b1": raw,
        "target": data["test_y"],
        "gaussian_b1": data["test_gaussian"],
        "tsvd_b1": data["test_tsvd"],
        "raw_b10": data["test_raw10"],
        "target_b10": data["test_target10"],
    }.items():
        if value.ndim != 4 or value.shape[1] != len(OUTPUT_FIELDS) or not np.all(np.isfinite(value)):
            raise ValueError(f"Invalid or non-finite MV8 field array: {name}")
    return assembly, data, protocol


def representative_seed(
    condition: str,
    conditions: np.ndarray,
    identities: np.ndarray,
    protocol: Mapping[str, Any],
) -> int:
    available = sorted(int(value) for value in np.unique(identities[conditions == condition, 0]))
    if not available:
        raise ValueError(f"No MV8 B=1 seed is available for {condition}")
    execution = protocol["execution_matrix"]
    locked = int(execution["representative_contour_seed"])
    if condition == str(execution["primary_condition"]) and locked in available:
        return locked
    return available[0]


def select_condition_payload(
    condition: str,
    data: Mapping[str, np.ndarray],
    protocol: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], np.ndarray, int, int]:
    conditions = np.asarray(data["test_condition"]).astype(str)
    identities = np.asarray(data["test_identity"], dtype=np.int64)
    conditions10 = np.asarray(data["test_condition10"]).astype(str)
    identities10 = np.asarray(data["test_identity10"], dtype=np.int64)
    seed = representative_seed(condition, conditions, identities, protocol)
    preferred_block = int(protocol["execution_matrix"].get("representative_contour_block", 0))
    mask = (
        (conditions == condition)
        & (identities[:, 0] == seed)
        & (identities[:, 1] == preferred_block)
    )
    mask10 = (conditions10 == condition) & (identities10[:, 0] == seed)
    if np.count_nonzero(mask) != 1 or np.count_nonzero(mask10) != 1:
        raise ValueError(f"Unique paired MV8 B=1/B=10 identity is absent for {condition}, seed {seed}")
    index = int(np.flatnonzero(mask)[0])
    index10 = int(np.flatnonzero(mask10)[0])
    reference = np.asarray(data["test_y"])[index]
    reference10 = np.asarray(data["test_target10"])[index10]
    if not np.array_equal(reference, reference10):
        raise ValueError(f"Cross-fit targets differ between B=1 and B=10 for {condition}")
    scale = np.asarray(data["test_scale"])[index]
    scale10 = np.asarray(data["test_scale10"])[index10]
    if not np.allclose(scale, scale10, rtol=1.0e-12, atol=0.0):
        raise ValueError(f"Physical scales differ between B=1 and B=10 for {condition}")
    payload = {
        "reference": reference,
        "raw_b1": np.asarray(data["test_x"])[index, : len(OUTPUT_FIELDS)],
        "gaussian_b1": np.asarray(data["test_gaussian"])[index],
        "tsvd_b1": np.asarray(data["test_tsvd"])[index],
        "raw_b10": np.asarray(data["test_raw10"])[index10],
    }
    shape = reference.shape
    if len(shape) != 3 or any(value.shape != shape for value in payload.values()):
        raise ValueError(f"Physical field shapes differ for {condition}")
    return payload, np.asarray(scale, dtype=np.float64), seed, preferred_block


def condition_math_label(condition: str) -> str:
    parts = condition.split("_")
    kn = parts[0].removeprefix("kn").replace("p", ".")
    speed = parts[1].removeprefix("u")
    return rf"$Kn={kn}$, $U_{{\mathrm{{lid}}}}={speed}\ \mathrm{{m\,s^{{-1}}}}$"


def physical_figure(
    output: Path,
    condition: str,
    field_index: int,
    payload: Mapping[str, np.ndarray],
    scale: float,
    seed: int,
    block: int,
    audit_only: bool,
) -> tuple[list[Path], list[dict[str, Any]], dict[str, Any]]:
    key, symbol, unit, error_label = FIELD_SPECS[field_index]
    normalized = {name: np.asarray(payload[name][field_index], dtype=np.float64) for name, _ in METHOD_SPECS}
    reference = normalized["reference"]
    physical = {name: value * scale for name, value in normalized.items()}
    errors = {
        name: 100.0 * (value - reference)
        for name, value in normalized.items()
        if name != "reference"
    }
    physical_values = np.concatenate([np.abs(value[np.isfinite(value)]).ravel() for value in physical.values()])
    error_values = np.concatenate([np.abs(value[np.isfinite(value)]).ravel() for value in errors.values()])
    physical_limit = nice_ceiling(float(np.percentile(physical_values, 99.5)), 1.0e-12)
    error_limit = nice_ceiling(float(np.percentile(error_values, 99.5)), 1.0e-4)
    physical_norm = TwoSlopeNorm(vmin=-physical_limit, vcenter=0.0, vmax=physical_limit)
    error_norm = TwoSlopeNorm(vmin=-error_limit, vcenter=0.0, vmax=error_limit)

    figure = plt.figure(figsize=(13.0, 7.15))
    grid = figure.add_gridspec(
        2,
        len(METHOD_SPECS) + 1,
        width_ratios=[1.0] * len(METHOD_SPECS) + [0.065],
        left=0.055,
        right=0.95,
        bottom=0.105,
        top=0.86,
        wspace=0.18,
        hspace=0.30,
    )
    axes = np.empty((2, len(METHOD_SPECS)), dtype=object)
    for row in range(2):
        for column in range(len(METHOD_SPECS)):
            axes[row, column] = figure.add_subplot(grid[row, column])
    physical_cax = figure.add_subplot(grid[0, -1])
    error_cax = figure.add_subplot(grid[1, -1])
    physical_artist = error_artist = None
    rows: list[dict[str, Any]] = []
    for column, (name, title) in enumerate(METHOD_SPECS):
        top = axes[0, column]
        physical_artist = top.imshow(
            physical[name],
            origin="lower",
            extent=(0.0, 1.0, 0.0, 1.0),
            cmap="RdBu_r",
            norm=physical_norm,
            interpolation="bilinear",
            rasterized=True,
        )
        top.set_title(title, pad=4.5, linespacing=0.90)
        bottom = axes[1, column]
        if name == "reference":
            bottom.set_facecolor("#F7F7F7")
            bottom.text(0.5, 0.5, "Reference", transform=bottom.transAxes, ha="center", va="center", color="0.45")
            error_rms = 0.0
            error_abs_995 = 0.0
        else:
            error = errors[name]
            error_artist = bottom.imshow(
                error,
                origin="lower",
                extent=(0.0, 1.0, 0.0, 1.0),
                cmap="RdBu_r",
                norm=error_norm,
                interpolation="bilinear",
                rasterized=True,
            )
            error_rms = float(np.sqrt(np.mean(error**2)))
            error_abs_995 = float(np.percentile(np.abs(error), 99.5))
        for axis in (top, bottom):
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(0.0, 1.0)
            axis.set_aspect("equal")
            axis.set_xticks((0.0, 0.5, 1.0))
            axis.set_yticks((0.0, 0.5, 1.0))
        top.tick_params(labelbottom=False)
        bottom.set_xlabel(r"$x/L$", labelpad=2.0)
        if column == 0:
            top.set_ylabel(r"$y/L$", labelpad=2.0)
            bottom.set_ylabel(r"$y/L$", labelpad=2.0)
        else:
            top.tick_params(labelleft=False)
            bottom.tick_params(labelleft=False)
        difference = normalized[name] - reference
        denominator = max(float(np.sqrt(np.mean(reference**2))), 1.0e-12)
        rows.append(
            {
                "condition": condition,
                "seed": seed,
                "block": block,
                "field": OUTPUT_FIELDS[field_index],
                "method": name,
                "physical_min": float(np.min(physical[name])),
                "physical_max": float(np.max(physical[name])),
                "nrmse": float(np.sqrt(np.mean(difference**2)) / denominator),
                "fixed_scale_error_rms_percent": error_rms,
                "fixed_scale_error_abs_99p5_percent": error_abs_995,
            }
        )
    if physical_artist is None or error_artist is None:
        raise RuntimeError("Physical moment figure did not create colorbar artists")
    physical_bar = figure.colorbar(physical_artist, cax=physical_cax, extend="both")
    physical_bar.set_label(f"{symbol} [{unit}]", labelpad=7.0)
    physical_formatter = ScalarFormatter(useMathText=True)
    physical_formatter.set_powerlimits((-2, 3))
    physical_bar.ax.yaxis.set_major_formatter(physical_formatter)
    error_bar = figure.colorbar(error_artist, cax=error_cax, extend="both")
    error_bar.set_label(error_label, labelpad=7.0)
    error_bar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    figure.suptitle(condition_math_label(condition), y=0.965, fontsize=13.0)
    if audit_only:
        figure.text(
            0.5,
            0.018,
            "MV8 information-audit visualization; neural models were not trained because the heat-flux consistency gate did not pass.",
            ha="center",
            va="bottom",
            fontsize=8.7,
            color="0.30",
        )
    stem = output / f"mv8_audit_{key}_physical_{condition}"
    pdf, png = save_figure(figure, stem)
    record = {
        "condition": condition,
        "seed": seed,
        "block": block,
        "field": OUTPUT_FIELDS[field_index],
        "physical_scale": scale,
        "physical_limit": physical_limit,
        "error_percent_limit": error_limit,
        "pdf": pdf.name,
        "png": png.name,
    }
    return [pdf, png], rows, record


def audit_summary_figure(output: Path, assembly: Mapping[str, Any], protocol: Mapping[str, Any]) -> tuple[Path, Path]:
    validation = assembly["development_validation_information_test"]
    raw_b1 = validation["raw_B1"]
    raw_b10 = validation["raw_B10"]
    labels = (r"$P_{xy}$", r"$P_{xx}-P_{yy}$", r"$q_x$", r"$q_y$")
    x = np.arange(len(labels), dtype=float)
    width = 0.34
    figure, axes = plt.subplots(2, 2, figsize=(12.4, 8.2))
    figure.subplots_adjust(left=0.075, right=0.98, bottom=0.09, top=0.90, wspace=0.30, hspace=0.43)

    b1_values = [float(raw_b1["per_field_nrmse"][field]) for field in OUTPUT_FIELDS]
    b10_values = [float(raw_b10["per_field_nrmse"][field]) for field in OUTPUT_FIELDS]
    axes[0, 0].bar(x - width / 2, b1_values, width, label=r"Raw $B=1$", color="#0072B2")
    axes[0, 0].bar(x + width / 2, b10_values, width, label=r"Raw $B=10$", color="#D55E00")
    axes[0, 0].set_xticks(x, labels)
    axes[0, 0].set_ylabel("RMS-normalized RMSE")
    axes[0, 0].set_title("Information content by moment field")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].grid(axis="y", color="0.88", linewidth=0.8)

    composite = [float(raw_b1["composite_nrmse"]), float(raw_b10["composite_nrmse"])]
    bars = axes[0, 1].bar((0, 1), composite, color=("#0072B2", "#D55E00"), width=0.62)
    axes[0, 1].set_xticks((0, 1), (r"Raw $B=1$", r"Raw $B=10$"))
    axes[0, 1].set_ylabel("Composite NRMSE")
    axes[0, 1].set_title("Ten-block information improvement")
    axes[0, 1].grid(axis="y", color="0.88", linewidth=0.8)
    for bar, value in zip(bars, composite):
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}", ha="center", va="bottom")

    gates = protocol["pre_model_feasibility_gates"]
    values = (
        float(assembly["maximum_block_full_additive_moment_fixed_scale_relative_linf"]),
        float(assembly["maximum_q_reconstruction_relative_difference"]),
    )
    tolerances = (
        float(gates["block_full_additive_moment_fixed_scale_relative_linf_tolerance"]),
        float(gates["stored_and_reconstructed_heat_flux_relative_tolerance"]),
    )
    positions = np.arange(2, dtype=float)
    axes[1, 0].bar(positions - width / 2, values, width, label="Observed", color="#6A3D9A")
    axes[1, 0].bar(positions + width / 2, tolerances, width, label="Locked tolerance", color="#999999")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xticks(positions, ("Block/full\nadditivity", "Stored/rebuilt\nheat flux"))
    axes[1, 0].set_ylabel("Relative discrepancy")
    axes[1, 0].set_title("Locked integrity gates")
    axes[1, 0].legend(frameon=False)
    axes[1, 0].grid(axis="y", which="both", color="0.88", linewidth=0.8)

    axes[1, 1].axis("off")
    checks = assembly["checks"]
    lines = [
        "MV8 pre-model information audit",
        "",
        f"Finite reconstructed fields: {'PASS' if checks['all_reconstructed_moment_fields_finite'] else 'FAIL'}",
        f"Pressure covariance PSD: {'PASS' if checks['pressure_covariance_positive_semidefinite'] else 'FAIL'}",
        f"Block/full additive moments: {'PASS' if checks['block_sums_match_full_additive_accumulators_with_fixed_scale_tolerance'] else 'FAIL'}",
        f"Stored/reconstructed heat flux: {'PASS' if checks['stored_and_reconstructed_heat_flux_match'] else 'FAIL'}",
        f"Fields improved at B=10: {int(validation['individual_fields_improved'])}/4",
        "",
        f"Minimum covariance eigenvalue ratio: {float(assembly['minimum_covariance_eigenvalue_ratio']):.6g}",
        f"Model decision: {assembly['decision']}",
    ]
    axes[1, 1].text(
        0.03,
        0.97,
        "\n".join(lines),
        transform=axes[1, 1].transAxes,
        ha="left",
        va="top",
        fontsize=11.0,
        linespacing=1.35,
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "#F7F7F7", "edgecolor": "0.75"},
    )
    figure.suptitle("MV8 kinetic-moment reconstruction audit", fontsize=14.0)
    return save_figure(figure, output / "mv8_audit_gate_and_information_summary")


def write_readme(output: Path, assembly: Mapping[str, Any], conditions: Sequence[str]) -> Path:
    path = output / "README.md"
    path.write_text(
        "# MV8 audit-only kinetic-moment figures\n\n"
        "These figures visualize pre-model reconstructed moment fields from the immutable MV8 assembly dataset. "
        "They do not contain neural-network predictions.\n\n"
        "Columns are Reference, Raw DSMC at B=1, the development-selected Gaussian and TSVD/POD baselines at B=1, "
        "and paired Raw DSMC at B=10. The upper row is the physical field. The lower row is the signed difference "
        "as a percentage of the fixed p_ref or q_ref scale; pointwise division by a locally vanishing moment is not used.\n\n"
        f"MV8 assembly decision: `{assembly['decision']}`.\n\n"
        f"Conditions: {', '.join(conditions)}.\n\n"
        "The heat-flux consistency gate remains failed, so qx and qy panels are diagnostic audit candidates and must not be "
        "reported as validated model results until that discrepancy is resolved.\n",
        encoding="utf-8",
    )
    return path


def create_archive(output: Path, files: Sequence[Path]) -> tuple[Path, Path]:
    archive = output / "MOHAMMADZADEH_MV8_AUDIT_PHYSICAL_FIGURES.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for path in files:
            stream.add(path, arcname=path.name)
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    checksum.write_text(f"{sha256(archive)}  {archive.name}\n", encoding="utf-8")
    return archive, checksum


def run(mv8_root: Path, output: Path, repo_root: Path, conditions: Sequence[str] | None) -> dict[str, Any]:
    configure_matplotlib()
    assembly, data, protocol = load_inputs(mv8_root, repo_root)
    available = tuple(str(value) for value in np.unique(np.asarray(data["test_condition"]).astype(str)))
    requested = tuple(conditions) if conditions else available
    unknown = set(requested) - set(available)
    if unknown:
        raise ValueError(f"Requested conditions are absent from MV8: {sorted(unknown)}")
    output.mkdir(parents=True, exist_ok=False)
    audit_only = assembly["decision"] == "hold_MV8_models_return_information_audit_only"
    generated: list[Path] = []
    metric_rows: list[dict[str, Any]] = []
    figure_records: list[dict[str, Any]] = []
    for condition in requested:
        payload, scale, seed, block = select_condition_payload(condition, data, protocol)
        for field_index in range(len(OUTPUT_FIELDS)):
            files, rows, record = physical_figure(
                output,
                condition,
                field_index,
                payload,
                float(scale[field_index]),
                seed,
                block,
                audit_only,
            )
            generated.extend(files)
            metric_rows.extend(rows)
            figure_records.append(record)
    summary_files = audit_summary_figure(output, assembly, protocol)
    generated.extend(summary_files)
    metrics = output / "mv8_audit_physical_figure_metrics.csv"
    with metrics.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    generated.append(metrics)
    readme = write_readme(output, assembly, requested)
    generated.append(readme)
    metadata = {
        "stage": "MV8_audit_only_kinetic_moment_physical_figures",
        "status": "complete_MV8_audit_only_physical_figure_suite",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_mv8_root": str(mv8_root.resolve()),
        "source_assembly_sha256": sha256(mv8_root / "assembly_summary.json"),
        "source_dataset_sha256": sha256(mv8_root / "dataset.npz"),
        "repository_head": git_head(repo_root),
        "assembly_decision": assembly["decision"],
        "neural_predictions_included": False,
        "heat_flux_validation_guard": "diagnostic_audit_candidate_until_stored_reconstructed_heat_flux_gate_passes",
        "conditions": list(requested),
        "columns": [title.replace("\n", " ") for _, title in METHOD_SPECS],
        "error_definition": "100*(candidate-reference)/fixed_p_ref_or_q_ref",
        "figures": figure_records,
    }
    metadata_path = output / "figure_metadata.json"
    atomic_json(metadata_path, metadata)
    generated.append(metadata_path)
    manifest_path = output / "artifact_manifest.json"
    atomic_json(
        manifest_path,
        {
            "stage": metadata["stage"],
            "files": {
                path.name: {"sha256": sha256(path), "size_bytes": path.stat().st_size}
                for path in generated
            },
        },
    )
    generated.append(manifest_path)
    archive, checksum = create_archive(output, generated)
    result = {
        "status": metadata["status"],
        "output": str(output.resolve()),
        "archive": str(archive.resolve()),
        "checksum": str(checksum.resolve()),
        "conditions": list(requested),
        "physical_figures": 4 * len(requested),
        "neural_predictions_included": False,
    }
    atomic_json(output / "completion.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mv8-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--condition", action="append", dest="conditions")
    args = parser.parse_args()
    print(json.dumps(run(args.mv8_root, args.output, args.repo_root, args.conditions), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
