#!/usr/bin/env python3
"""Create the compact MV6 publication figure suite from completed task artifacts."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter


ARCHITECTURES = (
    "corrected_unet",
    "nafnet_small",
    "mambairv2_tiny_adapted",
    "fno_residual_small",
)
TRAINING_SEEDS = (2608091, 2608092, 2608093)
CONDITIONS = (
    "kn0p075_u150",
    "kn0p075_u300",
    "kn0p1_u200",
    "kn0p1_u400",
)
CONDITION_LABELS = {
    "kn0p075_u150": r"$Kn=0.075, U_{\rm lid}=150\ {\rm m\,s^{-1}}$",
    "kn0p075_u300": r"$Kn=0.075, U_{\rm lid}=300\ {\rm m\,s^{-1}}$",
    "kn0p1_u200": r"$Kn=0.100, U_{\rm lid}=200\ {\rm m\,s^{-1}}$",
    "kn0p1_u400": r"$Kn=0.100, U_{\rm lid}=400\ {\rm m\,s^{-1}}$",
}
LID_SPEEDS = {
    "kn0p075_u150": 150.0,
    "kn0p075_u300": 300.0,
    "kn0p1_u200": 200.0,
    "kn0p1_u400": 400.0,
}
METHOD_KEYS = (
    "reference",
    "raw",
    "tsvd_pod_type",
    *ARCHITECTURES,
)
METHOD_LABELS = {
    "reference": "Reference",
    "raw": "Raw DSMC",
    "gaussian_like": "Gaussian",
    "tsvd_pod_type": "TSVD/POD",
    "corrected_unet": "Corrected U-Net",
    "nafnet_small": "NAFNet-Small",
    "mambairv2_tiny_adapted": "MambaIRv2",
    "fno_residual_small": "FNO",
}
METHOD_COLORS = {
    "reference": "#000000",
    "raw": "#777777",
    "gaussian_like": "#A0A0A0",
    "tsvd_pod_type": "#6A3D9A",
    "corrected_unet": "#0072B2",
    "nafnet_small": "#009E73",
    "mambairv2_tiny_adapted": "#D55E00",
    "fno_residual_small": "#CC79A7",
}
METHOD_MARKERS = {
    "gaussian_like": "s",
    "tsvd_pod_type": "^",
    "corrected_unet": "o",
    "nafnet_small": "D",
    "mambairv2_tiny_adapted": "P",
    "fno_residual_small": "X",
}
PROFILE_KEYS = (
    "vertical_temperature_x08",
    "macroscopic_lid_temperature",
    "macroscopic_lid_slip",
)
PROFILE_LABELS = {
    "vertical_temperature_x08": r"Vertical $T$ profile, $x/L=0.8$",
    "macroscopic_lid_temperature": "Lid-temperature profile",
    "macroscopic_lid_slip": "Lid-slip profile",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10.5,
            "axes.labelsize": 11.5,
            "axes.titlesize": 12.0,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.0,
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


def task_dir(root: Path, architecture: str, seed: int) -> Path:
    return root / "tasks" / architecture / f"training_seed_{seed}"


def validate_inputs(root: Path) -> None:
    required = [root / "summary.json", root / "verification.json"]
    for architecture in ARCHITECTURES:
        for seed in TRAINING_SEEDS:
            directory = task_dir(root, architecture, seed)
            required.extend(
                (directory / "summary.json", directory / "predictions.npz")
            )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required MV6 artifacts:\n" + "\n".join(missing))
    verification = json.loads((root / "verification.json").read_text(encoding="utf-8"))
    if verification.get("decision") != "verified":
        raise ValueError("MV6 verification.json is not in the verified state")


def scalar_identity(value: np.ndarray | object) -> str:
    array = np.asarray(value)
    if array.ndim == 0:
        return str(array.item())
    return ",".join(str(item) for item in array.reshape(-1).tolist())


def load_payloads(root: Path) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, str]]:
    baseline_path = task_dir(root, ARCHITECTURES[0], TRAINING_SEEDS[0]) / "predictions.npz"
    payloads: dict[str, dict[str, np.ndarray]] = {}
    evaluation_ids: dict[str, str] = {}
    indices: dict[str, int] = {}
    with np.load(baseline_path, allow_pickle=False) as data:
        labels = np.asarray(data["identity_condition"]).astype(str)
        identities = np.asarray(data["identity_numeric"])
        for condition in CONDITIONS:
            matches = np.flatnonzero(labels == condition)
            if len(matches) == 0:
                raise ValueError(f"Condition absent from predictions: {condition}")
            index = int(matches[0])
            indices[condition] = index
            evaluation_ids[condition] = scalar_identity(identities[index])
            payloads[condition] = {
                "reference": np.asarray(data["target"][index], dtype=np.float64),
                "raw": np.asarray(data["raw"][index], dtype=np.float64),
                "tsvd_pod_type": np.asarray(
                    data["tsvd_pod_type"][index], dtype=np.float64
                ),
            }
    for architecture in ARCHITECTURES:
        accumulated: dict[str, list[np.ndarray]] = {condition: [] for condition in CONDITIONS}
        for seed in TRAINING_SEEDS:
            path = task_dir(root, architecture, seed) / "predictions.npz"
            with np.load(path, allow_pickle=False) as data:
                labels = np.asarray(data["identity_condition"]).astype(str)
                identities = np.asarray(data["identity_numeric"])
                for condition in CONDITIONS:
                    index = indices[condition]
                    if labels[index] != condition:
                        raise ValueError(f"Prediction identity order differs in {path}")
                    if scalar_identity(identities[index]) != evaluation_ids[condition]:
                        raise ValueError(f"Evaluation identity differs in {path}")
                    accumulated[condition].append(
                        np.asarray(data["architecture_prediction"][index], dtype=np.float64)
                    )
        for condition in CONDITIONS:
            payloads[condition][architecture] = np.mean(
                np.stack(accumulated[condition], axis=0), axis=0
            )
    for condition, methods in payloads.items():
        reference_shape = methods["reference"].shape
        if len(reference_shape) != 3 or reference_shape[0] < 2:
            raise ValueError(f"Unexpected target shape for {condition}: {reference_shape}")
        if any(value.shape != reference_shape for value in methods.values()):
            raise ValueError(f"Method field shapes differ for {condition}")
    return payloads, evaluation_ids


def nice_ceiling(value: float, minimum: float = 0.5) -> float:
    value = max(float(value), minimum)
    exponent = 10.0 ** math.floor(math.log10(value))
    scaled = value / exponent
    for multiplier in (1.0, 2.0, 2.5, 5.0, 10.0):
        if scaled <= multiplier + 1.0e-12:
            return float(multiplier * exponent)
    return float(10.0 * exponent)


def save_figure(figure: plt.Figure, stem: Path) -> list[str]:
    outputs = []
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    figure.savefig(pdf, dpi=300, bbox_inches="tight", pad_inches=0.03)
    figure.savefig(png, dpi=400, bbox_inches="tight", pad_inches=0.03)
    outputs.extend((pdf.name, png.name))
    plt.close(figure)
    return outputs


def field_error(
    field_name: str,
    candidate: np.ndarray,
    reference: np.ndarray,
    lid_speed: float,
) -> np.ndarray:
    if field_name == "T":
        denominator = np.maximum(np.abs(reference), 1.0e-12)
        return 100.0 * (candidate - reference) / denominator
    return 100.0 * (candidate - reference) / lid_speed


def field_map_figure(
    output: Path,
    condition: str,
    payload: dict[str, np.ndarray],
    field_name: str,
) -> tuple[list[str], dict[str, float | str]]:
    field_index = 0 if field_name == "T" else 1
    speed = LID_SPEEDS[condition]
    reference = payload["reference"][field_index]
    plotted: dict[str, np.ndarray] = {}
    if field_name == "T":
        for method in METHOD_KEYS:
            plotted[method] = payload[method][field_index]
        absolute_vmin = float(np.nanmin(reference))
        absolute_vmax = float(np.nanmax(reference))
        absolute_label = r"$T$ [K]"
        error_label = "Signed relative error [%]"
        stem = output / f"mv6_temperature_percent_error_{condition}"
    else:
        for method in METHOD_KEYS:
            plotted[method] = payload[method][field_index] / speed
        absolute_bound = max(float(np.nanmax(np.abs(reference / speed))), 1.0e-12)
        absolute_vmin, absolute_vmax = -absolute_bound, absolute_bound
        absolute_label = r"$u/U_{\rm lid}$"
        error_label = r"Signed error / $U_{\rm lid}$ [%]"
        stem = output / f"mv6_velocity_percent_error_{condition}"
    errors = {
        method: field_error(
            field_name, payload[method][field_index], reference, speed
        )
        for method in METHOD_KEYS
        if method != "reference"
    }
    finite_values = np.concatenate(
        [np.abs(value[np.isfinite(value)]).reshape(-1) for value in errors.values()]
    )
    percentile_value = float(np.percentile(finite_values, 99.5))
    error_limit = nice_ceiling(percentile_value)

    figure = plt.figure(figsize=(19.0, 7.0))
    grid = figure.add_gridspec(
        2,
        len(METHOD_KEYS) + 1,
        width_ratios=[1.0] * len(METHOD_KEYS) + [0.075],
        left=0.045,
        right=0.955,
        bottom=0.10,
        top=0.93,
        wspace=0.26,
        hspace=0.34,
    )
    axes = np.empty((2, len(METHOD_KEYS)), dtype=object)
    for row in range(2):
        for column in range(len(METHOD_KEYS)):
            axes[row, column] = figure.add_subplot(grid[row, column])
    cax_absolute = figure.add_subplot(grid[0, -1])
    cax_error = figure.add_subplot(grid[1, -1])

    absolute_image = None
    error_image = None
    for column, method in enumerate(METHOD_KEYS):
        axis = axes[0, column]
        absolute_image = axis.imshow(
            plotted[method],
            origin="lower",
            extent=(0.0, 1.0, 0.0, 1.0),
            cmap="coolwarm",
            vmin=absolute_vmin,
            vmax=absolute_vmax,
            interpolation="bilinear",
            rasterized=True,
        )
        axis.set_title(METHOD_LABELS[method], pad=5.0)
        bottom = axes[1, column]
        if method == "reference":
            bottom.set_facecolor("#F7F7F7")
        else:
            error_image = bottom.imshow(
                errors[method],
                origin="lower",
                extent=(0.0, 1.0, 0.0, 1.0),
                cmap="RdBu_r",
                vmin=-error_limit,
                vmax=error_limit,
                interpolation="bilinear",
                rasterized=True,
            )
            contour_levels = np.linspace(-error_limit, error_limit, 9)[1:-1]
            bottom.contour(
                np.linspace(0.0, 1.0, errors[method].shape[1]),
                np.linspace(0.0, 1.0, errors[method].shape[0]),
                errors[method],
                levels=contour_levels,
                colors="0.35",
                linewidths=0.28,
                alpha=0.45,
            )
        for row_axis in (axis, bottom):
            row_axis.set_xlim(0.0, 1.0)
            row_axis.set_ylim(0.0, 1.0)
            row_axis.set_aspect("equal")
            row_axis.set_xticks((0.0, 0.5, 1.0))
            row_axis.set_yticks((0.0, 0.5, 1.0))
        axis.tick_params(labelbottom=False)
        bottom.set_xlabel(r"$x/L$")
        if column == 0:
            axis.set_ylabel(r"$y/L$")
            bottom.set_ylabel(r"$y/L$")
        else:
            axis.tick_params(labelleft=False)
            bottom.tick_params(labelleft=False)

    if absolute_image is None or error_image is None:
        raise RuntimeError("Field-map figure did not create its colorbar images")
    absolute_ticks = np.linspace(absolute_vmin, absolute_vmax, 5)
    cb_absolute = figure.colorbar(
        absolute_image, cax=cax_absolute, ticks=absolute_ticks
    )
    cb_absolute.set_label(absolute_label, labelpad=8.0)
    error_ticks = np.linspace(-error_limit, error_limit, 5)
    cb_error = figure.colorbar(
        error_image, cax=cax_error, ticks=error_ticks, extend="both"
    )
    cb_error.set_label(error_label, labelpad=8.0)
    cb_error.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    outputs = save_figure(figure, stem)
    return outputs, {
        "field": field_name,
        "condition": condition,
        "absolute_vmin": absolute_vmin,
        "absolute_vmax": absolute_vmax,
        "error_percentile": 99.5,
        "error_percentile_value": percentile_value,
        "symmetric_error_limit": error_limit,
        "temperature_error_definition": "100*(T-T_reference)/abs(T_reference)",
        "velocity_error_definition": "100*(u-u_reference)/U_lid",
    }


def profile_figure(
    output: Path, payloads: dict[str, dict[str, np.ndarray]]
) -> list[str]:
    figure, axes = plt.subplots(
        2, 4, figsize=(14.2, 6.2), sharey="row", constrained_layout=False
    )
    figure.subplots_adjust(left=0.065, right=0.985, bottom=0.105, top=0.82, wspace=0.25, hspace=0.34)
    line_styles = {
        "reference": "-",
        "raw": ":",
        "tsvd_pod_type": "--",
        "corrected_unet": "-.",
        "nafnet_small": "-",
        "mambairv2_tiny_adapted": "--",
        "fno_residual_small": ":",
    }
    handles = []
    labels = []
    for column, condition in enumerate(CONDITIONS):
        payload = payloads[condition]
        height, width = payload["reference"].shape[-2:]
        y = np.linspace(0.0, 1.0, height)
        x_index_temperature = int(np.argmin(np.abs(np.linspace(0.0, 1.0, width) - 0.8)))
        x_index_velocity = int(np.argmin(np.abs(np.linspace(0.0, 1.0, width) - 0.5)))
        for method in METHOD_KEYS:
            kwargs = {
                "color": METHOD_COLORS[method],
                "linestyle": line_styles[method],
                "linewidth": 1.9 if method == "reference" else 1.25,
                "alpha": 0.82 if method == "raw" else 1.0,
            }
            temperature = payload[method][0, :, x_index_temperature]
            velocity = payload[method][1, :, x_index_velocity] / LID_SPEEDS[condition]
            line = axes[0, column].plot(temperature, y, **kwargs)[0]
            axes[1, column].plot(velocity, y, **kwargs)
            if column == 0:
                handles.append(line)
                labels.append(METHOD_LABELS[method])
        axes[0, column].set_title(CONDITION_LABELS[condition], pad=5.0)
        axes[0, column].set_xlabel(r"$T$ [K] at $x/L=0.8$")
        axes[1, column].set_xlabel(r"$u/U_{\rm lid}$ at $x/L=0.5$")
        for row in range(2):
            axes[row, column].set_ylim(0.0, 1.0)
            axes[row, column].set_yticks((0.0, 0.25, 0.5, 0.75, 1.0))
            axes[row, column].grid(True, color="0.88", linewidth=0.55)
        if column == 0:
            axes[0, column].set_ylabel(r"$y/L$")
            axes[1, column].set_ylabel(r"$y/L$")
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.975),
        ncol=4,
        frameon=False,
        columnspacing=1.4,
        handlelength=2.8,
    )
    return save_figure(figure, output / "mv6_temperature_velocity_profiles_all_conditions")


def read_task_summaries(root: Path) -> dict[str, list[dict[str, object]]]:
    result: dict[str, list[dict[str, object]]] = {name: [] for name in ARCHITECTURES}
    for architecture in ARCHITECTURES:
        for seed in TRAINING_SEEDS:
            path = task_dir(root, architecture, seed) / "summary.json"
            result[architecture].append(json.loads(path.read_text(encoding="utf-8")))
    return result


def composite_ratio_figure(
    output: Path, aggregate: dict[str, object]
) -> tuple[list[str], list[dict[str, object]]]:
    statistics = aggregate["statistics"]
    x = np.arange(len(CONDITIONS), dtype=float)
    methods = ("gaussian_like", "tsvd_pod_type", *ARCHITECTURES)
    offsets = np.linspace(-0.24, 0.24, len(methods))
    records: list[dict[str, object]] = []
    figure, axis = plt.subplots(figsize=(8.7, 4.8), constrained_layout=True)
    for offset, method in zip(offsets, methods):
        means = []
        errors = []
        for condition in CONDITIONS:
            if method in ("gaussian_like", "tsvd_pod_type"):
                value = float(
                    statistics["baselines"][condition][method]["vision_over_raw_composite"]
                )
                mean, std = value, 0.0
            else:
                value = statistics["architectures"][method]["by_condition"][condition][
                    "model_over_raw"
                ]
                mean, std = float(value["mean"]), float(value["std"])
            means.append(mean)
            errors.append(std)
            records.append(
                {
                    "metric": "composite_nrmse_over_raw",
                    "method": method,
                    "condition": condition,
                    "mean": mean,
                    "sample_std": std,
                    "error_reduction_percent": 100.0 * (1.0 - mean),
                }
            )
        axis.errorbar(
            x + offset,
            means,
            yerr=errors if method in ARCHITECTURES else None,
            marker=METHOD_MARKERS[method],
            markersize=5.5,
            capsize=2.5,
            linewidth=1.25,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    axis.axhline(1.0, color="black", linewidth=1.0, label="Raw DSMC")
    axis.set_xticks(
        x,
        ("0.075 / 150", "0.075 / 300", "0.100 / 200", "0.100 / 400"),
    )
    axis.set_xlabel(r"$Kn\;/\;U_{\rm lid}\ [{\rm m\,s^{-1}}]$")
    axis.set_ylabel("Composite NRMSE / Raw DSMC")
    axis.set_ylim(0.0, 1.08)
    axis.grid(axis="y", color="0.87", linewidth=0.6)
    axis.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.19))
    return save_figure(figure, output / "mv6_composite_nrmse_ratio_all_conditions"), records


def profile_ratio_figure(
    output: Path,
    aggregate: dict[str, object],
    summaries: dict[str, list[dict[str, object]]],
) -> tuple[list[str], list[dict[str, object]]]:
    statistics = aggregate["statistics"]
    methods = ("gaussian_like", "tsvd_pod_type", *ARCHITECTURES)
    x = np.arange(len(CONDITIONS), dtype=float)
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.35), sharey=True)
    figure.subplots_adjust(left=0.07, right=0.985, bottom=0.19, top=0.76, wspace=0.16)
    records: list[dict[str, object]] = []
    legend_handles = []
    legend_labels = []
    for panel, profile in enumerate(PROFILE_KEYS):
        axis = axes[panel]
        for method in methods:
            means = []
            stds = []
            for condition in CONDITIONS:
                if method in ("gaussian_like", "tsvd_pod_type"):
                    mean = float(
                        statistics["baselines"][condition][method]["validated_profiles"][
                            profile
                        ]["ratio"]
                    )
                    std = 0.0
                else:
                    values = [
                        float(
                            item["methods_by_condition"][method][condition][
                                "validated_profiles"
                            ][profile]["ratio"]
                        )
                        for item in summaries[method]
                    ]
                    mean = float(np.mean(values))
                    std = float(np.std(values, ddof=1))
                means.append(mean)
                stds.append(std)
                records.append(
                    {
                        "metric": profile + "_nrmse_over_raw",
                        "method": method,
                        "condition": condition,
                        "mean": mean,
                        "sample_std": std,
                    }
                )
            line = axis.errorbar(
                x,
                means,
                yerr=stds if method in ARCHITECTURES else None,
                marker=METHOD_MARKERS[method],
                markersize=4.8,
                capsize=2.0,
                linewidth=1.15,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
            if panel == 0:
                legend_handles.append(line)
                legend_labels.append(METHOD_LABELS[method])
        axis.axhline(1.0, color="black", linewidth=0.9)
        axis.set_yscale("log")
        axis.set_title(f"({chr(97 + panel)}) {PROFILE_LABELS[profile]}")
        axis.set_xticks(x, ("0.075/150", "0.075/300", "0.100/200", "0.100/400"), rotation=22)
        axis.set_xlabel(r"$Kn/U_{\rm lid}$")
        axis.grid(axis="y", which="both", color="0.88", linewidth=0.55)
        axis.yaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
        axis.yaxis.set_minor_formatter(NullFormatter())
    axes[0].set_ylabel("Profile NRMSE / Raw DSMC (log scale)")
    figure.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.975),
        ncol=3,
        frameon=False,
        columnspacing=1.5,
    )
    return save_figure(figure, output / "mv6_validated_profile_nrmse_ratios"), records


def write_csv(path: Path, records: list[dict[str, object]]) -> None:
    fields = ("metric", "method", "condition", "mean", "sample_std", "error_reduction_percent")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fields})


def git_head() -> str:
    try:
        return subprocess.check_output(
            ("git", "rev-parse", "HEAD"), stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unavailable"


def write_readme(output: Path, evaluation_ids: dict[str, str]) -> None:
    selected = "\n".join(
        f"- `{condition}`: first locked evaluation identity `{evaluation_ids[condition]}`"
        for condition in CONDITIONS
    )
    text = f"""# Mohammadzadeh MV6 publication figure suite

This compact package was generated from the recursively verified MV6 task artifacts.
It contains figures and provenance only; model weights and `predictions.npz` files are
intentionally excluded.

## Figure set

- `mv6_temperature_percent_error_<condition>`: temperature and signed local relative-error contours.
- `mv6_velocity_percent_error_<condition>`: horizontal velocity and signed error normalized by lid speed.
- `mv6_temperature_velocity_profiles_all_conditions`: vertical temperature and velocity profiles.
- `mv6_composite_nrmse_ratio_all_conditions`: composite NRMSE normalized by Raw DSMC; neural error bars are sample SD over three training initializations.
- `mv6_validated_profile_nrmse_ratios`: predeclared profile metrics, including adverse ratios above one.

Each figure is supplied as a compact PDF and a 400-dpi PNG.

## Exact error definitions

- Temperature: `100 * (T_method - T_reference) / abs(T_reference)` [%].
- Horizontal velocity: `100 * (u_method - u_reference) / U_lid` [%].

Local relative velocity error is deliberately not used because the reference velocity
crosses zero and would create singular, visually misleading percentages.

Within each contour figure, all methods use the same absolute color scale and the same
symmetric error scale. The error bound is the 99.5th percentile over all non-reference
methods, rounded upward to a readable 1-2-2.5-5 value; values outside it are shown with
extended colorbar ends. Exact limits are stored in `figure_metadata.json`.

## Sampling/aggregation shown in field figures

{selected}

Raw DSMC, TSVD/POD and the reference use that same evaluation identity. Each neural
field is the arithmetic mean of the corresponding prediction over the three locked
training initialization seeds. No averaging over distinct DSMC evaluation inputs is
performed in these field maps.

## Provenance

- Source MV6 output root: `{os.environ['MV6_OUTPUT_ROOT']}`
- Local repository HEAD at figure generation: `{git_head()}`
- Generated UTC: `{datetime.now(timezone.utc).isoformat()}`
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def create_zip(output: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_directory = Path(
        os.environ.get("MV6_ARCHIVE_DIR", str(Path.home()))
    ).expanduser()
    archive_directory.mkdir(parents=True, exist_ok=True)
    archive = archive_directory / f"MOHAMMADZADEH_MV6_PUBLICATION_FIGURES_{timestamp}.zip"
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as zipped:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                zipped.write(path, arcname=f"{output.name}/{path.relative_to(output)}")
    return archive


def main() -> None:
    configure_matplotlib()
    root_value = os.environ.get("MV6_OUTPUT_ROOT")
    if not root_value:
        raise SystemExit("MV6_OUTPUT_ROOT is not exported; source the job env and export it")
    root = Path(root_value).expanduser().resolve()
    output = root / "publication_suite_20260811"
    output.mkdir(parents=True, exist_ok=True)
    validate_inputs(root)
    payloads, evaluation_ids = load_payloads(root)
    aggregate = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    summaries = read_task_summaries(root)

    generated: list[str] = []
    map_metadata: list[dict[str, float | str]] = []
    for condition in CONDITIONS:
        for field_name in ("T", "u"):
            files, metadata = field_map_figure(
                output, condition, payloads[condition], field_name
            )
            generated.extend(files)
            map_metadata.append(metadata)
    generated.extend(profile_figure(output, payloads))
    files, composite_records = composite_ratio_figure(output, aggregate)
    generated.extend(files)
    files, profile_records = profile_ratio_figure(output, aggregate, summaries)
    generated.extend(files)
    metric_records = composite_records + profile_records
    write_csv(output / "publication_metrics.csv", metric_records)
    generated.append("publication_metrics.csv")

    metadata = {
        "stage": "MV6_publication_figure_suite",
        "status": "complete",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_mv6_root": str(root),
        "source_summary_sha256": sha256(root / "summary.json"),
        "source_verification_sha256": sha256(root / "verification.json"),
        "selected_evaluation_identities": evaluation_ids,
        "neural_field_aggregation": "mean_over_three_locked_training_initialization_seeds",
        "field_map_scales": map_metadata,
        "files": generated,
    }
    (output / "figure_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.copy2(root / "summary.json", output / "provenance_mv6_summary.json")
    shutil.copy2(root / "verification.json", output / "provenance_mv6_verification.json")
    script_source = Path(__file__).resolve()
    shutil.copy2(script_source, output / script_source.name)
    write_readme(output, evaluation_ids)

    manifest_candidates = sorted(
        path for path in output.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    manifest_lines = [
        f"{sha256(path)}  {path.relative_to(output)}" for path in manifest_candidates
    ]
    (output / "SHA256SUMS").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    archive = create_zip(output)
    archive_size = archive.stat().st_size
    limit = 450 * 1024 * 1024
    print(f"PUBLICATION_DIR={output}")
    print(f"ARCHIVE={archive}")
    print(f"ARCHIVE_SIZE_MIB={archive_size / 1024**2:.2f}")
    print(f"ARCHIVE_SHA256={sha256(archive)}")
    if archive_size > limit:
        raise SystemExit("Archive exceeds the 450 MiB upload limit")


if __name__ == "__main__":
    main()
