#!/usr/bin/env python3
"""Create immutable, journal-quality figures from the completed MV7 matrix."""

from __future__ import annotations

import argparse
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
import time
from typing import Any, Mapping
import zipfile

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, ScalarFormatter


BUDGETS = (1, 2, 5, 10)
ARCHITECTURES = (
    "corrected_unet",
    "nafnet_small",
    "mambairv2_tiny_adapted",
    "fno_residual_small",
)
TRAINING_SEEDS = (2608091, 2608092, 2608093)
METHODS = (
    "raw",
    "gaussian_like",
    "tsvd_pod_type",
    *ARCHITECTURES,
)
DISPLAY_NAMES = {
    "raw": "Raw DSMC",
    "gaussian_like": "Gaussian",
    "tsvd_pod_type": "TSVD/POD",
    "corrected_unet": "Corrected U-Net",
    "nafnet_small": "NAFNet-Small",
    "mambairv2_tiny_adapted": "MambaIRv2-Tiny",
    "fno_residual_small": "FNO-residual Small",
}
SHORT_NAMES = {
    "raw": "Raw",
    "gaussian_like": "Gaussian",
    "tsvd_pod_type": "TSVD/POD",
    "corrected_unet": "U-Net",
    "nafnet_small": "NAFNet",
    "mambairv2_tiny_adapted": "MambaIRv2",
    "fno_residual_small": "FNO",
}
COLORS = {
    "raw": "#111111",
    "gaussian_like": "#7F7F7F",
    "tsvd_pod_type": "#6A3D9A",
    "corrected_unet": "#0072B2",
    "nafnet_small": "#009E73",
    "mambairv2_tiny_adapted": "#D55E00",
    "fno_residual_small": "#CC79A7",
}
MARKERS = {
    "raw": "o",
    "gaussian_like": "s",
    "tsvd_pod_type": "^",
    "corrected_unet": "o",
    "nafnet_small": "D",
    "mambairv2_tiny_adapted": "P",
    "fno_residual_small": "X",
}
NONINFERIORITY_MARGIN = 1.10
EPSILON = 1.0e-12
CONDITIONS = (
    "kn0p075_u150",
    "kn0p075_u300",
    "kn0p1_u200",
    "kn0p1_u400",
)
PHYSICAL_COLUMN_SPECS = (
    ("reference", "Reference"),
    ("raw_b1", "Raw DSMC\n$B=1$"),
    ("gaussian_b1", "Gaussian\n$B=1$"),
    ("tsvd_b1", "TSVD/POD\n$B=1$"),
    ("corrected_unet", "Corrected U-Net\n$B=1$"),
    ("nafnet_small", "NAFNet-Small\n$B=1$"),
    ("mambairv2_tiny_adapted", "MambaIRv2\n$B=1$"),
    ("fno_residual_small", "FNO\n$B=1$"),
    ("raw_b10", "Raw DSMC\n$B=10$"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head(repo_root: Path | None = None) -> str:
    try:
        command = ("git", "-C", str(repo_root), "rev-parse", "HEAD") if repo_root else ("git", "rev-parse", "HEAD")
        return subprocess.check_output(command, stderr=subprocess.DEVNULL, text=True).strip()
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
            "legend.fontsize": 9.0,
            "axes.linewidth": 0.9,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(figure: plt.Figure, stem: Path) -> list[str]:
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    figure.savefig(pdf, bbox_inches="tight", pad_inches=0.035)
    figure.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.035)
    plt.close(figure)
    return [pdf.name, png.name]


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        0.012,
        0.985,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=12.0,
        fontweight="bold",
    )


def load_verified_summary(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    summary_path = root / "summary.json"
    verification_path = root / "verification.json"
    if not summary_path.is_file() or not verification_path.is_file():
        raise FileNotFoundError("MV7 summary.json or verification.json is absent")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete_MV7_JCP_full_budget_matrix":
        raise ValueError("MV7 summary is not complete")
    if verification.get("decision") != "verified":
        raise ValueError("MV7 verification is not verified")
    checks = summary.get("checks", {})
    if not checks or not all(bool(value) for value in checks.values()):
        raise ValueError("MV7 locked checks are not all true")
    if set(summary.get("curves", {})) != set(METHODS):
        raise ValueError("MV7 method set or order differs from the locked matrix")
    return summary, verification


def scalar_identity(value: np.ndarray | object) -> str:
    array = np.asarray(value)
    if array.ndim == 0:
        return str(array.item())
    return ",".join(str(item) for item in array.reshape(-1).tolist())


def evaluation_seed(value: np.ndarray | object) -> str:
    array = np.asarray(value).reshape(-1)
    if len(array) == 0:
        raise ValueError("Evaluation identity is empty")
    seed = array[0].item()
    if isinstance(seed, (int, np.integer)):
        return str(int(seed))
    if isinstance(seed, (float, np.floating)) and np.isfinite(seed) and float(seed).is_integer():
        return str(int(seed))
    return str(seed)


def condition_index(data: Mapping[str, np.ndarray], condition: str, seed: str | None = None) -> int:
    labels = np.asarray(data["identity_condition"]).astype(str)
    matches = np.flatnonzero(labels == condition)
    if seed is not None:
        identities = np.asarray(data["identity_numeric"])
        matches = np.asarray(
            [index for index in matches if evaluation_seed(identities[index]) == seed],
            dtype=int,
        )
    if len(matches) == 0:
        suffix = "" if seed is None else f" with evaluation seed {seed}"
        raise ValueError(f"Condition {condition}{suffix} is absent from predictions")
    return int(matches[0])


def nice_ceiling(value: float, minimum: float = 0.5) -> float:
    value = max(float(value), minimum)
    exponent = 10.0 ** math.floor(math.log10(value))
    scaled = value / exponent
    for multiplier in (1.0, 2.0, 2.5, 5.0, 10.0):
        if scaled <= multiplier + 1.0e-12:
            return float(multiplier * exponent)
    return float(10.0 * exponent)


def load_temperature_payload(
    mv7_root: Path,
    mv6_root: Path,
    condition: str,
) -> tuple[dict[str, np.ndarray], str]:
    payload: dict[str, np.ndarray] = {}
    baseline_b1 = mv7_root / "baselines" / "budget_1" / "predictions.npz"
    baseline_b10 = mv7_root / "baselines" / "budget_10" / "predictions.npz"
    if not baseline_b1.is_file() or not baseline_b10.is_file():
        raise FileNotFoundError("MV7 baseline predictions for B=1 or B=10 are absent")

    with np.load(baseline_b1, allow_pickle=False) as data:
        index = condition_index(data, condition)
        identity = scalar_identity(np.asarray(data["identity_numeric"])[index])
        evaluation_seed_id = evaluation_seed(np.asarray(data["identity_numeric"])[index])
        payload["reference"] = np.asarray(data["target"][index, 0], dtype=np.float64)
        payload["raw_b1"] = np.asarray(data["raw"][index, 0], dtype=np.float64)
        payload["gaussian_b1"] = np.asarray(data["gaussian_like"][index, 0], dtype=np.float64)
        payload["tsvd_b1"] = np.asarray(data["tsvd_pod_type"][index, 0], dtype=np.float64)

    with np.load(baseline_b10, allow_pickle=False) as data:
        index = condition_index(data, condition, evaluation_seed_id)
        target_b10 = np.asarray(data["target"][index, 0], dtype=np.float64)
        if not np.array_equal(target_b10, payload["reference"]):
            raise ValueError(f"Reference temperature changed between B=1 and B=10 for {condition}")
        payload["raw_b10"] = np.asarray(data["raw"][index, 0], dtype=np.float64)

    for architecture in ARCHITECTURES:
        predictions: list[np.ndarray] = []
        for training_seed in TRAINING_SEEDS:
            path = prediction_path(mv7_root, mv6_root, 1, architecture, training_seed)
            if not path.is_file():
                raise FileNotFoundError(f"MV6 reused B=1 prediction is absent: {path}")
            with np.load(path, allow_pickle=False) as data:
                index = condition_index(data, condition, evaluation_seed_id)
                target = np.asarray(data["target"][index, 0], dtype=np.float64)
                if not np.array_equal(target, payload["reference"]):
                    raise ValueError(f"Neural reference identity changed in {path}")
                predictions.append(
                    np.asarray(data["architecture_prediction"][index, 0], dtype=np.float64)
                )
        payload[architecture] = np.mean(np.stack(predictions, axis=0), axis=0)

    reference_shape = payload["reference"].shape
    if len(reference_shape) != 2 or any(field.shape != reference_shape for field in payload.values()):
        raise ValueError(f"Temperature-field shapes differ for {condition}")
    return payload, identity


def temperature_physical_figure(
    output: Path,
    mv7_root: Path,
    mv6_root: Path,
    condition: str,
) -> tuple[list[str], list[dict[str, Any]]]:
    payload, identity = load_temperature_payload(mv7_root, mv6_root, condition)
    reference = payload["reference"]
    errors = {
        key: 100.0 * (field - reference) / np.maximum(np.abs(reference), EPSILON)
        for key, field in payload.items()
        if key != "reference"
    }
    finite_error = np.concatenate(
        [np.abs(value[np.isfinite(value)]).reshape(-1) for value in errors.values()]
    )
    percentile_value = float(np.percentile(finite_error, 99.5))
    error_limit = nice_ceiling(percentile_value)
    temperature_min = float(np.nanmin(reference))
    temperature_max = float(np.nanmax(reference))

    figure = plt.figure(figsize=(22.4, 7.15))
    grid = figure.add_gridspec(
        2,
        len(PHYSICAL_COLUMN_SPECS) + 1,
        width_ratios=[1.0] * len(PHYSICAL_COLUMN_SPECS) + [0.065],
        left=0.038,
        right=0.965,
        bottom=0.105,
        top=0.915,
        wspace=0.18,
        hspace=0.30,
    )
    axes = np.empty((2, len(PHYSICAL_COLUMN_SPECS)), dtype=object)
    for row in range(2):
        for column in range(len(PHYSICAL_COLUMN_SPECS)):
            axes[row, column] = figure.add_subplot(grid[row, column])
    temperature_cax = figure.add_subplot(grid[0, -1])
    error_cax = figure.add_subplot(grid[1, -1])

    temperature_image = None
    error_image = None
    metadata_rows: list[dict[str, Any]] = []
    for column, (key, title) in enumerate(PHYSICAL_COLUMN_SPECS):
        field = payload[key]
        top = axes[0, column]
        temperature_image = top.imshow(
            field,
            origin="lower",
            extent=(0.0, 1.0, 0.0, 1.0),
            cmap="coolwarm",
            vmin=temperature_min,
            vmax=temperature_max,
            interpolation="bilinear",
            rasterized=True,
        )
        top.set_title(title, pad=4.5, linespacing=0.90)
        bottom = axes[1, column]
        if key == "reference":
            bottom.set_facecolor("#F7F7F7")
            bottom.text(
                0.5,
                0.5,
                "Reference",
                transform=bottom.transAxes,
                ha="center",
                va="center",
                color="0.45",
                fontsize=9.0,
            )
            error_rms = 0.0
            error_abs_995 = 0.0
        else:
            error = errors[key]
            error_image = bottom.imshow(
                error,
                origin="lower",
                extent=(0.0, 1.0, 0.0, 1.0),
                cmap="RdBu_r",
                vmin=-error_limit,
                vmax=error_limit,
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
        metadata_rows.append(
            {
                "condition": condition,
                "evaluation_identity": identity,
                "column_key": key,
                "column_label": title.replace("\n", " "),
                "temperature_min_K": float(np.nanmin(field)),
                "temperature_max_K": float(np.nanmax(field)),
                "signed_relative_error_rms_percent": error_rms,
                "signed_relative_error_abs_99p5_percent": error_abs_995,
            }
        )

    if temperature_image is None or error_image is None:
        raise RuntimeError("Physical temperature figure did not create colorbar images")
    temperature_ticks = np.linspace(temperature_min, temperature_max, 5)
    temperature_bar = figure.colorbar(
        temperature_image,
        cax=temperature_cax,
        ticks=temperature_ticks,
        extend="both",
    )
    temperature_bar.set_label(r"Temperature, $T$ [K]", labelpad=8.0)
    temperature_bar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f}"))
    error_ticks = np.linspace(-error_limit, error_limit, 5)
    error_bar = figure.colorbar(
        error_image,
        cax=error_cax,
        ticks=error_ticks,
        extend="both",
    )
    error_bar.set_label("Signed relative error [%]", labelpad=8.0)
    error_bar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    stem = output / f"mv7_jcp_temperature_physical_{condition}"
    return save_figure(figure, stem), metadata_rows


def sampling_efficiency_figure(output: Path, summary: Mapping[str, Any]) -> list[str]:
    curves = summary["curves"]
    equivalence = summary["raw_scaling_and_effective_variance_reduction"]
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.25))
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.13, top=0.82, wspace=0.27)

    for method in METHODS:
        mean = [float(curves[method][str(b)]["mean_composite_nrmse"]) for b in BUDGETS]
        sd = [float(curves[method][str(b)].get("training_seed_sd") or 0.0) for b in BUDGETS]
        axes[0].errorbar(
            BUDGETS,
            mean,
            yerr=sd if method in ARCHITECTURES else None,
            color=COLORS[method],
            marker=MARKERS[method],
            markersize=5.6,
            linewidth=1.55,
            elinewidth=0.9,
            capsize=2.4,
            label=DISPLAY_NAMES[method],
            zorder=4 if method in ARCHITECTURES else 3,
        )
        gain = [
            float(
                equivalence["by_method_budget"][method][str(b)][
                    "empirical_equivalent_budget_over_consumed_budget"
                ]
            )
            for b in BUDGETS
        ]
        axes[1].plot(
            BUDGETS,
            gain,
            color=COLORS[method],
            marker=MARKERS[method],
            markersize=5.6,
            linewidth=1.55,
            zorder=4 if method in ARCHITECTURES else 3,
        )

    raw_one = float(curves["raw"]["1"]["mean_composite_nrmse"])
    guide = [raw_one / math.sqrt(b) for b in BUDGETS]
    guide_line = axes[0].plot(
        BUDGETS,
        guide,
        color="0.45",
        linestyle=(0, (4, 2)),
        linewidth=1.25,
        label=r"$B^{-1/2}$ guide",
        zorder=1,
    )[0]
    axes[1].axhline(1.0, color="0.25", linewidth=1.0, linestyle=(0, (4, 2)))

    for axis in axes:
        axis.set_xscale("log")
        axis.set_xticks(BUDGETS, [str(value) for value in BUDGETS])
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.set_xlabel(r"DSMC sampling blocks, $B$")
        axis.grid(True, which="major", color="0.86", linewidth=0.65)
        axis.grid(True, which="minor", color="0.93", linewidth=0.45)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Composite NRMSE")
    axes[1].set_yscale("log")
    axes[1].set_ylabel(r"Raw-equivalent budget / consumed $B$")
    axes[1].set_ylim(0.8, 26.0)
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    axes[0].text(
        0.02,
        0.055,
        "Neural error bars: SD over three training initializations",
        transform=axes[0].transAxes,
        fontsize=8.0,
        color="0.30",
    )

    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index(DISPLAY_NAMES[method]) for method in METHODS] + [labels.index(r"$B^{-1/2}$ guide")]
    figure.legend(
        [handles[index] for index in order],
        [labels[index] for index in order],
        loc="upper center",
        bbox_to_anchor=(0.53, 0.985),
        ncol=4,
        frameon=False,
        columnspacing=1.5,
        handlelength=2.6,
    )
    return save_figure(figure, output / "mv7_jcp_sampling_accuracy_and_efficiency")


def noninferiority_figure(output: Path, summary: Mapping[str, Any]) -> list[str]:
    values = summary["noninferiority_to_raw_budget_10"]
    method_order = (
        "raw",
        "gaussian_like",
        "tsvd_pod_type",
        "corrected_unet",
        "nafnet_small",
        "mambairv2_tiny_adapted",
        "fno_residual_small",
    )
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), sharex=True, sharey=True)
    figure.subplots_adjust(left=0.145, right=0.985, bottom=0.105, top=0.86, wspace=0.10, hspace=0.25)
    positions = np.arange(len(method_order), dtype=float)
    savings = {1: "10× fewer", 2: "5× fewer", 5: "2× fewer", 10: "reference budget"}

    for panel, (axis, budget) in enumerate(zip(axes.flat, BUDGETS)):
        for position, method in zip(positions, method_order):
            record = values[method][str(budget)]
            center = float(record["geometric_mean_ratio_to_raw10"])
            upper = float(record["one_sided_95_upper_ratio_to_raw10"])
            passed = bool(record["noninferior"])
            axis.hlines(position, center, upper, color=COLORS[method], linewidth=1.45)
            axis.vlines(upper, position - 0.12, position + 0.12, color=COLORS[method], linewidth=1.2)
            axis.plot(
                center,
                position,
                marker=MARKERS[method],
                markersize=7.2,
                markerfacecolor=COLORS[method] if passed else "white",
                markeredgecolor=COLORS[method],
                markeredgewidth=1.4,
                linestyle="none",
                zorder=4,
            )
        axis.axvline(1.0, color="black", linewidth=1.0)
        axis.axvline(NONINFERIORITY_MARGIN, color="#D55E00", linewidth=1.2, linestyle=(0, (4, 2)))
        axis.set_xscale("log")
        axis.set_xlim(0.50, 3.10)
        axis.set_ylim(len(method_order) - 0.5, -0.5)
        axis.set_title(rf"$B={budget}$ ({savings[budget]})", pad=6.0)
        axis.grid(True, axis="x", which="major", color="0.87", linewidth=0.6)
        panel_label(axis, f"({chr(97 + panel)})")
    axes[0, 0].set_yticks(positions, [DISPLAY_NAMES[method] for method in method_order])
    axes[1, 0].set_yticks(positions, [DISPLAY_NAMES[method] for method in method_order])
    tick_values = (0.6, 0.8, 1.0, 1.1, 1.5, 2.0, 3.0)
    for axis in axes.flat:
        axis.xaxis.set_major_locator(FixedLocator(tick_values))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axis.xaxis.set_minor_formatter(NullFormatter())
    axes[1, 0].set_xlabel("Paired composite-error ratio to Raw@B=10")
    axes[1, 1].set_xlabel("Paired composite-error ratio to Raw@B=10")
    figure.legend(
        handles=[
            Line2D([0], [0], marker="o", color="0.2", markerfacecolor="0.2", linestyle="none", label="Non-inferior"),
            Line2D([0], [0], marker="o", color="0.2", markerfacecolor="white", linestyle="none", label="Not established"),
            Line2D([0], [0], color="#D55E00", linestyle=(0, (4, 2)), label="Locked 1.10 margin"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.56, 0.975),
        ncol=3,
        frameon=False,
    )
    figure.text(
        0.56,
        0.905,
        "Point = geometric mean; right cap = one-sided 95% condition-clustered upper bound",
        ha="center",
        fontsize=9.0,
        color="0.28",
    )
    return save_figure(figure, output / "mv7_jcp_noninferiority_forest")


def bias_floor_figure(output: Path, summary: Mapping[str, Any]) -> list[str]:
    diagnostics = summary["bias_floor_diagnostics"]
    changes = np.asarray(
        [100.0 * (float(diagnostics[method]["B10_over_B5_error"]) - 1.0) for method in ARCHITECTURES]
    )
    figure, axis = plt.subplots(figsize=(7.4, 4.5), constrained_layout=True)
    positions = np.arange(len(ARCHITECTURES))
    bars = axis.bar(
        positions,
        changes,
        color=[COLORS[method] for method in ARCHITECTURES],
        width=0.64,
        edgecolor="black",
        linewidth=0.45,
    )
    axis.axhline(0.0, color="black", linewidth=1.0)
    axis.set_xticks(positions, [SHORT_NAMES[method] for method in ARCHITECTURES])
    axis.set_ylabel(r"Change in composite NRMSE, $B=5\rightarrow10$ [%]")
    axis.set_ylim(-6.4, 33.2)
    axis.grid(True, axis="y", color="0.88", linewidth=0.6)
    axis.set_axisbelow(True)
    for bar, value in zip(bars, changes):
        if value < 0:
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                value / 2.0,
                f"{value:+.1f}%",
                ha="center",
                va="center",
                color="white",
                fontsize=9.2,
                fontweight="bold",
            )
            continue
        axis.annotate(
            f"{value:+.1f}%",
            (bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.2,
        )
    return save_figure(figure, output / "mv7_jcp_large_budget_bias_floor")


def prediction_path(mv7_root: Path, mv6_root: Path, budget: int, architecture: str, seed: int) -> Path:
    if budget == 1:
        directory = mv6_root / "tasks" / architecture / f"training_seed_{seed}"
    else:
        directory = mv7_root / "tasks" / f"budget_{budget}" / architecture / f"training_seed_{seed}"
    return directory / "predictions.npz"


def normalized_error_arrays(
    mv7_root: Path,
    mv6_root: Path,
    budget: int,
    architecture: str,
    condition: str = "kn0p1_u400",
) -> np.ndarray:
    values: list[np.ndarray] = []
    for seed in TRAINING_SEEDS:
        path = prediction_path(mv7_root, mv6_root, budget, architecture, seed)
        with np.load(path, allow_pickle=False) as data:
            labels = np.asarray(data["identity_condition"]).astype(str)
            mask = labels == condition
            prediction = np.asarray(data["architecture_prediction"], dtype=np.float64)[mask]
            target = np.asarray(data["target"], dtype=np.float64)[mask]
        if len(prediction) == 0:
            raise ValueError(f"Condition {condition} is absent from {path}")
        temperature_scale = max(float(np.ptp(target[:, 0])), 1.0)
        error = np.empty_like(prediction, dtype=np.float64)
        error[:, 0] = (prediction[:, 0] - target[:, 0]) / temperature_scale
        error[:, 1] = (prediction[:, 1] - target[:, 1]) / 400.0
        values.append(error)
    return np.concatenate(values, axis=0)


def radial_spectrum(errors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = errors.shape[-2:]
    yy, xx = np.indices((height, width))
    radius = np.sqrt((yy - height // 2) ** 2 + (xx - width // 2) ** 2)
    bins = np.floor(radius).astype(int)
    maximum = min(height, width) // 2
    power = np.zeros(maximum, dtype=np.float64)
    count = np.zeros(maximum, dtype=np.float64)
    for sample in errors:
        for field in sample:
            spectrum = np.abs(np.fft.fftshift(np.fft.fft2(field))) ** 2
            for index in range(maximum):
                mask = bins == index
                power[index] += float(np.sum(spectrum[mask]))
                count[index] += int(np.sum(mask))
    power /= np.maximum(count, 1.0)
    power /= max(float(power.sum()), EPSILON)
    return np.arange(maximum, dtype=float), power


def fno_diagnostic_figure(output: Path, mv7_root: Path, mv6_root: Path) -> tuple[list[str], list[dict[str, Any]]]:
    selected_architectures = ("mambairv2_tiny_adapted", "fno_residual_small")
    rows: list[dict[str, Any]] = []
    spectra: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    for architecture in selected_architectures:
        for budget in (1, 10):
            errors = normalized_error_arrays(mv7_root, mv6_root, budget, architecture)
            height, width = errors.shape[-2:]
            boundary = np.zeros((height, width), dtype=bool)
            boundary[:2] = True
            boundary[-2:] = True
            boundary[:, :2] = True
            boundary[:, -2:] = True
            squared = np.mean(errors**2, axis=(0, 1))
            boundary_mse = float(np.mean(squared[boundary]))
            interior_mse = float(np.mean(squared[~boundary]))
            rows.append(
                {
                    "architecture": architecture,
                    "budget_blocks": budget,
                    "boundary_band_mse": boundary_mse,
                    "interior_mse": interior_mse,
                    "boundary_over_interior": boundary_mse / max(interior_mse, EPSILON),
                }
            )
            spectra[(architecture, budget)] = radial_spectrum(errors)

    with (output / "mv7_jcp_fno_absolute_error_diagnostics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "mv7_jcp_fno_radial_spectra.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("architecture", "budget_blocks", "radial_wavenumber", "normalized_error_power"))
        for (architecture, budget), (wave, power) in spectra.items():
            writer.writerows((architecture, budget, float(x), float(y)) for x, y in zip(wave, power))

    figure, axes = plt.subplots(1, 2, figsize=(12.6, 4.85))
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.14, top=0.80, wspace=0.28)
    for architecture in selected_architectures:
        selected = [row for row in rows if row["architecture"] == architecture]
        budgets = [int(row["budget_blocks"]) for row in selected]
        axes[0].plot(
            budgets,
            [float(row["boundary_band_mse"]) for row in selected],
            color=COLORS[architecture],
            marker=MARKERS[architecture],
            linewidth=1.6,
            label=f"{SHORT_NAMES[architecture]}: boundary",
        )
        axes[0].plot(
            budgets,
            [float(row["interior_mse"]) for row in selected],
            color=COLORS[architecture],
            marker=MARKERS[architecture],
            linewidth=1.45,
            linestyle=(0, (4, 2)),
            markerfacecolor="white",
            label=f"{SHORT_NAMES[architecture]}: interior",
        )
        for budget, style in ((1, "-"), (10, (0, (4, 2)))):
            wave, power = spectra[(architecture, budget)]
            axes[1].semilogy(
                wave[1:],
                power[1:],
                color=COLORS[architecture],
                linestyle=style,
                linewidth=1.55,
                label=f"{SHORT_NAMES[architecture]}, B={budget}",
            )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xticks((1, 10), ("1", "10"))
    axes[0].xaxis.set_minor_formatter(NullFormatter())
    axes[0].set_xlabel(r"DSMC sampling blocks, $B$")
    axes[0].set_ylabel("Normalized absolute error MSE")
    axes[0].grid(True, which="both", color="0.88", linewidth=0.6)
    axes[0].legend(frameon=False, ncol=2, fontsize=8.4, loc="upper right")
    axes[1].set_xlabel("Radial Fourier wavenumber")
    axes[1].set_ylabel("Normalized error power")
    axes[1].grid(True, which="both", color="0.88", linewidth=0.6)
    axes[1].legend(frameon=False, fontsize=8.6, loc="upper right")
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    figure.text(
        0.53,
        0.96,
        r"Diagnostic condition: $Kn=0.1$, $U_{\rm lid}=400\ {\rm m\,s^{-1}}$",
        ha="center",
        fontsize=11.0,
    )
    return save_figure(figure, output / "mv7_jcp_fno_error_structure"), rows


def torch_load_checkpoint(torch: Any, path: Path) -> Mapping[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def benchmark_budget_one_inference(
    summary: Mapping[str, Any],
    mv7_root: Path,
    mv6_root: Path,
    existing_m3_root: Path,
    mv3_root: Path,
    reference_root: Path,
    repo_root: Path,
    repeats: int = 7,
) -> dict[str, Any]:
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from vgdsmc import mohammadzadeh_architecture_screen as mv6
    from vgdsmc import mohammadzadeh_mv7_jcp_budget_matrix as mv7

    del mv7_root
    _, _, test, _, _ = mv7._budget_data(existing_m3_root, mv3_root, reference_root, 1)
    test_x = np.asarray(test[0], dtype=np.float32)
    torch, _, _ = mv6._torch_components()
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    records: list[dict[str, Any]] = []
    for architecture in ARCHITECTURES:
        for seed in TRAINING_SEEDS:
            directory = mv6_root / "tasks" / architecture / f"training_seed_{seed}"
            checkpoint = torch_load_checkpoint(torch, directory / "model.pt")
            scaling = {key: np.asarray(value) for key, value in checkpoint["scaling"].items()}
            model = mv6.build_architecture(architecture, int(test_x.shape[1]))
            model.load_state_dict(checkpoint["state_dict"])
            prediction, _ = mv6.predict_bounded(model, test_x, scaling, 6)
            if (
                prediction.shape[0] != test_x.shape[0]
                or prediction.shape[2:] != test_x.shape[2:]
                or not np.all(np.isfinite(prediction))
            ):
                raise ValueError(f"B=1 inference benchmark output invalid: {architecture}/{seed}")
            timings = []
            for _ in range(repeats):
                started = time.perf_counter()
                mv6.predict_bounded(model, test_x, scaling, 6)
                timings.append(time.perf_counter() - started)
            records.append(
                {
                    "architecture": architecture,
                    "training_seed": seed,
                    "confirmatory_arrays": int(len(test_x)),
                    "repeats": repeats,
                    "median_total_seconds": float(np.median(timings)),
                    "mean_total_seconds": float(np.mean(timings)),
                    "median_seconds_per_array": float(np.median(timings) / len(test_x)),
                }
            )

    block_seconds = float(summary["cost_accounting"]["reference_wall_seconds_per_block_including_amortized_burn_in"])
    by_architecture: dict[str, Any] = {}
    for architecture in ARCHITECTURES:
        selected = [record for record in records if record["architecture"] == architecture]
        inference_per_array = float(np.mean([record["median_seconds_per_array"] for record in selected]))
        training_seconds = float(
            summary["cost_accounting"]["by_architecture_budget"][architecture]["1"]["training_wall_seconds_mean"]
        )
        denominator = 9.0 * block_seconds - inference_per_array
        by_architecture[architecture] = {
            "inference_seconds_per_array_mean_of_seed_medians": inference_per_array,
            "training_wall_seconds_mean": training_seconds,
            "marginal_seconds_saved_per_use_vs_raw10": denominator,
            "training_only_break_even_uses_lower_bound": training_seconds / denominator if denominator > 0 else None,
            "shared_training_data_cost_included": False,
        }
    return {
        "status": "complete_MV7_reused_budget_one_inference_timing_closure",
        "device": "CPU",
        "timing_rule": f"one warm-up followed by {repeats} repetitions; report per-seed median",
        "records": records,
        "by_architecture": by_architecture,
        "interpretation_guard": "training-only break-even excludes shared training-data generation and is a lower bound",
    }


def reusable_budget_one_benchmark(mv7_root: Path) -> dict[str, Any] | None:
    candidates = sorted(
        mv7_root.glob("publication_suite_*/mv7_b1_inference_cost_closure.json"),
        reverse=True,
    )
    for path in candidates:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if record.get("status") != "complete_MV7_reused_budget_one_inference_timing_closure":
            continue
        if len(record.get("records", [])) != len(ARCHITECTURES) * len(TRAINING_SEEDS):
            continue
        reused = dict(record)
        reused["reused_from"] = str(path)
        reused["reuse_rule"] = "immutable completed CPU timing closure from the same verified MV7 root"
        return reused
    return None


def write_key_results(output: Path, summary: Mapping[str, Any]) -> None:
    with (output / "mv7_jcp_key_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("method", "minimum_noninferior_budget", "sampling_reduction_vs_raw10", "upper_ratio_at_minimum_budget"))
        for method in METHODS:
            passing = [b for b in BUDGETS if summary["noninferiority_to_raw_budget_10"][method][str(b)]["noninferior"]]
            minimum = min(passing) if passing else None
            upper = None if minimum is None else summary["noninferiority_to_raw_budget_10"][method][str(minimum)]["one_sided_95_upper_ratio_to_raw10"]
            writer.writerow((method, minimum, None if minimum is None else 10.0 / minimum, upper))


def write_readme(output: Path, source_root: Path, benchmark_status: str) -> None:
    text = f"""# Mohammadzadeh MV7 JCP publication suite

This package is an immutable postprocessing product of the recursively verified
MV7 budget matrix. It does not replace or edit any locked MV7 result.

## Figures

- `mv7_jcp_sampling_accuracy_and_efficiency`: mean composite NRMSE and empirical
  Raw-equivalent budget per consumed DSMC block. Neural error bars are the
  sample SD across three locked training initializations.
- `mv7_jcp_noninferiority_forest`: the locked paired comparison with Raw@B=10.
  Points are geometric mean ratios; the right cap is the one-sided 95%
  condition-clustered upper confidence bound. Filled points pass the locked
  1.10 non-inferiority margin.
- `mv7_jcp_large_budget_bias_floor`: percentage change from B=5 to B=10 for the
  four neural architectures. Positive values denote degradation, not an
  automatic task failure.
- `mv7_jcp_fno_error_structure`: absolute normalized boundary/interior MSE and
  radial error spectra at Kn=0.1, U_lid=400 m/s. These diagnostics describe the
  error structure but do not prove a spectral-bias or periodic-boundary cause.
- `mv7_jcp_temperature_physical_<condition>`: the physical temperature field
  for Reference, budget-one Raw/Gaussian/TSVD and four neural architectures,
  plus Raw@B=10. The lower row is the signed local relative error in percent.
  The same absolute and error color scales are used across every method within
  each condition.

Each figure is supplied as vector PDF and 600-dpi PNG. Fonts are embedded in the
PDF, legends are outside dense data regions, method colors/markers are consistent,
and inferential and descriptive uncertainty are not conflated.

## Cost closure

Budget-one inference timing status: `{benchmark_status}`. The reported
training-only break-even is explicitly a lower bound because shared training-data
generation is not included.

## Provenance

- Source MV7 root: `{source_root}`
- Generated UTC: `{datetime.now(timezone.utc).isoformat()}`
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def create_archive(output: Path, archive_directory: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_directory.mkdir(parents=True, exist_ok=True)
    archive = archive_directory / f"MOHAMMADZADEH_MV7_JCP_PUBLICATION_FIGURES_{timestamp}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zipped:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                zipped.write(path, arcname=f"{output.name}/{path.relative_to(output)}")
    return archive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mv7-root", type=Path)
    parser.add_argument("--mv6-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-prediction-diagnostics", action="store_true")
    parser.add_argument("--skip-b1-benchmark", action="store_true")
    parser.add_argument("--force-b1-benchmark", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    mv7_root = (args.mv7_root or Path(os.environ["MV7_OUTPUT_ROOT"])).expanduser().resolve()
    mv6_root = (args.mv6_root or Path(os.environ.get("MV7_MV6_ROOT", "."))).expanduser().resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output or (mv7_root / f"publication_suite_{timestamp}")).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary, verification = load_verified_summary(mv7_root)

    generated: list[str] = []
    print("STAGE=physical_temperature_contours", flush=True)
    physical_rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        print(f"PHYSICAL_CONDITION={condition}", flush=True)
        files, rows = temperature_physical_figure(output, mv7_root, mv6_root, condition)
        generated.extend(files)
        physical_rows.extend(rows)
    with (output / "mv7_jcp_temperature_physical_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(physical_rows[0]))
        writer.writeheader()
        writer.writerows(physical_rows)
    generated.append("mv7_jcp_temperature_physical_metrics.csv")

    print("STAGE=statistical_summary_figures", flush=True)
    generated.extend(sampling_efficiency_figure(output, summary))
    generated.extend(noninferiority_figure(output, summary))
    generated.extend(bias_floor_figure(output, summary))
    fno_rows: list[dict[str, Any]] = []
    if not args.skip_prediction_diagnostics:
        print("STAGE=fno_diagnostics", flush=True)
        files, fno_rows = fno_diagnostic_figure(output, mv7_root, mv6_root)
        generated.extend(files)
        generated.extend(("mv7_jcp_fno_absolute_error_diagnostics.csv", "mv7_jcp_fno_radial_spectra.csv"))

    benchmark: dict[str, Any]
    if args.skip_b1_benchmark:
        benchmark = {"status": "skipped_by_explicit_command_line_flag"}
    else:
        benchmark = None if args.force_b1_benchmark else reusable_budget_one_benchmark(mv7_root)
        if benchmark is None:
            print("STAGE=budget_one_inference_benchmark", flush=True)
            repo_root = Path(os.environ["MV7_REPO_ROOT"]).resolve()
            benchmark = benchmark_budget_one_inference(
                summary,
                mv7_root,
                mv6_root,
                Path(os.environ["MV7_M3_ROOT"]).resolve(),
                Path(os.environ["MV7_MV3_ROOT"]).resolve(),
                Path(os.environ["MV7_REFERENCE_ROOT"]).resolve(),
                repo_root,
            )
        else:
            print(f"STAGE=reuse_budget_one_inference_benchmark SOURCE={benchmark['reused_from']}", flush=True)
    (output / "mv7_b1_inference_cost_closure.json").write_text(
        json.dumps(benchmark, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_key_results(output, summary)
    shutil.copy2(mv7_root / "summary.json", output / "provenance_mv7_summary.json")
    shutil.copy2(mv7_root / "verification.json", output / "provenance_mv7_verification.json")
    shutil.copy2(Path(__file__).resolve(), output / Path(__file__).name)
    write_readme(output, mv7_root, str(benchmark["status"]))

    metadata = {
        "stage": "MV7_JCP_publication_figure_and_budget_one_cost_closure",
        "status": "complete",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_mv7_root": str(mv7_root),
        "source_summary_sha256": sha256(mv7_root / "summary.json"),
        "source_verification_sha256": sha256(mv7_root / "verification.json"),
        "source_verification_decision": verification["decision"],
        "repository_head": git_head(Path(os.environ["MV7_REPO_ROOT"]) if "MV7_REPO_ROOT" in os.environ else None),
        "figure_standard": {
            "pdf": "vector with embedded TrueType fonts",
            "png_dpi": 600,
            "method_encoding": "fixed color plus redundant marker/linestyle",
            "uncertainty": "descriptive training-seed SD only on curve; locked inferential upper bounds in forest figure",
        },
        "fno_interpretation_guard": "diagnostics describe but do not prove a causal spectral or boundary mechanism",
        "fno_rows": fno_rows,
        "physical_temperature_rows": physical_rows,
        "files": generated,
    }
    (output / "figure_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_paths = sorted(path for path in output.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    (output / "SHA256SUMS").write_text(
        "\n".join(f"{sha256(path)}  {path.relative_to(output)}" for path in manifest_paths) + "\n",
        encoding="utf-8",
    )
    print("STAGE=archive", flush=True)
    archive = create_archive(output, Path(os.environ.get("MV7_ARCHIVE_DIR", str(Path.home()))).expanduser())
    if archive.stat().st_size > 450 * 1024 * 1024:
        raise SystemExit("Publication archive exceeds the 450 MiB upload limit")
    print(f"PUBLICATION_DIR={output}")
    print(f"ARCHIVE={archive}")
    print(f"ARCHIVE_SIZE_MIB={archive.stat().st_size / 1024**2:.2f}")
    print(f"ARCHIVE_SHA256={sha256(archive)}")


if __name__ == "__main__":
    main()
