#!/usr/bin/env python3
"""Regenerate cavity figures 3--8 from the locked JCP1 field archive."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.fft import dctn


NAVY = "#17365D"
BLUE = "#2468B4"
TEAL = "#168AAD"
RED = "#D1495B"
GRAY = "#8A939E"
PURPLE = "#6F3C8E"

METHODS = (
    ("target", "Independent reference", "#111111", "-", 3.0),
    ("method_raw_b3", r"Raw DSMC, $B=3$", GRAY, ":", 2.6),
    ("method_pnet_alone", "Vision prior", PURPLE, "--", 2.7),
    ("method_pnet_cross_block", r"Mamba–Wiener, $B=3$", RED, "-", 3.2),
    ("method_raw_b10", r"Raw DSMC, $B=10$", BLUE, "-.", 2.7),
)


def set_style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "mathtext.fontset": "stixsans",
        "font.size": 21, "axes.labelsize": 24, "axes.titlesize": 23,
        "xtick.labelsize": 19, "ytick.labelsize": 19, "legend.fontsize": 19,
        "axes.linewidth": 1.8, "lines.linewidth": 3.5, "pdf.fonttype": 42,
        "ps.fonttype": 42, "savefig.facecolor": "white",
    })


def save(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.pdf", dpi=300, bbox_inches="tight", pad_inches=0.035, facecolor="white")
    fig.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.035, facecolor="white")
    plt.close(fig)


def robust_limit(*arrays: np.ndarray, percentile: float = 99.6) -> float:
    merged = np.concatenate([np.abs(np.asarray(a)[np.isfinite(a)]).ravel() for a in arrays])
    return float(np.percentile(merged, percentile))


def condition_indices(data: np.lib.npyio.NpzFile, condition: str) -> np.ndarray:
    return np.flatnonzero(np.asarray(data["conditions"]) == condition)


def qy_index(data: np.lib.npyio.NpzFile) -> int:
    return int(np.flatnonzero(np.asarray(data["field_names"]) == "qy")[0])


def panel_axes(ax: plt.Axes, index: int, title: str, show_x: bool, show_y: bool) -> None:
    ax.set_title(f"({'abcdefghijklmnopqrst'[index]}) {title}", loc="left", pad=9, fontweight="bold", color=NAVY)
    ax.set_xticks([0, 0.5, 1.0]); ax.set_yticks([0, 0.5, 1.0])
    ax.tick_params(direction="out", length=5.0, width=1.0)
    if show_x:
        ax.set_xlabel(r"$x/L$")
    else:
        ax.set_xticklabels([])
    if show_y:
        ax.set_ylabel(r"$y/L$")
    else:
        ax.set_yticklabels([])


def ensemble_figure(data: np.lib.npyio.NpzFile, output: Path) -> None:
    qi = qy_index(data)
    conditions = ("kn0p1_u400", "kn0p08_u350")
    row_labels = (r"A: $Kn=0.10$, $U_{\rm lid}=400$", r"B: $Kn=0.08$, $U_{\rm lid}=350$")
    rows = []
    for condition in conditions:
        idx = condition_indices(data, condition)
        target = np.asarray(data["target"])[idx, qi]
        proposed = np.asarray(data["method_pnet_frozen_gain"])[idx, qi]
        comparator = np.asarray(data["method_raw_b10"])[idx, qi]
        target_mean = np.mean(target, axis=0)
        proposed_mean = np.mean(proposed, axis=0)
        bias = proposed_mean - target_mean
        gain = np.sqrt(np.mean((comparator - target) ** 2, axis=0)) - np.sqrt(np.mean((proposed - target) ** 2, axis=0))
        rows.append((target_mean, proposed_mean, bias, gain))
    field_limit = robust_limit(*(row[j] for row in rows for j in (0, 1)), percentile=99.7)
    bias_limit = robust_limit(*(row[2] for row in rows), percentile=99.5)
    gain_limit = robust_limit(*(row[3] for row in rows), percentile=99.5)

    fig, axes = plt.subplots(2, 4, figsize=(19.2, 9.3))
    titles = ("Reference mean", "Proposed mean", "Signed bias", "RMSE reduction")
    artists = []
    for row in range(2):
        for col in range(4):
            ax = axes[row, col]; values = rows[row][col]
            if col < 2:
                limit, cmap = field_limit, "RdBu_r"
            elif col == 2:
                limit, cmap = bias_limit, "PuOr_r"
            else:
                limit, cmap = gain_limit, "PuOr_r"
            artist = ax.imshow(values, origin="lower", extent=(0, 1, 0, 1), cmap=cmap, norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit), interpolation="nearest", aspect="equal")
            artists.append(artist)
            panel_axes(ax, row * 4 + col, titles[col], row == 1, col == 0)
            ax.title.set_fontsize(21)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.84, bottom=0.18, wspace=0.15, hspace=0.72)
    # Horizontal condition banners avoid collision with the shared y-axis and
    # remain readable after the landscape page is scaled by LaTeX.
    fig.text(0.51, 0.94, "Condition " + row_labels[0], ha="center", va="center", fontsize=22, color=NAVY, fontweight="bold")
    fig.text(0.51, 0.51, "Condition " + row_labels[1], ha="center", va="center", fontsize=22, color=NAVY, fontweight="bold")
    specs = ((artists[0], [0.08, 0.065, 0.25, 0.034], r"Mean $q_y$ ($10^6$ W m$^{-2}$)", field_limit), (artists[2], [0.39, 0.065, 0.25, 0.034], r"Bias $\Delta q_y$ ($10^6$ W m$^{-2}$)", bias_limit), (artists[3], [0.70, 0.065, 0.25, 0.034], r"RMSE reduction ($10^6$ W m$^{-2}$)", gain_limit))
    for artist, bounds, label, limit in specs:
        cb = fig.colorbar(artist, cax=fig.add_axes(bounds), orientation="horizontal")
        ticks = np.linspace(-0.75 * limit, 0.75 * limit, 3)
        cb.set_ticks(ticks)
        cb.set_ticklabels([f"{value:.2g}" for value in ticks])
        cb.set_label(label, fontsize=23); cb.ax.tick_params(labelsize=21, length=5.2)
    save(fig, output, "cavity_ensemble_physical_diagnostics")


def condition_field_figure(data: np.lib.npyio.NpzFile, output: Path, condition: str, stem: str) -> None:
    qi = qy_index(data)
    index = int(condition_indices(data, condition)[0])
    order = ("target", "method_raw_b3", "method_pnet_alone", "method_pnn_eb", "method_pnet_cross_block", "method_raw_b10")
    titles = (
        "Reference", r"Raw $B=3$", "Vision prior", "DC inverse", "Proposed", r"Raw $B=10$",
    )
    fields = [np.asarray(data[name])[index, qi] for name in order]
    reference = fields[0]
    rms = float(np.sqrt(np.mean(reference**2)))
    errors = [100.0 * (field - reference) / rms for field in fields]
    field_limit = robust_limit(*fields, percentile=99.7)
    error_limit = robust_limit(*errors[1:], percentile=99.3)
    field_norm = TwoSlopeNorm(vmin=-field_limit, vcenter=0.0, vmax=field_limit)
    error_norm = TwoSlopeNorm(vmin=-error_limit, vcenter=0.0, vmax=error_limit)

    fig = plt.figure(figsize=(19.6, 6.8))
    grid = fig.add_gridspec(2, 7, width_ratios=[1, 1, 1, 1, 1, 1, 0.075], left=0.045, right=0.968, bottom=0.12, top=0.91, wspace=0.12, hspace=0.25)
    axes = np.asarray([[fig.add_subplot(grid[row, col]) for col in range(6)] for row in range(2)])
    cax_top = fig.add_subplot(grid[0, 6]); cax_bottom = fig.add_subplot(grid[1, 6])
    top_artist = bottom_artist = None
    for col, (field, error, title) in enumerate(zip(fields, errors, titles, strict=True)):
        top_artist = axes[0, col].imshow(field, origin="lower", extent=(0, 1, 0, 1), cmap="RdBu_r", norm=field_norm, interpolation="nearest", aspect="equal")
        bottom_artist = axes[1, col].imshow(error, origin="lower", extent=(0, 1, 0, 1), cmap="PuOr_r", norm=error_norm, interpolation="nearest", aspect="equal")
        panel_axes(axes[0, col], col, title, False, col == 0)
        lower_title = ("Zero", r"Raw $B=3$", "Prior", "DC inverse", "Proposed", r"Raw $B=10$")[col]
        panel_axes(axes[1, col], 6 + col, lower_title, True, col == 0)
        axes[0, col].title.set_fontsize(21); axes[1, col].title.set_fontsize(20)
    if top_artist is None or bottom_artist is None:
        raise RuntimeError("cavity field figure incomplete")
    cb1 = fig.colorbar(top_artist, cax=cax_top); cb2 = fig.colorbar(bottom_artist, cax=cax_bottom)
    cb1.set_label(r"$q_y$ ($10^6$ W m$^{-2}$)", fontsize=23, labelpad=9)
    cb2.set_label("Signed error (%)", fontsize=23, labelpad=9)
    cb1.set_ticks(np.linspace(-0.75 * field_limit, 0.75 * field_limit, 5))
    cb2.set_ticks(np.linspace(-0.75 * error_limit, 0.75 * error_limit, 5))
    for cb in (cb1, cb2): cb.ax.tick_params(labelsize=21, length=5.2)
    save(fig, output, stem)


def mean_field(data: np.lib.npyio.NpzFile, name: str, indices: np.ndarray, qi: int) -> np.ndarray:
    return np.mean(np.asarray(data[name])[indices, qi], axis=0)


def profiles_figure(data: np.lib.npyio.NpzFile, output: Path, vertical: bool) -> None:
    qi = qy_index(data)
    conditions = ("kn0p1_u400", "kn0p08_u350")
    condition_short = (r"$Kn=0.10$, $U_{\rm lid}=400$", r"$Kn=0.08$, $U_{\rm lid}=350$")
    cuts = (0.20, 0.50, 0.80) if vertical else (0.15, 0.50, 0.85)
    coordinate = np.linspace(0.0, 1.0, 100)
    fig, axes = plt.subplots(2, 3, figsize=(19.0, 9.8), sharex=False, sharey=vertical)
    handles = []
    for row, condition in enumerate(conditions):
        indices = condition_indices(data, condition)
        means = {name: mean_field(data, name, indices, qi) for name, *_ in METHODS}
        for col, cut in enumerate(cuts):
            ax = axes[row, col]
            grid_index = int(round(cut * 99))
            for name, label, color, linestyle, linewidth in METHODS:
                profile = means[name][:, grid_index] if vertical else means[name][grid_index, :]
                xval, yval = (profile, coordinate) if vertical else (coordinate, profile)
                line, = ax.plot(xval, yval, color=color, linestyle=linestyle, linewidth=linewidth, label=label)
                if row == 0 and col == 0: handles.append(line)
            cut_label = rf"$x/L={cut:.2f}$" if vertical else rf"$y/L={cut:.2f}$"
            ax.set_title(f"({'abcdef'[row*3+col]}) " + condition_short[row] + "\n" + cut_label, loc="left", pad=9, fontsize=22, fontweight="bold", color=NAVY)
            ax.grid(alpha=0.24); ax.tick_params(direction="out", length=5.0, width=1.0)
            if vertical:
                ax.set_xlabel(r"Transverse heat flux, $q_y$ ($10^6$ W m$^{-2}$)")
                pass
            else:
                ax.set_xlabel(r"Dimensionless position, $x/L$")
                pass
    if vertical:
        fig.supylabel(r"Dimensionless height, $y/L$", x=0.012, fontsize=24)
    else:
        fig.supylabel(r"Transverse heat flux, $q_y$ ($10^6$ W m$^{-2}$)", x=0.012, fontsize=24)
    fig.legend(handles=handles, labels=[item[1] for item in METHODS], loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=5, frameon=False, handlelength=3.2, columnspacing=1.7)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.08, top=0.82, wspace=0.15, hspace=0.68)
    save(fig, output, "cavity_qy_vertical_profiles" if vertical else "cavity_qy_horizontal_profiles")


def inferred_wiener_gain(data: np.lib.npyio.NpzFile, qi: int) -> np.ndarray:
    ratios = []
    for index in range(len(data["conditions"])):
        residual = np.asarray(data["method_raw_b3"])[index, qi] - np.asarray(data["method_pnet_alone"])[index, qi]
        correction = np.asarray(data["method_pnet_cross_block"])[index, qi] - np.asarray(data["method_pnet_alone"])[index, qi]
        source = dctn(residual, type=2, norm="ortho")
        target = dctn(correction, type=2, norm="ortho")
        threshold = np.percentile(np.abs(source), 30)
        ratio = np.full_like(source, np.nan)
        valid = np.abs(source) > threshold
        ratio[valid] = np.abs(target[valid] / source[valid])
        ratios.append(ratio)
    return np.clip(np.nanmedian(np.asarray(ratios), axis=0), 0.0, 1.0)


def spectral_figure(data: np.lib.npyio.NpzFile, output: Path) -> None:
    qi = qy_index(data)
    gain = inferred_wiener_gain(data, qi)
    conditions = ("kn0p1_u400", "kn0p08_u350")
    biases, improvements = [], []
    for condition in conditions:
        idx = condition_indices(data, condition)
        target = np.asarray(data["target"])[idx, qi]
        proposed = np.asarray(data["method_pnet_cross_block"])[idx, qi]
        comparator = np.asarray(data["method_raw_b10"])[idx, qi]
        biases.append(np.mean(proposed - target, axis=0))
        improvements.append(np.sqrt(np.mean((comparator - target) ** 2, axis=0)) - np.sqrt(np.mean((proposed - target) ** 2, axis=0)))
    bias_limit = robust_limit(*biases, percentile=99.5)
    improvement_limit = robust_limit(*improvements, percentile=99.5)
    arrays = (gain, biases[0], improvements[0], biases[1], improvements[1])
    titles = ("Wiener gain", "Bias — A", "RMSE — A", "Bias — B", "RMSE — B")
    fig, axes = plt.subplots(1, 5, figsize=(19.4, 5.3))
    artists = []
    for index, (ax, values, title) in enumerate(zip(axes, arrays, titles, strict=True)):
        if index == 0:
            artist = ax.imshow(values, origin="lower", extent=(0, 1, 0, 1), cmap="viridis", vmin=0, vmax=1, interpolation="nearest", aspect="equal")
        elif index in (1, 3):
            artist = ax.imshow(values, origin="lower", extent=(0, 1, 0, 1), cmap="PuOr_r", norm=TwoSlopeNorm(vmin=-bias_limit, vcenter=0, vmax=bias_limit), interpolation="nearest", aspect="equal")
        else:
            artist = ax.imshow(values, origin="lower", extent=(0, 1, 0, 1), cmap="PuOr_r", norm=TwoSlopeNorm(vmin=-improvement_limit, vcenter=0, vmax=improvement_limit), interpolation="nearest", aspect="equal")
        artists.append(artist)
        panel_axes(ax, index, title, True, index == 0)
        ax.title.set_fontsize(21)
    fig.subplots_adjust(left=0.052, right=0.995, top=0.80, bottom=0.26, wspace=0.16)
    fig.text(0.5, 0.925, r"Condition A: $Kn=0.10$, $U_{\rm lid}=400$ m s$^{-1}$     •     Condition B: $Kn=0.08$, $U_{\rm lid}=350$ m s$^{-1}$", ha="center", va="center", fontsize=23, color=NAVY, fontweight="bold")
    specs = ((artists[0], [0.07, 0.08, 0.25, 0.035], "Modal observation gain (0–1)", None), (artists[1], [0.39, 0.08, 0.25, 0.035], r"Signed bias ($10^6$ W m$^{-2}$)", bias_limit), (artists[2], [0.70, 0.08, 0.25, 0.035], r"RMSE reduction ($10^6$ W m$^{-2}$)", improvement_limit))
    for artist, bounds, label, limit in specs:
        cb = fig.colorbar(artist, cax=fig.add_axes(bounds), orientation="horizontal")
        ticks = np.linspace(0.0, 1.0, 3) if limit is None else np.linspace(-0.75 * limit, 0.75 * limit, 3)
        cb.set_ticks(ticks)
        cb.set_ticklabels([f"{value:.2g}" for value in ticks])
        cb.set_label(label, fontsize=23); cb.ax.tick_params(labelsize=21, length=5.2)
    save(fig, output, "cavity_spectral_and_cancellation")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fields", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    set_style()
    with np.load(args.fields, allow_pickle=False) as data:
        ensemble_figure(data, args.output)
        condition_field_figure(data, args.output, "kn0p08_u350", "cavity_kn0p08_u350_qy_fields_and_errors")
        condition_field_figure(data, args.output, "kn0p1_u400", "cavity_kn0p1_u400_qy_fields_and_errors")
        profiles_figure(data, args.output, vertical=True)
        profiles_figure(data, args.output, vertical=False)
        spectral_figure(data, args.output)


if __name__ == "__main__":
    main()
