#!/usr/bin/env python3
"""Regenerate the Mach-10 cylinder figures from the locked MV17B arrays.

The layout deliberately reserves separate bands for titles, panel letters,
legends, and colour bars.  No label is placed on a data curve or on the solid
cylinder.  The numerical operations reproduce the manuscript definitions:
area-weighted angular profiles, signed errors normalised by the independent
reference RMS, native-grid vector magnitudes, and radial heat flux
q_r = q_x cos(theta) + q_y sin(theta).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
from matplotlib.patches import Arc, Circle
from matplotlib.ticker import MaxNLocator
import numpy as np


NAVY = "#17365D"
BLUE = "#2468B4"
TEAL = "#168AAD"
RED = "#D1495B"
GRAY = "#8A939E"
PURPLE = "#6F3C8E"

METHODS = (
    ("target", "Independent reference", "#111111", "-", 3.1),
    ("raw_b3", r"Raw DSMC, $B=3$", GRAY, ":", 2.7),
    ("prior", "Frozen prior", TEAL, "--", 2.8),
    ("selected", r"Proposed estimator, $B=3$", RED, "-", 3.3),
    ("raw_b10", r"Raw DSMC, $B=10$", BLUE, "-.", 2.8),
)


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "stixsans",
            "font.size": 21,
            "axes.labelsize": 24,
            "axes.titlesize": 23,
            "xtick.labelsize": 19,
            "ytick.labelsize": 19,
            "legend.fontsize": 19,
            "axes.linewidth": 1.8,
            "lines.linewidth": 3.5,
            "lines.markersize": 9.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def save(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.pdf", dpi=300, bbox_inches="tight", pad_inches=0.035, facecolor="white")
    fig.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.035, facecolor="white")
    plt.close(fig)


def geometry(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray, float, mtri.Triangulation]:
    # The locked DS2V deck uses a diameter D=0.2 m and centre (0,0).
    diameter = 0.2
    x = np.asarray(data["x_m"], dtype=float) / diameter
    y = np.asarray(data["y_m"], dtype=float) / diameter
    tri = mtri.Triangulation(x, y)
    tx = x[tri.triangles]
    ty = y[tri.triangles]
    centroid_radius = np.sqrt(np.mean(tx, axis=1) ** 2 + np.mean(ty, axis=1) ** 2)
    edge = np.maximum.reduce(
        (
            np.hypot(tx[:, 0] - tx[:, 1], ty[:, 0] - ty[:, 1]),
            np.hypot(tx[:, 1] - tx[:, 2], ty[:, 1] - ty[:, 2]),
            np.hypot(tx[:, 2] - tx[:, 0], ty[:, 2] - ty[:, 0]),
        )
    )
    tri.set_mask((centroid_radius < 0.495) | (edge > 0.22))
    return x, y, diameter, tri


def angular_profiles(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    edges = np.linspace(0.0, np.pi, 61)
    angle = np.rad2deg(0.5 * (edges[:-1] + edges[1:]))
    area = np.asarray(data["area_m2"], dtype=float)
    profiles: dict[str, np.ndarray] = {}
    for method, *_ in METHODS:
        values = np.full((6, len(angle)), np.nan)
        for pair in range(6):
            qn = (
                np.asarray(data[f"{method}_qx"])[pair] * np.asarray(data["cos_theta"])[pair]
                + np.asarray(data[f"{method}_qy"])[pair] * np.asarray(data["sin_theta"])[pair]
            )
            theta = np.asarray(data["theta"])[pair]
            near = np.asarray(data["near_wall_mask"])[pair]
            for j, (left, right) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
                mask = near & (theta >= left) & (theta < right)
                if np.any(mask):
                    values[pair, j] = np.sum(area[mask] * qn[mask]) / np.sum(area[mask])
        profiles[method] = values
    return angle, profiles


def common_panel_format(ax: plt.Axes, pair: int, letter: str) -> None:
    ax.set_title(f"({letter}) Fresh pair {pair + 1:02d}", loc="left", pad=10, fontweight="bold", color=NAVY)
    ax.set_xlim(0, 180)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.grid(alpha=0.24, linewidth=0.8)
    ax.tick_params(direction="out", length=5.5, width=1.1)


def near_wall_profile_figures(data: np.lib.npyio.NpzFile, output: Path) -> None:
    angle, profiles = angular_profiles(data)
    letters = "abcdef"

    fig, axes = plt.subplots(2, 3, figsize=(19.2, 10.6), sharex=True, sharey=True)
    handles = []
    for pair, ax in enumerate(axes.flat):
        for method, label, color, linestyle, linewidth in METHODS:
            line, = ax.plot(angle, profiles[method][pair], color=color, linestyle=linestyle, linewidth=linewidth, label=label)
            if pair == 0:
                handles.append(line)
        common_panel_format(ax, pair, letters[pair])
    fig.supylabel(r"Near-wall normal heat flux, $q_n^*$", x=0.014, fontsize=24)
    for ax in axes[1, :]:
        ax.set_xlabel(r"Surface angle, $\theta$ (deg)")
    fig.legend(handles=handles, labels=[item[1] for item in METHODS], loc="upper center", bbox_to_anchor=(0.5, 0.992), ncol=5, frameon=False, handlelength=3.2, columnspacing=1.45)
    fig.subplots_adjust(left=0.074, right=0.995, bottom=0.083, top=0.885, wspace=0.12, hspace=0.22)
    save(fig, output, "cylinder_near_wall_normal_heat_flux_profiles")

    fig, axes = plt.subplots(2, 3, figsize=(19.2, 10.6), sharex=True, sharey=True)
    error_methods = METHODS[1:]
    handles = []
    for pair, ax in enumerate(axes.flat):
        reference = profiles["target"][pair]
        reference_rms = np.sqrt(np.nanmean(reference**2))
        for method, label, color, linestyle, linewidth in error_methods:
            error = 100.0 * (profiles[method][pair] - reference) / reference_rms
            line, = ax.plot(angle, error, color=color, linestyle=linestyle, linewidth=linewidth, label=label)
            if pair == 0:
                handles.append(line)
        ax.axhline(0.0, color="#333333", linewidth=1.25, zorder=0)
        common_panel_format(ax, pair, letters[pair])
    fig.supylabel(r"Signed error, $100\Delta q_n/\mathrm{RMS}(q_{n,\mathrm{ref}})$ (\%)", x=0.014, fontsize=24)
    for ax in axes[1, :]:
        ax.set_xlabel(r"Surface angle, $\theta$ (deg)")
    fig.legend(handles=handles, labels=[item[1] for item in error_methods], loc="upper center", bbox_to_anchor=(0.5, 0.992), ncol=4, frameon=False, handlelength=3.3, columnspacing=2.1)
    fig.subplots_adjust(left=0.082, right=0.995, bottom=0.083, top=0.885, wspace=0.13, hspace=0.22)
    save(fig, output, "cylinder_near_wall_normal_heat_flux_errors")


def add_cylinder(ax: plt.Axes) -> None:
    ax.add_patch(Circle((0.0, 0.0), 0.5, facecolor="white", edgecolor="#222222", linewidth=1.8, zorder=20))
    ax.set_xlim(-1.08, 1.68)
    ax.set_ylim(0.0, 1.30)
    ax.set_aspect("equal")
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.tick_params(direction="out", length=5.0, width=1.0)


def robust_limit(*arrays: np.ndarray, percentile: float = 99.6) -> float:
    merged = np.concatenate([np.abs(np.asarray(a)[np.isfinite(a)]).ravel() for a in arrays])
    return float(np.percentile(merged, percentile))


def native_representation_figure(data: np.lib.npyio.NpzFile, output: Path) -> None:
    x, y, _, tri = geometry(data)
    area = np.asarray(data["area_m2"], dtype=float)
    theta_deg = np.rad2deg(np.asarray(data["theta"])[0])
    near = np.asarray(data["near_wall_mask"])[0].astype(float)

    fig, axes = plt.subplots(1, 3, figsize=(19.0, 6.1))
    area_artist = axes[0].tripcolor(tri, area, shading="flat", cmap="cividis")
    theta_artist = axes[1].tripcolor(tri, theta_deg, shading="flat", cmap="magma", vmin=0.0, vmax=180.0)
    mask_map = ListedColormap(["#E7EBF0", RED])
    mask_norm = BoundaryNorm([-0.5, 0.5, 1.5], mask_map.N)
    mask_artist = axes[2].tripcolor(tri, near, shading="flat", cmap=mask_map, norm=mask_norm)
    titles = (
        "Native cell area",
        r"Surface-relative angle, $\theta$",
        r"Locked near-wall band, $d_w/D\leq0.05$",
    )
    for index, (ax, title) in enumerate(zip(axes, titles, strict=True)):
        add_cylinder(ax)
        ax.set_title(f"({'abc'[index]}) {title}", loc="left", pad=11, fontweight="bold", color=NAVY, fontsize=21)
        ax.set_xlabel(r"$(x-x_c)/D$")
        if index == 0:
            ax.set_ylabel(r"$(y-y_c)/D$")
        else:
            ax.set_yticklabels([])
    fig.subplots_adjust(left=0.055, right=0.99, top=0.90, bottom=0.24, wspace=0.14)
    cax1 = fig.add_axes([0.075, 0.095, 0.25, 0.035])
    cax2 = fig.add_axes([0.385, 0.095, 0.25, 0.035])
    cax3 = fig.add_axes([0.695, 0.095, 0.25, 0.035])
    cb1 = fig.colorbar(area_artist, cax=cax1, orientation="horizontal")
    cb1.set_label(r"Native cell area (m$^2$)", fontsize=23)
    cb2 = fig.colorbar(theta_artist, cax=cax2, orientation="horizontal", ticks=[0, 45, 90, 135, 180])
    cb2.set_label(r"Surface angle, $\theta$ (deg)", fontsize=23)
    cb3 = fig.colorbar(mask_artist, cax=cax3, orientation="horizontal", ticks=[0, 1])
    cb3.ax.set_xticklabels(["Outside band", "Inside band"])
    cb3.set_label(r"Near-wall selection; $d_w$ is wall-normal distance", fontsize=22)
    for cb in (cb1, cb2, cb3):
        cb.ax.tick_params(labelsize=13.5, length=4.5)
    save(fig, output, "cylinder_native_representation")


def qy_fields_and_errors_figure(data: np.lib.npyio.NpzFile, output: Path) -> None:
    _, _, _, tri = geometry(data)
    pair = 0
    order = ("target", "raw_b3", "prior", "selected", "phase", "raw_b10")
    fields = [np.asarray(data[f"{name}_qy"])[pair] for name in order]
    reference = fields[0]
    reference_rms = float(np.sqrt(np.mean(reference**2)))
    errors = [100.0 * (field - reference) / reference_rms for field in fields]
    field_limit = robust_limit(*fields, percentile=99.5)
    error_limit = robust_limit(*errors[1:], percentile=99.2)
    field_levels = np.linspace(-field_limit, field_limit, 47)
    error_levels = np.linspace(-error_limit, error_limit, 47)
    field_norm = TwoSlopeNorm(vmin=-field_limit, vcenter=0.0, vmax=field_limit)
    error_norm = TwoSlopeNorm(vmin=-error_limit, vcenter=0.0, vmax=error_limit)
    titles = ("Reference", r"Raw $B=3$", "Frozen prior", r"Proposed $B=3$", "Phase control", r"Raw $B=10$")

    fig = plt.figure(figsize=(19.6, 6.9))
    grid = fig.add_gridspec(2, 7, width_ratios=[1, 1, 1, 1, 1, 1, 0.075], left=0.045, right=0.968, bottom=0.115, top=0.91, wspace=0.12, hspace=0.24)
    axes = np.asarray([[fig.add_subplot(grid[row, col]) for col in range(6)] for row in range(2)])
    cax_top = fig.add_subplot(grid[0, 6]); cax_bottom = fig.add_subplot(grid[1, 6])
    top_artist = bottom_artist = None
    for col, (field, error, title) in enumerate(zip(fields, errors, titles, strict=True)):
        top_artist = axes[0, col].tricontourf(tri, field, levels=field_levels, cmap="RdBu_r", norm=field_norm, extend="both")
        bottom_artist = axes[1, col].tricontourf(tri, error, levels=error_levels, cmap="PuOr_r", norm=error_norm, extend="both")
        for row in range(2):
            add_cylinder(axes[row, col])
            axes[row, col].set_xlim(-1.05, 1.62); axes[row, col].set_ylim(0.0, 1.18)
            if col > 0:
                axes[row, col].set_yticklabels([])
        axes[0, col].set_title(f"({'abcdef'[col]}) {title}", loc="left", pad=8, fontsize=18, fontweight="bold", color=NAVY)
        axes[0, col].set_xticklabels([])
        lower_title = ("Zero error", r"Raw $B=3$", "Prior", r"Proposed $B=3$", "Phase control", r"Raw $B=10$")[col]
        axes[1, col].set_title(f"({'ghijkl'[col]}) {lower_title}", loc="left", pad=7, fontsize=18, fontweight="bold", color=NAVY)
        axes[1, col].set_xlabel(r"$(x-x_c)/D$", fontsize=22)
    axes[0, 0].set_ylabel(r"$(y-y_c)/D$", fontsize=23)
    axes[1, 0].set_ylabel(r"$(y-y_c)/D$", fontsize=23)
    if top_artist is None or bottom_artist is None:
        raise RuntimeError("cylinder qy figure incomplete")
    cb1 = fig.colorbar(top_artist, cax=cax_top); cb2 = fig.colorbar(bottom_artist, cax=cax_bottom)
    cb1.set_label(r"$q_y^*$", fontsize=24, labelpad=9)
    cb2.set_label("Signed error (%)", fontsize=23, labelpad=9)
    cb1.set_ticks(np.linspace(-0.75 * field_limit, 0.75 * field_limit, 5))
    cb2.set_ticks(np.linspace(-0.75 * error_limit, 0.75 * error_limit, 5))
    for cb in (cb1, cb2): cb.ax.tick_params(labelsize=21, length=5.2)
    save(fig, output, "cylinder_fresh_pair01_qy_fields_and_errors")


def cylinder_ensemble_figure(data: np.lib.npyio.NpzFile, output: Path) -> None:
    _, _, _, tri = geometry(data)
    reference = np.asarray(data["target_qy"])
    proposed = np.asarray(data["selected_qy"])
    comparator = np.asarray(data["raw_b10_qy"])
    reference_mean = np.mean(reference, axis=0)
    proposed_mean = np.mean(proposed, axis=0)
    bias = proposed_mean - reference_mean
    proposed_rmse = np.sqrt(np.mean((proposed - reference) ** 2, axis=0))
    comparator_rmse = np.sqrt(np.mean((comparator - reference) ** 2, axis=0))
    improvement = comparator_rmse - proposed_rmse
    field_limit = robust_limit(reference_mean, proposed_mean, percentile=99.6)
    bias_limit = robust_limit(bias, percentile=99.5)
    improvement_limit = robust_limit(improvement, percentile=99.5)
    arrays = (reference_mean, proposed_mean, bias, improvement)
    titles = ("Reference mean", r"Proposed mean, $B=3$", "Signed bias", "RMSE reduction")

    fig, axes = plt.subplots(1, 4, figsize=(19.2, 5.7))
    artists = []
    for index, (ax, values, title) in enumerate(zip(axes, arrays, titles, strict=True)):
        if index < 2:
            limit, cmap = field_limit, "RdBu_r"
        elif index == 2:
            limit, cmap = bias_limit, "PuOr_r"
        else:
            limit, cmap = improvement_limit, "PuOr_r"
        levels = np.linspace(-limit, limit, 47)
        artist = ax.tricontourf(tri, values, levels=levels, cmap=cmap, norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit), extend="both")
        artists.append(artist)
        add_cylinder(ax)
        ax.set_xlim(-1.05, 1.62); ax.set_ylim(0.0, 1.18)
        ax.set_title(f"({'abcd'[index]}) {title}", loc="left", pad=10, fontsize=20, fontweight="bold", color=NAVY)
        ax.set_xlabel(r"$(x-x_c)/D$")
        if index == 0:
            ax.set_ylabel(r"$(y-y_c)/D$")
        else:
            ax.set_yticklabels([])
    fig.subplots_adjust(left=0.052, right=0.992, top=0.89, bottom=0.25, wspace=0.13)
    cbar_specs = ((artists[0], [0.07, 0.08, 0.25, 0.035], r"Mean $q_y^*$", field_limit), (artists[2], [0.385, 0.08, 0.25, 0.035], r"Bias $\Delta q_y^*$", bias_limit), (artists[3], [0.70, 0.08, 0.25, 0.035], r"RMSE reduction $\Delta\mathrm{RMSE}^*$", improvement_limit))
    for artist, bounds, label, limit in cbar_specs:
        cb = fig.colorbar(artist, cax=fig.add_axes(bounds), orientation="horizontal")
        ticks = np.linspace(-0.75 * limit, 0.75 * limit, 3)
        cb.set_ticks(ticks); cb.set_ticklabels([f"{value:.2g}" for value in ticks])
        cb.set_label(label, fontsize=23); cb.ax.tick_params(labelsize=21, length=5.2)
    save(fig, output, "cylinder_ensemble_physical_diagnostics")


def prior_correction_figure(data: np.lib.npyio.NpzFile, output: Path) -> None:
    _, _, _, tri = geometry(data)
    pair = 0
    prior_qy = np.asarray(data["prior_qy"])[pair]
    selected_dqy = np.asarray(data["selected_qy"])[pair] - prior_qy
    phase_dqy = np.asarray(data["phase_qy"])[pair] - prior_qy
    cosine = np.asarray(data["cos_theta"])[pair]
    sine = np.asarray(data["sin_theta"])[pair]
    near = np.asarray(data["near_wall_mask"])[pair]
    prior_qn = np.asarray(data["prior_qx"])[pair] * cosine + prior_qy * sine
    selected_qn = np.asarray(data["selected_qx"])[pair] * cosine + np.asarray(data["selected_qy"])[pair] * sine
    phase_qn = np.asarray(data["phase_qx"])[pair] * cosine + np.asarray(data["phase_qy"])[pair] * sine
    selected_dqn = selected_qn - prior_qn
    phase_dqn = phase_qn - prior_qn
    near_tri = mtri.Triangulation(tri.x, tri.y, triangles=tri.triangles.copy())
    base_mask = np.zeros(len(tri.triangles), dtype=bool) if tri.mask is None else np.asarray(tri.mask, dtype=bool)
    near_tri.set_mask(base_mask | ~np.all(near[tri.triangles], axis=1))

    field_limit = robust_limit(prior_qy, prior_qn[near], percentile=99.3)
    correction_limit = robust_limit(selected_dqy, phase_dqy, selected_dqn[near], phase_dqn[near], percentile=99.3)
    field_levels = np.linspace(-field_limit, field_limit, 45)
    correction_levels = np.linspace(-correction_limit, correction_limit, 45)
    field_norm = TwoSlopeNorm(vmin=-field_limit, vcenter=0.0, vmax=field_limit)
    correction_norm = TwoSlopeNorm(vmin=-correction_limit, vcenter=0.0, vmax=correction_limit)

    fig = plt.figure(figsize=(19.2, 7.6))
    grid = fig.add_gridspec(2, 3, wspace=0.15, hspace=0.38, left=0.055, right=0.992, bottom=0.20, top=0.94)
    axes = np.asarray([[fig.add_subplot(grid[row, col]) for col in range(3)] for row in range(2)])
    fields = (prior_qy, selected_dqy, phase_dqy, prior_qn, selected_dqn, phase_dqn)
    titles = (
        r"Prior: $q_y^*$", r"Proposed: $\Delta q_y^*$", r"Phase control: $\Delta q_y^*$",
        r"Prior: near-wall $q_n^*$", r"Proposed: near-wall $\Delta q_n^*$", r"Phase control: near-wall $\Delta q_n^*$",
    )
    artists = []
    for index, (ax, values, title) in enumerate(zip(axes.flat, fields, titles, strict=True)):
        use_field = index in (0, 3)
        tri_used = tri if index < 3 else near_tri
        artist = ax.tricontourf(tri_used, values, levels=field_levels if use_field else correction_levels, cmap="RdBu_r" if use_field else "PuOr_r", norm=field_norm if use_field else correction_norm, extend="both")
        artists.append(artist)
        add_cylinder(ax)
        ax.set_title(f"({'abcdef'[index]}) {title}", loc="left", pad=9, fontweight="bold", color=NAVY, fontsize=19)
        if index // 3 == 1:
            ax.set_xlabel(r"$(x-x_c)/D$")
        else:
            ax.set_xticklabels([])
        if index % 3 == 0:
            ax.set_ylabel(r"$(y-y_c)/D$")
        else:
            ax.set_yticklabels([])
    cax_field = fig.add_axes([0.12, 0.065, 0.34, 0.034])
    cax_corr = fig.add_axes([0.57, 0.065, 0.34, 0.034])
    cb1 = fig.colorbar(artists[0], cax=cax_field, orientation="horizontal")
    cb2 = fig.colorbar(artists[1], cax=cax_corr, orientation="horizontal")
    for cb, label, limit in ((cb1, r"Prior heat-flux component, $q^*$", field_limit), (cb2, r"Observation correction, $\Delta q^*$", correction_limit)):
        cb.set_ticks(np.linspace(-0.75 * limit, 0.75 * limit, 5))
        cb.ax.tick_params(labelsize=21, width=1.2, length=5.2)
        cb.set_label(label, fontsize=23, labelpad=9)
    save(fig, output, "cylinder_prior_correction_fields")


def interpolate_profile(tri: mtri.Triangulation, values: np.ndarray, xcut: float, ygrid: np.ndarray) -> np.ndarray:
    interpolator = mtri.LinearTriInterpolator(tri, np.asarray(values, dtype=float))
    result = interpolator(np.full_like(ygrid, xcut), ygrid)
    return np.asarray(np.ma.filled(result, np.nan), dtype=float)


def wake_profiles_figure(data: np.lib.npyio.NpzFile, output: Path) -> None:
    _, _, _, tri = geometry(data)
    ygrid = np.linspace(0.02, 1.30, 190)
    cuts = (0.75, 1.00, 1.50)
    fig, axes = plt.subplots(1, 3, figsize=(19.2, 6.8), sharey=True)
    handles = []
    for col, (ax, cut) in enumerate(zip(axes, cuts, strict=True)):
        reference_ensemble = np.asarray([interpolate_profile(tri, np.asarray(data["target_qy"])[pair], cut, ygrid) for pair in range(6)])
        low = np.nanquantile(reference_ensemble, 0.10, axis=0)
        high = np.nanquantile(reference_ensemble, 0.90, axis=0)
        band = ax.fill_betweenx(ygrid, low, high, color="#D8DDE3", alpha=0.85, linewidth=0, label="Independent-reference 10–90% range")
        if col == 0:
            handles.append(band)
        for method, label, color, linestyle, linewidth in METHODS:
            profile = interpolate_profile(tri, np.asarray(data[f"{method}_qy"])[0], cut, ygrid)
            line, = ax.plot(profile, ygrid, color=color, linestyle=linestyle, linewidth=linewidth, label=label)
            if col == 0:
                handles.append(line)
        ax.set_title(f"({'abc'[col]}) " + rf"Wake cut $(x-x_c)/D={cut:.2f}$", loc="left", pad=10, fontweight="bold", color=NAVY)
        ax.set_xlabel(r"Transverse heat flux, $q_y^*$")
        ax.set_ylim(0.02, 1.30)
        ax.grid(alpha=0.24)
        ax.tick_params(direction="out", length=5.0, width=1.0)
    axes[0].set_ylabel(r"Dimensionless height, $(y-y_c)/D$")
    labels = ["Independent-reference 10–90% range"] + [item[1] for item in METHODS]
    fig.legend(handles=handles, labels=labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3, frameon=False, handlelength=3.2, columnspacing=1.65)
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.11, top=0.80, wspace=0.14)
    save(fig, output, "cylinder_wake_qy_profiles")


def magnitude_figure(data: np.lib.npyio.NpzFile, output: Path) -> None:
    _, _, _, tri = geometry(data)
    pair = 0
    order = ("target", "raw_b3", "selected", "raw_b10")
    magnitudes = {name: np.hypot(np.asarray(data[f"{name}_qx"])[pair], np.asarray(data[f"{name}_qy"])[pair]) for name in order}
    reference = magnitudes["target"]
    reference_rms = float(np.sqrt(np.mean(reference**2)))
    errors = {name: 100.0 * (magnitudes[name] - reference) / reference_rms for name in order[1:]}
    field_max = float(np.percentile(reference, 99.7))
    error_max = robust_limit(*errors.values(), percentile=99.4)
    field_levels = np.linspace(0.0, field_max, 48)
    error_levels = np.linspace(-error_max, error_max, 49)
    error_norm = TwoSlopeNorm(vmin=-error_max, vcenter=0.0, vmax=error_max)

    fig = plt.figure(figsize=(19.4, 6.9))
    grid = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 0.065], hspace=0.22, wspace=0.13, left=0.055, right=0.965, bottom=0.115, top=0.92)
    axes = np.asarray([[fig.add_subplot(grid[row, col]) for col in range(4)] for row in range(2)])
    cax_top = fig.add_subplot(grid[0, 4]); cax_bottom = fig.add_subplot(grid[1, 4])
    titles = ("Reference", r"Raw $B=3$", r"Proposed $B=3$", r"Raw $B=10$")
    top_artists = []
    for col, (name, title) in enumerate(zip(order, titles, strict=True)):
        artist = axes[0, col].tricontourf(tri, magnitudes[name], levels=field_levels, cmap="magma", vmin=0.0, vmax=field_max, extend="max")
        artist.set_rasterized(True)
        top_artists.append(artist)
        add_cylinder(axes[0, col])
        axes[0, col].set_title(f"({'abcd'[col]}) {title}", loc="left", pad=9, fontweight="bold", color=NAVY, fontsize=20)
        axes[0, col].set_xticklabels([])
        if col == 0:
            axes[0, col].set_ylabel(r"$(y-y_c)/D$")
        else:
            axes[0, col].set_yticklabels([])
    bottom_titles = ("Reference baseline\n(zero error)", r"Raw $B=3$ error", r"Proposed $B=3$ error", r"Raw $B=10$ error")
    zero = np.zeros_like(reference)
    bottom_values = (zero, errors["raw_b3"], errors["selected"], errors["raw_b10"])
    bottom_artists = []
    for col, (values, title) in enumerate(zip(bottom_values, bottom_titles, strict=True)):
        artist = axes[1, col].tricontourf(tri, values, levels=error_levels, cmap="PuOr_r", norm=error_norm, extend="both")
        artist.set_rasterized(True)
        bottom_artists.append(artist)
        add_cylinder(axes[1, col])
        axes[1, col].set_title(f"({'efgh'[col]}) {title}", loc="left", pad=9, fontweight="bold", color=NAVY, fontsize=19)
        axes[1, col].set_xlabel(r"$(x-x_c)/D$")
        if col == 0:
            axes[1, col].set_ylabel(r"$(y-y_c)/D$")
        else:
            axes[1, col].set_yticklabels([])
    cb1 = fig.colorbar(top_artists[0], cax=cax_top)
    cb1.set_label(r"$|\boldsymbol{q}|^*$", fontsize=24, labelpad=9)
    cb2 = fig.colorbar(bottom_artists[1], cax=cax_bottom)
    cb2.set_label("Signed error (%)", fontsize=23, labelpad=9)
    cb1.set_ticks(np.linspace(0.0, field_max, 5))
    cb2.set_ticks(np.linspace(-0.75 * error_max, 0.75 * error_max, 5))
    for cb in (cb1, cb2): cb.ax.tick_params(labelsize=21, length=5.2, width=1.2)
    save(fig, output, "cylinder_fresh_pair01_heat_flux_magnitude")


def polar_figure(data: np.lib.npyio.NpzFile, output: Path) -> None:
    pair = 0
    theta = np.asarray(data["theta"])[pair]
    x = np.asarray(data["x_m"], dtype=float) / 0.2
    y = np.asarray(data["y_m"], dtype=float) / 0.2
    radius = np.hypot(x, y)
    mask = (radius >= 0.5) & (radius <= 0.9) & (theta >= 0.0) & (theta <= np.pi)
    xx = radius[mask] * np.cos(theta[mask])
    yy = radius[mask] * np.sin(theta[mask])
    polar_tri = mtri.Triangulation(xx, yy)
    tt = theta[mask][polar_tri.triangles]
    rr = radius[mask][polar_tri.triangles]
    polar_tri.set_mask((np.ptp(tt, axis=1) > 0.16) | (np.ptp(rr, axis=1) > 0.10))
    names = ("target", "raw_b3", "selected")
    titles = ("Independent reference", r"Raw DSMC, $B=3$", r"Proposed estimator, $B=3$")
    values = []
    cosine = np.asarray(data["cos_theta"])[pair]
    sine = np.asarray(data["sin_theta"])[pair]
    for name in names:
        qn = np.asarray(data[f"{name}_qx"])[pair] * cosine + np.asarray(data[f"{name}_qy"])[pair] * sine
        values.append(qn[mask])
    limit = robust_limit(*values, percentile=99.4)
    levels = np.linspace(-limit, limit, 51)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    fig, axes = plt.subplots(1, 3, figsize=(19.2, 5.25))
    artist = None
    for index, (ax, field, title) in enumerate(zip(axes, values, titles, strict=True)):
        artist = ax.tricontourf(polar_tri, field, levels=levels, cmap="RdBu_r", norm=norm, extend="both")
        for rad in (0.5, 0.6, 0.7, 0.8, 0.9):
            ax.add_patch(Arc((0.0, 0.0), 2 * rad, 2 * rad, theta1=0, theta2=180, color="#A5B1BD", alpha=0.60, linewidth=0.85, zorder=5))
        for angle_deg in (0, 45, 90, 135, 180):
            angle_rad = np.deg2rad(angle_deg)
            ax.plot([0.0, 0.9 * np.cos(angle_rad)], [0.0, 0.9 * np.sin(angle_rad)], color="#A5B1BD", alpha=0.60, linewidth=0.85, zorder=5)
            label_radius = 0.975
            ax.text(label_radius * np.cos(angle_rad), label_radius * np.sin(angle_rad), rf"${angle_deg}^\circ$", ha="center", va="center", fontsize=22, color="#222222")
        ax.plot([-0.9, 0.9], [0.0, 0.0], color="#222222", linewidth=1.3, zorder=6)
        ax.set_xlim(-1.02, 1.02); ax.set_ylim(-0.08, 1.03); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"({'abc'[index]}) {title}", loc="left", pad=14, fontsize=21, fontweight="bold", color=NAVY)
    fig.subplots_adjust(left=0.025, right=0.985, bottom=0.28, top=0.92, wspace=0.12)
    if artist is None:
        raise RuntimeError("polar figure has no contour artist")
    cax = fig.add_axes([0.24, 0.065, 0.52, 0.05])
    cb = fig.colorbar(artist, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=14, length=4.5, width=1.0)
    cb.set_label(r"Radial heat flux, $q_r^*=q_x^*\cos\theta+q_y^*\sin\theta$", fontsize=23, labelpad=9)
    fig.text(0.5, 0.175, r"Radial coordinate $r/D$: cylinder surface = 0.5; outer displayed radius = 0.9", ha="center", va="center", fontsize=22, color=NAVY)
    save(fig, output, "cylinder_polar_qn_structure")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fields", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    set_style()
    with np.load(args.fields, allow_pickle=False) as data:
        native_representation_figure(data, args.output)
        qy_fields_and_errors_figure(data, args.output)
        cylinder_ensemble_figure(data, args.output)
        near_wall_profile_figures(data, args.output)
        prior_correction_figure(data, args.output)
        wake_profiles_figure(data, args.output)
        magnitude_figure(data, args.output)
        polar_figure(data, args.output)


if __name__ == "__main__":
    main()
