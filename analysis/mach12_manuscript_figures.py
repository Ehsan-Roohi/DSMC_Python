#!/usr/bin/env python3
"""Generate manuscript figures from the locked Mach-12 artifacts.

The stagnation-point trend is not a generic smoother.  It is a weighted even
quadratic in delta = pi - theta, which enforces the cylinder symmetry condition
dq/dtheta = 0 at the upstream stagnation point.  Raw angular-bin means remain
visible in the figure and the replacement is restricted to theta >= 149 deg.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from final_check import FIELDS, load_reference, zones_for
from jcp10_postprocess import load_walls, pooled_acf, integrated_time


METHODS = ("zero_shot", "calibration_prior", "calibrated_B3")
METHOD_LABELS = {
    "zero_shot": "Frozen Mach-8/10 zero-shot estimator",
    "calibration_prior": "Mach-12 calibration prior (80 blocks)",
    "calibrated_B3": "Calibrated three-block estimator",
}
METHOD_COLORS = {
    "zero_shot": "#c44e52",
    "calibration_prior": "#4c72b0",
    "calibrated_B3": "#55a868",
}
FIELD_LABELS = {
    "n": r"Number density, $n$",
    "u": r"Streamwise velocity, $u$",
    "v": r"Transverse velocity, $v$",
    "T": r"Translational temperature, $T$",
    "Pxx": r"Streamwise normal stress, $P_{xx}$",
    "Pxy": r"Shear stress, $P_{xy}$",
    "Pyy": r"Transverse normal stress, $P_{yy}$",
    "qx": r"Streamwise heat flux, $q_x$",
    "qy": r"Transverse heat flux, $q_y$",
}


def style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "stixsans",
        "font.size": 11,
        "axes.labelsize": 11.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "axes.linewidth": 1.05,
        "lines.linewidth": 2.1,
        "lines.markersize": 7.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def geometric(values: np.ndarray) -> float:
    return float(np.exp(np.mean(np.log(np.maximum(values, 1.0e-15)))))


def bootstrap_geometric(values: np.ndarray, seed: int) -> tuple[float, float]:
    logs = np.log(np.maximum(np.asarray(values, dtype=float), 1.0e-15))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(logs), size=(20000, len(logs)))
    samples = np.exp(np.mean(logs[indices], axis=1))
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def save(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(output / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(output / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def read_ratios(metrics_csv: Path) -> dict[str, dict[str, np.ndarray]]:
    records: dict[tuple[int, str, str], float] = {}
    with metrics_csv.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            records[(int(row["seed"]), row["method"], row["field"])] = float(row["nrmse_observed"])
    seeds = sorted({key[0] for key in records})
    ratios: dict[str, dict[str, np.ndarray]] = {}
    for method in METHODS:
        ratios[method] = {}
        for field in (*FIELDS, "qn_near_wall"):
            ratios[method][field] = np.asarray([
                records[(seed, method, field)] / records[(seed, "raw_B10", field)]
                for seed in seeds
            ])
        ratios[method]["all_nine"] = np.asarray([
            geometric(np.asarray([
                records[(seed, method, field)] / records[(seed, "raw_B10", field)]
                for field in FIELDS
            ]))
            for seed in seeds
        ])
    return ratios


def complete_hierarchy_figure(ratios: dict[str, dict[str, np.ndarray]], output: Path) -> None:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.8), gridspec_kw={"width_ratios": [1.45, 0.82]}, constrained_layout=True)
    y = np.arange(len(FIELDS))
    offsets = (-0.20, 0.0, 0.20)
    markers = ("X", "o", "s")
    for offset, marker, method in zip(offsets, markers, METHODS, strict=True):
        means = np.asarray([geometric(ratios[method][field]) for field in FIELDS])
        bounds = np.asarray([bootstrap_geometric(ratios[method][field], 26083101 + j) for j, field in enumerate(FIELDS)])
        ax.errorbar(
            means, y + offset,
            xerr=np.vstack((means - bounds[:, 0], bounds[:, 1] - means)),
            fmt=marker, color=METHOD_COLORS[method], ecolor=METHOD_COLORS[method],
            capsize=4.0, capthick=1.4, elinewidth=1.7, markersize=8.0,
            label=METHOD_LABELS[method], zorder=3,
        )
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.6)
    ax.set_yticks(y, [FIELD_LABELS[field] for field in FIELDS])
    ax.invert_yaxis()
    ax.set_xlim(0.42, 1.75)
    ax.set_xlabel(r"Observed NRMSE ratio to Raw-$B=10$")
    ax.grid(axis="x", alpha=0.25)
    ax.text(0.012, 0.985, "(a)", transform=ax.transAxes, va="top", fontweight="bold", fontsize=14, color="#17365D")

    endpoints = ("all_nine", "qy", "qn_near_wall")
    endpoint_labels = ("All nine fields", r"Transverse heat flux, $q_y$", r"Near-wall normal heat flux, $q_n$")
    x = np.arange(len(endpoints))
    for j, endpoint in enumerate(endpoints):
        prior = ratios["calibration_prior"][endpoint]
        calibrated = ratios["calibrated_B3"][endpoint]
        jitter = np.linspace(-0.06, 0.06, len(prior))
        for k in range(len(prior)):
            ax2.plot([j - 0.12 + jitter[k], j + 0.12 + jitter[k]], [prior[k], calibrated[k]], color="0.73", linewidth=1.0, zorder=1)
        ax2.scatter(j - 0.12 + jitter, prior, color=METHOD_COLORS["calibration_prior"], s=48, alpha=0.90, edgecolor="white", linewidth=0.5, zorder=2)
        ax2.scatter(j + 0.12 + jitter, calibrated, color=METHOD_COLORS["calibrated_B3"], marker="s", s=48, alpha=0.90, edgecolor="white", linewidth=0.5, zorder=3)
    ax2.set_xticks(x, endpoint_labels, rotation=12, ha="right")
    ax2.set_ylim(0.425, 0.505)
    ax2.set_ylabel(r"Seed-level NRMSE ratio to Raw-$B=10$")
    ax2.grid(axis="y", alpha=0.25)
    ax2.text(0.018, 0.985, "(b)", transform=ax2.transAxes, va="top", fontweight="bold", fontsize=14, color="#17365D")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.50, 1.035), frameon=False, ncol=3, columnspacing=1.6)
    save(fig, output, "mach12_complete_moment_hierarchy")


def effective_sampling_figure(acf_json: Path, reference_zip: Path, output: Path) -> None:
    payload = json.loads(acf_json.read_text(encoding="utf-8"))
    lags = np.asarray(payload["lags"])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.7), gridspec_kw={"width_ratios": [1.0, 1.34]}, constrained_layout=True)
    curves = (
        ("n", r"Number density, $n$", "#4c72b0"),
        ("T", r"Temperature, $T$", "#8172b3"),
        ("qy", r"Cell-centred heat flux, $q_y$", "#55a868"),
    )
    for key, label, color in curves:
        ax.plot(lags, payload["all_four_seed_acf"][key], marker="o", markersize=6.2, label=label, color=color)
    wall_units, _, _, _ = load_walls(reference_zip)
    wall_acf = pooled_acf([wall_units[seed] for seed in (26082801, 26082802, 26082803, 26082804)])
    ax.plot(lags[: len(wall_acf)], wall_acf, marker="s", markersize=6.6, label=r"Wall-collision heat flux, $q_w$", color="#c44e52")
    ax.axhline(0.0, color="black", linewidth=1.1)
    ax.set_xlim(0, 15)
    ax.set_xlabel("Sampling-block lag")
    ax.set_ylabel("Pooled temporal autocorrelation")
    ax.grid(alpha=0.25)
    ax.text(0.018, 0.97, "(a)", transform=ax.transAxes, va="top", fontweight="bold", fontsize=14, color="#17365D")

    keys = [*FIELDS, "qn", "qw"]
    labels = [FIELD_LABELS[key] for key in FIELDS] + [r"Nearest-cell normal heat flux, $q_n$", r"Wall-collision heat flux, $q_w$"]
    values = [payload["all_four_seed_effective_blocks"][key] for key in FIELDS]
    values += [payload["all_four_seed_qn_effective_blocks"], payload["all_four_seed_wall_effective_blocks"]]
    colors = ["#4c72b0"] * 7 + ["#55a868", "#55a868", "#55a868", "#c44e52"]
    yy = np.arange(len(keys))
    ax2.barh(yy, values, color=colors)
    ax2.set_yticks(yy, labels)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 170)
    ax2.set_xlabel(r"Effective independent blocks, $B_{\mathrm{eff}}$ (out of 160)")
    ax2.grid(axis="x", alpha=0.25)
    for y0, value in zip(yy, values, strict=True):
        ax2.text(value + 2.5, y0, f"{value:.1f}", va="center", fontsize=9.5, fontweight="bold")
    ax2.text(0.018, 0.97, "(b)", transform=ax2.transAxes, va="top", fontweight="bold", fontsize=14, color="#17365D")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.50, 1.035), frameon=False, ncol=4, columnspacing=1.3)
    save(fig, output, "mach12_effective_sampling")


def nearest_indices(coords: np.ndarray, wall_x: np.ndarray, wall_y: np.ndarray) -> np.ndarray:
    return np.asarray([
        int(np.argmin((coords[:, 1] - wx) ** 2 + (coords[:, 2] - wy) ** 2))
        for wx, wy in zip(wall_x, wall_y, strict=True)
    ])


def symmetry_constrained_trend(angle: np.ndarray, values: np.ndarray, standard_error: np.ndarray, fit_start: float = 149.0, blend_end: float = 159.0) -> tuple[np.ndarray, np.ndarray]:
    mask = angle >= fit_start
    delta = np.deg2rad(180.0 - angle[mask])
    matrix = np.column_stack((np.ones(mask.sum()), delta**2))
    weights = 1.0 / np.maximum(standard_error[mask], np.nanmedian(standard_error[mask]) * 0.25) ** 2
    beta, *_ = np.linalg.lstsq(matrix * np.sqrt(weights[:, None]), values[mask] * np.sqrt(weights), rcond=None)
    fitted = matrix @ beta
    trend = values.copy()
    transition = np.clip((angle[mask] - fit_start) / (blend_end - fit_start), 0.0, 1.0)
    transition = transition * transition * (3.0 - 2.0 * transition)
    trend[mask] = (1.0 - transition) * values[mask] + transition * fitted
    return trend, beta


def wall_heat_flux_figure(reference_zip: Path, output: Path, audit_path: Path) -> None:
    coords, moment_units = load_reference(reference_zip)
    wall_units, angle, wall_x, wall_y = load_walls(reference_zip)
    _, ex, ey, _ = zones_for(coords)
    heldout = (26082803, 26082804)
    moment_blocks = np.concatenate([moment_units[seed] for seed in heldout], axis=0)
    wall_blocks = np.concatenate([wall_units[seed] for seed in heldout], axis=0)
    nearest = nearest_indices(coords, wall_x, wall_y)
    qn_blocks = moment_blocks[:, nearest, 7] * ex[nearest] + moment_blocks[:, nearest, 8] * ey[nearest]
    toward_wall_blocks = -qn_blocks

    wall_acf = pooled_acf([wall_units[seed] for seed in heldout])
    wall_beff = 80.0 / integrated_time(wall_acf)
    qn_acf = pooled_acf([toward_wall_blocks[:40], toward_wall_blocks[40:]])
    qn_beff = 80.0 / integrated_time(qn_acf)
    wall_mean = np.mean(wall_blocks, axis=0)
    qn_mean = np.mean(toward_wall_blocks, axis=0)
    wall_se = np.std(wall_blocks, axis=0, ddof=1) / np.sqrt(wall_beff)
    qn_se = np.std(toward_wall_blocks, axis=0, ddof=1) / np.sqrt(qn_beff)
    wall_trend, wall_beta = symmetry_constrained_trend(angle, wall_mean, wall_se)
    qn_trend, qn_beta = symmetry_constrained_trend(angle, qn_mean, qn_se)

    radians = np.deg2rad(angle)
    wall_average_raw = float(np.trapezoid(wall_mean, radians) / np.pi)
    wall_average_trend = float(np.trapezoid(wall_trend, radians) / np.pi)
    qn_average_trend = float(np.trapezoid(qn_trend, radians) / np.pi)
    audit = {
        "angle_definition": "theta is measured counter-clockwise from the downstream +x direction; theta=180 deg is the upstream stagnation point",
        "fit_model": "q(theta)=a+b*[pi-theta]^2",
        "fit_start_deg": 149.0,
        "blend_end_deg": 159.0,
        "constraint": "dq/dtheta=0 at theta=180 deg",
        "raw_bins_retained_in_figure": True,
        "wall_beta": wall_beta.tolist(),
        "nearest_cell_qn_beta": qn_beta.tolist(),
        "wall_effective_blocks_out_of_80": wall_beff,
        "qn_effective_blocks_out_of_80": qn_beff,
        "surface_average_wall_raw_W_m2": wall_average_raw,
        "surface_average_wall_trend_W_m2": wall_average_trend,
        "surface_average_wall_change_percent": 100.0 * (wall_average_trend / wall_average_raw - 1.0),
        "surface_average_toward_wall_nearest_cell_W_m2": qn_average_trend,
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    with (audit_path.parent / "mach12_wall_heat_flux_profiles.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["theta_deg", "qw_raw_mean_W_m2", "qw_symmetry_trend_W_m2", "qw_standard_error_W_m2", "minus_qn_raw_mean_W_m2", "minus_qn_symmetry_trend_W_m2", "minus_qn_standard_error_W_m2"])
        writer.writerows(zip(angle, wall_mean, wall_trend, wall_se, qn_mean, qn_trend, qn_se, strict=True))

    fig = plt.figure(figsize=(14.5, 6.7), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.08, 1.65, 1.12])
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[0, 1])
    ax2 = fig.add_subplot(grid[0, 2])

    phi = np.linspace(0.0, 2.0 * np.pi, 400)
    ax0.fill(np.cos(phi), np.sin(phi), color="#F1F3F5", zorder=0)
    ax0.plot(np.cos(phi), np.sin(phi), color="#17365D", linewidth=2.5)
    # Flow direction and angular convention.
    ax0.annotate("", xy=(-1.15, 0.64), xytext=(-2.15, 0.64), arrowprops={"arrowstyle": "-|>", "mutation_scale": 17, "lw": 2.5, "color": "#3B6FB6"})
    ax0.text(-2.12, 0.82, r"Freestream velocity $U_\infty$", color="#3B6FB6", fontsize=10.5)
    th = np.deg2rad(122.0)
    surface = np.array([np.cos(th), np.sin(th)])
    nearest_cell = 1.34 * surface
    ax0.plot([0, nearest_cell[0]], [0, nearest_cell[1]], color="#8B95A1", linestyle=":", linewidth=1.5)
    ax0.scatter(*nearest_cell, s=44, color="#2E8B57", edgecolor="white", linewidth=0.8, zorder=5)
    ax0.annotate("", xy=1.72 * surface, xytext=surface, arrowprops={"arrowstyle": "-|>", "mutation_scale": 15, "lw": 2.2, "color": "#2E8B57"})
    ax0.text(-0.84, 1.55, r"outward unit normal $\boldsymbol{n}$", color="#2E8B57", fontsize=10, ha="left")
    ax0.annotate("", xy=0.90 * surface, xytext=1.42 * surface, arrowprops={"arrowstyle": "-|>", "mutation_scale": 15, "lw": 2.2, "color": "#D8742F"})
    ax0.text(-0.34, 1.39, r"positive wall-directed flux", color="#D8742F", fontsize=10, ha="left")
    ax0.text(nearest_cell[0] + 0.10, nearest_cell[1] - 0.02, r"nearest fluid-cell centre, $r_1$", color="#2E8B57", fontsize=9.7)
    arc = np.linspace(0, th, 90)
    ax0.plot(0.48 * np.cos(arc), 0.48 * np.sin(arc), color="#7A5195", linewidth=2.0)
    ax0.text(0.22, 0.43, r"$\theta$", color="#7A5195", fontsize=15)
    ax0.text(-1.02, -0.14, "upstream stagnation point\n" + r"$\theta=180^{\circ}$", ha="right", va="top", fontsize=9.8)
    ax0.text(1.02, -0.14, "downstream point\n" + r"$\theta=0^{\circ}$", ha="left", va="top", fontsize=9.8)
    ax0.text(0.0, -1.34, r"$q_n(r_1,\theta)=\boldsymbol{q}(r_1,\theta)\!\cdot\!\boldsymbol{n}$", ha="center", fontsize=10.3, color="#17365D")
    ax0.text(0.0, -1.59, r"$-q_n>0$: cell-centred transport toward wall", ha="center", fontsize=9.7)
    ax0.text(0.0, -1.81, r"$q_w>0$: molecular energy delivered to wall", ha="center", fontsize=9.7)
    ax0.set_aspect("equal")
    ax0.set_xlim(-2.25, 2.18)
    ax0.set_ylim(-1.92, 1.82)
    ax0.axis("off")
    ax0.text(0.015, 0.985, "(a)", transform=ax0.transAxes, va="top", fontweight="bold", fontsize=14, color="#17365D")

    scale = 1000.0
    ax1.fill_between(angle, (wall_mean - 1.96 * wall_se) / scale, (wall_mean + 1.96 * wall_se) / scale, color="#D8742F", alpha=0.16, linewidth=0)
    ax1.plot(angle, wall_mean / scale, "o", color="#D8742F", markersize=3.8, alpha=0.52, label=r"Raw wall-collision tally $q_w$")
    ax1.plot(angle, wall_trend / scale, color="#B84B13", linewidth=2.4, label=r"Symmetry-constrained wall trend $\widetilde q_w$")
    ax1.plot(angle, qn_trend / scale, color="#2E8B57", linestyle="--", linewidth=2.3, label=r"Nearest-cell trend toward wall $-\widetilde q_n(r_1,\theta)$")
    ax1.axvline(180.0, color="0.25", linestyle=":", linewidth=1.4)
    ax1.set_xlim(0, 180)
    ax1.set_xlabel(r"Surface angle, $\theta$ (deg)")
    ax1.set_ylabel(r"Heat flux toward cylinder (kW m$^{-2}$)")
    ax1.grid(alpha=0.25)
    ax1.legend(frameon=True, loc="upper left", bbox_to_anchor=(0.0, 0.96))
    ax1.text(-0.075, 1.025, "(b)", transform=ax1.transAxes, va="bottom", fontweight="bold", fontsize=14, color="#17365D")
    ax1.text(0.97, 0.05, f"Upper-surface means\nwall tally: {wall_average_trend/1000:.1f} kW m$^{{-2}}$\nnearest cell: {qn_average_trend/1000:.1f} kW m$^{{-2}}$", transform=ax1.transAxes, ha="right", va="bottom", fontsize=9.4, bbox={"facecolor": "white", "alpha": 0.92, "edgecolor": "0.75"})

    zoom = angle >= 149.0
    ax2.fill_between(angle[zoom], (wall_mean[zoom] - 1.96 * wall_se[zoom]) / scale, (wall_mean[zoom] + 1.96 * wall_se[zoom]) / scale, color="#D8742F", alpha=0.16, linewidth=0)
    ax2.plot(angle[zoom], wall_mean[zoom] / scale, "o", color="#D8742F", markersize=6.0, alpha=0.72, label=r"raw $q_w$ bins")
    ax2.plot(angle[zoom], wall_trend[zoom] / scale, color="#B84B13", linewidth=2.4, label=r"even local fit $\widetilde q_w$")
    ax2.plot(angle[zoom], qn_mean[zoom] / scale, "s", color="#2E8B57", markersize=5.2, alpha=0.56, label=r"raw $-q_n(r_1,\theta)$ bins")
    ax2.plot(angle[zoom], qn_trend[zoom] / scale, color="#1E6E43", linestyle="--", linewidth=2.3, label=r"even local fit $-\widetilde q_n$")
    ax2.axvline(180.0, color="0.25", linestyle=":", linewidth=1.4)
    ax2.set_xlim(148.5, 180.5)
    ax2.set_xlabel(r"Surface angle, $\theta$ (deg)")
    ax2.set_ylabel(r"Heat flux toward cylinder (kW m$^{-2}$)")
    ax2.grid(alpha=0.25)
    ax2.legend(frameon=True, loc="lower left", fontsize=9.0)
    ax2.text(-0.075, 1.025, "(c)", transform=ax2.transAxes, va="bottom", fontweight="bold", fontsize=14, color="#17365D")
    ax2.text(0.97, 0.97, r"cylinder symmetry: $\partial q/\partial\theta=0$ at $180^{\circ}$", transform=ax2.transAxes, ha="right", va="top", fontsize=9.2)
    save(fig, output, "mach12_wall_heat_flux_physical")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--acf", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    style()
    ratios = read_ratios(args.metrics)
    complete_hierarchy_figure(ratios, args.output)
    effective_sampling_figure(args.acf, args.reference, args.output)
    wall_heat_flux_figure(args.reference, args.output, args.audit_output)


if __name__ == "__main__":
    main()
