#!/usr/bin/env python3
"""Create the publication DSMC grid/PPC-sensitivity figure.

The figure is built only from the reduced eight-realisation Run-1/Run-3 products.
It deliberately separates low-order and high-order field errors so the
sampling-limited fourth moment does not compress the converged fields.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-jfm-run3-sensitivity")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
RELEASE_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DATA_DIR = RELEASE_ROOT / "data" / "reduced"
DEFAULT_OUTPUT_DIR = RELEASE_ROOT / "figures" / "generated"

COMPARISON_CSV = "dsmc_sensitivity_vs_N160_ppc128.csv"
COMPARISON_JSON = "dsmc_sensitivity_vs_N160_ppc128.json"
SCALAR_CSV = "dsmc_grid_particle_sensitivity.csv"
SCALAR_JSON = "dsmc_grid_particle_sensitivity.json"

CASE_ORDER = (
    ("120x120", 128, "$120^2$\n128 ppc", "grid"),
    ("200x200", 128, "$200^2$\n128 ppc", "grid"),
    ("160x160", 64, "$160^2$\n64 ppc", "particles"),
    ("160x160", 256, "$160^2$\n256 ppc", "particles"),
)

LOW_FIELDS = (
    ("relL2_temperature", r"$T$", "#3B78A8", ""),
    ("relL2_velocity", r"$\boldsymbol{u}$", "#4C9A82", "//"),
    ("relL2_stress", r"$\boldsymbol{\sigma}$", "#D79A35", "xx"),
)

HIGH_FIELDS = (
    ("relL2_heat_flux", r"$\boldsymbol{q}$", "#3B78A8", ""),
    ("relL2_R", r"$R^{\mathrm{cl}}$", "#B36A8C", "//"),
    ("relL2_Delta", r"$\Delta$", "#D2763D", "xx"),
)

SCALAR_METRICS = (
    ("f_AF_domain", r"$f_{AF\mid\Omega}$", "#3B78A8", "o"),
    ("mean_IAF_AF", r"$\langle I_{AF}\rangle_{AF}$", "#4C9A82", "s"),
    ("PDelta_over_PR", r"$P_\Delta/P_R$", "#D2763D", "^"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"empty CSV: {path}")
    return rows


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12.0,
            "axes.titlesize": 13.2,
            "axes.labelsize": 12.4,
            "legend.fontsize": 11.0,
            "xtick.labelsize": 11.2,
            "ytick.labelsize": 11.0,
            "figure.titlesize": 14.0,
            "axes.linewidth": 0.9,
            "lines.linewidth": 1.9,
            "lines.markersize": 6.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def candidate_key(row: dict[str, str]) -> tuple[str, int]:
    return row["candidate_grid"], int(round(float(row["candidate_mean_particle_count"])))


def order_comparisons(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_key = {candidate_key(row): row for row in rows}
    expected = {(grid, ppc) for grid, ppc, _, _ in CASE_ORDER}
    if set(by_key) != expected:
        raise RuntimeError(
            f"unexpected comparison cases: {sorted(by_key)}; expected {sorted(expected)}"
        )
    return [by_key[(grid, ppc)] for grid, ppc, _, _ in CASE_ORDER]


def order_scalars(
    rows: list[dict[str, str]], reference: dict[str, float]
) -> list[dict[str, float]]:
    by_key: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        key = (
            row["source_grid"],
            int(round(float(row["mean_particle_count"]))),
        )
        by_key[key] = row
    ordered: list[dict[str, float]] = []
    for grid, ppc, label, group in CASE_ORDER:
        row = by_key[(grid, ppc)]
        item: dict[str, float | str] = {
            "case": f"N{grid.split('x')[0]}_ppc{ppc}",
            "case_label": label.replace("$", "").replace("\n", " / "),
            "sweep_group": group,
            "grid": grid,
            "ppc": ppc,
        }
        for metric, _, _, _ in SCALAR_METRICS:
            value = float(row[metric])
            item[metric] = value
            item[f"{metric}_over_baseline"] = value / float(reference[metric])
        ordered.append(item)  # type: ignore[arg-type]
    return ordered


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.018,
        0.975,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        fontsize=12.5,
    )


def grouped_bars(
    ax: plt.Axes,
    rows: list[dict[str, str]],
    definitions: tuple[tuple[str, str, str, str], ...],
) -> None:
    x = np.arange(len(rows), dtype=float)
    width = 0.235
    offsets = np.linspace(-width, width, len(definitions))
    for offset, (key, label, color, hatch) in zip(offsets, definitions):
        values = np.asarray([100.0 * float(row[key]) for row in rows])
        ax.bar(
            x + offset,
            values,
            width=width,
            color=color,
            edgecolor="#35404A",
            linewidth=0.55,
            hatch=hatch,
            label=label,
            zorder=3,
        )
    ax.set_xticks(x, [label for _, _, label, _ in CASE_ORDER])
    ax.axvline(1.5, color="0.72", linewidth=0.8, zorder=1)
    ax.grid(axis="y", color="0.88", linewidth=0.6, zorder=0)
    ax.set_xlim(-0.58, len(rows) - 0.42)


def make_figure(
    comparisons: list[dict[str, str]],
    scalar_rows: list[dict[str, float]],
    reference: dict[str, float],
    output_stem: Path,
) -> None:
    configure_matplotlib()
    # A wide canvas preserves legibility after inclusion at full manuscript
    # width and leaves dedicated space above each axis for its legend.
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.55), constrained_layout=False)
    ax0, ax1, ax2 = axes

    grouped_bars(ax0, comparisons, LOW_FIELDS)
    ax0.axhline(5.0, color="0.25", linestyle="--", linewidth=0.9, label="5% guide")
    ax0.set_ylim(0.0, 6.1)
    ax0.set_ylabel("relative $L_2$ difference (%)")
    ax0.set_title("")
    panel_label(ax0, "(a)")
    handles, labels = ax0.get_legend_handles_labels()
    legend0 = (handles, labels)

    grouped_bars(ax1, comparisons, HIGH_FIELDS)
    ax1.axhline(10.0, color="0.25", linestyle="--", linewidth=0.9, label="10% guide")
    ax1.axhline(15.0, color="0.25", linestyle=":", linewidth=1.0, label="15% guide")
    ax1.set_ylim(0.0, 124.0)
    ax1.set_ylabel("relative $L_2$ difference (%)")
    ax1.set_title("")
    panel_label(ax1, "(b)")
    handles, labels = ax1.get_legend_handles_labels()
    legend1 = (handles, labels)

    x = np.arange(len(scalar_rows), dtype=float)
    ax2.axhspan(0.9, 1.1, color="0.88", alpha=0.75, zorder=0, label=r"$\pm10\%$")
    ax2.axhline(1.0, color="0.25", linestyle="--", linewidth=0.9, zorder=1)
    for metric, label, color, marker in SCALAR_METRICS:
        y = np.asarray([float(row[f"{metric}_over_baseline"]) for row in scalar_rows])
        ax2.plot(x, y, color=color, marker=marker, label=label, zorder=3)
    ax2.set_xticks(x, [label for _, _, label, _ in CASE_ORDER])
    ax2.axvline(1.5, color="0.72", linewidth=0.8, zorder=1)
    ax2.set_xlim(-0.25, len(scalar_rows) - 0.75)
    ax2.set_ylim(0.80, 1.23)
    ax2.set_ylabel("diagnostic / baseline")
    ax2.set_title("")
    ax2.grid(axis="y", color="0.88", linewidth=0.6, zorder=0)
    panel_label(ax2, "(c)")
    legend2 = ax2.get_legend_handles_labels()
    for ax in axes:
        ax.tick_params(direction="out", width=0.7, length=3.0)

    centers = (0.178, 0.508, 0.842)
    titles = ("Lower-order fields", "Heat flux and fourth moments", "Anti-Fourier diagnostics")
    for center, title in zip(centers, titles):
        fig.text(center, 0.80, title, ha="center", va="center", fontsize=12.4)
    for center, (handles, labels) in zip(centers, (legend0, legend1, legend2)):
        fig.legend(
            handles,
            labels,
            frameon=False,
            ncol=2,
            loc="center",
            bbox_to_anchor=(center, 0.705),
            handlelength=2.1,
            columnspacing=1.25,
        )
    fig.subplots_adjust(left=0.065, right=0.992, bottom=0.19, top=0.60, wspace=0.34)

    metadata = {
        "Title": "Eight-seed DSMC grid and particle sensitivity",
        "Creator": "make_dsmc_sensitivity_figure.py",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(
        output_stem.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.025,
        metadata=metadata,
    )
    fig.savefig(
        output_stem.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.025,
    )
    plt.close(fig)


def write_figure_data(
    path: Path,
    comparisons: list[dict[str, str]],
    scalar_rows: list[dict[str, float]],
    reference: dict[str, float],
) -> None:
    scalar_by_case = {str(row["case"]): row for row in scalar_rows}
    output: list[dict[str, Any]] = [
        {
            "case": "N160_ppc128",
            "case_label": "160^2 / 128 ppc",
            "sweep_group": "baseline",
            "baseline": "N160_ppc128",
            "relL2_T_percent": 0.0,
            "relL2_u_percent": 0.0,
            "relL2_sigma_percent": 0.0,
            "relL2_q_percent": 0.0,
            "relL2_R_percent": 0.0,
            "relL2_Delta_percent": 0.0,
            "f_AF_domain": reference["f_AF_domain"],
            "f_AF_domain_over_baseline": 1.0,
            "mean_IAF_AF": reference["mean_IAF_AF"],
            "mean_IAF_AF_over_baseline": 1.0,
            "PDelta_over_PR": reference["PDelta_over_PR"],
            "PDelta_over_PR_over_baseline": 1.0,
        }
    ]
    for source_row, (grid, ppc, label, group) in zip(comparisons, CASE_ORDER):
        case = f"N{grid.split('x')[0]}_ppc{ppc}"
        scalar = scalar_by_case[case]
        output.append(
            {
                "case": case,
                "case_label": label.replace("$", "").replace("\n", " / "),
                "sweep_group": group,
                "baseline": "N160_ppc128",
                "relL2_T_percent": 100.0 * float(source_row["relL2_temperature"]),
                "relL2_u_percent": 100.0 * float(source_row["relL2_velocity"]),
                "relL2_sigma_percent": 100.0 * float(source_row["relL2_stress"]),
                "relL2_q_percent": 100.0 * float(source_row["relL2_heat_flux"]),
                "relL2_R_percent": 100.0 * float(source_row["relL2_R"]),
                "relL2_Delta_percent": 100.0 * float(source_row["relL2_Delta"]),
                "f_AF_domain": scalar["f_AF_domain"],
                "f_AF_domain_over_baseline": scalar["f_AF_domain_over_baseline"],
                "mean_IAF_AF": scalar["mean_IAF_AF"],
                "mean_IAF_AF_over_baseline": scalar["mean_IAF_AF_over_baseline"],
                "PDelta_over_PR": scalar["PDelta_over_PR"],
                "PDelta_over_PR_over_baseline": scalar["PDelta_over_PR_over_baseline"],
            }
        )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)


def write_audit(
    path: Path,
    inputs: list[Path],
    reference: dict[str, float],
    comparisons: list[dict[str, str]],
) -> None:
    extrema: dict[str, tuple[float, float]] = {}
    for key in (
        "relL2_temperature",
        "relL2_velocity",
        "relL2_stress",
        "relL2_heat_flux",
        "relL2_R",
        "relL2_Delta",
    ):
        values = [100.0 * float(row[key]) for row in comparisons]
        extrema[key] = (min(values), max(values))
    hashes = "\n".join(f"- `{item.name}`: `{sha256_file(item)}`" for item in inputs)
    text = f"""# DSMC sensitivity figure audit

## Scope

- Eight independent seeds per case, 8501 accumulated samples per cell.
- VHS argon, `omega=0.81`, `Kn=0.05`, `Uwall=100 m/s`, `Twall=300 K`.
- Baseline: `N160_ppc128`; all field differences are relative L2 norms on the common 160x160 target grid.
- Plot gates reproduce the predeclared campaign guides: 5% for low-order fields, 10% for heat flux/primary AF scalars, and 15% for fourth moments.

## Audited ranges

- T: {extrema['relL2_temperature'][0]:.3f}--{extrema['relL2_temperature'][1]:.3f}%.
- velocity: {extrema['relL2_velocity'][0]:.3f}--{extrema['relL2_velocity'][1]:.3f}%.
- stress: {extrema['relL2_stress'][0]:.3f}--{extrema['relL2_stress'][1]:.3f}%.
- heat flux: {extrema['relL2_heat_flux'][0]:.2f}--{extrema['relL2_heat_flux'][1]:.2f}%.
- R: {extrema['relL2_R'][0]:.2f}--{extrema['relL2_R'][1]:.2f}%.
- Delta: {extrema['relL2_Delta'][0]:.2f}--{extrema['relL2_Delta'][1]:.2f}%.
- Baseline scalars: f_AF_domain={reference['f_AF_domain']:.8f}, mean_IAF_AF={reference['mean_IAF_AF']:.8f}, PDelta_over_PR={reference['PDelta_over_PR']:.8f}.

## Caveats

- The N120-to-N160 remapping uses linear boundary extrapolation at 636 target points because cell-centre extents differ.
- The plotted field errors use raw ensemble-mean fields. Heat flux and R are marginally outside their desired 10%/15% guides; Delta is sampling-limited and is not pointwise converged.
- The stable result is the qualitative tensorial dominance: PDelta/PR remains approximately 0.038--0.049. Exact AF support remains threshold- and sampling-sensitive.
- Panel (c) divides each scalar by its baseline value to place the three diagnostics on one non-misleading scale; absolute values are retained in `figure_data.csv`.
- A stale inherited `interpolation.caveat` string in the scalar JSON calls the fields legacy/single-realisation. It is contradicted by the same file's audited paths/schema and by the reducer outputs (eight independent realisations per case), so it is not propagated into the figure or caption.

## Input hashes

{hashes}
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    paths = [
        args.data_dir / COMPARISON_CSV,
        args.data_dir / COMPARISON_JSON,
        args.data_dir / SCALAR_CSV,
        args.data_dir / SCALAR_JSON,
    ]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    comparisons = order_comparisons(load_csv(paths[0]))
    comparison_report = load_json(paths[1])
    scalar_csv_rows = load_csv(paths[2])
    scalar_report = load_json(paths[3])
    if len(comparisons) != 4 or len(scalar_csv_rows) != 4:
        raise RuntimeError("expected exactly four sensitivity candidates")
    if len(scalar_report.get("results", [])) != 4:
        raise RuntimeError("scalar JSON does not contain four audited results")

    reference = {
        key: float(value)
        for key, value in comparison_report["reference_metrics"].items()
    }
    scalar_rows = order_scalars(scalar_csv_rows, reference)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "fig_dsmc_grid_ppc_sensitivity"
    make_figure(comparisons, scalar_rows, reference, stem)
    write_figure_data(
        args.output_dir / "figure_data.csv", comparisons, scalar_rows, reference
    )
    write_audit(
        args.output_dir / "AUDIT.md", paths, reference, comparisons
    )
    print(stem.with_suffix(".pdf"))
    print(stem.with_suffix(".png"))
    print(args.output_dir / "figure_data.csv")
    print(args.output_dir / "AUDIT.md")


if __name__ == "__main__":
    main()
