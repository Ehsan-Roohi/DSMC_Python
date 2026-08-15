#!/usr/bin/env python3
"""Replot MV15C-A1 q_y in the locked MV8 publication style.

The script performs no prediction, fitting, or DSMC.  It reads the already
locked fresh predictions and constructs the same leave-one-seed-out reference
used by MV15C-A1.  Each condition is rendered as six columns by two rows:
physical q_y fields above and signed q_y/q_ref error fields below.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np


REQUIRED_KEYS = (
    "conditions",
    "seeds",
    "raw_b3_qy",
    "vision_b3_qy",
    "selected_b3_qy",
    "tsvd_b3_qy",
    "raw_b10_qy",
    "q_ref_scales",
)
METHOD_ORDER = (
    "reference",
    "raw_b3",
    "vision_b3",
    "selected_b3",
    "tsvd_b3",
    "raw_b10",
)
METHOD_TITLES = (
    "Reference",
    "Raw DSMC\n$B=3$",
    "MambaIRv2\n$B=3$",
    "DCIR-QY\n$B=3$",
    "TSVD/POD\n$B=3$",
    "Raw DSMC\n$B=10$",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def leave_one_seed_out_targets(
    raw_b10: np.ndarray, conditions: np.ndarray, seeds: np.ndarray
) -> np.ndarray:
    targets = np.empty_like(raw_b10, dtype=np.float64)
    for index, (condition, seed) in enumerate(zip(conditions, seeds, strict=True)):
        peers = (conditions == condition) & (seeds != seed)
        if np.count_nonzero(peers) != 3:
            raise ValueError(
                f"condition {condition!r} seed {seed} does not have exactly three peers"
            )
        targets[index] = np.mean(raw_b10[peers], axis=0, dtype=np.float64)
    return targets


def condition_title(condition: str) -> str:
    match = re.fullmatch(r"kn(\d+)p(\d+)_u(\d+)", condition)
    if match is None:
        return condition
    whole, fraction, speed = match.groups()
    kn = f"{whole}.{fraction}"
    return rf"$Kn={kn},\ U_{{\rm lid}}={int(speed)}\ \mathrm{{m\,s^{{-1}}}}$"


def robust_symmetric_limit(values: list[np.ndarray], quantile: float, floor: float) -> float:
    flattened = np.concatenate([np.abs(np.asarray(value)).ravel() for value in values])
    return max(float(np.quantile(flattened, quantile)), floor)


def plot_condition(
    output_directory: Path,
    condition: str,
    seed: int,
    normalized_fields: dict[str, np.ndarray],
    reference: np.ndarray,
    q_ref_scale: float,
    display_quantile: float,
) -> dict[str, object]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "mathtext.fontset": "dejavuserif",
            "font.size": 9.5,
            "axes.linewidth": 0.85,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    normalized = {"reference": np.asarray(reference, dtype=np.float64)}
    normalized.update(
        {name: np.asarray(normalized_fields[name], dtype=np.float64) for name in METHOD_ORDER[1:]}
    )
    physical = {name: value * float(q_ref_scale) for name, value in normalized.items()}
    errors = {
        name: np.zeros_like(reference, dtype=np.float64)
        if name == "reference"
        else 100.0 * (value - reference)
        for name, value in normalized.items()
    }

    physical_limit = robust_symmetric_limit(
        [physical[name] for name in METHOD_ORDER], display_quantile, 1.0e-12
    )
    error_limit = robust_symmetric_limit(
        [errors[name] for name in METHOD_ORDER[1:]], display_quantile, 1.0e-4
    )
    physical_clipped = int(
        sum(np.count_nonzero(np.abs(physical[name]) > physical_limit) for name in METHOD_ORDER)
    )
    error_clipped = int(
        sum(np.count_nonzero(np.abs(errors[name]) > error_limit) for name in METHOD_ORDER[1:])
    )

    figure, axes = plt.subplots(
        2,
        len(METHOD_ORDER),
        figsize=(15.7, 5.55),
        constrained_layout=True,
    )
    physical_norm = TwoSlopeNorm(vmin=-physical_limit, vcenter=0.0, vmax=physical_limit)
    error_norm = TwoSlopeNorm(vmin=-error_limit, vcenter=0.0, vmax=error_limit)
    physical_levels = np.linspace(-physical_limit, physical_limit, 41)
    error_levels = np.linspace(-error_limit, error_limit, 41)
    physical_artist = None
    error_artist = None
    ny, nx = reference.shape

    for column, (name, title) in enumerate(zip(METHOD_ORDER, METHOD_TITLES, strict=True)):
        physical_artist = axes[0, column].contourf(
            physical[name],
            levels=physical_levels,
            cmap="RdBu_r",
            norm=physical_norm,
            extend="both",
        )
        axes[0, column].set_title(title, pad=6)
        error_artist = axes[1, column].contourf(
            errors[name],
            levels=error_levels,
            cmap="RdBu_r",
            norm=error_norm,
            extend="both",
        )
        if name == "reference":
            axes[1, column].text(
                0.5,
                0.5,
                "Reference",
                transform=axes[1, column].transAxes,
                ha="center",
                va="center",
                color="0.45",
            )
        for row in range(2):
            axis = axes[row, column]
            axis.set_aspect("equal")
            axis.set_xlim(0, nx - 1)
            axis.set_ylim(0, ny - 1)
            axis.set_xticks([0, (nx - 1) / 2, nx - 1])
            axis.set_yticks([0, (ny - 1) / 2, ny - 1])
            axis.set_xticklabels(["0", "0.5", "1"] if row == 1 else [])
            axis.set_yticklabels(["0", "0.5", "1"] if column == 0 else [])
            if row == 1:
                axis.set_xlabel(r"$x/L$")
            if column == 0:
                axis.set_ylabel(r"$y/L$")

    if physical_artist is None or error_artist is None:
        raise RuntimeError("no contour artist was generated")
    physical_bar = figure.colorbar(
        physical_artist, ax=axes[0, :], shrink=0.91, pad=0.012
    )
    physical_bar.set_label(r"$q_y$ [W m$^{-2}$]")
    error_bar = figure.colorbar(error_artist, ax=axes[1, :], shrink=0.91, pad=0.012)
    error_bar.set_label(r"$100\,\Delta q_y/q_{ref}$ [\%]")
    figure.suptitle(condition_title(condition), y=1.075, fontsize=11)

    stem = f"mv15c_a1_qy_six_panel_{condition}_seed_{seed}"
    png = output_directory / f"{stem}.png"
    pdf = output_directory / f"{stem}.pdf"
    figure.savefig(png, dpi=600, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return {
        "condition": condition,
        "seed": int(seed),
        "png": png.name,
        "pdf": pdf.name,
        "method_order": list(METHOD_ORDER),
        "colormap": "RdBu_r",
        "display_quantile": float(display_quantile),
        "physical_limit_W_m2": physical_limit,
        "physical_values_clipped": physical_clipped,
        "error_definition": "100 * (candidate_over_q_ref - reference_over_q_ref)",
        "error_percent_limit": error_limit,
        "error_values_clipped": error_clipped,
        "q_ref_W_m2": float(q_ref_scale),
    }


def run(output_root: Path, return_directory: Path, display_quantile: float) -> dict[str, object]:
    output_root = output_root.resolve()
    prediction_file = output_root / "locked_fresh_predictions.npz"
    if not prediction_file.is_file():
        raise FileNotFoundError(prediction_file)
    if not 0.90 <= display_quantile < 1.0:
        raise ValueError("display quantile must be in [0.90, 1.0)")

    with np.load(prediction_file, allow_pickle=False) as source:
        missing = sorted(set(REQUIRED_KEYS).difference(source.files))
        if missing:
            raise ValueError(f"locked prediction file is missing {missing}")
        locked = {key: np.asarray(source[key]) for key in REQUIRED_KEYS}
    conditions = locked["conditions"]
    seeds = locked["seeds"]
    reference = leave_one_seed_out_targets(locked["raw_b10_qy"], conditions, seeds)
    fields = {
        "raw_b3": locked["raw_b3_qy"],
        "vision_b3": locked["vision_b3_qy"],
        "selected_b3": locked["selected_b3_qy"],
        "tsvd_b3": locked["tsvd_b3_qy"],
        "raw_b10": locked["raw_b10_qy"],
    }

    figure_directory = output_root / "mv15c_a1_publication_figures"
    figure_directory.mkdir(parents=True, exist_ok=True)
    records = []
    for condition in sorted(str(value) for value in np.unique(conditions)):
        condition_indices = np.flatnonzero(conditions == condition)
        index = int(condition_indices[np.argmin(seeds[condition_indices])])
        records.append(
            plot_condition(
                figure_directory,
                condition,
                int(seeds[index]),
                {name: value[index] for name, value in fields.items()},
                reference[index],
                float(locked["q_ref_scales"][index]),
                display_quantile,
            )
        )

    manifest = {
        "stage": "MV15C_A1_qy_six_panel_replot",
        "status": "complete_no_prediction_no_fitting_no_DSMC",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(prediction_file),
        "input_sha256": sha256(prediction_file),
        "reference": "leave-one-seed-out mean of the other three Raw B10 fields",
        "layout": "2_rows_by_6_columns",
        "records": records,
    }
    manifest_path = figure_directory / "mv15c_a1_qy_six_panel_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    return_directory = return_directory.resolve()
    return_directory.mkdir(parents=True, exist_ok=True)
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = return_directory / f"MV15C_A1_QY_SIX_PANEL_FIGURES_{tag}.zip"
    files = [manifest_path]
    for record in records:
        files.extend((figure_directory / str(record["png"]), figure_directory / str(record["pdf"])))
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as stream:
        for path in files:
            stream.write(path, arcname=path.name)
    archive_sha = sha256(archive)
    pointer = return_directory / "LAST_MV15C_A1_QY_SIX_PANEL_RESULT.env"
    pointer.write_text(
        "\n".join(
            (
                f"MV15C_A1_SIX_PANEL_OUTPUT={figure_directory}",
                f"MV15C_A1_SIX_PANEL_ARCHIVE={archive}",
                f"MV15C_A1_SIX_PANEL_ARCHIVE_SHA256={archive_sha}",
                "",
            )
        ),
        encoding="utf-8",
    )
    result = dict(manifest)
    result.update(
        {
            "archive": str(archive),
            "archive_sha256": archive_sha,
            "pointer": str(pointer),
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--return-directory", type=Path, required=True)
    parser.add_argument("--display-quantile", type=float, default=0.995)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.output_root, args.return_directory, args.display_quantile),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
