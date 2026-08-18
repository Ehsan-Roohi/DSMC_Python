"""Compare separately completed cavity runs and plot their lid profiles."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dsmc_cavity.validation import load_reference, metrics


DEFAULT_MODELS = ("ntc-prescan", "sbt", "gbt", "ssbt", "sgbt")


def relative_l2(value: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(value - reference) / max(np.linalg.norm(reference), 1.0e-30))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--reference-model", default="ntc-prescan")
    args = parser.parse_args()

    missing = []
    for model in args.models:
        for name in ("metadata.json", "lid_profile.csv", "fields.npz"):
            path = args.root / model / name
            if not path.exists():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing completed-run files:\n" + "\n".join(missing))

    reference_dir = args.root / args.reference_model
    reference_profile = np.genfromtxt(
        reference_dir / "lid_profile.csv", delimiter=",", names=True
    )
    with np.load(reference_dir / "fields.npz") as archive:
        reference_u = archive["u"].copy()
        reference_temperature = archive["temperature"].copy()
    reference_metadata = json.loads((reference_dir / "metadata.json").read_text())
    kn = float(reference_metadata["config"]["kn"])
    publication_csv = ROOT / "reference" / "mohammadzadeh_2012_lid_profiles.csv"

    rows = []
    profiles = {}
    for model in args.models:
        run_dir = args.root / model
        profile = np.genfromtxt(run_dir / "lid_profile.csv", delimiter=",", names=True)
        profiles[model] = profile
        with np.load(run_dir / "fields.npz") as archive:
            u = archive["u"].copy()
            temperature = archive["temperature"].copy()
        metadata = json.loads((run_dir / "metadata.json").read_text())
        config = metadata["config"]
        collision = metadata["collision_statistics"]
        published = metrics(run_dir / "lid_profile.csv", publication_csv, kn)
        slip_on_reference = np.interp(
            reference_profile["x_over_L"],
            profile["x_over_L"],
            profile["macro_slip_over_Uwall"],
        )
        temperature_on_reference = np.interp(
            reference_profile["x_over_L"],
            profile["x_over_L"],
            profile["macro_T_K"],
        )
        rows.append(
            {
                "model": model,
                "backend": metadata["backend_resolved"],
                "seed": config["seed"],
                "dt_seconds": config["resolved_dt"],
                "steps": config["steps"],
                "end_time_seconds": config["steps"] * config["resolved_dt"],
                "runtime_seconds": metadata["runtime_seconds"],
                "selected": collision["selected"],
                "accepted": collision["accepted"],
                "max_probability": collision["max_probability"],
                "probability_exceedances": collision["probability_exceedances"],
                "slip_rmse_vs_ntc_prescan": float(
                    np.sqrt(
                        np.mean(
                            (
                                slip_on_reference
                                - reference_profile["macro_slip_over_Uwall"]
                            )
                            ** 2
                        )
                    )
                ),
                "temperature_rmse_K_vs_ntc_prescan": float(
                    np.sqrt(
                        np.mean(
                            (
                                temperature_on_reference
                                - reference_profile["macro_T_K"]
                            )
                            ** 2
                        )
                    )
                ),
                "u_relative_l2_vs_ntc_prescan": relative_l2(u, reference_u),
                "temperature_relative_l2_vs_ntc_prescan": relative_l2(
                    temperature, reference_temperature
                ),
                "mohammadzadeh_slip_rmse": published["slip_rmse"],
                "mohammadzadeh_temperature_rmse_K": published["temperature_rmse_K"],
                "mohammadzadeh_overall_pass": published["overall_pass"],
            }
        )

    args.root.mkdir(parents=True, exist_ok=True)
    csv_path = args.root / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    markdown = [
        "# Collision-model comparison",
        "",
        f"Kn = {kn:g}; reference model = `{args.reference_model}`.",
        "",
        "| Model | Slip RMSE vs NTC | T RMSE vs NTC [K] | Mohammadzadeh slip RMSE | Mohammadzadeh T RMSE [K] | Gate | P exceed. |",
        "|---|---:|---:|---:|---:|:---:|---:|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['model']} | {row['slip_rmse_vs_ntc_prescan']:.5f} | "
            f"{row['temperature_rmse_K_vs_ntc_prescan']:.3f} | "
            f"{row['mohammadzadeh_slip_rmse']:.5f} | "
            f"{row['mohammadzadeh_temperature_rmse_K']:.3f} | "
            f"{'PASS' if row['mohammadzadeh_overall_pass'] else 'FAIL'} | "
            f"{row['probability_exceedances']} |"
        )
    (args.root / "comparison.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    import matplotlib.pyplot as plt

    publication = load_reference(publication_csv, kn)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for model, profile in profiles.items():
        axes[0].plot(
            profile["x_over_L"], profile["macro_slip_over_Uwall"], label=model
        )
        axes[1].plot(profile["x_over_L"], profile["macro_T_K"], label=model)
    axes[0].errorbar(
        publication["x"],
        publication["slip"],
        yerr=publication["slip_digitization_uncertainty"],
        fmt="ko",
        fillstyle="none",
        label="Mohammadzadeh et al.",
    )
    axes[1].errorbar(
        publication["x"],
        publication["temperature"],
        yerr=publication["temperature_digitization_uncertainty"],
        fmt="ko",
        fillstyle="none",
        label="Mohammadzadeh et al.",
    )
    axes[0].set(xlabel="x/L", ylabel=r"$u_{slip}/U_{wall}$", title=f"Kn = {kn:g}")
    axes[1].set(xlabel="x/L", ylabel="T [K]", title=f"Kn = {kn:g}")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    figure.savefig(args.root / "comparison.png", dpi=220)
    plt.close(figure)
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
