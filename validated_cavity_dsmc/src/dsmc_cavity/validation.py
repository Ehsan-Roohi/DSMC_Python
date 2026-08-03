"""Quantitative validation against Mohammadzadeh et al. (PRE 2012)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_reference(path: Path, kn: float) -> dict[str, np.ndarray]:
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if abs(float(row["kn"]) - kn) < 1e-12:
                rows.append(row)
    if not rows:
        raise ValueError(f"No reference data for Kn={kn}")
    return {
        "x": np.asarray([float(row["x_over_L"]) for row in rows]),
        "slip": np.asarray([float(row["macro_slip_over_Uwall"]) for row in rows]),
        "temperature": np.asarray([float(row["macro_T_K"]) for row in rows]),
        "slip_digitization_uncertainty": np.asarray(
            [float(row["estimated_digitization_uncertainty_slip"]) for row in rows]
        ),
        "temperature_digitization_uncertainty": np.asarray(
            [float(row["estimated_digitization_uncertainty_T_K"]) for row in rows]
        ),
    }


def metrics(result_csv: Path, reference_csv: Path, kn: float) -> dict[str, float | bool]:
    result = np.genfromtxt(result_csv, delimiter=",", names=True)
    ref = load_reference(reference_csv, kn)
    slip = np.interp(ref["x"], result["x_over_L"], result["macro_slip_over_Uwall"])
    temperature = np.interp(ref["x"], result["x_over_L"], result["macro_T_K"])
    slip_error = slip - ref["slip"]
    temp_error = temperature - ref["temperature"]
    slip_rmse = float(np.sqrt(np.mean(slip_error**2)))
    temp_rmse = float(np.sqrt(np.mean(temp_error**2)))
    return {
        "kn": kn,
        "points": len(ref["x"]),
        "slip_rmse": slip_rmse,
        "slip_max_abs": float(np.max(np.abs(slip_error))),
        "temperature_rmse_K": temp_rmse,
        "temperature_max_abs_K": float(np.max(np.abs(temp_error))),
        "slip_pass": slip_rmse <= 0.08,
        "temperature_pass": temp_rmse <= 2.0,
        "overall_pass": slip_rmse <= 0.08 and temp_rmse <= 2.0,
    }


def plot_validation(result_csv: Path, reference_csv: Path, kn: float, output: Path) -> None:
    import matplotlib.pyplot as plt

    result = np.genfromtxt(result_csv, delimiter=",", names=True)
    ref = load_reference(reference_csv, kn)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    axes[0].plot(result["x_over_L"], result["macro_slip_over_Uwall"], "-", label="Python DSMC")
    axes[0].errorbar(
        ref["x"], ref["slip"], yerr=ref["slip_digitization_uncertainty"],
        fmt="o", fillstyle="none", label="Mohammadzadeh et al."
    )
    axes[0].set(xlabel="x/L", ylabel=r"$u_{slip}/U_{wall}$", title=f"Kn = {kn:g}")
    axes[1].plot(result["x_over_L"], result["macro_T_K"], "-", label="Python DSMC")
    axes[1].errorbar(
        ref["x"], ref["temperature"], yerr=ref["temperature_digitization_uncertainty"],
        fmt="o", fillstyle="none", label="Mohammadzadeh et al."
    )
    axes[1].set(xlabel="x/L", ylabel="T [K]", title=f"Kn = {kn:g}")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_csv", type=Path)
    parser.add_argument("--kn", type=float, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "reference" / "mohammadzadeh_2012_lid_profiles.csv",
    )
    parser.add_argument("--output", type=Path, default=Path("validation_metrics.json"))
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args(argv)
    report = metrics(args.result_csv, args.reference, args.kn)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.plot:
        plot_validation(args.result_csv, args.reference, args.kn, args.plot)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["overall_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
