#!/usr/bin/env python3
"""Extract the lid-adjacent SPARTA profile and compare it with PRE data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np


def latest_dump(run_dir: Path) -> Path:
    candidates = list(run_dir.glob("grid.final.*"))
    if not candidates:
        raise FileNotFoundError(f"No grid.final.* dump was found in {run_dir}")

    def step(path: Path) -> int:
        match = re.search(r"(\d+)$", path.name)
        return int(match.group(1)) if match else -1

    return max(candidates, key=step)


def read_last_snapshot(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header_positions = [i for i, line in enumerate(lines) if line.startswith("ITEM: CELLS")]
    if not header_positions:
        raise ValueError(f"No ITEM: CELLS block in {path}")
    start = header_positions[-1]
    columns = lines[start].split()[2:]
    rows: list[list[float]] = []
    for line in lines[start + 1 :]:
        if line.startswith("ITEM:"):
            break
        if line.strip():
            rows.append([float(value) for value in line.split()])
    data = np.asarray(rows, dtype=float)
    if data.ndim != 2 or data.shape[1] != len(columns):
        raise ValueError(f"Malformed grid block in {path}")
    return columns, data


def load_reference(path: Path, kn: float) -> dict[str, np.ndarray]:
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if abs(float(row["kn"]) - kn) < 1.0e-12:
                rows.append(row)
    if not rows:
        raise ValueError(f"No reference rows for Kn={kn:g}")
    return {
        "x": np.asarray([float(row["x_over_L"]) for row in rows]),
        "slip": np.asarray([float(row["macro_slip_over_Uwall"]) for row in rows]),
        "temperature": np.asarray([float(row["macro_T_K"]) for row in rows]),
    }


def process(run_dir: Path, reference_csv: Path) -> dict[str, float | bool | str]:
    metadata = json.loads((run_dir / "case_metadata.json").read_text(encoding="utf-8"))
    columns, values = read_last_snapshot(latest_dump(run_dir))
    index = {name: i for i, name in enumerate(columns)}
    required = ["xc", "yc", "f_flowavg[2]", "f_flowavg[5]"]
    missing = [name for name in required if name not in index]
    if missing:
        raise ValueError(f"Missing columns in SPARTA dump: {missing}; got {columns}")

    yc = values[:, index["yc"]]
    top_y = float(np.max(yc))
    lid = values[np.isclose(yc, top_y, rtol=0.0, atol=max(1e-18, abs(top_y) * 1e-12))]
    lid = lid[np.argsort(lid[:, index["xc"]])]
    length = float(metadata["length_m"])
    wall_speed = float(metadata["lid_velocity_m_per_s"])
    x = lid[:, index["xc"]] / length
    u = lid[:, index["f_flowavg[2]"]]
    temperature = lid[:, index["f_flowavg[5]"]]
    slip = (wall_speed - u) / wall_speed

    profile_path = run_dir / "lid_profile.csv"
    np.savetxt(
        profile_path,
        np.column_stack((x, slip, temperature, u)),
        delimiter=",",
        header="x_over_L,macro_slip_over_Uwall,macro_T_K,macro_u_m_per_s",
        comments="",
    )

    ref = load_reference(reference_csv, float(metadata["kn"]))
    interp_slip = np.interp(ref["x"], x, slip)
    interp_temp = np.interp(ref["x"], x, temperature)
    slip_rmse = float(np.sqrt(np.mean((interp_slip - ref["slip"]) ** 2)))
    temp_rmse = float(np.sqrt(np.mean((interp_temp - ref["temperature"]) ** 2)))
    report: dict[str, float | bool | str] = {
        "kn": float(metadata["kn"]),
        "dump": latest_dump(run_dir).name,
        "profile": profile_path.name,
        "slip_rmse": slip_rmse,
        "temperature_rmse_K": temp_rmse,
        "slip_gate": 0.08,
        "temperature_gate_K": 2.0,
        "slip_pass": slip_rmse <= 0.08,
        "temperature_pass": temp_rmse <= 2.0,
        "overall_pass": slip_rmse <= 0.08 and temp_rmse <= 2.0,
    }
    (run_dir / "validation_metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
        axes[0].plot(x, slip, label="SPARTA, macroscopic")
        axes[0].plot(ref["x"], ref["slip"], "o", fillstyle="none", label="PRE Fig. 4")
        axes[0].set(xlabel="x/L", ylabel="Slip/Uwall")
        axes[1].plot(x, temperature, label="SPARTA, macroscopic")
        axes[1].plot(ref["x"], ref["temperature"], "o", fillstyle="none", label="PRE Fig. 5")
        axes[1].set(xlabel="x/L", ylabel="Temperature [K]")
        for axis in axes:
            axis.grid(alpha=0.25)
            axis.legend(frameon=False)
        fig.savefig(run_dir / "validation.png", dpi=220)
        plt.close(fig)
    except ImportError:
        report["plot_note"] = "Install requirements.txt to create validation.png"
        (run_dir / "validation_metrics.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "reference"
        / "mohammadzadeh_2012_lid_profiles.csv",
    )
    args = parser.parse_args()
    report = process(args.run_dir.resolve(), args.reference.resolve())
    return 0 if bool(report["overall_pass"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
