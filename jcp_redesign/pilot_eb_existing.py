#!/usr/bin/env python3
"""Development-only EB fusion pilot on the archived cavity arrays.

This script is deliberately not a confirmatory analysis.  It estimates the
Raw-B3 noise power from the overlapping Raw-B3/Raw-B10 identity

    E |X_3 - X_10|^2 = sigma^2 (1/3 - 1/10),

using leave-one-seed-out pooling within each condition.  It then applies a
blockwise empirical-Bayes gain in an orthonormal 2-D DCT.  The purpose is to
decide whether adaptive prior/observation fusion is promising enough to merit
new DSMC trajectories.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.fft import dctn, idctn


METHOD_KEYS = {
    "raw_b3": "method_raw_b3",
    "raw_b10": "method_raw_b10",
    "prior": "method_development_prior_only",
    "frozen_selected": "method_selected_b3",
    "pure_wiener": "method_pure_continuous_Wiener_B3",
    "mamba_wiener": "method_Mamba_plus_continuous_Wiener_residual",
}


def nrmse(field: np.ndarray, target: np.ndarray) -> float:
    den = float(np.sum(target * target))
    if den <= 0.0:
        raise ValueError("Reference energy is not positive")
    return math.sqrt(float(np.sum((field - target) ** 2)) / den)


def block_slices(n: int, width: int, minimum_width: int = 8) -> list[slice]:
    edges = list(range(0, n, width)) + [n]
    if len(edges) >= 3 and edges[-1] - edges[-2] < minimum_width:
        edges.pop(-2)
    return [slice(a, b) for a, b in zip(edges[:-1], edges[1:], strict=True)]


def estimate_noise_bins(
    raw_b3: np.ndarray,
    raw_b10: np.ndarray,
    peer_indices: np.ndarray,
    bin_width: int,
) -> np.ndarray:
    """Return coefficient-noise power at B=3, constant within each bin."""
    if peer_indices.size < 2:
        raise ValueError("At least two peer seeds are required")
    delta = raw_b3[peer_indices] - raw_b10[peer_indices]
    coeff = np.stack([dctn(x, norm="ortho") for x in delta], axis=0)
    # Overlapping B3 subset of B10: Var(X3-X10)=sigma^2*(1/3-1/10).
    b3_from_delta = (10.0 / 7.0) * coeff**2
    noise = np.zeros(coeff.shape[1:], dtype=np.float64)
    ys = block_slices(noise.shape[0], bin_width)
    xs = block_slices(noise.shape[1], bin_width)
    for sy in ys:
        for sx in xs:
            noise[sy, sx] = float(np.mean(b3_from_delta[:, sy, sx]))
    return noise


def eb_fuse(
    observation: np.ndarray,
    prior: np.ndarray,
    noise_power: np.ndarray,
    bin_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    z = dctn(observation, norm="ortho")
    mu = dctn(prior, norm="ortho")
    residual = z - mu
    gain = np.zeros_like(residual)
    ys = block_slices(residual.shape[0], bin_width)
    xs = block_slices(residual.shape[1], bin_width)
    for sy in ys:
        for sx in xs:
            r = float(np.mean(residual[sy, sx] ** 2))
            n = float(np.mean(noise_power[sy, sx]))
            h = 0.0 if r <= 0.0 else float(np.clip((r - n) / r, 0.0, 1.0))
            gain[sy, sx] = h
    estimate = idctn(mu + gain * residual, norm="ortho")
    return estimate, gain


def geometric_mean(values: list[float]) -> float:
    a = np.asarray(values, dtype=np.float64)
    if np.any(a <= 0.0):
        raise ValueError("Geometric mean requires positive values")
    return float(np.exp(np.mean(np.log(a))))


def run(data_path: Path, output_dir: Path, bin_width: int) -> dict:
    arrays = np.load(data_path, allow_pickle=True)
    conditions = arrays["conditions"].astype(str)
    seeds = arrays["seeds"].astype(np.int64)
    targets = arrays["target_qy"]
    raw_b3 = arrays["method_raw_b3"]
    raw_b10 = arrays["method_raw_b10"]
    pnn = arrays["method_development_prior_only"]
    pnet = arrays["method_vision_b3"]

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    gain_records: list[np.ndarray] = []

    for i, (condition, seed) in enumerate(zip(conditions, seeds, strict=True)):
        peers = np.flatnonzero((conditions == condition) & (np.arange(len(seeds)) != i))
        noise = estimate_noise_bins(raw_b3, raw_b10, peers, bin_width)
        pnn_eb, pnn_gain = eb_fuse(raw_b3[i], pnn[i], noise, bin_width)
        pnet_eb, pnet_gain = eb_fuse(raw_b3[i], pnet[i], noise, bin_width)
        p0_eb, p0_gain = eb_fuse(raw_b3[i], np.zeros_like(raw_b3[i]), noise, bin_width)
        gain_records.append(np.stack([pnn_gain, pnet_gain, p0_gain], axis=0))

        fields = {name: arrays[key][i] for name, key in METHOD_KEYS.items()}
        fields["pnn_eb"] = pnn_eb
        fields["pnet_eb"] = pnet_eb
        fields["p0_eb"] = p0_eb
        errors = {name: nrmse(field, targets[i]) for name, field in fields.items()}
        raw10_error = errors["raw_b10"]
        envelope_error = min(errors["prior"], errors["p0_eb"])
        for name, error in errors.items():
            rows.append(
                {
                    "condition": condition,
                    "seed": int(seed),
                    "method": name,
                    "nrmse": error,
                    "ratio_to_raw_b10": error / raw10_error,
                    "ratio_to_prior_p0_envelope": error / envelope_error,
                }
            )

    csv_path = output_dir / "pilot_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, object] = {
        "classification": "development_only_retrospective_pilot",
        "data_path": str(data_path),
        "bin_width": bin_width,
        "minimum_modes_per_full_bin": bin_width * bin_width,
        "noise_identity": "Var(Raw-B3 - Raw-B10) = sigma^2*(1/3 - 1/10)",
        "conditions": {},
    }
    for condition in sorted(set(conditions)):
        subset = [r for r in rows if r["condition"] == condition]
        method_names = sorted({str(r["method"]) for r in subset})
        cond_summary: dict[str, object] = {}
        for method in method_names:
            mr = [r for r in subset if r["method"] == method]
            cond_summary[method] = {
                "geometric_ratio_to_raw_b10": geometric_mean(
                    [float(r["ratio_to_raw_b10"]) for r in mr]
                ),
                "geometric_ratio_to_prior_p0_envelope": geometric_mean(
                    [float(r["ratio_to_prior_p0_envelope"]) for r in mr]
                ),
                "unit_count_below_raw_b10": sum(
                    float(r["ratio_to_raw_b10"]) < 1.0 for r in mr
                ),
                "unit_count_below_0p95_envelope": sum(
                    float(r["ratio_to_prior_p0_envelope"]) < 0.95 for r in mr
                ),
                "n_units": len(mr),
            }
        summary["conditions"][condition] = cond_summary

    summary["provisional_gates"] = {}
    for method, label in (("pnn_eb", "PNN+EB"), ("pnet_eb", "PNET+EB")):
        gate_rows = [r for r in rows if r["method"] == method]
        pass_count = sum(
            float(r["ratio_to_prior_p0_envelope"]) < 0.95 for r in gate_rows
        )
        summary["provisional_gates"][method] = {
            "criterion": f"{label} below 0.95*min(prior-only,P0) in at least 75% of units",
            "pass_count": pass_count,
            "n_units": len(gate_rows),
            "pass": pass_count >= math.ceil(0.75 * len(gate_rows)),
        }
    summary["warning"] = (
        "This pilot uses already-seen evaluation conditions and is not publication evidence."
    )

    summary_path = output_dir / "pilot_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        output_dir / "pilot_gains.npz",
        conditions=conditions,
        seeds=seeds,
        gains=np.stack(gain_records, axis=0),
        gain_names=np.array(["pnn_eb", "pnet_eb", "p0_eb"]),
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("repro_v2/data_arrays/cavity_audit_fields.npz"),
    )
    parser.add_argument("--output", type=Path, default=Path("jcp_redesign/pilot_out"))
    parser.add_argument("--bin-width", type=int, default=10)
    args = parser.parse_args()
    if args.bin_width < 8:
        raise SystemExit("bin-width must be at least 8 so each full bin has >=64 modes")
    result = run(args.data, args.output, args.bin_width)
    print(json.dumps(result["provisional_gates"], indent=2))


if __name__ == "__main__":
    main()
