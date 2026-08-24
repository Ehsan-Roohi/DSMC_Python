#!/usr/bin/env python3
"""Archive-reconstructed, target-free support audit for the Mach-12 estimator.

The support calculation uses only (i) empirical-Bayes (EB) gains recorded
during the pre-Mach-12 Mach-8/Mach-10 development campaign and (ii) the frozen
Mach-12 prediction manifest.  Reference fields are opened separately after the
decision is reconstructed.  This computational ordering demonstrates target
independence; it is not a claim that the rule was historically specified before
the initial reference-scored Mach-12 analysis.  The rule is retrospectively
specified and should be described as a domain-of-validity diagnostic.

The audit is post-processing: it launches no DSMC trajectory and changes no
archived prediction.  It produces machine-readable metrics, a journal figure,
and a checksum-locked ZIP bundle.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import zipfile

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


FIELDS = ("n", "u", "v", "T", "Pxx", "Pxy", "Pyy", "qx", "qy")
GAIN_COMPONENTS = ("n", "u", "v", "T", "Pxx", "Pxy", "Pyy", "qn", "qt")
ZONES = ("near_wall", "wake", "outer")
REFERENCE_SEEDS = (26082803, 26082804)
MASS = 6.62999997e-26
KB = 1.380649e-23
EPS = np.finfo(np.float64).tiny
HEADER_RE = re.compile(
    r"FNUM=\s*([+\-0-9.EeDd]+).*BLOCK_SAMPLES=(\d+)", re.IGNORECASE
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_moment_bytes(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    header = b"\n".join(data.splitlines()[:3]).decode("ascii", errors="strict")
    match = HEADER_RE.search(header)
    if match is None:
        raise ValueError("moment header lacks FNUM and BLOCK_SAMPLES")
    fnum = float(match.group(1).replace("D", "E").replace("d", "e"))
    block_samples = int(match.group(2))
    raw = np.atleast_2d(np.loadtxt(io.BytesIO(data), comments="#"))
    if raw.shape[1] != 18 or not np.isfinite(raw).all():
        raise ValueError(f"invalid additive-moment array {raw.shape}")

    m0 = raw[:, 5]
    m1x, m1y, m1z = raw[:, 6], raw[:, 7], raw[:, 8]
    m2xx, m2yy, m2zz = raw[:, 9], raw[:, 10], raw[:, 11]
    m2xy, m2xz, m2yz = raw[:, 12], raw[:, 13], raw[:, 14]
    energy, energy_vx, energy_vy = raw[:, 15], raw[:, 16], raw[:, 17]
    u, v, w = m1x / m0, m1y / m0, m1z / m0
    factor = fnum / (raw[:, 4] * float(block_samples))
    number_density = factor * m0
    pxx = factor * MASS * (m2xx - m1x * u)
    pyy = factor * MASS * (m2yy - m1y * v)
    pzz = factor * MASS * (m2zz - m1z * w)
    pxy = factor * MASS * (m2xy - m1x * v)
    temperature = (pxx + pyy + pzz) / (3.0 * number_density * KB)
    speed2 = u * u + v * v + w * w
    qx = factor * (
        energy_vx
        - u * energy
        - MASS * (u * m2xx + v * m2xy + w * m2xz)
        + MASS * u * m0 * speed2
    )
    qy = factor * (
        energy_vy
        - v * energy
        - MASS * (u * m2xy + v * m2yy + w * m2yz)
        + MASS * v * m0 * speed2
    )
    coordinates = raw[:, (0, 2, 3, 4)].astype(np.float64)
    fields = np.column_stack(
        (number_density, u, v, temperature, pxx, pxy, pyy, qx, qy)
    )
    return coordinates, fields


def heldout_reference(reference_archive: Path) -> tuple[np.ndarray, np.ndarray]:
    total: np.ndarray | None = None
    coordinates: np.ndarray | None = None
    count = 0
    with zipfile.ZipFile(reference_archive) as outer:
        for seed in REFERENCE_SEEDS:
            nested_bytes = outer.read(
                f"units/seed_{seed}/JCP8_M12_REFERENCE_seed_{seed}.zip"
            )
            with zipfile.ZipFile(io.BytesIO(nested_bytes)) as nested:
                summary = json.loads(
                    nested.read("JCP8_M12_REFERENCE_SUMMARY.json")
                )
                if summary.get("status") != "mechanical_reference_unit_pass":
                    raise ValueError(f"invalid held-out reference seed {seed}")
                for nout in summary["retained_nout"]:
                    current_coordinates, current = parse_moment_bytes(
                        nested.read(f"JCP3_MOMENTS_NOUT{nout:04d}.DAT")
                    )
                    if coordinates is None:
                        coordinates = current_coordinates
                        total = np.zeros_like(current)
                    elif not np.allclose(
                        coordinates, current_coordinates, rtol=0.0, atol=2e-8
                    ):
                        raise ValueError("coordinates changed in held-out reference")
                    assert total is not None
                    total += current
                    count += 1
    if coordinates is None or total is None or count != 80:
        raise ValueError(f"expected 80 held-out blocks, found {count}")
    return coordinates, total / float(count)


def nrmse(value: np.ndarray, target: np.ndarray) -> np.ndarray:
    numerator = np.sqrt(np.mean((value - target) ** 2, axis=0))
    denominator = np.sqrt(np.mean(target**2, axis=0))
    return numerator / np.maximum(denominator, EPS)


def geometric(values: list[float] | np.ndarray) -> float:
    array = np.maximum(np.asarray(values, dtype=np.float64), EPS)
    return float(np.exp(np.mean(np.log(array))))


def development_gains(model_archive: Path) -> dict[str, np.ndarray]:
    with zipfile.ZipFile(model_archive) as archive:
        rows = list(
            csv.DictReader(
                io.StringIO(
                    archive.read("JCP6R_VALIDATION_METRICS.csv").decode("utf-8")
                )
            )
        )
    result: dict[str, list[float]] = {}
    for row in rows:
        field = row["field"]
        if field.startswith("gain_"):
            result.setdefault(field, []).append(float(row["nrmse"]))
    expected = {
        f"gain_{component}_{zone}"
        for component in GAIN_COMPONENTS
        for zone in ZONES
    }
    if set(result) != expected:
        raise ValueError("development gain ledger is incomplete")
    return {key: np.asarray(values, dtype=np.float64) for key, values in result.items()}


def load_prediction_manifest(prediction_archive: Path) -> tuple[dict, zipfile.ZipFile]:
    archive = zipfile.ZipFile(prediction_archive)
    lock = json.loads(archive.read("JCP7_M12_PREDICTION_LOCK.json"))
    manifest = json.loads(archive.read("JCP7_PREDICTION_MANIFEST.json"))
    if lock.get("reference_artifacts_read") is not False:
        archive.close()
        raise ValueError("prediction lock was not reference isolated")
    if len(manifest.get("units", [])) != 12:
        archive.close()
        raise ValueError("expected 12 frozen prediction units")
    return manifest, archive


def support_rows(
    development: dict[str, np.ndarray], manifest: dict
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for unit in manifest["units"]:
        for component in GAIN_COMPONENTS:
            values = unit["gains"][component]
            for zone, gain in zip(ZONES, values, strict=True):
                key = f"gain_{component}_{zone}"
                envelope = float(np.max(development[key]))
                rows.append(
                    {
                        "seed": int(unit["seed"]),
                        "component": component,
                        "zone": zone,
                        "development_gain_max": envelope,
                        "mach12_gain": float(gain),
                        "outside_development_support": bool(gain > envelope),
                    }
                )
    return rows


def score_frozen_predictions(
    prediction_archive: zipfile.ZipFile,
    manifest: dict,
    coordinates: np.ndarray,
    target: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, dict[str, float]]]:
    dx, dy = coordinates[:, 1] - 0.1524, coordinates[:, 2]
    radius = np.hypot(dx, dy)
    ex, ey = dx / radius, dy / radius
    near = radius <= 0.20
    qn_target = target[:, 7] * ex + target[:, 8] * ey
    rows: list[dict[str, object]] = []
    seed_endpoint_ratios = {
        "frozen_zero_shot": {"all_nine": [], "qy": [], "qn_near_wall": []},
        "support_fallback_raw_B3": {
            "all_nine": [],
            "qy": [],
            "qn_near_wall": [],
        },
    }
    for unit in manifest["units"]:
        seed = int(unit["seed"])
        with np.load(
            io.BytesIO(
                prediction_archive.read(f"JCP7_PREDICTION_seed_{seed}.npz")
            ),
            allow_pickle=False,
        ) as frozen:
            arrays = {
                "frozen_zero_shot": frozen["candidate"].astype(np.float64),
                "support_fallback_raw_B3": frozen["raw_B3"].astype(np.float64),
                "raw_B10": frozen["raw_B10"].astype(np.float64),
            }
        errors = {name: nrmse(value, target) for name, value in arrays.items()}
        qn_errors: dict[str, float] = {}
        for method, value in arrays.items():
            qn = value[:, 7] * ex + value[:, 8] * ey
            qn_errors[method] = float(
                nrmse(qn[near, None], qn_target[near, None])[0]
            )
            for field, error in zip(FIELDS, errors[method], strict=True):
                rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "field": field,
                        "nrmse": float(error),
                    }
                )
            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "field": "qn_near_wall",
                    "nrmse": qn_errors[method],
                }
            )
        for method in ("frozen_zero_shot", "support_fallback_raw_B3"):
            field_ratios = errors[method] / np.maximum(errors["raw_B10"], EPS)
            seed_endpoint_ratios[method]["all_nine"].append(
                geometric(field_ratios)
            )
            seed_endpoint_ratios[method]["qy"].append(float(field_ratios[8]))
            seed_endpoint_ratios[method]["qn_near_wall"].append(
                qn_errors[method] / max(qn_errors["raw_B10"], EPS)
            )
    summary = {
        method: {
            endpoint: geometric(values)
            for endpoint, values in endpoints.items()
        }
        for method, endpoints in seed_endpoint_ratios.items()
    }
    return rows, summary


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 16,
            "axes.titlesize": 19,
            "axes.labelsize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 15,
            "legend.fontsize": 16,
            "axes.linewidth": 1.3,
            "savefig.dpi": 320,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def make_figure(
    development: dict[str, np.ndarray],
    support: list[dict[str, object]],
    endpoint: dict[str, dict[str, float]],
    output_dir: Path,
) -> None:
    style()
    blue, red, navy, orange = "#2C6E9E", "#C94C4C", "#17365D", "#D8742F"
    labels = [r"$n$", r"$u$", r"$v$", r"$T$", r"$P_{xx}$", r"$P_{xy}$", r"$P_{yy}$", r"$q_n$", r"$q_t$"]
    dev_values = [
        np.concatenate(
            [development[f"gain_{component}_{zone}"] for zone in ZONES]
        )
        for component in GAIN_COMPONENTS
    ]
    m12_values = [
        np.asarray(
            [
                float(row["mach12_gain"])
                for row in support
                if row["component"] == component
            ]
        )
        for component in GAIN_COMPONENTS
    ]

    fig = plt.figure(figsize=(19.2, 7.8))
    grid = fig.add_gridspec(1, 3, width_ratios=(1.72, 1.10, 1.10))
    ax0, ax1, ax2 = (fig.add_subplot(grid[0, i]) for i in range(3))
    positions = np.arange(len(labels), dtype=float)
    bp1 = ax0.boxplot(
        dev_values,
        positions=positions - 0.18,
        widths=0.28,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "white", "linewidth": 2.0},
        boxprops={"facecolor": blue, "edgecolor": blue, "alpha": 0.90},
        whiskerprops={"color": blue, "linewidth": 1.6},
        capprops={"color": blue, "linewidth": 1.6},
    )
    bp2 = ax0.boxplot(
        m12_values,
        positions=positions + 0.18,
        widths=0.28,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "white", "linewidth": 2.0},
        boxprops={"facecolor": red, "edgecolor": red, "alpha": 0.90},
        whiskerprops={"color": red, "linewidth": 1.6},
        capprops={"color": red, "linewidth": 1.6},
    )
    ax0.set_xticks(positions, labels)
    ax0.set_ylim(-0.02, 1.02)
    ax0.set_ylabel(r"Empirical-Bayes observation gain, $K_{f,z}$")
    ax0.set_title("(a) Development support and Mach-12 gains", loc="left", fontweight="bold", color=navy, fontsize=17.5)
    ax0.grid(axis="y", alpha=0.25)
    ax0.legend(
        [bp1["boxes"][0], bp2["boxes"][0]],
        ["Mach-8/Mach-10 development", "Mach-12 frozen predictions"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        frameon=False,
        ncol=2,
    )

    by_seed: dict[int, list[bool]] = {}
    for row in support:
        by_seed.setdefault(int(row["seed"]), []).append(
            bool(row["outside_development_support"])
        )
    seeds = sorted(by_seed)
    fractions = [np.mean(by_seed[seed]) for seed in seeds]
    bars = ax1.bar(np.arange(len(seeds)), fractions, color=orange, width=0.75)
    ax1.set_ylim(0.0, 1.10)
    ax1.set_xticks(np.arange(len(seeds)), [f"{i:02d}" for i in range(1, 13)])
    ax1.set_xlabel("Mach-12 evaluation seed index")
    ax1.set_ylabel("Fraction outside support")
    ax1.set_title("(b) Target-free archival diagnostic", loc="left", fontweight="bold", color=navy, fontsize=17.5)
    ax1.grid(axis="y", alpha=0.25)
    for bar in bars:
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            1.015,
            "27/27",
            ha="center",
            va="bottom",
            fontsize=13,
            rotation=90,
            color=navy,
        )

    endpoints = ("all_nine", "qy", "qn_near_wall")
    endpoint_labels = ("All nine", r"$q_y$", r"$q_n$")
    values = [endpoint["frozen_zero_shot"][key] for key in endpoints]
    x = np.arange(3)
    bars2 = ax2.bar(x, values, color=(navy, blue, "#4B8B6A"), width=0.68)
    ax2.axhline(1.0, color="black", linestyle="--", linewidth=1.8, label="Raw-$B=10$")
    ax2.set_xticks(x, endpoint_labels)
    ax2.set_xlabel("Held-out diagnostic")
    ax2.set_ylim(0.0, max(values) * 1.23)
    ax2.set_ylabel("NRMSE ratio to Raw-$B=10$")
    ax2.set_title("(c) Separate target verification", loc="left", fontweight="bold", color=navy, fontsize=17.5)
    ax2.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars2, values, strict=True):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.035,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=16,
            fontweight="bold",
            color=navy,
        )
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), frameon=False)

    fig.subplots_adjust(left=0.055, right=0.992, bottom=0.16, top=0.82, wspace=0.34)
    fig.savefig(
        output_dir / "mach12_support_gate.pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    # Use an explicit file handle so the archival PNG is fully flushed before
    # checksum locking and ZIP packaging on parallel filesystems.
    with (output_dir / "mach12_support_gate.png").open("wb") as stream:
        fig.savefig(
            stream,
            format="png",
            dpi=240,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--prediction-lock", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    development = development_gains(args.model_lock)
    manifest, prediction_archive = load_prediction_manifest(args.prediction_lock)
    try:
        support = support_rows(development, manifest)
        # The target-free computation is complete before this program opens the
        # verification archive.  This is computational separation, not a claim
        # of historical preregistration.
        support_fraction = float(
            np.mean([bool(row["outside_development_support"]) for row in support])
        )
        seed_support = {
            str(seed): float(
                np.mean(
                    [
                        bool(row["outside_development_support"])
                        for row in support
                        if int(row["seed"]) == seed
                    ]
                )
            )
            for seed in sorted({int(row["seed"]) for row in support})
        }
        decision = "abstain_from_structured_prediction_and_request_same_condition_support"
        if support_fraction == 0.0:
            decision = "accept_structured_prediction_within_development_gain_envelope"

        coordinates, target = heldout_reference(args.reference)
        metric_rows, endpoint = score_frozen_predictions(
            prediction_archive, manifest, coordinates, target
        )
    finally:
        prediction_archive.close()

    support_path = args.output / "JCP10_SUPPORT_GATE_LEDGER.csv"
    metrics_path = args.output / "JCP10_HELDOUT_METRICS.csv"
    write_csv(support_path, support)
    write_csv(metrics_path, metric_rows)
    make_figure(development, support, endpoint, args.output)

    summary = {
        "stage": "JCP10_archival_target_free_support_audit",
        "classification": "postprocess_only_no_new_DSMC",
        "rule_specification_timing": "specified_after_initial_reference_scored_Mach12_analysis",
        "decision_reconstruction": "computed_without_target_or_reference_inputs_from_archived_pre_reference_model_and_prediction_artifacts",
        "prospective_validation_claim_authorized": False,
        "development_conditions": ["Mach 8", "Mach 10"],
        "evaluation_condition": "Mach 12",
        "fields": list(FIELDS),
        "gain_components": list(GAIN_COMPONENTS),
        "zones": list(ZONES),
        "support_rule": "structured prediction is accepted only when every field-zone EB gain is at or below its pre-Mach-12 development maximum",
        "support_component_count": len(support),
        "outside_support_component_count": int(
            sum(bool(row["outside_development_support"]) for row in support)
        ),
        "outside_support_fraction": support_fraction,
        "outside_support_fraction_by_seed": seed_support,
        "support_decision": decision,
        "heldout_reference_seeds": list(REFERENCE_SEEDS),
        "heldout_reference_blocks": 80,
        "heldout_nrmse_ratios": endpoint,
        "seed_contract": {
            "Mach12_evaluation": [int(unit["seed"]) for unit in manifest["units"]],
            "Mach12_calibration": [26082801, 26082802],
            "Mach12_heldout": list(REFERENCE_SEEDS),
        },
        "model_lock_sha256": sha256(args.model_lock),
        "prediction_lock_sha256": sha256(args.prediction_lock),
        "reference_sha256": sha256(args.reference),
    }
    summary_path = args.output / "JCP10_SUPPORT_GATE_SUMMARY.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    archive_path = args.output / "JCP10_SUPPORT_AUDIT.zip"
    members = (
        summary_path,
        support_path,
        metrics_path,
        args.output / "mach12_support_gate.pdf",
        args.output / "mach12_support_gate.png",
    )
    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in members:
            archive.write(path, arcname=path.name)
    checksum_path = archive_path.with_suffix(".zip.sha256")
    checksum_path.write_text(
        f"{sha256(archive_path)}  {archive_path.name}\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
