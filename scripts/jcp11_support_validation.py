#!/usr/bin/env python3
"""JCP11 support specificity and prospective Mach-12 validation.

The support rule is calibrated only from the pre-existing Mach-8/Mach-10
development gain ledger.  Six previously held-out Mach-10 observation seeds
measure specificity without opening their reference partners.  Four fresh
Mach-12 observation trajectories are then generated after the rule lock; their
support decisions and structured predictions are frozen before the independent
archived Mach-12 reference is opened by the final score command.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable
import zipfile

import numpy as np


FIELDS = ("n", "u", "v", "T", "Pxx", "Pxy", "Pyy", "qx", "qy")
GAIN_COMPONENTS = ("n", "u", "v", "T", "Pxx", "Pxy", "Pyy", "qn", "qt")
ZONES = ("near_wall", "wake", "outer")
GAIN_KEYS = tuple(
    f"gain_{component}_{zone}"
    for component in GAIN_COMPONENTS
    for zone in ZONES
)
M10_PAIRS = tuple(
    (f"pair_{index:02d}", 171699 + 2 * index, 171700 + 2 * index)
    for index in range(1, 7)
)
M10_B3_NOUT = (100, 108, 116)
M12_EVALUATION_SEEDS = (26082901, 26082902, 26082903, 26082904)
M12_HELDOUT_REFERENCE_SEEDS = (26082803, 26082804)
MASS = 6.62999997e-26
KB = 1.380649e-23
EPS = np.finfo(np.float64).tiny
NOUT_RE = re.compile(r"NOUT(\d+)\.DAT$")
HEADER_RE = re.compile(
    r"FNUM=\s*([+\-0-9.EeDd]+).*BLOCK_SAMPLES=(\d+)", re.IGNORECASE
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV {path}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_moment_bytes(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    header = b"\n".join(data.splitlines()[:3]).decode("ascii", errors="strict")
    match = HEADER_RE.search(header)
    if match is None:
        raise ValueError("moment header lacks FNUM and BLOCK_SAMPLES")
    fnum = float(match.group(1).replace("D", "E").replace("d", "e"))
    block_samples = int(match.group(2))
    raw = np.atleast_2d(np.loadtxt(io.BytesIO(data), comments="#"))
    if raw.shape[1] != 18 or raw.shape[0] < 1000 or not np.isfinite(raw).all():
        raise ValueError(f"invalid additive-moment array {raw.shape}")
    m0 = raw[:, 5]
    if np.any(m0 <= 0.0):
        raise ValueError("non-positive additive mass")
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


def model_from_archive(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any], bytes]:
    with zipfile.ZipFile(path) as archive:
        lock = json.loads(archive.read("JCP6R_MODEL_LOCK.json"))
        model_bytes = archive.read("JCP6R_MODEL.npz")
    if lock.get("status") != "repair_model_lock_complete_gate_pass":
        raise ValueError("JCP6R development gate did not pass")
    if lock.get("model_sha256") != sha256_bytes(model_bytes):
        raise ValueError("JCP6R internal model checksum mismatch")
    with np.load(io.BytesIO(model_bytes), allow_pickle=False) as frozen:
        model = {name: frozen[name].copy() for name in frozen.files}
    required = {
        "coordinates", "zones", "prior_m8", "prior_m10",
        "block_variance_m8", "block_variance_m10",
    }
    if not required.issubset(model):
        raise ValueError("JCP6R model arrays are incomplete")
    for name, value in model.items():
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError(f"non-finite model array {name}")
    return model, lock, model_bytes


def development_samples(model_archive: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(model_archive) as archive:
        rows = list(
            csv.DictReader(
                io.StringIO(
                    archive.read("JCP6R_VALIDATION_METRICS.csv").decode("utf-8")
                )
            )
        )
    grouped: dict[tuple[int, int, int, float], dict[str, float]] = {}
    for row in rows:
        field = row["field"]
        if not field.startswith("gain_"):
            continue
        identity = (
            int(row["unit"]), int(row["draw"]), int(row["seed"]), float(row["mach"])
        )
        grouped.setdefault(identity, {})[field] = float(row["nrmse"])
    samples = []
    for (unit, draw, seed, mach), gains in sorted(grouped.items()):
        if set(gains) != set(GAIN_KEYS):
            raise ValueError(f"incomplete development gain record unit={unit} draw={draw}")
        samples.append(
            {"unit": unit, "draw": draw, "seed": seed, "mach": mach, "gains": gains}
        )
    if len(samples) != 32 or len({sample["unit"] for sample in samples}) != 8:
        raise ValueError(f"expected 32 draws from eight development units, found {len(samples)}")
    return samples


def calibrated_rule(model_archive: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    samples = development_samples(model_archive)
    units = sorted({int(sample["unit"]) for sample in samples})
    full_envelope = {
        key: max(float(sample["gains"][key]) for sample in samples)
        for key in GAIN_KEYS
    }
    loo_rows: list[dict[str, Any]] = []
    for unit in units:
        held = [sample for sample in samples if int(sample["unit"]) == unit]
        training = [sample for sample in samples if int(sample["unit"]) != unit]
        envelope = {
            key: max(float(sample["gains"][key]) for sample in training)
            for key in GAIN_KEYS
        }
        flags = [
            float(sample["gains"][key]) > envelope[key]
            for sample in held
            for key in GAIN_KEYS
        ]
        loo_rows.append(
            {
                "heldout_unit": unit,
                "mach": held[0]["mach"],
                "seed": held[0]["seed"],
                "heldout_draws": len(held),
                "component_tests": len(flags),
                "outside_count": int(sum(flags)),
                "outside_fraction": float(np.mean(flags)),
            }
        )
    # With eight independent development units and alpha=0.10, the finite-sample
    # upper conformal rank is the maximum leave-one-unit-out score.
    threshold = max(float(row["outside_fraction"]) for row in loo_rows)
    envelope_rows = [
        {"gain_key": key, "development_max": full_envelope[key]}
        for key in GAIN_KEYS
    ]
    rule = {
        "stage": "JCP11_support_rule_lock",
        "status": "support_rule_frozen_before_fresh_M12",
        "classification": "development_only_calibration",
        "created_utc": utc_now(),
        "development_conditions": ["Mach 8", "Mach 10"],
        "development_units": 8,
        "draws_per_unit": 4,
        "gain_component_count": len(GAIN_KEYS),
        "score": "fraction of 27 field-zone EB gains exceeding the full-development componentwise maximum",
        "calibration": "leave-one-development-seed-out familywise score; alpha=0.10 finite-sample upper rank",
        "primary_threshold_outside_fraction": threshold,
        "acceptance_rule": "accept structured prediction when outside_fraction <= locked threshold; otherwise abstain",
        "development_false_abstention_count_at_locked_threshold": int(
            sum(float(row["outside_fraction"]) > threshold for row in loo_rows)
        ),
        "model_lock_archive_sha256": sha256(model_archive),
        "gain_envelope": full_envelope,
        "historical_Mach12_or_reference_artifacts_read": False,
    }
    return rule, envelope_rows, loo_rows


def flatten_gains(gains: dict[str, list[float]]) -> dict[str, float]:
    result = {
        f"gain_{component}_{zone}": float(value)
        for component, values in gains.items()
        for zone, value in zip(ZONES, values, strict=True)
    }
    if set(result) != set(GAIN_KEYS):
        raise ValueError("computed gain vector is incomplete")
    return result


def support_score(gains: dict[str, float], rule: dict[str, Any]) -> dict[str, Any]:
    envelope = rule["gain_envelope"]
    flags = {key: bool(float(gains[key]) > float(envelope[key])) for key in GAIN_KEYS}
    fraction = float(np.mean(list(flags.values())))
    ratios = []
    for key in GAIN_KEYS:
        denominator = float(envelope[key])
        numerator = float(gains[key])
        ratios.append(0.0 if numerator == 0.0 and denominator == 0.0 else numerator / max(denominator, EPS))
    threshold = float(rule["primary_threshold_outside_fraction"])
    accept = bool(fraction <= threshold + 1.0e-15)
    return {
        "outside_count": int(sum(flags.values())),
        "component_count": len(flags),
        "outside_fraction": fraction,
        "maximum_gain_to_envelope_ratio": float(max(ratios)),
        "decision": "accept_structured_prediction" if accept else "abstain_outside_support",
        "accepted": accept,
        "outside_flags": flags,
    }


def zones_for(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dx, dy = coords[:, 1] - 0.1524, coords[:, 2]
    radius = np.hypot(dx, dy)
    if np.any(radius <= 0.0):
        raise ValueError("cylinder-relative radius contains zero")
    zones = np.full(len(coords), 2, dtype=np.int8)
    zones[radius <= 0.20] = 0
    zones[(radius > 0.20) & (dx >= 0.0)] = 1
    return zones, dx / radius, dy / radius, radius


def field_fuse(
    observation: np.ndarray,
    prior: np.ndarray,
    block_variance: np.ndarray,
    zones: np.ndarray,
    budget: int,
) -> tuple[np.ndarray, list[float]]:
    output = prior.copy()
    gains: list[float] = []
    for zone in range(len(ZONES)):
        mask = zones == zone
        residual = observation[mask] - prior[mask]
        residual_power = float(np.mean(residual * residual))
        noise_power = float(np.mean(block_variance[mask])) / float(budget)
        gain = 0.0 if residual_power <= EPS else float(
            np.clip((residual_power - noise_power) / residual_power, 0.0, 1.0)
        )
        output[mask] = prior[mask] + gain * residual
        gains.append(gain)
    return output, gains


def fuse_candidate(
    observation: np.ndarray,
    prior: np.ndarray,
    block_variance: np.ndarray,
    zones: np.ndarray,
    ex: np.ndarray,
    ey: np.ndarray,
    budget: int = 3,
) -> tuple[np.ndarray, dict[str, list[float]]]:
    estimate = np.empty_like(observation, dtype=np.float64)
    gains: dict[str, list[float]] = {}
    for field in range(7):
        estimate[:, field], gains[FIELDS[field]] = field_fuse(
            observation[:, field], prior[:, field], block_variance[:, field], zones, budget
        )
    obs_qn = observation[:, 7] * ex + observation[:, 8] * ey
    obs_qt = -observation[:, 7] * ey + observation[:, 8] * ex
    prior_qn = prior[:, 7] * ex + prior[:, 8] * ey
    prior_qt = -prior[:, 7] * ey + prior[:, 8] * ex
    var_qn = block_variance[:, 7] * ex**2 + block_variance[:, 8] * ey**2
    var_qt = block_variance[:, 7] * ey**2 + block_variance[:, 8] * ex**2
    qn, gains["qn"] = field_fuse(obs_qn, prior_qn, var_qn, zones, budget)
    qt, gains["qt"] = field_fuse(obs_qt, prior_qt, var_qt, zones, budget)
    estimate[:, 7] = qn * ex - qt * ey
    estimate[:, 8] = qn * ey + qt * ex
    return estimate, gains


def average_moment_paths(paths: Iterable[Path]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    coordinates = None
    blocks = []
    ledger = []
    for path in paths:
        data = Path(path).read_bytes()
        current_coordinates, fields = parse_moment_bytes(data)
        if coordinates is None:
            coordinates = current_coordinates
        elif not np.allclose(coordinates, current_coordinates, rtol=0.0, atol=2.0e-8):
            raise ValueError(f"native mesh changed in {path}")
        blocks.append(fields)
        ledger.append({"path": str(path), "sha256": sha256(Path(path))})
    if coordinates is None:
        raise ValueError("no moment files supplied")
    return coordinates, np.mean(np.asarray(blocks), axis=0), ledger


def command_freeze(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "JCP11_SUPPORT_RULE_LOCK.json").exists():
        raise FileExistsError("refusing to overwrite an existing JCP11 rule lock")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("status") != "locked_before_JCP11_fresh_M12":
        raise ValueError("JCP11 protocol is not locked")
    rule, envelope_rows, loo_rows = calibrated_rule(args.model_lock)
    rule["protocol_sha256"] = sha256(args.protocol)
    rule_path = args.output / "JCP11_SUPPORT_RULE_LOCK.json"
    envelope_path = args.output / "JCP11_GAIN_ENVELOPE.csv"
    loo_path = args.output / "JCP11_DEVELOPMENT_LOO_CALIBRATION.csv"
    write_json(rule_path, rule)
    write_csv(envelope_path, envelope_rows)
    write_csv(loo_path, loo_rows)
    archive_path = args.output / "JCP11_SUPPORT_RULE_LOCK.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in (args.protocol, rule_path, envelope_path, loo_path):
            archive.write(path, arcname=path.name)
    archive_path.with_suffix(".zip.sha256").write_text(
        f"{sha256(archive_path)}  {archive_path.name}\n", encoding="utf-8"
    )
    print(json.dumps(rule, indent=2, sort_keys=True))


def command_specificity(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    rule = json.loads(args.rule.read_text(encoding="utf-8"))
    model, _, _ = model_from_archive(args.model_lock)
    coordinates = model["coordinates"].astype(np.float64)
    prior = model["prior_m10"].astype(np.float64)
    variance = model["block_variance_m10"].astype(np.float64)
    zones, ex, ey, _ = zones_for(coordinates)
    rows, component_rows, sources = [], [], []
    for pair_id, observation_seed, reference_seed in M10_PAIRS:
        case = args.campaign / "cases" / f"{pair_id}_observation"
        metadata_path = case / "CASE_METADATA.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if int(metadata.get("seed", -1)) != observation_seed:
                raise ValueError(f"Mach-10 seed mismatch in {metadata_path}")
        paths = [
            case / "results" / "moments" / f"MV11_MOMENTS_NOUT{nout:04d}.DAT"
            for nout in M10_B3_NOUT
        ]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        coords_seen, raw_b3, ledger = average_moment_paths(paths)
        if not np.allclose(coordinates, coords_seen, rtol=0.0, atol=2.0e-8):
            raise ValueError(f"Mach-10 mesh differs from model for {pair_id}")
        _, gain_lists = fuse_candidate(raw_b3, prior, variance, zones, ex, ey)
        gains = flatten_gains(gain_lists)
        score = support_score(gains, rule)
        rows.append(
            {
                "pair_id": pair_id,
                "observation_seed": observation_seed,
                "reference_seed_not_read": reference_seed,
                "outside_count": score["outside_count"],
                "component_count": score["component_count"],
                "outside_fraction": score["outside_fraction"],
                "locked_threshold": rule["primary_threshold_outside_fraction"],
                "maximum_gain_to_envelope_ratio": score["maximum_gain_to_envelope_ratio"],
                "decision": score["decision"],
            }
        )
        for key in GAIN_KEYS:
            component_rows.append(
                {
                    "pair_id": pair_id,
                    "observation_seed": observation_seed,
                    "gain_key": key,
                    "gain": gains[key],
                    "development_max": rule["gain_envelope"][key],
                    "outside": score["outside_flags"][key],
                }
            )
        sources.extend(ledger)
    accepted = sum(row["decision"] == "accept_structured_prediction" for row in rows)
    summary = {
        "stage": "JCP11_Mach10_specificity",
        "status": "specificity_complete_without_reference_access",
        "classification": "heldout_postprocess_specificity",
        "pair_count": len(rows),
        "accepted_count": accepted,
        "false_abstention_count": len(rows) - accepted,
        "predeclared_specificity_gate": "at least five of six supported Mach-10 observations accepted",
        "specificity_gate_pass": bool(accepted >= 5),
        "reference_partner_files_read": False,
        "rule_lock_sha256": sha256(args.rule),
        "model_lock_archive_sha256": sha256(args.model_lock),
        "source_moment_files": sources,
    }
    ledger_path = args.output / "JCP11_M10_SPECIFICITY_LEDGER.csv"
    component_path = args.output / "JCP11_M10_SPECIFICITY_COMPONENTS.csv"
    summary_path = args.output / "JCP11_M10_SPECIFICITY_SUMMARY.json"
    write_csv(ledger_path, rows)
    write_csv(component_path, component_rows)
    write_json(summary_path, summary)
    archive_path = args.output / "JCP11_M10_SPECIFICITY.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in (ledger_path, component_path, summary_path):
            archive.write(path, arcname=path.name)
    archive_path.with_suffix(".zip.sha256").write_text(
        f"{sha256(archive_path)}  {archive_path.name}\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def indexed(case: Path, prefix: str) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in case.glob(f"{prefix}NOUT*.DAT"):
        match = NOUT_RE.search(path.name)
        if match:
            result[int(match.group(1))] = path
    return result


def block_diagnostic(moment_path: Path, wall_path: Path) -> dict[str, Any]:
    raw = np.atleast_2d(np.loadtxt(moment_path, comments="#"))
    wall = np.atleast_2d(np.loadtxt(wall_path, skiprows=2))
    if raw.shape[0] < 1000 or raw.shape[1] != 18 or not np.isfinite(raw).all():
        raise ValueError(f"invalid moment output {moment_path}")
    if wall.shape[0] < 20 or wall.shape[1] < 16 or not np.isfinite(wall).all():
        raise ValueError(f"invalid wall output {wall_path}")
    m0 = raw[:, 5]
    total = float(m0.sum())
    return {
        "nout": int(NOUT_RE.search(moment_path.name).group(1)),
        "populated_cells": int(raw.shape[0]),
        "wall_elements": int(wall.shape[0]),
        "global_u_m_per_s": float(raw[:, 6].sum() / total),
        "energy_per_weight_J": float(raw[:, 15].sum() / total),
        "wall_heat_flux_mean_W_m2": float(wall[:, 15].mean()),
        "wall_heat_flux_max_abs_W_m2": float(np.max(np.abs(wall[:, 15]))),
    }


def drift_z(values: np.ndarray) -> float:
    first, last = values[:7], values[7:]
    se = math.sqrt(float(np.var(first, ddof=1) / 7.0 + np.var(last, ddof=1) / 7.0))
    difference = abs(float(np.mean(first) - np.mean(last)))
    return (0.0 if difference == 0.0 else 1.0e300) if se == 0.0 else difference / se


def command_verify_eval(args: argparse.Namespace) -> None:
    patch = json.loads(args.patch_report.read_text(encoding="utf-8"))
    if int(patch.get("changed_token_count", -1)) != 1 or not math.isclose(
        float(patch.get("new_speed_m_per_s", 0.0)), 3160.92, abs_tol=1.0e-8
    ):
        raise ValueError("invalid Mach-12 source patch")
    moments = indexed(args.case, "JCP3_MOMENTS_")
    walls = indexed(args.case, "JCP3_WALL_")
    paired = sorted(set(moments) & set(walls))
    if len(paired) < 40:
        raise ValueError(f"only {len(paired)} complete paired blocks")
    retained = paired[-14:]
    if any(b != a + 1 for a, b in zip(retained[:-1], retained[1:], strict=True)):
        raise ValueError("retained evaluation blocks are not consecutive")
    diagnostics = [block_diagnostic(moments[nout], walls[nout]) for nout in retained]
    if any(float(row["wall_heat_flux_max_abs_W_m2"]) <= 0.0 for row in diagnostics):
        raise ValueError("zero direct wall heat-flux tally")
    drift = {
        name: drift_z(np.asarray([float(row[name]) for row in diagnostics]))
        for name in ("global_u_m_per_s", "energy_per_weight_J", "wall_heat_flux_mean_W_m2")
    }
    summary = {
        "stage": "JCP11_fresh_M12_evaluation",
        "status": "mechanically_complete",
        "classification": "fresh_observation_generated_after_support_rule_lock",
        "seed": args.seed,
        "paired_completed_blocks": len(paired),
        "retained_nout": retained,
        "block_roles": {"B3": retained[:3], "guard": retained[3], "Raw_B10": retained[4:]},
        "stationarity_sensitivity_not_a_selection_gate": {
            "first7_vs_last7_drift_z": drift,
            "all_below_3p5": bool(max(drift.values()) <= 3.5),
        },
        "patch_report_sha256": sha256(args.patch_report),
        "block_diagnostics": diagnostics,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.summary, summary)
    members = [args.patch_report, args.summary]
    for optional in ("RNG_SEED_USED.txt", "DS2VD.TXT", "JCP11_PRE_RUN_RULE_LOCK.sha256"):
        path = args.case / optional
        if path.is_file():
            members.append(path)
    members.extend(moments[nout] for nout in retained)
    members.extend(walls[nout] for nout in retained)
    with zipfile.ZipFile(args.archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in members:
            archive.write(path, arcname=path.name)
    args.archive.with_suffix(".zip.sha256").write_text(
        f"{sha256(args.archive)}  {args.archive.name}\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def wall_q(data: bytes) -> np.ndarray:
    table = np.atleast_2d(np.loadtxt(io.BytesIO(data), skiprows=2))
    if table.shape[0] < 20 or table.shape[1] < 16 or not np.isfinite(table).all():
        raise ValueError(f"invalid wall table {table.shape}")
    return table[:, 15].astype(np.float64)


def command_collect_lock(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    rule = json.loads(args.rule.read_text(encoding="utf-8"))
    if rule.get("status") != "support_rule_frozen_before_fresh_M12":
        raise ValueError("invalid JCP11 support-rule lock")
    model, model_lock, _ = model_from_archive(args.model_lock)
    coords = model["coordinates"].astype(np.float64)
    prior = 2.0 * model["prior_m10"].astype(np.float64) - model["prior_m8"].astype(np.float64)
    variance = np.maximum(
        model["block_variance_m8"], model["block_variance_m10"]
    ).astype(np.float64)
    zones, ex, ey, _ = zones_for(coords)
    units, unit_paths = [], []
    for seed in M12_EVALUATION_SEEDS:
        unit = args.work / "units" / f"seed_{seed}"
        summary_path = unit / "JCP11_M12_EVALUATION_SUMMARY.json"
        archive_path = unit / f"JCP11_M12_EVALUATION_seed_{seed}.zip"
        checksum_path = archive_path.with_suffix(".zip.sha256")
        for path in (summary_path, archive_path, checksum_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "mechanically_complete" or int(summary["seed"]) != seed:
            raise ValueError(f"invalid fresh M12 unit {seed}")
        if checksum_path.read_text(encoding="utf-8").split()[0] != sha256(archive_path):
            raise ValueError(f"fresh M12 unit checksum mismatch {seed}")
        units.append({"seed": seed, "archive_sha256": sha256(archive_path), "retained_nout": summary["retained_nout"]})
        unit_paths.append((summary_path, archive_path, checksum_path))
    audit = {
        "stage": "JCP11_fresh_M12_evaluation",
        "status": "four_fresh_units_locked_without_selection",
        "selected_seeds": list(M12_EVALUATION_SEEDS),
        "selection_rule": "all four predeclared seeds; no outcome-dependent replacement",
        "support_rule_sha256": sha256(args.rule),
        "reference_artifacts_read": False,
        "units": units,
    }
    audit_path = args.output / "JCP11_M12_EVALUATION_AUDIT.json"
    write_json(audit_path, audit)
    evaluation_archive = args.output / "JCP11_M12_EVALUATION.zip"
    with zipfile.ZipFile(evaluation_archive, "w", zipfile.ZIP_STORED) as outer:
        outer.write(args.protocol, arcname=args.protocol.name)
        outer.write(audit_path, arcname=audit_path.name)
        for summary_path, archive_path, checksum_path in unit_paths:
            prefix = f"units/{summary_path.parent.name}"
            outer.write(summary_path, arcname=f"{prefix}/{summary_path.name}")
            outer.write(archive_path, arcname=f"{prefix}/{archive_path.name}")
            outer.write(checksum_path, arcname=f"{prefix}/{checksum_path.name}")
    evaluation_archive.with_suffix(".zip.sha256").write_text(
        f"{sha256(evaluation_archive)}  {evaluation_archive.name}\n", encoding="utf-8"
    )

    predictions, manifest, component_rows = [], [], []
    with zipfile.ZipFile(evaluation_archive) as outer:
        for seed in M12_EVALUATION_SEEDS:
            nested_bytes = outer.read(
                f"units/seed_{seed}/JCP11_M12_EVALUATION_seed_{seed}.zip"
            )
            with zipfile.ZipFile(io.BytesIO(nested_bytes)) as nested:
                summary = json.loads(nested.read("JCP11_M12_EVALUATION_SUMMARY.json"))
                nouts = list(summary["retained_nout"])
                blocks, walls, coords_seen = [], [], None
                for nout in nouts:
                    current_coords, fields = parse_moment_bytes(
                        nested.read(f"JCP3_MOMENTS_NOUT{nout:04d}.DAT")
                    )
                    if coords_seen is None:
                        coords_seen = current_coords
                    elif not np.allclose(coords_seen, current_coords, rtol=0.0, atol=2.0e-8):
                        raise ValueError("native mesh changed within fresh M12 unit")
                    blocks.append(fields)
                    walls.append(wall_q(nested.read(f"JCP3_WALL_NOUT{nout:04d}.DAT")))
                if coords_seen is None or not np.allclose(coords, coords_seen, rtol=0.0, atol=2.0e-8):
                    raise ValueError("fresh M12 mesh differs from JCP6R model")
                block_array = np.asarray(blocks)
                wall_array = np.asarray(walls)
                raw_b3 = np.mean(block_array[:3], axis=0)
                raw_b10 = np.mean(block_array[4:], axis=0)
                candidate, gain_lists = fuse_candidate(raw_b3, prior, variance, zones, ex, ey)
                for field in (0, 3, 4, 6):
                    candidate[:, field] = np.maximum(candidate[:, field], 0.05 * raw_b3[:, field])
                gains = flatten_gains(gain_lists)
                score = support_score(gains, rule)
                prediction_path = args.output / f"JCP11_PREDICTION_seed_{seed}.npz"
                np.savez_compressed(
                    prediction_path,
                    candidate=candidate.astype(np.float32),
                    raw_B3=raw_b3.astype(np.float32),
                    raw_B10=raw_b10.astype(np.float32),
                    wall_raw_B3=np.mean(wall_array[:3], axis=0),
                    wall_raw_B10=np.mean(wall_array[4:], axis=0),
                    retained_nout=np.asarray(nouts),
                )
                predictions.append(prediction_path)
                manifest.append(
                    {
                        "seed": seed,
                        "evaluation_unit_sha256": sha256_bytes(nested_bytes),
                        "prediction_sha256": sha256(prediction_path),
                        "B3_nout": nouts[:3],
                        "guard_nout": nouts[3],
                        "Raw_B10_nout": nouts[4:],
                        "gains": gain_lists,
                        "support": {key: value for key, value in score.items() if key != "outside_flags"},
                    }
                )
                for key in GAIN_KEYS:
                    component_rows.append(
                        {
                            "seed": seed,
                            "gain_key": key,
                            "gain": gains[key],
                            "development_max": rule["gain_envelope"][key],
                            "outside": score["outside_flags"][key],
                        }
                    )
    manifest_path = args.output / "JCP11_M12_DECISION_MANIFEST.json"
    component_path = args.output / "JCP11_M12_SUPPORT_COMPONENTS.csv"
    write_json(manifest_path, {"units": manifest})
    write_csv(component_path, component_rows)
    lock = {
        "stage": "JCP11_fresh_M12_prediction_and_support_lock",
        "status": "decision_lock_complete_before_reference_scoring",
        "classification": "prospective_support_decision",
        "created_utc": utc_now(),
        "fresh_seed_count": len(manifest),
        "fresh_seeds": list(M12_EVALUATION_SEEDS),
        "abstained_count": int(sum(not item["support"]["accepted"] for item in manifest)),
        "accepted_count": int(sum(item["support"]["accepted"] for item in manifest)),
        "rule_lock_sha256": sha256(args.rule),
        "model_lock_archive_sha256": sha256(args.model_lock),
        "model_sha256": model_lock["model_sha256"],
        "protocol_sha256": sha256(args.protocol),
        "evaluation_archive_sha256": sha256(evaluation_archive),
        "manifest_sha256": sha256(manifest_path),
        "reference_or_target_artifacts_read": False,
        "next_stage": "open archived independent M12 reference only in dependent score job",
    }
    lock_path = args.output / "JCP11_M12_DECISION_LOCK.json"
    write_json(lock_path, lock)
    decision_archive = args.output / "JCP11_M12_DECISION_LOCK.zip"
    with zipfile.ZipFile(decision_archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in (args.protocol, args.rule, manifest_path, component_path, lock_path, *predictions):
            archive.write(path, arcname=path.name)
    decision_archive.with_suffix(".zip.sha256").write_text(
        f"{sha256(decision_archive)}  {decision_archive.name}\n", encoding="utf-8"
    )
    print(json.dumps(lock, indent=2, sort_keys=True))


def heldout_reference(reference_archive: Path) -> tuple[np.ndarray, np.ndarray]:
    coordinates, total, count = None, None, 0
    with zipfile.ZipFile(reference_archive) as outer:
        for seed in M12_HELDOUT_REFERENCE_SEEDS:
            nested_bytes = outer.read(f"units/seed_{seed}/JCP8_M12_REFERENCE_seed_{seed}.zip")
            with zipfile.ZipFile(io.BytesIO(nested_bytes)) as nested:
                summary = json.loads(nested.read("JCP8_M12_REFERENCE_SUMMARY.json"))
                if summary.get("status") != "mechanical_reference_unit_pass":
                    raise ValueError(f"invalid independent reference seed {seed}")
                for nout in summary["retained_nout"]:
                    current_coords, fields = parse_moment_bytes(
                        nested.read(f"JCP3_MOMENTS_NOUT{nout:04d}.DAT")
                    )
                    if coordinates is None:
                        coordinates = current_coords
                        total = np.zeros_like(fields)
                    elif not np.allclose(coordinates, current_coords, rtol=0.0, atol=2.0e-8):
                        raise ValueError("independent reference mesh changed")
                    total += fields
                    count += 1
    if coordinates is None or total is None or count != 80:
        raise ValueError(f"expected 80 held-out reference blocks, found {count}")
    return coordinates, total / float(count)


def nrmse(value: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((value - target) ** 2, axis=0)) / np.maximum(
        np.sqrt(np.mean(target**2, axis=0)), EPS
    )


def geometric(values: Iterable[float]) -> float:
    array = np.maximum(np.asarray(list(values), dtype=np.float64), EPS)
    return float(np.exp(np.mean(np.log(array))))


def make_figure(
    loo_csv: Path,
    specificity_csv: Path,
    manifest: dict[str, Any],
    metric_summary: dict[str, Any],
    output: Path,
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans", "font.size": 16,
            "axes.titlesize": 18, "axes.labelsize": 17,
            "xtick.labelsize": 14, "ytick.labelsize": 14,
            "legend.fontsize": 14, "axes.linewidth": 1.25,
            "pdf.fonttype": 42, "ps.fonttype": 42,
        }
    )
    with loo_csv.open(encoding="utf-8") as stream:
        loo = list(csv.DictReader(stream))
    with specificity_csv.open(encoding="utf-8") as stream:
        specificity = list(csv.DictReader(stream))
    threshold = float(specificity[0]["locked_threshold"])
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 6.2))
    navy, green, red, grey = "#17365D", "#278253", "#C94C4C", "#8A96A3"
    axes[0].bar(np.arange(8), [float(row["outside_fraction"]) for row in loo], color=grey)
    axes[0].axhline(threshold, color=navy, lw=2.2, ls="--", label="Locked threshold")
    axes[0].set_xticks(np.arange(8), [str(i + 1) for i in range(8)])
    axes[0].set_xlabel("Held-out development seed index")
    axes[0].set_ylabel("Fraction of gains outside envelope")
    axes[0].set_title("(a) Development-only calibration", loc="left", fontweight="bold")
    axes[0].legend(loc="upper left", frameon=False)

    axes[1].bar(np.arange(6), [float(row["outside_fraction"]) for row in specificity], color=green)
    axes[1].axhline(threshold, color=navy, lw=2.2, ls="--")
    axes[1].set_xticks(np.arange(6), [f"{i + 1}" for i in range(6)])
    axes[1].set_xlabel("Held-out Mach-10 observation index")
    axes[1].set_title("(b) Supported-condition specificity", loc="left", fontweight="bold")

    m12 = manifest["units"]
    axes[2].bar(
        np.arange(4), [float(unit["support"]["outside_fraction"]) for unit in m12], color=red
    )
    axes[2].axhline(threshold, color=navy, lw=2.2, ls="--")
    axes[2].set_xticks(np.arange(4), [f"{i + 1}" for i in range(4)])
    axes[2].set_xlabel("Fresh Mach-12 observation index")
    axes[2].set_title("(c) Prospective shift detection", loc="left", fontweight="bold")
    for ax in axes:
        ax.set_ylim(0.0, 1.05)
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle(
        "Frozen empirical-Bayes support monitor: calibration, specificity, and prospective detection",
        fontsize=19, fontweight="bold", color=navy, y=0.99,
    )
    fig.text(
        0.5, 0.015,
        f"Post-lock diagnostic candidate/Raw-B10 all-field NRMSE ratio: {metric_summary['candidate_to_raw_B10']['all_nine']:.3f}",
        ha="center", va="bottom", fontsize=15, color=navy,
    )
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.18, top=0.84, wspace=0.28)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def command_score_pack(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_value = json.loads(args.protocol.read_text(encoding="utf-8"))
    expected_reference = protocol_value["reference_reuse"]["expected_archive_sha256"]
    if sha256(args.reference) != expected_reference:
        raise ValueError("independent JCP8 reference checksum differs from the locked protocol")
    specificity = json.loads(args.specificity_summary.read_text(encoding="utf-8"))
    coordinates, target = heldout_reference(args.reference)
    dx, dy = coordinates[:, 1] - 0.1524, coordinates[:, 2]
    radius = np.hypot(dx, dy)
    ex, ey, near = dx / radius, dy / radius, radius <= 0.20
    qn_target = target[:, 7] * ex + target[:, 8] * ey
    metric_rows: list[dict[str, Any]] = []
    ratios = {
        method: {endpoint: [] for endpoint in ("all_nine", "qy", "qn_near_wall")}
        for method in ("candidate", "raw_B3")
    }
    with zipfile.ZipFile(args.decision_lock) as archive:
        lock = json.loads(archive.read("JCP11_M12_DECISION_LOCK.json"))
        manifest = json.loads(archive.read("JCP11_M12_DECISION_MANIFEST.json"))
        if lock.get("reference_or_target_artifacts_read") is not False:
            raise ValueError("decision lock is not target isolated")
        for unit in manifest["units"]:
            seed = int(unit["seed"])
            with np.load(
                io.BytesIO(archive.read(f"JCP11_PREDICTION_seed_{seed}.npz")),
                allow_pickle=False,
            ) as frozen:
                arrays = {
                    "candidate": frozen["candidate"].astype(np.float64),
                    "raw_B3": frozen["raw_B3"].astype(np.float64),
                    "raw_B10": frozen["raw_B10"].astype(np.float64),
                }
            errors = {name: nrmse(value, target) for name, value in arrays.items()}
            qn_errors = {}
            for method, value in arrays.items():
                qn = value[:, 7] * ex + value[:, 8] * ey
                qn_errors[method] = float(nrmse(qn[near, None], qn_target[near, None])[0])
                for field, error in zip(FIELDS, errors[method], strict=True):
                    metric_rows.append({"seed": seed, "method": method, "field": field, "nrmse": float(error)})
                metric_rows.append({"seed": seed, "method": method, "field": "qn_near_wall", "nrmse": qn_errors[method]})
            for method in ("candidate", "raw_B3"):
                field_ratios = errors[method] / np.maximum(errors["raw_B10"], EPS)
                ratios[method]["all_nine"].append(geometric(field_ratios))
                ratios[method]["qy"].append(float(field_ratios[8]))
                ratios[method]["qn_near_wall"].append(qn_errors[method] / max(qn_errors["raw_B10"], EPS))
    ratio_summary = {
        f"{method}_to_raw_B10": {
            endpoint: geometric(values) for endpoint, values in endpoints.items()
        }
        for method, endpoints in ratios.items()
    }
    abstained = int(sum(not unit["support"]["accepted"] for unit in manifest["units"]))
    primary_pass = bool(specificity["accepted_count"] >= 5 and abstained == 4)
    summary = {
        "stage": "JCP11_support_specificity_and_prospective_validation",
        "status": "complete",
        "classification": "four_fresh_M12_observations_generated_after_rule_lock_then_scored_against_archived_independent_reference",
        "supported_Mach10_specificity": {
            "accepted_count": specificity["accepted_count"],
            "observation_count": specificity["pair_count"],
            "reference_partner_files_read_for_specificity": False,
        },
        "fresh_Mach12_detection": {
            "abstained_count": abstained,
            "observation_count": len(manifest["units"]),
            "all_decisions_locked_before_reference_read": True,
        },
        "predeclared_primary_monitor_gate": "accept at least 5/6 held-out Mach-10 observations and abstain on 4/4 fresh Mach-12 observations",
        "primary_monitor_gate_pass": primary_pass,
        "post_lock_diagnostic_nrmse_ratios": ratio_summary,
        "heldout_reference_seeds": list(M12_HELDOUT_REFERENCE_SEEDS),
        "heldout_reference_blocks": 80,
        "rule_lock_archive_sha256": sha256(args.rule_archive),
        "evaluation_archive_sha256": sha256(args.evaluation),
        "decision_lock_archive_sha256": sha256(args.decision_lock),
        "reference_archive_sha256": sha256(args.reference),
        "protocol_sha256": sha256(args.protocol),
    }
    metrics_path = args.output / "JCP11_POSTLOCK_METRICS.csv"
    summary_path = args.output / "JCP11_SUPPORT_VALIDATION_SUMMARY.json"
    write_csv(metrics_path, metric_rows)
    write_json(summary_path, summary)
    make_figure(
        args.loo_calibration,
        args.specificity_ledger,
        manifest,
        ratio_summary,
        args.output / "jcp11_support_validation",
    )
    final_archive = args.output / "JCP11_SUPPORT_VALIDATION.zip"
    members = [
        args.protocol, args.rule_archive, args.specificity_archive,
        args.evaluation, args.decision_lock, metrics_path, summary_path,
        args.output / "jcp11_support_validation.pdf",
        args.output / "jcp11_support_validation.png",
    ]
    pre_submission = args.output / "JCP11_PRE_SUBMISSION_LOCK.json"
    if pre_submission.is_file():
        members.append(pre_submission)
    with zipfile.ZipFile(final_archive, "w", zipfile.ZIP_STORED) as archive:
        for path in members:
            archive.write(path, arcname=path.name)
    final_archive.with_suffix(".zip.sha256").write_text(
        f"{sha256(final_archive)}  {final_archive.name}\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze-rule")
    freeze.add_argument("--model-lock", type=Path, required=True)
    freeze.add_argument("--protocol", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.set_defaults(function=command_freeze)

    specificity = commands.add_parser("specificity")
    specificity.add_argument("--model-lock", type=Path, required=True)
    specificity.add_argument("--rule", type=Path, required=True)
    specificity.add_argument("--campaign", type=Path, required=True)
    specificity.add_argument("--output", type=Path, required=True)
    specificity.set_defaults(function=command_specificity)

    verify = commands.add_parser("verify-eval")
    verify.add_argument("--case", type=Path, required=True)
    verify.add_argument("--seed", type=int, required=True)
    verify.add_argument("--patch-report", type=Path, required=True)
    verify.add_argument("--summary", type=Path, required=True)
    verify.add_argument("--archive", type=Path, required=True)
    verify.set_defaults(function=command_verify_eval)

    collect = commands.add_parser("collect-lock")
    collect.add_argument("--work", type=Path, required=True)
    collect.add_argument("--model-lock", type=Path, required=True)
    collect.add_argument("--rule", type=Path, required=True)
    collect.add_argument("--protocol", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True)
    collect.set_defaults(function=command_collect_lock)

    score = commands.add_parser("score-pack")
    score.add_argument("--rule-archive", type=Path, required=True)
    score.add_argument("--loo-calibration", type=Path, required=True)
    score.add_argument("--specificity-archive", type=Path, required=True)
    score.add_argument("--specificity-summary", type=Path, required=True)
    score.add_argument("--specificity-ledger", type=Path, required=True)
    score.add_argument("--evaluation", type=Path, required=True)
    score.add_argument("--decision-lock", type=Path, required=True)
    score.add_argument("--reference", type=Path, required=True)
    score.add_argument("--protocol", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.set_defaults(function=command_score_pack)
    return root


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
