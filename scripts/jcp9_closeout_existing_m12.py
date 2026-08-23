#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np


FIELDS = ("n", "u", "v", "T", "Pxx", "Pxy", "Pyy", "qx", "qy")
MASS = 6.62999997e-26
KB = 1.380649e-23
EPS = np.finfo(np.float64).tiny
HEADER_RE = re.compile(r"FNUM=\s*([+\-0-9.EeDd]+).*BLOCK_SAMPLES=(\d+)", re.I)
REF_SEEDS = (26082801, 26082802, 26082803, 26082804)
EXPECTED_PREDICTION_SHA256 = "54db6c0be71764df87f9912090821d4676625ea7ccd8da1f4c069e7edd2ac0d8"
EXPECTED_REFERENCE_SHA256 = "340dd425239d3df48a056b618caf49b1af22348e384ae9ab3ae597c5ba587f12"
EXPECTED_SCORE_SHA256 = "0469f5d78f4a8e4b075a73c781c9d14a2074b16378c6fa2a5545633c14431866"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_moment_bytes(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    text = b"\n".join(data.splitlines()[:3]).decode("ascii")
    match = HEADER_RE.search(text)
    if match is None:
        raise ValueError("missing moment header")
    fnum = float(match.group(1).replace("D", "E").replace("d", "e"))
    block_samples = int(match.group(2))
    raw = np.loadtxt(io.BytesIO(data), comments="#")
    m0 = raw[:, 5]
    m1x, m1y, m1z = raw[:, 6], raw[:, 7], raw[:, 8]
    m2xx, m2yy, m2zz = raw[:, 9], raw[:, 10], raw[:, 11]
    m2xy, m2xz, m2yz = raw[:, 12], raw[:, 13], raw[:, 14]
    energy, energy_vx, energy_vy = raw[:, 15], raw[:, 16], raw[:, 17]
    u, v, w = m1x / m0, m1y / m0, m1z / m0
    factor = fnum / (raw[:, 4] * float(block_samples))
    n = factor * m0
    cxx = m2xx - m1x * u
    cyy = m2yy - m1y * v
    czz = m2zz - m1z * w
    cxy = m2xy - m1x * v
    pxx = factor * MASS * cxx
    pyy = factor * MASS * cyy
    pzz = factor * MASS * czz
    pxy = factor * MASS * cxy
    temperature = (pxx + pyy + pzz) / (3.0 * n * KB)
    speed2 = u * u + v * v + w * w
    qx_sum = energy_vx - u * energy - MASS * (u * m2xx + v * m2xy + w * m2xz) + MASS * u * m0 * speed2
    qy_sum = energy_vy - v * energy - MASS * (u * m2xy + v * m2yy + w * m2yz) + MASS * v * m0 * speed2
    fields = np.column_stack((n, u, v, temperature, pxx, pxy, pyy, factor * qx_sum, factor * qy_sum))
    return raw[:, (0, 2, 3, 4)].copy(), fields.astype(np.float64)


def load_reference(path: Path) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    coords_ref = None
    units: dict[int, np.ndarray] = {}
    with zipfile.ZipFile(path) as outer:
        for seed in REF_SEEDS:
            nested_bytes = outer.read(f"units/seed_{seed}/JCP8_M12_REFERENCE_seed_{seed}.zip")
            with zipfile.ZipFile(io.BytesIO(nested_bytes)) as nested:
                summary = json.loads(nested.read("JCP8_M12_REFERENCE_SUMMARY.json"))
                blocks = []
                for nout in summary["retained_nout"]:
                    coords, fields = parse_moment_bytes(nested.read(f"JCP3_MOMENTS_NOUT{nout:04d}.DAT"))
                    if coords_ref is None:
                        coords_ref = coords
                    elif not np.allclose(coords_ref, coords, rtol=0.0, atol=2e-8):
                        raise ValueError("coordinate mismatch")
                    blocks.append(fields)
                units[seed] = np.asarray(blocks)
    assert coords_ref is not None
    return coords_ref, units


def nrmse(value: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((value - target) ** 2, axis=0)) / np.maximum(np.sqrt(np.mean(target**2, axis=0)), EPS)


def geometric(values: np.ndarray) -> float:
    return float(np.exp(np.mean(np.log(np.maximum(np.asarray(values), EPS)))))


def zones_for(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dx, dy = coords[:, 1] - 0.1524, coords[:, 2]
    radius = np.hypot(dx, dy)
    zones = np.full(len(coords), 2, dtype=np.int8)
    zones[radius <= 0.20] = 0
    zones[(radius > 0.20) & (dx >= 0.0)] = 1
    return zones, dx / radius, dy / radius, radius <= 0.20


def fit_alphas(units: dict[int, np.ndarray], coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit zone/field shrinkage from calibration seeds only.

    Each calibration seed alternates between being the prior and the disjoint
    B3/remaining-B37 observation/target unit.  Fixed draws are deterministic.
    Heat flux is fitted in the cylinder-local normal/tangential basis.
    """
    zones, ex, ey, _ = zones_for(coords)
    alpha = np.zeros((9, 3), dtype=np.float64)
    alpha_q = np.zeros((2, 3), dtype=np.float64)
    cal = (26082801, 26082802)
    records: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray]]] = {}
    qrecords: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray]]] = {}
    for target_seed, prior_seed in ((cal[0], cal[1]), (cal[1], cal[0])):
        prior = np.mean(units[prior_seed], axis=0)
        for draw in range(20):
            rng = np.random.default_rng(target_seed + 104729 * (draw + 1))
            order = rng.permutation(40)
            obs = np.mean(units[target_seed][order[:3]], axis=0)
            target = np.mean(units[target_seed][order[3:]], axis=0)
            for field in range(7):
                for zone in range(3):
                    mask = zones == zone
                    records.setdefault((field, zone), []).append(((obs[mask, field] - prior[mask, field]).ravel(), (target[mask, field] - prior[mask, field]).ravel()))
            for comp, (oi, pi, ti) in enumerate((
                (obs[:, 7] * ex + obs[:, 8] * ey, prior[:, 7] * ex + prior[:, 8] * ey, target[:, 7] * ex + target[:, 8] * ey),
                (-obs[:, 7] * ey + obs[:, 8] * ex, -prior[:, 7] * ey + prior[:, 8] * ex, -target[:, 7] * ey + target[:, 8] * ex),
            )):
                for zone in range(3):
                    mask = zones == zone
                    qrecords.setdefault((comp, zone), []).append(((oi[mask] - pi[mask]).ravel(), (ti[mask] - pi[mask]).ravel()))
    for field in range(7):
        for zone in range(3):
            xs = np.concatenate([x for x, _ in records[(field, zone)]])
            ys = np.concatenate([y for _, y in records[(field, zone)]])
            alpha[field, zone] = np.clip(np.dot(xs, ys) / max(np.dot(xs, xs), EPS), 0.0, 1.0)
    for comp in range(2):
        for zone in range(3):
            xs = np.concatenate([x for x, _ in qrecords[(comp, zone)]])
            ys = np.concatenate([y for _, y in qrecords[(comp, zone)]])
            alpha_q[comp, zone] = np.clip(np.dot(xs, ys) / max(np.dot(xs, xs), EPS), 0.0, 1.0)
    return alpha, alpha_q


def fuse(obs: np.ndarray, prior: np.ndarray, coords: np.ndarray, alpha: np.ndarray, alpha_q: np.ndarray) -> np.ndarray:
    zones, ex, ey, _ = zones_for(coords)
    out = prior.copy()
    for field in range(7):
        for zone in range(3):
            mask = zones == zone
            out[mask, field] += alpha[field, zone] * (obs[mask, field] - prior[mask, field])
    obs_qn, prior_qn = obs[:, 7] * ex + obs[:, 8] * ey, prior[:, 7] * ex + prior[:, 8] * ey
    obs_qt, prior_qt = -obs[:, 7] * ey + obs[:, 8] * ex, -prior[:, 7] * ey + prior[:, 8] * ex
    qn, qt = prior_qn.copy(), prior_qt.copy()
    for zone in range(3):
        mask = zones == zone
        qn[mask] += alpha_q[0, zone] * (obs_qn[mask] - prior_qn[mask])
        qt[mask] += alpha_q[1, zone] * (obs_qt[mask] - prior_qt[mask])
    out[:, 7] = qn * ex - qt * ey
    out[:, 8] = qn * ey + qt * ex
    return out


def bootstrap_ci(values: list[float], seed: int) -> list[float]:
    logs = np.log(np.maximum(np.asarray(values, dtype=np.float64), EPS))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(logs), size=(20000, len(logs)))
    samples = np.exp(np.mean(logs[indices], axis=1))
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Close the existing M12 campaign without new DSMC")
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--prospective-score", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    expected = (
        (args.prediction, EXPECTED_PREDICTION_SHA256),
        (args.reference, EXPECTED_REFERENCE_SHA256),
        (args.prospective_score, EXPECTED_SCORE_SHA256),
    )
    for path, digest in expected:
        if sha256(path) != digest:
            raise ValueError(f"locked checksum mismatch: {path}")
    with zipfile.ZipFile(args.prospective_score) as score_zip:
        prospective = json.loads(score_zip.read("JCP8_M12_SCORE_SUMMARY.json"))
    if prospective.get("primary_prospective_gate_pass") is not False:
        raise ValueError("JCP9 is valid only after the recorded zero-shot gate failure")

    coords, units = load_reference(args.reference)
    calibration_prior = np.mean(np.concatenate((units[26082801], units[26082802]), axis=0), axis=0)
    heldout_target = np.mean(np.concatenate((units[26082803], units[26082804]), axis=0), axis=0)
    alpha, alpha_q = fit_alphas(units, coords)
    zones, ex, ey, near = zones_for(coords)
    target_qn = heldout_target[:, 7] * ex + heldout_target[:, 8] * ey
    records = []
    metric_rows = []
    with zipfile.ZipFile(args.prediction) as prediction_zip:
        lock = json.loads(prediction_zip.read("JCP7_M12_PREDICTION_LOCK.json"))
        if lock.get("reference_artifacts_read") is not False or lock.get("prediction_count") != 12:
            raise ValueError("invalid prediction lock")
        for seed in lock["selected_seeds"]:
            with np.load(io.BytesIO(prediction_zip.read(f"JCP7_PREDICTION_seed_{seed}.npz"))) as z:
                raw3 = z["raw_B3"].astype(np.float64)
                raw10 = z["raw_B10"].astype(np.float64)
            methods = {
                "raw_B3": raw3,
                "raw_B10": raw10,
                "calibration_prior": calibration_prior,
                "calibrated_B3": fuse(raw3, calibration_prior, coords, alpha, alpha_q),
            }
            errors = {name: nrmse(value, heldout_target) for name, value in methods.items()}
            qn_errors = {}
            for name, value in methods.items():
                qn = value[:, 7] * ex + value[:, 8] * ey
                qn_errors[name] = float(nrmse(qn[near, None], target_qn[near, None])[0])
                for field, error in zip(FIELDS, errors[name], strict=True):
                    metric_rows.append({"seed": seed, "method": name, "field": field, "nrmse": float(error)})
                metric_rows.append({"seed": seed, "method": name, "field": "qn_near_wall", "nrmse": qn_errors[name]})
            for method in ("calibration_prior", "calibrated_B3"):
                ratio = errors[method] / errors["raw_B10"]
                records.append({
                    "seed": seed,
                    "method": method,
                    "field_ratios": ratio.tolist(),
                    "all_nine_ratio": geometric(ratio),
                    "qy_ratio": float(ratio[8]),
                    "qn_ratio": qn_errors[method] / qn_errors["raw_B10"],
                })
    summary = {
        "stage": "JCP9_M12_support_aware_campaign_closeout",
        "classification": "post_failure_split_reanalysis_no_new_DSMC_not_prospective",
        "calibration_reference_seeds": [26082801, 26082802],
        "heldout_reference_seeds": [26082803, 26082804],
        "evaluation_seed_count": 12,
        "alpha": alpha.tolist(),
        "alpha_qn_qt": alpha_q.tolist(),
        "locked_input_sha256": {
            "prediction": sha256(args.prediction),
            "reference": sha256(args.reference),
            "prospective_score": sha256(args.prospective_score),
            "protocol": sha256(args.protocol),
        },
        "prospective_zero_shot_result_retained": prospective["endpoints"],
        "methods": {},
    }
    for method in ("calibration_prior", "calibrated_B3"):
        selected = [r for r in records if r["method"] == method]
        field_matrix = np.asarray([r["field_ratios"] for r in selected])
        all_nine_values = [r["all_nine_ratio"] for r in selected]
        qy_values = [r["qy_ratio"] for r in selected]
        qn_values = [r["qn_ratio"] for r in selected]
        summary["methods"][method] = {
            "all_nine_ratio_geometric": geometric(np.asarray([r["all_nine_ratio"] for r in selected])),
            "all_nine_ratio_bootstrap_95pct_CI": bootstrap_ci(all_nine_values, 26082911),
            "field_ratio_geometric": dict(zip(FIELDS, [geometric(field_matrix[:, j]) for j in range(9)], strict=True)),
            "qy_ratio_geometric": geometric(np.asarray([r["qy_ratio"] for r in selected])),
            "qy_ratio_bootstrap_95pct_CI": bootstrap_ci(qy_values, 26082912),
            "qn_ratio_geometric": geometric(np.asarray([r["qn_ratio"] for r in selected])),
            "qn_ratio_bootstrap_95pct_CI": bootstrap_ci(qn_values, 26082913),
            "all_nine_seed_count_improved": int(sum(r["all_nine_ratio"] < 1.0 for r in selected)),
            "qy_seed_count_improved": int(sum(r["qy_ratio"] < 1.0 for r in selected)),
            "qn_seed_count_improved": int(sum(r["qn_ratio"] < 1.0 for r in selected)),
        }
    promoted = summary["methods"]["calibrated_B3"]
    gate = bool(
        promoted["all_nine_ratio_geometric"] < 0.95
        and promoted["qy_ratio_geometric"] < 0.95
        and promoted["qn_ratio_geometric"] < 0.95
        and promoted["all_nine_seed_count_improved"] >= 9
        and promoted["qy_seed_count_improved"] >= 9
        and promoted["qn_seed_count_improved"] >= 9
    )
    summary["primary_closeout_gate_pass"] = gate
    summary["status"] = "closeout_complete_gate_pass" if gate else "closeout_complete_gate_fail"
    summary["interpretation_guard"] = (
        "Report the original zero-shot failure and this post-failure calibration split together; "
        "do not relabel JCP9 as prospective or as unseen-condition extrapolation."
    )
    summary_path = args.output / "JCP9_M12_CLOSEOUT_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    metrics_path = args.output / "JCP9_M12_CLOSEOUT_METRICS.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("seed", "method", "field", "nrmse"))
        writer.writeheader(); writer.writerows(metric_rows)
    archive_path = args.output / "JCP9_M12_CLOSEOUT.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in (args.protocol, summary_path, metrics_path):
            archive.write(path, arcname=path.name)
    archive_path.with_suffix(".zip.sha256").write_text(f"{sha256(archive_path)}  {archive_path.name}\n")
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
