#!/usr/bin/env python3
"""Validate chemistry-off geometry smoke and coarse shock-siting cases."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

from prepare_gate5_geometry_cases import (
    BACK_PRESSURE_RATIOS,
    CASES,
    MAIN_LENGTH_M,
    NBX,
    NCX,
    NCY,
    P0_PA,
    POST_LENGTH_M,
    TARGET_HEIGHT_M,
    T0_K,
    TBACK_K,
)


FATAL = re.compile(
    r"fatal|segmentation|floating|backtrace|\*+error|error in xc|"
    r"error in enter|y\s*<\s*ylb|error in reflect",
    re.I,
)
BOUNDARY_GUARD = re.compile(r"(?:x|y)\s+coord\s+outside\s+flow", re.I)
EXPECTED = {name for name, _ in CASES}
ACTIVE_CELLS = 4200
DESIGN_LAMBDA1_M = 3.359425365560068e-7
DESIGN_LAMBDA2_M = 1.0078401076229616e-7


def key_values(path: Path) -> dict[str, int | float | str]:
    result: dict[str, int | float | str] = {}
    if not path.exists():
        return result
    for line in path.read_text(errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        try:
            result[key.strip()] = (
                float(value) if any(c in value.lower() for c in (".", "e")) else int(value)
            )
        except ValueError:
            result[key.strip()] = value
    return result


def numeric_rows(path: Path, expected_columns: int) -> list[list[float]]:
    if not path.exists():
        return []
    rows: list[list[float]] = []
    for line in path.read_text(errors="replace").splitlines()[1:]:
        fields = line.replace(",", " ").split()
        if len(fields) != expected_columns:
            continue
        try:
            rows.append([float(value) for value in fields])
        except ValueError:
            pass
    return rows


def flow_centerline(rows: list[list[float]]) -> list[list[float]]:
    """Follow the symmetry/top row through both nozzle and duct zones."""
    by_x: dict[float, list[float]] = {}
    for row in rows:
        if int(row[1]) not in (1, 2):
            continue
        x = row[2]
        old = by_x.get(x)
        if old is None or row[3] > old[3]:
            by_x[x] = row
    return [by_x[x] for x in sorted(by_x)]


def shock_metrics(line: list[list[float]]) -> dict[str, Any]:
    if len(line) < 30:
        return {"detected": False, "x_m": None, "reason": "short_centerline"}
    candidates: list[tuple[float, int, float, float, float]] = []
    for index in range(3, len(line) - 4):
        x = line[index][2]
        if x <= 55.0e-6:
            continue
        upstream = line[index - 2]
        downstream = line[index + 2]
        rho_jump = (downstream[4] - upstream[4]) / max(
            0.5 * (abs(downstream[4]) + abs(upstream[4])), 1.0e-300
        )
        pressure_jump = (downstream[12] - upstream[12]) / max(
            0.5 * (abs(downstream[12]) + abs(upstream[12])), 1.0e-300
        )
        velocity_drop = (upstream[8] - downstream[8]) / max(abs(upstream[8]), 1.0e-300)
        score = max(rho_jump, 0.0) + max(pressure_jump, 0.0) + max(velocity_drop, 0.0)
        candidates.append((score, index, rho_jump, pressure_jump, velocity_drop))
    if not candidates:
        return {"detected": False, "x_m": None, "reason": "no_candidates"}
    score, index, rho_jump, pressure_jump, velocity_drop = max(candidates)
    detected = rho_jump > 0.08 and pressure_jump > 0.08 and velocity_drop > 0.03
    spacing = statistics.median(
        line[i + 1][2] - line[i][2] for i in range(len(line) - 1)
    )
    upstream = line[index - 2]
    downstream = line[index + 2]
    return {
        "detected": detected,
        "x_m": line[index][2],
        "region": "duct" if line[index][2] >= MAIN_LENGTH_M else "divergent_nozzle",
        "grid_spacing_m": spacing,
        "rho_jump_fraction": rho_jump,
        "pressure_jump_fraction": pressure_jump,
        "velocity_drop_fraction": velocity_drop,
        "mach_upstream": upstream[11],
        "mach_downstream": downstream[11],
        "Ttr_upstream_K": upstream[5],
        "Ttr_downstream_K": downstream[5],
        "pressure_upstream_Pa": upstream[12],
        "pressure_downstream_Pa": downstream[12],
        "score": score,
    }


def monitor_metrics(rows: list[list[float]]) -> dict[str, Any]:
    if len(rows) < 20:
        return {"records": len(rows), "steady": False}
    tail = rows[-20:]
    throat = [row[4] for row in tail]
    inlet = [row[3] for row in tail]
    outlet = [row[5] for row in tail]
    mean_throat = statistics.fmean(throat)
    relative_std = statistics.pstdev(throat) / max(abs(mean_throat), 1.0e-300)
    imbalance = abs(statistics.fmean(inlet) - statistics.fmean(outlet)) / max(
        abs(mean_throat), 1.0e-300
    )
    return {
        "records": len(rows),
        "throat_relative_std_last20": relative_std,
        "mass_flux_imbalance_last20": imbalance,
        "steady": relative_std < 0.20 and imbalance < 0.30,
    }


def progress_rows(log: str) -> list[list[float]]:
    progress: list[list[float]] = []
    for line in log.splitlines():
        fields = line.split()
        if len(fields) != 4:
            continue
        try:
            progress.append(
                [float(int(fields[0])), float(int(fields[1])), float(fields[2]), float(fields[3])]
            )
        except ValueError:
            pass
    return progress


def resolution_audit() -> dict[str, Any]:
    dx = MAIN_LENGTH_M / NCX
    dy = TARGET_HEIGHT_M / NCY
    worst = max(dx, dy)
    ratio1 = worst / DESIGN_LAMBDA1_M
    ratio2 = worst / DESIGN_LAMBDA2_M
    return {
        "dx_m": dx,
        "envelope_dy_m": dy,
        "design_lambda1_m": DESIGN_LAMBDA1_M,
        "design_lambda2_m": DESIGN_LAMBDA2_M,
        "max_cell_over_lambda1": ratio1,
        "max_cell_over_lambda2": ratio2,
        "dsmc_cell_resolution_pass": max(ratio1, ratio2) <= 1.0,
        "classification": "coarse_geometry_and_shock_siting_only",
    }


def write_centerline(path: Path, rows: list[list[float]]) -> None:
    header = [
        "cell", "zone", "x_m", "y_m", "rho", "Ttr", "Trot", "T",
        "ux", "uy", "uz", "Mach", "pressure", "Kn",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def inspect_case(case: Path, outdir: Path) -> dict[str, Any]:
    log = (case / "run.log").read_text(errors="replace") if (case / "run.log").exists() else ""
    progress = progress_rows(log)
    events = key_values(case / "QK_GATE5_EVENTS.txt")
    metadata = key_values(case / "GEOMETRY_CASE.txt")
    flow = numeric_rows(case / "QK_PRODUCTION_FLOW_FIELD.dat", 14)
    monitor = numeric_rows(case / "QK_PRODUCTION_MONITOR.dat", 6)
    line = flow_centerline(flow)
    write_centerline(outdir / f"{case.name}_centerline.csv", line)
    shock = shock_metrics(line)
    through_flow = monitor_metrics(monitor)
    boundary_removals = len(BOUNDARY_GUARD.findall(log))
    reaction_events = sum(
        int(events.get(key, 0))
        for key in ("exchange_reactions", "recombinations", "dissociations")
    )
    audit = bool(
        int(events.get("atom_failures", -1)) == 0
        and int(events.get("capacity_failures", -1)) == 0
        and float(events.get("max_local_energy_error", 1.0)) < 2.0e-8
    )
    numerical = bool(
        len(progress) >= 280
        and progress[-1][3] >= 5.5e-7
        and max((int(row[1]) for row in progress), default=0) < 1800000
        and len(flow) >= ACTIVE_CELLS
        and len(line) >= NCX + NBX
        and not FATAL.search(log)
        and boundary_removals <= 20
        and audit
        and int(events.get("mode", 0)) == 2
        and reaction_events == 0
    )
    shock_x = shock.get("x_m") if shock.get("detected") else None
    siting = bool(
        numerical
        and shock_x is not None
        and 225.0e-6 <= float(shock_x) <= 300.0e-6
        and float(shock.get("mach_upstream") or 0.0) >= 2.5
    )
    return {
        "case": case.name,
        "condition": metadata,
        "progress_records": len(progress),
        "last_record": progress[-1] if progress else None,
        "max_particles": max((int(row[1]) for row in progress), default=0),
        "events": events,
        "reaction_events": reaction_events,
        "shock": shock,
        "through_flow": through_flow,
        "boundary_guard_removals": boundary_removals,
        "numerical_pass": numerical,
        "refinement_candidate": siting,
    }


def validate_smoke(case: Path, run_status: int, out: Path) -> None:
    log = (case / "run.log").read_text(errors="replace") if (case / "run.log").exists() else ""
    progress = progress_rows(log)
    events = key_values(case / "QK_GATE5_EVENTS.txt")
    reaction_events = sum(
        int(events.get(key, 0))
        for key in ("exchange_reactions", "recombinations", "dissociations")
    )
    result = {
        "scope": "Gate5 closed-duct chemistry-off geometry smoke",
        "case": case.name,
        "run_status": run_status,
        "progress_records": len(progress),
        "last_record": progress[-1] if progress else None,
        "max_particles": max((int(row[1]) for row in progress), default=0),
        "particle_limit": 1800000,
        "active_output_marker": "GATE5_ACTIVE_OUTPUT_CELLS" in log and "4200" in log,
        "prescribed_reservoir_marker": "GATE5_PRESCRIBED_RESERVOIR_BOUNDARIES" in log,
        "closed_duct_reflection_errors": len(re.findall(r"error in reflect", log, re.I)),
        "boundary_guard_removals": len(BOUNDARY_GUARD.findall(log)),
        "fatal_matches": FATAL.findall(log)[:20],
        "events": events,
        "reaction_events": reaction_events,
        "resolution": resolution_audit(),
    }
    result["pass"] = bool(
        run_status == 0
        and len(progress) >= 30
        and progress[-1][3] >= 5.9e-8
        and result["max_particles"] < result["particle_limit"]
        and result["active_output_marker"]
        and result["prescribed_reservoir_marker"]
        and not result["fatal_matches"]
        and result["boundary_guard_removals"] <= 10
        and int(events.get("mode", 0)) == 2
        and reaction_events == 0
        and int(events.get("capacity_failures", -1)) == 0
    )
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit("QK_GATE5_GEOMETRY_SMOKE_FAIL")
    print("QK_GATE5_GEOMETRY_SMOKE_PASS")


def validate_group(cases_root: Path, out: Path) -> None:
    found = {path.name for path in cases_root.iterdir() if path.is_dir()}
    cases = [inspect_case(cases_root / name, out.parent) for name in sorted(found & EXPECTED)]
    ranked = sorted(
        cases,
        key=lambda row: (
            0 if row["numerical_pass"] else 1,
            0 if row["refinement_candidate"] else 1,
            abs(float(row["shock"].get("x_m") or 0.0) - MAIN_LENGTH_M),
        ),
    )
    result = {
        "scope": "Gate5 Mach-3.5 closed-duct chemistry-off coarse shock siting",
        "conditions": {
            "p0_Pa": P0_PA,
            "T0_K": T0_K,
            "Tback_K": TBACK_K,
            "pb_over_p0": list(BACK_PRESSURE_RATIOS),
            "mixture": "2H2+O2+3Ar",
            "chemistry": "OFF",
            "wall": "specular",
            "nozzle_to_duct_x_m": MAIN_LENGTH_M,
            "post_duct_length_m": POST_LENGTH_M,
            "steps_per_case": 30000,
        },
        "resolution": resolution_audit(),
        "expected_case_count": len(EXPECTED),
        "completed_case_count": len(cases),
        "all_numerical_pass": len(cases) == len(EXPECTED) and all(
            row["numerical_pass"] for row in cases
        ),
        "refinement_candidates_ranked": [
            row["case"] for row in ranked if row["refinement_candidate"]
        ],
        "recommended_refinement_case": next(
            (row["case"] for row in ranked if row["refinement_candidate"]), None
        ),
        "publication_ready": False,
        "cases": cases,
    }
    result["pass"] = result["all_numerical_pass"]
    out.write_text(json.dumps(result, indent=2) + "\n")

    with (out.parent / "QK_GATE5_GEOMETRY_RANKING.csv").open("w", newline="") as handle:
        fields = [
            "rank", "case", "pb_over_p0", "numerical_pass", "steady",
            "shock_detected", "shock_x_um", "shock_region", "M1", "M2",
            "T1_K", "T2_K", "mass_flux_imbalance", "refinement_candidate",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(ranked, 1):
            shock = row["shock"]
            mon = row["through_flow"]
            condition = row["condition"]
            writer.writerow({
                "rank": rank,
                "case": row["case"],
                "pb_over_p0": condition.get("pb_over_p0"),
                "numerical_pass": row["numerical_pass"],
                "steady": mon.get("steady"),
                "shock_detected": shock.get("detected"),
                "shock_x_um": None if shock.get("x_m") is None else 1e6 * float(shock["x_m"]),
                "shock_region": shock.get("region"),
                "M1": shock.get("mach_upstream"),
                "M2": shock.get("mach_downstream"),
                "T1_K": shock.get("Ttr_upstream_K"),
                "T2_K": shock.get("Ttr_downstream_K"),
                "mass_flux_imbalance": mon.get("mass_flux_imbalance_last20"),
                "refinement_candidate": row["refinement_candidate"],
            })

    lines = [
        "QK Gate-5 closed-duct geometry preflight",
        f"Completed cases: {len(cases)}/{len(EXPECTED)}",
        f"All numerical pass: {result['all_numerical_pass']}",
        f"Recommended refinement case: {result['recommended_refinement_case']}",
        "",
        "Case diagnostics:",
    ]
    for row in ranked:
        shock = row["shock"]
        lines.append(
            f"  {row['case']}: numerical={row['numerical_pass']}, "
            f"steady={row['through_flow'].get('steady')}, shock={shock.get('detected')}, "
            f"x_um={None if shock.get('x_m') is None else 1e6*float(shock['x_m']):}, "
            f"region={shock.get('region')}, M1={shock.get('mach_upstream')}, "
            f"candidate={row['refinement_candidate']}"
        )
    lines += [
        "",
        "Guardrails:",
        "  This grid does not satisfy DSMC cell-size/mean-free-path resolution.",
        "  Use it only to verify the closed-duct topology and bracket shock position.",
        "  A refined chemistry-OFF grid/particle/time-step study is mandatory next.",
        "  Reacting DSMC is not authorized by this result.",
    ]
    (out.parent / "QK_GATE5_GEOMETRY_PREFLIGHT.txt").write_text("\n".join(lines) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, indent=2))
    if not result["pass"]:
        raise SystemExit("QK_GATE5_GEOMETRY_PREFLIGHT_NUMERICAL_FAIL")
    print("QK_GATE5_GEOMETRY_PREFLIGHT_PASS")


def self_test() -> None:
    line: list[list[float]] = []
    for index in range(140):
        x = (index + 0.5) * 2.5e-6
        downstream = x >= 250.0e-6
        rho = 4.0 if downstream else 1.0
        pressure = 14.0 if downstream else 1.0
        ux = 500.0 if downstream else 1900.0
        mach = 0.46 if downstream else 3.5
        line.append([index + 1, 1 if x < 250e-6 else 2, x, 90e-6, rho, 2400.0 if downstream else 665.0, 0.0, 0.0, ux, 0.0, 0.0, mach, pressure, 0.0])
    shock = shock_metrics(line)
    assert shock["detected"]
    assert abs(float(shock["x_m"]) - 250.0e-6) <= 7.5e-6
    assert shock["mach_upstream"] == 3.5
    assert resolution_audit()["dsmc_cell_resolution_pass"] is False
    print("GATE5_GEOMETRY_VALIDATOR_SELF_TEST_PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--smoke-case", type=Path)
    parser.add_argument("--run-status", type=int, default=0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.out is None:
        parser.error("--out is required")
    if args.smoke_case is not None:
        validate_smoke(args.smoke_case, args.run_status, args.out)
    elif args.cases is not None:
        validate_group(args.cases, args.out)
    else:
        parser.error("provide --smoke-case or --cases")


if __name__ == "__main__":
    main()
