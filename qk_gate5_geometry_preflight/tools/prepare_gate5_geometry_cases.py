#!/usr/bin/env python3
"""Generate the chemistry-off Mach-3.5 nozzle/duct siting cases."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


P0_PA = 5.0e6
T0_K = 2500.0
TBACK_K = 2395.2542123279973
THROAT_HEIGHT_M = 15.0e-6
TARGET_AREA_RATIO = 6.109855622339336
TARGET_HEIGHT_M = THROAT_HEIGHT_M * TARGET_AREA_RATIO
INLET_HEIGHT_M = 35.0e-6
MAIN_LENGTH_M = 250.0e-6
THROAT_X_M = 50.0e-6
POST_LENGTH_M = 100.0e-6
NCX = 100
NCY = 30
NBX = 40
NBY = 30
BACK_PRESSURE_RATIOS = (0.16, 0.20, 0.24)
CASES = [
    (f"m35_r{int(round(100 * ratio)):03d}_chem_off", ratio)
    for ratio in BACK_PRESSURE_RATIOS
]


REPLACEMENTS = {
    "!NSM": "150\t!NSM geometry-preflight burn-in cycles",
    "!NCX": f"{NCX}\t!NCX coarse shock-siting grid",
    "!NCY": f"{NCY}\t!NCY coarse shock-siting grid",
    "!NBX": f"{NBX}\t!NBX 0.10 mm constant-area duct",
    "!NBY": f"{NBY}\t!NBY equal to NCY: no artificial buffer step",
    "!FTMP": f"{T0_K:.8f}\t!FTMP inlet stagnation-reservoir temperature",
    "!PIN ": f"{P0_PA:.8e}\t!PIN ",
    "!VFX": "0.0\t!VFX reservoir injection; nozzle generates axial speed",
    "!VFY": "0.0\t!VFY",
    "!FNUM": "8.0E12\t!FNUM geometry-preflight particle weight",
    "!DTM is the time step": "2.0E-11\t!DTM is the time step",
    "!CB(1)": "0.0\t!CB(1)",
    "!CB(2)": f"{MAIN_LENGTH_M:.12e}\t!CB(2) nozzle-to-duct junction",
    "!CB(3)": "0.0\t!CB(3)",
    "!CB(4)-outlet": f"{TARGET_HEIGHT_M:.12e}\t!CB(4)-outlet target A/Astar",
    "!CB(5)the nozzle throat height": f"{THROAT_HEIGHT_M:.12e}\t!CB(5)the nozzle throat height",
    "!CB(6)-INLET HEIGHT": f"{TARGET_HEIGHT_M - INLET_HEIGHT_M:.12e}\t!CB(6)-INLET HEIGHT lower-wall coordinate",
    "!CB(9)": "0.0\t!CB(9) constant-area duct lower wall",
    "!INZ(1)": "0\t!INZ(1)",
    "!INZ(2)-Throat location": "20\t!INZ(2)-Throat location at 50 um",
    "!INZ(3)": "100\t!INZ(3) divergence ends at duct junction",
    "!IB(6)": "2\t!IB(6) specular lower wall of post-shock duct",
    "!NIS is the number of time steps between samples": "10\t!NIS is the number of time steps between samples",
    "!NSP is the number of samples between restart and output file updates": "10\t!NSP is the number of samples between restart and output file updates",
    "!NPS is the number of updates to reach assumed steady flow": "150\t!NPS is the number of updates to reach assumed steady flow",
    "!NPTT is the number of file updates to STOP": "300\t!NPTT is the number of file updates to STOP",
    "!ITypeQw": "0\t!ITypeQw fixed; TSURF<0 makes nozzle wall specular",
    "!Tw1": "300.0\t!Tw1 unused by specular wall",
    "!TW Buffer": f"{TBACK_K:.8f}\t!TW Buffer unused by specular duct wall",
    "!NSQ": "100\t!NSQ",
}


def rewrite(lines: list[str], ratio: float, *, smoke: bool = False) -> str:
    replacements = dict(REPLACEMENTS)
    replacements["!POUT"] = f"{P0_PA * ratio:.8e}\t!POUT"
    if smoke:
        replacements["!NSM"] = "1\t!NSM geometry smoke"
        replacements["!NPS is the number of updates to reach assumed steady flow"] = (
            "1\t!NPS is the number of updates to reach assumed steady flow"
        )
        replacements["!NPTT is the number of file updates to STOP"] = (
            "30\t!NPTT is the number of file updates to STOP"
        )

    output: list[str] = []
    seen: set[str] = set()
    skip_ghs = False
    tsurf_seen = 0
    for raw in lines:
        line = raw.rstrip("\n")
        if "!IMOLMODEL:" in line:
            output.append("1\t!IMOLMODEL: Q-K mixture uses validated pair VHS data")
            skip_ghs = True
            continue
        if skip_ghs and any(
            marker in line
            for marker in ("!GHS_SIG", "!GHS_EPSK", "!GHS_A1", "!GHS_A2", "!GHS_W1", "!GHS_W2")
        ):
            continue
        if "!TSURF(" in line:
            tsurf_seen += 1
            output.append(f"-1.0\t!TSURF({tsurf_seen}) specular for inviscid shock siting")
            continue
        for marker, value in replacements.items():
            if marker in line:
                output.append(value)
                seen.add(marker)
                break
        else:
            output.append(line)

    missing = sorted(set(replacements) - seen)
    if missing:
        raise RuntimeError(f"template markers not replaced: {missing}")
    if tsurf_seen != 3:
        raise RuntimeError(f"expected three TSURF lines, found {tsurf_seen}")
    text = "\n".join(output) + "\n"
    checks = {
        "active_cells": NCX * NCY + NBX * NBY == 4200,
        "duct_length": math.isclose(NBX * MAIN_LENGTH_M / NCX, POST_LENGTH_M),
        "no_buffer_step": NBY == NCY,
        "throat_index": math.isclose(20 * MAIN_LENGTH_M / NCX, THROAT_X_M),
        "area_ratio": math.isclose(TARGET_HEIGHT_M / THROAT_HEIGHT_M, TARGET_AREA_RATIO),
        "closed_duct": "2\t!IB(6)" in text,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"geometry invariants failed: {failed}")
    return text


def write_case(template: Path, outroot: Path, index: int, smoke: bool) -> Path:
    ratio = 0.20 if smoke else CASES[index][1]
    name = "m35_r020_chem_off_smoke" if smoke else CASES[index][0]
    case = outroot / name
    case.mkdir(parents=True, exist_ok=True)
    text = rewrite(template.read_text().splitlines(True), ratio, smoke=smoke)
    (case / "InputData.txt").write_text(text)
    (case / "ChemistryControl.txt").write_text(f"2\n95001\n{TBACK_K:.12f}\n")
    (case / "GEOMETRY_CASE.txt").write_text(
        f"index={index}\nname={name}\np0_Pa={P0_PA:.8e}\n"
        f"pb_over_p0={ratio:.8f}\nT0_K={T0_K:.8f}\nTback_K={TBACK_K:.12f}\n"
        "mixture=2H2+O2+3Ar\nchemistry=OFF\nwall=specular\n"
        f"target_M1=3.5\ntarget_A_over_Astar={TARGET_AREA_RATIO:.12f}\n"
        f"target_height_m={TARGET_HEIGHT_M:.12e}\nthroat_height_m={THROAT_HEIGHT_M:.12e}\n"
        f"nozzle_to_duct_x_m={MAIN_LENGTH_M:.12e}\npost_duct_length_m={POST_LENGTH_M:.12e}\n"
        f"grid_NCX={NCX}\ngrid_NCY={NCY}\ngrid_NBX={NBX}\ngrid_NBY={NBY}\n"
        "resolution_class=coarse_shock_siting_only\n"
    )
    return case


def self_test(template: Path) -> None:
    for ratio in BACK_PRESSURE_RATIOS:
        text = rewrite(template.read_text().splitlines(True), ratio)
        assert f"{P0_PA * ratio:.8e}\t!POUT" in text
        assert "2\t!IB(6)" in text
        assert text.count("specular for inviscid shock siting") == 3
    print("GATE5_GEOMETRY_CASE_PREPARATION_SELF_TEST_PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--outroot", type=Path)
    parser.add_argument("--index", type=int, choices=range(len(CASES)))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(args.template)
        return
    if args.outroot is None:
        parser.error("--outroot is required unless --self-test is used")
    if args.smoke:
        case = write_case(args.template, args.outroot, 1, True)
        print(f"GATE5_GEOMETRY_SMOKE_CASE_PREPARATION_PASS case={case.name}")
        return
    if args.index is None:
        parser.error("--index is required for a production siting case")
    case = write_case(args.template, args.outroot, args.index, False)
    print(f"GATE5_GEOMETRY_CASE_PREPARATION_PASS index={args.index} case={case.name}")


if __name__ == "__main__":
    main()
