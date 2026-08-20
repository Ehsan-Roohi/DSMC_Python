#!/usr/bin/env python3
"""Prepare the locked Mach-8 development-reference DS2V source and data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import prepare_jcp3_ds2v_m12 as base


M8_SPEED = 2107.28


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-source", type=Path, required=True)
    parser.add_argument("--output-data", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    # Reuse the mechanically verified additive-moment patch.  The pilot stop is
    # inactive unless JCP3_PILOT_ONLY exists; production jobs never create it.
    base.M12_SPEED = M8_SPEED
    report = {
        "stage": "JCP4_M8_development_reference",
        "classification": "development_only_not_prospective_evidence",
        "nominal_Mach": 8.0,
        **base.patch_source(args.source, args.output_source),
        **base.patch_data(args.data, args.output_data),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
