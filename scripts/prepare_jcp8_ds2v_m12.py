#!/usr/bin/env python3
"""Prepare an independent Mach-12 DS2V reference trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import prepare_jcp3_ds2v_m12 as base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-source", type=Path, required=True)
    parser.add_argument("--output-data", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "stage": "JCP8_M12_independent_reference",
        "classification": "prospective_reference_after_prediction_lock",
        "nominal_Mach": 12.0,
        **base.patch_source(args.source, args.output_source),
        **base.patch_data(args.data, args.output_data),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
