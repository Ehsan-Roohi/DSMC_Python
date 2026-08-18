"""Run one collision model for the physical time defined by a reference TOML."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dsmc_cavity.cli import load_config
from dsmc_cavity.collisions import SUPPORTED_MODELS
from dsmc_cavity.solver import CavitySolver


def aligned_count(target_time: float, dt: float) -> int:
    """Ceiling with a small tolerance for an already integral ratio."""
    return int(math.ceil(target_time / dt - 1.0e-12))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=SUPPORTED_MODELS, required=True)
    parser.add_argument("--backend", choices=("cpu", "gpu", "auto"), default="gpu")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    reference = load_config(str(args.config), {})
    reference_dt, _ = reference.resolved_dt()
    target_end_time = reference.steps * reference_dt
    target_warmup_time = reference.warmup_steps * reference_dt
    target_sample_interval = reference.sample_stride * reference_dt

    probe = replace(
        reference,
        model=args.model,
        backend=args.backend,
        seed=reference.seed if args.seed is None else args.seed,
        output_dir=str(args.output_dir),
    )
    model_dt, _ = probe.resolved_dt()
    warmup_steps = max(1, aligned_count(target_warmup_time, model_dt))
    steps = max(warmup_steps + 1, aligned_count(target_end_time, model_dt))
    sample_stride = max(1, round(target_sample_interval / model_dt))
    config = replace(
        probe,
        steps=steps,
        warmup_steps=warmup_steps,
        sample_stride=sample_stride,
    )
    plan = {
        "model": args.model,
        "reference_dt_seconds": reference_dt,
        "model_dt_seconds": model_dt,
        "steps": steps,
        "warmup_steps": warmup_steps,
        "sample_stride": sample_stride,
        "target_end_time_seconds": target_end_time,
        "actual_end_time_seconds": steps * model_dt,
        "output_dir": str(args.output_dir),
    }
    print(json.dumps(plan, indent=2, sort_keys=True))
    CavitySolver(config).run(progress=not args.no_progress)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
