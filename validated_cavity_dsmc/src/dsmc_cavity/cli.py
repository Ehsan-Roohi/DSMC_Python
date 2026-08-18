"""Command-line interface."""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
import tomllib

from .collisions import SUPPORTED_MODELS
from .config import SimulationConfig
from .solver import CavitySolver


def load_config(path: str | None, overrides: dict) -> SimulationConfig:
    values = {}
    if path:
        with Path(path).open("rb") as stream:
            document = tomllib.load(stream)
        values.update(document.get("simulation", document))
    valid = {field.name for field in fields(SimulationConfig)}
    values.update({key: value for key, value in overrides.items() if value is not None and key in valid})
    return SimulationConfig(**values)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", help="TOML configuration file")
    p.add_argument("--model", choices=SUPPORTED_MODELS)
    p.add_argument("--backend", choices=("cpu", "gpu", "auto"))
    p.add_argument("--kn", type=float)
    p.add_argument("--nx", type=int)
    p.add_argument("--ny", type=int)
    p.add_argument("--particles-per-cell", dest="particles_per_cell", type=int)
    p.add_argument("--steps", type=int)
    p.add_argument("--warmup-steps", dest="warmup_steps", type=int)
    p.add_argument("--sample-stride", dest="sample_stride", type=int)
    p.add_argument("--dt", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--output-dir", dest="output_dir")
    p.add_argument("--no-progress", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    values = vars(args).copy()
    config_path = values.pop("config")
    no_progress = values.pop("no_progress")
    config = load_config(config_path, values)
    solver = CavitySolver(config)
    result = solver.run(progress=not no_progress)
    print(result["output_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
