from __future__ import annotations
import argparse
from .dataset import generate_case
from .simulator import CavityConfig


def main() -> None:
    p = argparse.ArgumentParser(description="Generate a vision-guided DSMC pilot dataset")
    p.add_argument("--output", default="outputs/pilot")
    p.add_argument("--nx", type=int, default=24)
    p.add_argument("--ny", type=int, default=24)
    p.add_argument("--ppc", type=int, default=20)
    p.add_argument("--reference-ppc", type=int, default=120)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--sample-start", type=int, default=120)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    cfg = CavityConfig(nx=a.nx, ny=a.ny, particles_per_cell=a.ppc, steps=a.steps, sample_start=a.sample_start, seed=a.seed)
    path = generate_case(a.output, cfg, a.reference_ppc)
    print(path)


if __name__ == "__main__":
    main()
