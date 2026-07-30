from __future__ import annotations

import argparse
import json
import numpy as np

from .dvm_bgk import DVMReferenceConfig, save_dvm_reference


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a deterministic BGK-DVM thermal-cavity reference")
    parser.add_argument("--output", default="outputs/dvm/reference.npz")
    parser.add_argument("--nx", type=int, default=16)
    parser.add_argument("--ny", type=int, default=16)
    parser.add_argument("--nv", type=int, default=12)
    parser.add_argument("--kn", type=float, default=0.10)
    parser.add_argument("--t-left", type=float, default=330.0)
    parser.add_argument("--t-right", type=float, default=270.0)
    parser.add_argument("--t-top", type=float, default=300.0)
    parser.add_argument("--t-bottom", type=float, default=300.0)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--tolerance", type=float, default=2.0e-6)
    args = parser.parse_args()
    cfg = DVMReferenceConfig(
        nx=args.nx,
        ny=args.ny,
        nv=args.nv,
        knudsen=args.kn,
        t_left=args.t_left,
        t_right=args.t_right,
        t_top=args.t_top,
        t_bottom=args.t_bottom,
        max_steps=args.max_steps,
        tolerance=args.tolerance,
    )
    path = save_dvm_reference(args.output, cfg)
    with np.load(path) as data:
        summary = {
            "output": str(path),
            "iterations": int(data["iterations"]),
            "final_residual": float(data["residual_history"][-1]),
            "mean_temperature": float(np.mean(data["T"])),
            "left_temperature": float(np.mean(data["T"][:, 0])),
            "right_temperature": float(np.mean(data["T"][:, -1])),
        }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
