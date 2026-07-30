from __future__ import annotations

import argparse
import json
from pathlib import Path

from .physical_reference_pipeline import run_pipeline


def build_dataset(
    output_dir: str | Path,
    *,
    quick: bool = False,
    nx: int = 8,
    ny: int = 8,
    particles_per_cell: int = 8,
    dsmc_steps: int = 60,
    nv: int = 10,
    dvm_max_steps: int = 1200,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if quick:
        conditions = [
            (0.05, 25.0, 11),
            (0.10, 25.0, 22),
            (0.05, 45.0, 33),
            (0.10, 45.0, 44),
        ]
    else:
        conditions = [
            (kn, delta, seed)
            for kn in (0.05, 0.10, 0.20)
            for delta in (20.0, 40.0, 60.0)
            for seed in (11, 22)
        ]

    rows: list[dict[str, object]] = []
    for index, (knudsen, delta_temperature, seed) in enumerate(conditions):
        case_dir = output_dir / f"case_{index:03d}_kn{knudsen:.3f}_dt{delta_temperature:.0f}_s{seed}"
        summary = run_pipeline(
            case_dir,
            nx=nx,
            ny=ny,
            particles_per_cell=particles_per_cell,
            dsmc_steps=dsmc_steps,
            nv=nv,
            dvm_max_steps=dvm_max_steps,
            knudsen=knudsen,
            t_left=300.0 + delta_temperature,
            t_right=300.0 - delta_temperature,
            t_top=300.0,
            t_bottom=300.0,
            seed=seed,
        )
        rows.append({
            "index": index,
            "knudsen": knudsen,
            "delta_temperature": delta_temperature,
            "seed": seed,
            "case": str(case_dir / "supervised_case.npz"),
            "score_mean": summary["score_mean"],
            "score_max": summary["score_max"],
            "dvm_iterations": summary["dvm_iterations"],
            "dvm_final_residual": summary["dvm_final_residual"],
        })

    manifest = {
        "model_pair": "physical_vhs_sbt_dsmc_vs_deterministic_bgk_dvm",
        "quick": quick,
        "case_count": len(rows),
        "shape": [ny, nx],
        "conditions": rows,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DSMC/BGK-DVM supervised cases over several conditions")
    parser.add_argument("--output-dir", default="outputs/dvm_supervised_dataset")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--nx", type=int, default=8)
    parser.add_argument("--ny", type=int, default=8)
    parser.add_argument("--ppc", type=int, default=8)
    parser.add_argument("--dsmc-steps", type=int, default=60)
    parser.add_argument("--nv", type=int, default=10)
    parser.add_argument("--dvm-max-steps", type=int, default=1200)
    args = parser.parse_args()
    print(json.dumps(build_dataset(
        args.output_dir,
        quick=args.quick,
        nx=args.nx,
        ny=args.ny,
        particles_per_cell=args.ppc,
        dsmc_steps=args.dsmc_steps,
        nv=args.nv,
        dvm_max_steps=args.dvm_max_steps,
    ), indent=2))


if __name__ == "__main__":
    main()
