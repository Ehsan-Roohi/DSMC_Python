from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from .dvm_bgk import DVMReferenceConfig, save_dvm_reference
from .dvm_shakhov_corrected import ShakhovReferenceConfig, save_shakhov_reference
from .reference_adapter import build_supervised_reference_case
from .sbt_solver import run_physical_cavity
from .vhs_model import PhysicalCavityConfig


def run_pipeline(
    output_dir: str | Path,
    *,
    nx: int = 12,
    ny: int = 12,
    particles_per_cell: int = 12,
    dsmc_steps: int = 160,
    nv: int = 8,
    dvm_max_steps: int = 1800,
    knudsen: float = 0.10,
    t_left: float = 330.0,
    t_right: float = 270.0,
    t_top: float = 300.0,
    t_bottom: float = 300.0,
    seed: int = 7,
    reference_model: str = "shakhov",
) -> dict[str, object]:
    if reference_model not in {"bgk", "shakhov"}:
        raise ValueError("reference_model must be 'bgk' or 'shakhov'")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dsmc_cfg = PhysicalCavityConfig(
        nx=nx,
        ny=ny,
        particles_per_cell=particles_per_cell,
        knudsen=knudsen,
        t_left=t_left,
        t_right=t_right,
        t_top=t_top,
        t_bottom=t_bottom,
        steps=dsmc_steps,
        sample_start=max(1, dsmc_steps // 2),
        seed=seed,
    )
    coarse = run_physical_cavity(dsmc_cfg)
    sigma_t = np.asarray(coarse.get("sigma_T", np.zeros((ny, nx))), dtype=np.float64)
    x = np.stack([coarse["T"], coarse["u"], coarse["v"], sigma_t], axis=0).astype(np.float32)
    mean_wall_temperature = 0.25 * (t_left + t_right + t_top + t_bottom)
    context = np.array(
        [
            knudsen,
            (t_left - t_right) / max(mean_wall_temperature, 1.0e-12),
        ],
        dtype=np.float32,
    )
    coarse_path = output_dir / "coarse_dsmc.npz"
    np.savez_compressed(
        coarse_path,
        x=x,
        context=context,
        case_seed=np.int64(seed),
        **{
            f"coarse_{name}": np.asarray(coarse[name])
            for name in ("T", "rho", "u", "v")
        },
    )

    reference_path = output_dir / f"{reference_model}_reference.npz"
    if reference_model == "shakhov":
        reference_cfg = ShakhovReferenceConfig(
            nx=nx,
            ny=ny,
            nv=nv,
            knudsen=knudsen,
            t_left=t_left,
            t_right=t_right,
            t_top=t_top,
            t_bottom=t_bottom,
            max_steps=dvm_max_steps,
        )
        save_shakhov_reference(reference_path, reference_cfg)
    else:
        reference_cfg = DVMReferenceConfig(
            nx=nx,
            ny=ny,
            nv=nv,
            knudsen=knudsen,
            t_left=t_left,
            t_right=t_right,
            t_top=t_top,
            t_bottom=t_bottom,
            max_steps=dvm_max_steps,
        )
        save_dvm_reference(reference_path, reference_cfg)

    supervised_path = build_supervised_reference_case(
        coarse_path,
        reference_path,
        output_dir / "supervised_case.npz",
    )
    with np.load(reference_path) as reference, np.load(supervised_path) as supervised:
        summary = {
            "model": f"physical_vhs_sbt_dsmc_plus_deterministic_{reference_model}_dvm",
            "reference_model": reference_model,
            "temperature_reconstruction": (
                "inverse_discrete_Maxwellian_second_moment"
                if reference_model == "shakhov"
                else None
            ),
            "coarse_case": str(coarse_path),
            "reference": str(reference_path),
            "supervised_case": str(supervised_path),
            "shape": list(supervised["score"].shape),
            "context": supervised["context"].tolist(),
            "case_seed": int(supervised["case_seed"]),
            "class_counts": np.bincount(supervised["label"].ravel(), minlength=3).tolist(),
            "score_mean": float(np.mean(supervised["score"])),
            "score_max": float(np.max(supervised["score"])),
            "dvm_iterations": int(reference["iterations"]),
            "dvm_final_residual": float(reference["residual_history"][-1]),
            "dvm_left_temperature": float(np.mean(reference["T"][:, 0])),
            "dvm_right_temperature": float(np.mean(reference["T"][:, -1])),
            "dvm_mean_temperature": float(np.mean(reference["T"])),
            "dvm_raw_quadrature_mean_temperature": (
                float(np.mean(reference["T_raw_quadrature"]))
                if "T_raw_quadrature" in reference
                else None
            ),
            "dvm_mean_heat_flux_x": (
                float(np.mean(reference["qx"])) if "qx" in reference else None
            ),
        }
    (output_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build physical DSMC and deterministic kinetic-reference supervision"
    )
    parser.add_argument("--output-dir", default="outputs/physical_reference_pipeline")
    parser.add_argument(
        "--reference-model",
        choices=("shakhov", "bgk"),
        default="shakhov",
    )
    parser.add_argument("--nx", type=int, default=12)
    parser.add_argument("--ny", type=int, default=12)
    parser.add_argument("--ppc", type=int, default=12)
    parser.add_argument("--dsmc-steps", type=int, default=160)
    parser.add_argument("--nv", type=int, default=8)
    parser.add_argument("--dvm-max-steps", type=int, default=1800)
    parser.add_argument("--kn", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(
        json.dumps(
            run_pipeline(
                args.output_dir,
                nx=args.nx,
                ny=args.ny,
                particles_per_cell=args.ppc,
                dsmc_steps=args.dsmc_steps,
                nv=args.nv,
                dvm_max_steps=args.dvm_max_steps,
                knudsen=args.kn,
                seed=args.seed,
                reference_model=args.reference_model,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
