from __future__ import annotations

from pathlib import Path
import argparse
import json

from .lowfreq_closed_loop import ClosedLoopConfig, run_closed_loop_benchmark


DEFAULT_FOCUSED_SEEDS = (66, 77, 88, 99, 111, 122, 133, 144, 155, 166)


def run_focused_benchmark(
    model_path: str | Path,
    output_dir: str | Path,
    *,
    seeds: tuple[int, ...] = DEFAULT_FOCUSED_SEEDS,
    amplitude: float = 0.05,
    nx: int = 6,
    ny: int = 6,
    particles_per_cell: int = 20,
    warm_steps: int = 40,
    continuation_steps: int = 60,
    nv: int = 6,
    dvm_max_steps: int = 900,
) -> dict[str, object]:
    summary = run_closed_loop_benchmark(
        model_path,
        output_dir,
        ClosedLoopConfig(
            nx=nx,
            ny=ny,
            particles_per_cell=particles_per_cell,
            warm_steps=warm_steps,
            continuation_steps=continuation_steps,
            nv=nv,
            dvm_max_steps=dvm_max_steps,
            amplitudes=(amplitude,),
            seeds=seeds,
        ),
    )
    summary["stage"] = 18
    summary["description"] = (
        "Focused five-percent matched-budget closed loop on ten unseen seeds"
    )
    summary["focus_amplitude"] = amplitude
    summary["scientific_guard"] = (
        "A benefit is claimed only if the paired 95% interval lies entirely "
        "below one and the result is not driven by a single Knudsen condition."
    )
    output_dir = Path(output_dir)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a focused low-amplitude closed-loop ensemble on unseen seeds"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", default="outputs/stage18_focused_closed_loop")
    parser.add_argument("--amplitude", type=float, default=0.05)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_FOCUSED_SEEDS))
    parser.add_argument("--nx", type=int, default=6)
    parser.add_argument("--ny", type=int, default=6)
    parser.add_argument("--ppc", type=int, default=20)
    parser.add_argument("--warm-steps", type=int, default=40)
    parser.add_argument("--continuation-steps", type=int, default=60)
    parser.add_argument("--nv", type=int, default=6)
    parser.add_argument("--dvm-max-steps", type=int, default=900)
    args = parser.parse_args()
    summary = run_focused_benchmark(
        args.model,
        args.output_dir,
        seeds=tuple(args.seeds),
        amplitude=args.amplitude,
        nx=args.nx,
        ny=args.ny,
        particles_per_cell=args.ppc,
        warm_steps=args.warm_steps,
        continuation_steps=args.continuation_steps,
        nv=args.nv,
        dvm_max_steps=args.dvm_max_steps,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
