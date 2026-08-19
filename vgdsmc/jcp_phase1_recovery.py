"""QC-safe recovery entry point for the frozen JCP2 cavity experiment.

The JCP2 protocol preregistered primary seeds followed by spare seeds and
explicitly requires a failed primary to be replaced by the first passing
spare.  The frozen implementation correctly handled a completed seed that
failed its mechanical checks, but raised immediately when an array task left
no artifact directory.  This wrapper preserves the frozen scientific code and
seed order while treating missing, incomplete, or mechanically invalid runs
as rejected candidates.

Predictions still inspect evaluation runs only.  Reference selection occurs
inside the frozen score function, after that function verifies the prediction
lock and hash.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

from . import jcp_phase1_cavity as jcp2


SHIFT_CONDITION_ID = "S2_kn0p085_u350"
SHIFT_KNUDSEN = 0.085
SHIFT_LID_SPEED_M_PER_S = 350.0
RECOVERY_EVALUATION_SCOPE = "JCP2_condition_independent_completion_QC"
CORE_COMPLETION_FIELDS = ("rho", "u", "v", "T", "qx", "qy")
FINAL_ARTIFACT_NAMES = (
    "fields.npz",
    "block_fields.npz",
    "summary.json",
    "artifact_manifest.json",
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _finite_range(values: Any) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    return {
        "shape": list(array.shape),
        "all_finite": bool(np.all(finite)),
        "minimum": float(np.min(array[finite])) if np.any(finite) else "nan",
        "maximum": float(np.max(array[finite])) if np.any(finite) else "nan",
    }


def shifted_condition_evaluation(
    fields: Mapping[str, Any],
    cfg: Any,
) -> dict[str, Any]:
    """Report condition-independent completion diagnostics for locked JCP2 S2.

    The spatial-refinement engine historically called a profile validator that
    is valid only at Kn=0.05 and U_wall=100 m/s. JCP2 was prospectively locked
    at a different condition, so applying those profiles would be scientifically
    invalid. This evaluator deliberately makes no external-validation claim;
    seed acceptance remains governed by the engine's preregistered mechanical
    and stationarity checks.
    """

    if not np.isclose(float(cfg.knudsen), SHIFT_KNUDSEN) or not np.isclose(
        float(cfg.lid_velocity_x), SHIFT_LID_SPEED_M_PER_S
    ):
        raise ValueError(
            "checkpoint recovery accepts only locked JCP2 S2 "
            f"(Kn={SHIFT_KNUDSEN}, U_wall={SHIFT_LID_SPEED_M_PER_S:g} m/s)"
        )
    missing = [name for name in CORE_COMPLETION_FIELDS if name not in fields]
    if missing:
        raise ValueError(f"JCP2 recovered fields are incomplete: missing={missing}")

    diagnostics = {
        name: _finite_range(fields[name]) for name in CORE_COMPLETION_FIELDS
    }
    shapes = {tuple(record["shape"]) for record in diagnostics.values()}
    checks = {
        "core_fields_present": not missing,
        "core_fields_share_grid_shape": len(shapes) == 1
        and shapes == {(int(cfg.ny), int(cfg.nx))},
        "core_fields_finite": all(
            bool(record["all_finite"]) for record in diagnostics.values()
        ),
        "density_positive": bool(np.all(np.asarray(fields["rho"]) > 0.0)),
        "temperature_positive": bool(np.all(np.asarray(fields["T"]) > 0.0)),
    }
    return {
        "scope": RECOVERY_EVALUATION_SCOPE,
        "condition_id": SHIFT_CONDITION_ID,
        "knudsen": float(cfg.knudsen),
        "lid_speed_m_per_s": float(cfg.lid_velocity_x),
        "external_validation_claim": False,
        "legacy_kn0p05_u100_reference_applied": False,
        "seed_acceptance_source": (
            "preregistered event-mechanics, majorant, field-completeness, "
            "lid-coverage, stationarity, and checkpoint-roundtrip checks"
        ),
        "field_diagnostics": diagnostics,
        "checks": checks,
        "all_completion_diagnostics_pass": bool(all(checks.values())),
        "decision": "retain_locked_JCP2_mechanical_and_stationarity_QC",
    }


def _zero_crossing_count(values: Any) -> int:
    array = np.asarray(values, dtype=np.float64).ravel()
    signs = np.sign(array[np.isfinite(array)])
    signs = signs[signs != 0.0]
    if len(signs) < 2:
        return 0
    return int(np.count_nonzero(signs[1:] != signs[:-1]))


def shifted_heat_flux_topology(
    fields: Mapping[str, Any],
    cfg: Any,
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    """Record reference-free centerline heat-flux topology for provenance."""

    if evaluation.get("scope") != RECOVERY_EVALUATION_SCOPE:
        raise ValueError("unexpected JCP2 recovery evaluation scope")
    qx = np.asarray(fields["qx"], dtype=np.float64)
    qy = np.asarray(fields["qy"], dtype=np.float64)
    center_x = int(cfg.nx) // 2
    center_y = int(cfg.ny) // 2
    near_y_0p8 = min(
        int(cfg.ny) - 1,
        max(0, int(round(0.8 * int(cfg.ny) - 0.5))),
    )
    return {
        "scope": "reference_free_sign_topology",
        "qx_vertical_centerline": _zero_crossing_count(qx[:, center_x]),
        "qy_horizontal_centerline": _zero_crossing_count(qy[center_y, :]),
        "qy_near_y_over_l_0p8": _zero_crossing_count(qy[near_y_0p8, :]),
    }


def _preserve_partial_final_artifacts(directory: Path) -> Path | None:
    existing = [directory / name for name in FINAL_ARTIFACT_NAMES]
    existing = [path for path in existing if path.exists()]
    if not existing:
        return None
    backup = directory / "pre_checkpoint_recovery_partial"
    suffix = 1
    while backup.exists():
        suffix += 1
        backup = directory / f"pre_checkpoint_recovery_partial_{suffix}"
    backup.mkdir(parents=True)
    for path in existing:
        path.replace(backup / path.name)
    return backup


def recover_checkpoint(
    *,
    output_root: Path,
    group: str,
    seed: int,
) -> dict[str, Any]:
    """Resume a locked JCP2 checkpoint and create the missing final artifacts."""

    output_root = Path(output_root)
    seed = int(seed)
    if seed not in jcp2.group_seeds(group):
        raise ValueError(f"seed {seed} is not preregistered for JCP2 {group}")
    directory = output_root / group / f"seed_{seed}"
    try:
        existing = jcp2._verify_artifacts(directory)
    except (OSError, ValueError, KeyError, TypeError):
        existing = None
    existing_evaluation = existing.get("evaluation", {}) if existing else {}
    if (
        existing is not None
        and existing_evaluation.get("scope") == RECOVERY_EVALUATION_SCOPE
        and existing_evaluation.get("all_completion_diagnostics_pass") is True
        and existing_evaluation.get("legacy_kn0p05_u100_reference_applied") is False
    ):
        return {
            "status": "JCP2_checkpoint_recovery_already_complete",
            "group": group,
            "seed": seed,
            "directory": str(directory),
        }

    checkpoint_path = directory / "checkpoint.npz"
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size <= 0:
        raise FileNotFoundError(f"JCP2 checkpoint is missing: {checkpoint_path}")

    cfg, _, _, _, _ = jcp2.stage_configuration(f"{jcp2.STAGE}::{group}", seed)
    from .ntc_checkpoint import load_ntc_checkpoint

    checkpoint = load_ntc_checkpoint(checkpoint_path, cfg)
    if int(checkpoint.step_index) > int(cfg.steps):
        raise ValueError("JCP2 checkpoint step exceeds the locked trajectory length")
    resumed_from_step = int(checkpoint.step_index)
    del checkpoint
    partial_backup = _preserve_partial_final_artifacts(directory)

    engine = jcp2.engine
    original_evaluation = engine.evaluate_mohammadzadeh_fields
    original_topology = engine._heat_flux_zero_crossing_counts
    engine.evaluate_mohammadzadeh_fields = shifted_condition_evaluation
    engine._heat_flux_zero_crossing_counts = shifted_heat_flux_topology
    try:
        summary = jcp2.run_seed(
            group=group,
            seed=seed,
            output_root=output_root,
            resume=True,
        )
    finally:
        engine.evaluate_mohammadzadeh_fields = original_evaluation
        engine._heat_flux_zero_crossing_counts = original_topology

    verified = jcp2._verify_artifacts(directory)
    evaluation = verified.get("evaluation", {})
    if evaluation.get("scope") != RECOVERY_EVALUATION_SCOPE:
        raise ValueError("recovered JCP2 summary lacks the recovery evaluation scope")
    if evaluation.get("legacy_kn0p05_u100_reference_applied") is not False:
        raise ValueError("recovered JCP2 summary has an invalid validation provenance")
    if evaluation.get("all_completion_diagnostics_pass") is not True:
        raise ValueError("recovered JCP2 fields failed completion diagnostics")
    return {
        "status": "JCP2_checkpoint_recovery_complete",
        "group": group,
        "seed": seed,
        "resumed_from_step": resumed_from_step,
        "completed_step": int(cfg.steps),
        "remaining_steps_executed": int(cfg.steps) - resumed_from_step,
        "mechanical_checks": summary.get("mechanical_checks", {}),
        "decision": summary.get("decision"),
        "directory": str(directory),
        "preserved_partial_artifacts": (
            str(partial_backup) if partial_backup is not None else None
        ),
    }


def qc_audit(output_root: Path, group: str, required: int) -> dict[str, Any]:
    """Select the first required passing seeds in preregistered order."""

    output_root = Path(output_root)
    required = int(required)
    seed_bank = jcp2.load_seed_bank()
    if group == "evaluation":
        primary_count = len(seed_bank["evaluation_primary"])
    elif group == "reference":
        primary_count = len(seed_bank["reference_primary"])
    else:
        raise ValueError("JCP2 group must be evaluation or reference")

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    candidates = jcp2.group_seeds(group)
    for order, seed in enumerate(candidates):
        directory = output_root / group / f"seed_{seed}"
        role = "primary" if order < primary_count else "spare"
        try:
            summary = jcp2._verify_artifacts(directory)
            recorded_seed = int(summary.get("seed", -1))
            if recorded_seed != int(seed):
                raise ValueError(
                    f"summary seed {recorded_seed} differs from candidate seed {seed}"
                )
            checks = summary.get("mechanical_checks", {})
            failed_checks = sorted(
                str(name) for name, value in checks.items() if not bool(value)
            )
            if not checks:
                raise ValueError("mechanical_checks is empty")
            if failed_checks:
                rejected.append(
                    {
                        "order": order,
                        "seed": int(seed),
                        "role": role,
                        "reason": "mechanical_checks_failed",
                        "failed_checks": failed_checks,
                    }
                )
                continue
        except (OSError, ValueError, KeyError, TypeError) as error:
            rejected.append(
                {
                    "order": order,
                    "seed": int(seed),
                    "role": role,
                    "reason": type(error).__name__,
                    "detail": str(error),
                }
            )
            continue

        accepted.append(
            {
                "order": order,
                "seed": int(seed),
                "role": role,
                "directory": str(directory),
            }
        )
        if len(accepted) == required:
            break

    audit = {
        "stage": jcp2.STAGE,
        "group": group,
        "selection_rule": "first QC-pass candidates in locked primary-then-spare order",
        "required": required,
        "accepted_count": len(accepted),
        "accepted": accepted,
        "rejected_before_selection_completed": rejected,
        "candidate_count": len(candidates),
        "selection_complete": len(accepted) == required,
    }
    audit_root = os.environ.get("JCP2_SELECTION_AUDIT_ROOT")
    if audit_root:
        _atomic_json(
            Path(audit_root) / f"{group}_selection_audit.json",
            audit,
        )
    if len(accepted) != required:
        raise ValueError(
            f"JCP2 has only {len(accepted)} passing {group} seeds; "
            f"{required} are required; rejected={rejected}"
        )
    return audit


def _qc_selected_available(
    output_root: Path,
    group: str,
    required: int,
) -> list[Path]:
    audit = qc_audit(output_root, group, required)
    return [Path(record["directory"]) for record in audit["accepted"]]


def _audit_cli(arguments: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Audit JCP2 primary/spare selection without running prediction or score"
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--group", choices=("evaluation", "reference"), required=True)
    parser.add_argument("--required", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(arguments)
    audit = qc_audit(args.run_root, args.group, args.required)
    if args.output is not None:
        _atomic_json(args.output, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


def _recover_checkpoint_cli(arguments: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Resume a JCP2 trajectory checkpoint with S2-safe completion QC"
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--group", choices=("evaluation", "reference"), required=True
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--task-index", type=int)
    selection.add_argument("--seed", type=int)
    args = parser.parse_args(arguments)
    seed = (
        int(args.seed)
        if args.seed is not None
        else jcp2.task_from_index(args.group, int(args.task_index))
    )
    result = recover_checkpoint(
        output_root=args.run_root,
        group=args.group,
        seed=seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "audit":
        _audit_cli(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "recover-checkpoint":
        _recover_checkpoint_cli(sys.argv[2:])
        return
    jcp2._qc_selected = _qc_selected_available
    jcp2.main()


if __name__ == "__main__":
    main()
