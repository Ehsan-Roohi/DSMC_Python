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
from typing import Any

from . import jcp_phase1_cavity as jcp2


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


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


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "audit":
        _audit_cli(sys.argv[2:])
        return
    jcp2._qc_selected = _qc_selected_available
    jcp2.main()


if __name__ == "__main__":
    main()
