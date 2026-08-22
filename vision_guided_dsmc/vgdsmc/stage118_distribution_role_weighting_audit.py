from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .stage118_distribution_role_weighting_core import analyze

GRID = (64, 64)
KNUDSEN = 10.0
RULE = (40, 96)
PAIR_SECTORS = (5, 6)
DOMINANT_RADIAL_SHELL = 1
RADIAL_NODES = 10
PHI_ROLE_SPEED_POWER = 2
PSI_ROLE_SPEED_POWER = 0
PARENT_DECISION = (
    "stage117_stable_single_radial_transition_stage118_distribution_role_weighting_audit"
)


def validate_stage118_design(**overrides: object) -> None:
    frozen = {
        "grid": GRID,
        "kn0": KNUDSEN,
        "rule": RULE,
        "pair_sectors": PAIR_SECTORS,
        "dominant_radial_shell": DOMINANT_RADIAL_SHELL,
        "radial_nodes": RADIAL_NODES,
        "phi_role_speed_power": PHI_ROLE_SPEED_POWER,
        "psi_role_speed_power": PSI_ROLE_SPEED_POWER,
    }
    if any(k not in frozen or frozen[k] != v for k, v in overrides.items()):
        raise ValueError(
            "Stage 118 is fixed; reduced-distribution role powers and retained physical/numerical settings may not be retuned"
        )


def run(parent_dir: str | Path, output_dir: str | Path, **design: object) -> dict:
    validate_stage118_design(**design)
    parent = Path(parent_dir)
    summary = json.loads((parent / "summary.json").read_text())
    if (
        summary.get("stage") != 117
        or summary.get("decision") != PARENT_DECISION
        or summary.get("finite") is not True
    ):
        raise ValueError("Stage-117 parent mismatch")

    with np.load(parent / "radial_transition_profiles.npz") as data:
        phi = np.asarray(data["phi"], dtype=float)
        psi = np.asarray(data["psi"], dtype=float)
        speed = np.asarray(data["node_speed_mean"], dtype=float)

    metrics, aggregate, decision, phi_role, psi_role = analyze(phi, psi, speed)
    out = {
        "stage": 118,
        "finite": True,
        "configuration": {
            "grid": list(GRID),
            "kn0": KNUDSEN,
            "rule": list(RULE),
            "pair_sectors": list(PAIR_SECTORS),
            "dominant_radial_shell": DOMINANT_RADIAL_SHELL,
            "radial_nodes": RADIAL_NODES,
            "phi_role_speed_power": PHI_ROLE_SPEED_POWER,
            "psi_role_speed_power": PSI_ROLE_SPEED_POWER,
            "full_solver_endpoint_rerun": False,
            "physical_parameter_retuning": False,
            "collision_parameter_retuning": False,
            "correction_floor_retuning": False,
            "positivity_floor_retuning": False,
            "source_relaxation_retuning": False,
            "transport_parameter_retuning": False,
            "wall_model_retuning": False,
            "normalization_retuning": False,
            "limiter_retuning": False,
            "velocity_quadrature_retuning": False,
            "failed_muscl_endpoint_rehabilitated": False,
            "cross_knudsen_extension_permitted": False,
            "validation_claim_permitted": False,
            "solver_endpoint_claim_permitted": False,
        },
        "metrics": metrics,
        "aggregate": aggregate,
        "decision": decision,
        "scientific_conclusion": (
            "Stage 118 tests one fixed kinetic-theory explanation for the Stage-117 radial split: "
            "phi is weighted by c_perp^2 because it carries in-plane kinetic energy, while psi "
            "receives no extra speed power because it already represents the integrated transverse "
            "kinetic-energy moment. Alignment after this role weighting would support a kinematic "
            "reduced-distribution explanation; incomplete alignment would reject that explanation "
            "as sufficient. Neither outcome establishes MUSCL-instability causality, solver "
            "convergence, or validation."
        ),
        "design_guard": (
            "The powers 2 and 0 are fixed by the reduced-distribution energy moment, not fitted to "
            "Stage 117. No solver endpoint is advanced and no retained model, wall, quadrature, "
            "limiter, source, transport, or floor setting is changed."
        ),
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n"
    )
    np.savez_compressed(
        output / "distribution_role_weighted_profiles.npz",
        raw_phi=phi,
        raw_psi=psi,
        phi_energy_role=phi_role,
        psi_energy_role=psi_role,
        node_speed_mean=speed,
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run(args.parent_dir, args.output_dir)


if __name__ == "__main__":
    main()
