from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from vgdsmc import mohammadzadeh_mv14_kinetic_conservation_cavity as mv14


def test_protocol_lock_and_no_continuum_closure() -> None:
    record = mv14.verify_lock()
    assert record["stage"] == mv14.STAGE
    assert record["Fourier_law_used"] is False
    assert record["Navier_Stokes_closure_used"] is False
    assert record["wall_heat_flux_imposed"] is False
    assert record["DSMC_rerun"] is False


def test_raw_additive_moments_reconstruct_direct_kinetic_fields() -> None:
    velocities = np.asarray(
        (
            (2.0, -1.0, 0.5),
            (-1.0, 0.0, -0.5),
            (0.5, 2.0, 1.0),
            (1.0, -0.5, -1.0),
        )
    )
    speed2 = np.sum(velocities**2, axis=1)
    m0 = np.asarray((len(velocities),), dtype=np.float64)
    payload = {
        "samples": 1,
        "m0": m0,
        "m1": np.sum(velocities, axis=0)[None, :],
        "m2": np.einsum("ni,nj->ij", velocities, velocities)[None, :, :],
        "energy": np.asarray((np.sum(speed2),)),
        "energy_velocity": np.sum(speed2[:, None] * velocities, axis=0)[None, :],
    }
    cfg = SimpleNamespace(
        nx=1,
        ny=1,
        cell_volume=1.0,
        number_density=float(len(velocities)),
        t0=1.0,
        lid_velocity_x=2.0,
        vhs=SimpleNamespace(mass=2.0),
    )
    result = mv14.kinetic_fields_from_payload(payload, cfg, 1.0)
    mean = np.mean(velocities, axis=0)
    peculiar = velocities - mean
    number_density = float(len(velocities))
    pressure = 2.0 * number_density * np.mean(
        np.einsum("ni,nj->nij", peculiar, peculiar), axis=0
    )
    q = 0.5 * 2.0 * number_density * np.mean(
        np.sum(peculiar**2, axis=1)[:, None] * peculiar, axis=0
    )
    p_ref = number_density
    q_ref = p_ref * np.sqrt(0.5)
    np.testing.assert_allclose(result["pxx"][0, 0], pressure[0, 0] / p_ref)
    np.testing.assert_allclose(result["pxy"][0, 0], pressure[0, 1] / p_ref)
    np.testing.assert_allclose(result["pxz"][0, 0], pressure[0, 2] / p_ref)
    np.testing.assert_allclose(result["pyy"][0, 0], pressure[1, 1] / p_ref)
    np.testing.assert_allclose(result["pyz"][0, 0], pressure[1, 2] / p_ref)
    np.testing.assert_allclose(result["pzz"][0, 0], pressure[2, 2] / p_ref)
    np.testing.assert_allclose(result["w"][0, 0], mean[2] / cfg.lid_velocity_x)
    np.testing.assert_allclose(result["qx"][0, 0], q[0] / q_ref)
    np.testing.assert_allclose(result["qy"][0, 0], q[1] / q_ref)


def test_uniform_equilibrium_has_zero_exact_energy_rhs() -> None:
    shape = (9, 11)
    fields = {
        "rho": np.ones(shape),
        "u": np.zeros(shape),
        "v": np.zeros(shape),
        "w": np.zeros(shape),
        "temperature": np.ones(shape),
        "pxx": np.ones(shape),
        "pxy": np.zeros(shape),
        "pxz": np.zeros(shape),
        "pyy": np.ones(shape),
        "pyz": np.zeros(shape),
        "beta": 0.5,
    }
    rhs = mv14.exact_energy_rhs(fields, np.zeros(shape), macro_smoothing_passes=1)
    np.testing.assert_allclose(rhs, 0.0, atol=1.0e-14)


def test_weak_gls_reduces_balance_error_without_imposing_wall_value() -> None:
    ny, nx = 17, 7
    y = np.linspace(0.0, 1.0, ny)[:, None]
    exact = np.broadcast_to(0.4 * y**2 - 0.2 * y + 0.07, (ny, nx))
    rhs = np.broadcast_to(0.8 * y - 0.2, (ny, nx))
    observation = exact.copy()
    observation[3:-2] += 0.08 * np.sin(4.0 * np.pi * y[3:-2])
    observation[0] += 0.035
    observation[-1] -= 0.025
    projected, diagnostics = mv14.weak_gls_project_qy(
        observation,
        rhs,
        np.ones_like(observation),
        span_name="multiscale",
        lambda_strength=10.0,
    )
    assert diagnostics["weak_residual_ratio"] < 1.0
    assert diagnostics["bottom_boundary_correction_rms"] > 0.0
    assert diagnostics["top_boundary_correction_rms"] > 0.0
    assert not np.allclose(projected[0], 0.0)
    assert not np.allclose(projected[-1], 0.0)


def test_zero_strength_is_exact_observation_identity() -> None:
    rng = np.random.default_rng(20260814)
    qy = rng.normal(size=(8, 9))
    projected, _ = mv14.weak_gls_project_qy(
        qy,
        rng.normal(size=qy.shape),
        np.ones_like(qy),
        span_name="local",
        lambda_strength=0.0,
    )
    np.testing.assert_allclose(projected, qy, atol=1.0e-14)


def test_prediction_stage_cannot_index_legacy_targets() -> None:
    tree = ast.parse(inspect.getsource(mv14.run_prediction_stage))
    forbidden = {"test_y", "test_target10", "validation_target10", "test_raw10"}
    indexed = {
        node.slice.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    }
    assert not (indexed & forbidden)


def test_five_arm_ablation_and_machine_vision_claim_are_locked() -> None:
    protocol = json.loads(mv14.protocol_path().read_text(encoding="utf-8"))
    arms = protocol["ablation_contract"]["arms"]
    assert len(arms) == 5
    assert any("vision_only" in arm for arm in arms)
    assert any("physics_only" in arm for arm in arms)
    assert any("vision_plus" in arm for arm in arms)
    assert "beat both vision-only and physics-only" in protocol["ablation_contract"]["machine_vision_contribution_rule"]


def test_protocol_preserves_antifourier_and_forbids_pressure_closure() -> None:
    physics = json.loads(mv14.protocol_path().read_text(encoding="utf-8"))["kinetic_physics_contract"]
    assert physics["Fourier_law_used"] is False
    assert physics["Newtonian_stress_law_used"] is False
    assert physics["Pzz_equals_p_closure_used"] is False
    assert physics["anti_Fourier_heat_transfer_permitted"] is True
    assert physics["pointwise_hard_PDE_projection_used"] is False


def test_submit_chain_is_postprocessing_only() -> None:
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    submit = (scripts / "submit_mohammadzadeh_mv14_kinetic_cavity_unity.sh").read_text(encoding="utf-8")
    predict = (scripts / "unity_mohammadzadeh_mv14_kinetic_predict.sbatch").read_text(encoding="utf-8")
    assert "LAST_MOHAMMADZADEH_MV10_QY_JOB.env" in submit
    assert "LAST_MOHAMMADZADEH_MV12_SAGE_QY_JOB.env" in submit
    assert "--mv9-output-root" in predict
    assert "--mv12-output-root" in predict
    assert "DSMC trajectory" not in submit


def main() -> None:
    test_protocol_lock_and_no_continuum_closure()
    test_raw_additive_moments_reconstruct_direct_kinetic_fields()
    test_uniform_equilibrium_has_zero_exact_energy_rhs()
    test_weak_gls_reduces_balance_error_without_imposing_wall_value()
    test_zero_strength_is_exact_observation_identity()
    test_prediction_stage_cannot_index_legacy_targets()
    test_five_arm_ablation_and_machine_vision_claim_are_locked()
    test_protocol_preserves_antifourier_and_forbids_pressure_closure()
    test_submit_chain_is_postprocessing_only()
    print("MV14_KINETIC_CONSERVATION_TESTS_PASS count=9")


if __name__ == "__main__":
    main()
