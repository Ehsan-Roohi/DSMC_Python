from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np

from vgdsmc import mohammadzadeh_mv12_sage_qy as mv12


def test_protocol_lock() -> None:
    record = mv12.verify_lock()
    assert record["stage"] == mv12.STAGE
    assert record["legacy_labels_forbidden_during_prediction"] is True
    assert tuple(record["experts"]) == mv12.EXPERT_NAMES


def test_simplex_projection() -> None:
    projected = mv12.project_simplex(np.asarray((1.3, -0.4, 0.2, 2.0)))
    assert np.min(projected) >= 0.0
    assert abs(float(projected.sum()) - 1.0) < 1.0e-12
    np.testing.assert_allclose(
        mv12.project_simplex(np.ones(6) / 6.0), np.ones(6) / 6.0
    )


def test_convex_fit_recovers_safe_mixture() -> None:
    rng = np.random.default_rng(20260813)
    base = rng.normal(size=(7, 5, 4))
    experts = np.stack(
        (
            base + 0.8,
            base - 0.4,
            base + 0.05,
            base + rng.normal(scale=0.2, size=base.shape),
            base - 0.1,
            base + 0.3,
        )
    )
    target = 0.25 * experts[1] + 0.75 * experts[2]
    weights = mv12.fit_convex_weights(experts, target, 2, 0.0)
    prediction = np.tensordot(weights, experts, axes=(0, 0))
    assert mv12.component_nrmse(prediction, target) < 1.0e-7
    assert np.min(weights) >= -1.0e-12
    assert abs(float(weights.sum()) - 1.0) < 1.0e-10


def test_global_gate_transfers_across_disjoint_conditions_without_test_labels() -> None:
    protocol = mv12.locked_protocol()
    rng = np.random.default_rng(260812)
    target = rng.normal(size=(8, 6, 6))
    validation = np.stack(
        (
            target + rng.normal(scale=0.5, size=target.shape),
            target + rng.normal(scale=0.25, size=target.shape),
            target + rng.normal(scale=0.05, size=target.shape),
            target + rng.normal(scale=0.3, size=target.shape),
            target + rng.normal(scale=0.2, size=target.shape),
            target + rng.normal(scale=0.4, size=target.shape),
        )
    )
    validation_conditions = np.repeat(
        np.asarray(mv12.DEVELOPMENT_CONDITIONS, dtype="U32"), 2
    )
    test = validation[:, :4].copy()
    test_conditions = np.asarray(mv12.EVALUATION_CONDITIONS, dtype="U32")
    test[:, 1] += np.arange(6, dtype=np.float64)[:, None, None] * 20.0
    prediction, record = mv12.fit_global_gate(
        validation,
        target,
        validation_conditions,
        test,
        test_conditions,
        protocol,
    )
    assert prediction.shape == test.shape[1:]
    assert record["simplex_verified"] is True
    assert record["test_abstention_count"] >= 1
    assert set(record["development_validation_conditions"]).isdisjoint(
        record["evaluation_conditions"]
    )
    assert len(record["leave_one_development_condition_out_selection"]) == 8
    assert all(value >= -1.0e-12 for value in record["final_weights"].values())
    assert abs(sum(record["final_weights"].values()) - 1.0) < 1.0e-10


def test_manifest_verification() -> None:
    with tempfile.TemporaryDirectory() as directory_name:
        directory = Path(directory_name)
        payload = directory / "payload.txt"
        payload.write_text("locked\n", encoding="utf-8")
        manifest = {
            "files": {
                "payload.txt": {
                    "sha256": mv12._sha256(payload),
                    "size_bytes": payload.stat().st_size,
                }
            }
        }
        (directory / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        verified = mv12._verify_manifest(directory, "manifest.json")
        assert verified == manifest


def test_data_contract_verifies_disjoint_identities_without_targets() -> None:
    with tempfile.TemporaryDirectory() as directory_name:
        directory = Path(directory_name)
        np.savez_compressed(
            directory / "dataset.npz",
            validation_condition=np.repeat(
                np.asarray(mv12.DEVELOPMENT_CONDITIONS, dtype="U32"), 10
            ),
            test_condition=np.repeat(
                np.asarray(mv12.EVALUATION_CONDITIONS, dtype="U32"), 40
            ),
        )
        record = mv12.verify_data_contract(directory)
        assert record["development_sample_count"] == 40
        assert record["evaluation_sample_count"] == 160
        assert record["checks"]["targets_not_loaded"] is True
        assert all(record["checks"].values())


def test_slurm_jobs_reuse_the_verified_mv10_torch_environment() -> None:
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    submit = (scripts / "submit_mohammadzadeh_mv12_sage_unity.sh").read_text(
        encoding="utf-8"
    )
    predict = (scripts / "unity_mohammadzadeh_mv12_sage_predict.sbatch").read_text(
        encoding="utf-8"
    )
    post = (scripts / "unity_mohammadzadeh_mv12_sage_post.sbatch").read_text(
        encoding="utf-8"
    )
    assert 'MV12_VENV_DIR=${MV10_VENV_DIR}' in submit
    assert '"${MV10_VENV_DIR}/bin/python" -c' in submit
    assert 'verify-data --mv10-output-root "${MV10_OUTPUT_ROOT}"' in submit
    assert 'source "${MV12_VENV_DIR}/bin/activate"' in predict
    assert 'import torch' in predict
    assert 'source "${MV12_VENV_DIR}/bin/activate"' in post


def main() -> None:
    test_protocol_lock()
    test_simplex_projection()
    test_convex_fit_recovers_safe_mixture()
    test_global_gate_transfers_across_disjoint_conditions_without_test_labels()
    test_manifest_verification()
    test_data_contract_verifies_disjoint_identities_without_targets()
    test_slurm_jobs_reuse_the_verified_mv10_torch_environment()
    print("MV12_SAGE_TESTS_PASS count=7")


if __name__ == "__main__":
    main()
