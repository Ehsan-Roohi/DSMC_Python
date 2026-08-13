from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "make_mohammadzadeh_mv8_audit_figures.py"
SPEC = importlib.util.spec_from_file_location("mv8_audit_figures", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_audit_figure_suite(tmp_path: Path) -> None:
    mv8 = tmp_path / "mv8"
    repo = tmp_path / "repo"
    output = tmp_path / "figures"
    mv8.mkdir()
    protocol_dir = repo / "reference_data" / "mohammadzadeh_2012"
    protocol_dir.mkdir(parents=True)
    assembly = {
        "status": "complete_MV8_additive_moment_assembly_and_information_gate",
        "decision": "hold_MV8_models_return_information_audit_only",
        "checks": {
            "all_reconstructed_moment_fields_finite": True,
            "pressure_covariance_positive_semidefinite": True,
            "block_sums_match_full_additive_accumulators_with_fixed_scale_tolerance": True,
            "stored_and_reconstructed_heat_flux_match": False,
        },
        "development_validation_information_test": {
            "individual_fields_improved": 4,
            "raw_B1": {
                "composite_nrmse": 0.32,
                "per_field_nrmse": {name: 0.32 + index * 0.01 for index, name in enumerate(MODULE.OUTPUT_FIELDS)},
            },
            "raw_B10": {
                "composite_nrmse": 0.12,
                "per_field_nrmse": {name: 0.12 + index * 0.01 for index, name in enumerate(MODULE.OUTPUT_FIELDS)},
            },
        },
        "maximum_block_full_additive_moment_fixed_scale_relative_linf": 5.34e-14,
        "maximum_q_reconstruction_relative_difference": 5.78e-8,
        "minimum_covariance_eigenvalue_ratio": 0.449,
    }
    (mv8 / "assembly_summary.json").write_text(json.dumps(assembly), encoding="utf-8")
    protocol = {
        "moment_contract": {"outputs": list(MODULE.OUTPUT_FIELDS)},
        "execution_matrix": {
            "primary_condition": "kn0p1_u400",
            "representative_contour_seed": 94302,
            "representative_contour_block": 0,
        },
        "pre_model_feasibility_gates": {
            "block_full_additive_moment_fixed_scale_relative_linf_tolerance": 1e-9,
            "stored_and_reconstructed_heat_flux_relative_tolerance": 1e-10,
        },
    }
    (protocol_dir / "mv8_kinetic_moment_feasibility_protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    rng = np.random.default_rng(7)
    reference = rng.normal(0.0, 0.2, size=(1, 4, 12, 12)).astype(np.float32)
    raw_b1 = reference + rng.normal(0.0, 0.06, size=reference.shape).astype(np.float32)
    raw_b10 = reference + rng.normal(0.0, 0.02, size=reference.shape).astype(np.float32)
    test_x = np.concatenate((raw_b1, np.zeros((1, 6, 12, 12), dtype=np.float32)), axis=1)
    np.savez_compressed(
        mv8 / "dataset.npz",
        test_x=test_x,
        test_y=reference,
        test_condition=np.asarray(["kn0p1_u400"]),
        test_identity=np.asarray([[94302, 0, 1]], dtype=np.int64),
        test_scale=np.asarray([[2.0, 2.0, 3.0, 3.0]], dtype=np.float64),
        test_gaussian=reference + 0.04,
        test_tsvd=reference + 0.03,
        test_raw10=raw_b10,
        test_target10=reference.copy(),
        test_condition10=np.asarray(["kn0p1_u400"]),
        test_identity10=np.asarray([[94302, 0, 10]], dtype=np.int64),
        test_scale10=np.asarray([[2.0, 2.0, 3.0, 3.0]], dtype=np.float64),
    )
    result = MODULE.run(mv8, output, repo, None)
    assert result["physical_figures"] == 4
    assert result["neural_predictions_included"] is False
    assert (output / "mv8_audit_qy_physical_kn0p1_u400.pdf").is_file()
    assert (output / "mv8_audit_gate_and_information_summary.png").is_file()
    assert (output / "MOHAMMADZADEH_MV8_AUDIT_PHYSICAL_FIGURES.tar.gz").is_file()
    metadata = json.loads((output / "figure_metadata.json").read_text(encoding="utf-8"))
    assert metadata["neural_predictions_included"] is False

