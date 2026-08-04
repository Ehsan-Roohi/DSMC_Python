import json

import numpy as np

from vgdsmc.mohammadzadeh_validation import (
    SOURCE_PDF_SHA256,
    _profile_at_x,
    _profile_at_y,
    evaluate_mohammadzadeh_fields,
    mohammadzadeh_config,
    reference_directory,
)


def test_locked_protocol_and_source_hash_are_consistent():
    protocol = json.loads(
        (reference_directory() / "validation_protocol.json").read_text()
    )
    metadata = json.loads(
        (reference_directory() / "benchmark_metadata.json").read_text()
    )
    assert protocol["anti_circularity"]["source_pdf_sha256"] == SOURCE_PDF_SHA256
    assert metadata["source_pdf_sha256"] == SOURCE_PDF_SHA256
    assert protocol["primary_case"]["knudsen"] == 0.05
    assert protocol["primary_case"]["grid"] == [200, 200]


def test_m1_execution_lock_inherits_m0_and_keeps_seed_banks_disjoint():
    ref_dir = reference_directory()
    m0 = json.loads((ref_dir / "validation_protocol.json").read_text())
    m1 = json.loads((ref_dir / "m1_execution_protocol.json").read_text())
    seeds = json.loads((ref_dir / "m1_seed_bank.json").read_text())

    assert m1["scientific_gate_changes"] == "none"
    assert m0["protocol_version"] in m1["inherits_scientific_gates_from"]
    assert m1["runtime_contract"]["sample_stride"] == 20
    assert m1["runtime_contract"]["nonoverlapping_sampling_blocks"] == 8

    development = {
        seed
        for group in seeds["development"].values()
        for seed in group
    }
    confirmatory = set(seeds["confirmatory_200"])
    smoke = set(seeds["smoke_excluded"])
    assert development.isdisjoint(confirmatory)
    assert development.isdisjoint(smoke)
    assert confirmatory.isdisjoint(smoke)
    assert len(confirmatory) == 8


def test_cell_centered_profile_interpolation():
    field = np.arange(12, dtype=float).reshape(3, 4)
    assert np.allclose(_profile_at_x(field, 0.375), field[:, 1])
    assert np.allclose(_profile_at_y(field, 0.5), field[1])


def test_coarse_field_is_scored_but_never_called_validation():
    cfg = mohammadzadeh_config(
        grid=8,
        particles_per_cell=4,
        steps=20,
        sample_start=10,
        seed=8,
    )
    shape = (cfg.ny, cfg.nx)
    fields = {
        "T": np.full(shape, 303.0),
        "u": np.full(shape, 75.0),
        "qy": np.tile(np.linspace(-2.0, 1.0, cfg.nx), (cfg.ny, 1)),
        "microscopic_lid_slip_over_uwall": np.full(cfg.nx, 0.25),
        "microscopic_lid_T": np.full(cfg.nx, 303.0),
    }
    report = evaluate_mohammadzadeh_fields(fields, cfg)
    assert report["decision"] == "not_eligible_for_validation"
    assert not report["eligible_single_run_geometry_and_duration"]
    assert all(np.isfinite(value) for value in report["metrics"].values())
