import json

from vgdsmc import mohammadzadeh_mv5_reference_stability_repair as repair


def test_repair_task_mapping_is_locked_to_three_failed_seeds():
    assert [repair.task_from_index(index) for index in range(3)] == [
        ("kn0p075_u150", 94003),
        ("kn0p1_u200", 94201),
        ("kn0p1_u400", 94301),
    ]


def test_repair_protocol_preserves_original_sampling_horizon_and_gate():
    protocol = repair.load_protocol()
    contract = protocol["repair_contract"]
    assert contract["steps"] - contract["sample_start"] == 93750
    assert contract["sampling_horizon_steps"] == 93750
    assert contract["sample_count"] == 3000
    assert contract["nonoverlapping_sampling_blocks"] == 10
    assert contract["stationarity_z_limit"] == 2.0
    assert protocol["scope_guards"] == {
        "original_16_reference_directories_are_immutable": True,
        "only_the_three_preidentified_failed_seeds_are_repaired": True,
        "all_other_13_references_are_reused_byte_for_byte": True,
        "heat_flux_excluded_from_MV5_and_MV6_claims": True,
        "physics_changed": False,
        "grid_particles_sample_count_block_count_or_sampling_horizon_changed": False,
        "stationarity_gate_changed": False,
        "model_architecture_training_split_baselines_or_metrics_changed": False,
        "failed_model_attempts_created_no_model_outcomes": True,
    }


def test_diagnostic_lock_records_no_model_outcomes():
    protocol = json.loads(repair.protocol_path().read_text(encoding="utf-8"))
    diagnostics = protocol["diagnostic_basis"]
    assert diagnostics["completed_MV5_model_outcomes_before_lock"] == 0
    assert diagnostics["completed_MV6_model_outcomes_before_lock"] == 0
