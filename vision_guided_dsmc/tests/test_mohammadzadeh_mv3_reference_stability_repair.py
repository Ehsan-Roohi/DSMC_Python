from vgdsmc.mohammadzadeh_mv3_reference_stability_repair import (
    CONDITION_ID,
    SEED,
    load_protocol,
)


def test_stability_repair_is_one_seed_with_a_later_equal_horizon_window() -> None:
    contract = load_protocol()["repair_contract"]
    assert contract["condition_id"] == CONDITION_ID == "kn0p1_u100"
    assert contract["seed"] == SEED == 93202
    assert contract["steps"] - contract["sample_start"] == 93750
    assert contract["sample_count"] == 3000
    assert contract["nonoverlapping_sampling_blocks"] == 10
