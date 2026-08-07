from __future__ import annotations

from vgdsmc.mohammadzadeh_mv3_reference import (
    load_protocol,
    new_reference_tasks,
    task_from_index,
)


def test_mv3_reference_array_is_locked_to_twelve_unique_new_tasks() -> None:
    protocol = load_protocol()
    tasks = new_reference_tasks()
    assert len(tasks) == protocol["reference_contract"]["new_trajectory_count"] == 12
    assert len(set(tasks)) == 12
    assert task_from_index(0) == ("kn0p05_u200", 93001)
    assert task_from_index(11) == ("kn0p1_u100", 93204)
    existing = {
        int(seed)
        for condition in protocol["conditions"]
        if condition["source"] == "existing_M3_QY100"
        for seed in condition["evaluation_seeds"]
    }
    assert not ({seed for _, seed in tasks} & existing)


def test_mv3_protocol_excludes_heat_flux_and_expensive_unlocked_conditions() -> None:
    protocol = load_protocol()
    fields = protocol["model_contract"]["input_fields"] + protocol["model_contract"]["output_fields"]
    assert not any(str(name).lower().startswith("q") for name in fields)
    assert {condition["knudsen"] for condition in protocol["conditions"]} == {0.05, 0.1}
    assert {condition["lid_speed_m_per_s"] for condition in protocol["conditions"]} == {100.0, 200.0, 400.0}
