import json

import numpy as np

from vgdsmc import mohammadzadeh_mv5_reference as ref
from vgdsmc import mohammadzadeh_vision_mv5 as mv5


DEVELOPMENT = (
    {"id": "a", "knudsen": 0.05, "lid_speed_m_per_s": 100.0},
    {"id": "b", "knudsen": 0.05, "lid_speed_m_per_s": 200.0},
    {"id": "c", "knudsen": 0.05, "lid_speed_m_per_s": 400.0},
    {"id": "d", "knudsen": 0.10, "lid_speed_m_per_s": 100.0},
)


def test_task_indices_match_locked_budgets():
    assert [mv5.task_from_index(index) for index in range(4)] == [1, 2, 5, 10]


def test_convex_hull_gate_rejects_mv4_fold_zero_rectangle_false_positive():
    heldout = {"knudsen": 0.05, "lid_speed_m_per_s": 100.0}
    training = DEVELOPMENT[1:]
    report = mv5.convex_hull_support_report(heldout, training)
    assert not report["inside_development_hull"]
    assert report["normalized_distance_to_hull"] > 0.0


def test_preregistered_inside_condition_is_inside_joint_hull():
    heldout = {"knudsen": 0.075, "lid_speed_m_per_s": 150.0}
    report = mv5.convex_hull_support_report(heldout, DEVELOPMENT)
    assert report["inside_development_hull"]
    assert report["normalized_distance_to_hull"] == 0.0


def test_preregistered_ood_conditions_are_outside_joint_hull():
    for knudsen, speed in ((0.075, 300.0), (0.10, 200.0), (0.10, 400.0)):
        report = mv5.convex_hull_support_report(
            {"knudsen": knudsen, "lid_speed_m_per_s": speed}, DEVELOPMENT
        )
        assert not report["inside_development_hull"]
        assert np.isfinite(report["normalized_distance_to_hull"])


def test_reference_seed_bank_is_unique_and_complete():
    protocol = json.loads(ref.protocol_path().read_text(encoding="utf-8"))
    tasks = [
        (item["id"], seed)
        for item in protocol["confirmatory_conditions"]
        for seed in item["evaluation_seeds"]
    ]
    assert len(tasks) == 16
    assert len({seed for _, seed in tasks}) == 16


def test_convex_hull_is_counter_clockwise():
    points = np.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.5]])
    hull = mv5.convex_hull(points)
    assert len(hull) == 3
    assert all(
        mv5._cross(hull[index], hull[(index + 1) % 3], hull[(index + 2) % 3]) > 0.0
        for index in range(3)
    )
