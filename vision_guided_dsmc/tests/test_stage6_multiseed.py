import numpy as np

from vgdsmc.physical_multiseed import (
    multiseed_cases,
    summarize_rows,
)


def test_multiseed_case_matrix():
    cases = multiseed_cases((1, 2, 3))
    assert len(cases) == 9
    assert sorted({case.knudsen for case in cases}) == [
        0.05,
        0.1,
        0.2,
    ]
    assert sorted({case.seed for case in cases}) == [1, 2, 3]


def test_multiseed_summary_uses_matched_cost_ratios():
    rows = []
    for knudsen in (0.05, 0.1, 0.2):
        for seed, ratio in zip(
            (1, 2, 3),
            (0.9, 1.0, 1.1),
        ):
            rows.append(
                {
                    "knudsen": knudsen,
                    "seed": seed,
                    "adaptive_to_uniform_error_ratio": ratio,
                }
            )
    summary = summarize_rows(rows)
    assert summary["cases"] == 9
    assert summary["improved_cases"] == 3
    assert np.isclose(summary["mean_error_ratio"], 1.0)
    assert summary["adaptive_to_uniform_particle_ratio"] == 1.0
