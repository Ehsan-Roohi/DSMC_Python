from pathlib import Path

import numpy as np

from vgdsmc.mohammadzadeh_vision import (
    INPUT_FIELDS,
    OUTPUT_FIELDS,
    SEEDS,
    build_arrays,
    evaluate,
    fit_scaling,
    leave_one_seed_out_targets,
)


def test_heat_flux_is_completely_excluded():
    assert not any(name.startswith("q") for name in INPUT_FIELDS + OUTPUT_FIELDS)


def test_leave_one_seed_out_target_excludes_self():
    full = {seed: np.full((2, 2, 2), float(seed)) for seed in SEEDS}
    targets = leave_one_seed_out_targets(full)
    expected = np.mean([seed for seed in SEEDS if seed != SEEDS[-1]])
    assert np.allclose(targets[SEEDS[-1]], expected)


def test_seed_images_and_scaling_contract():
    blocks = {seed: np.full((2, 5, 4, 4), float(seed)) for seed in SEEDS}
    targets = {seed: np.full((2, 4, 4), float(seed + 1)) for seed in SEEDS}
    x, y, identity = build_arrays(blocks, targets, SEEDS[:2])
    assert x.shape == (4, 5, 4, 4)
    assert y.shape == (4, 2, 4, 4)
    assert set(identity[:, 0]) == set(SEEDS[:2])
    scaling = fit_scaling(x, y)
    assert scaling["input_mean"].shape == (1, 5, 1, 1)
    assert scaling["residual_std"].shape == (1, 2, 1, 1)


def test_evaluation_rewards_an_exact_reconstruction():
    rng = np.random.default_rng(4)
    target = rng.normal(size=(2, 2, 8, 8))
    raw = target + 0.2 * rng.normal(size=target.shape)
    metrics = evaluate(raw, target.copy(), target)
    assert metrics["vision_over_raw_composite"] == 0.0
