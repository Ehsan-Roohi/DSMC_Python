import numpy as np

from vgdsmc.mohammadzadeh_vision import SEEDS
from vgdsmc.mohammadzadeh_vision_mv2 import (
    BUDGETS,
    build_budget_arrays,
    fold_contained_targets,
    fold_split,
    gaussian_like,
    group_blocks,
    task_from_index,
    task_index,
    tsvd,
)


def test_all_folds_are_disjoint_and_cover_each_test_seed_once():
    observed = []
    for fold in range(len(SEEDS)):
        train, validation, test = fold_split(fold)
        assert len(train) == 6
        assert len(validation) == len(test) == 1
        assert not set(train) & set(validation + test)
        assert not set(validation) & set(test)
        observed.extend(test)
    assert tuple(observed) == SEEDS


def test_locked_task_mapping_roundtrips():
    observed = []
    for fold in range(len(SEEDS)):
        for budget in BUDGETS:
            index = task_index(fold, budget)
            assert task_from_index(index) == (fold, budget)
            observed.append(index)
    assert sorted(observed) == list(range(32))


def test_nonoverlapping_budget_grouping_has_exact_means():
    values = np.arange(10, dtype=np.float32).reshape(10, 1, 1, 1)
    grouped = group_blocks(values, 5)
    assert grouped.shape == (2, 1, 1, 1)
    assert np.allclose(grouped[:, 0, 0, 0], (2.0, 7.0))
    try:
        group_blocks(values, 3)
    except ValueError:
        pass
    else:
        raise AssertionError("an unlocked temporal budget was accepted")


def test_budget_arrays_repeat_only_the_fold_contained_target():
    blocks = {seed: np.full((10, 5, 3, 3), float(seed)) for seed in SEEDS}
    full = {seed: np.full((2, 3, 3), float(seed)) for seed in SEEDS}
    train, validation, test = fold_split(0)
    targets = fold_contained_targets(full, train, validation, test)
    x, y, identity = build_budget_arrays(blocks, targets, train[:2], 5)
    assert x.shape == (4, 5, 3, 3)
    assert y.shape == (4, 2, 3, 3)
    assert identity.shape == (4, 3)
    expected = np.mean([seed for seed in train if seed != train[0]])
    assert np.allclose(y[0], expected)


def test_validation_and_test_fields_cannot_change_any_target():
    train, validation, test = fold_split(3)
    full = {seed: np.full((2, 2, 2), float(seed)) for seed in SEEDS}
    before = fold_contained_targets(full, train, validation, test)
    full[validation[0]] = np.full((2, 2, 2), -1.0e9)
    full[test[0]] = np.full((2, 2, 2), 1.0e9)
    after = fold_contained_targets(full, train, validation, test)
    for seed in SEEDS:
        assert np.array_equal(before[seed], after[seed])


def test_spatial_filter_preserves_constant_fields():
    values = np.ones((2, 2, 9, 11), dtype=np.float32)
    assert np.array_equal(gaussian_like(values, 4), values)


def test_tsvd_recovers_rank_one_fields():
    left = np.arange(1, 7, dtype=np.float32)
    right = np.arange(1, 9, dtype=np.float32)
    field = np.outer(left, right)
    values = np.stack((field, 2.0 * field))[None]
    assert np.allclose(tsvd(values, 1), values, rtol=2.0e-6, atol=2.0e-6)
