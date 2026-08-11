import math

import numpy as np

from vgdsmc import mohammadzadeh_mv7_jcp_budget_matrix as mv7


def test_locked_execution_matrix_is_four_existing_plus_thirty_six_new_tasks():
    protocol = mv7.locked_protocol()
    execution = protocol["execution_matrix"]
    assert tuple(execution["budget_blocks"]) == (1, 2, 5, 10)
    assert tuple(execution["new_model_budget_blocks"]) == (2, 5, 10)
    assert execution["existing_budget_one_model_tasks"] == 12
    assert execution["new_model_tasks"] == 36
    assert execution["total_model_tasks_in_analysis"] == 48


def test_model_task_mapping_covers_each_new_budget_architecture_seed_once():
    tasks = [mv7.model_task_from_index(index) for index in range(36)]
    expected = {
        (budget, architecture, seed)
        for budget in mv7.NEW_MODEL_BUDGETS
        for architecture in mv7.ARCHITECTURES
        for seed in mv7.TRAINING_SEEDS
    }
    assert len(tasks) == 36
    assert set(tasks) == expected
    assert all(tasks.count(task) == 1 for task in expected)


def test_baseline_task_mapping_covers_every_budget_before_models():
    assert [mv7.baseline_task_from_index(index) for index in range(4)] == [1, 2, 5, 10]


def _constant_seed_errors(ratio):
    conditions = {
        "kn0p075_u150": ("94001", "94002", "94003", "94004"),
        "kn0p075_u300": ("94101", "94102", "94103", "94104"),
        "kn0p1_u200": ("94201", "94202", "94203", "94204"),
        "kn0p1_u400": ("94301", "94302", "94303", "94304"),
    }
    result = {}
    for method in mv7.METHODS:
        result[method] = {}
        for budget in mv7.BUDGETS:
            value = 1.0 if method == "raw" and budget == 10 else ratio
            result[method][budget] = {
                condition: {seed: value for seed in seeds}
                for condition, seeds in conditions.items()
            }
    return result


def test_noninferiority_uses_locked_ten_percent_margin_and_condition_clusters():
    passing = mv7._noninferiority(_constant_seed_errors(1.05))
    failing = mv7._noninferiority(_constant_seed_errors(1.11))
    assert passing["mambairv2_tiny_adapted"]["2"]["noninferior"]
    assert not failing["mambairv2_tiny_adapted"]["2"]["noninferior"]
    assert np.isclose(
        passing["mambairv2_tiny_adapted"]["2"]["geometric_mean_ratio_to_raw10"],
        1.05,
    )
    assert len(
        passing["mambairv2_tiny_adapted"]["2"]["condition_mean_log_ratios"]
    ) == 4


def test_raw_loglog_diagnostic_recovers_inverse_square_root_slope():
    curves = {}
    for method in mv7.METHODS:
        curves[method] = {
            str(budget): {"mean_composite_nrmse": 2.0 / math.sqrt(budget)}
            for budget in mv7.BUDGETS
        }
    result = mv7._scaling_and_equivalence(curves)
    assert np.isclose(result["raw_loglog_fit"]["slope"], -0.5)
    assert result["raw_loglog_fit"]["diagnostic_only"]


def test_generalization_claim_is_explicitly_limited_after_mv6_selection():
    scope = mv7.locked_protocol()["generalization_scope"]
    assert scope["evaluation_conditions_informed_the_completed_MV6_architecture_promotion"]
    assert "pristine external test after model selection" in scope["disallowed_claims"]
