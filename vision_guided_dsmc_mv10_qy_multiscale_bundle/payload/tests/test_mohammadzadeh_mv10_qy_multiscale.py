from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import textwrap

import numpy as np


PAYLOAD = Path(__file__).parents[1]
SCRIPT = PAYLOAD / "vgdsmc" / "mohammadzadeh_mv10_qy_multiscale.py"
MV9_PAYLOAD = PAYLOAD
if not (MV9_PAYLOAD / "vgdsmc" / "mohammadzadeh_mv9_heat_flux.py").is_file():
    MV9_PAYLOAD = (
        PAYLOAD.parents[1] / "vision_guided_dsmc_mv9_heat_flux_bundle" / "payload"
    )
SPEC = importlib.util.spec_from_file_location("mv10_qy", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
mv10 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mv10)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_task_matrix_is_exactly_three_locked_initializations():
    assert [mv10.task_from_index(index) for index in range(3)] == list(
        mv10.TRAINING_SEEDS
    )
    for invalid in (-1, 3):
        try:
            mv10.task_from_index(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid MV10 task index was accepted")


def test_component_metric_is_finite_at_zero_crossings():
    target = np.zeros((4, 8, 8), dtype=np.float32)
    target[:, 2:6, 2:6] = -0.05
    candidate = target + 0.01
    assert np.isfinite(mv10.component_nrmse(candidate, target))
    assert mv10.component_nrmse(target, target) == 0.0


def test_block_average_preserves_constant_and_partial_edges():
    constant = np.full((2, 10, 9), -3.5)
    coarse = mv10.numpy_block_average(constant, factor=4)
    assert coarse.shape == (2, 3, 3)
    assert np.array_equal(coarse, np.full((2, 3, 3), -3.5))

    ramp = np.arange(25, dtype=np.float64).reshape(5, 5)
    coarse_ramp = mv10.numpy_block_average(ramp, factor=3)
    assert coarse_ramp[0, 0] == np.mean(ramp[:3, :3])
    assert coarse_ramp[1, 1] == np.mean(ramp[3:, 3:])


def test_legacy_seed_identity_is_scoped_to_the_locked_primary_condition():
    conditions = np.asarray(
        ["kn0p075_u150", "kn0p075_u150"]
        + ["kn0p1_u400"] * 8,
        dtype="U32",
    )
    identities = np.asarray(
        [(94001, 0, 1), (94002, 0, 1)]
        + [
            (seed, block, 1)
            for seed in mv10.EXPECTED_LEGACY_SEEDS
            for block in (0, 1)
        ],
        dtype=np.int64,
    )
    by_condition = mv10.seed_identity_by_condition(conditions, identities)
    assert by_condition["kn0p075_u150"] == (94001, 94002)
    assert by_condition["kn0p1_u400"] == mv10.EXPECTED_LEGACY_SEEDS
    assert tuple(sorted(set(int(item) for item in identities[:, 0]))) != (
        mv10.EXPECTED_LEGACY_SEEDS
    )


def test_model_task_never_indexes_legacy_target_arrays():
    tree = ast.parse(textwrap.dedent(inspect.getsource(mv10.run_model_task)))
    indexed_names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        if isinstance(node.value, ast.Name) and node.value.id == "data":
            value = node.slice.value if isinstance(node.slice, ast.Constant) else None
            if isinstance(value, str):
                indexed_names.add(value)
    assert "test_y" not in indexed_names
    assert "test_target10" not in indexed_names
    assert {"train_x", "train_y", "validation_x", "validation_y", "test_x"} <= indexed_names


def test_protocol_discloses_post_outcome_design_and_fresh_seed_requirement():
    path = (
        PAYLOAD
        / "reference_data"
        / "mohammadzadeh_2012"
        / mv10.PROTOCOL_FILE
    )
    protocol = json.loads(path.read_text(encoding="utf-8"))
    assert protocol["stage"] == mv10.STAGE
    assert protocol["status"] == mv10.STATUS
    role = protocol["scientific_role"]
    assert role["MV9_outcomes_observed_before_lock"] is True
    assert role["old_evaluation_seeds_forbidden_as_confirmation"] is True
    assert role["fresh_unobserved_seed_followup_required_for_any_JCP_claim"] is True
    assert tuple(role["old_evaluation_seeds"]) == mv10.EXPECTED_LEGACY_SEEDS
    assert protocol["data_contract"]["legacy_targets_used_for_training_or_selection"] is False
    assert protocol["analysis_contract"]["maximum_primary_mean_qy_ratio_to_raw_B10"] == 1.0
    assert protocol["return_contract"]["compact_archive_written_directly_to_Unity_project_root"] is True


def test_protocol_locks_the_exact_MV9_ancestry_in_this_branch():
    path = (
        PAYLOAD
        / "reference_data"
        / "mohammadzadeh_2012"
        / mv10.PROTOCOL_FILE
    )
    protocol = json.loads(path.read_text(encoding="utf-8"))
    source = protocol["source_contract"]
    assert sha256(
        MV9_PAYLOAD / "vgdsmc" / "mohammadzadeh_mv9_heat_flux.py"
    ) == source["mv9_module_sha256"]
    assert sha256(
        MV9_PAYLOAD
        / "reference_data"
        / "mohammadzadeh_2012"
        / "mv9_heat_flux_noise2noise_protocol.json"
    ) == source["mv9_protocol_sha256"]
