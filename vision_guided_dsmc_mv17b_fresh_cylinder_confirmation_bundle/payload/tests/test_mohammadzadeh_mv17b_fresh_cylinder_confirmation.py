#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np

from vgdsmc import mohammadzadeh_mv17b_fresh_cylinder_confirmation as mv17b


def test_contract_is_locked() -> None:
    value = mv17b.verify_contract()
    assert value["status"] == "MV17B_contract_verified"
    assert value["pair_count"] == 6
    assert value["trajectory_count"] == 12


def test_fresh_seeds_are_unique_and_disjoint() -> None:
    fresh = [seed for _, left, right in mv17b.PAIRS for seed in (left, right)]
    assert len(fresh) == len(set(fresh)) == 12
    assert not set(fresh).intersection(mv17b.DEVELOPMENT_SEEDS)


def test_locked_budget_partition() -> None:
    assert not set(mv17b.B3_NOUT).intersection(mv17b.B10_NOUT)
    assert not set(mv17b.B3_NOUT).intersection(mv17b.GUARD_NOUT)
    assert not set(mv17b.B10_NOUT).intersection(mv17b.GUARD_NOUT)
    assert set(mv17b.B3_NOUT + mv17b.B10_NOUT + mv17b.GUARD_NOUT) == set(range(100, 117))


def test_two_endpoint_exact_power() -> None:
    unadjusted = 1.0 / 2**len(mv17b.PAIRS)
    adjusted = mv17b._holm_adjust({"global_qy": unadjusted, "near_wall_qn": unadjusted})
    assert unadjusted == 0.015625
    assert adjusted == {"global_qy": 0.03125, "near_wall_qn": 0.03125}


def test_holm_monotonicity() -> None:
    adjusted = mv17b._holm_adjust({"a": 0.01, "b": 0.04})
    assert adjusted["a"] == 0.02
    assert adjusted["b"] == 0.04


def test_case_identity() -> None:
    assert mv17b._case_id("pair_03", "observation") == "pair_03_observation"
    assert mv17b._case_id("pair_03", "reference") == "pair_03_reference"


def test_mesh_equality_is_fail_closed() -> None:
    source = {"x_m": np.arange(4.0), "y_m": np.arange(4.0), "area_m2": np.ones(4)}
    mv17b._assert_same_mesh(source, {name: value.copy() for name, value in source.items()})
    changed = {name: value.copy() for name, value in source.items()}
    changed["x_m"][0] = 1.0
    try:
        mv17b._assert_same_mesh(source, changed)
    except ValueError:
        pass
    else:
        raise AssertionError("mesh mismatch was accepted")


def test_manifest_detects_mutation() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        value = root / "value.txt"
        value.write_text("locked\n", encoding="utf-8")
        mv17b._write_manifest(root, "manifest.json", [value])
        mv17b._verify_manifest(root, "manifest.json")
        value.write_text("changed\n", encoding="utf-8")
        try:
            mv17b._verify_manifest(root, "manifest.json")
        except ValueError:
            pass
        else:
            raise AssertionError("mutated file passed manifest verification")


def test_json_numpy_scalars_are_serializable() -> None:
    encoded = mv17b._json_dumps({"boolean": np.bool_(True), "integer": np.int64(6)})
    assert json.loads(encoded) == {"boolean": True, "integer": 6}


def test_protocol_preserves_stationarity_warning() -> None:
    protocol = mv17b.locked_protocol()
    prohibited = protocol["interpretation_lock"]["successful_result_does_not_authorize"]
    assert "full_stationarity_at_tU_over_D_30" in prohibited
    assert any("tU/D=11.5" in text for text in protocol["known_limitations"])


def test_reference_cannot_enter_prediction_contract() -> None:
    protocol = mv17b.locked_protocol()
    assert protocol["fresh_pair_contract"]["reference_trajectory_never_enters_prediction"] is True
    source = Path(mv17b.__file__).read_text(encoding="utf-8")
    apply_start = source.index("def _apply_frozen(")
    apply_end = source.index("\ndef _holm_adjust", apply_start)
    assert "target" not in source[apply_start:apply_end]


def test_runner_uses_locked_window_and_fresh_IRUN3() -> None:
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    runner = (scripts / "unity_mohammadzadeh_mv17b_run_array.sbatch").read_text(encoding="utf-8")
    assert "MIN_NOUT=116" in runner and "MAX_NOUT=116" in runner
    assert "RANDOM_SEED.IN" in runner
    assert "restart_reused\": False" in runner
    assert "exit 22" in runner
    assert 'exit "${SOLVER_RC:-22}"' not in runner


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"MV17B_FRESH_CYLINDER_CONFIRMATION_TESTS_PASS count={len(tests)}")


if __name__ == "__main__":
    main()
