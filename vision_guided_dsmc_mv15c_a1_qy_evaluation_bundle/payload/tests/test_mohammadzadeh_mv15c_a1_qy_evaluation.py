#!/usr/bin/env python3
"""Synthetic contract tests for the MV15C-A1 q_y evaluation recovery."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "vgdsmc"
    / "mohammadzadeh_mv15c_a1_qy_evaluation.py"
)
SPEC = importlib.util.spec_from_file_location("mv15c_a1_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
mv15a1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mv15a1)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_manifest(root: Path, name: str, files: list[str]) -> None:
    _write_json(
        root / name,
        {
            "files": {
                relative: {
                    "sha256": mv15a1._sha256(root / relative),
                    "size_bytes": (root / relative).stat().st_size,
                }
                for relative in files
            }
        },
    )


class FakeMV15C:
    COMPLETE_REFERENCE_STATUS = "complete_MV15C_fresh_reference_seed"
    PROTOCOL_FILE = "mv15c_fresh_b3_confirmation_protocol.json"
    post_pass = True

    @staticmethod
    def fresh_tasks():
        return list(mv15a1.FRESH_TASKS)

    @staticmethod
    def verify_lock():
        return {"protocol_sha256": "synthetic-original-protocol"}

    @staticmethod
    def _verify_manifest(root: Path, name: str):
        return mv15a1._verify_manifest(root, name)

    @staticmethod
    def run_prediction_stage(output: Path, *, batch_size: int):
        assert batch_size > 0
        (output / "PREDICTION_LOCK_PASS").write_text("pass\n", encoding="utf-8")
        (output / "locked_fresh_predictions.npz").write_bytes(b"synthetic")
        _write_json(
            output / "prediction_summary.json",
            {
                "status": "MV15C_fresh_predictions_locked_before_target_construction",
                "Raw_B10_used_by_prediction": False,
            },
        )
        (output / "fresh_source_audit.csv").write_text(
            "condition,seed\n", encoding="utf-8"
        )
        _write_manifest(
            output,
            "prediction_manifest.json",
            [
                "PREDICTION_LOCK_PASS",
                "locked_fresh_predictions.npz",
                "prediction_summary.json",
                "fresh_source_audit.csv",
            ],
        )
        return {
            "status": "MV15C_fresh_predictions_locked_before_target_construction"
        }

    @classmethod
    def run_post(cls, output: Path):
        figures = [
            "mv15c_fresh_qy_confirmation_ratios.png",
            "mv15c_fresh_qy_confirmation_ratios.pdf",
            "mv15c_fresh_qy_kn0p1_u400_seed_151501.png",
            "mv15c_fresh_qy_kn0p1_u400_seed_151501.pdf",
            "mv15c_fresh_qy_kn0p08_u350_seed_151511.png",
            "mv15c_fresh_qy_kn0p08_u350_seed_151511.pdf",
        ]
        for name in figures:
            (output / name).write_bytes(b"figure")
        (output / "mv15c_fresh_qy_metrics.csv").write_text(
            "condition,seed,method,qy_nrmse,ratio_to_Raw_B10\n",
            encoding="utf-8",
        )
        ratios = {
            "selected_b3": {"kn0p1_u400": 0.95, "kn0p08_u350": 0.90},
            "raw_b10": {"kn0p1_u400": 1.0, "kn0p08_u350": 1.0},
        }
        value = {
            "stage": "MV15C_Mohammadzadeh_fresh_B3_confirmation",
            "status": "complete_MV15C_prospectively_locked_fresh_confirmation",
            "decision": (
                "MV15C_fresh_DSMC_confirms_B3_DCIR_QY"
                if cls.post_pass
                else "MV15C_fresh_DSMC_does_not_confirm_B3_DCIR_QY_no_retuning"
            ),
            "all_gates_pass": bool(cls.post_pass),
            "gates": {"primary_qy_no_worse_than_Raw_B10": bool(cls.post_pass)},
            "condition_mean_ratios_to_Raw_B10": ratios,
            "selected_per_seed_ratios_to_Raw_B10": {
                "kn0p1_u400": {"151501": 0.95},
                "kn0p08_u350": {"151511": 0.90},
            },
            "figures": figures,
            "fresh_outcomes_used_for_tuning": False,
        }
        _write_json(output / "summary.json", value)
        return value


mv15a1._mv15c_module = lambda: FakeMV15C


def _reference_summary(condition: str, seed: int, hold: bool) -> dict:
    failed = (
        "temperature_min_K"
        if (condition, seed) == ("kn0p1_u400", 151502)
        else "temperature_max_K"
    )
    checks = {
        "macroscopic_lid_slip_center": True,
        "microscopic_lid_slip_center": True,
        "qy_profile_max_normalized": False,
        "qy_profile_min_normalized": False,
        "temperature_max_K": True,
        "temperature_min_K": True,
    }
    if hold:
        checks[failed] = False
    tracked = {
        "temperature_max_K": {
            "first_half_mean": 400.0,
            "second_half_mean": 398.0,
            "drift": -2.0,
            "drift_standard_error": 0.6,
            "drift_z_score": -3.3 if hold and failed == "temperature_max_K" else -0.5,
            "relative_drift": 0.005 if hold and failed == "temperature_max_K" else 0.001,
            "max_abs_drift_z_score": 3.3 if hold and failed == "temperature_max_K" else 0.5,
        },
        "temperature_min_K": {
            "first_half_mean": 292.0,
            "second_half_mean": 291.0,
            "drift": -1.0,
            "drift_standard_error": 0.5,
            "drift_z_score": -2.03 if hold and failed == "temperature_min_K" else -0.5,
            "relative_drift": 0.0043 if hold and failed == "temperature_min_K" else 0.001,
            "max_abs_drift_z_score": 2.03 if hold and failed == "temperature_min_K" else 0.5,
        },
    }
    return {
        "status": FakeMV15C.COMPLETE_REFERENCE_STATUS,
        "decision": (
            "hold_MV15C_fresh_reference"
            if hold
            else "accept_MV15C_fresh_reference_for_cross_seed_qy_analysis"
        ),
        "mechanical_checks": {
            "all_event_mechanics_gates_pass": True,
            "checkpoint_roundtrip_bitwise_identity": True,
            "complete_lid_event_bin_coverage": True,
            "finite_nonempty_fields": True,
            "majorant_violations_equal_zero": True,
            "stationarity_pass": not hold,
        },
        "stationarity": {"checks": checks, "tracked": tracked, "z_limit": 2.0},
    }


def _build_output(root: Path) -> Path:
    output = root / "run"
    output.mkdir()
    _write_json(output / "submission_lock.json", {"status": "synthetic"})
    _write_json(output / FakeMV15C.PROTOCOL_FILE, {"status": "synthetic"})
    _write_manifest(
        output,
        "source_lock_manifest.json",
        ["submission_lock.json", FakeMV15C.PROTOCOL_FILE],
    )
    held_pairs = {("kn0p1_u400", 151502), ("kn0p08_u350", 151513)}
    for condition, seed in mv15a1.FRESH_TASKS:
        directory = output / "references" / condition / f"seed_{seed}"
        directory.mkdir(parents=True)
        _write_json(
            directory / "summary.json",
            _reference_summary(condition, seed, (condition, seed) in held_pairs),
        )
        (directory / "fields.npz").write_bytes(b"fields")
        (directory / "block_fields.npz").write_bytes(b"blocks")
        _write_manifest(
            directory,
            "artifact_manifest.json",
            ["summary.json", "fields.npz", "block_fields.npz"],
        )
    return output


def _assert_raises(error, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except error:
        return
    raise AssertionError(f"expected {error.__name__}")


def test_contract_is_unchanged_and_requires_no_dsmc() -> None:
    value = mv15a1.verify_contract()
    assert value["trajectory_count"] == 8
    assert value["DSMC_rerun_required"] is False
    assert value["B3_predictor_changed"] is False
    assert value["q_y_gates_changed"] is False


def test_qc_audit_preserves_exactly_two_original_holds() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        output = _build_output(Path(temporary))
        qc = mv15a1.collect_reference_qc(output)
        assert qc["trajectory_count"] == 8
        assert qc["original_reference_gate_pass_count"] == 6
        assert qc["original_reference_gate_hold_count"] == 2
        assert qc["original_reference_gate_all_pass"] is False
        assert qc["complete_mechanical_provenance_count"] == 8


def test_amendment_lock_precedes_predictions() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        output = _build_output(Path(temporary))
        lock = mv15a1.prepare_amendment(output)
        assert lock["model_predictions_seen_before_amendment"] is False
        assert lock["cross_seed_targets_seen_before_amendment"] is False
        assert lock["original_reference_gate_hold_count"] == 2
        assert (output / mv15a1.LOCK_MANIFEST).is_file()


def test_amendment_refuses_any_existing_prediction_artifact() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        output = _build_output(Path(temporary))
        (output / "prediction_summary.json").write_text("{}\n", encoding="utf-8")
        _assert_raises(RuntimeError, mv15a1.prepare_amendment, output)


def test_amendment_refuses_mechanical_failure() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        output = _build_output(Path(temporary))
        directory = output / "references" / "kn0p1_u400" / "seed_151501"
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        summary["mechanical_checks"]["majorant_violations_equal_zero"] = False
        _write_json(directory / "summary.json", summary)
        _write_manifest(
            directory,
            "artifact_manifest.json",
            ["summary.json", "fields.npz", "block_fields.npz"],
        )
        _assert_raises(ValueError, mv15a1.prepare_amendment, output)


def test_prediction_attestation_keeps_raw_b10_out() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        output = _build_output(Path(temporary))
        mv15a1.prepare_amendment(output)
        value = mv15a1.run_prediction(output, batch_size=4)
        assert value["Raw_B10_used_by_prediction"] is False
        assert value["cross_seed_targets_constructed"] is False
        assert value["B3_predictor_or_weights_changed"] is False


def test_positive_qy_result_is_support_not_unamended_confirmation() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        output = _build_output(Path(temporary))
        mv15a1.prepare_amendment(output)
        mv15a1.run_prediction(output, batch_size=4)
        FakeMV15C.post_pass = True
        value = mv15a1.run_post(output)
        assert value["all_q_y_gates_pass"] is True
        assert value["all_gates_pass"] is False
        assert "supports" in value["decision"]
        assert "inconclusive" in value["unamended_MV15C_confirmatory_status"]


def test_negative_qy_result_is_reported_without_retuning() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        output = _build_output(Path(temporary))
        mv15a1.prepare_amendment(output)
        mv15a1.run_prediction(output, batch_size=4)
        FakeMV15C.post_pass = False
        value = mv15a1.run_post(output)
        assert value["all_q_y_gates_pass"] is False
        assert "does_not_support" in value["decision"]
        assert value["fresh_q_y_outcomes_used_for_tuning"] is False
    FakeMV15C.post_pass = True


def test_compact_package_is_written_to_return_root() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        output = _build_output(root)
        returned = root / "returned"
        mv15a1.prepare_amendment(output)
        mv15a1.run_prediction(output, batch_size=4)
        FakeMV15C.post_pass = True
        mv15a1.run_post(output)
        result = mv15a1.package_results(output, returned)
        archive = Path(result["archive"])
        assert archive.parent == returned
        assert archive.is_file()
        assert mv15a1._sha256(archive) == result["archive_sha256"]
        assert (returned / mv15a1.RESULT_POINTER).is_file()


def test_locked_reference_mutation_is_detected() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        output = _build_output(Path(temporary))
        mv15a1.prepare_amendment(output)
        path = output / "references" / "kn0p1_u400" / "seed_151501" / "summary.json"
        path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
        _assert_raises(ValueError, mv15a1.verify_amendment_lock, output)


def main() -> None:
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in sorted(tests, key=lambda value: value.__name__):
        test()
    print(f"MV15C_A1_QY_EVALUATION_TESTS_PASS count={len(tests)}")


if __name__ == "__main__":
    main()
