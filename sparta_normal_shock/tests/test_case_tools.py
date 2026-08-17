from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generate = load_module("shock_generate", "scripts/generate_case.py")
post = load_module("shock_post", "scripts/postprocess.py")


class GeneratorTests(unittest.TestCase):
    def test_mach_three_rankine_hugoniot(self):
        state = generate.rankine_hugoniot(3.0, 300.0, 1.0e25)
        self.assertAlmostEqual(state["density_ratio"], 3.0)
        self.assertAlmostEqual(state["pressure_ratio"], 11.0)
        self.assertAlmostEqual(state["temperature_ratio"], 11.0 / 3.0)
        self.assertAlmostEqual(state["velocity_2"] / state["velocity_1"], 1.0 / 3.0)

    def test_generated_contract_and_boundary_conditions(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)
            metadata = generate.write_case(target, "production", 3.0, 12345)
            deck = (target / "in.shock").read_text(encoding="utf-8")
            self.assertEqual(metadata["schema_version"], 2)
            self.assertEqual(metadata["protocol"], "sparta_normal_shock_v2")
            self.assertEqual(metadata["nx"], 1200)
            self.assertEqual(metadata["half_span_lambda"], 30.0)
            self.assertEqual(metadata["warmup_steps"], 80000)
            self.assertEqual(metadata["sample_steps"], 320000)
            self.assertEqual(metadata["dump_frequency_steps"], 80000)
            self.assertEqual(metadata["restart_frequency_steps"], 80000)
            self.assertEqual(metadata["target_upstream_particles_per_cell"], 64)
            self.assertAlmostEqual(metadata["dx_over_lambda_1"], 0.05)
            self.assertLessEqual(metadata["transport_cfl_bound"], 0.2)
            self.assertIn("boundary             o p p", deck)
            self.assertIn("create_grid          1200 1 1", deck)
            self.assertIn("inject_right emit/face downstream xhi twopass", deck)
            self.assertIn("compute              pflux pflux/grid", deck)
            self.assertIn("compute              eflux eflux/grid", deck)
            self.assertIn("restart              80000", deck)
            self.assertIn("run                  80000", deck)
            self.assertIn("run                  320000", deck)


class PostprocessTests(unittest.TestCase):
    @staticmethod
    def ideal_profile(scale: float = 1.0):
        n1 = 1.0e20
        t1 = 300.0
        base = generate.rankine_hugoniot(3.0, t1, n1)
        meta = {
            **base,
            "argon_mass_kg": generate.MASS,
            "mean_free_path_1": 1.0,
            "half_span_lambda": 30.0,
        }
        rows = []
        for index in range(1200):
            x = -29.975 + 0.05 * index
            downstream = x >= 0.0
            n = scale * (base["number_density_2"] if downstream else n1)
            temp = base["temperature_2"] if downstream else t1
            velocity = base["velocity_2"] if downstream else base["velocity_1"]
            pressure = n * post.K_B * temp
            rows.append({
                "x": x,
                "number_density": n,
                "u": velocity,
                "v": 0.0,
                "w": 0.0,
                "temperature": temp,
                "pressure": pressure,
                "Pxx": pressure,
                "Pyy": pressure,
                "Pzz": pressure,
                "qx": 0.0,
            })
        return rows, meta

    def test_parser_uses_last_snapshot(self):
        content = """ITEM: TIMESTEP
1
ITEM: NUMBER OF CELLS
1
ITEM: CELLS id xc f_avg[1] f_avg[2] f_avg[3] f_avg[4] f_avg[5] f_avg[6] f_avg[7] f_avg[8] f_avg[9] f_avg[10]
1 -1 1 2 3 4 5 6 7 8 9 10
ITEM: TIMESTEP
2
ITEM: NUMBER OF CELLS
1
ITEM: CELLS id xc f_avg[1] f_avg[2] f_avg[3] f_avg[4] f_avg[5] f_avg[6] f_avg[7] f_avg[8] f_avg[9] f_avg[10]
1 1 11 12 13 14 15 16 17 18 19 20
"""
        with tempfile.TemporaryDirectory() as temporary:
            dump = Path(temporary) / "dump"
            dump.write_text(content, encoding="utf-8")
            rows = post.read_last_grid_snapshot(dump)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["x"], 1.0)
        self.assertEqual(rows[0]["qx"], 20.0)

    def test_required_crossing_rejects_invalid_profile(self):
        with self.assertRaisesRegex(ValueError, "does not cross"):
            post.crossing_x([-1.0, 0.0, 1.0], [1.0, 1.0, 1.0], 2.0, require_crossing=True)

    def test_reversed_density_profile_rejects_thickness(self):
        rows, meta = self.ideal_profile()
        reversed_rows = sorted(
            ({**row, "x": -row["x"]} for row in rows), key=lambda row: row["x"]
        )
        with self.assertRaisesRegex(ValueError, "Invalid density-thickness crossings"):
            post.normalize_rows(reversed_rows, meta)

    def test_directional_temperature_from_pressure_tensor(self):
        rows, meta = self.ideal_profile()
        normalized, metrics = post.normalize_rows(rows, meta)
        self.assertAlmostEqual(normalized[0]["Tx_over_T1"], 1.0)
        self.assertAlmostEqual(normalized[-1]["Tperp_over_T1"], 11.0 / 3.0)
        self.assertLess(metrics["gates"]["far_field_max_relative_error"], 1.0e-12)
        self.assertLess(
            metrics["gates"]["flux_maximum_relative_error_from_upstream"], 1.0e-12
        )
        self.assertTrue(metrics["physics_gate_pass"])
        self.assertEqual(
            metrics["validation_windows_x_over_lambda_1"]["upstream"], [-28.0, -24.0]
        )

    def test_far_field_error_above_three_percent_fails(self):
        rows, meta = self.ideal_profile()
        for row in rows:
            if -28.0 <= row["x"] <= -24.0:
                row["temperature"] *= 1.031
        _, metrics = post.normalize_rows(rows, meta)
        self.assertGreater(metrics["gates"]["far_field_max_relative_error"], 0.03)
        self.assertFalse(metrics["physics_gate_pass"])

    def test_constant_one_percent_flux_bias_fails(self):
        rows, meta = self.ideal_profile(scale=0.99)
        _, metrics = post.normalize_rows(rows, meta)
        self.assertLess(metrics["gates"]["far_field_max_relative_error"], 0.03)
        self.assertGreater(
            metrics["gates"]["flux_maximum_relative_error_from_upstream"], 0.005
        )
        self.assertFalse(metrics["physics_gate_pass"])

    def test_student_t_three_seed_confidence_interval(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dirs = []
            for seed, value in zip((11, 22, 33), (1.0, 2.0, 3.0)):
                run_dir = root / f"seed_{seed}"
                run_dir.mkdir()
                rows = []
                for x in (-1.0, 0.0, 1.0):
                    row = {key: 1.0 for key in post.NORMALIZED_NAMES}
                    row["x_over_lambda_1"] = x
                    row["temperature_over_T1"] = value
                    rows.append(row)
                post.write_csv(run_dir / "profile_normalized.csv", rows)
                (run_dir / "case_metadata.json").write_text(
                    json.dumps({"mach_1": 3.0, "seed": seed, "dx_over_lambda_1": 0.05}),
                    encoding="utf-8",
                )
                (run_dir / "validation_metrics.json").write_text(
                    json.dumps({"physics_gate_pass": True}), encoding="utf-8"
                )
                run_dirs.append(run_dir)
            summary = post.process_ensemble(root / "ensemble", run_dirs)
            ensemble_rows = post.read_csv(root / "ensemble" / "ensemble_profile.csv")
            expected = 4.30265272991 / math.sqrt(3.0)
            self.assertAlmostEqual(
                ensemble_rows[0]["temperature_over_T1_ci95"], expected, places=10
            )
            self.assertEqual(summary["uncertainty"]["degrees_of_freedom"], 2)
            self.assertAlmostEqual(summary["uncertainty"]["critical_value"], 4.30265272991)

    def test_checkpoint_stability_gate(self):
        far_field = {
            key: {"upstream_mean": upstream, "downstream_mean": downstream}
            for key, upstream, downstream in (
                ("number_density_over_n1", 1.0, 3.0),
                ("u_over_u1", 1.0, 1.0 / 3.0),
                ("temperature_over_T1", 1.0, 11.0 / 3.0),
            )
        }
        current = {
            "shock_center_over_lambda_1_before_alignment": 0.1,
            "density_10_90_thickness_over_lambda_1": 6.0,
            "far_field": far_field,
        }
        previous = {
            "shock_center_over_lambda_1_before_alignment": 0.0,
            "density_10_90_thickness_over_lambda_1": 6.05,
            "far_field": far_field,
        }
        profile = []
        for x in (-1.0, 0.0, 1.0):
            row = {key: 1.0 for key in post.NORMALIZED_NAMES}
            row["x_over_lambda_1"] = x
            profile.append(row)
        metrics = post.checkpoint_stability_metrics(current, previous, profile, profile)
        self.assertTrue(metrics["pass"])
        previous["shock_center_over_lambda_1_before_alignment"] = -0.5
        metrics = post.checkpoint_stability_metrics(current, previous, profile, profile)
        self.assertFalse(metrics["pass"])

    def test_ensemble_rejects_failed_member_and_duplicate_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dirs = []
            for index, seed in enumerate((11, 22, 33)):
                run_dir = root / f"run_{index}"
                run_dir.mkdir()
                rows = []
                for x in (-1.0, 1.0):
                    row = {key: 1.0 for key in post.NORMALIZED_NAMES}
                    row["x_over_lambda_1"] = x
                    rows.append(row)
                post.write_csv(run_dir / "profile_normalized.csv", rows)
                (run_dir / "case_metadata.json").write_text(
                    json.dumps({"mach_1": 3.0, "seed": seed, "dx_over_lambda_1": 0.05}),
                    encoding="utf-8",
                )
                (run_dir / "validation_metrics.json").write_text(
                    json.dumps({"physics_gate_pass": index != 2}), encoding="utf-8"
                )
                run_dirs.append(run_dir)
            with self.assertRaisesRegex(ValueError, "failed validation"):
                post.process_ensemble(root / "failed", run_dirs)
            (run_dirs[2] / "validation_metrics.json").write_text(
                json.dumps({"physics_gate_pass": True}), encoding="utf-8"
            )
            (run_dirs[2] / "case_metadata.json").write_text(
                json.dumps({"mach_1": 3.0, "seed": 22, "dx_over_lambda_1": 0.05}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "seeds must be distinct"):
                post.process_ensemble(root / "duplicate", run_dirs)


class HPCContractTests(unittest.TestCase):
    def test_v2_submission_and_validation_contract(self):
        array = (ROOT / "hpc" / "unity_sparta_shock_array.slurm").read_text(encoding="utf-8")
        collect = (ROOT / "hpc" / "unity_sparta_shock_collect.slurm").read_text(encoding="utf-8")
        bootstrap = (ROOT / "hpc" / "bootstrap_unity_sparta_normal_shock.sh").read_text(encoding="utf-8")
        self.assertIn('FINAL_DUMP_NAME="profile.final.00320000"', array)
        self.assertIn('FINAL_DUMP_NAME="profile.final.00320000"', collect)
        self.assertIn('export OMPI_MCA_pml="${OPENMPI_PML}"', array)
        self.assertIn('status=physics_failed', array)
        self.assertIn("exit 9", array)
        self.assertIn("physics_gate_pass=true", collect)
        self.assertIn("validated_member_count=%s", collect)
        self.assertIn('--dependency="afterany:${ARRAY_JOB_ID}"', bootstrap)
        self.assertIn("agent/sparta-normal-shock-v2", bootstrap)
        self.assertIn("LAST_SPARTA_NORMAL_SHOCK_V2_JOBS.env", bootstrap)


if __name__ == "__main__":
    unittest.main()
