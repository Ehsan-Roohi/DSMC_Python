from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dsmc_cavity.backend import ArrayBackend
from dsmc_cavity.collisions import SUPPORTED_MODELS, TrialBatch, apply_trials
from dsmc_cavity.collisions import generate_bt_trials
from dsmc_cavity.config import SimulationConfig
from dsmc_cavity.physics import total_momentum_energy, vhs_sigma_g_scalar
from dsmc_cavity.solver import CavitySolver


class PhysicsTests(unittest.TestCase):
    def test_vhs_sigma_g_is_finite_positive(self):
        value = vhs_sigma_g_scalar(500.0, 4.17e-10, 273.0, 0.81, 6.63e-26)
        self.assertTrue(math.isfinite(value))
        self.assertGreater(value, 0.0)

    def test_elastic_scattering_conserves_pair_momentum_and_energy(self):
        config = SimulationConfig(
            nx=2,
            ny=2,
            particles_per_cell=2,
            steps=2,
            warmup_steps=1,
            strict_probability=False,
        )
        backend = ArrayBackend.create("cpu", seed=12)
        velocities = np.asarray([[500.0, -30.0, 10.0], [-120.0, 80.0, -40.0]])
        before_p, before_e = total_momentum_energy(velocities, config.mass)
        batch = TrialBatch(
            first=np.asarray([0]),
            second=np.asarray([1]),
            multiplier=np.asarray([1e9]),
            cell=np.asarray([0]),
            volumes=np.asarray([config.cell_volume]),
        )
        stats = apply_trials(batch, velocities, config, backend, dt=1.0)
        after_p, after_e = total_momentum_energy(velocities, config.mass)
        self.assertEqual(stats.accepted, 1)
        np.testing.assert_allclose(after_p, before_p, rtol=1e-13, atol=1e-35)
        self.assertAlmostEqual(after_e / before_e, 1.0, places=13)

    def test_bt_timestep_is_more_restrictive(self):
        ntc = SimulationConfig(model="ntc")
        sbt = replace(ntc, model="sbt")
        dt_ntc, _ = ntc.recommended_dt()
        dt_sbt, limits = sbt.recommended_dt()
        self.assertIn("bernoulli_probability", limits)
        self.assertLessEqual(dt_sbt, dt_ntc)

    def test_bt_trial_generators_are_rate_unbiased(self):
        n = 12
        source = np.random.default_rng(44)
        position = source.random((n, 2)) * 1e-6
        velocity = source.normal(0.0, 300.0, size=(n, 3))
        config = SimulationConfig(
            nx=1, ny=1, particles_per_cell=n, steps=2, warmup_steps=1
        )
        exact = sum(
            vhs_sigma_g_scalar(
                np.linalg.norm(velocity[i] - velocity[j]),
                config.diameter_ref,
                config.temperature_ref,
                config.viscosity_index,
                config.mass,
            )
            for i in range(n)
            for j in range(i + 1, n)
        )
        for model in ("sbt", "gbt", "ssbt", "sgbt"):
            estimates = []
            for seed in range(300):
                trials = generate_bt_trials(
                    model, [np.arange(n)], position, config, np.random.default_rng(seed)
                )
                estimate = 0.0
                for i, j, multiplier in zip(
                    trials.first, trials.second, trials.multiplier
                ):
                    estimate += multiplier * vhs_sigma_g_scalar(
                        np.linalg.norm(velocity[i] - velocity[j]),
                        config.diameter_ref,
                        config.temperature_ref,
                        config.viscosity_index,
                        config.mass,
                    )
                estimates.append(estimate)
            with self.subTest(model=model):
                self.assertLess(abs(np.mean(estimates) / exact - 1.0), 0.03)


class SolverSmokeTests(unittest.TestCase):
    def test_diffuse_wall_uses_post_collision_velocity_for_remaining_time(self):
        config = SimulationConfig(
            model="ntc",
            backend="cpu",
            nx=2,
            ny=2,
            particles_per_cell=2,
            steps=2,
            warmup_steps=1,
            seed=71,
        )
        solver = CavitySolver(config)
        solver.positions[:] = 0.5 * config.length
        solver.velocities[:] = 0.0
        incoming_vy = 1000.0
        solver.positions[0] = (0.5 * config.length, config.length - 0.5 * incoming_vy * solver.dt)
        solver.velocities[0] = (40.0, incoming_vy, 0.0)
        solver._move_and_reflect(collect_wall=True)
        expected_y = config.length + solver.velocities[0, 1] * (0.5 * solver.dt)
        self.assertAlmostEqual(solver.positions[0, 1], expected_y, places=18)
        self.assertEqual(solver.wall_hits["top"], 1)

    def test_every_collision_model_runs_and_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            for model in SUPPORTED_MODELS:
                with self.subTest(model=model):
                    output = Path(tmp) / model
                    config = SimulationConfig(
                        model=model,
                        backend="cpu",
                        nx=4,
                        ny=4,
                        particles_per_cell=4,
                        steps=8,
                        warmup_steps=4,
                        sample_stride=1,
                        seed=1234,
                        output_dir=str(output),
                    )
                    result = CavitySolver(config).run(progress=False)
                    self.assertTrue((output / "fields.npz").exists())
                    self.assertTrue((output / "lid_profile.csv").exists())
                    self.assertEqual(result["fields"]["u"].shape, (4, 4))
                    self.assertTrue(np.all(np.isfinite(result["fields"]["temperature"])))
                    self.assertEqual(
                        result["metadata"]["collision_statistics"]["probability_exceedances"],
                        0,
                    )

    def test_gpu_backend_smoke_when_cupy_is_available(self):
        try:
            import cupy as cp

            if cp.cuda.runtime.getDeviceCount() < 1:
                self.skipTest("No CUDA device")
        except Exception:
            self.skipTest("CuPy/CUDA unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            config = SimulationConfig(
                model="sgbt",
                backend="gpu",
                nx=4,
                ny=4,
                particles_per_cell=4,
                steps=8,
                warmup_steps=4,
                sample_stride=1,
                output_dir=tmp,
            )
            result = CavitySolver(config).run(progress=False)
            self.assertEqual(result["metadata"]["backend_resolved"], "gpu")


if __name__ == "__main__":
    unittest.main()
