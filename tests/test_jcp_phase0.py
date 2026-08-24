import unittest

import numpy as np

from vgdsmc.jcp_phase0 import (
    dct2,
    development_priors,
    direct_block_noise_power,
    eb_gain,
    fuse,
    idct2,
    overlapping_noise_power,
    pnet_cross_block_gain,
)


class TestJCPPhase0(unittest.TestCase):
    def test_dct_round_trip(self):
        rng = np.random.default_rng(20260817)
        value = rng.normal(size=(3, 16, 24))
        np.testing.assert_allclose(idct2(dct2(value)), value, rtol=1.0e-12, atol=1.0e-12)

    def test_overlap_identity_agrees_with_direct_scatter(self):
        rng = np.random.default_rng(314159)
        units, blocks, ny, nx = 320, 10, 8, 8
        truth = rng.normal(scale=0.25, size=(units, ny, nx))
        noise = rng.normal(scale=1.0, size=(units, blocks, ny, nx))
        samples = truth[:, None] + noise
        raw3 = np.mean(samples[:, :3], axis=1)
        raw10 = np.mean(samples, axis=1)
        peers = np.arange(units)
        overlap = overlapping_noise_power(raw3, raw10, peers, width=8)
        direct = direct_block_noise_power(samples, peers, budget=3, width=8)
        ratio = float(np.sum(overlap) / np.sum(direct))
        self.assertLess(abs(ratio - 1.0), 0.08)

    def test_eb_limits(self):
        rng = np.random.default_rng(271828)
        observation = rng.normal(size=(16, 16))
        prior = rng.normal(size=(16, 16))
        zero_noise = np.zeros((16, 16))
        gain = eb_gain(observation, prior, zero_noise, width=8)
        np.testing.assert_allclose(fuse(observation, prior, gain), observation, atol=1.0e-11)
        huge_noise = np.full((16, 16), 1.0e30)
        gain = eb_gain(observation, prior, huge_noise, width=8)
        np.testing.assert_allclose(fuse(observation, prior, gain), prior, atol=1.0e-11)

    def test_cross_block_gain_is_bounded(self):
        rng = np.random.default_rng(161803)
        truth = rng.normal(size=(16, 16))
        blocks = truth + rng.normal(scale=0.8, size=(3, 16, 16))
        priors = np.stack(
            [truth + rng.normal(scale=0.25, size=(16, 16)) for _ in range(3)]
        )
        gain = pnet_cross_block_gain(blocks, priors, width=8)
        self.assertTrue(np.all(np.isfinite(gain)))
        self.assertGreaterEqual(float(np.min(gain)), 0.0)
        self.assertLessEqual(float(np.max(gain)), 1.0)

    def test_development_priors_exclude_exact_target(self):
        rng = np.random.default_rng(141421)
        features = np.asarray(((-1.3, 2.0), (-1.0, 3.0), (-0.7, 4.0), (-0.4, 5.0)))
        fields = rng.normal(size=(4, 8, 16, 16))
        targets = np.asarray(((-1.0, 3.0), (-0.85, 3.5)))
        pnn, pnns, audit = development_priors(fields, features, targets, k=3)
        self.assertEqual(pnn.shape, (2, 8, 16, 16))
        self.assertEqual(pnns.shape, pnn.shape)
        self.assertTrue(all(record["target_condition_excluded"] for record in audit))
        self.assertNotIn(1, audit[0]["selected_development_indices"])


if __name__ == "__main__":
    unittest.main()
