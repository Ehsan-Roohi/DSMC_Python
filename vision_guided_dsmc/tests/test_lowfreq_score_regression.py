import json
import numpy as np

from vgdsmc.lowfreq_score_regression import (
    LowFrequencyScoreConfig,
    _percentile_rank_maps,
    _smooth,
    train_low_frequency_score_regression,
)


def _write_case(path, seed, context, offset):
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, 8),
        np.linspace(0.0, 1.0, 8),
        indexing="ij",
    )
    rng = np.random.default_rng(seed)
    x = np.stack(
        [
            300.0 + 20.0 * xx + offset + rng.normal(0.0, 0.2, xx.shape),
            2.0 * yy,
            -1.5 * xx,
            0.5 + 0.2 * yy,
        ]
    ).astype(np.float32)
    score = (
        0.02
        + 0.08 * np.exp(-((xx - 0.25) ** 2 + (yy - 0.65) ** 2) / 0.08)
        + 0.005 * rng.random(xx.shape)
    ).astype(np.float32)
    np.savez_compressed(
        path,
        x=x,
        score=score,
        context=np.asarray(context, dtype=np.float32),
        case_seed=np.int64(seed),
    )


def test_smoothing_and_rank_maps():
    values = np.zeros((2, 5, 5), dtype=np.float32)
    values[:, 2, 2] = 1.0
    smoothed = _smooth(values, 2)
    ranked = _percentile_rank_maps(smoothed)
    assert smoothed.shape == values.shape
    assert ranked.shape == values.shape
    assert np.all((ranked >= 0.0) & (ranked <= 1.0))
    assert float(smoothed[0, 2, 2]) < 1.0


def test_low_frequency_training(tmp_path):
    paths = []
    conditions = [(0.05, 0.2), (0.10, 0.4)]
    for condition_index, context in enumerate(conditions):
        for seed in (11, 22, 33):
            path = tmp_path / f"case_{condition_index}_{seed}.npz"
            _write_case(path, seed, context, float(condition_index))
            paths.append(path)
    model = train_low_frequency_score_regression(
        paths,
        tmp_path / "training",
        LowFrequencyScoreConfig(
            epochs=3,
            batch_size=2,
            validation_seed_count=1,
            smoothing_passes=1,
            seed=3,
        ),
    )
    assert model.exists()
    metrics = json.loads((tmp_path / "training" / "metrics.json").read_text())
    assert np.isfinite(metrics["rank_validation_mae"])
    assert np.isfinite(metrics["rank_validation_spearman"])
    assert np.isfinite(metrics["top_quartile_iou"])
