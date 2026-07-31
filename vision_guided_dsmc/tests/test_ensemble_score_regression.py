import json
import numpy as np

from vgdsmc.ensemble_score_regression import (
    EnsembleScoreConfig,
    train_ensemble_score_regression,
)


def _write_case(path, seed, knudsen, delta):
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, 6),
        np.linspace(0.0, 1.0, 6),
        indexing="ij",
    )
    rng = np.random.default_rng(seed + int(1000 * knudsen + delta))
    temperature = 300.0 + delta * (0.5 - xx) + rng.normal(0.0, 0.5, xx.shape)
    u = 2.0 * yy + rng.normal(0.0, 0.05, xx.shape)
    v = -1.5 * xx + rng.normal(0.0, 0.05, xx.shape)
    sigma = np.full_like(xx, 0.5 + 0.1 * knudsen)
    x = np.stack([temperature, u, v, sigma]).astype(np.float32)
    expected = 0.03 + 0.04 * xx + 0.02 * yy + 0.05 * knudsen + 0.0005 * delta
    score = np.maximum(expected + rng.normal(0.0, 0.004, xx.shape), 0.0).astype(np.float32)
    context = np.array([knudsen, 2.0 * delta / 300.0], dtype=np.float32)
    np.savez_compressed(
        path,
        x=x,
        score=score,
        context=context,
        case_seed=np.int64(seed),
    )


def test_ensemble_score_training(tmp_path):
    paths = []
    for knudsen in (0.05, 0.10):
        for delta in (20.0, 40.0):
            for seed in (11, 22, 33, 44, 55):
                path = tmp_path / f"k{knudsen}_d{delta}_s{seed}.npz"
                _write_case(path, seed, knudsen, delta)
                paths.append(path)
    model = train_ensemble_score_regression(
        paths,
        tmp_path / "training",
        EnsembleScoreConfig(
            epochs=4,
            batch_size=4,
            validation_seed_count=2,
            seed=3,
        ),
    )
    assert model.exists()
    metrics = json.loads((tmp_path / "training" / "metrics.json").read_text())
    assert metrics["case_count"] == 20
    assert metrics["condition_count"] == 4
    assert metrics["train_seeds"] == [11, 22, 33]
    assert metrics["validation_seeds"] == [44, 55]
    assert metrics["input_channels"] == 10
    assert np.isfinite(metrics["ensemble_group_mae"])
    assert np.isfinite(metrics["ensemble_group_spearman"])
