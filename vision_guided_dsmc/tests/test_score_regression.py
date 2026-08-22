import json
import numpy as np

from vgdsmc.score_regression import ScoreTrainConfig, load_score_cases, train_score_regression


def _write_case(path, offset):
    yy, xx = np.meshgrid(np.linspace(0, 1, 8), np.linspace(0, 1, 8), indexing="ij")
    x = np.stack([
        300.0 + 20.0 * xx + offset,
        2.0 * yy,
        -1.5 * xx,
        0.5 + 0.2 * yy,
    ]).astype(np.float32)
    score = (0.02 + 0.03 * xx + 0.01 * yy + 0.001 * offset).astype(np.float32)
    np.savez_compressed(path, x=x, score=score, label=np.zeros((8, 8), dtype=np.int64))


def test_score_case_loading_and_training(tmp_path):
    paths = []
    for index in range(4):
        path = tmp_path / f"case_{index}.npz"
        _write_case(path, float(index))
        paths.append(path)
    x, score = load_score_cases(paths)
    assert x.shape == (4, 4, 8, 8)
    assert score.shape == (4, 8, 8)
    model = train_score_regression(
        paths,
        tmp_path / "training",
        ScoreTrainConfig(epochs=3, batch_size=2, seed=3),
    )
    assert model.exists()
    metrics = json.loads((tmp_path / "training" / "metrics.json").read_text())
    assert np.isfinite(metrics["validation_mae"])
    assert np.isfinite(metrics["validation_spearman"])
    with np.load(tmp_path / "training" / "validation_predictions.npz") as data:
        assert data["truth"].shape == data["prediction"].shape
