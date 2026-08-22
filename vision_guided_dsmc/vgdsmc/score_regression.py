from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import json
import numpy as np


@dataclass(frozen=True)
class ScoreTrainConfig:
    epochs: int = 30
    learning_rate: float = 1.0e-3
    batch_size: int = 4
    validation_fraction: float = 0.25
    split_mode: str = "seed"
    seed: int = 7


def _load_records(paths: list[str | Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    seeds: list[int] = []
    expected_shape: tuple[int, int] | None = None
    expected_channels: int | None = None
    for index, path in enumerate(paths):
        with np.load(path) as data:
            x = np.asarray(data["x"], dtype=np.float32)
            score = np.asarray(data["score"], dtype=np.float32)
            context = np.asarray(data["context"], dtype=np.float32) if "context" in data else None
            case_seed = int(data["case_seed"]) if "case_seed" in data else index
        if x.ndim != 3 or x.shape[0] < 4 or score.shape != x.shape[1:]:
            raise ValueError(f"Invalid supervised case shape in {path}")
        if context is not None:
            if context.ndim != 1 or not np.isfinite(context).all():
                raise ValueError(f"Invalid physical context in {path}")
            maps = np.broadcast_to(context[:, None, None], (len(context), *score.shape))
            x = np.concatenate([x, maps.astype(np.float32)], axis=0)
        if expected_shape is None:
            expected_shape = score.shape
            expected_channels = x.shape[0]
        elif score.shape != expected_shape or x.shape[0] != expected_channels:
            raise ValueError("All supervised cases must use the same grid and channels")
        if not np.isfinite(x).all() or not np.isfinite(score).all() or np.any(score < 0.0):
            raise ValueError(f"Non-finite or negative training data in {path}")
        xs.append(x)
        ys.append(score)
        seeds.append(case_seed)
    if len(xs) < 2:
        raise ValueError("At least two supervised cases are required")
    return np.stack(xs), np.stack(ys), np.asarray(seeds, dtype=np.int64)


def load_score_cases(paths: list[str | Path]) -> tuple[np.ndarray, np.ndarray]:
    x, score, _ = _load_records(paths)
    return x, score


def _rank_average(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        ranks[order[index:end]] = 0.5 * (index + end - 1)
        index = end
    return ranks


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float).ravel()
    second = np.asarray(second, dtype=float).ravel()
    if np.std(first) <= 1.0e-14 or np.std(second) <= 1.0e-14:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def _split_indices(seeds: np.ndarray, cfg: ScoreTrainConfig) -> tuple[np.ndarray, np.ndarray, int | None]:
    if cfg.split_mode == "seed" and len(np.unique(seeds)) >= 2:
        holdout_seed = int(np.max(np.unique(seeds)))
        validation = np.flatnonzero(seeds == holdout_seed)
        training = np.flatnonzero(seeds != holdout_seed)
        if len(training) and len(validation):
            return training, validation, holdout_seed
    if cfg.split_mode not in {"seed", "tail"}:
        raise ValueError("split_mode must be 'seed' or 'tail'")
    count = len(seeds)
    validation_count = min(max(1, int(round(cfg.validation_fraction * count))), count - 1)
    return np.arange(count - validation_count), np.arange(count - validation_count, count), None


def train_score_regression(case_paths, output_dir, cfg: ScoreTrainConfig = ScoreTrainConfig()) -> Path:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise ImportError("Install optional ML dependencies: pip install -e '.[ml]'") from exc

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    x, score, case_seeds = _load_records(case_paths)
    training_indices, validation_indices, holdout_seed = _split_indices(case_seeds, cfg)
    train_x, validation_x = x[training_indices], x[validation_indices]
    train_score, validation_score = score[training_indices], score[validation_indices]
    mean = train_x.mean(axis=(0, 2, 3), keepdims=True)
    std = np.maximum(train_x.std(axis=(0, 2, 3), keepdims=True), 1.0e-6)
    train_x = (train_x - mean) / std
    validation_x = (validation_x - mean) / std
    positive = train_score[train_score > 0.0]
    target_scale = max(float(np.median(positive)) if positive.size else 1.0, 1.0e-8)
    train_target = np.log1p(train_score / target_scale).astype(np.float32)
    input_channels = train_x.shape[1]

    class ScoreCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Conv2d(input_channels, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=2, dilation=2), nn.ReLU(),
                nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
                nn.Conv2d(16, 1, 1),
            )
        def forward(self, values):
            return self.network(values).squeeze(1)

    model = ScoreCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    loss_function = torch.nn.SmoothL1Loss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_target)),
        batch_size=min(cfg.batch_size, len(train_x)), shuffle=True,
    )
    history: list[float] = []
    model.train()
    for _ in range(cfg.epochs):
        total = 0.0
        for features, target in loader:
            optimizer.zero_grad()
            loss = loss_function(model(features), target)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(features)
        history.append(total / len(train_x))

    model.eval()
    with torch.no_grad():
        predicted_log = model(torch.from_numpy(validation_x)).numpy()
    prediction = np.maximum(target_scale * np.expm1(predicted_log), 0.0)
    truth = validation_score
    mae = float(np.mean(np.abs(prediction - truth)))
    rmse = float(np.sqrt(np.mean((prediction - truth) ** 2)))
    constant = float(np.median(train_score))
    baseline_mae = float(np.mean(np.abs(truth - constant)))
    pearson = _correlation(prediction, truth)
    spearman = _correlation(_rank_average(prediction.ravel()), _rank_average(truth.ravel()))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "score_model.pt"
    torch.save({
        "state_dict": model.state_dict(), "mean": mean, "std": std,
        "target_scale": target_scale, "history": history,
        "input_channels": input_channels,
        "architecture": "context_conditioned_dilated_score_cnn",
    }, model_path)
    metrics = {
        "config": asdict(cfg), "case_count": len(x),
        "train_cases": len(training_indices), "validation_cases": len(validation_indices),
        "train_seeds": sorted(set(case_seeds[training_indices].tolist())),
        "validation_seeds": sorted(set(case_seeds[validation_indices].tolist())),
        "holdout_seed": holdout_seed, "input_channels": input_channels,
        "final_training_loss": history[-1], "validation_mae": mae,
        "validation_rmse": rmse, "baseline_constant_mae": baseline_mae,
        "mae_ratio_to_constant": mae / max(baseline_mae, 1.0e-14),
        "validation_pearson": pearson, "validation_spearman": spearman,
        "target_scale": target_scale,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez_compressed(
        output_dir / "validation_predictions.npz",
        truth=truth, prediction=prediction,
        case_seeds=case_seeds[validation_indices],
    )
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train continuous DSMC local-error regression against deterministic references")
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--output-dir", default="outputs/score_regression")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--split-mode", choices=("seed", "tail"), default="seed")
    args = parser.parse_args()
    model_path = train_score_regression(
        args.cases, args.output_dir,
        ScoreTrainConfig(
            epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.learning_rate, split_mode=args.split_mode,
        ),
    )
    metrics = json.loads((Path(args.output_dir) / "metrics.json").read_text(encoding="utf-8"))
    print(json.dumps({"model": str(model_path), **metrics}, indent=2))


if __name__ == "__main__":
    main()
