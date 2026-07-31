from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import json
import numpy as np


@dataclass(frozen=True)
class EnsembleScoreConfig:
    epochs: int = 80
    learning_rate: float = 5.0e-4
    batch_size: int = 6
    validation_seed_count: int = 2
    gradient_weight: float = 0.20
    mean_weight: float = 0.10
    seed: int = 7


def _rank_average(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).ravel()
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float).ravel()
    second = np.asarray(second, dtype=float).ravel()
    if np.std(first) <= 1.0e-14 or np.std(second) <= 1.0e-14:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def _condition_key(context: np.ndarray) -> tuple[float, ...]:
    return tuple(np.round(np.asarray(context, dtype=float), 8).tolist())


def _augment_features(x: np.ndarray, context: np.ndarray) -> np.ndarray:
    if x.ndim != 3 or x.shape[0] < 4:
        raise ValueError("Expected x with shape (at least 4, ny, nx)")
    ny, nx = x.shape[1:]
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, ny, dtype=np.float32),
        np.linspace(-1.0, 1.0, nx, dtype=np.float32),
        indexing="ij",
    )
    temperature = np.asarray(x[0], dtype=np.float64)
    sigma_temperature = np.asarray(x[3], dtype=np.float64)
    temperature_scale = max(float(np.mean(np.abs(temperature))), 1.0e-6)
    gradient_y, gradient_x = np.gradient(temperature)
    gradient = np.hypot(gradient_x, gradient_y) / temperature_scale
    relative_sigma = sigma_temperature / temperature_scale
    context_maps = np.broadcast_to(
        context[:, None, None], (len(context), ny, nx)
    ).astype(np.float32)
    derived = np.stack(
        [xx, yy, gradient.astype(np.float32), relative_sigma.astype(np.float32)],
        axis=0,
    )
    return np.concatenate([x.astype(np.float32), context_maps, derived], axis=0)


def _load_cases(paths: list[str | Path]):
    features: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    contexts: list[np.ndarray] = []
    seeds: list[int] = []
    shape: tuple[int, int] | None = None
    channels: int | None = None
    for index, path in enumerate(paths):
        with np.load(path) as data:
            if "x" not in data or "score" not in data or "context" not in data:
                raise ValueError(f"Missing x, score, or context in {path}")
            x = np.asarray(data["x"], dtype=np.float32)
            score = np.asarray(data["score"], dtype=np.float32)
            context = np.asarray(data["context"], dtype=np.float32)
            case_seed = int(data["case_seed"]) if "case_seed" in data else index
        if score.ndim != 2 or x.shape[1:] != score.shape:
            raise ValueError(f"Invalid spatial shape in {path}")
        if context.ndim != 1 or len(context) < 2 or not np.isfinite(context).all():
            raise ValueError(f"Invalid context in {path}")
        augmented = _augment_features(x, context)
        if shape is None:
            shape = score.shape
            channels = augmented.shape[0]
        elif score.shape != shape or augmented.shape[0] != channels:
            raise ValueError("All cases must share grid shape and feature count")
        if not np.isfinite(augmented).all() or not np.isfinite(score).all():
            raise ValueError(f"Non-finite data in {path}")
        if np.any(score < 0.0):
            raise ValueError(f"Negative score in {path}")
        features.append(augmented)
        scores.append(score)
        contexts.append(context)
        seeds.append(case_seed)
    if len(features) < 4:
        raise ValueError("At least four cases are required")
    return (
        np.stack(features),
        np.stack(scores),
        np.stack(contexts),
        np.asarray(seeds, dtype=np.int64),
    )


def _split_seeds(seeds: np.ndarray, validation_seed_count: int):
    unique = np.unique(seeds)
    if len(unique) < validation_seed_count + 1:
        raise ValueError("Not enough unique seeds for ensemble train/validation split")
    validation_seeds = unique[-validation_seed_count:]
    validation = np.flatnonzero(np.isin(seeds, validation_seeds))
    training = np.flatnonzero(~np.isin(seeds, validation_seeds))
    return training, validation, validation_seeds


def _ensemble_targets(
    score: np.ndarray,
    contexts: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, dict[tuple[float, ...], np.ndarray]]:
    templates: dict[tuple[float, ...], np.ndarray] = {}
    for key in sorted({_condition_key(contexts[index]) for index in indices}):
        members = [
            index for index in indices if _condition_key(contexts[index]) == key
        ]
        templates[key] = np.mean(score[members], axis=0)
    assigned = np.stack([templates[_condition_key(contexts[index])] for index in indices])
    return assigned.astype(np.float32), templates


def _group_validation(
    prediction: np.ndarray,
    truth: np.ndarray,
    contexts: np.ndarray,
):
    grouped_prediction: list[np.ndarray] = []
    grouped_truth: list[np.ndarray] = []
    keys = sorted({_condition_key(context) for context in contexts})
    for key in keys:
        members = [index for index, context in enumerate(contexts) if _condition_key(context) == key]
        grouped_prediction.append(np.mean(prediction[members], axis=0))
        grouped_truth.append(np.mean(truth[members], axis=0))
    return np.stack(grouped_prediction), np.stack(grouped_truth), keys


def train_ensemble_score_regression(
    case_paths: list[str | Path],
    output_dir: str | Path,
    cfg: EnsembleScoreConfig = EnsembleScoreConfig(),
) -> Path:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise ImportError("Install optional ML dependencies: pip install -e '.[ml]'") from exc

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    x, score, contexts, seeds = _load_cases(case_paths)
    training_indices, validation_indices, validation_seeds = _split_seeds(
        seeds, cfg.validation_seed_count
    )
    train_target, templates = _ensemble_targets(score, contexts, training_indices)
    train_x = x[training_indices]
    validation_x = x[validation_indices]
    validation_truth = score[validation_indices]
    validation_context = contexts[validation_indices]

    mean = train_x.mean(axis=(0, 2, 3), keepdims=True)
    std = np.maximum(train_x.std(axis=(0, 2, 3), keepdims=True), 1.0e-6)
    train_x = ((train_x - mean) / std).astype(np.float32)
    validation_x = ((validation_x - mean) / std).astype(np.float32)

    positive = train_target[train_target > 0.0]
    target_scale = max(float(np.median(positive)) if positive.size else 1.0, 1.0e-8)
    train_log_target = np.log1p(train_target / target_scale).astype(np.float32)
    input_channels = train_x.shape[1]

    class ResidualBlock(nn.Module):
        def __init__(self, channels: int, dilation: int) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
                nn.GroupNorm(8, channels),
                nn.GELU(),
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.GroupNorm(8, channels),
            )
            self.activation = nn.GELU()

        def forward(self, values):
            return self.activation(values + self.block(values))

    class EnsembleScoreCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(input_channels, 48, 3, padding=1),
                nn.GroupNorm(8, 48),
                nn.GELU(),
            )
            self.blocks = nn.Sequential(
                ResidualBlock(48, 1),
                ResidualBlock(48, 2),
                ResidualBlock(48, 3),
            )
            self.head = nn.Sequential(
                nn.Conv2d(48, 24, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(24, 1, 1),
                nn.Softplus(),
            )

        def forward(self, values):
            return self.head(self.blocks(self.stem(values))).squeeze(1)

    model = EnsembleScoreCNN()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=1.0e-4
    )
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_log_target)),
        batch_size=min(cfg.batch_size, len(train_x)),
        shuffle=True,
    )
    smooth_l1 = nn.SmoothL1Loss()
    history: list[float] = []
    model.train()
    for _ in range(cfg.epochs):
        total = 0.0
        for features, target in loader:
            optimizer.zero_grad()
            predicted = model(features)
            value_loss = smooth_l1(predicted, target)
            gradient_loss = (
                torch.mean(torch.abs(torch.diff(predicted, dim=1) - torch.diff(target, dim=1)))
                + torch.mean(torch.abs(torch.diff(predicted, dim=2) - torch.diff(target, dim=2)))
            )
            mean_loss = torch.mean(
                torch.abs(predicted.mean(dim=(1, 2)) - target.mean(dim=(1, 2)))
            )
            loss = value_loss + cfg.gradient_weight * gradient_loss + cfg.mean_weight * mean_loss
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(features)
        history.append(total / len(train_x))

    model.eval()
    with torch.no_grad():
        predicted_log = model(torch.from_numpy(validation_x)).numpy()
    prediction = np.maximum(target_scale * np.expm1(predicted_log), 0.0)

    group_prediction, group_truth, group_keys = _group_validation(
        prediction, validation_truth, validation_context
    )
    group_mae = float(np.mean(np.abs(group_prediction - group_truth)))
    group_rmse = float(np.sqrt(np.mean((group_prediction - group_truth) ** 2)))
    group_pearson = _correlation(group_prediction, group_truth)
    group_spearman = _correlation(
        _rank_average(group_prediction), _rank_average(group_truth)
    )

    global_constant = float(np.median(train_target))
    global_constant_mae = float(np.mean(np.abs(group_truth - global_constant)))
    condition_scalar_maps: list[np.ndarray] = []
    spatial_template_maps: list[np.ndarray] = []
    for key in group_keys:
        template = templates[key]
        condition_scalar_maps.append(np.full_like(template, np.median(template)))
        spatial_template_maps.append(template)
    condition_scalar = np.stack(condition_scalar_maps)
    spatial_template = np.stack(spatial_template_maps)
    condition_scalar_mae = float(np.mean(np.abs(group_truth - condition_scalar)))
    spatial_template_mae = float(np.mean(np.abs(group_truth - spatial_template)))

    raw_mae = float(np.mean(np.abs(prediction - validation_truth)))
    raw_spearman = _correlation(
        _rank_average(prediction), _rank_average(validation_truth)
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "ensemble_score_model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "mean": mean,
            "std": std,
            "target_scale": target_scale,
            "history": history,
            "input_channels": input_channels,
            "architecture": "coordinate_conditioned_residual_ensemble_score_cnn",
            "training_seeds": sorted(set(seeds[training_indices].tolist())),
            "validation_seeds": validation_seeds.tolist(),
        },
        model_path,
    )
    metrics = {
        "config": asdict(cfg),
        "case_count": len(x),
        "condition_count": len(group_keys),
        "train_cases": len(training_indices),
        "validation_cases": len(validation_indices),
        "train_seeds": sorted(set(seeds[training_indices].tolist())),
        "validation_seeds": validation_seeds.tolist(),
        "input_channels": input_channels,
        "final_training_loss": history[-1],
        "raw_validation_mae": raw_mae,
        "raw_validation_spearman": raw_spearman,
        "ensemble_group_mae": group_mae,
        "ensemble_group_rmse": group_rmse,
        "ensemble_group_pearson": group_pearson,
        "ensemble_group_spearman": group_spearman,
        "global_constant_mae": global_constant_mae,
        "condition_scalar_mae": condition_scalar_mae,
        "training_spatial_template_mae": spatial_template_mae,
        "mae_ratio_to_global_constant": group_mae / max(global_constant_mae, 1.0e-14),
        "mae_ratio_to_condition_scalar": group_mae / max(condition_scalar_mae, 1.0e-14),
        "target_scale": target_scale,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        output_dir / "validation_predictions.npz",
        raw_truth=validation_truth,
        raw_prediction=prediction,
        group_truth=group_truth,
        group_prediction=group_prediction,
        condition_scalar=condition_scalar,
        spatial_template=spatial_template,
        validation_seeds=seeds[validation_indices],
        validation_context=validation_context,
    )
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train coordinate-conditioned regression on ensemble-averaged DSMC error targets"
    )
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--output-dir", default="outputs/ensemble_score_regression")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--validation-seed-count", type=int, default=2)
    args = parser.parse_args()
    model_path = train_ensemble_score_regression(
        args.cases,
        args.output_dir,
        EnsembleScoreConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            validation_seed_count=args.validation_seed_count,
        ),
    )
    metrics = json.loads(
        (Path(args.output_dir) / "metrics.json").read_text(encoding="utf-8")
    )
    print(json.dumps({"model": str(model_path), **metrics}, indent=2))


if __name__ == "__main__":
    main()
