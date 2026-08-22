from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import json
import numpy as np

from .ensemble_score_regression import (
    _condition_key,
    _correlation,
    _ensemble_targets,
    _group_validation,
    _load_cases,
    _rank_average,
    _split_seeds,
)


@dataclass(frozen=True)
class LowFrequencyScoreConfig:
    epochs: int = 100
    learning_rate: float = 4.0e-4
    batch_size: int = 6
    validation_seed_count: int = 2
    smoothing_passes: int = 2
    pairwise_weight: float = 0.20
    mean_weight: float = 0.10
    seed: int = 7


def _smooth(field: np.ndarray, passes: int) -> np.ndarray:
    result = np.asarray(field, dtype=np.float32)
    for _ in range(max(0, passes)):
        padded = np.pad(result, ((0, 0), (1, 1), (1, 1)), mode="edge")
        updated = np.zeros_like(result)
        for j in range(3):
            for i in range(3):
                updated += padded[:, j : j + result.shape[1], i : i + result.shape[2]]
        result = updated / 9.0
    return result


def _percentile_rank_maps(values: np.ndarray) -> np.ndarray:
    ranked = np.empty_like(values, dtype=np.float32)
    cell_count = values.shape[1] * values.shape[2]
    denominator = max(cell_count - 1, 1)
    for index, field in enumerate(values):
        ranked[index] = (_rank_average(field) / denominator).reshape(field.shape)
    return ranked


def _top_fraction_iou(prediction: np.ndarray, truth: np.ndarray, fraction: float = 0.25) -> float:
    scores: list[float] = []
    count = prediction.shape[1] * prediction.shape[2]
    selected = max(1, int(round(count * fraction)))
    for predicted_field, truth_field in zip(prediction, truth):
        predicted_ids = set(np.argpartition(predicted_field.ravel(), -selected)[-selected:].tolist())
        truth_ids = set(np.argpartition(truth_field.ravel(), -selected)[-selected:].tolist())
        scores.append(len(predicted_ids & truth_ids) / max(len(predicted_ids | truth_ids), 1))
    return float(np.mean(scores))


def train_low_frequency_score_regression(
    case_paths: list[str | Path],
    output_dir: str | Path,
    cfg: LowFrequencyScoreConfig = LowFrequencyScoreConfig(),
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
    train_ensemble, templates = _ensemble_targets(score, contexts, training_indices)
    train_smoothed = _smooth(train_ensemble, cfg.smoothing_passes)
    train_rank = _percentile_rank_maps(train_smoothed)

    train_x = x[training_indices]
    validation_x = x[validation_indices]
    validation_truth_raw = score[validation_indices]
    validation_context = contexts[validation_indices]

    mean = train_x.mean(axis=(0, 2, 3), keepdims=True)
    std = np.maximum(train_x.std(axis=(0, 2, 3), keepdims=True), 1.0e-6)
    train_x = ((train_x - mean) / std).astype(np.float32)
    validation_x = ((validation_x - mean) / std).astype(np.float32)
    input_channels = train_x.shape[1]

    class ResidualBlock(nn.Module):
        def __init__(self, channels: int, dilation: int) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
                nn.GroupNorm(8, channels),
                nn.GELU(),
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.GroupNorm(8, channels),
            )
            self.activation = nn.GELU()

        def forward(self, values):
            return self.activation(values + self.layers(values))

    class LowFrequencyCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(input_channels, 48, 3, padding=1),
                nn.GroupNorm(8, 48),
                nn.GELU(),
            )
            self.body = nn.Sequential(
                ResidualBlock(48, 1),
                ResidualBlock(48, 2),
                ResidualBlock(48, 3),
            )
            self.head = nn.Sequential(
                nn.Conv2d(48, 24, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(24, 1, 1),
                nn.Sigmoid(),
            )

        def forward(self, values):
            return self.head(self.body(self.stem(values))).squeeze(1)

    model = LowFrequencyCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1.0e-4)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_rank)),
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
            flat_predicted = predicted.flatten(1)
            flat_target = target.flatten(1)
            permutation = torch.randperm(flat_predicted.shape[1])
            difference_predicted = flat_predicted - flat_predicted[:, permutation]
            difference_target = flat_target - flat_target[:, permutation]
            sign = torch.sign(difference_target)
            valid = sign != 0
            pairwise = torch.nn.functional.softplus(-difference_predicted * sign)
            pairwise_loss = pairwise[valid].mean() if torch.any(valid) else value_loss * 0.0
            mean_loss = torch.mean(
                torch.abs(predicted.mean(dim=(1, 2)) - target.mean(dim=(1, 2)))
            )
            loss = value_loss + cfg.pairwise_weight * pairwise_loss + cfg.mean_weight * mean_loss
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(features)
        history.append(total / len(train_x))

    model.eval()
    with torch.no_grad():
        raw_prediction = model(torch.from_numpy(validation_x)).numpy()

    group_prediction, group_truth_raw, group_keys = _group_validation(
        raw_prediction, validation_truth_raw, validation_context
    )
    group_truth_smoothed = _smooth(group_truth_raw, cfg.smoothing_passes)
    group_truth_rank = _percentile_rank_maps(group_truth_smoothed)

    rank_mae = float(np.mean(np.abs(group_prediction - group_truth_rank)))
    rank_rmse = float(np.sqrt(np.mean((group_prediction - group_truth_rank) ** 2)))
    rank_pearson = _correlation(group_prediction, group_truth_rank)
    rank_spearman = _correlation(_rank_average(group_prediction), _rank_average(group_truth_rank))
    top_quartile_iou = _top_fraction_iou(group_prediction, group_truth_rank)

    constant_rank = np.full_like(group_truth_rank, 0.5)
    constant_rank_mae = float(np.mean(np.abs(group_truth_rank - constant_rank)))

    condition_scalar_maps: list[np.ndarray] = []
    training_template_rank_maps: list[np.ndarray] = []
    for key in group_keys:
        template = _smooth(templates[key][None], cfg.smoothing_passes)[0]
        condition_scalar_maps.append(np.full_like(template, 0.5))
        training_template_rank_maps.append(_percentile_rank_maps(template[None])[0])
    condition_scalar_rank = np.stack(condition_scalar_maps)
    training_template_rank = np.stack(training_template_rank_maps)
    condition_scalar_rank_mae = float(np.mean(np.abs(group_truth_rank - condition_scalar_rank)))
    training_template_rank_mae = float(np.mean(np.abs(group_truth_rank - training_template_rank)))
    training_template_spearman = _correlation(
        _rank_average(training_template_rank), _rank_average(group_truth_rank)
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "lowfreq_score_model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "mean": mean,
            "std": std,
            "history": history,
            "input_channels": input_channels,
            "smoothing_passes": cfg.smoothing_passes,
            "target": "within_case_percentile_rank_of_smoothed_ensemble_error",
            "architecture": "coordinate_conditioned_low_frequency_rank_cnn",
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
        "rank_validation_mae": rank_mae,
        "rank_validation_rmse": rank_rmse,
        "rank_validation_pearson": rank_pearson,
        "rank_validation_spearman": rank_spearman,
        "top_quartile_iou": top_quartile_iou,
        "constant_rank_mae": constant_rank_mae,
        "condition_scalar_rank_mae": condition_scalar_rank_mae,
        "training_template_rank_mae": training_template_rank_mae,
        "training_template_spearman": training_template_spearman,
        "mae_ratio_to_constant_rank": rank_mae / max(constant_rank_mae, 1.0e-14),
        "mae_ratio_to_training_template_rank": rank_mae / max(training_template_rank_mae, 1.0e-14),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez_compressed(
        output_dir / "validation_predictions.npz",
        group_truth_raw=group_truth_raw,
        group_truth_smoothed=group_truth_smoothed,
        group_truth_rank=group_truth_rank,
        group_prediction_rank=group_prediction,
        training_template_rank=training_template_rank,
        validation_context=validation_context,
    )
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train low-frequency rank regression on ensemble-averaged DSMC error targets"
    )
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--output-dir", default="outputs/lowfreq_score_regression")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=4.0e-4)
    parser.add_argument("--validation-seed-count", type=int, default=2)
    parser.add_argument("--smoothing-passes", type=int, default=2)
    args = parser.parse_args()
    model_path = train_low_frequency_score_regression(
        args.cases,
        args.output_dir,
        LowFrequencyScoreConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            validation_seed_count=args.validation_seed_count,
            smoothing_passes=args.smoothing_passes,
        ),
    )
    metrics = json.loads((Path(args.output_dir) / "metrics.json").read_text(encoding="utf-8"))
    print(json.dumps({"model": str(model_path), **metrics}, indent=2))


if __name__ == "__main__":
    main()
