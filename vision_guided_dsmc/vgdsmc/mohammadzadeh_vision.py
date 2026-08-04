"""Leakage-safe image-to-image denoising for the validated Mohammadzadeh fields.

The stage deliberately excludes every heat-flux component.  Ten temporal block
fields from each M3 seed are treated as noisy images.  A seed-wise split and a
leave-one-seed-out target prevent blocks from the held-out seed leaking into its
reference image.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

SEEDS = tuple(range(91901, 91909))
TRAIN_SEEDS = SEEDS[:6]
VALIDATION_SEEDS = (SEEDS[6],)
TEST_SEEDS = (SEEDS[7],)
INPUT_FIELDS = ("T", "u", "v", "rho", "count")
OUTPUT_FIELDS = ("T", "u")
STAGE = "MV1_Mohammadzadeh_validated_field_vision"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_m3_images(root: Path) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Load 80 block images and eight converged fields without q data."""
    blocks: dict[int, np.ndarray] = {}
    full: dict[int, np.ndarray] = {}
    expected_shape: tuple[int, int] | None = None
    for seed in SEEDS:
        directory = root / f"seed_{seed}"
        with np.load(directory / "block_fields.npz", allow_pickle=False) as data:
            missing = set(INPUT_FIELDS) - set(data.files)
            if missing:
                raise ValueError(f"seed {seed} block fields missing {sorted(missing)}")
            image = np.stack([np.asarray(data[name], dtype=np.float32) for name in INPUT_FIELDS], axis=1)
        with np.load(directory / "fields.npz", allow_pickle=False) as data:
            missing = set(OUTPUT_FIELDS) - set(data.files)
            if missing:
                raise ValueError(f"seed {seed} fields missing {sorted(missing)}")
            target = np.stack([np.asarray(data[name], dtype=np.float32) for name in OUTPUT_FIELDS], axis=0)
        if image.ndim != 4 or target.ndim != 3 or image.shape[1] != len(INPUT_FIELDS):
            raise ValueError(f"seed {seed} has an invalid image contract")
        if image.shape[-2:] != target.shape[-2:]:
            raise ValueError(f"seed {seed} block/full grids differ")
        if expected_shape is None:
            expected_shape = target.shape[-2:]
        if target.shape[-2:] != expected_shape or not np.all(np.isfinite(image)) or not np.all(np.isfinite(target)):
            raise ValueError(f"seed {seed} has inconsistent or non-finite fields")
        blocks[seed], full[seed] = image, target
    return blocks, full


def leave_one_seed_out_targets(full: Mapping[int, np.ndarray]) -> dict[int, np.ndarray]:
    targets: dict[int, np.ndarray] = {}
    for seed in SEEDS:
        others = [np.asarray(full[other], dtype=np.float64) for other in SEEDS if other != seed]
        targets[seed] = np.mean(others, axis=0).astype(np.float32)
    return targets


def build_arrays(
    blocks: Mapping[int, np.ndarray], targets: Mapping[int, np.ndarray], seeds: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, identity = [], [], []
    for seed in seeds:
        for block_index, image in enumerate(blocks[seed]):
            x.append(image)
            y.append(targets[seed])
            identity.append((seed, block_index))
    return np.stack(x), np.stack(y), np.asarray(identity, dtype=np.int64)


def fit_scaling(train_x: np.ndarray, train_y: np.ndarray) -> dict[str, np.ndarray]:
    input_mean = train_x.mean(axis=(0, 2, 3), keepdims=True)
    input_std = np.maximum(train_x.std(axis=(0, 2, 3), keepdims=True), 1.0e-6)
    residual = train_y - train_x[:, : len(OUTPUT_FIELDS)]
    residual_std = np.maximum(residual.std(axis=(0, 2, 3), keepdims=True), 1.0e-4)
    return {
        "input_mean": input_mean.astype(np.float32),
        "input_std": input_std.astype(np.float32),
        "residual_std": residual_std.astype(np.float32),
    }


def build_model(in_channels: int = 5, out_channels: int = 2, base: int = 12):
    try:
        import torch
        from torch import nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError("Install PyTorch with: python -m pip install -e '.[ml]'") from exc

    class Block(nn.Module):
        def __init__(self, cin: int, cout: int):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.SiLU(),
                nn.Conv2d(cout, cout, 3, padding=1), nn.SiLU(),
            )

        def forward(self, value):
            return self.layers(value)

    class ResidualUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.e1, self.e2 = Block(in_channels, base), Block(base, 2 * base)
            self.middle = Block(2 * base, 4 * base)
            self.pool = nn.MaxPool2d(2)
            self.up2 = nn.ConvTranspose2d(4 * base, 2 * base, 2, 2)
            self.up1 = nn.ConvTranspose2d(2 * base, base, 2, 2)
            self.d2, self.d1 = Block(4 * base, 2 * base), Block(2 * base, base)
            self.output = nn.Conv2d(base, out_channels, 1)

        @staticmethod
        def match(value, reference):
            return value if value.shape[-2:] == reference.shape[-2:] else F.interpolate(
                value, size=reference.shape[-2:], mode="bilinear", align_corners=False
            )

        def forward(self, value):
            a = self.e1(value)
            b = self.e2(self.pool(a))
            z = self.middle(self.pool(b))
            z = self.d2(torch.cat((self.match(self.up2(z), b), b), dim=1))
            z = self.d1(torch.cat((self.match(self.up1(z), a), a), dim=1))
            return self.output(z)

    return ResidualUNet()


def _loss(prediction, target):
    import torch

    pixel = torch.mean((prediction - target) ** 2)
    grad_x = torch.mean(((prediction[..., 1:] - prediction[..., :-1]) - (target[..., 1:] - target[..., :-1])) ** 2)
    grad_y = torch.mean(((prediction[..., 1:, :] - prediction[..., :-1, :]) - (target[..., 1:, :] - target[..., :-1, :])) ** 2)
    lid = torch.mean((prediction[..., -1, :] - target[..., -1, :]) ** 2)
    return pixel + 0.10 * (grad_x + grad_y) + 0.50 * lid


def train(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    scaling: Mapping[str, np.ndarray],
    *,
    epochs: int,
    batch_size: int,
    seed: int,
    output: Path,
) -> tuple[Any, dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.use_deterministic_algorithms(True, warn_only=True)

    def tensors(x: np.ndarray, y: np.ndarray):
        normalized_x = (x - scaling["input_mean"]) / scaling["input_std"]
        residual_y = (y - x[:, : len(OUTPUT_FIELDS)]) / scaling["residual_std"]
        return torch.from_numpy(normalized_x), torch.from_numpy(residual_y)

    tx, ty = tensors(train_x, train_y)
    vx, vy = tensors(validation_x, validation_y)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(tx, ty), batch_size=min(batch_size, len(tx)), shuffle=True,
        generator=generator, num_workers=0,
    )
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-5)
    best_state, best_value, best_epoch, stale = None, float("inf"), 0, 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            value = _loss(model(xb), yb)
            value.backward()
            optimizer.step()
            running += float(value.detach().cpu()) * len(xb)
        model.eval()
        with torch.no_grad():
            validation = float(_loss(model(vx.to(device)), vy.to(device)).cpu())
        history.append({"epoch": epoch, "train_loss": running / len(tx), "validation_loss": validation})
        if validation < best_value - 1.0e-7:
            best_value, best_epoch, stale = validation, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= 25:
            break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    model.to("cpu").eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage": STAGE, "state_dict": best_state,
            "input_fields": INPUT_FIELDS, "output_fields": OUTPUT_FIELDS,
            "scaling": {key: np.asarray(value) for key, value in scaling.items()},
            "train_seeds": TRAIN_SEEDS, "validation_seeds": VALIDATION_SEEDS,
            "test_seeds": TEST_SEEDS, "best_epoch": best_epoch,
        }, output,
    )
    return model, {
        "device": str(device), "epochs_completed": len(history),
        "best_epoch": best_epoch, "best_validation_loss": best_value,
        "history": history,
    }


def predict(model: Any, x: np.ndarray, scaling: Mapping[str, np.ndarray], batch_size: int = 8) -> np.ndarray:
    import torch

    normalized = ((x - scaling["input_mean"]) / scaling["input_std"]).astype(np.float32)
    results = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            residual = model(torch.from_numpy(normalized[start : start + batch_size])).numpy()
            results.append(x[start : start + batch_size, : len(OUTPUT_FIELDS)] + residual * scaling["residual_std"])
    return np.concatenate(results)


def nrmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm((prediction - target).ravel()) / max(np.linalg.norm((target - target.mean()).ravel()), 1.0e-12))


def evaluate(raw: np.ndarray, corrected: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    per_field = {}
    for index, name in enumerate(OUTPUT_FIELDS):
        baseline = nrmse(raw[:, index], target[:, index])
        vision = nrmse(corrected[:, index], target[:, index])
        per_field[name] = {"raw_nrmse": baseline, "vision_nrmse": vision, "ratio": vision / max(baseline, 1.0e-12)}
    raw_composite = float(np.mean([item["raw_nrmse"] for item in per_field.values()]))
    vision_composite = float(np.mean([item["vision_nrmse"] for item in per_field.values()]))
    vertical_index = int(round(0.8 * raw.shape[-1] - 0.5))
    profiles = {
        "macroscopic_lid_temperature": (
            raw[:, 0, -1, :], corrected[:, 0, -1, :], target[:, 0, -1, :]
        ),
        "vertical_temperature_x08": (
            raw[:, 0, :, vertical_index], corrected[:, 0, :, vertical_index], target[:, 0, :, vertical_index]
        ),
        "macroscopic_lid_slip": (
            1.0 - raw[:, 1, -1, :] / 100.0,
            1.0 - corrected[:, 1, -1, :] / 100.0,
            1.0 - target[:, 1, -1, :] / 100.0,
        ),
    }
    profile_metrics = {}
    for name, (raw_profile, vision_profile, target_profile) in profiles.items():
        baseline = nrmse(raw_profile, target_profile)
        vision = nrmse(vision_profile, target_profile)
        profile_metrics[name] = {
            "raw_nrmse": baseline, "vision_nrmse": vision,
            "ratio": vision / max(baseline, 1.0e-12),
        }
    return {
        "per_field": per_field,
        "validated_profiles": profile_metrics,
        "raw_composite_nrmse": raw_composite,
        "vision_composite_nrmse": vision_composite,
        "vision_over_raw_composite": vision_composite / max(raw_composite, 1.0e-12),
    }


def _draw(raw: np.ndarray, corrected: np.ndarray, target: np.ndarray, output: Path) -> None:
    import matplotlib.pyplot as plt

    sample = 0
    figure, axes = plt.subplots(2, 3, figsize=(12.0, 7.3), constrained_layout=True)
    titles = ("Low-sample block", "Vision reconstruction", "LOSO M3 target")
    for column, value in enumerate((raw[sample, 0], corrected[sample, 0], target[sample, 0])):
        levels = np.linspace(min(raw[sample, 0].min(), corrected[sample, 0].min(), target[sample, 0].min()), max(raw[sample, 0].max(), corrected[sample, 0].max(), target[sample, 0].max()), 21)
        plot = axes[0, column].contourf(value, levels=levels, cmap="coolwarm", extend="both")
        axes[0, column].set_title(titles[column])
        figure.colorbar(plot, ax=axes[0, column], label="T (K)")
    x = (np.arange(raw.shape[-1]) + 0.5) / raw.shape[-1]
    for column, value in enumerate((raw[sample], corrected[sample], target[sample])):
        axes[1, column].plot(x, 1.0 - value[1, -1] / 100.0, label="lid slip")
        axes[1, column].plot(x, (value[0, -1] - 300.0) / 10.0, label="(lid T-300)/10")
        axes[1, column].set(xlabel="x/L", xlim=(0, 1), title=titles[column])
        axes[1, column].grid(alpha=0.2)
        axes[1, column].legend(fontsize=8)
    figure.suptitle("Mohammadzadeh MV1: unsmoothed held-out-seed comparison")
    figure.savefig(output, dpi=220)
    plt.close(figure)


def run(m3_root: Path, output_dir: Path, *, epochs: int, batch_size: int, seed: int) -> dict[str, Any]:
    blocks, full = load_m3_images(m3_root)
    targets = leave_one_seed_out_targets(full)
    train_x, train_y, train_id = build_arrays(blocks, targets, TRAIN_SEEDS)
    val_x, val_y, val_id = build_arrays(blocks, targets, VALIDATION_SEEDS)
    test_x, test_y, test_id = build_arrays(blocks, targets, TEST_SEEDS)
    scaling = fit_scaling(train_x, train_y)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, training = train(
        train_x, train_y, val_x, val_y, scaling, epochs=epochs,
        batch_size=batch_size, seed=seed, output=output_dir / "model.pt",
    )
    corrected = predict(model, test_x, scaling, batch_size=batch_size)
    metrics = evaluate(test_x[:, :3], corrected, test_y)
    np.savez_compressed(
        output_dir / "heldout_predictions.npz", identity=test_id,
        raw=test_x[:, : len(OUTPUT_FIELDS)], corrected=corrected, target=test_y,
    )
    _draw(test_x[:, : len(OUTPUT_FIELDS)], corrected, test_y, output_dir / "heldout_contours_and_profiles.png")
    checks = {
        "seed_disjoint_split": not bool(set(TRAIN_SEEDS) & set(VALIDATION_SEEDS + TEST_SEEDS)) and not bool(set(VALIDATION_SEEDS) & set(TEST_SEEDS)),
        "heat_flux_excluded": not any(name.lower().startswith("q") for name in INPUT_FIELDS + OUTPUT_FIELDS),
        "heldout_composite_improves_5pct": metrics["vision_over_raw_composite"] <= 0.95,
        "heldout_temperature_not_worse": metrics["per_field"]["T"]["ratio"] <= 1.0,
        "heldout_lid_temperature_not_worse": metrics["validated_profiles"]["macroscopic_lid_temperature"]["ratio"] <= 1.0,
        "heldout_vertical_temperature_not_worse": metrics["validated_profiles"]["vertical_temperature_x08"]["ratio"] <= 1.0,
        "heldout_lid_slip_not_worse": metrics["validated_profiles"]["macroscopic_lid_slip"]["ratio"] <= 1.0,
    }
    summary = {
        "stage": STAGE, "status": "complete",
        "scientific_scope": "single-case denoising; no cross-condition generalization claim",
        "heat_flux_policy": "excluded_from_inputs_targets_loss_metrics_and_decision",
        "split": {"train": list(TRAIN_SEEDS), "validation": list(VALIDATION_SEEDS), "test": list(TEST_SEEDS)},
        "samples": {"train": len(train_x), "validation": len(val_x), "test": len(test_x)},
        "target_contract": "leave-one-seed-out mean of converged M3 fields",
        "training": training, "heldout_metrics": metrics, "checks": checks,
        "decision": "accept_MV1_single_case_denoiser" if all(checks.values()) else "hold_MV1_model",
    }
    _atomic_json(output_dir / "summary.json", summary)
    artifact_names = ("model.pt", "heldout_predictions.npz", "heldout_contours_and_profiles.png", "summary.json")
    manifest = {"stage": STAGE, "files": {name: {"sha256": _sha256(output_dir / name), "size_bytes": (output_dir / name).stat().st_size} for name in artifact_names}}
    _atomic_json(output_dir / "artifact_manifest.json", manifest)
    bundle = output_dir / "MOHAMMADZADEH_MV1_RETURN_BUNDLE.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        for name in artifact_names + ("artifact_manifest.json",):
            archive.add(output_dir / name, arcname=name)
    (output_dir / f"{bundle.name}.sha256").write_text(f"{_sha256(bundle)}  {bundle.name}\n", encoding="utf-8")
    return summary


def verify(output_dir: Path) -> dict[str, Any]:
    manifest = json.loads((output_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    for name, record in manifest["files"].items():
        path = output_dir / name
        if not path.is_file() or _sha256(path) != record["sha256"] or path.stat().st_size != record["size_bytes"]:
            raise ValueError(f"artifact verification failed for {name}")
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    if summary["heat_flux_policy"] != "excluded_from_inputs_targets_loss_metrics_and_decision":
        raise ValueError("heat-flux exclusion contract was violated")
    return {"status": "complete_MV1_artifacts_verified", "decision": summary["decision"], "summary_sha256": _sha256(output_dir / "summary.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m3-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    result = verify(args.output_dir) if args.verify_only else run(
        args.m3_root if args.m3_root is not None else Path("results/mohammadzadeh_2012/m3_qy_precision"),
        args.output_dir, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
