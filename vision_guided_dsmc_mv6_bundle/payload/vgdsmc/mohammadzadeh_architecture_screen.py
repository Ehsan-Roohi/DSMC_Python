"""Locked four-architecture screen for low-budget Mohammadzadeh recovery.

The screen reuses the MV5 confirmatory trajectories and evaluates exactly one
DSMC sampling block.  Four approximately parameter-matched residual models are
trained with the same data split, targets, optimizer, loss, physical scaling,
bounded output head, and three initialization seeds:

* corrected conditioned residual U-Net;
* NAFNet-Small;
* a dependency-free MambaIRv2-Tiny adaptation with semantic prompts and
  four-direction prefix-state mixing;
* a compact residual Fourier neural operator.

The confirmatory targets are never used for training, tuning, early stopping,
or baseline selection.  This module does not launch or modify DSMC references.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import tarfile
import time
from typing import Any, Mapping

import numpy as np

from . import mohammadzadeh_vision_mv3 as mv3
from . import mohammadzadeh_vision_mv4 as mv4
from . import mohammadzadeh_vision_mv5 as mv5
from . import mohammadzadeh_mv5_reference as mv5ref
from .mohammadzadeh_vision import OUTPUT_FIELDS, _loss, build_model, fit_scaling
from .mohammadzadeh_vision_mv2 import (
    GAUSSIAN_PASSES,
    TSVD_RANKS,
    _atomic_json,
    _portable_tarinfo,
    _sha256,
    gaussian_like,
    select_baseline,
    tsvd,
)


STAGE = "MV6_Mohammadzadeh_four_architecture_budget_one_screen"
PROTOCOL_FILE = "mv6_four_architecture_screen_protocol.json"
BUDGET = 1
ARCHITECTURES = (
    "corrected_unet",
    "nafnet_small",
    "mambairv2_tiny_adapted",
    "fno_residual_small",
)
DISPLAY_NAMES = {
    "corrected_unet": "Corrected U-Net",
    "nafnet_small": "NAFNet-Small",
    "mambairv2_tiny_adapted": "MambaIRv2-Tiny adapted",
    "fno_residual_small": "FNO-residual small",
}
TRAINING_SEEDS = (2608091, 2608092, 2608093)
RESIDUAL_CAP_SIGMA = 4.0
PARAMETER_RATIO_LIMIT = 1.10
CONDITION_CHANNELS = ("log10_Kn", "U_lid_over_100")
CONDITION_CENTERS = (-1.1505149978319906, 2.5)
CONDITION_SCALES = (0.3010299956639812, 3.0)
METHODS = (
    "raw",
    "gaussian_like",
    "tsvd_pod_type",
    *ARCHITECTURES,
)


def protocol_path() -> Path:
    return mv5ref.protocol_path().parent / PROTOCOL_FILE


def locked_protocol() -> dict[str, Any]:
    value = json.loads(protocol_path().read_text(encoding="utf-8"))
    if value.get("stage") != STAGE or value.get("status") != (
        "locked_before_MV6_model_outcomes_and_before_MV5_reference_completion"
    ):
        raise ValueError("MV6 architecture-screen protocol is not locked")
    contract = value["comparison_contract"]
    if tuple(contract["architectures"]) != ARCHITECTURES:
        raise ValueError("MV6 architecture list differs from the locked protocol")
    if tuple(contract["training_initialization_seeds"]) != TRAINING_SEEDS:
        raise ValueError("MV6 training seeds differ from the locked protocol")
    if int(contract["budget_blocks"]) != BUDGET:
        raise ValueError("MV6 initial screen must use budget=1 only")
    if float(contract["residual_cap_sigma"]) != RESIDUAL_CAP_SIGMA:
        raise ValueError("MV6 residual cap differs from the locked protocol")
    scaling = contract["condition_scaling"]
    if (
        tuple(scaling["channels"]) != CONDITION_CHANNELS
        or not np.allclose(scaling["centers"], CONDITION_CENTERS)
        or not np.allclose(scaling["scales"], CONDITION_SCALES)
    ):
        raise ValueError("MV6 physical condition scaling differs from code")
    return value


def task_from_index(index: int) -> tuple[str, int]:
    total = len(ARCHITECTURES) * len(TRAINING_SEEDS)
    if not 0 <= index < total:
        raise ValueError(f"task index must be in [0,{total - 1}]")
    architecture = ARCHITECTURES[index // len(TRAINING_SEEDS)]
    seed = TRAINING_SEEDS[index % len(TRAINING_SEEDS)]
    return architecture, seed


def _task_directory(root: Path, architecture: str, seed: int) -> Path:
    return root / "tasks" / architecture / f"training_seed_{seed}"


def fixed_physical_scaling(
    train_x: np.ndarray, train_y: np.ndarray
) -> dict[str, np.ndarray]:
    """Use data scaling for fields and fixed nondimensional condition scaling."""
    scaling = {key: np.asarray(value).copy() for key, value in fit_scaling(train_x, train_y).items()}
    if train_x.shape[1] != len(mv3.MODEL_INPUT_FIELDS):
        raise ValueError("unexpected MV6 input channel count")
    indices = tuple(mv3.MODEL_INPUT_FIELDS.index(name) for name in CONDITION_CHANNELS)
    for index, center, scale in zip(indices, CONDITION_CENTERS, CONDITION_SCALES):
        scaling["input_mean"][0, index, 0, 0] = float(center)
        scaling["input_std"][0, index, 0, 0] = float(scale)
    return {key: np.asarray(value, dtype=np.float32) for key, value in scaling.items()}


def _torch_components():
    try:
        import torch
        from torch import nn
        import torch.nn.functional as functional
    except ImportError as exc:
        raise ImportError("MV6 architecture screen requires PyTorch") from exc
    return torch, nn, functional


def build_architecture(name: str, in_channels: int, out_channels: int = 2):
    """Build one of four locked approximately parameter-matched models."""
    if name not in ARCHITECTURES:
        raise ValueError(f"unknown architecture: {name}")
    torch, nn, functional = _torch_components()

    if name == "corrected_unet":
        return build_model(in_channels=in_channels, out_channels=out_channels, base=12)

    class LayerNorm2d(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(channels))
            self.bias = nn.Parameter(torch.zeros(channels))

        def forward(self, value):
            value = value.permute(0, 2, 3, 1)
            value = functional.layer_norm(
                value, (value.shape[-1],), self.weight, self.bias, 1.0e-6
            )
            return value.permute(0, 3, 1, 2)

    class SimpleGate(nn.Module):
        def forward(self, value):
            first, second = value.chunk(2, dim=1)
            return first * second

    class NAFBlock(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.norm1 = LayerNorm2d(channels)
            self.conv1 = nn.Conv2d(channels, 2 * channels, 1)
            self.depthwise = nn.Conv2d(
                2 * channels, 2 * channels, 3, padding=1, groups=2 * channels
            )
            self.gate = SimpleGate()
            self.sca = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1)
            )
            self.conv3 = nn.Conv2d(channels, channels, 1)
            self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
            self.norm2 = LayerNorm2d(channels)
            self.conv4 = nn.Conv2d(channels, 2 * channels, 1)
            self.conv5 = nn.Conv2d(channels, channels, 1)
            self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

        def forward(self, value):
            residual = self.conv1(self.norm1(value))
            residual = self.gate(self.depthwise(residual))
            residual = self.conv3(residual * self.sca(residual))
            value = value + self.beta * residual
            residual = self.conv5(self.gate(self.conv4(self.norm2(value))))
            return value + self.gamma * residual

    class NAFNetSmall(nn.Module):
        def __init__(self, width: int = 32, depth: int = 8):
            super().__init__()
            self.intro = nn.Conv2d(in_channels, width, 3, padding=1)
            self.body = nn.Sequential(*(NAFBlock(width) for _ in range(depth)))
            self.ending = nn.Conv2d(width, out_channels, 3, padding=1)

        def forward(self, value):
            return self.ending(self.body(self.intro(value)))

    def directional_prefix_mean(value, dimension: int, reverse: bool):
        if reverse:
            value = torch.flip(value, dims=(dimension,))
        length = value.shape[dimension]
        denominator_shape = [1] * value.ndim
        denominator_shape[dimension] = length
        denominator = torch.arange(
            1, length + 1, device=value.device, dtype=value.dtype
        ).reshape(denominator_shape)
        result = torch.cumsum(value, dim=dimension) / denominator
        return torch.flip(result, dims=(dimension,)) if reverse else result

    class AttentiveStateBlock(nn.Module):
        """Tiny ASE-inspired block using prompts and vectorized directional states."""

        def __init__(self, channels: int, prompts: int = 4):
            super().__init__()
            self.norm1 = LayerNorm2d(channels)
            self.in_projection = nn.Conv2d(channels, 2 * channels, 1)
            self.local = nn.Conv2d(
                channels, channels, 3, padding=1, groups=channels
            )
            self.prompt_score = nn.Conv2d(channels, prompts, 1)
            self.prompt_bank = nn.Parameter(torch.randn(prompts, channels) * 0.02)
            self.direction_logits = nn.Parameter(torch.zeros(4, channels, 1, 1))
            self.state_decay = nn.Parameter(torch.zeros(1, channels, 1, 1))
            self.out_projection = nn.Conv2d(channels, channels, 1)
            self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
            self.norm2 = LayerNorm2d(channels)
            self.ffn_in = nn.Conv2d(channels, 2 * channels, 1)
            self.ffn_out = nn.Conv2d(channels, channels, 1)
            self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

        def forward(self, value):
            normalized = self.norm1(value)
            local, gate = self.in_projection(normalized).chunk(2, dim=1)
            local = self.local(local)
            policy = torch.softmax(self.prompt_score(normalized), dim=1)
            prompt = torch.einsum("bphw,pc->bchw", policy, self.prompt_bank)
            states = torch.stack(
                (
                    directional_prefix_mean(local, -1, False),
                    directional_prefix_mean(local, -1, True),
                    directional_prefix_mean(local, -2, False),
                    directional_prefix_mean(local, -2, True),
                ),
                dim=0,
            )
            weights = torch.softmax(self.direction_logits, dim=0).unsqueeze(1)
            directional = torch.sum(weights * states, dim=0)
            decay = torch.sigmoid(self.state_decay)
            mixed = (1.0 - decay) * local + decay * directional + prompt
            value = value + self.beta * self.out_projection(
                mixed * torch.sigmoid(gate)
            )
            hidden = self.ffn_in(self.norm2(value))
            hidden = self.ffn_out(SimpleGate()(hidden))
            return value + self.gamma * hidden

    class MambaIRv2TinyAdapted(nn.Module):
        def __init__(self, width: int = 32, depth: int = 9):
            super().__init__()
            self.intro = nn.Conv2d(in_channels, width, 3, padding=1)
            self.body = nn.Sequential(
                *(AttentiveStateBlock(width) for _ in range(depth))
            )
            self.ending = nn.Conv2d(width, out_channels, 3, padding=1)

        def forward(self, value):
            return self.ending(self.body(self.intro(value)))

    class SpectralConv2d(nn.Module):
        def __init__(self, channels: int, modes: int):
            super().__init__()
            scale = 1.0 / max(1, channels)
            shape = (channels, channels, modes, modes, 2)
            self.top = nn.Parameter(scale * torch.randn(*shape))
            self.bottom = nn.Parameter(scale * torch.randn(*shape))
            self.modes = modes

        @staticmethod
        def multiply(value, weight):
            return torch.einsum("bixy,ioxy->boxy", value, weight)

        def forward(self, value):
            batch, _, height, width = value.shape
            transformed = torch.fft.rfft2(value, norm="ortho")
            output = torch.zeros(
                batch,
                transformed.shape[1],
                height,
                width // 2 + 1,
                device=value.device,
                dtype=transformed.dtype,
            )
            modes_y = min(self.modes, height // 2)
            modes_x = min(self.modes, width // 2 + 1)
            top = torch.view_as_complex(self.top.contiguous())
            bottom = torch.view_as_complex(self.bottom.contiguous())
            output[:, :, :modes_y, :modes_x] = self.multiply(
                transformed[:, :, :modes_y, :modes_x],
                top[:, :, :modes_y, :modes_x],
            )
            output[:, :, -modes_y:, :modes_x] = self.multiply(
                transformed[:, :, -modes_y:, :modes_x],
                bottom[:, :, :modes_y, :modes_x],
            )
            return torch.fft.irfft2(
                output, s=(height, width), norm="ortho"
            )

    class FNOResidualSmall(nn.Module):
        def __init__(self, width: int = 9, modes: int = 8, depth: int = 3):
            super().__init__()
            self.lift = nn.Conv2d(in_channels, width, 1)
            self.spectral = nn.ModuleList(
                SpectralConv2d(width, modes) for _ in range(depth)
            )
            self.local = nn.ModuleList(
                nn.Conv2d(width, width, 1) for _ in range(depth)
            )
            self.norm = nn.ModuleList(
                nn.GroupNorm(1, width) for _ in range(depth)
            )
            self.project = nn.Conv2d(width, out_channels, 1)

        def forward(self, value):
            value = self.lift(value)
            for spectral, local, norm in zip(self.spectral, self.local, self.norm):
                value = functional.gelu(norm(spectral(value) + local(value)))
            return self.project(value)

    if name == "nafnet_small":
        return NAFNetSmall()
    if name == "mambairv2_tiny_adapted":
        return MambaIRv2TinyAdapted()
    return FNOResidualSmall()


def parameter_report(in_channels: int) -> dict[str, Any]:
    counts = {
        name: int(
            sum(
                parameter.numel()
                for parameter in build_architecture(name, in_channels).parameters()
                if parameter.requires_grad
            )
        )
        for name in ARCHITECTURES
    }
    minimum, maximum = min(counts.values()), max(counts.values())
    ratio = float(maximum / minimum)
    return {
        "trainable_parameters": counts,
        "maximum_to_minimum_ratio": ratio,
        "limit": PARAMETER_RATIO_LIMIT,
        "pass": bool(ratio <= PARAMETER_RATIO_LIMIT + 1.0e-12),
    }


def _training_tensors(
    x: np.ndarray, y: np.ndarray, scaling: Mapping[str, np.ndarray]
):
    torch, _, _ = _torch_components()
    normalized = ((x - scaling["input_mean"]) / scaling["input_std"]).astype(
        np.float32
    )
    residual = (
        (y - x[:, : len(OUTPUT_FIELDS)]) / scaling["residual_std"]
    ).astype(np.float32)
    return torch.from_numpy(normalized), torch.from_numpy(residual)


def train_architecture(
    architecture: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    scaling: Mapping[str, np.ndarray],
    *,
    seed: int,
    epochs: int,
    batch_size: int,
) -> tuple[Any, dict[str, Any]]:
    torch, _, _ = _torch_components()
    from torch.utils.data import DataLoader, TensorDataset

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tx, ty = _training_tensors(train_x, train_y, scaling)
    vx, vy = _training_tensors(validation_x, validation_y, scaling)
    loader = DataLoader(
        TensorDataset(tx, ty),
        batch_size=min(batch_size, len(tx)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    model = build_architecture(architecture, int(train_x.shape[1])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-5)
    best_state: dict[str, Any] | None = None
    best_value, best_epoch, stale = float("inf"), 0, 0
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            latent = model(xb)
            bounded = RESIDUAL_CAP_SIGMA * torch.tanh(
                latent / RESIDUAL_CAP_SIGMA
            )
            loss = _loss(bounded, yb)
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu()) * len(xb)
        model.eval()
        with torch.no_grad():
            latent = model(vx.to(device))
            bounded = RESIDUAL_CAP_SIGMA * torch.tanh(
                latent / RESIDUAL_CAP_SIGMA
            )
            validation = float(_loss(bounded, vy.to(device)).cpu())
        history.append(
            {
                "epoch": epoch,
                "train_loss": running / len(tx),
                "validation_loss": validation,
            }
        )
        if validation < best_value - 1.0e-7:
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_value, best_epoch, stale = validation, epoch, 0
        else:
            stale += 1
        if stale >= 25:
            break
    if best_state is None:
        raise RuntimeError("MV6 training produced no checkpoint")
    model.load_state_dict(best_state)
    model.to("cpu").eval()
    return model, {
        "architecture": architecture,
        "training_seed": seed,
        "device": str(device),
        "seconds": time.perf_counter() - started,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_loss": best_value,
        "optimizer": "AdamW(lr=1e-3,weight_decay=1e-5)",
        "early_stopping_patience": 25,
        "history": history,
    }


def predict_bounded(
    model: Any,
    x: np.ndarray,
    scaling: Mapping[str, np.ndarray],
    batch_size: int,
) -> tuple[np.ndarray, dict[str, float]]:
    torch, _, _ = _torch_components()
    normalized = ((x - scaling["input_mean"]) / scaling["input_std"]).astype(
        np.float32
    )
    outputs, latent_max, bounded_max = [], 0.0, 0.0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            latent = model(torch.from_numpy(normalized[start : start + batch_size]))
            bounded = RESIDUAL_CAP_SIGMA * torch.tanh(
                latent / RESIDUAL_CAP_SIGMA
            )
            latent_max = max(latent_max, float(torch.max(torch.abs(latent))))
            bounded_max = max(bounded_max, float(torch.max(torch.abs(bounded))))
            outputs.append(
                x[start : start + batch_size, : len(OUTPUT_FIELDS)]
                + bounded.numpy() * scaling["residual_std"]
            )
    return np.concatenate(outputs).astype(np.float32), {
        "latent_normalized_residual_abs_max": latent_max,
        "bounded_normalized_residual_abs_max": bounded_max,
    }


def _data_arrays(existing_m3_root: Path, mv3_root: Path, mv5_reference_root: Path):
    protocol = locked_protocol()
    mv3_protocol = mv3.locked_protocol()
    mv5_protocol = mv5.locked_protocol()
    development_specs = mv3._condition_map(mv3_protocol)
    confirmatory_specs = mv5ref.condition_map(mv5_protocol)
    development_blocks, development_full = mv3.load_condition_data(
        existing_m3_root, mv3_root, mv3_protocol
    )
    confirmatory_blocks, confirmatory_full = mv5.load_confirmatory_data(
        mv5_reference_root
    )
    train_split = {
        key: tuple(int(seed) for seed in value)
        for key, value in mv5_protocol["development_seed_split"]["train"].items()
    }
    validation_split = {
        key: tuple(int(seed) for seed in value)
        for key, value in mv5_protocol["development_seed_split"]["validation"].items()
    }
    development_targets = mv5._development_targets(
        development_full, train_split, validation_split
    )
    confirmatory_targets = mv5._confirmatory_targets(confirmatory_full)
    test_split = {
        key: tuple(int(seed) for seed in value["evaluation_seeds"])
        for key, value in confirmatory_specs.items()
    }
    train = mv3.build_budget_arrays(
        development_blocks,
        development_targets,
        train_split,
        development_specs,
        BUDGET,
    )
    validation = mv3.build_budget_arrays(
        development_blocks,
        development_targets,
        validation_split,
        development_specs,
        BUDGET,
    )
    test = mv3.build_budget_arrays(
        confirmatory_blocks,
        confirmatory_targets,
        test_split,
        confirmatory_specs,
        BUDGET,
    )
    if protocol["source_contract"]["confirmatory_conditions"] != list(
        confirmatory_specs
    ):
        raise ValueError("MV6 confirmatory conditions differ from MV5")
    return train, validation, test, development_specs, confirmatory_specs


def run_task(
    existing_m3_root: Path,
    mv3_root: Path,
    mv5_reference_root: Path,
    output_dir: Path,
    *,
    architecture: str,
    training_seed: int,
    epochs: int,
    batch_size: int,
) -> dict[str, Any]:
    protocol = locked_protocol()
    if architecture not in ARCHITECTURES or training_seed not in TRAINING_SEEDS:
        raise ValueError("architecture/training seed is outside the locked MV6 matrix")
    (
        (train_x, train_y, train_conditions, train_identity),
        (validation_x, validation_y, validation_conditions, validation_identity),
        (test_x, test_y, test_conditions, test_identity),
        development_specs,
        confirmatory_specs,
    ) = _data_arrays(existing_m3_root, mv3_root, mv5_reference_root)
    del train_conditions, train_identity, validation_identity
    scaling = fixed_physical_scaling(train_x, train_y)
    parameters = parameter_report(int(train_x.shape[1]))
    if not parameters["pass"]:
        raise ValueError("MV6 architecture parameter parity gate failed")
    model, training = train_architecture(
        architecture,
        train_x,
        train_y,
        validation_x,
        validation_y,
        scaling,
        seed=training_seed,
        epochs=epochs,
        batch_size=batch_size,
    )
    validation_candidate, validation_diagnostics = predict_bounded(
        model, validation_x, scaling, batch_size
    )
    test_candidate, test_diagnostics = predict_bounded(
        model, test_x, scaling, batch_size
    )
    validation_raw = validation_x[:, : len(OUTPUT_FIELDS)]
    raw = test_x[:, : len(OUTPUT_FIELDS)]
    validation_speeds = np.asarray(
        [
            float(development_specs[str(item)]["lid_speed_m_per_s"])
            for item in validation_conditions
        ]
    )
    alpha, alpha_records = mv3.select_residual_gate(
        validation_raw,
        validation_candidate,
        validation_y,
        validation_speeds,
        tuple(
            float(value)
            for value in protocol["comparison_contract"][
                "residual_alpha_candidates"
            ]
        ),
    )
    candidate = raw + float(alpha) * (test_candidate - raw)
    gaussian_passes, gaussian_records = select_baseline(
        validation_raw, validation_y, GAUSSIAN_PASSES, gaussian_like
    )
    tsvd_rank, tsvd_records = select_baseline(
        validation_raw, validation_y, TSVD_RANKS, tsvd
    )
    gaussian = gaussian_like(raw, gaussian_passes)
    pod_type = tsvd(raw, tsvd_rank)
    gaussian, gaussian_projection = mv5._project_by_condition(
        gaussian, raw, test_conditions, confirmatory_specs
    )
    pod_type, tsvd_projection = mv5._project_by_condition(
        pod_type, raw, test_conditions, confirmatory_specs
    )
    candidate, candidate_projection = mv5._project_by_condition(
        candidate, raw, test_conditions, confirmatory_specs
    )
    candidates = {
        "raw": raw,
        "gaussian_like": gaussian,
        "tsvd_pod_type": pod_type,
        architecture: candidate,
    }
    methods_by_condition = {
        method: mv5._metric_by_condition(
            raw, value, test_y, test_conditions, confirmatory_specs
        )
        for method, value in candidates.items()
    }
    per_seed: dict[str, Any] = {}
    for condition_id, condition in confirmatory_specs.items():
        per_seed[condition_id] = {}
        speed = float(condition["lid_speed_m_per_s"])
        for evaluation_seed in condition["evaluation_seeds"]:
            mask = (test_conditions == condition_id) & (
                test_identity[:, 0] == int(evaluation_seed)
            )
            per_seed[condition_id][str(evaluation_seed)] = {
                method: mv3.evaluate_fields(
                    raw[mask], value[mask], test_y[mask], speed
                )
                for method, value in candidates.items()
            }
    output_dir.mkdir(parents=True, exist_ok=True)
    torch, _, _ = _torch_components()
    torch.save(
        {
            "stage": STAGE,
            "architecture": architecture,
            "training_seed": training_seed,
            "state_dict": model.state_dict(),
            "scaling": {key: np.asarray(value) for key, value in scaling.items()},
            "residual_gate_alpha": alpha,
            "input_fields": mv3.MODEL_INPUT_FIELDS,
            "output_fields": OUTPUT_FIELDS,
        },
        output_dir / "model.pt",
    )
    np.savez_compressed(
        output_dir / "predictions.npz",
        identity_condition=test_conditions,
        identity_numeric=test_identity,
        raw=raw,
        gaussian_like=gaussian,
        tsvd_pod_type=pod_type,
        architecture_prediction=candidate,
        target=test_y,
    )
    checks = {
        "budget_is_one": BUDGET == 1,
        "same_locked_split_and_target_for_all_architectures": True,
        "three_locked_training_initializations": len(TRAINING_SEEDS) == 3,
        "parameter_parity": bool(parameters["pass"]),
        "bounded_residual": bool(
            test_diagnostics["bounded_normalized_residual_abs_max"]
            <= RESIDUAL_CAP_SIGMA + 1.0e-6
        ),
        "confirmatory_target_not_used_for_training_or_selection": True,
        "finite_candidate": bool(np.all(np.isfinite(candidate))),
        "positive_temperature": bool(np.min(candidate[:, 0]) >= 1.0),
    }
    summary = {
        "stage": STAGE,
        "status": "complete_MV6_architecture_seed_task",
        "architecture": architecture,
        "architecture_display_name": DISPLAY_NAMES[architecture],
        "training_seed": training_seed,
        "budget_blocks": BUDGET,
        "protocol_sha256": _sha256(protocol_path()),
        "sample_counts": {
            "train": len(train_x),
            "validation": len(validation_x),
            "confirmatory": len(test_x),
        },
        "parameter_parity": parameters,
        "scaling_contract": protocol["comparison_contract"]["condition_scaling"],
        "selection": {
            "residual_alpha": {"selected": alpha, "candidates": alpha_records},
            "gaussian_like": {
                "selected_passes": gaussian_passes,
                "candidates": gaussian_records,
            },
            "tsvd_pod_type": {
                "selected_rank": tsvd_rank,
                "candidates": tsvd_records,
            },
        },
        "training": training,
        "prediction_diagnostics": {
            "validation": validation_diagnostics,
            "confirmatory": test_diagnostics,
            "projection": {
                "gaussian_like": gaussian_projection,
                "tsvd_pod_type": tsvd_projection,
                architecture: candidate_projection,
            },
        },
        "methods_by_condition": methods_by_condition,
        "per_evaluation_seed_metrics": per_seed,
        "checks": checks,
        "decision": "accept_MV6_task" if all(checks.values()) else "hold_MV6_task",
    }
    _atomic_json(output_dir / "summary.json", summary)
    _atomic_json(
        output_dir / "artifact_manifest.json",
        {
            "stage": STAGE,
            "files": {
                name: {
                    "sha256": _sha256(output_dir / name),
                    "size_bytes": (output_dir / name).stat().st_size,
                }
                for name in ("model.pt", "predictions.npz", "summary.json")
            },
        },
    )
    return summary


def _records(root: Path) -> list[dict[str, Any]]:
    values = []
    for architecture in ARCHITECTURES:
        for seed in TRAINING_SEEDS:
            directory = _task_directory(root, architecture, seed)
            summary_path = directory / "summary.json"
            manifest_path = directory / "artifact_manifest.json"
            if not summary_path.is_file() or not manifest_path.is_file():
                raise FileNotFoundError(f"incomplete MV6 task: {directory}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for name, record in manifest["files"].items():
                path = directory / name
                if (
                    not path.is_file()
                    or path.stat().st_size != int(record["size_bytes"])
                    or _sha256(path) != record["sha256"]
                ):
                    raise ValueError(f"MV6 task artifact failed verification: {path}")
            value = json.loads(summary_path.read_text(encoding="utf-8"))
            if (
                value.get("status") != "complete_MV6_architecture_seed_task"
                or value.get("architecture") != architecture
                or int(value.get("training_seed", -1)) != seed
                or value.get("decision") != "accept_MV6_task"
            ):
                raise ValueError(f"MV6 task summary contract failed: {summary_path}")
            values.append(value)
    return values


def _mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=1))}


def _screen_statistics(records: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    condition_ids = list(
        records[0]["methods_by_condition"]["raw"].keys()
    )
    baseline = {}
    for condition in condition_ids:
        raw_metric = records[0]["methods_by_condition"]["raw"][condition]
        gaussian_metric = records[0]["methods_by_condition"]["gaussian_like"][condition]
        tsvd_metric = records[0]["methods_by_condition"]["tsvd_pod_type"][condition]
        for record in records[1:]:
            if record["methods_by_condition"]["raw"][condition] != raw_metric:
                raise ValueError("Raw metrics differ between architecture tasks")
            if record["methods_by_condition"]["gaussian_like"][condition] != gaussian_metric:
                raise ValueError("Gaussian metrics differ between architecture tasks")
            if record["methods_by_condition"]["tsvd_pod_type"][condition] != tsvd_metric:
                raise ValueError("TSVD metrics differ between architecture tasks")
        classical = {
            "gaussian_like": gaussian_metric,
            "tsvd_pod_type": tsvd_metric,
        }
        best_name = min(
            classical,
            key=lambda name: classical[name]["vision_composite_nrmse"],
        )
        baseline[condition] = {
            "raw": raw_metric,
            "gaussian_like": gaussian_metric,
            "tsvd_pod_type": tsvd_metric,
            "best_classical_method": best_name,
            "best_classical_composite_nrmse": classical[best_name][
                "vision_composite_nrmse"
            ],
        }
    architectures: dict[str, Any] = {}
    promoted = []
    for architecture in ARCHITECTURES:
        selected = [item for item in records if item["architecture"] == architecture]
        by_condition = {}
        beats_raw_count = 0
        beats_classical_count = 0
        for condition in condition_ids:
            ratios = [
                item["methods_by_condition"][architecture][condition][
                    "vision_over_raw_composite"
                ]
                for item in selected
            ]
            errors = [
                item["methods_by_condition"][architecture][condition][
                    "vision_composite_nrmse"
                ]
                for item in selected
            ]
            ratio_stats = _mean_std(ratios)
            error_stats = _mean_std(errors)
            beats_raw = ratio_stats["mean"] < 1.0
            beats_classical = error_stats["mean"] < baseline[condition][
                "best_classical_composite_nrmse"
            ]
            beats_raw_count += int(beats_raw)
            beats_classical_count += int(beats_classical)
            by_condition[condition] = {
                "model_over_raw": ratio_stats,
                "model_composite_nrmse": error_stats,
                "beats_raw_on_three_seed_mean": beats_raw,
                "beats_best_classical_on_three_seed_mean": beats_classical,
                "best_classical_method": baseline[condition][
                    "best_classical_method"
                ],
            }
        pass_screen = (
            beats_raw_count == len(condition_ids) and beats_classical_count >= 3
        )
        architectures[architecture] = {
            "display_name": DISPLAY_NAMES[architecture],
            "by_condition": by_condition,
            "conditions_beating_raw": beats_raw_count,
            "conditions_beating_best_classical": beats_classical_count,
            "promotion_gate": {
                "requires_beating_raw_conditions": len(condition_ids),
                "requires_beating_best_classical_min_conditions": 3,
                "pass": pass_screen,
            },
        }
        if pass_screen:
            promoted.append(architecture)
    return {
        "conditions": condition_ids,
        "baselines": baseline,
        "architectures": architectures,
    }, promoted


def _write_csv(root: Path, statistics: Mapping[str, Any]) -> str:
    name = "mv6_four_architecture_budget1_comparison.csv"
    with (root / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "architecture",
                "condition",
                "model_over_raw_mean",
                "model_over_raw_std",
                "model_composite_nrmse_mean",
                "model_composite_nrmse_std",
                "best_classical_method",
                "beats_raw",
                "beats_best_classical",
            )
        )
        for architecture in ARCHITECTURES:
            for condition, value in statistics["architectures"][architecture][
                "by_condition"
            ].items():
                writer.writerow(
                    (
                        architecture,
                        condition,
                        value["model_over_raw"]["mean"],
                        value["model_over_raw"]["std"],
                        value["model_composite_nrmse"]["mean"],
                        value["model_composite_nrmse"]["std"],
                        value["best_classical_method"],
                        value["beats_raw_on_three_seed_mean"],
                        value["beats_best_classical_on_three_seed_mean"],
                    )
                )
    return name


def _comparison_figure(root: Path, statistics: Mapping[str, Any]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = statistics["conditions"]
    x = np.arange(len(conditions), dtype=float)
    offsets = np.linspace(-0.24, 0.24, len(ARCHITECTURES))
    colors = ("#0072B2", "#009E73", "#D55E00", "#CC79A7")
    figure, axis = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    gaussian = [
        statistics["baselines"][condition]["gaussian_like"][
            "vision_over_raw_composite"
        ]
        for condition in conditions
    ]
    pod = [
        statistics["baselines"][condition]["tsvd_pod_type"][
            "vision_over_raw_composite"
        ]
        for condition in conditions
    ]
    axis.plot(x, gaussian, "--", color="0.45", marker="s", label="Gaussian/Raw")
    axis.plot(x, pod, ":", color="0.10", marker="^", label="TSVD/Raw")
    for offset, color, architecture in zip(offsets, colors, ARCHITECTURES):
        values = [
            statistics["architectures"][architecture]["by_condition"][condition][
                "model_over_raw"
            ]
            for condition in conditions
        ]
        axis.errorbar(
            x + offset,
            [item["mean"] for item in values],
            yerr=[item["std"] for item in values],
            marker="o",
            capsize=3,
            linewidth=1.4,
            color=color,
            label=DISPLAY_NAMES[architecture],
        )
    axis.axhline(1.0, color="black", linewidth=1.0, label="Raw")
    axis.set_xticks(x, conditions, rotation=18, ha="right")
    axis.set_ylabel("Composite NRMSE / Raw")
    axis.set_title("Locked budget=1 architecture screen; mean ± SD over 3 training seeds")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    names = []
    for suffix in ("png", "pdf"):
        name = f"mv6_architecture_screen_ratios.{suffix}"
        figure.savefig(root / name, dpi=300)
        names.append(name)
    plt.close(figure)
    return names


def _physical_figures(root: Path, records: list[dict[str, Any]]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = list(records[0]["methods_by_condition"]["raw"])
    files = []
    columns = (
        "Raw",
        "Gaussian",
        "TSVD/POD",
        *(DISPLAY_NAMES[name] for name in ARCHITECTURES),
        "Reference",
    )
    for condition in conditions:
        architecture_values = {}
        first_payload = None
        identity_index = None
        for architecture in ARCHITECTURES:
            predictions = []
            for seed in TRAINING_SEEDS:
                directory = _task_directory(root, architecture, seed)
                with np.load(directory / "predictions.npz", allow_pickle=False) as data:
                    labels = np.asarray(data["identity_condition"]).astype(str)
                    indices = np.flatnonzero(labels == condition)
                    if len(indices) == 0:
                        raise ValueError(f"condition missing from predictions: {condition}")
                    index = int(indices[0])
                    predictions.append(np.asarray(data["architecture_prediction"][index]))
                    if first_payload is None:
                        first_payload = {
                            name: np.asarray(data[name][index])
                            for name in (
                                "raw",
                                "gaussian_like",
                                "tsvd_pod_type",
                                "target",
                            )
                        }
                        identity_index = index
            architecture_values[architecture] = np.mean(predictions, axis=0)
        if first_payload is None or identity_index is None:
            raise RuntimeError("MV6 physical figure has no payload")
        values = (
            first_payload["raw"],
            first_payload["gaussian_like"],
            first_payload["tsvd_pod_type"],
            *(architecture_values[name] for name in ARCHITECTURES),
            first_payload["target"],
        )
        figure, axes = plt.subplots(
            2, len(columns), figsize=(20.0, 5.8), constrained_layout=True
        )
        for field, label in enumerate(("T [K]", "u [m/s]")):
            target = first_payload["target"][field]
            if field == 0:
                vmin, vmax = float(target.min()), float(target.max())
                cmap = "inferno"
            else:
                bound = max(float(np.max(np.abs(target))), 1.0e-12)
                vmin, vmax, cmap = -bound, bound, "coolwarm"
            for column, (title, value) in enumerate(zip(columns, values)):
                image = axes[field, column].imshow(
                    value[field], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax
                )
                axes[field, column].set_xticks([])
                axes[field, column].set_yticks([])
                if field == 0:
                    axes[field, column].set_title(title, fontsize=8)
                if column == 0:
                    axes[field, column].set_ylabel(label)
            figure.colorbar(image, ax=axes[field, :], shrink=0.78, pad=0.01)
        figure.suptitle(
            f"{condition}, budget=1; neural fields are means over 3 training seeds",
            fontsize=11,
        )
        for suffix in ("png", "pdf"):
            name = f"mv6_physical_fields_{condition}.{suffix}"
            figure.savefig(root / name, dpi=300)
            files.append(name)
        plt.close(figure)
    return files


def aggregate(root: Path) -> dict[str, Any]:
    protocol = locked_protocol()
    records = _records(root)
    statistics, promoted = _screen_statistics(records)
    parameter_reports = [record["parameter_parity"] for record in records]
    if any(value != parameter_reports[0] for value in parameter_reports[1:]):
        raise ValueError("architecture parameter reports differ between tasks")
    csv_name = _write_csv(root, statistics)
    figure_names = _comparison_figure(root, statistics)
    physical_names = _physical_figures(root, records)
    checks = {
        "all_12_architecture_seed_tasks_complete": len(records) == 12,
        "four_architectures": len(ARCHITECTURES) == 4,
        "three_training_seeds_each": all(
            sum(item["architecture"] == name for item in records) == 3
            for name in ARCHITECTURES
        ),
        "budget_one_only": all(item["budget_blocks"] == 1 for item in records),
        "parameter_parity": bool(parameter_reports[0]["pass"]),
        "all_task_checks_pass": all(
            all(item["checks"].values()) for item in records
        ),
    }
    summary = {
        "stage": STAGE,
        "status": "complete_MV6_four_architecture_screen",
        "protocol_sha256": _sha256(protocol_path()),
        "scientific_question": protocol["scientific_question"],
        "comparison_contract": protocol["comparison_contract"],
        "parameter_parity": parameter_reports[0],
        "statistics": statistics,
        "promotion_rule": protocol["promotion_rule"],
        "architectures_eligible_for_full_budget_matrix": promoted,
        "automatic_full_matrix_submission": False,
        "artifacts": [csv_name, *figure_names, *physical_names],
        "checks": checks,
        "decision": (
            "architecture_candidates_ready_for_user_decision"
            if promoted
            else "no_architecture_passed_locked_budget_one_screen"
        ),
    }
    _atomic_json(root / "summary.json", summary)
    return summary


def verify(root: Path) -> dict[str, Any]:
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    records = _records(root)
    expected = [
        "mv6_four_architecture_budget1_comparison.csv",
        "mv6_architecture_screen_ratios.png",
        "mv6_architecture_screen_ratios.pdf",
        *(
            f"mv6_physical_fields_{condition}.{suffix}"
            for condition in summary["statistics"]["conditions"]
            for suffix in ("png", "pdf")
        ),
    ]
    checks = {
        "summary_complete": summary.get("status")
        == "complete_MV6_four_architecture_screen",
        "summary_checks_pass": all(summary.get("checks", {}).values()),
        "twelve_recursively_verified_tasks": len(records) == 12,
        "all_report_artifacts_exist": all((root / name).is_file() for name in expected),
        "protocol_hash_matches": summary.get("protocol_sha256")
        == _sha256(protocol_path()),
    }
    value = {
        "stage": STAGE,
        "status": "complete_MV6_architecture_screen_artifacts_and_metrics_verified"
        if all(checks.values())
        else "failed_MV6_architecture_screen_verification",
        "summary_sha256": _sha256(root / "summary.json"),
        "checks": checks,
        "decision": "verified" if all(checks.values()) else "hold",
    }
    _atomic_json(root / "verification.json", value)
    if not all(checks.values()):
        raise ValueError("MV6 architecture-screen verification failed")
    return value


def package(root: Path) -> dict[str, Any]:
    verify(root)
    bundle = root / "MOHAMMADZADEH_MV6_ARCHITECTURE_SCREEN_BUNDLE.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        for name in (
            "summary.json",
            "verification.json",
            "mv6_four_architecture_budget1_comparison.csv",
            "mv6_architecture_screen_ratios.png",
            "mv6_architecture_screen_ratios.pdf",
        ):
            archive.add(root / name, arcname=name, filter=_portable_tarinfo)
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        for condition in summary["statistics"]["conditions"]:
            for suffix in ("png", "pdf"):
                name = f"mv6_physical_fields_{condition}.{suffix}"
                archive.add(root / name, arcname=name, filter=_portable_tarinfo)
        archive.add(
            protocol_path(),
            arcname=f"provenance/{PROTOCOL_FILE}",
            filter=_portable_tarinfo,
        )
        for architecture in ARCHITECTURES:
            for seed in TRAINING_SEEDS:
                directory = _task_directory(root, architecture, seed)
                for name in ("summary.json", "artifact_manifest.json", "predictions.npz"):
                    archive.add(
                        directory / name,
                        arcname=f"tasks/{architecture}/training_seed_{seed}/{name}",
                        filter=_portable_tarinfo,
                    )
    checksum = _sha256(bundle)
    (root / f"{bundle.name}.sha256").write_text(
        f"{checksum}  {bundle.name}\n", encoding="utf-8"
    )
    return {"bundle": str(bundle), "sha256": checksum}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("task", "post"), required=True)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--existing-m3-root", type=Path)
    parser.add_argument("--mv3-root", type=Path)
    parser.add_argument("--mv5-reference-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=6)
    args = parser.parse_args()
    if args.mode == "task":
        if args.task_index is None:
            parser.error("task mode requires --task-index")
        for name in ("existing_m3_root", "mv3_root", "mv5_reference_root"):
            if getattr(args, name) is None:
                parser.error(f"task mode requires --{name.replace('_', '-')}")
        architecture, seed = task_from_index(args.task_index)
        directory = _task_directory(args.output_dir, architecture, seed)
        if (directory / "summary.json").exists():
            raise SystemExit(f"refusing to overwrite completed MV6 task: {directory}")
        result = run_task(
            args.existing_m3_root,
            args.mv3_root,
            args.mv5_reference_root,
            directory,
            architecture=architecture,
            training_seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        print(json.dumps({"decision": result["decision"], "task": args.task_index}))
    else:
        aggregate(args.output_dir)
        verification = verify(args.output_dir)
        packaged = package(args.output_dir)
        print(json.dumps({"verification": verification, "package": packaged}))


if __name__ == "__main__":
    main()
