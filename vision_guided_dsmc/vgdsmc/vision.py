from __future__ import annotations

import numpy as np


def robust_unit_scale(field: np.ndarray, lower: float = 0.10, upper: float = 0.90) -> np.ndarray:
    """Robustly map an image-like field to [0, 1] using two quantiles."""
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError("Require 0 <= lower < upper <= 1")
    low, high = np.quantile(field, [lower, upper])
    return np.clip((field - low) / max(float(high - low), 1.0e-12), 0.0, 1.0)


def gradient_magnitude(field: np.ndarray) -> np.ndarray:
    grad_y, grad_x = np.gradient(field)
    return np.hypot(grad_x, grad_y)


def physics_vision_score(
    fields: dict[str, np.ndarray],
    mode: str = "temperature_gradient",
) -> np.ndarray:
    """Create a reference-free image score from coarse DSMC fields.

    The temperature-gradient mode is the current validated pilot baseline. Other
    modes are retained as ablations for future comparison against learned models.
    """
    temperature_gradient = robust_unit_scale(gradient_magnitude(fields["T"]))
    temperature_noise = robust_unit_scale(
        fields["sigma_T"] / (np.abs(fields["T"]) + 0.05)
    )
    speed = np.hypot(fields["u"], fields["v"])
    speed_gradient = robust_unit_scale(gradient_magnitude(speed))
    density_gradient = robust_unit_scale(gradient_magnitude(fields["rho"]))

    if mode == "temperature_gradient":
        return temperature_gradient
    if mode == "temperature_noise":
        return temperature_noise
    if mode == "noise_gradient":
        return 0.60 * temperature_noise + 0.40 * temperature_gradient
    if mode == "multi_field":
        return (
            0.35 * temperature_noise
            + 0.35 * temperature_gradient
            + 0.15 * speed_gradient
            + 0.15 * density_gradient
        )
    raise ValueError(
        "Unknown mode. Choose temperature_gradient, temperature_noise, "
        "noise_gradient, or multi_field"
    )
