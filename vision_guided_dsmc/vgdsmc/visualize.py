from __future__ import annotations

from pathlib import Path
import numpy as np


def plot_case(case_path: str | Path, output_path: str | Path) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Install plotting dependencies: pip install -e '.[plot]'") from exc

    with np.load(case_path) as data:
        panels = [
            ("Coarse temperature", data["coarse_T"]),
            ("Reference temperature", data["reference_T"]),
            ("Combined error score", data["score"]),
            ("Particle-allocation class", data["label"]),
        ]

    figure, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
    for axis, (title, field) in zip(axes.ravel(), panels):
        image = axis.imshow(field, origin="lower")
        axis.set_title(title)
        axis.set_xlabel("cell i")
        axis.set_ylabel("cell j")
        figure.colorbar(image, ax=axis, shrink=0.8)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output
