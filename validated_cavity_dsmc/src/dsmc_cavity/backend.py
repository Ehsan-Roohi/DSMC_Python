"""Small NumPy/CuPy compatibility layer.

The collision trial list is generated on the CPU because it is discrete and
cell-local.  Relative velocities, VHS probabilities, collision scattering,
particle motion, and moment sampling operate on the selected array backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ArrayBackend:
    name: str
    xp: Any
    rng: Any

    @classmethod
    def create(cls, name: str, seed: int) -> "ArrayBackend":
        requested = name.lower()
        if requested == "auto":
            try:
                import cupy as cp

                if cp.cuda.runtime.getDeviceCount() > 0:
                    return cls("gpu", cp, cp.random.RandomState(seed))
            except Exception:
                pass
            requested = "cpu"
        if requested == "cpu":
            return cls("cpu", np, np.random.default_rng(seed))
        if requested == "gpu":
            try:
                import cupy as cp
            except ImportError as exc:
                raise RuntimeError(
                    "GPU backend requested but CuPy is not installed. Install the "
                    "matching package, for example `pip install cupy-cuda12x`."
                ) from exc
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("GPU backend requested but CUDA reports no device.")
            return cls("gpu", cp, cp.random.RandomState(seed))
        raise ValueError(f"Unknown backend {name!r}; choose cpu, gpu, or auto.")

    def asnumpy(self, value: Any) -> np.ndarray:
        if self.name == "gpu":
            return self.xp.asnumpy(value)
        return np.asarray(value)

    def uniform(self, size: int | tuple[int, ...]) -> Any:
        if self.name == "gpu":
            return self.rng.random_sample(size)
        return self.rng.random(size)

    def normal(self, size: int | tuple[int, ...]) -> Any:
        return self.rng.normal(0.0, 1.0, size=size)

    def synchronize(self) -> None:
        if self.name == "gpu":
            self.xp.cuda.Stream.null.synchronize()
