# Vision-Guided DSMC Pilot

A deliberately small, reproducible first case for **vision-guided particle allocation** in a 2-D rarefied thermal cavity.

## Physics pilot

- unit square cavity;
- diffuse thermal walls;
- left wall `T=1.10`, right wall `T=0.90`, top/bottom `T=1.00`;
- particle transport plus stochastic cell collisions;
- coarse and reference runs;
- four-channel vision input: `T, u, v, sigma_T`;
- three-class target: reduce / retain / increase particles.

This is an educational DSMC-like research prototype, not yet a production DSMC solver. Its purpose is to validate the complete data and ML pipeline before coupling the method to the high-fidelity DSMC code.

## Install and test

```bash
cd vision_guided_dsmc
python -m pip install -e '.[dev]'
pytest -q
vgdsmc-generate --output outputs/pilot --nx 24 --ny 24 --ppc 20 --reference-ppc 120
```

The generated `case.npz` contains the network input, label map, error score, and coarse/reference fields.

## Stage 2: training and adaptive allocation

Install the full optional dependencies:

```bash
python -m pip install -e '.[full]'
```

Train a compact U-Net from several generated cases:

```python
from pathlib import Path
from vgdsmc.training import TrainConfig, train_model

cases = sorted(Path("outputs/dataset").glob("*/case.npz"))
model_path = train_model(cases, "outputs/training", TrainConfig(epochs=20))
```

Convert predicted classes to a particles-per-cell map:

```python
import numpy as np
from vgdsmc.adaptive import allocation_summary, label_to_target_ppc

with np.load("outputs/pilot/case.npz") as data:
    label = data["label"]

target_ppc = label_to_target_ppc(label, base_ppc=20)
print(allocation_summary(label, base_ppc=20))
```

Create a diagnostic figure:

```python
from vgdsmc.visualize import plot_case
plot_case("outputs/pilot/case.npz", "outputs/pilot/diagnostics.png")
```

## Current limitations

- particle splitting/merging currently uses unweighted resampling;
- newly populated empty cells use a unit-temperature Maxwellian;
- the pilot collision kernel is not yet the validated VHS/SBT DSMC implementation;
- conservation corrections and particle weights are the next physics upgrade.

## Next milestones

1. generate an ensemble dataset over `Kn`, wall-temperature ratio, and random seed;
2. measure segmentation accuracy and class-wise recall;
3. run a closed-loop adaptive cavity simulation;
4. replace pilot resampling with conservative weighted particles;
5. compare error versus particle updates against uniform DSMC.
