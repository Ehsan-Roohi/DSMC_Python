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

## Run

```bash
cd vision_guided_dsmc
python -m pip install -e '.[dev]'
pytest -q
vgdsmc-generate --output outputs/pilot --nx 24 --ny 24 --ppc 20 --reference-ppc 120
```

The generated `case.npz` contains the network input, label map, error score, and coarse/reference fields.

## Next milestones

1. ensemble averaging over independent seeds;
2. local particle weights/splitting and merging;
3. training script and validation plots for the U-Net;
4. replacement of the pilot collision kernel by the validated DSMC implementation;
5. cost-versus-error comparison against uniform particle allocation.
