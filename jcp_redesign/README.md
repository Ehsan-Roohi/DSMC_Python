# JCP redesign — Phase 0

The existing-data pilot promotes **P-NET + adaptive EB fusion** and rejects
**P-NN + EB** as the central estimator.  The pilot is retrospective and is used
only to decide the next development step.

## Run first on Unity

1. Upload `JCP0_collect.sh` to Unity.
2. Run:

```bash
bash JCP0_collect.sh
```

3. Upload the two paths printed under `UPLOAD THESE TWO FILES`:
   `JCP0_src.tar.gz` and `JCP0_src.tar.gz.sha256`.

This command does not submit a DSMC job.  It collects the exact existing code,
Slurm scripts, protocols, and verified Mach-10 DS2V source needed to construct
the source-specific Phase-0 implementation.  Heavy cavity and Mach-shift runs
start only after the Phase-0 dry-run gate passes.

## Contents

- `JCP_run_plan.md`: corrected gated computation plan.
- `pilot_eb_existing.py`: reproducible archived-array pilot.
- `pilot_findings.md`: decision table and interpretation.
- `pilot_out/`: detailed metrics and gain arrays for the 10x10 DCT partition.
- `JCP0_collect.sh`: source snapshot command for Unity.
