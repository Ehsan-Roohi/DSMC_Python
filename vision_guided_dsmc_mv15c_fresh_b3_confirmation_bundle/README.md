# MV15C — prospectively locked fresh-B3 confirmation

MV15C is the confirmatory successor to MV15B. MV15B found that the smallest
passing data-consistent budget was B3, but that result used legacy evaluation
conditions. MV15C therefore freezes the exact MV15B B3 rule before generating
any new trajectory or observing any new outcome.

## Confirmatory design

- Eight new DSMC trajectories, using the exact verified MV3 trajectory engine.
- Four unseen seeds at the former corner `kn0p1_u400`.
- Four unseen seeds at the new condition `kn0p08_u350`.
- A prediction receives only additive blocks 0, 1, and 2 from its indexed seed.
- Its target is built *after prediction locking* from the exact Raw B10 mean of
  the other three seeds at the same condition.
- The MV15B DCT weight map, threshold `0.97`, strength `0.25`, Mamba ensemble,
  TSVD rank, metrics, and gates are immutable.
- Failure is reported without retuning. This stage does not train a network and
  does not impose a continuum heat-flux closure.

The Slurm chain is an eight-task reference array (at most four concurrent), a
locked prediction job, and a post/package job. The returned ZIP contains JSON,
CSV, PNG, and vector PDF evidence but excludes raw checkpoints and model files.

## Unity installation and submission

Clone this branch with sparse checkout for the MV9, MV14, MV15A, MV15B, and
MV15C bundles, then run `install_and_submit_unity.sh` with the canonical Unity
project root. The installer selects `MV15C_PYTHON` when supplied, verifies the
entire ancestry lock and ten tests, and submits the dependency chain.

Monitor with:

```bash
source LAST_MOHAMMADZADEH_MV15C_FRESH_B3_JOB.env && squeue -j "$MV15C_JOB_IDS"
```

After completion, the post job writes
`LAST_MOHAMMADZADEH_MV15C_FRESH_B3_RESULT.env` and a SHA256-verified compact
archive in the Unity project root.
