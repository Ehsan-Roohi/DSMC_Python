# JFM 80M single-seed restartable checkpoint workflow

This deadline-oriented bundle replaces the seven long JFM GPU jobs with one
heavy realization for each distinct physical case. Every job writes an
interim, analysis-ready raw field at 1,000,000 completed steps and continues
in the same process to a 1,500,000-step final result.

At 1,500,000 steps it also writes a complete particle/RNG/collision/moment
restart. If the profiles remain noisy, the same realization can be continued
to 2,000,000 or 3,000,000 total steps without repeating the first segment.

## Numerical settings

- 7 distinct cases: Figure 5 (3), Figure 6 (3), and Figure 2(d) (1)
- 80,000,000 particles per case; seed 104729
- quarter-domain symmetry; 200 x 200 reconstructed full field
- sampling from step 100,000, every 2 steps
- interim checkpoint at step 1,000,000 (450,000 sampled instants)
- final output at step 1,500,000 (700,000 sampled instants)
- 14 contiguous time blocks, each spanning about 100,000 steps
- float64 sampled-moment accumulators
- no spatial smoothing and no velocity projection
- broad `vram48` GPU constraint; no `qos=long`

The checkpoint is written from the live accumulated moments and does not reset,
fork, or restart the simulation. It creates `*_step1000000_raw.npz`,
`*_step1000000_raw.dat`, and `*_step1000000_checkpoint.json`. Final outputs use
the original production names expected by the summary tool.

The final restart directory is approximately 2.1 GiB per case and ends in
`_restart_step1500000`. A valid restart always contains `manifest.json`.

After all seven 1.5M runs finish, a continuation can be submitted with:

```bash
bash scripts/submit_continuation.sh 2000000
```

or, for a larger extension:

```bash
bash scripts/submit_continuation.sh 3000000
```

The continuation keeps the original 50,000 sampled instants per temporal block,
loads the original RNG and collision state, and writes a new complete restart at
the requested final step.

## Replacement safety

Pass `replace` to `submit_from_github.sh` to cancel only the superseded JFM GPU
workflows, and only after both the new production array and its summary job have
been accepted by Slurm. The unrelated CPU JFM revision jobs are not cancelled.

The block diagnostics measure temporal stability within one heavy realization;
they are not independent-seed uncertainty.
