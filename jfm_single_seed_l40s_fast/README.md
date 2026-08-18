# JFM single-seed fast completion on L40S

This deadline-oriented bundle completes the seven high-statistics physical cases
using one heavy realization per case. It deliberately reuses the three first-seed
Figure 5 jobs already running on RTX 8000 GPUs and submits only the four missing
physical cases to exact L40S GPUs.

## Numerical settings

- 80,000,000 particles per case
- seed 104729
- sampling begins at step 100,000 (not 1,000,000)
- sampling every 2 steps
- 3,000,000 total steps for the four new L40S jobs
- 29 contiguous time blocks (about 100,000 steps per block)
- no spatial filtering or velocity projection
- exact Slurm constraint `l40s`; no `qos=long`
- four simultaneous GPU tasks

The three retained Figure 5 jobs were already submitted with 5,000,000 steps,
the same 80M particles and sampling from step 100,000. Their Slurm jobs are
62606478 (HS), 62611438 (BGK), and 62611907 (Shakhov).

## Statistical limitation

The summary reports variation across contiguous time blocks. These block-based
diagnostics assess temporal stability but are **not independent-seed uncertainty**.
Do not describe the final fields as a multi-seed ensemble. The previously planned
BGK/Shakhov Kn=0.01 cross-seed endpoints are excluded from this fast workflow;
omit or visibly flag them in the Ek plot.

## Submit

Run `submit_from_github.sh` from the Unity login node. The final summary job waits
for the four new L40S cases and the three retained Figure 5 jobs. Override the
three dependency job IDs with `JFM_FIG5_DEPENDENCY_IDS` only if needed.

## Cancel superseded work

The cancellation command is printed by the bootstrap but is not executed
automatically. Review it before running it. It preserves array tasks 0, 3 and 6.
