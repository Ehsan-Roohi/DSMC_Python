# R26 recovery source

This directory contains the recovered R26 solver and the guarded recovery patch
prepared for the reviewer-requested R26 calculation.

## What changed

- fresh-Jacobian fallback from direct LU to regularized LSMR directions;
- Armijo backtracking based on the actual clipped step;
- positivity-preserving secant prediction in Knudsen number;
- merit-guarded restart reconciliation when Kn changes;
- strict final residual/physics acceptance gates are retained.

## Validation

- 87 unit tests pass locally;
- a three-stage 5x5 continuation smoke test at Kn = 0.050, 0.055, and
  0.056 reaches `target_accepted`;
- the GitHub workflow runs the same unit-test and continuation smoke gates.

The production N=40 calculation is intentionally not started by CI. A standard
GitHub-hosted job is limited to six hours, while the earlier full continuation
ran for more than twenty-two hours. Production runs must be checkpointed by
stage or submitted to the Unity/self-hosted runner with the private restart
state.

