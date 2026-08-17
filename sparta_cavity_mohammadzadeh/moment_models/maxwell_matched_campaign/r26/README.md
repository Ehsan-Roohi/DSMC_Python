# R26 pure-Maxwell rerun bundle

This bundle is a fail-closed correction of the exact R26 source family used by the accepted cavity runs. It does not edit the manuscript or reuse an old field as a final answer.

## Audit result

The hash-matched collision-relevant sources identify the model unambiguously:

- `r26_bulk_equations.py` states that the nonlinear R26 equations are the Gu--Emerson (2009) Maxwell-molecule equations.
- `r26_tensor_closures.py` retains the final `jfm2009` coefficient set `(C1,C2,Y1,Y2,Y3)=(2.097,0.291,1.698,1.203,0.854)`.
- `r26_wall_conditions.py` implements the Gu--Emerson smooth-wall equations; `accommodation=1` is the fully diffuse wall used here.
- The previous `jfm_observability_cavity_case` supplied `mu/mu0=(T/T0)^0.81`. Therefore those accepted R26 fields were a Maxwell-equation/VHS-transport hybrid, not a pure-Maxwell comparison.

The correction adds a separate `jfm-maxwell` family. It hard-locks `mu/mu0=T/T0`, `closure_mode=jfm2009`, Gu's equilibrium `Kn=lambda/L` normalization, and wall accommodation 1. Passing `--vhs-omega` in this mode is an error.

The fully recovered solver source is the N40 source family (`r26_solver.py` SHA-256 `6da3588cf78033f10d304e64fd2e451172d9497c8e4dcfdd2157d84e8a514b6d`), together with post-processing hash `ac32ca20a387227c9d61b16cc4215134c5e59032b72fac307dc3428b88e20c6a` and pre-patch driver hash `ec1d10fa25ecdc0b48b1f66bb95462162c02be9661994d32085a29c73c0c0d0d`. The older N20 summary records different solver/post-processing/driver hashes, but those exact three historical files were not recovered. The collision-relevant Maxwell sources are hash-identical. This bundle deliberately reruns both Kn values with the same fully recovered N40 solver family, eliminating a solver-version difference from the new comparison.

## Production cases

The production reruns preserve the grids used in the current comparison so the molecular-model change is isolated:

| case | nodes | beta | mu0* | lid* |
|---|---:|---:|---:|---:|
| KnGu=0.05 | 40 | 1.25 | 0.039894228040143274 | 0.40032038451271784 |
| KnGu=0.20 | 20 | 0.0 | 0.1595769121605731 | 0.40032038451271784 |

The old accepted states may be supplied only as nonlinear initial guesses. Both jobs force `--reconcile-initial`, solve the pure-Maxwell equations again at the physical lid, and promote a state only if the raw residual, positivity, wall-pressure, momentum-balance and energy-balance gates pass.

## Submit

Set the two seed paths and submit both independent jobs:

```bash
export R26_MAXWELL_SEED_KN005=/absolute/path/to/kn005_N40/last_accepted_state.npz
export R26_MAXWELL_SEED_KN020=/absolute/path/to/kn020_N20/last_accepted_state.npz
bash hpc/submit_maxwell_pair.sh
```

The validator prints `R26_MAXWELL_VALIDATION_PASS` only for an accepted full target. Do not regenerate centreline or anti-Fourier figures from a run without that marker.

## Verification already completed

- 85 existing R26 unit tests: PASS.
- 5 Maxwell locking/provenance tests: PASS.
- N=5 end-to-end smoke continuation at KnGu=0.05: PASS.
- N=5 end-to-end smoke continuation at KnGu=0.20: PASS.

Full N40/N20 production solves remain to be executed on the cluster. Exact hashes and smoke diagnostics are in `audit/model_audit.json`.
