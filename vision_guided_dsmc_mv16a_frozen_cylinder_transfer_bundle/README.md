# MV16A — frozen cavity-to-cylinder q_y transfer audit

MV16A reuses the four completed MV11 Bird/DS2V Mach-10 argon-cylinder
trajectories. It submits **no DSMC job** and performs **no training or tuning**.
The successful MV15C-A1 Mamba ensemble, MV15B B3 DCIR-QY weight map, and TSVD
rank are inherited unchanged.

## Why the MV11 data can be reused

The original MV11 `tU/D >= 30` completion gate is false because the four runs
ended near `tU/D=11.3--11.6`. The solver and Slurm jobs did not fail. All four
trajectories contain a common, grid-stable NOUT 100--116 interval. Raw scalar
diagnostics in that interval showed less than one-percent fitted change in
`q_y` RMS and 0.142% cross-seed dispersion. MV16A preserves the original
completion warning and is labelled a retrospective stationarity-amended
transfer audit, never an unamended MV11 confirmation.

## Locked split

- Raw B3 input: NOUT `100,108,116`.
- Raw B10 source: NOUT `101--105,109--113`.
- Unused guard/QC blocks: NOUT `106,107,114,115`.
- Additive moments are summed before one centralisation.
- The target for each seed is the mean of the other three seeds' disjoint Raw
  B10 fields.
- Predictions are recursively hashed before any B10 target is constructed.

## Geometry transfer

MV11 cells are unstructured. They are deterministically interpolated onto the
exact frozen MV15B DCT-weight shape, with an analytic cylinder mask and a
minimum 90% linear-interpolation coverage gate. The original functional
conditioning form is retained: `log10(Kn)` and characteristic speed divided by
100. The geometry interface makes one explicit, predeclared semantic
substitution, `U_lid -> U_inf`. The Mach-10 value is outside the cavity training
range and is neither clipped nor selected after seeing cylinder outcomes.

## Returned result

The post job creates a SHA256-verified `MV16A_FROZEN_CYLINDER_TRANSFER_BUNDLE_*.zip`
in the Unity project root. It contains per-seed metrics and four publication-
style six-panel `q_y` figures in the fixed order:

1. Reference;
2. Raw DSMC B3;
3. MambaIRv2 B3;
4. DCIR-QY B3;
5. TSVD/POD B3;
6. Raw DSMC B10.

All panels use `RdBu_r` and shared field/error colour limits across the four
seeds.

## Scientific guardrail

A positive result is evidence of frozen cross-geometry transfer under a
transparent late-stationarity amendment. A negative result is reported without
retuning these four observed seeds. A later tU/D=30 fresh-cylinder campaign is
only warranted after this audit if an unamended prospective confirmation is
needed.
