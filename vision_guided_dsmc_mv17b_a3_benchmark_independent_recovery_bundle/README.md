# MV17B-A3 — fixed-endpoint, benchmark-independent recovery

MV17B-A3 repairs the seven mechanically incomplete MV17B cylinder
trajectories without changing the frozen estimator, seeds, DSMC physics, or
scientific endpoints.

The A2 diagnosis is now explicit. A2 extended the main loop to
`(TIME < TLIM) OR (NOUT < 116)`, but DS2V still executed the legacy active
statement

```fortran
IF (SUM_heat_ER.LE.Heat_er) CONV_PAR=1
```

and the main loop subsequently followed `IF (CONV_PAR.EQ.1) GO TO 2512`.
Therefore all seven A2 reruns again stopped at NOUT 105--115. The warning
`PROGRAM COULD NOT ALLOCATE VAR,IC 5014` was not the termination mechanism.

A3 applies exactly two audited source transformations to the original MV17B
DS2V source:

1. the unique acquisition loop becomes `DO WHILE (NOUT < 116)`;
2. the unique active benchmark-dependent `CONV_PAR` setter is disabled.

The fixed endpoint is independent of `HEAT-BENCH.TXT`. The benchmark file is
still supplied because the legacy solver reads it, but its values cannot end
the acquisition. A3 keeps fresh `IRUN=3`, all original seeds, the DS2VD deck,
collision model, sampling schedule, B3/B10 split, frozen estimator, endpoints,
and acceptance gates unchanged.

Fail-closed provenance rules:

- rerun only indices `1,2,3,4,9,10,11`; reuse the five complete trajectories;
- stage A3 output separately under `campaign/recovery_a3`;
- require every existing overlapping moment block to be byte-identical before
  installing any missing block;
- never overwrite an existing moment block;
- preserve both A1 and A2 failure locks, source reports, and run logs;
- read no q_y target, prediction, or performance value before recovery;
- stop the dependency chain on any patch-anchor, build, overlap, provenance,
  or endpoint failure.

This is an outcome-blind same-seed acquisition repair. It is not a new model,
parameter selection, seed replacement, or alteration of the scientific
acceptance endpoints.
