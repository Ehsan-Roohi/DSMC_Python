# MV17B-A2 — deterministic locked-window recovery

MV17B-A2 corrects the failed A1 acquisition repair without changing the
preregistered MV17B scientific experiment. A1 incorrectly supplied `IRUN=4`;
DS2V accepts only 1, 2, or 3, reprompted, and then stopped at stdin EOF before
any trajectory was produced.

A2 keeps fresh `IRUN=3`, the original twelve seeds, the DS2VD deck, collision
model, sampling schedule, frozen estimator, endpoints, and gates unchanged.
Only the unique main-loop termination guard is extended from
`TIME < TLIM` to `(TIME < TLIM) OR (NOUT < 116)`. This lets the same
deterministic fresh trajectory reach the already locked endpoint.

Safety and provenance gates:

- only the same seven incomplete trajectories are rerun;
- the five complete trajectories are reused byte-for-byte;
- all previously stored overlapping moment blocks must match the A2 rerun
  byte-for-byte before any missing block is installed;
- existing moment blocks are never overwritten;
- A1's failure logs and hashes are retained in the final archive;
- no heat-flux values, predictions, targets, or performance metrics are read
  while A2 is locked or while the source termination guard is patched.

After recovery, the original frozen MV17B analysis and packaging code runs on
the unchanged B3/B10/guard split. A failed source anchor, build, overlap check,
or provenance check stops the dependency chain.
