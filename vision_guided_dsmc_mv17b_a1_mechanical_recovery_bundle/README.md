# MV17B-A1 — outcome-blind mechanical recovery

MV17B-A1 repairs the acquisition failure that occurred before any MV17B
prediction or reference target was constructed.  Seven of the twelve locked
DS2V trajectories exited normally before `NOUT=116`; the other five are reused
unchanged.

The recovery keeps every scientific choice fixed:

- the six observation/reference pairs and all twelve seeds are unchanged;
- the frozen MV17B model, B3/B10/guard split, endpoints, and gates are unchanged;
- only the seven mechanically incomplete trajectories are rerun;
- the rerun uses the same executable, deck, and seed, changing only fresh
  `IRUN=3` to `IRUN=4` so that the predeclared window is reached;
- every overlapping moment block must be byte-identical before missing blocks
  are installed into the campaign;
- no heat-flux field, prediction, target, or metric is read while the recovery
  lock is created.

After recovery, the original frozen MV17B analysis and packaging code runs on
the original preregistered NOUT split.  This is an acquisition repair, not a
new experiment, seed replacement, parameter search, or model update.

