# MV15C-A1 — q_y evaluation from the eight completed MV15C trajectories

MV15C-A1 evaluates the unchanged, frozen MV15B B3 DCIR-QY rule without
rerunning DSMC. The original eight MV15C trajectories all reached the exact
locked final step and pass their mechanical, provenance, finite-field, and
checkpoint checks. Two trajectories missed the original stochastic
temperature-extremum stationarity z gate, which caused Slurm `afterok` to
cancel prediction and packaging before any model outcome was produced.

Before constructing any prediction or leave-one-seed-out target, MV15C-A1:

- freezes the original two QC holds and every reference hash;
- includes all eight original seeds without replacement or selection;
- leaves B3 blocks, weights, network checkpoints, TSVD rank, targets, metrics,
  and q_y acceptance gates unchanged;
- reports the original reference-QC result separately from the q_y result;
- never labels a positive result as the unamended preregistered confirmation;
- writes a compact SHA256-verified ZIP to the Unity project root.

The submitted chain contains only prediction and post/package jobs. It submits
no DSMC reference job.

