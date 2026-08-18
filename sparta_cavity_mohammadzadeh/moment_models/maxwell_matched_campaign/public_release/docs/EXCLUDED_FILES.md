# Deliberate exclusions

The following material is not part of this public release.

## Redistribution hold

- the original MATLAB R13 cavity source supplied by A. S. Rana;
- `rana_original_coefficients.py`, a transcription/adapter derived from that
  supplied source; and
- `rana_original_reference_solver.py`, a solver derived from that supplied
  source.

These files require explicit redistribution permission. Their absence is a
licensing/provenance decision, not a missing numerical run. The accepted R13
states, run records, public printed-equation module and validators are
included.

Older paths elsewhere on the development branch may still contain the two
derived Python files. They are outside `public_release`, are not covered by
its BSD licence, and should be removed or quarantined before a repository-wide
licence is asserted.

## Not required for reproduction

- SPARTA source and binaries (use the pinned official upstream revision);
- any unverified external R13/R26 solver or patch artifact;
- raw particle trajectories;
- restart and checkpoint dumps;
- intermediate SPARTA grids;
- scheduler, MPI-preflight, terminal-screen, progress and failure logs;
- hostnames, queue records and cluster packaging metadata;
- `__pycache__`, test caches and editor files;
- obsolete absolute-path checksum manifests; and
- historical common-mask/model-support and moment-grid tables no longer used
  by the revised manuscript.

The compact eight-realization DSMC sensitivity table is retained because it
reproduces a published numerical-sensitivity figure; it is labelled as the
earlier VHS sensitivity campaign and is not substituted for the matched
Maxwell cases.
