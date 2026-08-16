# Instructions for Codex

This directory contains a validation-oriented SPARTA normal-shock benchmark for
the independent rarefied-gas-flow book project.

- Keep the case shock-fixed and one-dimensional.  Do not replace it with a
  transient shock-tube problem.
- Preserve the argon molecular constants in `data/` across solver ports.
- Treat shock recentering as coordinate alignment, never as field smoothing.
- Do not label a result validated unless far-field Rankine-Hugoniot and
  conservation gates are recorded in machine-readable output.
- Run `python3 -m unittest discover -s tests -v` and the real SPARTA smoke case
  before submitting production jobs.
- Preserve previous run directories and refuse accidental overwrites.

