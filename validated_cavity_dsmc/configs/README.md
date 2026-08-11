# Configuration levels

- `quick_cpu.toml` checks installation and output generation. It is not a
  physical validation.
- `student_kn01.toml` gives a moderate classroom run.
- `production_mohammadzadeh_kn*.toml` reproduces the 200x200, 32-particle
  baseline described in PRE 85, 056310 (2012). Use at least three independent
  seeds for book figures.
- The `_bt` file demonstrates the smaller automatically selected time step for
  a Bernoulli-trial method. The solver aborts if any trial probability exceeds
  unity under strict mode.

The paper reports a five-neighbor output filter. This repository validates raw
profiles first. Any filtered comparison must be reported as a separate
post-processing sensitivity rather than silently changing the solver output.
