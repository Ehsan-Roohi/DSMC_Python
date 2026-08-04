# Instructions for Codex

This directory is a teaching and validation-ready SPARTA case for the
Mohammadzadeh et al. lid-driven cavity benchmark.

- Preserve the physical constants in `data/` and the reference values in
  `reference/` unless the user explicitly requests a documented sensitivity
  study.
- Run `python3 -m unittest discover -s tests -v` before a SPARTA run.
- Run the `smoke` level before `student` or `production`.
- A smoke or student run is not publication validation.  Only label a result
  validated after the 200x200, 32-particle-per-cell production case passes the
  quantitative gates and repeat-seed uncertainty has been reported.
- Do not hide a failed gate by smoothing, filtering, or changing the reference
  data.  If a five-cell filter is applied to reproduce the paper, report the
  raw and filtered profiles separately.
- Do not use destructive cleanup commands.  Preserve completed `runs/`
  directories unless the user explicitly names one to remove.
