# Release audit

## Scope decision

The public tree is intentionally narrower than the working campaign tree. It
contains only redistributable original source, published-equation
transcriptions, validators, run generators, processed states and reduced data.
The private supplied/derived R13 solver lineage is hash-identified but not
republished.

## Checks performed

- R26 unit suite: 85/85 tests passed.
- R26 Maxwell source-lock contract: 5/5 tests passed.
- R26 `Kn_Gu = 0.05`, 40-node validator: passed.
- R26 `Kn_Gu = 0.20`, 20-node validator: passed.
- R13 public Appendix-A equation contract: passed.
- R13 `Kn_Gu = 0.05`, 60-node numerical-state validator: passed.
- R13 `Kn_Gu = 0.20`, 60-node numerical-state validator: passed.
- DSMC `Kn_Gu = 0.05` 15-field final-grid validator: passed.
- DSMC `Kn_Gu = 0.20` 15-field final-grid validator: passed.
- public pure-Maxwell reducer: regenerated four numerical tables and all six
  matched-model PDF/600-dpi PNG figures without importing excluded source.
- DSMC sensitivity figure script: deterministic PDF/600-dpi PNG generation
  checked.
- portable SHA-256 manifest: regenerated from release-relative paths.

The final command transcript is written to `VALIDATION.txt` when
`analysis/validate_release.py` is run by the release packager.

## Claims not made by this audit

Passing these checks does not establish that R13 or R26 is uniformly accurate
against DSMC, does not erase model-form differences in higher moments, and
does not promote a single DSMC realization to an uncertainty ensemble. It
does establish that the released matched cases use the declared molecular,
wall, grid, sampling and Knudsen-number contracts and that each accepted
moment state satisfies its recorded numerical gates.
