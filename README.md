# DSMC Python research archive

Direct-simulation Monte Carlo, kinetic-model, and machine-learning experiments
for rarefied-gas dynamics.  The repository contains historical notebooks plus
reproducible gated campaigns for relaxation, cavity, shock, expansion, and
high-temperature collision studies.

## Default-branch map

- Root notebooks and scripts are historical research prototypes; filenames
  record their original experiment and are intentionally not bulk-renamed.
- [`qk_gate2_bundle/`](qk_gate2_bundle/) through the Gate-5 bundles contain
  staged Bird-QK/coupled-nozzle workflows.
- [`unity_gate2/`](unity_gate2/) contains the corresponding Unity submission
  and status scripts.
- The root data/image files are retained for provenance and are not a general
  package API.

Later DSMC, SPARTA, R13/R26, and reconstruction campaigns remain on their
original research branches and pull requests until their scientific status is
resolved.  Branch existence does not mean the result is accepted.

## Protected JFM code

R13/R26 solvers, reference fields, restart workflows, and numerical evidence
for the active JFM article are protected.  Read
[`JFM_R13_R26_PROTECTION.md`](JFM_R13_R26_PROTECTION.md).  Repository cleanup
must not merge or rewrite those assets simply to reduce branch or PR counts.

## Reproducibility policy

- Preserve the exact solver, random seed, mesh, timestep, sample count, and
  restart lineage for a quantitative claim.
- Distinguish exploratory, held, failed, and validated stages in filenames and
  reports.
- Keep large trajectories and cluster output outside GitHub; retain compact
  configuration, provenance, and summaries needed to reproduce them.
- Treat notebooks containing “not working” or similar labels as negative
  research records, not supported entry points.

## License and citation

No repository-wide license or citation metadata is declared yet.  These must
be chosen only after the article authorship and release scope are confirmed.
