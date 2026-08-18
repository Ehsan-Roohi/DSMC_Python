# R13/R26 JFM code-protection policy

R13/R26 solver code, reference fields, SPARTA/DSMC comparisons, restart
lineage, and associated scripts for the active JFM article are **read-only
during repository cleanup**.

## Protected scope

- every branch or pull request identified as R13, R26, JFM, Rana, or
  Maxwell-transport evidence;
- solver source, patches, boundary conditions, closure coefficients, meshes,
  generated inputs, validators, checkpoints, compact reference data, and
  restart/continuation scripts used by those studies;
- stage decisions and negative or held results.

## Rules

1. Do not reformat, rename, move, squash, delete, or opportunistically merge
   protected material during general cleanup.
2. Keep competing scientific branches separate until a documented numerical
   comparison selects one.
3. A scientific change needs a narrowly scoped pull request recording the
   baseline commit, physical/numerical justification, reproduction command,
   and before/after evidence.
4. Never relabel a diagnostic, held, failed, or incomplete run as validated.
5. Keep large HPC outputs external unless a compact artifact is necessary to
   audit a numerical claim.

This policy protects the article code from repository-maintenance changes; it
does not certify every historical branch.
