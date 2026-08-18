# R13 component

The public R13 source is limited to the collision-production module
transcribed from the printed 2013 Appendix A and independent tests. The legacy
R13 cavity solver supplied by A. S. Rana, and Python files derived from it,
are not redistributed without explicit permission.

`equations/r13_maxwell_production.py` exposes two named branches:

- `production_maxwell`: Appendix-A matrix times `rho/Kn_Rana`, consistent
  with the stated Maxwell viscosity law and nondimensional relaxation rate;
- `production_appendix_literal`: the matrix divided by `Kn_Rana`, matching a
  literal reading of the displayed equation.

The released states used the first branch. The branch choice is explicit and
was not fit to DSMC. Run the public equation checks with:

```bash
python tests/test_r13_equation_contract.py
```

The processed states and their numerical run records are under `../data/r13`.
See `../docs/PROVENANCE.md` for the starting point, modifications and
redistribution boundary.

