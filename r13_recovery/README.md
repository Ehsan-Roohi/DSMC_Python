# R13 recovery and N80 verification

This directory contains the traceable R13 solver recovered for the
reviewer-requested `Kn_Gu = 0.2` comparison and the Unity submission script for
the same-Kn `N60 -> N80` grid-refinement run.

## Scientific scope

The 17-state bulk matrices and wall machinery are derived from the supplied
Rana source associated with:

> Rana, Torrilhon & Struchtrup, *Journal of Computational Physics* 236 (2013)
> 169--186.

The accepted N60 calculation uses:

- the published two-point linear wall extrapolation (`paper-linear`);
- the tangential-stress effective wall pressure printed in the paper
  (`paper-tangential`);
- a conservative shared-face continuity discretization;
- defect-Newton linearization and a globally guarded JFNK solve.

Consequently the defensible current claim is:

> an independently implemented, converged, physically admissible R13
> prediction based on the Rana equations and recovered coefficient matrices.

It must **not** yet be called an exact reproduction of the original Rana
MATLAB/archive result. Exact archive reproduction, resolution of the printed
paper-versus-archive boundary/coefficient choices, independent comparison with
the original output, and same-Kn grid convergence remain separate gates.

## N80 run

`unity_submit_r13_n80.sh` verifies the accepted N60 state/report pair and the
local coefficient-module hash, copies an immutable runtime bundle, and submits
an N80 JFNK solve. The solver transfers the accepted N60 state internally onto
the N80 interior-node grid and records that operation as
`initial_guess_only_not_an_accepted_solution`; only the completed N80 nonlinear
solve can promote the result.

The recovered coefficient module `rana_original_coefficients.py` was not
included in the uploaded recovery capture. The Unity submitter therefore reads
the campaign copy from `code/r13/` and verifies it against the coefficient hash
recorded by the accepted N60 report before submission. That module must be
added to this public directory before claiming a fully self-contained public
R13 release.
