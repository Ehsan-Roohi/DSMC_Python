# Third-party notices

## SPARTA

The DSMC input decks were run with the official
[SPARTA](https://github.com/sparta/sparta) source at commit
`912c9e163c38ea5c3562d039e65215f6e2a4f3f8`. SPARTA is distributed upstream
under GPL-2.0. No SPARTA source or binary is included in this directory; only
original input generators, run scripts and outputs are provided.

## Published mathematical specifications

The R26 implementation follows the equations, closure coefficients and wall
conditions published by X.-J. Gu and D. R. Emerson, *Journal of Fluid
Mechanics* **636** (2009), DOI
[10.1017/S002211200900768X](https://doi.org/10.1017/S002211200900768X).
It is an independent implementation; no source from the paper's authors is
bundled.

The public R13 equation module transcribes the printed Appendix-A production
matrix in A. S. Rana, M. Torrilhon and H. Struchtrup, *Journal of
Computational Physics* **236** (2013), DOI
[10.1016/j.jcp.2012.11.023](https://doi.org/10.1016/j.jcp.2012.11.023).
The legacy cavity implementation supplied by A. S. Rana was used as the
starting point for the numerical R13 work, but neither that supplied source
nor reconstructed source derived from it is redistributed here.

No source from `rgd-software/fenicsR13`, or from any other public R13/R26
solver repository, is included or required by this release.

## Python dependencies

NumPy, SciPy and Matplotlib are runtime dependencies installed separately and
retain their respective upstream licences. Their source is not bundled.

