# JFM manuscript source

## Current draft

**Title:** *An emergent collective coordinate in the stochastic dynamics of a rarefied hypersonic bow shock*

**Authors:** Ahmad Shoja-Sani (first author) and Ehsan Roohi (corresponding author).

The draft is deliberately separated from the companion mean-flow preprint. The companion paper establishes mean inflation, standoff, thickness and parameter-space similarity. This manuscript instead concerns temporal covariance, collective displacement, angular kinematics, body-scale memory and full-field multi-moment validation.

Companion preprint: [arXiv:2605.17099](https://arxiv.org/abs/2605.17099).

## Restoring `main.tex`

The exact LaTeX source is stored as a compressed Base64 text file so that it can be transferred through the GitHub connector without altering special characters:

```bash
python restore_main_tex.py
```

This creates `main.tex` from `main.tex.gz.b64`.

## Build

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Required template files (`JFM-FLM_Au.cls` and `jfm.bst`) are part of the complete source bundle delivered with this repository release. Processed data and figure-generation scripts are under `data/` and `code/`.

## Main non-repetitive physics

- complete-field fluctuations remain strongly high rank;
- the physical marker covariance contains a much weaker collective coordinate at `Kn_D=0.01` and `0.025`;
- the two independent mode shapes correlate by `0.972`;
- translation plus one curvature harmonic represents `97-98%` of the mode shape;
- displacement amplitude is only `0.29-0.45%` of the mean layer thickness;
- memory lasts `6.5-10.7` local layer-crossing times;
- density and pressure recover the same temporal amplitude independently from full-field matched filters;
- larger-Kn cases are reported as statistically unresolved rather than as proven physical disappearance.
