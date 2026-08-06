# Collective dynamics of a rarefied hypersonic bow shock

This repository contains the manuscript, processed data, figures, and analysis code for:

> **An emergent collective coordinate in the stochastic dynamics of a rarefied hypersonic bow shock**  
> Ahmad Shoja-Sani and Ehsan Roohi (corresponding author)

## Physical result

The time-resolved DSMC fields are strongly high dimensional, but the physical covariance of the density half-jump marker contains a weak, reproducible forebody displacement coordinate at `Kn_D = 0.01` and `0.025`.

The result is distinct from the companion mean-flow preprint:

- **Companion paper:** mean bow-shock inflation, standoff, thickness, and parameter-space similarity.
- **Present paper:** temporal covariance, collective displacement, angular kinematics, body-scale memory, and full-field multi-moment validation.

Companion preprint: [Rarefaction-induced inflation and similarity breakdown of hypersonic bow shocks over a circular cylinder](https://arxiv.org/abs/2605.17099).

Key dynamic findings:

- the independently inferred low-Kn modes correlate by `0.972`;
- a uniform translation plus one curvature harmonic represents `97-98%` of the mode shape;
- the displacement standard deviation is only `0.29-0.45%` of the mean 10-90 layer thickness;
- memory persists for `6.5-10.7` local layer-crossing times;
- density and pressure recover the same amplitude independently from the complete fields;
- higher-Kn non-detection is reported as a loss of statistical identifiability, not as proven physical disappearance.

## Repository layout

- `manuscript/` -- JFM LaTeX source, compiled PDF, vector figures, processed plotting data, and figure scripts.
- `analysis/unified_all_kn_pipeline/` -- common-200 POD, marker extraction, temporal coarse graining, covariance inference, QC, and collection scripts.
- `analysis/displacement_template_validation/` -- full-field translation-template validation.
- `analysis/final_statistical_gate/` -- sliding-window, synthetic-control, and power/exclusion analysis, including the corrected post-processing outputs.
- `dsmc_output/` -- DS2V modal-output patch, run-control generation, validation, and output packaging scripts.
- `release/` -- complete-source download instructions.

## Build the manuscript

```bash
cd manuscript
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The repository includes the JFM class and bibliography style. On systems where the `bibtex` alternative is broken, use the installed executable directly, for example `bibtex.original main`.

## Reproduce the enhanced physics figures

```bash
cd manuscript
python code/enhanced_physics_figures.py
```

The script uses the processed covariance and template products stored under `manuscript/source_support/`.

## Raw data

The raw DS2V `DS2FF` snapshot campaign is too large for GitHub. It is therefore not duplicated here. The repository contains the DS2V output-control source and run plan, the exact post-processing configuration, processed covariance products, all tabulated values, figure-generation scripts, and validation/data-lineage notes.

## Authors

- **Ahmad Shoja-Sani** -- first author; DSMC simulations, data curation, software, validation, analysis, and visualization.
- **Ehsan Roohi** -- corresponding author; conceptualization, methodology, supervision, interpretation, and manuscript review.

## Code license

Analysis and utility code are released under the MIT License. The manuscript remains an author draft and should be cited rather than redistributed as a journal-formatted published version.
