# Existing-data pilot: decision record

Classification: retrospective development-only mechanism check.  These results
are not submission evidence.

The literal P-NN+EB proposal fails its provisional gate for every tested DCT
partition.  The development-neighbour prior is strongly biased at the two
available conditions, and adaptive EB fusion merely falls back toward the
prior-free result.

| DCT bin width | Condition | P-NN+EB / Raw-B10 | P-NET+EB / Raw-B10 | P-NET+EB / min(prior,P-0) |
| ---: | --- | ---: | ---: | ---: |
| 8 | Kn=0.08, U=350 | 0.820 | 0.610 | 0.749 |
| 8 | Kn=0.10, U=400 | 0.926 | 0.635 | 0.725 |
| 10 | Kn=0.08, U=350 | 0.812 | 0.604 | 0.742 |
| 10 | Kn=0.10, U=400 | 0.921 | 0.630 | 0.720 |
| 20 | Kn=0.08, U=350 | 0.823 | 0.627 | 0.759 |
| 20 | Kn=0.10, U=400 | 0.928 | 0.658 | 0.744 |
| 25 | Kn=0.08, U=350 | 0.840 | 0.650 | 0.779 |
| 25 | Kn=0.10, U=400 | 0.946 | 0.687 | 0.767 |

For the 10x10 partition, P-NET+EB improves on the prior/P-0 envelope in all
eight seed units.  The promoted JCP method is therefore P-NET plus adaptive,
data-consistent fusion, with P-0 as the out-of-support fallback.  P-NN remains
an ablation rather than the central estimator.
