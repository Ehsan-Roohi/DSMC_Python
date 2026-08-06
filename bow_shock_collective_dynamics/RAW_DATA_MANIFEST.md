# Raw-data manifest

The full raw DSMC campaign is not hosted on GitHub because it contains thousands of large Tecplot `DS2FF` files.

## Cases

`Kn_D = 0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.25, 0.50, 1.0`, all at `M_inf = 10` for nitrogen.

## Snapshot counts

| Kn_D | available snapshots | Delta t* |
|---:|---:|---:|
| 0.010 | 200 | 0.288 |
| 0.025 | 600 | 0.240 |
| 0.050 | 419 | 0.431 |
| 0.075 | 674 | 0.491 |
| 0.100 | 400 | 0.369 |
| 0.150 | 703 | 0.389 |
| 0.250 | 462 | 0.443 |
| 0.500 | 350 | 0.508 |
| 1.000 | 283 | 0.789 |

## Reproduction route

1. Apply the DS2V output-control patch to the validated cylinder source.
2. Generate campaign controls with `make_campaign_control_files.py`.
3. Validate each run with `validate_modal_run.py`.
4. Edit raw-data paths in `all_kn_campaign_config.json`.
5. Run QC, corrected common-200 POD, temporal coarse graining, and covariance inference.
6. Run the displacement-template and final statistical-gate packages.

Processed products sufficient to reproduce the manuscript physics tables and enhanced figures are included in this repository. The complete manuscript/source/figure bundle is also provided as a downloadable ZIP in the ChatGPT project delivery.
