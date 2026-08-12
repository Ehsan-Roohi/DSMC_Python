# MV6 postprocessor-only baseline-verification fix

This bundle repairs the MV6 aggregation failure observed after all twelve model
tasks completed successfully. The model predictions, targets, identities, Raw
fields, Gaussian fields, and TSVD fields were identical across all tasks. The
failure came from exact JSON float-dictionary equality across heterogeneous CPU
nodes.

The patched aggregator now:

- verifies the six shared arrays exactly across all twelve task artifacts;
- compares derived JSON metric trees with a strict relative tolerance of
  `1e-6`, absolute tolerance of `1e-12`, and equal-NaN handling;
- retains all existing artifact, task, protocol, and decision checks.

The installer updates only the aggregator and its focused test, then submits
only the postprocessor. It does not submit DSMC references or model training.

Run from the existing Unity checkout with:

```bash
TMP="$(mktemp -d)"
git clone --depth 1 --filter=blob:none --sparse \
  --branch agent/mv6-reference-stability-repair \
  https://github.com/Ehsan-Roohi/DSMC_Python.git "${TMP}/repo"
git -C "${TMP}/repo" sparse-checkout set \
  vision_guided_dsmc_mv6_bundle \
  vision_guided_dsmc_mv6_postfix_bundle
bash "${TMP}/repo/vision_guided_dsmc_mv6_postfix_bundle/install_and_submit_unity.sh" "$PWD"
```
