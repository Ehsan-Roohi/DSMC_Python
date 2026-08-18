# Publishing checklist

Recommended repository target:

```text
Ehsan-Roohi/DSMC_Python
branch: agent/maxwell-matched-antifourier
sparta_cavity_mohammadzadeh/
  moment_models/maxwell_matched_campaign/public_release/
```

The independent `public_release` subtree prevents the clean licence and
provenance boundary from being confused with older working files already on
the branch.

Before publication:

1. copy this subtree without merging it with the sibling development bundle;
2. run `python analysis/validate_release.py`;
3. run both figure scripts and inspect their generated PDF/PNG files;
4. regenerate `MANIFEST.sha256` after any edit;
5. confirm no supplied/derived R13 legacy source appears in the commit;
6. create an immutable GitHub release tag, recommended
   `jfm-antifourier-v1.0.0`;
7. archive the tag with Zenodo, then add the assigned DOI to
   `CITATION.cff` and the manuscript data-availability statement; and
8. avoid using the mutable branch URL as the sole permanent data citation.

This preparation tree was not pushed by the release audit.

