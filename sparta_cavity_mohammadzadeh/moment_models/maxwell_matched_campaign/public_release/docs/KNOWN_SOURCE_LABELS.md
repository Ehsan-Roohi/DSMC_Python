# Source-lock note

The exact R26 files are distributed byte-for-byte with the numerical source
manifest. A few identifiers and docstrings retain development-era names for
tensor component packing and access status. No external package bearing those
names is bundled, imported or required. The scientific lineage of this
release is the independently implemented Gu--Emerson 2009 equation set stated
in `PROVENANCE.md`.

These comments were not mechanically renamed because doing so would break the
source hashes embedded in both accepted run records. User-facing paper text,
the main README and citation metadata do not use those development labels as
provenance.

