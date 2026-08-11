#!/usr/bin/env python3
"""Make the legacy downstream buffer a closed constant-area duct.

The Gate-5 baseline uses boundary 6 as an open receiver surface.  For the
Mach--residence design, boundary 6 must be a wall; otherwise the nominal
post-shock length is an open plenum rather than a residence duct.  The input
deck selects IB(6)=2, and this patch adds the missing K=6 specular reflection
branch to MOVE2.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXPECTED_BEFORE_SHA256 = "3002a264b456a28178f9e3ca55c0703f1fdd36d947ea61b342fa91d014a70c4a"
EXPECTED_AFTER_SHA256 = "da24ff8d41de877ab68a04ec038e182f32b22090b806f7b888f4384f76e5dd2f"


NEEDLE = """\t      ELSEIF(K==4)THEN
\t\t\t  Y=2.*CB(K)-Y
\t\t\t  PV(2,N)=-PV(2,N)
"""

REPLACEMENT = """\t      ELSEIF(K==6)THEN
C             Gate 5 geometry preflight: boundary 6 is the flat lower wall
C             of the constant-area post-shock duct when IB(6)=2.
\t\t\t  Y=2.*YS-Y
\t\t\t  PV(2,N)=-PV(2,N)
\t      ELSEIF(K==4)THEN
\t\t\t  Y=2.*CB(K)-Y
\t\t\t  PV(2,N)=-PV(2,N)
"""


def patch_source(path: Path) -> tuple[str, str]:
    before = path.read_bytes()
    before_sha = hashlib.sha256(before).hexdigest()
    if before_sha != EXPECTED_BEFORE_SHA256:
        raise RuntimeError(
            f"unexpected base source SHA-256: {before_sha}; "
            f"expected {EXPECTED_BEFORE_SHA256}"
        )
    text = before.decode()
    if "Gate 5 geometry preflight: boundary 6" in text:
        raise RuntimeError("geometry patch is already present")
    count = text.count(NEEDLE)
    if count != 1:
        raise RuntimeError(f"expected one MOVE2 K=4 block, found {count}")
    text = text.replace(NEEDLE, REPLACEMENT)
    path.write_text(text)
    after = path.read_bytes()
    after_sha = hashlib.sha256(after).hexdigest()
    if after_sha != EXPECTED_AFTER_SHA256:
        raise RuntimeError(
            f"patched source SHA-256 mismatch: {after_sha}; "
            f"expected {EXPECTED_AFTER_SHA256}"
        )
    return before_sha, after_sha


def self_test() -> None:
    sample = "prefix\n" + NEEDLE + "\t\t  ENDIF\nsuffix\n"
    patched = sample.replace(NEEDLE, REPLACEMENT)
    assert patched.count("ELSEIF(K==6)THEN") == 1
    assert patched.count("ELSEIF(K==4)THEN") == 1
    assert "Y=2.*YS-Y" in patched
    print("GATE5_GEOMETRY_SOURCE_PATCH_SELF_TEST_PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    if args.source is not None:
        before, after = patch_source(args.source)
        print(f"GATE5_GEOMETRY_SOURCE_PATCH_PASS before={before} after={after}")
    elif not args.self_test:
        parser.error("provide --source or --self-test")


if __name__ == "__main__":
    main()
