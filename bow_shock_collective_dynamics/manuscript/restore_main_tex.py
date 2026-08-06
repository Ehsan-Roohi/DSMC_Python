#!/usr/bin/env python3
from __future__ import annotations

import base64
import gzip
from pathlib import Path

HERE = Path(__file__).resolve().parent
encoded = (HERE / "main.tex.gz.b64").read_text(encoding="ascii")
source = gzip.decompress(base64.b64decode(encoded))
(HERE / "main.tex").write_bytes(source)
print(f"Restored {HERE / 'main.tex'} ({len(source)} bytes)")
