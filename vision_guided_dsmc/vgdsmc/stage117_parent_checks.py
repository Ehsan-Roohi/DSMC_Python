from __future__ import annotations
import hashlib,sys
from pathlib import Path
EXPECTED={'summary.json':'6907dc4be24a3bb987bd80c6b57f1876ccbfba69a70ec5226a6213d4da6aebbd','pair_resolved_radial_node_profiles.npz':'309b17f927b32719be8aa6634508919b9d786b9b2321f2116779d9c3460bb019'}
p=Path(sys.argv[1])
for name,want in EXPECTED.items():
    got=hashlib.sha256((p/name).read_bytes()).hexdigest()
    if got!=want: raise SystemExit(f'checksum mismatch: {name}: {got}')
