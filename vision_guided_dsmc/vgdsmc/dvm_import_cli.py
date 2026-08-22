from __future__ import annotations

import argparse
from pathlib import Path

from .dvm_import import import_dvm_table, write_reference_npz


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a DVM/Shakhov moment table to the standard reference NPZ contract"
    )
    parser.add_argument("--input", required=True, help="Tecplot/CSV/whitespace DVM moment table")
    parser.add_argument("--output", required=True, help="Output reference .npz")
    parser.add_argument(
        "--columns",
        default=None,
        help="Comma-separated column order when the source has no usable header, e.g. x,y,rho,u,v,T,qx,qy",
    )
    parser.add_argument("--case", default="DVM", help="Reference case label")
    parser.add_argument("--knudsen", type=float, default=None)
    parser.add_argument("--lid-speed", type=float, default=None)
    parser.add_argument("--wall-temperature", type=float, default=None)
    args = parser.parse_args()

    grid = import_dvm_table(args.input, columns=args.columns)
    metadata = {
        "source": str(Path(args.input)),
        "case": args.case,
        "knudsen": args.knudsen,
        "lid_speed": args.lid_speed,
        "wall_temperature": args.wall_temperature,
    }
    output = write_reference_npz(grid, args.output, metadata=metadata)
    print(output)


if __name__ == "__main__":
    main()
