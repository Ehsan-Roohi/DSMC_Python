from __future__ import annotations

import argparse
from .reference_adapter import build_supervised_reference_case


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a DVM/deterministic-reference supervised DSMC case"
    )
    parser.add_argument("--coarse-case", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(build_supervised_reference_case(args.coarse_case, args.reference, args.output))


if __name__ == "__main__":
    main()
