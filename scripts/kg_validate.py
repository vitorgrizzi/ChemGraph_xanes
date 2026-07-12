#!/usr/bin/env python
"""Validate a persisted literature KG and its provenance references."""

from __future__ import annotations

import argparse
import json

from chemgraph.kg.validation import validate_kg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kg", required=True, help="Built KG directory.")
    parser.add_argument("--skip-hashes", action="store_true")
    args = parser.parse_args()
    result = validate_kg(args.kg, verify_hashes=not args.skip_hashes)
    print(json.dumps(result, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
