#!/usr/bin/env python
"""Build the literature KG from CatalystRecord JSONL."""

from __future__ import annotations

import argparse
import json

from chemgraph.kg.extract import read_records_jsonl
from chemgraph.kg.store import build_kg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, help="Input CatalystRecord JSONL.")
    parser.add_argument("--out", required=True, help="Output KG directory.")
    parser.add_argument("--synonyms", default=None, help="Optional synonym YAML path.")
    args = parser.parse_args()

    records = read_records_jsonl(args.records)
    result = build_kg(records, args.out, synonyms_path=args.synonyms)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
