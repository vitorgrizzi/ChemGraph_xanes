#!/usr/bin/env python
"""Evaluate extracted CatalystRecord JSONL against a manually labeled gold set."""

from __future__ import annotations

import argparse
import json

from chemgraph.kg.benchmark import evaluate_extractions
from chemgraph.kg.extract import read_records_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predicted", required=True, help="Predicted CatalystRecord JSONL.")
    parser.add_argument("--gold", required=True, help="Gold CatalystRecord JSONL.")
    args = parser.parse_args()
    result = evaluate_extractions(
        read_records_jsonl(args.predicted),
        read_records_jsonl(args.gold),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
