#!/usr/bin/env python
"""Temporal backtest for active-metal/support missing-link predictions."""

from __future__ import annotations

import argparse
import json

from chemgraph.kg.evaluation import temporal_link_backtest
from chemgraph.kg.ingest import ingest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papers", required=True, help="Paper file or directory.")
    parser.add_argument("--split-year", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--task", default="link_prediction", choices=["link_prediction"])
    args = parser.parse_args()
    chunks = ingest_path(args.papers)
    result = temporal_link_backtest(
        chunks,
        split_year=args.split_year,
        top_k=args.top_k,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
