#!/usr/bin/env python
"""Suggest evidence-backed hypotheses from a built literature KG."""

from __future__ import annotations

import argparse
import json

from chemgraph.kg.hypotheses import suggest_hypotheses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kg", required=True, help="Built KG directory.")
    parser.add_argument("--goal", required=True, help="Scientific objective.")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    print(
        json.dumps(
            suggest_hypotheses(args.kg, goal=args.goal, top_k=args.top_k),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
