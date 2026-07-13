#!/usr/bin/env python
"""Query a built literature KG."""

from __future__ import annotations

import argparse
import json

from chemgraph.kg.query import hybrid_query


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kg", required=True, help="Built KG directory.")
    parser.add_argument("--q", required=True, help="Natural-language question.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--response-mode",
        choices=("full", "compact"),
        default="full",
        help="Full audit output or compact model-facing output.",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Optional sentence-transformers model for vector retrieval.",
    )
    args = parser.parse_args()

    print(
        json.dumps(
            hybrid_query(
                args.kg,
                args.q,
                top_k=args.top_k,
                embedding_model=args.embedding_model,
                response_mode=args.response_mode,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
