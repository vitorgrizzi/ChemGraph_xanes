#!/usr/bin/env python
"""Temporal backtesting scaffold for literature KG link prediction."""

from __future__ import annotations

import argparse
import json
import re

from chemgraph.kg.ingest import ingest_path


def _year_from_chunk(chunk) -> int | None:
    for value in [
        chunk.metadata.get("year"),
        chunk.metadata.get("publication_year"),
        chunk.source_path or "",
        chunk.text[:500],
    ]:
        match = re.search(r"\b(19|20)\d{2}\b", str(value))
        if match:
            return int(match.group(0))
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papers", required=True, help="Paper file or directory.")
    parser.add_argument("--split-year", type=int, required=True)
    parser.add_argument("--task", default="link_prediction", choices=["link_prediction"])
    args = parser.parse_args()

    chunks = ingest_path(args.papers)
    before = 0
    after = 0
    unknown = 0
    for chunk in chunks:
        year = _year_from_chunk(chunk)
        if year is None:
            unknown += 1
        elif year <= args.split_year:
            before += 1
        else:
            after += 1
    result = {
        "ok": True,
        "task": args.task,
        "split_year": args.split_year,
        "chunks_before_or_equal_split": before,
        "chunks_after_split": after,
        "chunks_without_year": unknown,
        "note": "MVP scaffold: use these splits to build G_<=t and score future edges.",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
