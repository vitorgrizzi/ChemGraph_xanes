#!/usr/bin/env python
"""Ingest papers into literature KG chunk JSONL."""

from __future__ import annotations

import argparse
import json

from chemgraph.kg.ingest import ingest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input paper file or directory.")
    parser.add_argument("--out", required=True, help="Output chunk JSONL path.")
    parser.add_argument("--chunk-size", type=int, default=1500)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    args = parser.parse_args()

    chunks = ingest_path(
        args.input,
        out=args.out,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(json.dumps({"ok": True, "out": args.out, "n_chunks": len(chunks)}, indent=2))


if __name__ == "__main__":
    main()
