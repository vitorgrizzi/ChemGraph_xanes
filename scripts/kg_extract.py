#!/usr/bin/env python
"""Extract CatalystRecord JSONL from chunk JSONL."""

from __future__ import annotations

import argparse
import json
import os

from chemgraph.kg.extract import extract_records_from_chunks, write_records_jsonl
from chemgraph.kg.ingest import read_chunks_jsonl


def _load_llm(model: str):
    if model in {"deterministic", "regex", "offline", "none"}:
        return None
    from chemgraph.models.openai import load_openai_model

    return load_openai_model(
        model_name=model,
        temperature=0.0,
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", required=True, help="Input chunk JSONL path.")
    parser.add_argument("--out", required=True, help="Output extraction JSONL path.")
    parser.add_argument(
        "--model",
        default="deterministic",
        help="LLM model name, or deterministic/regex/offline for local extraction.",
    )
    parser.add_argument("--retries", type=int, default=1)
    args = parser.parse_args()

    llm = _load_llm(args.model)
    chunks = read_chunks_jsonl(args.chunks)
    records = extract_records_from_chunks(chunks, llm=llm, retries=args.retries)
    write_records_jsonl(records, args.out)
    print(
        json.dumps(
            {
                "ok": True,
                "chunks": args.chunks,
                "out": args.out,
                "model": args.model,
                "n_records": len(records),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
