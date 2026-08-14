#!/usr/bin/env python
"""Extract CatalystRecord JSONL from chunk JSONL."""

from __future__ import annotations

import argparse
import json
from chemgraph.kg.extract import (
    extract_records_from_chunks,
    load_extraction_llm,
    write_records_jsonl,
)
from chemgraph.kg.ingest import read_chunks_jsonl
from chemgraph.kg.profiles import profile_name_for_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", required=True, help="Input chunk JSONL path.")
    parser.add_argument("--out", required=True, help="Output extraction JSONL path.")
    parser.add_argument(
        "--model",
        default="deterministic",
        help=(
            "LLM model name, deterministic for generic regex extraction, or "
            "co2_methanol_regex for the explicit pilot profile."
        ),
    )
    parser.add_argument(
        "--profile",
        default="general",
        help="Extraction vocabulary profile (default: general).",
    )
    parser.add_argument(
        "--profiles-config",
        help="Optional YAML file defining extraction profiles.",
    )
    parser.add_argument("--retries", type=int, default=1)
    args = parser.parse_args()

    profile = profile_name_for_model(args.model, args.profile)
    llm = load_extraction_llm(args.model)
    chunks = read_chunks_jsonl(args.chunks)
    records = extract_records_from_chunks(
        chunks,
        llm=llm,
        retries=args.retries,
        profile=profile,
        profiles_config=args.profiles_config,
    )
    write_records_jsonl(records, args.out)
    print(
        json.dumps(
            {
                "ok": True,
                "chunks": args.chunks,
                "out": args.out,
                "model": args.model,
                "profile": profile,
                "n_records": len(records),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
