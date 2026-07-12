#!/usr/bin/env python
"""Verify extracted records and write only evidence-grounded records."""

from __future__ import annotations

import argparse
import json

from chemgraph.kg.extract import read_records_jsonl, write_records_jsonl
from chemgraph.kg.verify import verify_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    records = read_records_jsonl(args.records)
    results = verify_records(records)
    accepted = [result.record for result in results if result.accepted]
    write_records_jsonl(accepted, args.out)
    report = {
        "ok": True,
        "n_input": len(records),
        "n_accepted": len(accepted),
        "n_rejected": len(records) - len(accepted),
        "issues": [
            issue.model_dump(mode="json")
            for result in results
            for issue in result.issues
        ],
        "out": args.out,
    }
    print(json.dumps(report, indent=2))
    if len(accepted) != len(records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
