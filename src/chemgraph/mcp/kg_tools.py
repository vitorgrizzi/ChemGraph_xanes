"""FastMCP server exposing literature KG tools."""

from __future__ import annotations

from typing import Literal

from mcp.server.fastmcp import FastMCP

from chemgraph.kg.extract import (
    extract_records_from_chunks,
    load_extraction_llm,
    read_records_jsonl,
    write_records_jsonl,
)
from chemgraph.kg.hypotheses import suggest_hypotheses
from chemgraph.kg.ingest import ingest_path, read_chunks_jsonl
from chemgraph.kg.query import export_training_table, get_evidence, hybrid_query
from chemgraph.kg.store import build_kg
from chemgraph.kg.verify import verify_records
from chemgraph.kg.validation import validate_kg
from chemgraph.mcp.server_utils import run_mcp_server


mcp = FastMCP(
    name="ChemGraph Literature KG Tools",
    instructions="""
        You expose provenance-first literature knowledge-graph tools.
        Use these tools to ingest papers, extract catalyst records, build
        evidence-backed graphs, answer hybrid graph/RAG questions, retrieve
        evidence spans, suggest hypotheses, export observation-level tables,
        and validate artifact integrity after builds.
        Do not launch computations or experiments from hypotheses without
        explicit human approval.
    """,
)


@mcp.tool(
    name="kg_ingest_papers",
    description="Ingest PDF/text/JSONL papers into chunk JSONL.",
)
def kg_ingest_papers(
    input_path: str,
    out_path: str,
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
) -> dict:
    chunks = ingest_path(
        input_path,
        out=out_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return {"ok": True, "out_path": out_path, "n_chunks": len(chunks)}


@mcp.tool(
    name="kg_extract_records",
    description="Extract CatalystRecord JSONL with a selected model or deterministic fallback.",
)
def kg_extract_records(
    chunks_path: str,
    out_path: str,
    model: str = "deterministic",
    retries: int = 1,
) -> dict:
    chunks = read_chunks_jsonl(chunks_path)
    llm = load_extraction_llm(model)
    records = extract_records_from_chunks(chunks, llm=llm, retries=retries)
    write_records_jsonl(records, out_path)
    return {
        "ok": True,
        "out_path": out_path,
        "n_records": len(records),
        "model": model,
    }


@mcp.tool(
    name="kg_verify_records",
    description="Verify grounding and write only accepted CatalystRecord objects.",
)
def kg_verify_records(records_path: str, verified_out: str) -> dict:
    records = read_records_jsonl(records_path)
    results = verify_records(records)
    accepted = [result.record for result in results if result.accepted]
    write_records_jsonl(accepted, verified_out)
    return {
        "ok": True,
        "verified_records_path": verified_out,
        "n_input": len(records),
        "n_accepted": len(accepted),
        "issues": [
            issue.model_dump(mode="json")
            for result in results
            for issue in result.issues
        ],
    }


@mcp.tool(
    name="kg_build_graph",
    description="Build nodes, edges, evidence SQLite, and graph JSON from CatalystRecord JSONL.",
)
def kg_build_graph(records_path: str, kg_dir: str) -> dict:
    records = read_records_jsonl(records_path)
    return build_kg(records, kg_dir)


@mcp.tool(
    name="kg_hybrid_query",
    description="Run a hybrid KG query with compact model-facing output by default.",
)
def kg_hybrid_query(
    kg_dir: str,
    query: str,
    top_k: int = 10,
    embedding_model: str | None = None,
    response_mode: Literal["compact", "full"] = "compact",
) -> dict:
    return hybrid_query(
        kg_dir,
        query,
        top_k=top_k,
        embedding_model=embedding_model,
        response_mode=response_mode,
    )


@mcp.tool(
    name="kg_get_evidence",
    description="Fetch one evidence span by evidence_id.",
)
def kg_get_evidence(kg_dir: str, evidence_id: str) -> dict:
    return get_evidence(kg_dir, evidence_id)


@mcp.tool(
    name="kg_suggest_hypotheses",
    description="Generate evidence-backed catalyst hypothesis cards.",
)
def kg_suggest_hypotheses(kg_dir: str, goal: str, top_k: int = 5) -> dict:
    return suggest_hypotheses(kg_dir, goal=goal, top_k=top_k)


@mcp.tool(
    name="kg_export_training_table",
    description="Export a CSV training table from a built KG.",
)
def kg_export_training_table(
    kg_dir: str,
    out_path: str,
    target_quantity: str = "methanol_selectivity",
) -> dict:
    return export_training_table(
        kg_dir,
        out_path,
        target_quantity=target_quantity,
    )


@mcp.tool(
    name="kg_validate_graph",
    description="Validate KG artifact hashes, references, and observation linkage.",
)
def kg_validate_graph(kg_dir: str, verify_hashes: bool = True) -> dict:
    return validate_kg(kg_dir, verify_hashes=verify_hashes)


if __name__ == "__main__":
    run_mcp_server(mcp, default_port=9011)
