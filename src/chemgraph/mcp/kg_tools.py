"""FastMCP server exposing literature KG tools."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from chemgraph.kg.extract import (
    extract_records_from_chunks,
    read_records_jsonl,
    write_records_jsonl,
)
from chemgraph.kg.hypotheses import suggest_hypotheses
from chemgraph.kg.ingest import ingest_path, read_chunks_jsonl
from chemgraph.kg.query import export_training_table, get_evidence, hybrid_query
from chemgraph.kg.store import build_kg
from chemgraph.mcp.server_utils import run_mcp_server


mcp = FastMCP(
    name="ChemGraph Literature KG Tools",
    instructions="""
        You expose provenance-first literature knowledge-graph tools.
        Use these tools to ingest papers, extract catalyst records, build
        evidence-backed graphs, answer hybrid graph/RAG questions, retrieve
        evidence spans, suggest hypotheses, and export ML-ready tables.
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
    description="Extract CatalystRecord JSONL from chunk JSONL with the deterministic MVP extractor.",
)
def kg_extract_records(chunks_path: str, out_path: str) -> dict:
    chunks = read_chunks_jsonl(chunks_path)
    records = extract_records_from_chunks(chunks)
    write_records_jsonl(records, out_path)
    return {"ok": True, "out_path": out_path, "n_records": len(records)}


@mcp.tool(
    name="kg_build_graph",
    description="Build nodes, edges, evidence SQLite, and graph JSON from CatalystRecord JSONL.",
)
def kg_build_graph(records_path: str, kg_dir: str) -> dict:
    records = read_records_jsonl(records_path)
    return build_kg(records, kg_dir)


@mcp.tool(
    name="kg_hybrid_query",
    description="Run a hybrid graph and evidence-span query over a built KG.",
)
def kg_hybrid_query(kg_dir: str, query: str, top_k: int = 10) -> dict:
    return hybrid_query(kg_dir, query, top_k=top_k)


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


if __name__ == "__main__":
    run_mcp_server(mcp, default_port=9011)
