"""Agent wrappers and LangChain tools for the literature KG workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from chemgraph.kg.extract import (
    extract_records_from_chunks,
    load_extraction_llm,
    read_records_jsonl,
    write_records_jsonl,
)
from chemgraph.kg.hypotheses import score_hypothesis, suggest_hypotheses
from chemgraph.kg.ingest import ingest_path, read_chunks_jsonl
from chemgraph.kg.query import (
    evidence_search,
    export_training_table,
    get_evidence,
    graph_query,
    hybrid_query,
    semantic_search,
)
from chemgraph.kg.schema import HypothesisCard
from chemgraph.kg.store import build_kg
from chemgraph.kg.verify import verify_records
from chemgraph.kg.validation import validate_kg


class IngestionAgent:
    def run(self, input_path: str, chunks_out: str, **kwargs) -> dict[str, Any]:
        chunks = ingest_path(input_path, out=chunks_out, **kwargs)
        return {"ok": True, "chunks_path": chunks_out, "n_chunks": len(chunks)}


class ExtractionAgent:
    def run(
        self,
        chunks_path: str,
        records_out: str,
        llm=None,
        retries: int = 1,
    ) -> dict[str, Any]:
        chunks = read_chunks_jsonl(chunks_path)
        records = extract_records_from_chunks(chunks, llm=llm, retries=retries)
        write_records_jsonl(records, records_out)
        return {"ok": True, "records_path": records_out, "n_records": len(records)}


class VerifierAgent:
    def run(self, records_path: str, verified_out: str) -> dict[str, Any]:
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


class ResolverAgent:
    def run(self, records_path: str) -> dict[str, Any]:
        return {"ok": True, "records_path": records_path, "message": "Normalization is applied during graph build."}


class GraphBuilderAgent:
    def run(self, records_path: str, kg_dir: str) -> dict[str, Any]:
        records = read_records_jsonl(records_path)
        return build_kg(records, kg_dir)


class QueryAgent:
    def retrieval(
        self,
        kg_dir: str,
        query: str,
        top_k: int = 5,
        embedding_model: str | None = None,
    ) -> dict[str, Any]:
        return evidence_search(
            kg_dir,
            query,
            top_k=top_k,
            embedding_model=embedding_model,
        )

    def semantic(self, kg_dir: str, query: str, top_k: int = 5) -> dict[str, Any]:
        """Backward-compatible alias for evidence retrieval."""
        return semantic_search(kg_dir, query, top_k=top_k)

    def graph(self, kg_dir: str, **kwargs) -> dict[str, Any]:
        return graph_query(kg_dir, **kwargs)

    def hybrid(self, kg_dir: str, query: str, top_k: int = 10) -> dict[str, Any]:
        return hybrid_query(kg_dir, query, top_k=top_k)


class InsightAgent:
    def run(self, kg_dir: str, goal: str, top_k: int = 5) -> dict[str, Any]:
        return suggest_hypotheses(kg_dir, goal=goal, top_k=top_k)


class CriticAgent:
    def run(self, hypothesis: dict[str, Any]) -> dict[str, Any]:
        card = HypothesisCard.model_validate(hypothesis)
        independent_papers = {
            path.get("paper_id")
            for path in card.supporting_paths
            if path.get("paper_id")
        }
        approved = bool(
            card.supporting_edge_ids
            and card.suggested_validation
            and len(independent_papers) >= 2
        )
        concerns = []
        if not card.supporting_edge_ids:
            concerns.append("No supporting KG edge IDs are attached.")
        if not card.counter_evidence_ids:
            concerns.append("No counter-evidence was found; treat this as an evidence gap, not proof of safety.")
        if len(independent_papers) < 2:
            concerns.append("Fewer than two independent papers support the candidate.")
        return {
            "approved": approved,
            "major_concerns": concerns,
            "missing_evidence": [] if approved else ["independent_paper_support"],
            "suggested_validation": card.suggested_validation,
            "revised_claim": card.claim,
        }


class SimulationPlannerAgent:
    def run(self, hypothesis: dict[str, Any]) -> dict[str, Any]:
        card = HypothesisCard.model_validate(hypothesis)
        return {
            "ok": True,
            "requires_human_approval": True,
            "tasks": card.structured_tasks,
        }


class OrchestratorAgent:
    """Small deterministic orchestrator used by scripts and docs demos."""

    def run(
        self,
        input_path: str,
        work_dir: str,
        *,
        query: str | None = None,
        goal: str | None = None,
        extraction_model: str = "deterministic",
        extraction_retries: int = 1,
    ) -> dict[str, Any]:
        work = Path(work_dir)
        chunks_path = work / "chunks.jsonl"
        records_path = work / "extractions.jsonl"
        verified_path = work / "verified_records.jsonl"
        kg_dir = work / "graph"

        ingestion = IngestionAgent().run(input_path, str(chunks_path))
        extraction_llm = load_extraction_llm(extraction_model)
        extraction = ExtractionAgent().run(
            str(chunks_path),
            str(records_path),
            llm=extraction_llm,
            retries=extraction_retries,
        )
        extraction["model"] = extraction_model
        verification = VerifierAgent().run(str(records_path), str(verified_path))
        graph = GraphBuilderAgent().run(str(verified_path), str(kg_dir))
        output: dict[str, Any] = {
            "ok": True,
            "ingestion": ingestion,
            "extraction": extraction,
            "verification": verification,
            "graph": graph,
        }
        if query:
            output["query"] = QueryAgent().hybrid(str(kg_dir), query)
        if goal:
            output["hypotheses"] = InsightAgent().run(str(kg_dir), goal)
        return output


class KGIngestInput(BaseModel):
    input_path: str = Field(description="File or directory containing PDF/text/JSONL papers.")
    out_path: str = Field(description="JSONL path for produced chunks.")
    chunk_size: int = Field(default=1500)
    chunk_overlap: int = Field(default=200)


class KGExtractInput(BaseModel):
    chunks_path: str = Field(description="Path to chunk JSONL produced by kg_ingest_papers.")
    out_path: str = Field(description="JSONL path for extracted CatalystRecord objects.")
    model: str = Field(default="deterministic")
    retries: int = Field(default=1, ge=0, le=5)


class KGBuildInput(BaseModel):
    records_path: str = Field(description="Path to CatalystRecord JSONL.")
    kg_dir: str = Field(description="Output directory for nodes/edges/evidence store.")


class KGVerifyInput(BaseModel):
    records_path: str = Field(description="Path to extracted CatalystRecord JSONL.")
    verified_out: str = Field(description="Output path for accepted records.")


class KGQueryInput(BaseModel):
    kg_dir: str = Field(description="Directory containing a built literature KG.")
    query: str = Field(description="Natural-language graph/RAG question.")
    top_k: int = Field(default=10)
    embedding_model: str | None = Field(
        default=None,
        description="Optional sentence-transformers model for vector evidence retrieval.",
    )


class KGEvidenceInput(BaseModel):
    kg_dir: str
    evidence_id: str


class KGHypothesisInput(BaseModel):
    kg_dir: str
    goal: str
    top_k: int = Field(default=5)


class KGExportInput(BaseModel):
    kg_dir: str
    out_path: str
    target_quantity: str = Field(default="methanol_selectivity")


class KGValidateInput(BaseModel):
    kg_dir: str
    verify_hashes: bool = True


@tool(args_schema=KGIngestInput)
def kg_ingest_papers(
    input_path: str,
    out_path: str,
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
) -> dict:
    """Ingest papers into provenance-preserving text chunks."""
    chunks = ingest_path(
        input_path,
        out=out_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return {"ok": True, "out_path": out_path, "n_chunks": len(chunks)}


@tool(args_schema=KGExtractInput)
def kg_extract_records(
    chunks_path: str,
    out_path: str,
    model: str = "deterministic",
    retries: int = 1,
) -> dict:
    """Extract CatalystRecord JSONL with a selected model or offline fallback."""
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


@tool(args_schema=KGBuildInput)
def kg_build_graph(records_path: str, kg_dir: str) -> dict:
    """Build the literature KG from CatalystRecord JSONL."""
    records = read_records_jsonl(records_path)
    return build_kg(records, kg_dir)


@tool(args_schema=KGVerifyInput)
def kg_verify_records(records_path: str, verified_out: str) -> dict:
    """Verify grounding and write only accepted records."""
    return VerifierAgent().run(records_path, verified_out)


@tool(args_schema=KGQueryInput)
def kg_hybrid_query(
    kg_dir: str,
    query: str,
    top_k: int = 10,
    embedding_model: str | None = None,
) -> dict:
    """Ask a hybrid graph + evidence-retrieval question."""
    return hybrid_query(
        kg_dir,
        query,
        top_k=top_k,
        embedding_model=embedding_model,
    )


@tool(args_schema=KGEvidenceInput)
def kg_get_evidence(kg_dir: str, evidence_id: str) -> dict:
    """Fetch one evidence span by ID."""
    return get_evidence(kg_dir, evidence_id)


@tool(args_schema=KGHypothesisInput)
def kg_suggest_hypotheses(kg_dir: str, goal: str, top_k: int = 5) -> dict:
    """Suggest evidence-backed catalyst hypotheses."""
    return suggest_hypotheses(kg_dir, goal=goal, top_k=top_k)


@tool(args_schema=KGExportInput)
def kg_export_training_table(
    kg_dir: str,
    out_path: str,
    target_quantity: str = "methanol_selectivity",
) -> dict:
    """Export a CSV table for ML modeling from the KG."""
    return export_training_table(kg_dir, out_path, target_quantity=target_quantity)


@tool(args_schema=KGValidateInput)
def kg_validate_graph(kg_dir: str, verify_hashes: bool = True) -> dict:
    """Validate artifact hashes, graph references, and observation linkage."""
    return validate_kg(kg_dir, verify_hashes=verify_hashes)


def kg_langchain_tools() -> list:
    """Return literature KG tools for LangGraph ReAct workflows."""
    return [
        kg_ingest_papers,
        kg_extract_records,
        kg_verify_records,
        kg_build_graph,
        kg_hybrid_query,
        kg_get_evidence,
        kg_suggest_hypotheses,
        kg_export_training_table,
        kg_validate_graph,
    ]
