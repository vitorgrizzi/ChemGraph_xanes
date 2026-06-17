"""Deterministic literature KG pipeline used by scripts and agent tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from chemgraph.kg.agents import OrchestratorAgent


class LiteratureKGWorkflowConfig(BaseModel):
    input_path: str = Field(description="Paper file or directory to ingest.")
    work_dir: str = Field(description="Directory where chunks, records, and KG are written.")
    query: str | None = None
    goal: str | None = None


def run_literature_kg_workflow(
    input_path: str,
    work_dir: str,
    *,
    query: str | None = None,
    goal: str | None = None,
) -> dict[str, Any]:
    """Run ingest -> extract -> verify -> normalize/build -> query/insight."""
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    return OrchestratorAgent().run(
        input_path=input_path,
        work_dir=work_dir,
        query=query,
        goal=goal,
    )
