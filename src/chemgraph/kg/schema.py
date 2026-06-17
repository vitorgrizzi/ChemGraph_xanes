"""Pydantic schemas for the literature knowledge-graph workflow."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


class EvidenceSpan(BaseModel):
    """A source-backed text span used as provenance for extracted facts."""

    evidence_id: str = Field(default_factory=lambda: _new_id("span"))
    paper_id: str
    chunk_id: str
    page: Optional[int] = None
    section: Optional[str] = None
    text: str
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    source_path: Optional[str] = None
    doi: Optional[str] = None
    extraction_model: Optional[str] = None
    extraction_time: str = Field(default_factory=_utc_now_iso)

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("EvidenceSpan.text must not be empty.")
        return value


class PaperChunk(BaseModel):
    """Chunk-level representation produced by ingestion."""

    paper_id: str
    chunk_id: str
    text: str
    page: Optional[int] = None
    section: Optional[str] = None
    source_path: Optional[str] = None
    doi: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def chunk_text_must_not_be_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("PaperChunk.text must not be empty.")
        return value


class Measurement(BaseModel):
    """A numerical or categorical catalysis measurement with provenance."""

    measurement_id: str = Field(default_factory=lambda: _new_id("meas"))
    quantity: str
    value: Optional[float] = None
    unit: Optional[str] = None
    raw_value: Optional[str] = None
    uncertainty: Optional[float] = None
    condition_id: Optional[str] = None
    evidence_span_id: Optional[str] = None
    confidence: float = 0.5
    attributes: dict[str, Any] = Field(default_factory=dict)

    @field_validator("confidence")
    @classmethod
    def confidence_in_unit_interval(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")
        return value


class ReactionCondition(BaseModel):
    """Reaction conditions associated with one or more measurements."""

    condition_id: str = Field(default_factory=lambda: _new_id("cond"))
    temperature: Optional[float] = None
    temperature_unit: Optional[str] = "degC"
    pressure: Optional[float] = None
    pressure_unit: Optional[str] = "bar"
    feed_composition: Optional[str] = None
    h2_co2_ratio: Optional[float] = None
    ghsv: Optional[float] = None
    whsv: Optional[float] = None
    catalyst_mass: Optional[float] = None
    reactor_type: Optional[str] = None
    time_on_stream: Optional[float] = None
    time_on_stream_unit: Optional[str] = "h"
    pretreatment: Optional[str] = None
    evidence_span_id: Optional[str] = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class SynthesisStep(BaseModel):
    """One ordered step in a catalyst synthesis recipe."""

    step_id: str = Field(default_factory=lambda: _new_id("step"))
    order: int = 0
    operation: str
    material: Optional[str] = None
    solvent: Optional[str] = None
    temperature: Optional[float] = None
    temperature_unit: Optional[str] = "degC"
    duration: Optional[float] = None
    duration_unit: Optional[str] = "h"
    atmosphere: Optional[str] = None
    evidence_span_id: Optional[str] = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class CatalystRecord(BaseModel):
    """Structured extraction result for one catalyst system in one paper/chunk."""

    record_id: str = Field(default_factory=lambda: _new_id("rec"))
    paper_id: str
    catalyst_name: str
    canonical_catalyst_name: Optional[str] = None
    reaction: Optional[str] = None
    active_metals: list[str] = Field(default_factory=list)
    promoters: list[str] = Field(default_factory=list)
    dopants: list[str] = Field(default_factory=list)
    support: Optional[str] = None
    precursors: list[str] = Field(default_factory=list)
    synthesis_method: Optional[str] = None
    synthesis_steps: list[SynthesisStep] = Field(default_factory=list)
    reaction_conditions: list[ReactionCondition] = Field(default_factory=list)
    performance_metrics: list[Measurement] = Field(default_factory=list)
    characterization_methods: list[str] = Field(default_factory=list)
    characterization_results: list[str] = Field(default_factory=list)
    material_properties: list[Measurement] = Field(default_factory=list)
    mechanistic_claims: list[str] = Field(default_factory=list)
    evidence_spans: list[EvidenceSpan] = Field(default_factory=list)
    confidence: float = 0.5
    extractor_version: str = "literature_kg_mvp"
    attributes: dict[str, Any] = Field(default_factory=dict)

    @field_validator("confidence")
    @classmethod
    def confidence_in_unit_interval(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")
        return value

    @model_validator(mode="after")
    def fill_canonical_name(self):
        if not self.canonical_catalyst_name:
            self.canonical_catalyst_name = self.catalyst_name
        return self


class KGNode(BaseModel):
    """Typed attributed node for the literature KG."""

    node_id: str
    node_type: str
    name: str
    canonical_name: Optional[str] = None
    aliases: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)
    source_count: int = 1
    confidence: float = 0.5
    created_at: str = Field(default_factory=_utc_now_iso)
    updated_at: str = Field(default_factory=_utc_now_iso)

    @field_validator("confidence")
    @classmethod
    def confidence_in_unit_interval(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")
        return value


class KGEdge(BaseModel):
    """Typed attributed edge with required confidence and provenance IDs."""

    edge_id: str
    source_node_id: str
    relation: str
    target_node_id: str
    attributes: dict[str, Any] = Field(default_factory=dict)
    evidence_ids: list[str]
    confidence: float
    extractor_version: str = "literature_kg_mvp"
    created_at: str = Field(default_factory=_utc_now_iso)
    updated_at: str = Field(default_factory=_utc_now_iso)

    @field_validator("confidence")
    @classmethod
    def confidence_in_unit_interval(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")
        return value

    @field_validator("evidence_ids")
    @classmethod
    def evidence_ids_required(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("KGEdge.evidence_ids must not be empty.")
        return value


class HypothesisCard(BaseModel):
    """Evidence-backed hypothesis emitted by the insight workflow."""

    hypothesis_id: str = Field(default_factory=lambda: _new_id("hyp"))
    claim: str
    hypothesis_type: Literal["missing_link", "trend", "contradiction", "gap"] = (
        "missing_link"
    )
    novelty: float = 0.5
    plausibility: float = 0.5
    expected_utility: float = 0.5
    risk: float = 0.5
    cost: float = 0.5
    score: Optional[float] = None
    supporting_paths: list[dict[str, Any]] = Field(default_factory=list)
    supporting_edge_ids: list[str] = Field(default_factory=list)
    counter_evidence_ids: list[str] = Field(default_factory=list)
    suggested_validation: list[str] = Field(default_factory=list)
    structured_tasks: list[dict[str, Any]] = Field(default_factory=list)
    status: Literal["proposed", "critic_approved", "rejected", "validated"] = (
        "proposed"
    )
    created_at: str = Field(default_factory=_utc_now_iso)

    @field_validator("novelty", "plausibility", "expected_utility", "risk", "cost")
    @classmethod
    def scores_in_unit_interval(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("hypothesis scores must be between 0 and 1.")
        return value

    @model_validator(mode="after")
    def fill_score(self):
        if self.score is None:
            self.score = (
                0.30 * self.plausibility
                + 0.25 * self.expected_utility
                + 0.25 * self.novelty
                - 0.10 * self.risk
                - 0.10 * self.cost
            )
        return self
