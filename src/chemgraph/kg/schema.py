"""Pydantic schemas for the literature knowledge-graph workflow.

Identifiers in this module are content-derived. Reprocessing the same source
therefore produces the same record, evidence, condition, and measurement IDs.
Run timestamps belong in the artifact manifest rather than scientific rows.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from chemgraph.kg.ontology import NODE_TYPES, RELATION_TYPES


SCHEMA_VERSION = "literature_kg_v2"


def _stable_id(prefix: str, *parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


class EvidenceSpan(BaseModel):
    """A source-backed text span used as provenance for extracted facts."""

    evidence_id: str = ""
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
    extraction_time: Optional[str] = None

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("EvidenceSpan.text must not be empty.")
        return value

    @model_validator(mode="after")
    def fill_stable_id_and_validate_offsets(self):
        if self.start_char is not None and self.start_char < 0:
            raise ValueError("EvidenceSpan.start_char must be non-negative.")
        if (
            self.start_char is not None
            and self.end_char is not None
            and self.end_char < self.start_char
        ):
            raise ValueError("EvidenceSpan.end_char must be >= start_char.")
        if not self.evidence_id:
            self.evidence_id = _stable_id(
                "span",
                self.paper_id,
                self.chunk_id,
                self.start_char,
                self.end_char,
                self.text,
            )
        return self


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

    measurement_id: str = ""
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

    @model_validator(mode="after")
    def fill_stable_id(self):
        if not self.measurement_id:
            self.measurement_id = _stable_id(
                "meas",
                self.quantity,
                self.value,
                self.unit,
                self.raw_value,
                self.uncertainty,
                self.condition_id,
                self.evidence_span_id,
                self.attributes,
            )
        return self


class ReactionCondition(BaseModel):
    """Reaction conditions associated with one or more measurements."""

    condition_id: str = ""
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

    @model_validator(mode="after")
    def fill_stable_id(self):
        if not self.condition_id:
            self.condition_id = _stable_id(
                "cond",
                self.temperature,
                self.temperature_unit,
                self.pressure,
                self.pressure_unit,
                self.feed_composition,
                self.h2_co2_ratio,
                self.ghsv,
                self.whsv,
                self.catalyst_mass,
                self.reactor_type,
                self.time_on_stream,
                self.time_on_stream_unit,
                self.pretreatment,
                self.evidence_span_id,
            )
        return self


class SynthesisStep(BaseModel):
    """One ordered step in a catalyst synthesis recipe."""

    step_id: str = ""
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

    @model_validator(mode="after")
    def fill_stable_id(self):
        if not self.step_id:
            self.step_id = _stable_id(
                "step",
                self.order,
                self.operation,
                self.material,
                self.solvent,
                self.temperature,
                self.duration,
                self.atmosphere,
                self.evidence_span_id,
            )
        return self


class CatalystRecord(BaseModel):
    """Structured extraction result for one catalyst system in one paper/chunk."""

    record_id: str = ""
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
    field_evidence_ids: dict[str, list[str]] = Field(default_factory=dict)
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
    def fill_canonical_name_and_stable_id(self):
        if not self.canonical_catalyst_name:
            self.canonical_catalyst_name = self.catalyst_name
        if not self.record_id:
            self.record_id = _stable_id(
                "rec",
                self.paper_id,
                self.canonical_catalyst_name,
                self.reaction,
                self.active_metals,
                self.promoters,
                self.dopants,
                self.support,
                self.precursors,
                self.synthesis_method,
                [span.evidence_id for span in self.evidence_spans],
                [condition.condition_id for condition in self.reaction_conditions],
                [metric.measurement_id for metric in self.performance_metrics],
                [metric.measurement_id for metric in self.material_properties],
                self.characterization_methods,
                self.characterization_results,
                self.mechanistic_claims,
            )
        return self


class KGNode(BaseModel):
    """Typed attributed node for the literature KG."""

    node_id: str
    node_type: str
    name: str
    canonical_name: Optional[str] = None
    aliases: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)
    source_paper_ids: list[str] = Field(default_factory=list)
    source_count: int = 1
    confidence: float = 0.5
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    @field_validator("confidence")
    @classmethod
    def confidence_in_unit_interval(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")
        return value

    @field_validator("node_type")
    @classmethod
    def node_type_must_be_known(cls, value: str) -> str:
        if value not in NODE_TYPES:
            raise ValueError(f"Unknown KG node type: {value}")
        return value

    @model_validator(mode="after")
    def normalize_source_count(self):
        self.source_paper_ids = sorted(set(self.source_paper_ids))
        self.source_count = len(self.source_paper_ids) or self.source_count
        return self


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
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

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

    @field_validator("relation")
    @classmethod
    def relation_must_be_known(cls, value: str) -> str:
        if value not in RELATION_TYPES:
            raise ValueError(f"Unknown KG relation: {value}")
        return value


class HypothesisCard(BaseModel):
    """Evidence-backed hypothesis emitted by the insight workflow."""

    hypothesis_id: str = ""
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
    created_at: Optional[str] = None

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
        if not self.hypothesis_id:
            self.hypothesis_id = _stable_id(
                "hyp",
                self.claim,
                self.hypothesis_type,
                self.supporting_edge_ids,
                self.counter_evidence_ids,
            )
        return self
