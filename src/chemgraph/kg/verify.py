"""Evidence and range checks for extracted catalyst records."""

from __future__ import annotations

from pydantic import BaseModel, Field

from chemgraph.kg.ontology import PERCENT_QUANTITIES
from chemgraph.kg.schema import CatalystRecord, Measurement


class VerificationIssue(BaseModel):
    record_id: str
    field: str
    message: str
    severity: str = "warning"


class VerificationResult(BaseModel):
    record: CatalystRecord
    accepted: bool
    confidence: float
    issues: list[VerificationIssue] = Field(default_factory=list)


def _metric_issues(record: CatalystRecord, metric: Measurement) -> list[VerificationIssue]:
    issues: list[VerificationIssue] = []
    if metric.evidence_span_id is None:
        issues.append(
            VerificationIssue(
                record_id=record.record_id,
                field=f"performance_metrics.{metric.measurement_id}",
                message="Measurement is missing evidence_span_id.",
                severity="error",
            )
        )

    quantity = metric.quantity.lower()
    unit = (metric.unit or "").lower()
    if metric.value is not None and quantity in PERCENT_QUANTITIES | {"conversion", "selectivity"}:
        if unit in {"percent", "%", ""} and not 0.0 <= metric.value <= 100.0:
            issues.append(
                VerificationIssue(
                    record_id=record.record_id,
                    field=f"performance_metrics.{metric.measurement_id}",
                    message="Percent-like metric must be between 0 and 100.",
                    severity="error",
                )
            )
        if unit == "dimensionless" and not 0.0 <= metric.value <= 1.0:
            issues.append(
                VerificationIssue(
                    record_id=record.record_id,
                    field=f"performance_metrics.{metric.measurement_id}",
                    message="Dimensionless selectivity/conversion must be between 0 and 1.",
                    severity="error",
                )
            )
    return issues


def verify_record(record: CatalystRecord) -> VerificationResult:
    """Verify provenance and deterministic physical/unit constraints."""
    issues: list[VerificationIssue] = []
    evidence_ids = {span.evidence_id for span in record.evidence_spans}

    if not evidence_ids:
        issues.append(
            VerificationIssue(
                record_id=record.record_id,
                field="evidence_spans",
                message="Record has no evidence spans.",
                severity="error",
            )
        )

    for metric in record.performance_metrics:
        issues.extend(_metric_issues(record, metric))
        if metric.evidence_span_id and metric.evidence_span_id not in evidence_ids:
            issues.append(
                VerificationIssue(
                    record_id=record.record_id,
                    field=f"performance_metrics.{metric.measurement_id}",
                    message="Measurement evidence_span_id is not present in record evidence.",
                    severity="error",
                )
            )

    for condition in record.reaction_conditions:
        if (
            condition.temperature is not None
            or condition.pressure is not None
            or condition.feed_composition is not None
        ) and condition.evidence_span_id is None:
            issues.append(
                VerificationIssue(
                    record_id=record.record_id,
                    field=f"reaction_conditions.{condition.condition_id}",
                    message="Non-empty reaction condition is missing evidence_span_id.",
                    severity="error",
                )
            )

    n_errors = sum(issue.severity == "error" for issue in issues)
    confidence = max(0.0, record.confidence - 0.15 * n_errors - 0.05 * (len(issues) - n_errors))
    accepted = n_errors == 0
    data = record.model_dump()
    data["confidence"] = confidence
    return VerificationResult(
        record=CatalystRecord.model_validate(data),
        accepted=accepted,
        confidence=confidence,
        issues=issues,
    )


def verify_records(records: list[CatalystRecord]) -> list[VerificationResult]:
    return [verify_record(record) for record in records]
