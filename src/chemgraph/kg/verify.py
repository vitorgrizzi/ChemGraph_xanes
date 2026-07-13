"""Grounding, reference-integrity, and physical checks for KG records."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, Field

from chemgraph.kg.ontology import PERCENT_QUANTITIES
from chemgraph.kg.schema import CatalystRecord, Measurement, ReactionCondition


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


GROUNDED_FIELDS = (
    "catalyst_name",
    "reaction",
    "active_metals",
    "promoters",
    "dopants",
    "support",
    "precursors",
    "synthesis_method",
    "characterization_methods",
    "characterization_results",
    "mechanistic_claims",
)


def _issue(record: CatalystRecord, field: str, message: str, severity: str = "error"):
    return VerificationIssue(
        record_id=record.record_id,
        field=field,
        message=message,
        severity=severity,
    )


def _normalized_text(value: Any) -> str:
    text = str(value).lower().replace("co₂", "co2").replace("h₂", "h2")
    return re.sub(r"[^a-z0-9]+", "", text)


def _iter_field_values(value: Any) -> Iterable[Any]:
    if isinstance(value, list):
        yield from value
    elif value is not None and value != "":
        yield value


def _value_is_supported(field: str, value: Any, evidence_text: str) -> bool:
    needle = _normalized_text(value)
    haystack = _normalized_text(evidence_text)
    if not needle:
        return True
    if needle in haystack:
        return True
    if field == "reaction":
        if "co2hydrogenationtomethanol" in needle and "methanolsynthesis" in haystack:
            return True
        if "co2hydrogenation" in needle and "carbondioxidehydrogenation" in haystack:
            return True
        tokens = [
            token
            for token in ("co2", "hydrogenation", "methanol", "synthesis")
            if token in needle
        ]
        return bool(tokens) and sum(token in haystack for token in tokens) >= min(2, len(tokens))
    return False


def _number_is_supported(value: float, text: str) -> bool:
    candidates = {f"{value:g}", str(value)}
    return any(
        re.search(rf"(?<![\d.]){re.escape(candidate)}(?![\d.])", text)
        for candidate in candidates
    )


def _metric_condition_are_co_located(
    metric: Measurement,
    condition: ReactionCondition,
    evidence_text: str,
) -> bool:
    """Require metric and linked numerical conditions in one sentence."""
    condition_values = [
        value
        for value in (
            condition.temperature,
            condition.pressure,
            condition.h2_co2_ratio,
        )
        if value is not None
    ]
    if metric.value is None or not condition_values:
        return True
    quantity_token = metric.quantity.lower().replace("_", " ").split()[-1]
    segments = re.split(r"(?<=[.!?])\s+|\n+", evidence_text)
    return any(
        _number_is_supported(metric.value, segment)
        and quantity_token in segment.lower()
        and all(_number_is_supported(value, segment) for value in condition_values)
        for segment in segments
    )


def _metric_issues(
    record: CatalystRecord,
    metric: Measurement,
    evidence_by_id: dict[str, Any],
    conditions_by_id: dict[str, ReactionCondition],
    field_prefix: str = "performance_metrics",
) -> list[VerificationIssue]:
    issues: list[VerificationIssue] = []
    field = f"{field_prefix}.{metric.measurement_id}"
    if metric.evidence_span_id is None:
        issues.append(_issue(record, field, "Measurement is missing evidence_span_id."))
    elif metric.evidence_span_id not in evidence_by_id:
        issues.append(_issue(record, field, "Measurement evidence_span_id is not present in record evidence."))
    else:
        evidence_text = evidence_by_id[metric.evidence_span_id].text
        if metric.value is not None and not _number_is_supported(metric.value, evidence_text):
            issues.append(_issue(record, field, "Measurement value is not present in its evidence text."))
        if metric.uncertainty is not None and not _number_is_supported(
            metric.uncertainty, evidence_text
        ):
            issues.append(
                _issue(record, field, "Measurement uncertainty is not present in its evidence text.")
            )
        quantity_token = metric.quantity.lower().replace("_", " ").split()[-1]
        if quantity_token and quantity_token not in evidence_text.lower():
            issues.append(_issue(record, field, "Measurement quantity is not stated in its evidence text."))
        comparator = str(metric.attributes.get("comparator") or "=").lower()
        if comparator == "range":
            range_max = metric.attributes.get("range_max")
            if range_max is None or not _number_is_supported(float(range_max), evidence_text):
                issues.append(_issue(record, field, "Measurement range maximum is not present in evidence."))
            elif metric.value is not None and float(range_max) < metric.value:
                issues.append(_issue(record, field, "Measurement range maximum is below its minimum."))
        if comparator in {"above", "over", ">", "below", "under", "<", "~", "approximately", "about", "around"}:
            comparator_tokens = {
                "above": ("above", "over", ">"),
                "over": ("above", "over", ">"),
                ">": ("above", "over", ">"),
                "below": ("below", "under", "<"),
                "under": ("below", "under", "<"),
                "<": ("below", "under", "<"),
                "~": ("~", "approximately", "about", "around"),
                "approximately": ("~", "approximately", "about", "around"),
                "about": ("~", "approximately", "about", "around"),
                "around": ("~", "approximately", "about", "around"),
            }[comparator]
            if not any(token in evidence_text.lower() for token in comparator_tokens):
                issues.append(_issue(record, field, "Measurement comparator is not present in its evidence text."))

    if metric.condition_id is not None and metric.condition_id not in conditions_by_id:
        issues.append(_issue(record, field, "Measurement condition_id does not resolve within the record."))
    if conditions_by_id and metric.condition_id is None:
        issues.append(_issue(record, field, "Measurement is not linked to a reaction condition."))
    if metric.condition_id in conditions_by_id and metric.evidence_span_id in evidence_by_id:
        if not _metric_condition_are_co_located(
            metric,
            conditions_by_id[metric.condition_id],
            evidence_by_id[metric.evidence_span_id].text,
        ):
            issues.append(
                _issue(
                    record,
                    field,
                    "Measurement and linked numerical condition are not stated in the same sentence.",
                )
            )

    quantity = metric.quantity.lower()
    unit = (metric.unit or "").lower()
    if metric.value is not None and quantity in PERCENT_QUANTITIES | {"conversion", "selectivity"}:
        if unit in {"percent", "%", ""} and not 0.0 <= metric.value <= 100.0:
            issues.append(_issue(record, field, "Percent-like metric must be between 0 and 100."))
        if unit == "dimensionless" and not 0.0 <= metric.value <= 1.0:
            issues.append(_issue(record, field, "Dimensionless selectivity/conversion must be between 0 and 1."))
    return issues


def verify_record(record: CatalystRecord) -> VerificationResult:
    """Verify that a record is internally consistent and text-grounded."""
    issues: list[VerificationIssue] = []
    evidence_by_id = {span.evidence_id: span for span in record.evidence_spans}
    if len(evidence_by_id) != len(record.evidence_spans):
        issues.append(_issue(record, "evidence_spans", "Duplicate evidence IDs are present."))
    if not evidence_by_id:
        issues.append(_issue(record, "evidence_spans", "Record has no evidence spans."))

    for span in record.evidence_spans:
        if span.paper_id != record.paper_id:
            issues.append(_issue(record, "evidence_spans", "Evidence paper_id does not match the record."))
        if span.source_path and span.start_char is not None and span.end_char is not None:
            source = Path(span.source_path)
            if source.exists() and source.suffix.lower() in {".txt", ".md"}:
                source_text = source.read_text(encoding="utf-8")
                if source_text[span.start_char : span.end_char] != span.text:
                    issues.append(_issue(record, "evidence_spans", "Evidence offsets do not reproduce the source text."))

    for field in GROUNDED_FIELDS:
        value = getattr(record, field)
        values = list(_iter_field_values(value))
        if not values:
            continue
        evidence_ids = record.field_evidence_ids.get(field, [])
        if not evidence_ids:
            issues.append(_issue(record, field, "Field is missing field-level evidence IDs."))
            continue
        missing = [eid for eid in evidence_ids if eid not in evidence_by_id]
        if missing:
            issues.append(_issue(record, field, f"Field references missing evidence IDs: {missing}."))
            continue
        combined_text = "\n".join(evidence_by_id[eid].text for eid in evidence_ids)
        for item in values:
            if not _value_is_supported(field, item, combined_text):
                issues.append(_issue(record, field, f"Extracted value is not stated in evidence: {item!r}."))

    conditions_by_id = {
        condition.condition_id: condition for condition in record.reaction_conditions
    }
    for condition in record.reaction_conditions:
        field = f"reaction_conditions.{condition.condition_id}"
        if condition.evidence_span_id is None:
            issues.append(_issue(record, field, "Reaction condition is missing evidence_span_id."))
        elif condition.evidence_span_id not in evidence_by_id:
            issues.append(_issue(record, field, "Reaction-condition evidence_span_id is not present in record evidence."))
        else:
            text = evidence_by_id[condition.evidence_span_id].text
            for name, value in (
                ("temperature", condition.temperature),
                ("pressure", condition.pressure),
                ("h2_co2_ratio", condition.h2_co2_ratio),
            ):
                if value is not None and not _number_is_supported(value, text):
                    issues.append(_issue(record, field, f"Condition {name} is not present in its evidence text."))
        if condition.temperature is not None and condition.temperature_unit == "degC" and condition.temperature < -273.15:
            issues.append(_issue(record, field, "Temperature is below absolute zero."))
        if condition.pressure is not None and condition.pressure < 0:
            issues.append(_issue(record, field, "Pressure must be non-negative."))
        if condition.h2_co2_ratio is not None and condition.h2_co2_ratio <= 0:
            issues.append(_issue(record, field, "H2/CO2 ratio must be positive."))

    for metric in record.performance_metrics:
        issues.extend(_metric_issues(record, metric, evidence_by_id, conditions_by_id))
    for metric in record.material_properties:
        issues.extend(
            _metric_issues(
                record,
                metric,
                evidence_by_id,
                {},
                field_prefix="material_properties",
            )
        )
    for step in record.synthesis_steps:
        if not step.evidence_span_id or step.evidence_span_id not in evidence_by_id:
            issues.append(_issue(record, f"synthesis_steps.{step.step_id}", "Synthesis step has unresolved evidence."))

    n_errors = sum(issue.severity == "error" for issue in issues)
    n_warnings = len(issues) - n_errors
    confidence = max(0.0, record.confidence - 0.15 * n_errors - 0.05 * n_warnings)
    data = record.model_dump()
    data["confidence"] = confidence
    return VerificationResult(
        record=CatalystRecord.model_validate(data),
        accepted=n_errors == 0,
        confidence=confidence,
        issues=issues,
    )


def verify_records(records: list[CatalystRecord]) -> list[VerificationResult]:
    return [verify_record(record) for record in records]
