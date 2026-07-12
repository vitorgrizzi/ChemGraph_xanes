"""Gold-set evaluation for literature-KG extraction quality."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

from chemgraph.kg.normalize import normalize_records
from chemgraph.kg.schema import CatalystRecord
from chemgraph.kg.verify import verify_records


def _facts(record: CatalystRecord) -> dict[str, set[tuple[Any, ...]]]:
    key = (record.paper_id, record.canonical_catalyst_name or record.catalyst_name)
    facts: dict[str, set[tuple[Any, ...]]] = defaultdict(set)
    facts["catalyst"].add((*key, record.catalyst_name))
    if record.reaction:
        facts["reaction"].add((*key, record.reaction))
    if record.support:
        facts["support"].add((*key, record.support))
    for field in ("active_metals", "promoters", "dopants", "precursors"):
        for value in getattr(record, field):
            facts[field].add((*key, value))
    for condition in record.reaction_conditions:
        facts["conditions"].add(
            (
                *key,
                condition.temperature,
                condition.temperature_unit,
                condition.pressure,
                condition.pressure_unit,
                condition.h2_co2_ratio,
            )
        )
    for metric in record.performance_metrics:
        facts["metrics"].add(
            (
                *key,
                metric.quantity,
                metric.value,
                metric.unit,
                metric.attributes.get("comparator", "="),
            )
        )
        facts["metric_condition_links"].add(
            (*key, metric.quantity, metric.value, metric.condition_id)
        )
    return facts


def _aggregate(records: Iterable[CatalystRecord]) -> dict[str, set[tuple[Any, ...]]]:
    aggregate: dict[str, set[tuple[Any, ...]]] = defaultdict(set)
    for record in records:
        for category, values in _facts(record).items():
            aggregate[category].update(values)
    return aggregate


def evaluate_extractions(
    predicted: list[CatalystRecord],
    gold: list[CatalystRecord],
) -> dict[str, Any]:
    """Compute exact fact precision/recall plus grounding-gate diagnostics."""
    predicted_facts = _aggregate(normalize_records(predicted))
    gold_facts = _aggregate(normalize_records(gold))
    categories = sorted(set(predicted_facts) | set(gold_facts))
    per_category = {}
    total_tp = total_predicted = total_gold = 0
    for category in categories:
        predicted_values = predicted_facts[category]
        gold_values = gold_facts[category]
        true_positive = len(predicted_values & gold_values)
        precision = true_positive / len(predicted_values) if predicted_values else 0.0
        recall = true_positive / len(gold_values) if gold_values else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_category[category] = {
            "true_positive": true_positive,
            "predicted": len(predicted_values),
            "gold": len(gold_values),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        total_tp += true_positive
        total_predicted += len(predicted_values)
        total_gold += len(gold_values)

    precision = total_tp / total_predicted if total_predicted else 0.0
    recall = total_tp / total_gold if total_gold else 0.0
    verification = verify_records(predicted)
    issue_counts = Counter(
        issue.field.split(".", 1)[0]
        for result in verification
        for issue in result.issues
        if issue.severity == "error"
    )
    return {
        "ok": True,
        "predicted_records": len(predicted),
        "gold_records": len(gold),
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "grounding_acceptance_rate": (
            sum(result.accepted for result in verification) / len(verification)
            if verification
            else 0.0
        ),
        "grounding_error_counts": dict(issue_counts),
        "per_category": per_category,
    }
