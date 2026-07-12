from chemgraph.kg.benchmark import evaluate_extractions
from chemgraph.kg.schema import CatalystRecord, EvidenceSpan, Measurement


def _record(value):
    span = EvidenceSpan(
        paper_id="paper1",
        chunk_id="chunk1",
        text=f"Cu/ZnO methanol selectivity was {value}%.",
    )
    return CatalystRecord(
        paper_id="paper1",
        catalyst_name="Cu/ZnO",
        active_metals=["Cu"],
        support="ZnO",
        performance_metrics=[
            Measurement(
                quantity="methanol_selectivity",
                value=value,
                unit="percent",
                evidence_span_id=span.evidence_id,
            )
        ],
        evidence_spans=[span],
        field_evidence_ids={
            "catalyst_name": [span.evidence_id],
            "active_metals": [span.evidence_id],
            "support": [span.evidence_id],
        },
    )


def test_gold_benchmark_reports_fact_precision_and_recall():
    result = evaluate_extractions([_record(83)], [_record(83)])

    assert result["micro_precision"] == 1.0
    assert result["micro_recall"] == 1.0
    assert result["grounding_acceptance_rate"] == 1.0
