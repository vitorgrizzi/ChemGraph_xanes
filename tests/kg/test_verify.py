from chemgraph.kg.schema import CatalystRecord, EvidenceSpan, Measurement
from chemgraph.kg.verify import verify_record


def test_verify_rejects_percent_metric_out_of_range():
    span = EvidenceSpan(paper_id="paper1", chunk_id="chunk1", text="bad metric")
    record = CatalystRecord(
        paper_id="paper1",
        catalyst_name="Cu/ZnO",
        evidence_spans=[span],
        performance_metrics=[
            Measurement(
                quantity="methanol_selectivity",
                value=130.0,
                unit="percent",
                evidence_span_id=span.evidence_id,
            )
        ],
    )

    result = verify_record(record)

    assert not result.accepted
    assert any("between 0 and 100" in issue.message for issue in result.issues)
