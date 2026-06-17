import pytest

from chemgraph.kg.schema import (
    CatalystRecord,
    EvidenceSpan,
    KGEdge,
    Measurement,
)


def test_schema_validates_record_with_evidence():
    span = EvidenceSpan(
        paper_id="paper1",
        chunk_id="chunk1",
        text="Cu/ZnO reports methanol selectivity of 83%.",
    )
    metric = Measurement(
        quantity="methanol_selectivity",
        value=83,
        unit="percent",
        evidence_span_id=span.evidence_id,
    )
    record = CatalystRecord(
        paper_id="paper1",
        catalyst_name="Cu/ZnO",
        performance_metrics=[metric],
        evidence_spans=[span],
    )
    assert record.canonical_catalyst_name == "Cu/ZnO"


def test_edge_requires_evidence_ids():
    with pytest.raises(ValueError):
        KGEdge(
            edge_id="edge1",
            source_node_id="node1",
            relation="reports",
            target_node_id="node2",
            evidence_ids=[],
            confidence=0.5,
        )


def test_confidence_must_be_unit_interval():
    with pytest.raises(ValueError):
        Measurement(quantity="co2_conversion", confidence=1.2)
