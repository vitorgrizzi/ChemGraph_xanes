from chemgraph.kg.hypotheses import suggest_hypotheses
from chemgraph.kg.schema import CatalystRecord, EvidenceSpan, Measurement
from chemgraph.kg.store import build_kg


def test_suggest_hypotheses_returns_cards(tmp_path):
    span = EvidenceSpan(
        paper_id="paper1",
        chunk_id="chunk1",
        text="Cu/ZnO/Al2O3 methanol selectivity was 83%.",
    )
    record = CatalystRecord(
        paper_id="paper1",
        catalyst_name="Cu/ZnO/Al2O3",
        active_metals=["Cu"],
        support="Al2O3",
        performance_metrics=[
            Measurement(
                quantity="methanol_selectivity",
                value=83,
                unit="percent",
                evidence_span_id=span.evidence_id,
            )
        ],
        evidence_spans=[span],
    )
    build_kg([record], tmp_path)

    result = suggest_hypotheses(
        tmp_path,
        goal="low-temperature CO2 hydrogenation to methanol",
    )

    assert result["num_hypotheses"] == 1
    card = result["hypotheses"][0]
    assert card["supporting_edge_ids"]
    assert card["structured_tasks"]
