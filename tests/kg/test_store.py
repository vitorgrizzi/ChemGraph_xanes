from chemgraph.kg.schema import CatalystRecord, EvidenceSpan, Measurement, ReactionCondition
from chemgraph.kg.store import LiteratureKGStore, build_kg


def test_store_builds_nodes_edges_and_evidence(tmp_path):
    span = EvidenceSpan(
        paper_id="paper1",
        chunk_id="chunk1",
        text="Cu/ZnO/Al2O3 reached methanol selectivity of 83% at 210 C.",
    )
    condition = ReactionCondition(temperature=210, evidence_span_id=span.evidence_id)
    metric = Measurement(
        quantity="methanol_selectivity",
        value=83,
        unit="percent",
        evidence_span_id=span.evidence_id,
        condition_id=condition.condition_id,
    )
    record = CatalystRecord(
        paper_id="paper1",
        catalyst_name="Cu/ZnO/Al2O3",
        reaction="CO2 hydrogenation to methanol",
        active_metals=["Cu"],
        support="Al2O3",
        reaction_conditions=[condition],
        performance_metrics=[metric],
        evidence_spans=[span],
    )

    result = build_kg([record], tmp_path)
    store = LiteratureKGStore(tmp_path)

    assert result["n_nodes"] >= 4
    assert result["n_edges"] >= 4
    assert store.get_evidence(span.evidence_id).text.startswith("Cu/ZnO")
    assert (tmp_path / "nodes.parquet").exists()
    assert (tmp_path / "edges.parquet").exists()
    assert (tmp_path / "evidence.sqlite").exists()
