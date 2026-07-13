from pathlib import Path

from chemgraph.kg.schema import CatalystRecord, EvidenceSpan, Measurement, ReactionCondition
from chemgraph.kg.store import LiteratureKGStore, build_kg


def test_store_builds_nodes_edges_and_evidence(tmp_path):
    span = EvidenceSpan(
        paper_id="paper1",
        chunk_id="chunk1",
        doi="10.1234/example",
        text=(
            "Cu/ZnO/Al2O3 reached methanol selectivity of 83% at 210 C "
            "during CO2 hydrogenation to methanol."
        ),
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
        field_evidence_ids={
            "catalyst_name": [span.evidence_id],
            "reaction": [span.evidence_id],
            "active_metals": [span.evidence_id],
            "support": [span.evidence_id],
        },
    )

    result = build_kg([record], tmp_path)
    store = LiteratureKGStore(tmp_path)

    assert result["n_nodes"] >= 4
    assert result["n_edges"] >= 4
    assert store.get_evidence(span.evidence_id).text.startswith("Cu/ZnO")
    paper = next(node for node in store.load_nodes() if node.node_type == "Paper")
    assert paper.name == "10.1234/example"
    assert paper.attributes["dois"] == ["10.1234/example"]
    assert Path(result["paths"]["nodes"]).exists()
    assert Path(result["paths"]["edges"]).exists()
    assert (tmp_path / "evidence.sqlite").exists()
    assert (tmp_path / "manifest.json").exists()
