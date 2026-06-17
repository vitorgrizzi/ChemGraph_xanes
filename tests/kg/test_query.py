from chemgraph.kg.query import export_training_table, hybrid_query
from chemgraph.kg.schema import CatalystRecord, EvidenceSpan, Measurement, ReactionCondition
from chemgraph.kg.store import build_kg


def _build_demo_kg(tmp_path):
    span = EvidenceSpan(
        paper_id="paper1",
        chunk_id="chunk1",
        text="Cu/ZnO/Al2O3 gave methanol selectivity of 83% at 210 C and 50 bar.",
    )
    condition = ReactionCondition(
        temperature=210,
        pressure=50,
        evidence_span_id=span.evidence_id,
    )
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
    build_kg([record], tmp_path)


def test_hybrid_query_returns_graph_and_evidence(tmp_path):
    _build_demo_kg(tmp_path)

    result = hybrid_query(
        tmp_path,
        "Which catalysts report methanol selectivity above 70% below 220 C?",
    )

    assert result["graph"]["num_results"] == 1
    assert result["graph"]["results"][0]["source"]["name"] == "Cu/ZnO/Al2O3"
    assert result["semantic"]["num_results"] >= 1


def test_export_training_table(tmp_path):
    _build_demo_kg(tmp_path)
    out = tmp_path / "table.csv"

    result = export_training_table(tmp_path, out)

    assert result["n_rows"] == 1
    assert out.exists()
    assert "methanol_selectivity" in out.read_text(encoding="utf-8")
