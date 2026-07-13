import csv

import pytest

from chemgraph.kg.query import export_training_table, hybrid_query
from chemgraph.kg.schema import CatalystRecord, EvidenceSpan, Measurement, ReactionCondition
from chemgraph.kg.store import build_kg


def _build_demo_kg(tmp_path):
    span = EvidenceSpan(
        paper_id="paper1",
        chunk_id="chunk1",
        doi="10.1234/example",
        text=(
            "Cu/ZnO/Al2O3 gave methanol selectivity of 83% at 210 C and 50 bar "
            "during CO2 hydrogenation to methanol."
        ),
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
        field_evidence_ids={
            "catalyst_name": [span.evidence_id],
            "reaction": [span.evidence_id],
            "active_metals": [span.evidence_id],
            "support": [span.evidence_id],
        },
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
    assert result["retrieval"]["num_results"] >= 1
    assert "semantic" not in result


def test_compact_query_returns_model_facing_answers(tmp_path):
    _build_demo_kg(tmp_path)

    result = hybrid_query(
        tmp_path,
        "Which catalysts report methanol selectivity above 70% below 220 C?",
        response_mode="compact",
    )

    assert result["response_mode"] == "compact"
    assert result["answer_count"] == 1
    assert result["retrieval_context"] == []
    assert result["warnings"] == []
    assert not {"graph", "retrieval", "fused"} & result.keys()
    answer = result["answers"][0]
    assert answer["catalyst"] == "Cu/ZnO/Al2O3"
    assert answer["metric_quantity"] == "methanol_selectivity"
    assert answer["value"] == 83
    assert answer["temperature"] == 210
    assert answer["pressure"] == 50
    assert answer["doi"] == "10.1234/example"
    assert answer["evidence_ids"]
    assert answer["supporting_edge_ids"]
    assert "83%" in answer["evidence_excerpt"]
    assert len(answer["evidence_excerpt"]) <= 322


def test_compact_query_uses_retrieval_only_as_explicit_fallback(tmp_path):
    _build_demo_kg(tmp_path)

    result = hybrid_query(
        tmp_path,
        "methanol selectivity above 99% below 220 C",
        response_mode="compact",
    )

    assert result["answer_count"] == 0
    assert result["answers"] == []
    assert result["warnings"]
    assert result["retrieval_context"]
    assert len(result["retrieval_context"]) <= 3
    assert all(
        context["graph_supported"] is False
        for context in result["retrieval_context"]
    )
    assert all("text" not in context for context in result["retrieval_context"])


def test_hybrid_query_rejects_unknown_response_mode(tmp_path):
    _build_demo_kg(tmp_path)

    with pytest.raises(ValueError, match="response_mode"):
        hybrid_query(tmp_path, "methanol selectivity", response_mode="brief")


def test_export_training_table(tmp_path):
    _build_demo_kg(tmp_path)
    out = tmp_path / "table.csv"

    result = export_training_table(tmp_path, out)

    assert result["n_rows"] == 1
    assert out.exists()
    assert "methanol_selectivity" in out.read_text(encoding="utf-8")


def test_export_does_not_mix_observations_for_same_catalyst(tmp_path):
    records = []
    for paper_id, temperature, value in (("paper1", 210, 83), ("paper2", 300, 61)):
        span = EvidenceSpan(
            paper_id=paper_id,
            chunk_id=f"{paper_id}_chunk",
            text=(
                f"Cu/ZnO/Al2O3 gave methanol selectivity of {value}% at "
                f"{temperature} C during CO2 hydrogenation to methanol."
            ),
        )
        condition = ReactionCondition(
            temperature=temperature,
            evidence_span_id=span.evidence_id,
        )
        metric = Measurement(
            quantity="methanol_selectivity",
            value=value,
            unit="percent",
            condition_id=condition.condition_id,
            evidence_span_id=span.evidence_id,
        )
        records.append(
            CatalystRecord(
                paper_id=paper_id,
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
        )
    build_kg(records, tmp_path)
    out = tmp_path / "table.csv"

    export_training_table(tmp_path, out)
    with out.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    assert {(row["paper_id"], row["reaction_temperature"], row["target"]) for row in rows} == {
        ("paper1", "210.0", "83.0"),
        ("paper2", "300.0", "61.0"),
    }
