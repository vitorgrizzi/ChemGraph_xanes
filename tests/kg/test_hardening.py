import json

import pandas as pd
import pytest

from chemgraph.kg.extract import regex_extract_record
from chemgraph.kg.query import graph_query, hybrid_query
from chemgraph.kg.schema import (
    CatalystRecord,
    EvidenceSpan,
    KGEdge,
    KGNode,
    Measurement,
    PaperChunk,
    ReactionCondition,
)
from chemgraph.kg.store import build_kg
from chemgraph.kg.validation import validate_kg
from chemgraph.kg.verify import verify_record


def _grounded_record(*, high_temperature=300, include_low_condition=False):
    text = (
        "Cu/ZnO/Al2O3 reached methanol selectivity of 83% at "
        f"{high_temperature} C and 50 bar during CO2 hydrogenation to methanol."
    )
    if include_low_condition:
        text += " A separate screening condition was 200 C."
    span = EvidenceSpan(paper_id="paper1", chunk_id="chunk1", text=text)
    high = ReactionCondition(
        temperature=high_temperature,
        pressure=50,
        evidence_span_id=span.evidence_id,
    )
    conditions = [high]
    if include_low_condition:
        conditions.insert(
            0,
            ReactionCondition(temperature=200, evidence_span_id=span.evidence_id),
        )
    metric = Measurement(
        quantity="methanol_selectivity",
        value=83,
        unit="percent",
        condition_id=high.condition_id,
        evidence_span_id=span.evidence_id,
    )
    return CatalystRecord(
        paper_id="paper1",
        catalyst_name="Cu/ZnO/Al2O3",
        reaction="CO2 hydrogenation to methanol",
        active_metals=["Cu"],
        support="Al2O3",
        reaction_conditions=conditions,
        performance_metrics=[metric],
        evidence_spans=[span],
        field_evidence_ids={
            "catalyst_name": [span.evidence_id],
            "reaction": [span.evidence_id],
            "active_metals": [span.evidence_id],
            "support": [span.evidence_id],
        },
    )


def test_content_ids_are_deterministic():
    first = _grounded_record()
    second = _grounded_record()

    assert first.record_id == second.record_id
    assert first.evidence_spans[0].evidence_id == second.evidence_spans[0].evidence_id
    assert first.reaction_conditions[0].condition_id == second.reaction_conditions[0].condition_id
    assert first.performance_metrics[0].measurement_id == second.performance_metrics[0].measurement_id


def test_regex_extractor_classifies_components_and_unicode_temperature():
    record = regex_extract_record(
        PaperChunk(
            paper_id="paper1",
            chunk_id="chunk1",
            text=(
                "Cu/ZnO/Al2O3 was tested for CO2 hydrogenation to methanol. "
                "Methanol selectivity reached 83% at 210 °C and 50 bar."
            ),
        )
    )

    assert record is not None
    assert record.active_metals == ["Cu"]
    assert record.promoters == ["Zn"]
    assert record.support == "Al2O3"
    assert record.reaction_conditions[0].temperature == 210


def test_regex_extractor_preserves_measurement_ranges():
    record = regex_extract_record(
        PaperChunk(
            paper_id="paper1",
            chunk_id="chunk1",
            text=(
                "Cu/ZnO methanol selectivity was 70-80% at 210 C during "
                "CO2 hydrogenation to methanol."
            ),
        )
    )

    metric = record.performance_metrics[0]
    assert metric.value == 70
    assert metric.attributes["comparator"] == "range"
    assert metric.attributes["range_max"] == 80
    assert verify_record(record).accepted


def test_verifier_rejects_unentailed_metric_and_missing_condition_reference():
    span = EvidenceSpan(paper_id="paper1", chunk_id="chunk1", text="Cu/ZnO was tested.")
    condition = ReactionCondition(temperature=200, evidence_span_id="missing")
    metric = Measurement(
        quantity="methanol_selectivity",
        value=83,
        unit="percent",
        condition_id=condition.condition_id,
        evidence_span_id=span.evidence_id,
    )
    record = CatalystRecord(
        paper_id="paper1",
        catalyst_name="Cu/ZnO",
        reaction_conditions=[condition],
        performance_metrics=[metric],
        evidence_spans=[span],
        field_evidence_ids={"catalyst_name": [span.evidence_id]},
    )

    result = verify_record(record)

    assert not result.accepted
    messages = " ".join(issue.message for issue in result.issues)
    assert "not present in its evidence text" in messages
    assert "not present in record evidence" in messages


def test_build_rejects_unverified_records(tmp_path):
    span = EvidenceSpan(paper_id="paper1", chunk_id="chunk1", text="Cu/ZnO was tested.")
    record = CatalystRecord(
        paper_id="paper1",
        catalyst_name="Cu/ZnO",
        evidence_spans=[span],
    )

    with pytest.raises(ValueError, match="rejected unverified records"):
        build_kg([record], tmp_path)


def test_temperature_filter_uses_metric_condition_only(tmp_path):
    build_kg([_grounded_record(include_low_condition=True)], tmp_path)

    too_low = graph_query(
        tmp_path,
        relation="achieves",
        metric_quantity="methanol_selectivity",
        max_temperature=220,
    )
    high_enough = graph_query(
        tmp_path,
        relation="achieves",
        metric_quantity="methanol_selectivity",
        max_temperature=320,
    )

    assert too_low["num_results"] == 0
    assert high_enough["num_results"] == 1
    assert high_enough["results"][0]["condition"]["attributes"]["temperature"] == 300


def test_hybrid_query_fuses_graph_and_retrieval_results(tmp_path):
    build_kg([_grounded_record(high_temperature=210)], tmp_path)

    result = hybrid_query(
        tmp_path,
        "methanol selectivity above 70% below 220 C",
    )

    assert result["graph"]["num_results"] == 1
    assert result["retrieval"]["method"] == "bm25"
    assert result["fused"]["num_results"] >= 1
    assert result["fused"]["results"][0]["graph_hits"]


def test_artifact_validator_detects_hash_tampering(tmp_path):
    result = build_kg([_grounded_record()], tmp_path)
    assert validate_kg(tmp_path)["ok"]

    graph_path = tmp_path / "graph.json"
    graph_path.write_text(graph_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    validation = validate_kg(tmp_path)
    assert not validation["ok"]
    assert "hash mismatch" in " ".join(validation["errors"]).lower()
    assert result["validation"]["ok"]


def test_equivalent_rebuilds_have_identical_artifact_hashes(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    build_kg([_grounded_record()], first_dir)
    build_kg([_grounded_record()], second_dir)
    first_manifest = json.loads((first_dir / "manifest.json").read_text(encoding="utf-8"))
    second_manifest = json.loads((second_dir / "manifest.json").read_text(encoding="utf-8"))

    assert first_manifest["sha256"] == second_manifest["sha256"]


def test_parquet_fallback_uses_truthful_jsonl_suffix(tmp_path, monkeypatch):
    def missing_engine(*args, **kwargs):
        raise ImportError("Unable to find a usable parquet engine; pyarrow is missing")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", missing_engine)

    result = build_kg([_grounded_record()], tmp_path)

    assert result["paths"]["nodes"].endswith("nodes.jsonl")
    assert result["paths"]["edges"].endswith("edges.jsonl")
    assert not (tmp_path / "nodes.parquet").exists()
    assert validate_kg(tmp_path)["ok"]


def test_node_and_relation_types_are_constrained():
    with pytest.raises(ValueError, match="Unknown KG node type"):
        KGNode(node_id="bad", node_type="Typo", name="bad")
    with pytest.raises(ValueError, match="Unknown KG relation"):
        KGEdge(
            edge_id="bad",
            source_node_id="a",
            relation="typo",
            target_node_id="b",
            evidence_ids=["span"],
            confidence=0.5,
        )
