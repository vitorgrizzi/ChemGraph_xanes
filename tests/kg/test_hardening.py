import json

import pandas as pd
import pytest

from chemgraph.kg.extract import extract_records_from_chunks, regex_extract_record
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


def test_regex_extractor_links_only_sentence_local_conditions():
    record = regex_extract_record(
        PaperChunk(
            paper_id="paper1",
            chunk_id="chunk1",
            text=(
                "A high selectivity towards methanol was maintained at around 80% "
                "during recycling at 250 °C for Cu/ZnO/Al2O3 catalyst. "
                "Conclusion. Hexanol reached the highest methanol selectivity of "
                "91% among the pure solvents."
            ),
        )
    )

    assert [metric.value for metric in record.performance_metrics] == [80.0]
    metric = record.performance_metrics[0]
    assert metric.attributes["comparator"] == "around"
    condition = record.reaction_conditions[0]
    assert metric.condition_id == condition.condition_id
    assert condition.temperature == 250
    assert verify_record(record).accepted


def test_observed_80_and_91_percent_case_queries_without_cross_linking(tmp_path):
    record = regex_extract_record(
        PaperChunk(
            paper_id="paper1",
            chunk_id="chunk1",
            text=(
                "A high selectivity towards methanol was maintained at around 80% "
                "during recycling at 250 °C for Cu/ZnO/Al2O3 catalyst. "
                "Hexanol reached the highest methanol selectivity of 91% among "
                "the pure solvents."
            ),
        )
    )
    build_kg([record], tmp_path)

    strict = hybrid_query(
        tmp_path,
        "methanol selectivity above 70% below 250 C",
    )
    inclusive = hybrid_query(
        tmp_path,
        "methanol selectivity above 70% at or below 250 C",
    )

    assert strict["graph"]["num_results"] == 0
    assert inclusive["graph"]["num_results"] == 1
    assert inclusive["graph"]["results"][0]["edge"]["attributes"]["value"] == 80


def test_multivalue_passage_restores_graph_results_with_paper_context(tmp_path):
    context_chunk = PaperChunk(
        paper_id="paper1",
        chunk_id="context",
        text=(
            "The Cu/ZnO/Al2O3 catalyst was used for CO2 hydrogenation "
            "to methanol."
        ),
    )
    metric_chunk = PaperChunk(
        paper_id="paper1",
        chunk_id="metrics",
        text=(
            "Conversions increased from 11% at 170 °C to 51% at 230 °C, "
            "while methanol selectivity increased from 69% at 170 °C and "
            "reached 84% at 230 °C. Compared with the first solvent, the "
            "second solvent gave 15% conversion and 82% methanol selectivity "
            "at 170 °C."
        ),
    )

    unlinked = regex_extract_record(metric_chunk)
    assert unlinked.catalyst_name.startswith("unknown_catalyst_")

    records = extract_records_from_chunks(
        [context_chunk, metric_chunk],
        profile="co2_methanol",
    )
    metric_record = next(record for record in records if record.performance_metrics)
    assert metric_record.catalyst_name == "Cu/ZnO/Al2O3"
    assert metric_record.attributes["catalyst_context_propagated"]
    assert verify_record(metric_record).accepted
    build_kg(records, tmp_path)

    result = hybrid_query(
        tmp_path,
        "methanol selectivity above 70% below 250 C",
    )

    observations = {
        (
            item["source"]["name"],
            item["edge"]["attributes"]["value"],
            item["condition"]["attributes"]["temperature"],
        )
        for item in result["graph"]["results"]
    }
    assert observations == {
        ("Cu/ZnO/Al2O3", 84.0, 230.0),
        ("Cu/ZnO/Al2O3", 82.0, 170.0),
    }


def test_paper_context_is_not_propagated_when_multiple_catalysts_are_named():
    chunks = [
        PaperChunk(
            paper_id="paper1",
            chunk_id="cu",
            text="Cu/ZnO was tested for CO2 hydrogenation.",
        ),
        PaperChunk(
            paper_id="paper1",
            chunk_id="pd",
            text="Pd/Al2O3 was tested for CO2 hydrogenation.",
        ),
        PaperChunk(
            paper_id="paper1",
            chunk_id="metrics",
            text="Methanol selectivity reached 84% at 230 °C.",
        ),
    ]

    records = extract_records_from_chunks(chunks, profile="co2_methanol")
    metric_record = next(record for record in records if record.performance_metrics)

    assert metric_record.catalyst_name.startswith("unknown_catalyst_")
    assert not metric_record.attributes.get("catalyst_context_propagated", False)


def test_estimated_equilibrium_metric_is_not_an_observed_performance_fact():
    record = regex_extract_record(
        PaperChunk(
            paper_id="paper1",
            chunk_id="chunk1",
            text=(
                "For Cu/ZnO, the equilibrium methanol selectivity was estimated "
                "as 92% at 170 °C."
            ),
        )
    )

    assert record.performance_metrics == []


def test_graph_query_deduplicates_overlapping_chunk_facts(tmp_path):
    chunks = [
        PaperChunk(
            paper_id="paper1",
            chunk_id="chunk1",
            text=(
                "First run: Cu/ZnO/Al2O3 reached methanol selectivity of 82% "
                "at 170 °C during CO2 hydrogenation to methanol."
            ),
        ),
        PaperChunk(
            paper_id="paper1",
            chunk_id="chunk2",
            text=(
                "Cu/ZnO/Al2O3 reached methanol selectivity of 82% at 170 °C "
                "during CO2 hydrogenation to methanol, as shown in the first run."
            ),
        ),
    ]
    records = extract_records_from_chunks(chunks)
    build_kg(records, tmp_path)

    result = hybrid_query(
        tmp_path,
        "methanol selectivity above 70% below 250 C",
    )

    assert result["graph"]["num_results"] == 1
    graph_result = result["graph"]["results"][0]
    assert len(graph_result["evidence"]) == 2
    assert len(graph_result["supporting_edge_ids"]) == 2


def test_co2_profile_preserves_mpa_and_scopes_reaction_inference():
    chunk = PaperChunk(
        paper_id="paper1",
        chunk_id="chunk1",
        text=(
            "The Cu-Zn/Al catalyst delivered CO2 conversion of 9.9% and "
            "methanol selectivity of 82.7% under 3 MPa and 250 °C."
        ),
    )
    general_record = regex_extract_record(chunk)
    record = regex_extract_record(chunk, profile="co2_methanol")

    assert general_record.reaction is None
    assert record.reaction == "CO2 hydrogenation to methanol"
    assert record.reaction_conditions[0].pressure == 3
    assert record.reaction_conditions[0].pressure_unit == "MPa"
    assert record.reaction_conditions[0].temperature == 250
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
    first = result["fused"]["results"][0]
    assert first["graph_hits"]
    assert first["graph_supported"]
    assert first["origins"] == ["graph", "retrieval"]
    assert first["graph_rank"] == 1
    assert first["retrieval_rank"] == 1
    assert "semantic" not in result


def test_hybrid_query_preserves_strict_and_inclusive_language(tmp_path):
    build_kg([_grounded_record(high_temperature=210)], tmp_path)

    strict = hybrid_query(
        tmp_path,
        "methanol selectivity above 83% below 210 C",
    )
    inclusive = hybrid_query(
        tmp_path,
        "methanol selectivity at least 83% at or below 210 C",
    )

    assert strict["parsed_filters"]["min_value_operator"] == ">"
    assert strict["parsed_filters"]["max_temperature_operator"] == "<"
    assert strict["graph"]["num_results"] == 0
    assert inclusive["parsed_filters"]["min_value_operator"] == ">="
    assert inclusive["parsed_filters"]["max_temperature_operator"] == "<="
    assert inclusive["graph"]["num_results"] == 1


def test_verifier_rejects_cross_sentence_metric_condition_link():
    span = EvidenceSpan(
        paper_id="paper1",
        chunk_id="chunk1",
        text=(
            "Cu/ZnO/Al2O3 was recycled at 250 °C. "
            "Methanol selectivity reached 91% among the tested solvents."
        ),
    )
    condition = ReactionCondition(
        temperature=250,
        evidence_span_id=span.evidence_id,
    )
    metric = Measurement(
        quantity="methanol_selectivity",
        value=91,
        unit="percent",
        condition_id=condition.condition_id,
        evidence_span_id=span.evidence_id,
    )
    record = CatalystRecord(
        paper_id="paper1",
        catalyst_name="Cu/ZnO/Al2O3",
        reaction_conditions=[condition],
        performance_metrics=[metric],
        evidence_spans=[span],
        field_evidence_ids={"catalyst_name": [span.evidence_id]},
    )

    result = verify_record(record)

    assert not result.accepted
    assert "not stated in the same sentence" in " ".join(
        issue.message for issue in result.issues
    )


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
