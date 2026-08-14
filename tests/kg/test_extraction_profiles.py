import pytest

from chemgraph.kg.extract import extract_records_from_chunks, regex_extract_record
from chemgraph.kg.profiles import (
    load_extraction_profile,
    profile_name_for_model,
)
from chemgraph.kg.schema import PaperChunk
from chemgraph.kg.verify import verify_record


@pytest.mark.parametrize(
    ("text", "reaction", "quantity", "value", "unit"),
    [
        (
            "Pt/Al2O3 reached CO conversion of 88% at 150 C during CO oxidation.",
            "CO oxidation",
            "co_conversion",
            88.0,
            "percent",
        ),
        (
            "Ru/MgO achieved ammonia yield of 12% at 400 C and 10 MPa "
            "during ammonia synthesis.",
            "ammonia synthesis",
            "ammonia_yield",
            12.0,
            "percent",
        ),
        (
            "Ni/Al2O3 gave methane conversion of 75% at 700 C during "
            "methane steam reforming.",
            "methane steam reforming",
            "methane_conversion",
            75.0,
            "percent",
        ),
        (
            "Ni/FeOx showed an overpotential of 280 mV during oxygen "
            "evolution reaction.",
            "oxygen evolution reaction",
            "overpotential",
            280.0,
            "mV",
        ),
        (
            "Pt/C showed overpotential of 30 mV during hydrogen evolution "
            "reaction.",
            "hydrogen evolution reaction",
            "overpotential",
            30.0,
            "mV",
        ),
    ],
)
def test_general_regex_extracts_unrelated_reaction_families(
    text,
    reaction,
    quantity,
    value,
    unit,
):
    record = regex_extract_record(
        PaperChunk(paper_id="paper1", chunk_id="chunk1", text=text)
    )

    assert record is not None
    assert record.reaction == reaction
    assert [
        (metric.quantity, metric.value, metric.unit)
        for metric in record.performance_metrics
    ] == [(quantity, value, unit)]
    assert record.attributes["extraction_profile"] == "general"
    assert verify_record(record).accepted


def test_co2_metric_combination_is_inferred_only_by_named_profile():
    chunk = PaperChunk(
        paper_id="paper1",
        chunk_id="chunk1",
        text=(
            "Cu/ZnO delivered CO2 conversion of 9.9% and methanol "
            "selectivity of 82.7% at 250 C."
        ),
    )

    general = regex_extract_record(chunk)
    specialized = regex_extract_record(chunk, profile="co2_methanol")

    assert general.reaction is None
    assert specialized.reaction == "CO2 hydrogenation to methanol"
    assert specialized.extractor_version == "co2_methanol_regex_v1"
    assert {metric.quantity for metric in specialized.performance_metrics} == {
        "co2_conversion",
        "methanol_selectivity",
    }


def test_co2_ratio_and_cross_chunk_context_are_profile_scoped():
    chunks = [
        PaperChunk(
            paper_id="paper1",
            chunk_id="context",
            text=(
                "Cu/ZnO was tested for CO2 hydrogenation at 250 C and "
                "H2/CO2 = 3."
            ),
        ),
        PaperChunk(
            paper_id="paper1",
            chunk_id="metric",
            text="Methanol selectivity reached 80% at 230 C.",
        ),
    ]

    general = extract_records_from_chunks(chunks)
    specialized = extract_records_from_chunks(chunks, profile="co2_methanol")

    assert general[0].reaction_conditions[0].h2_co2_ratio is None
    metric_record = next(record for record in specialized if record.performance_metrics)
    assert metric_record.catalyst_name == "Cu/ZnO"
    assert metric_record.attributes["catalyst_context_propagated"]
    assert specialized[0].reaction_conditions[0].h2_co2_ratio == 3


def test_profile_loader_and_model_alias_are_explicit():
    profile = load_extraction_profile("co2_methanol")

    assert profile.name == "co2_methanol"
    assert profile.propagate_unique_catalyst
    assert profile_name_for_model("co2_methanol_regex", "general") == "co2_methanol"
    with pytest.raises(ValueError, match="cannot be combined"):
        profile_name_for_model("co2_methanol_regex", "unrelated_profile")


def test_custom_yaml_profile_extends_quantity_vocabulary(tmp_path):
    config = tmp_path / "profiles.yaml"
    config.write_text(
        """
profiles:
  oxygenate_pilot:
    extractor_version: oxygenate_regex_v1
    quantity_aliases:
      desired_product_selectivity:
        - desired product share
    reaction_aliases:
      water oxidation:
        - water oxidation
""".strip(),
        encoding="utf-8",
    )
    profile = load_extraction_profile("oxygenate_pilot", config)
    record = regex_extract_record(
        PaperChunk(
            paper_id="paper1",
            chunk_id="chunk1",
            text=(
                "Pt/Al2O3 desired product share was 60% at 300 C during "
                "water oxidation."
            ),
        ),
        profile=profile,
    )

    assert record.reaction == "water oxidation"
    assert record.performance_metrics[0].quantity == "desired_product_selectivity"
    assert record.extractor_version == "oxygenate_regex_v1"
    assert verify_record(record).accepted
