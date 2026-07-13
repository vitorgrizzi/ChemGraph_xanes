import json

from chemgraph.kg.extract import extract_records_from_chunks
from chemgraph.kg.schema import PaperChunk
from chemgraph.kg.verify import verify_record


class FakeLLM:
    model_name = "fake-model"

    def invoke(self, prompt):
        payload = {
            "paper_id": "paper1",
            "catalyst_name": "Cu/ZnO",
            "reaction": "CO2 hydrogenation to methanol",
            "active_metals": ["Cu"],
            "support": "ZnO",
            "performance_metrics": [
                {
                    "quantity": "methanol_selectivity",
                    "value": 83.0,
                    "unit": "percent",
                    "evidence_span_id": "span1",
                    "confidence": 0.9,
                }
            ],
            "evidence_spans": [
                {
                    "evidence_id": "span1",
                    "paper_id": "paper1",
                    "chunk_id": "chunk1",
                    "text": "model-controlled evidence must be ignored",
                }
            ],
            "confidence": 0.9,
        }
        return json.dumps(payload)


def test_extract_records_with_mock_llm():
    chunk = PaperChunk(
        paper_id="paper1",
        chunk_id="chunk1",
        text=(
            "Cu/ZnO methanol selectivity was 83% during CO2 hydrogenation "
            "to methanol."
        ),
    )

    records = extract_records_from_chunks([chunk], llm=FakeLLM())

    assert len(records) == 1
    assert records[0].catalyst_name == "Cu/ZnO"
    assert records[0].performance_metrics[0].value == 83.0
    assert records[0].paper_id == chunk.paper_id
    assert records[0].evidence_spans[0].text == chunk.text
    assert records[0].evidence_spans[0].evidence_id != "span1"
    assert verify_record(records[0]).accepted


class HallucinatingLLM(FakeLLM):
    def invoke(self, prompt):
        payload = json.loads(super().invoke(prompt))
        payload["performance_metrics"][0]["value"] = 99.0
        return json.dumps(payload)


def test_llm_hallucinated_value_fails_grounding():
    chunk = PaperChunk(
        paper_id="paper1",
        chunk_id="chunk1",
        text=(
            "Cu/ZnO methanol selectivity was 83% during CO2 hydrogenation "
            "to methanol."
        ),
    )

    record = extract_records_from_chunks([chunk], llm=HallucinatingLLM())[0]

    assert not verify_record(record).accepted


class ConditionLinkingLLM(FakeLLM):
    def invoke(self, prompt):
        payload = json.loads(super().invoke(prompt))
        payload["reaction_conditions"] = [
            {
                "condition_id": "condition_1",
                "temperature": 210,
                "temperature_unit": "degC",
            }
        ]
        payload["performance_metrics"][0]["condition_id"] = "condition_1"
        return json.dumps(payload)


def test_llm_local_condition_labels_are_remapped_without_inference():
    chunk = PaperChunk(
        paper_id="paper1",
        chunk_id="chunk1",
        text=(
            "Cu/ZnO methanol selectivity was 83% at 210 C during CO2 "
            "hydrogenation to methanol."
        ),
    )

    record = extract_records_from_chunks([chunk], llm=ConditionLinkingLLM())[0]
    condition = record.reaction_conditions[0]
    metric = record.performance_metrics[0]

    assert condition.condition_id != "condition_1"
    assert metric.condition_id == condition.condition_id
    assert verify_record(record).accepted
