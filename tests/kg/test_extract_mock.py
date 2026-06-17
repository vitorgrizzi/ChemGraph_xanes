import json

from chemgraph.kg.extract import extract_records_from_chunks
from chemgraph.kg.schema import PaperChunk


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
                    "text": "Cu/ZnO methanol selectivity was 83%.",
                }
            ],
            "confidence": 0.9,
        }
        return json.dumps(payload)


def test_extract_records_with_mock_llm():
    chunk = PaperChunk(
        paper_id="paper1",
        chunk_id="chunk1",
        text="Cu/ZnO methanol selectivity was 83%.",
    )

    records = extract_records_from_chunks([chunk], llm=FakeLLM())

    assert len(records) == 1
    assert records[0].catalyst_name == "Cu/ZnO"
    assert records[0].performance_metrics[0].value == 83.0
