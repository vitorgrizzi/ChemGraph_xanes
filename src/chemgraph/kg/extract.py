"""Schema-constrained extraction for literature KG records."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from chemgraph.kg.ontology import CATALYSIS_ELEMENTS
from chemgraph.kg.schema import (
    CatalystRecord,
    EvidenceSpan,
    Measurement,
    PaperChunk,
    ReactionCondition,
)

EXTRACTOR_VERSION = "literature_kg_mvp_regex"


EXTRACTION_PROMPT = """You are extracting structured catalysis data from scientific text.

Rules:
- Extract only facts explicitly stated in the provided evidence.
- Do not infer missing values.
- Do not use outside knowledge.
- If a field is not present, use null or [].
- Every non-null field must include an evidence_span_id.
- Preserve raw text for numerical values.
- Normalize only when unambiguous.
- Return valid JSON matching the CatalystRecord schema.

Evidence:
{chunk_text}

Schema:
{schema}
"""


def _message_text(response: Any) -> str:
    if hasattr(response, "content"):
        return str(response.content)
    if isinstance(response, dict) and "content" in response:
        return str(response["content"])
    return str(response)


def _extract_json_object(text: str) -> dict:
    """Extract the first JSON object from a model response."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return json.loads(stripped[start : end + 1])
    raise ValueError("No valid JSON object found in extraction response.")


def _canonical_quantity(raw: str) -> str:
    text = raw.lower()
    text = text.replace("co₂", "co2")
    if "methanol" in text and "select" in text:
        return "methanol_selectivity"
    if "co2" in text and "conversion" in text:
        return "co2_conversion"
    if "co " in text and "select" in text:
        return "co_selectivity"
    if "methane" in text and "select" in text:
        return "methane_selectivity"
    if "space-time" in text or "sty" in text:
        return "space_time_yield"
    if "time-on-stream" in text or "time on stream" in text:
        return "time_on_stream"
    if "conversion" in text:
        return "conversion"
    if "select" in text:
        return "selectivity"
    return text.strip().replace(" ", "_")


def _find_catalyst_name(text: str) -> str | None:
    patterns = [
        r"\b[A-Z][a-z]?(?:[-/][A-Z][A-Za-z0-9]*){1,4}\b",
        r"\b[A-Z][a-z]?/[A-Z][A-Za-z0-9]+/[A-Z][A-Za-z0-9]+\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    catalyst_match = re.search(
        r"(?:catalyst|sample)\s+([A-Z][A-Za-z0-9/._-]{2,})",
        text,
        flags=re.IGNORECASE,
    )
    if catalyst_match:
        return catalyst_match.group(1)
    return None


def _find_elements(text: str) -> list[str]:
    found = set()
    for symbol in CATALYSIS_ELEMENTS:
        if re.search(rf"(?<![A-Za-z]){re.escape(symbol)}(?![a-z])", text):
            found.add(symbol)
    return sorted(found)


def _detect_support(catalyst_name: str | None) -> str | None:
    if not catalyst_name or "/" not in catalyst_name:
        return None
    parts = [part for part in catalyst_name.split("/") if part]
    if len(parts) >= 2:
        return parts[-1]
    return None


def _detect_reaction(text: str) -> str | None:
    lower = text.lower()
    if "co2" in lower and "hydrogenation" in lower:
        if "methanol" in lower:
            return "CO2 hydrogenation to methanol"
        return "CO2 hydrogenation"
    if "methanol synthesis" in lower:
        return "CO2 hydrogenation to methanol"
    return None


def _extract_conditions(text: str, evidence_id: str) -> list[ReactionCondition]:
    temperatures = [
        float(match.group(1))
        for match in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:°\s*)?C\b", text)
    ]
    pressures = [
        float(match.group(1))
        for match in re.finditer(r"(\d+(?:\.\d+)?)\s*bar\b", text, flags=re.I)
    ]
    ratios = [
        float(match.group(1))
        for match in re.finditer(r"H2\s*/\s*CO2\s*(?:=|ratio)?\s*(\d+(?:\.\d+)?)", text, flags=re.I)
    ]
    if not temperatures and not pressures and not ratios:
        return []
    return [
        ReactionCondition(
            temperature=temperatures[0] if temperatures else None,
            pressure=pressures[0] if pressures else None,
            h2_co2_ratio=ratios[0] if ratios else None,
            evidence_span_id=evidence_id,
        )
    ]


def _extract_measurements(
    text: str,
    evidence_id: str,
    condition_id: str | None,
) -> list[Measurement]:
    metrics: list[Measurement] = []
    metric_pattern = re.compile(
        r"((?:CO2|CO|methanol|methane|higher alcohol|carbon)?\s*"
        r"(?:conversion|selectivity|yield|space-time yield|STY|time-on-stream|time on stream))"
        r"(?:\s*(?:of|=|:|was|is|reached|above|below|~))?\s*"
        r"(\d+(?:\.\d+)?)\s*(%|percent|h|bar|g(?:MeOH)?\s*gcat-1\s*h-1|mmol\s*g-1\s*h-1)?",
        flags=re.IGNORECASE,
    )
    for match in metric_pattern.finditer(text):
        raw_quantity, raw_value, raw_unit = match.groups()
        unit = raw_unit or ("percent" if "select" in raw_quantity.lower() else None)
        metrics.append(
            Measurement(
                quantity=_canonical_quantity(raw_quantity),
                value=float(raw_value),
                unit=unit,
                raw_value=match.group(0),
                condition_id=condition_id,
                evidence_span_id=evidence_id,
                confidence=0.65,
            )
        )
    return metrics


def _extract_characterization_methods(text: str) -> list[str]:
    methods = []
    for method in ["XANES", "EXAFS", "XAS", "XRD", "TEM", "STEM", "XPS", "Raman", "FTIR", "DRIFTS", "NMR", "ICP"]:
        if re.search(rf"\b{method}\b", text, flags=re.IGNORECASE):
            methods.append(method.upper())
    return sorted(set(methods))


def regex_extract_record(chunk: PaperChunk) -> CatalystRecord | None:
    """Offline extraction fallback for tests and first-pass demos."""
    text = chunk.text
    catalyst_name = _find_catalyst_name(text)
    metrics_or_reaction = _extract_measurements(text, "placeholder", None) or _detect_reaction(text)
    if not catalyst_name and not metrics_or_reaction:
        return None

    evidence = EvidenceSpan(
        paper_id=chunk.paper_id,
        chunk_id=chunk.chunk_id,
        page=chunk.page,
        section=chunk.section,
        text=text,
        start_char=chunk.metadata.get("start_char"),
        end_char=chunk.metadata.get("end_char"),
        source_path=chunk.source_path,
        doi=chunk.doi,
        extraction_model=EXTRACTOR_VERSION,
    )
    conditions = _extract_conditions(text, evidence.evidence_id)
    condition_id = conditions[0].condition_id if conditions else None
    metrics = _extract_measurements(text, evidence.evidence_id, condition_id)
    elements = _find_elements(catalyst_name or text)
    support = _detect_support(catalyst_name)
    active_metals = elements[:1]
    promoters = [element for element in elements[1:] if element != support]

    return CatalystRecord(
        paper_id=chunk.paper_id,
        catalyst_name=catalyst_name or f"unknown_catalyst_{chunk.chunk_id}",
        reaction=_detect_reaction(text),
        active_metals=active_metals,
        promoters=promoters,
        support=support,
        reaction_conditions=conditions,
        performance_metrics=metrics,
        characterization_methods=_extract_characterization_methods(text),
        mechanistic_claims=[],
        evidence_spans=[evidence],
        confidence=0.55 if catalyst_name else 0.35,
        extractor_version=EXTRACTOR_VERSION,
    )


def llm_extract_record(chunk: PaperChunk, llm, retries: int = 1) -> CatalystRecord:
    """Extract a CatalystRecord by invoking a LangChain-style chat model."""
    schema = CatalystRecord.model_json_schema()
    prompt = EXTRACTION_PROMPT.format(
        chunk_text=chunk.text,
        schema=json.dumps(schema, indent=2),
    )
    last_error: Exception | None = None
    for _ in range(retries + 1):
        try:
            response = llm.invoke(prompt)
            payload = _extract_json_object(_message_text(response))
            record = CatalystRecord.model_validate(payload)
            if not record.evidence_spans:
                evidence = EvidenceSpan(
                    paper_id=chunk.paper_id,
                    chunk_id=chunk.chunk_id,
                    page=chunk.page,
                    section=chunk.section,
                    text=chunk.text,
                    source_path=chunk.source_path,
                    doi=chunk.doi,
                    extraction_model=getattr(llm, "model_name", "llm"),
                )
                data = record.model_dump()
                data["evidence_spans"] = [evidence.model_dump()]
                record = CatalystRecord.model_validate(data)
            return record
        except Exception as exc:
            last_error = exc
    raise ValueError(f"LLM extraction failed after retries: {last_error}") from last_error


def extract_records_from_chunks(
    chunks: Iterable[PaperChunk],
    *,
    llm=None,
    retries: int = 1,
) -> list[CatalystRecord]:
    """Extract records from chunks using an LLM or deterministic fallback."""
    records: list[CatalystRecord] = []
    for chunk in chunks:
        record = llm_extract_record(chunk, llm, retries=retries) if llm else regex_extract_record(chunk)
        if record is not None:
            records.append(record)
    return records


def write_records_jsonl(records: Iterable[CatalystRecord], out: str | Path) -> Path:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")
    return out_path


def read_records_jsonl(path: str | Path) -> list[CatalystRecord]:
    records: list[CatalystRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(CatalystRecord.model_validate_json(line))
    return records
