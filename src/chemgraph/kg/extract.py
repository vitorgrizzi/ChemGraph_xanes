"""Schema-constrained extraction for literature KG records."""

from __future__ import annotations

import json
import os
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
- Do not invent paper, chunk, evidence, record, condition, or measurement IDs;
  the application attaches source-controlled provenance after extraction.
- Preserve raw text for numerical values.
- Preserve comparators such as above, below, approximately, and ranges.
- Keep each measurement linked to the reaction condition under which it was measured.
- Normalize only when unambiguous.
- Return valid JSON matching the CatalystRecord schema.

Evidence:
{chunk_text}

Schema:
{schema}
"""


def load_extraction_llm(model: str):
    """Load an extraction model, or return ``None`` for the offline fallback."""
    if model.lower() in {"deterministic", "regex", "offline", "none"}:
        return None
    from chemgraph.models.openai import load_openai_model

    return load_openai_model(
        model_name=model,
        temperature=0.0,
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )


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
    matches: list[tuple[int, str]] = []
    for symbol in CATALYSIS_ELEMENTS:
        match = re.search(rf"(?<![A-Za-z]){re.escape(symbol)}(?![a-z])", text)
        if match:
            matches.append((match.start(), symbol))
    return [symbol for _, symbol in sorted(matches)]


def _detect_support(catalyst_name: str | None) -> str | None:
    if not catalyst_name or "/" not in catalyst_name:
        return None
    parts = [part for part in catalyst_name.split("/") if part]
    if len(parts) >= 2:
        return parts[-1]
    return None


def _classify_catalyst_components(
    catalyst_name: str | None,
) -> tuple[list[str], list[str], str | None]:
    """Conservatively classify slash-delimited catalyst components.

    The first component is treated as the active phase, the last as support,
    and middle components as promoters. This remains a heuristic, but avoids
    the previous alphabetical-element assignment (for example, Al becoming
    the active metal in Cu/ZnO/Al2O3).
    """
    if not catalyst_name:
        return [], [], None
    parts = [part for part in catalyst_name.split("/") if part]
    if not parts:
        return [], [], None
    active_metals = _find_elements(parts[0])
    support = parts[-1] if len(parts) >= 2 else None
    promoters: list[str] = []
    for part in parts[1:-1]:
        promoters.extend(_find_elements(part))
    return list(dict.fromkeys(active_metals)), list(dict.fromkeys(promoters)), support


def _detect_reaction(text: str) -> str | None:
    lower = text.lower().replace("co₂", "co2").replace("h₂", "h2")
    if "co2" in lower and "hydrogenation" in lower:
        if "methanol" in lower:
            return "CO2 hydrogenation to methanol"
        return "CO2 hydrogenation"
    if "methanol synthesis" in lower:
        return "CO2 hydrogenation to methanol"
    return None


def _extract_conditions(text: str, evidence_id: str) -> list[ReactionCondition]:
    normalized = text.replace("CO₂", "CO2").replace("H₂", "H2")
    temperatures = [
        float(match.group(1))
        for match in re.finditer(
            r"(-?\d+(?:\.\d+)?)\s*(?:(?:°|º)\s*|deg(?:ree)?s?\s*)?C\b",
            normalized,
            flags=re.I,
        )
    ]
    pressures = [
        float(match.group(1))
        for match in re.finditer(r"(\d+(?:\.\d+)?)\s*bar\b", normalized, flags=re.I)
    ]
    ratios = [
        float(match.group(1))
        for match in re.finditer(
            r"H2\s*/\s*CO2\s*(?:(?:=|:)|ratio\s*(?:=|:)?)?\s*(\d+(?:\.\d+)?)",
            normalized,
            flags=re.I,
        )
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
        r"(?:\s*(?:of|was|is|reached))?\s*"
        r"(?:(=|:|above|over|>|below|under|<|~|approximately|about)\s*)?"
        r"(\d+(?:\.\d+)?)"
        r"(?:\s*(?:-|–|to)\s*(\d+(?:\.\d+)?))?\s*"
        r"(?:\s*(?:±|\+/-)\s*(\d+(?:\.\d+)?))?\s*"
        r"(%|percent|h|bar|g(?:MeOH)?\s*gcat-1\s*h-1|mmol\s*g-1\s*h-1)?",
        flags=re.IGNORECASE,
    )
    for match in metric_pattern.finditer(text):
        raw_quantity, comparator, raw_value, range_max, uncertainty, raw_unit = match.groups()
        unit = raw_unit or ("percent" if "select" in raw_quantity.lower() else None)
        attributes: dict[str, Any] = {"comparator": comparator or "="}
        if range_max is not None:
            attributes.update(
                {
                    "comparator": "range",
                    "range_min": float(raw_value),
                    "range_max": float(range_max),
                }
            )
        metrics.append(
            Measurement(
                quantity=_canonical_quantity(raw_quantity),
                value=float(raw_value),
                unit=unit,
                raw_value=match.group(0),
                uncertainty=float(uncertainty) if uncertainty is not None else None,
                condition_id=condition_id,
                evidence_span_id=evidence_id,
                confidence=0.65,
                attributes=attributes,
            )
        )
    return metrics


def _extract_characterization_methods(text: str) -> list[str]:
    methods = []
    for method in ["XANES", "EXAFS", "XAS", "XRD", "TEM", "STEM", "XPS", "Raman", "FTIR", "DRIFTS", "NMR", "ICP"]:
        if re.search(rf"\b{method}\b", text, flags=re.IGNORECASE):
            methods.append(method.upper())
    return sorted(set(methods))


def _extract_condition_metric_pairs(
    text: str,
    evidence_id: str,
) -> tuple[list[ReactionCondition], list[Measurement]]:
    """Associate metrics with conditions from the same sentence when possible."""
    segments = [
        segment.strip()
        for segment in re.split(r"(?<=[.!?])\s+|\n+", text)
        if segment.strip()
    ]
    conditions: list[ReactionCondition] = []
    pending_metrics: list[Measurement] = []
    metrics: list[Measurement] = []
    for segment in segments:
        segment_conditions = _extract_conditions(segment, evidence_id)
        conditions.extend(segment_conditions)
        condition_id = segment_conditions[0].condition_id if segment_conditions else None
        for metric in _extract_measurements(segment, evidence_id, condition_id):
            if condition_id:
                metrics.append(metric)
            else:
                pending_metrics.append(metric)
    unique_conditions = list({item.condition_id: item for item in conditions}.values())
    fallback_condition_id = (
        unique_conditions[0].condition_id if len(unique_conditions) == 1 else None
    )
    for metric in pending_metrics:
        data = metric.model_dump()
        data["measurement_id"] = ""
        data["condition_id"] = fallback_condition_id
        metrics.append(Measurement.model_validate(data))
    return unique_conditions, metrics


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
    conditions, metrics = _extract_condition_metric_pairs(text, evidence.evidence_id)
    active_metals, promoters, support = _classify_catalyst_components(catalyst_name)
    field_evidence_ids: dict[str, list[str]] = {
        "catalyst_name": [evidence.evidence_id],
    }
    if _detect_reaction(text):
        field_evidence_ids["reaction"] = [evidence.evidence_id]
    if active_metals:
        field_evidence_ids["active_metals"] = [evidence.evidence_id]
    if promoters:
        field_evidence_ids["promoters"] = [evidence.evidence_id]
    if support:
        field_evidence_ids["support"] = [evidence.evidence_id]
    methods = _extract_characterization_methods(text)
    if methods:
        field_evidence_ids["characterization_methods"] = [evidence.evidence_id]

    return CatalystRecord(
        paper_id=chunk.paper_id,
        catalyst_name=catalyst_name or f"unknown_catalyst_{chunk.chunk_id}",
        reaction=_detect_reaction(text),
        active_metals=active_metals,
        promoters=promoters,
        support=support,
        reaction_conditions=conditions,
        performance_metrics=metrics,
        characterization_methods=methods,
        mechanistic_claims=[],
        evidence_spans=[evidence],
        field_evidence_ids=field_evidence_ids,
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
            payload["paper_id"] = chunk.paper_id
            payload["record_id"] = ""
            payload["evidence_spans"] = []
            payload["field_evidence_ids"] = {}
            record = CatalystRecord.model_validate(payload)
            model_name = getattr(llm, "model_name", "llm")
            evidence = EvidenceSpan(
                paper_id=chunk.paper_id,
                chunk_id=chunk.chunk_id,
                page=chunk.page,
                section=chunk.section,
                text=chunk.text,
                start_char=chunk.metadata.get("start_char"),
                end_char=chunk.metadata.get("end_char"),
                source_path=chunk.source_path,
                doi=chunk.doi,
                extraction_model=str(model_name),
            )
            data = record.model_dump()
            data["record_id"] = ""
            data["paper_id"] = chunk.paper_id
            data["evidence_spans"] = [evidence.model_dump()]
            data["extractor_version"] = f"literature_kg_llm:{model_name}"
            condition_id_map: dict[str, str] = {}
            conditions = []
            for item in data.get("reaction_conditions", []):
                old_id = str(item.get("condition_id") or "")
                item["condition_id"] = ""
                item["evidence_span_id"] = evidence.evidence_id
                condition = ReactionCondition.model_validate(item)
                conditions.append(condition)
                if old_id:
                    condition_id_map[old_id] = condition.condition_id
            data["reaction_conditions"] = [item.model_dump() for item in conditions]
            condition_ids = [item.condition_id for item in conditions]
            for metric_group in ("performance_metrics", "material_properties"):
                for item in data.get(metric_group, []):
                    old_condition_id = str(item.get("condition_id") or "")
                    item["measurement_id"] = ""
                    item["evidence_span_id"] = evidence.evidence_id
                    item["condition_id"] = condition_id_map.get(old_condition_id)
                    if len(condition_ids) == 1 and not item.get("condition_id"):
                        item["condition_id"] = condition_ids[0]
            for item in data.get("synthesis_steps", []):
                item["step_id"] = ""
                item["evidence_span_id"] = evidence.evidence_id
            grounded_fields = {}
            for field in (
                "catalyst_name",
                "reaction",
                "active_metals",
                "promoters",
                "dopants",
                "support",
                "precursors",
                "synthesis_method",
                "characterization_methods",
                "characterization_results",
                "mechanistic_claims",
            ):
                if data.get(field):
                    grounded_fields[field] = [evidence.evidence_id]
            data["field_evidence_ids"] = grounded_fields
            return CatalystRecord.model_validate(data)
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
