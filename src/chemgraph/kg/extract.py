"""Schema-constrained extraction for literature KG records."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

from chemgraph.kg.ontology import CATALYSIS_ELEMENTS
from chemgraph.kg.profiles import ExtractionProfile, load_extraction_profile
from chemgraph.kg.schema import (
    CatalystRecord,
    EvidenceSpan,
    Measurement,
    PaperChunk,
    ReactionCondition,
)

OFFLINE_EXTRACTION_MODELS = {
    "deterministic",
    "regex",
    "offline",
    "none",
    "co2_methanol_regex",
}


EXTRACTION_PROMPT = """You are extracting structured catalysis data from scientific text.

Rules:
- Extract only facts explicitly stated in the provided evidence.
- Do not infer missing values.
- Do not use outside knowledge.
- If a field is not present, use null or [].
- Do not invent persistent paper, chunk, evidence, record, or measurement IDs;
  the application attaches source-controlled provenance after extraction.
- Use local condition labels such as condition_1 only to link each measurement
  to an explicitly stated reaction condition. The application replaces these
  labels with stable condition IDs.
- Preserve raw text for numerical values.
- Preserve comparators such as above, below, approximately, and ranges.
- Keep each measurement linked to the reaction condition under which it was measured.
- Normalize only when unambiguous.
- Return valid JSON matching the CatalystRecord schema.

Evidence:
{chunk_text}

Schema:
{schema}

Optional normalization profile:
{profile}
"""


def load_extraction_llm(model: str):
    """Load an extraction model, or return ``None`` for the offline fallback."""
    if model.lower().replace("-", "_") in OFFLINE_EXTRACTION_MODELS:
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


_SUBSCRIPT_TRANSLATION = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_METRIC_BASES = (
    "faradaic efficiency",
    "space-time yield",
    "time-on-stream",
    "time on stream",
    "turnover frequency",
    "current density",
    "overpotential",
    "productivity",
    "conversion",
    "selectivity",
    "activity",
    "yield",
    "sty",
    "tof",
)
_METRIC_BASE_PATTERN = "|".join(
    re.escape(item) for item in sorted(_METRIC_BASES, key=len, reverse=True)
)
_SUBJECT_TOKEN_PATTERN = r"[A-Za-z][A-Za-z0-9₀-₉+.-]*"
_GENERIC_QUANTITY_PATTERN = (
    rf"(?:(?<![A-Za-z0-9/])(?:{_SUBJECT_TOKEN_PATTERN}\s+){{0,3}}"
    rf"(?:{_METRIC_BASE_PATTERN})s?)"
)


def _normalized_phrase(value: str) -> str:
    text = value.translate(_SUBSCRIPT_TRANSLATION).lower()
    text = text.replace("₂", "2").replace("₃", "3")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _slug(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value)).strip("_")


def _quantity_pattern(profile: ExtractionProfile) -> str:
    aliases = {
        alias.strip()
        for values in profile.quantity_aliases.values()
        for alias in values
        if alias.strip()
    }
    if not aliases:
        return _GENERIC_QUANTITY_PATTERN
    alias_pattern = "|".join(
        re.escape(alias) for alias in sorted(aliases, key=len, reverse=True)
    )
    return rf"(?:{alias_pattern}|{_GENERIC_QUANTITY_PATTERN})"


def _canonical_quantity(
    raw: str,
    profile: ExtractionProfile | None = None,
) -> str:
    """Normalize an arbitrary ``subject + metric`` phrase without a product list."""
    selected = profile or load_extraction_profile()
    normalized = _normalized_phrase(raw)
    for canonical, aliases in selected.quantity_aliases.items():
        if normalized in {_normalized_phrase(alias) for alias in aliases}:
            return canonical

    product_first = re.fullmatch(
        rf"(?P<base>selectivity|yield)\s+(?:towards?|to|for|of)\s+"
        rf"(?P<subject>.+)",
        normalized,
    )
    if product_first:
        subject = re.split(
            r"\b(?:was|is|were|are|reached|remained|maintained|increased|decreased)\b",
            product_first.group("subject"),
            maxsplit=1,
        )[0].strip()
        return f"{_slug(subject)}_{product_first.group('base')}"

    base_match = re.search(rf"(?P<base>{_METRIC_BASE_PATTERN})s?$", normalized)
    if not base_match:
        return _slug(normalized)
    base = base_match.group("base")
    base = {
        "sty": "space time yield",
        "tof": "turnover frequency",
        "time-on-stream": "time on stream",
    }.get(base, base)
    subject = normalized[: base_match.start()].strip()
    if subject:
        subject_tokens = subject.split()
        leading_context = {
            "a",
            "an",
            "and",
            "the",
            "while",
            "its",
            "catalyst",
            "sample",
            "reached",
            "achieved",
            "gave",
            "delivered",
            "showed",
            "reported",
            "observed",
        }
        while subject_tokens and subject_tokens[0] in leading_context:
            subject_tokens.pop(0)
        if len(subject_tokens) > 3:
            subject_tokens = subject_tokens[-3:]
        subject = " ".join(subject_tokens)
        if not subject:
            return _slug(base)
        return f"{_slug(subject)}_{_slug(base)}"
    return _slug(base)


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
        r"(?i:catalyst|sample)\s+([A-Z][A-Za-z0-9/._-]{2,})",
        text,
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


_REACTION_PROCESS_PATTERN = (
    r"(?:steam\s+reforming|dry\s+reforming|hydrogenation|dehydrogenation|"
    r"oxidation|reduction|reforming|synthesis|decomposition|"
    r"(?:oxygen|hydrogen)\s+evolution(?:\s+reaction)?)"
)


def _clean_reaction_phrase(value: str) -> str:
    phrase = value.translate(_SUBSCRIPT_TRANSLATION).strip(" ,;:.")
    phrase = re.split(
        r"\s+(?:at|under|over|using|with)\s+(?=(?:-?\d|the\b|a\b|an\b))",
        phrase,
        maxsplit=1,
        flags=re.I,
    )[0]
    phrase = re.sub(r"^(?:the\s+)?(?:catalytic\s+)?", "", phrase, flags=re.I)
    return re.sub(r"\s+", " ", phrase).strip()


def _explicit_general_reaction(text: str) -> str | None:
    """Return only an explicitly named reaction; do not infer from metrics."""
    normalized = text.translate(_SUBSCRIPT_TRANSLATION)
    marked = re.compile(
        rf"\b(?:during|for|via)\s+(?P<reaction>[^.!?;]{{0,100}}?"
        rf"\b{_REACTION_PROCESS_PATTERN}\b"
        rf"(?:\s+(?:to|into)\s+[A-Za-z0-9₀-₉+.-]+(?:\s+[A-Za-z0-9₀-₉+.-]+){{0,2}})?)",
        flags=re.I,
    )
    match = marked.search(normalized)
    if match:
        return _clean_reaction_phrase(match.group("reaction"))

    standalone = re.compile(
        rf"(?<![A-Za-z0-9/])(?P<reaction>"
        rf"(?:[A-Za-z0-9][A-Za-z0-9₀-₉+./-]*\s+){{1,3}}"
        rf"{_REACTION_PROCESS_PATTERN}"
        rf"(?:\s+(?:to|into)\s+[A-Za-z0-9₀-₉+.-]+(?:\s+[A-Za-z0-9₀-₉+.-]+){{0,2}})?)",
        flags=re.I,
    )
    stopwords = {"during", "for", "via", "tested", "used", "the"}
    for candidate in standalone.finditer(normalized):
        phrase = _clean_reaction_phrase(candidate.group("reaction"))
        tokens = phrase.split()
        while tokens and tokens[0].lower() in stopwords:
            tokens.pop(0)
        if tokens:
            return " ".join(tokens)
    return None


def _detect_reaction(
    text: str,
    profile: ExtractionProfile | None = None,
    quantities: Iterable[str] = (),
) -> str | None:
    selected = profile or load_extraction_profile()
    normalized_text = _normalized_phrase(text)

    # Profile aliases are canonicalization rules, not global chemistry facts.
    for canonical, aliases in selected.reaction_aliases.items():
        for alias in aliases:
            normalized_alias = _normalized_phrase(alias)
            if re.search(rf"\b{re.escape(normalized_alias)}\b", normalized_text):
                return canonical

    explicit = _explicit_general_reaction(text)
    if explicit:
        return explicit

    quantity_set = set(quantities)
    for rule in selected.reaction_inference:
        if not set(rule.all_quantities).issubset(quantity_set):
            continue
        if rule.any_quantities and not quantity_set.intersection(rule.any_quantities):
            continue
        if any(_normalized_phrase(term) not in normalized_text for term in rule.all_terms):
            continue
        return rule.reaction
    return None


def _extract_profile_ratios(
    text: str,
    profile: ExtractionProfile,
) -> dict[str, float]:
    values: dict[str, float] = {}
    for field, aliases in profile.condition_ratio_aliases.items():
        for alias in aliases:
            match = re.search(
                rf"{re.escape(alias)}\s*(?:(?:=|:)|ratio\s*(?:=|:)?)?\s*"
                r"(\d+(?:\.\d+)?)",
                text,
                flags=re.I,
            )
            if match:
                values[field] = float(match.group(1))
                break
    return values


def _extract_conditions(
    text: str,
    evidence_id: str,
    profile: ExtractionProfile,
) -> list[ReactionCondition]:
    normalized = text.translate(_SUBSCRIPT_TRANSLATION)
    temperature_matches = list(re.finditer(
        r"(-?\d+(?:\.\d+)?)\s*(?:(?:°|º)\s*|deg(?:ree)?s?\s*)?(C|K)\b",
        normalized,
        flags=re.I,
    ))
    pressure_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(bar|MPa|kPa|Pa)\b",
        normalized,
        flags=re.I,
    )
    ratios = _extract_profile_ratios(normalized, profile)
    if not temperature_matches and pressure_match is None and not ratios:
        return []
    pressure_unit = pressure_match.group(2) if pressure_match else None
    if pressure_unit:
        pressure_unit = {
            "bar": "bar",
            "mpa": "MPa",
            "kpa": "kPa",
            "pa": "Pa",
        }[pressure_unit.lower()]
    conditions = []
    for temperature_match in temperature_matches or [None]:
        temperature_unit = None
        if temperature_match:
            temperature_unit = (
                "K" if temperature_match.group(2).lower() == "k" else "degC"
            )
        conditions.append(ReactionCondition(
            temperature=(
                float(temperature_match.group(1)) if temperature_match else None
            ),
            temperature_unit=temperature_unit,
            pressure=float(pressure_match.group(1)) if pressure_match else None,
            pressure_unit=pressure_unit,
            h2_co2_ratio=ratios.get("h2_co2_ratio"),
            attributes={
                key: value
                for key, value in ratios.items()
                if key != "h2_co2_ratio"
            },
            evidence_span_id=evidence_id,
        ))
    return list({condition.condition_id: condition for condition in conditions}.values())


def _condition_from_temperature(
    text: str,
    evidence_id: str,
    temperature: str,
    temperature_unit: str,
    temperature_position: int,
    profile: ExtractionProfile,
) -> ReactionCondition:
    """Build a condition around one explicitly paired temperature mention."""
    pressure_matches = list(
        re.finditer(r"(\d+(?:\.\d+)?)\s*(bar|MPa|kPa|Pa)\b", text, flags=re.I)
    )
    pressure_match = min(
        pressure_matches,
        key=lambda match: abs(match.start() - temperature_position),
        default=None,
    )
    pressure_unit = pressure_match.group(2) if pressure_match else None
    if pressure_unit:
        pressure_unit = {
            "bar": "bar",
            "mpa": "MPa",
            "kpa": "kPa",
            "pa": "Pa",
        }[pressure_unit.lower()]
    ratios = _extract_profile_ratios(
        text.translate(_SUBSCRIPT_TRANSLATION),
        profile,
    )
    return ReactionCondition(
        temperature=float(temperature),
        temperature_unit=(
            "K" if temperature_unit.lower() == "k" else "degC"
        ),
        pressure=float(pressure_match.group(1)) if pressure_match else None,
        pressure_unit=pressure_unit,
        h2_co2_ratio=ratios.get("h2_co2_ratio"),
        attributes={
            key: value for key, value in ratios.items() if key != "h2_co2_ratio"
        },
        evidence_span_id=evidence_id,
    )


def _extract_explicit_condition_metric_pairs(
    text: str,
    evidence_id: str,
    profile: ExtractionProfile,
) -> tuple[list[ReactionCondition], list[Measurement]]:
    """Extract metric values explicitly paired with nearby temperatures."""
    if re.search(
        r"\d+(?:\.\d+)?\s*(?:-|–|to)\s*\d+(?:\.\d+)?\s*(?:%|percent)",
        text,
        flags=re.I,
    ):
        # The general measurement parser preserves these as one range; do not
        # reinterpret the upper bound as an independent point measurement.
        return [], []
    temperature_pattern = re.compile(
        r"(?P<temperature>-?\d+(?:\.\d+)?)\s*"
        r"(?:(?:°|º)\s*|deg(?:ree)?s?\s*)?"
        r"(?P<temperature_unit>C|K)\b",
        flags=re.I,
    )
    conditions: dict[str, ReactionCondition] = {}
    metrics: dict[tuple[str, float, str], Measurement] = {}
    quantity_pattern = _quantity_pattern(profile)

    def add_metric(
        raw_quantity: str,
        raw_value: str,
        raw_text: str,
        temperature: str,
        temperature_unit: str,
        temperature_position: int,
    ) -> None:
        if re.search(
            r"\b(?:estimated|predicted|calculated|equilibrium)\b",
            raw_text,
            flags=re.I,
        ):
            return
        condition = _condition_from_temperature(
            text,
            evidence_id,
            temperature,
            temperature_unit,
            temperature_position,
            profile,
        )
        quantity = _canonical_quantity(raw_quantity, profile)
        value = float(raw_value)
        conditions[condition.condition_id] = condition
        metrics[(quantity, value, condition.condition_id)] = Measurement(
            quantity=quantity,
            value=value,
            unit="percent",
            raw_value=raw_text,
            condition_id=condition.condition_id,
            evidence_span_id=evidence_id,
            confidence=0.7,
            attributes={"comparator": "=", "raw_quantity": raw_quantity},
        )

    series_pattern = re.compile(
        rf"(?P<quantity>{quantity_pattern})\s+"
        r"(?:increased|decreased|rose|declined)\s+from\s+"
        r"(?P<value1>\d+(?:\.\d+)?)\s*(?:%|percent)\s+at\s+"
        r"(?P<temperature1>-?\d+(?:\.\d+)?)\s*"
        r"(?:(?:°|º)\s*)?(?P<temperature_unit1>C|K)\b\s*"
        r"(?:to|and\s+(?:reached|fell\s+to|decreased\s+to|increased\s+to))\s*"
        r"(?P<value2>\d+(?:\.\d+)?)\s*(?:%|percent)\s+at\s+"
        r"(?P<temperature2>-?\d+(?:\.\d+)?)\s*"
        r"(?:(?:°|º)\s*)?(?P<temperature_unit2>C|K)\b",
        flags=re.I,
    )
    for match in series_pattern.finditer(text):
        for index in (1, 2):
            add_metric(
                match.group("quantity"),
                match.group(f"value{index}"),
                match.group(0),
                match.group(f"temperature{index}"),
                match.group(f"temperature_unit{index}"),
                match.start(f"temperature{index}"),
            )

    direct_pattern = re.compile(
        rf"(?P<quantity>{quantity_pattern})"
        r"(?:(?![.!?]).){0,50}?"
        r"(?P<value>\d+(?:\.\d+)?)\s*(?:%|percent)\s+at\s+"
        r"(?P<temperature>-?\d+(?:\.\d+)?)\s*"
        r"(?:(?:°|º)\s*)?(?P<temperature_unit>C|K)\b",
        flags=re.I,
    )
    for match in direct_pattern.finditer(text):
        add_metric(
            match.group("quantity"),
            match.group("value"),
            match.group(0),
            match.group("temperature"),
            match.group("temperature_unit"),
            match.start("temperature"),
        )

    value_first_pattern = re.compile(
        r"(?P<value>\d+(?:\.\d+)?)\s*(?:%|percent)\s*"
        rf"(?P<quantity>{quantity_pattern})\b",
        flags=re.I,
    )
    temperature_matches = list(temperature_pattern.finditer(text))
    for match in value_first_pattern.finditer(text):
        if _normalized_phrase(match.group("quantity")).split()[0] in {
            "and",
            "while",
            "but",
        }:
            continue
        nearby_temperatures = [
            candidate
            for candidate in temperature_matches
            if min(
                abs(candidate.end() - match.start()),
                abs(candidate.start() - match.end()),
            ) <= 120
        ]
        if not nearby_temperatures:
            continue
        temperature_match = min(
            nearby_temperatures,
            key=lambda candidate: min(
                abs(candidate.end() - match.start()),
                abs(candidate.start() - match.end()),
            ),
        )
        add_metric(
            match.group("quantity"),
            match.group("value"),
            match.group(0),
            temperature_match.group("temperature"),
            temperature_match.group("temperature_unit"),
            temperature_match.start(),
        )

    return list(conditions.values()), list(metrics.values())


def _extract_measurements(
    text: str,
    evidence_id: str,
    condition_id: str | None,
    profile: ExtractionProfile,
) -> list[Measurement]:
    metrics: list[Measurement] = []
    quantity_pattern = _quantity_pattern(profile)
    metric_pattern = re.compile(
        rf"({quantity_pattern})"
        r"(?:\s*(?:of|was|is|reached))?\s*"
        r"(?:(=|:|above|over|>|below|under|<|~|approximately|about|around)\s*)?"
        r"(\d+(?:\.\d+)?)"
        r"(?:\s*(?:-|–|to)\s*(\d+(?:\.\d+)?))?\s*"
        r"(?:\s*(?:±|\+/-)\s*(\d+(?:\.\d+)?))?\s*"
        r"(%|percent|h|bar|mV|V|mA\s*cm-2|A\s*g-1|s-1|"
        r"g(?:MeOH)?\s*gcat-1\s*h-1|mmol\s*g-1\s*h-1)?",
        flags=re.IGNORECASE,
    )
    for match in metric_pattern.finditer(text):
        raw_quantity, comparator, raw_value, range_max, uncertainty, raw_unit = match.groups()
        quantity = _canonical_quantity(raw_quantity, profile)
        percent_like = quantity.endswith(
            ("_conversion", "_selectivity", "_yield", "_faradaic_efficiency")
        ) or quantity in {"conversion", "selectivity", "yield", "faradaic_efficiency"}
        unit = raw_unit or ("percent" if percent_like else None)
        if unit == "%":
            unit = "percent"
        attributes: dict[str, Any] = {
            "comparator": comparator or "=",
            "raw_quantity": raw_quantity,
        }
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
                quantity=quantity,
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
    product_first_pattern = re.compile(
        r"(selectivity\s+(?:towards?|to|for|of)\s+"
        rf"{_SUBJECT_TOKEN_PATTERN}(?:\s+{_SUBJECT_TOKEN_PATTERN}){{0,2}})"
        r"(?P<preamble>(?:(?![.!?]).){0,100}?)"
        r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>%|percent\b)",
        flags=re.IGNORECASE,
    )
    for match in product_first_pattern.finditer(text):
        preamble = match.group("preamble")
        comparator_match = re.search(
            r"(above|over|below|under|approximately|about|around|~|>|<)",
            preamble,
            flags=re.IGNORECASE,
        )
        comparator = comparator_match.group(1).lower() if comparator_match else "="
        metrics.append(
            Measurement(
                quantity=_canonical_quantity(match.group(1), profile),
                value=float(match.group("value")),
                unit="percent" if match.group("unit") == "%" else match.group("unit"),
                raw_value=match.group(0),
                condition_id=condition_id,
                evidence_span_id=evidence_id,
                confidence=0.65,
                attributes={
                    "comparator": comparator,
                    "raw_quantity": match.group(1),
                },
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
    profile: ExtractionProfile,
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
        explicit_conditions, explicit_metrics = _extract_explicit_condition_metric_pairs(
            segment,
            evidence_id,
            profile,
        )
        if explicit_metrics:
            conditions.extend(explicit_conditions)
            metrics.extend(explicit_metrics)
        segment_conditions = _extract_conditions(segment, evidence_id, profile)
        conditions.extend(segment_conditions)
        condition_id = (
            segment_conditions[0].condition_id
            if len(segment_conditions) == 1
            else None
        )
        for metric in _extract_measurements(
            segment,
            evidence_id,
            condition_id,
            profile,
        ):
            if condition_id:
                metrics.append(metric)
            else:
                pending_metrics.append(metric)
    unique_conditions = list({item.condition_id: item for item in conditions}.values())
    # A condition elsewhere in a chunk does not establish that it applies to
    # an unconditioned measurement. Keep such measurements only when the
    # entire chunk contains no reaction condition.
    if not unique_conditions:
        metrics.extend(pending_metrics)
    unique_metrics = {
        (metric.quantity, metric.value, metric.unit, metric.condition_id): metric
        for metric in metrics
    }
    return unique_conditions, list(unique_metrics.values())


def regex_extract_record(
    chunk: PaperChunk,
    profile: str | ExtractionProfile | None = None,
    *,
    profiles_config: str | Path | None = None,
) -> CatalystRecord | None:
    """Run conservative domain-neutral regex extraction plus an optional profile."""
    selected_profile = load_extraction_profile(profile, profiles_config)
    text = chunk.text
    catalyst_name = _find_catalyst_name(text)
    explicit_metrics = _extract_explicit_condition_metric_pairs(
        text,
        "placeholder",
        selected_profile,
    )[1]
    metrics_or_reaction = (
        _extract_measurements(text, "placeholder", None, selected_profile)
        or explicit_metrics
        or _detect_reaction(text, selected_profile)
    )
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
        extraction_model=selected_profile.extractor_version,
    )
    conditions, metrics = _extract_condition_metric_pairs(
        text,
        evidence.evidence_id,
        selected_profile,
    )
    reaction = _detect_reaction(
        text,
        selected_profile,
        (metric.quantity for metric in metrics),
    )
    active_metals, promoters, support = _classify_catalyst_components(catalyst_name)
    field_evidence_ids: dict[str, list[str]] = {
        "catalyst_name": [evidence.evidence_id],
    }
    if reaction:
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
        reaction=reaction,
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
        extractor_version=selected_profile.extractor_version,
        attributes={"extraction_profile": selected_profile.name},
    )


def llm_extract_record(
    chunk: PaperChunk,
    llm,
    retries: int = 1,
    profile: str | ExtractionProfile | None = None,
    *,
    profiles_config: str | Path | None = None,
) -> CatalystRecord:
    """Extract a CatalystRecord by invoking a LangChain-style chat model."""
    selected_profile = load_extraction_profile(profile, profiles_config)
    schema = CatalystRecord.model_json_schema()
    prompt = EXTRACTION_PROMPT.format(
        chunk_text=chunk.text,
        schema=json.dumps(schema, indent=2),
        profile=(
            json.dumps(
                {
                    "name": selected_profile.name,
                    "quantity_aliases": selected_profile.quantity_aliases,
                    "reaction_aliases": selected_profile.reaction_aliases,
                    "note": (
                        "Normalization vocabulary only. Do not infer a reaction "
                        "that is not explicit in the evidence."
                    ),
                },
                indent=2,
            )
            if selected_profile.name != "general"
            else "None. Extract domain-neutral terms as written in the evidence."
        ),
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
            data["attributes"] = {
                **data.get("attributes", {}),
                "extraction_profile": selected_profile.name,
            }
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
            for metric_group in ("performance_metrics", "material_properties"):
                for item in data.get(metric_group, []):
                    old_condition_id = str(item.get("condition_id") or "")
                    item["measurement_id"] = ""
                    item["evidence_span_id"] = evidence.evidence_id
                    item["condition_id"] = condition_id_map.get(old_condition_id)
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


def _propagate_unambiguous_paper_catalyst_context(
    records: list[CatalystRecord],
) -> list[CatalystRecord]:
    """Ground metric-only chunks with a unique catalyst named in that paper."""
    records_by_paper: dict[str, list[CatalystRecord]] = {}
    for record in records:
        records_by_paper.setdefault(record.paper_id, []).append(record)

    updated: list[CatalystRecord] = []
    for paper_records in records_by_paper.values():
        named_records = [
            record
            for record in paper_records
            if not record.catalyst_name.startswith("unknown_catalyst_")
        ]
        catalyst_names = {record.catalyst_name for record in named_records}
        exemplar = None
        if len(catalyst_names) == 1:
            exemplar = max(
                named_records,
                key=lambda record: (
                    len(record.active_metals),
                    bool(record.support),
                    record.confidence,
                ),
            )

        for record in paper_records:
            if (
                exemplar is None
                or not record.catalyst_name.startswith("unknown_catalyst_")
                or not record.performance_metrics
            ):
                updated.append(record)
                continue

            data = record.model_dump()
            data["record_id"] = ""
            for field in (
                "catalyst_name",
                "canonical_catalyst_name",
                "active_metals",
                "promoters",
                "dopants",
                "support",
            ):
                data[field] = getattr(exemplar, field)

            evidence_by_id = {
                span.evidence_id: span.model_dump()
                for span in [*record.evidence_spans, *exemplar.evidence_spans]
            }
            data["evidence_spans"] = list(evidence_by_id.values())
            field_evidence_ids = dict(data.get("field_evidence_ids") or {})
            for field in (
                "catalyst_name",
                "active_metals",
                "promoters",
                "dopants",
                "support",
            ):
                if getattr(exemplar, field):
                    field_evidence_ids[field] = exemplar.field_evidence_ids.get(
                        field,
                        [span.evidence_id for span in exemplar.evidence_spans],
                    )
            data["field_evidence_ids"] = field_evidence_ids
            data["attributes"] = {
                **data.get("attributes", {}),
                "catalyst_context_propagated": True,
                "catalyst_context_record_id": exemplar.record_id,
            }
            updated.append(CatalystRecord.model_validate(data))
    return updated


def extract_records_from_chunks(
    chunks: Iterable[PaperChunk],
    *,
    llm=None,
    retries: int = 1,
    profile: str | ExtractionProfile | None = None,
    profiles_config: str | Path | None = None,
) -> list[CatalystRecord]:
    """Extract records with schema-constrained LLM or profiled regex parsing."""
    selected_profile = load_extraction_profile(profile, profiles_config)
    records: list[CatalystRecord] = []
    for chunk in chunks:
        record = (
            llm_extract_record(
                chunk,
                llm,
                retries=retries,
                profile=selected_profile,
            )
            if llm
            else regex_extract_record(chunk, selected_profile)
        )
        if record is not None:
            records.append(record)
    if selected_profile.propagate_unique_catalyst:
        return _propagate_unambiguous_paper_catalyst_context(records)
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
