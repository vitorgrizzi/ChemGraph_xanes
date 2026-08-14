"""Normalization helpers for catalyst records."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from chemgraph.kg.schema import CatalystRecord, Measurement


DEFAULT_SYNONYMS = {
    # Domain vocabularies belong in an explicit synonyms/profile file.
    "molecules": {},
    "reactions": {},
    "units": {
        "%": "percent",
        "pct": "percent",
        "c": "degC",
        "deg c": "degC",
        "bar": "bar",
        "h": "h",
    },
}


def load_synonyms(path: str | Path | None = None) -> dict[str, Any]:
    """Load synonym mappings from YAML when available, else defaults."""
    if path is None:
        return DEFAULT_SYNONYMS

    yaml_path = Path(path)
    if not yaml_path.exists():
        return DEFAULT_SYNONYMS
    try:
        import yaml
    except ImportError:
        return DEFAULT_SYNONYMS

    with yaml_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    merged = {section: dict(values) for section, values in DEFAULT_SYNONYMS.items()}
    for section, values in loaded.items():
        if isinstance(values, dict):
            merged.setdefault(section, {}).update(values)
    return merged


def normalize_unit(unit: str | None, synonyms: dict[str, Any] | None = None) -> str | None:
    if unit is None:
        return None
    syn = synonyms or DEFAULT_SYNONYMS
    key = unit.strip().lower()
    return syn.get("units", {}).get(key, unit.strip())


def normalize_catalyst_name(
    name: str,
    synonyms: dict[str, Any] | None = None,
) -> str:
    """Normalize common catalyst separators without changing chemistry."""
    clean = re.sub(r"\s+", "", name.strip())
    clean = clean.replace("\\", "/")
    clean = re.sub(r"-supported-on-", "/", clean, flags=re.IGNORECASE)
    clean = re.sub(r"supportedon", "/", clean, flags=re.IGNORECASE)
    syn = synonyms or DEFAULT_SYNONYMS
    return syn.get("catalysts", {}).get(clean.lower(), clean)


def normalize_reaction_name(
    reaction: str | None,
    synonyms: dict[str, Any] | None = None,
) -> str | None:
    if reaction is None:
        return None
    syn = synonyms or DEFAULT_SYNONYMS
    key = reaction.strip().lower()
    return syn.get("reactions", {}).get(key, reaction.strip())


def normalize_measurement(
    measurement: Measurement,
    synonyms: dict[str, Any] | None = None,
) -> Measurement:
    data = measurement.model_dump()
    data["unit"] = normalize_unit(measurement.unit, synonyms)
    quantity = measurement.quantity.strip().lower()
    quantity = quantity.replace(" ", "_").replace("-", "_")
    quantity = quantity.replace("co₂", "co2")
    data["quantity"] = quantity
    return Measurement.model_validate(data)


def normalize_record(
    record: CatalystRecord,
    synonyms: dict[str, Any] | None = None,
) -> CatalystRecord:
    """Return a normalized copy of a catalyst record."""
    syn = synonyms or DEFAULT_SYNONYMS
    data = record.model_dump()
    data["canonical_catalyst_name"] = normalize_catalyst_name(
        record.canonical_catalyst_name or record.catalyst_name,
        syn,
    )
    data["reaction"] = normalize_reaction_name(record.reaction, syn)
    data["active_metals"] = sorted({item.strip() for item in record.active_metals if item})
    data["promoters"] = sorted({item.strip() for item in record.promoters if item})
    data["dopants"] = sorted({item.strip() for item in record.dopants if item})
    if record.support:
        data["support"] = normalize_catalyst_name(record.support, syn)
    data["performance_metrics"] = [
        normalize_measurement(metric, syn).model_dump()
        for metric in record.performance_metrics
    ]
    data["material_properties"] = [
        normalize_measurement(metric, syn).model_dump()
        for metric in record.material_properties
    ]
    return CatalystRecord.model_validate(data)


def normalize_records(
    records: list[CatalystRecord],
    synonyms_path: str | Path | None = None,
) -> list[CatalystRecord]:
    synonyms = load_synonyms(synonyms_path)
    return [normalize_record(record, synonyms) for record in records]
