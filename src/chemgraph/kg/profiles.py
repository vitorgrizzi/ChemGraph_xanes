"""Configurable vocabularies for deterministic literature extraction."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class ReactionInferenceRule(BaseModel):
    """Profile-scoped rule for inferring a reaction from extracted evidence."""

    reaction: str
    all_quantities: list[str] = Field(default_factory=list)
    any_quantities: list[str] = Field(default_factory=list)
    all_terms: list[str] = Field(default_factory=list)


class ExtractionProfile(BaseModel):
    """Domain vocabulary layered on top of the generic regex extractor."""

    name: str = "general"
    description: str = "Domain-neutral deterministic extraction."
    extractor_version: str = "literature_kg_general_regex_v1"
    quantity_aliases: dict[str, list[str]] = Field(default_factory=dict)
    reaction_aliases: dict[str, list[str]] = Field(default_factory=dict)
    reaction_inference: list[ReactionInferenceRule] = Field(default_factory=list)
    condition_ratio_aliases: dict[str, list[str]] = Field(default_factory=dict)
    propagate_unique_catalyst: bool = False


GENERAL_EXTRACTION_PROFILE = ExtractionProfile()
DEFAULT_PROFILE_CONFIG = Path(__file__).with_name("data") / "extraction_profiles.yaml"


def normalize_profile_name(name: str | None) -> str:
    normalized = (name or "general").strip().lower().replace("-", "_")
    aliases = {
        "default": "general",
        "none": "general",
        "co2_methanol_regex": "co2_methanol",
    }
    return aliases.get(normalized, normalized)


def profile_name_for_model(model: str, profile: str | None = None) -> str:
    """Resolve explicit regex model aliases without changing LLM model names."""
    model_name = model.strip().lower().replace("-", "_")
    if model_name == "co2_methanol_regex":
        requested = normalize_profile_name(profile)
        if requested not in {"general", "co2_methanol"}:
            raise ValueError(
                "co2_methanol_regex cannot be combined with profile "
                f"{profile!r}."
            )
        return "co2_methanol"
    return normalize_profile_name(profile)


def load_extraction_profile(
    profile: str | ExtractionProfile | None = None,
    config_path: str | Path | None = None,
) -> ExtractionProfile:
    """Load an extraction profile, keeping the generic path dependency-free."""
    if isinstance(profile, ExtractionProfile):
        return profile

    name = normalize_profile_name(profile)
    if name == "general" and config_path is None:
        return GENERAL_EXTRACTION_PROFILE

    path = Path(config_path) if config_path is not None else DEFAULT_PROFILE_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"Extraction profile config does not exist: {path}")
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "YAML extraction profiles require PyYAML; install the kg extra."
        ) from exc

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    profiles = payload.get("profiles", payload)
    if not isinstance(profiles, dict) or name not in profiles:
        available = sorted(profiles) if isinstance(profiles, dict) else []
        raise ValueError(
            f"Unknown extraction profile {name!r}. Available profiles: {available}."
        )
    data = profiles[name]
    if not isinstance(data, dict):
        raise ValueError(f"Extraction profile {name!r} must be a mapping.")
    return ExtractionProfile.model_validate({"name": name, **data})
