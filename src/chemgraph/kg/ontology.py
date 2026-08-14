"""Small catalysis ontology used by the literature KG MVP."""

NODE_TYPES = {
    "Paper",
    "Chunk",
    "Observation",
    "Reaction",
    "CatalystSystem",
    "CatalystComponent",
    "ActiveMetal",
    "Promoter",
    "Dopant",
    "Support",
    "Precursor",
    "SynthesisMethod",
    "SynthesisStep",
    "Treatment",
    "ReactionCondition",
    "PerformanceMetric",
    "CharacterizationMethod",
    "CharacterizationResult",
    "MaterialProperty",
    "MechanisticClaim",
    "Intermediate",
    "Descriptor",
    "Computation",
    "Spectrum",
    "Dataset",
    "EvidenceSpan",
    "Hypothesis",
}

RELATION_TYPES = {
    "reports",
    "uses_catalyst",
    "studies_reaction",
    "contains_chunk",
    "supports_fact",
    "has_component",
    "has_active_metal",
    "has_promoter",
    "has_dopant",
    "supported_on",
    "synthesized_by",
    "uses_precursor",
    "has_step",
    "has_treatment",
    "tested_for",
    "tested_under",
    "achieves",
    "characterized_by",
    "produces",
    "has_property",
    "has_descriptor",
    "involves",
    "supported_by",
    "contradicted_by",
    "computes",
    "indicates",
    "suggests_validation",
}

# Backward-compatible extension point for exact names that do not follow one
# of the generic suffixes below. Domain profiles should not mutate this set.
PERCENT_QUANTITIES: set[str] = set()

PERCENT_QUANTITY_SUFFIXES = (
    "_conversion",
    "_selectivity",
    "_yield",
    "_faradaic_efficiency",
)


def is_percent_quantity(quantity: str) -> bool:
    """Recognize percent-like metric families without enumerating products."""
    normalized = quantity.strip().lower()
    return normalized in PERCENT_QUANTITIES | {
        "conversion",
        "selectivity",
        "yield",
        "faradaic_efficiency",
    } or normalized.endswith(PERCENT_QUANTITY_SUFFIXES)

CATALYSIS_ELEMENTS = {
    "Ag",
    "Al",
    "Au",
    "Ce",
    "Co",
    "Cr",
    "Cu",
    "Fe",
    "Ga",
    "In",
    "K",
    "La",
    "Mn",
    "Mo",
    "Ni",
    "Pd",
    "Pt",
    "Re",
    "Rh",
    "Ru",
    "Sn",
    "Ti",
    "V",
    "W",
    "Y",
    "Zn",
    "Zr",
}
