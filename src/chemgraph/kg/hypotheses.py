"""Conservative, goal-conditioned, evidence-gated hypothesis generation."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from chemgraph.kg.ontology import PERCENT_QUANTITIES
from chemgraph.kg.query import _graph_context, _node_map, _temperature_deg_c
from chemgraph.kg.schema import HypothesisCard, KGEdge, KGNode
from chemgraph.kg.store import LiteratureKGStore


def _goal_quantity(goal: str) -> str | None:
    lower = goal.lower()
    if "methanol" in lower and "select" in lower:
        return "methanol_selectivity"
    if "co2" in lower and "conversion" in lower:
        return "co2_conversion"
    if "conversion" in lower:
        return "conversion"
    if "select" in lower:
        return "selectivity"
    if "yield" in lower:
        return "yield"
    if "stability" in lower or "time on stream" in lower:
        return "time_on_stream"
    return None


def _goal_max_temperature(goal: str) -> float | None:
    match = re.search(
        r"(?:below|under|<|at most)\s*(\d+(?:\.\d+)?)\s*(?:(?:°|º)\s*)?c\b",
        goal.lower(),
    )
    return float(match.group(1)) if match else None


def _comparable_value(edge: KGEdge) -> float | None:
    value = edge.attributes.get("value")
    if value is None:
        return None
    value = float(value)
    quantity = str(edge.attributes.get("quantity") or "").lower()
    unit = str(edge.attributes.get("unit") or "").lower()
    if quantity in PERCENT_QUANTITIES | {"conversion", "selectivity", "yield"}:
        if unit == "dimensionless":
            return 100.0 * value
        if unit not in {"percent", "%", ""}:
            return None
    return value


def _component_summary(
    edges: list[KGEdge], nodes_by_id: dict[str, KGNode]
) -> dict[str, list[str]]:
    components: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        if edge.relation in {"has_active_metal", "has_promoter", "has_dopant", "supported_on"}:
            target = nodes_by_id.get(edge.target_node_id)
            if target and target.name not in components[edge.source_node_id]:
                components[edge.source_node_id].append(target.name)
    return components


def suggest_hypotheses(
    kg_dir: str | Path,
    *,
    goal: str,
    top_k: int = 5,
) -> dict[str, Any]:
    """Rank comparable, goal-relevant trends and expose counter-evidence."""
    store = LiteratureKGStore(kg_dir)
    nodes = store.load_nodes()
    edges = store.load_edges()
    nodes_by_id = _node_map(nodes)
    observation_to_catalyst, observation_to_paper, conditions, _ = _graph_context(
        edges, nodes_by_id
    )
    components = _component_summary(edges, nodes_by_id)
    requested_quantity = _goal_quantity(goal)
    requested_max_temperature = _goal_max_temperature(goal)

    grouped: dict[tuple[str, str], list[tuple[KGEdge, float]]] = defaultdict(list)
    for edge in edges:
        if edge.relation != "achieves" or edge.source_node_id not in observation_to_catalyst:
            continue
        quantity = str(edge.attributes.get("quantity") or "").lower()
        if requested_quantity and requested_quantity not in quantity:
            continue
        if requested_max_temperature is not None:
            condition_id = edge.attributes.get("condition_id")
            condition = (
                conditions[edge.source_node_id].get(str(condition_id))
                if condition_id
                else None
            )
            temperature = _temperature_deg_c(condition)
            if temperature is None or temperature > requested_max_temperature:
                continue
        value = _comparable_value(edge)
        if value is None:
            continue
        catalyst = observation_to_catalyst[edge.source_node_id]
        grouped[(catalyst.node_id, quantity)].append((edge, value))

    candidates: list[tuple[float, HypothesisCard]] = []
    for (catalyst_id, quantity), observations in grouped.items():
        catalyst = nodes_by_id[catalyst_id]
        values = [value for _, value in observations]
        best_value = max(values)
        central_value = median(values)
        supporting = [item for item in observations if item[1] >= best_value * 0.9]
        counter = [item for item in observations if item[1] < central_value * 0.75]
        paper_ids = {
            observation_to_paper[edge.source_node_id].name
            for edge, _ in supporting
            if edge.source_node_id in observation_to_paper
        }
        support_edge_ids = [edge.edge_id for edge, _ in supporting]
        support_evidence_ids = sorted(
            {eid for edge, _ in supporting for eid in edge.evidence_ids}
        )
        counter_evidence_ids = sorted(
            {eid for edge, _ in counter for eid in edge.evidence_ids}
        )
        comp_text = ", ".join(components.get(catalyst_id, [])) or catalyst.name
        source_phrase = (
            f"{len(paper_ids)} independent papers report"
            if len(paper_ids) != 1
            else "one paper reports"
        )
        unit = str(supporting[0][0].attributes.get("unit") or "")
        claim = (
            f"{comp_text} in {catalyst.name} should be prioritized for controlled validation "
            f"toward {goal}: {source_phrase} {quantity} up to {best_value:g} {unit}. "
            "This is a literature-supported trend candidate, not a causal conclusion."
        )
        mean_confidence = sum(edge.confidence for edge, _ in supporting) / len(supporting)
        plausibility = min(0.9, 0.35 + 0.35 * mean_confidence + 0.08 * min(len(paper_ids), 3))
        novelty = 0.35
        utility = 0.75 if requested_quantity and requested_quantity in quantity else 0.55
        risk = min(0.9, 0.65 - 0.12 * min(len(paper_ids), 3) + 0.08 * bool(counter))
        card = HypothesisCard(
            claim=claim,
            hypothesis_type="trend",
            novelty=novelty,
            plausibility=plausibility,
            expected_utility=utility,
            risk=risk,
            cost=0.45,
            supporting_paths=[
                {
                    "edge_id": edge.edge_id,
                    "observation_id": edge.source_node_id,
                    "paper_id": observation_to_paper.get(edge.source_node_id).name
                    if edge.source_node_id in observation_to_paper
                    else None,
                    "catalyst": catalyst.name,
                    "quantity": quantity,
                    "value": value,
                    "condition_id": edge.attributes.get("condition_id"),
                    "condition": (
                        conditions[edge.source_node_id]
                        .get(str(edge.attributes.get("condition_id")))
                        .attributes
                        if edge.attributes.get("condition_id")
                        and conditions[edge.source_node_id].get(str(edge.attributes.get("condition_id")))
                        else None
                    ),
                    "evidence_ids": edge.evidence_ids,
                }
                for edge, value in supporting
            ],
            supporting_edge_ids=support_edge_ids,
            counter_evidence_ids=counter_evidence_ids,
            suggested_validation=[
                "Re-evaluate all supporting and counter-evidence under one normalized condition schema.",
                "Compute adsorption energies for CO2, HCOO*, CH3O*, and CO on representative surfaces.",
                "Run a fixed-condition stability and carbon-balance check before causal interpretation.",
            ],
            structured_tasks=[
                {
                    "task_type": "evidence_reconciliation",
                    "catalyst": catalyst.name,
                    "supporting_evidence_ids": support_evidence_ids,
                    "counter_evidence_ids": counter_evidence_ids,
                    "requires_human_approval": True,
                },
                {
                    "task_type": "adsorption_energy_screen",
                    "catalyst": catalyst.name,
                    "adsorbates": ["CO2", "HCOO", "CH3O", "CO"],
                    "method": "UMA_then_DFT",
                    "outputs": ["E_ads", "relaxed_structure", "uncertainty"],
                    "requires_human_approval": True,
                },
            ],
        )
        candidates.append((float(card.score or 0.0), card))

    candidates.sort(key=lambda item: (-item[0], item[1].hypothesis_id))
    cards = [card for _, card in candidates[:top_k]]
    return {
        "ok": True,
        "goal": goal,
        "requested_quantity": requested_quantity,
        "requested_max_temperature": requested_max_temperature,
        "num_hypotheses": len(cards),
        "hypotheses": [card.model_dump(mode="json") for card in cards],
    }


def score_hypothesis(card: HypothesisCard) -> dict[str, Any]:
    return {
        "ok": True,
        "hypothesis_id": card.hypothesis_id,
        "score": card.score,
        "components": {
            "novelty": card.novelty,
            "plausibility": card.plausibility,
            "expected_utility": card.expected_utility,
            "risk": card.risk,
            "cost": card.cost,
        },
    }
