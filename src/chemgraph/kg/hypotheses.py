"""Autonomous but evidence-gated hypothesis generation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from chemgraph.kg.schema import HypothesisCard, KGEdge, KGNode
from chemgraph.kg.store import LiteratureKGStore


def _node_map(nodes: list[KGNode]) -> dict[str, KGNode]:
    return {node.node_id: node for node in nodes}


def _component_summary(edges: list[KGEdge], nodes_by_id: dict[str, KGNode]) -> dict[str, list[str]]:
    components: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        if edge.relation in {"has_active_metal", "has_promoter", "has_dopant", "supported_on"}:
            target = nodes_by_id.get(edge.target_node_id)
            if target:
                components[edge.source_node_id].append(target.name)
    return components


def suggest_hypotheses(
    kg_dir: str | Path,
    *,
    goal: str,
    top_k: int = 5,
) -> dict[str, Any]:
    """Generate simple missing-link hypothesis cards from graph evidence."""
    store = LiteratureKGStore(kg_dir)
    nodes = store.load_nodes()
    edges = store.load_edges()
    nodes_by_id = _node_map(nodes)
    components = _component_summary(edges, nodes_by_id)

    metric_edges = [
        edge
        for edge in edges
        if edge.relation == "achieves"
        and any(term in str(edge.attributes.get("quantity", "")).lower() for term in ["selectivity", "conversion", "yield"])
    ]
    metric_edges.sort(
        key=lambda edge: (
            float(edge.attributes.get("value") or 0.0),
            edge.confidence,
        ),
        reverse=True,
    )

    cards: list[HypothesisCard] = []
    for edge in metric_edges[:top_k]:
        catalyst = nodes_by_id.get(edge.source_node_id)
        metric = nodes_by_id.get(edge.target_node_id)
        if not catalyst or not metric:
            continue
        comp_text = ", ".join(components.get(catalyst.node_id, [])) or catalyst.name
        quantity = edge.attributes.get("quantity", "performance")
        value = edge.attributes.get("value")
        unit = edge.attributes.get("unit") or ""
        claim = (
            f"{comp_text} motifs related to {catalyst.name} should be prioritized for "
            f"{goal} because the KG contains evidence for {quantity}={value} {unit}."
        )
        plausibility = min(0.95, 0.45 + 0.5 * edge.confidence)
        novelty = 0.65 if len(components.get(catalyst.node_id, [])) >= 2 else 0.45
        utility = 0.70 if "methanol" in goal.lower() or "selectivity" in goal.lower() else 0.55
        risk = 0.35 if edge.evidence_ids else 0.75
        cost = 0.45
        cards.append(
            HypothesisCard(
                claim=claim,
                hypothesis_type="missing_link",
                novelty=novelty,
                plausibility=plausibility,
                expected_utility=utility,
                risk=risk,
                cost=cost,
                supporting_paths=[
                    {
                        "edge_id": edge.edge_id,
                        "source": catalyst.name,
                        "relation": edge.relation,
                        "target": metric.name,
                        "evidence_ids": edge.evidence_ids,
                    }
                ],
                supporting_edge_ids=[edge.edge_id],
                counter_evidence_ids=[],
                suggested_validation=[
                    "Compute adsorption energies for CO2, HCOO*, CH3O*, and CO on representative surfaces.",
                    "Simulate XANES/EXAFS descriptors for the proposed oxidation-state fingerprint.",
                    "Run a fixed-condition literature or experiment check for stability and carbon balance.",
                ],
                structured_tasks=[
                    {
                        "task_type": "adsorption_energy_screen",
                        "catalyst": catalyst.name,
                        "adsorbates": ["CO2", "HCOO", "CH3O", "CO"],
                        "method": "UMA_then_DFT",
                        "outputs": ["E_ads", "relaxed_structure", "uncertainty"],
                    },
                    {
                        "task_type": "xanes_descriptor_generation",
                        "catalyst": catalyst.name,
                        "outputs": ["oxidation_state_fingerprint", "white_line_intensity"],
                    },
                ],
            )
        )

    return {
        "ok": True,
        "goal": goal,
        "num_hypotheses": len(cards),
        "hypotheses": [card.model_dump(mode="json") for card in cards],
    }


def score_hypothesis(card: HypothesisCard) -> dict[str, Any]:
    """Return the scalar score and score components for one hypothesis card."""
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
