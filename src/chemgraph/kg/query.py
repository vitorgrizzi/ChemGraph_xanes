"""Graph, semantic, and hybrid query helpers for the literature KG."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from chemgraph.kg.schema import EvidenceSpan, KGEdge, KGNode
from chemgraph.kg.store import LiteratureKGStore


def _terms(query: str) -> list[str]:
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z0-9]+", query)
        if len(token) > 2
    ]


def _score_text(query_terms: list[str], text: str) -> float:
    lower = text.lower()
    if not query_terms:
        return 0.0
    return sum(lower.count(term) for term in query_terms) / len(query_terms)


def _node_map(nodes: list[KGNode]) -> dict[str, KGNode]:
    return {node.node_id: node for node in nodes}


def _evidence_map(evidence: list[EvidenceSpan]) -> dict[str, EvidenceSpan]:
    return {span.evidence_id: span for span in evidence}


def semantic_search(
    kg_dir: str | Path,
    query: str,
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    """Lexical semantic fallback over evidence spans."""
    store = LiteratureKGStore(kg_dir)
    terms = _terms(query)
    scored = []
    for span in store.load_evidence():
        score = _score_text(terms, span.text)
        if score > 0:
            scored.append((score, span))
    scored.sort(key=lambda item: item[0], reverse=True)
    return {
        "ok": True,
        "query": query,
        "num_results": min(len(scored), top_k),
        "results": [
            {
                "score": score,
                "evidence": span.model_dump(mode="json"),
            }
            for score, span in scored[:top_k]
        ],
    }


def graph_query(
    kg_dir: str | Path,
    *,
    relation: str | None = None,
    catalyst_contains: str | None = None,
    metric_quantity: str | None = None,
    min_value: float | None = None,
    max_temperature: float | None = None,
    top_k: int = 20,
) -> dict[str, Any]:
    """Filter KG edges and return source/target/evidence paths."""
    store = LiteratureKGStore(kg_dir)
    nodes = store.load_nodes()
    edges = store.load_edges()
    evidence_lookup = _evidence_map(store.load_evidence())
    nodes_by_id = _node_map(nodes)
    condition_by_catalyst: dict[str, list[KGEdge]] = defaultdict(list)
    for edge in edges:
        if edge.relation == "tested_under":
            condition_by_catalyst[edge.source_node_id].append(edge)

    results = []
    for edge in edges:
        if relation and edge.relation != relation:
            continue
        source = nodes_by_id.get(edge.source_node_id)
        target = nodes_by_id.get(edge.target_node_id)
        if source is None or target is None:
            continue
        if catalyst_contains and catalyst_contains.lower() not in source.name.lower():
            continue
        if metric_quantity:
            quantity = str(edge.attributes.get("quantity") or target.attributes.get("quantity") or "")
            if metric_quantity.lower() not in quantity.lower():
                continue
        if min_value is not None:
            value = edge.attributes.get("value", target.attributes.get("value"))
            if value is None or float(value) < min_value:
                continue
        if max_temperature is not None:
            temperatures = []
            for cond_edge in condition_by_catalyst.get(edge.source_node_id, []):
                cond_node = nodes_by_id.get(cond_edge.target_node_id)
                if cond_node:
                    temp = cond_node.attributes.get("temperature")
                    if temp is not None:
                        temperatures.append(float(temp))
            if temperatures and min(temperatures) > max_temperature:
                continue

        evidence = [
            evidence_lookup[eid].model_dump(mode="json")
            for eid in edge.evidence_ids
            if eid in evidence_lookup
        ]
        results.append(
            {
                "edge": edge.model_dump(mode="json"),
                "source": source.model_dump(mode="json"),
                "target": target.model_dump(mode="json"),
                "evidence": evidence,
            }
        )
        if len(results) >= top_k:
            break
    return {"ok": True, "num_results": len(results), "results": results}


def _parse_metric_query(query: str) -> dict[str, Any]:
    lower = query.lower()
    parsed: dict[str, Any] = {}
    if "selectivity" in lower:
        parsed["metric_quantity"] = "selectivity"
    if "methanol" in lower and "selectivity" in lower:
        parsed["metric_quantity"] = "methanol_selectivity"
    if "conversion" in lower:
        parsed["metric_quantity"] = "conversion"
    above = re.search(r"(?:above|over|>|greater than)\s*(\d+(?:\.\d+)?)\s*%?", lower)
    if above:
        parsed["min_value"] = float(above.group(1))
    below_temp = re.search(r"(?:below|under|<)\s*(\d+(?:\.\d+)?)\s*(?:c|°c)", lower)
    if below_temp:
        parsed["max_temperature"] = float(below_temp.group(1))
    return parsed


def hybrid_query(
    kg_dir: str | Path,
    query: str,
    *,
    top_k: int = 10,
) -> dict[str, Any]:
    """Combine graph filters with evidence-span lexical retrieval."""
    parsed = _parse_metric_query(query)
    graph_results = graph_query(
        kg_dir,
        relation="achieves" if parsed.get("metric_quantity") else None,
        top_k=top_k,
        **parsed,
    )
    semantic_results = semantic_search(kg_dir, query, top_k=top_k)
    return {
        "ok": True,
        "query": query,
        "graph": graph_results,
        "semantic": semantic_results,
    }


def get_evidence(kg_dir: str | Path, evidence_id: str) -> dict[str, Any]:
    store = LiteratureKGStore(kg_dir)
    span = store.get_evidence(evidence_id)
    if span is None:
        return {"ok": False, "error": f"Evidence not found: {evidence_id}"}
    return {"ok": True, "evidence": span.model_dump(mode="json")}


def export_training_table(
    kg_dir: str | Path,
    out: str | Path,
    *,
    target_quantity: str = "methanol_selectivity",
) -> dict[str, Any]:
    """Export a simple catalyst-performance table for ML prototypes."""
    store = LiteratureKGStore(kg_dir)
    nodes_by_id = _node_map(store.load_nodes())
    rows: dict[str, dict[str, Any]] = {}
    for edge in store.load_edges():
        if edge.relation not in {"achieves", "has_active_metal", "has_promoter", "supported_on", "tested_under"}:
            continue
        source = nodes_by_id.get(edge.source_node_id)
        target = nodes_by_id.get(edge.target_node_id)
        if not source or source.node_type != "CatalystSystem" or not target:
            continue
        row = rows.setdefault(
            source.node_id,
            {
                "catalyst_id": source.node_id,
                "catalyst_name": source.name,
                "active_metal": "",
                "promoter": "",
                "support": "",
                "reaction_temperature": "",
                "pressure": "",
                "literature_confidence": source.confidence,
                "evidence_count": 0,
            },
        )
        row["evidence_count"] += len(edge.evidence_ids)
        if edge.relation == "has_active_metal":
            row["active_metal"] = target.name
        elif edge.relation == "has_promoter":
            row["promoter"] = ";".join(filter(None, [row.get("promoter"), target.name]))
        elif edge.relation == "supported_on":
            row["support"] = target.name
        elif edge.relation == "tested_under":
            row["reaction_temperature"] = target.attributes.get("temperature", "")
            row["pressure"] = target.attributes.get("pressure", "")
        elif edge.relation == "achieves":
            quantity = edge.attributes.get("quantity")
            if quantity:
                row[quantity] = edge.attributes.get("value")
                row[f"{quantity}_unit"] = edge.attributes.get("unit")
            if quantity == target_quantity:
                row["target"] = edge.attributes.get("value")

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows.values() for key in row})
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows.values():
            writer.writerow(row)
    return {"ok": True, "path": str(out_path), "n_rows": len(rows)}
