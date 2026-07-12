"""Observation-aware graph, retrieval, and export helpers for the literature KG."""

from __future__ import annotations

import csv
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from chemgraph.kg.schema import EvidenceSpan, KGEdge, KGNode
from chemgraph.kg.store import LiteratureKGStore


def _terms(query: str) -> list[str]:
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z0-9]+", query.replace("CO₂", "CO2"))
        if len(token) > 2
    ]


def _node_map(nodes: list[KGNode]) -> dict[str, KGNode]:
    return {node.node_id: node for node in nodes}


def _evidence_map(evidence: list[EvidenceSpan]) -> dict[str, EvidenceSpan]:
    return {span.evidence_id: span for span in evidence}


def _bm25_scores(query: str, spans: list[EvidenceSpan]) -> list[tuple[float, EvidenceSpan]]:
    query_terms = _terms(query)
    if not query_terms or not spans:
        return []
    documents = [_terms(span.text) for span in spans]
    avg_length = sum(len(document) for document in documents) / max(1, len(documents))
    document_frequency = {
        term: sum(term in set(document) for document in documents) for term in set(query_terms)
    }
    scored: list[tuple[float, EvidenceSpan]] = []
    k1 = 1.5
    b = 0.75
    for span, document in zip(spans, documents):
        frequencies = {term: document.count(term) for term in query_terms}
        score = 0.0
        for term in query_terms:
            frequency = frequencies[term]
            if not frequency:
                continue
            df = document_frequency[term]
            idf = math.log(1.0 + (len(documents) - df + 0.5) / (df + 0.5))
            denominator = frequency + k1 * (
                1.0 - b + b * len(document) / max(avg_length, 1.0)
            )
            score += idf * frequency * (k1 + 1.0) / denominator
        if score > 0:
            scored.append((score, span))
    return sorted(scored, key=lambda item: (-item[0], item[1].evidence_id))


def _vector_scores(
    query: str,
    spans: list[EvidenceSpan],
    model_name: str,
) -> list[tuple[float, EvidenceSpan]]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Vector KG retrieval requires sentence-transformers; install the kg extra."
        ) from exc
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        [query, *[span.text for span in spans]],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    query_embedding = embeddings[0]
    scored = [
        (float(query_embedding @ embedding), span)
        for embedding, span in zip(embeddings[1:], spans)
    ]
    return sorted(scored, key=lambda item: (-item[0], item[1].evidence_id))


def semantic_search(
    kg_dir: str | Path,
    query: str,
    *,
    top_k: int = 5,
    embedding_model: str | None = None,
) -> dict[str, Any]:
    """Retrieve evidence with BM25 or an explicitly requested embedding model."""
    store = LiteratureKGStore(kg_dir)
    spans = store.load_evidence()
    model_name = embedding_model or os.environ.get("CHEMGRAPH_KG_EMBEDDING_MODEL")
    if model_name:
        scored = _vector_scores(query, spans, model_name)
        method = "vector"
    else:
        scored = _bm25_scores(query, spans)
        method = "bm25"
    return {
        "ok": True,
        "query": query,
        "method": method,
        "embedding_model": model_name,
        "num_results": min(len(scored), top_k),
        "results": [
            {
                "rank": rank,
                "score": score,
                "evidence": span.model_dump(mode="json"),
            }
            for rank, (score, span) in enumerate(scored[:top_k], start=1)
        ],
    }


def _graph_context(edges: list[KGEdge], nodes_by_id: dict[str, KGNode]):
    observation_to_catalyst: dict[str, KGNode] = {}
    observation_to_paper: dict[str, KGNode] = {}
    conditions_by_observation: dict[str, dict[str, KGNode]] = defaultdict(dict)
    reactions_by_observation: dict[str, list[KGNode]] = defaultdict(list)
    for edge in edges:
        target = nodes_by_id.get(edge.target_node_id)
        source = nodes_by_id.get(edge.source_node_id)
        if not source or not target:
            continue
        if edge.relation == "uses_catalyst" and source.node_type == "Observation":
            observation_to_catalyst[source.node_id] = target
        elif edge.relation == "reports" and target.node_type == "Observation":
            observation_to_paper[target.node_id] = source
        elif edge.relation == "tested_under" and source.node_type == "Observation":
            condition_id = str(target.attributes.get("condition_id") or target.canonical_name)
            conditions_by_observation[source.node_id][condition_id] = target
        elif edge.relation == "tested_for" and source.node_type == "Observation":
            reactions_by_observation[source.node_id].append(target)
    return (
        observation_to_catalyst,
        observation_to_paper,
        conditions_by_observation,
        reactions_by_observation,
    )


def _temperature_deg_c(condition: KGNode | None) -> float | None:
    if condition is None:
        return None
    value = condition.attributes.get("temperature")
    if value is None:
        return None
    value = float(value)
    unit = str(condition.attributes.get("temperature_unit") or "degC").lower()
    if unit in {"k", "kelvin"}:
        return value - 273.15
    return value


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
    """Filter graph paths while preserving observation-condition linkage."""
    store = LiteratureKGStore(kg_dir)
    nodes = store.load_nodes()
    edges = store.load_edges()
    evidence_lookup = _evidence_map(store.load_evidence())
    nodes_by_id = _node_map(nodes)
    (
        observation_to_catalyst,
        observation_to_paper,
        conditions_by_observation,
        reactions_by_observation,
    ) = _graph_context(edges, nodes_by_id)

    results = []
    for edge in edges:
        if relation and edge.relation != relation:
            continue
        raw_source = nodes_by_id.get(edge.source_node_id)
        target = nodes_by_id.get(edge.target_node_id)
        if raw_source is None or target is None:
            continue
        observation = raw_source if raw_source.node_type == "Observation" else None
        catalyst = observation_to_catalyst.get(raw_source.node_id) if observation else None
        display_source = catalyst or raw_source
        if catalyst_contains and catalyst_contains.lower() not in display_source.name.lower():
            continue
        if metric_quantity:
            quantity = str(edge.attributes.get("quantity") or target.attributes.get("quantity") or "")
            if metric_quantity.lower() not in quantity.lower():
                continue
        if min_value is not None:
            value = edge.attributes.get("value", target.attributes.get("value"))
            comparator = str(edge.attributes.get("attributes", {}).get("comparator") or edge.attributes.get("comparator") or "=")
            if value is None or comparator in {"below", "under", "<"} or float(value) < min_value:
                continue

        condition = None
        if observation and edge.relation == "achieves":
            condition_id = edge.attributes.get("condition_id") or target.attributes.get("condition_id")
            if condition_id:
                condition = conditions_by_observation[observation.node_id].get(str(condition_id))
        if max_temperature is not None:
            temperature = _temperature_deg_c(condition)
            if temperature is None or temperature > max_temperature:
                continue

        evidence = [
            evidence_lookup[eid].model_dump(mode="json")
            for eid in edge.evidence_ids
            if eid in evidence_lookup
        ]
        results.append(
            {
                "edge": edge.model_dump(mode="json"),
                "source": display_source.model_dump(mode="json"),
                "observation": observation.model_dump(mode="json") if observation else None,
                "paper": (
                    observation_to_paper.get(observation.node_id).model_dump(mode="json")
                    if observation and observation.node_id in observation_to_paper
                    else None
                ),
                "condition": condition.model_dump(mode="json") if condition else None,
                "reactions": [
                    node.model_dump(mode="json")
                    for node in reactions_by_observation.get(observation.node_id, [])
                ] if observation else [],
                "target": target.model_dump(mode="json"),
                "evidence": evidence,
            }
        )
    results.sort(
        key=lambda item: (
            -float(item["edge"].get("confidence") or 0.0),
            -float(item["edge"].get("attributes", {}).get("value") or 0.0),
            item["edge"]["edge_id"],
        )
    )
    results = results[:top_k]
    return {"ok": True, "num_results": len(results), "results": results}


def _parse_metric_query(query: str) -> dict[str, Any]:
    lower = query.lower().replace("co₂", "co2")
    parsed: dict[str, Any] = {}
    if "selectivity" in lower:
        parsed["metric_quantity"] = "selectivity"
    if "methanol" in lower and "selectivity" in lower:
        parsed["metric_quantity"] = "methanol_selectivity"
    if "conversion" in lower:
        parsed["metric_quantity"] = "conversion"
    above = re.search(r"(?:above|over|>|greater than|at least)\s*(\d+(?:\.\d+)?)\s*%?", lower)
    if above:
        parsed["min_value"] = float(above.group(1))
    below_temp = re.search(
        r"(?:below|under|<|at most)\s*(\d+(?:\.\d+)?)\s*(?:(?:°|º)\s*)?c\b",
        lower,
    )
    if below_temp:
        parsed["max_temperature"] = float(below_temp.group(1))
    return parsed


def hybrid_query(
    kg_dir: str | Path,
    query: str,
    *,
    top_k: int = 10,
    embedding_model: str | None = None,
) -> dict[str, Any]:
    """Fuse graph-path and evidence-retrieval rankings with reciprocal rank fusion."""
    parsed = _parse_metric_query(query)
    graph_results = (
        graph_query(
            kg_dir,
            relation="achieves" if parsed.get("metric_quantity") else None,
            top_k=top_k,
            **parsed,
        )
        if parsed
        else {"ok": True, "num_results": 0, "results": []}
    )
    retrieval_results = semantic_search(
        kg_dir,
        query,
        top_k=top_k,
        embedding_model=embedding_model,
    )
    fused: dict[str, dict[str, Any]] = {}
    for rank, result in enumerate(graph_results["results"], start=1):
        for evidence in result["evidence"]:
            item = fused.setdefault(
                evidence["evidence_id"],
                {"evidence": evidence, "score": 0.0, "graph_hits": []},
            )
            item["score"] += 1.0 / (60 + rank)
            item["graph_hits"].append(result["edge"]["edge_id"])
    for rank, result in enumerate(retrieval_results["results"], start=1):
        evidence = result["evidence"]
        item = fused.setdefault(
            evidence["evidence_id"],
            {"evidence": evidence, "score": 0.0, "graph_hits": []},
        )
        item["score"] += 1.0 / (60 + rank)
    fused_results = sorted(
        fused.values(),
        key=lambda item: (-item["score"], item["evidence"]["evidence_id"]),
    )[:top_k]
    return {
        "ok": True,
        "query": query,
        "parsed_filters": parsed,
        "graph": graph_results,
        "retrieval": retrieval_results,
        "semantic": retrieval_results,
        "fused": {"num_results": len(fused_results), "results": fused_results},
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
    """Export one condition-linked observation row per target measurement."""
    store = LiteratureKGStore(kg_dir)
    nodes_by_id = _node_map(store.load_nodes())
    edges = store.load_edges()
    observation_to_catalyst, observation_to_paper, conditions, reactions = _graph_context(
        edges, nodes_by_id
    )
    components: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    metrics_by_observation: dict[str, list[KGEdge]] = defaultdict(list)
    for edge in edges:
        if edge.relation == "achieves":
            metrics_by_observation[edge.source_node_id].append(edge)
        elif edge.relation in {"has_active_metal", "has_promoter", "supported_on"}:
            target = nodes_by_id.get(edge.target_node_id)
            if target:
                components[edge.source_node_id][edge.relation].append(target.name)

    rows: list[dict[str, Any]] = []
    for observation_id, metric_edges in metrics_by_observation.items():
        catalyst = observation_to_catalyst.get(observation_id)
        if not catalyst:
            continue
        paper = observation_to_paper.get(observation_id)
        for target_edge in metric_edges:
            quantity = str(target_edge.attributes.get("quantity") or "")
            if quantity != target_quantity:
                continue
            condition_id = target_edge.attributes.get("condition_id")
            condition = conditions[observation_id].get(str(condition_id)) if condition_id else None
            row: dict[str, Any] = {
                "observation_id": observation_id,
                "paper_id": paper.name if paper else "",
                "catalyst_id": catalyst.node_id,
                "catalyst_name": catalyst.name,
                "active_metal": ";".join(
                    sorted(set(components[catalyst.node_id]["has_active_metal"]))
                ),
                "promoter": ";".join(
                    sorted(set(components[catalyst.node_id]["has_promoter"]))
                ),
                "support": ";".join(
                    sorted(set(components[catalyst.node_id]["supported_on"]))
                ),
                "reaction": ";".join(node.name for node in reactions.get(observation_id, [])),
                "condition_id": condition_id or "",
                "reaction_temperature": condition.attributes.get("temperature", "") if condition else "",
                "temperature_unit": condition.attributes.get("temperature_unit", "") if condition else "",
                "pressure": condition.attributes.get("pressure", "") if condition else "",
                "pressure_unit": condition.attributes.get("pressure_unit", "") if condition else "",
                "target_quantity": target_quantity,
                "target": target_edge.attributes.get("value"),
                "target_unit": target_edge.attributes.get("unit"),
                "target_comparator": target_edge.attributes.get("attributes", {}).get("comparator", "="),
                "target_range_min": target_edge.attributes.get("attributes", {}).get("range_min", ""),
                "target_range_max": target_edge.attributes.get("attributes", {}).get("range_max", ""),
                "literature_confidence": target_edge.confidence,
                "evidence_ids": ";".join(sorted(set(target_edge.evidence_ids))),
            }
            for feature_edge in metric_edges:
                if feature_edge.attributes.get("condition_id") != condition_id:
                    continue
                feature_quantity = feature_edge.attributes.get("quantity")
                if feature_quantity:
                    row[str(feature_quantity)] = feature_edge.attributes.get("value")
                    row[f"{feature_quantity}_unit"] = feature_edge.attributes.get("unit")
            rows.append(row)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        if fieldnames:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return {"ok": True, "path": str(out_path), "n_rows": len(rows)}
