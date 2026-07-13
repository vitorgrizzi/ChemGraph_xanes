"""Observation-aware graph, retrieval, and export helpers for the literature KG."""

from __future__ import annotations

import csv
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

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


def evidence_search(
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


def semantic_search(
    kg_dir: str | Path,
    query: str,
    *,
    top_k: int = 5,
    embedding_model: str | None = None,
) -> dict[str, Any]:
    """Backward-compatible alias for :func:`evidence_search`.

    The historical name is imprecise because retrieval defaults to lexical
    BM25. New callers should use ``evidence_search`` and inspect ``method``.
    """
    return evidence_search(
        kg_dir,
        query,
        top_k=top_k,
        embedding_model=embedding_model,
    )


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


def _result_fact_key(result: dict[str, Any]) -> tuple[Any, ...]:
    """Identify one same-paper observation fact independent of chunk overlap."""
    edge_attributes = result["edge"].get("attributes", {})
    measurement_attributes = edge_attributes.get("attributes", {})
    condition_attributes = (
        result["condition"].get("attributes", {}) if result["condition"] else {}
    )
    condition_values = {
        key: condition_attributes.get(key)
        for key in (
            "temperature",
            "temperature_unit",
            "pressure",
            "pressure_unit",
            "h2_co2_ratio",
            "ghsv",
            "whsv",
            "time_on_stream",
            "time_on_stream_unit",
        )
    }
    return (
        result["paper"].get("canonical_name") if result["paper"] else None,
        result["source"]["node_id"],
        result["edge"]["relation"],
        edge_attributes.get("quantity"),
        edge_attributes.get("value"),
        edge_attributes.get("unit"),
        measurement_attributes.get("comparator", "="),
        measurement_attributes.get("range_min"),
        measurement_attributes.get("range_max"),
        json.dumps(condition_values, sort_keys=True),
        tuple(sorted(node["canonical_name"] for node in result["reactions"])),
    )


def graph_query(
    kg_dir: str | Path,
    *,
    relation: str | None = None,
    catalyst_contains: str | None = None,
    metric_quantity: str | None = None,
    min_value: float | None = None,
    min_value_operator: str = ">=",
    max_temperature: float | None = None,
    max_temperature_operator: str = "<=",
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

    if min_value_operator not in {">", ">="}:
        raise ValueError("min_value_operator must be '>' or '>='.")
    if max_temperature_operator not in {"<", "<="}:
        raise ValueError("max_temperature_operator must be '<' or '<='.")

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
            below_minimum = (
                float(value) <= min_value
                if min_value_operator == ">"
                else float(value) < min_value
            ) if value is not None else True
            if value is None or comparator in {"below", "under", "<"} or below_minimum:
                continue

        condition = None
        if observation and edge.relation == "achieves":
            condition_id = edge.attributes.get("condition_id") or target.attributes.get("condition_id")
            if condition_id:
                condition = conditions_by_observation[observation.node_id].get(str(condition_id))
        if max_temperature is not None:
            temperature = _temperature_deg_c(condition)
            above_maximum = (
                temperature >= max_temperature
                if max_temperature_operator == "<"
                else temperature > max_temperature
            ) if temperature is not None else True
            if temperature is None or above_maximum:
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
                "supporting_edge_ids": [edge.edge_id],
            }
        )
    deduplicated: dict[tuple[Any, ...], dict[str, Any]] = {}
    for result in results:
        key = _result_fact_key(result)
        existing = deduplicated.get(key)
        if existing is None:
            deduplicated[key] = result
            continue
        evidence_by_id = {
            span["evidence_id"]: span
            for span in [*existing["evidence"], *result["evidence"]]
        }
        existing["evidence"] = list(evidence_by_id.values())
        existing["supporting_edge_ids"] = sorted(
            set(existing["supporting_edge_ids"] + result["supporting_edge_ids"])
        )
    results = list(deduplicated.values())
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
    above = re.search(
        r"(at or above|greater than|at least|above|over|>)\s*"
        r"(\d+(?:\.\d+)?)\s*%?",
        lower,
    )
    if above:
        parsed["min_value"] = float(above.group(2))
        parsed["min_value_operator"] = (
            ">=" if above.group(1) in {"at least", "at or above"} else ">"
        )
    below_temp = re.search(
        r"(at or below|at most|below|under|<)\s*(\d+(?:\.\d+)?)\s*"
        r"(?:(?:°|º)\s*)?c\b",
        lower,
    )
    if below_temp:
        parsed["max_temperature"] = float(below_temp.group(2))
        parsed["max_temperature_operator"] = (
            "<=" if below_temp.group(1) in {"at most", "at or below"} else "<"
        )
    return parsed


def _compact_excerpt(
    text: str,
    *,
    value: float | None = None,
    quantity: str | None = None,
    query: str | None = None,
    limit: int = 320,
) -> str:
    """Select a short evidence window around the relevant metric or query terms."""
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= limit:
        return normalized

    protected = re.sub(
        r"\b(Figs?|Eqs?|Refs?)\.",
        r"\1<period>",
        normalized,
        flags=re.I,
    )
    sentences = [
        sentence.replace("<period>", ".").strip()
        for sentence in re.split(r"(?<=[.!?])\s+", protected)
        if sentence.strip()
    ]
    quantity_token = (quantity or "").replace("_", " ").split()[-1:]
    value_text = f"{value:g}" if value is not None else None
    selected = None
    for sentence in sentences:
        if value_text and value_text in sentence and (
            not quantity_token or quantity_token[0].lower() in sentence.lower()
        ):
            selected = sentence
            break
    if selected is None and query:
        query_terms = _terms(query)
        selected = max(
            sentences,
            key=lambda sentence: sum(term in sentence.lower() for term in query_terms),
            default=normalized,
        )
    selected = selected or normalized
    if len(selected) <= limit:
        return selected

    anchors = [value_text] if value_text else []
    anchors.extend(quantity_token)
    if query:
        anchors.extend(_terms(query))
    anchor_index = next(
        (
            selected.lower().find(anchor.lower())
            for anchor in anchors
            if anchor and selected.lower().find(anchor.lower()) >= 0
        ),
        0,
    )
    start = max(0, anchor_index - limit // 3)
    stop = min(len(selected), start + limit)
    start = max(0, stop - limit)
    excerpt = selected[start:stop].strip()
    return f"{'…' if start else ''}{excerpt}{'…' if stop < len(selected) else ''}"


def compact_query_result(full_result: dict[str, Any]) -> dict[str, Any]:
    """Project a full hybrid result into a token-efficient, model-facing form."""
    answers = []
    for result in full_result["graph"]["results"]:
        edge_attributes = result["edge"].get("attributes", {})
        measurement_attributes = edge_attributes.get("attributes", {})
        condition_attributes = (
            result["condition"].get("attributes", {}) if result["condition"] else {}
        )
        evidence = result.get("evidence", [])
        evidence_ids = sorted({span["evidence_id"] for span in evidence})
        paper_attributes = result["paper"].get("attributes", {}) if result["paper"] else {}
        dois = list(paper_attributes.get("dois") or [])
        if not dois:
            dois = sorted({span.get("doi") for span in evidence if span.get("doi")})
        value = edge_attributes.get("value")
        quantity = edge_attributes.get("quantity")
        best_evidence = next(
            (
                span
                for span in evidence
                if value is not None and f"{float(value):g}" in span.get("text", "")
            ),
            evidence[0] if evidence else None,
        )
        answer = {
            "catalyst": result["source"]["name"],
            "metric_quantity": quantity,
            "value": value,
            "unit": edge_attributes.get("unit"),
            "comparator": measurement_attributes.get("comparator", "="),
            "temperature": condition_attributes.get("temperature"),
            "temperature_unit": condition_attributes.get("temperature_unit"),
            "pressure": condition_attributes.get("pressure"),
            "pressure_unit": condition_attributes.get("pressure_unit"),
            "h2_co2_ratio": condition_attributes.get("h2_co2_ratio"),
            "reaction": (
                result["reactions"][0]["name"] if result.get("reactions") else None
            ),
            "paper_id": (
                paper_attributes.get("paper_id")
                or (result["paper"].get("canonical_name") if result["paper"] else None)
            ),
            "doi": dois[0] if dois else None,
            "confidence": result["edge"].get("confidence"),
            "evidence_ids": evidence_ids,
            "evidence_excerpt": (
                _compact_excerpt(
                    best_evidence["text"],
                    value=float(value) if value is not None else None,
                    quantity=str(quantity or ""),
                )
                if best_evidence
                else None
            ),
            "supporting_edge_ids": result.get(
                "supporting_edge_ids",
                [result["edge"]["edge_id"]],
            ),
        }
        answers.append({key: item for key, item in answer.items() if item is not None})

    retrieval_context = []
    warnings = []
    if not answers:
        warnings.append(
            "No graph-supported answers matched; retrieval_context is unfiltered evidence."
        )
        for result in full_result["retrieval"]["results"][:3]:
            evidence = result["evidence"]
            retrieval_context.append(
                {
                    "graph_supported": False,
                    "retrieval_rank": result["rank"],
                    "score": result["score"],
                    "evidence_id": evidence["evidence_id"],
                    "paper_id": evidence["paper_id"],
                    "doi": evidence.get("doi"),
                    "evidence_excerpt": _compact_excerpt(
                        evidence["text"],
                        query=full_result["query"],
                    ),
                }
            )

    return {
        "ok": full_result["ok"],
        "query": full_result["query"],
        "response_mode": "compact",
        "parsed_filters": full_result["parsed_filters"],
        "answer_count": len(answers),
        "answers": answers,
        "retrieval_context": retrieval_context,
        "warnings": warnings,
    }


def hybrid_query(
    kg_dir: str | Path,
    query: str,
    *,
    top_k: int = 10,
    embedding_model: str | None = None,
    response_mode: Literal["full", "compact"] = "full",
) -> dict[str, Any]:
    """Fuse graph and evidence rankings, returning a full or compact response."""
    if response_mode not in {"full", "compact"}:
        raise ValueError("response_mode must be 'full' or 'compact'.")
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
    retrieval_results = evidence_search(
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
                {
                    "evidence": evidence,
                    "score": 0.0,
                    "graph_hits": [],
                    "graph_rank": None,
                    "retrieval_rank": None,
                },
            )
            item["graph_rank"] = min(item["graph_rank"] or rank, rank)
            for edge_id in result.get(
                "supporting_edge_ids",
                [result["edge"]["edge_id"]],
            ):
                if edge_id not in item["graph_hits"]:
                    item["graph_hits"].append(edge_id)
    for item in fused.values():
        item["score"] += 1.0 / (60 + item["graph_rank"])
    for rank, result in enumerate(retrieval_results["results"], start=1):
        evidence = result["evidence"]
        item = fused.setdefault(
            evidence["evidence_id"],
            {
                "evidence": evidence,
                "score": 0.0,
                "graph_hits": [],
                "graph_rank": None,
                "retrieval_rank": None,
            },
        )
        item["score"] += 1.0 / (60 + rank)
        item["retrieval_rank"] = rank
    for item in fused.values():
        item["graph_supported"] = bool(item["graph_hits"])
        item["origins"] = [
            origin
            for origin, present in (
                ("graph", item["graph_rank"] is not None),
                ("retrieval", item["retrieval_rank"] is not None),
            )
            if present
        ]
    fused_results = sorted(
        fused.values(),
        key=lambda item: (-item["score"], item["evidence"]["evidence_id"]),
    )[:top_k]
    full_result = {
        "ok": True,
        "query": query,
        "parsed_filters": parsed,
        "graph": graph_results,
        "retrieval": retrieval_results,
        "fused": {"num_results": len(fused_results), "results": fused_results},
    }
    return compact_query_result(full_result) if response_mode == "compact" else full_result


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
