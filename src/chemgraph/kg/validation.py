"""Artifact-level integrity validation for persisted literature KGs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from chemgraph.kg.schema import SCHEMA_VERSION
from chemgraph.kg.store import LiteratureKGStore


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_kg(kg_dir: str | Path, *, verify_hashes: bool = True) -> dict[str, Any]:
    """Validate artifact hashes, graph references, and observation linkage."""
    store = LiteratureKGStore(kg_dir)
    errors: list[str] = []
    warnings: list[str] = []
    manifest: dict[str, Any] = {}
    if not store.manifest_path.exists():
        errors.append("manifest.json is missing.")
    else:
        manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SCHEMA_VERSION:
            errors.append(
                f"Schema version is {manifest.get('schema_version')!r}; expected {SCHEMA_VERSION!r}."
            )
        for name, raw_path in manifest.get("artifacts", {}).items():
            path = store.root / raw_path
            if not path.exists():
                errors.append(f"Artifact is missing: {name}={path}.")
            elif verify_hashes:
                expected = manifest.get("sha256", {}).get(name)
                if expected and _sha256(path) != expected:
                    errors.append(f"Artifact hash mismatch: {name}.")

    try:
        nodes = store.load_nodes()
        edges = store.load_edges()
        evidence = store.load_evidence()
    except Exception as exc:
        errors.append(f"Artifacts could not be loaded: {exc}")
        return {"ok": False, "errors": errors, "warnings": warnings}

    node_ids = {node.node_id for node in nodes}
    evidence_ids = {span.evidence_id for span in evidence}
    observation_ids = {node.node_id for node in nodes if node.node_type == "Observation"}
    catalysts_by_observation: dict[str, set[str]] = {node_id: set() for node_id in observation_ids}
    papers_by_observation: dict[str, set[str]] = {node_id: set() for node_id in observation_ids}
    conditions_by_observation: dict[str, set[str]] = {node_id: set() for node_id in observation_ids}

    for edge in edges:
        if edge.source_node_id not in node_ids:
            errors.append(f"Edge {edge.edge_id} has missing source {edge.source_node_id}.")
        if edge.target_node_id not in node_ids:
            errors.append(f"Edge {edge.edge_id} has missing target {edge.target_node_id}.")
        missing_evidence = sorted(set(edge.evidence_ids) - evidence_ids)
        if missing_evidence:
            errors.append(f"Edge {edge.edge_id} has missing evidence {missing_evidence}.")
        if edge.relation == "uses_catalyst" and edge.source_node_id in observation_ids:
            catalysts_by_observation[edge.source_node_id].add(edge.target_node_id)
        elif edge.relation == "reports" and edge.target_node_id in observation_ids:
            papers_by_observation[edge.target_node_id].add(edge.source_node_id)
        elif edge.relation == "tested_under" and edge.source_node_id in observation_ids:
            conditions_by_observation[edge.source_node_id].add(edge.target_node_id)

    nodes_by_id = {node.node_id: node for node in nodes}
    for observation_id in observation_ids:
        if len(catalysts_by_observation[observation_id]) != 1:
            errors.append(f"Observation {observation_id} must link to exactly one catalyst.")
        if len(papers_by_observation[observation_id]) != 1:
            errors.append(f"Observation {observation_id} must link to exactly one paper.")
    for edge in edges:
        if edge.relation != "achieves" or edge.source_node_id not in observation_ids:
            continue
        condition_id = edge.attributes.get("condition_id")
        if condition_id:
            linked_condition_ids = {
                str(nodes_by_id[node_id].attributes.get("condition_id"))
                for node_id in conditions_by_observation[edge.source_node_id]
            }
            if str(condition_id) not in linked_condition_ids:
                errors.append(
                    f"Metric edge {edge.edge_id} condition_id does not resolve on its observation."
                )

    duplicate_node_ids = len(nodes) - len(node_ids)
    duplicate_evidence_ids = len(evidence) - len(evidence_ids)
    if duplicate_node_ids:
        errors.append(f"Found {duplicate_node_ids} duplicate node IDs.")
    if duplicate_evidence_ids:
        errors.append(f"Found {duplicate_evidence_ids} duplicate evidence IDs.")
    if manifest:
        counts = manifest.get("counts", {})
        for name, actual in (("nodes", len(nodes)), ("edges", len(edges)), ("evidence", len(evidence))):
            if counts.get(name) != actual:
                errors.append(f"Manifest count mismatch for {name}: {counts.get(name)} != {actual}.")

    return {
        "ok": not errors,
        "schema_version": SCHEMA_VERSION,
        "counts": {"nodes": len(nodes), "edges": len(edges), "evidence": len(evidence)},
        "errors": errors,
        "warnings": warnings,
    }
