"""Persistence layer for the literature knowledge graph."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from chemgraph.kg.normalize import normalize_records
from chemgraph.kg.schema import CatalystRecord, EvidenceSpan, KGEdge, KGNode


def _slug(value: str) -> str:
    clean = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
    clean = "_".join(part for part in clean.split("_") if part)
    return clean[:80] or "unnamed"


def _stable_id(prefix: str, *parts: Any) -> str:
    payload = "|".join(str(part) for part in parts)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _jsonify(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _maybe_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


class LiteratureKGStore:
    """SQLite + table + optional NetworkX store for literature KG data."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.nodes_path = self.root / "nodes.parquet"
        self.edges_path = self.root / "edges.parquet"
        self.evidence_path = self.root / "evidence.sqlite"
        self.graph_path = self.root / "graph.json"

    def save(
        self,
        nodes: Iterable[KGNode],
        edges: Iterable[KGEdge],
        evidence_spans: Iterable[EvidenceSpan],
    ) -> dict[str, str]:
        """Persist nodes, edges, evidence, and a NetworkX JSON graph."""
        self.root.mkdir(parents=True, exist_ok=True)
        node_rows = [node.model_dump(mode="json") for node in nodes]
        edge_rows = [edge.model_dump(mode="json") for edge in edges]
        evidence_rows = [span.model_dump(mode="json") for span in evidence_spans]

        self._write_table(pd.DataFrame(node_rows), self.nodes_path)
        self._write_table(pd.DataFrame(edge_rows), self.edges_path)
        self._write_evidence(evidence_rows)
        self._write_networkx_graph(node_rows, edge_rows)

        return {
            "nodes": str(self.nodes_path),
            "edges": str(self.edges_path),
            "evidence": str(self.evidence_path),
            "graph": str(self.graph_path),
        }

    def load_nodes(self) -> list[KGNode]:
        if not self.nodes_path.exists():
            return []
        rows = self._read_table(self.nodes_path).to_dict(orient="records")
        return [KGNode.model_validate(self._decode_row(row)) for row in rows]

    def load_edges(self) -> list[KGEdge]:
        if not self.edges_path.exists():
            return []
        rows = self._read_table(self.edges_path).to_dict(orient="records")
        return [KGEdge.model_validate(self._decode_row(row)) for row in rows]

    def load_evidence(self) -> list[EvidenceSpan]:
        if not self.evidence_path.exists():
            return []
        with sqlite3.connect(self.evidence_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM evidence").fetchall()
        return [EvidenceSpan.model_validate(dict(row)) for row in rows]

    def get_evidence(self, evidence_id: str) -> EvidenceSpan | None:
        if not self.evidence_path.exists():
            return None
        with sqlite3.connect(self.evidence_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM evidence WHERE evidence_id = ?",
                (evidence_id,),
            ).fetchone()
        return EvidenceSpan.model_validate(dict(row)) if row else None

    def to_networkx(self):
        """Return a NetworkX MultiDiGraph when networkx is installed."""
        try:
            import networkx as nx
        except ImportError as exc:
            raise ImportError("networkx is required for graph export.") from exc

        graph = nx.MultiDiGraph()
        for node in self.load_nodes():
            graph.add_node(node.node_id, **node.model_dump(mode="json"))
        for edge in self.load_edges():
            graph.add_edge(
                edge.source_node_id,
                edge.target_node_id,
                key=edge.edge_id,
                **edge.model_dump(mode="json"),
            )
        return graph

    def _write_table(self, df: pd.DataFrame, path: Path) -> None:
        """Write parquet when possible; fall back to JSONL at the same path."""
        encoded = df.copy()
        for col in encoded.columns:
            if encoded[col].map(lambda x: isinstance(x, (dict, list))).any():
                encoded[col] = encoded[col].map(_jsonify)
        try:
            encoded.to_parquet(path, index=False)
        except Exception:
            encoded.to_json(path, orient="records", lines=True)

    def _read_table(self, path: Path) -> pd.DataFrame:
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.read_json(path, orient="records", lines=True)

    def _write_evidence(self, rows: list[dict[str, Any]]) -> None:
        with sqlite3.connect(self.evidence_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS evidence (
                    evidence_id TEXT PRIMARY KEY,
                    paper_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    page INTEGER,
                    section TEXT,
                    text TEXT NOT NULL,
                    start_char INTEGER,
                    end_char INTEGER,
                    source_path TEXT,
                    doi TEXT,
                    extraction_model TEXT,
                    extraction_time TEXT
                )
                """
            )
            conn.execute("DELETE FROM evidence")
            for row in rows:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO evidence (
                        evidence_id, paper_id, chunk_id, page, section, text,
                        start_char, end_char, source_path, doi, extraction_model,
                        extraction_time
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row.get("evidence_id"),
                        row.get("paper_id"),
                        row.get("chunk_id"),
                        row.get("page"),
                        row.get("section"),
                        row.get("text"),
                        row.get("start_char"),
                        row.get("end_char"),
                        row.get("source_path"),
                        row.get("doi"),
                        row.get("extraction_model"),
                        row.get("extraction_time"),
                    ),
                )

    def _write_networkx_graph(
        self,
        node_rows: list[dict[str, Any]],
        edge_rows: list[dict[str, Any]],
    ) -> None:
        try:
            import networkx as nx
            from networkx.readwrite import json_graph
        except ImportError:
            payload = {"nodes": node_rows, "edges": edge_rows}
            self.graph_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return

        graph = nx.MultiDiGraph()
        for row in node_rows:
            graph.add_node(row["node_id"], **row)
        for row in edge_rows:
            graph.add_edge(
                row["source_node_id"],
                row["target_node_id"],
                key=row["edge_id"],
                **row,
            )
        payload = json_graph.node_link_data(graph, edges="edges")
        self.graph_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _decode_row(self, row: dict[str, Any]) -> dict[str, Any]:
        decoded = dict(row)
        for key in ("aliases", "attributes", "evidence_ids"):
            if key in decoded:
                decoded[key] = _maybe_json(decoded[key])
        for key in ("created_at", "updated_at"):
            if key in decoded and not isinstance(decoded[key], str):
                decoded[key] = str(decoded[key])
        return decoded


class KGBuilder:
    """Convert validated catalyst records into nodes, edges, and evidence."""

    def __init__(self):
        self.nodes: OrderedDict[str, KGNode] = OrderedDict()
        self.edges: OrderedDict[str, KGEdge] = OrderedDict()
        self.evidence: OrderedDict[str, EvidenceSpan] = OrderedDict()

    def build(self, records: Iterable[CatalystRecord]):
        for record in records:
            self.add_record(record)
        return list(self.nodes.values()), list(self.edges.values()), list(self.evidence.values())

    def add_record(self, record: CatalystRecord) -> None:
        evidence_ids = [span.evidence_id for span in record.evidence_spans]
        for span in record.evidence_spans:
            self.evidence[span.evidence_id] = span

        paper = self._node("Paper", record.paper_id, record.paper_id, record.confidence)
        catalyst = self._node(
            "CatalystSystem",
            record.canonical_catalyst_name or record.catalyst_name,
            record.catalyst_name,
            record.confidence,
            aliases=[record.catalyst_name],
        )
        self._edge(paper, "reports", catalyst, evidence_ids, record.confidence)

        if record.reaction:
            reaction = self._node("Reaction", record.reaction, record.reaction, record.confidence)
            self._edge(paper, "studies_reaction", reaction, evidence_ids, record.confidence)
            self._edge(catalyst, "tested_for", reaction, evidence_ids, record.confidence)

        for active_metal in record.active_metals:
            node = self._node("ActiveMetal", active_metal, active_metal, record.confidence)
            self._edge(catalyst, "has_active_metal", node, evidence_ids, record.confidence)
        for promoter in record.promoters:
            node = self._node("Promoter", promoter, promoter, record.confidence)
            self._edge(catalyst, "has_promoter", node, evidence_ids, record.confidence)
        for dopant in record.dopants:
            node = self._node("Dopant", dopant, dopant, record.confidence)
            self._edge(catalyst, "has_dopant", node, evidence_ids, record.confidence)
        if record.support:
            support = self._node("Support", record.support, record.support, record.confidence)
            self._edge(catalyst, "supported_on", support, evidence_ids, record.confidence)

        for condition in record.reaction_conditions:
            condition_evidence = [condition.evidence_span_id] if condition.evidence_span_id else evidence_ids
            node = self._node(
                "ReactionCondition",
                condition.condition_id,
                condition.condition_id,
                record.confidence,
                attributes=condition.model_dump(mode="json"),
            )
            self._edge(catalyst, "tested_under", node, condition_evidence, record.confidence)

        for metric in record.performance_metrics:
            metric_evidence = [metric.evidence_span_id] if metric.evidence_span_id else evidence_ids
            name = f"{metric.quantity}:{metric.value}:{metric.unit}"
            node = self._node(
                "PerformanceMetric",
                name,
                name,
                metric.confidence,
                attributes=metric.model_dump(mode="json"),
            )
            self._edge(
                catalyst,
                "achieves",
                node,
                metric_evidence,
                min(record.confidence, metric.confidence),
                attributes=metric.model_dump(mode="json"),
            )

        for method in record.characterization_methods:
            method_node = self._node(
                "CharacterizationMethod",
                method,
                method,
                record.confidence,
            )
            self._edge(
                catalyst,
                "characterized_by",
                method_node,
                evidence_ids,
                record.confidence,
            )

        for claim in record.mechanistic_claims:
            claim_node = self._node("MechanisticClaim", claim, claim, record.confidence)
            self._edge(claim_node, "supported_by", catalyst, evidence_ids, record.confidence)

    def _node(
        self,
        node_type: str,
        canonical_name: str,
        name: str,
        confidence: float,
        *,
        aliases: list[str] | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> KGNode:
        node_id = f"{node_type}:{_slug(canonical_name)}"
        if node_id in self.nodes:
            existing = self.nodes[node_id]
            merged_aliases = sorted(set(existing.aliases + (aliases or [])))
            data = existing.model_dump()
            data["aliases"] = merged_aliases
            data["source_count"] = existing.source_count + 1
            data["confidence"] = max(existing.confidence, confidence)
            if attributes:
                data["attributes"] = {**existing.attributes, **attributes}
            self.nodes[node_id] = KGNode.model_validate(data)
            return self.nodes[node_id]

        node = KGNode(
            node_id=node_id,
            node_type=node_type,
            name=name,
            canonical_name=canonical_name,
            aliases=aliases or [],
            attributes=attributes or {},
            confidence=confidence,
        )
        self.nodes[node_id] = node
        return node

    def _edge(
        self,
        source: KGNode,
        relation: str,
        target: KGNode,
        evidence_ids: list[str],
        confidence: float,
        *,
        attributes: dict[str, Any] | None = None,
    ) -> KGEdge:
        clean_evidence = [eid for eid in evidence_ids if eid]
        edge_id = _stable_id("edge", source.node_id, relation, target.node_id, clean_evidence)
        if edge_id in self.edges:
            return self.edges[edge_id]
        edge = KGEdge(
            edge_id=edge_id,
            source_node_id=source.node_id,
            relation=relation,
            target_node_id=target.node_id,
            attributes=attributes or {},
            evidence_ids=clean_evidence,
            confidence=confidence,
        )
        self.edges[edge_id] = edge
        return edge


def build_kg(
    records: Iterable[CatalystRecord],
    out_dir: str | Path,
    *,
    normalize: bool = True,
    synonyms_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build and persist the KG from catalyst records."""
    record_list = list(records)
    if normalize:
        record_list = normalize_records(record_list, synonyms_path=synonyms_path)
    builder = KGBuilder()
    nodes, edges, evidence = builder.build(record_list)
    store = LiteratureKGStore(out_dir)
    paths = store.save(nodes, edges, evidence)
    return {
        "ok": True,
        "n_records": len(record_list),
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "n_evidence": len(evidence),
        "paths": paths,
    }
