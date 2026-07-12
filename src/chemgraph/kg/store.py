"""Persistence layer for the literature knowledge graph."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from chemgraph.kg.normalize import normalize_records
from chemgraph.kg.schema import SCHEMA_VERSION, CatalystRecord, EvidenceSpan, KGEdge, KGNode
from chemgraph.kg.verify import verify_records


def _slug(value: str) -> str:
    clean = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
    clean = "_".join(part for part in clean.split("_") if part)
    return clean[:80] or "unnamed"


def _stable_id(prefix: str, *parts: Any) -> str:
    payload = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
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
        self.nodes_jsonl_path = self.root / "nodes.jsonl"
        self.edges_path = self.root / "edges.parquet"
        self.edges_jsonl_path = self.root / "edges.jsonl"
        self.evidence_path = self.root / "evidence.sqlite"
        self.graph_path = self.root / "graph.json"
        self.manifest_path = self.root / "manifest.json"

    def save(
        self,
        nodes: Iterable[KGNode],
        edges: Iterable[KGEdge],
        evidence_spans: Iterable[EvidenceSpan],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        """Persist nodes, edges, evidence, and a NetworkX JSON graph."""
        self.root.mkdir(parents=True, exist_ok=True)
        node_rows = [node.model_dump(mode="json") for node in nodes]
        edge_rows = [edge.model_dump(mode="json") for edge in edges]
        evidence_rows = [span.model_dump(mode="json") for span in evidence_spans]

        nodes_written = self._write_table(
            pd.DataFrame(node_rows), self.nodes_path, self.nodes_jsonl_path
        )
        edges_written = self._write_table(
            pd.DataFrame(edge_rows), self.edges_path, self.edges_jsonl_path
        )
        self._write_evidence(evidence_rows)
        self._write_networkx_graph(node_rows, edge_rows)

        artifacts = {
            "nodes": str(nodes_written),
            "edges": str(edges_written),
            "evidence": str(self.evidence_path),
            "graph": str(self.graph_path),
        }
        manifest_artifacts = {
            name: Path(path).name for name, path in artifacts.items()
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "counts": {
                "nodes": len(node_rows),
                "edges": len(edge_rows),
                "evidence": len(evidence_rows),
            },
            "artifacts": manifest_artifacts,
            "sha256": {
                name: self._file_hash(Path(path)) for name, path in artifacts.items()
            },
            "metadata": metadata or {},
        }
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )

        return {**artifacts, "manifest": str(self.manifest_path)}

    def load_nodes(self) -> list[KGNode]:
        path = self.nodes_path if self.nodes_path.exists() else self.nodes_jsonl_path
        if not path.exists():
            return []
        rows = self._read_table(path).to_dict(orient="records")
        return [KGNode.model_validate(self._decode_row(row)) for row in rows]

    def load_edges(self) -> list[KGEdge]:
        path = self.edges_path if self.edges_path.exists() else self.edges_jsonl_path
        if not path.exists():
            return []
        rows = self._read_table(path).to_dict(orient="records")
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

    def _write_table(
        self,
        df: pd.DataFrame,
        parquet_path: Path,
        jsonl_path: Path,
    ) -> Path:
        """Write Parquet, or a truthfully named JSONL fallback."""
        encoded = df.copy()
        for col in encoded.columns:
            if encoded[col].map(lambda x: isinstance(x, (dict, list))).any():
                encoded[col] = encoded[col].map(_jsonify)
        try:
            encoded.to_parquet(parquet_path, index=False)
            if jsonl_path.exists():
                jsonl_path.unlink()
            return parquet_path
        except (ImportError, ModuleNotFoundError, ValueError) as exc:
            message = str(exc).lower()
            if not any(term in message for term in ("parquet", "pyarrow", "fastparquet", "engine")):
                raise
            encoded.to_json(jsonl_path, orient="records", lines=True)
            if parquet_path.exists():
                parquet_path.unlink()
            return jsonl_path

    def _read_table(self, path: Path) -> pd.DataFrame:
        if path.suffix.lower() == ".jsonl":
            return pd.read_json(path, orient="records", lines=True)
        with path.open("rb") as handle:
            first_byte = handle.read(1)
        if first_byte in {b"{", b"["}:
            # Backward compatibility for v1 artifacts that stored JSONL under
            # a .parquet filename. New writes always use a truthful suffix.
            return pd.read_json(path, orient="records", lines=True)
        try:
            return pd.read_parquet(path)
        except (ImportError, ModuleNotFoundError, ValueError) as exc:
            raise ValueError(f"Could not read Parquet artifact {path}: {exc}") from exc

    @staticmethod
    def _file_hash(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

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
        for key in ("aliases", "attributes", "evidence_ids", "source_paper_ids"):
            if key in decoded:
                decoded[key] = _maybe_json(decoded[key])
        for key in ("created_at", "updated_at"):
            if key in decoded and decoded[key] is not None and not isinstance(decoded[key], str):
                decoded[key] = None if pd.isna(decoded[key]) else str(decoded[key])
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
        def field_evidence(field: str) -> list[str]:
            return record.field_evidence_ids.get(field) or evidence_ids

        for span in record.evidence_spans:
            self.evidence[span.evidence_id] = span

        paper = self._node(
            "Paper",
            record.paper_id,
            record.paper_id,
            record.confidence,
            paper_id=record.paper_id,
        )
        catalyst = self._node(
            "CatalystSystem",
            record.canonical_catalyst_name or record.catalyst_name,
            record.catalyst_name,
            record.confidence,
            aliases=[record.catalyst_name],
            paper_id=record.paper_id,
        )
        observation = self._node(
            "Observation",
            record.record_id,
            record.record_id,
            record.confidence,
            paper_id=record.paper_id,
            attributes={
                "record_id": record.record_id,
                "paper_id": record.paper_id,
                "extractor_version": record.extractor_version,
            },
        )
        self._edge(paper, "reports", observation, evidence_ids, record.confidence)
        self._edge(
            observation,
            "uses_catalyst",
            catalyst,
            field_evidence("catalyst_name"),
            record.confidence,
        )

        if record.reaction:
            reaction = self._node(
                "Reaction",
                record.reaction,
                record.reaction,
                record.confidence,
                paper_id=record.paper_id,
            )
            self._edge(paper, "studies_reaction", reaction, field_evidence("reaction"), record.confidence)
            self._edge(observation, "tested_for", reaction, field_evidence("reaction"), record.confidence)

        for active_metal in record.active_metals:
            node = self._node("ActiveMetal", active_metal, active_metal, record.confidence, paper_id=record.paper_id)
            self._edge(catalyst, "has_active_metal", node, field_evidence("active_metals"), record.confidence)
        for promoter in record.promoters:
            node = self._node("Promoter", promoter, promoter, record.confidence, paper_id=record.paper_id)
            self._edge(catalyst, "has_promoter", node, field_evidence("promoters"), record.confidence)
        for dopant in record.dopants:
            node = self._node("Dopant", dopant, dopant, record.confidence, paper_id=record.paper_id)
            self._edge(catalyst, "has_dopant", node, field_evidence("dopants"), record.confidence)
        if record.support:
            support = self._node("Support", record.support, record.support, record.confidence, paper_id=record.paper_id)
            self._edge(catalyst, "supported_on", support, field_evidence("support"), record.confidence)

        for precursor in record.precursors:
            precursor_node = self._node(
                "Precursor",
                precursor,
                precursor,
                record.confidence,
                paper_id=record.paper_id,
            )
            self._edge(
                observation,
                "uses_precursor",
                precursor_node,
                field_evidence("precursors"),
                record.confidence,
            )
        if record.synthesis_method:
            method_node = self._node(
                "SynthesisMethod",
                record.synthesis_method,
                record.synthesis_method,
                record.confidence,
                paper_id=record.paper_id,
            )
            self._edge(
                observation,
                "synthesized_by",
                method_node,
                field_evidence("synthesis_method"),
                record.confidence,
            )
        for step in record.synthesis_steps:
            step_evidence = [step.evidence_span_id] if step.evidence_span_id else evidence_ids
            step_node = self._node(
                "SynthesisStep",
                step.step_id,
                f"step_{step.order}:{step.operation}",
                record.confidence,
                attributes=step.model_dump(mode="json"),
                paper_id=record.paper_id,
            )
            self._edge(
                observation,
                "has_step",
                step_node,
                step_evidence,
                record.confidence,
            )

        for condition in record.reaction_conditions:
            condition_evidence = [condition.evidence_span_id] if condition.evidence_span_id else evidence_ids
            node = self._node(
                "ReactionCondition",
                condition.condition_id,
                condition.condition_id,
                record.confidence,
                attributes=condition.model_dump(mode="json"),
                paper_id=record.paper_id,
            )
            self._edge(observation, "tested_under", node, condition_evidence, record.confidence)

        for metric in record.performance_metrics:
            metric_evidence = [metric.evidence_span_id] if metric.evidence_span_id else evidence_ids
            name = f"{metric.quantity}:{metric.value}:{metric.unit}"
            node = self._node(
                "PerformanceMetric",
                metric.measurement_id,
                name,
                metric.confidence,
                attributes=metric.model_dump(mode="json"),
                paper_id=record.paper_id,
            )
            self._edge(
                observation,
                "achieves",
                node,
                metric_evidence,
                min(record.confidence, metric.confidence),
                attributes=metric.model_dump(mode="json"),
            )

        for metric in record.material_properties:
            metric_evidence = [metric.evidence_span_id] if metric.evidence_span_id else evidence_ids
            name = f"{metric.quantity}:{metric.value}:{metric.unit}"
            node = self._node(
                "MaterialProperty",
                metric.measurement_id,
                name,
                metric.confidence,
                attributes=metric.model_dump(mode="json"),
                paper_id=record.paper_id,
            )
            self._edge(
                observation,
                "has_property",
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
                paper_id=record.paper_id,
            )
            self._edge(
                observation,
                "characterized_by",
                method_node,
                field_evidence("characterization_methods"),
                record.confidence,
            )

        for result in record.characterization_results:
            result_node = self._node(
                "CharacterizationResult",
                result,
                result,
                record.confidence,
                paper_id=record.paper_id,
            )
            self._edge(
                observation,
                "produces",
                result_node,
                field_evidence("characterization_results"),
                record.confidence,
            )

        for claim in record.mechanistic_claims:
            claim_node = self._node(
                "MechanisticClaim", claim, claim, record.confidence, paper_id=record.paper_id
            )
            self._edge(
                claim_node,
                "supported_by",
                observation,
                field_evidence("mechanistic_claims"),
                record.confidence,
            )

    def _node(
        self,
        node_type: str,
        canonical_name: str,
        name: str,
        confidence: float,
        *,
        aliases: list[str] | None = None,
        attributes: dict[str, Any] | None = None,
        paper_id: str | None = None,
    ) -> KGNode:
        node_id = f"{node_type}:{_slug(canonical_name)}:{_stable_id('id', canonical_name).split('_', 1)[1]}"
        if node_id in self.nodes:
            existing = self.nodes[node_id]
            merged_aliases = sorted(set(existing.aliases + (aliases or [])))
            data = existing.model_dump()
            data["aliases"] = merged_aliases
            source_paper_ids = set(existing.source_paper_ids)
            if paper_id:
                source_paper_ids.add(paper_id)
            data["source_paper_ids"] = sorted(source_paper_ids)
            data["source_count"] = len(source_paper_ids) or existing.source_count
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
            source_paper_ids=[paper_id] if paper_id else [],
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
    allow_unverified: bool = False,
) -> dict[str, Any]:
    """Verify, normalize, build, integrity-check, and persist the KG.

    Verification is mandatory by default for every caller, including scripts,
    LangChain tools, MCP, and the UI. ``allow_unverified`` is an explicit
    debugging escape hatch and is recorded in the artifact manifest.
    """
    record_list = list(records)
    chunk_texts: dict[tuple[str, str], set[str]] = {}
    for record in record_list:
        for span in record.evidence_spans:
            chunk_texts.setdefault((span.paper_id, span.chunk_id), set()).add(span.text)
    chunk_collisions = [key for key, texts in chunk_texts.items() if len(texts) > 1]
    if chunk_collisions:
        raise ValueError(
            "Source identity collision detected: "
            f"chunk_ids={chunk_collisions}."
        )
    verification = verify_records(record_list)
    rejected = [result for result in verification if not result.accepted]
    if rejected and not allow_unverified:
        messages = [
            f"{result.record.record_id}: "
            + "; ".join(issue.message for issue in result.issues if issue.severity == "error")
            for result in rejected
        ]
        raise ValueError(
            "KG build rejected unverified records. "
            + " | ".join(messages[:10])
        )
    if not allow_unverified:
        record_list = [result.record for result in verification if result.accepted]
    if normalize:
        record_list = normalize_records(record_list, synonyms_path=synonyms_path)
    builder = KGBuilder()
    nodes, edges, evidence = builder.build(record_list)
    node_ids = {node.node_id for node in nodes}
    evidence_ids = {span.evidence_id for span in evidence}
    integrity_errors = []
    for edge in edges:
        if edge.source_node_id not in node_ids or edge.target_node_id not in node_ids:
            integrity_errors.append(f"Edge {edge.edge_id} has an orphan endpoint.")
        missing = sorted(set(edge.evidence_ids) - evidence_ids)
        if missing:
            integrity_errors.append(f"Edge {edge.edge_id} references missing evidence {missing}.")
    if integrity_errors:
        raise ValueError("KG integrity validation failed: " + " | ".join(integrity_errors[:10]))
    store = LiteratureKGStore(out_dir)
    paths = store.save(
        nodes,
        edges,
        evidence,
        metadata={
            "normalization_enabled": normalize,
            "synonyms_path": str(synonyms_path) if synonyms_path else None,
            "allow_unverified": allow_unverified,
            "verification": {
                "accepted": sum(result.accepted for result in verification),
                "rejected": len(rejected),
                "issue_count": sum(len(result.issues) for result in verification),
            },
        },
    )
    from chemgraph.kg.validation import validate_kg

    artifact_validation = validate_kg(out_dir)
    if not artifact_validation["ok"]:
        raise ValueError(
            "Persisted KG integrity validation failed: "
            + " | ".join(artifact_validation["errors"][:10])
        )
    return {
        "ok": True,
        "n_records": len(record_list),
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "n_evidence": len(evidence),
        "n_rejected": len(rejected),
        "validation": artifact_validation,
        "paths": paths,
    }
