"""Text chunking helpers for literature ingestion."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from chemgraph.kg.schema import PaperChunk


def chunk_text(
    text: str,
    *,
    paper_id: str,
    source_path: str | None = None,
    page: int | None = None,
    section: str | None = None,
    doi: str | None = None,
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
) -> list[PaperChunk]:
    """Split text into overlapping character chunks with paper metadata."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    if not text or not text.strip():
        return []

    chunks: list[PaperChunk] = []
    start = 0
    idx = 0
    while start < len(text):
        stop = min(len(text), start + chunk_size)
        if stop < len(text):
            split_at = max(text.rfind("\n\n", start, stop), text.rfind(". ", start, stop))
            if split_at > start + max(200, chunk_size // 3):
                stop = split_at + 1
        raw_piece = text[start:stop]
        left_trim = len(raw_piece) - len(raw_piece.lstrip())
        right_trimmed = raw_piece.rstrip()
        piece_start = start + left_trim
        piece_stop = start + len(right_trimmed)
        piece = text[piece_start:piece_stop]
        if piece:
            page_part = f"_p{page:04d}" if page is not None else ""
            digest = hashlib.sha256(piece.encode("utf-8")).hexdigest()[:10]
            chunks.append(
                PaperChunk(
                    paper_id=paper_id,
                    chunk_id=f"{paper_id}{page_part}_chunk_{idx:05d}_{digest}",
                    page=page,
                    section=section,
                    text=piece,
                    source_path=source_path,
                    doi=doi,
                    metadata={"start_char": piece_start, "end_char": piece_stop},
                )
            )
            idx += 1
        if stop >= len(text):
            break
        start = max(0, stop - chunk_overlap)
    return chunks


def iter_jsonl_chunks(rows: Iterable[dict]) -> list[PaperChunk]:
    """Convert JSONL dictionaries into ``PaperChunk`` objects."""
    chunks: list[PaperChunk] = []
    for index, row in enumerate(rows):
        text = str(row.get("text") or row.get("content") or "").strip()
        if not text:
            continue
        paper_id = str(row.get("paper_id") or row.get("document_id") or f"paper_{index}")
        chunk_id = str(row.get("chunk_id") or f"{paper_id}_chunk_{index:05d}")
        chunks.append(
            PaperChunk(
                paper_id=paper_id,
                chunk_id=chunk_id,
                page=row.get("page"),
                section=row.get("section"),
                text=text,
                source_path=row.get("source_path"),
                doi=row.get("doi"),
                metadata=dict(row.get("metadata") or {}),
            )
        )
    return chunks
