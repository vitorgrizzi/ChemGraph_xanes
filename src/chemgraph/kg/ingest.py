"""Paper ingestion for the literature knowledge-graph workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from chemgraph.kg.chunk import chunk_text, iter_jsonl_chunks
from chemgraph.kg.schema import PaperChunk

SUPPORTED_INPUT_SUFFIXES = {".txt", ".md", ".jsonl", ".json", ".pdf"}


def _paper_id_from_path(path: Path) -> str:
    return path.stem.lower().replace(" ", "_").replace("-", "_")


def _read_pdf_pages(path: Path) -> list[tuple[int, str]]:
    try:
        import fitz
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is required for PDF ingestion. Install with "
            "`pip install -e .[rag]` or `pip install pymupdf`."
        ) from exc

    pages: list[tuple[int, str]] = []
    with fitz.open(path) as doc:
        for idx, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if text:
                pages.append((idx, text))
    return pages


def _read_json_records(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("chunks"), list):
        return payload["chunks"]
    if isinstance(payload, dict):
        return [payload]
    raise ValueError(f"Unsupported JSON shape in {path}.")


def iter_input_files(input_path: str | Path) -> Iterable[Path]:
    """Yield supported files from a file or directory."""
    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() in SUPPORTED_INPUT_SUFFIXES:
            yield path
        return
    if not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    for candidate in sorted(path.rglob("*")):
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_INPUT_SUFFIXES:
            yield candidate


def ingest_file(
    path: str | Path,
    *,
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
) -> list[PaperChunk]:
    """Ingest one PDF/text/JSONL file into ``PaperChunk`` objects."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    paper_id = _paper_id_from_path(file_path)

    if suffix in {".jsonl", ".json"}:
        rows = _read_json_records(file_path)
        chunks = iter_jsonl_chunks(rows)
        for chunk in chunks:
            if not chunk.source_path:
                chunk.source_path = str(file_path)
        return chunks

    if suffix == ".pdf":
        chunks: list[PaperChunk] = []
        for page_num, page_text in _read_pdf_pages(file_path):
            chunks.extend(
                chunk_text(
                    page_text,
                    paper_id=paper_id,
                    source_path=str(file_path),
                    page=page_num,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            )
        return chunks

    if suffix in {".txt", ".md"}:
        text = file_path.read_text(encoding="utf-8")
        return chunk_text(
            text,
            paper_id=paper_id,
            source_path=str(file_path),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    raise ValueError(f"Unsupported input file type: {file_path.suffix}")


def ingest_path(
    input_path: str | Path,
    *,
    out: str | Path | None = None,
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
) -> list[PaperChunk]:
    """Ingest all supported files under ``input_path`` and optionally write JSONL."""
    chunks: list[PaperChunk] = []
    for file_path in iter_input_files(input_path):
        chunks.extend(
            ingest_file(
                file_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )

    if out is not None:
        write_chunks_jsonl(chunks, out)
    return chunks


def write_chunks_jsonl(chunks: Iterable[PaperChunk], out: str | Path) -> Path:
    """Write chunks as one JSON object per line."""
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(chunk.model_dump_json() + "\n")
    return out_path


def read_chunks_jsonl(path: str | Path) -> list[PaperChunk]:
    """Read chunks from JSONL."""
    chunks: list[PaperChunk] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                chunks.append(PaperChunk.model_validate_json(line))
    return chunks
