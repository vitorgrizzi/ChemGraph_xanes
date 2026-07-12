from chemgraph.kg.ingest import ingest_path, read_chunks_jsonl


def test_ingest_text_file_to_chunks(tmp_path):
    paper = tmp_path / "paper.txt"
    paper.write_text(
        "Cu/ZnO/Al2O3 was tested for CO2 hydrogenation. "
        "Methanol selectivity reached 83% at 210 C and 50 bar.",
        encoding="utf-8",
    )
    out = tmp_path / "chunks.jsonl"

    chunks = ingest_path(paper, out=out, chunk_size=80, chunk_overlap=10)

    assert out.exists()
    assert chunks
    assert read_chunks_jsonl(out)[0].paper_id.startswith("paper_")
    for chunk in chunks:
        start = chunk.metadata["start_char"]
        end = chunk.metadata["end_char"]
        assert paper.read_text(encoding="utf-8")[start:end] == chunk.text


def test_same_filename_with_different_content_gets_distinct_paper_ids(tmp_path):
    first = tmp_path / "a" / "paper.txt"
    second = tmp_path / "b" / "paper.txt"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("Cu/ZnO was tested.", encoding="utf-8")
    second.write_text("In2O3 was tested.", encoding="utf-8")

    chunks = ingest_path(tmp_path)

    assert len({chunk.paper_id for chunk in chunks}) == 2
