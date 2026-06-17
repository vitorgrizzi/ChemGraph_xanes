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
    assert read_chunks_jsonl(out)[0].paper_id == "paper"
