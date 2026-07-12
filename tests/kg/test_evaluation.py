from chemgraph.kg.evaluation import temporal_link_backtest
from chemgraph.kg.schema import PaperChunk


def test_temporal_link_backtest_scores_future_missing_link():
    chunks = [
        PaperChunk(
            paper_id="p2018",
            chunk_id="c2018",
            text=(
                "Published 2018. Cu/Al2O3 methanol selectivity reached 80% "
                "at 220 C during CO2 hydrogenation to methanol."
            ),
            metadata={"year": 2018},
        ),
        PaperChunk(
            paper_id="p2019",
            chunk_id="c2019",
            text=(
                "Published 2019. Ni/ZnO methanol selectivity reached 70% "
                "at 220 C during CO2 hydrogenation to methanol."
            ),
            metadata={"year": 2019},
        ),
        PaperChunk(
            paper_id="p2021",
            chunk_id="c2021",
            text=(
                "Published 2021. Cu/ZnO methanol selectivity reached 85% "
                "at 220 C during CO2 hydrogenation to methanol."
            ),
            metadata={"year": 2021},
        ),
    ]

    result = temporal_link_backtest(chunks, split_year=2020, top_k=1)

    assert result["predictions"][0]["active_metal"] == "Cu"
    assert result["predictions"][0]["support"] == "ZnO"
    assert result["precision_at_k"] == 1.0
    assert result["recall_at_k"] == 1.0
