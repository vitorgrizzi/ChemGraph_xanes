from chemgraph.kg.agents import kg_langchain_tools
from chemgraph.workflows.literature_kg import run_literature_kg_workflow


def test_end_to_end_workflow_builds_a_validated_observation_graph(tmp_path):
    paper = tmp_path / "paper.txt"
    paper.write_text(
        "Cu/ZnO/Al2O3 methanol selectivity reached 83% at 210 C and 50 bar "
        "during CO2 hydrogenation to methanol.",
        encoding="utf-8",
    )
    work_dir = tmp_path / "work"

    result = run_literature_kg_workflow(
        str(paper),
        str(work_dir),
        query="methanol selectivity above 70% below 220 C",
    )

    assert result["verification"]["n_accepted"] == 1
    assert result["graph"]["validation"]["ok"]
    assert result["query"]["graph"]["num_results"] == 1


def test_langchain_surface_exposes_guarded_validation_tool():
    names = {tool.name for tool in kg_langchain_tools()}

    assert "kg_validate_graph" in names
    assert "kg_build_graph" in names
    assert "kg_extract_records" in names
    assert "kg_verify_records" in names
