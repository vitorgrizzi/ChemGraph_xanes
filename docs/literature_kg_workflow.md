# Literature Knowledge Graph Workflow

ChemGraph includes a first-pass `literature_kg` workflow for turning a small
paper set into a provenance-first catalysis knowledge graph. The MVP is designed
for 20-50 papers and one reaction family before scaling extraction quality or
distributed inference.

## Install

```bash
pip install -e ".[rag,kg]"
```

The core package imports without the optional `kg` extra. PyMuPDF, PyYAML,
NetworkX, FAISS, and Parquet support become available through the extra.

## Minimal Demo

```bash
python scripts/kg_ingest.py \
  --input data/demo_papers \
  --out data/kg_demo/chunks.jsonl

python scripts/kg_extract.py \
  --chunks data/kg_demo/chunks.jsonl \
  --out data/kg_demo/extractions.jsonl \
  --model deterministic

python scripts/kg_build.py \
  --records data/kg_demo/extractions.jsonl \
  --out data/kg_demo/graph \
  --synonyms configs/kg_synonyms.yaml

python scripts/kg_query.py \
  --kg data/kg_demo/graph \
  --q "Which catalysts report methanol selectivity above 70% for CO2 hydrogenation?"

python scripts/kg_suggest.py \
  --kg data/kg_demo/graph \
  --goal "low-temperature CO2 hydrogenation to methanol with high selectivity and stability"
```

Use `--model gpt-4o-mini` in `kg_extract.py` to call the existing OpenAI model
loader. The deterministic extractor is an offline fallback for tests and quick
pipeline checks; production extraction should use an LLM and stronger
verification prompts.

## Stored Artifacts

`kg_build.py` writes:

- `nodes.parquet`: typed nodes such as `Paper`, `CatalystSystem`, `Reaction`,
  `PerformanceMetric`, and `ReactionCondition`.
- `edges.parquet`: typed edges such as `reports`, `tested_for`,
  `tested_under`, and `achieves`. Every edge has `confidence` and
  `evidence_ids`.
- `evidence.sqlite`: source spans with paper/chunk/page/source metadata.
- `graph.json`: NetworkX node-link JSON when NetworkX is installed.

If no Parquet engine is installed, the table writer falls back to JSON lines at
the same `*.parquet` paths so the workflow remains usable in lightweight
environments.

## Agent Workflow

The registered `literature_kg` ChemGraph workflow exposes LangChain tools for:

- `kg_ingest_papers`
- `kg_extract_records`
- `kg_build_graph`
- `kg_hybrid_query`
- `kg_get_evidence`
- `kg_suggest_hypotheses`
- `kg_export_training_table`

The FastMCP server in `chemgraph.mcp.kg_tools` exposes the same operations for
external agents. Hypothesis cards include supporting KG paths, evidence IDs,
risk/utility/novelty scores, and structured validation tasks. Computational or
experimental actions are proposed only as plans and require human approval.

## Temporal Backtesting Scaffold

```bash
python scripts/kg_eval_temporal.py \
  --papers data/demo_papers \
  --split-year 2020 \
  --task link_prediction
```

The scaffold splits chunks into pre/post-year groups. A fuller evaluation should
build `G_<=t`, predict missing links, and check whether those links appear in
future papers.
