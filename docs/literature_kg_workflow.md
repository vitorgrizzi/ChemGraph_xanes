# Literature Knowledge Graph Workflow

ChemGraph's `literature_kg` workflow turns a small, curated paper collection
into an evidence-gated catalysis graph. Start with 20-50 papers from one
reaction family. Extraction recall may be imperfect, but unsupported claims are
rejected before persistence unless a caller explicitly enables the unsafe
debugging override.

## Install

```bash
pip install -e ".[rag,kg]"
```

The core TXT/Markdown/JSONL and deterministic-test path uses the base package.
The `kg` extra adds layout-aware PDF ingestion, Parquet, NetworkX, YAML, and
optional sentence-transformer retrieval.

## Data model

Canonical entities such as catalysts, supports, and reactions are shared.
Source-specific facts are not attached directly to those merged entities.
Every extracted `CatalystRecord` becomes an `Observation`:

```text
Paper -> reports -> Observation -> uses_catalyst -> CatalystSystem
                             |-> tested_for -> Reaction
                             |-> tested_under -> ReactionCondition
                             `-> achieves -> PerformanceMetric
```

A metric retains its exact `condition_id`, and queries/exports resolve only
that condition. Every edge carries confidence and resolvable evidence IDs.

## Minimal verified demo

```bash
python scripts/kg_ingest.py \
  --input data/demo_papers \
  --out data/kg_demo/chunks.jsonl

python scripts/kg_extract.py \
  --chunks data/kg_demo/chunks.jsonl \
  --out data/kg_demo/extractions.jsonl \
  --model deterministic

python scripts/kg_verify.py \
  --records data/kg_demo/extractions.jsonl \
  --out data/kg_demo/verified_records.jsonl

python scripts/kg_build.py \
  --records data/kg_demo/verified_records.jsonl \
  --out data/kg_demo/graph \
  --synonyms configs/kg_synonyms.yaml

python scripts/kg_validate.py --kg data/kg_demo/graph

python scripts/kg_query.py \
  --kg data/kg_demo/graph \
  --q "Which catalysts report methanol selectivity above 70% below 220 C?"
```

`kg_build.py` always runs grounding and reference verification. It fails closed
when a field lacks evidence, a numerical value is absent from its cited text,
a condition/evidence ID does not resolve, or a physical constraint fails. The
`--allow-unverified` option exists only for debugging and is recorded in the
manifest.

The deterministic extractor is a conservative offline fallback for tests and
plumbing checks. Use `--model MODEL_NAME` for schema-constrained LLM extraction;
the application, not the model, creates provenance IDs and source metadata.
The deterministic path preserves explicit value-temperature pairs in repeated
series and value-first phrases. For metric-only chunks, it carries catalyst
context from another chunk only when the same paper names exactly one catalyst,
and attaches that source chunk as evidence. It does not guess when multiple
catalysts are present. Estimated, predicted, calculated, and equilibrium values
are excluded from observed-performance edges. Query results collapse identical
same-paper facts introduced by overlapping chunks while retaining every
supporting evidence span and edge ID.

## Stored artifacts

Each build writes:

- `nodes.parquet` and `edges.parquet` when a Parquet engine is available;
  otherwise truthfully named `nodes.jsonl` and `edges.jsonl` files.
- `evidence.sqlite`, containing source-controlled evidence spans.
- `graph.json`, containing the graph representation.
- `manifest.json`, containing schema version, build configuration, counts,
  artifact filenames, and SHA-256 hashes.

Content-derived record, evidence, condition, measurement, node, edge, and
hypothesis IDs make equivalent rebuilds comparable. Build time is stored only
in the manifest.

## Retrieval and querying

Natural-language metric filters are converted to typed graph constraints by a
deterministic parser; querying does not require an LLM. Strict language such as
`above` and `below` is preserved as `>` and `<`, while `at least`, `at most`,
and `at or below` remain inclusive.
Temperature predicates use the condition linked to the returned metric, not
another condition reported for the same catalyst.

Evidence retrieval uses BM25 by default. To add semantic vector retrieval:

```bash
python scripts/kg_query.py \
  --kg data/kg_demo/graph \
  --q "low-temperature methanol selectivity" \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2
```

The response uses one `retrieval` object whose `method` reports `bm25` or
`vector`; it does not duplicate lexical results under a misleading `semantic`
key. Graph-path and evidence rankings are combined with reciprocal-rank fusion.
Each fused result reports its `origins`, `graph_supported`, `graph_rank`, and
`retrieval_rank`, so retrieval-only context cannot be mistaken for an answer
that passed graph filters. The response retains graph paths, observation and
condition objects, paper IDs, and evidence spans rather than emitting an
unsupported free-form answer.

## Hypotheses

Hypothesis cards are conservative trend candidates. Candidate measurements are
filtered by the stated goal and compared only within compatible
quantity/unit families. Cards report independent-paper counts,
condition-specific supporting paths, counter-evidence IDs, uncertainty-aware
scores, and human-gated validation tasks. The critic does not approve a card
without at least two independent supporting papers.

## Validation

Run the offline regression and integrity suites with:

```bash
pytest tests/kg -q
python scripts/kg_validate.py --kg data/kg_demo/graph
```

Evaluate a manually labeled `CatalystRecord` JSONL set with:

```bash
python scripts/kg_eval_gold.py \
  --predicted data/kg_demo/extractions.jsonl \
  --gold data/kg_demo/gold_records.jsonl
```

The report includes micro and per-field precision/recall/F1, exact metric and
condition-link accuracy, grounding acceptance rate, and grounding-error counts.

For scientific validation, build a manually labeled pilot and measure:

- exact page/chunk/source traceability;
- field-level precision and recall;
- numerical value, unit, comparator, and condition-link accuracy;
- evidence-entailment precision and unsupported-claim rate;
- graph referential integrity and deterministic rebuild IDs;
- query precision/recall on expected answers and expected empty answers;
- export correctness at the observation-condition level.

Adversarial examples should include multiple catalysts and conditions in one
passage, inequalities/ranges, Unicode units, conflicting papers, review papers
quoting primary work, negative results, and duplicate filenames.

## Temporal backtesting

```bash
python scripts/kg_eval_temporal.py \
  --papers data/demo_papers \
  --split-year 2020 \
  --top-k 10
```

The backtest builds a pre-cutoff candidate ranking for unseen
active-metal/support links, then measures precision@k, recall@k, and reciprocal
rank against links first observed after the cutoff. Papers without a reliable
year are excluded and counted explicitly.

## Agent surfaces

LangChain and FastMCP expose the same guarded operations:

- `kg_ingest_papers`
- `kg_extract_records`
- `kg_verify_records`
- `kg_build_graph`
- `kg_hybrid_query`
- `kg_get_evidence`
- `kg_suggest_hypotheses`
- `kg_export_training_table`
- `kg_validate_graph`

Computational and experimental tasks remain proposals and require explicit
human approval.
