# GLiNERGraphRetriever

A LangChain-compatible retriever that builds a knowledge graph from your documents using [GLiNER-ReLEx](https://huggingface.co/knowledgator/gliner-relex-large-v1.0) and uses it to retrieve richer, more connected context than vector search alone.

---

## What problem does this solve?

Standard RAG retrieves documents by embedding similarity. This works well when the answer lives neatly inside a single chunk, but breaks down when answering a question requires connecting information spread across multiple documents — for example, reasoning about relationships between entities, or following a chain of facts.

`GLiNERGraphRetriever` augments similarity search with a knowledge graph built at index time. After retrieving seed chunks by vector search, it traverses the graph to pull in related chunks that share entities or have explicit relational connections — chunks that a pure similarity search would miss.

---

## How it works

### Indexing

When you call `from_documents()`, three things happen:

**1. Parent → child splitting**

Each document is split into small child chunks (default 512 characters) sized to fit GLiNER's context window. Every child carries a `parent_id` reference back to its source document. The full parent documents are stored separately.

**2. GLiNER extraction**

GLiNER runs over each child chunk to extract two things:

- **Named entities** — e.g. `sam altman (person)`, `openai (organization)`
- **Typed relations** — e.g. `sam altman → founded → openai`

You configure which entity labels and relation types to extract. GLiNER is a zero-shot model, so no fine-tuning is required.

**3. Two graphs are built**

- **Entity-link graph** (undirected bipartite): connects entity nodes to the child chunks they appear in. Two chunks that mention the same entity are 2 hops apart in this graph.
- **Triplet graph** (directed multigraph): connects entity nodes via typed relation edges. Each edge records which child chunk the relation was extracted from.

Both graphs are persisted to disk alongside the edge data as a parquet file.

```
parent doc
    │
    ├── child chunk A  ──── "sam altman" ──── child chunk B
    │         │                  │
    │     [entity]           [founded]
    │                            │
    └── child chunk C  ──── "openai" ─────── child chunk D
```

---

### Retrieval

Retrieval always returns **full parent documents**, not chunks. Two paths are available, selected per call.

#### Path 1 — Auto traversal (default)

```python
docs = retriever.invoke("Who founded OpenAI?")
docs = retriever.invoke("Who founded OpenAI?", k=6, traversal_depth=2)
```

1. **Seed search** — vector similarity search retrieves `k` child chunks.
2. **Entity-link expansion** — for each seed chunk, find all other chunks that share an entity (2-hop traversal on the entity-link graph).
3. **Triplet expansion** — find the entity nodes connected to the seed chunks, then BFS the triplet graph up to `traversal_depth` hops, collecting all child chunks referenced by traversed edges.
4. **Parent resolution** — the union of all child IDs is resolved to deduplicated parent documents.

#### Path 2 — LLM-assisted traversal

```python
docs = retriever.invoke("Who founded OpenAI?", filter_llm=ChatOpenAI(model="gpt-4o"))
```

Same graph, but two LLM filtering steps are inserted to prune noise before returning parents:

1. **Seed search** — same as path 1. Optionally, `expand_llm` generates a richer keyword query first (see [Query expansion](#query-expansion) below).
2. **Entity filtering** — the LLM is shown the seed chunks as context and asked to select which candidate entities are relevant to the query.
3. **Triple filtering** — the LLM is shown the triples reachable from the selected entities and asked to select which ones are relevant.
4. **Parent resolution** — only chunks referenced by the selected triples (plus seed and entity-link neighbours) are resolved to parents.

This reduces irrelevant context reaching the LLM in the final generation step, at the cost of two extra LLM calls. Path selection depends **only** on whether `filter_llm` is provided.

#### Query expansion

An optional `expand_llm` kwarg generates a richer keyword/synonym string from the natural-language query before seed search. This improves recall for both BM25 and dense retrieval without affecting which path is taken.

```python
# Expand query only (Path 1)
docs = retriever.invoke("Who founded OpenAI?", expand_llm=fast_model)

# Expand query and filter graph (Path 2)
docs = retriever.invoke("Who founded OpenAI?",
                        expand_llm=fast_model,
                        filter_llm=smart_model)
```

`expand_llm` and `filter_llm` are fully independent — you can use either, both, or neither, and you can supply different models for each role.

---

### BM25 + dense ensemble

By default the retriever uses dense-only similarity search. Setting `bm25_weight > 0` enables a BM25 + dense ensemble with `dense_weight = 1 - bm25_weight`. BM25 is never instantiated when `bm25_weight == 0`.

```python
retriever = GLiNERGraphRetriever(
    ...,
    bm25_weight=0.3,   # 30% BM25, 70% dense
)

# Override per call
docs = retriever.invoke("Who founded OpenAI?", bm25_weight=0.5)
```

BM25 works independently of LLM usage and benefits from query expansion when `expand_llm` is provided.

---

### Debugging with `RetrieverTrace`

Every `invoke` call populates `retriever.last_trace` with the full decision trail:

```python
docs  = retriever.invoke("Who founded OpenAI?",
                         expand_llm=fast_model,
                         filter_llm=smart_model)
trace = retriever.last_trace

print(trace.path)              # "llm" or "auto"
print(trace.expanded_query)    # keyword string used for seed search, or None
print(trace.bm25_weight)       # effective BM25 weight (None if disabled)

print(trace.candidate_entities)  # all entities found in seed chunks
print(trace.selected_entities)   # entities kept after LLM filter (path 2)
                                  # == candidate_entities for path 1

print(trace.candidate_triples)   # all triples reachable from selected entities
print(trace.selected_triples)    # triples kept after LLM filter (path 2)
                                  # == candidate_triples for path 1
```

In path 1, `candidate_*` and `selected_*` are always identical — nothing is filtered out — so you can directly compare what the graph would expose to an LLM versus what path 1 uses automatically.

---

## Installation

```bash
pip install gliner langchain langchain-core networkx pandas tqdm
```

For BM25 ensemble support:

```bash
pip install langchain-community rank_bm25
```

Install your preferred vectorstore, e.g.:

```bash
pip install langchain-chroma
```

---

## Quick start

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document

from gliner_graph_retriever import GLiNERGraphRetriever

# 1. Instantiate
retriever = GLiNERGraphRetriever(
    vectorstore=Chroma(embedding_function=OpenAIEmbeddings()),
    model_path="knowledgator/gliner-relex-large-v1.0",
    collection_name="my_docs",
    labels=["person", "organization", "location"],
    relations=["founded", "located_in", "acquired"],
    persist_directory="./graph_store",
    bm25_weight=0.3,   # optional: enable BM25 ensemble (default 0 = dense-only)
)

# 2. Index documents (run once)
docs = [Document(page_content="Sam Altman co-founded OpenAI in 2015."), ...]
retriever.from_documents(docs, gliner_batch_size=8)

# 3. Retrieve — path 1 (auto)
results = retriever.invoke("Who founded OpenAI?")

# 4. Retrieve — path 2 (LLM-assisted graph filtering)
smart_llm = ChatOpenAI(model="gpt-4o")
results   = retriever.invoke("Who founded OpenAI?", filter_llm=smart_llm)

# 5. Retrieve — path 2 with query expansion using a cheaper model
fast_llm = ChatOpenAI(model="gpt-4o-mini")
results  = retriever.invoke("Who founded OpenAI?",
                            expand_llm=fast_llm,
                            filter_llm=smart_llm)

# 6. Inspect the trace
trace = retriever.last_trace
print(trace.expanded_query)
print(trace.candidate_entities)
print(trace.selected_triples)
```

---

## Configuration reference

### Constructor

| Parameter | Default | Description |
|---|---|---|
| `vectorstore` | required | Any LangChain `VectorStore` |
| `model_path` | required | HuggingFace ID or local path for GLiNER |
| `collection_name` | required | Prefix for persisted files |
| `labels` | required | Entity types for GLiNER to extract |
| `relations` | `[]` | Relation types for GLiNER to extract |
| `persist_directory` | `None` | Directory for parquet/manifest/parent store. `None` = in-memory only |
| `add_inverse_relations` | `False` | Add reverse edges to the triplet graph |
| `threshold` | `0.7` | GLiNER entity confidence threshold |
| `relation_threshold` | `0.5` | GLiNER relation confidence threshold |
| `child_chunk_size` | `512` | Max characters per child chunk |
| `child_chunk_overlap` | `64` | Character overlap between child chunks |
| `k` | `4` | Default number of seed chunks from similarity search |
| `traversal_depth` | `1` | Default BFS hops on the triplet graph |
| `bm25_weight` | `0.0` | BM25 share of the ensemble. `0` = dense-only (BM25 never built). `> 0` enables BM25; dense weight is `1 - bm25_weight` |

### `from_documents` / `add_documents`

| Parameter | Default | Description |
|---|---|---|
| `gliner_batch_size` | `1` | GLiNER inference batch size. `1` = one-by-one, `0` = full batch, `N` = mini-batches |

### `invoke` kwargs

| Parameter | Default | Description |
|---|---|---|
| `k` | `self.k` | Override seed chunk count for this call |
| `traversal_depth` | `self.traversal_depth` | Override BFS depth for this call |
| `expand_llm` | `None` | LLM used for query keyword expansion. Improves seed search quality. Does not affect path selection |
| `filter_llm` | `None` | LLM used for entity and triple filtering. Providing this activates path 2 |
| `bm25_weight` | `self.bm25_weight` | Override BM25 weight for this call. `0` = dense-only |

---

## Persistence and reloading

```python
# After from_documents(), the graph is saved automatically.
# To restore it in a new session:
retriever = GLiNERGraphRetriever(
    vectorstore=Chroma(
        persist_directory="./graph_store",
        embedding_function=OpenAIEmbeddings(),
    ),
    persist_directory="./graph_store",
    # ... other params
)
retriever.load()
```

The vectorstore must be restored separately (pass `persist_directory` to Chroma or equivalent). The retriever manages the edge parquet, manifest, and parent store.

---

## Manual LangGraph wiring

For full control over the path 2 steps inside a LangGraph workflow:

```python
seed_children    = retriever.seed_search(query, k=6)
seed_ids         = [doc.id for doc in seed_children]

entities         = retriever.get_entry_entities(seed_ids)
selected_ents    = retriever.filter_entities_with_llm(query, entities, seed_children, llm)

triples          = retriever.get_reachable_triples(selected_ents, traversal_depth=2)
selected_triples = retriever.filter_triples_with_llm(query, triples, llm)

parent_docs      = retriever.resolve_parents_from_triples(selected_triples)
```

---

## Design notes

**Why parent/child splitting?** GLiNER has a limited context window, so extraction happens on small chunks. But returning small chunks to the final LLM loses surrounding context. The parent/child design gives you accurate extraction on short text and full context at generation time.

**Why two graphs?** The entity-link graph is fast to traverse and good at finding chunks that co-mention the same entities regardless of whether a typed relation was extracted. The triplet graph is more precise — it only fires when GLiNER extracted a confident typed relation — and supports directional BFS. Both are used together in path 1; path 2 operates only on the triplet graph since the LLM filters triples directly.

**Why split `expand_llm` and `filter_llm`?** The two LLM roles have different requirements. Query expansion benefits from a fast, cheap model since it only needs to generate synonyms. Graph filtering benefits from a more capable model since it reasons over entity and triple candidates in context. Splitting the roles lets you right-size each call and use either independently.

**Why LLM filtering as a separate path?** Auto traversal can pull in loosely related context. For deep or ambiguous queries, letting an LLM prune the entity and triple candidates before resolving parents keeps the final context window tighter. The tradeoff is latency from two additional structured-output calls.

**Why a single `bm25_weight` parameter?** `bm25_weight == 0` cleanly means "BM25 off" — no object is ever built, no extra dependency is required. For non-zero values, `dense_weight = 1 - bm25_weight` is a natural constraint that removes a redundant degree of freedom.
