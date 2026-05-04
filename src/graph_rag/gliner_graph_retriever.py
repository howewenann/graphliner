"""
GLiNERGraphRetriever
====================
A LangChain-compatible retriever that combines:
  - Internal parent→child splitting (GLiNER has a small context window)
  - GLiNER NER/relation extraction → a NetworkX knowledge graph
  - A child vectorstore for candidate retrieval
  - Two graph-traversal expansion strategies (auto and LLM-assisted)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DESIGN: PARENT / CHILD SPLIT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You pass parent documents.  Internally the retriever:
  1. Splits each parent into small child chunks (≤ child_chunk_size
     characters) so they fit in GLiNER's context window.
  2. Stamps metadata["parent_id"] on every child.
  3. Adds children to the vectorstore and captures the assigned IDs.
  4. Runs GLiNER on each child — building the knowledge graph with child IDs.
  5. Stores parent docs in an internal dict keyed by parent_id.

At retrieval time graph traversal operates on child IDs, then the final
step resolves those children → their parent_ids → deduplicated parent docs.
The LLM always receives full parent context, not small chunks.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PIPELINE OVERVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 0 — Ingestion
------------------
Pass parent documents.  The retriever handles splitting, embedding, and
graph construction internally.

    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings

    retriever = GLiNERGraphRetriever(
        vectorstore=Chroma(embedding_function=OpenAIEmbeddings()),
        model_path="urchade/gliner_mediumv2.1",
        collection_name="my_docs",
        labels=["person", "organization", "location"],
        relations=["founded", "located_in"],
        persist_directory="./graph_store",
    )
    retriever.from_documents(parent_docs, gliner_batch_size=8)

    # Incremental additions — batch size is per-call
    retriever.add_documents(more_parent_docs, gliner_batch_size=0)  # 0 = full batch

    # Restore a persisted graph later (vectorstore must be re-populated
    # separately, e.g. by passing persist_directory to Chroma)
    retriever.load()

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Path 1 — Auto traversal  (no llm provided)
-------------------------------------------
    docs = retriever.invoke("Who founded OpenAI?")
    docs = retriever.invoke("Who founded OpenAI?", k=6, traversal_depth=2)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Path 2 — LLM-assisted traversal  (llm provided at invoke time)
---------------------------------------------------------------
    docs = retriever.invoke("Who founded OpenAI?", llm=ChatOpenAI(model="gpt-4o"))

    # Inspect what was selected:
    trace = retriever.last_trace
    print(trace.selected_entities)
    print(trace.selected_triples)

    # Manual LangGraph wiring (no llm arg needed):
    seed_children = retriever.seed_search(query, k=4)
    seed_ids      = [doc.id for doc in seed_children]

    entities      = retriever.get_entry_entities(seed_ids)
    selected_ents = retriever.filter_entities_with_llm(query, entities, seed_children, llm)

    triples          = retriever.get_reachable_triples(selected_ents, traversal_depth=2)
    selected_triples = retriever.filter_triples_with_llm(query, triples, llm)

    parent_docs = retriever.resolve_parents_from_triples(selected_triples)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import textwrap
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Type, Union

import networkx as nx
import pandas as pd
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field, create_model
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Column-name constants — single source of truth
# ---------------------------------------------------------------------------
EDGE_COLS          = ["head", "head_type", "relation", "tail", "tail_type", "score", "graph_type", "edge_source"]
ENTITY_DEDUP_COLS  = ["head", "head_type", "relation", "tail", "tail_type"]
TRIPLET_DEDUP_COLS = ["head", "head_type", "relation", "tail", "tail_type", "edge_source"]

GRAPH_TYPE_ENTITY  = "ENTITY_LINK"
GRAPH_TYPE_TRIPLET = "TRIPLET"
RELATION_IN_CHUNK  = "__IN_CHUNK__"
NODE_TYPE_CHUNK    = "_CHUNK_"
PARENT_ID_KEY      = "parent_id"


# ---------------------------------------------------------------------------
# Trace dataclass — populated on every invoke for debugging
# ---------------------------------------------------------------------------

@dataclass
class RetrieverTrace:
    """
    Snapshot of a single retrieval call.  Available via ``retriever.last_trace``
    immediately after ``invoke``.

    Both paths populate ``candidate_entities`` and ``candidate_triples`` so you
    can always inspect what the graph found.  The ``selected_*`` fields show
    what was kept after filtering:

    - Path 1 (auto):  ``selected_entities == candidate_entities`` and
      ``selected_triples == candidate_triples`` — nothing is filtered out.
    - Path 2 (llm):   ``selected_*`` is the LLM's subset of ``candidate_*``.

    Attributes:
        path:                ``"auto"`` or ``"llm"``.
        seed_child_ids:      Vectorstore IDs of seed children from similarity search.
        candidate_entities:  All entities found in seed children (both paths).
        selected_entities:   Entities used for triple traversal (both paths).
        candidate_triples:   All triples reachable from selected entities (both paths).
        selected_triples:    Triples used for parent resolution (both paths).
        returned_parent_ids: IDs of parent documents returned to the caller.
    """
    path:                str                    = ""
    seed_child_ids:      List[str]              = field(default_factory=list)
    candidate_entities:  Dict[str, str]         = field(default_factory=dict)
    selected_entities:   Dict[str, str]         = field(default_factory=dict)
    candidate_triples:   List["TripleRecord"]   = field(default_factory=list)
    selected_triples:    List["TripleRecord"]   = field(default_factory=list)
    returned_parent_ids: List[str]              = field(default_factory=list)


# ---------------------------------------------------------------------------
# Small data-carrier used by the LLM-assist path
# ---------------------------------------------------------------------------

class TripleRecord(BaseModel):
    """
    A single graph triple together with a stable ID and a human-readable
    description that can be shown to an LLM for filtering.
    """
    id:          str
    text:        str
    head:        str
    head_type:   str
    relation:    str
    tail:        str
    tail_type:   str
    edge_source: str


# ---------------------------------------------------------------------------
# Schema builders — List[Literal[...]] approach
# ---------------------------------------------------------------------------

def _build_entity_filter_schema(entities: Dict[str, str]) -> Type[BaseModel]:
    """
    Build a structured-output schema that asks the LLM to select a subset of
    entities by returning their names in a list.
    """
    if not entities:
        return create_model(
            "EntityFilterSchema",
            selected_entities=(List[str], Field(default_factory=list, description="No entities available.")),
        )
    names    = tuple(entities.keys())
    lit_type = Literal[names]  # type: ignore[valid-type]
    description = (
        "Select the entity names that are relevant to answering the query. "
        "Available entities and their types:\n"
        + "\n".join(f"  • {name}  ({etype})" for name, etype in entities.items())
    )
    return create_model(
        "EntityFilterSchema",
        selected_entities=(List[lit_type], Field(default_factory=list, description=description)),  # type: ignore[valid-type]
    )


def _build_triple_filter_schema(triples: List[TripleRecord]) -> Type[BaseModel]:
    """
    Build a structured-output schema that asks the LLM to select a subset of
    triples by returning their IDs in a list.
    """
    if not triples:
        return create_model(
            "TripleFilterSchema",
            selected_triple_ids=(List[str], Field(default_factory=list, description="No triples available.")),
        )
    ids      = tuple(t.id for t in triples)
    lit_type = Literal[ids]  # type: ignore[valid-type]
    description = (
        "Select the triple IDs that contain information relevant to answering the query. "
        "Available triples:\n"
        + "\n".join(f"  • {t.id}: {t.text}" for t in triples)
    )
    return create_model(
        "TripleFilterSchema",
        selected_triple_ids=(List[lit_type], Field(default_factory=list, description=description)),  # type: ignore[valid-type]
    )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class GLiNERGraphRetriever(BaseRetriever):
    """
    Retriever that splits parent documents into child chunks, builds a
    knowledge graph over those children using GLiNER, and returns full
    parent documents at retrieval time.

    Retrieval path is selected per-call:
      - ``llm`` kwarg provided  → Path 2 (LLM-assisted traversal)
      - no ``llm``              → Path 1 (auto graph traversal)

    Per-call overrides (all optional kwargs to ``invoke``):
      - ``k``               — overrides ``self.k`` for this call
      - ``traversal_depth`` — overrides ``self.traversal_depth`` for this call
      - ``llm``             — triggers Path 2 for this call

    After every ``invoke``, ``retriever.last_trace`` holds a ``RetrieverTrace``
    with the full decision trail for debugging.

    Args:
        vectorstore:           Any LangChain VectorStore (Chroma, FAISS, Qdrant …).
        model_path:            HuggingFace path or local path to the GLiNER model.
        collection_name:       Identifier for persisted files.
        labels:                Entity labels for GLiNER — list[str] or dict[str, str].
        persist_directory:     Directory for parquet / manifest / parent-store files.
                               ``None`` → in-memory only.
        relations:             Relation types for GLiNER relation extraction.
        add_inverse_relations: Add reverse edges to the triplet graph.
        threshold:             Confidence threshold for entity extraction.
        relation_threshold:    Confidence threshold for relation extraction.
        child_chunk_size:      Maximum character length of each child chunk.
        child_chunk_overlap:   Character overlap between adjacent child chunks.
        k:                     Default number of seed children from similarity search.
        traversal_depth:       Default BFS hops from each seed entity node.
    """

    # Pydantic fields
    vectorstore:           VectorStore                    = Field(...)
    model_path:            str
    collection_name:       str
    labels:                Union[List[str], Dict[str, str]]
    persist_directory:     Optional[str]                  = None
    relations:             List[str]                      = Field(default_factory=list)
    add_inverse_relations: bool                           = False
    threshold:             float                          = 0.7
    relation_threshold:    float                          = 0.5
    child_chunk_size:      int                            = 512
    child_chunk_overlap:   int                            = 64
    k:                     int                            = 4
    traversal_depth:       int                            = 1

    # Private runtime state
    _ner_extractor:     object                            = None
    _child_splitter:    object                            = None
    _edge_df:           Optional[pd.DataFrame]            = None
    _graph_entity_link: Optional[nx.Graph]                = None
    _graph_triplet:     Optional[nx.MultiDiGraph]         = None
    _parent_store:      Dict[str, Document]               = {}
    _last_trace:        Optional[RetrieverTrace]          = None

    model_config = {"arbitrary_types_allowed": True}

    # ------------------------------------------------------------------
    # Public trace accessor
    # ------------------------------------------------------------------

    @property
    def last_trace(self) -> Optional[RetrieverTrace]:
        """The ``RetrieverTrace`` from the most recent ``invoke`` call."""
        return self._last_trace

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def model_post_init(self, __context: Any) -> None:
        self._validate_labels(self.labels)
        from gliner import GLiNER
        self._ner_extractor = GLiNER.from_pretrained(self.model_path)
        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.child_chunk_size,
            chunk_overlap=self.child_chunk_overlap,
        )
        self._parent_store = {}

    @staticmethod
    def _validate_labels(labels: Union[List[str], Dict[str, str]]) -> None:
        if isinstance(labels, dict):
            if not all(isinstance(k, str) and k.strip() for k in labels):
                raise ValueError("Dict keys (labels) must be non-empty strings.")
        elif isinstance(labels, list):
            if not all(isinstance(x, str) and x.strip() for x in labels):
                raise ValueError("List elements (labels) must be non-empty strings.")
        else:
            raise TypeError("labels must be a list[str] or dict[str, str].")

    # ------------------------------------------------------------------
    # Parent / child splitting
    # ------------------------------------------------------------------

    def _split_to_children(self, parents: List[Document]) -> List[Document]:
        """
        Split parent documents into child chunks, stamp ``parent_id`` on each
        child's metadata, and register parents in ``_parent_store``.
        """
        children: List[Document] = []
        for parent in parents:
            parent_id             = parent.id or str(uuid.uuid4())
            parent_with_id        = parent.model_copy()
            parent_with_id.id     = parent_id
            self._parent_store[parent_id] = parent_with_id

            chunks = self._child_splitter.create_documents(
                texts=[parent.page_content],
                metadatas=[{**parent.metadata, PARENT_ID_KEY: parent_id}],
            )
            children.extend(chunks)
        return children

    # ------------------------------------------------------------------
    # Parent resolution
    # ------------------------------------------------------------------

    def _parents_from_child_ids(self, child_ids: List[str]) -> List[Document]:
        """
        Resolve a list of child vectorstore IDs → deduplicated parent documents.
        """
        seen:    set[str]       = set()
        parents: List[Document] = []
        for child in self._fetch_children_by_ids(child_ids):
            parent_id = child.metadata.get(PARENT_ID_KEY)
            if parent_id and parent_id not in seen:
                seen.add(parent_id)
                parent = self._parent_store.get(parent_id)
                if parent is not None:
                    parents.append(parent)
        return parents

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(str(text).strip().lower().split())

    def _fetch_children_by_ids(self, child_ids: List[str]) -> List[Document]:
        if hasattr(self.vectorstore, "get_by_ids"):
            return self.vectorstore.get_by_ids(child_ids)
        raise NotImplementedError(
            "The configured vectorstore does not support get_by_ids. "
            "Use Chroma, FAISS, or another store that implements this method."
        )

    # ------------------------------------------------------------------
    # Seed retrieval
    # ------------------------------------------------------------------

    def seed_search(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        Run similarity search against the child vectorstore.

        Args:
            query: The user's query string.
            k:     Number of results; falls back to ``self.k``.

        Returns:
            Child chunk documents carrying ``parent_id`` metadata.
        """
        return self.vectorstore.similarity_search(query, k=k or self.k)

    # ------------------------------------------------------------------
    # GLiNER inference — batched
    # ------------------------------------------------------------------

    def _run_inference_batch(self, texts: List[str], batch_size: int = 1) -> List[tuple[list, list]]:
        """
        Run GLiNER NER + relation inference over a list of texts.

        Args:
            texts:      Texts to process.
            batch_size: ``1`` = one-by-one, ``0`` = full batch, ``N`` = mini-batches.

        Returns a list of ``(entities, relations)`` tuples, one per input text.
        """
        def _unwrap(x: Any) -> list:
            if isinstance(x, list) and len(x) == 1 and isinstance(x[0], list):
                return x[0]
            return x or []

        effective_batch = batch_size if batch_size > 0 else len(texts)
        batches         = [texts[i : i + effective_batch] for i in range(0, len(texts), effective_batch)]
        results: List[tuple[list, list]] = []

        for batch in tqdm(batches, desc="Extracting graph edges", unit="batch"):
            raw_entities, raw_relations = self._ner_extractor.inference(
                batch,
                labels=self.labels,
                relations=self.relations,
                threshold=self.threshold,
                relation_threshold=self.relation_threshold,
                return_relations=True,
            )
            for ents, rels in zip(raw_entities, raw_relations):
                results.append((_unwrap(ents), _unwrap(rels)))

        return results

    # ------------------------------------------------------------------
    # Edge extraction
    # ------------------------------------------------------------------

    def _extract_entity_edges(self, entities: list, child_id: str) -> list[dict]:
        return [
            {
                "head":        self._normalize(e["text"]),
                "head_type":   e["label"],
                "relation":    RELATION_IN_CHUNK,
                "tail":        child_id,
                "tail_type":   NODE_TYPE_CHUNK,
                "score":       float(e.get("score", 0.0)),
                "graph_type":  GRAPH_TYPE_ENTITY,
                "edge_source": "NONE",
            }
            for e in entities
            if e.get("text") and e.get("label")
        ]

    def _extract_relation_edges(self, relations: list, child_id: str) -> list[dict]:
        return [
            {
                "head":        self._normalize(r["head"]["text"]),
                "head_type":   r["head"].get("type", ""),
                "relation":    r["relation"],
                "tail":        self._normalize(r["tail"]["text"]),
                "tail_type":   r["tail"].get("type", ""),
                "score":       float(r.get("score", 0.0)),
                "graph_type":  GRAPH_TYPE_TRIPLET,
                "edge_source": child_id,
            }
            for r in relations
            if r.get("head") and r.get("tail") and r.get("relation")
            and r["head"].get("text") and r["tail"].get("text")
        ]

    # ------------------------------------------------------------------
    # Core extraction loop
    # ------------------------------------------------------------------

    def build_edge_list(
        self,
        children: List[Document],
        child_ids: List[str],
        gliner_batch_size: int = 1,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run GLiNER over every child chunk and produce two edge DataFrames:
          - ``entity_df``   — entity ↔ child bipartite links
          - ``relation_df`` — entity → entity typed triplets

        Args:
            children:          Child chunk documents (post-split).
            child_ids:         Vectorstore-assigned IDs in the same order.
            gliner_batch_size: ``1`` = one-by-one (default), ``0`` = full batch,
                               ``N`` = mini-batches of size N.
        """
        texts = [c.page_content for c in children]

        entity_rows:   list[dict] = []
        relation_rows: list[dict] = []

        inferences = self._run_inference_batch(texts, batch_size=gliner_batch_size)

        for (raw_entities, raw_relations), child_id in zip(inferences, child_ids):
            entity_rows.extend(self._extract_entity_edges(raw_entities, child_id))
            relation_rows.extend(self._extract_relation_edges(raw_relations, child_id))

        return (
            self._rows_to_df(entity_rows,   ENTITY_DEDUP_COLS),
            self._rows_to_df(relation_rows, TRIPLET_DEDUP_COLS),
        )

    @staticmethod
    def _rows_to_df(rows: list[dict], dedup_cols: list[str]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=EDGE_COLS)
        return (
            pd.DataFrame(rows)
            .drop_duplicates(dedup_cols)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_entity_link_graph(self, entity_edges: pd.DataFrame) -> nx.Graph:
        return nx.from_pandas_edgelist(
            df=entity_edges,
            source="head",
            target="tail",
            edge_attr=["relation"],
            create_using=nx.Graph(),
        )

    def _build_triplet_graph(self, triplet_edges: pd.DataFrame) -> Optional[nx.MultiDiGraph]:
        if triplet_edges.empty:
            return None
        G = nx.from_pandas_edgelist(
            df=triplet_edges,
            source="head",
            target="tail",
            edge_attr=["relation", "edge_source"],
            create_using=nx.MultiDiGraph(),
        )
        if self.add_inverse_relations:
            G.add_edges_from([
                (v, u, {**data, "relation": f"inverse_of:{data['relation']}", "is_inverse": True})
                for u, v, data in G.edges(data=True)
            ])
        return G

    def _rebuild_graphs(self) -> None:
        if self._edge_df is None or self._edge_df.empty:
            self._graph_entity_link = None
            self._graph_triplet     = None
            return
        entity_edges  = self._edge_df.query(f'graph_type == "{GRAPH_TYPE_ENTITY}"')
        triplet_edges = self._edge_df.query(f'graph_type == "{GRAPH_TYPE_TRIPLET}"')
        self._graph_entity_link = self._build_entity_link_graph(entity_edges)
        self._graph_triplet     = self._build_triplet_graph(triplet_edges)

    # ------------------------------------------------------------------
    # Edge DataFrame merging
    # ------------------------------------------------------------------

    def _merge_edges(self, new_edges: pd.DataFrame) -> pd.DataFrame:
        if self._edge_df is None or self._edge_df.empty:
            return new_edges
        return (
            pd.concat([self._edge_df, new_edges], ignore_index=True)
            .drop_duplicates()
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _parquet_path(self)  -> Path: return Path(self.persist_directory) / f"{self.collection_name}.parquet"
    def _manifest_path(self) -> Path: return Path(self.persist_directory) / f"{self.collection_name}.json"
    def _parent_store_path(self) -> Path: return Path(self.persist_directory) / f"{self.collection_name}.parents.json"

    _MANIFEST_KEYS = ["labels", "relations", "threshold", "relation_threshold",
                      "child_chunk_size", "child_chunk_overlap"]

    def _persist(self) -> None:
        if self.persist_directory is None or self._edge_df is None:
            return
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        self._edge_df.to_parquet(self._parquet_path())
        self._manifest_path().write_text(
            json.dumps({k: getattr(self, k) for k in self._MANIFEST_KEYS}, indent=2),
            encoding="utf-8",
        )
        self._parent_store_path().write_text(
            json.dumps({
                pid: {"page_content": doc.page_content, "metadata": doc.metadata, "id": doc.id}
                for pid, doc in self._parent_store.items()
            }, indent=2),
            encoding="utf-8",
        )

    def load(self) -> "GLiNERGraphRetriever":
        """
        Restore a previously persisted graph store from disk.

        Note: the child vectorstore must be re-populated separately.
        """
        if self.persist_directory is None:
            raise ValueError("persist_directory is None; cannot load.")

        parquet_path = self._parquet_path()
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        self._edge_df = pd.read_parquet(parquet_path)

        manifest_path = self._manifest_path()
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        missing  = [k for k in self._MANIFEST_KEYS if k not in manifest]
        if missing:
            raise ValueError(f"Manifest is missing required keys: {missing}")
        for k in self._MANIFEST_KEYS:
            setattr(self, k, manifest[k])

        parent_store_path = self._parent_store_path()
        if not parent_store_path.exists():
            raise FileNotFoundError(f"Parent store not found: {parent_store_path}")
        self._parent_store = {
            pid: Document(page_content=e["page_content"], metadata=e["metadata"], id=e["id"])
            for pid, e in json.loads(parent_store_path.read_text(encoding="utf-8")).items()
        }

        self._rebuild_graphs()
        return self

    # ------------------------------------------------------------------
    # Public ingestion API
    # ------------------------------------------------------------------

    def from_documents(self, documents: List[Document], gliner_batch_size: int = 1) -> None:
        """
        Index a fresh set of parent documents.

        Args:
            documents:         Parent documents to index.
            gliner_batch_size: ``1`` = one-by-one (default), ``0`` = full batch,
                               ``N`` = mini-batches of size N.
        """
        children  = self._split_to_children(documents)
        child_ids = self.vectorstore.add_documents(children)
        entity_df, relation_df = self.build_edge_list(children, child_ids, gliner_batch_size)
        self._edge_df = pd.concat([entity_df, relation_df], ignore_index=True)
        self._rebuild_graphs()
        self._persist()

    def add_documents(self, documents: List[Document], gliner_batch_size: int = 1) -> None:
        """
        Incrementally add parent documents to an already-fitted store.

        Args:
            documents:         Additional parent documents to index.
            gliner_batch_size: ``1`` = one-by-one (default), ``0`` = full batch,
                               ``N`` = mini-batches of size N.
        """
        children  = self._split_to_children(documents)
        child_ids = self.vectorstore.add_documents(children)
        entity_df, relation_df = self.build_edge_list(children, child_ids, gliner_batch_size)
        new_edges     = pd.concat([entity_df, relation_df], ignore_index=True)
        self._edge_df = self._merge_edges(new_edges)
        self._rebuild_graphs()
        self._persist()

    # ------------------------------------------------------------------
    # Traversal primitives (shared by both paths)
    # ------------------------------------------------------------------

    @staticmethod
    def _nodes_at_exact_depth(graph: nx.Graph, source: Any, depth: int) -> set:
        lengths = nx.single_source_shortest_path_length(graph, source, cutoff=depth)
        return {node for node, dist in lengths.items() if dist == depth}

    def _entity_nodes_for_chunks(self, child_ids: List[str]) -> set[str]:
        if self._graph_entity_link is None:
            return set()
        return set().union(*[
            self._nodes_at_exact_depth(self._graph_entity_link, cid, depth=1)
            for cid in child_ids
            if cid in self._graph_entity_link
        ])

    @staticmethod
    def _child_ids_reachable_from(
        graph: nx.MultiDiGraph,
        entry_node: Any,
        depth: int,
    ) -> set[str]:
        queue         = deque([(entry_node, 0)])
        visited_nodes = {entry_node}
        child_ids:    set[str] = set()
        while queue:
            node, current_depth = queue.popleft()
            if current_depth >= depth:
                continue
            for _, neighbour, edge_data in graph.out_edges(node, data=True):
                source_id = edge_data.get("edge_source")
                if source_id:
                    child_ids.add(source_id)
                if neighbour not in visited_nodes:
                    visited_nodes.add(neighbour)
                    queue.append((neighbour, current_depth + 1))
        return child_ids

    # ------------------------------------------------------------------
    # Path 2 — LLM-assisted traversal helpers
    # ------------------------------------------------------------------

    def get_entry_entities(self, seed_child_ids: List[str]) -> Dict[str, str]:
        """
        Collect candidate entities from seed children.

        Returns a ``{entity_text: entity_type}`` mapping for all entities
        found in the seed child chunks.
        """
        entity_nodes = self._entity_nodes_for_chunks(seed_child_ids)
        if self._edge_df is None or not entity_nodes:
            return {}
        entity_edges = self._edge_df.query(f'graph_type == "{GRAPH_TYPE_ENTITY}"')
        return {
            entity_text: entity_edges.loc[entity_edges["head"] == entity_text, "head_type"].iloc[0]
            for entity_text in entity_nodes
            if not entity_edges.loc[entity_edges["head"] == entity_text, "head_type"].empty
        }

    def filter_entities_with_llm(
        self,
        query: str,
        entities: Dict[str, str],
        seed_children: List[Document],
        llm: Any,
    ) -> Dict[str, str]:
        """
        Use the LLM to select relevant entities from candidates.

        Args:
            query:          The user's query string.
            entities:       Output of ``get_entry_entities``.
            seed_children:  Child documents from ``seed_search`` for grounding.
            llm:            A LangChain chat model with ``with_structured_output``.

        Returns:
            Filtered ``{entity_text: entity_type}`` dict.
        """
        if not entities:
            return {}
        chunk_context = "\n\n".join(
            f"[Chunk {i}]\n{textwrap.shorten(doc.page_content, width=400, placeholder=' …')}"
            for i, doc in enumerate(seed_children, 1)
        )
        prompt = (
            f"Query: {query}\n\n"
            f"Retrieved context:\n{chunk_context}\n\n"
            "From the candidate entities listed in the schema description, "
            "select those that are relevant to answering the query. "
            "Use the retrieved context above to inform your selection."
        )
        schema   = _build_entity_filter_schema(entities)
        decision = llm.with_structured_output(schema).invoke(prompt)
        selected = set(decision.selected_entities)
        return {name: etype for name, etype in entities.items() if name in selected}

    def get_reachable_triples(
        self,
        selected_entities: Dict[str, str],
        traversal_depth: Optional[int] = None,
    ) -> List[TripleRecord]:
        """
        Fetch triples reachable from selected entities up to ``traversal_depth`` hops.

        Args:
            selected_entities: ``{entity_text: entity_type}`` from the LLM decision.
            traversal_depth:   Hops; falls back to ``self.traversal_depth``.

        Returns:
            List of ``TripleRecord`` objects.
        """
        if self._graph_triplet is None:
            return []

        depth           = traversal_depth or self.traversal_depth
        seen_edge_keys: set[tuple]         = set()
        records:        List[TripleRecord] = []

        for entity in selected_entities:
            if entity not in self._graph_triplet:
                continue
            queue         = deque([(entity, 0)])
            visited_nodes = {entity}
            while queue:
                node, current_depth = queue.popleft()
                if current_depth >= depth:
                    continue
                for _, neighbour, edge_data in self._graph_triplet.out_edges(node, data=True):
                    edge_key = (node, neighbour, edge_data.get("relation"), edge_data.get("edge_source"))
                    if edge_key not in seen_edge_keys:
                        seen_edge_keys.add(edge_key)
                        head_type = selected_entities.get(node, "")
                        tail_type = ""
                        if self._edge_df is not None:
                            tail_row = self._edge_df.loc[self._edge_df["head"] == neighbour, "head_type"]
                            if not tail_row.empty:
                                tail_type = tail_row.iloc[0]
                        records.append(TripleRecord(
                            id          = str(uuid.uuid4()),
                            text        = f"{node} ({head_type}) → {edge_data.get('relation')} → {neighbour} ({tail_type})",
                            head        = node,
                            head_type   = head_type,
                            relation    = edge_data.get("relation", ""),
                            tail        = neighbour,
                            tail_type   = tail_type,
                            edge_source = edge_data.get("edge_source", ""),
                        ))
                    if neighbour not in visited_nodes:
                        visited_nodes.add(neighbour)
                        queue.append((neighbour, current_depth + 1))
        return records

    def filter_triples_with_llm(
        self,
        query: str,
        triples: List[TripleRecord],
        llm: Any,
    ) -> List[TripleRecord]:
        """
        Use the LLM to select relevant triples.

        Args:
            query:   The user's query string.
            triples: Output of ``get_reachable_triples``.
            llm:     A LangChain chat model with ``with_structured_output``.

        Returns:
            Filtered list of ``TripleRecord`` objects.
        """
        if not triples:
            return []
        schema   = _build_triple_filter_schema(triples)
        prompt   = (
            f"Query: {query}\n\n"
            "From the candidate triples listed in the schema description, "
            "select those that contain information relevant to answering the query."
        )
        decision     = llm.with_structured_output(schema).invoke(prompt)
        selected_ids = set(decision.selected_triple_ids)
        return [t for t in triples if t.id in selected_ids]

    def resolve_parents_from_triples(self, selected_triples: List[TripleRecord]) -> List[Document]:
        """
        Resolve child IDs from selected triples → parent documents.
        """
        child_ids = list({t.edge_source for t in selected_triples if t.edge_source})
        return self._parents_from_child_ids(child_ids)

    # ------------------------------------------------------------------
    # LangChain BaseRetriever interface
    # ------------------------------------------------------------------

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        k:               Optional[int] = None,
        traversal_depth: Optional[int] = None,
        llm:             Optional[Any] = None,
    ) -> List[Document]:
        """
        Unified retrieval dispatch — always returns parent documents.

        Per-call overrides (all optional):
          - ``k``               — number of seed children
          - ``traversal_depth`` — BFS hops for graph expansion
          - ``llm``             — triggers Path 2 (LLM-assisted) when provided

        A ``RetrieverTrace`` is stored in ``self.last_trace`` after every call.
        Both paths populate ``candidate_entities`` and ``candidate_triples``
        so you can always inspect what the graph found.

        Path 1 (no ``llm``):
          1. Seed search → child chunks
          2. ``get_entry_entities`` + ``get_reachable_triples`` (no filtering)
          3. Union of triplet child IDs + entity-link 2-hop neighbours → parent docs

        Path 2 (``llm`` provided):
          1. Seed search → child chunks
          2. ``get_entry_entities`` → ``filter_entities_with_llm``
          3. ``get_reachable_triples`` → ``filter_triples_with_llm``
          4. ``resolve_parents_from_triples``
          (falls back to seed parents if LLM selects nothing)
        """
        effective_k     = k               or self.k
        effective_depth = traversal_depth or self.traversal_depth
        trace           = RetrieverTrace()

        seed_children       = self.seed_search(query, k=effective_k)
        seed_ids            = [doc.id for doc in seed_children if doc.id]
        trace.seed_child_ids = seed_ids

        # ---- Path 2 — LLM-assisted -----------------------------------
        if llm is not None:
            trace.path = "llm"

            entities                  = self.get_entry_entities(seed_ids)
            trace.candidate_entities  = entities

            selected_ents             = self.filter_entities_with_llm(query, entities, seed_children, llm)
            trace.selected_entities   = selected_ents

            triples                   = self.get_reachable_triples(selected_ents, traversal_depth=effective_depth)
            trace.candidate_triples   = triples

            selected_triples          = self.filter_triples_with_llm(query, triples, llm)
            trace.selected_triples    = selected_triples

            if not selected_triples:
                parents = self._parents_from_child_ids(seed_ids)
            else:
                parents = self.resolve_parents_from_triples(selected_triples)

            trace.returned_parent_ids = [p.id for p in parents if p.id]
            self._last_trace          = trace
            return parents

        # ---- Path 1 — auto traversal ---------------------------------
        # Reuses the same graph-query methods as path 2 so the trace is
        # fully populated.  candidate_* == selected_* (no LLM filter).
        trace.path = "auto"

        entities                 = self.get_entry_entities(seed_ids)
        trace.candidate_entities = entities
        trace.selected_entities  = entities          # all kept — no filter

        triples                  = self.get_reachable_triples(entities, traversal_depth=effective_depth)
        trace.candidate_triples  = triples
        trace.selected_triples   = triples           # all kept — no filter

        # Also pull in entity-link neighbours (2-hop undirected) which are
        # not captured by the triplet BFS above.
        entity_linked_ids = set().union(*[
            self._nodes_at_exact_depth(self._graph_entity_link, cid, depth=2)
            for cid in seed_ids
            if self._graph_entity_link and cid in self._graph_entity_link
        ]) if self._graph_entity_link else set()

        triple_child_ids = {t.edge_source for t in triples if t.edge_source}
        all_child_ids    = list(set(seed_ids) | entity_linked_ids | triple_child_ids)

        parents                   = self._parents_from_child_ids(all_child_ids)
        trace.returned_parent_ids = [p.id for p in parents if p.id]
        self._last_trace          = trace
        return parents
