"""
GLiNERGraphRetriever
====================
A LangChain-compatible retriever that combines:
  - GLiNER NER/relation extraction → a NetworkX knowledge graph
  - A pluggable vectorstore for candidate retrieval
  - Two graph-traversal expansion strategies (auto and LLM-assisted)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PIPELINE OVERVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 0 — Ingestion (always required first)
------------------------------------------
Build the knowledge graph and populate the vectorstore in one call.
The retriever uses the vectorstore's own internal document IDs as the
graph node identifiers — no extra metadata is stamped on documents.

    retriever = GLiNERGraphRetriever(
        vectorstore=my_chroma_store,
        model_path="urchade/gliner_mediumv2.1",
        collection_name="my_docs",
        labels=["person", "organization", "location"],
        relations=["founded", "located_in"],
        persist_directory="./graph_store",
    )
    ids = retriever.from_documents(docs)

    # Later — add more documents incrementally
    new_ids = retriever.add_documents(more_docs)

    # Or restore a previously persisted graph without re-running GLiNER
    retriever.load()

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Path 1 — Auto traversal  (use_llm_filter=False, the default)
-------------------------------------------------------------
The simplest path.  Call ``invoke`` (or ``get_relevant_documents``) as
with any LangChain retriever.  Graph expansion happens automatically
using both entity-link and triplet traversal.

    docs = retriever.invoke("Who founded OpenAI?")

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Path 2 — LLM-assisted traversal  (use_llm_filter=True)
-------------------------------------------------------
When ``llm`` is provided, the full path-2 pipeline runs automatically
inside ``invoke``/``get_relevant_documents``.

When ``llm`` is None, ``_get_relevant_documents`` raises
``NotImplementedError`` with step-by-step guidance for wiring the
helper methods manually inside a LangGraph node.

    retriever = GLiNERGraphRetriever(
        ...,
        use_llm_filter=True,
        llm=ChatOpenAI(model="gpt-4o"),
    )
    docs = retriever.invoke("Who founded OpenAI?")

    # Or wire it manually in LangGraph:

    seed_docs = retriever.seed_search(query)
    seed_ids  = [doc.id for doc in seed_docs]

    entities      = retriever.get_entry_entities(seed_ids)
    selected_ents = retriever.filter_entities_with_llm(query, entities, seed_docs)

    triples          = retriever.get_reachable_triples(selected_ents)
    selected_triples = retriever.filter_triples_with_llm(query, triples)

    docs = retriever.resolve_docs_from_triples(selected_triples)

"""

import json
import textwrap
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Type, Union, get_args

import networkx as nx
import pandas as pd
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel, Field, create_model
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Column-name constants — single source of truth, no magic strings in logic
# ---------------------------------------------------------------------------
EDGE_COLS          = ["head", "head_type", "relation", "tail", "tail_type", "score", "graph_type", "edge_source"]
ENTITY_DEDUP_COLS  = ["head", "head_type", "relation", "tail", "tail_type"]
TRIPLET_DEDUP_COLS = ["head", "head_type", "relation", "tail", "tail_type", "edge_source"]

GRAPH_TYPE_ENTITY  = "ENTITY_LINK"
GRAPH_TYPE_TRIPLET = "TRIPLET"
RELATION_IN_CHUNK  = "__IN_CHUNK__"
NODE_TYPE_CHUNK    = "_CHUNK_"


# ---------------------------------------------------------------------------
# Small data-carrier used by the LLM-assist path
# ---------------------------------------------------------------------------

class TripleRecord(BaseModel):
    """
    A single graph triple together with a stable ID and a human-readable
    description that can be shown to an LLM for filtering.
    """
    id:          str   # stable key used in the filter schema
    text:        str   # natural-language form shown to the LLM
    head:        str
    head_type:   str
    relation:    str
    tail:        str
    tail_type:   str
    edge_source: str   # vectorstore document ID this triple was extracted from


# ---------------------------------------------------------------------------
# Schema builders — List[Literal[...]] approach
# ---------------------------------------------------------------------------

def _build_entity_filter_schema(entities: Dict[str, str]) -> Type[BaseModel]:
    """
    Build a structured-output schema that asks the LLM to select a subset of
    entities by returning their names in a list.

    A single ``List[Literal[name1, name2, ...]]`` field is used rather than
    one bool per entity.  This lets the LLM make a single, coherent selection
    decision instead of N independent binary ones, which is both faster and
    more accurate.

    The field description includes the entity type for each candidate so the
    LLM has full context without needing to infer types from names alone.

    Args:
        entities: ``{entity_text: entity_type}`` mapping from ``get_entry_entities``.

    Returns:
        A dynamically-created Pydantic model class with a single
        ``selected_entities`` field.
    """
    if not entities:
        # Degenerate case: no candidates — return a model with an empty-only field
        return create_model(
            "EntityFilterSchema",
            selected_entities=(List[str], Field(default_factory=list, description="No entities available.")),
        )

    # Build a Literal type whose values are the entity names
    names      = tuple(entities.keys())
    lit_type   = Literal[names]  # type: ignore[valid-type]

    description = (
        "Select the entity names that are relevant to answering the query. "
        "Available entities and their types:\n"
        + "\n".join(f"  • {name}  ({etype})" for name, etype in entities.items())
    )

    return create_model(
        "EntityFilterSchema",
        selected_entities=(
            List[lit_type],  # type: ignore[valid-type]
            Field(default_factory=list, description=description),
        ),
    )


def _build_triple_filter_schema(triples: List[TripleRecord]) -> Type[BaseModel]:
    """
    Build a structured-output schema that asks the LLM to select a subset of
    triples by returning their IDs in a list.

    Same design as ``_build_entity_filter_schema``: one
    ``List[Literal[id1, id2, ...]]`` field instead of one bool per triple.

    Args:
        triples: Output of ``get_reachable_triples``.

    Returns:
        A dynamically-created Pydantic model class with a single
        ``selected_triple_ids`` field.
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
        selected_triple_ids=(
            List[lit_type],  # type: ignore[valid-type]
            Field(default_factory=list, description=description),
        ),
    )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class GLiNERGraphRetriever(BaseRetriever):
    """
    Retriever that builds a knowledge graph from documents using GLiNER,
    then expands vectorstore retrieval results via graph traversal.

    The graph uses the vectorstore's own internal document IDs as node
    identifiers.  Documents must have a populated ``id`` field (assigned
    automatically by most LangChain vectorstores) before being passed to
    ``from_documents`` or ``add_documents``.

    Args:
        vectorstore:           Any LangChain VectorStore (Chroma, FAISS, Qdrant …).
                               Always used for ingestion (``add_documents``) and
                               ``get_by_ids``.
        model_path:            HuggingFace path or local path to the GLiNER model.
        collection_name:       Identifier for persisted files (parquet + manifest).
        labels:                Entity labels for GLiNER — list[str] or dict[str, str].
        persist_directory:     Directory for parquet / manifest files.  ``None`` →
                               in-memory only, no persistence.
        relations:             Relation types for GLiNER relation extraction.
        add_inverse_relations: Add reverse edges to the triplet graph.
        threshold:             Confidence threshold for entity extraction.
        relation_threshold:    Confidence threshold for relation extraction.
        k:                     Number of vectorstore candidates before graph expansion.
        traversal_depth:       Hops to traverse from each seed entity node.
        use_llm_filter:        If True, enables the LLM-assisted traversal path.
                               Requires ``llm`` to be set for automatic execution;
                               otherwise raises ``NotImplementedError`` with manual
                               wiring instructions.  Defaults to False (auto traversal).
        llm:                   Optional LangChain chat model.  When set and
                               ``use_llm_filter=True``, the full path-2 pipeline runs
                               automatically inside ``invoke``.
    """

    # Pydantic fields — LangChain BaseRetriever is a Pydantic v1 model
    vectorstore:           VectorStore                    = Field(...)
    model_path:            str
    collection_name:       str
    labels:                Union[List[str], Dict[str, str]]
    persist_directory:     Optional[str]                  = None
    relations:             List[str]                      = Field(default_factory=list)
    add_inverse_relations: bool                           = False
    threshold:             float                          = 0.7
    relation_threshold:    float                          = 0.5
    k:                     int                            = 4
    traversal_depth:       int                            = 1
    # LLM-assisted path
    use_llm_filter:        bool                           = False
    llm:                   Optional[Any]                  = None

    # Private runtime state — excluded from the Pydantic schema
    _ner_extractor:     object                       = None
    _edge_df:           Optional[pd.DataFrame]       = None
    _graph_entity_link: Optional[nx.Graph]           = None
    _graph_triplet:     Optional[nx.MultiDiGraph]    = None

    class Config:
        arbitrary_types_allowed      = True
        underscore_attrs_are_private = True

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def model_post_init(self, __context: Any) -> None:
        """Validate labels and load the GLiNER model after Pydantic init."""
        self._validate_labels(self.labels)

        from gliner import GLiNER
        self._ner_extractor = GLiNER.from_pretrained(self.model_path)

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
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        """Lowercase and collapse internal whitespace."""
        return " ".join(str(text).strip().lower().split())

    # ------------------------------------------------------------------
    # Seed retrieval
    # ------------------------------------------------------------------

    def seed_search(self, query: str) -> List[Document]:
        """
        Run the initial similarity search that produces seed documents.

        Args:
            query: The user's query string.

        Returns:
            Up to ``self.k`` seed documents.
        """
        return self.vectorstore.similarity_search(query, k=self.k)

    # ------------------------------------------------------------------
    # GLiNER inference
    # ------------------------------------------------------------------

    def _run_inference(self, text: str) -> tuple[list, list]:
        """
        Run GLiNER NER + relation inference on ``text``.

        GLiNER can return ``[items]`` or ``[[items]]`` depending on batching
        mode; ``_unwrap`` normalises both shapes to a flat list.
        """
        def _unwrap(x: Any) -> list:
            if isinstance(x, list) and len(x) == 1 and isinstance(x[0], list):
                return x[0]
            return x or []

        raw_entities, raw_relations = self._ner_extractor.inference(
            text,
            labels=self.labels,
            relations=self.relations,
            threshold=self.threshold,
            relation_threshold=self.relation_threshold,
            return_relations=True,
        )
        return _unwrap(raw_entities), _unwrap(raw_relations)

    # ------------------------------------------------------------------
    # Edge extraction
    # ------------------------------------------------------------------

    def _extract_entity_edges(self, entities: list, doc_id: str) -> list[dict]:
        """
        Convert GLiNER entity dicts → ``__IN_CHUNK__`` edge records.
        Each edge links an entity node to the document it was found in,
        identified by the vectorstore's internal document ID.
        """
        return [
            {
                "head":        self._normalize(e["text"]),
                "head_type":   e["label"],
                "relation":    RELATION_IN_CHUNK,
                "tail":        doc_id,
                "tail_type":   NODE_TYPE_CHUNK,
                "score":       float(e.get("score", 0.0)),
                "graph_type":  GRAPH_TYPE_ENTITY,
                "edge_source": "NONE",
            }
            for e in entities
            if e.get("text") and e.get("label")
        ]

    def _extract_relation_edges(self, relations: list, doc_id: str) -> list[dict]:
        """
        Convert GLiNER relation dicts → typed triplet edge records.
        ``edge_source`` records the vectorstore document ID for the document
        each relation was extracted from.
        """
        return [
            {
                "head":        self._normalize(r["head"]["text"]),
                "head_type":   r["head"].get("type", ""),
                "relation":    r["relation"],
                "tail":        self._normalize(r["tail"]["text"]),
                "tail_type":   r["tail"].get("type", ""),
                "score":       float(r.get("score", 0.0)),
                "graph_type":  GRAPH_TYPE_TRIPLET,
                "edge_source": doc_id,
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
        documents: list[Document],
        ids: list[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run GLiNER over every document and produce two edge DataFrames:
          - ``entity_df``   — entity ↔ document bipartite links
          - ``relation_df`` — entity → entity typed triplets

        Args:
            documents: The documents to process.
            ids:       Vectorstore-assigned IDs in the same order as ``documents``.

        Returns:
            (entity_df, relation_df)
        """
        entity_rows:   list[dict] = []
        relation_rows: list[dict] = []

        for doc, doc_id in tqdm(zip(documents, ids), desc="Extracting graph edges", total=len(documents)):
            raw_entities, raw_relations = self._run_inference(doc.page_content)

            entity_rows.extend(self._extract_entity_edges(raw_entities, doc_id))
            relation_rows.extend(self._extract_relation_edges(raw_relations, doc_id))

        entity_df   = self._rows_to_df(entity_rows,   dedup_cols=ENTITY_DEDUP_COLS)
        relation_df = self._rows_to_df(relation_rows, dedup_cols=TRIPLET_DEDUP_COLS)

        return entity_df, relation_df

    @staticmethod
    def _rows_to_df(rows: list[dict], dedup_cols: list[str]) -> pd.DataFrame:
        """Convert a list of edge dicts to a deduplicated DataFrame."""
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
        """
        Build an undirected bipartite graph:
            entity_node — __IN_CHUNK__ — doc_id (vectorstore internal ID)

        Two documents that share an entity are 2 hops apart in this graph,
        which is how entity-link traversal discovers related documents.
        """
        return nx.from_pandas_edgelist(
            df=entity_edges,
            source="head",
            target="tail",
            edge_attr=["relation"],
            create_using=nx.Graph(),
        )

    def _build_triplet_graph(self, triplet_edges: pd.DataFrame) -> Optional[nx.MultiDiGraph]:
        """
        Build a directed multigraph of entity → entity typed relations.
        Optionally adds inverse edges for bidirectional traversal.
        Returns ``None`` when there are no triplet edges.
        """
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
            inverse_edges = [
                (v, u, {**data, "relation": f"inverse_of:{data['relation']}", "is_inverse": True})
                for u, v, data in G.edges(data=True)
            ]
            G.add_edges_from(inverse_edges)

        return G

    def _rebuild_graphs(self) -> None:
        """Reconstruct both in-memory graphs from the current edge DataFrame."""
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
        """Append ``new_edges`` to the existing store and deduplicate."""
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

    def _parquet_path(self) -> Path:
        return Path(self.persist_directory) / f"{self.collection_name}.parquet"

    def _manifest_path(self) -> Path:
        return Path(self.persist_directory) / f"{self.collection_name}.json"

    def _persist(self) -> None:
        """Write the edge DataFrame and config manifest to disk."""
        if self.persist_directory is None or self._edge_df is None:
            return
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        self._edge_df.to_parquet(self._parquet_path())
        self._save_manifest()

    def _save_manifest(self) -> None:
        manifest = {
            "labels":             self.labels,
            "relations":          self.relations,
            "threshold":          self.threshold,
            "relation_threshold": self.relation_threshold,
        }
        self._manifest_path().write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _load_manifest(self) -> None:
        path = self._manifest_path()
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")

        manifest = json.loads(path.read_text(encoding="utf-8"))
        required = ["labels", "relations", "threshold", "relation_threshold"]
        missing  = [k for k in required if k not in manifest]
        if missing:
            raise ValueError(f"Manifest is missing required keys: {missing}")

        self.labels             = manifest["labels"]
        self.relations          = manifest["relations"]
        self.threshold          = manifest["threshold"]
        self.relation_threshold = manifest["relation_threshold"]

    def load(self) -> "GLiNERGraphRetriever":
        """Restore a previously persisted graph store from disk."""
        if self.persist_directory is None:
            raise ValueError("persist_directory is None; cannot load.")

        parquet_path = self._parquet_path()
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        self._edge_df = pd.read_parquet(parquet_path)
        self._load_manifest()
        self._rebuild_graphs()
        return self

    # ------------------------------------------------------------------
    # Public ingestion API
    # ------------------------------------------------------------------

    def from_documents(self, documents: list[Document]) -> list[str]:
        """
        Index a fresh set of documents:
          1. Add documents to the vectorstore — returns the assigned IDs
          2. Run GLiNER to build the knowledge graph using those IDs
          3. Persist graph to disk

        Returns the list of vectorstore-assigned document IDs.
        """
        ids = self.vectorstore.add_documents(documents)

        entity_df, relation_df = self.build_edge_list(documents, ids)
        self._edge_df = pd.concat([entity_df, relation_df], ignore_index=True)
        self._rebuild_graphs()
        self._persist()

        return ids

    def add_documents(self, documents: list[Document]) -> list[str]:
        """
        Incrementally add documents to an already-fitted store.
        New edges are merged with existing ones and deduplicated.

        Returns the list of vectorstore-assigned document IDs.
        """
        ids = self.vectorstore.add_documents(documents)

        entity_df, relation_df = self.build_edge_list(documents, ids)
        new_edges     = pd.concat([entity_df, relation_df], ignore_index=True)
        self._edge_df = self._merge_edges(new_edges)
        self._rebuild_graphs()
        self._persist()

        return ids

    # ------------------------------------------------------------------
    # Traversal primitives (shared by both paths)
    # ------------------------------------------------------------------

    @staticmethod
    def _nodes_at_exact_depth(graph: nx.Graph, source: Any, depth: int) -> set:
        """
        Return nodes whose shortest-path distance from ``source`` is exactly
        ``depth``.  BFS up to ``depth`` then keep only the frontier.
        """
        lengths = nx.single_source_shortest_path_length(graph, source, cutoff=depth)
        return {node for node, dist in lengths.items() if dist == depth}

    def _entity_nodes_for_chunks(self, doc_ids: list[str]) -> set[str]:
        """
        Return all entity nodes that are direct (1-hop) neighbours of the
        given document IDs in the entity-link graph.
        """
        if self._graph_entity_link is None:
            return set()

        return set().union(*[
            self._nodes_at_exact_depth(self._graph_entity_link, doc_id, depth=1)
            for doc_id in doc_ids
            if doc_id in self._graph_entity_link
        ])

    @staticmethod
    def _chunk_ids_reachable_from(
        graph: nx.MultiDiGraph,
        entry_node: Any,
        depth: int,
    ) -> set[str]:
        """
        BFS over the triplet graph starting from ``entry_node``.
        Returns all ``edge_source`` document IDs found along traversed edges,
        up to ``depth`` hops.
        """
        queue         = deque([(entry_node, 0)])
        visited_nodes = {entry_node}
        doc_ids:      set[str] = set()

        while queue:
            node, current_depth = queue.popleft()

            if current_depth >= depth:
                continue

            for _, neighbour, edge_data in graph.out_edges(node, data=True):
                source_id = edge_data.get("edge_source")
                if source_id:
                    doc_ids.add(source_id)

                if neighbour not in visited_nodes:
                    visited_nodes.add(neighbour)
                    queue.append((neighbour, current_depth + 1))

        return doc_ids

    def _fetch_docs_by_ids(self, doc_ids: list[str]) -> list[Document]:
        """
        Retrieve Document objects from the vectorstore by their internal IDs.
        Raises ``NotImplementedError`` for stores that don't support ``get_by_ids``.
        """
        if hasattr(self.vectorstore, "get_by_ids"):
            return self.vectorstore.get_by_ids(doc_ids)

        raise NotImplementedError(
            "The configured vectorstore does not support get_by_ids. "
            "Use Chroma, FAISS, or another store that implements this method."
        )

    # ------------------------------------------------------------------
    # Path 1 — Auto graph traversal
    # ------------------------------------------------------------------

    def _auto_graph_expand(self, seed_doc_ids: list[str]) -> list[str]:
        """
        Expand a set of seed document IDs automatically by traversing both graphs.

        Entity-link expansion (undirected bipartite, always 2 hops):
            seed doc → shared entity node → neighbouring doc

        Triplet expansion (directed multigraph, up to ``traversal_depth`` hops):
            seed doc → entity nodes → relation edges → connected doc IDs
        """
        # Docs that share at least one entity with a seed doc
        entity_linked_doc_ids = set().union(*[
            self._nodes_at_exact_depth(self._graph_entity_link, doc_id, depth=2)
            for doc_id in seed_doc_ids
            if self._graph_entity_link and doc_id in self._graph_entity_link
        ])

        # Entity nodes attached to the seed docs (entry points for triplet traversal)
        entry_entity_nodes = self._entity_nodes_for_chunks(seed_doc_ids)

        # Follow relation edges from those entities to discover more doc IDs
        relation_linked_doc_ids: set[str] = set()
        if self._graph_triplet is not None:
            relation_linked_doc_ids = set().union(*[
                self._chunk_ids_reachable_from(self._graph_triplet, entity, self.traversal_depth)
                for entity in entry_entity_nodes
                if entity in self._graph_triplet
            ])

        return list(set(seed_doc_ids) | entity_linked_doc_ids | relation_linked_doc_ids)

    # ------------------------------------------------------------------
    # Path 2 — LLM-assisted traversal helpers
    # ------------------------------------------------------------------

    def get_entry_entities(self, seed_doc_ids: list[str]) -> Dict[str, str]:
        """
        **LangGraph step 1 of 4** — Collect candidate entities from seed documents.

        Returns a ``{entity_text: entity_type}`` mapping for all entities
        found in the seed documents.  Pass this to ``filter_entities_with_llm``
        (or ``build_entity_filter_schema`` for manual wiring).

        Args:
            seed_doc_ids: Vectorstore internal IDs of the documents returned
                          by similarity search (i.e. ``[doc.id for doc in seed_docs]``).

        Returns:
            Dict mapping normalised entity text → entity type label.
        """
        entity_nodes = self._entity_nodes_for_chunks(seed_doc_ids)

        if self._edge_df is None or not entity_nodes:
            return {}

        entity_edges = self._edge_df.query(f'graph_type == "{GRAPH_TYPE_ENTITY}"')

        # Resolve the entity type (GLiNER label) for each entity node text
        entities: Dict[str, str] = {}
        for entity_text in entity_nodes:
            matching = entity_edges.loc[entity_edges["head"] == entity_text, "head_type"]
            if not matching.empty:
                entities[entity_text] = matching.iloc[0]

        return entities

    # ---- Entity filtering ---------------------------------------------------

    @staticmethod
    def build_entity_filter_schema(entities: Dict[str, str]) -> Type[BaseModel]:
        """
        Build the LLM structured-output schema for entity selection.

        Returns a Pydantic model with a single ``selected_entities``
        ``List[Literal[...]]`` field.  The LLM selects a subset of entity
        names rather than answering N independent boolean questions.

        Args:
            entities: Output of ``get_entry_entities``.

        Returns:
            A dynamically-created Pydantic model class.
        """
        return _build_entity_filter_schema(entities)

    @staticmethod
    def _format_seed_chunks(seed_docs: List[Document]) -> str:
        """
        Format retrieved seed documents into a readable context block for the
        entity-filter prompt.  Truncates each chunk to 400 characters to keep
        the prompt concise while still giving the LLM grounding.
        """
        lines = []
        for i, doc in enumerate(seed_docs, 1):
            snippet = textwrap.shorten(doc.page_content, width=400, placeholder=" …")
            lines.append(f"[Chunk {i}]\n{snippet}")
        return "\n\n".join(lines)

    def filter_entities_with_llm(
        self,
        query: str,
        entities: Dict[str, str],
        seed_docs: List[Document],
    ) -> Dict[str, str]:
        """
        **LangGraph step 2 of 4** — Use the LLM to select relevant entities.

        The prompt includes both the candidate entity list *and* the retrieved
        seed chunks so the LLM can ground its decision in actual content rather
        than picking from abstract names.

        Args:
            query:      The user's query string.
            entities:   Output of ``get_entry_entities``.
            seed_docs:  The seed documents returned by the similarity search —
                        their text is injected into the prompt as context.

        Returns:
            A filtered ``{entity_text: entity_type}`` dict containing only the
            entities the LLM judged relevant.
        """
        if not entities:
            return {}

        schema        = _build_entity_filter_schema(entities)
        chunk_context = self._format_seed_chunks(seed_docs)

        prompt = (
            f"Query: {query}\n\n"
            f"Retrieved context:\n{chunk_context}\n\n"
            "From the candidate entities listed in the schema description, "
            "select those that are relevant to answering the query. "
            "Use the retrieved context above to inform your selection."
        )

        decision       = self.llm.with_structured_output(schema).invoke(prompt)
        selected_names = set(decision.selected_entities)

        return {name: etype for name, etype in entities.items() if name in selected_names}

    # ---- Triple filtering ---------------------------------------------------

    def get_reachable_triples(self, selected_entities: Dict[str, str]) -> List[TripleRecord]:
        """
        **LangGraph step 3 of 4** — Fetch triples reachable from selected entities.

        Walks the triplet graph up to ``traversal_depth`` hops from each selected
        entity and returns all traversed edges as ``TripleRecord`` objects.

        The ``text`` field of each record is human-readable::

            sam altman (person) → founded → openai (organization)

        Args:
            selected_entities: ``{entity_text: entity_type}`` from the LLM decision.

        Returns:
            List of ``TripleRecord`` objects, one per unique traversed edge.
        """
        if self._graph_triplet is None:
            return []

        seen_edge_keys: set[tuple] = set()
        records: List[TripleRecord] = []

        for entity in selected_entities:
            if entity not in self._graph_triplet:
                continue

            queue         = deque([(entity, 0)])
            visited_nodes = {entity}

            while queue:
                node, depth = queue.popleft()

                if depth >= self.traversal_depth:
                    continue

                for _, neighbour, edge_data in self._graph_triplet.out_edges(node, data=True):
                    edge_key = (node, neighbour, edge_data.get("relation"), edge_data.get("edge_source"))

                    if edge_key not in seen_edge_keys:
                        seen_edge_keys.add(edge_key)

                        head_type = selected_entities.get(node, "")

                        # Look up the neighbour's type from the entity-link edges
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
                        queue.append((neighbour, depth + 1))

        return records

    @staticmethod
    def build_triple_filter_schema(triples: List[TripleRecord]) -> Type[BaseModel]:
        """
        Build the LLM structured-output schema for triple selection.

        Returns a Pydantic model with a single ``selected_triple_ids``
        ``List[Literal[...]]`` field.

        Args:
            triples: Output of ``get_reachable_triples``.

        Returns:
            A dynamically-created Pydantic model class.
        """
        return _build_triple_filter_schema(triples)

    def filter_triples_with_llm(
        self,
        query: str,
        triples: List[TripleRecord],
    ) -> List[TripleRecord]:
        """
        **LangGraph step 4 of 4** — Use the LLM to select relevant triples.

        Unlike entity filtering, triples are self-describing (each has a
        human-readable ``text`` field), so no extra chunk context is needed.

        Args:
            query:   The user's query string.
            triples: Output of ``get_reachable_triples``.

        Returns:
            A filtered list of ``TripleRecord`` objects.
        """
        if not triples:
            return []

        schema   = _build_triple_filter_schema(triples)
        prompt   = (
            f"Query: {query}\n\n"
            "From the candidate triples listed in the schema description, "
            "select those that contain information relevant to answering the query."
        )

        decision         = self.llm.with_structured_output(schema).invoke(prompt)
        selected_ids     = set(decision.selected_triple_ids)

        return [t for t in triples if t.id in selected_ids]

    def resolve_docs_from_triples(self, selected_triples: List[TripleRecord]) -> List[Document]:
        """
        Terminal step of the LLM-assisted path.

        Takes the triples the LLM selected and fetches the source Documents
        from the vectorstore using the internal IDs stored in ``edge_source``.

        Args:
            selected_triples: Filtered list of ``TripleRecord`` objects.

        Returns:
            Deduplicated list of ``Document`` objects.
        """
        doc_ids = list({t.edge_source for t in selected_triples if t.edge_source})
        return self._fetch_docs_by_ids(doc_ids)

    # ------------------------------------------------------------------
    # LangChain BaseRetriever interface
    # ------------------------------------------------------------------

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """
        Unified retrieval dispatch:

        Path 1 (``use_llm_filter=False``):
          1. Seed search (vectorstore or pluggable retriever)
          2. Auto-expand via entity-link + triplet graphs
          3. Fetch and return expanded docs

        Path 2 (``use_llm_filter=True``, ``llm`` provided):
          1. Seed search
          2. ``get_entry_entities`` → ``filter_entities_with_llm``
          3. ``get_reachable_triples`` → ``filter_triples_with_llm``
          4. ``resolve_docs_from_triples``

        When ``use_llm_filter=True`` but ``llm`` is None, raises
        ``NotImplementedError`` with step-by-step manual wiring instructions.
        """
        # ---- Path 2, LLM provided — run automatically ----------------
        if self.use_llm_filter and self.llm is not None:
            seed_docs = self.seed_search(query)
            seed_ids  = [doc.id for doc in seed_docs if doc.id]

            entities      = self.get_entry_entities(seed_ids)
            selected_ents = self.filter_entities_with_llm(query, entities, seed_docs)

            triples          = self.get_reachable_triples(selected_ents)
            selected_triples = self.filter_triples_with_llm(query, triples)

            # Fall back to seed docs if the LLM selected nothing
            if not selected_triples:
                return seed_docs

            return self.resolve_docs_from_triples(selected_triples)

        # ---- Path 2, no LLM — raise with guidance --------------------
        if self.use_llm_filter:
            raise NotImplementedError(
                "use_llm_filter=True but no llm was provided.\n\n"
                "Option A — pass an llm to the constructor for automatic execution:\n"
                "    GLiNERGraphRetriever(..., use_llm_filter=True, llm=ChatOpenAI(...))\n\n"
                "Option B — wire the steps manually in LangGraph:\n\n"
                "    seed_docs        = retriever.seed_search(query)\n"
                "    seed_ids         = [doc.id for doc in seed_docs]\n"
                "    entities         = retriever.get_entry_entities(seed_ids)\n"
                "    selected_ents    = retriever.filter_entities_with_llm(query, entities, seed_docs)\n"
                "    triples          = retriever.get_reachable_triples(selected_ents)\n"
                "    selected_triples = retriever.filter_triples_with_llm(query, triples)\n"
                "    docs             = retriever.resolve_docs_from_triples(selected_triples)\n"
            )

        # ---- Path 1 — auto traversal ---------------------------------
        seed_docs = self.seed_search(query)
        seed_ids  = [doc.id for doc in seed_docs if doc.id]

        # Graph not built yet — fall back to raw vector results
        if not seed_ids or self._graph_entity_link is None:
            return seed_docs

        expanded_ids = self._auto_graph_expand(seed_ids)
        return self._fetch_docs_by_ids(expanded_ids)