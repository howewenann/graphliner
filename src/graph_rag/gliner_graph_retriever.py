"""
GLiNERGraphRetriever
====================
A LangChain-compatible retriever that combines:
  - GLiNER NER/relation extraction → a NetworkX knowledge graph
  - A pluggable vectorstore for initial candidate retrieval
  - Two graph-traversal expansion strategies (auto and LLM-assisted)

Auto traversal  (use_llm_filter=False, the default)
----------------------------------------------------
    retriever.invoke("Who founded OpenAI?")
    # vector search → seed chunks → entity-link expand + triplet expand → docs

LLM-assisted traversal  (use_llm_filter=True)
----------------------------------------------
The retriever exposes four methods that a LangGraph node can call in
sequence.  The LLM is never imported here; this class only produces the
inputs the LLM needs and consumes its decisions.

    Step 1  retriever.get_entry_entities(seed_ids)
            → {"sam altman": "person", "openai": "organization", ...}

    Step 2  retriever.build_entity_filter_schema(entities)
            → a Pydantic model; pass to llm.with_structured_output(schema)
            → LLM decides which entities are relevant to the query

    Step 3  retriever.get_reachable_triples(selected_entities)
            → [TripleRecord(text="sam altman (person) → founded → openai (organization)"), ...]

    Step 4  retriever.build_triple_filter_schema(triples)
            → a Pydantic model; pass to llm.with_structured_output(schema)
            → LLM decides which triples are relevant to the query

    Step 5  retriever.resolve_docs_from_triples(selected_triples)
            → list[Document]

Typical ingestion
-----------------
    retriever = GLiNERGraphRetriever(
        vectorstore=my_chroma_store,
        model_path="urchade/gliner_mediumv2.1",
        collection_name="my_docs",
        labels=["person", "organization", "location"],
        relations=["founded", "located_in"],
        persist_directory="./graph_store",
    )
    retriever.from_documents(docs)  # adds docs to vectorstore + builds graph
"""

import json
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

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
    edge_source: str   # chunk ID this triple was extracted from


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class GLiNERGraphRetriever(BaseRetriever):
    """
    Retriever that builds a knowledge graph from documents using GLiNER,
    then expands vectorstore retrieval results via graph traversal.

    Args:
        vectorstore:           Any LangChain VectorStore (Chroma, FAISS, Qdrant …).
        model_path:            HuggingFace path or local path to the GLiNER model.
        collection_name:       Identifier for persisted files (parquet + manifest).
        labels:                Entity labels for GLiNER — list[str] or dict[str, str].
        persist_directory:     Directory for parquet / manifest files.  ``None`` →
                               in-memory only, no persistence.
        relations:             Relation types for GLiNER relation extraction.
        add_inverse_relations: Add reverse edges to the triplet graph.
        threshold:             Confidence threshold for entity extraction.
        relation_threshold:    Confidence threshold for relation extraction.
        graph_id_key:          Metadata key used to store the per-document graph ID.
        k:                     Number of vectorstore candidates before graph expansion.
        traversal_depth:       Hops to traverse from each seed entity node.
        use_llm_filter:        If True, ``_get_relevant_documents`` raises
                               ``NotImplementedError`` with instructions to use the
                               LangGraph helper methods instead.  Defaults to False
                               (auto traversal).
    """

    # Pydantic fields — LangChain BaseRetriever is a Pydantic v1 model
    vectorstore:           VectorStore               = Field(...)
    model_path:            str
    collection_name:       str
    labels:                Union[List[str], Dict[str, str]]
    persist_directory:     Optional[str]             = None
    relations:             List[str]                 = Field(default_factory=list)
    add_inverse_relations: bool                      = False
    threshold:             float                     = 0.7
    relation_threshold:    float                     = 0.5
    graph_id_key:          str                       = "graph_id"
    k:                     int                       = 4
    traversal_depth:       int                       = 1
    use_llm_filter:        bool                      = False

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

    def _extract_entity_edges(self, entities: list, graph_id: str) -> list[dict]:
        """
        Convert GLiNER entity dicts → ``__IN_CHUNK__`` edge records.
        Each edge links an entity node to the chunk it was found in.
        """
        return [
            {
                "head":        self._normalize(e["text"]),
                "head_type":   e["label"],
                "relation":    RELATION_IN_CHUNK,
                "tail":        graph_id,
                "tail_type":   NODE_TYPE_CHUNK,
                "score":       float(e.get("score", 0.0)),
                "graph_type":  GRAPH_TYPE_ENTITY,
                "edge_source": "NONE",
            }
            for e in entities
            if e.get("text") and e.get("label")
        ]

    def _extract_relation_edges(self, relations: list, graph_id: str) -> list[dict]:
        """
        Convert GLiNER relation dicts → typed triplet edge records.
        ``edge_source`` records which chunk each relation was extracted from.
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
                "edge_source": graph_id,
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
        keep_existing_graph_ids: bool = True,
    ) -> tuple[list[Document], pd.DataFrame, pd.DataFrame]:
        """
        Run GLiNER over every document and produce two edge DataFrames:
          - ``entity_df``   — entity ↔ chunk bipartite links
          - ``relation_df`` — entity → entity typed triplets

        Each document receives a ``graph_id`` in its metadata so it can be
        located during traversal.

        Returns:
            (enriched_documents, entity_df, relation_df)
        """
        entity_rows:   list[dict] = []
        relation_rows: list[dict] = []

        for doc in tqdm(documents, desc="Extracting graph edges"):
            doc.metadata = doc.metadata or {}

            # Assign a stable graph ID unless one already exists and we're keeping it
            if (not keep_existing_graph_ids) or (self.graph_id_key not in doc.metadata):
                doc.metadata[self.graph_id_key] = str(uuid.uuid4())

            graph_id = doc.metadata[self.graph_id_key]
            raw_entities, raw_relations = self._run_inference(doc.page_content)

            entity_rows.extend(self._extract_entity_edges(raw_entities, graph_id))
            relation_rows.extend(self._extract_relation_edges(raw_relations, graph_id))

        entity_df   = self._rows_to_df(entity_rows,   dedup_cols=ENTITY_DEDUP_COLS)
        relation_df = self._rows_to_df(relation_rows, dedup_cols=TRIPLET_DEDUP_COLS)

        return documents, entity_df, relation_df

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
            entity_node — __IN_CHUNK__ — chunk_node (graph_id)

        Two chunks that share an entity are 2 hops apart in this graph,
        which is how entity-link traversal discovers related chunks.
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
            "graph_id_key":       self.graph_id_key,
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
        required = ["graph_id_key", "labels", "relations", "threshold", "relation_threshold"]
        missing  = [k for k in required if k not in manifest]
        if missing:
            raise ValueError(f"Manifest is missing required keys: {missing}")

        self.graph_id_key       = manifest["graph_id_key"]
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

    def from_documents(
        self,
        documents: list[Document],
        keep_existing_graph_ids: bool = True,
    ) -> list[Document]:
        """
        Index a fresh set of documents:
          1. Run GLiNER to build the knowledge graph
          2. Add the enriched documents to the vectorstore
          3. Persist graph to disk

        Documents are added to the vectorstore *after* ``graph_id`` has been
        stamped onto their metadata, so the ID is always present at query time.

        Returns the enriched documents.
        """
        enriched, entity_df, relation_df = self.build_edge_list(
            documents=documents,
            keep_existing_graph_ids=keep_existing_graph_ids,
        )
        self._edge_df = pd.concat([entity_df, relation_df], ignore_index=True)
        self._rebuild_graphs()
        self._persist()

        # Add to vectorstore after graph_id is stamped on metadata
        self.vectorstore.add_documents(enriched)

        return enriched

    def add_documents(
        self,
        documents: list[Document],
        keep_existing_graph_ids: bool = True,
    ) -> list[Document]:
        """
        Incrementally add documents to an already-fitted store.
        New edges are merged with existing ones and deduplicated.
        """
        enriched, entity_df, relation_df = self.build_edge_list(
            documents=documents,
            keep_existing_graph_ids=keep_existing_graph_ids,
        )
        new_edges     = pd.concat([entity_df, relation_df], ignore_index=True)
        self._edge_df = self._merge_edges(new_edges)
        self._rebuild_graphs()
        self._persist()

        self.vectorstore.add_documents(enriched)

        return enriched

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

    def _entity_nodes_for_chunks(self, chunk_ids: list[str]) -> set[str]:
        """
        Return all entity nodes that are direct (1-hop) neighbours of the
        given chunk IDs in the entity-link graph.
        """
        if self._graph_entity_link is None:
            return set()

        return set().union(*[
            self._nodes_at_exact_depth(self._graph_entity_link, chunk_id, depth=1)
            for chunk_id in chunk_ids
            if chunk_id in self._graph_entity_link
        ])

    @staticmethod
    def _chunk_ids_reachable_from(
        graph: nx.MultiDiGraph,
        entry_node: Any,
        depth: int,
    ) -> set[str]:
        """
        BFS over the triplet graph starting from ``entry_node``.
        Returns all ``edge_source`` chunk IDs found along traversed edges,
        up to ``depth`` hops.
        """
        queue         = deque([(entry_node, 0)])
        visited_nodes = {entry_node}
        chunk_ids:    set[str] = set()

        while queue:
            node, current_depth = queue.popleft()

            if current_depth >= depth:
                continue

            for _, neighbour, edge_data in graph.out_edges(node, data=True):
                source_id = edge_data.get("edge_source")
                if source_id:
                    chunk_ids.add(source_id)

                if neighbour not in visited_nodes:
                    visited_nodes.add(neighbour)
                    queue.append((neighbour, current_depth + 1))

        return chunk_ids

    def _fetch_docs_by_chunk_ids(self, chunk_ids: list[str]) -> list[Document]:
        """
        Retrieve Document objects from the vectorstore by their graph IDs.
        Raises ``NotImplementedError`` for stores that don't support ``get_by_ids``.
        """
        if hasattr(self.vectorstore, "get_by_ids"):
            return self.vectorstore.get_by_ids(chunk_ids)

        raise NotImplementedError(
            "The configured vectorstore does not support get_by_ids. "
            "Use Chroma, FAISS, or another store that implements this method."
        )

    # ------------------------------------------------------------------
    # Path 1 — Auto graph traversal
    # ------------------------------------------------------------------

    def _auto_graph_expand(self, seed_chunk_ids: list[str]) -> list[str]:
        """
        Expand a set of seed chunk IDs automatically by traversing both graphs.

        Entity-link expansion (undirected bipartite, always 2 hops):
            seed chunk → shared entity node → neighbouring chunk

        Triplet expansion (directed multigraph, up to ``traversal_depth`` hops):
            seed chunk → entity nodes → relation edges → connected chunk IDs
        """
        # Chunks that share at least one entity with a seed chunk
        entity_linked_chunk_ids = set().union(*[
            self._nodes_at_exact_depth(self._graph_entity_link, chunk_id, depth=2)
            for chunk_id in seed_chunk_ids
            if self._graph_entity_link and chunk_id in self._graph_entity_link
        ])

        # Entity nodes attached to the seed chunks (entry points for triplet traversal)
        entry_entity_nodes = self._entity_nodes_for_chunks(seed_chunk_ids)

        # Follow relation edges from those entities to discover more chunk IDs
        relation_linked_chunk_ids: set[str] = set()
        if self._graph_triplet is not None:
            relation_linked_chunk_ids = set().union(*[
                self._chunk_ids_reachable_from(self._graph_triplet, entity, self.traversal_depth)
                for entity in entry_entity_nodes
                if entity in self._graph_triplet
            ])

        return list(set(seed_chunk_ids) | entity_linked_chunk_ids | relation_linked_chunk_ids)

    # ------------------------------------------------------------------
    # Path 2 — LLM-assisted traversal (stateless helpers for LangGraph)
    # ------------------------------------------------------------------

    def get_entry_entities(self, seed_chunk_ids: list[str]) -> Dict[str, str]:
        """
        **LangGraph step 1 of 4** — Collect candidate entities from seed chunks.

        Returns a ``{entity_text: entity_type}`` mapping for all entities
        found in the seed chunks.  Pass this to ``build_entity_filter_schema``
        to create the structured-output schema for the LLM.

        Args:
            seed_chunk_ids: Graph IDs of the chunks returned by vector search.

        Returns:
            Dict mapping normalised entity text → entity type label.
        """
        entity_nodes = self._entity_nodes_for_chunks(seed_chunk_ids)

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

    @staticmethod
    def build_entity_filter_schema(entities: Dict[str, str]) -> Type[BaseModel]:
        """
        **LangGraph step 2 of 4** — Build the LLM structured-output schema for entities.

        Creates a Pydantic model on the fly with one ``bool`` field per entity.
        The field name is a safe Python identifier (spaces → underscores); the
        field description includes the entity type so the LLM has full context.

        Usage::

            schema   = retriever.build_entity_filter_schema(entities)
            decision = llm.with_structured_output(schema).invoke(query)

            # Recover selected entities, mapping safe field names back to originals
            selected = {
                entity_text: entity_type
                for entity_text, entity_type in entities.items()
                if getattr(decision, entity_text.replace(" ", "_"), False)
            }

        Args:
            entities: Output of ``get_entry_entities``.

        Returns:
            A dynamically-created Pydantic model class.
        """
        fields = {
            entity_text.replace(" ", "_"): (
                bool,
                Field(default=False, description=f"Include '{entity_text}' ({entity_type})?"),
            )
            for entity_text, entity_type in entities.items()
        }
        return create_model("EntityFilterSchema", **fields)

    def get_reachable_triples(self, selected_entities: Dict[str, str]) -> List[TripleRecord]:
        """
        **LangGraph step 3 of 4** — Fetch triples reachable from selected entities.

        Walks the triplet graph up to ``traversal_depth`` hops from each selected
        entity and returns all traversed edges as ``TripleRecord`` objects.

        The ``text`` field of each record is human-readable::

            sam altman (person) → founded → openai (organization)

        Pass the result to ``build_triple_filter_schema``.

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
        **LangGraph step 4 of 4** — Build the LLM structured-output schema for triples.

        Same pattern as ``build_entity_filter_schema``: one ``bool`` field per
        triple, keyed by the triple's ``id`` (hyphens replaced with underscores).
        The field description is the human-readable ``text`` field.

        Usage::

            schema   = retriever.build_triple_filter_schema(triples)
            decision = llm.with_structured_output(schema).invoke(query)

            selected_triples = [
                t for t in triples
                if getattr(decision, t.id.replace("-", "_"), False)
            ]
            docs = retriever.resolve_docs_from_triples(selected_triples)

        Args:
            triples: Output of ``get_reachable_triples``.

        Returns:
            A dynamically-created Pydantic model class.
        """
        fields = {
            triple.id.replace("-", "_"): (
                bool,
                Field(default=False, description=f"Include: {triple.text}"),
            )
            for triple in triples
        }
        return create_model("TripleFilterSchema", **fields)

    def resolve_docs_from_triples(self, selected_triples: List[TripleRecord]) -> List[Document]:
        """
        Terminal step of the LLM-assisted path.

        Takes the triples the LLM selected and fetches the source Documents
        from the vectorstore.

        Args:
            selected_triples: Filtered list of ``TripleRecord`` objects.

        Returns:
            Deduplicated list of ``Document`` objects.
        """
        chunk_ids = list({t.edge_source for t in selected_triples if t.edge_source})
        return self._fetch_docs_by_chunk_ids(chunk_ids)

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
        Retrieval pipeline (auto traversal, ``use_llm_filter=False``):
          1. Similarity search → seed docs
          2. Extract graph IDs from seed docs
          3. Auto-expand via entity-link + triplet graphs
          4. Fetch and return expanded docs

        When ``use_llm_filter=True``, raises ``NotImplementedError`` with a
        step-by-step guide for wiring the LangGraph helper methods.
        """
        if self.use_llm_filter:
            raise NotImplementedError(
                "use_llm_filter=True: orchestrate retrieval via LangGraph instead of .invoke().\n\n"
                "  seed_docs    = vectorstore.similarity_search(query, k=k)\n"
                "  seed_ids     = [doc.metadata[graph_id_key] for doc in seed_docs]\n\n"
                "  # Step 1 — collect candidate entities from seed chunks\n"
                "  entities     = retriever.get_entry_entities(seed_ids)\n\n"
                "  # Step 2 — LLM selects relevant entities\n"
                "  schema       = retriever.build_entity_filter_schema(entities)\n"
                "  decision     = llm.with_structured_output(schema).invoke(query)\n"
                "  selected_ents = {e: t for e, t in entities.items()\n"
                "                   if getattr(decision, e.replace(' ', '_'), False)}\n\n"
                "  # Step 3 — collect triples reachable from selected entities\n"
                "  triples      = retriever.get_reachable_triples(selected_ents)\n\n"
                "  # Step 4 — LLM selects relevant triples\n"
                "  schema       = retriever.build_triple_filter_schema(triples)\n"
                "  decision     = llm.with_structured_output(schema).invoke(query)\n"
                "  selected_triples = [t for t in triples\n"
                "                      if getattr(decision, t.id.replace('-','_'), False)]\n\n"
                "  docs         = retriever.resolve_docs_from_triples(selected_triples)\n"
            )

        # --- Auto path ---
        seed_docs = self.vectorstore.similarity_search(query, k=self.k)

        seed_ids = [
            doc.metadata[self.graph_id_key]
            for doc in seed_docs
            if self.graph_id_key in doc.metadata
        ]

        # Graph not built yet — fall back to raw vector results
        if not seed_ids or self._graph_entity_link is None:
            return seed_docs

        expanded_ids = self._auto_graph_expand(seed_ids)
        return self._fetch_docs_by_chunk_ids(expanded_ids)
