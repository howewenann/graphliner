import json
import uuid
import networkx as nx
import pandas as pd

from tqdm import tqdm
from typing import Optional
from pathlib import Path
from langchain_core.documents import Document

class GLiNERGraphStore:

	def __init__(
		self,
		model_path: str,
		collection_name: str,
		labels: list[str],
		persist_directory: Optional[str] = None,
		relations: Optional[list[str]] = None,
		threshold: float = 0.7,
		relation_threshold: float = 0.5,
		graph_id_key: str = "graph_id"
		) -> None:

		"""Initialize the GLiNERGraphStore with configuration parameters.

		Args:
		    model_path (str): Path to the GLiNER model
		    collection_name (str): Name of the collection to store data
		    labels (list[str]): List of entity labels to extract
		    persist_directory (Optional[str]): Directory to persist data, defaults to None
		    relations (Optional[list[str]]): List of relation types to extract, defaults to None
		    threshold (float): Confidence threshold for entity extraction, defaults to 0.7
		    relation_threshold (float): Confidence threshold for relation extraction, defaults to 0.5
		    graph_id_key (str): Key name for graph identifiers, defaults to "graph_id"
		"""

		# ---- Validate required config early ---- #
		if not labels or not isinstance(labels, list) or not all(isinstance(x, str) and x.strip() for x in labels):
			raise ValueError("labels must be a non-empty list of non-empty strings")

		self.model_path = model_path
		self.collection_name = collection_name
		self.persist_directory = persist_directory
		
		from gliner import GLiNER
		self.ner_extractor = GLiNER.from_pretrained(self.model_path)

		# fitted artifacts
		self.edge_df: Optional[pd.DataFrame] = None
		self.graph_entity_link: Optional[nx.Graph] = None
		self.graph_triplet: Optional[nx.MultiDiGraph] = None

		# store level defaults
		self.labels_: list[str] = labels
		self.relations_: list[str] = relations or []
		self.threshold_: float = float(threshold)
		self.relation_threshold_: float = float(relation_threshold)
		self.graph_id_key_: str = graph_id_key

	# --------------------------------------------------------------------
	# Paths + manifest helpers
	# --------------------------------------------------------------------
	def _parquet_path(self) -> Path:
		"""Return the path to the parquet file for this collection."""
		return Path(self.persist_directory) / f"{self.collection_name}.parquet"
	
	def _manifest_path(self) -> Path:
		"""Return the path to the manifest file for this collection."""
		return Path(self.persist_directory) / f"{self.collection_name}.json"
	
	def _save_manifest(self) -> None:
		"""Save the current configuration to a manifest file."""
		if self.persist_directory is None:
			return
		
		manifest = {
			"graph_id_key": self.graph_id_key_,
			"labels": self.labels_,
			"relations": self.relations_,
			"threshold": self.threshold_,
			"relation_threshold": self.relation_threshold_,
		}

		self._manifest_path().write_text(json.dumps(manifest, indent=2), encoding="utf-8")

	def _load_manifest(self) -> None:
		"""Load configuration from the manifest file."""
		if self.persist_directory is None:
			raise ValueError("persist_directory is None;  cannot load manifest")
		
		path = self._manifest_path()
		if not path.exists():
			raise ValueError(f"Missing manifest file {path}")
		
		manifest = json.loads(path.read_text(encoding="utf-8"))
		self.graph_id_key_ = manifest.get("graph_id_key")
		self.labels_ = manifest.get("labels")
		self.relations_ = manifest.get("relations")
		self.threshold_ = manifest.get("threshold")
		self.relation_threshold_ = manifest.get("relation_threshold")

		if self.graph_id_key_ is None or self.labels_ is None or self.relations_ is None or self.threshold_ is None or self.relation_threshold_ is None:
			raise ValueError('Manifest missing required keys')

	# ----- Extract Entities and enrich metadata ----- #
	@staticmethod
	def _norm(s: str) -> str:
		"""Normalize string by lowercasing and removing extra whitespace."""
		return " ".join(str(s).strip().lower().split())
	
	@staticmethod
	def _unwrap_singleton(x):
		"""
		Unwrap nested lists from GLiNER output.
		GLiNER may return [items] or [[items]] depending on mode.
		"""
		if isinstance(x, list) and len(x) == 1 and isinstance(x[0], list):
			return x[0]
		return x

	def build_edge_list(
		self,
		documents: list[Document],
		keep_existing_graph_ids: bool = True,
		):

		"""Build edge lists from documents using GLiNER entity and relation extraction.

		Args:
			documents (list[Document]): List of documents to process
			keep_existing_graph_ids (bool): Whether to preserve existing graph IDs, defaults to True

		Returns:
			tuple[list[Document], pd.DataFrame, pd.DataFrame]: Updated documents and edge dataframes for entities and relations
		"""
		
		entities_edge_lst: list[dict] = []
		relations_edge_lst: list[dict] = []

		for doc in tqdm(documents):
			doc.metadata = doc.metadata or {}

			# graph_ids is seperate from vector store ids
			if (not keep_existing_graph_ids) or (self.graph_id_key_ not in doc.metadata):
				doc.metadata[self.graph_id_key_] = str(uuid.uuid4())

			# ----- GLiNER inference ----- #
			raw_entities, raw_relations = self.ner_extractor.inference(
				doc.page_content,
				labels = self.labels_,
				relations = self.relations_,
				threshold = self.threshold_,
				relation_threshold = self.relation_threshold_,
				return_relations = True
			)

			raw_entities = self._unwrap_singleton(raw_entities) or []
			raw_relations = self._unwrap_singleton(raw_relations) or []

			# ----- build entity linked edges ----- #
			if raw_entities:
				entities_ = [
					{
						"head": self._norm(e["text"]),
						"head_type": e["label"],
						"relation": "__IN_CHUNK__",
						"tail": doc.metadata[self.graph_id_key_],
						"tail_type": "_CHUNK_",
						"score": float(e.get("score", 0.0)),
						"graph_type": "ENTITY_LINK",
						"edge_source": "NONE",
					}
					for e in raw_entities
					if e.get("text") and e.get("label")
				]
				entities_edge_lst.extend(entities_)

			# ----- build relations edges ----- #
			if raw_relations:
				relations_ = [
					{
						"head": self._norm(r["head"]["text"]),
						"head_type": r["head"].get("type", ""),
						"relation": r["relation"],
						"tail": self._norm(r["tail"]["text"]),
						"tail_type": r["tail"].get("type", ""),
						"score": float(r.get("score", 0.0)),
						"graph_type": "TRIPLET",
						"edge_source": doc.metadata[self.graph_id_key_],							
					}
					for r in raw_relations
					if r.get("head")
					and r.get("tail")
					and r.get("relation")
					and r["head"].get("text")
					and r["tail"].get("text")
				]
				relations_edge_lst.extend(relations_)

		# ----- Build dataframe after loop ----- #
		if entities_edge_lst:
			entities_edge_df = (
				pd.DataFrame(entities_edge_lst)
				.drop_duplicates(["head", "head_type", "relation", "tail", "tail_type"])
				.reset_index(drop=True)
			)
		else:
			entities_edge_df = (
				pd.DataFrame(columns=["head", "head_type", "relation", "tail", "tail_type", "score", "graph_type", "edge_source"])
			)

		if relations_edge_lst:
			relations_edge_df = (
				pd.DataFrame(relations_edge_lst)
				.drop_duplicates(["head", "head_type", "relation", "tail", "tail_type", "edge_source"]) # allow for edges to belong to multiple sources
				.reset_index(drop=True)
			)
		else:
			relations_edge_df = (
				pd.DataFrame(columns=["head", "head_type", "relation", "tail", "tail_type", "score", "graph_type", "edge_source"])
			)

		return documents, entities_edge_df, relations_edge_df


	def _rebuild_graph(self) -> None:

		if self.edge_df is None:
			self.graph_entity_link = None
			self.graph_triplet = None
			return
		
		entity_link = self.edge_df.query('graph_type == "ENTITY_LINK"')
		triplet = self.edge_df.query('graph_type == "TRIPLET"')

		self.graph_entity_link = nx.from_pandas_edgelist(
			df = entity_link,
			source='head',
			target='tail',
			edge_attr=['relation'],
			create_using=nx.Graph() 
			)
		
		if len(triplet) > 0:
			self.graph_triplet = nx.from_pandas_edgelist(
				df = triplet,
				source='head',
				target='tail',
				edge_attr=['relation', 'edge_source'],
				create_using=nx.MultiDiGraph() 
			)
		
	# ----- Persistence: parquet + manifest (same base name) ------ #
	def _persist(self) -> None:

		if self.persist_directory is None or self.edge_df is None:
			return
		
		Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

		# 1) Save parquet artiface
		self.edge_df.to_parquet(self._parquet_path())

		# 2) Save fitted config manifest
		self._save_manifest()

	def load(self) -> "GLiNERGraphStore":

		if self.persist_directory is None:
			raise ValueError("persist_directory is None; cannot load.")

		parquet_path = self._parquet_path()
		if not parquet_path.exists():
			raise FileNotFoundError(f"Missing parquet file: {parquet_path}")
		
		# Load parquet edge list
		self.edge_df = pd.read_parquet(parquet_path)

		# Load manifest
		self._load_manifest()

		# rebuild graph
		self._rebuild_graph()

		return self
	
	# --------------------------------------------------------------------
	# LangChain-ish API: from_documents / add_documents
	# --------------------------------------------------------------------

	def from_documents(
			self, 
			documents: list[Document],
			keep_existing_graph_ids: bool = True
			):
		
		enriched, entities_edge_df, relations_edge_df = self.build_edge_list(
			documents=documents,
			keep_existing_graph_ids=keep_existing_graph_ids,
		)

		self.edge_df = pd.concat([entities_edge_df, relations_edge_df], axis=0, ignore_index=True)
		self._rebuild_graph()

		# does not persist if persist_directory is None
		self._persist()

		return enriched
	
	def add_documents(
			self, 
			documents: list[Document],
			keep_existing_graph_ids: bool = True
			):
		
		if self.graph_id_key_ is None or self.labels_ is None or self.relations_ is None or self.threshold_ is None or self.relation_threshold_ is None:
			raise ValueError("Store is not fitted. Call from_documents(...) first or load().")
		
		# Entrich + extract new edges using store defaults
		enriched, entities_edge_df, relations_edge_df = self.build_edge_list(
			documents=documents,
			keep_existing_graph_ids=keep_existing_graph_ids,
		)		
		
		new_edges = pd.concat([entities_edge_df, relations_edge_df], axis=0, ignore_index=True)

		if self.edge_df is None or self.edge_df.empty:
			self.edge_df = new_edges
		else:
			# Append + deduplicate
			self.edge_df = (
				pd.concat([self.edge_df, new_edges], axis=0, ignore_index=True)
				.drop_duplicates()
				.reset_index(drop=True)
			)
		
		self._rebuild_graph()
		self._persist()

		return enriched
	
	#TODO: Add methods for graph traversal
