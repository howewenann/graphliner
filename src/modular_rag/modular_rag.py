import uuid
from typing import List, Tuple, Optional, Any, Callable, Dict
from langchain_core.documents import Document

class ModularRAG:

	def __init__(
		self,
		vectorstore,
		graphstore=None,
		reranker=None,
		id_fn: Optional[Callable[[Document], str]] = None
	) -> None:
		
		"""
        Initializes the ModularRAG engine.

        Args:
            vectorstore: Vector database instance (e.g., Chroma, Pinecone).
            graphstore: Optional Graph database instance for traversal-based retrieval.
            reranker: Optional reranking model to refine retrieved document relevance.
            id_fn: Callable to generate unique document IDs; defaults to uuid4.
        """
		
		self.vectorstore = vectorstore
		self.graphstore = graphstore
		self.reranker = reranker
		self.id_fn = id_fn or (lambda _: str(uuid.uuid4()))

	def ingest(
		self,
		docs: List[Document],
		ids: Optional[List[str]] = None
	) -> List[str]:

		"""
        Processes and adds documents to the RAG system.

        If a graphstore is provided, documents are first processed to build 
        graph relationships. Documents are then assigned IDs (via id_fn or 
        provided list) and indexed into the vectorstore.

        Args:
            docs: A list of LangChain Document objects to index.
            ids: Optional list of unique string IDs corresponding to the docs. 
                 If None, id_fn is used to generate them.

        Returns:
            A list of the IDs assigned to the ingested documents.
        """
		
		#  if graph store is available
		if self.graphstore is not None:
			# build graph and return docs with graph_id
			docs = self.graphstore.from_documents(
				documents=docs
			)

		if ids is None:
			ids = [self.id_fn(d) for d in docs]

		return self.vectorstore.add_documents(documents=docs, ids=ids)

	def query(
		self,
		query_text: str,
		top_k: int = 20,
		graph_depth: int = 1,
		top_reranked_k: int = 5,
		use_scores: bool = True,
		metadata_filter: Optional[Dict[str, Any]] = None,
		**kwargs: Any
	) -> List[Tuple[Document, float]]:
		
		"""
        Executes a multi-stage retrieval process: Vector Search -> (Optional) Graph Traversal -> (Optional) Reranking.

        Args:
            query_text: The user input prompt.
            top_k: Initial number of documents to retrieve via semantic search.
            graph_depth: Hop-count for graph-based expansion (if graphstore exists).
            top_reranked_k: Final number of documents to return after reranking.
            use_scores: Whether to retrieve and return vector similarity scores.
            metadata_filter: Dictionary for pre-filtering in the vectorstore.
            **kwargs: Additional provider-specific arguments for the vectorstore.

        Returns:
            A list of (Document, score) tuples ranked by relevance.
        """
		
		if metadata_filter is not None and "filter" not in kwargs:
			kwargs["filter"] = metadata_filter

		# --- 1) Retrieve from vectorstore ---
		if use_scores:
			candidate_docs = self.vectorstore.similarity_search_with_score(
				query=query_text,
				k=top_k,
				**kwargs
			)

		else:
			retrieved = self.vectorstore.similarity_search(
				query=query_text,
				k=top_k,
				**kwargs
			)
			candidate_docs = [(doc, 0.0) for doc in retrieved]

		# --- Optional graphstore ---
		if self.graphstore is not None:

			start_doc_ids = [
				doc.metadata[self.graphstore.graph_id_key_]
				for doc, score_ in candidate_docs
			]

			candidate_docs_ids = self.graphstore.traversal_search(
				start_doc_ids=start_doc_ids,
				depth=graph_depth
			)

			# retrieve actual chunks from vector store
			candidate_strings = self.vectorstore.get(
				where={self.graphstore.graph_id_key_: {"$in": candidate_docs_ids}},
				include=["documents", "metadatas"]
			)

			graph_docs = [
				Document(page_content=doc_text, metadata=meta or {}, id=doc_id)
				for doc_id, doc_text, meta in zip(
					candidate_strings["ids"],
					candidate_strings["documents"],
					candidate_strings["metadatas"]
				)
			]

			candidate_docs = [(d, 0.0) for d in graph_docs]

		# --- 3) Optional reranking ---
		if self.reranker is not None:
			reranked = self.reranker.rerank(
				query_text, 
				[item[0] for item in candidate_docs]
				)
			return reranked[:top_reranked_k]
		
		return candidate_docs[:top_reranked_k]
