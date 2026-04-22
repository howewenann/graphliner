from typing import List, Tuple

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from .interface import Reranker

class CrossEncoderReranker(Reranker):

    """
    A reranker that uses a cross-encoder model to rank documents based on their relevance to a query.
    """

    def __init__(self, model_name_or_path: str):
        """
        Initializes the reranker with a pre-trained cross-encoder model.

        Args:
            model_name_or_path (str): The name or path of the pre-trained cross-encoder model.
        """

        self.model_name_or_path = model_name_or_path
        self.cross_encoder = CrossEncoder(model_name_or_path)

    def rerank(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        """
        Ranks a list of documents based on their relevance to a query using the cross-encoder model.

        Args:
            query (str): The query string.
            documents (List[Document]): A list of documents to rank.
        Returns:
            List[Tuple[Document, float]]: A list of tuples containing the ranked documents and their corresponding scores.
        """

        if not documents:
            return []

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.cross_encoder.predict(pairs)

        reranked = list(zip(documents, scores))
        reranked.sort(key=lambda x: x[1], reverse=True)

        return reranked