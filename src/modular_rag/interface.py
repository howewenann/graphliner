from typing import List, Tuple, Optional, Any
from langchain_core.documents import Document

#============================================
# Base Interface
#============================================
class VectorStore:

    def __init__(self, embeddings: Any, **kwargs:Any):
        """Initialize a vector store with the given embeddings.

        Args:
            embeddings: The embeddings to use for similarity searches.
            kwargs: Additional keyword arguments.
        """
        raise NotImplementedError('VectorStore subclasses must implement __init__()')
    
    def add_documents(self, documents: List[Document], ids: Optional[List[str]] = None, **kwargs: Any) -> List[str]:
        """Add the given documents to the vector store.

        Args:
            documents: The list of documents to add.
            ids: Optional list of IDs for each document. If not provided, will generate unique IDs.
            kwargs: Additional keyword arguments.

        Returns:
            A list of IDs for the added documents.
        """
        raise NotImplementedError('VectorStore subclasses must implement add_documents()')
    
    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        """Search for the most similar documents to the given query.

        Args:
            query: The search query.
            k: The number of results to return. Default is 4.
            kwargs: Additional keyword arguments.

        Returns:
            A list of the k most similar documents.
        """
        raise NotImplementedError('VectorStore subclasses must implement similarity_search()')
    
    def similarity_search_with_score(self, query: str, k: int = 4, **kwargs: Any) -> List[Tuple[Document, float]]:
        """Search for the most similar documents to the given query and return their scores.

        Args:
            query: The search query.
            k: The number of results to return. Default is 4.
            kwargs: Additional keyword arguments.

        Returns:
            A list of tuples containing the k most similar documents and their similarity scores.
        """
        raise NotImplementedError('VectorStore subclasses must implement similarity_search_with_score()')
    

class Reranker:

    def rerank(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        """Rerank the given documents based on the query.

        Args:
            query: The search query.
            documents: The list of documents to rerank.

        Returns:
            A list of tuples containing the reranked documents and their scores.
        """
        raise NotImplementedError('Reranker subclasses must implement rerank()')
    

class GraphStore:

    def from_documents(self, documents: List[Document]) -> List[Document]:
        """Create a graph representation of the given documents.

        Args:
            documents: The list of documents to use for creating the graph.

        Returns:
            A list of documents representing the graph.
        """
        raise NotImplementedError('GraphStore subclasses must implement from_documents()')

    def traversal_search(self, start_doc_ids: list[str], depth=1) -> List[str]:
        """Search the graph using a traversal algorithm.

        Args:
            start_doc_ids: The IDs of the starting document(s).
            depth: The maximum depth of the traversal. Default is 1.

        Returns:
            A list of document IDs found during the traversal.
        """
        raise NotImplementedError('GraphStore subclasses must implement traversal_search()')
    


#============================================
# Optional Interface
#============================================

class Parser:

    def __init__(self, **kwargs: Any):
        """Initialize a parser with the given keyword arguments.

        Args:
            kwargs: Additional keyword arguments.
        """
        raise NotImplementedError('Parser subclasses must implement __init__()')

    def parse(self, source: Any, **kwargs: Any) -> str:
        """Parse the given source into a string representation.

        Args:
            source: The source to parse.
            kwargs: Additional keyword arguments.

        Returns:
            A string representation of the parsed source.
        """
        raise NotImplementedError('Parser subclasses must implement parse()')
    

class Chunker:

    def __init__(self, **kwargs: Any):
        """Initialize a chunker with the given keyword arguments.

        Args:
            kwargs: Additional keyword arguments.
        """
        raise NotImplementedError('Chunker subclasses must implement __init__()')

    def chunk(self, markdown: str, **kwargs: Any) -> List[Document]:
        """Split the given markdown text into chunks of documents.

        Args:
            markdown: The markdown text to split.
            kwargs: Additional keyword arguments.

        Returns:
            A list of Document objects representing the chunks.
        """
        raise NotImplementedError('Chunker subclasses must implement chunk()')
