from typing import List, Tuple, Optional, Any
from langchain_core.documents import Document

#============================================
# Base Interface
#============================================
class VectorStore:

    def __init__(self, embeddings: Any, **kwargs:Any):
        raise NotImplementedError('VectorStore subclasses must implement __init__()')
    
    def add_documents(self, documents: List[Document], ids: Optional[List[str]] = None, **kwargs: Any) -> List[str]:
        raise NotImplementedError('VectorStore subclasses must implement add_documents()')
    
    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        raise NotImplementedError('VectorStore subclasses must implement similarity_search()')
    
    def similarity_search_with_score(self, query: str, k: int = 4, **kwargs: Any) -> List[Tuple[Document, float]]:
        raise NotADirectoryError('VectorStore subclasses must implement similarity_search_with_score()')
    

class Reranker:

    def rerank(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        raise NotImplementedError('Reranker subclasses must implement rerank()')
    

class GraphStore:

    def from_documents(self, documents: List[Document]) -> List[Document]:
        raise NotImplementedError('GraphStore subclasses must implement from_documents()')

    def traversal_search(self, start_doc_ids: list[str], depth=1) -> List[str]:
        raise NotImplementedError('GraphStore subclasses must implement traversal_search()')
    


#============================================
# Optional Interface
#============================================

class Parser:

    def __init__(self, **kwargs: Any):
        raise NotImplementedError('Parser subclasses must implement __init__()')

    def parse(self, source: Any, **kwargs: Any) -> str:
        raise NotImplementedError('Parser subclasses must implement parse()')
    

class Chunker:

    def __init__(self, **kwargs: Any):
        raise NotImplementedError('Chunker subclasses must implement __init__()')

    def chunk(self, markdown: str, **kwargs: Any) -> List[Document]:
        raise NotImplementedError('Chunker subclasses must implement chunk()')
