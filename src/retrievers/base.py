"""
Base retriever interface for RAG agents.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class RetrievedDocument:
    """Dataclass for retrieved document information."""
    content: str
    source: str
    similarity_score: float
    metadata: Dict[str, Any]


class BaseRetriever(ABC):
    """Abstract base class for document retrievers."""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        similarity_top_k: int = 3
    ):
        """
        Initialize the retriever with configuration.

        Args:
            embedding_model: Name of the embedding model to use
            similarity_top_k: Number of documents to retrieve
        """
        self.embedding_model = embedding_model
        self.similarity_top_k = similarity_top_k
        self._vector_store = None
        self._embeddings = None

    @abstractmethod
    async def initialize(self, documents_path: str) -> None:
        """
        Initialize the retriever with documents from the given path.

        Args:
            documents_path: Path to the directory containing documents
        """
        pass

    @abstractmethod
    async def retrieve(self, query: str) -> List[RetrievedDocument]:
        """
        Retrieve relevant documents for a given query.

        Args:
            query: The query to search for

        Returns:
            List of retrieved documents with similarity scores
        """
        pass

    @abstractmethod
    async def add_documents(self, documents: List[str], metadata: List[Dict[str, Any]]) -> None:
        """
        Add new documents to the retriever.

        Args:
            documents: List of document contents
            metadata: List of metadata dictionaries for each document
        """
        pass

    @abstractmethod
    def get_document_count(self) -> int:
        """
        Get the total number of documents in the retriever.

        Returns:
            Number of documents
        """
        pass