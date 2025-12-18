"""
Base agent interface for specialized department agents.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class AgentResponse:
    """Dataclass for agent responses."""
    answer: str
    confidence: float
    source_documents: list
    metadata: Dict[str, Any]


class BaseAgent(ABC):
    """Abstract base class for specialized department agents."""

    def __init__(
        self,
        name: str,
        department: str,
        model_name: str = "google/gemini-2.0-flash-exp:free"
    ):
        """
        Initialize the agent with configuration.

        Args:
            name: Name of the agent
            department: Department specialization (hr, tech, finance)
            model_name: Name of the LLM model to use
        """
        self.name = name
        self.department = department
        self.model_name = model_name
        self._retriever = None
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the agent and its components.
        This should be called before processing any queries.
        """
        pass

    @abstractmethod
    async def process_query(self, query: str) -> AgentResponse:
        """
        Process a query and return a specialized response.

        Args:
            query: The input query to process

        Returns:
            AgentResponse with answer, confidence, and metadata
        """
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for this agent.

        Returns:
            System prompt string
        """
        pass

    def is_initialized(self) -> bool:
        """
        Check if the agent is initialized.

        Returns:
            True if initialized, False otherwise
        """
        return self._initialized

    async def _generate_contextual_prompt(
        self,
        query: str,
        retrieved_docs: list
    ) -> str:
        """
        Generate a contextual prompt with retrieved documents.

        Args:
            query: The original query
            retrieved_docs: List of retrieved documents

        Returns:
            Contextual prompt string
        """
        system_prompt = self.get_system_prompt()

        if not retrieved_docs:
            return f"{system_prompt}\n\nQuestion: {query}"

        # Format retrieved documents
        context = "\n\n---\n\n".join([
            f"Document {i+1} (Source: {doc.source}, Relevance: {doc.similarity_score:.3f}):\n{doc.content}"
            for i, doc in enumerate(retrieved_docs)
        ])

        contextual_prompt = f"""{system_prompt}

CONTEXT:
{context}

Question: {query}

Please provide a comprehensive answer based on the context above. If the context doesn't contain sufficient information to answer the question completely, please indicate what information is missing."""

        return contextual_prompt
