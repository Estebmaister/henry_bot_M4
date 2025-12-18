"""
FAISS-based retriever implementation for efficient semantic search.
"""

from src.utils import observe
import os
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from .base import BaseRetriever, RetrievedDocument
from src.config import settings
from src.utils import langfuse_client


class FAISSRetriever(BaseRetriever):
    """
    FAISS-based retriever using Sentence Transformers for embeddings.
    Provides efficient similarity search for document retrieval with persistent storage support.
    """

    def __init__(
        self,
        embedding_model: str = None,
        similarity_top_k: int = None,
        department_name: str = None,
        use_persistent_storage: bool = None,
        force_rebuild: bool = None
    ):
        """
        Initialize FAISS retriever with configuration.

        Args:
            embedding_model: Name of the embedding model
            similarity_top_k: Number of documents to retrieve
            department_name: Name of the department for persistent storage
            use_persistent_storage: Whether to use persistent storage
            force_rebuild: Whether to force rebuild the index
        """
        super().__init__(
            embedding_model or settings.embedding_model,
            similarity_top_k or settings.similarity_top_k
        )
        self._documents = []
        self._metadata = []
        self._index = None
        self._embeddings = None

        # Persistent storage configuration
        self.department_name = department_name
        self.use_persistent_storage = use_persistent_storage if use_persistent_storage is not None else settings.use_persistent_storage
        self.force_rebuild = force_rebuild if force_rebuild is not None else settings.force_rebuild_indices

        if self.use_persistent_storage and self.department_name:
            self._setup_persistent_storage_paths()

    def _setup_persistent_storage_paths(self):
        """Setup paths for persistent storage based on department name."""
        if not self.department_name:
            return

        # Create department-specific directories
        dept_faiss_dir = Path(settings.faiss_indices_dir) / \
            self.department_name
        dept_embeddings_dir = Path(
            settings.embeddings_dir) / self.department_name
        dept_metadata_dir = Path(settings.metadata_dir) / self.department_name

        # Create directories if they don't exist
        for dir_path in [dept_faiss_dir, dept_embeddings_dir, dept_metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # File paths for persistent storage
        self.index_file = dept_faiss_dir / "faiss.index"
        self.embeddings_file = dept_embeddings_dir / "embeddings.npy"
        self.metadata_file = dept_metadata_dir / "metadata.json"
        self.documents_file = dept_metadata_dir / "documents.json"

        print(
            f"Persistent storage paths configured for department: {self.department_name}")

    def _should_rebuild_index(self) -> bool:
        """
        Determine if the index should be rebuilt based on file existence and timestamps.

        Returns:
            True if index should be rebuilt, False otherwise
        """
        if not self.use_persistent_storage or not self.department_name:
            return True

        if self.force_rebuild:
            print("Force rebuild flag set, rebuilding index...")
            return True

        # Check if all required files exist
        required_files = [self.index_file, self.embeddings_file,
                          self.metadata_file, self.documents_file]
        if not all(file.exists() for file in required_files):
            print("Some persistent files missing, rebuilding index...")
            return True

        return False

    def _save_index_data(self, embeddings_array):
        """Save index data to persistent storage."""
        if not self.use_persistent_storage or not self.department_name:
            return

        try:
            # Save FAISS index
            faiss.write_index(self._index, str(self.index_file))

            # Save embeddings
            np.save(self.embeddings_file, embeddings_array)

            # Save metadata
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self._metadata, f, indent=2, ensure_ascii=False)

            # Save documents
            with open(self.documents_file, 'w', encoding='utf-8') as f:
                json.dump(self._documents, f, indent=2, ensure_ascii=False)

            print(
                f"Index data saved to persistent storage for {self.department_name}")

        except Exception as e:
            print(f"Error saving index data: {e}")

    def _load_index_data(self) -> bool:
        """
        Load index data from persistent storage.

        Returns:
            True if successfully loaded, False otherwise
        """
        if not self.use_persistent_storage or not self.department_name:
            return False

        try:
            # Load FAISS index
            if self.index_file.exists():
                self._index = faiss.read_index(str(self.index_file))
                print(f"FAISS index loaded from {self.index_file}")
            else:
                return False

            # Load embeddings
            if self.embeddings_file.exists():
                embeddings = np.load(self.embeddings_file)
                print(f"Embeddings loaded from {self.embeddings_file}")
            else:
                return False

            # Load metadata and documents
            if self.metadata_file.exists() and self.documents_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self._metadata = json.load(f)
                with open(self.documents_file, 'r', encoding='utf-8') as f:
                    self._documents = json.load(f)
                print(
                    f"Metadata and documents loaded for {self.department_name}")
            else:
                return False

            # Initialize embedding model
            self._embeddings = SentenceTransformer(self.embedding_model)

            print(
                f"✅ Successfully loaded persistent index for {self.department_name}")
            print(f"   Documents: {len(self._documents)}")
            print(
                f"   Index dimension: {embeddings.shape[1] if len(embeddings.shape) > 0 else 'unknown'}")

            return True

        except Exception as e:
            print(f"Error loading index data: {e}")
            return False

    async def initialize(self, documents_path: str) -> None:
        """
        Initialize the retriever with documents from the given path.
        Uses persistent storage if enabled and available.

        Args:
            documents_path: Path to the directory containing documents
        """
        start_time = asyncio.get_event_loop().time()

        try:
            # Try to load from persistent storage first
            if self._should_rebuild_index():
                # Need to rebuild the index
                await self._build_new_index(documents_path, start_time)
            else:
                # Try to load existing index
                if self._load_index_data():
                    end_time = asyncio.get_event_loop().time()
                    print(
                        f"FAISS retriever loaded from cache in {end_time - start_time:.2f} seconds")
                else:
                    # Failed to load, build new index
                    print("Failed to load persistent index, building new one...")
                    await self._build_new_index(documents_path, start_time)

        except Exception as e:
            print(f"Error initializing FAISS retriever: {e}")
            raise

    async def _build_new_index(self, documents_path: str, start_time: float) -> None:
        """
        Build a new index from documents.

        Args:
            documents_path: Path to the directory containing documents
            start_time: Initialization start time
        """
        # Load documents from the specified path
        await self._load_documents(documents_path)

        if not self._documents:
            print(f"Warning: No documents found in {documents_path}")
            return

        # Initialize embedding model
        self._embeddings = SentenceTransformer(self.embedding_model)

        # Generate embeddings for all documents
        print(f"Generating embeddings for {len(self._documents)} documents...")
        embeddings = self._embeddings.encode(
            self._documents,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Create FAISS index
        dimension = embeddings.shape[1]
        # Inner product for cosine similarity
        self._index = faiss.IndexFlatIP(dimension)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self._index.add(embeddings)

        # Save to persistent storage if enabled
        if self.use_persistent_storage and self.department_name:
            self._save_index_data(embeddings)

        end_time = asyncio.get_event_loop().time()
        print(
            f"FAISS retriever initialized in {end_time - start_time:.2f} seconds")
        print(
            f"Indexed {len(self._documents)} documents with dimension {dimension}")

    @observe(name="document_retrieval", as_type="retriever")
    async def retrieve(self, query: str) -> List[RetrievedDocument]:
        """
        Retrieve relevant documents for a given query.

        Args:
            query: The query to search for

        Returns:
            List of retrieved documents with similarity scores
        """
        if not self._index or not self._embeddings:
            return []

        trace = langfuse_client.create_trace(
            name="faiss_retriever_query",
            input=f"Query: {query[:100]}...",
            metadata={
                'embedding_model': self.embedding_model,
                'similarity_top_k': self.similarity_top_k
            }
        )

        try:
            # Create embedding generation span
            with trace.span(
                name="query_embedding_generation",
                input={'query': query, 'model': self.embedding_model},
                metadata={
                    'embedding_model': self.embedding_model,
                    'query_length': len(query),
                    'embedding_stage': 'query_processing'
                }
            ) as embedding_span:
                # Generate query embedding
                query_embedding = self._embeddings.encode([query])
                query_embedding = query_embedding.astype(np.float32)

                # Update embedding span
                embedding_span.update(
                    output={'embedding_shape': query_embedding.shape},
                    metadata={'embedding_dimension': query_embedding.shape[1]}
                )

            # Normalize for cosine similarity
            faiss.normalize_L2(query_embedding)

            # Search the index
            scores, indices = self._index.search(
                query_embedding,
                min(self.similarity_top_k, len(self._documents))
            )

            # Create retrieved documents
            retrieved_docs = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self._documents):
                    doc = RetrievedDocument(
                        content=self._documents[idx],
                        source=self._metadata[idx].get("source", "unknown"),
                        similarity_score=float(score),
                        metadata=self._metadata[idx]
                    )
                    retrieved_docs.append(doc)

            return retrieved_docs

        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []

    async def add_documents(self, documents: List[str], metadata: List[Dict[str, Any]]) -> None:
        """
        Add new documents to the retriever.

        Args:
            documents: List of document contents
            metadata: List of metadata dictionaries for each document
        """
        if not self._embeddings:
            raise RuntimeError(
                "Retriever not initialized. Call initialize() first.")

        try:
            # Generate embeddings for new documents
            new_embeddings = self._embeddings.encode(
                documents,
                batch_size=32,
                convert_to_numpy=True
            )

            # Normalize embeddings
            faiss.normalize_L2(new_embeddings)

            # Add to index
            self._index.add(new_embeddings)

            # Add to document lists
            self._documents.extend(documents)
            self._metadata.extend(metadata)

            print(f"Added {len(documents)} new documents to retriever")

        except Exception as e:
            print(f"Error adding documents: {e}")
            raise

    def get_document_count(self) -> int:
        """
        Get the total number of documents in the retriever.

        Returns:
            Number of documents
        """
        return len(self._documents)

    async def _load_documents(self, documents_path: str) -> None:
        """
        Load documents from the specified path.

        Args:
            documents_path: Path to the directory containing documents
        """
        doc_path = Path(documents_path)

        if not doc_path.exists():
            print(f"Warning: Documents path {documents_path} does not exist")
            return

        # Supported file extensions
        supported_extensions = {'.txt', '.md', '.markdown'}

        documents = []
        metadata = []

        # Load all supported documents
        for file_path in doc_path.rglob('*'):
            if file_path.suffix.lower() in supported_extensions:
                try:
                    # Read file content
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()

                    if content:
                        documents.append(content)
                        metadata.append({
                            'source': str(file_path.relative_to(doc_path)),
                            'file_name': file_path.name,
                            'file_size': len(content),
                            'file_type': file_path.suffix.lower()
                        })

                except Exception as e:
                    print(f"Error loading file {file_path}: {e}")

        self._documents = documents
        self._metadata = metadata

        print(f"Loaded {len(documents)} documents from {documents_path}")
