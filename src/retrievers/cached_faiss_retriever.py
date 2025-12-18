"""
Enhanced FAISS-based retriever with persistent caching for embeddings and indices.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from .base import BaseRetriever, RetrievedDocument
from src.config import settings
import shutil


class CachedFAISSRetriever(BaseRetriever):
    """
    Enhanced FAISS retriever with persistent caching for embeddings and indices.
    Provides efficient similarity search with fast startup times through caching.
    """

    def __init__(
        self,
        embedding_model: str = None,
        similarity_top_k: int = None
    ):
        """
        Initialize cached FAISS retriever with configuration.

        Args:
            embedding_model: Name of the embedding model
            similarity_top_k: Number of documents to retrieve
        """
        super().__init__(
            embedding_model or settings.embedding_model,
            similarity_top_k or settings.similarity_top_k
        )

        self._documents = []
        self._metadata = []
        self._index = None
        self._embeddings = None
        self._cache_dir = Path(settings.cache_dir)
        self._department_cache_name = None

        # Cache file paths
        self._embeddings_cache_path = None
        self._faiss_index_path = None
        self._metadata_cache_path = None
        self._fingerprint_path = None

    async def initialize(self, documents_path: str) -> None:
        """
        Initialize the retriever with documents from the given path.
        Uses cached data if available and up-to-date.

        Args:
            documents_path: Path to the directory containing documents
        """
        start_time = asyncio.get_event_loop().time()

        # Create department-specific cache name
        self._department_cache_name = Path(documents_path).name
        self._setup_cache_paths()

        try:
            # Load documents from the specified path
            await self._load_documents(documents_path)

            if not self._documents:
                print(f"Warning: No documents found in {documents_path}")
                return

            # Check if cached data is available and valid
            if await self._try_load_cache():
                end_time = asyncio.get_event_loop().time()
                print(
                    f"FAISS retriever loaded from cache in {end_time - start_time:.2f} seconds")
                print(
                    f"Indexed {len(self._documents)} documents with dimension {self._embeddings.shape[1] if hasattr(self._embeddings, 'shape') else 'unknown'}")
                return

            # Initialize embedding model and generate new embeddings
            print("Cache miss or invalid - generating new embeddings...")
            await self._generate_new_embeddings()

            # Create FAISS index
            await self._create_faiss_index()

            # Save to cache
            await self._save_cache()

            end_time = asyncio.get_event_loop().time()
            print(
                f"FAISS retriever initialized in {end_time - start_time:.2f} seconds")
            print(
                f"Indexed {len(self._documents)} documents with dimension {self._embeddings.shape[1]}")

        except Exception as e:
            print(f"Error initializing FAISS retriever: {e}")
            raise

    async def retrieve(self, query: str) -> List[RetrievedDocument]:
        """
        Retrieve relevant documents for a given query.

        Args:
            query: The query to search for

        Returns:
            List of retrieved documents with similarity scores
        """
        if self._index is None or self._embeddings is None:
            return []

        try:
            # Generate query embedding
            query_embedding = self._model.encode([query])
            query_embedding = query_embedding.astype(np.float32)

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
                    # Handle both string and dict document formats
                    if isinstance(self._documents[idx], dict):
                        content = self._documents[idx].get(
                            'content', str(self._documents[idx]))
                    else:
                        content = str(self._documents[idx])

                    doc = RetrievedDocument(
                        content=content,
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
        Add new documents to the retriever and update cache.

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

            # Save updated cache
            await self._save_cache()

            print(
                f"Added {len(documents)} new documents to retriever and updated cache")

        except Exception as e:
            print(f"Error adding documents: {e}")
            raise

    def get_document_count(self) -> int:
        """Get the total number of documents in the retriever."""
        return len(self._documents)

    def _setup_cache_paths(self) -> None:
        """Setup cache file paths for the current department."""
        dept_cache_dir = self._cache_dir / self._department_cache_name
        dept_cache_dir.mkdir(parents=True, exist_ok=True)

        self._embeddings_cache_path = dept_cache_dir / settings.embeddings_cache_file
        self._faiss_index_path = dept_cache_dir / settings.faiss_index_file
        self._metadata_cache_path = dept_cache_dir / settings.metadata_cache_file
        self._fingerprint_path = dept_cache_dir / "fingerprint.json"

    async def _try_load_cache(self) -> bool:
        """
        Try to load cached data if available and valid.

        Returns:
            True if cache was loaded successfully, False otherwise
        """
        try:
            # Check if all cache files exist
            if not all(path.exists() for path in [
                self._embeddings_cache_path,
                self._faiss_index_path,
                self._metadata_cache_path,
                self._fingerprint_path
            ]):
                return False

            # Check if cache is up-to-date by comparing fingerprints
            if not await self._is_cache_valid():
                print("Cache exists but is outdated - regenerating...")
                return False

            # Initialize the embedding model
            self._model = SentenceTransformer(self.embedding_model)

            # Load embeddings (computed embeddings as numpy array)
            self._embeddings = np.load(self._embeddings_cache_path)

            # Load FAISS index
            self._index = faiss.read_index(str(self._faiss_index_path))

            # Load metadata
            with open(self._metadata_cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                self._documents = cache_data['documents']
                self._metadata = cache_data['metadata']

            # Initialize embedding model for query processing
            self._model = SentenceTransformer(self.embedding_model)

            print(f"Loaded {len(self._documents)} documents from cache")
            return True

        except Exception as e:
            print(f"Error loading cache: {e}")
            return False

    async def _is_cache_valid(self) -> bool:
        """
        Check if cached data is still valid by comparing file fingerprints.

        Returns:
            True if cache is valid, False otherwise
        """
        try:
            # Load cached fingerprint
            with open(self._fingerprint_path, 'r', encoding='utf-8') as f:
                cached_fingerprint = json.load(f)

            # Generate current fingerprint
            current_fingerprint = await self._generate_fingerprint()

            # Compare fingerprints
            return cached_fingerprint == current_fingerprint

        except Exception as e:
            print(f"Error validating cache: {e}")
            return False

    async def _generate_fingerprint(self) -> Dict[str, Any]:
        """
        Generate a fingerprint based on document file paths and modification times.

        Returns:
            Dictionary containing fingerprint data
        """
        fingerprint = {
            'embedding_model': self.embedding_model,
            'files': {}
        }

        for doc_metadata in self._metadata:
            file_path = doc_metadata.get('source')
            if file_path:
                # Get full path to the file
                full_path = Path(doc_metadata.get('full_path', file_path))
                if full_path.exists():
                    fingerprint['files'][file_path] = {
                        'mtime': full_path.stat().st_mtime,
                        'size': full_path.stat().st_size
                    }

        return fingerprint

    async def _generate_new_embeddings(self) -> None:
        """Generate new embeddings for all documents."""
        # Initialize embedding model
        self._model = SentenceTransformer(self.embedding_model)

        # Generate embeddings for all documents
        print(f"Generating embeddings for {len(self._documents)} documents...")
        self._embeddings = self._model.encode(
            self._documents,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )

    async def _create_faiss_index(self) -> None:
        """Create FAISS index from embeddings."""
        dimension = self._embeddings.shape[1]
        # Inner product for cosine similarity
        self._index = faiss.IndexFlatIP(dimension)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self._embeddings)
        self._index.add(self._embeddings)

    async def _save_cache(self) -> None:
        """Save embeddings, index, and metadata to cache files."""
        try:
            # Ensure cache directory exists
            self._cache_dir.mkdir(parents=True, exist_ok=True)

            # Save embeddings
            np.save(self._embeddings_cache_path, self._embeddings)

            # Save FAISS index
            faiss.write_index(self._index, str(self._faiss_index_path))

            # Save metadata
            cache_data = {
                'documents': self._documents,
                'metadata': self._metadata,
                'timestamp': time.time(),
                'embedding_model': self.embedding_model
            }
            with open(self._metadata_cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)

            # Save fingerprint for cache validation
            fingerprint = await self._generate_fingerprint()
            with open(self._fingerprint_path, 'w', encoding='utf-8') as f:
                json.dump(fingerprint, f, indent=2)

            print(
                f"Cache saved to {self._cache_dir / self._department_cache_name}")

        except Exception as e:
            print(f"Error saving cache: {e}")

    async def _load_documents(self, documents_path: str) -> None:
        """
        Load documents from the specified path.
        Enhanced to store full file paths for fingerprinting.

        Args:
            documents_path: Path to the directory containing documents
        """
        doc_path = Path(documents_path).resolve()

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
                        relative_path = str(file_path.relative_to(doc_path))
                        documents.append(content)
                        metadata.append({
                            'source': relative_path,
                            # Store full path for fingerprinting
                            'full_path': str(file_path),
                            'file_name': file_path.name,
                            'file_size': len(content),
                            'file_type': file_path.suffix.lower(),
                            'modified_time': file_path.stat().st_mtime
                        })

                except Exception as e:
                    print(f"Error loading file {file_path}: {e}")

        self._documents = documents
        self._metadata = metadata

        print(f"Loaded {len(documents)} documents from {documents_path}")

    def clear_cache(self) -> None:
        """Clear all cached data for this retriever."""
        try:
            if self._department_cache_name:
                dept_cache_dir = self._cache_dir / self._department_cache_name
                if dept_cache_dir.exists():
                    shutil.rmtree(dept_cache_dir)
                    print(f"Cleared cache for {self._department_cache_name}")
        except Exception as e:
            print(f"Error clearing cache: {e}")

    @staticmethod
    def clear_all_caches() -> None:
        """Clear all cached data for all retrievers."""
        try:
            cache_dir = Path(settings.cache_dir)
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
                print("Cleared all caches")
        except Exception as e:
            print(f"Error clearing all caches: {e}")
