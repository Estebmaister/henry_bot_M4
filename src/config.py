"""
Configuration management for the multi-agent routing system.

AI Assistant Notes:
- Central configuration using Pydantic Settings for type safety and validation
- All settings can be overridden via environment variables (.env file)
- Key categories: LLM, Langfuse, RAG, Storage, Performance
- Use settings.model_name, settings.embedding_model for common access patterns
- Storage paths support both persistent and cache-based configurations
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Optional


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # LLM Configuration
    openrouter_api_key: str
    openrouter_base_url: Optional[str] = None
    model_name: str = "anthropic/claude-3.5-haiku"

    # Langfuse Configuration
    langfuse_secret_key: Optional[str] = Field(None)
    langfuse_public_key: Optional[str] = Field(None)
    langfuse_base_url: str = Field(
        "https://cloud.langfuse.com")

    # RAG Configuration
    embedding_model: str = Field("all-MiniLM-L6-v2")
    similarity_top_k: int = Field(3)
    chunk_size: int = Field(1000)
    chunk_overlap: int = Field(200)

    # Document Paths
    data_dir: str = Field("./data")
    hr_docs_dir: str = Field("./data/hr_docs")
    tech_docs_dir: str = Field("./data/tech_docs")
    finance_docs_dir: str = Field(
        "./data/finance_docs")

    # Application Configuration
    debug: bool = Field(False)
    log_level: str = Field("INFO")

    # Legacy Agent Configuration (for M3 compatibility)
    department_classes: List[str] = ["hr", "tech", "finance"]
    confidence_threshold: float = Field(0.7)

    # M4 E-commerce Configuration
    # Database Configuration
    database_url: str = Field("sqlite:///database/orders.db")
    database_pool_size: int = Field(5)
    database_timeout: int = Field(30)
    database_path: str = Field("database/orders.db")

    # Function Calling Configuration
    function_calling_model: str = Field("gpt-4")
    function_calling_timeout: int = Field(30)
    max_function_calls_per_turn: int = Field(3)
    function_calling_temperature: float = Field(0.1)

    # Order Management Configuration
    order_status_update_interval: int = Field(300)  # seconds
    default_delivery_window_days: int = Field(7)
    max_order_value: float = Field(10000.00)
    order_number_prefix: str = Field("ORD")

    # Product Catalog Configuration
    products_dir: str = Field("./data/products")
    min_product_name_length: int = Field(1)
    max_product_name_length: int = Field(200)
    min_product_description_length: int = Field(0)
    max_product_description_length: int = Field(2000)

    # Conversation Memory Configuration
    conversation_memory_size: int = Field(20)
    session_timeout_minutes: int = Field(60)
    max_conversation_tokens: int = Field(2000)
    conversation_cleanup_days: int = Field(30)

    # M4 Agent Configuration
    product_agent_similarity_top_k: int = Field(5)
    order_agent_max_tokens: int = Field(1500)
    product_agent_max_tokens: int = Field(1200)

    # E-commerce Business Logic
    low_stock_threshold: int = Field(5)
    max_items_per_order: int = Field(100)
    max_quantity_per_item: int = Field(100)
    tax_rate: float = Field(0.08)  # 8% tax rate
    shipping_cost_threshold: float = Field(50.00)  # Free shipping over $50
    standard_shipping_cost: float = Field(5.99)

    # API Rate Limiting
    function_calls_per_minute: int = Field(60)
    queries_per_minute: int = Field(120)

    # Quality and Performance
    min_agent_confidence_threshold: float = Field(0.3)
    enable_order_extraction: bool = Field(True)
    enable_conversation_memory: bool = Field(True)

    # Evaluator Configuration
    evaluator_model: str = Field("gpt-3.5-turbo")
    quality_dimensions: List[str] = ["relevance", "completeness", "accuracy"]

    # Mock Mode Configuration
    mock_mode: bool = Field(False)

    # Cache Configuration
    cache_dir: str = Field("./cache")
    embeddings_cache_file: str = Field("embeddings.npy")
    faiss_index_file: str = Field("faiss.index")
    metadata_cache_file: str = Field("metadata.json")

    # Persistent FAISS Storage Configuration
    store_dir: str = Field("./store")
    faiss_indices_dir: str = Field("./store/faiss_indices")
    embeddings_dir: str = Field("./store/embeddings")
    metadata_dir: str = Field("./store/metadata")
    use_persistent_storage: bool = Field(True)
    force_rebuild_indices: bool = Field(False)

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"  # Ignore extra fields in .env
    }


# Global settings instance
settings = Settings()
