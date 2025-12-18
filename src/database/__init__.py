"""
Database package for M4 delivery chatbot.

This package provides SQLite database management for the e-commerce delivery system,
including products, orders, customers, and conversation tracking.

AI Assistant Notes:
- Uses SQLAlchemy ORM with SQLite for simplicity and reliability
- Migration system for schema versioning
- Repository pattern for clean data access
- Connection pooling for performance
"""

from .connection import DatabaseConnection
from .models import ProductModel, OrderModel, OrderItemModel, ConversationModel
from .repositories import ProductRepository, OrderRepository, ConversationRepository
from .migrations import DatabaseMigrations

__all__ = [
    "DatabaseConnection",
    "ProductModel",
    "OrderModel",
    "OrderItemModel",
    "ConversationModel",
    "ProductRepository",
    "OrderRepository",
    "ConversationRepository",
    "DatabaseMigrations",
]