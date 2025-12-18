"""
Database migration management for M4 delivery chatbot.

AI Assistant Notes:
- Version-controlled schema changes using migration files
- Automatic migration detection and execution
- Rollback support for development
- Migration history tracking
"""

import sqlite3
from typing import List, Dict, Any
from pathlib import Path
import logging

from .connection import DatabaseConnection

logger = logging.getLogger(__name__)


class DatabaseMigrations:
    """
    Manages database schema migrations with version tracking.

    Features:
    - Version-controlled migrations
    - Automatic migration detection
    - Rollback capabilities
    - Migration history tracking
    """

    def __init__(self, db_connection: DatabaseConnection):
        """
        Initialize migration manager.

        Args:
            db_connection: Database connection instance
        """
        self.db = db_connection
        self.migrations_dir = Path(__file__).parent / "migrations"

    def _ensure_migrations_table(self) -> None:
        """Create migrations tracking table if it doesn't exist."""
        if not self.db.table_exists("schema_migrations"):
            migration_sql = """
            CREATE TABLE schema_migrations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version INTEGER NOT NULL UNIQUE,
                name TEXT NOT NULL,
                executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            self.db.execute_script(migration_sql)
            logger.info("Created migrations tracking table")

    def _get_applied_migrations(self) -> List[int]:
        """Get list of applied migration versions."""
        if not self.db.table_exists("schema_migrations"):
            return []

        query = "SELECT version FROM schema_migrations ORDER BY version"
        result = self.db.execute_query(query)
        return [row['version'] for row in result]

    def get_pending_migrations(self) -> List[Dict[str, Any]]:
        """
        Get list of pending migrations that need to be applied.

        Returns:
            List of migration dictionaries with version, name, and sql
        """
        self._ensure_migrations_table()
        applied_versions = set(self._get_applied_migrations())

        # Define migrations in order
        all_migrations = [
            {
                "version": 1,
                "name": "initial_schema",
                "sql": self._get_initial_schema_sql()
            },
            {
                "version": 2,
                "name": "add_indexes",
                "sql": self._get_indexes_sql()
            },
            {
                "version": 3,
                "name": "add_triggers",
                "sql": self._get_triggers_sql()
            }
        ]

        pending = [
            migration for migration in all_migrations
            if migration["version"] not in applied_versions
        ]

        return pending

    def run_migrations(self) -> None:
        """Run all pending migrations."""
        pending = self.get_pending_migrations()

        if not pending:
            logger.info("No pending migrations to apply")
            return

        logger.info(f"Applying {len(pending)} migrations")

        with self.db.get_cursor() as cursor:
            for migration in pending:
                try:
                    # Execute migration SQL
                    if migration["sql"]:
                        cursor.executescript(migration["sql"])

                    # Record migration
                    cursor.execute(
                        "INSERT INTO schema_migrations (version, name) VALUES (?, ?)",
                        (migration["version"], migration["name"])
                    )

                    logger.info(f"Applied migration {migration['version']}: {migration['name']}")

                except Exception as e:
                    logger.error(f"Migration {migration['version']} failed: {e}")
                    raise

        logger.info("All migrations applied successfully")

    def rollback_to_version(self, target_version: int) -> None:
        """
        Rollback database to a specific version (development only).

        Args:
            target_version: Target version to rollback to
        """
        logger.warning("Rollback functionality is for development only")
        # In production, we'd implement proper rollback migrations
        raise NotImplementedError("Rollback not implemented for production safety")

    def _get_initial_schema_sql(self) -> str:
        """Get SQL for initial database schema."""
        return """
        -- Products Table
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sku TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            price DECIMAL(10,2) NOT NULL,
            category TEXT NOT NULL,
            stock_count INTEGER NOT NULL DEFAULT 0,
            weight_kg DECIMAL(5,2),
            dimensions TEXT, -- JSON: {length, width, height}
            is_active BOOLEAN DEFAULT true,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Orders Table
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_number TEXT UNIQUE NOT NULL,
            customer_name TEXT NOT NULL,
            customer_email TEXT,
            customer_phone TEXT,
            delivery_address TEXT NOT NULL,
            order_status TEXT DEFAULT 'pending',
            total_amount DECIMAL(10,2) NOT NULL,
            items TEXT NOT NULL, -- JSON array of order items
            special_instructions TEXT,
            estimated_delivery TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Order Items Table (for detailed tracking)
        CREATE TABLE IF NOT EXISTS order_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL,
            unit_price DECIMAL(10,2) NOT NULL,
            total_price DECIMAL(10,2) NOT NULL,
            FOREIGN KEY (order_id) REFERENCES orders(id) ON DELETE CASCADE,
            FOREIGN KEY (product_id) REFERENCES products(id)
        );

        -- Conversations Table (for memory management)
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            user_id TEXT,
            message_sequence INTEGER,
            message_type TEXT, -- user, assistant, system
            content TEXT,
            metadata TEXT, -- JSON for additional context
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Product Categories Table
        CREATE TABLE IF NOT EXISTS product_categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            parent_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (parent_id) REFERENCES product_categories(id)
        );

        -- Customers Table (for order history and preferences)
        CREATE TABLE IF NOT EXISTS customers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            phone TEXT,
            default_address TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

    def _get_indexes_sql(self) -> str:
        """Get SQL for performance indexes."""
        return """
        -- Product indexes
        CREATE INDEX IF NOT EXISTS idx_products_sku ON products(sku);
        CREATE INDEX IF NOT EXISTS idx_products_category ON products(category);
        CREATE INDEX IF NOT EXISTS idx_products_active ON products(is_active);
        CREATE INDEX IF NOT EXISTS idx_products_name ON products(name);

        -- Order indexes
        CREATE INDEX IF NOT EXISTS idx_orders_number ON orders(order_number);
        CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_email);
        CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(order_status);
        CREATE INDEX IF NOT EXISTS idx_orders_created ON orders(created_at);

        -- Order items indexes
        CREATE INDEX IF NOT EXISTS idx_order_items_order ON order_items(order_id);
        CREATE INDEX IF NOT EXISTS idx_order_items_product ON order_items(product_id);

        -- Conversation indexes
        CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id);
        CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id);
        CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(timestamp);

        -- Customer indexes
        CREATE INDEX IF NOT EXISTS idx_customers_email ON customers(email);
        """

    def _get_triggers_sql(self) -> str:
        """Get SQL for database triggers."""
        return """
        -- Update product updated_at timestamp
        CREATE TRIGGER IF NOT EXISTS update_products_timestamp
        AFTER UPDATE ON products
        FOR EACH ROW
        BEGIN
            UPDATE products SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
        END;

        -- Update order updated_at timestamp
        CREATE TRIGGER IF NOT EXISTS update_orders_timestamp
        AFTER UPDATE ON orders
        FOR EACH ROW
        BEGIN
            UPDATE orders SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
        END;

        -- Update customer updated_at timestamp
        CREATE TRIGGER IF NOT EXISTS update_customers_timestamp
        AFTER UPDATE ON customers
        FOR EACH ROW
        BEGIN
            UPDATE customers SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
        END;

        -- Prevent negative stock
        CREATE TRIGGER IF NOT EXISTS check_stock_count
        BEFORE UPDATE OF stock_count ON products
        FOR EACH ROW
        WHEN NEW.stock_count < 0
        BEGIN
            SELECT RAISE(ABORT, 'Stock count cannot be negative');
        END;

        -- Prevent negative prices
        CREATE TRIGGER IF NOT EXISTS check_product_price
        BEFORE UPDATE OF price ON products
        FOR EACH ROW
        WHEN NEW.price < 0
        BEGIN
            SELECT RAISE(ABORT, 'Product price cannot be negative');
        END;
        """