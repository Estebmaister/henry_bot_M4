"""
Database connection management for SQLite.

AI Assistant Notes:
- Manages SQLite database connections with connection pooling
- Handles database initialization and schema migrations
- Provides context managers for safe database operations
- Configurable timeout and connection settings
"""

import sqlite3
import threading
from pathlib import Path
from typing import Optional, Generator
from contextlib import contextmanager
import logging

from src.config import settings

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """
    Manages SQLite database connections with thread-safe connection pooling.

    Features:
    - Thread-local connections for safety
    - Automatic database creation
    - Connection timeout management
    - Context manager support
    """

    def __init__(self, database_path: Optional[str] = None):
        """
        Initialize database connection manager.

        Args:
            database_path: Path to SQLite database file. If None, uses config.
        """
        self.database_path = Path(database_path or "database/orders.db")
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        # Thread-local storage for connections
        self._local = threading.local()

        # Connection settings
        self.timeout = getattr(settings, 'database_timeout', 30)
        self.pool_size = getattr(settings, 'database_pool_size', 5)

        logger.info(f"Database connection initialized: {self.database_path}")

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection with proper settings."""
        conn = sqlite3.connect(
            self.database_path,
            timeout=self.timeout,
            check_same_thread=False  # We handle thread safety ourselves
        )

        # Enable foreign keys and WAL mode for better performance
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA cache_size = 10000")

        # Set row factory to return dict-like rows
        conn.row_factory = sqlite3.Row

        return conn

    def get_connection(self) -> sqlite3.Connection:
        """
        Get a thread-local database connection.
        Creates new connection if needed for current thread.
        """
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = self._create_connection()
            logger.debug(f"Created new database connection for thread {threading.get_ident()}")

        return self._local.connection

    @contextmanager
    def get_cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        """
        Context manager for database cursor operations.
        Automatically commits or rolls back transactions.
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database operation failed: {e}")
            raise
        finally:
            cursor.close()

    def execute_script(self, script: str) -> None:
        """
        Execute a SQL script (useful for migrations).

        Args:
            script: SQL script to execute
        """
        conn = self.get_connection()
        try:
            conn.executescript(script)
            conn.commit()
            logger.info("SQL script executed successfully")
        except Exception as e:
            conn.rollback()
            logger.error(f"SQL script execution failed: {e}")
            raise

    def execute_query(self, query: str, params: tuple = ()) -> list:
        """
        Execute a SELECT query and return results.

        Args:
            query: SQL query to execute
            params: Query parameters

        Returns:
            List of rows as dictionaries
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def execute_update(self, query: str, params: tuple = ()) -> int:
        """
        Execute an INSERT/UPDATE/DELETE query.

        Args:
            query: SQL query to execute
            params: Query parameters

        Returns:
            Number of affected rows
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            return cursor.rowcount

    def execute_insert(self, query: str, params: tuple = ()) -> int:
        """
        Execute an INSERT query and return the last row ID.

        Args:
            query: SQL INSERT query
            params: Query parameters

        Returns:
            ID of the inserted row
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            return cursor.lastrowid

    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.

        Args:
            table_name: Name of the table to check

        Returns:
            True if table exists, False otherwise
        """
        query = """
        SELECT name FROM sqlite_master
        WHERE type='table' AND name=?
        """
        result = self.execute_query(query, (table_name,))
        return len(result) > 0

    def close_connection(self) -> None:
        """Close the current thread's database connection."""
        if hasattr(self._local, 'connection') and self._local.connection is not None:
            self._local.connection.close()
            self._local.connection = None
            logger.debug(f"Closed database connection for thread {threading.get_ident()}")

    def close_all_connections(self) -> None:
        """Close all database connections (useful for shutdown)."""
        # Close current thread connection
        self.close_connection()

        # Note: SQLite doesn't have a way to close all thread connections
        # This would typically be called during application shutdown
        logger.info("Database connections closed")


# Global database instance
db = DatabaseConnection()