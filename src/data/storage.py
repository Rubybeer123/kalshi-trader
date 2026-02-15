"""SQLite storage management for local data persistence."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import structlog

logger = structlog.get_logger(__name__)


class DatabaseManager:
    """Manages SQLite database connections and operations."""
    
    def __init__(self, db_path: Union[str, Path]):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info("Database manager initialized", db_path=str(self.db_path))
    
    @contextmanager
    def get_connection(self):
        """
        Get a database connection as context manager.
        
        Yields:
            sqlite3.Connection object
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def execute(self, query: str, parameters: tuple = ()) -> int:
        """
        Execute a query and return row count.
        
        Args:
            query: SQL query
            parameters: Query parameters
            
        Returns:
            Number of rows affected
        """
        with self.get_connection() as conn:
            cursor = conn.execute(query, parameters)
            conn.commit()
            return cursor.rowcount
    
    def fetch_one(self, query: str, parameters: tuple = ()) -> Optional[Dict[str, Any]]:
        """
        Fetch a single row.
        
        Args:
            query: SQL query
            parameters: Query parameters
            
        Returns:
            Row as dictionary or None
        """
        with self.get_connection() as conn:
            cursor = conn.execute(query, parameters)
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def fetch_all(self, query: str, parameters: tuple = ()) -> List[Dict[str, Any]]:
        """
        Fetch all rows.
        
        Args:
            query: SQL query
            parameters: Query parameters
            
        Returns:
            List of rows as dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.execute(query, parameters)
            return [dict(row) for row in cursor.fetchall()]


def init_database(db_path: Union[str, Path]) -> None:
    """
    Initialize database with required tables.
    
    Args:
        db_path: Path to database file
    """
    db = DatabaseManager(db_path)
    
    with db.get_connection() as conn:
        # Market data cache
        conn.execute("""
            CREATE TABLE IF NOT EXISTS market_cache (
                ticker TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                cached_at INTEGER NOT NULL
            )
        """)
        
        # Trade history
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_ticker TEXT NOT NULL,
                side TEXT NOT NULL,
                price REAL NOT NULL,
                count INTEGER NOT NULL,
                timestamp INTEGER NOT NULL
            )
        """)
        
        # Create indexes
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_ticker_time 
            ON trades (market_ticker, timestamp)
        """)
        
        conn.commit()
    
    logger.info("Database initialized", db_path=str(db_path))
