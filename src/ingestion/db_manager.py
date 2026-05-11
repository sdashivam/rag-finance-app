"""
SQLite database manager for financial document table storage.

Handles structured storage of extracted PDF tables with metadata
for keyword-based retrieval and structured data queries.
"""

import sqlite3
import json
import os


class SQLiteManager:
    """Manages persistent storage of extracted financial tables.

    Responsibilities:
    - Create and maintain financial_tables schema
    - Insert parsed table data with source and page metadata
    - Handle duplicate cleanup on re-parsing

    Attributes:
        conn: SQLite connection instance
    """
    def __init__(self, db_path: str):
        """Initialize SQLiteManager with database path.

        Args:
            db_path: Path to SQLite database file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._setup_db()

    def _setup_db(self):
        """Create financial_tables schema if not exists."""
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS financial_tables (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_source TEXT,
                    page_number INTEGER,
                    table_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

    def insert_tables(self, file_path: str, tables: list):
        """Insert parsed table data with duplicate cleanup.

        Args:
            file_path: Source PDF file path.
            tables: List of table dicts from PDFParser.
        """
        normalized_source = os.path.abspath(file_path)
        with self.conn:
            self.conn.execute(
                "DELETE FROM financial_tables WHERE file_source IN (?, ?)",
                (file_path, normalized_source),
            )
            for table in tables:
                self.conn.execute(
                    "INSERT INTO financial_tables (file_source, page_number, table_data) VALUES (?, ?, ?)",
                    (normalized_source, table["page"], json.dumps(table))
                )
        print(f"Successfully stored {len(tables)} tables in SQLite.")

    def close(self):
        """Close SQLite connection."""
        if self.conn:
            self.conn.close()
