import sqlite3
import json
import os

class SQLiteManager:
    """
    Handles structured storage of financial tables in SQLite.
    """
    def __init__(self, db_path: str):
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._setup_db()

    def _setup_db(self):
        """Creates the necessary table structure for financial data."""
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
        """Inserts list of table dictionaries into the database."""
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
        if self.conn:
            self.conn.close()
