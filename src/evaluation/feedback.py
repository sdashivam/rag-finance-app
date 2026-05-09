import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

class FeedbackManager:
    """
    Manages a continuous feedback loop by logging RAG interactions 
    and capturing user evaluations.
    """
    def __init__(self, db_path: str):
        """
        Initializes the FeedbackManager with a path to the SQLite database.

        Args:
            db_path (str): The file system path where the SQLite database is (or will be) stored.
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initializes the feedback table if it doesn't exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rag_interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    query TEXT,
                    answer TEXT,
                    retrieved_contexts TEXT,
                    user_score INTEGER,
                    user_comments TEXT,
                    metadata TEXT
                )
            """)

    def log_interaction(self, query: str, answer: str, contexts: List[Dict[str, Any]], metadata: Optional[Dict] = None) -> int:
        """Logs a RAG interaction before user feedback is provided."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO rag_interactions (timestamp, query, answer, retrieved_contexts, metadata) VALUES (?, ?, ?, ?, ?)",
                (datetime.now().isoformat(), query, answer, json.dumps(contexts), json.dumps(metadata or {}))
            )
            return cursor.lastrowid

    def submit_feedback(self, interaction_id: int, score: int, comments: str = ""):
        """Updates a specific interaction with user feedback (e.g., 1 for Good, 0 for Bad)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE rag_interactions SET user_score = ?, user_comments = ? WHERE id = ?",
                (score, comments, interaction_id)
            )

    def get_low_performance_samples(self, threshold: int = 1) -> List[Dict]:
        """Retrieves samples where users gave poor feedback for manual analysis."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM rag_interactions WHERE user_score < ? AND user_score IS NOT NULL", (threshold,))
            return [dict(row) for row in cursor.fetchall()]

    def get_feedback_stats(self) -> Dict[str, Any]:
        """Returns summary statistics of the captured feedback."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), AVG(user_score) FROM rag_interactions WHERE user_score IS NOT NULL")
            count, avg_score = cursor.fetchone()
            return {
                "total_feedback_entries": count,
                "average_user_score": avg_score if avg_score is not None else 0.0
            }

    def export_feedback_to_jsonl(self, output_file: str):
        """Exports all interactions to a JSONL file for fine-tuning or further RAGAS evaluation."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM rag_interactions")
            rows = cursor.fetchall()
            
        with open(output_file, 'w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(dict(row)) + '\n')