"""
Feedback management for RAG system quality improvement.

Captures user ratings, stores interaction history, and provides
export utilities for model fine-tuning and regression analysis.
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional


class FeedbackManager:
    """Manages interaction logging and user feedback capture for RAG quality monitoring.

    Responsibilities:
    - Store query/answer/context triples with timestamps
    - Accept binary quality ratings (correct/incorrect)
    - Aggregate feedback statistics for dashboarding
    - Export data for model fine-tuning pipelines
    """
    def __init__(self, db_path: str):
        """Initialize FeedbackManager with database path.

        Args:
            db_path: Absolute or relative path to SQLite database file.
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create rag_interactions table if not exists."""
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
        """Log a RAG interaction before user provides feedback.

        Args:
            query: Original user query.
            answer: Generated answer text.
            contexts: Retrieved context chunks.
            metadata: Optional run metadata (e.g., run_type, model).

        Returns:
            Interaction ID for subsequent feedback submission.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO rag_interactions (timestamp, query, answer, retrieved_contexts, metadata) VALUES (?, ?, ?, ?, ?)",
                (datetime.now().isoformat(), query, answer, json.dumps(contexts), json.dumps(metadata or {}))
            )
            return cursor.lastrowid

    def submit_feedback(self, interaction_id: int, score: int, comments: str = ""):
        """Submit user feedback for a logged interaction.

        Args:
            interaction_id: ID returned from log_interaction().
            score: Binary rating (1 = correct, 0 = incorrect/hallucination).
            comments: Optional user comments explaining the rating.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE rag_interactions SET user_score = ?, user_comments = ? WHERE id = ?",
                (score, comments, interaction_id)
            )

    def get_low_performance_samples(self, threshold: int = 1) -> List[Dict]:
        """Retrieve interactions with poor feedback for regression analysis.

        Args:
            threshold: Maximum score considered "poor" (default 1 means score 0).

        Returns:
            List of interaction records with score below threshold.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM rag_interactions WHERE user_score < ? AND user_score IS NOT NULL", (threshold,))
            return [dict(row) for row in cursor.fetchall()]

    def get_feedback_stats(self) -> Dict[str, Any]:
        """Compute aggregate feedback statistics.

        Returns:
            Dict with total feedback count and average user score.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), AVG(user_score) FROM rag_interactions WHERE user_score IS NOT NULL")
            count, avg_score = cursor.fetchone()
            return {
                "total_feedback_entries": count,
                "average_user_score": avg_score if avg_score is not None else 0.0
            }

    def export_feedback_to_jsonl(self, output_file: str):
        """Export all interactions to JSONL for model fine-tuning.

        Args:
            output_file: Target path for JSONL export.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM rag_interactions")
            rows = cursor.fetchall()

        with open(output_file, 'w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(dict(row)) + '\n')