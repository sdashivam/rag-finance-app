import json
import pickle
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

import faiss
from sentence_transformers import SentenceTransformer


class FAISSRetriever:
    """Loads a FAISS index and metadata to retrieve relevant chunks for a query."""

    def __init__(
        self,
        index_path: str,
        metadata_path: str,
        model_name: str = "all-MiniLM-L6-v2",
        top_k: int = 3,
    ):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.metadata = []
        self.top_k = top_k
        self._load_index()

    def _load_index(self):
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index file not found at {self.index_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {self.metadata_path}")

        self.index = faiss.read_index(str(self.index_path))
        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

    def embed(self, text: str):
        return self.model.encode([text], convert_to_numpy=True).astype("float32")

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        if self.index is None:
            self._load_index()

        assert self.index is not None, "FAISS index must be loaded before searching"
        top_k = top_k or self.top_k
        vector = self.embed(query)
        distances, indices = self.index.search(vector.reshape(1, -1), top_k)

        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            entry = self.metadata[idx]
            results.append(
                {
                    "query": query,
                    "score": float(score),
                    "text": entry.get("text", ""),
                    "metadata": entry.get("metadata", {}),
                    "source_type": "faiss",
                }
            )
        return results

    def retrieve_for_queries(self, queries: List[str], top_k: Optional[int] = None) -> Dict[str, List[Dict]]:
        return {query: self.retrieve(query, top_k) for query in queries}


class SQLiteRetriever:
    """Retrieves candidate table rows from SQLite based on query keywords."""

    def __init__(self, db_path: str, top_k: int = 3):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"SQLite database not found at {self.db_path}")
        self.top_k = top_k

    def _query_keywords(self, query: str) -> List[str]:
        tokens = re.findall(r"\b[a-zA-Z]{3,}\b", query.lower())
        stop_words = {
            "what", "why", "how", "is", "the", "for", "and", "from",
            "to", "in", "of", "a", "an", "on", "with", "by", "year",
            "fy", "q", "icici",
        }
        keywords = [tok for tok in tokens if tok not in stop_words]
        return keywords[:6]

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        top_k = top_k or self.top_k
        keywords = self._query_keywords(query)
        if not keywords:
            return []

        clauses = ["lower(table_data) LIKE ?" for _ in keywords]
        params: List[object] = [f"%{keyword}%" for keyword in keywords]
        sql = f"SELECT file_source, page_number, table_data FROM financial_tables WHERE {' OR '.join(clauses)} LIMIT ?"
        params.append(top_k)

        results = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(sql, params)
            for row in cursor.fetchall():
                try:
                    table_data = json.loads(row["table_data"])
                    pretty_text = json.dumps(table_data, indent=2)
                except Exception:
                    pretty_text = row["table_data"]

                results.append(
                    {
                        "query": query,
                        "score": 0.0,
                        "text": pretty_text,
                        "metadata": {
                            "source": row["file_source"],
                            "page": row["page_number"],
                            "type": "table",
                        },
                        "source_type": "sqlite",
                    }
                )

        return results


class HybridRetriever:
    """Combines FAISS and SQLite retrieval results into a unified ranked result set."""

    def __init__(
        self,
        faiss_retriever: FAISSRetriever,
        sqlite_retriever: Optional[SQLiteRetriever] = None,
        top_k: int = 3,
    ):
        self.faiss_retriever = faiss_retriever
        self.sqlite_retriever = sqlite_retriever
        self.top_k = top_k

    def _merge(self, faiss_results: List[Dict], sqlite_results: List[Dict]) -> List[Dict]:
        combined = []
        seen_texts = set()

        for item in faiss_results + sqlite_results:
            text_key = item.get("text", "").strip()
            if not text_key or text_key in seen_texts:
                continue
            seen_texts.add(text_key)
            combined.append(item)

        # If SQLite results don't have score, keep FAISS scores first but preserve SQLite results too.
        combined.sort(key=lambda x: (0 if x.get("source_type") == "faiss" else 1, x.get("score", 0.0)))
        return combined[: self.top_k]

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        top_k = top_k or self.top_k
        faiss_results = self.faiss_retriever.retrieve(query, top_k)
        sqlite_results = []
        if self.sqlite_retriever:
            sqlite_results = self.sqlite_retriever.retrieve(query, top_k)

        merged = self._merge(faiss_results, sqlite_results)
        return merged

    def retrieve_for_queries(self, queries: List[str], top_k: Optional[int] = None) -> Dict[str, List[Dict]]:
        return {query: self.retrieve(query, top_k) for query in queries}


class AnswerAggregator:
    """Aggregates retrieved content and produces a structured answer."""

    def __init__(self, llm=None):
        self.llm = llm

    def _build_prompt(self, query: str, retrieval_results: List[Dict]) -> str:
        prompt = [
            "You are an expert financial assistant.",
            "Use only the retrieved evidence to answer the query accurately.",
            "If the evidence is insufficient, say so clearly.",
            "Provide a concise structured answer and list the sources.",
            "",
            f"Query: {query}",
            "",
            "Retrieved evidence:"
        ]

        for item in retrieval_results:
            source_meta = item.get("metadata", {})
            prompt.append(
                f"- Source: {item.get('source_type', 'unknown')} | page: {source_meta.get('page')} | section: {source_meta.get('type')} | text: {item.get('text','').strip()[:400]}"
            )

        prompt.append("")
        prompt.append("Answer:")
        return "\n".join(prompt)

    def aggregate_answer(self, query: str, retrieval_results: List[Dict]) -> str:
        if not retrieval_results:
            return "No relevant evidence was retrieved to answer this query."

        if not self.llm:
            lines = [f"Query: {query}"]
            for item in retrieval_results:
                lines.append(f"- {item.get('source_type')} source page {item.get('metadata', {}).get('page')}: {item.get('text', '')[:200]}...")
            return "\n".join(lines)

        try:
            prompt = self._build_prompt(query, retrieval_results)
            
            # Robust invocation: Try .invoke() (standard) or direct call (fallback)
            if hasattr(self.llm, "invoke"):
                response = self.llm.invoke(prompt)
            elif callable(self.llm):
                response = self.llm(prompt)
            else:
                raise AttributeError("The provided LLM object is neither callable nor has an 'invoke' method.")
            
            # Handle different return types (e.g., LangChain AIMessage vs raw string)
            if hasattr(response, "content"):
                res_text = str(response.content).strip()
            else:
                res_text = str(response).strip()

            if not res_text:
                raise ValueError("LLM returned an empty response.")
            
            return res_text
        except Exception as e:
            print(f"Warning: LLM invocation failed: {e}. Falling back to summary mode.")
            # Fallback to summary mode when LLM fails
            lines = [f"Query: {query}"]
            lines.append("LLM unavailable - showing retrieved evidence:")
            for item in retrieval_results:
                lines.append(f"- {item.get('source_type')} source page {item.get('metadata', {}).get('page')}: {item.get('text', '')[:200]}...")
            return "\n".join(lines)

    def aggregate_all(self, queries: List[str], retrieval_map: Dict[str, List[Dict]]) -> Dict[str, str]:
        return {
            query: self.aggregate_answer(query, retrieval_map.get(query, []))
            for query in queries
        }
