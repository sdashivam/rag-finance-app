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
        model_name: str,
        top_k: int = 3,
    ):
        """
        Initializes the FAISS retriever.

        Args:
            index_path (str): Path to the saved .faiss index file.
            metadata_path (str): Path to the pickled metadata file.
            model_name (str): SentenceTransformer model name for embeddings.
            top_k (int): Default number of documents to retrieve.
        """
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.metadata = []
        self.top_k = top_k
        self._load_index()

    def _load_index(self):
        """
        Loads the FAISS index from disk and unpickles the associated metadata.

        Raises:
            FileNotFoundError: If index or metadata files are missing.
        """
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index file not found at {self.index_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {self.metadata_path}")

        self.index = faiss.read_index(str(self.index_path))
        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

    def embed(self, text: str):
        """
        Generates a vector embedding for the given text.

        Args:
            text (str): The input string to embed.

        Returns:
            numpy.ndarray: The float32 embedding vector.
        """
        return self.model.encode([text], convert_to_numpy=True).astype("float32")

    def _matches_filters(self, metadata: Dict, filters: Dict) -> bool:
        """Checks if the metadata dictionary matches the provided filters."""
        for key, value in filters.items():
            if metadata.get(key) != value:
                return False
        return True

    def retrieve(self, query: str, filters: Optional[Dict] = None, top_k: Optional[int] = None) -> List[Dict]:
        """
        Retrieves the most similar text chunks for a single query using FAISS.
        Supports metadata filtering by scanning a larger candidate pool.

        Args:
            query (str): The query string.
            filters (Optional[Dict]): Metadata filters (e.g., {"page": 5}).
            top_k (Optional[int]): Number of results to return. Defaults to self.top_k.

        Returns:
            List[Dict]: A list of result dictionaries containing text, score, and metadata.
        """
        if self.index is None:
            self._load_index()

        assert self.index is not None, "FAISS index must be loaded before searching"
        top_k = top_k or self.top_k
        
        # If filters are applied, search for more candidates to ensure we find enough matches
        search_k = top_k * 10 if filters else top_k
        vector = self.embed(query)
        distances, indices = self.index.search(vector.reshape(1, -1), search_k)

        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            entry = self.metadata[idx]
            
            # Apply metadata-aware filtering
            if filters and not self._matches_filters(entry.get("metadata", {}), filters):
                continue

            results.append(
                {
                    "query": query,
                    "score": float(score),
                    "text": entry.get("text", ""),
                    "metadata": entry.get("metadata", {}),
                    "source_type": "faiss",
                }
            )
            
            if len(results) >= top_k:
                break
        return results

    def retrieve_for_queries(self, queries: List[str], filters: Optional[Dict] = None, top_k: Optional[int] = None) -> Dict[str, List[Dict]]:
        """
        Batch retrieves results for multiple sub-queries.

        Args:
            queries (List[str]): List of query strings.
            filters (Optional[Dict]): Metadata filters to apply to all queries.
            top_k (Optional[int]): Number of results per query.

        Returns:
            Dict[str, List[Dict]]: Mapping of query string to its list of results.
        """
        return {query: self.retrieve(query, filters, top_k) for query in queries}


class BM25Retriever:
    """Keyword-based retriever using the BM25 algorithm on document chunks."""

    def __init__(self, corpus_metadata: List[Dict], top_k: int = 3):
        """
        Initializes the BM25 retriever.

        Args:
            corpus_metadata (List[Dict]): List of chunks with 'text' and 'metadata'.
            top_k (int): Number of results to retrieve.
        """
        self.corpus_metadata = corpus_metadata
        self.top_k = top_k
        self.bm25 = None

        try:
            from rank_bm25 import BM25Okapi
            tokenized_corpus = [self._tokenize(doc.get("text", "")) for doc in corpus_metadata]
            if tokenized_corpus:
                self.bm25 = BM25Okapi(tokenized_corpus)
        except ImportError:
            print("Warning: rank_bm25 not installed. BM25 retrieval will be disabled.")

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace and punctuation tokenizer."""
        return re.findall(r"\w+", text.lower())

    def _matches_filters(self, metadata: Dict, filters: Dict) -> bool:
        """Checks if the metadata dictionary matches the provided filters."""
        for key, value in filters.items():
            if metadata.get(key) != value:
                return False
        return True

    def retrieve(self, query: str, filters: Optional[Dict] = None, top_k: Optional[int] = None) -> List[Dict]:
        """
        Retrieves the most relevant chunks using BM25 ranking.

        Args:
            query (str): User query.
            filters (Optional[Dict]): Metadata filters.
            top_k (Optional[int]): Results count.

        Returns:
            List[Dict]: Ranked results.
        """
        if not self.bm25:
            return []

        top_k = top_k or self.top_k
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in ranked_indices:
            if score <= 0:
                continue
            entry = self.corpus_metadata[idx]
            if filters and not self._matches_filters(entry.get("metadata", {}), filters):
                continue

            results.append({
                "query": query,
                "score": float(score),
                "text": entry.get("text", ""),
                "metadata": entry.get("metadata", {}),
                "source_type": "bm25",
            })
            if len(results) >= top_k:
                break
        return results


class SQLiteRetriever:
    """Retrieves candidate table rows from SQLite based on query keywords."""

    def __init__(self, db_path: str, top_k: int = 3):
        """
        Initializes the SQLite retriever.

        Args:
            db_path (str): Path to the SQLite database.
            top_k (int): Number of table rows to retrieve.
        """
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"SQLite database not found at {self.db_path}")
        self.top_k = top_k

    def _query_keywords(self, query: str) -> List[str]:
        """
        Extracts significant keywords from a query for SQL LIKE matching.

        Args:
            query (str): The user query.

        Returns:
            List[str]: A list of lowercase keywords, filtered for stop words.
        """
        tokens = re.findall(r"\b[a-zA-Z]{3,}\b", query.lower())
        stop_words = {
            "what", "why", "how", "is", "the", "for", "and", "from",
            "to", "in", "of", "a", "an", "on", "with", "by", "year",
            "fy", "q", "icici",
        }
        keywords = [tok for tok in tokens if tok not in stop_words]
        return keywords[:6]

    def retrieve(self, query: str, filters: Optional[Dict] = None, top_k: Optional[int] = None) -> List[Dict]:
        """
        Retrieves table rows from SQLite using a keyword-based OR search.

        Args:
            query (str): The query string.
            filters (Optional[Dict]): Metadata filters mapped to SQL columns.
            top_k (Optional[int]): Max number of rows to return.

        Returns:
            List[Dict]: Formatted table rows as text with metadata.
        """
        top_k = top_k or self.top_k
        keywords = self._query_keywords(query)
        if not keywords:
            return []

        clauses = ["lower(table_data) LIKE ?" for _ in keywords]
        params: List[object] = [f"%{keyword}%" for keyword in keywords]
        
        sql = f"SELECT file_source, page_number, table_data FROM financial_tables WHERE ({' OR '.join(clauses)})"
        
        # Metadata-aware pre-filtering via SQL WHERE clause
        if filters:
            for key, val in filters.items():
                # Map common metadata keys to table column names
                col = "page_number" if key == "page" else "file_source" if key == "source" else key
                sql += f" AND {col} = ?"
                params.append(val)
        
        sql += " LIMIT ?"
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
        bm25_retriever: Optional[BM25Retriever] = None,
        sqlite_retriever: Optional[SQLiteRetriever] = None,
        top_k: int = 3,
    ):
        """
        Initializes the HybridRetriever.

        Args:
            faiss_retriever (FAISSRetriever): Instance for semantic search.
            bm25_retriever (Optional[BM25Retriever]): Instance for keyword-based ranking.
            sqlite_retriever (Optional[SQLiteRetriever]): Instance for structured table search.
            top_k (int): Total results to return after merging.
        """
        self.faiss_retriever = faiss_retriever
        self.bm25_retriever = bm25_retriever
        self.sqlite_retriever = sqlite_retriever
        self.top_k = top_k

    def _merge(self, faiss_results: List[Dict], bm25_results: List[Dict], sqlite_results: List[Dict]) -> List[Dict]:
        """
        Deduplicates and sorts results from different sources.

        Args:
            faiss_results (List[Dict]): Results from semantic search.
            bm25_results (List[Dict]): Results from BM25 ranking.
            sqlite_results (List[Dict]): Results from SQL keyword search.

        Returns:
            List[Dict]: Combined list sorted primarily by source then score.
        """
        combined = []
        seen_texts = set()

        for item in faiss_results + bm25_results + sqlite_results:
            text_key = item.get("text", "").strip()
            if not text_key or text_key in seen_texts:
                continue
            seen_texts.add(text_key)
            combined.append(item)

        # Prioritize by source type then score. 
        # FAISS (0) -> BM25 (1) -> SQLite (2).
        # For FAISS (L2), lower is better. For BM25, higher is better.
        combined.sort(key=lambda x: (
            0 if x.get("source_type") == "faiss" else 
            1 if x.get("source_type") == "bm25" else 2,
            x.get("score", 0.0) if x.get("source_type") == "faiss" else -x.get("score", 0.0)
        ))
        return combined[: self.top_k]

    def retrieve(self, query: str, filters: Optional[Dict] = None, top_k: Optional[int] = None) -> List[Dict]:
        """
        Performs hybrid retrieval by calling both FAISS and SQLite.

        Args:
            query (str): The query string.
            filters (Optional[Dict]): Metadata filters for both retrieval engines.
            top_k (Optional[int]): Number of results per source.

        Returns:
            List[Dict]: Merged and deduplicated results.
        """
        top_k = top_k or self.top_k
        faiss_results = self.faiss_retriever.retrieve(query, filters, top_k)
        bm25_results = []
        if self.bm25_retriever:
            bm25_results = self.bm25_retriever.retrieve(query, filters, top_k)
            
        sqlite_results = []
        if self.sqlite_retriever:
            sqlite_results = self.sqlite_retriever.retrieve(query, filters, top_k)

        merged = self._merge(faiss_results, bm25_results, sqlite_results)
        return merged

    def retrieve_for_queries(self, queries: List[str], filters: Optional[Dict] = None, top_k: Optional[int] = None) -> Dict[str, List[Dict]]:
        """
        Performs hybrid retrieval for a batch of queries.

        Args:
            queries (List[str]): List of query strings.
            filters (Optional[Dict]): Metadata filters to apply.
            top_k (Optional[int]): Number of results per source per query.

        Returns:
            Dict[str, List[Dict]]: Mapping of query to merged results.
        """
        return {query: self.retrieve(query, filters, top_k) for query in queries}


class AnswerAggregator:
    """Aggregates retrieved content and produces a structured answer."""

    def __init__(self, llm=None):
        """
        Initializes the AnswerAggregator.

        Args:
            llm (Optional[Any]): The language model instance (e.g., ChatOllama).
        """
        self.llm = llm

    def _build_prompt(self, query: str, retrieval_results: List[Dict]) -> str:
        """
        Constructs the instruction and context prompt for the LLM.

        Args:
            query (str): The original or sub-query.
            retrieval_results (List[Dict]): List of retrieved context chunks.

        Returns:
            str: The fully formatted prompt.
        """
        prompt = [
            "You are an expert financial assistant.",
            "Your task is to answer the query based ONLY on the provided evidence.",
            "For every statement or claim you make, you MUST cite the source index in brackets, e.g., [1].",
            "If multiple sources support a claim, list them all, e.g., [1, 3].",
            "If the evidence is insufficient, say so clearly.",
            "Format the output as a concise answer followed by a 'References' list mapping indices to source metadata.",
            "",
            f"Query: {query}",
            "",
            "Retrieved evidence:"
        ]

        for i, item in enumerate(retrieval_results, 1):
            source_meta = item.get("metadata", {})
            prompt.append(
                f"[{i}] Source: {item.get('source_type', 'unknown')} (Page: {source_meta.get('page')}, Type: {source_meta.get('type')})"
            )
            prompt.append(f"Content: {item.get('text','').strip()}")
            prompt.append("")

        prompt.append("")
        prompt.append("Answer:")
        return "\n".join(prompt)

    def aggregate_answer(self, query: str, retrieval_results: List[Dict]) -> str:
        """
        Generates a final answer for a single query based on retrieval context.

        Args:
            query (str): The query to answer.
            retrieval_results (List[Dict]): The evidence chunks.

        Returns:
            str: The LLM-generated answer or a formatted summary if LLM fails.
        """
        if not retrieval_results:
            return "No relevant evidence was retrieved to answer this query."

        if not self.llm:
            lines = [f"Query: {query}"]
            for i, item in enumerate(retrieval_results, 1):
                lines.append(f"[{i}] {item.get('source_type')} (Page {item.get('metadata', {}).get('page')}): {item.get('text', '')[:150]}...")
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
        """
        Processes a batch of queries and their respective retrieval results into answers.

        Args:
            queries (List[str]): List of sub-queries.
            retrieval_map (Dict[str, List[Dict]]): Map of query to retrieval list.

        Returns:
            Dict[str, str]: Map of query to final answer string.
        """
        return {
            query: self.aggregate_answer(query, retrieval_map.get(query, []))
            for query in queries
        }
