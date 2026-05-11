"""
Multi-source retrieval and answer aggregation for RAG pipeline.

Implements semantic (FAISS), lexical (BM25), and structured (SQLite) retrieval
with hybrid merging and LLM-based answer synthesis.
"""

import json
import pickle
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import torch
from sentence_transformers import SentenceTransformer


class FAISSRetriever:
    """Loads a FAISS index and metadata to retrieve relevant chunks for a query.

    Responsibilities:
    - Load persisted FAISS index and chunk metadata
    - Generate query embeddings with GPU acceleration
    - Execute similarity search with metadata filtering
    - Return ranked results with source attribution
    """

    def __init__(
        self,
        index_path: str,
        metadata_path: str,
        model_name: str,
        top_k: int = None,
        config: dict = None,
    ):
        """Initialize FAISS retriever with optional config.

        Args:
            index_path: Path to saved .faiss index file.
            metadata_path: Path to pickled metadata file.
            model_name: SentenceTransformer model name for embeddings.
            top_k: Default number of documents to retrieve.
            config: Optional config dict with device, search_k_multiplier settings.
        """
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)

        # Device configuration
        device = "cpu"
        if config:
            device_setting = config.get('device', 'auto')
            if device_setting == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            elif device_setting in ("cuda", "cpu"):
                device = device_setting

        self.model = SentenceTransformer(model_name, device=device)
        self.index = None
        self.metadata = []

        # Retrieval settings from config or defaults
        self.top_k = top_k
        self.search_multiplier = config.get('search_k_multiplier', 10) if config else 10
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
        """
        Checks if the metadata dictionary matches the provided filters.
        Supports pdf_page fallback: if filtering by "page" but metadata has "pdf_page",
        also check pdf_page as fallback for visible page number matching.
        """
        for key, value in filters.items():
            # Direct match
            if metadata.get(key) == value:
                continue
            # Fallback: if filtering by "page", also check "pdf_page"
            if key == "page" and metadata.get("pdf_page") == value:
                continue
            return False
        return True

    def retrieve(self, query: str, filters: Optional[Dict] = None, top_k: Optional[int] = None) -> List[Dict]:
        """Retrieve most similar text chunks for a query using FAISS.

        Args:
            query: The query string.
            filters: Metadata filters (e.g., {"page": 5}).
            top_k: Number of results to return. Defaults to self.top_k.

        Returns:
            List of result dicts containing text, score, and metadata.
        """
        if self.index is None:
            self._load_index()

        assert self.index is not None, "FAISS index must be loaded before searching"
        top_k = top_k or self.top_k

        # Use config multiplier or default 10 for filtered searches
        search_multiplier = getattr(self, 'search_multiplier', 10)
        search_k = top_k * search_multiplier if filters else top_k
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

    def __init__(self, corpus_metadata: List[Dict], top_k: int = None, config: dict = None):
        """Initialize BM25 retriever with optional config.

        Args:
            corpus_metadata: List of chunks with 'text' and 'metadata'.
            top_k: Number of results to retrieve.
            config: Optional config dict.
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
        """
        Checks if the metadata dictionary matches the provided filters.
        Supports pdf_page fallback: if filtering by "page" but metadata has "pdf_page",
        also check pdf_page as fallback for visible page number matching.
        """
        for key, value in filters.items():
            # Direct match
            if metadata.get(key) == value:
                continue
            # Fallback: if filtering by "page", also check "pdf_page"
            if key == "page" and metadata.get("pdf_page") == value:
                continue
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

    def __init__(self, db_path: str, top_k: int = None, config: dict = None):
        """Initialize SQLite retriever with optional config.

        Args:
            db_path: Path to the SQLite database.
            top_k: Number of table rows to retrieve.
            config: Optional config dict with sql_stop_words and sql_keywords_limit.
        """
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"SQLite database not found at {self.db_path}")
        self.top_k = top_k

        # Load stop words and keywords limit from config
        default_stop_words = {"what", "why", "how", "is", "the", "for", "and", "from",
                              "to", "in", "of", "a", "an", "on", "with", "by", "year",
                              "fy", "q", "icici"}
        self.stop_words = config.get('sql_stop_words', default_stop_words) if config else default_stop_words
        self.keywords_limit = config.get('sql_keywords_limit', 6) if config else 6

    def _query_keywords(self, query: str) -> List[str]:
        """Extract significant keywords from query for SQL LIKE matching.

        Args:
            query: The user query.

        Returns:
            List of lowercase keywords filtered for stop words, capped by keywords_limit.
        """
        tokens = re.findall(r"\b[a-zA-Z]{3,}\b", query.lower())
        keywords = [tok for tok in tokens if tok not in self.stop_words]
        return keywords[:self.keywords_limit]

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
                # Handle "page" filter with pdf_page fallback
                if key == "page":
                    # Try page_number column (internal), fallback to pdf_page if needed
                    # Note: SQLite stores internal page number; pdf_page would need separate column
                    col = "page_number"
                elif key == "pdf_page":
                    col = "page_number"  # Map pdf_page to page_number for now
                else:
                    col = "file_source" if key == "source" else key
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
        top_k: int = None,
        config: dict = None,
    ):
        """Initialize HybridRetriever with optional config.

        Args:
            faiss_retriever: Instance for semantic search.
            bm25_retriever: Instance for keyword-based ranking.
            sqlite_retriever: Instance for structured table search.
            top_k: Total results to return after merging.
            config: Optional config dict passed to retrievers.
        """
        self.faiss_retriever = faiss_retriever
        self.bm25_retriever = bm25_retriever
        self.sqlite_retriever = sqlite_retriever
        self.top_k = top_k

    def _merge(self, faiss_results: List[Dict], bm25_results: List[Dict], sqlite_results: List[Dict]) -> List[Dict]:
        """Deduplicate and rank results from multiple retrievers.

        Args:
            faiss_results: Semantic search results (inner product, higher=better).
            bm25_results: Keyword search results (TF-IDF score, higher=better).
            sqlite_results: Structured table results (unscored).

        Returns:
            Combined list sorted by source priority (FAISS > BM25 > SQLite) then score.
        """
        combined = []
        seen_texts = set()

        # Deduplicate by exact text match
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
            # Use pdf_page if available (visible page number), fallback to internal page
            pdf_page = source_meta.get("pdf_page")
            display_page = pdf_page if pdf_page is not None else source_meta.get("page")
            prompt.append(
                f"[{i}] Source: {item.get('source_type', 'unknown')} (Page: {display_page}, Type: {source_meta.get('type')})"
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
                meta = item.get("metadata", {})
                pdf_page = meta.get("pdf_page")
                display_page = pdf_page if pdf_page is not None else meta.get("page")
                lines.append(f"[{i}] {item.get('source_type')} (Page {display_page}): {item.get('text', '')[:150]}...")
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
                meta = item.get("metadata", {})
                pdf_page = meta.get("pdf_page")
                display_page = pdf_page if pdf_page is not None else meta.get("page")
                lines.append(f"- {item.get('source_type')} source page {display_page}: {item.get('text', '')[:200]}...")
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