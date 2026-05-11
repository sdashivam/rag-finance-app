"""
Query preprocessing and decomposition for financial RAG pipeline.

Handles query normalization, financial term expansion, and LLM-driven
decomposition of complex multi-part questions into atomic sub-queries.
"""

import re
from typing import List


class QueryProcessor:
    """Normalizes and decomposes user queries for multi-stage retrieval.

    Responsibilities:
    - Clean and normalize query text
    - Expand financial abbreviations (FY25, YoY, EPS, etc.)
    - Use LLM to decompose complex queries into atomic sub-queries
    - Fallback to regex decomposition if LLM unavailable

    Attributes:
        llm: Optional LLM instance for decomposition (e.g., ChatOllama)
        similarity_threshold: Threshold for detecting failed decomposition

    Args:
        llm: Optional LLM instance (e.g., ChatOllama).
        config: Optional config dict with decomposition_similarity_threshold.
    """
    def __init__(self, llm=None, config: dict = None):
        """Initialize QueryProcessor with optional LLM and config."""
        self.llm = llm
        self.similarity_threshold = 0.75
        if config:
            self.similarity_threshold = config.get('decomposition_similarity_threshold', self.similarity_threshold)

    def preprocess(self, query: str) -> str:
        """Normalize query and expand financial abbreviations.

        Args:
            query: Raw user query string.

        Returns:
            Cleaned and expanded query string.
        """
        # Basic text cleaning
        query = query.strip()
        query = re.sub(r'\s+', ' ', query)

        # Financial term expansion - aligns informal queries with formal PDF language
        replacements = {
            r'\bFY(\d{4})\b': r'FY\1 year ended March 31 \1',
            r'\bFY(\d{2})\b': r'FY20\1 year ended March 31 20\1',
            r'\bYoY\b': 'Year-over-Year',
            r'\bQ([1-4])\b': r'Quarter \1',
            r'\bEPS\b': 'Earnings Per Share',
            r'\bCAGR\b': 'Compound Annual Growth Rate'
        }

        for pattern, replacement in replacements.items():
            query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)

        return query

    def decompose(self, query: str) -> List[str]:
        """Split complex query into atomic sub-queries using LLM.

        Args:
            query: Preprocessed query string.

        Returns:
            List of atomic sub-queries suitable for independent retrieval.
        """
        if not self.llm:
            return [query]
        
        # Combined prompt that extracts structure and generates sub-queries in one call
        decompose_prompt = (
            "You are a Financial Query Expert. Your task is to break down complex financial questions into simple, atomic sub-queries.\n\n"
            
            "STEP 1 - ANALYZE the query to identify:\n"
            "- Company/Entity name(s)\n"
            "- Financial metrics (revenue, assets, liabilities, etc.)\n"
            "- Time periods (fiscal years, quarters)\n"
            "- Any comparisons or calculations needed\n\n"
            
            "STEP 2 - DECOMPOSE into simple queries where EACH query:\n"
            "✓ Asks about ONLY ONE metric\n"
            "✓ References ONLY ONE time period\n"
            "✓ Includes the company name when mentioned\n"
            "✓ Is independent and retrievable\n\n"
            
            "Do not invent placeholder company names like XYZ Company, [Company Name], or [Bank Name]. "
            "If the original query does not name a company, omit the company name.\n\n"

            "STEP 3 - OUTPUT: Provide ONLY the final list of sub-queries, one per line.\n"
            "Format each as a natural question. Do NOT use bullet points or numbering.\n\n"
            
            "EXAMPLES:\n\n"
            "Example 1:\n"
            "Input: 'What is the revenue and net income for Fiscal Year 2024?'\n"
            "Output:\n"
            "What is the revenue for Fiscal Year 2024?\n"
            "What is the net income for Fiscal Year 2024?\n\n"
            
            "Example 2:\n"
            "Input: 'Compare total assets between Fiscal Year 2023 and Fiscal Year 2024 for ICICI Bank.'\n"
            "Output:\n"
            "What are the total assets for ICICI Bank in Fiscal Year 2023?\n"
            "What are the total assets for ICICI Bank in Fiscal Year 2024?\n\n"
            
            "Example 3:\n"
            "Input: 'What was the total capital and liabilities for FY24? What is the difference in revenue from FY24 to FY25 for ICICI Bank?'\n"
            "Output:\n"
            "What was the total capital and liabilities for Fiscal Year 2024?\n"
            "What is the revenue for ICICI Bank in Fiscal Year 2024?\n"
            "What is the revenue for ICICI Bank in Fiscal Year 2025?\n\n"
            
            "Now process this query - Output ONLY the sub-queries:\n"
            f"Input: '{query}'\n"
            "Output:\n"
        )
        
        try:
            # Robust invocation: Try .invoke() or direct call
            if hasattr(self.llm, "invoke"):
                response = self.llm.invoke(decompose_prompt)
            elif callable(self.llm):
                response = self.llm(decompose_prompt)
            else:
                raise AttributeError("The provided LLM object is neither callable nor has an 'invoke' method.")
            
            # Extract string content safely (handles LangChain AIMessage or raw strings)
            content = response.content if hasattr(response, "content") else str(response)
            
            # Parse the response - each line is a sub-query
            lines = content.strip().split('\n')
            sub_queries = []
            
            for line in lines:
                line = line.strip()
                # Skip empty lines and common artifacts
                if not line:
                    continue
                # Remove common formatting artifacts
                line = re.sub(r"^\s*[-*•]?\s*\d*[\).\s-]*", "", line).strip()
                line = line.strip()
                lower_line = line.lower()

                artifact_prefixes = (
                    "output:", "input:", "example", "here are", "note:",
                    "since ", "assuming ", "if ", "let me", "i'll ", "i will ",
                )
                placeholder_fragments = (
                    "[company", "[bank", "[specific", "[insert", "xyz company",
                    "company name", "bank name", "specific company", "specific sector",
                    "insert metric",
                )
                if lower_line.startswith(artifact_prefixes):
                    continue
                if any(fragment in lower_line for fragment in placeholder_fragments):
                    continue
                if not line.endswith("?"):
                    if lower_line.startswith(("what", "how", "which", "why", "when", "where")):
                        line += "?"
                    else:
                        continue

                # Ensure it's a meaningful query (at least some minimum length)
                if len(line) > 10:
                    sub_queries.append(line)
            
            # If we got meaningful sub-queries, return them
            if sub_queries and len(sub_queries) > 1:
                return sub_queries
            # If only one query returned, verify it's not just the original
            elif sub_queries and len(sub_queries) == 1:
                # If the single query is too similar to original, try to force decomposition
                if self._is_single_query_similar_to_original(query, sub_queries[0]):
                    return self._fallback_regex_decompose(query)
                return sub_queries
            else:
                # No valid queries parsed, use regex fallback
                return self._fallback_regex_decompose(query)
                
        except Exception as e:
            print(f"Error in decompose: {e}")
            return self._fallback_regex_decompose(query)
    
    def _is_single_query_similar_to_original(self, original: str, single_query: str) -> bool:
        """Check if decomposition failed (returned query too similar to original).

        Args:
            original: Original preprocessed query.
            single_query: Single sub-query from LLM.

        Returns:
            True if Jaccard similarity exceeds similarity_threshold.
        """
        original_words = set(original.lower().split())
        query_words = set(single_query.lower().split())

        if not original_words or not query_words:
            return False

        intersection = len(original_words & query_words)
        union = len(original_words | query_words)
        similarity = intersection / union if union > 0 else 0

        return similarity > self.similarity_threshold
    
    def _fallback_regex_decompose(self, query: str) -> List[str]:
        """Fallback decomposition using regex when LLM is unavailable.

        Args:
            query: Preprocessed query string.

        Returns:
            List of sub-queries (or single-item list if splitting fails).
        """
        sub_queries = []

        # Split by "?" and "and" while preserving context
        parts = re.split(r'\?\s+(?:and\s+)?(?:what\s+)', query, flags=re.IGNORECASE)
        
        if len(parts) > 1:
            # First part
            first_part = parts[0].strip()
            if not first_part.lower().startswith('what'):
                first_part = 'what ' + first_part
            if not first_part.endswith('?'):
                first_part += '?'
            sub_queries.append(first_part)
            
            # Remaining parts
            for part in parts[1:]:
                part = part.strip()
                if part:
                    if not part.lower().startswith('what'):
                        part = 'what ' + part
                    if not part.endswith('?'):
                        part += '?'
                    sub_queries.append(part)
        
        # If regex decomposition didn't work, return original as single query
        if not sub_queries:
            return [query]
        
        return sub_queries

    def run(self, raw_query: str) -> List[str]:
        """Execute full preprocessing and decomposition pipeline.

        Args:
            raw_query: Raw user query from UI or CLI.

        Returns:
            List of decomposed sub-queries for retrieval.
        """
        clean_query = self.preprocess(raw_query)
        return self.decompose(clean_query)
        clean_query = self.preprocess(raw_query)
        return self.decompose(clean_query)
