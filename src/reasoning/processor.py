import re
from typing import List

class QueryProcessor:
    """
    Handles query cleaning, normalization, and decomposition into sub-queries.
    This is the first stage of the Query Pipeline.
    """
    def __init__(self, llm=None):
        """
        Initialize with an optional LLM instance. 
        Decomposition requires an LLM; if None, it returns the preprocessed query as a single item.
        """
        self.llm = llm

    def preprocess(self, query: str) -> str:
        """
        Normalizes the user query and expands common financial abbreviations 
        to improve semantic search matching against report text.
        """
        # 1. Basic text cleaning
        query = query.strip()
        query = re.sub(r'\s+', ' ', query)
        
        # 2. Financial Term Expansion (e.g., FY25 -> Fiscal Year 2025)
        # This aligns the query language with the formal language usually found in PDFs
        replacements = {
            r'\bFY(\d{2})\b': r'Fiscal Year 20\1',
            r'\bFY(\d{4})\b': r'Fiscal Year \1',
            r'\bYoY\b': 'Year-over-Year',
            r'\bQ([1-4])\b': r'Quarter \1',
            r'\bEPS\b': 'Earnings Per Share',
            r'\bCAGR\b': 'Compound Annual Growth Rate'
        }
        
        for pattern, replacement in replacements.items():
            query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
            
        return query

    def decompose(self, query: str) -> List[str]:
        """
        Uses a direct two-level LLM approach to split complex financial queries into standalone sub-queries.
        Level 1: Identify query components (metrics, time periods, companies, comparisons)
        Level 2: Generate atomic sub-queries for each component combination
        """
        if not self.llm:
            # Fallback if no LLM is integrated yet
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
                line = line.lstrip('- •*123456789. ')
                line = line.strip()
                
                # Ensure it's a meaningful query (at least some minimum length)
                if len(line) > 10 and not line.lower().startswith(('output:', 'input:', 'example')):
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
        """Check if the returned query is too similar to the original (not properly decomposed)."""
        # Simple heuristic: if they share more than 80% of words, they're too similar
        original_words = set(original.lower().split())
        query_words = set(single_query.lower().split())
        
        if not original_words or not query_words:
            return False
            
        intersection = len(original_words & query_words)
        union = len(original_words | query_words)
        similarity = intersection / union if union > 0 else 0
        
        return similarity > 0.75
    
    def _fallback_regex_decompose(self, query: str) -> List[str]:
        """Fallback decomposition using regex patterns for common multi-part queries."""
        sub_queries = []
        
        # Pattern 1: Handle "and" separated clauses that look like separate questions
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
        """Executes the full preprocessing and decomposition pipeline."""
        clean_query = self.preprocess(raw_query)
        return self.decompose(clean_query)