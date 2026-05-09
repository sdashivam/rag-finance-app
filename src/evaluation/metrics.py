import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Set, Any

class RAGEvaluator:
    """
    Evaluator for RAG (Retrieval-Augmented Generation) metrics.
    """

    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the evaluator with an embedding model for similarity calculations.

        Args:
            embedding_model_name: Name of the sentence transformer model to use.
        """
        self.embedding_model = SentenceTransformer(embedding_model_name)

    def precision_at_k(self, retrieved_docs: List[str], relevant_docs: Set[str], k: int) -> float:
        """
        Calculate Precision@k for retrieval evaluation.

        Precision@k = (Number of relevant documents in top k) / k

        Args:
            retrieved_docs: List of retrieved document IDs/texts in order
            relevant_docs: Set of relevant document IDs/texts
            k: Number of top documents to consider

        Returns:
            Precision@k score (0.0 to 1.0)
        """
        if k <= 0 or not retrieved_docs:
            return 0.0

        top_k = retrieved_docs[:k]
        relevant_in_top_k = len([doc for doc in top_k if doc in relevant_docs])
        return relevant_in_top_k / k

    def relevance_score(self, query: str, answer: str) -> float:
        """
        Calculate relevance score between query and generated answer.
        Uses cosine similarity of embeddings.

        Args:
            query: The original query string
            answer: The generated answer string

        Returns:
            Relevance score (0.0 to 1.0, higher is more relevant)
        """
        if not query or not answer:
            return 0.0

        query_embedding = self.embedding_model.encode([query])[0]
        answer_embedding = self.embedding_model.encode([answer])[0]

        # Cosine similarity
        similarity = np.dot(query_embedding, answer_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(answer_embedding)
        )
        return float(similarity)

    def groundedness_score(self, answer: str, contexts: List[str]) -> float:
        """
        Calculate groundedness score - how well the answer is supported by the retrieved contexts.
        Measures average similarity between answer and each context.

        Args:
            answer: The generated answer string
            contexts: List of retrieved context strings

        Returns:
            Groundedness score (0.0 to 1.0, higher is more grounded)
        """
        if not answer or not contexts:
            return 0.0

        answer_embedding = self.embedding_model.encode([answer])[0]
        context_embeddings = self.embedding_model.encode(contexts)

        similarities = []
        for ctx_emb in context_embeddings:
            sim = np.dot(answer_embedding, ctx_emb) / (
                np.linalg.norm(answer_embedding) * np.linalg.norm(ctx_emb)
            )
            similarities.append(sim)

        return float(np.mean(similarities))

    def faithfulness_score(self, answer: str, contexts: List[str]) -> float:
        """
        Calculate faithfulness score - how factually consistent the answer is with the contexts.
        For simplicity, uses the same calculation as groundedness (cosine similarity).
        In practice, this might require more sophisticated NLP models like entailment checkers.

        Args:
            answer: The generated answer string
            contexts: List of retrieved context strings

        Returns:
            Faithfulness score (0.0 to 1.0, higher is more faithful)
        """
        # For this implementation, faithfulness is similar to groundedness
        # In advanced setups, use models like BERT for entailment
        return self.groundedness_score(answer, contexts)

    def evaluate_retrieval(self, retrieved_results: List[Dict[str, Any]], relevant_docs: Set[str], k: int) -> Dict[str, float]:
        """
        Evaluate retrieval performance.

        Args:
            retrieved_results: List of retrieved results with 'text' or 'id' field
            relevant_docs: Set of relevant document texts/IDs
            k: Top k to evaluate

        Returns:
            Dict with 'precision_at_k' score
        """
        # Extract document identifiers (using text as ID for simplicity)
        retrieved_docs = [result.get('text', str(result)) for result in retrieved_results]

        precision = self.precision_at_k(retrieved_docs, relevant_docs, k)

        return {
            'precision_at_k': precision
        }

    def evaluate_generation(self, query: str, answer: str, contexts: List[str]) -> Dict[str, float]:
        """
        Evaluate generation quality.

        Args:
            query: The original query
            answer: The generated answer
            contexts: List of retrieved contexts

        Returns:
            Dict with relevance, groundedness, and faithfulness scores
        """
        relevance = self.relevance_score(query, answer)
        groundedness = self.groundedness_score(answer, contexts)
        faithfulness = self.faithfulness_score(answer, contexts)

        return {
            'relevance': relevance,
            'groundedness': groundedness,
            'faithfulness': faithfulness
        }