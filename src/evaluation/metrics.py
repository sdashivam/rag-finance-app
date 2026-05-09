import time
import torch
import logging
from typing import Dict, Any, List, Optional, Set

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevance,
        context_precision,
        context_recall,
    )
    from datasets import Dataset
    HAS_RAGAS = True
except ImportError:
    HAS_RAGAS = False

logger = logging.getLogger(__name__)

class RAGMetrics:
    """
    Handles performance monitoring for the RAG system, including:
    - Embedding generation latency
    - End-to-end response time
    - Token generation speed
    - GPU utilization %
    - VRAM usage
    - Faithfulness
    - Answer Relevancy
    - Context Precision
    - Context Recall
    """
    def __init__(self):
        pass

    def get_hardware_metrics(self) -> Dict[str, float]:
        """Returns GPU utilization % and VRAM usage in MB."""
        metrics = {"gpu_utilization_pct": 0.0, "vram_usage_mb": 0.0}
        try:
            if torch.cuda.is_available():
                # utilization and device_memory_used abstract away the differences 
                # between pynvml (NVIDIA) and amdsmi (AMD/ROCm)
                metrics["gpu_utilization_pct"] = float(torch.cuda.utilization(0))
                metrics["vram_usage_mb"] = torch.cuda.device_memory_used(0) / (1024**2)
        except Exception:
            pass
        return metrics

    def measure_duration(self, start_time: float) -> float:
        """Measures latency in seconds from start_time to now."""
        return time.perf_counter() - start_time

    def calculate_token_speed(self, response_text: str, duration: float) -> float:
        """Calculates token generation speed (tokens/sec)."""
        if duration <= 0:
            return 0.0
        # Rough estimation of tokens using word count
        token_estimate = len(response_text.split())
        return token_estimate / duration

    def get_quality_scores(
        self, 
        query: str, 
        answer: str, 
        contexts: List[str], 
        llm: Any, 
        embeddings: Optional[Any] = None,
        ground_truth: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Calculates RAG quality metrics using RAGAS.
        """
        default_scores = {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0
        }

        if not HAS_RAGAS:
            return default_scores

        try:
            data = {
                "question": [query],
                "answer": [answer],
                "contexts": [contexts],
                "ground_truth": [ground_truth if ground_truth else ""]
            }
            dataset = Dataset.from_dict(data)

            results = evaluate(
                dataset=dataset,
                metrics=[faithfulness, answer_relevance, context_precision, context_recall],
                llm=llm,
                embeddings=embeddings
            )

            return {
                "faithfulness": float(results.get("faithfulness", 0.0)),
                "answer_relevancy": float(results.get("answer_relevance", 0.0)),
                "context_precision": float(results.get("context_precision", 0.0)),
                "context_recall": float(results.get("context_recall", 0.0)),
            }
        except Exception as e:
            logger.warning(f"Error calculating RAGAS quality scores: {e}")
            return default_scores

    def evaluate_retrieval(self, results: List[Dict[str, Any]], relevant_docs: Set[str], top_k: int) -> Dict[str, float]:
        """
        Evaluates retrieval quality using Precision@K metric.
        
        Args:
            results: List of retrieved results with metadata
            relevant_docs: Set of relevant document IDs
            top_k: Number of top results to consider
            
        Returns:
            Dictionary with precision_at_k metric
        """
        try:
            if not results or not relevant_docs:
                return {"precision_at_k": 0.0}
            
            # Extract document IDs from results
            retrieved_ids = set()
            for i, result in enumerate(results[:top_k]):
                doc_id = result.get("source", result.get("id", str(i)))
                retrieved_ids.add(doc_id)
            
            # Calculate precision@k: (relevant AND retrieved) / k
            relevant_retrieved = len(retrieved_ids.intersection(relevant_docs))
            precision_at_k = relevant_retrieved / min(top_k, len(results)) if results else 0.0
            
            return {"precision_at_k": float(precision_at_k)}
        except Exception as e:
            logger.warning(f"Error evaluating retrieval: {e}")
            return {"precision_at_k": 0.0}

    def evaluate_generation(self, query: str, answer: str, contexts: List[str]) -> Dict[str, float]:
        """
        Evaluates generation quality using multiple metrics.
        
        Args:
            query: The user query
            answer: The generated answer
            contexts: List of context documents used to generate the answer
            
        Returns:
            Dictionary with relevance, groundedness, and faithfulness scores
        """
        try:
            relevance_score = self._calculate_relevance(answer, query)
            groundedness_score = self._calculate_groundedness(answer, contexts)
            faithfulness_score = self._calculate_faithfulness(answer, contexts)
            
            return {
                "relevance": float(relevance_score),
                "groundedness": float(groundedness_score),
                "faithfulness": float(faithfulness_score)
            }
        except Exception as e:
            logger.warning(f"Error evaluating generation: {e}")
            return {
                "relevance": 0.0,
                "groundedness": 0.0,
                "faithfulness": 0.0
            }

    def _calculate_relevance(self, answer: str, query: str) -> float:
        """
        Simple relevance scoring: checks if answer contains key terms from query.
        In production, consider using semantic similarity or LLM-based scoring.
        """
        if not answer or not query:
            return 0.0
        
        query_terms = set(query.lower().split())
        answer_lower = answer.lower()
        
        # Count how many query terms appear in the answer
        matching_terms = sum(1 for term in query_terms if term in answer_lower)
        relevance = matching_terms / len(query_terms) if query_terms else 0.0
        
        return min(relevance, 1.0)  # Clamp to [0, 1]

    def _calculate_groundedness(self, answer: str, contexts: List[str]) -> float:
        """
        Groundedness scoring: measures how much of the answer is supported by contexts.
        Simple approach: checks word overlap between answer and contexts.
        """
        if not answer or not contexts:
            return 0.0
        
        answer_words = set(answer.lower().split())
        context_text = " ".join(contexts).lower()
        context_words = set(context_text.split())
        
        # Calculate word overlap ratio
        overlap = len(answer_words.intersection(context_words))
        groundedness = overlap / len(answer_words) if answer_words else 0.0
        
        return min(groundedness, 1.0)  # Clamp to [0, 1]

    def _calculate_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """
        Faithfulness scoring: measures consistency between answer and contexts.
        Simple approach: word overlap with context sources.
        """
        if not answer or not contexts:
            return 0.0
        
        # Combine all contexts
        combined_contexts = " ".join(contexts)
        
        # Split into sentences/fragments for more granular comparison
        answer_lower = answer.lower()
        context_lower = combined_contexts.lower()
        
        # Simple heuristic: if key information from answer appears in contexts
        faithfulness_score = 0.0
        
        # Count overlapping n-grams (bigrams)
        answer_bigrams = set()
        answer_words = answer_lower.split()
        for i in range(len(answer_words) - 1):
            bigram = f"{answer_words[i]} {answer_words[i+1]}"
            answer_bigrams.add(bigram)
        
        context_bigrams = set()
        context_words = context_lower.split()
        for i in range(len(context_words) - 1):
            bigram = f"{context_words[i]} {context_words[i+1]}"
            context_bigrams.add(bigram)
        
        # Calculate bigram overlap ratio
        overlap = len(answer_bigrams.intersection(context_bigrams))
        faithfulness_score = overlap / len(answer_bigrams) if answer_bigrams else 0.0
        
        return min(faithfulness_score, 1.0)  # Clamp to [0, 1]

    def collect_all_metrics(
        self, 
        query: str, 
        answer: str, 
        contexts: List[str],
        embedding_latency: float,
        gen_latency: float,
        e2e_latency: float,
        llm: Optional[Any] = None,
        embeddings: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Collects all available metrics in one comprehensive call.
        
        Returns a dictionary with all performance and quality metrics.
        """
        hw_metrics = self.get_hardware_metrics()
        token_speed = self.calculate_token_speed(answer, gen_latency)
        quality_scores = self.get_quality_scores(query, answer, contexts, llm, embeddings)
        
        comprehensive_metrics = {
            "performance": {
                "embedding_latency_sec": embedding_latency,
                "generation_latency_sec": gen_latency,
                "end_to_end_latency_sec": e2e_latency,
                "token_generation_speed": token_speed,
                "gpu_utilization_pct": hw_metrics["gpu_utilization_pct"],
                "vram_usage_mb": hw_metrics["vram_usage_mb"]
            },
            "quality": quality_scores
        }
        
        return comprehensive_metrics