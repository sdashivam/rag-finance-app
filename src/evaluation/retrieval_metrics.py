"""
Retrieval and generation quality evaluation metrics.

Provides embedding-based metrics for RAG pipeline assessment.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class RetrievalMetrics:
    """Evaluation utilities for retrieval and generation quality.

    Args:
        config: Optional config dict with evaluation_model and device settings.
    """

    DEFAULT_SCORES = {
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "context_precision": 0.0,
        "context_recall": 0.0,
        "groundedness": 0.0,
    }

    def __init__(self, config: dict = None):
        """Initialize RetrievalMetrics with optional config."""
        self.config = config or {}

    def _clean_contexts(self, contexts: List[str]) -> List[str]:
        if not isinstance(contexts, list):
            return []
        return [str(ctx).strip() for ctx in contexts if str(ctx or "").strip()]

    def get_quality_scores(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        llm: Any = None,
        embeddings: Any = None,
        ground_truth: Optional[str] = None,
    ) -> dict:
        """Calculate quality scores using custom embedding-based evaluation.

        Args:
            query: User query string.
            answer: Generated answer string.
            contexts: List of retrieved context strings.
            llm: Optional LLM instance (unused in current implementation).
            embeddings: Optional pre-loaded embeddings model.
            ground_truth: Optional ground truth answer for recall calculation.

        Returns:
            Dict with faithfulness, answer_relevancy, context_precision,
            context_recall, groundedness scores.
        """
        scores = dict(self.DEFAULT_SCORES)
        clean_contexts = self._clean_contexts(contexts)

        if not answer or not clean_contexts:
            return scores

        # Get embeddings model - prefer provided, fallback to config
        emb_model = None
        if embeddings is not None and hasattr(embeddings, 'encode'):
            emb_model = embeddings

        if emb_model is None:
            try:
                # Load from config or use defaults
                model_name = self.config.get('evaluation_model', 'all-MiniLM-L6-v2')
                device_setting = self.config.get('device', 'auto')

                if device_setting == "auto":
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                elif device_setting in ("cuda", "cpu"):
                    device = device_setting
                else:
                    device = "cpu"

                emb_model = SentenceTransformer(model_name, device=device)
            except Exception:
                logger.warning("Could not load embeddings model")

        if emb_model:
            custom_scores = self.evaluate_with_embeddings(
                query=query,
                answer=answer,
                contexts=clean_contexts,
                embeddings_model=emb_model,
                ground_truth=ground_truth
            )
            scores.update(custom_scores)

        return scores

    def evaluate_with_embeddings(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        embeddings_model: Any = None,
        ground_truth: Optional[str] = None,
    ) -> dict:
        """
        Calculates semantic-based quality metrics without RAGAS.
        Uses embeddings for similarity calculations.

        Metrics:
        - Faithfulness: Whether the generated answer is supported by the retrieved context. 
                        {Answer -> Retrieved Context}
        - Answer Relevancy: Semantic similarity between answer and query
        - Groundedness: Whether the response is grounded in evidence/source material. 
                        {Answer -> Source Evidence}
        - Context Precision: How well retrieved contexts match the ground truth.
                            {Relevant Retrieved Contexts / Total Retrieved Contexts​}
        - Context Recall: How much of ground truth is covered by contexts.
                        {Relevant Retrieved Contexts / Total Relevant Contexts Available}
        """
        clean_contexts = self._clean_contexts(contexts)
        scores = dict(self.DEFAULT_SCORES)

        if not answer or not clean_contexts:
            return scores

        # Load sentence transformer if not provided
        if embeddings_model is None:
            try:
                embeddings_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            except Exception:
                logger.warning("Could not load sentence transformer for evaluation")
                return scores

        try:
            # Combine all contexts into one
            combined_context = " ".join(clean_contexts)

            # Get embeddings
            query_emb = embeddings_model.encode(str(query), convert_to_numpy=True)
            answer_emb = embeddings_model.encode(str(answer), convert_to_numpy=True)
            context_emb = embeddings_model.encode(combined_context, convert_to_numpy=True)

            # Normalize for cosine similarity
            query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
            answer_emb = answer_emb / (np.linalg.norm(answer_emb) + 1e-8)
            context_emb = context_emb / (np.linalg.norm(context_emb) + 1e-8)

            # Answer Relevancy: cosine similarity between answer and query
            answer_relevancy = float(np.dot(answer_emb, query_emb))
            scores["answer_relevancy"] = max(0.0, answer_relevancy)

            # Faithfulness/Groundedness: How much answer is supported by context
            answer_in_context = float(np.dot(answer_emb, context_emb))
            scores["faithfulness"] = max(0.0, answer_in_context)
            scores["groundedness"] = max(0.0, answer_in_context)

            # Check if key numbers/entities from answer appear in context
            answer_text = str(answer).lower()
            numbers_in_answer = re.findall(r'-?[\d,]+\.?\d*', answer_text)
            context_text = combined_context.lower()

            # Number match for faithfulness
            number_matches = 0
            for num in numbers_in_answer:
                num_clean = num.replace(',', '')
                if num_clean in context_text or num in context_text:
                    number_matches += 1

            if numbers_in_answer:
                number_match_score = number_matches / len(numbers_in_answer)
                scores["faithfulness"] = (scores["faithfulness"] + number_match_score) / 2
                scores["groundedness"] = scores["faithfulness"]

            # Context Precision: Do the retrieved contexts contain relevant info?
            # Compare each context to the query
            context_precision_scores = []
            for ctx in clean_contexts:
                ctx_emb = embeddings_model.encode(str(ctx), convert_to_numpy=True)
                ctx_emb = ctx_emb / (np.linalg.norm(ctx_emb) + 1e-8)
                ctx_similarity = float(np.dot(ctx_emb, query_emb))
                context_precision_scores.append(max(0.0, ctx_similarity))

            scores["context_precision"] = np.mean(context_precision_scores) if context_precision_scores else 0.0 # pyright: ignore[reportArgumentType]

            # Context Recall: How much of ground truth is in contexts?
            if ground_truth:
                gt_emb = embeddings_model.encode(str(ground_truth), convert_to_numpy=True)
                gt_emb = gt_emb / (np.linalg.norm(gt_emb) + 1e-8)
                gt_in_context = float(np.dot(gt_emb, context_emb))
                scores["context_recall"] = max(0.0, gt_in_context)

                # Also check if ground truth numbers appear in contexts
                gt_text = str(ground_truth).lower()
                gt_numbers = re.findall(r'-?[\d,]+\.?\d*', gt_text)
                gt_number_matches = 0
                for num in gt_numbers:
                    num_clean = num.replace(',', '')
                    if num_clean in context_text or num in context_text:
                        gt_number_matches += 1

                if gt_numbers:
                    gt_number_score = gt_number_matches / len(gt_numbers)
                    scores["context_recall"] = (scores["context_recall"] + gt_number_score) / 2
            else:
                # Without ground truth, estimate based on answer-content overlap
                answer_in_context_score = float(np.dot(answer_emb, context_emb))
                scores["context_recall"] = max(0.0, answer_in_context_score)

        except Exception as e:
            logger.warning(f"Error in embedding-based evaluation: {e}")

        return scores

    def evaluate_retrieval(self, results: List[Dict], relevant_docs: Set[str], top_k: int) -> dict:
        """Calculates IR metrics like Precision@K for the retrieved chunks."""
        if not results:
            return {"precision_at_k": 0.0}
        if not relevant_docs:
            return {"precision_at_k": 0.0}

        hits = 0
        for res in results[:top_k]:
            doc_text = res.get("text", "")
            if any(truth in doc_text for truth in relevant_docs):
                hits += 1

        denominator = min(top_k, len(results)) or 1
        return {"precision_at_k": hits / denominator}

    def evaluate_generation(self, query: str, answer: str, contexts: List[str]) -> dict:
        """Calculates lightweight generation scores for quick local smoke tests."""
        clean_contexts = self._clean_contexts(contexts)
        if not answer or not clean_contexts:
            return {
                "relevance": 0.0,
                "groundedness": 0.0,
                "faithfulness": 0.0,
            }

        query_terms = set(str(query).lower().split())
        answer_terms = set(str(answer).lower().split())
        context_terms = set(" ".join(clean_contexts).lower().split())

        relevance = len(query_terms & answer_terms) / len(query_terms) if query_terms else 0.0
        groundedness = len(answer_terms & context_terms) / len(answer_terms) if answer_terms else 0.0
        return {
            "relevance": min(float(relevance), 1.0),
            "groundedness": min(float(groundedness), 1.0),
            "faithfulness": min(float(groundedness), 1.0),
        }