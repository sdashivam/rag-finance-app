import logging
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

try:
    from datasets import Dataset
    from langchain_core.embeddings import Embeddings
    from ragas import evaluate
    from ragas.metrics.collections import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    HAS_RAGAS = True
except ImportError:
    HAS_RAGAS = False
    Dataset = None
    Embeddings = object


class SentenceTransformerEmbeddings(Embeddings):
    """LangChain-compatible adapter for sentence-transformers models."""

    def __init__(self, model: Any):
        self.model = model

    def _encode(self, texts: List[str]) -> List[List[float]]:
        try:
            vectors = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except TypeError:
            vectors = self.model.encode(texts, convert_to_numpy=True)
        return vectors.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._encode([str(text or "") for text in texts])

    def embed_query(self, text: str) -> List[float]:
        return self._encode([str(text or "")])[0]


class RetrievalMetrics:
    """Evaluation utilities for retrieval and generation quality."""

    DEFAULT_SCORES = {
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "context_precision": 0.0,
        "context_recall": 0.0,
    }

    def _clean_contexts(self, contexts: List[str]) -> List[str]:
        if not isinstance(contexts, list):
            return []
        return [str(ctx).strip() for ctx in contexts if str(ctx or "").strip()]

    def _coerce_embeddings(self, embeddings: Any) -> Any:
        if embeddings is None:
            return None
        if hasattr(embeddings, "embed_query") and hasattr(embeddings, "embed_documents"):
            return embeddings
        if hasattr(embeddings, "encode"):
            return SentenceTransformerEmbeddings(embeddings)
        return embeddings

    def _scores_from_result(self, result: Any) -> Dict[str, float]:
        row = {}
        try:
            if hasattr(result, "to_pandas"):
                df = result.to_pandas()
                row = df.iloc[0].to_dict() if len(df) else {}
            elif hasattr(result, "scores"):
                scores = result.scores
                if hasattr(scores, "to_pandas"):
                    df = scores.to_pandas()
                    row = df.iloc[0].to_dict() if len(df) else {}
                elif isinstance(scores, list) and scores:
                    row = scores[0]
            elif isinstance(result, dict):
                row = result
        except Exception as exc:
            logger.warning("Could not parse RAGAS result object: %s", exc)

        scores = dict(self.DEFAULT_SCORES)
        scores["faithfulness"] = float(row.get("faithfulness", 0.0) or 0.0)
        scores["answer_relevancy"] = float(row.get("answer_relevancy", row.get("answer_relevance", 0.0)) or 0.0)
        scores["context_precision"] = float(row.get("context_precision", 0.0) or 0.0)
        scores["context_recall"] = float(row.get("context_recall", 0.0) or 0.0)
        return scores

    def validate_ragas_input(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        clean_contexts = self._clean_contexts(contexts)
        issues = []
        if not str(query or "").strip():
            issues.append("empty_question")
        if not str(answer or "").strip():
            issues.append("empty_answer")
        if not clean_contexts:
            issues.append("empty_contexts")
        if any(not isinstance(ctx, str) for ctx in contexts or []):
            issues.append("non_string_context")

        return {
            "valid": not issues,
            "issues": issues,
            "context_count": len(clean_contexts),
            "answer_is_no_evidence": "no relevant evidence" in str(answer or "").lower()
            or "no evidence" in str(answer or "").lower()
            or "insufficient evidence" in str(answer or "").lower(),
            "has_reference": bool(str(ground_truth or "").strip()),
        }

    def get_quality_scores(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        llm: Any,
        embeddings: Any,
        ground_truth: Optional[str] = None,
    ) -> dict:
        """
        Calculates RAGAS scores using the current single-turn schema:
        user_input, response, retrieved_contexts, and optional reference.
        """
        scores = dict(self.DEFAULT_SCORES)
        validation = self.validate_ragas_input(query, answer, contexts, ground_truth)

        if not HAS_RAGAS:
            logger.warning("RAGAS is not installed; returning zero quality scores.")
            return scores
        if llm is None:
            logger.warning("No evaluator LLM was provided; returning zero RAGAS quality scores.")
            return scores
        if not validation["valid"]:
            logger.warning("Skipping RAGAS evaluation because input is invalid: %s", validation["issues"])
            return scores

        clean_contexts = self._clean_contexts(contexts)
        reference = str(ground_truth or "").strip()
        selected_metrics = [faithfulness, answer_relevancy]
        if reference:
            selected_metrics.extend([context_precision, context_recall])

        try:
            dataset = Dataset.from_dict({
                "user_input": [str(query).strip()],
                "response": [str(answer).strip()],
                "retrieved_contexts": [clean_contexts],
                "reference": [reference],
            })

            result = evaluate(
                dataset=dataset,
                metrics=selected_metrics,
                llm=llm,
                embeddings=self._coerce_embeddings(embeddings),
                raise_exceptions=False,
                show_progress=False,
            )
            return self._scores_from_result(result)
        except Exception as e:
            logger.warning("Error calculating RAGAS quality scores: %s", e)
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
