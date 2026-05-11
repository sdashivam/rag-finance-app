import yaml
import os
import time
import numpy as np
import json
import logging
from langchain_ollama import ChatOllama
from src.ingestion.db_manager import SQLiteManager
from src.ingestion.parsing import PDFParser
from src.ingestion.indexing import SectionAwareChunker, FAISSIndexManager
from src.reasoning.processor import QueryProcessor
from src.reasoning.retrieval import (FAISSRetriever, BM25Retriever, SQLiteRetriever, HybridRetriever, AnswerAggregator)
from src.evaluation.feedback import FeedbackManager
from src.evaluation.runtime_metrics import RuntimeMetrics
from src.evaluation.retrieval_metrics import RetrievalMetrics


"""
Main script for the Financial Report RAG QA System.

Orchestrates the complete RAG pipeline:
1. Configuration loading
2. PDF parsing and table extraction
3. SQLite storage for structured data
4. Section-aware chunking and FAISS indexing
5. Query preprocessing and decomposition
6. Hybrid retrieval (FAISS + BM25 + SQLite)
7. LLM-based answer aggregation
8. Quality metrics evaluation
9. Results logging to JSON

Usage:
    python main.py
"""

def main():
    """Execute end-to-end RAG pipeline with logging and evaluation."""
    # Get the directory where main.py is located to resolve paths reliably
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, 'config.yaml')

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Constructing input path and retrieving output directory
    input_path = config.get('input_path')
    if not input_path:
        raise ValueError("input_path must be set in config.yaml.")
    if not os.path.isabs(input_path):
        input_path = os.path.join(base_dir, input_path)

    input_file_name = config.get('input_file_name')
    if not input_file_name:
        raise ValueError("input_file_name must be set in config.yaml.")
    input_file = os.path.join(input_path, input_file_name)

    output_path = config.get('output_path')
    if not output_path:
        raise ValueError("output_path must be set in config.yaml.")
    if not os.path.isabs(output_path):
        output_path = os.path.join(base_dir, output_path)

    output_file_name = config.get('output_file')
    if not output_file_name:
        raise ValueError("output_file must be set in config.yaml.")

    metadata_file_name = config.get('metadata_file')
    if not metadata_file_name:
        raise ValueError("metadata_file must be set in config.yaml.")

    db_path = config.get('db_path')
    if not db_path:
        raise ValueError("db_path must be set in config.yaml.")
    if not os.path.isabs(db_path):
        db_path = os.path.join(base_dir, db_path)

    # Setup Logging
    os.makedirs(output_path, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_path, "pipeline.log")),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # Check whether the persisted index is present and usable.
    index_report = FAISSIndexManager.inspect_saved(output_path, output_file_name, metadata_file_name)

    if index_report["valid"]:
        logger.info(
            "Existing FAISS artifacts are valid. vectors=%s metadata=%s non_empty_text=%s",
            index_report["faiss_ntotal"],
            index_report["metadata_count"],
            index_report["non_empty_text_count"],
        )
    else:
        logger.warning("FAISS artifacts are missing or invalid: %s", index_report["errors"])
        logger.info("Rebuilding parsing, chunking, and FAISS artifacts.")
        # 1. Parsing
        logger.info(f"Parsing document: {input_file}")
        parser = PDFParser(input_file)
        parsed_data = parser.parse()

        # 1.5 Structured Storage (SQLite)
        logger.info(f"Saving structured data to SQLite: {db_path}")
        if db_path and parsed_data.get("tables"):
            db_mgr = SQLiteManager(db_path)
            db_mgr.insert_tables(input_file, parsed_data["tables"])
            db_mgr.close()

        # 2. Chunking & Indexing
        logger.info(f"Section-aware Chunking document and building FAISS index.")
        chunker = SectionAwareChunker(config=config)
        chunks = chunker.chunk(parsed_data)

        embedding_model = config.get("embedding_model")
        if not embedding_model:
            raise ValueError("embedding_model must be set in config.yaml.")

        logger.info(f"Saving Semantic Retrival to FAISS index :{output_file_name}")
        indexer = FAISSIndexManager(config=config)
        indexer.build_index(chunks) # This builds the index in memory
        indexer.save(output_path, output_file_name, metadata_file_name) # This saves the index to disk

    # 3. Query Processing (Demonstration)
    logger.info("--- Starting Query Reasoning Phase ---")
    
    # Configuration-driven LLM setup
    model_name = config.get('llm_model')
    temp = config.get('llm_temperature')
    base_url = config.get('llm_base_url')

    if not model_name or not base_url:
        raise ValueError("llm_model and llm_base_url must be set in config.yaml.")

    try:
        llm_model = ChatOllama(
            model=model_name,
            temperature=temp,
            base_url=base_url
        )
    except Exception as e:
        logger.warning(f"Failed to initialize ChatOllama: {e}. Falling back to single-query mode.")
        llm_model = None
    
    # Pass the initialized LLM to the QueryProcessor
    query_processor = QueryProcessor(llm=llm_model, config=config)
    runtime_engine = RuntimeMetrics()
    retrieval_engine = RetrievalMetrics(config=config)

    # Capture Hardware Metrics
    hw_metrics = runtime_engine.get_hardware_metrics()
    logger.info(f"Hardware Stats - GPU Utilization: {hw_metrics['gpu_utilization_pct']}%, VRAM Usage: {hw_metrics['vram_usage_mb']:.2f} MB")

    sample_query = config.get("sample_query", "")
    e2e_start_time = time.perf_counter()

    if not sample_query:
        raise ValueError("The 'sample_query' value must be set in config.yaml.")

    logger.info(f"Processing Query: '{sample_query}'")
    processed_queries = query_processor.run(sample_query)
    
    logger.info("Decomposed Sub-queries for Retrieval:")
    for i, sub_q in enumerate(processed_queries, 1):
        logger.info(f"  {i}. {sub_q}")

    logger.info("--- Starting Hybrid Retrieval Phase ---")
    retriever_model = config.get("retriever_model")
    if not retriever_model:
        raise ValueError("retriever_model must be set in config.yaml.")

    metadata_file_name = config.get("metadata_file")
    if not metadata_file_name:
        raise ValueError("metadata_file must be set in config.yaml.")

    retrieval_top_k = config.get("retrieval_top_k")
    if retrieval_top_k is None:
        raise ValueError("retrieval_top_k must be set in config.yaml.")

    faiss_retriever = FAISSRetriever(
        index_path=os.path.join(output_path, output_file_name),
        metadata_path=os.path.join(output_path, metadata_file_name),
        model_name=retriever_model,
        top_k=retrieval_top_k,
        config=config,
    )

    bm25_retriever = BM25Retriever(
        corpus_metadata=faiss_retriever.metadata,
        top_k=retrieval_top_k,
        config=config,
    )

    sqlite_retriever = None
    if os.path.exists(db_path):
        try:
            sqlite_retriever = SQLiteRetriever(db_path=db_path, top_k=retrieval_top_k, config=config)
        except Exception as e:
            logger.warning(f"SQLite retrieval disabled: {e}")

    hybrid_retriever = HybridRetriever(
        faiss_retriever=faiss_retriever,
        bm25_retriever=bm25_retriever,
        sqlite_retriever=sqlite_retriever,
        top_k=retrieval_top_k,
        config=config,
    )

    # Measure Retrieval Phase (includes embedding generation)
    retrieval_start_time = time.perf_counter()
    retrieval_results = hybrid_retriever.retrieve_for_queries(processed_queries)
    embedding_latency = runtime_engine.measure_duration(retrieval_start_time)
    logger.info(f"Embedding/Retrieval Latency: {embedding_latency:.4f}s")

    # Measure Aggregation/Generation phase
    gen_start_time = time.perf_counter()
    answer_aggregator = AnswerAggregator(llm=llm_model)
    aggregated_answers = answer_aggregator.aggregate_all(processed_queries, retrieval_results)
    gen_duration = runtime_engine.measure_duration(gen_start_time)
    
    e2e_response_time = runtime_engine.measure_duration(e2e_start_time)
    logger.info(f"End-to-End Response Time: {e2e_response_time:.4f}s")
    
    # Initialize FeedbackManager for logging evaluation runs
    feedback_mgr = FeedbackManager(db_path=db_path)

    final_answer = "\n\n".join([f"Q: {q}\nA: {aggregated_answers.get(q, 'No answer generated.')}" for q in processed_queries])

    for i, sub_q in enumerate(processed_queries, 1):
        logger.info(f"Sub-query {i}: {sub_q}")
        results = retrieval_results.get(sub_q, [])
        if not results:
            logger.info("  No relevant evidence found.")
            continue
        for j, result in enumerate(results, 1):
            metadata = result.get("metadata", {})
            logger.info(
                f"  Result {j}: source={result.get('source_type')} score={result.get('score', 0):.4f} "
                f"page={metadata.get('page')} section={metadata.get('type')}"
            )
            logger.info(f"    {result['text'][:320]}{'...' if len(result['text']) > 320 else ''}")

        logger.info("  Aggregated Answer:")
        logger.info(f"    {aggregated_answers.get(sub_q, 'No answer generated.')}")

    token_speed = runtime_engine.calculate_token_speed(final_answer, gen_duration)
    logger.info(f"Token Generation Speed: {token_speed:.2f} tokens/sec")

    # Log the overall session to the feedback database for future trend analysis
    feedback_mgr.log_interaction(
        query=sample_query,
        answer=final_answer,
        contexts=[ctx for res in retrieval_results.values() for ctx in res],
        metadata={"run_type": "eval_script"}
    )

    # Evaluation
    logger.info("--- Evaluation Phase ---")

    # For demonstration, using empty relevant docs since no ground truth available
    # In practice, provide actual relevant document IDs/texts
    relevant_docs = set()  # TODO: Replace with actual ground truth relevant documents

    all_metrics_data = {
        "query": sample_query,
        "final_answer": final_answer,
        "sub_queries": [],
        "overall_retrieval_scores": {},
        "overall_generation_scores": {},
        "performance_metrics": {
            "embedding_generation_latency": embedding_latency,
            "end_to_end_response_time": e2e_response_time,
            "token_generation_speed": token_speed,
            "gpu_utilization_pct": hw_metrics["gpu_utilization_pct"],
            "vram_usage_mb": hw_metrics["vram_usage_mb"]
        },
        "quality_metrics": {}
    }

    # Calculate RAGAS Quality Metrics (Faithfulness, Relevancy, Precision, Recall)
    all_contexts = [ctx['text'] for q_res in retrieval_results.values() for ctx in q_res if ctx.get("text")]
    
    logger.info("Calculating RAGAS quality scores...")
    quality_scores = retrieval_engine.get_quality_scores(
        query=sample_query,
        answer=final_answer,
        contexts=all_contexts,
        llm=llm_model,
        embeddings=faiss_retriever.model
    )
    all_metrics_data["quality_metrics"] = quality_scores
    logger.info(f"Quality Metrics: {json.dumps(quality_scores, indent=2)}")

    overall_retrieval_scores = []
    overall_generation_scores = []

    for i, sub_q in enumerate(processed_queries, 1):
        logger.info(f"Evaluating Sub-query {i}: {sub_q}")
        answer = aggregated_answers.get(sub_q, '')
        sub_query_metrics = {
            "sub_query": sub_q, 
            "answer": answer, 
            "retrieval": {"precision_at_k": 0.0}, 
            "generation": {"relevance": 0.0, "groundedness": 0.0, "faithfulness": 0.0}
        }

        # Retrieval Evaluation
        results = retrieval_results.get(sub_q, [])
        if results:
            ret_scores = retrieval_engine.evaluate_retrieval(results, relevant_docs, retrieval_top_k)
            logger.info(f"  Retrieval - Precision@{retrieval_top_k}: {ret_scores['precision_at_k']:.4f}")
            sub_query_metrics["retrieval"] = ret_scores
            overall_retrieval_scores.append(ret_scores['precision_at_k'])
        else:
            logger.info("  Retrieval - No results to evaluate")
            sub_query_metrics["retrieval"] = {"precision_at_k": 0.0}

        # Generation Evaluation
        sub_contexts = [res['text'] for res in results]
        if answer and sub_contexts:
            gen_scores = retrieval_engine.evaluate_generation(sub_q, answer, sub_contexts)
            logger.info(f"  Generation - Relevance: {gen_scores['relevance']:.4f}")
            logger.info(f"  Generation - Groundedness: {gen_scores['groundedness']:.4f}")
            logger.info(f"  Generation - Faithfulness: {gen_scores['faithfulness']:.4f}")
            sub_query_metrics["generation"] = gen_scores
            overall_generation_scores.append(gen_scores)
        else:
            logger.info("  Generation - No answer or contexts to evaluate")
            sub_query_metrics["generation"] = {"relevance": 0.0, "groundedness": 0.0, "faithfulness": 0.0}

        all_metrics_data["sub_queries"].append(sub_query_metrics)

    # Overall Summary
    avg_precision = np.mean(overall_retrieval_scores) if overall_retrieval_scores else 0.0
    all_metrics_data["overall_retrieval_scores"] = {"average_precision_at_k": avg_precision}
    logger.info(f"Overall Retrieval - Average Precision@{retrieval_top_k}: {avg_precision:.4f}")
    
    if overall_generation_scores:
        avg_relevance = np.mean([s['relevance'] for s in overall_generation_scores])
        avg_groundedness = np.mean([s['groundedness'] for s in overall_generation_scores])
        avg_faithfulness = np.mean([s['faithfulness'] for s in overall_generation_scores])

        logger.info(f"Overall Generation - Average Relevance: {avg_relevance:.4f}")
        logger.info(f"Overall Generation - Average Groundedness: {avg_groundedness:.4f}")
        logger.info(f"Overall Generation - Average Faithfulness: {avg_faithfulness:.4f}")

        all_metrics_data["overall_generation_scores"] = {
            "average_relevance": avg_relevance,
            "average_groundedness": avg_groundedness,
            "average_faithfulness": avg_faithfulness
        }
    else:
        all_metrics_data["overall_generation_scores"] = {
            "average_relevance": 0.0, "average_groundedness": 0.0, "average_faithfulness": 0.0
        }

    # Save all metrics to a JSON file
    metrics_output_file = os.path.join(output_path, "rag_metrics.json")
    with open(metrics_output_file, 'w', encoding='utf-8') as f:
        json.dump(all_metrics_data, f, indent=4)
    logger.info(f"All RAG metrics saved to {metrics_output_file}")
    logger.info(f"Pipeline execution complete. Artifacts stored in {output_path}")

if __name__ == "__main__":
    main()
