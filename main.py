import yaml
import os
import numpy as np
import json
import logging
from src.ingestion.parsing import PDFParser
from src.ingestion.indexing import SectionAwareChunker, FAISSIndexManager
from src.reasoning.processor import QueryProcessor
from src.reasoning.retrieval import (FAISSRetriever, SQLiteRetriever, HybridRetriever, AnswerAggregator)
from src.evaluation.metrics import RAGEvaluator
from langchain_ollama import ChatOllama
from src.evaluation.feedback import FeedbackManager
from src.ingestion.db_manager import SQLiteManager

def main():
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

    # Check if index and metadata already exist
    index_exists = os.path.exists(os.path.join(output_path, output_file_name))
    metadata_exists = os.path.exists(os.path.join(output_path, metadata_file_name))

    if index_exists and metadata_exists:
        logger.info(f"Existing index and metadata found in {output_path}. Skipping parsing and indexing.")
    else:
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
        chunker = SectionAwareChunker()
        chunks = chunker.chunk(parsed_data)

        embedding_model = config.get("embedding_model")
        if not embedding_model:
            raise ValueError("embedding_model must be set in config.yaml.")

        logger.info(f"Saving Semantic Retrival to FAISS index :{output_file_name}")
        indexer = FAISSIndexManager(model_name=embedding_model)
        indexer.build_index(chunks) # This builds the index in memory
        indexer.save(output_path, output_file_name) # This saves the index to disk

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
    query_processor = QueryProcessor(llm=llm_model)
    sample_query = config.get("sample_query", "")
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
    )

    sqlite_retriever = None
    if os.path.exists(db_path):
        try:
            sqlite_retriever = SQLiteRetriever(db_path=db_path, top_k=retrieval_top_k)
        except Exception as e:
            logger.warning(f"SQLite retrieval disabled: {e}")

    hybrid_retriever = HybridRetriever(
        faiss_retriever=faiss_retriever,
        sqlite_retriever=sqlite_retriever,
        top_k=retrieval_top_k,
    )

    retrieval_results = hybrid_retriever.retrieve_for_queries(processed_queries)
    answer_aggregator = AnswerAggregator(llm=llm_model)
    aggregated_answers = answer_aggregator.aggregate_all(processed_queries, retrieval_results)
    
    # Initialize FeedbackManager for logging evaluation runs
    feedback_mgr = FeedbackManager(db_path=db_path)

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

    # Log the overall session to the feedback database for future trend analysis
    feedback_mgr.log_interaction(
        query=sample_query,
        answer="\n\n".join([f"Q: {q}\nA: {aggregated_answers.get(q)}" for q in processed_queries]),
        contexts=[ctx for res in retrieval_results.values() for ctx in res],
        metadata={"run_type": "eval_script"}
    )

    # Evaluation
    logger.info("--- Evaluation Phase ---")
    evaluator = RAGEvaluator()

    # For demonstration, using empty relevant docs since no ground truth available
    # In practice, provide actual relevant document IDs/texts
    relevant_docs = set()  # TODO: Replace with actual ground truth relevant documents

    overall_retrieval_scores = []
    overall_generation_scores = []

    # Construct a consolidated final answer from sub-query answers for logging
    final_answer = "\n\n".join([f"Q: {q}\nA: {aggregated_answers.get(q, 'No answer generated.')}" for q in processed_queries])

    all_metrics_data = {
        "query": sample_query,
        "final_answer": final_answer,
        "sub_queries": [],
        "overall_retrieval_scores": {},
        "overall_generation_scores": {}
    }

    for i, sub_q in enumerate(processed_queries, 1):
        logger.info(f"Evaluating Sub-query {i}: {sub_q}")
        answer = aggregated_answers.get(sub_q, '')
        sub_query_metrics = {"sub_query": sub_q, "answer": answer, "retrieval": {}, "generation": {}}

        # Retrieval Evaluation
        results = retrieval_results.get(sub_q, [])
        if results:
            retrieval_scores = evaluator.evaluate_retrieval(results, relevant_docs, retrieval_top_k)
            logger.info(f"  Retrieval - Precision@{retrieval_top_k}: {retrieval_scores['precision_at_k']:.4f}")
            sub_query_metrics["retrieval"] = retrieval_scores
            overall_retrieval_scores.append(retrieval_scores['precision_at_k'])
        else:
            logger.info("  Retrieval - No results to evaluate")
            sub_query_metrics["retrieval"] = {"precision_at_k": 0.0} # Indicate no results

        # Generation Evaluation
        contexts = [result['text'] for result in results] if results else []

        if answer and contexts:
            generation_scores = evaluator.evaluate_generation(sub_q, answer, contexts)
            logger.info(f"  Generation - Relevance: {generation_scores['relevance']:.4f}")
            logger.info(f"  Generation - Groundedness: {generation_scores['groundedness']:.4f}")
            logger.info(f"  Generation - Faithfulness: {generation_scores['faithfulness']:.4f}")
            sub_query_metrics["generation"] = generation_scores
            overall_generation_scores.append(generation_scores)
        else:
            logger.info("  Generation - No answer or contexts to evaluate")
            sub_query_metrics["generation"] = {"relevance": 0.0, "groundedness": 0.0, "faithfulness": 0.0} # Indicate no results

        all_metrics_data["sub_queries"].append(sub_query_metrics)

    # Overall Scores
    overall_retrieval_summary = {}
    overall_generation_summary = {}

    if overall_retrieval_scores:
        avg_precision = np.mean(overall_retrieval_scores)
        logger.info(f"Overall Retrieval - Average Precision@{retrieval_top_k}: {avg_precision:.4f}")

    if overall_generation_scores:
        avg_relevance = np.mean([s['relevance'] for s in overall_generation_scores])
        avg_groundedness = np.mean([s['groundedness'] for s in overall_generation_scores])
        avg_faithfulness = np.mean([s['faithfulness'] for s in overall_generation_scores])

        logger.info(f"Overall Generation - Average Relevance: {avg_relevance:.4f}")
        logger.info(f"Overall Generation - Average Groundedness: {avg_groundedness:.4f}")
        logger.info(f"Overall Generation - Average Faithfulness: {avg_faithfulness:.4f}")

        overall_retrieval_summary = {"average_precision_at_k": avg_precision}
        overall_generation_summary = {
            "average_relevance": avg_relevance,
            "average_groundedness": avg_groundedness,
            "average_faithfulness": avg_faithfulness
        }

    all_metrics_data["overall_retrieval_scores"] = overall_retrieval_summary
    all_metrics_data["overall_generation_scores"] = overall_generation_summary

    # Save all metrics to a JSON file
    metrics_output_file = os.path.join(output_path, "rag_metrics.json")
    with open(metrics_output_file, 'w', encoding='utf-8') as f:
        json.dump(all_metrics_data, f, indent=4)
    logger.info(f"All RAG metrics saved to {metrics_output_file}")
    logger.info(f"Pipeline execution complete. Artifacts stored in {output_path}")

if __name__ == "__main__":
    main()