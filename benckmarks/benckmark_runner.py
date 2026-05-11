import yaml
import os
import time
import json
import logging
import sys
import numpy as np

# Add project root to path to resolve src imports correctly
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from langchain_ollama import ChatOllama
from src.ingestion.db_manager import SQLiteManager
from src.ingestion.parsing import PDFParser
from src.ingestion.indexing import SectionAwareChunker, FAISSIndexManager
from src.reasoning.processor import QueryProcessor
from src.reasoning.retrieval import (FAISSRetriever, BM25Retriever, SQLiteRetriever, HybridRetriever, AnswerAggregator)
from src.evaluation.runtime_metrics import RuntimeMetrics
from src.evaluation.retrieval_metrics import RetrievalMetrics

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder to handle NumPy types during JSON serialization."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def load_config():
    """Loads configuration from config.yaml in the project root."""
    config_path = os.path.join(root_dir, 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_benckmark():
    """
    Executes the end-to-end RAG pipeline for every entry in the golden set.
    Results and logs are saved inside the 'benckmark_result' directory.
    """
    config = load_config()

    # Resolve paths relative to root_dir
    output_path = config.get('output_path', 'output')
    if not os.path.isabs(output_path):
        output_path = os.path.join(root_dir, output_path)
    
    # Folder for benchmark outputs as per request
    benckmark_result_dir = os.path.join(output_path, 'benckmark_result')
    os.makedirs(benckmark_result_dir, exist_ok=True)
    
    db_path = config.get('db_path')
    if not os.path.isabs(db_path):
        db_path = os.path.join(root_dir, db_path)

    # Setup Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(benckmark_result_dir, "benckmark.log")),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # 1. Ingestion Phase (Ensure index and DB are ready)
    input_path = config.get('input_path')
    if not os.path.isabs(input_path):
        input_path = os.path.join(root_dir, input_path)
    input_file = os.path.join(input_path, config.get('input_file_name'))
    
    output_file_name = config.get('output_file')
    metadata_file_name = config.get('metadata_file')
    
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
        logger.info("Rebuilding parsing, chunking, and FAISS artifacts before benchmark.")
        logger.info(f"Pipeline Step: Parsing document {input_file}")
        parser = PDFParser(input_file)
        parsed_data = parser.parse()

        if db_path and parsed_data.get("tables"):
            logger.info(f"Pipeline Step: Saving tables to SQLite: {db_path}")
            db_mgr = SQLiteManager(db_path)
            db_mgr.insert_tables(input_file, parsed_data["tables"])
            db_mgr.close()

        logger.info("Pipeline Step: Chunking and Indexing.")
        chunker = SectionAwareChunker()
        chunks = chunker.chunk(parsed_data)

        indexer = FAISSIndexManager(model_name=config.get("embedding_model"))
        indexer.build_index(chunks)
        indexer.save(output_path, output_file_name, metadata_file_name)

    # 2. Loading Golden Set
    golden_set_path = os.path.join(root_dir, 'benckmarks', 'golden_qa_dataset.json')
    if not os.path.exists(golden_set_path):
        logger.error(f"Golden set file not found at {golden_set_path}. Benchmark aborted.")
        return

    with open(golden_set_path, 'r') as f:
        golden_set = json.load(f)

    # 3. Pipeline Initialization
    llm = ChatOllama(model=config['llm_model'], temperature=config['llm_temperature'], base_url=config['llm_base_url'])
    query_processor = QueryProcessor(llm=llm)
    runtime_engine = RuntimeMetrics()
    retrieval_engine = RetrievalMetrics()

    faiss_retriever = FAISSRetriever(
        index_path=os.path.join(output_path, output_file_name),
        metadata_path=os.path.join(output_path, metadata_file_name),
        model_name=config['retriever_model'],
        top_k=config['retrieval_top_k'],
    )
    bm25_retriever = BM25Retriever(corpus_metadata=faiss_retriever.metadata, top_k=config['retrieval_top_k'])
    sqlite_retriever = SQLiteRetriever(db_path=db_path, top_k=config['retrieval_top_k']) if os.path.exists(db_path) else None

    hybrid_retriever = HybridRetriever(faiss_retriever=faiss_retriever, bm25_retriever=bm25_retriever, 
                                     sqlite_retriever=sqlite_retriever, top_k=config['retrieval_top_k'])
    answer_aggregator = AnswerAggregator(llm=llm)

    # 4. Run Benchmark Comparison Loop
    modes = ["with_bm25", "without_bm25"]
    for mode in modes:
        logger.info(f"--- Starting Benchmark Mode: {mode} ({len(golden_set)} items) ---")
        
        # Configure hybrid retriever for the current mode
        # Passing None to bm25_retriever disables the BM25 branch in the hybrid retrieval logic
        current_bm25 = bm25_retriever if mode == "with_bm25" else None
        hybrid_retriever = HybridRetriever(
            faiss_retriever=faiss_retriever, 
            bm25_retriever=current_bm25, 
            sqlite_retriever=sqlite_retriever, 
            top_k=config['retrieval_top_k']
        )

        results = []
        for i, item in enumerate(golden_set, 1):
            query = str(item.get('question', '')).strip()
            if not query: continue
            
            logger.info(f"[{mode}] Processing Item {i}: {query}")
            try:
                start_time = time.perf_counter()
                
                # Run RAG steps
                sub_queries = query_processor.run(query)
                retrieval_results = hybrid_retriever.retrieve_for_queries(sub_queries)

                source_counts = {}
                for q_res in retrieval_results.values():
                    for ctx in q_res:
                        source = ctx.get("source_type", "unknown")
                        source_counts[source] = source_counts.get(source, 0) + 1

                logger.info(
                    "[%s] Retrieved %s contexts across %s sub-queries. sources=%s",
                    mode,
                    sum(len(q_res) for q_res in retrieval_results.values()),
                    len(sub_queries),
                    source_counts,
                )
                for sub_q, q_res in retrieval_results.items():
                    if not q_res:
                        logger.info("  No retrieval results for sub-query: %s", sub_q)
                        continue
                    for rank, ctx in enumerate(q_res[:3], 1):
                        metadata = ctx.get("metadata", {})
                        snippet = " ".join(str(ctx.get("text", "")).split())[:220]
                        logger.info(
                            "  %s #%s source=%s sources=%s fused=%.5f raw=%.5f page=%s text=%s",
                            sub_q,
                            rank,
                            ctx.get("source_type"),
                            ctx.get("source_types", [ctx.get("source_type")]),
                            float(ctx.get("fused_score", 0.0)),
                            float(ctx.get("score", 0.0)),
                            metadata.get("page"),
                            snippet,
                        )

                aggregated_answers = answer_aggregator.aggregate_all(sub_queries, retrieval_results)
                
                latency = runtime_engine.measure_duration(start_time)
                final_answer = "\n\n".join([f"Q: {q}\nA: {aggregated_answers.get(q, 'N/A')}" for q in sub_queries])
                
                # Evaluation metrics (RAGAS)
                all_contexts = [ctx['text'] for q_res in retrieval_results.values() for ctx in q_res if ctx.get("text")]
                quality_scores = retrieval_engine.get_quality_scores(query=query, answer=final_answer, 
                                                                contexts=all_contexts, llm=llm,
                                                                embeddings=faiss_retriever.model,
                                                                ground_truth=item.get("ground_truth", ""))
                
                results.append({
                    "query": query, 
                    "ground_truth": item.get("ground_truth", ""), 
                    "generated_answer": final_answer, 
                    "latency_s": latency, 
                    "quality_metrics": quality_scores,
                    "retrieval_debug": {
                        "sub_queries": sub_queries,
                        "context_count": len(all_contexts),
                        "source_counts": source_counts,
                        "contexts_preview": [
                            {
                                "source_type": ctx.get("source_type"),
                                "source_types": ctx.get("source_types", [ctx.get("source_type")]),
                                "score": ctx.get("score"),
                                "fused_score": ctx.get("fused_score"),
                                "metadata": ctx.get("metadata", {}),
                                "text_preview": " ".join(str(ctx.get("text", "")).split())[:300],
                            }
                            for q_res in retrieval_results.values()
                            for ctx in q_res[:2]
                        ],
                    },
                })
            except Exception as e:
                logger.error(f"Error processing item {i}: {str(e)}")
                continue

        # 5. Save benchmark result file for this specific execution mode
        results_file = os.path.join(benckmark_result_dir, f"benckmark_output_{mode}.json")
        logger.info(f"Saving {len(results)} results to {results_file}...")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, cls=NumpyEncoder, ensure_ascii=False)
        
        logger.info(f"Benchmark mode '{mode}' finished. Output saved to: {results_file}")

if __name__ == "__main__":
    run_benckmark()
