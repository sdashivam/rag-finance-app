import streamlit as st
import yaml
import os
import sys

# """
# Streamlit application for the Financial Report RAG QA System.
# This app provides a user interface to interact with the RAG pipeline, submit queries,
# and provide feedback on the generated answers.
# """

# Add project root to path to resolve src imports correctly
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)

from src.reasoning.processor import QueryProcessor
from src.reasoning.retrieval import (FAISSRetriever, SQLiteRetriever, HybridRetriever, AnswerAggregator)
from src.evaluation.feedback import FeedbackManager
from langchain_ollama import ChatOllama

def load_config():
    """
    Loads the configuration from the config.yaml file.
    Returns: dict: The loaded configuration.
    """
    config_path = os.path.join(root_dir, 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

@st.cache_resource
def initialize_rag(_config):
    """Initializes and caches the RAG components for the UI."""
    # Resolve absolute paths
    output_path = _config['output_path']
    if not os.path.isabs(output_path):
        output_path = os.path.join(root_dir, output_path)
        
    db_path = _config['db_path']
    if not os.path.isabs(db_path):
        db_path = os.path.join(root_dir, db_path)

    llm = ChatOllama(
        model=_config['llm_model'],
        temperature=_config['llm_temperature'],
        base_url=_config['llm_base_url']
    )
    
    faiss_retriever = FAISSRetriever(
        index_path=os.path.join(output_path, _config['output_file']),
        metadata_path=os.path.join(output_path, _config['metadata_file']),
        model_name=_config['retriever_model'],
        top_k=_config['retrieval_top_k'],
    )
    
    sqlite_retriever = None
    if os.path.exists(db_path):
        sqlite_retriever = SQLiteRetriever(db_path=db_path, top_k=_config['retrieval_top_k'])

    hybrid_retriever = HybridRetriever(
        faiss_retriever=faiss_retriever,
        sqlite_retriever=sqlite_retriever,
        top_k=_config['retrieval_top_k'],
    )
    
    return (
        QueryProcessor(llm=llm),
        hybrid_retriever,
        AnswerAggregator(llm=llm),
        FeedbackManager(db_path=db_path)
    )

def main():
    """
    Main function to run the Streamlit application.
    """
    st.set_page_config(page_title="FinChat RAG", page_icon="📉", layout="wide")
    st.title("📊 Financial Report RAG System")
    
    config = load_config()
    processor, retriever, aggregator, feedback_mgr = initialize_rag(config)

    # Initialize session state for feedback
    if "interaction_id" not in st.session_state:
        st.session_state.interaction_id = None

    query = st.text_input("Enter your financial query:", placeholder="e.g. What was the revenue growth in 2023?")

    if st.button("Generate Answer") and query:
        with st.spinner("Analyzing document..."):
            # 1. Reasoning
            sub_queries = processor.run(query)
            
            # 2. Retrieval
            results = retriever.retrieve_for_queries(sub_queries)
            
            # 3. Aggregation
            answers = aggregator.aggregate_all(sub_queries, results)
            
            # Display breakdown
            st.subheader("Analysis Results")
            final_response_parts = []
            for i, q in enumerate(sub_queries, 1):
                with st.expander(f"Step {i}: {q}", expanded=True):
                    ans = answers.get(q, "No answer generated.")
                    st.write(ans)
                    final_response_parts.append(f"Q: {q}\nA: {ans}")
            
            # Log interaction for feedback
            all_contexts = [ctx for q_res in results.values() for ctx in q_res]
            st.session_state.interaction_id = feedback_mgr.log_interaction(
                query=query,
                answer="\n\n".join(final_response_parts),
                contexts=all_contexts
            )
            st.success("Analysis complete. Please provide feedback below.")

    # Feedback Loop UI
    if st.session_state.interaction_id:
        st.divider()
        st.write("### Quality Feedback")
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("👍 Correct"):
                feedback_mgr.submit_feedback(st.session_state.interaction_id, 1)
                st.balloons()
        with col2:
            if st.button("👎 Incorrect/Hallucination"):
                feedback_mgr.submit_feedback(st.session_state.interaction_id, 0)
                st.warning("Feedback logged. We will analyze the retrieval failure.")

if __name__ == "__main__":
    main()