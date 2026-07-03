"""
Streamlit demo application for embedding and reranker fine-tuning.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
from typing import Dict, List
import numpy as np

from src.inference.pipeline import SearchPipeline, create_demo_corpus
from src.data.dataset_loader import DatasetLoader
from src.evaluation.metrics import RankingMetrics


# Page configuration
st.set_page_config(
    page_title="Embedding & Reranker Fine-tuning Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    margin-bottom: 2rem;
    background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #4ECDC4;
    margin: 0.5rem 0;
}

.search-result {
    background-color: #ffffff;
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #e0e0e0;
    margin: 0.5rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.score-bar {
    height: 20px;
    background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
    border-radius: 10px;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    """Load the search pipeline with caching."""
    try:
        # Try to load fine-tuned models
        embedding_path = "./models/embedding/final_model"
        reranker_path = "./models/reranker/final_model"
        
        # Create demo corpus
        corpus = create_demo_corpus()
        
        pipeline = SearchPipeline(
            embedding_model_path=embedding_path,
            reranker_model_path=reranker_path,
            corpus=corpus
        )
        
        return pipeline, True
    except Exception as e:
        st.error(f"Error loading pipeline: {e}")
        return None, False


@st.cache_data
def load_evaluation_results():
    """Load evaluation results if available."""
    results_path = "./evaluation_results/comprehensive_evaluation.json"
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            return json.load(f)
    return None


def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 Embedding & Reranker Fine-tuning Demo</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        use_reranker = st.checkbox("Use Reranker", value=True, 
                                  help="Enable reranking after initial retrieval")
        
        # Search parameters
        st.subheader("Search Parameters")
        top_k = st.slider("Top K Results", min_value=1, max_value=20, value=10)
        retrieval_k = st.slider("Retrieval K", min_value=10, max_value=100, value=50,
                               help="Number of candidates to retrieve before reranking")
        
        # Display options
        st.subheader("Display Options")
        show_scores = st.checkbox("Show Scores", value=True)
        show_timing = st.checkbox("Show Timing", value=True)
        show_metadata = st.checkbox("Show Metadata", value=False)
    
    # Load pipeline
    pipeline, pipeline_loaded = load_pipeline()
    
    if not pipeline_loaded:
        st.error("Failed to load pipeline. Please ensure models are trained and available.")
        st.stop()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Search Demo", "📊 Model Comparison", "📈 Evaluation", "🔧 Training"])
    
    with tab1:
        st.header("Interactive Search Demo")
        
        # Search interface
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_input("Enter your search query:", 
                                 placeholder="e.g., What is machine learning?",
                                 key="search_query")
        
        with col2:
            search_button = st.button("🔍 Search", type="primary")
        
        if query and (search_button or query):
            # Perform search
            with st.spinner("Searching..."):
                start_time = time.time()
                
                # Search with and without reranker for comparison
                results_with_reranker = pipeline.search(
                    query, top_k=top_k, retrieval_k=retrieval_k, use_reranker=True
                )
                
                results_without_reranker = pipeline.search(
                    query, top_k=top_k, retrieval_k=retrieval_k, use_reranker=False
                )
                
                search_time = time.time() - start_time
            
            # Display results
            if use_reranker:
                current_results = results_with_reranker
                st.subheader("🎯 Results with Reranker")
            else:
                current_results = results_without_reranker
                st.subheader("📋 Results without Reranker")
            
            # Metrics
            if show_timing:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Time", f"{current_results['metadata']['total_time']:.3f}s")
                with col2:
                    st.metric("Retrieval Time", f"{current_results['metadata']['retrieval_time']:.3f}s")
                with col3:
                    st.metric("Rerank Time", f"{current_results['metadata']['rerank_time']:.3f}s")
                with col4:
                    st.metric("Results Found", len(current_results['results']))
            
            # Results display
            for i, result in enumerate(current_results['results']):
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        st.markdown(f"**Rank {result['rank']}**")
                        st.markdown(result['document'])
                    
                    with col2:
                        if show_scores:
                            score = result['score']
                            st.metric("Score", f"{score:.4f}")
                            
                            # Score visualization
                            fig = go.Figure(go.Indicator(
                                mode = "gauge+number",
                                value = score,
                                domain = {'x': [0, 1], 'y': [0, 1]},
                                title = {'text': "Relevance"},
                                gauge = {
                                    'axis': {'range': [None, 1]},
                                    'bar': {'color': "#4ECDC4"},
                                    'steps': [
                                        {'range': [0, 0.3], 'color': "#FFE5E5"},
                                        {'range': [0.3, 0.7], 'color': "#FFF8DC"},
                                        {'range': [0.7, 1], 'color': "#E5F5F5"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': 0.8
                                    }
                                }
                            ))
                            fig.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=20))
                            st.plotly_chart(fig, use_container_width=True)
                    
                    st.divider()
            
            # Show metadata
            if show_metadata:
                with st.expander("🔍 Search Metadata"):
                    st.json(current_results['metadata'])
    
    with tab2:
        st.header("Model Comparison")
        
        if query:
            # Compare results side by side
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📋 Without Reranker")
                for result in results_without_reranker['results'][:5]:
                    with st.container():
                        st.markdown(f"**Rank {result['rank']} (Score: {result['score']:.4f})**")
                        st.markdown(result['document'])
                        st.divider()
            
            with col2:
                st.subheader("🎯 With Reranker")
                for result in results_with_reranker['results'][:5]:
                    with st.container():
                        st.markdown(f"**Rank {result['rank']} (Score: {result['score']:.4f})**")
                        st.markdown(result['document'])
                        st.divider()
            
            # Performance comparison
            st.subheader("⚡ Performance Comparison")
            
            comparison_data = {
                'Model': ['Embedding Only', 'Embedding + Reranker'],
                'Total Time (s)': [
                    results_without_reranker['metadata']['total_time'],
                    results_with_reranker['metadata']['total_time']
                ],
                'Retrieval Time (s)': [
                    results_without_reranker['metadata']['retrieval_time'],
                    results_with_reranker['metadata']['retrieval_time']
                ],
                'Rerank Time (s)': [
                    0,
                    results_with_reranker['metadata']['rerank_time']
                ]
            }
            
            df_comparison = pd.DataFrame(comparison_data)
            
            # Timing comparison chart
            fig = px.bar(df_comparison, x='Model', y=['Retrieval Time (s)', 'Rerank Time (s)'],
                        title="Search Time Breakdown", barmode='stack')
            st.plotly_chart(fig, use_container_width=True)
            
            # Score distribution comparison
            scores_embedding = [r['score'] for r in results_without_reranker['results']]
            scores_reranker = [r['score'] for r in results_with_reranker['results']]
            
            fig = make_subplots(rows=1, cols=2, 
                              subplot_titles=('Embedding Scores', 'Reranker Scores'))
            
            fig.add_trace(go.Histogram(x=scores_embedding, name="Embedding", nbinsx=10), 
                         row=1, col=1)
            fig.add_trace(go.Histogram(x=scores_reranker, name="Reranker", nbinsx=10), 
                         row=1, col=2)
            
            fig.update_layout(title="Score Distribution Comparison", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Enter a search query in the Search Demo tab to see model comparison.")
    
    with tab3:
        st.header("Model Evaluation Results")
        
        # Load evaluation results
        eval_results = load_evaluation_results()
        
        if eval_results:
            st.success("✅ Evaluation results loaded successfully!")
            
            # Metrics comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Embedding Model Metrics")
                embedding_metrics = eval_results.get('embedding_model_results', {})
                for metric, value in embedding_metrics.items():
                    st.metric(metric.upper(), f"{value:.4f}")
            
            with col2:
                st.subheader("🎯 Reranker Model Metrics")
                reranker_metrics = eval_results.get('reranker_model_results', {})
                for metric, value in reranker_metrics.items():
                    st.metric(metric.upper(), f"{value:.4f}")
            
            # Improvement visualization
            if 'model_comparison' in eval_results and 'improvements' in eval_results['model_comparison']:
                st.subheader("📈 Performance Improvements")
                
                improvements = eval_results['model_comparison']['improvements']
                
                metrics = list(improvements.keys())
                values = list(improvements.values())
                
                fig = px.bar(x=metrics, y=values, 
                           title="Performance Improvements by Reranker (%)",
                           color=values,
                           color_continuous_scale="RdYlGn")
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
                # Metrics table
                df_improvements = pd.DataFrame({
                    'Metric': metrics,
                    'Improvement (%)': [f"{v:+.2f}%" for v in values]
                })
                st.dataframe(df_improvements, use_container_width=True)
            
            # Pipeline results
            if 'pipeline_results' in eval_results and eval_results['pipeline_results']:
                st.subheader("🔄 End-to-End Pipeline Results")
                pipeline_results = eval_results['pipeline_results']
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("NDCG@5", f"{pipeline_results.get('ndcg@5', 0):.4f}")
                with col2:
                    st.metric("NDCG@10", f"{pipeline_results.get('ndcg@10', 0):.4f}")
                with col3:
                    st.metric("MRR", f"{pipeline_results.get('mrr', 0):.4f}")
                with col4:
                    st.metric("MAP", f"{pipeline_results.get('map', 0):.4f}")
        else:
            st.warning("⚠️ No evaluation results found. Run the evaluation script first.")
            
            # Show evaluation commands
            with st.expander("🔧 How to run evaluation"):
                st.code("""
# Run comprehensive evaluation
python scripts/evaluate.py

# Or run individual model training and evaluation
python scripts/train_embedding.py
python scripts/train_reranker.py
python scripts/evaluate.py
                """, language="bash")
    
    with tab4:
        st.header("Training Pipeline")
        
        st.markdown("""
        ### 🚀 Quick Start Training
        
        Follow these steps to train your own embedding and reranker models:
        """)
        
        # Training steps
        with st.expander("1️⃣ Prepare Data", expanded=True):
            st.markdown("""
            ```bash
            python scripts/prepare_data.py --max_samples 5000 --eval_queries 200
            ```
            
            This will:
            - Download/create training dataset
            - Prepare data for embedding and reranker training  
            - Create evaluation dataset
            - Save processed data to `./data/` directory
            """)
            
            if st.button("📊 Show Data Statistics"):
                data_stats_path = "./data/data_statistics.json"
                if os.path.exists(data_stats_path):
                    with open(data_stats_path, 'r') as f:
                        stats = json.load(f)
                    st.json(stats)
                else:
                    st.info("Run data preparation first to see statistics.")
        
        with st.expander("2️⃣ Train Embedding Model"):
            st.markdown("""
            ```bash
            python scripts/train_embedding.py --epochs 3 --batch_size 16 --max_samples 2000
            ```
            
            Configuration options:
            - `--epochs`: Number of training epochs (default: 3)
            - `--batch_size`: Training batch size (default: 16)
            - `--learning_rate`: Learning rate (default: 2e-5)
            - `--max_samples`: Maximum training samples (default: 1000)
            """)
        
        with st.expander("3️⃣ Train Reranker Model"):
            st.markdown("""
            ```bash
            python scripts/train_reranker.py --epochs 3 --batch_size 8 --max_samples 2000
            ```
            
            Configuration options:
            - `--epochs`: Number of training epochs (default: 3)
            - `--batch_size`: Training batch size (default: 8)
            - `--learning_rate`: Learning rate (default: 2e-5)
            - `--max_samples`: Maximum training samples (default: 1000)
            """)
        
        with st.expander("4️⃣ Run Evaluation"):
            st.markdown("""
            ```bash
            python scripts/evaluate.py --num_queries 100
            ```
            
            This will:
            - Evaluate both embedding and reranker models
            - Compare performance improvements
            - Test end-to-end pipeline
            - Save results to `./evaluation_results/`
            """)
        
        # Training status
        st.subheader("📋 Training Status")
        
        # Check if models exist
        embedding_model_exists = os.path.exists("./models/embedding/final_model")
        reranker_model_exists = os.path.exists("./models/reranker/final_model")
        eval_results_exist = os.path.exists("./evaluation_results/comprehensive_evaluation.json")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if embedding_model_exists:
                st.success("✅ Embedding Model Trained")
            else:
                st.error("❌ Embedding Model Not Found")
        
        with col2:
            if reranker_model_exists:
                st.success("✅ Reranker Model Trained")
            else:
                st.error("❌ Reranker Model Not Found")
        
        with col3:
            if eval_results_exist:
                st.success("✅ Evaluation Completed")
            else:
                st.error("❌ Evaluation Not Run")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🔍 Embedding & Reranker Fine-tuning Demo | Built with Streamlit & Hugging Face</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
