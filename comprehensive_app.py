"""
Comprehensive Streamlit UI for Embedding & Reranker Fine-tuning Pipeline
Full control over fine-tuning, testing, evaluation, and inference
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
import subprocess
import threading
from datetime import datetime
import numpy as np
from pathlib import Path

# Import our modules
from src.inference.pipeline import SearchPipeline, create_demo_corpus
from src.data.dataset_loader import DatasetLoader
from src.evaluation.metrics import RankingMetrics

# Page configuration
st.set_page_config(
    page_title="🔍 Embedding & Reranker Fine-tuning Control Center",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
    margin-bottom: 1rem;
    color: #1f77b4;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    /* Fallback for systems that don't support background-clip */
    background-clip: text;
}

/* Alternative header style for better visibility */
.main-header-alt {
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
    margin-bottom: 1rem;
    color: #2E86AB;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    border-bottom: 3px solid #4ECDC4;
    padding-bottom: 10px;
}

.control-panel {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    border: 2px solid #dee2e6;
    margin: 0.5rem 0;
}

.status-success {
    background-color: #d4edda;
    border-color: #c3e6cb;
    color: #155724;
    padding: 0.5rem;
    border-radius: 0.25rem;
    margin: 0.25rem 0;
}

.status-warning {
    background-color: #fff3cd;
    border-color: #ffeaa7;
    color: #856404;
    padding: 0.5rem;
    border-radius: 0.25rem;
    margin: 0.25rem 0;
}

.status-error {
    background-color: #f8d7da;
    border-color: #f5c6cb;
    color: #721c24;
    padding: 0.5rem;
    border-radius: 0.25rem;
    margin: 0.25rem 0;
}

.metric-card {
    background-color: #ffffff;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #4ECDC4;
    margin: 0.5rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.tab-content {
    padding: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'training_status' not in st.session_state:
    st.session_state.training_status = {
        'embedding': 'Not Started',
        'reranker': 'Not Started',
        'evaluation': 'Not Started'
    }

if 'logs' not in st.session_state:
    st.session_state.logs = []

if 'current_pipeline' not in st.session_state:
    st.session_state.current_pipeline = None

def add_log(message, level="INFO"):
    """Add log message to session state"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append({
        'timestamp': timestamp,
        'level': level,
        'message': message
    })
    # Keep only last 100 logs
    if len(st.session_state.logs) > 100:
        st.session_state.logs = st.session_state.logs[-100:]

def check_system_status():
    """Check current system status"""
    status = {}
    
    # Check directories
    status['data_dir'] = os.path.exists('./data')
    status['models_dir'] = os.path.exists('./models')
    status['eval_dir'] = os.path.exists('./evaluation_results')
    
    # Check data files
    status['training_data'] = os.path.exists('./data/embedding_training_data.json')
    status['eval_data'] = os.path.exists('./data/evaluation_data.json')
    
    # Check models
    status['embedding_model'] = os.path.exists('./models/embedding/final_model')
    status['reranker_model'] = os.path.exists('./models/reranker/final_model')
    
    # Check evaluation results
    status['eval_results'] = os.path.exists('./evaluation_results/comprehensive_evaluation.json')
    
    return status

def run_command_with_logging(command, description):
    """Run command and log output"""
    add_log(f"Starting: {description}")
    add_log(f"Command: {command}")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            add_log(f"✅ {description} completed successfully", "SUCCESS")
            if result.stdout:
                add_log(f"Output: {result.stdout[:500]}...")
            return True, result.stdout
        else:
            add_log(f"❌ {description} failed", "ERROR")
            if result.stderr:
                add_log(f"Error: {result.stderr}")
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        add_log(f"⏰ {description} timed out after 5 minutes", "WARNING")
        return False, "Timeout"
    except Exception as e:
        add_log(f"❌ {description} failed with exception: {str(e)}", "ERROR")
        return False, str(e)

def generate_embeddings_for_texts(texts):
    """Generate embeddings for the given texts"""
    try:
        # Import here to avoid issues
        from src.models.embedding_model import EmbeddingModelInference
        
        with st.spinner("🔄 Loading model and generating embeddings..."):
            # Load model
            model = EmbeddingModelInference("./models/embedding/final_model")
            
            # Generate embeddings
            embeddings = model.encode(texts)
            
            # Display results
            st.success(f"✅ Generated embeddings for {len(texts)} texts!")
            
            # Show embedding details
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Number of Texts", len(texts))
            with col2:
                st.metric("Embedding Dimension", embeddings.shape[1])
            with col3:
                st.metric("Total Values", embeddings.size)
            
            # Show embeddings
            st.subheader("🔢 Generated Embeddings")
            
            # Create a DataFrame for better display
            df = pd.DataFrame(embeddings)
            # Set proper column names for embedding dimensions
            df.columns = [f'Dim_{i}' for i in range(embeddings.shape[1])]
            df.insert(0, 'Text', texts)
            df.insert(1, 'Text_Length', [len(text) for text in texts])
            
            # Show first few dimensions
            display_cols = ['Text', 'Text_Length'] + [f'Dim_{i}' for i in range(min(10, embeddings.shape[1]))]
            # Filter columns that actually exist in the DataFrame
            existing_cols = [col for col in display_cols if col in df.columns]
            st.dataframe(df[existing_cols], use_container_width=True)
            
            # Show full embeddings in expandable section
            with st.expander("📊 View All Embedding Dimensions"):
                st.dataframe(df, use_container_width=True)
            
            # Similarity matrix
            if len(texts) > 1:
                st.subheader("🔍 Text Similarity Matrix")
                similarity_matrix = []
                for i, text1 in enumerate(texts):
                    row = []
                    for j, text2 in enumerate(texts):
                        if i == j:
                            similarity = 1.0
                        else:
                            similarity = model.compute_similarity(text1, text2)
                        row.append(similarity)
                    similarity_matrix.append(row)
                
                # Create similarity heatmap
                similarity_df = pd.DataFrame(
                    similarity_matrix,
                    index=[f"Text {i+1}" for i in range(len(texts))],
                    columns=[f"Text {i+1}" for i in range(len(texts))]
                )
                
                fig = px.imshow(
                    similarity_df,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="viridis",
                    title="Text Similarity Heatmap"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            # Download options
            st.subheader("💾 Download Results")
            
            # Save as JSON
            embeddings_data = {
                "texts": texts,
                "embeddings": embeddings.tolist(),
                "shape": embeddings.shape,
                "model_path": "./models/embedding/final_model",
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            json_str = json.dumps(embeddings_data, indent=2)
            st.download_button(
                label="📥 Download as JSON",
                data=json_str,
                file_name=f"embeddings_{len(texts)}_texts.json",
                mime="application/json"
            )
            
            # Save as CSV
            csv_str = df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv_str,
                file_name=f"embeddings_{len(texts)}_texts.csv",
                mime="text/csv"
            )
            
    except Exception as e:
        st.error(f"❌ Error generating embeddings: {str(e)}")
        st.error("Make sure your embedding model is properly trained and available.")

def main():
    # Header
    st.markdown('<h1 class="main-header-alt">🎛️ Embedding & Reranker Fine-tuning Control Center</h1>', 
                unsafe_allow_html=True)
    
    # System Status Check
    with st.sidebar:
        st.header("🔍 System Status")
        status = check_system_status()
        
        # Status indicators
        if status['data_dir']:
            st.success("📁 Data Directory")
        else:
            st.error("📁 Data Directory")
            
        if status['training_data']:
            st.success("📊 Training Data")
        else:
            st.error("📊 Training Data")
            
        if status['embedding_model']:
            st.success("🤖 Embedding Model")
        else:
            st.warning("🤖 Embedding Model (Base)")
            
        if status['reranker_model']:
            st.success("🎯 Reranker Model")
        else:
            st.warning("🎯 Reranker Model (Base)")
            
        if status['eval_results']:
            st.success("📈 Evaluation Results")
        else:
            st.info("📈 Evaluation Results (Not Available)")
        
        st.divider()
        
        # Quick Actions
        st.header("⚡ Quick Actions")
        if st.button("🔄 Refresh Status"):
            st.rerun()
        
        if st.button("🧹 Clear Logs"):
            st.session_state.logs = []
            st.rerun()
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🏗️ Data Preparation", 
        "🚀 Model Training", 
        "📊 Evaluation", 
        "🎯 Generate Embeddings",
        "🔍 Inference Demo", 
        "📋 Model Details",
        "📈 Analytics", 
        "📋 Logs"
    ])
    
    # Tab 1: Data Preparation
    with tab1:
        st.header("🏗️ Data Preparation & Management")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 Dataset Configuration")
            
            dataset_type = st.selectbox(
                "Dataset Type",
                ["sample", "ms_marco"],
                help="Choose dataset type for training"
            )
            
            max_samples = st.slider(
                "Maximum Samples",
                min_value=50,
                max_value=10000,
                value=500,
                step=50,
                help="Number of samples to use for training"
            )
            
            eval_queries = st.slider(
                "Evaluation Queries",
                min_value=10,
                max_value=500,
                value=100,
                step=10,
                help="Number of queries for evaluation dataset"
            )
            
            if st.button("🔄 Prepare Dataset", type="primary"):
                with st.spinner("Preparing dataset..."):
                    command = f"python scripts/prepare_data.py --dataset {dataset_type} --max_samples {max_samples} --eval_queries {eval_queries}"
                    success, output = run_command_with_logging(command, "Dataset Preparation")
                    
                    if success:
                        st.success("Dataset prepared successfully!")
                        st.rerun()
                    else:
                        st.error("Dataset preparation failed!")
        
        with col2:
            st.subheader("📁 Current Data Status")
            
            if status['training_data']:
                try:
                    with open('./data/data_statistics.json', 'r') as f:
                        stats = json.load(f)
                    
                    st.metric("Total Samples", stats.get('total_samples', 0))
                    st.metric("Embedding Examples", stats.get('embedding_examples', 0))
                    st.metric("Reranker Examples", stats.get('reranker_examples', 0))
                    st.metric("Evaluation Queries", stats.get('evaluation_queries', 0))
                    
                except Exception as e:
                    st.error(f"Error loading statistics: {e}")
            else:
                st.info("No training data available")
        
        # Data Preview
        if status['training_data']:
            with st.expander("👀 Preview Training Data"):
                try:
                    with open('./data/embedding_training_data.json', 'r') as f:
                        data = json.load(f)
                    
                    df = pd.DataFrame(data[:10])  # Show first 10 examples
                    st.dataframe(df, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error loading data preview: {e}")
    
    # Tab 2: Model Training
    with tab2:
        st.header("🚀 Model Training Control Center")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🤖 Embedding Model Training")
            
            embedding_epochs = st.slider("Epochs", 1, 10, 3, key="emb_epochs")
            embedding_batch_size = st.selectbox("Batch Size", [8, 16, 32, 64], index=1, key="emb_batch")
            embedding_lr = st.selectbox("Learning Rate", [1e-5, 2e-5, 5e-5, 1e-4], index=1, key="emb_lr")
            
            if st.button("🚀 Train Embedding Model", key="train_emb"):
                with st.spinner("Training embedding model..."):
                    command = f"python scripts/train_embedding.py --epochs {embedding_epochs} --batch_size {embedding_batch_size} --learning_rate {embedding_lr}"
                    success, output = run_command_with_logging(command, "Embedding Model Training")
                    
                    if success:
                        st.session_state.training_status['embedding'] = 'Completed'
                        st.success("Embedding model training completed!")
                    else:
                        st.session_state.training_status['embedding'] = 'Failed'
                        st.error("Embedding model training failed!")
        
        with col2:
            st.subheader("🎯 Reranker Model Training")
            
            reranker_epochs = st.slider("Epochs", 1, 10, 3, key="rer_epochs")
            reranker_batch_size = st.selectbox("Batch Size", [4, 8, 16, 32], index=1, key="rer_batch")
            reranker_lr = st.selectbox("Learning Rate", [1e-5, 2e-5, 5e-5, 1e-4], index=1, key="rer_lr")
            
            if st.button("🚀 Train Reranker Model", key="train_rer"):
                with st.spinner("Training reranker model..."):
                    command = f"python scripts/train_reranker.py --epochs {reranker_epochs} --batch_size {reranker_batch_size} --learning_rate {reranker_lr}"
                    success, output = run_command_with_logging(command, "Reranker Model Training")
                    
                    if success:
                        st.session_state.training_status['reranker'] = 'Completed'
                        st.success("Reranker model training completed!")
                    else:
                        st.session_state.training_status['reranker'] = 'Failed'
                        st.error("Reranker model training failed!")
        
        # Training Status
        st.subheader("📋 Training Status")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status_color = {
                'Not Started': 'gray',
                'In Progress': 'orange',
                'Completed': 'green',
                'Failed': 'red'
            }
            st.metric("Embedding Model", st.session_state.training_status['embedding'])
        
        with col2:
            st.metric("Reranker Model", st.session_state.training_status['reranker'])
        
        with col3:
            st.metric("Evaluation", st.session_state.training_status['evaluation'])
        
        # Batch Training
        st.subheader("⚡ Batch Training Pipeline")
        if st.button("🚀 Run Complete Training Pipeline", type="primary"):
            with st.spinner("Running complete training pipeline..."):
                # Update status
                st.session_state.training_status['embedding'] = 'In Progress'
                st.session_state.training_status['reranker'] = 'In Progress'
                
                # Run training commands
                commands = [
                    ("python scripts/train_embedding.py --epochs 3 --batch_size 16", "Embedding Training"),
                    ("python scripts/train_reranker.py --epochs 3 --batch_size 8", "Reranker Training")
                ]
                
                all_success = True
                for cmd, desc in commands:
                    success, output = run_command_with_logging(cmd, desc)
                    if not success:
                        all_success = False
                
                if all_success:
                    st.session_state.training_status['embedding'] = 'Completed'
                    st.session_state.training_status['reranker'] = 'Completed'
                    st.success("Complete training pipeline finished!")
                else:
                    st.error("Some training steps failed!")
    
    # Tab 3: Evaluation
    with tab3:
        st.header("📊 Model Evaluation & Benchmarking")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🔍 Evaluation Configuration")
            
            num_queries = st.slider("Number of Evaluation Queries", 10, 200, 50)
            use_finetuned = st.checkbox("Use Fine-tuned Models (if available)", value=True)
            
            if st.button("📊 Run Evaluation", type="primary"):
                with st.spinner("Running evaluation..."):
                    command = f"python scripts/evaluate.py --num_queries {num_queries}"
                    success, output = run_command_with_logging(command, "Model Evaluation")
                    
                    if success:
                        st.session_state.training_status['evaluation'] = 'Completed'
                        st.success("Evaluation completed successfully!")
                        st.rerun()
                    else:
                        st.error("Evaluation failed!")
        
        with col2:
            st.subheader("📈 Current Results")
            
            if status['eval_results']:
                try:
                    with open('./evaluation_results/comprehensive_evaluation.json', 'r') as f:
                        results = json.load(f)
                    
                    # Show key metrics
                    if 'embedding_model_results' in results:
                        emb_ndcg = results['embedding_model_results'].get('ndcg@10', 0)
                        st.metric("Embedding NDCG@10", f"{emb_ndcg:.4f}")
                    
                    if 'reranker_model_results' in results:
                        rer_ndcg = results['reranker_model_results'].get('ndcg@10', 0)
                        st.metric("Reranker NDCG@10", f"{rer_ndcg:.4f}")
                    
                    if 'model_comparison' in results and 'improvements' in results['model_comparison']:
                        improvements = results['model_comparison']['improvements']
                        if 'ndcg@10' in improvements:
                            improvement = improvements['ndcg@10']
                            st.metric("NDCG@10 Improvement", f"{improvement:+.2f}%")
                            
                except Exception as e:
                    st.error(f"Error loading results: {e}")
            else:
                st.info("No evaluation results available")
        
        # Results Visualization
        if status['eval_results']:
            with st.expander("📊 Detailed Results Visualization"):
                try:
                    with open('./evaluation_results/comprehensive_evaluation.json', 'r') as f:
                        results = json.load(f)
                    
                    # Create comparison chart
                    if 'embedding_model_results' in results and 'reranker_model_results' in results:
                        emb_metrics = results['embedding_model_results']
                        rer_metrics = results['reranker_model_results']
                        
                        # Filter common metrics
                        common_metrics = [k for k in emb_metrics.keys() if k in rer_metrics]
                        
                        if common_metrics:
                            comparison_data = {
                                'Metric': common_metrics,
                                'Embedding Model': [emb_metrics[m] for m in common_metrics],
                                'Reranker Model': [rer_metrics[m] for m in common_metrics]
                            }
                            
                            df = pd.DataFrame(comparison_data)
                            
                            fig = px.bar(df, x='Metric', y=['Embedding Model', 'Reranker Model'],
                                       title="Model Performance Comparison", barmode='group')
                            st.plotly_chart(fig, use_container_width=True)
                            
                except Exception as e:
                                                st.error(f"Error creating visualization: {e}")
    
    # Tab 4: Generate Embeddings
    with tab4:
        st.header("🎯 Generate Embeddings")
        st.markdown("Generate embeddings for any text using your fine-tuned models!")
        
        # Check if models are available
        if not status['embedding_model']:
            st.error("❌ Embedding model not found! Please train your embedding model first.")
        else:
            # Input section
            st.subheader("📝 Input Your Text")
            
            input_method = st.selectbox(
                "Choose input method:",
                ["Single Text", "Multiple Texts", "Upload File", "Sample Examples"],
                key="embedding_input_method"
            )
            
            if input_method == "Single Text":
                text_input = st.text_area(
                    "Enter your text:",
                    placeholder="Type or paste your text here...",
                    height=100,
                    key="single_text"
                )
                texts = [text_input] if text_input.strip() else []
                
            elif input_method == "Multiple Texts":
                text_input = st.text_area(
                    "Enter multiple texts (one per line):",
                    placeholder="Text 1\nText 2\nText 3\n...",
                    height=150,
                    key="multiple_texts"
                )
                texts = [line.strip() for line in text_input.split('\n') if line.strip()]
                
            elif input_method == "Upload File":
                uploaded_file = st.file_uploader(
                    "Upload a text file:",
                    type=['txt', 'csv'],
                    help="Upload a .txt file with one text per line, or a .csv file with a 'text' column",
                    key="embedding_file"
                )
                
                if uploaded_file is not None:
                    if uploaded_file.name.endswith('.txt'):
                        content = uploaded_file.read().decode('utf-8')
                        texts = [line.strip() for line in content.split('\n') if line.strip()]
                    elif uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                        if 'text' in df.columns:
                            texts = df['text'].dropna().tolist()
                        else:
                            st.error("CSV file must have a 'text' column")
                            texts = []
                else:
                    texts = []
                    
            else:  # Sample Examples
                sample_texts = [
                    "What is artificial intelligence?",
                    "How do machine learning algorithms work?",
                    "Explain deep learning neural networks",
                    "What is natural language processing?",
                    "Describe computer vision applications"
                ]
                texts = sample_texts
                st.info("Using sample texts for demonstration")
            
            # Display input texts
            if texts:
                st.subheader("📋 Input Texts")
                for i, text in enumerate(texts):
                    st.write(f"**{i+1}.** {text}")
                
                # Generate embeddings button
                if st.button("🚀 Generate Embeddings", type="primary", key="generate_embeddings"):
                    if texts:
                        generate_embeddings_for_texts(texts)
                    else:
                        st.error("Please enter some text first!")
            else:
                st.info("Enter some text above to generate embeddings")
    
    # Tab 5: Inference Demo
    with tab4:
        st.header("🔍 Interactive Inference Demo")
        
        # Initialize pipeline
        if st.session_state.current_pipeline is None:
            try:
                corpus = create_demo_corpus()
                pipeline = SearchPipeline(
                    embedding_model_path="./models/embedding/final_model",
                    reranker_model_path="./models/reranker/final_model",
                    corpus=corpus
                )
                st.session_state.current_pipeline = pipeline
                st.success("✅ Pipeline initialized successfully!")
            except Exception as e:
                st.warning(f"⚠️ Using base models: {e}")
                try:
                    corpus = create_demo_corpus()
                    pipeline = SearchPipeline(
                        embedding_model_path="sentence-transformers/all-MiniLM-L6-v2",
                        reranker_model_path="cross-encoder/ms-marco-MiniLM-L-6-v2",
                        corpus=corpus
                    )
                    st.session_state.current_pipeline = pipeline
                    st.success("✅ Pipeline initialized with base models!")
                except Exception as e2:
                    st.error(f"❌ Failed to initialize pipeline: {e2}")
                    st.stop()
        
        # Search Interface
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_input(
                "🔍 Enter your search query:",
                placeholder="e.g., What is machine learning?",
                key="search_query"
            )
        
        with col2:
            search_button = st.button("🔍 Search", type="primary")
        
        # Search Parameters
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            top_k = st.slider("Top K Results", 1, 20, 10)
        
        with col2:
            retrieval_k = st.slider("Retrieval K", 10, 100, 50)
        
        with col3:
            use_reranker = st.checkbox("Use Reranker", value=True)
        
        with col4:
            show_timing = st.checkbox("Show Timing", value=True)
        
        # Perform Search
        if query and (search_button or query):
            if st.session_state.current_pipeline:
                with st.spinner("Searching..."):
                    try:
                        # Search with and without reranker
                        results_with_reranker = st.session_state.current_pipeline.search(
                            query, top_k=top_k, retrieval_k=retrieval_k, use_reranker=True
                        )
                        
                        results_without_reranker = st.session_state.current_pipeline.search(
                            query, top_k=top_k, retrieval_k=retrieval_k, use_reranker=False
                        )
                        
                        # Display Results
                        if use_reranker:
                            current_results = results_with_reranker
                            st.subheader("🎯 Results with Reranker")
                        else:
                            current_results = results_without_reranker
                            st.subheader("📋 Results without Reranker")
                        
                        # Performance Metrics
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
                        
                        # Results Display
                        for i, result in enumerate(current_results['results']):
                            with st.container():
                                col1, col2 = st.columns([4, 1])
                                
                                with col1:
                                    st.markdown(f"**Rank {result['rank']}**")
                                    st.markdown(result['document'])
                                
                                with col2:
                                    score = result['score']
                                    st.metric("Score", f"{score:.4f}")
                                    
                                    # Score gauge
                                    fig = go.Figure(go.Indicator(
                                        mode="gauge+number",
                                        value=score,
                                        domain={'x': [0, 1], 'y': [0, 1]},
                                        title={'text': "Relevance"},
                                        gauge={
                                            'axis': {'range': [None, 1]},
                                            'bar': {'color': "#4ECDC4"},
                                            'steps': [
                                                {'range': [0, 0.3], 'color': "#FFE5E5"},
                                                {'range': [0.3, 0.7], 'color': "#FFF8DC"},
                                                {'range': [0.7, 1], 'color': "#E5F5F5"}
                                            ]
                                        }
                                    ))
                                    fig.update_layout(height=150, margin=dict(l=20, r=20, t=30, b=20))
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                st.divider()
                        
                        # Model Comparison
                        if use_reranker:
                            with st.expander("📊 Model Comparison"):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.subheader("📋 Without Reranker")
                                    for result in results_without_reranker['results'][:5]:
                                        st.markdown(f"**Rank {result['rank']} (Score: {result['score']:.4f})**")
                                        st.markdown(result['document'][:100] + "...")
                                        st.divider()
                                
                                with col2:
                                    st.subheader("🎯 With Reranker")
                                    for result in results_with_reranker['results'][:5]:
                                        st.markdown(f"**Rank {result['rank']} (Score: {result['score']:.4f})**")
                                        st.markdown(result['document'][:100] + "...")
                                        st.divider()
                        
                    except Exception as e:
                        st.error(f"Search failed: {e}")
            else:
                st.error("Pipeline not initialized!")
    
    # Tab 6: Model Details
    with tab6:
        st.header("📋 Model Details & Information")
        
        # Model overview
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🤖 Embedding Model")
            
            if status['embedding_model']:
                st.success("✅ Fine-tuned model available")
                
                # Get model info
                model_info = {
                    "base_model": "sentence-transformers/all-MiniLM-L6-v2",
                    "architecture": "Transformer-based (DistilBERT)",
                    "embedding_dimension": 384,
                    "max_sequence_length": 256,
                    "vocabulary_size": 30523,
                    "model_size": "~87MB",
                    "training_data": "Custom fine-tuned dataset",
                    "use_case": "Text embeddings and similarity search",
                    "features": [
                        "Contrastive learning with positive/negative pairs",
                        "Semantic similarity computation",
                        "Batch processing support",
                        "Cosine similarity scoring"
                    ]
                }
                
                st.markdown(f"""
                **Base Model:** `{model_info['base_model']}`  
                **Architecture:** {model_info['architecture']}  
                **Embedding Dimension:** {model_info['embedding_dimension']}  
                **Max Sequence Length:** {model_info['max_sequence_length']}  
                **Vocabulary Size:** {model_info['vocabulary_size']:,}  
                **Model Size:** {model_info['model_size']}  
                **Training Data:** {model_info['training_data']}  
                **Use Case:** {model_info['use_case']}
                """)
                
                st.subheader("✨ Features")
                for feature in model_info['features']:
                    st.markdown(f"• {feature}")
            else:
                st.warning("⚠️ Using base model")
                st.info("Train your embedding model to see detailed information")
        
        with col2:
            st.subheader("🎯 Reranker Model")
            
            if status['reranker_model']:
                st.success("✅ Fine-tuned model available")
                
                # Get model info
                model_info = {
                    "base_model": "microsoft/MiniLM-L12-H384-uncased",
                    "architecture": "Transformer-based (MiniLM)",
                    "hidden_size": 384,
                    "num_layers": 12,
                    "num_attention_heads": 12,
                    "max_sequence_length": 512,
                    "vocabulary_size": 30523,
                    "model_size": "~120MB",
                    "training_data": "Custom fine-tuned dataset",
                    "use_case": "Query-passage relevance scoring",
                    "features": [
                        "Cross-encoder architecture",
                        "Relevance scoring (0-1)",
                        "Batch prediction support",
                        "Fine-tuned for retrieval tasks"
                    ]
                }
                
                st.markdown(f"""
                **Base Model:** `{model_info['base_model']}`  
                **Architecture:** {model_info['architecture']}  
                **Hidden Size:** {model_info['hidden_size']}  
                **Number of Layers:** {model_info['num_layers']}  
                **Attention Heads:** {model_info['num_attention_heads']}  
                **Max Sequence Length:** {model_info['max_sequence_length']}  
                **Vocabulary Size:** {model_info['vocabulary_size']:,}  
                **Model Size:** {model_info['model_size']}  
                **Training Data:** {model_info['training_data']}  
                **Use Case:** {model_info['use_case']}
                """)
                
                st.subheader("✨ Features")
                for feature in model_info['features']:
                    st.markdown(f"• {feature}")
            else:
                st.warning("⚠️ Using base model")
                st.info("Train your reranker model to see detailed information")
        
        # Model summary
        st.subheader("📊 Model Summary")
        
        summary_data = {
            "Model Type": ["Embedding", "Reranker"],
            "Base Model": [
                "sentence-transformers/all-MiniLM-L6-v2",
                "microsoft/MiniLM-L12-H384-uncased"
            ],
            "Architecture": [
                "Transformer-based (DistilBERT)",
                "Transformer-based (MiniLM)"
            ],
            "Output": [
                "384D vectors",
                "Relevance scores (0-1)"
            ],
            "Training Approach": [
                "Contrastive Learning",
                "Cross-encoder Fine-tuning"
            ],
            "Use Case": [
                "Text embeddings and similarity",
                "Query-passage relevance scoring"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        # Model comparison chart
        st.subheader("📈 Model Comparison")
        
        # Create comparison metrics
        comparison_data = {
            "Model": ["Embedding", "Reranker"],
            "Model Size (MB)": [87, 120],
            "Sequence Length": [256, 512],
            "Vocabulary Size": [30523, 30523]
        }
        
        comp_df = pd.DataFrame(comparison_data)
        
        # Model size comparison
        fig1 = px.bar(
            comp_df, 
            x="Model", 
            y="Model Size (MB)",
            title="Model Size Comparison",
            color="Model",
            color_discrete_sequence=["#1f77b4", "#ff7f0e"]
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Sequence length comparison
        fig2 = px.bar(
            comp_df, 
            x="Model", 
            y="Sequence Length",
            title="Maximum Sequence Length",
            color="Model",
            color_discrete_sequence=["#2ca02c", "#d62728"]
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Technical details
        st.subheader("🔧 Technical Details")
        
        tech_details = {
            "Framework": "PyTorch + Transformers",
            "Training Method": "Fine-tuning with custom dataset",
            "Loss Function": "Contrastive Loss (Embedding), Cross-Entropy (Reranker)",
            "Optimizer": "AdamW",
            "Learning Rate": "2e-5",
            "Batch Size": "16 (Embedding), 8 (Reranker)",
            "Training Epochs": "2-3",
            "Hardware": "CPU/GPU compatible"
        }
        
        for key, value in tech_details.items():
            st.markdown(f"**{key}:** {value}")
    
    # Tab 7: Analytics
    with tab7:
        st.header("📈 Analytics & Insights")
        
        # System Overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📊 Data Statistics")
            if status['training_data']:
                try:
                    with open('./data/data_statistics.json', 'r') as f:
                        stats = json.load(f)
                    
                    st.metric("Total Samples", stats.get('total_samples', 0))
                    st.metric("Training Examples", stats.get('embedding_examples', 0))
                    st.metric("Evaluation Queries", stats.get('evaluation_queries', 0))
                    
                except Exception as e:
                    st.error(f"Error loading stats: {e}")
            else:
                st.info("No data available")
        
        with col2:
            st.subheader("🤖 Model Status")
            st.metric("Embedding Model", "✅ Ready" if status['embedding_model'] else "⚠️ Base Model")
            st.metric("Reranker Model", "✅ Ready" if status['reranker_model'] else "⚠️ Base Model")
            st.metric("Pipeline Status", "✅ Active" if st.session_state.current_pipeline else "❌ Inactive")
        
        with col3:
            st.subheader("📈 Performance")
            if status['eval_results']:
                try:
                    with open('./evaluation_results/comprehensive_evaluation.json', 'r') as f:
                        results = json.load(f)
                    
                    if 'embedding_model_results' in results:
                        emb_ndcg = results['embedding_model_results'].get('ndcg@10', 0)
                        st.metric("Best NDCG@10", f"{emb_ndcg:.4f}")
                    
                    if 'model_comparison' in results and 'improvements' in results['model_comparison']:
                        improvements = results['model_comparison']['improvements']
                        if improvements:
                            best_improvement = max(improvements.values())
                            st.metric("Best Improvement", f"{best_improvement:+.2f}%")
                    
                except Exception as e:
                    st.error(f"Error loading performance: {e}")
            else:
                st.info("No performance data")
        
        # Training History
        st.subheader("📚 Training History")
        
        # Check for training logs
        training_logs = []
        if os.path.exists('./models/embedding/training_info.json'):
            try:
                with open('./models/embedding/training_info.json', 'r') as f:
                    emb_info = json.load(f)
                    training_logs.append({
                        'Model': 'Embedding',
                        'Status': 'Completed',
                        'Epochs': emb_info.get('epochs', 0),
                        'Samples': emb_info.get('num_train_examples', 0),
                        'Timestamp': emb_info.get('timestamp', 'Unknown')
                    })
            except:
                pass
        
        if os.path.exists('./models/reranker/training_info.json'):
            try:
                with open('./models/reranker/training_info.json', 'r') as f:
                    rer_info = json.load(f)
                    training_logs.append({
                        'Model': 'Reranker',
                        'Status': 'Completed',
                        'Epochs': rer_info.get('epochs', 0),
                        'Samples': rer_info.get('num_train_examples', 0),
                        'Timestamp': rer_info.get('timestamp', 'Unknown')
                    })
            except:
                pass
        
        if training_logs:
            df_logs = pd.DataFrame(training_logs)
            st.dataframe(df_logs, use_container_width=True)
        else:
            st.info("No training history available")
    
    # Tab 8: Logs
    with tab8:
        st.header("📋 System Logs & Monitoring")
        
        # Log Controls
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            log_level_filter = st.selectbox(
                "Filter by Level",
                ["ALL", "INFO", "SUCCESS", "WARNING", "ERROR"],
                key="log_filter"
            )
        
        with col2:
            if st.button("🔄 Refresh Logs"):
                st.rerun()
        
        with col3:
            if st.button("📥 Export Logs"):
                if st.session_state.logs:
                    df_logs = pd.DataFrame(st.session_state.logs)
                    csv = df_logs.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"system_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        
        # Display Logs
        st.subheader("📝 Recent Logs")
        
        if st.session_state.logs:
            # Filter logs
            if log_level_filter != "ALL":
                filtered_logs = [log for log in st.session_state.logs if log['level'] == log_level_filter]
            else:
                filtered_logs = st.session_state.logs
            
            # Display logs in reverse chronological order
            for log in reversed(filtered_logs[-50:]):  # Show last 50 logs
                level_colors = {
                    'INFO': 'blue',
                    'SUCCESS': 'green',
                    'WARNING': 'orange',
                    'ERROR': 'red'
                }
                
                color = level_colors.get(log['level'], 'gray')
                st.markdown(
                    f"<span style='color: {color}; font-weight: bold;'>[{log['timestamp']}] {log['level']}:</span> {log['message']}",
                    unsafe_allow_html=True
                )
        else:
            st.info("No logs available")
        
        # System Information
        st.subheader("💻 System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Python Environment**")
            st.code(f"Python Version: {sys.version}")
            st.code(f"Working Directory: {os.getcwd()}")
        
        with col2:
            st.markdown("**Project Status**")
            st.code(f"Data Directory: {'✅' if status['data_dir'] else '❌'}")
            st.code(f"Models Directory: {'✅' if status['models_dir'] else '❌'}")
            st.code(f"Evaluation Results: {'✅' if status['eval_results'] else '❌'}")

if __name__ == "__main__":
    main()
