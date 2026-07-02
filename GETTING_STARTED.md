# Getting Started Guide

Welcome to the Embedding & Reranker Fine-tuning Demo! This guide will help you get up and running quickly.

## 🚀 Quick Start

### Option 1: One-Command Setup (Recommended)
```bash
# Install dependencies and run the full pipeline
python run_demo.py --mode full --max_samples 1000 --epochs 2
```

This will:
1. Install requirements
2. Prepare training data
3. Train embedding model
4. Train reranker model
5. Run evaluation
6. Launch the Streamlit demo

### Option 2: Step-by-Step Setup

#### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 2. Prepare Data
```bash
python scripts/prepare_data.py --max_samples 2000 --eval_queries 200
```

#### 3. Train Models
```bash
# Train embedding model
python scripts/train_embedding.py --epochs 3 --batch_size 16 --max_samples 1000

# Train reranker model
python scripts/train_reranker.py --epochs 3 --batch_size 8 --max_samples 1000
```

#### 4. Evaluate Models
```bash
python scripts/evaluate.py --num_queries 100
```

#### 5. Launch Demo
```bash
streamlit run demo/app.py
```

## 🎯 Demo Only (Skip Training)

If you want to see the demo with base models (no fine-tuning):

```bash
# Install requirements
pip install -r requirements.txt

# Launch demo directly
python run_demo.py --mode demo
```

## 📊 What You'll See

### 1. Search Demo Tab
- Interactive search interface
- Real-time comparison of embedding vs reranker results
- Performance metrics and timing information
- Score visualizations

### 2. Model Comparison Tab
- Side-by-side result comparison
- Performance timing breakdown
- Score distribution analysis

### 3. Evaluation Tab
- Comprehensive model metrics (NDCG, MAP, MRR, etc.)
- Performance improvement visualizations
- End-to-end pipeline evaluation

### 4. Training Tab
- Training status and configuration
- Step-by-step training guide
- Model availability checks

## 🔧 Configuration

### Training Parameters
Edit `configs/embedding_config.yaml` and `configs/reranker_config.yaml`:

```yaml
# Example configuration
epochs: 3
batch_size: 16
learning_rate: 2e-5
max_samples: 2000
```

### Model Selection
Change base models in the configuration files:

```yaml
# For embeddings
model_name: "sentence-transformers/all-MiniLM-L6-v2"

# For rerankers  
model_name: "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

## 📁 Project Structure

```
├── src/                    # Core source code
│   ├── data/              # Data loading utilities
│   ├── models/            # Model implementations
│   ├── evaluation/        # Evaluation metrics
│   └── inference/         # Inference pipeline
├── scripts/               # Training and evaluation scripts
├── demo/                  # Streamlit demo application
├── configs/               # Configuration files
├── notebooks/             # Jupyter analysis notebooks
├── data/                  # Generated datasets
├── models/                # Saved model checkpoints
└── evaluation_results/    # Evaluation outputs
```

## 🎛️ Command Line Options

### run_demo.py Options
```bash
python run_demo.py --help

Options:
  --mode {full,demo,train,eval}  Run mode (default: demo)
  --max_samples INT              Training samples (default: 1000)
  --epochs INT                   Training epochs (default: 2)
  --skip_data_prep              Skip data preparation
```

### Training Script Options
```bash
python scripts/train_embedding.py --help
python scripts/train_reranker.py --help
python scripts/evaluate.py --help
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `batch_size` in config files
   - Reduce `max_samples` parameter
   - Use CPU by setting `device: cpu` in configs

2. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Streamlit Won't Start**
   ```bash
   # Try different port
   streamlit run demo/app.py --server.port 8502
   ```

4. **Model Loading Errors**
   - Check if models exist in `./models/` directory
   - Demo will fallback to base models if fine-tuned models aren't found

### Performance Tips

1. **Faster Training**
   - Use GPU if available
   - Reduce `max_samples` for quick testing
   - Use smaller batch sizes if memory constrained

2. **Better Results**
   - Increase `epochs` and `max_samples`
   - Use domain-specific data
   - Tune learning rates

3. **Production Deployment**
   - Use FAISS for large-scale retrieval
   - Cache embeddings for repeated queries
   - Consider model quantization

## 📚 Next Steps

1. **Customize Data**: Replace the sample dataset with your own domain-specific data
2. **Experiment**: Try different model architectures and hyperparameters
3. **Scale Up**: Use larger datasets and more training epochs
4. **Deploy**: Integrate the pipeline into your application

## 🤝 Contributing

Feel free to:
- Add new evaluation metrics
- Implement additional model architectures
- Improve the demo interface
- Add more datasets

## 📖 Learn More

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [FAISS for Similarity Search](https://faiss.ai/)
- [Streamlit Documentation](https://docs.streamlit.io/)

Happy fine-tuning! 🎉
