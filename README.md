# Embedding and Reranker Fine-tuning Demo

A comprehensive pipeline for fine-tuning embedding models and rerankers with real-time evaluation and demo interface.

## Features

- 🔧 Fine-tuning pipeline for embedding models (sentence-transformers)
- 🎯 Reranker model fine-tuning with cross-encoder architecture
- 📊 Comprehensive evaluation metrics (NDCG, MAP, MRR, Recall@K)
- 🚀 Real-time demo interface with Streamlit
- 📈 Training monitoring with Weights & Biases
- 🔍 Interactive search and ranking comparison

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your dataset (or use the provided sample):
```bash
python scripts/prepare_data.py
```

3. Fine-tune embedding model:
```bash
python scripts/train_embedding.py --config configs/embedding_config.yaml
```

4. Fine-tune reranker:
```bash
python scripts/train_reranker.py --config configs/reranker_config.yaml
```

5. Run evaluation:
```bash
python scripts/evaluate.py
```

6. Launch demo:
```bash
streamlit run demo/app.py
```

## Project Structure

```
├── data/                   # Datasets and processed data
├── models/                 # Saved models and checkpoints
├── configs/               # Configuration files
├── scripts/               # Training and evaluation scripts
├── demo/                  # Streamlit demo application
├── src/                   # Core source code
│   ├── data/             # Data processing utilities
│   ├── models/           # Model implementations
│   ├── training/         # Training utilities
│   └── evaluation/       # Evaluation metrics
└── notebooks/            # Jupyter notebooks for analysis
```

## Models Used

- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2 (base)
- **Reranker Model**: cross-encoder/ms-marco-MiniLM-L-6-v2 (base)

## Dataset

Using MS MARCO passage ranking dataset for demonstration, with options to use custom datasets.
