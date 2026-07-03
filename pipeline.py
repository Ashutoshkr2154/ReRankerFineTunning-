"""
Complete inference pipeline combining embedding retrieval and reranking.
"""

import os
import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime
import logging

from src.models.embedding_model import EmbeddingModelInference
from src.models.reranker_model import RerankerModelInference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SearchPipeline:
    """Complete search pipeline with embedding retrieval and reranking."""
    
    def __init__(self, 
                 embedding_model_path: str,
                 reranker_model_path: str,
                 corpus: Optional[List[str]] = None,
                 index_path: Optional[str] = None):
        
        self.embedding_model_path = embedding_model_path
        self.reranker_model_path = reranker_model_path
        self.corpus = corpus or []
        self.index_path = index_path
        
        # Load models
        self._load_models()
        
        # Initialize FAISS index
        self.index = None
        self.embeddings = None
        
        if corpus:
            self._build_index()
        elif index_path and os.path.exists(index_path):
            self._load_index()
    
    def _load_models(self):
        """Load embedding and reranker models."""
        try:
            # Try loading fine-tuned models first
            if os.path.exists(self.embedding_model_path):
                self.embedding_model = EmbeddingModelInference(self.embedding_model_path)
                logger.info(f"Loaded fine-tuned embedding model from {self.embedding_model_path}")
            else:
                # Fallback to base model
                from src.models.embedding_model import EmbeddingModelTrainer
                self.embedding_model = EmbeddingModelTrainer("sentence-transformers/all-MiniLM-L6-v2")
                logger.info("Using base embedding model")
                
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            from src.models.embedding_model import EmbeddingModelTrainer
            self.embedding_model = EmbeddingModelTrainer("sentence-transformers/all-MiniLM-L6-v2")
        
        try:
            if os.path.exists(self.reranker_model_path):
                self.reranker_model = RerankerModelInference(self.reranker_model_path)
                logger.info(f"Loaded fine-tuned reranker model from {self.reranker_model_path}")
            else:
                # Fallback to base model
                from src.models.reranker_model import RerankerModelTrainer
                self.reranker_model = RerankerModelTrainer("cross-encoder/ms-marco-MiniLM-L-6-v2")
                logger.info("Using base reranker model")
                
        except Exception as e:
            logger.error(f"Error loading reranker model: {e}")
            from src.models.reranker_model import RerankerModelTrainer
            self.reranker_model = RerankerModelTrainer("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    def _build_index(self):
        """Build FAISS index from corpus."""
        logger.info(f"Building FAISS index for {len(self.corpus)} documents...")
        
        # Get embeddings for all documents
        if hasattr(self.embedding_model, 'encode'):
            self.embeddings = self.embedding_model.encode(self.corpus)
        else:
            self.embeddings = self.embedding_model.encode_texts(self.corpus)
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        
        # Add to index
        self.index.add(self.embeddings.astype('float32'))
        
        logger.info(f"Built FAISS index with {self.index.ntotal} documents")
    
    def _load_index(self):
        """Load pre-built FAISS index."""
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            
            # Load embeddings if available
            embeddings_path = self.index_path.replace('.index', '_embeddings.npy')
            if os.path.exists(embeddings_path):
                self.embeddings = np.load(embeddings_path)
            
            # Load corpus if available
            corpus_path = self.index_path.replace('.index', '_corpus.json')
            if os.path.exists(corpus_path):
                with open(corpus_path, 'r', encoding='utf-8') as f:
                    self.corpus = json.load(f)
            
            logger.info(f"Loaded FAISS index with {self.index.ntotal} documents")
    
    def save_index(self, save_path: str):
        """Save FAISS index, embeddings, and corpus."""
        if self.index is not None:
            # Save index
            faiss.write_index(self.index, save_path)
            
            # Save embeddings
            if self.embeddings is not None:
                embeddings_path = save_path.replace('.index', '_embeddings.npy')
                np.save(embeddings_path, self.embeddings)
            
            # Save corpus
            if self.corpus:
                corpus_path = save_path.replace('.index', '_corpus.json')
                with open(corpus_path, 'w', encoding='utf-8') as f:
                    json.dump(self.corpus, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved index to {save_path}")
    
    def add_documents(self, documents: List[str]):
        """Add new documents to the corpus and index."""
        if not documents:
            return
        
        # Add to corpus
        start_idx = len(self.corpus)
        self.corpus.extend(documents)
        
        # Get embeddings for new documents
        if hasattr(self.embedding_model, 'encode'):
            new_embeddings = self.embedding_model.encode(documents)
        else:
            new_embeddings = self.embedding_model.encode_texts(documents)
        
        # Normalize embeddings
        faiss.normalize_L2(new_embeddings)
        
        if self.index is None:
            # Create new index
            dimension = new_embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.embeddings = new_embeddings
        else:
            # Add to existing index
            if self.embeddings is not None:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])
            else:
                self.embeddings = new_embeddings
        
        # Add to FAISS index
        self.index.add(new_embeddings.astype('float32'))
        
        logger.info(f"Added {len(documents)} documents to index")
    
    def search(self, 
               query: str, 
               top_k: int = 10, 
               retrieval_k: int = 50,
               use_reranker: bool = True) -> List[Dict]:
        """
        Complete search pipeline: retrieval + reranking.
        
        Args:
            query: Search query
            top_k: Final number of results to return
            retrieval_k: Number of candidates to retrieve before reranking
            use_reranker: Whether to use reranker for final ranking
        
        Returns:
            List of search results with scores and metadata
        """
        if self.index is None or len(self.corpus) == 0:
            return []
        
        start_time = datetime.now()
        
        # Step 1: Retrieval with embedding model
        if hasattr(self.embedding_model, 'encode'):
            query_embedding = self.embedding_model.encode([query])
        else:
            query_embedding = self.embedding_model.encode_texts([query])
        
        # Normalize query embedding
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS index
        retrieval_k = min(retrieval_k, len(self.corpus))
        similarities, indices = self.index.search(query_embedding.astype('float32'), retrieval_k)
        
        # Get retrieved documents
        retrieved_docs = [self.corpus[idx] for idx in indices[0] if idx < len(self.corpus)]
        retrieval_scores = similarities[0].tolist()
        
        retrieval_time = (datetime.now() - start_time).total_seconds()
        
        # Step 2: Reranking (optional)
        if use_reranker and len(retrieved_docs) > 0:
            rerank_start = datetime.now()
            
            try:
                if hasattr(self.reranker_model, 'rerank_passages'):
                    reranked_results = self.reranker_model.rerank_passages(query, retrieved_docs)
                    reranked_docs = [doc for doc, _ in reranked_results]
                    rerank_scores = [score for _, score in reranked_results]
                else:
                    rerank_scores = self.reranker_model.predict(query, retrieved_docs)
                    # Sort by rerank scores
                    sorted_pairs = sorted(zip(retrieved_docs, rerank_scores), 
                                        key=lambda x: x[1], reverse=True)
                    reranked_docs = [doc for doc, _ in sorted_pairs]
                    rerank_scores = [score for _, score in sorted_pairs]
                
                final_docs = reranked_docs[:top_k]
                final_scores = rerank_scores[:top_k]
                
            except Exception as e:
                logger.error(f"Reranking failed: {e}")
                final_docs = retrieved_docs[:top_k]
                final_scores = retrieval_scores[:top_k]
            
            rerank_time = (datetime.now() - start_time).total_seconds() - retrieval_time
        else:
            final_docs = retrieved_docs[:top_k]
            final_scores = retrieval_scores[:top_k]
            rerank_time = 0.0
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Format results
        results = []
        for i, (doc, score) in enumerate(zip(final_docs, final_scores)):
            results.append({
                'rank': i + 1,
                'document': doc,
                'score': float(score),
                'doc_id': self.corpus.index(doc) if doc in self.corpus else -1
            })
        
        # Add metadata
        metadata = {
            'query': query,
            'total_results': len(results),
            'retrieval_k': retrieval_k,
            'used_reranker': use_reranker,
            'retrieval_time': retrieval_time,
            'rerank_time': rerank_time,
            'total_time': total_time,
            'corpus_size': len(self.corpus)
        }
        
        return {
            'results': results,
            'metadata': metadata
        }
    
    def batch_search(self, 
                    queries: List[str], 
                    top_k: int = 10, 
                    retrieval_k: int = 50,
                    use_reranker: bool = True) -> List[Dict]:
        """Batch search for multiple queries."""
        batch_results = []
        
        for query in queries:
            result = self.search(query, top_k, retrieval_k, use_reranker)
            batch_results.append(result)
        
        return batch_results
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        stats = {
            'corpus_size': len(self.corpus),
            'index_size': self.index.ntotal if self.index else 0,
            'embedding_model_path': self.embedding_model_path,
            'reranker_model_path': self.reranker_model_path,
            'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else 0
        }
        
        return stats


def create_demo_corpus() -> List[str]:
    """Create a sample corpus for demonstration."""
    demo_corpus = [
        "Machine learning is a method of data analysis that automates analytical model building.",
        "Deep learning is part of machine learning methods based on artificial neural networks.",
        "Natural language processing helps computers understand and interpret human language.",
        "Computer vision enables machines to interpret and make decisions based on visual data.",
        "Artificial intelligence is intelligence demonstrated by machines, in contrast to natural intelligence.",
        "Neural networks are computing systems inspired by biological neural networks.",
        "Supervised learning uses labeled training data to learn a mapping from inputs to outputs.",
        "Unsupervised learning finds hidden patterns in data without labeled examples.",
        "Reinforcement learning is about taking suitable actions to maximize reward in a particular situation.",
        "Feature engineering is the process of selecting and transforming variables for your model.",
        "Cross-validation is a resampling procedure used to evaluate machine learning models.",
        "Overfitting occurs when a model learns the training data too well and performs poorly on new data.",
        "Gradient descent is an optimization algorithm used to minimize the cost function in machine learning.",
        "Support vector machines are supervised learning models used for classification and regression.",
        "Random forests combine multiple decision trees to create a more robust predictive model.",
        "Clustering is the task of grouping a set of objects in such a way that objects in the same group are more similar.",
        "Dimensionality reduction techniques reduce the number of input variables in a dataset.",
        "Ensemble methods combine multiple learning algorithms to improve predictive performance.",
        "Transfer learning leverages knowledge gained from one task to improve performance on a related task.",
        "Attention mechanisms allow models to focus on specific parts of the input when making predictions."
    ]
    
    return demo_corpus
