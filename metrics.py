"""
Evaluation metrics for embedding models and rerankers.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import average_precision_score
import math
from collections import defaultdict


class RankingMetrics:
    """Comprehensive ranking evaluation metrics."""
    
    @staticmethod
    def ndcg_at_k(relevance_scores: List[float], k: int = 10) -> float:
        """Calculate Normalized Discounted Cumulative Gain at k."""
        if not relevance_scores:
            return 0.0
        
        # Truncate to k items
        relevance_scores = relevance_scores[:k]
        
        # Calculate DCG
        dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance_scores))
        
        # Calculate IDCG (ideal DCG)
        ideal_relevance = sorted(relevance_scores, reverse=True)
        idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_relevance))
        
        # Return NDCG
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def precision_at_k(relevance_scores: List[float], k: int = 10) -> float:
        """Calculate Precision at k."""
        if not relevance_scores or k <= 0:
            return 0.0
        
        relevant_items = sum(1 for score in relevance_scores[:k] if score > 0)
        return relevant_items / min(k, len(relevance_scores))
    
    @staticmethod
    def recall_at_k(relevance_scores: List[float], k: int = 10) -> float:
        """Calculate Recall at k."""
        if not relevance_scores:
            return 0.0
        
        total_relevant = sum(1 for score in relevance_scores if score > 0)
        if total_relevant == 0:
            return 0.0
        
        relevant_at_k = sum(1 for score in relevance_scores[:k] if score > 0)
        return relevant_at_k / total_relevant
    
    @staticmethod
    def mean_reciprocal_rank(relevance_scores: List[float]) -> float:
        """Calculate Mean Reciprocal Rank."""
        for i, score in enumerate(relevance_scores):
            if score > 0:
                return 1.0 / (i + 1)
        return 0.0
    
    @staticmethod
    def average_precision(relevance_scores: List[float]) -> float:
        """Calculate Average Precision."""
        if not relevance_scores:
            return 0.0
        
        relevant_items = 0
        precision_sum = 0.0
        
        for i, score in enumerate(relevance_scores):
            if score > 0:
                relevant_items += 1
                precision_sum += relevant_items / (i + 1)
        
        total_relevant = sum(1 for score in relevance_scores if score > 0)
        return precision_sum / total_relevant if total_relevant > 0 else 0.0
    
    @staticmethod
    def hit_rate_at_k(relevance_scores: List[float], k: int = 10) -> float:
        """Calculate Hit Rate at k (whether any relevant item is in top-k)."""
        if not relevance_scores:
            return 0.0
        
        return 1.0 if any(score > 0 for score in relevance_scores[:k]) else 0.0


class EmbeddingEvaluator:
    """Evaluator for embedding models."""
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
    
    def evaluate_retrieval(self, 
                          queries: List[str], 
                          candidates_list: List[List[str]], 
                          relevance_list: List[List[float]],
                          k_values: List[int] = [1, 5, 10]) -> Dict:
        """Evaluate embedding model on retrieval task."""
        
        assert len(queries) == len(candidates_list) == len(relevance_list)
        
        metrics = defaultdict(list)
        
        for query, candidates, relevance in zip(queries, candidates_list, relevance_list):
            # Get embeddings
            query_embedding = self.embedding_model.encode([query])
            candidate_embeddings = self.embedding_model.encode(candidates)
            
            # Calculate similarities
            similarities = np.dot(query_embedding, candidate_embeddings.T).flatten()
            
            # Sort by similarity (descending)
            sorted_indices = np.argsort(similarities)[::-1]
            sorted_relevance = [relevance[i] for i in sorted_indices]
            
            # Calculate metrics
            for k in k_values:
                metrics[f'ndcg@{k}'].append(RankingMetrics.ndcg_at_k(sorted_relevance, k))
                metrics[f'precision@{k}'].append(RankingMetrics.precision_at_k(sorted_relevance, k))
                metrics[f'recall@{k}'].append(RankingMetrics.recall_at_k(sorted_relevance, k))
                metrics[f'hit_rate@{k}'].append(RankingMetrics.hit_rate_at_k(sorted_relevance, k))
            
            metrics['mrr'].append(RankingMetrics.mean_reciprocal_rank(sorted_relevance))
            metrics['map'].append(RankingMetrics.average_precision(sorted_relevance))
        
        # Average all metrics
        avg_metrics = {metric: np.mean(values) for metric, values in metrics.items()}
        
        return avg_metrics
    
    def evaluate_similarity(self, 
                           text_pairs: List[Tuple[str, str]], 
                           similarity_scores: List[float]) -> Dict:
        """Evaluate embedding model on similarity task."""
        
        predicted_similarities = []
        for text1, text2 in text_pairs:
            embedding1 = self.embedding_model.encode([text1])
            embedding2 = self.embedding_model.encode([text2])
            similarity = np.dot(embedding1, embedding2.T).flatten()[0]
            predicted_similarities.append(similarity)
        
        # Calculate correlation
        correlation = np.corrcoef(predicted_similarities, similarity_scores)[0, 1]
        
        return {
            'spearman_correlation': correlation,
            'predicted_similarities': predicted_similarities
        }


class RerankerEvaluator:
    """Evaluator for reranker models."""
    
    def __init__(self, reranker_model):
        self.reranker_model = reranker_model
    
    def evaluate_ranking(self, 
                        queries: List[str], 
                        candidates_list: List[List[str]], 
                        relevance_list: List[List[float]],
                        k_values: List[int] = [1, 5, 10]) -> Dict:
        """Evaluate reranker model on ranking task."""
        
        assert len(queries) == len(candidates_list) == len(relevance_list)
        
        metrics = defaultdict(list)
        
        for query, candidates, relevance in zip(queries, candidates_list, relevance_list):
            # Get reranker scores
            scores = self.reranker_model.predict(query, candidates)
            
            # Sort by reranker scores (descending)
            sorted_indices = np.argsort(scores)[::-1]
            sorted_relevance = [relevance[i] for i in sorted_indices]
            
            # Calculate metrics
            for k in k_values:
                metrics[f'ndcg@{k}'].append(RankingMetrics.ndcg_at_k(sorted_relevance, k))
                metrics[f'precision@{k}'].append(RankingMetrics.precision_at_k(sorted_relevance, k))
                metrics[f'recall@{k}'].append(RankingMetrics.recall_at_k(sorted_relevance, k))
                metrics[f'hit_rate@{k}'].append(RankingMetrics.hit_rate_at_k(sorted_relevance, k))
            
            metrics['mrr'].append(RankingMetrics.mean_reciprocal_rank(sorted_relevance))
            metrics['map'].append(RankingMetrics.average_precision(sorted_relevance))
        
        # Average all metrics
        avg_metrics = {metric: np.mean(values) for metric, values in metrics.items()}
        
        return avg_metrics


class ComparisonEvaluator:
    """Compare embedding model and reranker performance."""
    
    def __init__(self, embedding_model, reranker_model):
        self.embedding_evaluator = EmbeddingEvaluator(embedding_model)
        self.reranker_evaluator = RerankerEvaluator(reranker_model)
    
    def compare_models(self, 
                      queries: List[str], 
                      candidates_list: List[List[str]], 
                      relevance_list: List[List[float]],
                      k_values: List[int] = [1, 5, 10]) -> Dict:
        """Compare embedding model vs reranker performance."""
        
        embedding_metrics = self.embedding_evaluator.evaluate_retrieval(
            queries, candidates_list, relevance_list, k_values
        )
        
        reranker_metrics = self.reranker_evaluator.evaluate_ranking(
            queries, candidates_list, relevance_list, k_values
        )
        
        comparison = {
            'embedding_model': embedding_metrics,
            'reranker_model': reranker_metrics,
            'improvements': {}
        }
        
        # Calculate improvements
        for metric in embedding_metrics:
            if metric in reranker_metrics:
                improvement = (reranker_metrics[metric] - embedding_metrics[metric]) / embedding_metrics[metric] * 100
                comparison['improvements'][metric] = improvement
        
        return comparison
    
    def pipeline_evaluation(self, 
                           queries: List[str], 
                           corpus: List[str],
                           relevance_list: List[List[float]],
                           retrieval_k: int = 20,
                           rerank_k: int = 10) -> Dict:
        """Evaluate full retrieval + reranking pipeline."""
        
        pipeline_metrics = defaultdict(list)
        
        for i, query in enumerate(queries):
            # Step 1: Retrieval with embedding model
            similarities = []
            query_embedding = self.embedding_evaluator.embedding_model.encode([query])
            
            for doc in corpus:
                doc_embedding = self.embedding_evaluator.embedding_model.encode([doc])
                similarity = np.dot(query_embedding, doc_embedding.T).flatten()[0]
                similarities.append(similarity)
            
            # Get top-k from retrieval
            top_retrieval_indices = np.argsort(similarities)[::-1][:retrieval_k]
            retrieved_docs = [corpus[idx] for idx in top_retrieval_indices]
            
            # Step 2: Reranking
            rerank_scores = self.reranker_evaluator.reranker_model.predict(query, retrieved_docs)
            rerank_indices = np.argsort(rerank_scores)[::-1][:rerank_k]
            
            # Final ranking
            final_doc_indices = [top_retrieval_indices[idx] for idx in rerank_indices]
            final_relevance = [relevance_list[i][idx] for idx in final_doc_indices]
            
            # Calculate metrics
            pipeline_metrics['ndcg@5'].append(RankingMetrics.ndcg_at_k(final_relevance, 5))
            pipeline_metrics['ndcg@10'].append(RankingMetrics.ndcg_at_k(final_relevance, 10))
            pipeline_metrics['mrr'].append(RankingMetrics.mean_reciprocal_rank(final_relevance))
            pipeline_metrics['map'].append(RankingMetrics.average_precision(final_relevance))
        
        return {metric: np.mean(values) for metric, values in pipeline_metrics.items()}
