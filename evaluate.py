"""
Comprehensive evaluation script for both embedding and reranker models.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from src.data.dataset_loader import DatasetLoader
from src.models.embedding_model import EmbeddingModelInference
from src.models.reranker_model import RerankerModelInference
from src.evaluation.metrics import ComparisonEvaluator, EmbeddingEvaluator, RerankerEvaluator


def main():
    parser = argparse.ArgumentParser(description='Evaluate embedding and reranker models')
    parser.add_argument('--embedding_model', type=str, 
                       default='./models/embedding/final_model',
                       help='Path to fine-tuned embedding model')
    parser.add_argument('--reranker_model', type=str, 
                       default='./models/reranker/final_model',
                       help='Path to fine-tuned reranker model')
    parser.add_argument('--base_embedding_model', type=str,
                       default='sentence-transformers/all-MiniLM-L6-v2',
                       help='Base embedding model for comparison')
    parser.add_argument('--base_reranker_model', type=str,
                       default='cross-encoder/ms-marco-MiniLM-L-6-v2',
                       help='Base reranker model for comparison')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Output directory for evaluation results')
    parser.add_argument('--num_queries', type=int, default=100,
                       help='Number of queries for evaluation')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Starting comprehensive evaluation...")
    
    # Load evaluation dataset
    print("Loading evaluation dataset...")
    dataset_loader = DatasetLoader()
    
    # Use sample dataset directly for now to avoid MS MARCO loading issues
    print("Using sample dataset for evaluation...")
    dataset = dataset_loader._create_sample_dataset()
    
    eval_data = dataset_loader.create_evaluation_dataset(dataset, num_queries=args.num_queries)
    
    print(f"Evaluation dataset: {len(eval_data['queries'])} queries")
    
    # Load models
    print("Loading models...")
    
    # Try to load fine-tuned models, fallback to base models
    try:
        if os.path.exists(args.embedding_model):
            embedding_model = EmbeddingModelInference(args.embedding_model)
            print(f"Loaded fine-tuned embedding model from {args.embedding_model}")
        else:
            from src.models.embedding_model import EmbeddingModelTrainer
            embedding_model = EmbeddingModelTrainer(args.base_embedding_model)
            print(f"Using base embedding model: {args.base_embedding_model}")
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        from src.models.embedding_model import EmbeddingModelTrainer
        embedding_model = EmbeddingModelTrainer(args.base_embedding_model)
        print(f"Fallback to base embedding model: {args.base_embedding_model}")
    
    try:
        if os.path.exists(args.reranker_model):
            reranker_model = RerankerModelInference(args.reranker_model)
            print(f"Loaded fine-tuned reranker model from {args.reranker_model}")
        else:
            from src.models.reranker_model import RerankerModelTrainer
            reranker_model = RerankerModelTrainer(args.base_reranker_model)
            print(f"Using base reranker model: {args.base_reranker_model}")
    except Exception as e:
        print(f"Error loading reranker model: {e}")
        from src.models.reranker_model import RerankerModelTrainer
        reranker_model = RerankerModelTrainer(args.base_reranker_model)
        print(f"Fallback to base reranker model: {args.base_reranker_model}")
    
    # Evaluate embedding model
    print("Evaluating embedding model...")
    embedding_evaluator = EmbeddingEvaluator(embedding_model)
    embedding_results = embedding_evaluator.evaluate_retrieval(
        queries=eval_data['queries'],
        candidates_list=eval_data['candidates'],
        relevance_list=eval_data['relevance'],
        k_values=[1, 3, 5, 10]
    )
    
    print("Embedding model results:")
    for metric, value in embedding_results.items():
        print(f"  {metric}: {value:.4f}")
    
    # Evaluate reranker model
    print("Evaluating reranker model...")
    reranker_evaluator = RerankerEvaluator(reranker_model)
    reranker_results = reranker_evaluator.evaluate_ranking(
        queries=eval_data['queries'],
        candidates_list=eval_data['candidates'],
        relevance_list=eval_data['relevance'],
        k_values=[1, 3, 5, 10]
    )
    
    print("Reranker model results:")
    for metric, value in reranker_results.items():
        print(f"  {metric}: {value:.4f}")
    
    # Compare models
    print("Comparing models...")
    comparison_evaluator = ComparisonEvaluator(embedding_model, reranker_model)
    comparison_results = comparison_evaluator.compare_models(
        queries=eval_data['queries'],
        candidates_list=eval_data['candidates'],
        relevance_list=eval_data['relevance'],
        k_values=[1, 3, 5, 10]
    )
    
    print("Model comparison (improvements by reranker):")
    for metric, improvement in comparison_results['improvements'].items():
        print(f"  {metric}: {improvement:+.2f}%")
    
    # Evaluate full pipeline
    print("Evaluating full retrieval + reranking pipeline...")
    # Create a larger corpus for pipeline evaluation
    corpus = []
    for candidates in eval_data['candidates']:
        corpus.extend(candidates)
    corpus = list(set(corpus))  # Remove duplicates
    
    # Create relevance mapping for corpus
    corpus_relevance = []
    for i, candidates in enumerate(eval_data['candidates']):
        relevance_map = {}
        for j, candidate in enumerate(candidates):
            relevance_map[candidate] = eval_data['relevance'][i][j]
        
        corpus_rel = [relevance_map.get(doc, 0) for doc in corpus]
        corpus_relevance.append(corpus_rel)
    
    try:
        pipeline_results = comparison_evaluator.pipeline_evaluation(
            queries=eval_data['queries'][:20],  # Limit for performance
            corpus=corpus,
            relevance_list=corpus_relevance[:20],
            retrieval_k=20,
            rerank_k=10
        )
        
        print("Pipeline evaluation results:")
        for metric, value in pipeline_results.items():
            print(f"  {metric}: {value:.4f}")
    except Exception as e:
        print(f"Pipeline evaluation failed: {e}")
        pipeline_results = {}
    
    # Save all results
    all_results = {
        'embedding_model_results': embedding_results,
        'reranker_model_results': reranker_results,
        'model_comparison': comparison_results,
        'pipeline_results': pipeline_results,
        'evaluation_config': {
            'embedding_model_path': args.embedding_model,
            'reranker_model_path': args.reranker_model,
            'num_queries': args.num_queries,
            'base_embedding_model': args.base_embedding_model,
            'base_reranker_model': args.base_reranker_model
        }
    }
    
    results_path = os.path.join(args.output_dir, 'comprehensive_evaluation.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nEvaluation completed! Results saved to: {results_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    if embedding_results:
        print(f"Embedding Model NDCG@10: {embedding_results.get('ndcg@10', 0):.4f}")
    if reranker_results:
        print(f"Reranker Model NDCG@10: {reranker_results.get('ndcg@10', 0):.4f}")
    if 'improvements' in comparison_results:
        ndcg10_improvement = comparison_results['improvements'].get('ndcg@10', 0)
        print(f"NDCG@10 Improvement: {ndcg10_improvement:+.2f}%")
    
    print("="*50)


if __name__ == "__main__":
    main()
