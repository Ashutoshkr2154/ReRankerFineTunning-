"""
Data preparation script for embedding and reranker training.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from src.data.dataset_loader import DatasetLoader
import json


def main():
    parser = argparse.ArgumentParser(description='Prepare training data')
    parser.add_argument('--dataset', type=str, default='ms_marco',
                       choices=['ms_marco', 'sample'],
                       help='Dataset to use')
    parser.add_argument('--max_samples', type=int, default=5000,
                       help='Maximum number of samples to prepare')
    parser.add_argument('--output_dir', type=str, default='./data',
                       help='Output directory for prepared data')
    parser.add_argument('--eval_queries', type=int, default=200,
                       help='Number of queries for evaluation dataset')
    
    args = parser.parse_args()
    
    print("Preparing training data...")
    print(f"Dataset: {args.dataset}")
    print(f"Max samples: {args.max_samples}")
    print(f"Output directory: {args.output_dir}")
    
    # Initialize dataset loader
    dataset_loader = DatasetLoader(cache_dir=args.output_dir)
    
    # Load raw dataset
    if args.dataset == 'ms_marco':
        print("Loading MS MARCO dataset...")
        try:
            dataset = dataset_loader.load_ms_marco_dataset(max_samples=args.max_samples)
            print(f"Loaded {len(dataset)} samples from MS MARCO")
        except Exception as e:
            print(f"Failed to load MS MARCO: {e}")
            print("Using sample dataset instead...")
            dataset = dataset_loader._create_sample_dataset()
    else:
        print("Creating sample dataset...")
        dataset = dataset_loader._create_sample_dataset()
    
    print(f"Dataset size: {len(dataset)}")
    
    # Prepare embedding training data
    print("Preparing embedding training data...")
    embedding_data = dataset_loader.prepare_embedding_data(dataset)
    
    # Convert to serializable format - save in the format expected by training script
    embedding_data_dict = []
    for example in embedding_data:
        embedding_data_dict.append({
            'query': example.texts[0],
            'positive': example.texts[1] if example.label == 1.0 else None,
            'negative': example.texts[1] if example.label == 0.0 else None,
            'label': example.label
        })
    
    # Also create the format expected by the training script
    training_format = []
    for item in dataset:
        training_format.append({
            'query': item['query'],
            'positive': item['positive'],
            'negative': item['negative']
        })
    
    # Save both formats
    embedding_path = os.path.join(args.output_dir, 'embedding_training_data.json')
    with open(embedding_path, 'w', encoding='utf-8') as f:
        json.dump(training_format, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(training_format)} embedding examples to {embedding_path}")
    
    # Save the original format for compatibility
    embedding_original_path = os.path.join(args.output_dir, 'embedding_training_data_original.json')
    with open(embedding_original_path, 'w', encoding='utf-8') as f:
        json.dump(embedding_data_dict, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(embedding_data_dict)} embedding examples (original format) to {embedding_original_path}")
    
    # Prepare reranker training data
    print("Preparing reranker training data...")
    reranker_data = dataset_loader.prepare_reranker_data(dataset)
    
    # Save reranker data
    reranker_path = os.path.join(args.output_dir, 'reranker_training_data.json')
    with open(reranker_path, 'w', encoding='utf-8') as f:
        json.dump(reranker_data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(reranker_data)} reranker examples to {reranker_path}")
    
    # Create evaluation dataset
    print("Creating evaluation dataset...")
    eval_data = dataset_loader.create_evaluation_dataset(dataset, num_queries=args.eval_queries)
    
    # Save evaluation data
    eval_path = os.path.join(args.output_dir, 'evaluation_data.json')
    dataset_loader.save_dataset(eval_data, 'evaluation_data.json')
    print(f"Saved evaluation dataset with {len(eval_data['queries'])} queries to {eval_path}")
    
    # Create data statistics
    stats = {
        'dataset_type': args.dataset,
        'total_samples': len(dataset),
        'embedding_examples': len(embedding_data_dict),
        'reranker_examples': len(reranker_data),
        'evaluation_queries': len(eval_data['queries']),
        'avg_candidates_per_query': sum(len(candidates) for candidates in eval_data['candidates']) / len(eval_data['candidates']),
        'files_created': [
            'embedding_training_data.json',
            'reranker_training_data.json',
            'evaluation_data.json'
        ]
    }
    
    stats_path = os.path.join(args.output_dir, 'data_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\nData preparation completed!")
    print("="*50)
    print("STATISTICS:")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Embedding examples: {stats['embedding_examples']}")
    print(f"Reranker examples: {stats['reranker_examples']}")
    print(f"Evaluation queries: {stats['evaluation_queries']}")
    print(f"Avg candidates per query: {stats['avg_candidates_per_query']:.1f}")
    print("="*50)
    print("Files created:")
    for file in stats['files_created']:
        print(f"  - {os.path.join(args.output_dir, file)}")
    print(f"  - {stats_path}")
    print("="*50)


if __name__ == "__main__":
    main()
