"""
Training script for embedding model fine-tuning.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
from src.data.dataset_loader import DatasetLoader
from src.models.embedding_model import EmbeddingModelTrainer
from src.evaluation.metrics import EmbeddingEvaluator
import json


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Fine-tune embedding model')
    parser.add_argument('--config', type=str, default='configs/embedding_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model_name', type=str, 
                       default='sentence-transformers/all-MiniLM-L6-v2',
                       help='Base model name')
    parser.add_argument('--output_dir', type=str, default='./models/embedding',
                       help='Output directory for saved models')
    parser.add_argument('--max_samples', type=int, default=1000,
                       help='Maximum number of training samples')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    # Load configuration if exists
    config = {}
    if os.path.exists(args.config):
        config = load_config(args.config)
        print(f"Loaded configuration from {args.config}")
    
    # Override config with command line arguments
    config.update({
        'model_name': args.model_name,
        'output_dir': args.output_dir,
        'max_samples': args.max_samples,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate
    })
    
    print("Starting embedding model training...")
    print(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Load dataset
    print("Loading dataset...")
    dataset_loader = DatasetLoader()
    
    # Use sample dataset directly for now to avoid MS MARCO loading issues
    print("Using sample dataset for training...")
    dataset = dataset_loader._create_sample_dataset()
    
    # Prepare training data
    train_examples = dataset_loader.prepare_embedding_data(dataset)
    print(f"Prepared {len(train_examples)} training examples")
    
    # Split data for validation (80-20 split)
    split_idx = int(0.8 * len(train_examples))
    train_data = train_examples[:split_idx]
    val_data = train_examples[split_idx:]
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Initialize trainer
    trainer = EmbeddingModelTrainer(
        model_name=config['model_name'],
        output_dir=config['output_dir']
    )
    
    # Fine-tune the model
    training_info = trainer.fine_tune(
        train_examples=train_data,
        val_examples=val_data,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        warmup_steps=config.get('warmup_steps', 100),
        evaluation_steps=config.get('evaluation_steps', 1000)
    )
    
    print("Training completed!")
    print(f"Training info: {json.dumps(training_info, indent=2)}")
    
    # Evaluate the model
    print("Evaluating model...")
    eval_data = dataset_loader.create_evaluation_dataset(dataset, num_queries=50)
    
    evaluator = EmbeddingEvaluator(trainer)
    eval_results = evaluator.evaluate_retrieval(
        queries=eval_data['queries'],
        candidates_list=eval_data['candidates'],
        relevance_list=eval_data['relevance']
    )
    
    print("Evaluation results:")
    for metric, value in eval_results.items():
        print(f"{metric}: {value:.4f}")
    
    # Save evaluation results
    eval_path = os.path.join(config['output_dir'], 'evaluation_results.json')
    with open(eval_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"Evaluation results saved to: {eval_path}")
    print("Embedding model training completed successfully!")


if __name__ == "__main__":
    main()
