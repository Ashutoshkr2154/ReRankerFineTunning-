"""
Embedding model fine-tuning implementation using sentence-transformers.
"""

import os
import torch
from sentence_transformers import SentenceTransformer, losses, evaluation
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
from typing import List, Dict, Optional, Tuple
import numpy as np
from datetime import datetime
import json


class EmbeddingModelTrainer:
    """Fine-tuning trainer for embedding models using sentence-transformers."""
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 output_dir: str = "./models/embedding",
                 device: str = None):
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the base model
        self.model = SentenceTransformer(model_name, device=self.device)
        print(f"Loaded model: {model_name} on {self.device}")
    
    def fine_tune(self, 
                  train_examples: List,
                  val_examples: Optional[List] = None,
                  epochs: int = 3,
                  batch_size: int = 16,
                  learning_rate: float = 2e-5,
                  warmup_steps: int = 100,
                  evaluation_steps: int = 1000,
                  save_best_model: bool = True) -> Dict:
        """Fine-tune the embedding model."""
        
        print(f"Starting fine-tuning with {len(train_examples)} training examples")
        
        # Create data loader
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        
        # Define loss function
        train_loss = losses.CosineSimilarityLoss(self.model)
        
        # Setup evaluator if validation data is provided
        evaluator = None
        if val_examples:
            evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
                val_examples, name='val'
            )
        
        # Training arguments
        model_save_path = os.path.join(self.output_dir, f"finetuned-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        
        # Fine-tune the model
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=epochs,
            evaluation_steps=evaluation_steps,
            warmup_steps=warmup_steps,
            output_path=model_save_path if save_best_model else None,
            save_best_model=save_best_model,
            optimizer_params={'lr': learning_rate}
        )
        
        # Save final model
        final_model_path = os.path.join(self.output_dir, "final_model")
        try:
            self.model.save(final_model_path)
        except Exception as e:
            print(f"Warning: Could not save to {final_model_path}: {e}")
            # Try alternative save location
            alt_path = os.path.join(self.output_dir, f"final_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            print(f"Trying alternative location: {alt_path}")
            try:
                self.model.save(alt_path)
                final_model_path = alt_path
                print(f"Model saved to alternative location: {alt_path}")
            except Exception as e2:
                print(f"Error saving to alternative location: {e2}")
                # Use the best model path instead
                final_model_path = model_save_path if save_best_model else alt_path
                print(f"Using best model path: {final_model_path}")
        
        # Save training info
        training_info = {
            "base_model": self.model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "warmup_steps": warmup_steps,
            "num_train_examples": len(train_examples),
            "num_val_examples": len(val_examples) if val_examples else 0,
            "device": self.device,
            "final_model_path": final_model_path,
            "best_model_path": model_save_path if save_best_model else None,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(os.path.join(self.output_dir, "training_info.json"), 'w') as f:
            json.dump(training_info, f, indent=2)
        
        print(f"Fine-tuning completed. Model saved to: {final_model_path}")
        return training_info
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        return self.model.encode(texts, convert_to_tensor=False)
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings (alias for encode_texts for compatibility)."""
        return self.encode_texts(texts)
    
    def similarity_search(self, query: str, candidates: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Perform similarity search."""
        query_embedding = self.model.encode([query])
        candidate_embeddings = self.model.encode(candidates)
        
        # Calculate cosine similarities
        similarities = torch.nn.functional.cosine_similarity(
            torch.tensor(query_embedding), 
            torch.tensor(candidate_embeddings)
        ).numpy()
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(candidates[i], float(similarities[i])) for i in top_indices]
        
        return results
    
    def load_model(self, model_path: str):
        """Load a saved model."""
        self.model = SentenceTransformer(model_path, device=self.device)
        print(f"Loaded model from: {model_path}")
    
    def evaluate_model(self, test_examples: List) -> Dict:
        """Evaluate the model on test data."""
        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
            test_examples, name='test'
        )
        
        score = evaluator(self.model)
        return {"test_score": score}


class EmbeddingModelInference:
    """Inference class for embedding models."""
    
    def __init__(self, model_path: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_path, device=self.device)
        print(f"Loaded inference model from: {model_path}")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        return self.model.encode(texts, convert_to_tensor=False)
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts."""
        embeddings = self.model.encode([text1, text2])
        similarity = torch.nn.functional.cosine_similarity(
            torch.tensor(embeddings[0:1]), 
            torch.tensor(embeddings[1:2])
        ).item()
        return similarity
    
    def find_most_similar(self, query: str, candidates: List[str]) -> Tuple[str, float]:
        """Find the most similar candidate to the query."""
        similarities = []
        query_embedding = self.model.encode([query])
        
        for candidate in candidates:
            candidate_embedding = self.model.encode([candidate])
            similarity = torch.nn.functional.cosine_similarity(
                torch.tensor(query_embedding), 
                torch.tensor(candidate_embedding)
            ).item()
            similarities.append((candidate, similarity))
        
        # Return the most similar
        return max(similarities, key=lambda x: x[1])
