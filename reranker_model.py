"""
Reranker model fine-tuning implementation using cross-encoder architecture.
"""

import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from datasets import Dataset
from typing import List, Dict, Optional, Tuple
import numpy as np
from datetime import datetime
import json
from sklearn.metrics import ndcg_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RerankerDataset(torch.utils.data.Dataset):
    """Dataset class for reranker training."""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        query = item['query']
        passage = item['passage']
        label = float(item['label'])
        
        # Combine query and passage
        text = f"{query} [SEP] {passage}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }


class RerankerModelTrainer:
    """Fine-tuning trainer for reranker models using cross-encoder architecture."""
    
    def __init__(self, 
                 model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 output_dir: str = "./models/reranker",
                 device: str = None):
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=1,  # Regression task
            torch_dtype=torch.float32
        )
        
        # Move model to device
        self.model.to(self.device)
        
        print(f"Loaded reranker model: {model_name} on {self.device}")
    
    def prepare_dataset(self, data: List[Dict]) -> Dataset:
        """Prepare dataset for training."""
        dataset = RerankerDataset(data, self.tokenizer)
        return dataset
    
    def fine_tune(self, 
                  train_data: List[Dict],
                  val_data: Optional[List[Dict]] = None,
                  epochs: int = 3,
                  batch_size: int = 8,
                  learning_rate: float = 2e-5,
                  warmup_steps: int = 100,
                  evaluation_steps: int = 500,
                  save_steps: int = 1000,
                  logging_steps: int = 100) -> Dict:
        """Fine-tune the reranker model."""
        
        print(f"Starting reranker fine-tuning with {len(train_data)} training examples")
        
        # Prepare datasets
        train_dataset = self.prepare_dataset(train_data)
        val_dataset = self.prepare_dataset(val_data) if val_data else None
        
        # Training arguments
        model_save_path = os.path.join(self.output_dir, f"finetuned-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        
        training_args = TrainingArguments(
            output_dir=model_save_path,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            eval_steps=evaluation_steps if val_dataset else None,
            save_steps=save_steps,
            save_total_limit=2,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(self.tokenizer)
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Train the model
        trainer.train()
        
        # Save final model
        final_model_path = os.path.join(self.output_dir, "final_model")
        try:
            trainer.save_model(final_model_path)
            self.tokenizer.save_pretrained(final_model_path)
        except Exception as e:
            print(f"Warning: Could not save to {final_model_path}: {e}")
            # Try alternative save location
            alt_path = os.path.join(self.output_dir, f"final_model_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
            print(f"Trying alternative location: {alt_path}")
            try:
                trainer.save_model(alt_path)
                self.tokenizer.save_pretrained(alt_path)
                final_model_path = alt_path
                print(f"Model saved to alternative location: {alt_path}")
            except Exception as e2:
                print(f"Error saving to alternative location: {e2}")
                final_model_path = model_save_path
                print(f"Using best model path: {final_model_path}")
        
        # Save training info
        training_info = {
            "base_model": self.model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "warmup_steps": warmup_steps,
            "num_train_examples": len(train_data),
            "num_val_examples": len(val_data) if val_data else 0,
            "device": self.device,
            "final_model_path": final_model_path,
            "best_model_path": model_save_path,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(os.path.join(self.output_dir, "training_info.json"), 'w') as f:
            json.dump(training_info, f, indent=2)
        
        print(f"Reranker fine-tuning completed. Model saved to: {final_model_path}")
        return training_info
    
    def predict(self, query: str, passages: List[str]) -> List[float]:
        """Predict relevance scores for query-passage pairs."""
        self.model.eval()
        scores = []
        
        with torch.no_grad():
            for passage in passages:
                text = f"{query} [SEP] {passage}"
                inputs = self.tokenizer(
                    text,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(self.device)
                
                outputs = self.model(**inputs)
                score = torch.sigmoid(outputs.logits).cpu().item()
                scores.append(score)
        
        return scores
    
    def rerank(self, query: str, passages: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Rerank passages for a given query."""
        scores = self.predict(query, passages)
        
        # Sort by scores in descending order
        ranked_results = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
        
        return ranked_results[:top_k]


class RerankerModelInference:
    """Inference class for reranker models."""
    
    def __init__(self, model_path: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded reranker inference model from: {model_path}")
    
    def predict_score(self, query: str, passage: str) -> float:
        """Predict relevance score for a query-passage pair."""
        text = f"{query} [SEP] {passage}"
        
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            outputs = self.model(**inputs)
            score = torch.sigmoid(outputs.logits).cpu().item()
        
        return score
    
    def rerank_passages(self, query: str, passages: List[str]) -> List[Tuple[str, float]]:
        """Rerank a list of passages for a given query."""
        scores = []
        
        for passage in passages:
            score = self.predict_score(query, passage)
            scores.append((passage, score))
        
        # Sort by score in descending order
        return sorted(scores, key=lambda x: x[1], reverse=True)
    
    def batch_predict(self, queries: List[str], passages: List[str]) -> List[float]:
        """Batch prediction for multiple query-passage pairs."""
        assert len(queries) == len(passages), "Queries and passages must have same length"
        
        scores = []
        for query, passage in zip(queries, passages):
            score = self.predict_score(query, passage)
            scores.append(score)
        
        return scores
    
    def predict(self, query: str, passages: List[str]) -> List[float]:
        """Predict relevance scores for query-passage pairs (compatibility method)."""
        scores = []
        for passage in passages:
            score = self.predict_score(query, passage)
            scores.append(score)
        return scores
