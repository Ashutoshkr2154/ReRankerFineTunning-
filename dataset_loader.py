"""
Dataset loading and preprocessing utilities for fine-tuning embedding models and rerankers.
"""

import json
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datasets import Dataset, DatasetDict, load_dataset
from sentence_transformers import InputExample
import random
import os


class DatasetLoader:
    """Handles loading and preprocessing of datasets for embedding and reranker training."""
    
    def __init__(self, dataset_name: str = "ms_marco", cache_dir: str = "./data"):
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def load_ms_marco_dataset(self, split: str = "train", max_samples: Optional[int] = 10000) -> Dataset:
        """Load MS MARCO passage ranking dataset."""
        try:
            # Load MS MARCO dataset
            dataset = load_dataset("ms_marco", "v1.1", split=split, cache_dir=self.cache_dir)
            
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            
            return dataset
        except Exception as e:
            print(f"Error loading MS MARCO dataset: {e}")
            return self._create_sample_dataset()
    
    def _create_sample_dataset(self) -> Dataset:
        """Create a sample dataset for demonstration purposes."""
        sample_data = [
            {
                "query": "What is machine learning?",
                "positive": "Machine learning is a method of data analysis that automates analytical model building using algorithms that iteratively learn from data.",
                "negative": "Cooking is the practice of preparing food by combining ingredients and applying heat."
            },
            {
                "query": "How does neural network work?",
                "positive": "Neural networks are computing systems inspired by biological neural networks. They learn to perform tasks by considering examples.",
                "negative": "Gardening involves growing and cultivating plants as part of horticulture."
            },
            {
                "query": "What is deep learning?",
                "positive": "Deep learning is part of machine learning methods based on artificial neural networks with representation learning.",
                "negative": "Swimming is a sport involving movement through water using arms and legs."
            },
            {
                "query": "Explain natural language processing",
                "positive": "Natural language processing is a subfield of AI that helps computers understand, interpret and manipulate human language.",
                "negative": "Photography is the art and science of creating durable images by recording light."
            },
            {
                "query": "What is computer vision?",
                "positive": "Computer vision is an interdisciplinary field that deals with how computers can gain understanding from digital images or videos.",
                "negative": "Music theory is the study of the practices and possibilities of music composition and performance."
            }
        ]
        
        # Expand the dataset with variations
        expanded_data = []
        for item in sample_data:
            expanded_data.append(item)
            # Add some variations
            for i in range(3):
                expanded_data.append({
                    "query": item["query"] + f" (variation {i+1})",
                    "positive": item["positive"],
                    "negative": random.choice([d["negative"] for d in sample_data if d != item])
                })
        
        return Dataset.from_list(expanded_data)
    
    def prepare_embedding_data(self, dataset: Dataset) -> List[InputExample]:
        """Prepare data for embedding model training."""
        examples = []
        
        for item in dataset:
            # Positive example
            examples.append(InputExample(
                texts=[item["query"], item["positive"]], 
                label=1.0
            ))
            
            # Negative example
            examples.append(InputExample(
                texts=[item["query"], item["negative"]], 
                label=0.0
            ))
        
        return examples
    
    def prepare_reranker_data(self, dataset: Dataset) -> List[Dict]:
        """Prepare data for reranker training."""
        examples = []
        
        for item in dataset:
            # Positive example
            examples.append({
                "query": item["query"],
                "passage": item["positive"],
                "label": 1
            })
            
            # Negative example
            examples.append({
                "query": item["query"],
                "passage": item["negative"],
                "label": 0
            })
        
        return examples
    
    def create_evaluation_dataset(self, dataset: Dataset, num_queries: int = 50) -> Dict:
        """Create evaluation dataset with multiple candidates per query."""
        eval_data = {"queries": [], "candidates": [], "relevance": []}
        
        dataset_list = list(dataset)
        selected_items = random.sample(dataset_list, min(num_queries, len(dataset_list)))
        
        for i, item in enumerate(selected_items):
            query = item["query"]
            positive = item["positive"]
            
            # Get some random negatives from other items
            other_items = [d for d in dataset_list if d != item]
            negatives = random.sample([d["negative"] for d in other_items], 
                                    min(4, len(other_items)))
            
            candidates = [positive] + negatives
            relevance_scores = [1] + [0] * len(negatives)
            
            # Shuffle candidates and relevance together
            combined = list(zip(candidates, relevance_scores))
            random.shuffle(combined)
            candidates, relevance_scores = zip(*combined)
            
            eval_data["queries"].append(query)
            eval_data["candidates"].append(list(candidates))
            eval_data["relevance"].append(list(relevance_scores))
        
        return eval_data
    
    def save_dataset(self, data: Dict, filename: str):
        """Save dataset to JSON file."""
        filepath = os.path.join(self.cache_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Dataset saved to {filepath}")
    
    def load_dataset_from_file(self, filename: str) -> Dict:
        """Load dataset from JSON file."""
        filepath = os.path.join(self.cache_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)


def create_triplet_dataset(queries: List[str], positives: List[str], negatives: List[str]) -> List[Dict]:
    """Create triplet dataset from queries, positives, and negatives."""
    assert len(queries) == len(positives) == len(negatives), "All lists must have same length"
    
    triplets = []
    for q, p, n in zip(queries, positives, negatives):
        triplets.append({
            "query": q,
            "positive": p,
            "negative": n
        })
    
    return triplets
