#!/usr/bin/env python3
"""
Generate embeddings for any data using your trained models
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.embedding_model import EmbeddingModelInference
from src.models.reranker_model import RerankerModelInference

def load_models():
    """Load your trained models"""
    print("🔄 Loading trained models...")
    
    # Load embedding model
    embedding_model = EmbeddingModelInference("./models/embedding/final_model")
    print("✅ Embedding model loaded successfully!")
    
    # Load reranker model
    reranker_model = RerankerModelInference("./models/reranker/final_model")
    print("✅ Reranker model loaded successfully!")
    
    return embedding_model, reranker_model

def generate_embeddings_example():
    """Example of generating embeddings for different types of data"""
    
    # Load models
    embedding_model, reranker_model = load_models()
    
    print("\n" + "="*60)
    print("🎯 GENERATING EMBEDDINGS FOR YOUR DATA")
    print("="*60)
    
    # Example 1: Single text embedding
    print("\n📝 Example 1: Single Text Embedding")
    text = "What is artificial intelligence?"
    embedding = embedding_model.encode([text])
    print(f"Text: {text}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding (first 10 values): {embedding[0][:10]}")
    
    # Example 2: Multiple texts
    print("\n📚 Example 2: Multiple Texts")
    texts = [
        "Machine learning algorithms",
        "Deep learning neural networks", 
        "Natural language processing",
        "Computer vision applications",
        "Data science techniques"
    ]
    embeddings = embedding_model.encode(texts)
    print(f"Number of texts: {len(texts)}")
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Example 3: Similarity between texts
    print("\n🔍 Example 3: Text Similarity")
    text1 = "What is machine learning?"
    text2 = "How do ML algorithms work?"
    text3 = "What is cooking?"
    
    similarity_12 = embedding_model.compute_similarity(text1, text2)
    similarity_13 = embedding_model.compute_similarity(text1, text3)
    
    print(f"Similarity between '{text1}' and '{text2}': {similarity_12:.4f}")
    print(f"Similarity between '{text1}' and '{text3}': {similarity_13:.4f}")
    
    # Example 4: Reranker scoring
    print("\n🎯 Example 4: Reranker Scoring")
    query = "What is machine learning?"
    passages = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
        "Cooking is the art of preparing food using various techniques and ingredients.",
        "Machine learning algorithms can identify patterns in data to make predictions.",
        "Swimming is a sport that involves moving through water using arms and legs."
    ]
    
    scores = reranker_model.predict(query, passages)
    print(f"Query: {query}")
    for i, (passage, score) in enumerate(zip(passages, scores)):
        print(f"Passage {i+1} (Score: {score:.4f}): {passage[:80]}...")
    
    # Example 5: Save embeddings to file
    print("\n💾 Example 5: Saving Embeddings")
    save_embeddings_to_file(texts, embeddings, "example_embeddings.json")
    
    return embedding_model, reranker_model

def save_embeddings_to_file(texts, embeddings, filename):
    """Save embeddings to a JSON file"""
    data = {
        "texts": texts,
        "embeddings": embeddings.tolist(),
        "shape": embeddings.shape,
        "model_path": "./models/embedding/final_model"
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Embeddings saved to {filename}")

def generate_embeddings_for_custom_data():
    """Generate embeddings for your own custom data"""
    
    print("\n" + "="*60)
    print("🚀 CUSTOM DATA EMBEDDING GENERATOR")
    print("="*60)
    
    # Load models
    embedding_model, reranker_model = load_models()
    
    # Your custom data here
    custom_texts = [
        "Your custom text 1",
        "Your custom text 2", 
        "Your custom text 3"
    ]
    
    print(f"\n📝 Generating embeddings for {len(custom_texts)} custom texts...")
    
    # Generate embeddings
    embeddings = embedding_model.encode(custom_texts)
    
    # Save results
    save_embeddings_to_file(custom_texts, embeddings, "custom_embeddings.json")
    
    print("\n🎉 Custom embeddings generated successfully!")
    print("📁 Check 'custom_embeddings.json' for results")

def main():
    """Main function"""
    print("🎛️ EMBEDDING GENERATION TOOL")
    print("Using your fine-tuned models!")
    
    try:
        # Generate example embeddings
        embedding_model, reranker_model = generate_embeddings_example()
        
        # Ask if user wants to generate custom embeddings
        print("\n" + "="*60)
        print("❓ Do you want to generate embeddings for your own data?")
        print("Edit the 'custom_texts' list in the script and run:")
        print("python scripts/generate_embeddings.py --custom")
        
        # Show model information
        print("\n" + "="*60)
        print("📁 MODEL LOCATIONS:")
        print(f"🤖 Embedding Model: ./models/embedding/final_model/")
        print(f"🎯 Reranker Model: ./models/reranker/final_model/")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure your models are trained and available!")

if __name__ == "__main__":
    main()
