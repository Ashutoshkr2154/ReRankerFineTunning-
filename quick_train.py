#!/usr/bin/env python3
"""
Quick training script that handles file conflicts and trains both models
"""

import os
import sys
import time
import shutil
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def clear_model_locks():
    """Clear any locked model files"""
    models_dir = Path("./models")
    if models_dir.exists():
        # Remove final_model directories that might be locked
        for model_type in ["embedding", "reranker"]:
            final_model_path = models_dir / model_type / "final_model"
            if final_model_path.exists():
                try:
                    shutil.rmtree(final_model_path)
                    print(f"✅ Cleared locked {model_type} model")
                except Exception as e:
                    print(f"⚠️ Could not clear {model_type} model: {e}")

def quick_train():
    """Quick training with conflict resolution"""
    print("🚀 Starting Quick Training Pipeline...")
    
    # Clear any locked models
    clear_model_locks()
    
    # Wait a moment for file system
    time.sleep(2)
    
    # Train embedding model
    print("\n🤖 Training Embedding Model...")
    try:
        os.system("python scripts/train_embedding.py --epochs 2 --batch_size 16 --learning_rate 2e-5")
        print("✅ Embedding model training completed!")
    except Exception as e:
        print(f"❌ Embedding training failed: {e}")
    
    # Wait between trainings
    time.sleep(3)
    
    # Train reranker model
    print("\n🎯 Training Reranker Model...")
    try:
        os.system("python scripts/train_reranker.py --epochs 2 --batch_size 8 --learning_rate 2e-5")
        print("✅ Reranker model training completed!")
    except Exception as e:
        print(f"❌ Reranker training failed: {e}")
    
    print("\n🎉 Quick training pipeline completed!")
    print("You can now restart your Streamlit app!")

if __name__ == "__main__":
    quick_train()
