"""
Quick startup script for the embedding and reranker fine-tuning demo.
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=False, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error: {e}")
        return False


def check_requirements():
    """Check if requirements are installed."""
    try:
        import torch
        import transformers
        import sentence_transformers
        import streamlit
        import datasets
        print("✅ All required packages are available!")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False


def main():
    parser = argparse.ArgumentParser(description='Run embedding and reranker demo')
    parser.add_argument('--mode', choices=['full', 'demo', 'train', 'eval'], 
                       default='demo',
                       help='Run mode: full (train+eval+demo), demo (just demo), train (just training), eval (just evaluation)')
    parser.add_argument('--max_samples', type=int, default=1000,
                       help='Maximum samples for training')
    parser.add_argument('--epochs', type=int, default=2,
                       help='Training epochs')
    parser.add_argument('--skip_data_prep', action='store_true',
                       help='Skip data preparation step')
    
    args = parser.parse_args()
    
    print("🔍 Embedding & Reranker Fine-tuning Demo")
    print("="*50)
    
    # Check requirements
    if not check_requirements():
        return
    
    # Create directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("models/embedding", exist_ok=True)
    os.makedirs("models/reranker", exist_ok=True)
    os.makedirs("evaluation_results", exist_ok=True)
    os.makedirs("configs", exist_ok=True)
    
    success = True
    
    if args.mode in ['full', 'train']:
        # Step 1: Prepare data
        if not args.skip_data_prep:
            success &= run_command(
                f"python scripts/prepare_data.py --max_samples {args.max_samples * 2} --eval_queries 200",
                "Data Preparation"
            )
        
        if success:
            # Step 2: Train embedding model
            success &= run_command(
                f"python scripts/train_embedding.py --epochs {args.epochs} --max_samples {args.max_samples} --batch_size 16",
                "Embedding Model Training"
            )
        
        if success:
            # Step 3: Train reranker model
            success &= run_command(
                f"python scripts/train_reranker.py --epochs {args.epochs} --max_samples {args.max_samples} --batch_size 8",
                "Reranker Model Training"
            )
    
    if args.mode in ['full', 'eval']:
        if success:
            # Step 4: Run evaluation
            success &= run_command(
                "python scripts/evaluate.py --num_queries 100",
                "Model Evaluation"
            )
    
    if args.mode in ['full', 'demo']:
        if success or args.mode == 'demo':
            # Step 5: Launch demo
            print(f"\n{'='*50}")
            print("🚀 Launching Streamlit Demo")
            print("The demo will open in your browser automatically.")
            print("If it doesn't open, go to: http://localhost:8501")
            print(f"{'='*50}")
            
            try:
                subprocess.run("streamlit run demo/app.py", shell=True, check=True)
            except KeyboardInterrupt:
                print("\n👋 Demo stopped by user")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to launch demo: {e}")
                print("You can manually run: streamlit run demo/app.py")
    
    if not success:
        print("\n❌ Some steps failed. Check the error messages above.")
        print("You can still run the demo with base models by running:")
        print("streamlit run demo/app.py")
    else:
        print("\n✅ All steps completed successfully!")


if __name__ == "__main__":
    main()
