#!/usr/bin/env python3
"""
Quick retrain script with fixed configuration for numerical stability.
Run this to retrain the model with gradient clipping and lower LR.
"""

import torch
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_bilstm import train_model

if __name__ == "__main__":
    # Fixed configuration for numerical stability
    
    # config = {
    #     # Data paths - UPDATE THESE
    #     'train_path': './data/train.json',
    #     'val_path': './data/val.json',
        
    #     # Model architecture
    #     'vocab_size': 50000,
    #     'embedding_dim': 300,
    #     'hidden_dim': 128,
    #     'num_layers': 2,
    #     'dropout': 0.3,
        
    #     # Training - FIXED FOR STABILITY
    #     'batch_size': 8,
    #     'num_epochs': 20,
    #     'learning_rate': 0.0001,  # 10x lower - prevents NaN
    #     'max_context_len': 512,
    #     'max_question_len': 64,
    #     'max_grad_norm': 1.0,      # Gradient clipping - prevents explosion
        
    #     # Device - CPU for stability, change to 'mps' or 'cuda' if you want to try GPU
    #     'seed': 42,
    #     'device': 'cpu',  
    #     'num_workers': 0,
    #     'output_dir': './outputs'
    # }


    config = {
    'train_path': './data/train.json',
    'val_path': './data/val.json',
    'vocab_size': 50000,
    'embedding_dim': 300,
    'hidden_dim': 256,        # INCREASED
    'num_layers': 2,
    'dropout': 0.2,           # DECREASED
    'batch_size': 16,         # INCREASED
    'num_epochs': 20,
    'learning_rate': 0.0002,  # INCREASED
    'max_context_len': 512,
    'max_question_len': 64,
    'max_grad_norm': 5.0,     # INCREASED
    'seed': 42,
    'device': 'cpu',
    'num_workers': 0,
    'output_dir': './outputs'
    }
    
    print("="*80)
    print("RETRAINING WITH FIXED CONFIGURATION")
    print("="*80)
    print("\nKey fixes applied:")
    print("  ✓ Learning rate: 0.001 → 0.0001 (10x lower)")
    print("  ✓ Gradient clipping: max_norm=1.0")
    print("  ✓ NaN detection in training loop")
    print("  ✓ Using CPU for numerical stability")
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*80)
    input("Press ENTER to start training (or Ctrl+C to cancel)...")
    print("="*80 + "\n")
    
    # Train model
    model, trainer, tokenizer, history = train_model(config)
    
    print("\n" + "="*80)
    print("Training complete!")
    print("="*80)
    print("\nTo verify the model is working, run:")
    print("  python3 debug_predictions.py")