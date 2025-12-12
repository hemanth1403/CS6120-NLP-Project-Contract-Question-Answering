#!/usr/bin/env python3
"""
Simple wrapper to run category analysis with correct paths
"""

import sys
import os

# Add Project directory to Python path
PROJECT_DIR = '/Users/hemu/Desktop/Desktop items/NU/Sem_2/NLP - siwu/Project'
sys.path.insert(0, PROJECT_DIR)

# Now import and run the analysis
from analuze_by_category import analyze_all_categories, print_results
import json

# Your paths
TEST_PATH = '/Users/hemu/Desktop/Desktop items/NU/Sem_2/NLP - siwu/Project/data/test.json'
MODEL_PATH = '/Users/hemu/Desktop/Desktop items/NU/Sem_2/NLP - siwu/Project/outputs/bilstm_answerability_20251202_203753/best_model.pt'
TOKENIZER_PATH = '/Users/hemu/Desktop/Desktop items/NU/Sem_2/NLP - siwu/Project/outputs/bilstm_answerability_20251202_203753/tokenizer.json'
OUTPUT_PATH = './bilstm_category_analysis.json'

print("="*80)
print("BiLSTM Category Analysis")
print("="*80)
print(f"\nTest data: {TEST_PATH}")
print(f"Model: {MODEL_PATH}")
print(f"Tokenizer: {TOKENIZER_PATH}")
print(f"Output: {OUTPUT_PATH}")
print()

try:
    # Run analysis
    results = analyze_all_categories(
        TEST_PATH,
        MODEL_PATH,
        TOKENIZER_PATH,
        device='cpu'
    )
    
    # Print results
    print_results(results)
    
    # Save results
    output_data = {
        'original_questions': results
    }
    
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Results saved to: {OUTPUT_PATH}")
    
except Exception as e:
    print(f"✗ Error: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)