"""
READY-TO-USE: BiLSTM Clause Category Analysis
==============================================
This script is ready for you to plug in your BiLSTM model predictions.
Simply replace the marked sections with your model code.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from datasets import load_dataset
from tqdm import tqdm

# ============================================================================
# SECTION 1: Evaluation Metrics (Keep as-is)
# ============================================================================

def normalize_answer(s):
    """Normalize answer text for comparison"""
    import re
    import string
    
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction, truth):
    """Compute exact match score"""
    return int(normalize_answer(prediction) == normalize_answer(truth))

def compute_f1(prediction, truth):
    """Compute F1 score"""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(truth).split()
    
    if len(pred_tokens) == 0 and len(truth_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return 0.0
    
    common = set(pred_tokens) & set(truth_tokens)
    num_common = sum(min(pred_tokens.count(t), truth_tokens.count(t)) for t in common)
    
    if num_common == 0:
        return 0.0
    
    precision = num_common / len(pred_tokens)
    recall = num_common / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1

# ============================================================================
# SECTION 2: Your Clause Category Analysis Function (Keep as-is)
# ============================================================================

def analyze_by_clause_category(dataset, predictions):
    """Analyze performance by legal clause category"""
    clause_categories = {
        'Document Name': ['document name'],
        'Parties': ['parties'],
        'Agreement Date': ['agreement date', 'effective date'],
        'Expiration Date': ['expiration date', 'renewal term'],
        'Governing Law': ['governing law'],
        'Termination': ['termination', 'can be terminated'],
        'IP Rights': ['intellectual property', 'ip ownership'],
        'Confidentiality': ['confidential information', 'confidentiality'],
        'Liability': ['liability', 'cap on liability'],
        'Payment Terms': ['payment', 'price', 'cost'],
        'Non-Compete': ['non-compete', 'competitive restriction'],
        'Insurance': ['insurance'],
        'Warranties': ['warranties', 'representations'],
        'Indemnification': ['indemnification', 'indemnify'],
        'Audit Rights': ['audit', 'auditing'],
    }

    category_results = defaultdict(lambda: {'em': [], 'f1': [], 'count': 0})

    for i, example in enumerate(dataset):
        question = example['question'].lower()
        truth = example['answers']['text'][0] if example['answers']['text'] and len(example['answers']['text']) > 0 else ""
        pred = predictions[i]

        category = 'Other'
        for cat_name, keywords in clause_categories.items():
            if any(kw in question for kw in keywords):
                category = cat_name
                break

        em = compute_exact_match(pred, truth)
        f1 = compute_f1(pred, truth)

        category_results[category]['em'].append(em)
        category_results[category]['f1'].append(f1)
        category_results[category]['count'] += 1

    results = []
    for category, metrics in category_results.items():
        results.append({
            'Category': category,
            'Count': metrics['count'],
            'EM': np.mean(metrics['em']) * 100,
            'F1': np.mean(metrics['f1']) * 100
        })

    df = pd.DataFrame(results).sort_values('F1', ascending=False)
    return df

# ============================================================================
# SECTION 3: MODEL LOADING - REPLACE THIS WITH YOUR CODE
# ============================================================================

def load_your_bilstm_model():
    """
    TODO: Replace this function with your actual model loading code
    
    Example structure:
    """
    import torch
    # from your_model_file import BiLSTMQA
    
    # Load checkpoint
    # checkpoint = torch.load('path/to/best_model.pt', map_location='cpu')
    
    # Initialize model
    # model = BiLSTMQA(
    #     vocab_size=checkpoint['vocab_size'],
    #     embedding_dim=checkpoint['embedding_dim'],
    #     hidden_dim=checkpoint['hidden_dim'],
    #     ...
    # )
    
    # Load weights
    # model.load_state_dict(checkpoint['model_state_dict'])
    # model.eval()
    
    # return model, checkpoint['tokenizer']
    
    print("⚠ WARNING: Using placeholder. Replace load_your_bilstm_model() with your code!")
    return None, None

def generate_predictions(model, tokenizer, test_dataset):
    """
    TODO: Replace this function with your actual prediction generation code
    
    Example structure:
    """
    predictions = []
    
    # with torch.no_grad():
    #     for example in tqdm(test_dataset, desc="Generating predictions"):
    #         context = example['context']
    #         question = example['question']
    #         
    #         # Tokenize
    #         context_ids = tokenizer.encode(context)
    #         question_ids = tokenizer.encode(question)
    #         
    #         # Get prediction
    #         start_logits, end_logits, answerability = model(context_ids, question_ids)
    #         
    #         # Decode
    #         if answerability > 0.5:
    #             start = torch.argmax(start_logits).item()
    #             end = torch.argmax(end_logits).item()
    #             pred_text = tokenizer.decode(context_ids[start:end+1])
    #         else:
    #             pred_text = ""
    #         
    #         predictions.append(pred_text)
    
    print("⚠ WARNING: Using placeholder. Replace generate_predictions() with your code!")
    return [""] * len(test_dataset)

# ============================================================================
# ALTERNATIVE: Load Pre-saved Predictions
# ============================================================================

def load_saved_predictions(filepath):
    """
    If you've already generated predictions, load them here
    
    Supports: .pkl, .json, .txt (one per line)
    """
    import pickle
    import json
    
    if filepath.endswith('.pkl'):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    elif filepath.endswith('.json'):
        with open(filepath, 'r') as f:
            return json.load(f)
    elif filepath.endswith('.txt'):
        with open(filepath, 'r') as f:
            return [line.strip() for line in f]
    else:
        raise ValueError(f"Unsupported file format: {filepath}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*80)
    print("BiLSTM CLAUSE CATEGORY ANALYSIS")
    print("="*80)
    print()
    
    # Load test dataset
    print("Loading CUAD test dataset...")
    test_dataset = load_dataset("cuad", split="test")
    print(f"✓ Loaded {len(test_dataset)} test examples")
    print()
    
    # OPTION 1: Generate predictions from model
    print("="*80)
    print("CHOOSE PREDICTION METHOD:")
    print("="*80)
    print("Option 1: Generate predictions from BiLSTM model")
    print("Option 2: Load pre-saved predictions from file")
    print()
    
    use_saved = input("Use saved predictions? (y/n): ").lower().strip() == 'y'
    
    if use_saved:
        filepath = input("Enter path to predictions file (.pkl/.json/.txt): ").strip()
        predictions = load_saved_predictions(filepath)
        print(f"✓ Loaded {len(predictions)} predictions from {filepath}")
    else:
        print("Loading BiLSTM model...")
        model, tokenizer = load_your_bilstm_model()
        
        if model is None:
            print("\n" + "="*80)
            print("ERROR: Model loading not implemented yet!")
            print("="*80)
            print("\nPlease edit this script and replace:")
            print("  1. load_your_bilstm_model() function")
            print("  2. generate_predictions() function")
            print("\nOr use Option 2 to load pre-saved predictions.")
            print("="*80)
            return
        
        print("Generating predictions...")
        predictions = generate_predictions(model, tokenizer, test_dataset)
    
    print()
    
    # Validate predictions
    if len(predictions) != len(test_dataset):
        print(f"ERROR: Mismatch in lengths!")
        print(f"  Dataset: {len(test_dataset)} examples")
        print(f"  Predictions: {len(predictions)} predictions")
        return
    
    # Run clause category analysis
    print("="*80)
    print("RUNNING CLAUSE CATEGORY ANALYSIS")
    print("="*80)
    print()
    
    results_df = analyze_by_clause_category(test_dataset, predictions)
    
    # Display results
    print("\n" + "="*80)
    print("RESULTS BY CLAUSE CATEGORY")
    print("="*80)
    print()
    print(results_df.to_string(index=False))
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Overall Average F1:     {results_df['F1'].mean():.2f}%")
    print(f"Overall Average EM:     {results_df['EM'].mean():.2f}%")
    print(f"Best Category (F1):     {results_df.iloc[0]['Category']} ({results_df.iloc[0]['F1']:.2f}%)")
    print(f"Worst Category (F1):    {results_df.iloc[-1]['Category']} ({results_df.iloc[-1]['F1']:.2f}%)")
    print(f"Std Dev F1:             {results_df['F1'].std():.2f}%")
    print(f"Std Dev EM:             {results_df['EM'].std():.2f}%")
    
    # Category insights
    print("\n" + "="*80)
    print("CATEGORY INSIGHTS")
    print("="*80)
    
    high_performers = results_df[results_df['F1'] > results_df['F1'].mean()]
    low_performers = results_df[results_df['F1'] < results_df['F1'].mean()]
    
    print(f"\nHigh-performing categories (above average):")
    for _, row in high_performers.iterrows():
        print(f"  • {row['Category']:20s} - F1: {row['F1']:5.1f}% (n={row['Count']})")
    
    print(f"\nLow-performing categories (below average):")
    for _, row in low_performers.iterrows():
        print(f"  • {row['Category']:20s} - F1: {row['F1']:5.1f}% (n={row['Count']})")
    
    # EM vs F1 gap analysis
    print("\n" + "="*80)
    print("EM-F1 GAP ANALYSIS")
    print("="*80)
    results_df['Gap'] = results_df['F1'] - results_df['EM']
    large_gap = results_df.nlargest(3, 'Gap')
    
    print("\nCategories with largest F1-EM gap (partial matches):")
    for _, row in large_gap.iterrows():
        print(f"  • {row['Category']:20s} - Gap: {row['Gap']:5.1f}% (F1={row['F1']:.1f}%, EM={row['EM']:.1f}%)")
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    output_file = 'bilstm_clause_category_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"✓ Saved results to: {output_file}")
    
    # Generate summary report
    with open('bilstm_clause_summary.txt', 'w') as f:
        f.write("BiLSTM CLAUSE CATEGORY ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Overall F1: {results_df['F1'].mean():.2f}%\n")
        f.write(f"Overall EM: {results_df['EM'].mean():.2f}%\n\n")
        f.write("Top 5 Categories:\n")
        for _, row in results_df.head(5).iterrows():
            f.write(f"  {row['Category']:20s} - F1: {row['F1']:5.1f}%\n")
        f.write("\nBottom 5 Categories:\n")
        for _, row in results_df.tail(5).iterrows():
            f.write(f"  {row['Category']:20s} - F1: {row['F1']:5.1f}%\n")
    
    print(f"✓ Saved summary to: bilstm_clause_summary.txt")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Review the results CSV for detailed breakdown")
    print("  2. Identify categories needing improvement")
    print("  3. Compare with BERT results when available")
    print("  4. Use insights to guide RAG implementation")
    print("="*80)

if __name__ == "__main__":
    main()