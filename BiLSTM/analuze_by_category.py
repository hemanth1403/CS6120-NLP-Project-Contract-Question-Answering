"""
Category-wise Performance Analysis for BiLSTM on CUAD
Analyzes BiLSTM performance across all 41 CUAD clause categories

This script loads your trained BiLSTM model and evaluates it on each category separately.
"""

import torch
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict


def load_test_data(test_path):
    """Load and organize test data by category"""
    with open(test_path, 'r') as f:
        data = json.load(f)
    
    # Organize by category (extract from question ID)
    category_data = defaultdict(list)
    
    for item in data['data']:
        for paragraph in item['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                # Extract category from question ID
                # Format: ContractName__CategoryName
                qa_id = qa['id']
                if '__' in qa_id:
                    category = qa_id.split('__')[1]
                else:
                    category = 'Unknown'
                
                category_data[category].append({
                    'id': qa_id,
                    'question': qa['question'],
                    'context': context,
                    'answers': qa.get('answers', []),
                    'is_impossible': qa.get('is_impossible', False)
                })
    
    return category_data


def compute_f1_score(prediction, ground_truth):
    """Compute F1 score between prediction and ground truth strings"""
    pred_tokens = prediction.lower().split()
    truth_tokens = ground_truth.lower().split()
    
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    
    if len(common_tokens) == 0:
        return 0
    
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(truth_tokens)
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def compute_exact_match(prediction, ground_truth):
    """Compute exact match between prediction and ground truth"""
    return int(prediction.strip().lower() == ground_truth.strip().lower())


def evaluate_category(examples, model, tokenizer, device='cpu'):
    """
    Evaluate model on a specific category
    
    Args:
        examples: List of examples for this category
        model: Trained BiLSTM model
        tokenizer: Tokenizer
        device: Device to run on
    
    Returns:
        Dictionary with metrics for this category
    """
    model.eval()
    
    total_f1 = 0.0
    total_em = 0.0
    total_examples = 0
    correct_answerability = 0
    
    predictions_list = []
    
    with torch.no_grad():
        for example in tqdm(examples, desc="Evaluating", leave=False):
            # Tokenize
            question_ids = tokenizer.encode(example['question'], max_len=64)
            context_ids = tokenizer.encode(example['context'], max_len=512)
            
            question_mask = [1] * len(question_ids)
            context_mask = [1] * len(context_ids)
            
            # Convert to tensors
            question_ids = torch.tensor([question_ids], dtype=torch.long).to(device)
            context_ids = torch.tensor([context_ids], dtype=torch.long).to(device)
            question_mask = torch.tensor([question_mask], dtype=torch.float).to(device)
            context_mask = torch.tensor([context_mask], dtype=torch.float).to(device)
            
            # Get predictions
            outputs = model(context_ids, question_ids, context_mask, question_mask)
            
            # Handle different return formats
            if isinstance(outputs, tuple) and len(outputs) == 3:
                start_logits, end_logits, ans_logits = outputs
            else:
                # If it's a different format, try to extract
                start_logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                end_logits = outputs[1] if isinstance(outputs, (tuple, list)) and len(outputs) > 1 else outputs
                ans_logits = outputs[2] if isinstance(outputs, (tuple, list)) and len(outputs) > 2 else None
            
            # Get predicted positions
            start_pos = torch.argmax(start_logits, dim=1).item()
            end_pos = torch.argmax(end_logits, dim=1).item()
            
            # Answerability prediction
            if ans_logits is not None:
                # Flatten to handle various tensor shapes: [batch, 1] or [1, 1] or [1]
                ans_score = torch.sigmoid(ans_logits.flatten()[0]).item()
                is_answerable_pred = ans_score > 0.5
                is_answerable_true = not example['is_impossible']
                
                if is_answerable_pred == is_answerable_true:
                    correct_answerability += 1
            else:
                # If no answerability logits, assume answerable
                is_answerable_pred = True
            
            # Extract predicted text
            if start_pos <= end_pos and end_pos < len(context_ids[0]):
                pred_tokens = context_ids[0][start_pos:end_pos+1].tolist()
                pred_text = tokenizer.decode(pred_tokens)
            else:
                pred_text = ""
            
            # Get ground truth
            if example['answers']:
                ground_truth = example['answers'][0]['text']
            else:
                ground_truth = ""
            
            # Initialize metrics
            f1 = 0.0
            em = 0.0
            
            # Compute metrics only for answerable questions
            if not example['is_impossible']:
                f1 = compute_f1_score(pred_text, ground_truth)
                em = compute_exact_match(pred_text, ground_truth)
                
                total_f1 += f1
                total_em += em
                total_examples += 1
            
            predictions_list.append({
                'id': example['id'],
                'prediction': pred_text,
                'ground_truth': ground_truth,
                'f1': f1 if not example['is_impossible'] else None,
                'em': em if not example['is_impossible'] else None
            })
    
    # Calculate averages
    avg_f1 = (total_f1 / total_examples * 100) if total_examples > 0 else 0.0
    avg_em = (total_em / total_examples * 100) if total_examples > 0 else 0.0
    answerability_acc = (correct_answerability / len(examples) * 100) if len(examples) > 0 else 0.0
    
    return {
        'count': len(examples),
        'answerable': sum(1 for ex in examples if not ex['is_impossible']),
        'f1': avg_f1,
        'em': avg_em,
        'answerability': answerability_acc,
        'predictions': predictions_list
    }


def analyze_all_categories(test_path, model_path, tokenizer_path, device='cpu'):
    """
    Analyze model performance across all categories
    
    Args:
        test_path: Path to test data JSON
        model_path: Path to saved model checkpoint
        tokenizer_path: Path to saved tokenizer
        device: Device to run on
    
    Returns:
        Dictionary with results for each category
    """
    print("Loading test data...")
    category_data = load_test_data(test_path)
    
    print(f"Found {len(category_data)} categories")
    print(f"Total examples: {sum(len(examples) for examples in category_data.values())}")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    import sys
    import os
    
    # Add parent directory to path to find modules
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # Try to import - handle different possible locations
    try:
        from cuad_dataloader import Tokenizer
    except ImportError:
        try:
            # If running from Project directory
            sys.path.insert(0, '/Users/hemu/Desktop/Desktop items/NU/Sem_2/NLP - siwu/Project')
            from cuad_dataloader import Tokenizer
        except ImportError:
            print("Error: Cannot find cuad_dataloader module")
            print("Please ensure cuad_dataloader.py is in the same directory or parent directory")
            raise
    
    tokenizer = Tokenizer(vocab_size=50000, max_len=512)
    tokenizer.load(tokenizer_path)
    
    # Load model
    print("Loading model...")
    try:
        from bilstm_attention_with_answerability import BiLSTMQAWithAnswerability
    except ImportError:
        print("Error: Cannot find bilstm_attention_with_answerability module")
        print("Please ensure bilstm_attention_with_answerability.py is in the same directory or parent directory")
        raise
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model = BiLSTMQAWithAnswerability(
        vocab_size=len(tokenizer.word2idx),
        embedding_dim=300,
        hidden_dim=256,
        num_layers=2,
        dropout=0.2
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    
    # Evaluate each category
    print("\nEvaluating categories...")
    results = {}
    
    for category, examples in tqdm(category_data.items(), desc="Categories"):
        print(f"\nEvaluating {category} ({len(examples)} examples)...")
        results[category] = evaluate_category(examples, model, tokenizer, device)
    
    return results


def print_results(results):
    """Print formatted results"""
    print("\n" + "="*100)
    print("BiLSTM Category Performance")
    print("="*100)
    print()
    print(f"{'Category':<35} {'Count':>8} {'EM':>10} {'F1':>10} {'Answerability':>15}")
    print("-"*100)
    
    # Sort by F1 score
    sorted_results = sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)
    
    for category, metrics in sorted_results:
        print(f"{category:<35} {metrics['count']:>8} {metrics['em']:>9.2f}% {metrics['f1']:>9.2f}% {metrics['answerability']:>14.2f}%")
    
    print("="*100)
    
    # Calculate overall statistics
    total_examples = sum(m['count'] for m in results.values())
    avg_em = np.mean([m['em'] for m in results.values()])
    avg_f1 = np.mean([m['f1'] for m in results.values()])
    avg_ans = np.mean([m['answerability'] for m in results.values()])
    
    print()
    print("Summary Statistics:")
    print(f"  Total Examples:     {total_examples:,}")
    print(f"  Total Categories:   {len(results)}")
    print(f"  Average EM:         {avg_em:.2f}%")
    print(f"  Average F1:         {avg_f1:.2f}%")
    print(f"  Average Answerability: {avg_ans:.2f}%")
    
    # Best and worst categories
    best_f1 = max(results.items(), key=lambda x: x[1]['f1'])
    worst_f1 = min(results.items(), key=lambda x: x[1]['f1'])
    
    print()
    print(f"Best Category:  {best_f1[0]} (F1: {best_f1[1]['f1']:.2f}%)")
    print(f"Worst Category: {worst_f1[0]} (F1: {worst_f1[1]['f1']:.2f}%)")


def main():
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze BiLSTM performance by category')
    parser.add_argument('--test_path', type=str, required=True, help='Path to test data JSON')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to tokenizer')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu/cuda)')
    parser.add_argument('--output', type=str, default='./bilstm_category_analysis.json', help='Output path')
    
    args = parser.parse_args()
    
    try:
        # Run analysis
        results = analyze_all_categories(
            args.test_path,
            args.model_path,
            args.tokenizer_path,
            args.device
        )
        
        # Print results
        print_results(results)
        
        # Save results
        output_data = {
            'original_questions': results
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✓ Results saved to: {args.output}")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()