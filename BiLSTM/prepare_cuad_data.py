"""
Script to prepare CUAD dataset for training.
Handles the actual CUAD data structure and creates train/val split.
"""

import json
import os
from sklearn.model_selection import train_test_split
import argparse


def create_train_val_split(cuad_path, output_dir, val_size=0.2, random_state=42):
    """
    Create train/validation split from CUADv1.json
    
    Args:
        cuad_path: Path to CUADv1.json
        output_dir: Directory to save train.json and val.json
        val_size: Fraction of data to use for validation
        random_state: Random seed for reproducibility
    """
    print(f"Loading CUAD data from {cuad_path}...")
    with open(cuad_path, 'r') as f:
        data = json.load(f)
    
    print(f"Dataset version: {data.get('version', 'unknown')}")
    print(f"Total articles: {len(data['data'])}")
    
    # Count total QA pairs
    total_qas = 0
    for article in data['data']:
        for paragraph in article['paragraphs']:
            total_qas += len(paragraph['qas'])
    print(f"Total QA pairs: {total_qas}")
    
    # Split articles into train/val
    articles = data['data']
    train_articles, val_articles = train_test_split(
        articles, 
        test_size=val_size, 
        random_state=random_state
    )
    
    print(f"\nSplit summary:")
    print(f"  Training articles: {len(train_articles)}")
    print(f"  Validation articles: {len(val_articles)}")
    
    # Count QA pairs in each split
    train_qas = sum(len(qa) for article in train_articles 
                    for paragraph in article['paragraphs'] 
                    for qa in [paragraph['qas']])
    val_qas = sum(len(qa) for article in val_articles 
                  for paragraph in article['paragraphs'] 
                  for qa in [paragraph['qas']])
    
    print(f"  Training QA pairs: {train_qas}")
    print(f"  Validation QA pairs: {val_qas}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save train split
    train_data = {
        'data': train_articles, 
        'version': data.get('version', 'CUADv1')
    }
    train_path = os.path.join(output_dir, 'train.json')
    with open(train_path, 'w') as f:
        json.dump(train_data, f)
    print(f"\n✓ Saved training data to {train_path}")
    
    # Save validation split
    val_data = {
        'data': val_articles, 
        'version': data.get('version', 'CUADv1')
    }
    val_path = os.path.join(output_dir, 'val.json')
    with open(val_path, 'w') as f:
        json.dump(val_data, f)
    print(f"✓ Saved validation data to {val_path}")
    
    return train_path, val_path


def analyze_dataset(json_path):
    """
    Analyze CUAD dataset structure and statistics.
    """
    print(f"\n{'='*60}")
    print(f"Analyzing: {os.path.basename(json_path)}")
    print(f"{'='*60}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    articles = data['data']
    print(f"\nArticles: {len(articles)}")
    
    # Collect statistics
    total_qas = 0
    answerable_qas = 0
    unanswerable_qas = 0
    context_lengths = []
    question_lengths = []
    answer_lengths = []
    
    clause_types = {}
    
    for article in articles:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            context_lengths.append(len(context.split()))
            
            for qa in paragraph['qas']:
                total_qas += 1
                question = qa['question']
                question_lengths.append(len(question.split()))
                
                # Track clause types from question
                # CUAD questions typically start with "Highlight the parts..."
                if 'question' in qa:
                    q_text = qa['question']
                    # Extract clause type (this is approximate)
                    clause_type = q_text.split('that')[0].strip() if 'that' in q_text else 'unknown'
                    clause_types[clause_type] = clause_types.get(clause_type, 0) + 1
                
                is_impossible = qa.get('is_impossible', False)
                if is_impossible or not qa.get('answers'):
                    unanswerable_qas += 1
                else:
                    answerable_qas += 1
                    if qa['answers']:
                        answer_text = qa['answers'][0]['text']
                        answer_lengths.append(len(answer_text.split()))
    
    print(f"\nQA Pairs:")
    print(f"  Total: {total_qas}")
    print(f"  Answerable: {answerable_qas} ({answerable_qas/total_qas*100:.1f}%)")
    print(f"  Unanswerable: {unanswerable_qas} ({unanswerable_qas/total_qas*100:.1f}%)")
    
    print(f"\nContext Length (words):")
    print(f"  Mean: {sum(context_lengths)/len(context_lengths):.0f}")
    print(f"  Min: {min(context_lengths)}")
    print(f"  Max: {max(context_lengths)}")
    print(f"  Median: {sorted(context_lengths)[len(context_lengths)//2]}")
    
    print(f"\nQuestion Length (words):")
    print(f"  Mean: {sum(question_lengths)/len(question_lengths):.1f}")
    print(f"  Min: {min(question_lengths)}")
    print(f"  Max: {max(question_lengths)}")
    
    if answer_lengths:
        print(f"\nAnswer Length (words):")
        print(f"  Mean: {sum(answer_lengths)/len(answer_lengths):.1f}")
        print(f"  Min: {min(answer_lengths)}")
        print(f"  Max: {max(answer_lengths)}")
        print(f"  Median: {sorted(answer_lengths)[len(answer_lengths)//2]}")
    
    print(f"\nTop 10 Most Common Question Types:")
    sorted_clause_types = sorted(clause_types.items(), key=lambda x: x[1], reverse=True)[:10]
    for clause_type, count in sorted_clause_types:
        print(f"  {count:4d} | {clause_type[:70]}...")


def main():
    parser = argparse.ArgumentParser(description='Prepare CUAD dataset for training')
    parser.add_argument('--cuad_file', type=str, default='CUADv1.json',
                       help='Path to CUADv1.json file')
    parser.add_argument('--output_dir', type=str, default='./data',
                       help='Output directory for train/val splits')
    parser.add_argument('--val_size', type=float, default=0.2,
                       help='Validation set size (default: 0.2)')
    parser.add_argument('--analyze', action='store_true',
                       help='Run dataset analysis')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for train/val split')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.cuad_file):
        print(f"Error: File not found: {args.cuad_file}")
        print("\nPlease provide the correct path to CUADv1.json")
        print("Example: python prepare_cuad_data.py --cuad_file /path/to/CUADv1.json")
        return
    
    # Analyze original dataset
    if args.analyze:
        analyze_dataset(args.cuad_file)
    
    # Create train/val split
    print(f"\n{'='*60}")
    print("Creating Train/Validation Split")
    print(f"{'='*60}")
    
    train_path, val_path = create_train_val_split(
        args.cuad_file,
        args.output_dir,
        val_size=args.val_size,
        random_state=args.random_seed
    )
    
    # Analyze splits
    if args.analyze:
        analyze_dataset(train_path)
        analyze_dataset(val_path)
    
    print(f"\n{'='*60}")
    print("✓ Data preparation complete!")
    print(f"{'='*60}")
    print(f"\nTo train your model, update train_bilstm.py with these paths:")
    print(f"  'train_path': '{train_path}',")
    print(f"  'val_path': '{val_path}',")


if __name__ == "__main__":
    main()