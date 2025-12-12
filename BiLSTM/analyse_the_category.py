"""
Category-wise Performance Analysis for BiLSTM
Grade Contract Milestone A.2: Deeper performance analysis

This script analyzes BiLSTM performance across all 41 CUAD clause categories
to identify strengths, weaknesses, and patterns.
"""

import torch
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from bilstm_attention_with_answerability import BiLSTMQAWithAnswerability, BiLSTMQATrainerWithAnswerability
from cuad_dataloader import Tokenizer


class CategoryPerformanceAnalyzer:
    """Analyze model performance by clause category"""
    
    def __init__(self, model_path, tokenizer_path, device='cpu'):
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = Tokenizer(vocab_size=50000, max_len=512)
        self.tokenizer.load(tokenizer_path)
        
        # Load model
        print("Loading model...")
        checkpoint = torch.load(model_path, weights_only=False)
        
        self.model = BiLSTMQAWithAnswerability(
            vocab_size=len(self.tokenizer.word2idx),
            embedding_dim=300,
            hidden_dim=256,
            num_layers=2,
            dropout=0.2
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.trainer = BiLSTMQATrainerWithAnswerability(self.model, device=device)
        self.model.eval()
        self.device = device
        
        print(f"Model loaded from epoch {checkpoint['epoch']}")
    
    def extract_category_from_id(self, qa_id):
        """
        Extract category from question ID
        Format: ContractName__CategoryName
        """
        if '__' in qa_id:
            return qa_id.split('__')[1]
        return 'Unknown'
    
    def preprocess_example(self, context, question, max_context_len=512, max_question_len=64):
        """Preprocess single example"""
        question_ids = self.tokenizer.encode(question, max_question_len)
        context_ids = self.tokenizer.encode(context, max_context_len)
        
        question_mask = [1] * len(question_ids)
        context_mask = [1] * len(context_ids)
        
        question_ids += [0] * (max_question_len - len(question_ids))
        question_mask += [0] * (max_question_len - len(question_mask))
        context_ids += [0] * (max_context_len - len(context_ids))
        context_mask += [0] * (max_context_len - len(context_mask))
        
        return {
            'question_ids': torch.tensor([question_ids]),
            'context_ids': torch.tensor([context_ids]),
            'question_mask': torch.tensor([question_mask]),
            'context_mask': torch.tensor([context_mask])
        }
    
    def predict(self, context, question):
        """Make prediction on single example"""
        batch = self.preprocess_example(context, question)
        
        with torch.no_grad():
            question_ids = batch['question_ids'].to(self.device)
            context_ids = batch['context_ids'].to(self.device)
            question_mask = batch['question_mask'].to(self.device)
            context_mask = batch['context_mask'].to(self.device)
            
            start_logits, end_logits, answerability_logits, _ = self.model(
                question_ids, context_ids, question_mask, context_mask
            )
            
            start_pred = torch.argmax(start_logits, dim=-1).item()
            end_pred = torch.argmax(end_logits, dim=-1).item()
            answerability_pred = torch.argmax(answerability_logits, dim=-1).item()
            
            if answerability_pred == 1:  # unanswerable
                return "", 0, 0, True
            
            context_words = context.lower().split()
            if 0 <= start_pred < len(context_words) and 0 <= end_pred < len(context_words) and start_pred <= end_pred:
                answer_text = ' '.join(context_words[start_pred:end_pred+1])
            else:
                answer_text = ""
            
            return answer_text, start_pred, end_pred, False
    
    def compute_f1(self, prediction, ground_truth):
        """Compute token-level F1"""
        pred_tokens = set(prediction.lower().split())
        true_tokens = set(ground_truth.lower().split())
        
        if len(pred_tokens) == 0 and len(true_tokens) == 0:
            return 1.0
        elif len(pred_tokens) == 0 or len(true_tokens) == 0:
            return 0.0
        
        common = pred_tokens & true_tokens
        if len(common) == 0:
            return 0.0
        
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(true_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        
        return f1
    
    def analyze_by_category(self, data_path, use_refactored=False):
        """
        Analyze performance by clause category
        
        Returns:
            dict: Category -> {count, em, f1, answerability_acc}
        """
        print(f"\nAnalyzing by category ({'refactored' if use_refactored else 'original'} questions)...")
        
        # Load data
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # Initialize category metrics
        category_metrics = defaultdict(lambda: {
            'count': 0,
            'f1_scores': [],
            'em_scores': [],
            'answerability_correct': 0
        })
        
        # Process each example
        for article in tqdm(data['data'], desc="Processing contracts"):
            for paragraph in article['paragraphs']:
                context = paragraph.get('context', '')
                
                for qa in paragraph['qas']:
                    # Extract category
                    category = self.extract_category_from_id(qa['id'])
                    
                    # Choose question
                    if use_refactored and 'refactored_question' in qa:
                        question = qa['refactored_question']
                    else:
                        question = qa['question']
                    
                    # Ground truth
                    is_impossible = qa.get('is_impossible', False)
                    if is_impossible or not qa.get('answers'):
                        true_answer = ""
                    else:
                        true_answer = qa['answers'][0]['text']
                    
                    # Predict
                    pred_answer, _, _, pred_unanswerable = self.predict(context, question)
                    
                    # Compute metrics
                    f1 = self.compute_f1(pred_answer, true_answer)
                    em = 1.0 if pred_answer.strip().lower() == true_answer.strip().lower() else 0.0
                    
                    # Answerability
                    ans_correct = (pred_unanswerable and is_impossible) or (not pred_unanswerable and not is_impossible)
                    
                    # Store
                    category_metrics[category]['count'] += 1
                    category_metrics[category]['f1_scores'].append(f1)
                    category_metrics[category]['em_scores'].append(em)
                    if ans_correct:
                        category_metrics[category]['answerability_correct'] += 1
        
        # Compute averages
        category_results = {}
        for category, metrics in category_metrics.items():
            if metrics['count'] > 0:
                category_results[category] = {
                    'count': metrics['count'],
                    'f1': np.mean(metrics['f1_scores']) * 100,
                    'em': np.mean(metrics['em_scores']) * 100,
                    'answerability_acc': (metrics['answerability_correct'] / metrics['count']) * 100
                }
        
        return category_results
    
    def print_category_results(self, results, title="Category Performance"):
        """Print formatted category results"""
        print("\n" + "="*80)
        print(title.center(80))
        print("="*80)
        
        # Sort by F1 score (descending)
        sorted_categories = sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)
        
        # Print header
        print(f"\n{'Category':<25} {'Count':>8} {'EM':>12} {'F1':>12} {'Ans.Acc':>12}")
        print("-"*80)
        
        # Print results
        for category, metrics in sorted_categories:
            print(f"{category:<25} {metrics['count']:>8} "
                  f"{metrics['em']:>11.2f}% {metrics['f1']:>11.2f}% "
                  f"{metrics['answerability_acc']:>11.2f}%")
        
        # Print summary
        print("-"*80)
        all_f1 = [m['f1'] for m in results.values()]
        all_em = [m['em'] for m in results.values()]
        all_ans = [m['answerability_acc'] for m in results.values()]
        
        print(f"{'AVERAGE':<25} {sum(m['count'] for m in results.values()):>8} "
              f"{np.mean(all_em):>11.2f}% {np.mean(all_f1):>11.2f}% "
              f"{np.mean(all_ans):>11.2f}%")
        print("="*80)
    
    def save_results_latex(self, results, output_path):
        """Save results as LaTeX table"""
        sorted_categories = sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)
        
        latex = []
        latex.append("\\begin{table}[H]")
        latex.append("\\centering")
        latex.append("\\caption{BiLSTM Performance by Clause Category}")
        latex.append("\\begin{tabular}{lrrrr}")
        latex.append("\\toprule")
        latex.append("\\textbf{Category} & \\textbf{Count} & \\textbf{EM (\\%)} & \\textbf{F1 (\\%)} & \\textbf{Ans. Acc. (\\%)} \\\\")
        latex.append("\\midrule")
        
        for category, metrics in sorted_categories:
            latex.append(f"{category.replace('_', ' ')} & {metrics['count']} & "
                        f"{metrics['em']:.2f} & {metrics['f1']:.2f} & "
                        f"{metrics['answerability_acc']:.2f} \\\\")
        
        latex.append("\\midrule")
        all_f1 = [m['f1'] for m in results.values()]
        all_em = [m['em'] for m in results.values()]
        all_ans = [m['answerability_acc'] for m in results.values()]
        total_count = sum(m['count'] for m in results.values())
        
        latex.append(f"\\textbf{{Average}} & {total_count} & "
                    f"{np.mean(all_em):.2f} & {np.mean(all_f1):.2f} & "
                    f"{np.mean(all_ans):.2f} \\\\")
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\label{tab:category_performance}")
        latex.append("\\end{table}")
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(latex))
        
        print(f"\n✓ LaTeX table saved to: {output_path}")


def main():
    # Configuration
    model_path = './outputs/bilstm_answerability_20251202_203753/best_model.pt'
    tokenizer_path = './outputs/bilstm_answerability_20251202_203753/tokenizer.json'
    test_path = '/Users/hemu/Desktop/Desktop items/NU/Sem_2/NLP - siwu/Project/Refactored Data/test_refactored.json'
    
    print("="*80)
    print("BiLSTM CATEGORY-WISE PERFORMANCE ANALYSIS")
    print("Grade Contract Milestone A.2: Deeper performance analysis")
    print("="*80)
    
    # Initialize analyzer
    analyzer = CategoryPerformanceAnalyzer(model_path, tokenizer_path, device='cpu')
    
    # Analyze on original questions
    print("\n1. Analyzing performance on ORIGINAL legal questions...")
    original_results = analyzer.analyze_by_category(test_path, use_refactored=False)
    analyzer.print_category_results(original_results, "BiLSTM Performance by Category (Original Questions)")
    
    # Analyze on refactored questions
    print("\n2. Analyzing performance on REFACTORED paraphrased questions...")
    refactored_results = analyzer.analyze_by_category(test_path, use_refactored=True)
    analyzer.print_category_results(refactored_results, "BiLSTM Performance by Category (Refactored Questions)")
    
    # Compare robustness by category
    print("\n" + "="*80)
    print("ROBUSTNESS BY CATEGORY (Original → Refactored)")
    print("="*80)
    print(f"\n{'Category':<25} {'F1 Drop':>12} {'EM Drop':>12} {'Ans. Drop':>12}")
    print("-"*80)
    
    robustness_data = []
    for category in original_results.keys():
        if category in refactored_results:
            f1_drop = original_results[category]['f1'] - refactored_results[category]['f1']
            em_drop = original_results[category]['em'] - refactored_results[category]['em']
            ans_drop = original_results[category]['answerability_acc'] - refactored_results[category]['answerability_acc']
            
            robustness_data.append({
                'category': category,
                'f1_drop': f1_drop,
                'em_drop': em_drop,
                'ans_drop': ans_drop
            })
            
            print(f"{category:<25} {f1_drop:>11.2f}% {em_drop:>11.2f}% {ans_drop:>11.2f}%")
    
    # Identify most/least robust categories
    robustness_data.sort(key=lambda x: abs(x['f1_drop']))
    
    print("\n" + "="*80)
    print("Most Robust Categories (smallest F1 drop):")
    for item in robustness_data[:5]:
        print(f"  {item['category']:<25} F1 drop: {item['f1_drop']:>6.2f}%")
    
    print("\nLeast Robust Categories (largest F1 drop):")
    for item in robustness_data[-5:]:
        print(f"  {item['category']:<25} F1 drop: {item['f1_drop']:>6.2f}%")
    print("="*80)
    
    # Save all results
    output_data = {
        'original_questions': original_results,
        'refactored_questions': refactored_results,
        'robustness_by_category': robustness_data
    }
    
    output_file = './outputs/bilstm_category_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\n✓ Complete results saved to: {output_file}")
    
    # Generate LaTeX tables
    analyzer.save_results_latex(original_results, './outputs/bilstm_category_table_original.tex')
    analyzer.save_results_latex(refactored_results, './outputs/bilstm_category_table_refactored.tex')
    
    # Generate summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print("\nPerformance Range (Original Questions):")
    f1_values = [m['f1'] for m in original_results.values()]
    print(f"  Best F1:  {max(f1_values):.2f}% ({[k for k,v in original_results.items() if v['f1'] == max(f1_values)][0]})")
    print(f"  Worst F1: {min(f1_values):.2f}% ({[k for k,v in original_results.items() if v['f1'] == min(f1_values)][0]})")
    print(f"  Mean F1:  {np.mean(f1_values):.2f}%")
    print(f"  Std F1:   {np.std(f1_values):.2f}%")
    
    print("\nAnswerability Accuracy Range:")
    ans_values = [m['answerability_acc'] for m in original_results.values()]
    print(f"  Best:  {max(ans_values):.2f}%")
    print(f"  Worst: {min(ans_values):.2f}%")
    print(f"  Mean:  {np.mean(ans_values):.2f}%")
    
    print("\n✓ Category-wise analysis complete!")
    print("\nGrade Contract Milestone A.2: ✓ COMPLETE")


if __name__ == '__main__':
    main()