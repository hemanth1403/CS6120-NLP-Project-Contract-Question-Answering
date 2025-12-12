"""
Evaluate BiLSTM model on test set with both original and refactored questions
This addresses Grade Contract Milestone B.3 and A-.3
"""

import torch
import json
import numpy as np
from tqdm import tqdm
from bilstm_attention_with_answerability import BiLSTMQAWithAnswerability, BiLSTMQATrainerWithAnswerability
from cuad_dataloader import Tokenizer

class TestSetEvaluator:
    """Evaluate model on test set with original and refactored questions"""
    
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
        print(f"Using device: {device}")
    
    def preprocess_example(self, context, question, max_context_len=512, max_question_len=64):
        """Preprocess single example"""
        # Tokenize
        question_ids = self.tokenizer.encode(question, max_question_len)
        context_ids = self.tokenizer.encode(context, max_context_len)
        
        # Create masks
        question_mask = [1] * len(question_ids)
        context_mask = [1] * len(context_ids)
        
        # Pad to max length
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
            
            # Get predictions
            start_pred = torch.argmax(start_logits, dim=-1).item()
            end_pred = torch.argmax(end_logits, dim=-1).item()
            answerability_pred = torch.argmax(answerability_logits, dim=-1).item()
            
            # If predicted as unanswerable, return empty
            if answerability_pred == 1:  # 1 = unanswerable
                return "", 0, 0, True
            
            # Extract answer text
            context_words = context.lower().split()
            if 0 <= start_pred < len(context_words) and 0 <= end_pred < len(context_words) and start_pred <= end_pred:
                answer_text = ' '.join(context_words[start_pred:end_pred+1])
            else:
                answer_text = ""
            
            return answer_text, start_pred, end_pred, False
    
    def compute_f1(self, prediction, ground_truth):
        """Compute token-level F1 score"""
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
    
    def evaluate_dataset(self, test_path, use_refactored=False):
        """
        Evaluate on test set
        
        Args:
            test_path: Path to test JSON file
            use_refactored: If True, use refactored questions; otherwise use original
        """
        print(f"\nEvaluating on {'REFACTORED' if use_refactored else 'ORIGINAL'} questions...")
        
        # Load test data
        with open(test_path, 'r') as f:
            test_data = json.load(f)
        
        # Metrics
        all_f1 = []
        all_em = []
        answerability_correct = 0
        total = 0
        
        # Track by answerability
        answerable_f1 = []
        unanswerable_f1 = []
        
        # Process each example
        for article in tqdm(test_data['data'], desc="Evaluating"):
            for paragraph in article['paragraphs']:
                context = paragraph.get('context', '')
                
                for qa in paragraph['qas']:
                    # Choose question type
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
                    pred_answer, start, end, pred_unanswerable = self.predict(context, question)
                    
                    # Compute metrics
                    f1 = self.compute_f1(pred_answer, true_answer)
                    em = 1.0 if pred_answer.strip().lower() == true_answer.strip().lower() else 0.0
                    
                    all_f1.append(f1)
                    all_em.append(em)
                    
                    # Answerability accuracy
                    if (pred_unanswerable and is_impossible) or (not pred_unanswerable and not is_impossible):
                        answerability_correct += 1
                    
                    # Track by type
                    if is_impossible:
                        unanswerable_f1.append(f1)
                    else:
                        answerable_f1.append(f1)
                    
                    total += 1
        
        # Compute overall metrics
        results = {
            'f1': np.mean(all_f1) * 100,
            'em': np.mean(all_em) * 100,
            'answerability_acc': (answerability_correct / total) * 100 if total > 0 else 0,
            'answerable_f1': np.mean(answerable_f1) * 100 if answerable_f1 else 0,
            'unanswerable_f1': np.mean(unanswerable_f1) * 100 if unanswerable_f1 else 0,
            'total_examples': total
        }
        
        return results
    
    def compare_original_vs_refactored(self, test_path):
        """
        Compare performance on original vs refactored questions
        Addresses Grade Contract Milestone A-.3
        """
        print("\n" + "="*70)
        print("PARAPHRASE ROBUSTNESS EVALUATION")
        print("="*70)
        
        # Evaluate on original questions
        original_results = self.evaluate_dataset("/Users/hemu/Desktop/Desktop items/NU/Sem_2/NLP - siwu/Project/data/test.json", use_refactored=False)
        
        # Evaluate on refactored questions
        refactored_results = self.evaluate_dataset(test_path, use_refactored=True)
        
        # Print comparison
        print("\n" + "="*70)
        print("RESULTS COMPARISON")
        print("="*70)
        
        print("\nOriginal Legal Questions:")
        print(f"  F1 Score:              {original_results['f1']:.2f}%")
        print(f"  Exact Match:           {original_results['em']:.2f}%")
        print(f"  Answerability Acc.:    {original_results['answerability_acc']:.2f}%")
        print(f"  Answerable F1:         {original_results['answerable_f1']:.2f}%")
        print(f"  Total Examples:        {original_results['total_examples']}")
        
        print("\nRefactored Paraphrased Questions:")
        print(f"  F1 Score:              {refactored_results['f1']:.2f}%")
        print(f"  Exact Match:           {refactored_results['em']:.2f}%")
        print(f"  Answerability Acc.:    {refactored_results['answerability_acc']:.2f}%")
        print(f"  Answerable F1:         {refactored_results['answerable_f1']:.2f}%")
        print(f"  Total Examples:        {refactored_results['total_examples']}")
        
        # Compute robustness (performance drop)
        print("\nRobustness Analysis:")
        f1_drop = original_results['f1'] - refactored_results['f1']
        em_drop = original_results['em'] - refactored_results['em']
        ans_drop = original_results['answerability_acc'] - refactored_results['answerability_acc']
        
        print(f"  F1 Score Drop:         {f1_drop:+.2f}% ({f1_drop/original_results['f1']*100 if original_results['f1'] > 0 else 0:.1f}% relative)")
        print(f"  EM Drop:               {em_drop:+.2f}%")
        print(f"  Answerability Drop:    {ans_drop:+.2f}%")
        
        if abs(f1_drop) < 1.0:
            print("\n  ✓ Model is ROBUST to paraphrasing (< 1% drop)")
        elif abs(f1_drop) < 3.0:
            print("\n  ~ Model shows MODERATE robustness to paraphrasing")
        else:
            print("\n  ✗ Model shows POOR robustness to paraphrasing (> 3% drop)")
        
        print("\n" + "="*70)
        
        return {
            'original': original_results,
            'refactored': refactored_results,
            'drops': {
                'f1': f1_drop,
                'em': em_drop,
                'answerability': ans_drop
            }
        }


def main():
    # Configuration
    model_path = './outputs/bilstm_answerability_20251202_203753/best_model.pt'
    tokenizer_path = './outputs/bilstm_answerability_20251202_203753/tokenizer.json'
    test_path = '/Users/hemu/Desktop/Desktop items/NU/Sem_2/NLP - siwu/Project/Refactored Data/test_refactored.json'
    
    print("="*70)
    print("BiLSTM TEST SET EVALUATION")
    print("Grade Contract Milestones: B.3 (Paraphrase) + A-.3 (Robustness)")
    print("="*70)
    
    # Initialize evaluator
    evaluator = TestSetEvaluator(model_path, tokenizer_path, device='cpu')
    
    # Run comparison
    results = evaluator.compare_original_vs_refactored(test_path)
    
    # Save results
    output_file = './outputs/test_evaluation_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print("\n✓ Test evaluation complete!")
    print("\nGrade Contract Status:")
    print("  [✓] B.3  - Paraphrase annotation (test set has refactored questions)")
    print("  [✓] A-.3 - Paraphrase robustness evaluation (comparison complete)")


if __name__ == '__main__':
    main()