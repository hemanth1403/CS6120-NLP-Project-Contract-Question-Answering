import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import json
import os
from datetime import datetime

from bilstm_attention_with_answerability import BiLSTMQAWithAnswerability, BiLSTMQATrainerWithAnswerability
from cuad_dataloader import Tokenizer, create_dataloaders


class EvaluationMetrics:
    """
    Evaluation metrics for QA task.
    """
    @staticmethod
    def compute_exact_match(predictions: list, references: list) -> float:
        """Compute exact match score."""
        correct = sum(1 for pred, ref in zip(predictions, references) 
                     if pred.strip().lower() == ref.strip().lower())
        return correct / len(predictions) if predictions else 0.0
    
    @staticmethod
    def compute_f1(predictions: list, references: list) -> float:
        """Compute F1 score (token-level)."""
        f1_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = set(pred.lower().split())
            ref_tokens = set(ref.lower().split())
            
            if len(pred_tokens) == 0 and len(ref_tokens) == 0:
                f1_scores.append(1.0)
                continue
            elif len(pred_tokens) == 0 or len(ref_tokens) == 0:
                f1_scores.append(0.0)
                continue
            
            common = pred_tokens & ref_tokens
            precision = len(common) / len(pred_tokens)
            recall = len(common) / len(ref_tokens)
            
            if precision + recall == 0:
                f1_scores.append(0.0)
            else:
                f1 = 2 * precision * recall / (precision + recall)
                f1_scores.append(f1)
        
        return np.mean(f1_scores) if f1_scores else 0.0
    
    @staticmethod
    def compute_answerability_accuracy(pred_has_answer: list, 
                                      true_has_answer: list) -> float:
        """Compute accuracy of predicting whether question is answerable."""
        correct = sum(1 for pred, true in zip(pred_has_answer, true_has_answer)
                     if pred == true)
        return correct / len(pred_has_answer) if pred_has_answer else 0.0


def extract_answer_text(context_words: list, start_idx: int, end_idx: int) -> str:
    """Extract answer text from context given start and end indices."""
    if start_idx < 0 or end_idx < 0 or start_idx > end_idx:
        return ""
    
    end_idx = min(end_idx, len(context_words) - 1)
    answer_words = context_words[start_idx:end_idx + 1]
    return ' '.join(answer_words)


def evaluate(model, dataloader, trainer, tokenizer):
    """Evaluate model on validation set."""
    model.eval()
    
    all_predictions = []
    all_references = []
    pred_has_answer = []
    true_has_answer = []
    
    total_loss = 0
    total_span_loss = 0
    total_answerability_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Get predictions
            outputs = trainer.predict(batch)
            
            # Compute loss
            question_ids = batch['question_ids'].to(trainer.device)
            context_ids = batch['context_ids'].to(trainer.device)
            question_mask = batch['question_mask'].to(trainer.device)
            context_mask = batch['context_mask'].to(trainer.device)
            start_positions = batch['start_positions'].to(trainer.device)
            end_positions = batch['end_positions'].to(trainer.device)
            # Convert is_impossible from list of booleans to tensor
            is_impossible = torch.tensor([int(x) for x in batch['is_impossible']], dtype=torch.long, device=trainer.device)
            
            start_logits, end_logits, answerability_logits, _ = model(
                question_ids, context_ids, question_mask, context_mask
            )
            
            # Check for NaN
            if torch.isnan(start_logits).any() or torch.isnan(end_logits).any():
                print("Warning: NaN detected in evaluation!")
                continue
            
            loss, span_loss, answerability_loss = trainer.compute_loss(
                start_logits, end_logits, answerability_logits,
                start_positions, end_positions, is_impossible
            )
            
            total_loss += loss.item()
            total_span_loss += span_loss.item()
            total_answerability_loss += answerability_loss.item()
            num_batches += 1
            
            # Extract answer texts
            for i in range(len(batch['original_contexts'])):
                context = batch['original_contexts'][i]
                context_words = context.lower().split()
                
                # Predicted answer
                pred_start = outputs['start_positions'][i]
                pred_end = outputs['end_positions'][i]
                pred_text = extract_answer_text(context_words, pred_start, pred_end)
                
                # Reference answer
                ref_text = batch['answer_texts'][i]
                
                all_predictions.append(pred_text)
                all_references.append(ref_text)
                
                # Answerability (using model's answerability prediction)
                pred_answerable = outputs['answerability_pred'][i] == 0  # 0 = answerable
                true_answerable = not batch['is_impossible'][i]
                
                pred_has_answer.append(pred_answerable)
                true_has_answer.append(true_answerable)
    
    # Compute metrics
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_span_loss = total_span_loss / num_batches if num_batches > 0 else 0
    avg_answerability_loss = total_answerability_loss / num_batches if num_batches > 0 else 0
    exact_match = EvaluationMetrics.compute_exact_match(all_predictions, all_references)
    f1_score = EvaluationMetrics.compute_f1(all_predictions, all_references)
    answerability_acc = EvaluationMetrics.compute_answerability_accuracy(
        pred_has_answer, true_has_answer
    )
    
    metrics = {
        'loss': avg_loss,
        'span_loss': avg_span_loss,
        'answerability_loss': avg_answerability_loss,
        'exact_match': exact_match,
        'f1': f1_score,
        'answerability_accuracy': answerability_acc
    }
    
    return metrics


def train_epoch(model, dataloader, trainer, optimizer, epoch, max_grad_norm=1.0):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    total_span_loss = 0
    total_answerability_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in progress_bar:
        # Train step returns total_loss, span_loss, answerability_loss
        loss, span_loss, answerability_loss = trainer.train_step(batch, optimizer, max_grad_norm)
        
        # Check for NaN
        if np.isnan(loss):
            print("Warning: NaN detected in training loss!")
            continue
        
        total_loss += loss
        total_span_loss += span_loss
        total_answerability_loss += answerability_loss
        num_batches += 1
        
        # Update progress bar
        avg_loss = total_loss / num_batches
        progress_bar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'span': f'{total_span_loss/num_batches:.4f}',
            'ans': f'{total_answerability_loss/num_batches:.4f}'
        })
    
    return (total_loss / num_batches if num_batches > 0 else 0,
            total_span_loss / num_batches if num_batches > 0 else 0,
            total_answerability_loss / num_batches if num_batches > 0 else 0)


def train_model(config):
    """Main training function."""
    # Set random seeds
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config['output_dir'], f"bilstm_answerability_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Save config
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = Tokenizer(
        vocab_size=config['vocab_size'],
        max_len=config['max_context_len']
    )
    
    # Build vocabulary
    print("Building vocabulary from training data...")
    with open(config['train_path'], 'r') as f:
        train_data = json.load(f)
    
    all_texts = []
    for article in train_data['data']:
        for paragraph in article['paragraphs']:
            all_texts.append(paragraph['context'])
            for qa in paragraph['qas']:
                all_texts.append(qa['question'])
    
    tokenizer.build_vocab(all_texts)
    tokenizer.save(os.path.join(output_dir, 'tokenizer.json'))
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        train_path=config['train_path'],
        val_path=config['val_path'],
        tokenizer=tokenizer,
        batch_size=config['batch_size'],
        max_context_len=config['max_context_len'],
        max_question_len=config['max_question_len'],
        num_workers=config['num_workers']
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Initialize model
    print("Initializing model with answerability head...")
    model = BiLSTMQAWithAnswerability(
        vocab_size=len(tokenizer.word2idx),
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    
    trainer = BiLSTMQATrainerWithAnswerability(
        model, 
        device=config['device'],
        answerability_weight=config['answerability_weight']
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Using device: {config['device']}")
    print(f"Answerability loss weight: {config['answerability_weight']}")
    
    # Initialize optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )
    
    # Training loop
    print("\nStarting training...")
    best_f1 = 0.0
    training_history = []
    
    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{config['num_epochs']}")
        print(f"{'='*50}")
        
        # Train
        train_loss, train_span_loss, train_ans_loss = train_epoch(
            model, train_loader, trainer, optimizer, epoch, config['max_grad_norm']
        )
        print(f"Train Loss: {train_loss:.4f} (span: {train_span_loss:.4f}, ans: {train_ans_loss:.4f})")
        
        # Evaluate
        print("\nEvaluating on validation set...")
        val_metrics = evaluate(model, val_loader, trainer, tokenizer)
        
        print(f"\nValidation Metrics:")
        print(f"  Loss: {val_metrics['loss']:.4f} (span: {val_metrics['span_loss']:.4f}, ans: {val_metrics['answerability_loss']:.4f})")
        print(f"  Exact Match: {val_metrics['exact_match']:.4f}")
        print(f"  F1 Score: {val_metrics['f1']:.4f}")
        print(f"  Answerability Accuracy: {val_metrics['answerability_accuracy']:.4f}")
        
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_metrics['f1'])
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != current_lr:
            print(f"  Learning rate reduced: {current_lr:.6f} -> {new_lr:.6f}")
        
        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': config
            }, os.path.join(output_dir, 'best_model.pt'))
            print(f"  ✓ Saved best model (F1: {best_f1:.4f})")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_metrics': val_metrics,
            'config': config
        }, os.path.join(output_dir, 'last_checkpoint.pt'))
        
        # Record history
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_span_loss': train_span_loss,
            'train_answerability_loss': train_ans_loss,
            'val_loss': val_metrics['loss'],
            'val_span_loss': val_metrics['span_loss'],
            'val_answerability_loss': val_metrics['answerability_loss'],
            'val_exact_match': val_metrics['exact_match'],
            'val_f1': val_metrics['f1'],
            'val_answerability_accuracy': val_metrics['answerability_accuracy']
        })
        
        # Save training history
        with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
            json.dump(training_history, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"Best F1 Score: {best_f1:.4f}")
    print(f"Results saved to: {output_dir}")
    
    return model, trainer, tokenizer, training_history


if __name__ == "__main__":
    # Training configuration
    config = {
        # Data
        'train_path': './data/train.json',
        'val_path': './data/val.json',
        
        # Model architecture
        'vocab_size': 50000,
        'embedding_dim': 300,
        'hidden_dim': 256,
        'num_layers': 2,
        'dropout': 0.2,
        
        # Training
        'batch_size': 16,
        'num_epochs': 20,
        'learning_rate': 0.0002,
        'max_context_len': 512,
        'max_question_len': 64,
        'max_grad_norm': 5.0,
        'answerability_weight': 1.0,  # NEW: Weight for answerability loss
        
        # Other
        'seed': 42,
        'device': 'cpu',
        'num_workers': 0,
        'output_dir': './outputs'
    }
    
    print("="*70)
    print("TRAINING WITH ANSWERABILITY LOSS")
    print("="*70)
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Train model
    model, trainer, tokenizer, history = train_model(config)