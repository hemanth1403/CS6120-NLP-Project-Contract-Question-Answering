"""
Debug script to inspect BiLSTM predictions and identify why metrics are frozen.
"""

import torch
import json
from bilstm_attention import BiLSTMQA, BiLSTMQATrainer
from cuad_dataloader import Tokenizer, CUADDataset, collate_fn
from torch.utils.data import DataLoader

def debug_predictions(checkpoint_path='./outputs/bilstm_20251127_145007/best_model.pt',
                     val_path='./data/val.json'):
    """
    Load trained model and inspect predictions on validation set.
    """
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    
    print("\nLoading tokenizer...")
    tokenizer = Tokenizer(vocab_size=config['vocab_size'], max_len=config['max_context_len'])
    tokenizer.load('./outputs/bilstm_20251127_145007/tokenizer.json')
    
    print("Creating model...")
    model = BiLSTMQA(
        vocab_size=len(tokenizer.word2idx),
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    trainer = BiLSTMQATrainer(model, device='cpu')
    
    print("Loading validation data...")
    val_dataset = CUADDataset(val_path, tokenizer, 
                             max_context_len=config['max_context_len'],
                             max_question_len=config['max_question_len'])
    
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    
    print("\n" + "="*80)
    print("INSPECTING FIRST BATCH PREDICTIONS")
    print("="*80)
    
    # Get first batch
    batch = next(iter(val_loader))
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        question_ids = batch['question_ids']
        context_ids = batch['context_ids']
        question_mask = batch['question_mask']
        context_mask = batch['context_mask']
        
        start_logits, end_logits, attention_weights = model(
            question_ids, context_ids, question_mask, context_mask
        )
        
        print(f"\nBatch size: {len(batch['original_contexts'])}")
        print(f"Start logits shape: {start_logits.shape}")
        print(f"End logits shape: {end_logits.shape}")
        
        # Analyze each example
        for i in range(min(4, len(batch['original_contexts']))):
            print(f"\n{'='*80}")
            print(f"EXAMPLE {i+1}")
            print(f"{'='*80}")
            
            # Context info
            context = batch['original_contexts'][i]
            question = batch['original_questions'][i]
            true_answer = batch['answer_texts'][i]
            is_impossible = batch['is_impossible'][i]
            
            print(f"\nQuestion: {question[:100]}...")
            print(f"Context length: {len(context.split())} words")
            print(f"True answer: '{true_answer[:50]}...' (impossible={is_impossible})")
            
            # Logits analysis
            example_start_logits = start_logits[i].cpu().numpy()
            example_end_logits = end_logits[i].cpu().numpy()
            context_len = context_mask[i].sum().item()
            
            print(f"\nLogits info:")
            print(f"  Valid context length: {context_len}")
            print(f"  Start logits range: [{example_start_logits[:context_len].min():.4f}, {example_start_logits[:context_len].max():.4f}]")
            print(f"  End logits range: [{example_end_logits[:context_len].min():.4f}, {example_end_logits[:context_len].max():.4f}]")
            
            # Check for -inf (masked positions)
            num_inf_start = (example_start_logits == float('-inf')).sum()
            num_inf_end = (example_end_logits == float('-inf')).sum()
            print(f"  Positions with -inf: start={num_inf_start}, end={num_inf_end}")
            
            # Predicted positions
            pred_start = example_start_logits[:context_len].argmax()
            pred_end = example_end_logits[:context_len].argmax()
            
            print(f"\nPredicted positions:")
            print(f"  Start: {pred_start} (logit={example_start_logits[pred_start]:.4f})")
            print(f"  End: {pred_end} (logit={example_end_logits[pred_end]:.4f})")
            
            # Extract predicted answer
            context_words = context.lower().split()
            if pred_start < len(context_words) and pred_end < len(context_words):
                pred_answer = ' '.join(context_words[pred_start:pred_end+1])
                print(f"  Predicted answer: '{pred_answer[:100]}'")
            else:
                print(f"  Predicted answer: <OUT OF BOUNDS>")
            
            # True positions
            true_start = batch['start_positions'][i].item()
            true_end = batch['end_positions'][i].item()
            print(f"\nTrue positions:")
            print(f"  Start: {true_start}")
            print(f"  End: {true_end}")
            
            # Attention analysis
            attn = attention_weights[i].cpu().numpy()
            print(f"\nAttention weights shape: {attn.shape}")
            print(f"  Mean attention: {attn.mean():.6f}")
            print(f"  Max attention: {attn.max():.6f}")
            print(f"  Min attention: {attn.min():.6f}")
    
    # Summary statistics over more examples
    print("\n" + "="*80)
    print("SUMMARY STATISTICS (First 100 examples)")
    print("="*80)
    
    pred_starts = []
    pred_ends = []
    true_starts = []
    true_ends = []
    
    count = 0
    for batch in val_loader:
        if count >= 100:
            break
        
        with torch.no_grad():
            start_logits, end_logits, _ = model(
                batch['question_ids'], batch['context_ids'],
                batch['question_mask'], batch['context_mask']
            )
            
            for i in range(len(batch['original_contexts'])):
                context_len = batch['context_mask'][i].sum().item()
                pred_start = start_logits[i, :context_len].argmax().item()
                pred_end = end_logits[i, :context_len].argmax().item()
                
                pred_starts.append(pred_start)
                pred_ends.append(pred_end)
                true_starts.append(batch['start_positions'][i].item())
                true_ends.append(batch['end_positions'][i].item())
                
                count += 1
                if count >= 100:
                    break
    
    print(f"\nPredicted start positions:")
    print(f"  Mean: {sum(pred_starts)/len(pred_starts):.2f}")
    print(f"  Unique values: {len(set(pred_starts))}")
    print(f"  Most common: {max(set(pred_starts), key=pred_starts.count)} (occurs {pred_starts.count(max(set(pred_starts), key=pred_starts.count))} times)")
    
    print(f"\nPredicted end positions:")
    print(f"  Mean: {sum(pred_ends)/len(pred_ends):.2f}")
    print(f"  Unique values: {len(set(pred_ends))}")
    print(f"  Most common: {max(set(pred_ends), key=pred_ends.count)} (occurs {pred_ends.count(max(set(pred_ends), key=pred_ends.count))} times)")
    
    print(f"\nTrue start positions:")
    print(f"  Mean: {sum([s for s in true_starts if s >= 0])/len([s for s in true_starts if s >= 0]):.2f}")
    print(f"  Num answerable: {sum(1 for s in true_starts if s >= 0)}")
    print(f"  Num unanswerable: {sum(1 for s in true_starts if s < 0)}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
        val_path = sys.argv[2] if len(sys.argv) > 2 else './data/val.json'
    else:
        # Try to find latest checkpoint
        import os
        import glob
        
        output_dirs = sorted(glob.glob('./outputs/bilstm_*'), reverse=True)
        if output_dirs:
            checkpoint_path = os.path.join(output_dirs[0], 'best_model.pt')
            print(f"Using checkpoint: {checkpoint_path}")
        else:
            print("Error: No checkpoint found. Please provide checkpoint path.")
            print("Usage: python debug_predictions.py [checkpoint_path] [val_path]")
            sys.exit(1)
        
        val_path = './data/val.json'
    
    debug_predictions(checkpoint_path, val_path)