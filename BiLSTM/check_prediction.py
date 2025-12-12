import torch
import numpy as np
import json
from bilstm_attention import BiLSTMQA, BiLSTMQATrainer
from cuad_dataloader import Tokenizer, create_dataloaders

# Load checkpoint
checkpoint_path = './outputs/bilstm_20251202_161409/best_model.pt'
tokenizer_path = './outputs/bilstm_20251202_161409/tokenizer.json'

print("Loading checkpoint...")
checkpoint = torch.load(checkpoint_path, weights_only=False)

# Load tokenizer
print("Loading tokenizer...")
tokenizer = Tokenizer(vocab_size=50000, max_len=512)
tokenizer.load(tokenizer_path)

# Create model
print("Creating model...")
model = BiLSTMQA(
    vocab_size=len(tokenizer.word2idx),
    embedding_dim=300,
    hidden_dim=256,
    num_layers=2,
    dropout=0.2
)
model.load_state_dict(checkpoint['model_state_dict'])
trainer = BiLSTMQATrainer(model, device='cpu')

# Load validation data
print("Loading validation data...")
_, val_loader = create_dataloaders(
    train_path='./data/train.json',
    val_path='./data/val.json',
    tokenizer=tokenizer,
    batch_size=1,
    max_context_len=512,
    max_question_len=64,
    num_workers=0
)

# Collect predictions
print("\nAnalyzing predictions...")
all_starts = []
all_ends = []
span_lengths = []
answerable_predictions = []
answerable_ground_truth = []
sample_predictions = []

model.eval()
for i, batch in enumerate(val_loader):
    if i >= 100:  # Check first 100
        break
    
    outputs = trainer.predict(batch)
    start = outputs['start_positions'][0]
    end = outputs['end_positions'][0]
    
    all_starts.append(start)
    all_ends.append(end)
    span_lengths.append(end - start + 1)
    
    # Check if model predicts answerable (non-zero span)
    pred_answerable = not (start == 0 and end == 0)
    true_answerable = not batch['is_impossible'][0]
    
    answerable_predictions.append(pred_answerable)
    answerable_ground_truth.append(true_answerable)
    
    # Save first 10 examples for detailed inspection
    if i < 10:
        context_words = batch['original_contexts'][0].lower().split()
        pred_text = ' '.join(context_words[start:end+1]) if start <= end and end < len(context_words) else ""
        true_text = batch['answer_texts'][0]
        
        sample_predictions.append({
            'example': i + 1,
            'question': batch['original_questions'][0][:80] + "...",
            'predicted_start': int(start),
            'predicted_end': int(end),
            'predicted_text': pred_text[:100],
            'true_text': true_text[:100],
            'is_answerable': true_answerable
        })

# Convert to numpy for analysis
all_starts = np.array(all_starts)
all_ends = np.array(all_ends)
span_lengths = np.array(span_lengths)

print("="*70)
print("PREDICTION ANALYSIS (first 100 validation samples)")
print("="*70)

print("\n📊 START POSITIONS:")
print(f"  Mean: {np.mean(all_starts):.2f}")
print(f"  Median: {np.median(all_starts):.2f}")
print(f"  Std: {np.std(all_starts):.2f}")
print(f"  Min: {np.min(all_starts)}, Max: {np.max(all_starts)}")
print(f"  Unique values: {len(np.unique(all_starts))}")
print(f"  Most common: position {np.bincount(all_starts).argmax()} (occurs {np.bincount(all_starts).max()} times = {100*np.bincount(all_starts).max()/len(all_starts):.1f}%)")

print("\n📊 END POSITIONS:")
print(f"  Mean: {np.mean(all_ends):.2f}")
print(f"  Median: {np.median(all_ends):.2f}")
print(f"  Std: {np.std(all_ends):.2f}")
print(f"  Min: {np.min(all_ends)}, Max: {np.max(all_ends)}")
print(f"  Unique values: {len(np.unique(all_ends))}")
print(f"  Most common: position {np.bincount(all_ends).argmax()} (occurs {np.bincount(all_ends).max()} times = {100*np.bincount(all_ends).max()/len(all_ends):.1f}%)")

print("\n📊 SPAN LENGTHS:")
print(f"  Mean: {np.mean(span_lengths):.2f} tokens")
print(f"  Median: {np.median(span_lengths):.2f} tokens")
print(f"  Std: {np.std(span_lengths):.2f}")
print(f"  Min: {np.min(span_lengths)}, Max: {np.max(span_lengths)}")

print("\n📊 ANSWERABILITY:")
true_answerable_count = sum(answerable_ground_truth)
pred_answerable_count = sum(answerable_predictions)
correct_answerable = sum(p == t for p, t in zip(answerable_predictions, answerable_ground_truth))

print(f"  Ground truth answerable: {true_answerable_count}/100 ({100*true_answerable_count/100:.1f}%)")
print(f"  Model predicts answerable: {pred_answerable_count}/100 ({100*pred_answerable_count/100:.1f}%)")
print(f"  Answerability accuracy: {correct_answerable}/100 ({100*correct_answerable/100:.1f}%)")

# Diagnosis
print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)

if len(np.unique(all_starts)) < 10:
    print("❌ CRITICAL: Model is collapsing! Very few unique start positions.")
    print(f"   Only {len(np.unique(all_starts))} unique start positions out of 100 samples.")
else:
    print(f"✓ Model is exploring positions ({len(np.unique(all_starts))} unique starts).")

if np.bincount(all_starts).max() > 80:
    print(f"❌ CRITICAL: Model predicting same start position {100*np.bincount(all_starts).max()/len(all_starts):.1f}% of the time!")
elif np.bincount(all_starts).max() > 50:
    print(f"⚠️  WARNING: Model heavily biased to one start position ({100*np.bincount(all_starts).max()/len(all_starts):.1f}% of samples).")
else:
    print("✓ Start predictions are reasonably diverse.")

if abs(pred_answerable_count - true_answerable_count) > 30:
    print(f"❌ CRITICAL: Model heavily biased in answerability prediction!")
    print(f"   Ground truth: {true_answerable_count}% answerable, Model predicts: {pred_answerable_count}%")
else:
    print("✓ Answerability predictions are balanced.")

if np.mean(span_lengths) < 2:
    print(f"⚠️  WARNING: Very short predicted spans (mean={np.mean(span_lengths):.1f} tokens).")
elif np.mean(span_lengths) > 50:
    print(f"⚠️  WARNING: Very long predicted spans (mean={np.mean(span_lengths):.1f} tokens).")
else:
    print(f"✓ Reasonable span lengths (mean={np.mean(span_lengths):.1f} tokens).")

print("\n" + "="*70)
print("SAMPLE PREDICTIONS (first 10 examples)")
print("="*70)

for sample in sample_predictions:
    print(f"\n📝 Example {sample['example']}:")
    print(f"   Question: {sample['question']}")
    print(f"   Is Answerable: {sample['is_answerable']}")
    print(f"   Predicted: [{sample['predicted_start']}:{sample['predicted_end']}] '{sample['predicted_text']}'")
    print(f"   True: '{sample['true_text']}'")

print("\n" + "="*70)
print("Analysis complete!")
print("="*70)