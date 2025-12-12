"""
Generate visualization figures for BiLSTM training report
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load training history
with open('./outputs/bilstm_answerability_20251202_203753/training_history.json', 'r') as f:
    history = json.load(f)

# Extract data
epochs = [h['epoch'] for h in history]
train_loss = [h['train_loss'] for h in history]
val_loss = [h['val_loss'] for h in history]
train_span_loss = [h['train_span_loss'] for h in history]
train_ans_loss = [h['train_answerability_loss'] for h in history]
val_f1 = [h['val_f1'] * 100 for h in history]  # Convert to percentage
val_em = [h['val_exact_match'] * 100 for h in history]
val_ans_acc = [h['val_answerability_accuracy'] * 100 for h in history]

# Set style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

# Figure 1: Training and Validation Loss
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Overall loss
ax1.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
ax1.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
ax1.axvline(x=5, color='gray', linestyle='--', alpha=0.5, label='Best F1 (Epoch 5)')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training vs Validation Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Loss components
ax2.plot(epochs, train_span_loss, 'b-', label='Span Loss', linewidth=2)
ax2.plot(epochs, train_ans_loss, 'g-', label='Answerability Loss', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Training Loss Components')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/hemu/Desktop/Desktop items/NU/Sem_2/NLP - siwu/Project/outputs/bilstm_answerability_20251202_203753/training_loss.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/Users/hemu/Desktop/Desktop items/NU/Sem_2/NLP - siwu/Project/outputs/bilstm_answerability_20251202_203753/training_loss.png', dpi=300, bbox_inches='tight')
print("✓ Saved training_loss figures")
plt.close()

# Figure 2: Validation Metrics Over Time
fig, ax = plt.subplots(1, 1, figsize=(10, 5))

ax.plot(epochs, val_f1, 'b-', marker='o', label='F1 Score', linewidth=2, markersize=4)
ax.plot(epochs, val_em, 'r-', marker='s', label='Exact Match', linewidth=2, markersize=4)
ax.plot(epochs, val_ans_acc, 'g-', marker='^', label='Answerability Acc.', linewidth=2, markersize=4)
ax.axvline(x=5, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=15, color='gray', linestyle='--', alpha=0.5)
ax.text(5, 85, 'Best F1\n(Epoch 5)', ha='center', va='top', fontsize=9)
ax.text(15, 85, 'Peak F1\n(Epoch 15)', ha='center', va='top', fontsize=9)

ax.set_xlabel('Epoch')
ax.set_ylabel('Score (%)')
ax.set_title('Validation Metrics Over Training')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 90])

plt.tight_layout()
plt.savefig('/Users/hemu/Desktop/Desktop items/NU/Sem_2/NLP - siwu/Project/outputs/bilstm_answerability_20251202_203753/validation_metrics.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/Users/hemu/Desktop/Desktop items/NU/Sem_2/NLP - siwu/Project/outputs/bilstm_answerability_20251202_203753/validation_metrics.png', dpi=300, bbox_inches='tight')
print("✓ Saved validation_metrics figures")
plt.close()

# Figure 3: Model Comparison Bar Chart
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

models = ['BiLSTM\n(No Answerability)', 'BiLSTM\n(With Answerability)']
f1_scores = [3.68, 4.69]
em_scores = [0.14, 0.41]
ans_scores = [32.93, 80.56]

x = np.arange(len(models))
width = 0.25

bars1 = ax.bar(x - width, f1_scores, width, label='F1 Score', color='#3498db')
bars2 = ax.bar(x, em_scores, width, label='Exact Match', color='#e74c3c')
bars3 = ax.bar(x + width, ans_scores, width, label='Answerability Acc.', color='#2ecc71')

ax.set_ylabel('Score (%)')
ax.set_title('BiLSTM Model Comparison')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.grid(True, axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('/Users/hemu/Desktop/Desktop items/NU/Sem_2/NLP - siwu/Project/outputs/bilstm_answerability_20251202_203753/model_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/Users/hemu/Desktop/Desktop items/NU/Sem_2/NLP - siwu/Project/outputs/bilstm_answerability_20251202_203753/model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved model_comparison figures")
plt.close()

# Figure 4: Answerability Learning Curve
fig, ax = plt.subplots(1, 1, figsize=(10, 5))

ax.plot(epochs, val_ans_acc, 'g-', marker='o', linewidth=3, markersize=6)
ax.axhline(y=32.93, color='r', linestyle='--', linewidth=2, label='Baseline (No Answerability)')
ax.axhline(y=50, color='gray', linestyle=':', linewidth=1, label='Random Guess')
ax.fill_between(epochs, 70, 85, alpha=0.2, color='green', label='Target Range')

ax.text(1, 82, 'Learned in\n1 epoch!', fontsize=11, fontweight='bold', 
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

ax.set_xlabel('Epoch')
ax.set_ylabel('Answerability Accuracy (%)')
ax.set_title('Answerability Classification Learning Curve')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 100])

plt.tight_layout()
plt.savefig('/Users/hemu/Desktop/Desktop items/NU/Sem_2/NLP - siwu/Project/outputs/bilstm_answerability_20251202_203753answerability_learning.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/Users/hemu/Desktop/Desktop items/NU/Sem_2/NLP - siwu/Project/outputs/bilstm_answerability_20251202_203753/answerability_learning.png', dpi=300, bbox_inches='tight')
print("✓ Saved answerability_learning figures")
plt.close()

# Figure 5: Overfitting Visualization
fig, ax = plt.subplots(1, 1, figsize=(10, 5))

ax.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2.5)
ax.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2.5)

# Shade overfitting region
overfitting_start = 5
ax.axvspan(overfitting_start, 20, alpha=0.2, color='red', label='Overfitting Region')
ax.axvline(x=overfitting_start, color='orange', linestyle='--', linewidth=2)
ax.text(overfitting_start + 0.5, 4.5, 'Overfitting\nBegins', fontsize=10, fontweight='bold')

# Annotate key points
ax.annotate('Training loss decreases', xy=(20, train_loss[-1]), xytext=(15, 1.5),
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
            fontsize=9, color='blue')
ax.annotate('Validation loss increases', xy=(20, val_loss[-1]), xytext=(15, 3.5),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            fontsize=9, color='red')

ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Evidence of Overfitting After Epoch 5')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/hemu/Desktop/Desktop items/NU/Sem_2/NLP - siwu/Project/outputs/bilstm_answerability_20251202_203753/overfitting_analysis.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/Users/hemu/Desktop/Desktop items/NU/Sem_2/NLP - siwu/Project/outputs/bilstm_answerability_20251202_203753/overfitting_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved overfitting_analysis figures")
plt.close()

# Figure 6: Expected vs Actual Performance (for LegalBERT comparison)
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

models = ['BiLSTM\n(No Ans.)', 'BiLSTM\n(With Ans.)', 'Expected\nLegalBERT']
f1_scores = [3.68, 4.69, 66.57]
em_scores = [0.14, 0.41, 66.43]
ans_scores = [32.93, 80.56, 87]

x = np.arange(len(models))
width = 0.25

bars1 = ax.bar(x - width, f1_scores, width, label='F1 Score', color='#3498db')
bars2 = ax.bar(x, em_scores, width, label='Exact Match', color='#e74c3c')
bars3 = ax.bar(x + width, ans_scores, width, label='Answerability Acc.', color='#2ecc71')

ax.set_ylabel('Score (%)')
ax.set_title('BiLSTM Baseline vs Expected LegalBERT Performance')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.grid(True, axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9)

# Add annotation for performance gap
ax.annotate('', xy=(2, 75), xytext=(1, 4.69),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax.text(1.5, 40, '~17× gap', fontsize=12, fontweight='bold', color='red',
        ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig('/Users/hemu/Desktop/Desktop items/NU/Sem_2/NLP - siwu/Project/outputs/bilstm_answerability_20251202_203753/bert_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/Users/hemu/Desktop/Desktop items/NU/Sem_2/NLP - siwu/Project/outputs/bilstm_answerability_20251202_203753/bert_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved bert_comparison figures")
plt.close()

print("\n" + "="*50)
print("All figures generated successfully!")
print("="*50)
print("\nGenerated files:")
print("  1. training_loss.pdf/png")
print("  2. validation_metrics.pdf/png")
print("  3. model_comparison.pdf/png")
print("  4. answerability_learning.pdf/png")
print("  5. overfitting_analysis.pdf/png")
print("  6. bert_comparison.pdf/png")
print("\nThese can be included in your LaTeX report.")