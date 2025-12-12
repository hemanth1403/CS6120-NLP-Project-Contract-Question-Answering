import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    """
    Additive (Bahdanau) attention mechanism for question-aware context representation.
    """
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Attention weights
        self.W_c = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.W_q = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, context_output, question_output, context_mask=None):
        """
        Args:
            context_output: [batch, context_len, hidden_dim * 2]
            question_output: [batch, question_len, hidden_dim * 2]
            context_mask: [batch, context_len]
            
        Returns:
            attended_context: [batch, context_len, hidden_dim * 4]
            attention_weights: [batch, context_len, question_len]
        """
        batch_size = context_output.size(0)
        context_len = context_output.size(1)
        question_len = question_output.size(1)
        
        # Project context and question
        context_proj = self.W_c(context_output)
        question_proj = self.W_q(question_output)
        
        # Expand for broadcasting
        context_expanded = context_proj.unsqueeze(2)
        question_expanded = question_proj.unsqueeze(1)
        
        # Additive attention
        combined = torch.tanh(context_expanded + question_expanded)
        attention_scores = self.v(combined).squeeze(-1)
        
        # Clamp attention scores
        attention_scores = torch.clamp(attention_scores, min=-10, max=10)
        
        # Apply mask
        if context_mask is not None:
            attention_scores = attention_scores.masked_fill(
                context_mask.unsqueeze(2) == 0, -10.0
            )
        
        # Softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Check for NaN
        if torch.isnan(attention_weights).any():
            print("Warning: NaN in attention weights")
            attention_weights = torch.ones_like(attention_weights) / question_len
        
        # Weighted sum
        attended_question = torch.bmm(attention_weights, question_output)
        attended_context = torch.cat([context_output, attended_question], dim=-1)
        
        return attended_context, attention_weights


class BiLSTMQAWithAnswerability(nn.Module):
    """
    BiLSTM with Attention for Contract QA + Answerability Classification.
    Predicts: (1) start position, (2) end position, (3) has_answer (binary)
    """
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=128, 
                 num_layers=2, dropout=0.3, pretrained_embeddings=None):
        super(BiLSTMQAWithAnswerability, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = True
        else:
            nn.init.normal_(self.embedding.weight, mean=0, std=0.1)
        
        self.embedding_dropout = nn.Dropout(dropout)
        
        # BiLSTM for question encoding
        self.question_lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # BiLSTM for context encoding
        self.context_lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Initialize LSTM weights
        self._init_lstm_weights()
        
        # Attention layer
        self.attention = AttentionLayer(hidden_dim)
        
        # Output layers for start and end positions
        self.start_output = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        self.end_output = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # NEW: Answerability classification head
        # Uses pooled question representation + pooled context representation
        self.answerability_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),  # question + context pooled
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # Binary classification: has_answer or not
        )
        
        # Initialize output weights
        self._init_output_weights()
    
    def _init_lstm_weights(self):
        """Initialize LSTM weights with Xavier uniform"""
        for lstm in [self.question_lstm, self.context_lstm]:
            for name, param in lstm.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
    
    def _init_output_weights(self):
        """Initialize output layer weights"""
        for module in [self.start_output, self.end_output, self.answerability_classifier]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
        
    def forward(self, question_ids, context_ids, question_mask=None, context_mask=None):
        """
        Args:
            question_ids: [batch, question_len]
            context_ids: [batch, context_len]
            question_mask: [batch, question_len]
            context_mask: [batch, context_len]
            
        Returns:
            start_logits: [batch, context_len]
            end_logits: [batch, context_len]
            answerability_logits: [batch, 2]  # NEW
            attention_weights: [batch, context_len, question_len]
        """
        batch_size = context_ids.size(0)
        
        # Embed
        question_embedded = self.embedding_dropout(self.embedding(question_ids))
        context_embedded = self.embedding_dropout(self.embedding(context_ids))
        
        # Check for NaN
        if torch.isnan(question_embedded).any() or torch.isnan(context_embedded).any():
            print("Warning: NaN in embeddings")
            question_embedded = torch.nan_to_num(question_embedded, nan=0.0)
            context_embedded = torch.nan_to_num(context_embedded, nan=0.0)
        
        # Encode with BiLSTM
        question_output, _ = self.question_lstm(question_embedded)
        context_output, _ = self.context_lstm(context_embedded)
        
        # Check for NaN
        if torch.isnan(question_output).any():
            print("Warning: NaN in question LSTM")
            question_output = torch.nan_to_num(question_output, nan=0.0)
        if torch.isnan(context_output).any():
            print("Warning: NaN in context LSTM")
            context_output = torch.nan_to_num(context_output, nan=0.0)
        
        # NEW: Pool question and context for answerability
        # Mean pooling over sequence length
        question_pooled = torch.mean(question_output, dim=1)  # [batch, hidden_dim * 2]
        context_pooled = torch.mean(context_output, dim=1)    # [batch, hidden_dim * 2]
        
        # Concatenate pooled representations
        pooled_combined = torch.cat([question_pooled, context_pooled], dim=-1)  # [batch, hidden_dim * 4]
        
        # Predict answerability
        answerability_logits = self.answerability_classifier(pooled_combined)  # [batch, 2]
        
        # Apply attention
        attended_context, attention_weights = self.attention(
            context_output, question_output, context_mask
        )
        
        # Check for NaN
        if torch.isnan(attended_context).any():
            print("Warning: NaN in attended context")
            attended_context = torch.nan_to_num(attended_context, nan=0.0)
        
        # Predict start and end positions
        start_logits = self.start_output(attended_context).squeeze(-1)
        end_logits = self.end_output(attended_context).squeeze(-1)
        
        # Clamp logits
        start_logits = torch.clamp(start_logits, min=-10, max=10)
        end_logits = torch.clamp(end_logits, min=-10, max=10)
        
        # Apply mask
        if context_mask is not None:
            start_logits = start_logits.masked_fill(context_mask == 0, -10.0)
            end_logits = end_logits.masked_fill(context_mask == 0, -10.0)
        
        return start_logits, end_logits, answerability_logits, attention_weights


class BiLSTMQATrainerWithAnswerability:
    """
    Trainer for BiLSTM QA model with answerability classification.
    """
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu', 
                 answerability_weight=1.0):
        self.model = model.to(device)
        self.device = device
        self.answerability_weight = answerability_weight  # Weight for answerability loss
        
    def compute_loss(self, start_logits, end_logits, answerability_logits,
                    start_positions, end_positions, is_impossible):
        """
        Compute combined loss: span loss + answerability loss
        
        Args:
            start_logits: [batch, context_len]
            end_logits: [batch, context_len]
            answerability_logits: [batch, 2]
            start_positions: [batch] - positions or -1 for unanswerable
            end_positions: [batch] - positions or -1 for unanswerable
            is_impossible: [batch] - binary tensor (1 = unanswerable, 0 = answerable)
        """
        batch_size = start_logits.size(0)
        
        # 1. Answerability loss (always computed)
        answerability_labels = is_impossible.long()  # Convert to long for cross_entropy
        answerability_loss = F.cross_entropy(answerability_logits, answerability_labels)
        
        # 2. Span loss (only for answerable questions)
        valid_mask = (start_positions >= 0) & (end_positions >= 0)
        
        if valid_mask.sum() == 0:
            # No answerable examples - only answerability loss
            span_loss = torch.tensor(0.0, device=self.device)
        else:
            # Get valid examples
            valid_start_logits = start_logits[valid_mask]
            valid_end_logits = end_logits[valid_mask]
            valid_start_positions = start_positions[valid_mask]
            valid_end_positions = end_positions[valid_mask]
            
            # Compute span loss
            start_loss = F.cross_entropy(valid_start_logits, valid_start_positions)
            end_loss = F.cross_entropy(valid_end_logits, valid_end_positions)
            span_loss = (start_loss + end_loss) / 2
        
        # Combined loss
        total_loss = span_loss + self.answerability_weight * answerability_loss
        
        return total_loss, span_loss, answerability_loss
    
    def train_step(self, batch, optimizer, max_grad_norm=1.0):
        """
        Single training step with gradient clipping.
        """
        self.model.train()
        optimizer.zero_grad()
        
        # Move batch to device
        question_ids = batch['question_ids'].to(self.device)
        context_ids = batch['context_ids'].to(self.device)
        question_mask = batch['question_mask'].to(self.device)
        context_mask = batch['context_mask'].to(self.device)
        start_positions = batch['start_positions'].to(self.device)
        end_positions = batch['end_positions'].to(self.device)
        # Convert is_impossible from list of booleans to tensor
        is_impossible = torch.tensor([int(x) for x in batch['is_impossible']], dtype=torch.long, device=self.device)
        
        # Forward pass
        start_logits, end_logits, answerability_logits, _ = self.model(
            question_ids, context_ids, question_mask, context_mask
        )
        
        # Check for NaN
        if torch.isnan(start_logits).any() or torch.isnan(end_logits).any() or torch.isnan(answerability_logits).any():
            print("Warning: NaN detected in model outputs!")
            return 0.0, 0.0, 0.0
        
        # Compute loss
        total_loss, span_loss, answerability_loss = self.compute_loss(
            start_logits, end_logits, answerability_logits,
            start_positions, end_positions, is_impossible
        )
        
        # Check for NaN
        if torch.isnan(total_loss):
            print("Warning: NaN loss detected!")
            return 0.0, 0.0, 0.0
        
        # Backward pass
        total_loss.backward()
        
        # Check for NaN gradients
        has_nan_grad = False
        for name, param in self.model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"Warning: NaN gradient in {name}")
                has_nan_grad = True
                param.grad = torch.nan_to_num(param.grad, nan=0.0)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        return total_loss.item(), span_loss.item(), answerability_loss.item()
    
    def predict(self, batch):
        """
        Make predictions on a batch.
        """
        self.model.eval()
        
        with torch.no_grad():
            question_ids = batch['question_ids'].to(self.device)
            context_ids = batch['context_ids'].to(self.device)
            question_mask = batch['question_mask'].to(self.device)
            context_mask = batch['context_mask'].to(self.device)
            
            start_logits, end_logits, answerability_logits, attention_weights = self.model(
                question_ids, context_ids, question_mask, context_mask
            )
            
            # Get predicted positions
            start_pred = torch.argmax(start_logits, dim=-1)
            end_pred = torch.argmax(end_logits, dim=-1)
            
            # Get answerability prediction
            answerability_pred = torch.argmax(answerability_logits, dim=-1)  # 0=answerable, 1=unanswerable
            
            # If predicted as unanswerable, set positions to 0
            unanswerable_mask = answerability_pred == 1
            start_pred[unanswerable_mask] = 0
            end_pred[unanswerable_mask] = 0
            
            # Ensure end >= start
            end_pred = torch.max(end_pred, start_pred)
            
        return {
            'start_positions': start_pred.cpu().numpy(),
            'end_positions': end_pred.cpu().numpy(),
            'answerability_pred': answerability_pred.cpu().numpy(),
            'start_logits': start_logits.cpu().numpy(),
            'end_logits': end_logits.cpu().numpy(),
            'answerability_logits': answerability_logits.cpu().numpy(),
            'attention_weights': attention_weights.cpu().numpy()
        }


# For backward compatibility - alias to old name
BiLSTMQA = BiLSTMQAWithAnswerability
BiLSTMQATrainer = BiLSTMQATrainerWithAnswerability


if __name__ == "__main__":
    # Test the model
    vocab_size = 50000
    embedding_dim = 300
    hidden_dim = 128
    num_layers = 2
    dropout = 0.3
    
    model = BiLSTMQAWithAnswerability(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    )
    
    trainer = BiLSTMQATrainerWithAnswerability(model, answerability_weight=1.0)
    
    print("BiLSTM QA Model with Answerability:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Example forward pass
    batch_size = 4
    question_len = 20
    context_len = 256
    
    question_ids = torch.randint(0, vocab_size, (batch_size, question_len))
    context_ids = torch.randint(0, vocab_size, (batch_size, context_len))
    question_mask = torch.ones(batch_size, question_len)
    context_mask = torch.ones(batch_size, context_len)
    
    start_logits, end_logits, answerability_logits, attention_weights = model(
        question_ids, context_ids, question_mask, context_mask
    )
    
    print(f"\nOutput shapes:")
    print(f"  Start logits: {start_logits.shape}")
    print(f"  End logits: {end_logits.shape}")
    print(f"  Answerability logits: {answerability_logits.shape}")
    print(f"  Attention weights: {attention_weights.shape}")
    
    print("\n✓ Model with answerability head working correctly!")