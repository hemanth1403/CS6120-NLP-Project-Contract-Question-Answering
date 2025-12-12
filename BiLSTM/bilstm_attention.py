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
        self.W_c = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)  # for context (BiLSTM output)
        self.W_q = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)  # for question
        self.v = nn.Linear(hidden_dim, 1, bias=False)  # attention score
        
    def forward(self, context_output, question_output, context_mask=None):
        """
        Args:
            context_output: [batch, context_len, hidden_dim * 2] - BiLSTM output for context
            question_output: [batch, question_len, hidden_dim * 2] - BiLSTM output for question
            context_mask: [batch, context_len] - mask for padding tokens
            
        Returns:
            attended_context: [batch, context_len, hidden_dim * 2] - context with question attention
            attention_weights: [batch, context_len, question_len] - attention distribution
        """
        batch_size = context_output.size(0)
        context_len = context_output.size(1)
        question_len = question_output.size(1)
        
        # Project context and question
        # [batch, context_len, hidden_dim]
        context_proj = self.W_c(context_output)
        
        # [batch, question_len, hidden_dim]
        question_proj = self.W_q(question_output)
        
        # Expand for broadcasting: context_proj -> [batch, context_len, 1, hidden_dim]
        # question_proj -> [batch, 1, question_len, hidden_dim]
        context_expanded = context_proj.unsqueeze(2)
        question_expanded = question_proj.unsqueeze(1)
        
        # Additive attention: tanh(W_c * c + W_q * q)
        # [batch, context_len, question_len, hidden_dim]
        combined = torch.tanh(context_expanded + question_expanded)
        
        # Compute attention scores: [batch, context_len, question_len]
        attention_scores = self.v(combined).squeeze(-1)
        
        # ADDED: Clamp attention scores to prevent extreme values
        attention_scores = torch.clamp(attention_scores, min=-10, max=10)
        
        # Apply mask if provided (for padding)
        if context_mask is not None:
            attention_scores = attention_scores.masked_fill(
                context_mask.unsqueeze(2) == 0, -10.0  # CHANGED: Use -10 instead of -inf
            )
        
        # Softmax over question dimension
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # ADDED: Check for NaN in attention weights
        if torch.isnan(attention_weights).any():
            print("Warning: NaN in attention weights, replacing with uniform distribution")
            attention_weights = torch.ones_like(attention_weights) / question_len
        
        # Weighted sum of question representations
        # [batch, context_len, hidden_dim * 2]
        attended_question = torch.bmm(attention_weights, question_output)
        
        # Combine with original context
        attended_context = torch.cat([context_output, attended_question], dim=-1)
        
        return attended_context, attention_weights


class BiLSTMQA(nn.Module):
    """
    BiLSTM with Attention for Contract QA (CUAD dataset).
    Predicts start and end positions for answer spans.
    """
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=128, 
                 num_layers=2, dropout=0.3, pretrained_embeddings=None):
        super(BiLSTMQA, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = True  # Fine-tune embeddings
        
        # ADDED: Initialize embedding weights with smaller std
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
        
        # ADDED: Initialize LSTM weights properly
        self._init_lstm_weights()
        
        # Attention layer
        self.attention = AttentionLayer(hidden_dim)
        
        # Output layers for start and end positions
        # Input: concatenated context + attended question = hidden_dim * 4
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
        
        # ADDED: Initialize output layer weights
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
        for module in [self.start_output, self.end_output]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
        
    def forward(self, question_ids, context_ids, question_mask=None, context_mask=None):
        """
        Args:
            question_ids: [batch, question_len] - tokenized question
            context_ids: [batch, context_len] - tokenized contract context
            question_mask: [batch, question_len] - mask for question padding
            context_mask: [batch, context_len] - mask for context padding
            
        Returns:
            start_logits: [batch, context_len] - logits for start position
            end_logits: [batch, context_len] - logits for end position
        """
        batch_size = context_ids.size(0)
        
        # Embed question and context
        question_embedded = self.embedding_dropout(self.embedding(question_ids))
        context_embedded = self.embedding_dropout(self.embedding(context_ids))
        
        # ADDED: Check for NaN in embeddings
        if torch.isnan(question_embedded).any() or torch.isnan(context_embedded).any():
            print("Warning: NaN detected in embeddings!")
            question_embedded = torch.nan_to_num(question_embedded, nan=0.0)
            context_embedded = torch.nan_to_num(context_embedded, nan=0.0)
        
        # Encode question with BiLSTM
        # question_output: [batch, question_len, hidden_dim * 2]
        question_output, _ = self.question_lstm(question_embedded)
        
        # ADDED: Check for NaN in LSTM output
        if torch.isnan(question_output).any():
            print("Warning: NaN detected in question LSTM output!")
            question_output = torch.nan_to_num(question_output, nan=0.0)
        
        # Encode context with BiLSTM
        # context_output: [batch, context_len, hidden_dim * 2]
        context_output, _ = self.context_lstm(context_embedded)
        
        # ADDED: Check for NaN in LSTM output
        if torch.isnan(context_output).any():
            print("Warning: NaN detected in context LSTM output!")
            context_output = torch.nan_to_num(context_output, nan=0.0)
        
        # Apply attention mechanism
        # attended_context: [batch, context_len, hidden_dim * 4]
        attended_context, attention_weights = self.attention(
            context_output, question_output, context_mask
        )
        
        # ADDED: Check for NaN in attention output
        if torch.isnan(attended_context).any():
            print("Warning: NaN detected in attended context!")
            attended_context = torch.nan_to_num(attended_context, nan=0.0)
        
        # Predict start and end positions
        start_logits = self.start_output(attended_context).squeeze(-1)  # [batch, context_len]
        end_logits = self.end_output(attended_context).squeeze(-1)      # [batch, context_len]
        
        # ADDED: Clamp logits to prevent extreme values
        start_logits = torch.clamp(start_logits, min=-10, max=10)
        end_logits = torch.clamp(end_logits, min=-10, max=10)
        
        # Apply mask to logits (set padding positions to very negative value)
        if context_mask is not None:
            start_logits = start_logits.masked_fill(context_mask == 0, -10.0)  # CHANGED: Use -10 instead of -inf
            end_logits = end_logits.masked_fill(context_mask == 0, -10.0)      # CHANGED: Use -10 instead of -inf
        
        return start_logits, end_logits, attention_weights


class BiLSTMQATrainer:
    """
    Trainer class for BiLSTM QA model.
    """
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        
    def compute_loss(self, start_logits, end_logits, start_positions, end_positions):
        """
        Compute cross-entropy loss for start and end positions.
        Handles cases where answer doesn't exist (position = -1).
        """
        # Filter out examples with no answer (position = -1)
        valid_mask = (start_positions >= 0) & (end_positions >= 0)
        
        if valid_mask.sum() == 0:
            # No valid examples in batch
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Get valid examples
        valid_start_logits = start_logits[valid_mask]
        valid_end_logits = end_logits[valid_mask]
        valid_start_positions = start_positions[valid_mask]
        valid_end_positions = end_positions[valid_mask]
        
        # Compute cross-entropy loss
        start_loss = F.cross_entropy(valid_start_logits, valid_start_positions)
        end_loss = F.cross_entropy(valid_end_logits, valid_end_positions)
        
        total_loss = (start_loss + end_loss) / 2
        return total_loss
    
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
        
        # Forward pass
        start_logits, end_logits, _ = self.model(
            question_ids, context_ids, question_mask, context_mask
        )
        
        # Check for NaN in outputs
        if torch.isnan(start_logits).any() or torch.isnan(end_logits).any():
            print("Warning: NaN detected in model outputs!")
            return 0.0
        
        # Compute loss
        loss = self.compute_loss(start_logits, end_logits, start_positions, end_positions)
        
        # Check for NaN in loss
        if torch.isnan(loss):
            print("Warning: NaN loss detected!")
            return 0.0
        
        # Backward pass
        loss.backward()
        
        # ADDED: Check for NaN gradients before clipping
        has_nan_grad = False
        for name, param in self.model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"Warning: NaN gradient in {name}")
                has_nan_grad = True
                param.grad = torch.nan_to_num(param.grad, nan=0.0)
        
        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        return loss.item()
    
    def predict(self, batch):
        """
        Make predictions on a batch.
        Returns predicted start and end positions.
        """
        self.model.eval()
        
        with torch.no_grad():
            question_ids = batch['question_ids'].to(self.device)
            context_ids = batch['context_ids'].to(self.device)
            question_mask = batch['question_mask'].to(self.device)
            context_mask = batch['context_mask'].to(self.device)
            
            start_logits, end_logits, attention_weights = self.model(
                question_ids, context_ids, question_mask, context_mask
            )
            
            # Get predicted positions
            start_pred = torch.argmax(start_logits, dim=-1)
            end_pred = torch.argmax(end_logits, dim=-1)
            
            # Ensure end >= start
            end_pred = torch.max(end_pred, start_pred)
            
        return {
            'start_positions': start_pred.cpu().numpy(),
            'end_positions': end_pred.cpu().numpy(),
            'start_logits': start_logits.cpu().numpy(),
            'end_logits': end_logits.cpu().numpy(),
            'attention_weights': attention_weights.cpu().numpy()
        }


# Example usage
if __name__ == "__main__":
    # Model parameters
    vocab_size = 50000
    embedding_dim = 300
    hidden_dim = 128
    num_layers = 2
    dropout = 0.3
    
    # Create model
    model = BiLSTMQA(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    )
    
    # Create trainer
    trainer = BiLSTMQATrainer(model)
    
    print("BiLSTM QA Model Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Example forward pass
    batch_size = 4
    question_len = 20
    context_len = 512
    
    question_ids = torch.randint(0, vocab_size, (batch_size, question_len))
    context_ids = torch.randint(0, vocab_size, (batch_size, context_len))
    question_mask = torch.ones(batch_size, question_len)
    context_mask = torch.ones(batch_size, context_len)
    
    start_logits, end_logits, attention_weights = model(
        question_ids, context_ids, question_mask, context_mask
    )
    
    print(f"\nExample output shapes:")
    print(f"Start logits: {start_logits.shape}")
    print(f"End logits: {end_logits.shape}")
    print(f"Attention weights: {attention_weights.shape}")