import json
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np
from typing import List, Dict, Tuple, Optional

class Tokenizer:
    """
    Simple word-level tokenizer for BiLSTM.
    For better performance, you could use spaCy or NLTK.
    """
    def __init__(self, vocab_size=50000, max_len=512):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<START>', 3: '<END>'}
        self.vocab_built = False
        
    def build_vocab(self, texts: List[str]):
        """
        Build vocabulary from list of texts.
        """
        word_counts = Counter()
        
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)
        
        # Get most common words
        most_common = word_counts.most_common(self.vocab_size - len(self.word2idx))
        
        for idx, (word, _) in enumerate(most_common, start=len(self.word2idx)):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        self.vocab_built = True
        print(f"Vocabulary built with {len(self.word2idx)} tokens")
        
    def encode(self, text: str, max_len: Optional[int] = None) -> List[int]:
        """
        Convert text to token ids.
        """
        if not self.vocab_built:
            raise ValueError("Vocabulary not built. Call build_vocab() first.")
        
        max_len = max_len or self.max_len
        words = text.lower().split()
        
        # Convert words to indices
        token_ids = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in words]
        
        # Truncate if too long
        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Convert token ids back to text.
        """
        words = [self.idx2word.get(idx, '<UNK>') for idx in token_ids]
        return ' '.join(words)
    
    def save(self, path: str):
        """Save tokenizer to file."""
        state = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'vocab_size': self.vocab_size,
            'max_len': self.max_len
        }
        with open(path, 'w') as f:
            json.dump(state, f)
    
    def load(self, path: str):
        """Load tokenizer from file."""
        with open(path, 'r') as f:
            state = json.load(f)
        self.word2idx = state['word2idx']
        self.idx2word = {int(k): v for k, v in state['idx2word'].items()}
        self.vocab_size = state['vocab_size']
        self.max_len = state['max_len']
        self.vocab_built = True


class CUADDataset(Dataset):
    """
    Dataset class for CUAD contract QA.
    """
    def __init__(self, data_path: str, tokenizer: Tokenizer, 
                 max_context_len: int = 512, max_question_len: int = 64):
        """
        Args:
            data_path: Path to CUAD JSON file
            tokenizer: Tokenizer instance
            max_context_len: Maximum context length
            max_question_len: Maximum question length
        """
        self.tokenizer = tokenizer
        self.max_context_len = max_context_len
        self.max_question_len = max_question_len
        
        # Load CUAD data
        with open(data_path, 'r') as f:
            cuad_data = json.load(f)
        
        self.examples = self._process_cuad_data(cuad_data)
        print(f"Loaded {len(self.examples)} examples from {data_path}")
        
    def _process_cuad_data(self, cuad_data: Dict) -> List[Dict]:
        """
        Process CUAD JSON format into individual examples.
        CUAD format: {data: [{title: ..., paragraphs: [{context: ..., qas: [...]}]}]}
        """
        examples = []
        
        for article in cuad_data['data']:
            title = article.get('title', '')
            
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                
                for qa in paragraph['qas']:
                    question = qa['question']
                    question_id = qa['id']
                    is_impossible = qa.get('is_impossible', False)
                    
                    # Handle answer extraction
                    if is_impossible or not qa.get('answers'):
                        # No answer exists
                        example = {
                            'question_id': question_id,
                            'question': question,
                            'context': context,
                            'answer_text': '',
                            'answer_start': -1,
                            'answer_end': -1,
                            'is_impossible': True
                        }
                    else:
                        # Take first answer (CUAD may have multiple)
                        answer = qa['answers'][0]
                        answer_text = answer['text']
                        answer_start_char = answer['answer_start']
                        
                        example = {
                            'question_id': question_id,
                            'question': question,
                            'context': context,
                            'answer_text': answer_text,
                            'answer_start_char': answer_start_char,
                            'is_impossible': False
                        }
                    
                    examples.append(example)
        
        return examples
    
    def _find_token_positions(self, context: str, answer_start_char: int, 
                             answer_text: str) -> Tuple[int, int]:
        """
        Convert character positions to token positions.
        This is approximate for word-level tokenization.
        """
        if answer_start_char == -1:
            return -1, -1
        
        # Tokenize context
        context_words = context.lower().split()
        
        # Find approximate token position by counting words before answer
        char_pos = 0
        token_start = -1
        
        for idx, word in enumerate(context_words):
            if char_pos >= answer_start_char:
                token_start = idx
                break
            char_pos += len(word) + 1  # +1 for space
        
        if token_start == -1:
            return -1, -1
        
        # Estimate end position based on answer length
        answer_words = answer_text.lower().split()
        token_end = min(token_start + len(answer_words) - 1, len(context_words) - 1)
        
        return token_start, token_end
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize question and context
        question_ids = self.tokenizer.encode(example['question'], self.max_question_len)
        context_ids = self.tokenizer.encode(example['context'], self.max_context_len)
        
        # Get answer positions in tokens
        if example['is_impossible']:
            start_position = -1
            end_position = -1
        else:
            start_position, end_position = self._find_token_positions(
                example['context'],
                example.get('answer_start_char', -1),
                example['answer_text']
            )
            
            # Ensure positions are within bounds
            if start_position >= len(context_ids):
                start_position = -1
                end_position = -1
            elif end_position >= len(context_ids):
                end_position = len(context_ids) - 1
        
        return {
            'question_id': example['question_id'],
            'question_ids': question_ids,
            'context_ids': context_ids,
            'start_position': start_position,
            'end_position': end_position,
            'is_impossible': example['is_impossible'],
            'original_context': example['context'],
            'original_question': example['question'],
            'answer_text': example['answer_text']
        }


def collate_fn(batch):
    """
    Custom collate function to pad sequences to same length in batch.
    """
    # Find max lengths in batch
    max_question_len = max(len(item['question_ids']) for item in batch)
    max_context_len = max(len(item['context_ids']) for item in batch)
    
    batch_size = len(batch)
    
    # Initialize tensors
    question_ids = torch.zeros(batch_size, max_question_len, dtype=torch.long)
    context_ids = torch.zeros(batch_size, max_context_len, dtype=torch.long)
    question_mask = torch.zeros(batch_size, max_question_len, dtype=torch.long)
    context_mask = torch.zeros(batch_size, max_context_len, dtype=torch.long)
    start_positions = torch.full((batch_size,), -1, dtype=torch.long)
    end_positions = torch.full((batch_size,), -1, dtype=torch.long)
    
    # Fill tensors
    question_ids_list = []
    context_ids_list = []
    is_impossible_list = []
    original_contexts = []
    original_questions = []
    answer_texts = []
    
    for i, item in enumerate(batch):
        q_len = len(item['question_ids'])
        c_len = len(item['context_ids'])
        
        question_ids[i, :q_len] = torch.tensor(item['question_ids'])
        context_ids[i, :c_len] = torch.tensor(item['context_ids'])
        question_mask[i, :q_len] = 1
        context_mask[i, :c_len] = 1
        
        if item['start_position'] >= 0:
            start_positions[i] = item['start_position']
            end_positions[i] = item['end_position']
        
        question_ids_list.append(item['question_id'])
        is_impossible_list.append(item['is_impossible'])
        original_contexts.append(item['original_context'])
        original_questions.append(item['original_question'])
        answer_texts.append(item['answer_text'])
    
    return {
        'question_ids': question_ids,
        'context_ids': context_ids,
        'question_mask': question_mask,
        'context_mask': context_mask,
        'start_positions': start_positions,
        'end_positions': end_positions,
        'question_id_list': question_ids_list,
        'is_impossible': is_impossible_list,
        'original_contexts': original_contexts,
        'original_questions': original_questions,
        'answer_texts': answer_texts
    }


def create_dataloaders(train_path: str, val_path: str, tokenizer: Tokenizer,
                       batch_size: int = 8, max_context_len: int = 512,
                       max_question_len: int = 64, num_workers: int = 0):
    """
    Create train and validation dataloaders.
    """
    # Create datasets
    train_dataset = CUADDataset(train_path, tokenizer, max_context_len, max_question_len)
    val_dataset = CUADDataset(val_path, tokenizer, max_context_len, max_question_len)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    
    return train_loader, val_loader


# Example usage
if __name__ == "__main__":
    # Initialize tokenizer
    tokenizer = Tokenizer(vocab_size=50000, max_len=512)
    
    # Example: Build vocabulary (you'd do this with actual CUAD data)
    sample_texts = [
        "This agreement is made between party A and party B",
        "The contract term shall be for a period of one year",
        "Either party may terminate this agreement with written notice"
    ]
    tokenizer.build_vocab(sample_texts)
    
    # Example encoding
    text = "This is a sample contract clause"
    encoded = tokenizer.encode(text, max_len=20)
    decoded = tokenizer.decode(encoded)
    
    print(f"Original: {text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"\nVocabulary size: {len(tokenizer.word2idx)}")