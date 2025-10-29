"""
Train contextual phoneme embeddings using transformers.
Like BERT, but for phonemes - each phoneme's representation depends on context.

Architecture:
- Phoneme embeddings + positional encoding
- Transformer encoder (self-attention across phonemes in a word)
- Masked phoneme prediction (like BERT's MLM)

Result: Same phoneme gets different representations in different contexts.
Example: /t/ in "cat" vs "bat" vs "take" will have different contextual embeddings.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from tqdm import tqdm
import sys
import math
import random

sys.path.insert(0, str(Path(__file__).parent))
from src.phonolex.embeddings.english_data_loader import EnglishPhonologyLoader


class ContextualPhonemeDataset(Dataset):
    """Dataset for training contextual phoneme embeddings with masked language modeling."""

    def __init__(self, loader: EnglishPhonologyLoader, mask_prob: float = 0.15):
        self.loader = loader
        self.mask_prob = mask_prob

        # Create phoneme vocabulary with special tokens
        all_phonemes = set()
        for phonemes in loader.lexicon.values():
            all_phonemes.update(phonemes)

        # Special tokens
        self.PAD = '<PAD>'
        self.MASK = '<MASK>'
        self.CLS = '<CLS>'  # Start of sequence
        self.SEP = '<SEP>'  # End of sequence

        self.phoneme_to_id = {
            self.PAD: 0,
            self.MASK: 1,
            self.CLS: 2,
            self.SEP: 3,
        }
        self.phoneme_to_id.update({p: i + 4 for i, p in enumerate(sorted(all_phonemes))})
        self.id_to_phoneme = {i: p for p, i in self.phoneme_to_id.items()}

        # Create word vocabulary (for downstream use)
        self.word_to_id = {word: i for i, word in enumerate(loader.lexicon.keys())}
        self.id_to_word = {i: word for word, i in self.word_to_id.items()}

        # Max length (add 2 for CLS and SEP)
        self.max_length = max(len(phonemes) for phonemes in loader.lexicon.values()) + 2

        print(f"\n{'='*70}")
        print(f"CONTEXTUAL PHONEME DATASET")
        print(f"{'='*70}")
        print(f"Vocabulary: {len(self.word_to_id):,} words")
        print(f"Phonemes: {len(self.phoneme_to_id)} (including special tokens)")
        print(f"Max sequence length: {self.max_length}")
        print(f"Mask probability: {mask_prob}")

        # Prepare data
        self.data = []
        for word, phonemes in loader.lexicon.items():
            word_id = self.word_to_id[word]
            # Add CLS and SEP tokens
            phoneme_ids = [self.phoneme_to_id[self.CLS]] + \
                          [self.phoneme_to_id[p] for p in phonemes] + \
                          [self.phoneme_to_id[self.SEP]]
            self.data.append((word_id, phoneme_ids))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        word_id, phoneme_ids = self.data[idx]
        seq_len = len(phoneme_ids)

        # Create input and target
        input_ids = phoneme_ids.copy()
        target_ids = phoneme_ids.copy()
        mask_positions = []

        # Mask random phonemes (not CLS/SEP/PAD)
        for i in range(1, seq_len - 1):  # Skip CLS and SEP
            if random.random() < self.mask_prob:
                mask_positions.append(i)
                # 80% MASK, 10% random, 10% unchanged
                rand = random.random()
                if rand < 0.8:
                    input_ids[i] = self.phoneme_to_id[self.MASK]
                elif rand < 0.9:
                    # Random phoneme (excluding special tokens)
                    input_ids[i] = random.randint(4, len(self.phoneme_to_id) - 1)
                # else: keep unchanged

        # Pad sequences
        padding_length = self.max_length - seq_len
        input_ids += [self.phoneme_to_id[self.PAD]] * padding_length
        target_ids += [self.phoneme_to_id[self.PAD]] * padding_length

        # Attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * seq_len + [0] * padding_length

        return (
            word_id,
            torch.LongTensor(input_ids),
            torch.LongTensor(target_ids),
            torch.LongTensor(attention_mask),
            torch.LongTensor(mask_positions) if mask_positions else torch.LongTensor([0])
        )


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x + positional encoding
        """
        return x + self.pe[:x.size(1), :]


class ContextualPhonemeEmbedding(nn.Module):
    """
    Transformer-based contextual phoneme embeddings.

    Like BERT, but for phonemes:
    - Each phoneme gets a contextual representation based on surrounding phonemes
    - Trained with masked phoneme prediction
    """

    def __init__(self, num_phonemes: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 3, dim_feedforward: int = 512, max_len: int = 50):
        super().__init__()

        # Phoneme embeddings (input)
        self.phoneme_embeddings = nn.Embedding(num_phonemes, d_model, padding_idx=0)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection for masked phoneme prediction
        self.output_projection = nn.Linear(d_model, num_phonemes)

        # Initialize
        nn.init.xavier_uniform_(self.phoneme_embeddings.weight)
        nn.init.xavier_uniform_(self.output_projection.weight)

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: (batch, seq_len) phoneme indices
            attention_mask: (batch, seq_len) attention mask

        Returns:
            logits: (batch, seq_len, num_phonemes) predictions for each position
            contextual_embeddings: (batch, seq_len, d_model) contextual representations
        """
        # Embed phonemes
        x = self.phoneme_embeddings(input_ids)  # (batch, seq_len, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Create attention mask for transformer (True = ignore)
        # Transformer expects (seq_len, seq_len) or (batch, seq_len)
        src_key_padding_mask = (attention_mask == 0)  # (batch, seq_len)

        # Apply transformer
        contextual_embeddings = self.transformer_encoder(
            x,
            src_key_padding_mask=src_key_padding_mask
        )  # (batch, seq_len, d_model)

        # Project to phoneme vocabulary
        logits = self.output_projection(contextual_embeddings)  # (batch, seq_len, num_phonemes)

        return logits, contextual_embeddings

    def get_contextual_embeddings(self, input_ids, attention_mask):
        """Get contextual embeddings for phonemes."""
        with torch.no_grad():
            _, contextual_embeddings = self.forward(input_ids, attention_mask)
        return contextual_embeddings


def collate_fn(batch):
    """Custom collate function."""
    word_ids, input_ids, target_ids, attention_masks, mask_positions = zip(*batch)

    return (
        torch.LongTensor(word_ids),
        torch.stack(input_ids),
        torch.stack(target_ids),
        torch.stack(attention_masks),
        mask_positions  # Keep as tuple since lengths vary
    )


def train():
    # Configuration
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*70}")
    print(f"Device: {device}")

    # Load data
    loader = EnglishPhonologyLoader()
    dataset = ContextualPhonemeDataset(loader, mask_prob=0.15)

    dataloader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    # Create model
    model = ContextualPhonemeEmbedding(
        num_phonemes=len(dataset.phoneme_to_id),
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=512,
        max_len=dataset.max_length
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.phoneme_to_id[dataset.PAD])

    # Training loop
    num_epochs = 15
    print(f"\n{'='*70}")
    print(f"TRAINING")
    print(f"{'='*70}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {dataloader.batch_size}")
    print(f"Total batches per epoch: {len(dataloader):,}")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_predictions = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for word_ids, input_ids, target_ids, attention_masks, mask_positions in pbar:
            # Move to device
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            attention_masks = attention_masks.to(device)

            # Forward pass
            logits, _ = model(input_ids, attention_masks)

            # Compute loss (only on non-padded positions)
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1)
            )

            # Compute accuracy on masked positions
            predictions = logits.argmax(dim=-1)
            mask = attention_masks.bool()
            correct = (predictions == target_ids) & mask
            total_correct += correct.sum().item()
            total_predictions += mask.sum().item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            accuracy = 100.0 * total_correct / total_predictions if total_predictions > 0 else 0
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{accuracy:.2f}%'})

        avg_loss = total_loss / len(dataloader)
        avg_accuracy = 100.0 * total_correct / total_predictions
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.2f}%")

    # Save model
    output_dir = Path("models/contextual_phoneme_embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'word_to_id': dataset.word_to_id,
        'id_to_word': dataset.id_to_word,
        'phoneme_to_id': dataset.phoneme_to_id,
        'id_to_phoneme': dataset.id_to_phoneme,
        'd_model': 128,
        'nhead': 4,
        'num_layers': 3,
        'max_length': dataset.max_length,
    }, output_dir / "model.pt")

    print(f"\n{'='*70}")
    print(f"Model saved to: {output_dir / 'model.pt'}")
    print(f"{'='*70}")
    print(f"\nKey insight: This model learns CONTEXTUAL phoneme embeddings!")
    print(f"- Same phoneme in different contexts gets different representations")
    print(f"- /t/ in 'cat' vs 'take' vs 'bat' will have different embeddings")
    print(f"- Can aggregate contextual phoneme embeddings to get word embeddings")


if __name__ == "__main__":
    train()
