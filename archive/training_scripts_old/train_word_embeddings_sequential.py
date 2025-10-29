"""
Train word embeddings that respect phoneme SEQUENCE.
Each word embedding is learned by predicting its phoneme sequence in order.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent))
from src.phonolex.embeddings.english_data_loader import EnglishPhonologyLoader


class SequentialPhonologyDataset(Dataset):
    """Dataset that learns word embeddings from phoneme sequences."""

    def __init__(self, loader: EnglishPhonologyLoader):
        self.loader = loader

        # Create phoneme vocabulary (with special tokens)
        all_phonemes = set()
        for phonemes in loader.lexicon.values():
            all_phonemes.update(phonemes)

        # Add special tokens
        self.PAD = '<PAD>'
        self.phoneme_to_id = {self.PAD: 0}
        self.phoneme_to_id.update({p: i + 1 for i, p in enumerate(sorted(all_phonemes))})
        self.id_to_phoneme = {i: p for p, i in self.phoneme_to_id.items()}

        # Create word vocabulary
        self.word_to_id = {word: i for i, word in enumerate(loader.lexicon.keys())}
        self.id_to_word = {i: word for word, i in self.word_to_id.items()}

        # Find max phoneme sequence length
        self.max_length = max(len(phonemes) for phonemes in loader.lexicon.values())

        print(f"\n{'='*70}")
        print(f"SEQUENTIAL PHONOLOGY DATASET")
        print(f"{'='*70}")
        print(f"Vocabulary: {len(self.word_to_id):,} words")
        print(f"Phonemes: {len(self.phoneme_to_id)} unique phonemes (+ PAD)")
        print(f"Max phoneme length: {self.max_length}")

        # Prepare data
        self.data = []
        for word, phonemes in loader.lexicon.items():
            word_id = self.word_to_id[word]
            phoneme_ids = [self.phoneme_to_id[p] for p in phonemes]
            self.data.append((word_id, phoneme_ids))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        word_id, phoneme_ids = self.data[idx]

        # Pad phoneme sequence
        padded = phoneme_ids + [self.phoneme_to_id[self.PAD]] * (self.max_length - len(phoneme_ids))

        return word_id, torch.LongTensor(padded), len(phoneme_ids)


class SequentialWordEmbedding(nn.Module):
    """
    Word embeddings that encode phoneme sequence information.

    Architecture:
    - Word embedding (what we learn)
    - LSTM to process phoneme sequences
    - Word embedding must predict its phoneme sequence
    """

    def __init__(self, num_words: int, num_phonemes: int,
                 word_dim: int = 64, phoneme_dim: int = 32, hidden_dim: int = 64):
        super().__init__()

        # Word embeddings (main output)
        self.word_embeddings = nn.Embedding(num_words, word_dim)

        # Phoneme embeddings for sequence modeling
        self.phoneme_embeddings = nn.Embedding(num_phonemes, phoneme_dim)

        # LSTM to encode phoneme sequences
        self.phoneme_lstm = nn.LSTM(phoneme_dim, hidden_dim, batch_first=True)

        # Projection from word embedding to LSTM hidden state
        self.word_to_hidden = nn.Linear(word_dim, hidden_dim)

        # Output projection to predict next phoneme
        self.hidden_to_phoneme = nn.Linear(hidden_dim, num_phonemes)

        # Initialize
        nn.init.xavier_uniform_(self.word_embeddings.weight)
        nn.init.xavier_uniform_(self.phoneme_embeddings.weight)

    def forward(self, word_ids, phoneme_seqs, lengths):
        """
        Learn word embeddings by predicting phoneme sequences.

        Args:
            word_ids: (batch_size,) word indices
            phoneme_seqs: (batch_size, max_len) phoneme sequences
            lengths: (batch_size,) actual sequence lengths

        Returns:
            loss: reconstruction loss
        """
        batch_size = word_ids.size(0)
        max_len = phoneme_seqs.size(1)

        # Get word embeddings
        word_emb = self.word_embeddings(word_ids)  # (batch, word_dim)

        # Project to LSTM hidden state (this is the initialization)
        h0 = self.word_to_hidden(word_emb).unsqueeze(0)  # (1, batch, hidden)
        c0 = torch.zeros_like(h0)  # (1, batch, hidden)

        # Embed phoneme sequences
        phoneme_emb = self.phoneme_embeddings(phoneme_seqs)  # (batch, max_len, phoneme_dim)

        # Run LSTM
        lstm_out, _ = self.phoneme_lstm(phoneme_emb, (h0, c0))  # (batch, max_len, hidden)

        # Predict next phoneme at each position
        logits = self.hidden_to_phoneme(lstm_out)  # (batch, max_len, num_phonemes)

        # Compute loss (predict each phoneme from previous context)
        # Shift target by 1 (predict next phoneme)
        target = phoneme_seqs  # (batch, max_len)

        # Flatten for cross entropy
        logits_flat = logits.view(-1, logits.size(-1))  # (batch*max_len, num_phonemes)
        target_flat = target.view(-1)  # (batch*max_len,)

        # Mask out padding
        mask = torch.arange(max_len, device=phoneme_seqs.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask_flat = mask.view(-1)

        # Loss only on non-padded positions
        loss = nn.functional.cross_entropy(logits_flat[mask_flat], target_flat[mask_flat])

        return loss

    def get_word_embedding(self, word_id):
        """Get the learned word embedding."""
        return self.word_embeddings(word_id)


def collate_fn(batch):
    """Custom collate to handle variable-length sequences."""
    word_ids, phoneme_seqs, lengths = zip(*batch)

    word_ids = torch.LongTensor(word_ids)
    phoneme_seqs = torch.stack(phoneme_seqs)
    lengths = torch.LongTensor(lengths)

    return word_ids, phoneme_seqs, lengths


def train():
    # Configuration
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*70}")
    print(f"Device: {device}")

    # Load data
    loader = EnglishPhonologyLoader()
    dataset = SequentialPhonologyDataset(loader)

    dataloader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    # Create model
    model = SequentialWordEmbedding(
        num_words=len(dataset.word_to_id),
        num_phonemes=len(dataset.phoneme_to_id),
        word_dim=64,
        phoneme_dim=32,
        hidden_dim=64
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 20
    print(f"\n{'='*70}")
    print(f"TRAINING")
    print(f"{'='*70}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {dataloader.batch_size}")
    print(f"Total batches per epoch: {len(dataloader):,}")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for word_ids, phoneme_seqs, lengths in pbar:
            # Move to device
            word_ids = word_ids.to(device)
            phoneme_seqs = phoneme_seqs.to(device)
            lengths = lengths.to(device)

            # Forward pass
            loss = model(word_ids, phoneme_seqs, lengths)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} - Average loss: {avg_loss:.4f}")

    # Save model
    output_dir = Path("models/word_embeddings_sequential")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'word_to_id': dataset.word_to_id,
        'id_to_word': dataset.id_to_word,
        'phoneme_to_id': dataset.phoneme_to_id,
        'id_to_phoneme': dataset.id_to_phoneme,
        'word_dim': 64,
        'phoneme_dim': 32,
        'hidden_dim': 64,
    }, output_dir / "model.pt")

    print(f"\n{'='*70}")
    print(f"Model saved to: {output_dir / 'model.pt'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    train()
