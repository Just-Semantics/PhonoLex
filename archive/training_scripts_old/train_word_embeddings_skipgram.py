"""
Train word embeddings using intra-word phoneme context.
Similar to Word2Vec skip-gram, but at the phonological level.
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


class PhonologicalWordDataset(Dataset):
    """Generate training pairs from intra-word phoneme context."""

    def __init__(self, loader: EnglishPhonologyLoader, window_size: int = 2):
        self.loader = loader
        self.window_size = window_size

        # Create phoneme vocabulary
        all_phonemes = set()
        for phonemes in loader.lexicon.values():
            all_phonemes.update(phonemes)

        self.phoneme_to_id = {p: i for i, p in enumerate(sorted(all_phonemes))}
        self.id_to_phoneme = {i: p for p, i in self.phoneme_to_id.items()}

        # Create word vocabulary
        self.word_to_id = {word: i for i, word in enumerate(loader.lexicon.keys())}
        self.id_to_word = {i: word for word, i in self.word_to_id.items()}

        print(f"\n{'='*70}")
        print(f"PHONOLOGICAL WORD DATASET")
        print(f"{'='*70}")
        print(f"Vocabulary: {len(self.word_to_id):,} words")
        print(f"Phonemes: {len(self.phoneme_to_id)} unique phonemes")
        print(f"Window size: {window_size}")

        # Generate training examples
        self.examples = []
        self._generate_examples()

    def _generate_examples(self):
        """Generate (word, target_word) pairs based on phoneme context similarity."""

        for word, phonemes in tqdm(self.loader.lexicon.items(), desc="Generating examples"):
            word_id = self.word_to_id[word]
            phoneme_ids = [self.phoneme_to_id[p] for p in phonemes]

            # Generate context pairs: for each phoneme, predict nearby phonemes
            for i, center_phoneme in enumerate(phoneme_ids):
                # Get context phonemes within window
                start = max(0, i - self.window_size)
                end = min(len(phoneme_ids), i + self.window_size + 1)

                for j in range(start, end):
                    if j != i:
                        context_phoneme = phoneme_ids[j]
                        # Store (word_id, center_phoneme, context_phoneme)
                        self.examples.append((word_id, center_phoneme, context_phoneme))

        print(f"  ✓ Generated {len(self.examples):,} training examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        word_id, center_phoneme, context_phoneme = self.examples[idx]
        return word_id, center_phoneme, context_phoneme


class PhonologicalWordEmbedding(nn.Module):
    """
    Word embeddings that learn from intra-word phoneme context.

    Architecture:
    - Word embeddings (main output)
    - Phoneme embeddings (for context prediction)
    - Each word learns to predict its constituent phonemes
    """

    def __init__(self, num_words: int, num_phonemes: int,
                 word_dim: int = 64, phoneme_dim: int = 32):
        super().__init__()

        # Word-level embeddings (what we want to learn)
        self.word_embeddings = nn.Embedding(num_words, word_dim)

        # Phoneme embeddings for context prediction
        self.phoneme_center = nn.Embedding(num_phonemes, phoneme_dim)
        self.phoneme_context = nn.Embedding(num_phonemes, phoneme_dim)

        # Project word embedding to phoneme space for prediction
        self.word_to_phoneme = nn.Linear(word_dim, phoneme_dim)

        # Initialize embeddings
        nn.init.xavier_uniform_(self.word_embeddings.weight)
        nn.init.xavier_uniform_(self.phoneme_center.weight)
        nn.init.xavier_uniform_(self.phoneme_context.weight)

    def forward(self, word_ids, center_phonemes, context_phonemes):
        """
        Predict context phonemes from word + center phoneme.

        Args:
            word_ids: (batch_size,) word indices
            center_phonemes: (batch_size,) center phoneme indices
            context_phonemes: (batch_size,) context phoneme indices

        Returns:
            loss: negative log likelihood
        """
        # Get embeddings
        word_emb = self.word_embeddings(word_ids)  # (batch, word_dim)
        center_emb = self.phoneme_center(center_phonemes)  # (batch, phoneme_dim)

        # Project word to phoneme space and combine with center phoneme
        word_phon = self.word_to_phoneme(word_emb)  # (batch, phoneme_dim)
        combined = word_phon + center_emb  # (batch, phoneme_dim)

        # Predict context phoneme
        context_emb = self.phoneme_context(context_phonemes)  # (batch, phoneme_dim)

        # Compute dot product (unnormalized log probability)
        score = (combined * context_emb).sum(dim=1)  # (batch,)

        # Negative sampling loss (we'll use all phonemes as negatives)
        # Get all context embeddings
        all_context = self.phoneme_context.weight  # (num_phonemes, phoneme_dim)

        # Compute scores for all phonemes
        all_scores = torch.matmul(combined, all_context.t())  # (batch, num_phonemes)

        # Log softmax
        log_probs = torch.log_softmax(all_scores, dim=1)

        # Get log prob of true context phoneme
        loss = -log_probs[torch.arange(len(context_phonemes)), context_phonemes].mean()

        return loss

    def get_word_embedding(self, word_id):
        """Get the learned word embedding."""
        return self.word_embeddings(word_id)


def train():
    # Configuration
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*70}")
    print(f"Device: {device}")

    # Load data
    loader = EnglishPhonologyLoader()
    dataset = PhonologicalWordDataset(loader, window_size=2)

    dataloader = DataLoader(
        dataset,
        batch_size=1024,
        shuffle=True,
        num_workers=0,  # MPS doesn't support multiprocessing
        pin_memory=False
    )

    # Create model
    model = PhonologicalWordEmbedding(
        num_words=len(dataset.word_to_id),
        num_phonemes=len(dataset.phoneme_to_id),
        word_dim=64,
        phoneme_dim=32
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
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
        for word_ids, center_phonemes, context_phonemes in pbar:
            # Move to device and ensure correct dtypes
            word_ids = word_ids.to(device).long()
            center_phonemes = center_phonemes.to(device).long()
            context_phonemes = context_phonemes.to(device).long()

            # Forward pass
            loss = model(word_ids, center_phonemes, context_phonemes)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} - Average loss: {avg_loss:.4f}")

    # Save model
    output_dir = Path("models/word_embeddings_skipgram")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'word_to_id': dataset.word_to_id,
        'id_to_word': dataset.id_to_word,
        'phoneme_to_id': dataset.phoneme_to_id,
        'id_to_phoneme': dataset.id_to_phoneme,
        'word_dim': 64,
        'phoneme_dim': 32,
    }, output_dir / "model.pt")

    print(f"\n{'='*70}")
    print(f"Model saved to: {output_dir / 'model.pt'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    train()
