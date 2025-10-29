#!/usr/bin/env python3
"""
Word-Level Phonological Embeddings

Compose word embeddings from phoneme embeddings using various strategies:
1. Simple average
2. Weighted average (stress-sensitive)
3. RNN/LSTM encoding
4. Attention-based
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional
from pathlib import Path


class WordEmbedder:
    """
    Convert words to embeddings using phoneme embeddings
    """

    def __init__(self, model_path: str = "models/english_multitask/model.pt"):
        """
        Load phoneme embeddings

        Args:
            model_path: Path to trained phoneme embedding model
        """
        checkpoint = torch.load(model_path, map_location='cpu')

        self.phoneme_to_id = checkpoint['phoneme_to_id']
        self.id_to_phoneme = checkpoint['id_to_phoneme']
        self.embedding_dim = checkpoint['embedding_dim']

        # Extract phoneme embeddings
        if 'model_state_dict' in checkpoint:
            state = checkpoint['model_state_dict']
            if 'embeddings.weight' in state:
                self.phoneme_embeddings = state['embeddings.weight']
            else:
                raise ValueError("No embeddings found in model")
        else:
            raise ValueError("No model_state_dict in checkpoint")

        print(f"✓ Loaded phoneme embeddings: {len(self.phoneme_to_id)} phonemes × {self.embedding_dim}d")

    def phonemes_to_ids(self, phonemes: List[str]) -> List[int]:
        """Convert phoneme sequence to IDs"""
        ids = []
        for p in phonemes:
            if p in self.phoneme_to_id:
                ids.append(self.phoneme_to_id[p])
        return ids

    def average_embedding(self, phonemes: List[str]) -> Optional[np.ndarray]:
        """
        Simple average of phoneme embeddings

        Args:
            phonemes: List of phonemes

        Returns:
            embedding: (embedding_dim,) array, or None if no phonemes found
        """
        ids = self.phonemes_to_ids(phonemes)
        if not ids:
            return None

        ids_tensor = torch.tensor(ids, dtype=torch.long)
        embeddings = self.phoneme_embeddings[ids_tensor]

        return embeddings.mean(dim=0).numpy()

    def weighted_average_embedding(
        self,
        phonemes: List[str],
        weights: Optional[List[float]] = None
    ) -> Optional[np.ndarray]:
        """
        Weighted average (e.g., by stress or position)

        Args:
            phonemes: List of phonemes
            weights: Weight for each phoneme (if None, uses position-based weights)

        Returns:
            embedding: (embedding_dim,) array
        """
        ids = self.phonemes_to_ids(phonemes)
        if not ids:
            return None

        # Default: higher weight for start/end (psycholinguistic evidence)
        if weights is None:
            n = len(ids)
            if n == 1:
                weights = [1.0]
            elif n == 2:
                weights = [0.6, 0.4]
            else:
                # Emphasize first and last
                weights = [0.4] + [0.2 / (n - 2)] * (n - 2) + [0.4]

        weights = torch.tensor(weights, dtype=torch.float32)
        weights = weights / weights.sum()  # Normalize

        ids_tensor = torch.tensor(ids, dtype=torch.long)
        embeddings = self.phoneme_embeddings[ids_tensor]

        weighted = (embeddings * weights.unsqueeze(1)).sum(dim=0)

        return weighted.numpy()

    def onset_nucleus_coda_embedding(self, phonemes: List[str]) -> Optional[np.ndarray]:
        """
        Separate embeddings for onset, nucleus (vowel), coda

        Linguistically motivated: syllable structure matters
        """
        if not phonemes:
            return None

        # Simple vowel detection (crude but works)
        vowels = {'a', 'æ', 'e', 'ɛ', 'i', 'ɪ', 'o', 'ɔ', 'u', 'ʊ', 'ʌ', 'ɑ', 'ɝ', 'ə',
                  'aɪ', 'aʊ', 'eɪ', 'oʊ', 'ɔɪ'}

        # Find nucleus (first vowel)
        nucleus_idx = None
        for i, p in enumerate(phonemes):
            if p in vowels:
                nucleus_idx = i
                break

        if nucleus_idx is None:
            # No vowel, fall back to average
            return self.average_embedding(phonemes)

        # Split into onset / nucleus / coda
        onset = phonemes[:nucleus_idx]
        nucleus = [phonemes[nucleus_idx]]
        coda = phonemes[nucleus_idx + 1:]

        # Get embeddings
        parts = []
        for part in [onset, nucleus, coda]:
            if part:
                emb = self.average_embedding(part)
                if emb is not None:
                    parts.append(emb)

        if not parts:
            return None

        # Concatenate (this increases dimensionality)
        # Or could average with different weights
        # For now, weighted average: nucleus > onset > coda
        if len(parts) == 3:
            combined = 0.3 * parts[0] + 0.5 * parts[1] + 0.2 * parts[2]
        elif len(parts) == 2:
            if onset:
                combined = 0.4 * parts[0] + 0.6 * parts[1]
            else:
                combined = 0.6 * parts[0] + 0.4 * parts[1]
        else:
            combined = parts[0]

        return combined


class LSTMWordEmbedder(nn.Module):
    """
    Learn word embeddings using LSTM over phoneme sequence

    This is more powerful but requires training
    """

    def __init__(
        self,
        phoneme_embeddings: torch.Tensor,
        hidden_dim: int = 64,
        output_dim: int = 32
    ):
        super().__init__()

        self.phoneme_embeddings = nn.Embedding.from_pretrained(
            phoneme_embeddings,
            freeze=False  # Can fine-tune
        )

        self.lstm = nn.LSTM(
            input_size=phoneme_embeddings.size(1),
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, phoneme_ids: torch.Tensor, lengths: torch.Tensor):
        """
        Args:
            phoneme_ids: (batch, max_len)
            lengths: (batch,) actual lengths

        Returns:
            embeddings: (batch, output_dim)
        """
        # Embed phonemes
        embs = self.phoneme_embeddings(phoneme_ids)  # (batch, max_len, emb_dim)

        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            embs,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        # LSTM
        _, (hidden, _) = self.lstm(packed)

        # Use last hidden state
        hidden = hidden.squeeze(0)  # (batch, hidden_dim)

        # Project to output dim
        output = self.projection(hidden)

        return output


def demo():
    """Demo word embeddings"""

    print("=" * 70)
    print("WORD EMBEDDING DEMO")
    print("=" * 70)

    import sys
    sys.path.insert(0, '/Users/jneumann/Repos/PhonoLex')
    from src.phonolex.embeddings.english_data_loader import EnglishPhonologyLoader

    # Load
    embedder = WordEmbedder()
    loader = EnglishPhonologyLoader()

    # Test words
    test_words = ['cat', 'dog', 'phone', 'hello', 'world']

    print("\n1. Simple Average:")
    for word in test_words:
        if word in loader.lexicon:
            phonemes = loader.lexicon[word]
            emb = embedder.average_embedding(phonemes)
            if emb is not None:
                print(f"  {word:10s} /{' '.join(phonemes):20s} → {emb[:5]} (showing first 5 dims)")

    print("\n2. Weighted Average (position-based):")
    for word in test_words:
        if word in loader.lexicon:
            phonemes = loader.lexicon[word]
            emb = embedder.weighted_average_embedding(phonemes)
            if emb is not None:
                print(f"  {word:10s} /{' '.join(phonemes):20s} → {emb[:5]}")

    print("\n3. Onset-Nucleus-Coda:")
    for word in test_words:
        if word in loader.lexicon:
            phonemes = loader.lexicon[word]
            emb = embedder.onset_nucleus_coda_embedding(phonemes)
            if emb is not None:
                print(f"  {word:10s} /{' '.join(phonemes):20s} → {emb[:5]}")

    # Test similarity
    print("\n4. Word Similarity (using average embeddings):")

    def word_similarity(w1, w2):
        if w1 not in loader.lexicon or w2 not in loader.lexicon:
            return None
        e1 = embedder.average_embedding(loader.lexicon[w1])
        e2 = embedder.average_embedding(loader.lexicon[w2])
        if e1 is None or e2 is None:
            return None
        return np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))

    pairs = [
        ('cat', 'bat'),
        ('cat', 'dog'),
        ('phone', 'clone'),
        ('hello', 'yellow')
    ]

    for w1, w2 in pairs:
        sim = word_similarity(w1, w2)
        if sim is not None:
            print(f"  {w1} - {w2}: {sim:.4f}")


if __name__ == '__main__':
    demo()
