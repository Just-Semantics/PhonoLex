#!/usr/bin/env python3
"""
Train Word-Level Phonological Embeddings with MPS support

Fixed version that works with Apple Silicon GPU by ensuring
proper dtype handling.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from collections import defaultdict
import random

import sys
sys.path.insert(0, '/Users/jneumann/Repos/PhonoLex')
from src.phonolex.embeddings.english_data_loader import EnglishPhonologyLoader


class WordPhonologyDataset(Dataset):
    """
    Fast dataset - precompute all pairs upfront
    """

    def __init__(self, loader: EnglishPhonologyLoader, num_minimal_pairs: int = 50000):
        self.loader = loader
        self.word_to_id = {word: i for i, word in enumerate(loader.lexicon.keys())}
        self.id_to_word = {i: w for w, i in self.word_to_id.items()}

        print(f"\nWord vocabulary: {len(self.word_to_id):,} words")
        print(f"Generating {num_minimal_pairs:,} training pairs...")

        self.examples = []

        words_list = list(self.word_to_id.keys())

        # Sample pairs randomly and check similarity
        for _ in tqdm(range(num_minimal_pairs), desc="Generating pairs"):
            w1, w2 = random.sample(words_list, 2)

            p1 = loader.lexicon[w1]
            p2 = loader.lexicon[w2]

            # Quick similarity based on length and phonemes
            len_diff = abs(len(p1) - len(p2))

            if len_diff == 0:
                # Same length - check phoneme overlap
                overlap = sum(1 for a, b in zip(p1, p2) if a == b)
                similarity = overlap / len(p1)
            elif len_diff == 1:
                # Edit distance 1 - high similarity
                similarity = 0.8
            else:
                # Different lengths - lower similarity
                similarity = max(0.0, 1.0 - len_diff / max(len(p1), len(p2)))

            self.examples.append((w1, w2, similarity))

        print(f"✓ Generated {len(self.examples):,} pairs")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        w1, w2, similarity = self.examples[idx]
        return {
            'word1_id': self.word_to_id[w1],
            'word2_id': self.word_to_id[w2],
            'similarity': similarity
        }


def collate_fn(batch):
    """Collate with explicit dtypes for MPS"""
    return {
        'word1_ids': torch.tensor([b['word1_id'] for b in batch], dtype=torch.long),
        'word2_ids': torch.tensor([b['word2_id'] for b in batch], dtype=torch.long),
        'similarities': torch.tensor([b['similarity'] for b in batch], dtype=torch.float32)
    }


class WordEmbeddingModel(nn.Module):
    """
    Word embedding model - MPS compatible
    """

    def __init__(self, num_words: int, embedding_dim: int = 64):
        super().__init__()
        self.embeddings = nn.Embedding(num_words, embedding_dim)
        nn.init.xavier_uniform_(self.embeddings.weight)

    def forward(self, word_ids):
        """Ensure proper dtype"""
        # Convert to long if needed (MPS requirement)
        if word_ids.dtype != torch.long:
            word_ids = word_ids.long()
        return self.embeddings(word_ids)

    def similarity(self, word1_ids, word2_ids):
        """Compute cosine similarity with proper dtypes"""
        emb1 = self.forward(word1_ids)
        emb2 = self.forward(word2_ids)

        # Normalize
        emb1 = nn.functional.normalize(emb1, p=2, dim=1)
        emb2 = nn.functional.normalize(emb2, p=2, dim=1)

        # Cosine similarity (element-wise multiply then sum)
        return (emb1 * emb2).sum(dim=1)

    def loss(self, word1_ids, word2_ids, target_similarities):
        """MSE loss"""
        pred_sim = self.similarity(word1_ids, word2_ids)

        # Ensure both are float32
        pred_sim = pred_sim.float()
        target_similarities = target_similarities.float()

        return nn.functional.mse_loss(pred_sim, target_similarities)


def train(device='auto'):
    """Train word embeddings"""

    print("=" * 70)
    print("WORD-LEVEL PHONOLOGICAL EMBEDDINGS (MPS-COMPATIBLE)")
    print("=" * 70)

    # Auto-detect device
    if device == 'auto':
        if torch.backends.mps.is_available():
            device = 'mps'
            print("\n✓ Using Apple Silicon GPU (MPS)")
        elif torch.cuda.is_available():
            device = 'cuda'
            print("\n✓ Using CUDA GPU")
        else:
            device = 'cpu'
            print("\n✓ Using CPU")

    device = torch.device(device)

    # Load data
    loader = EnglishPhonologyLoader()
    dataset = WordPhonologyDataset(loader, num_minimal_pairs=100000)

    dataloader = DataLoader(
        dataset,
        batch_size=1024,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # MPS doesn't support multiprocessing
    )

    # Create model
    model = WordEmbeddingModel(
        num_words=len(dataset.word_to_id),
        embedding_dim=64
    ).to(device)

    print(f"\nModel: {len(dataset.word_to_id):,} words → 64-dim embeddings")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    num_epochs = 20
    print(f"\nTraining for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            # Move to device
            word1_ids = batch['word1_ids'].to(device)
            word2_ids = batch['word2_ids'].to(device)
            similarities = batch['similarities'].to(device)

            # Forward
            loss = model.loss(word1_ids, word2_ids, similarities)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches
        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}")

    # Save
    output_dir = Path("models/word_embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Move to CPU for saving
    model = model.cpu()

    torch.save({
        'model_state_dict': model.state_dict(),
        'word_to_id': dataset.word_to_id,
        'id_to_word': dataset.id_to_word,
        'embedding_dim': 64
    }, output_dir / "model.pt")

    print(f"\n✓ Model saved to {output_dir / 'model.pt'}")

    # Test
    print("\n" + "=" * 70)
    print("TESTING")
    print("=" * 70)

    model.eval()

    def get_similar_words(word, top_k=10):
        if word not in dataset.word_to_id:
            return []

        with torch.no_grad():
            word_id = torch.tensor([dataset.word_to_id[word]], dtype=torch.long)
            word_emb = model(word_id)

            all_ids = torch.arange(len(dataset.word_to_id), dtype=torch.long)
            all_embs = model(all_ids)

            # Compute similarities
            word_emb_norm = nn.functional.normalize(word_emb, p=2, dim=1)
            all_embs_norm = nn.functional.normalize(all_embs, p=2, dim=1)

            sims = (all_embs_norm * word_emb_norm).sum(dim=1)

            # Top K
            top_indices = sims.argsort(descending=True)[1:top_k+1]

            results = []
            for idx in top_indices:
                w = dataset.id_to_word[idx.item()]
                s = sims[idx].item()
                results.append((w, s))

        return results

    test_words = ['cat', 'dog', 'run', 'running', 'make', 'phone']

    for word in test_words:
        print(f"\nWords similar to '{word}':")
        similar = get_similar_words(word, top_k=10)
        for w, s in similar:
            phones = ''.join(loader.lexicon.get(w, []))
            print(f"  {w:15s} /{phones:20s}  {s:.4f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'mps', 'cuda'])
    args = parser.parse_args()

    train(device=args.device)
