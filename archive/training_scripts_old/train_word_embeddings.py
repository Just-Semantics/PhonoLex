#!/usr/bin/env python3
"""
Train Word-Level Phonological Embeddings

Like Word2Vec but for pronunciation patterns.

Training signal:
1. Phonological neighborhood (minimal pairs, rhymes)
2. Morphological relationships (lemma ↔ inflection)
3. Phonotactic patterns (similar phoneme sequences)

Goal: Learn dense word embeddings where phonologically similar words
      have similar representations, capturing English phonology.
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
    Dataset for learning word-level phonological embeddings

    Training examples:
    1. Minimal pairs (cat/bat, cat/cap) - should be similar
    2. Rhymes (cat/bat/mat) - should be similar
    3. Morphological pairs (walk/walked) - should be related
    4. Random negatives (cat/elephant) - should be dissimilar
    """

    def __init__(self, loader: EnglishPhonologyLoader, min_word_freq: int = 1):
        self.loader = loader

        # Build word vocabulary (just use all words from lexicon)
        self.word_to_id = {word: i for i, word in enumerate(loader.lexicon.keys())}

        self.id_to_word = {i: w for w, i in self.word_to_id.items()}

        print(f"\nWord vocabulary: {len(self.word_to_id):,} words")

        # Generate training examples
        self._generate_examples()

    def _edit_distance(self, seq1, seq2):
        """Compute edit distance"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

        return dp[m][n]

    def _generate_examples(self):
        """Generate training examples"""
        print("\nGenerating training examples...")

        self.examples = []

        # 1. Minimal pairs (1 phoneme difference)
        print("  Finding minimal pairs...")
        minimal_pairs = []
        words_list = list(self.word_to_id.keys())

        for i, w1 in enumerate(words_list):
            if w1 not in self.loader.lexicon:
                continue
            p1 = self.loader.lexicon[w1]

            # Sample some words to check (not all - too slow)
            for w2 in random.sample(words_list, min(500, len(words_list))):
                if w1 == w2 or w2 not in self.loader.lexicon:
                    continue

                p2 = self.loader.lexicon[w2]

                dist = self._edit_distance(p1, p2)

                if dist == 1:  # Minimal pair
                    minimal_pairs.append((w1, w2, 1.0))  # High similarity

            if i % 1000 == 0 and i > 0:
                print(f"    Processed {i}/{len(words_list)} words, found {len(minimal_pairs)} pairs")

        print(f"  ✓ Found {len(minimal_pairs):,} minimal pairs")

        # 2. Rhymes (same suffix)
        print("  Finding rhymes...")
        rhymes = []

        # Group by last 2 phonemes
        suffix_groups = defaultdict(list)
        for word in self.word_to_id.keys():
            if word not in self.loader.lexicon:
                continue
            phones = self.loader.lexicon[word]
            if len(phones) >= 2:
                suffix = tuple(phones[-2:])
                suffix_groups[suffix].append(word)

        # Create pairs from same suffix group
        for suffix, words in suffix_groups.items():
            if len(words) >= 2:
                for i in range(min(len(words), 5)):  # Limit per group
                    for j in range(i + 1, min(len(words), 5)):
                        rhymes.append((words[i], words[j], 0.8))  # Rhymes are similar

        print(f"  ✓ Found {len(rhymes):,} rhyme pairs")

        # 3. Morphological relationships
        print("  Finding morphological pairs...")
        morph_pairs = []

        for pair in self.loader.morphology:
            if (pair.lemma in self.word_to_id and
                pair.inflected in self.word_to_id and
                pair.lemma_phonemes and
                pair.inflected_phonemes):

                morph_pairs.append((pair.lemma, pair.inflected, 0.7))  # Related

        print(f"  ✓ Found {len(morph_pairs):,} morphological pairs")

        # 4. Phonologically similar (edit distance 2-3)
        print("  Finding similar words...")
        similar = []

        for i, w1 in enumerate(random.sample(words_list, min(5000, len(words_list)))):
            if w1 not in self.loader.lexicon:
                continue
            p1 = self.loader.lexicon[w1]

            for w2 in random.sample(words_list, min(200, len(words_list))):
                if w1 == w2 or w2 not in self.loader.lexicon:
                    continue

                p2 = self.loader.lexicon[w2]
                dist = self._edit_distance(p1, p2)

                if 2 <= dist <= 3:
                    similar.append((w1, w2, 0.5))  # Moderately similar

        print(f"  ✓ Found {len(similar):,} similar pairs")

        # 5. Random negatives (dissimilar)
        print("  Generating negative examples...")
        negatives = []

        num_negatives = len(minimal_pairs) + len(rhymes) + len(morph_pairs)

        for _ in range(num_negatives):
            w1, w2 = random.sample(words_list, 2)

            if w1 in self.loader.lexicon and w2 in self.loader.lexicon:
                p1 = self.loader.lexicon[w1]
                p2 = self.loader.lexicon[w2]

                dist = self._edit_distance(p1, p2)

                if dist > 4:  # Very different
                    negatives.append((w1, w2, 0.0))  # Dissimilar

        print(f"  ✓ Generated {len(negatives):,} negative pairs")

        # Combine all
        self.examples = minimal_pairs + rhymes + morph_pairs + similar + negatives
        random.shuffle(self.examples)

        print(f"\n✓ Total training examples: {len(self.examples):,}")

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
    """Collate batch"""
    return {
        'word1_ids': torch.tensor([b['word1_id'] for b in batch], dtype=torch.long),
        'word2_ids': torch.tensor([b['word2_id'] for b in batch], dtype=torch.long),
        'similarities': torch.tensor([b['similarity'] for b in batch], dtype=torch.float32)
    }


class WordEmbeddingModel(nn.Module):
    """
    Word embedding model with contrastive learning
    """

    def __init__(self, num_words: int, embedding_dim: int = 64):
        super().__init__()

        self.embeddings = nn.Embedding(num_words, embedding_dim)

        # Initialize
        nn.init.xavier_uniform_(self.embeddings.weight)

    def forward(self, word_ids):
        return self.embeddings(word_ids)

    def similarity(self, word1_ids, word2_ids):
        """Compute cosine similarity"""
        emb1 = self.embeddings(word1_ids)
        emb2 = self.embeddings(word2_ids)

        # Normalize
        emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
        emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)

        # Cosine similarity
        return (emb1 * emb2).sum(dim=1)

    def loss(self, word1_ids, word2_ids, target_similarities):
        """MSE loss on similarity"""
        pred_sim = self.similarity(word1_ids, word2_ids)
        return nn.functional.mse_loss(pred_sim, target_similarities)


def train():
    """Train word embeddings"""

    print("=" * 70)
    print("WORD-LEVEL PHONOLOGICAL EMBEDDING TRAINING")
    print("=" * 70)

    # Load data
    loader = EnglishPhonologyLoader()
    dataset = WordPhonologyDataset(loader)

    dataloader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=True,
        collate_fn=collate_fn
    )

    # Create model
    model = WordEmbeddingModel(
        num_words=len(dataset.word_to_id),
        embedding_dim=64
    )

    print(f"\nModel: {len(dataset.word_to_id):,} words → 64-dim embeddings")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

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
            loss = model.loss(
                batch['word1_ids'],
                batch['word2_ids'],
                batch['similarities']
            )

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

        word_id = dataset.word_to_id[word]
        word_emb = model(torch.tensor([word_id])).detach()

        all_ids = torch.arange(len(dataset.word_to_id))
        all_embs = model(all_ids).detach()

        # Compute similarities
        word_emb_norm = torch.nn.functional.normalize(word_emb, p=2, dim=1)
        all_embs_norm = torch.nn.functional.normalize(all_embs, p=2, dim=1)

        sims = (all_embs_norm * word_emb_norm).sum(dim=1)

        # Top K
        top_indices = sims.argsort(descending=True)[1:top_k+1]  # Skip self

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
    train()
