#!/usr/bin/env python3
"""
Train English-only phoneme embeddings

Focused dataset: 39 English phonemes, 125K words
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

from src.phonolex.embeddings.english_data_loader import EnglishPhonologyLoader


class EnglishPhonemeEmbedding(nn.Module):
    """
    Simple phoneme embedding model for English

    Just learns embeddings from context (skip-gram)
    """

    def __init__(self, num_phonemes: int, embedding_dim: int = 32):
        super().__init__()

        self.embeddings = nn.Embedding(num_phonemes, embedding_dim)
        self.context_proj = nn.Linear(embedding_dim, num_phonemes)

        # Xavier init
        nn.init.xavier_uniform_(self.embeddings.weight)

    def forward(self, phoneme_ids):
        """Get embeddings"""
        return self.embeddings(phoneme_ids)

    def predict_context(self, center_embedding):
        """Predict context phonemes"""
        return self.context_proj(center_embedding)


class EnglishPhonemeDataset(Dataset):
    """Dataset for English phoneme skip-gram training"""

    def __init__(self, loader: EnglishPhonologyLoader, window_size: int = 2):
        self.loader = loader
        self.window_size = window_size

        # Build phoneme vocabulary
        self.phoneme_to_id = {p: i for i, p in enumerate(sorted(loader.english_phonemes))}
        self.id_to_phoneme = {i: p for p, i in self.phoneme_to_id.items()}

        print(f"\nPhoneme vocabulary: {len(self.phoneme_to_id)} phonemes")

        # Generate skip-gram examples
        print("Generating context examples...")
        self.examples = []

        for word, phonemes in loader.lexicon.items():
            for i, center in enumerate(phonemes):
                # Get context window
                start = max(0, i - window_size)
                end = min(len(phonemes), i + window_size + 1)

                context = []
                for j in range(start, end):
                    if j != i:
                        context.append(phonemes[j])

                if context and center in self.phoneme_to_id:
                    # Convert to IDs
                    center_id = self.phoneme_to_id[center]
                    context_ids = [self.phoneme_to_id[c] for c in context if c in self.phoneme_to_id]

                    if context_ids:
                        self.examples.append((center_id, context_ids))

        print(f"  ✓ Generated {len(self.examples):,} context examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch):
    """Collate batch"""
    centers = []
    contexts = []

    for center, context in batch:
        centers.append(center)
        # Pad/truncate context to fixed length
        if len(context) > 4:
            context = context[:4]
        else:
            context = context + [context[0]] * (4 - len(context))
        contexts.append(context)

    return {
        'center_ids': torch.tensor(centers, dtype=torch.long),
        'context_ids': torch.tensor(contexts, dtype=torch.long)
    }


def train():
    """Train English phoneme embeddings"""

    print("=" * 70)
    print("ENGLISH PHONEME EMBEDDING TRAINING")
    print("=" * 70)

    # Load data
    loader = EnglishPhonologyLoader()
    dataset = EnglishPhonemeDataset(loader, window_size=2)

    dataloader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        collate_fn=collate_fn
    )

    # Create model
    num_phonemes = len(dataset.phoneme_to_id)
    model = EnglishPhonemeEmbedding(num_phonemes=num_phonemes, embedding_dim=32)

    print(f"\nModel: {num_phonemes} phonemes → 32-dim embeddings")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    print(f"\nTraining for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            center_ids = batch['center_ids']
            context_ids = batch['context_ids']

            # Forward
            center_emb = model(center_ids)
            logits = model.predict_context(center_emb)

            # Loss: predict each context position
            loss = 0
            for i in range(context_ids.size(1)):
                loss += nn.functional.cross_entropy(logits, context_ids[:, i])
            loss = loss / context_ids.size(1)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(dataloader)
        print(f"  Epoch {epoch+1} avg loss: {avg_loss:.4f}")

    # Save model
    output_dir = Path("models/english_phoneme_embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'phoneme_to_id': dataset.phoneme_to_id,
        'id_to_phoneme': dataset.id_to_phoneme,
        'embedding_dim': 32
    }, output_dir / "model.pt")

    print(f"\n✓ Model saved to {output_dir / 'model.pt'}")

    # Test embeddings
    print("\n" + "=" * 70)
    print("TESTING LEARNED EMBEDDINGS")
    print("=" * 70)

    model.eval()
    with torch.no_grad():
        all_embeddings = model(torch.arange(num_phonemes)).numpy()

    # Find similar phonemes
    def cosine_sim(e1, e2):
        return np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8)

    test_phonemes = ['p', 'b', 't', 's', 'z', 'æ', 'i']

    for phone in test_phonemes:
        if phone not in dataset.phoneme_to_id:
            continue

        idx = dataset.phoneme_to_id[phone]
        emb = all_embeddings[idx]

        # Find similar
        sims = []
        for other_phone, other_idx in dataset.phoneme_to_id.items():
            if other_phone == phone:
                continue
            sim = cosine_sim(emb, all_embeddings[other_idx])
            sims.append((other_phone, sim))

        sims.sort(key=lambda x: x[1], reverse=True)

        print(f"\n/{phone}/ most similar:")
        for p, s in sims[:5]:
            print(f"  /{p}/  {s:.4f}")


if __name__ == '__main__':
    train()
