#!/usr/bin/env python3
"""
MLM-Only Model: Let similarity emerge naturally from sequential structure

No contrastive learning. Just:
1. Phoible initialization (phonological prior)
2. Positional encoding (preserves order)
3. Transformer (learns context)
4. MLM training (predicts masked phonemes)

Hypothesis: Position + transformer should naturally discriminate anagrams.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import sys
import numpy as np
import random

sys.path.insert(0, str(Path.cwd()))
from src.phonolex.embeddings.english_data_loader import EnglishPhonologyLoader
from train_phonology_bert import PhonoLexBERT
from data.mappings.phoneme_vectorizer import PhonemeVectorizer, load_phoible_csv


def load_phoible_init(phoneme_to_id, d_model=128):
    """Initialize with Phoible features"""
    phoible_csv = Path('data/phoible/english/phoible-english.csv')
    phoible_data = load_phoible_csv(str(phoible_csv))
    vectorizer = PhonemeVectorizer(encoding_scheme='three_way')

    phoible_features = {}
    for phoneme_data in phoible_data:
        vec = vectorizer.vectorize(phoneme_data)
        if vec.phoneme not in phoible_features:
            phoible_features[vec.phoneme] = vec.endpoints_76d

    feature_projection = nn.Linear(76, d_model)
    embedding_matrix = torch.randn(len(phoneme_to_id), d_model) * 0.01

    initialized = 0
    with torch.no_grad():
        for phoneme, idx in phoneme_to_id.items():
            if phoneme in phoible_features:
                features = torch.FloatTensor(phoible_features[phoneme])
                embedding_matrix[idx] = feature_projection(features)
                initialized += 1

    print(f"✓ Initialized {initialized}/{len(phoneme_to_id)} phonemes with Phoible\n")
    return embedding_matrix


class MLMOnlyDataset(Dataset):
    """Simple dataset - just MLM, no contrastive pairs"""

    def __init__(self, loader, phoneme_to_id, mask_prob=0.15):
        self.loader = loader
        self.phoneme_to_id = phoneme_to_id
        self.mask_prob = mask_prob

        self.PAD = '<PAD>'
        self.CLS = '<CLS>'
        self.SEP = '<SEP>'
        self.MASK = '<MASK>'

        # Build word data
        self.words = list(loader.lexicon.keys())
        self.word_data = []

        for word in self.words:
            phonemes = loader.lexicon[word]
            phoneme_ids = [self.phoneme_to_id['<CLS>']] + \
                         [self.phoneme_to_id.get(p, self.phoneme_to_id['<PAD>']) for p in phonemes] + \
                         [self.phoneme_to_id['<SEP>']]

            self.word_data.append({
                'word': word,
                'phoneme_ids': phoneme_ids
            })

        self.max_length = max(len(d['phoneme_ids']) for d in self.word_data) + 2

        print(f"Dataset: {len(self.words):,} words, max_length={self.max_length}\n")

    def __len__(self):
        return len(self.word_data)

    def __getitem__(self, idx):
        data = self.word_data[idx]
        phoneme_ids = data['phoneme_ids']

        # Apply masking
        input_ids = phoneme_ids.copy()
        target_ids = [0] * len(phoneme_ids)

        for i in range(1, len(phoneme_ids) - 1):
            if random.random() < self.mask_prob:
                target_ids[i] = input_ids[i]
                input_ids[i] = self.phoneme_to_id[self.MASK]

        # Pad
        seq_len = len(input_ids)
        padded_input = input_ids + [self.phoneme_to_id[self.PAD]] * (self.max_length - seq_len)
        padded_target = target_ids + [0] * (self.max_length - seq_len)
        attention_mask = [1] * seq_len + [0] * (self.max_length - seq_len)

        return {
            'input_ids': torch.LongTensor(padded_input),
            'target_ids': torch.LongTensor(padded_target),
            'attention_mask': torch.LongTensor(attention_mask),
            'word_idx': idx
        }


def train_mlm_only(model, dataset, device, num_epochs=20):
    """Train with ONLY MLM - no contrastive loss"""
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    print("=" * 70)
    print("TRAINING (MLM ONLY)")
    print("=" * 70)
    print()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            mlm_logits, _, _ = model(input_ids, attention_mask)

            loss = criterion(mlm_logits.view(-1, mlm_logits.size(-1)), target_ids.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Accuracy
            preds = mlm_logits.argmax(dim=-1)
            mask = target_ids != 0
            correct += ((preds == target_ids) & mask).sum().item()
            total += mask.sum().item()

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total if total > 0 else 0

        print(f"  Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")

    print()
    return model


def main():
    print("\n" + "=" * 70)
    print("MLM-ONLY MODEL")
    print("=" * 70)
    print("No contrastive learning - similarity emerges from sequential structure\n")

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Load data
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        loader = EnglishPhonologyLoader()

    print(f"Loaded {len(loader.lexicon):,} words\n")

    # Build vocab
    phonemes = ['<PAD>', '<CLS>', '<SEP>', '<MASK>'] + sorted(loader.english_phonemes)
    phoneme_to_id = {p: i for i, p in enumerate(phonemes)}

    # Create dataset
    dataset = MLMOnlyDataset(loader, phoneme_to_id, mask_prob=0.15)

    # Initialize with Phoible
    print("Initializing with Phoible features...")
    phoible_init = load_phoible_init(phoneme_to_id, d_model=128)

    # Create model
    model = PhonoLexBERT(
        num_phonemes=len(phoneme_to_id),
        d_model=128, nhead=4, num_layers=3, dim_feedforward=512,
        max_len=dataset.max_length
    ).to(device)

    # Initialize with Phoible
    with torch.no_grad():
        model.phoneme_embeddings.weight.data = phoible_init.to(device)

    print("✓ Model initialized with Phoible features\n")

    # Train with MLM only (longer - 20 epochs since no contrastive)
    model = train_mlm_only(model, dataset, device, num_epochs=20)

    # Save
    Path('models/mlm_only').mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'phoneme_to_id': phoneme_to_id,
        'max_length': dataset.max_length
    }, 'models/mlm_only/final.pt')

    print("✓ Model saved: models/mlm_only/final.pt\n")

    # Evaluate
    print("=" * 70)
    print("EVALUATION")
    print("=" * 70)
    print()

    model.eval()

    def get_emb(word):
        if word not in loader.lexicon:
            return None
        phonemes = loader.lexicon[word]
        CLS, SEP, PAD = phoneme_to_id['<CLS>'], phoneme_to_id['<SEP>'], phoneme_to_id['<PAD>']
        ids = [CLS] + [phoneme_to_id.get(p, PAD) for p in phonemes] + [SEP]
        seq_len = len(ids)
        padded = ids + [PAD] * (dataset.max_length - seq_len)
        mask = [1] * seq_len + [0] * (dataset.max_length - seq_len)

        with torch.no_grad():
            _, word_emb, _ = model(torch.LongTensor([padded]).to(device), torch.LongTensor([mask]).to(device))
        return word_emb.squeeze(0).cpu().numpy()

    def sim(w1, w2):
        e1, e2 = get_emb(w1), get_emb(w2)
        if e1 is None or e2 is None:
            return None
        return np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8)

    tests = [
        ('cat', 'bat', 'rhyme'),
        ('cat', 'mat', 'rhyme'),
        ('cat', 'act', 'ANAGRAM'),
        ('dog', 'fog', 'rhyme'),
        ('make', 'take', 'rhyme'),
        ('run', 'running', 'morph'),
        ('cat', 'dog', 'unrelated'),
        ('cat', 'phone', 'unrelated'),
    ]

    for w1, w2, typ in tests:
        s = sim(w1, w2)
        if s is not None:
            print(f"{w1}-{w2:<14} {s:.3f}  ({typ})")

    print()
    print("=" * 70)
    print("HYPOTHESIS TEST:")
    print("=" * 70)
    print("If position + transformer work properly:")
    print("  - cat/act should be LOW (different positions)")
    print("  - cat/bat should be HIGH (same positions, similar phonemes)")
    print()


if __name__ == '__main__':
    main()
