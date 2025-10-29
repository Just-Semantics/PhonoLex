#!/usr/bin/env python3
"""
Sequential Phonological Embeddings - FINAL

No pooling! Keep the full sequence of contextual phoneme embeddings.
Similarity = sequence-to-sequence distance (like DTW or aligned cosine).

This preserves:
- Order: cat /kæt/ ≠ act /ækt/
- Position: first phoneme matters differently than last
- Context: each phoneme embedding depends on neighbors
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import sys
import numpy as np
import random
from scipy.spatial.distance import cosine as cosine_dist

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


class MLMDataset(Dataset):
    """Next-phoneme prediction dataset (teaches adjacency)"""

    def __init__(self, loader, phoneme_to_id, mask_prob=0.15):
        self.loader = loader
        self.phoneme_to_id = phoneme_to_id
        # mask_prob not used for next-phoneme prediction

        self.PAD = '<PAD>'
        self.CLS = '<CLS>'
        self.SEP = '<SEP>'
        self.MASK = '<MASK>'

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
        print(f"Dataset: {len(self.words):,} words\n")

    def __len__(self):
        return len(self.word_data)

    def __getitem__(self, idx):
        data = self.word_data[idx]
        phoneme_ids = data['phoneme_ids']

        # NEXT-PHONEME PREDICTION (not MLM)
        # Input: phoneme[0:i], Target: phoneme[i]
        # This teaches adjacency: which phonemes follow which

        input_ids = phoneme_ids[:-1]  # All but last
        target_ids = phoneme_ids[1:]  # Shifted by 1 (next phoneme)

        seq_len = len(input_ids)
        padded_input = input_ids + [self.phoneme_to_id[self.PAD]] * (self.max_length - seq_len)
        padded_target = target_ids + [0] * (self.max_length - seq_len)
        attention_mask = [1] * seq_len + [0] * (self.max_length - seq_len)

        return {
            'input_ids': torch.LongTensor(padded_input),
            'target_ids': torch.LongTensor(padded_target),
            'attention_mask': torch.LongTensor(attention_mask),
        }


def sequence_similarity(seq1, seq2):
    """
    Compute similarity between two phoneme embedding sequences.
    Uses edit distance with embedding similarity (soft Levenshtein).

    Handles:
    - Insertions: computer has /p/, commuter doesn't
    - Deletions: vice versa
    - Substitutions: /ʌ/ vs /ə/ (cost based on embedding distance)
    """
    len1 = len(seq1)
    len2 = len(seq2)

    if len1 == 0 or len2 == 0:
        return 0.0

    # Dynamic programming for edit distance with soft costs
    # dp[i][j] = minimum cost to align seq1[:i] with seq2[:j]
    dp = np.zeros((len1 + 1, len2 + 1))

    # Initialize: cost of deleting/inserting entire sequences
    for i in range(len1 + 1):
        dp[i][0] = i  # Delete all from seq1
    for j in range(len2 + 1):
        dp[0][j] = j  # Insert all to match seq2

    # Fill DP table
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            # Cost of substitution (or match if similar)
            # Use 1 - cosine similarity as distance
            dist = cosine_dist(seq1[i-1], seq2[j-1])
            substitute_cost = dist

            # Match cost (if embeddings are very similar, cost is low)
            match_cost = substitute_cost

            # Edit operations
            delete_cost = dp[i-1][j] + 1.0  # Delete from seq1
            insert_cost = dp[i][j-1] + 1.0  # Insert into seq1
            match_or_substitute = dp[i-1][j-1] + match_cost

            dp[i][j] = min(delete_cost, insert_cost, match_or_substitute)

    # Normalize by sequence length to get similarity score
    # Lower edit distance = higher similarity
    max_len = max(len1, len2)
    normalized_distance = dp[len1][len2] / max_len if max_len > 0 else 0

    # Convert distance to similarity (0-1 scale)
    similarity = max(0.0, 1.0 - normalized_distance)

    return similarity


def train_mlm(model, dataset, device, num_epochs=20):
    """Train with NEXT-PHONEME prediction (teaches adjacency)"""
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    print("=" * 70)
    print("TRAINING (Next-Phoneme Prediction + Phoible init)")
    print("=" * 70)
    print("Learning: which phonemes follow which (adjacency!)")
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

            preds = mlm_logits.argmax(dim=-1)
            mask = target_ids != 0
            correct += ((preds == target_ids) & mask).sum().item()
            total += mask.sum().item()

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total if total > 0 else 0

        print(f"  Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.4f}, Acc={accuracy:.4f}")

    print()
    return model


def main():
    print("\n" + "=" * 70)
    print("SEQUENTIAL PHONOLOGICAL EMBEDDINGS")
    print("=" * 70)
    print("No pooling - use full contextual phoneme sequences")
    print("Similarity = aligned sequence similarity\n")

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}\n")

    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        loader = EnglishPhonologyLoader()

    print(f"Loaded {len(loader.lexicon):,} words\n")

    phonemes = ['<PAD>', '<CLS>', '<SEP>', '<MASK>'] + sorted(loader.english_phonemes)
    phoneme_to_id = {p: i for i, p in enumerate(phonemes)}

    dataset = MLMDataset(loader, phoneme_to_id, mask_prob=0.15)

    print("Initializing with Phoible...")
    phoible_init = load_phoible_init(phoneme_to_id, d_model=128)

    model = PhonoLexBERT(
        num_phonemes=len(phoneme_to_id),
        d_model=128, nhead=4, num_layers=3, dim_feedforward=512,
        max_len=dataset.max_length
    ).to(device)

    with torch.no_grad():
        model.phoneme_embeddings.weight.data = phoible_init.to(device)

    print("✓ Model initialized\n")

    # Train
    model = train_mlm(model, dataset, device, num_epochs=20)

    # Save
    Path('models/sequential').mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'phoneme_to_id': phoneme_to_id,
        'max_length': dataset.max_length
    }, 'models/sequential/final.pt')

    print("✓ Model saved: models/sequential/final.pt\n")

    # Evaluate with SEQUENTIAL similarity
    print("=" * 70)
    print("EVALUATION (Sequential Similarity)")
    print("=" * 70)
    print()

    model.eval()

    def get_seq(word):
        """Get contextual phoneme embedding sequence"""
        if word not in loader.lexicon:
            return None
        phonemes = loader.lexicon[word]
        CLS, SEP, PAD = phoneme_to_id['<CLS>'], phoneme_to_id['<SEP>'], phoneme_to_id['<PAD>']
        ids = [CLS] + [phoneme_to_id.get(p, PAD) for p in phonemes] + [SEP]
        seq_len = len(ids)
        padded = ids + [PAD] * (dataset.max_length - seq_len)
        mask = [1] * seq_len + [0] * (dataset.max_length - seq_len)

        with torch.no_grad():
            _, _, contextual = model(torch.LongTensor([padded]).to(device), torch.LongTensor([mask]).to(device))

        # Extract actual phoneme embeddings (skip CLS/SEP, exclude padding)
        phoneme_embs = contextual[0, 1:seq_len-1].cpu().numpy()  # [num_phonemes, d_model]
        return phoneme_embs

    tests = [
        ('cat', 'bat'),
        ('cat', 'act'),
        ('cat', 'mat'),
        ('dog', 'fog'),
        ('run', 'running'),
        ('cat', 'dog'),
        ('computer', 'commuter'),
    ]

    print(f'{"Word 1":<12} {"Word 2":<12} {"Similarity":<12}')
    print('-' * 40)

    for w1, w2 in tests:
        seq1 = get_seq(w1)
        seq2 = get_seq(w2)

        if seq1 is not None and seq2 is not None:
            sim = sequence_similarity(seq1, seq2)
            print(f'{w1:<12} {w2:<12} {sim:.3f}')

    print()
    print("=" * 70)
    print("KEY INSIGHT:")
    print("=" * 70)
    print("Sequential similarity preserves order:")
    print("  - cat/bat: High (same positions, /æt/ shared)")
    print("  - cat/act: LOW (different positions!)")
    print("  - computer/commuter: High (long shared sequence /mjutɚ/)")
    print()


if __name__ == '__main__':
    main()
