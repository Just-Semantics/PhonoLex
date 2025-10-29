#!/usr/bin/env python3
"""
Test learned phoneme embeddings with concrete examples
"""

import torch
import numpy as np
from src.phonolex.embeddings.data_loader import PhonemeEmbeddingDataLoader
from src.phonolex.embeddings.model import create_model

# Load data
loader = PhonemeEmbeddingDataLoader()

# Load trained model
checkpoint = torch.load('models/phoneme_embeddings/best_model.pt', map_location='cpu')
model = create_model(
    num_phonemes=checkpoint['config']['num_phonemes'],
    embedding_dim=checkpoint['config']['embedding_dim'],
    num_allomorphs=10
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Get all embeddings
with torch.no_grad():
    all_ids = torch.arange(len(loader.phoneme_to_id))
    all_embeddings = model.get_embeddings(all_ids)

def cosine_similarity(emb1, emb2):
    """Compute cosine similarity"""
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)

def get_embedding(phoneme):
    """Get embedding for a phoneme"""
    if phoneme not in loader.phoneme_to_id:
        return None
    idx = loader.phoneme_to_id[phoneme]
    return all_embeddings[idx]

def find_similar(phoneme, top_k=10):
    """Find most similar phonemes"""
    emb = get_embedding(phoneme)
    if emb is None:
        return []

    similarities = []
    for p, idx in loader.phoneme_to_id.items():
        if p == phoneme:
            continue
        sim = cosine_similarity(emb, all_embeddings[idx])
        similarities.append((p, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

# Test with common phonemes
print("=" * 70)
print("PHONEME EMBEDDING SIMILARITY TEST")
print("=" * 70)

test_phonemes = [
    ('p', 'voiceless bilabial stop'),
    ('b', 'voiced bilabial stop'),
    ('t', 'voiceless alveolar stop'),
    ('d', 'voiced alveolar stop'),
    ('k', 'voiceless velar stop'),
    ('s', 'voiceless alveolar fricative'),
    ('z', 'voiced alveolar fricative'),
    ('a', 'low vowel'),
    ('i', 'high front vowel'),
    ('u', 'high back vowel')
]

for phoneme, description in test_phonemes:
    print(f"\n/{phoneme}/ ({description})")
    print("-" * 70)

    similar = find_similar(phoneme, top_k=10)
    if similar:
        print("Most similar phonemes:")
        for i, (p, sim) in enumerate(similar[:10], 1):
            print(f"  {i:2d}. /{p:5s}  similarity: {sim:.4f}")
    else:
        print(f"  (/{phoneme}/ not in vocabulary)")

# Test specific pairs
print("\n" + "=" * 70)
print("SPECIFIC PHONEME PAIRS")
print("=" * 70)

pairs = [
    ('p', 'b', 'voiceless vs voiced bilabial stops'),
    ('p', 't', 'voiceless bilabial vs alveolar stops'),
    ('p', 'k', 'voiceless labial vs velar stops'),
    ('s', 'z', 'voiceless vs voiced alveolar fricatives'),
    ('s', 'ʃ', 'alveolar vs postalveolar fricatives'),
    ('a', 'æ', 'low vowels'),
    ('i', 'ɪ', 'high front vowels (tense vs lax)'),
    ('a', 'i', 'low vs high vowels (very different)'),
    ('p', 'a', 'stop vs vowel (very different)'),
]

for p1, p2, description in pairs:
    emb1 = get_embedding(p1)
    emb2 = get_embedding(p2)

    if emb1 is not None and emb2 is not None:
        sim = cosine_similarity(emb1, emb2)
        print(f"\n/{p1}/ vs /{p2}/ ({description})")
        print(f"  Similarity: {sim:.4f}")
    else:
        print(f"\n/{p1}/ vs /{p2}/ - not in vocabulary")

# Show feature similarity vs embedding similarity
print("\n" + "=" * 70)
print("FEATURE SIMILARITY VS EMBEDDING SIMILARITY")
print("=" * 70)

def feature_similarity(p1, p2):
    """Compute feature-based similarity"""
    f1 = loader.get_phoneme_features(p1)
    f2 = loader.get_phoneme_features(p2)
    if f1 is None or f2 is None:
        return None
    return np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-8)

print(f"\n{'Pair':<15} {'Feature Sim':<15} {'Embedding Sim':<15} {'Match?'}")
print("-" * 70)

for p1, p2, desc in pairs:
    feat_sim = feature_similarity(p1, p2)
    emb1 = get_embedding(p1)
    emb2 = get_embedding(p2)

    if feat_sim is not None and emb1 is not None and emb2 is not None:
        emb_sim = cosine_similarity(emb1, emb2)
        match = "✓" if abs(feat_sim - emb_sim) < 0.3 else "✗"
        print(f"/{p1}/-/{p2}/<10s {feat_sim:>14.4f} {emb_sim:>15.4f}      {match}")
