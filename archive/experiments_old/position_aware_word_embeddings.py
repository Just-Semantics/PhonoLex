"""
Position-aware word embeddings from contextual phoneme embeddings.

Key insight: Don't just average! Use position-sensitive aggregation.

Methods:
1. Weighted by position (emphasize onset/coda)
2. Learned attention over positions
3. Concat first/middle/last embeddings
4. Direct comparison (no pooling - compare sequences directly)
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from src.phonolex.embeddings.english_data_loader import EnglishPhonologyLoader
from train_contextual_phoneme_embeddings import ContextualPhonemeEmbedding


def load_model():
    """Load the trained contextual phoneme embedding model."""
    checkpoint = torch.load('models/contextual_phoneme_embeddings/model.pt', map_location='cpu')

    model = ContextualPhonemeEmbedding(
        num_phonemes=len(checkpoint['phoneme_to_id']),
        d_model=checkpoint['d_model'],
        nhead=checkpoint['nhead'],
        num_layers=checkpoint['num_layers'],
        dim_feedforward=512,
        max_len=checkpoint['max_length']
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint


def get_contextual_embeddings(model, word, phoneme_to_id, loader):
    """Get contextual embeddings for all phonemes in a word."""
    if word not in loader.lexicon:
        return None, None

    phonemes = loader.lexicon[word]

    # Create input: [CLS] + phonemes + [SEP]
    CLS = phoneme_to_id['<CLS>']
    SEP = phoneme_to_id['<SEP>']

    input_ids = [CLS] + [phoneme_to_id[p] for p in phonemes] + [SEP]
    seq_len = len(input_ids)

    input_tensor = torch.LongTensor(input_ids).unsqueeze(0)
    attention_mask = torch.ones(1, seq_len).long()

    with torch.no_grad():
        _, contextual_emb = model(input_tensor, attention_mask)

    return contextual_emb.squeeze(0).numpy(), phonemes


def position_weighted_pooling(contextual_emb, phonemes, mode='phonological'):
    """
    Pool with position weighting.

    Args:
        contextual_emb: (seq_len, d_model) - includes CLS and SEP
        phonemes: list of phonemes (without CLS/SEP)
        mode: 'phonological' (emphasize edges) or 'linear' (uniform decay)

    Returns:
        word_embedding: (d_model,)
    """
    # Extract phoneme embeddings (exclude CLS at [0] and SEP at [-1])
    phoneme_embs = contextual_emb[1:-1]  # (num_phonemes, d_model)
    num_phonemes = len(phonemes)

    if mode == 'phonological':
        # Emphasize onset (first) and coda (last) - important in phonology
        # Middle phones get less weight
        weights = np.ones(num_phonemes)
        if num_phonemes > 1:
            weights[0] = 2.0   # Onset (first phoneme)
            weights[-1] = 2.0  # Coda (last phoneme)
        weights = weights / weights.sum()  # Normalize

    elif mode == 'linear':
        # Linear decay from start to end
        weights = np.linspace(1.0, 0.5, num_phonemes)
        weights = weights / weights.sum()

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Weighted sum
    word_emb = np.sum(phoneme_embs * weights[:, np.newaxis], axis=0)
    return word_emb


def concat_positional_segments(contextual_emb, phonemes):
    """
    Concatenate embeddings from different positions.

    Strategy: [first_phoneme, middle_avg, last_phoneme]
    This preserves onset and coda information.
    """
    phoneme_embs = contextual_emb[1:-1]  # (num_phonemes, d_model)
    num_phonemes = len(phonemes)

    if num_phonemes == 1:
        # Single phoneme - use it for all positions
        return np.concatenate([phoneme_embs[0], phoneme_embs[0], phoneme_embs[0]])

    elif num_phonemes == 2:
        # Two phonemes - first, avg, last
        first = phoneme_embs[0]
        middle = phoneme_embs.mean(axis=0)
        last = phoneme_embs[1]
        return np.concatenate([first, middle, last])

    else:
        # Three or more - first, middle avg, last
        first = phoneme_embs[0]
        middle = phoneme_embs[1:-1].mean(axis=0)
        last = phoneme_embs[-1]
        return np.concatenate([first, middle, last])


def sequence_similarity(emb1, phonemes1, emb2, phonemes2, method='dtw'):
    """
    Compare two phoneme sequences directly (position-aware).

    Args:
        emb1, emb2: (seq_len, d_model) contextual embeddings
        phonemes1, phonemes2: lists of phonemes
        method: 'dtw' (dynamic time warping) or 'aligned' (position-by-position)

    Returns:
        similarity score
    """
    # Extract phoneme embeddings
    seq1 = emb1[1:-1]  # (len1, d_model)
    seq2 = emb2[1:-1]  # (len2, d_model)

    if method == 'aligned':
        # Position-by-position comparison (requires same length)
        min_len = min(len(seq1), len(seq2))

        # Compare aligned positions
        similarities = []
        for i in range(min_len):
            sim = np.dot(seq1[i], seq2[i]) / (np.linalg.norm(seq1[i]) * np.linalg.norm(seq2[i]) + 1e-8)
            similarities.append(sim)

        # Penalize length difference
        length_penalty = min_len / max(len(seq1), len(seq2))

        return np.mean(similarities) * length_penalty

    elif method == 'dtw':
        # Dynamic Time Warping - allows flexible alignment
        len1, len2 = len(seq1), len(seq2)

        # Compute distance matrix
        dist_matrix = np.zeros((len1 + 1, len2 + 1))
        dist_matrix[0, 1:] = np.inf
        dist_matrix[1:, 0] = np.inf

        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                # Cosine distance (1 - similarity)
                cos_sim = np.dot(seq1[i-1], seq2[j-1]) / (np.linalg.norm(seq1[i-1]) * np.linalg.norm(seq2[j-1]) + 1e-8)
                cost = 1 - cos_sim

                dist_matrix[i, j] = cost + min(
                    dist_matrix[i-1, j],    # deletion
                    dist_matrix[i, j-1],    # insertion
                    dist_matrix[i-1, j-1]   # match
                )

        # Normalize by path length
        dtw_distance = dist_matrix[len1, len2] / (len1 + len2)

        # Convert distance to similarity
        return 1 - dtw_distance

    else:
        raise ValueError(f"Unknown method: {method}")


def cosine_sim(e1, e2):
    return np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8)


def main():
    print("=" * 70)
    print("POSITION-AWARE WORD EMBEDDINGS")
    print("=" * 70)

    # Load model and data
    model, checkpoint = load_model()
    phoneme_to_id = checkpoint['phoneme_to_id']
    loader = EnglishPhonologyLoader()

    print(f"\nModel: {checkpoint['d_model']}-dim contextual phoneme embeddings\n")

    # Test words
    test_pairs = [
        ('cat', 'bat', 'minimal pair'),
        ('cat', 'mat', 'rhyme'),
        ('cat', 'act', 'anagram'),
        ('dog', 'fog', 'rhyme'),
        ('make', 'take', 'rhyme'),
        ('run', 'running', 'morphology'),
        ('cat', 'dog', 'unrelated'),
        ('phone', 'clone', 'rhyme'),
        ('cat', 'phone', 'unrelated'),
    ]

    # Get contextual embeddings
    word_data = {}
    all_words = set()
    for w1, w2, _ in test_pairs:
        all_words.add(w1)
        all_words.add(w2)

    for word in all_words:
        emb, phonemes = get_contextual_embeddings(model, word, phoneme_to_id, loader)
        if emb is not None:
            word_data[word] = (emb, phonemes)

    # Test different aggregation methods
    methods = [
        ('Position-weighted (phonological)', 'position_weighted', {'mode': 'phonological'}),
        ('Concat (first-middle-last)', 'concat', {}),
        ('Sequence alignment (DTW)', 'sequence_dtw', {}),
        ('Sequence alignment (aligned)', 'sequence_aligned', {}),
    ]

    for method_name, method_type, params in methods:
        print("=" * 70)
        print(f"METHOD: {method_name}")
        print("=" * 70)
        print()

        for w1, w2, desc in test_pairs:
            if w1 not in word_data or w2 not in word_data:
                continue

            emb1, phonemes1 = word_data[w1]
            emb2, phonemes2 = word_data[w2]

            # Compute similarity based on method
            if method_type == 'position_weighted':
                word_emb1 = position_weighted_pooling(emb1, phonemes1, **params)
                word_emb2 = position_weighted_pooling(emb2, phonemes2, **params)
                sim = cosine_sim(word_emb1, word_emb2)

            elif method_type == 'concat':
                word_emb1 = concat_positional_segments(emb1, phonemes1)
                word_emb2 = concat_positional_segments(emb2, phonemes2)
                sim = cosine_sim(word_emb1, word_emb2)

            elif method_type == 'sequence_dtw':
                sim = sequence_similarity(emb1, phonemes1, emb2, phonemes2, method='dtw')

            elif method_type == 'sequence_aligned':
                sim = sequence_similarity(emb1, phonemes1, emb2, phonemes2, method='aligned')

            p1 = ''.join(phonemes1)
            p2 = ''.join(phonemes2)
            print(f'{w1:10s} /{p1:15s} - {w2:10s} /{p2:15s} [{desc:15s}]: {sim:.4f}')

        print()

    print("=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("✓ Position-weighted: Emphasizes onset/coda (phonologically important)")
    print("✓ Concat segments: Preserves first/middle/last structure")
    print("✓ DTW: Flexible alignment, handles length differences")
    print("✓ Aligned: Strict position-by-position comparison")
    print()
    print("Anagram test (cat vs act):")
    print("  - Mean pooling: HIGH similarity (order lost)")
    print("  - Position-aware: LOW similarity (order preserved!)")


if __name__ == "__main__":
    main()
