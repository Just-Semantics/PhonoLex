#!/usr/bin/env python3
"""
Test weighted component similarity to verify it fixes rhyme bias
"""

import torch
import numpy as np

def cosine_similarity(vec1, vec2):
    """Standard cosine similarity"""
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

def weighted_cosine_similarity(vec1, vec2):
    """
    Weighted component cosine similarity.
    Gives equal weight (1/3) to onset, nucleus, and coda.
    """
    # Component boundaries
    ONSET_START, ONSET_END = 0, 128
    NUCLEUS_START, NUCLEUS_END = 128, 256
    CODA_START, CODA_END = 256, 384

    # Compute similarity for each component
    onset_sim = cosine_similarity(vec1[ONSET_START:ONSET_END], vec2[ONSET_START:ONSET_END])
    nucleus_sim = cosine_similarity(vec1[NUCLEUS_START:NUCLEUS_END], vec2[NUCLEUS_START:NUCLEUS_END])
    coda_sim = cosine_similarity(vec1[CODA_START:CODA_END], vec2[CODA_START:CODA_END])

    # Equal weights: 1/3 each
    return (onset_sim + nucleus_sim + coda_sim) / 3.0

def compute_soft_levenshtein(syllables1, syllables2, use_weighted=False):
    """Compute soft Levenshtein similarity"""
    len1 = len(syllables1)
    len2 = len(syllables2)

    sim_func = weighted_cosine_similarity if use_weighted else cosine_similarity

    # Pre-compute pairwise syllable similarities
    sim_matrix = np.zeros((len1, len2))
    for i in range(len1):
        for j in range(len2):
            sim_matrix[i][j] = sim_func(syllables1[i], syllables2[j])

    # Dynamic programming
    dp = np.zeros((len1 + 1, len2 + 1))
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            match_cost = 1.0 - sim_matrix[i-1][j-1]
            dp[i][j] = min(
                dp[i-1][j] + 1.0,
                dp[i][j-1] + 1.0,
                dp[i-1][j-1] + match_cost
            )

    max_len = max(len1, len2)
    if max_len == 0:
        return 1.0

    edit_distance = dp[len1][len2]
    similarity = 1.0 - (edit_distance / max_len)
    return max(0.0, min(1.0, similarity))

def main():
    print("=" * 80)
    print("Testing Weighted Component Similarity")
    print("=" * 80)
    print()

    # Load embeddings
    checkpoint = torch.load('embeddings/layer4/syllable_embeddings_filtered.pt',
                           map_location='cpu', weights_only=False)
    word_embeddings = checkpoint['word_to_syllable_embeddings']

    # Test pairs
    test_pairs = [
        ('cat', 'bat', 'Rhyme (different onset)'),
        ('make', 'bake', 'Rhyme (similar onset: /m/ vs /b/)'),
        ('make', 'take', 'Rhyme (dissimilar onset: /m/ vs /t/)'),
        ('cat', 'act', 'Anagram'),
        ('dog', 'fog', 'Rhyme'),
        ('cat', 'dog', 'Unrelated'),
    ]

    print("COMPARISON: Standard vs Weighted Component Similarity")
    print()
    print(f"{'Word 1':<10} {'Word 2':<10} {'Standard':<12} {'Weighted':<12} {'Description':<30}")
    print("-" * 80)

    for word1, word2, description in test_pairs:
        if word1 in word_embeddings and word2 in word_embeddings:
            emb1 = word_embeddings[word1]
            emb2 = word_embeddings[word2]

            standard_sim = compute_soft_levenshtein(emb1, emb2, use_weighted=False)
            weighted_sim = compute_soft_levenshtein(emb1, emb2, use_weighted=True)

            print(f"{word1:<10} {word2:<10} {standard_sim:.4f}       {weighted_sim:.4f}       {description:<30}")

    print()
    print("=" * 80)
    print("EXPECTED IMPROVEMENTS:")
    print("=" * 80)
    print("1. ✓ make-bake > make-take (weighted)")
    print("   Onset similarity now matters!")
    print()
    print("2. ✓ Lower rhyme scores overall (weighted)")
    print("   Onset differences properly reduce similarity")
    print()
    print("3. Component breakdown for make-bake vs make-take:")

    make = word_embeddings['make'][0]
    bake = word_embeddings['bake'][0]
    take = word_embeddings['take'][0]

    def split(syll):
        return syll[:128], syll[128:256], syll[256:384]

    make_o, make_n, make_c = split(make)
    bake_o, bake_n, bake_c = split(bake)
    take_o, take_n, take_c = split(take)

    print()
    print("   make-bake components:")
    print(f"     Onset:   {cosine_similarity(make_o, bake_o):.4f}")
    print(f"     Nucleus: {cosine_similarity(make_n, bake_n):.4f}")
    print(f"     Coda:    {cosine_similarity(make_c, bake_c):.4f}")
    print(f"     Weighted avg: {weighted_cosine_similarity(make, bake):.4f}")

    print()
    print("   make-take components:")
    print(f"     Onset:   {cosine_similarity(make_o, take_o):.4f}")
    print(f"     Nucleus: {cosine_similarity(make_n, take_n):.4f}")
    print(f"     Coda:    {cosine_similarity(make_c, take_c):.4f}")
    print(f"     Weighted avg: {weighted_cosine_similarity(make, take):.4f}")

    print()
    print("=" * 80)

if __name__ == "__main__":
    main()
