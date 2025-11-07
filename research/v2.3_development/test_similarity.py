#!/usr/bin/env python3
"""
Quick test script to verify component-wise normalization fixes rhyme bias
"""

import torch
import numpy as np

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors"""
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

def compute_soft_levenshtein(syllables1, syllables2):
    """Compute soft Levenshtein similarity between syllable sequences"""
    len1 = len(syllables1)
    len2 = len(syllables2)

    # Pre-compute pairwise syllable similarities
    sim_matrix = np.zeros((len1, len2))
    for i in range(len1):
        for j in range(len2):
            sim_matrix[i][j] = cosine_similarity(syllables1[i], syllables2[j])

    # Dynamic programming for edit distance
    dp = np.zeros((len1 + 1, len2 + 1))

    # Initialize: cost of insertions/deletions
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    # Fill DP table
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            # Soft substitution cost (1 - similarity)
            match_cost = 1.0 - sim_matrix[i-1][j-1]

            dp[i][j] = min(
                dp[i-1][j] + 1.0,      # Deletion
                dp[i][j-1] + 1.0,      # Insertion
                dp[i-1][j-1] + match_cost  # Substitution
            )

    # Normalize to [0, 1] similarity
    max_len = max(len1, len2)
    if max_len == 0:
        return 1.0

    edit_distance = dp[len1][len2]
    similarity = 1.0 - (edit_distance / max_len)

    return max(0.0, min(1.0, similarity))

def main():
    print("=" * 80)
    print("Testing Component-Wise Normalization Fix")
    print("=" * 80)
    print()

    # Load embeddings
    checkpoint = torch.load('embeddings/layer4/syllable_embeddings_filtered.pt',
                           map_location='cpu', weights_only=False)

    word_embeddings = checkpoint['word_to_syllable_embeddings']

    print(f"Loaded {len(word_embeddings):,} words")
    print(f"Normalization: {checkpoint.get('normalization', 'UNKNOWN')}")
    print()

    # Test cases
    test_pairs = [
        ('cat', 'bat', 'Rhyme (different onset)'),
        ('make', 'bake', 'Rhyme (similar onset: /m/ vs /b/)'),
        ('make', 'take', 'Rhyme (dissimilar onset: /m/ vs /t/)'),
        ('cat', 'act', 'Anagram'),
        ('dog', 'fog', 'Rhyme'),
        ('cat', 'dog', 'Unrelated'),
    ]

    print(f"{'Word 1':<10} {'Word 2':<10} {'Similarity':<12} {'Description':<30}")
    print("-" * 80)

    for word1, word2, description in test_pairs:
        if word1 in word_embeddings and word2 in word_embeddings:
            emb1 = word_embeddings[word1]
            emb2 = word_embeddings[word2]

            similarity = compute_soft_levenshtein(emb1, emb2)

            print(f"{word1:<10} {word2:<10} {similarity:.4f}       {description:<30}")
        else:
            missing = word1 if word1 not in word_embeddings else word2
            print(f"{word1:<10} {word2:<10} {'N/A':<12} {description:<30} ({missing} not in vocab)")

    print()
    print("=" * 80)
    print("EXPECTED BEHAVIOR (Component-Wise Normalization):")
    print("=" * 80)
    print("1. make-bake > make-take   (Onset similarity matters)")
    print("   /m/-/b/ (bilabial, both voiced) > /m/-/t/ (bilabial vs alveolar, voiced vs voiceless)")
    print()
    print("2. Rhymes should score lower than before (onset differences now count)")
    print("   Old: cat-bat ~0.995 (too high!)")
    print("   New: cat-bat ~0.6-0.8 (more realistic)")
    print()
    print("3. Anagrams should still score low")
    print("   cat-act should remain low due to different syllable structure")
    print()

if __name__ == "__main__":
    main()
