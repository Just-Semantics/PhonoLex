#!/usr/bin/env python3
"""
Test using Phoible vectors directly (Layer 2) without Layer 3 training.

This bypasses the transformer entirely and uses pure phonological features
from Phoible to build syllable embeddings.
"""

import pickle
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd()))
from src.phonolex.embeddings.english_data_loader import EnglishPhonologyLoader
from src.phonolex.utils.syllabification import syllabify


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


def build_phoible_syllable_embeddings(word, phonemes_with_stress, phoible_features):
    """
    Build syllable embeddings directly from Layer 2 Phoible features.

    Uses the same onset-nucleus-coda structure but with 76-dim Phoible features
    instead of 128-dim learned embeddings.
    """
    try:
        # Syllabify
        syllables = syllabify(phonemes_with_stress)
        if not syllables:
            return None

        # Extract plain phonemes
        word_phonemes = [p.phoneme for p in phonemes_with_stress]

        syllable_embeddings = []
        phoneme_idx = 0

        for syl in syllables:
            # Onset (76-dim) - normalize individually
            if syl.onset:
                onset_vecs = []
                for _ in syl.onset:
                    phoneme = word_phonemes[phoneme_idx]
                    if phoneme not in phoible_features:
                        return None  # Unknown phoneme
                    onset_vecs.append(phoible_features[phoneme])
                    phoneme_idx += 1

                onset_emb = np.mean(onset_vecs, axis=0)  # Average if multiple consonants
                onset_norm = np.linalg.norm(onset_emb)
                if onset_norm > 0:
                    onset_emb = onset_emb / onset_norm  # Normalize separately
            else:
                onset_emb = np.zeros(76)  # Zero vector for empty onset

            # Nucleus (76-dim) - normalize individually
            nucleus_phoneme = word_phonemes[phoneme_idx]
            if nucleus_phoneme not in phoible_features:
                return None
            nucleus_emb = phoible_features[nucleus_phoneme].copy()
            nucleus_norm = np.linalg.norm(nucleus_emb)
            if nucleus_norm > 0:
                nucleus_emb = nucleus_emb / nucleus_norm
            phoneme_idx += 1

            # Coda (76-dim) - normalize individually
            if syl.coda:
                coda_vecs = []
                for _ in syl.coda:
                    phoneme = word_phonemes[phoneme_idx]
                    if phoneme not in phoible_features:
                        return None
                    coda_vecs.append(phoible_features[phoneme])
                    phoneme_idx += 1

                coda_emb = np.mean(coda_vecs, axis=0)
                coda_norm = np.linalg.norm(coda_emb)
                if coda_norm > 0:
                    coda_emb = coda_emb / coda_norm
            else:
                coda_emb = np.zeros(76)  # Zero vector for empty coda

            # Concatenate: 76 + 76 + 76 = 228-dim syllable embedding
            syllable_emb = np.concatenate([onset_emb, nucleus_emb, coda_emb])
            syllable_embeddings.append(syllable_emb)

        return syllable_embeddings

    except Exception as e:
        return None


def main():
    print("=" * 80)
    print("Testing Pure Phoible Vectors (No Training)")
    print("=" * 80)
    print()

    # Load Layer 2 Phoible features
    print("Loading Layer 2 Phoible features...")
    layer2_path = Path('embeddings/layer2/normalized_76d.pkl')
    with open(layer2_path, 'rb') as f:
        phoible_features = pickle.load(f)
    print(f"✓ Loaded {len(phoible_features)} phoneme feature vectors (76-dim)")
    print()

    # Load CMU dictionary
    print("Loading CMU dictionary...")
    loader = EnglishPhonologyLoader()
    print(f"✓ Loaded {len(loader.lexicon)} words")
    print()

    # Test words
    test_words = ['cat', 'bat', 'make', 'bake', 'take', 'act', 'dog', 'fog']

    # Build Phoible-only syllable embeddings
    print("Building syllable embeddings from pure Phoible features...")
    word_embeddings = {}
    for word in test_words:
        if word in loader.lexicon_with_stress:
            emb = build_phoible_syllable_embeddings(
                word,
                loader.lexicon_with_stress[word],
                phoible_features
            )
            if emb is not None:
                word_embeddings[word] = emb

    print(f"✓ Built embeddings for {len(word_embeddings)} words")
    print()

    # Test pairs
    test_pairs = [
        ('cat', 'bat', 'Rhyme (different onset)'),
        ('make', 'bake', 'Rhyme (similar onset: /m/ vs /b/)'),
        ('make', 'take', 'Rhyme (dissimilar onset: /m/ vs /t/)'),
        ('cat', 'act', 'Anagram'),
        ('dog', 'fog', 'Rhyme'),
        ('cat', 'dog', 'Unrelated'),
    ]

    print("=" * 80)
    print("SIMILARITY SCORES (Pure Phoible Features)")
    print("=" * 80)
    print()
    print(f"{'Word 1':<10} {'Word 2':<10} {'Similarity':<12} {'Description':<30}")
    print("-" * 80)

    for word1, word2, description in test_pairs:
        if word1 in word_embeddings and word2 in word_embeddings:
            emb1 = word_embeddings[word1]
            emb2 = word_embeddings[word2]

            similarity = compute_soft_levenshtein(emb1, emb2)

            print(f"{word1:<10} {word2:<10} {similarity:.4f}       {description:<30}")

    print()
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print("These scores use ONLY Phoible phonological features (38 features → 76-dim).")
    print("No transformer training, no learned patterns - pure linguistic theory.")
    print()
    print("Key question: Is make-bake > make-take with just Phoible?")
    print("=" * 80)

    # Component analysis for make-bake vs make-take
    print()
    print("Component-level analysis:")
    print()

    make = word_embeddings['make'][0]
    bake = word_embeddings['bake'][0]
    take = word_embeddings['take'][0]

    def split(syll):
        return syll[:76], syll[76:152], syll[152:228]

    make_o, make_n, make_c = split(make)
    bake_o, bake_n, bake_c = split(bake)
    take_o, take_n, take_c = split(take)

    print("make-bake components:")
    print(f"  Onset:   {cosine_similarity(make_o, bake_o):.4f}  (/m/ vs /b/)")
    print(f"  Nucleus: {cosine_similarity(make_n, bake_n):.4f}  (/eɪ/ vs /eɪ/)")
    print(f"  Coda:    {cosine_similarity(make_c, bake_c):.4f}  (/k/ vs /k/)")

    print()
    print("make-take components:")
    print(f"  Onset:   {cosine_similarity(make_o, take_o):.4f}  (/m/ vs /t/)")
    print(f"  Nucleus: {cosine_similarity(make_n, take_n):.4f}  (/eɪ/ vs /eɪ/)")
    print(f"  Coda:    {cosine_similarity(make_c, take_c):.4f}  (/k/ vs /k/)")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
