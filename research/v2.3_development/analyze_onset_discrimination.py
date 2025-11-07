#!/usr/bin/env python3
"""
Detailed analysis of onset discrimination across different phoneme pairs.

Tests how well each approach distinguishes phonemes based on phonological features.
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


def get_onset_embedding(word, phonemes_with_stress, embeddings_dict, is_phoible=False):
    """Extract onset embedding from a word."""
    syllables = syllabify(phonemes_with_stress)
    if not syllables or not syllables[0].onset:
        return None

    if is_phoible:
        # For Phoible: build syllable embedding first
        word_phonemes = [p.phoneme for p in phonemes_with_stress]
        phoneme = word_phonemes[0]  # First phoneme = onset
        if phoneme not in embeddings_dict:
            return None
        vec = embeddings_dict[phoneme].copy()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec[:76]  # Return just the onset part for comparison
    else:
        # For MLM: extract from pre-built syllable embeddings
        word_key = word if word in embeddings_dict else word.upper()
        if word_key not in embeddings_dict:
            return None
        syllable = embeddings_dict[word_key][0]  # First syllable
        return syllable[:128]  # Onset is first 128 dims


def main():
    print("=" * 80)
    print("ONSET DISCRIMINATION ANALYSIS")
    print("=" * 80)
    print()

    # Load embeddings
    print("Loading embeddings...")
    mlm_checkpoint = torch.load(
        'embeddings/layer4/syllable_embeddings_filtered.pt',
        map_location='cpu',
        weights_only=False
    )
    mlm_embeddings = mlm_checkpoint['word_to_syllable_embeddings']

    layer2_path = Path('embeddings/layer2/normalized_76d.pkl')
    with open(layer2_path, 'rb') as f:
        phoible_features = pickle.load(f)

    loader = EnglishPhonologyLoader()
    print("✓ Data loaded")
    print()

    # Test word pairs designed to test specific phonological contrasts
    test_groups = [
        {
            'name': 'Voicing Contrast',
            'pairs': [
                ('bat', 'pat', '/b/ vs /p/ (voiced vs voiceless bilabial)'),
                ('dad', 'tad', '/d/ vs /t/ (voiced vs voiceless alveolar)'),
                ('gap', 'cap', '/g/ vs /k/ (voiced vs voiceless velar)'),
            ]
        },
        {
            'name': 'Place of Articulation',
            'pairs': [
                ('bat', 'mat', '/b/ vs /m/ (both bilabial, both voiced)'),
                ('bat', 'gat', '/b/ vs /g/ (bilabial vs velar)'),
                ('pat', 'tat', '/p/ vs /t/ (bilabial vs alveolar)'),
            ]
        },
        {
            'name': 'Manner of Articulation',
            'pairs': [
                ('bat', 'mat', '/b/ vs /m/ (stop vs nasal)'),
                ('sat', 'chat', '/s/ vs /tʃ/ (fricative vs affricate)'),
            ]
        },
        {
            'name': 'Multi-Feature Differences',
            'pairs': [
                ('make', 'take', '/m/ vs /t/ (bilabial nasal vs alveolar stop)'),
                ('make', 'bake', '/m/ vs /b/ (both bilabial, both voiced)'),
                ('bat', 'that', '/b/ vs /ð/ (stop vs fricative)'),
            ]
        }
    ]

    for group in test_groups:
        print("=" * 80)
        print(f"{group['name']}")
        print("=" * 80)
        print()
        print(f"{'Pair':<20} {'MLM Onset':<15} {'Phoible Onset':<15} {'Description':<40}")
        print("-" * 80)

        for word1, word2, description in group['pairs']:
            # Get onset embeddings
            mlm_onset1 = get_onset_embedding(
                word1,
                loader.lexicon_with_stress.get(word1),
                mlm_embeddings,
                is_phoible=False
            )
            mlm_onset2 = get_onset_embedding(
                word2,
                loader.lexicon_with_stress.get(word2),
                mlm_embeddings,
                is_phoible=False
            )

            phoible_onset1 = get_onset_embedding(
                word1,
                loader.lexicon_with_stress.get(word1),
                phoible_features,
                is_phoible=True
            )
            phoible_onset2 = get_onset_embedding(
                word2,
                loader.lexicon_with_stress.get(word2),
                phoible_features,
                is_phoible=True
            )

            # Compute similarities
            mlm_sim = "N/A"
            if mlm_onset1 is not None and mlm_onset2 is not None:
                mlm_sim = f"{cosine_similarity(mlm_onset1, mlm_onset2):.4f}"

            phoible_sim = "N/A"
            if phoible_onset1 is not None and phoible_onset2 is not None:
                phoible_sim = f"{cosine_similarity(phoible_onset1, phoible_onset2):.4f}"

            pair_name = f"{word1}-{word2}"
            print(f"{pair_name:<20} {mlm_sim:<15} {phoible_sim:<15} {description:<40}")

        print()

    print("=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print()
    print("1. PHOIBLE STRENGTHS:")
    print("   • Clear phonological feature grounding")
    print("   • Predictable similarity patterns based on linguistic theory")
    print("   • High similarity for same-place-same-voicing (e.g., /b/-/m/)")
    print("   • Lower similarity for different features")
    print()
    print("2. MLM STRENGTHS:")
    print("   • Learns distributional patterns from data")
    print("   • May capture English-specific phonotactic constraints")
    print("   • Better anagram discrimination (seen in earlier tests)")
    print()
    print("3. RECOMMENDATION:")
    print("   • If you want instant deployment and linguistic transparency: Use Phoible")
    print("   • If you want learned patterns and better discrimination: Use MLM")
    print("   • Or: Offer both as options in the UI!")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
