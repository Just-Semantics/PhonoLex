#!/usr/bin/env python3
"""
Debug why some words don't get Phoible embeddings.
Check which phonemes are missing from Phoible features.
"""

import pickle
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd()))
from src.phonolex.embeddings.english_data_loader import EnglishPhonologyLoader


def main():
    print("=" * 80)
    print("DEBUGGING PHOIBLE COVERAGE")
    print("=" * 80)
    print()

    # Load Phoible features
    layer2_path = Path('embeddings/layer2/normalized_76d.pkl')
    with open(layer2_path, 'rb') as f:
        phoible_features = pickle.load(f)

    print(f"Phoible features loaded: {len(phoible_features)} phonemes")
    print(f"Phonemes in Phoible: {sorted(phoible_features.keys())}")
    print()

    # Load CMU dictionary
    loader = EnglishPhonologyLoader()

    # Test words
    test_words = ['cat', 'bat', 'make', 'bake', 'take', 'act', 'dog', 'fog']

    print("=" * 80)
    print("CHECKING TEST WORDS")
    print("=" * 80)
    print()

    for word in test_words:
        if word not in loader.lexicon_with_stress:
            print(f"{word}: NOT IN DICTIONARY")
            continue

        phonemes_with_stress = loader.lexicon_with_stress[word]
        phonemes = [p.phoneme for p in phonemes_with_stress]

        print(f"{word}: {' '.join(phonemes)}")

        # Check each phoneme
        missing = []
        for phoneme in phonemes:
            if phoneme not in phoible_features:
                missing.append(phoneme)

        if missing:
            print(f"  ❌ MISSING from Phoible: {missing}")
        else:
            print(f"  ✓ All phonemes in Phoible")
        print()


if __name__ == "__main__":
    main()
