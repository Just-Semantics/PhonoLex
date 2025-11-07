#!/usr/bin/env python3
"""
Phase 2: Normalized Feature Vectors for ALL PHOIBLE Languages

This script processes all 3,142 unique phonemes from the PHOIBLE database
(across 2,716 languages) and computes Phase 2 normalized vectors.

Output:
- embeddings/phase2/phoible_all_phonemes_76d.pkl
- embeddings/phase2/phoible_all_phonemes_152d.pkl
- embeddings/phase2/phoible_language_inventories.json

Each phoneme includes:
- IPA symbol
- 76-dim normalized vector (endpoints)
- 152-dim normalized vector (trajectory)
- List of languages that use this phoneme

This enables cross-linguistic phoneme difficulty analysis (Flege's SLM).
"""

from pathlib import Path
import sys
import numpy as np
import json
import pickle
from collections import defaultdict

sys.path.insert(0, str(Path.cwd()))
from data.mappings.phoneme_vectorizer import PhonemeVectorizer, load_phoible_csv


def process_all_phoible_phonemes():
    """Process all PHOIBLE phonemes and create language-aware embeddings."""

    print("=== Phase 2: All PHOIBLE Languages ===\n")

    # Load full PHOIBLE database
    print("Loading PHOIBLE database...")
    phoible_path = "data/phoible/phoible.csv"
    phoible_data = load_phoible_csv(phoible_path)

    print(f"Loaded {len(phoible_data):,} phoneme entries")

    # Initialize vectorizer
    vectorizer = PhonemeVectorizer(encoding_scheme="three_way")

    # Data structures
    phoneme_to_languages = defaultdict(set)  # phoneme -> set of ISO codes
    language_to_phonemes = defaultdict(set)  # ISO code -> set of phonemes
    language_names = {}  # ISO code -> language name
    language_glottocodes = {}  # ISO code -> glottocode
    unique_phoneme_data = {}  # phoneme -> PHOIBLE feature dict

    # Process all entries
    print("\nProcessing phoneme entries...")
    for entry in phoible_data:
        phoneme = entry["Phoneme"]
        iso_code = entry.get("ISO6393", "")
        lang_name = entry.get("LanguageName", "")
        glottocode = entry.get("Glottocode", "")

        # Skip if missing critical info
        if not phoneme or not iso_code:
            continue

        # Track phoneme-language relationships
        phoneme_to_languages[phoneme].add(iso_code)
        language_to_phonemes[iso_code].add(phoneme)

        # Store language metadata
        if iso_code not in language_names:
            language_names[iso_code] = lang_name
            language_glottocodes[iso_code] = glottocode

        # Store first occurrence of each phoneme for vectorization
        if phoneme not in unique_phoneme_data:
            unique_phoneme_data[phoneme] = entry

    print(f"Found {len(unique_phoneme_data):,} unique phonemes")
    print(f"Found {len(language_to_phonemes):,} languages")

    # Compute vectors for all unique phonemes
    print("\nComputing normalized vectors...")
    embeddings_76d = {}
    embeddings_152d = {}

    for i, (phoneme, data) in enumerate(unique_phoneme_data.items(), 1):
        if i % 500 == 0:
            print(f"  Processed {i:,} / {len(unique_phoneme_data):,} phonemes...")

        try:
            vec = vectorizer.vectorize(data)
            embeddings_76d[phoneme] = vec.endpoints_76d
            embeddings_152d[phoneme] = vec.trajectory_152d
        except Exception as e:
            print(f"  Warning: Failed to vectorize '{phoneme}': {e}")
            continue

    print(f"Successfully vectorized {len(embeddings_76d):,} phonemes")

    # Create language inventories structure
    print("\nCreating language inventories...")
    language_inventories = {}

    for iso_code, phoneme_set in language_to_phonemes.items():
        # Only include phonemes that were successfully vectorized
        valid_phonemes = [p for p in phoneme_set if p in embeddings_76d]

        if valid_phonemes:
            language_inventories[iso_code] = {
                "iso": iso_code,
                "name": language_names.get(iso_code, ""),
                "glottocode": language_glottocodes.get(iso_code, ""),
                "phonemes": sorted(valid_phonemes),
                "phoneme_count": len(valid_phonemes)
            }

    print(f"Created inventories for {len(language_inventories):,} languages")

    # Create phoneme metadata with language lists
    print("\nCreating phoneme metadata...")
    phoneme_metadata = {}

    for phoneme in embeddings_76d.keys():
        languages = sorted(phoneme_to_languages[phoneme])
        phoneme_metadata[phoneme] = {
            "ipa": phoneme,
            "languages": languages,
            "language_count": len(languages)
        }

    # Save embeddings
    print("\nSaving Phase 2 embeddings...")
    output_dir = Path("embeddings/phase2")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save pickle files (for Python use)
    with open(output_dir / "phoible_all_phonemes_76d.pkl", "wb") as f:
        pickle.dump(embeddings_76d, f)
    with open(output_dir / "phoible_all_phonemes_152d.pkl", "wb") as f:
        pickle.dump(embeddings_152d, f)

    print(f"  Saved {len(embeddings_76d):,} phoneme vectors:")
    print(f"    - phoible_all_phonemes_76d.pkl")
    print(f"    - phoible_all_phonemes_152d.pkl")

    # Save language inventories (JSON for frontend)
    with open(output_dir / "phoible_language_inventories.json", "w") as f:
        json.dump(language_inventories, f, indent=2, ensure_ascii=False)

    print(f"  Saved {len(language_inventories):,} language inventories:")
    print(f"    - phoible_language_inventories.json")

    # Save phoneme metadata (JSON for frontend)
    with open(output_dir / "phoible_phoneme_metadata.json", "w") as f:
        json.dump(phoneme_metadata, f, indent=2, ensure_ascii=False)

    print(f"  Saved metadata for {len(phoneme_metadata):,} phonemes:")
    print(f"    - phoible_phoneme_metadata.json")

    # Print statistics
    print("\n=== Statistics ===")
    print(f"Unique phonemes: {len(embeddings_76d):,}")
    print(f"Languages: {len(language_inventories):,}")
    print(f"Total phoneme-language pairs: {sum(len(v['phonemes']) for v in language_inventories.values()):,}")

    # Show some examples
    print("\n=== Sample Languages ===")
    sample_languages = ["eng", "spa", "fra", "deu", "cmn", "jpn", "ara", "hin"]
    for iso in sample_languages:
        if iso in language_inventories:
            lang = language_inventories[iso]
            print(f"{lang['name']:20} ({iso}): {lang['phoneme_count']:3} phonemes")

    print("\n=== Sample Phonemes ===")
    sample_phonemes = ["t", "d", "θ", "ʃ", "ŋ", "ɾ", "ʔ", "ɲ"]
    for phoneme in sample_phonemes:
        if phoneme in phoneme_metadata:
            meta = phoneme_metadata[phoneme]
            print(f"{phoneme:3} - Used in {meta['language_count']:4} languages")

    # Calculate file sizes
    import os
    size_76d = os.path.getsize(output_dir / "phoible_all_phonemes_76d.pkl") / (1024 * 1024)
    size_152d = os.path.getsize(output_dir / "phoible_all_phonemes_152d.pkl") / (1024 * 1024)
    size_lang = os.path.getsize(output_dir / "phoible_language_inventories.json") / (1024 * 1024)
    size_meta = os.path.getsize(output_dir / "phoible_phoneme_metadata.json") / (1024 * 1024)

    print(f"\n=== File Sizes ===")
    print(f"76-dim vectors:      {size_76d:.2f} MB")
    print(f"152-dim vectors:     {size_152d:.2f} MB")
    print(f"Language inventories: {size_lang:.2f} MB")
    print(f"Phoneme metadata:    {size_meta:.2f} MB")
    print(f"Total:               {size_76d + size_152d + size_lang + size_meta:.2f} MB")

    print("\n✓ Phase 2 complete! Ready for frontend export.")


if __name__ == "__main__":
    process_all_phoible_phonemes()
