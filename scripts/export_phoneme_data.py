#!/usr/bin/env python3
"""
Export phoneme data for client-side use
Generates phonemes.json with IPA, type, and Phoible features
ONLY includes phonemes actually used in the exported word data
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.mappings.phoneme_vectorizer import load_phoible_csv

def main():
    # Paths
    root = Path(__file__).parent.parent
    data_dir = root / 'data'
    word_metadata_path = root / 'webapp' / 'frontend' / 'public' / 'data' / 'word_metadata.json'
    english_csv = data_dir / 'phoible' / 'english' / 'phoible-english.csv'
    output_path = root / 'webapp' / 'frontend' / 'public' / 'data' / 'phonemes.json'

    print(f"Loading word data from {word_metadata_path}")

    # Load word metadata to get actual phonemes in use
    with open(word_metadata_path) as f:
        word_data = json.load(f)

    # Collect all unique phonemes from actual word data
    used_phonemes = set()
    for word_info in word_data.values():
        if 'phonemes' in word_info:
            for phoneme in word_info['phonemes']:
                used_phonemes.add(phoneme)

    print(f"Found {len(used_phonemes)} unique phonemes in word data")

    # Load Phoible data
    print(f"Loading Phoible features from {english_csv}")
    phoneme_data = load_phoible_csv(str(english_csv))
    print(f"Loaded {len(phoneme_data)} Phoible entries")

    # Build phoneme dict from Phoible, filtered to used phonemes
    phonemes_dict = {}
    for entry in phoneme_data:
        ipa = entry['Phoneme']

        # Only include if actually used in our word data
        if ipa not in used_phonemes:
            continue

        if ipa not in phonemes_dict:
            # Extract features (all columns except metadata)
            features = {k: v for k, v in entry.items() if k not in ['Phoneme', 'LanguageName', 'InventoryID', 'SegmentClass', 'Glottocode', 'ISO6393', 'SpecificDialect', 'GlyphID', 'Allophones', 'Marginal', 'Source']}

            # Determine type (vowel vs consonant)
            # Vowels have syllabic=+ or consonantal=-
            phoneme_type = 'vowel' if features.get('syllabic') == '+' else 'consonant'

            phonemes_dict[ipa] = {
                'ipa': ipa,
                'type': phoneme_type,
                'features': features
            }

    # Convert to sorted list
    phonemes = sorted(phonemes_dict.values(), key=lambda x: x['ipa'])

    # Separate by type for reporting
    consonants = [p for p in phonemes if p['type'] == 'consonant']
    vowels = [p for p in phonemes if p['type'] == 'vowel']

    data = {
        'phonemes': phonemes,
        'count': len(phonemes)
    }

    # Write JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Exported {len(phonemes)} phonemes ({len(consonants)} consonants, {len(vowels)} vowels)")
    print(f"✓ Output: {output_path}")
    print(f"✓ Size: {output_path.stat().st_size / 1024:.1f} KB")

if __name__ == '__main__':
    main()
