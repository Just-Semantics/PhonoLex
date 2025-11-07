#!/usr/bin/env python3
"""
Merge phonotactic probability into word_metadata.json

Adds the following fields to each word:
- phono_prob_avg: Mean biphone probability (main metric)
- phono_prob_sum_log: Sum of log probabilities (Vitevitch metric)
- positional_prob_avg: Mean positional segment probability
"""

import json
from pathlib import Path

# Paths
WEBAPP_DATA = Path(__file__).parent.parent / "webapp" / "frontend" / "public" / "data"
WORD_METADATA_PATH = WEBAPP_DATA / "word_metadata.json"
PHONO_PROB_PATH = Path(__file__).parent.parent / "data" / "phonotactic_probability_24k.json"

def main():
    print("=" * 70)
    print("Merging Phonotactic Probability into word_metadata.json")
    print("=" * 70)
    print()

    # Load word metadata
    print(f"Loading word metadata from {WORD_METADATA_PATH}...")
    with open(WORD_METADATA_PATH, 'r', encoding='utf-8') as f:
        word_metadata = json.load(f)
    print(f"  ✓ Loaded {len(word_metadata)} words")

    # Load phonotactic probabilities
    print(f"Loading phonotactic probabilities from {PHONO_PROB_PATH}...")
    with open(PHONO_PROB_PATH, 'r', encoding='utf-8') as f:
        phono_data = json.load(f)
    phono_probs = phono_data['word_probabilities']
    print(f"  ✓ Loaded {len(phono_probs)} word probabilities")
    print()

    # Merge data
    print("Merging data...")
    matched = 0
    missing = 0

    for word, metadata in word_metadata.items():
        if word in phono_probs:
            probs = phono_probs[word]

            # Add phonotactic probability fields
            metadata['phono_prob_avg'] = round(probs['phono_prob_avg'], 6)
            metadata['phono_prob_sum_log'] = round(probs['phono_prob_sum_log'], 4)
            metadata['positional_prob_avg'] = round(probs['positional_prob_avg'], 6)

            matched += 1
        else:
            # Set to null if not found
            metadata['phono_prob_avg'] = None
            metadata['phono_prob_sum_log'] = None
            metadata['positional_prob_avg'] = None
            missing += 1

    print(f"  ✓ Matched: {matched} words")
    if missing > 0:
        print(f"  ⚠ Missing: {missing} words (set to null)")
    print()

    # Show sample
    print("Sample entries:")
    print("-" * 70)
    for word in ['cat', 'strength', 'the', 'computer'][:4]:
        if word in word_metadata:
            m = word_metadata[word]
            print(f"{word}:")
            print(f"  phono_prob_avg: {m['phono_prob_avg']}")
            print(f"  phono_prob_sum_log: {m['phono_prob_sum_log']}")
            print(f"  positional_prob_avg: {m['positional_prob_avg']}")
    print()

    # Save updated metadata
    print(f"Saving updated metadata to {WORD_METADATA_PATH}...")
    with open(WORD_METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(word_metadata, f, indent=2, ensure_ascii=False)

    print(f"  ✓ Saved {len(word_metadata)} words with phonotactic probabilities")
    print()
    print("=" * 70)
    print("✓ Complete!")
    print("=" * 70)

if __name__ == '__main__':
    main()
