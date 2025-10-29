#!/usr/bin/env python3
"""
Layer 1: Extract Raw Phoible Features

This script extracts English phonemes from the Phoible database and saves them
as a CSV with their 38 distinctive features (ternary: +, -, 0).

Output: data/phoible/english/phoible-english.csv

This is NOT a training script - Layer 1 features are extracted from the database,
not learned.
"""

from pathlib import Path
import pandas as pd
import sys

sys.path.insert(0, str(Path.cwd()))
from data.mappings.phoneme_vectorizer import load_phoible_csv


def extract_english_phonemes():
    """Extract English phonemes from Phoible database."""

    print("Loading Phoible database...")
    phoible_path = 'data/phoible/phoible.csv'
    phoible_data = load_phoible_csv(phoible_path)

    # Filter for English (ISO 639-3 code: 'eng')
    english_data = [p for p in phoible_data if p.get('ISO6393') == 'eng']

    print(f"Found {len(english_data)} English phoneme entries")

    # Convert to DataFrame for easier saving
    df = pd.DataFrame(english_data)

    # Save to embeddings directory
    output_dir = Path('embeddings/layer1')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / 'phoible_features.csv'
    df.to_csv(output_path, index=False)

    print(f"\nSaved to: {output_path}")
    print(f"Phonemes: {len(df['Phoneme'].unique())}")

    # Show sample
    print("\nSample phonemes:")
    if len(df) > 0:
        print(df[['Phoneme']].head(10).to_string(index=False))


if __name__ == '__main__':
    extract_english_phonemes()
