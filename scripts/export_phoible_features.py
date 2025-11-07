#!/usr/bin/env python3
"""
Export PHOIBLE phoneme features for frontend lookup.

Reads phoible-segments-features.tsv and exports to JSON format
for use in SearchTool phoneme lookup.

Output: webapp/frontend/public/data/phoible_features.json
Contains: 2,162 phonemes with 37 articulatory features (+/-, 0)
"""

import csv
import json
import gzip
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

def load_phoible_features():
    """Load PHOIBLE segments features from TSV"""
    print("Loading PHOIBLE segments features...")

    tsv_path = project_root / "data/phoible/phoible-segments-features.tsv"

    phonemes = []

    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')

        for row in reader:
            ipa = row['segment']

            # Extract features (all columns except 'segment')
            features = {}
            for key, value in row.items():
                if key != 'segment':
                    features[key] = value

            # Determine type based on syllabic feature
            phoneme_type = 'vowel' if features.get('syllabic') == '+' else 'consonant'

            phonemes.append({
                'ipa': ipa,
                'type': phoneme_type,
                'features': features
            })

    print(f"  ✓ Loaded {len(phonemes):,} phonemes")
    return phonemes

def export_phoible_features():
    """Export PHOIBLE features to JSON"""
    print("\nExporting PHOIBLE features...")

    phonemes = load_phoible_features()

    output_dir = project_root / "webapp/frontend/public/data"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create export structure
    export_data = {
        "phonemes": phonemes,
        "count": len(phonemes),
        "source": "PHOIBLE segments-features.tsv",
        "description": "2,162 phonemes with 37 articulatory features"
    }

    # Export uncompressed
    output_path = output_dir / "phoible_features.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, separators=(',', ':'))

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  ✓ phoible_features.json ({size_mb:.1f} MB)")

    # Also export compressed
    output_path_gz = output_dir / "phoible_features.json.gz"
    with gzip.open(output_path_gz, 'wt', encoding='utf-8') as f:
        json.dump(export_data, f, separators=(',', ':'))

    size_mb_gz = output_path_gz.stat().st_size / 1024 / 1024
    print(f"  ✓ phoible_features.json.gz ({size_mb_gz:.1f} MB)")
    print(f"  ✓ Compression: {100 * (1 - size_mb_gz/size_mb):.1f}% reduction")

    print(f"\n✓ SUCCESS: Exported {len(phonemes):,} phonemes")
    print(f"  Output: {output_path}")
    print(f"  Output (gz): {output_path_gz}")

if __name__ == "__main__":
    export_phoible_features()
