#!/usr/bin/env python3
"""
Filter Phoneme Data to Supported Phonemes

This script filters the phoneme vectors and features to only include 
the phonemes supported in the English IPA to ARPA mappings.

It's the final step in the data processing pipeline to ensure we only 
distribute the phoneme data we actually use.

Usage:
    python filter_supported_phonemes.py
"""

import os
import sys
import json
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent.absolute()
DATA_DIR = SCRIPT_DIR.parent.absolute()
PROCESSED_DIR = DATA_DIR / "processed"
MAPPINGS_DIR = DATA_DIR / "mappings"

# Input files
IPA_TO_ARPA_FILE = MAPPINGS_DIR / "ipa_to_arpa.json"
PHONEME_VECTORS_FILE = PROCESSED_DIR / "phoneme_vectors.json"
PHONEME_FEATURES_FILE = PROCESSED_DIR / "phoneme_features.json"

# Output files (same names - will overwrite the originals)
OUTPUT_VECTORS_FILE = PROCESSED_DIR / "phoneme_vectors.json"
OUTPUT_FEATURES_FILE = PROCESSED_DIR / "phoneme_features.json"
# Also create a backup of the original files
BACKUP_VECTORS_FILE = PROCESSED_DIR / "phoneme_vectors_full.json"
BACKUP_FEATURES_FILE = PROCESSED_DIR / "phoneme_features_full.json"

def main():
    """Main function to filter the phoneme data"""
    print("Filtering phoneme data to supported phonemes...")
    
    if not os.path.exists(IPA_TO_ARPA_FILE):
        print(f"Error: Mapping file not found: {IPA_TO_ARPA_FILE}")
        return 1
        
    if not os.path.exists(PHONEME_VECTORS_FILE):
        print(f"Error: Phoneme vectors file not found: {PHONEME_VECTORS_FILE}")
        return 1
        
    if not os.path.exists(PHONEME_FEATURES_FILE):
        print(f"Error: Phoneme features file not found: {PHONEME_FEATURES_FILE}")
        return 1
    
    # Load the IPA to ARPAbet mapping
    print(f"Reading IPA to ARPA mapping from {IPA_TO_ARPA_FILE}")
    with open(IPA_TO_ARPA_FILE, 'r', encoding='utf-8') as f:
        ipa_to_arpa = json.load(f)
    
    # Get the list of supported IPA symbols
    supported_ipa = list(ipa_to_arpa.keys())
    print(f"Found {len(supported_ipa)} supported IPA symbols")
    
    # Load the full phoneme vectors
    print(f"Reading phoneme vectors from {PHONEME_VECTORS_FILE}")
    with open(PHONEME_VECTORS_FILE, 'r', encoding='utf-8') as f:
        all_phoneme_vectors = json.load(f)
    
    # Load the phoneme features
    print(f"Reading phoneme features from {PHONEME_FEATURES_FILE}")
    with open(PHONEME_FEATURES_FILE, 'r', encoding='utf-8') as f:
        all_phoneme_features = json.load(f)
    
    # Create backups of the original files
    print("Creating backups of the original files...")
    with open(BACKUP_VECTORS_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_phoneme_vectors, f, ensure_ascii=False)
    
    with open(BACKUP_FEATURES_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_phoneme_features, f, ensure_ascii=False)
    
    # Extract metadata from features
    metadata = all_phoneme_features.get('metadata', {})
    
    # Filter phoneme vectors to only include supported IPA symbols
    filtered_vectors = {}
    matched_count = 0
    missing_count = 0
    
    for ipa in supported_ipa:
        if ipa in all_phoneme_vectors:
            filtered_vectors[ipa] = all_phoneme_vectors[ipa]
            matched_count += 1
        else:
            print(f"Warning: No vector found for supported IPA symbol: {ipa}")
            missing_count += 1
    
    # Filter phoneme features to only include supported IPA symbols
    filtered_features = {'metadata': metadata}
    
    for ipa in supported_ipa:
        if ipa in all_phoneme_features:
            filtered_features[ipa] = all_phoneme_features[ipa]
        else:
            print(f"Warning: No features found for supported IPA symbol: {ipa}")
    
    # Write filtered files (overwriting the originals)
    print(f"Writing filtered phoneme vectors to {OUTPUT_VECTORS_FILE}")
    with open(OUTPUT_VECTORS_FILE, 'w', encoding='utf-8') as f:
        json.dump(filtered_vectors, f, ensure_ascii=False, indent=2)
    
    print(f"Writing filtered phoneme features to {OUTPUT_FEATURES_FILE}")
    with open(OUTPUT_FEATURES_FILE, 'w', encoding='utf-8') as f:
        json.dump(filtered_features, f, ensure_ascii=False, indent=2)
    
    # Summary
    total_vectors_before = len(all_phoneme_vectors)
    total_vectors_after = len(filtered_vectors)
    
    print("\nSUMMARY:")
    print(f"Total vectors before filtering: {total_vectors_before}")
    print(f"Total vectors after filtering: {total_vectors_after}")
    print(f"Matched supported IPA symbols: {matched_count}")
    print(f"Missing vectors for supported IPA symbols: {missing_count}")
    print(f"Backed up original files to {BACKUP_VECTORS_FILE} and {BACKUP_FEATURES_FILE}")
    
    data_reduction = (1 - (total_vectors_after / total_vectors_before)) * 100
    print(f"Data size reduction: {data_reduction:.1f}%")
    
    print("\nFiltering complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 