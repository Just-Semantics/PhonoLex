#!/usr/bin/env python3
"""
Filter Gold Standard Words

This script filters the CMU dictionary to create a gold standard dataset
containing only words that are present in WordNet.

Output:
- gold_standard.json: Words from CMU that are also in WordNet
- gold_standard_stats.json: Statistics about the filtered dataset
"""

import os
import sys
import json
import nltk
import logging
from pathlib import Path
from tqdm import tqdm
from nltk.corpus import wordnet as wn
from typing import Dict, List, Any, Set, Tuple

# Ensure NLTK data is available
try:
    wn.synsets('test')
except LookupError:
    nltk.download('wordnet')

# Configuration
DATA_DIR = Path(__file__).parent.parent
INPUT_FILE = DATA_DIR / "processed/cmu_words.json"
OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_FILE = OUTPUT_DIR / "gold_standard.json"
STATS_FILE = OUTPUT_DIR / "gold_standard_stats.json"

# Legacy file paths for backward compatibility
LEGACY_OUTPUT_FILE = OUTPUT_DIR / "wordnet_words.json"
LEGACY_STATS_FILE = OUTPUT_DIR / "wordnet_stats.json"

def main():
    """Filter CMU dictionary to only include WordNet words"""
    print("Loading CMU dictionary...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        cmu_words = json.load(f)
    
    print(f"Loaded {len(cmu_words)} words from CMU dictionary")
    
    # Filter to words in WordNet
    print("Filtering to words present in WordNet...")
    wordnet_words = {}
    pos_counts = {"NOUN": 0, "VERB": 0, "ADJ": 0, "ADJ_SAT": 0, "ADV": 0}
    
    for word, data in tqdm(cmu_words.items()):
        # Skip words with apostrophes, hyphens, or digits
        if "'" in word or "-" in word or any(c.isdigit() for c in word):
            continue
            
        # Check if word is in WordNet
        synsets = wn.synsets(word)
        if synsets:
            # Add POS tags
            pos_tags = []
            for synset in synsets:
                pos = synset.pos()
                # Map WordNet POS to our format
                if pos == 'n':
                    pos_tags.append("NOUN")
                    pos_counts["NOUN"] += 1
                elif pos == 'v':
                    pos_tags.append("VERB")
                    pos_counts["VERB"] += 1
                elif pos == 'a':
                    pos_tags.append("ADJ")
                    pos_counts["ADJ"] += 1
                elif pos == 's':
                    pos_tags.append("ADJ_SAT")
                    pos_counts["ADJ_SAT"] += 1
                elif pos == 'r':
                    pos_tags.append("ADV")
                    pos_counts["ADV"] += 1
            
            # Remove duplicates
            pos_tags = list(set(pos_tags))
            
            # Add to filtered dict with POS information
            word_data = cmu_words[word].copy()
            word_data["pos_tags"] = pos_tags
            wordnet_words[word] = word_data
    
    print(f"Found {len(wordnet_words)} words that are in WordNet")
    
    # Compute statistics
    stats = {
        "total_words": len(wordnet_words),
        "pos_counts": pos_counts
    }
    
    # Save filtered words
    print(f"Saving filtered words to {OUTPUT_FILE}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(wordnet_words, f)
    
    # Save legacy file for backward compatibility
    with open(LEGACY_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(wordnet_words, f)
    
    # Save statistics
    print(f"Saving statistics to {STATS_FILE}...")
    with open(STATS_FILE, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    # Save legacy stats file
    with open(LEGACY_STATS_FILE, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    print("Done!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 