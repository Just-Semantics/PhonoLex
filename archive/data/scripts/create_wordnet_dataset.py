#!/usr/bin/env python3
"""
Create PhonoLex Dataset

This script creates the complete PhonoLex dataset by:
1. Loading the gold standard word list
2. Loading phoneme vectors and features
3. Creating word vectors based on phoneme sequences
4. Computing syllable information
5. Saving the final dataset as a structured JSON file

The final dataset includes:
- Phoneme representations (IPA and ARPAbet)
- Phoneme feature vectors 
- Word vectors
- Syllable information
- Part of speech tags
"""

import os
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional

# Configuration
DATA_DIR = Path(__file__).parent.parent
INPUT_FILE = DATA_DIR / "processed/gold_standard.json"  # Using new name
VECTORS_FILE = DATA_DIR / "processed/english_phoneme_vectors_complete.json"
FEATURES_FILE = DATA_DIR / "processed/english_phoneme_features_complete.json"
OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_FILE = OUTPUT_DIR / "phonolex_dataset.json"  # Using new name

# Legacy paths for backward compatibility
LEGACY_INPUT_FILE = DATA_DIR / "processed/wordnet_words.json"
LEGACY_OUTPUT_FILE = OUTPUT_DIR / "wordnet_dataset.json"

def compute_word_vector(phonemes, phoneme_vectors):
    """Compute word vector by averaging phoneme vectors"""
    vectors = []
    for phoneme in phonemes:
        if phoneme in phoneme_vectors:
            vectors.append(phoneme_vectors[phoneme])
    
    if not vectors:
        return None
        
    # Average the vectors
    return np.mean(vectors, axis=0).tolist()

def identify_syllables(phonemes):
    """Identify syllables in phoneme sequence based on stress markers"""
    syllables = []
    current_syllable = []
    stress_pattern = []
    
    for phoneme in phonemes:
        # Check if this is a vowel with stress marked
        if len(phoneme) > 1 and phoneme[-1] in '012':
            stress = int(phoneme[-1])
            stress_pattern.append(stress)
            current_syllable.append(phoneme)
            syllables.append(current_syllable)
            current_syllable = []
        else:
            current_syllable.append(phoneme)
    
    # Add any remaining phonemes to the last syllable
    if current_syllable:
        if syllables:
            syllables[-1].extend(current_syllable)
        else:
            syllables.append(current_syllable)
    
    return {
        "syllable_count": len(syllables),
        "syllables": syllables,
        "stress_pattern": stress_pattern
    }

def main():
    """Create the complete dataset"""
    print("Loading gold standard words...")
    
    # Try loading from the new file path first
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            wordnet_words = json.load(f)
    except FileNotFoundError:
        # Fall back to legacy file if new one doesn't exist
        print(f"New file {INPUT_FILE} not found, using legacy file {LEGACY_INPUT_FILE}")
        with open(LEGACY_INPUT_FILE, 'r', encoding='utf-8') as f:
            wordnet_words = json.load(f)
    
    print(f"Loaded {len(wordnet_words)} words from gold standard")
    
    print("Loading phoneme vectors...")
    with open(VECTORS_FILE, 'r', encoding='utf-8') as f:
        phoneme_vectors = json.load(f)
    
    print(f"Loaded vectors for {len(phoneme_vectors)} phonemes")
    
    print("Loading phoneme features...")
    with open(FEATURES_FILE, 'r', encoding='utf-8') as f:
        phoneme_features = json.load(f)
    
    # Create the dataset
    print("Creating PhonoLex dataset...")
    
    # Extract vector dimensions
    sample_vec = next(iter(phoneme_vectors.values()))
    vector_dim = len(sample_vec)
    
    dataset = {
        "metadata": {
            "word_count": len(wordnet_words),
            "vector_dimensions": vector_dim,
            "source": "CMU dictionary filtered to WordNet words"
        },
        "words": {}
    }
    
    # Process each word
    for word, data in tqdm(wordnet_words.items()):
        # Get phonemes (without stress markers for vector)
        arpa_phonemes = data.get("phonemes", [])
        
        # Skip words with no phonemes
        if not arpa_phonemes:
            continue
            
        # Get base phonemes (no stress)
        base_phonemes = [p[:-1] if p[-1] in '012' else p for p in arpa_phonemes]
        
        # Compute word vector
        word_vector = compute_word_vector(base_phonemes, phoneme_vectors)
        
        # Identify syllables
        syllable_info = identify_syllables(arpa_phonemes)
        
        # Create word entry
        word_entry = {
            "arpa_phonemes": arpa_phonemes,
            "ipa_phonemes": data.get("ipa", []),
            "pos_tags": data.get("pos_tags", []),
            "phoneme_vectors": [phoneme_vectors.get(p, None) for p in base_phonemes],
            "word_vector": word_vector,
            "syllables": syllable_info
        }
        
        dataset["words"][word] = word_entry
    
    # Save dataset
    print(f"Saving dataset with {len(dataset['words'])} words to {OUTPUT_FILE}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(dataset, f)
        
    # Save to legacy path for backward compatibility
    with open(LEGACY_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(dataset, f)
    
    print("Done!")
    return 0

if __name__ == "__main__":
    main() 