#!/usr/bin/env python3
"""
PhonoLex Data Processing Pipeline

This script runs the complete phonological data processing pipeline:
1. Extract phoneme features from PHOIBLE data
2. Process CMU dictionary to create a phonological dataset
3. Generate feature vectors for all phonemes
4. Create a comprehensive gold standard dataset

Usage:
    python phonolex_pipeline.py [--clean]
"""

import os
import sys
import json
import time
import shutil
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional

# Import phoneme mappings
from phoneme_mappings import (
    ARPA_TO_IPA, 
    IPA_TO_ARPA, 
    save_mapping_files,
    convert_arpa_to_ipa,
    convert_ipa_to_arpa
)

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR = Path(__file__).parent.absolute()
DATA_DIR = SCRIPT_DIR.parent.absolute()
PROCESSED_DIR = DATA_DIR / "processed"
MAPPINGS_DIR = DATA_DIR / "mappings"

# Create necessary directories
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MAPPINGS_DIR, exist_ok=True)

# Source data files
PHOIBLE_CSV = DATA_DIR / "phoible/phoible.csv"
PHOIBLE_FEATURES = DATA_DIR / "phoible/phoible-segments-features.tsv"
CMU_DICT = DATA_DIR / "cmu/cmudict-0.7b"

# Output files
FEATURES_FILE = PROCESSED_DIR / "phoneme_features.json"
VECTORS_FILE = PROCESSED_DIR / "phoneme_vectors.json"
CMU_WORDS_FILE = PROCESSED_DIR / "cmu_words.json"
GOLD_STANDARD_FILE = PROCESSED_DIR / "gold_standard.json"
PHONOLEX_DATASET_FILE = PROCESSED_DIR / "phonolex_dataset.json"

# ============================================================================
# Step 1: Extract phoneme features
# ============================================================================

def extract_phoneme_features():
    """Extract phoneme features from PHOIBLE data"""
    print("\n=== Extracting Phoneme Features ===")
    
    # Check source files
    if not os.path.exists(PHOIBLE_FEATURES):
        print(f"ERROR: PHOIBLE features file not found: {PHOIBLE_FEATURES}")
        return False
        
    print(f"Reading PHOIBLE features: {PHOIBLE_FEATURES}")
    
    # Read the features data
    try:
        features_df = pd.read_csv(PHOIBLE_FEATURES, sep='\t')
        print(f"Loaded {len(features_df)} phonemes with {features_df.shape[1]-1} features")
    except Exception as e:
        print(f"ERROR: Failed to read PHOIBLE features: {e}")
        return False
    
    # Process the features data
    phoneme_features = {}
    feature_names = [col for col in features_df.columns if col != 'segment']
    
    for _, row in features_df.iterrows():
        phoneme = row['segment']
        features = {}
        
        for feature in feature_names:
            value = row[feature]
            if pd.notna(value):
                features[feature] = value
        
        phoneme_features[phoneme] = {
            "features": features
        }
    
    # Convert features to vectors
    phoneme_vectors = {}
    
    for phoneme, data in phoneme_features.items():
        vector = []
        
        for feature in feature_names:
            value = data["features"].get(feature)
            
            # Convert feature values to numbers
            if value == '+':
                vector.append(1.0)
            elif value == '-':
                vector.append(0.0)
            elif value == '0':
                vector.append(0.5)
            else:
                vector.append(0.5)  # Default for undefined features
        
        # Normalize the vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = [v / norm for v in vector]
            
        phoneme_vectors[phoneme] = vector
    
    # Save the data
    features_data = {
        "metadata": {
            "feature_names": feature_names,
            "source": "PHOIBLE"
        },
        "phonemes": phoneme_features
    }
    
    with open(FEATURES_FILE, 'w', encoding='utf-8') as f:
        json.dump(features_data, f, indent=2)
    
    with open(VECTORS_FILE, 'w', encoding='utf-8') as f:
        json.dump(phoneme_vectors, f, indent=2)
    
    print(f"Saved {len(phoneme_features)} phonemes with features to {FEATURES_FILE}")
    print(f"Saved {len(phoneme_vectors)} phoneme vectors to {VECTORS_FILE}")
    return True

# ============================================================================
# Step 2: Process CMU Dictionary
# ============================================================================

def process_cmu_dictionary():
    """Process the CMU Pronouncing Dictionary"""
    print("\n=== Processing CMU Dictionary ===")
    
    # Generate phoneme mapping files
    save_mapping_files()
    
    # Check source files
    if not os.path.exists(CMU_DICT):
        print(f"ERROR: CMU dictionary file not found: {CMU_DICT}")
        return False
    
    print(f"Reading CMU dictionary: {CMU_DICT}")
    
    # Read the features and vectors
    try:
        with open(FEATURES_FILE, 'r', encoding='utf-8') as f:
            features_data = json.load(f)
            
        with open(VECTORS_FILE, 'r', encoding='utf-8') as f:
            phoneme_vectors = json.load(f)
            
        print(f"Loaded {len(phoneme_vectors)} phoneme vectors")
    except Exception as e:
        print(f"ERROR: Failed to read phoneme data: {e}")
        return False
    
    # Process the CMU dictionary
    cmu_words = {}
    
    with open(CMU_DICT, 'r', encoding='latin-1') as f:
        for line in f:
            # Skip comments
            if line.startswith(';;;'):
                continue
                
            # Parse the line
            try:
                parts = line.strip().split()
                if not parts:
                    continue
                    
                word = parts[0].lower()
                
                # Handle alternate pronunciations
                if '(' in word and ')' in word:
                    word = word.split('(')[0]
                
                # Skip non-alphabetic words
                if not word.isalpha():
                    continue
                
                # Get the phonemes
                phonemes = parts[1:]
                
                # Convert to IPA
                ipa_phonemes = []
                for p in phonemes:
                    ipa = convert_arpa_to_ipa(p)
                    ipa_phonemes.append(ipa)
                
                # Add to the dataset
                cmu_words[word] = {
                    "word": word,
                    "arpa_phonemes": phonemes,
                    "ipa_phonemes": ipa_phonemes
                }
            except Exception as e:
                continue
    
    # Save the data
    with open(CMU_WORDS_FILE, 'w', encoding='utf-8') as f:
        json.dump(cmu_words, f)
    
    print(f"Processed {len(cmu_words)} words from CMU dictionary")
    print(f"Saved to {CMU_WORDS_FILE}")
    return True

# ============================================================================
# Step 3: Create Gold Standard Dataset
# ============================================================================

def create_gold_standard():
    """Create a gold standard dataset filtered to common words"""
    print("\n=== Creating Gold Standard Dataset ===")
    
    # Import NLTK for WordNet
    try:
        import nltk
        from nltk.corpus import wordnet as wn
        
        # Ensure WordNet is available
        try:
            wn.synsets('test')
        except LookupError:
            print("Downloading WordNet...")
            nltk.download('wordnet', quiet=True)
    except ImportError:
        print("ERROR: NLTK not installed. Please install with 'pip install nltk'")
        return False
    
    # Read the CMU dictionary data
    try:
        with open(CMU_WORDS_FILE, 'r', encoding='utf-8') as f:
            cmu_words = json.load(f)
            
        print(f"Loaded {len(cmu_words)} words from CMU dictionary")
    except Exception as e:
        print(f"ERROR: Failed to read CMU words: {e}")
        return False
    
    # Filter to words in WordNet
    print("Filtering to words present in WordNet...")
    gold_standard = {}
    pos_counts = {"NOUN": 0, "VERB": 0, "ADJ": 0, "ADJ_SAT": 0, "ADV": 0}
    
    for word, data in cmu_words.items():
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
            word_data = data.copy()
            word_data["pos_tags"] = pos_tags
            gold_standard[word] = word_data
    
    print(f"Found {len(gold_standard)} words that are in WordNet")
    
    # Compute statistics
    stats = {
        "total_words": len(gold_standard),
        "pos_counts": pos_counts
    }
    
    # Save the data
    with open(GOLD_STANDARD_FILE, 'w', encoding='utf-8') as f:
        json.dump(gold_standard, f)
    
    print(f"Saved gold standard dataset to {GOLD_STANDARD_FILE}")
    return True

# ============================================================================
# Step 4: Create PhonoLex Dataset
# ============================================================================

def create_phonolex_dataset():
    """Create the complete PhonoLex dataset"""
    print("\n=== Creating PhonoLex Dataset ===")
    
    # Read the required data
    try:
        with open(GOLD_STANDARD_FILE, 'r', encoding='utf-8') as f:
            gold_standard = json.load(f)
            
        with open(VECTORS_FILE, 'r', encoding='utf-8') as f:
            phoneme_vectors = json.load(f)
            
        print(f"Loaded {len(gold_standard)} words from gold standard")
        print(f"Loaded {len(phoneme_vectors)} phoneme vectors")
    except Exception as e:
        print(f"ERROR: Failed to read input data: {e}")
        return False
    
    # Create the dataset
    dataset = {
        "metadata": {
            "word_count": len(gold_standard),
            "source": "CMU dictionary filtered to WordNet words"
        },
        "words": {}
    }
    
    # Process each word
    words_processed = 0
    words_skipped = 0
    
    for word, data in gold_standard.items():
        # Get phonemes
        arpa_phonemes = data.get("arpa_phonemes", [])
        ipa_phonemes = data.get("ipa_phonemes", [])
        
        # Skip words with no phonemes
        if not arpa_phonemes:
            words_skipped += 1
            continue
            
        # Get base phonemes (no stress)
        base_phonemes = []
        for p in arpa_phonemes:
            if p[-1].isdigit():
                base_phonemes.append(p[:-1])
            else:
                base_phonemes.append(p)
        
        # Compute word vector
        phoneme_vecs = []
        for p in base_phonemes:
            ipa = convert_arpa_to_ipa(p)
            if ipa in phoneme_vectors:
                phoneme_vecs.append(phoneme_vectors[ipa])
        
        if not phoneme_vecs:
            words_skipped += 1
            continue
            
        # Average the vectors
        word_vector = np.mean(phoneme_vecs, axis=0).tolist()
        
        # Identify syllables
        syllables = []
        current_syllable = []
        stress_pattern = []
        
        for p in arpa_phonemes:
            # Check if this is a vowel with stress marked
            if p[-1].isdigit():
                stress = int(p[-1])
                stress_pattern.append(stress)
                current_syllable.append(p)
                syllables.append(current_syllable)
                current_syllable = []
            else:
                current_syllable.append(p)
        
        # Add any remaining phonemes to the last syllable
        if current_syllable:
            if syllables:
                syllables[-1].extend(current_syllable)
            else:
                syllables.append(current_syllable)
        
        syllable_info = {
            "syllable_count": len(syllables),
            "syllables": syllables,
            "stress_pattern": stress_pattern
        }
        
        # Create word entry
        word_entry = {
            "arpa_phonemes": arpa_phonemes,
            "ipa_phonemes": ipa_phonemes,
            "pos_tags": data.get("pos_tags", []),
            "word_vector": word_vector,
            "syllables": syllable_info
        }
        
        dataset["words"][word] = word_entry
        words_processed += 1
    
    # Update metadata
    dataset["metadata"]["word_count"] = words_processed
    
    print(f"Processed {words_processed} words successfully")
    print(f"Skipped {words_skipped} words due to missing phonemes or vectors")
    
    # Save the dataset
    with open(PHONOLEX_DATASET_FILE, 'w', encoding='utf-8') as f:
        json.dump(dataset, f)
    
    print(f"Saved dataset with {words_processed} words to {PHONOLEX_DATASET_FILE}")
    return True

# ============================================================================
# Pipeline Runner
# ============================================================================

def clean_directory():
    """Clean the processed directory to start fresh"""
    print("Cleaning processed directory...")
    
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
        print("Created fresh processed directory")
        return
    
    files = list(PROCESSED_DIR.glob("*.json"))
    if not files:
        print("Processed directory is already empty")
        return
    
    # Remove existing files
    for file in files:
        try:
            os.remove(file)
            print(f"Removed: {file.name}")
        except Exception as e:
            print(f"Error removing {file.name}: {e}")
    
    print("Processed directory cleaned")

def main():
    """Run the complete PhonoLex pipeline"""
    start_time = time.time()
    print("Starting PhonoLex data processing pipeline...")
    
    # Check if --clean flag is provided
    if "--clean" in sys.argv:
        clean_directory()
    
    # Run each step in sequence
    if not extract_phoneme_features():
        print("Pipeline failed at step 1: Extract phoneme features")
        return 1
    
    if not process_cmu_dictionary():
        print("Pipeline failed at step 2: Process CMU dictionary")
        return 1
    
    if not create_gold_standard():
        print("Pipeline failed at step 3: Create gold standard dataset")
        return 1
    
    if not create_phonolex_dataset():
        print("Pipeline failed at step 4: Create PhonoLex dataset")
        return 1
    
    # Verify output files
    output_files = [
        FEATURES_FILE,
        VECTORS_FILE,
        CMU_WORDS_FILE,
        GOLD_STANDARD_FILE,
        PHONOLEX_DATASET_FILE
    ]
    
    print("\nVerifying output files...")
    all_found = True
    
    for file in output_files:
        if os.path.exists(file):
            size_mb = os.path.getsize(file) / (1024 * 1024)
            print(f"✓ Found: {file.name} ({size_mb:.1f} MB)")
        else:
            print(f"❌ Missing: {file.name}")
            all_found = False
    
    # Show dataset summary
    try:
        with open(PHONOLEX_DATASET_FILE, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        words = dataset.get('words', {})
        
        print("\nDATASET SUMMARY:")
        print(f"Total words: {len(words)}")
        
        # Count words with specific properties
        pos_counts = {}
        syllable_counts = {}
        
        for word, info in words.items():
            # Count part of speech
            for pos in info.get('pos_tags', []):
                pos_counts[pos] = pos_counts.get(pos, 0) + 1
            
            # Count syllable patterns
            syllable_count = info.get('syllables', {}).get('syllable_count', 0)
            syllable_counts[syllable_count] = syllable_counts.get(syllable_count, 0) + 1
        
        # Print POS stats
        print("\nPart of speech distribution:")
        for pos, count in sorted(pos_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(words)) * 100
            print(f"  {pos}: {count} words ({percentage:.1f}%)")
        
        # Print syllable stats
        print("\nSyllable count distribution:")
        for syllables, count in sorted(syllable_counts.items()):
            if syllables > 0:  # Only count valid syllable counts
                percentage = (count / len(words)) * 100
                print(f"  {syllables} syllables: {count} words ({percentage:.1f}%)")
    except Exception as e:
        print(f"Error generating summary: {e}")
    
    # Done!
    elapsed = time.time() - start_time
    print(f"\nPipeline completed in {elapsed:.1f} seconds.")
    print("All processed data is available in the data/processed directory")
    return 0 if all_found else 1

if __name__ == "__main__":
    sys.exit(main()) 