#!/usr/bin/env python3
"""
Inspect Data Files

A debugging script to examine the structure of the data files
and identify why words are missing phonemes.
"""

import json
import os
from pathlib import Path

# Configuration
DATA_DIR = Path(__file__).parent.parent
GOLD_STANDARD_FILE = DATA_DIR / "processed/gold_standard.json"
WORDNET_WORDS_FILE = DATA_DIR / "processed/wordnet_words.json"
VECTORS_FILE = DATA_DIR / "processed/english_phoneme_vectors_complete.json"

def inspect_gold_standard():
    """Inspect the gold_standard.json file"""
    print("\n=== GOLD STANDARD FILE ===")
    try:
        with open(GOLD_STANDARD_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        print(f"Total words: {len(data)}")
        
        # Sample words
        print("\nSample words from gold_standard.json:")
        for i, (word, word_data) in enumerate(data.items()):
            if i >= 1:
                break
            print(f"  {word}: {word_data}")
            
            # Print keys in the data
            print(f"\nKeys in the first word data: {list(word_data.keys())}")
            
        # Check how many words have arpa_phonemes
        words_with_arpa = sum(1 for word_data in data.values() if 'arpa_phonemes' in word_data)
        print(f"\nWords with arpa_phonemes key: {words_with_arpa} ({words_with_arpa/len(data)*100:.1f}%)")
        
        # Check how many words have phonemes
        words_with_phonemes = sum(1 for word_data in data.values() if 'phonemes' in word_data)
        print(f"Words with phonemes key: {words_with_phonemes} ({words_with_phonemes/len(data)*100:.1f}%)")
        
        # Check for alternate phoneme keys
        if words_with_phonemes == 0:
            # Look for all possible keys 
            all_keys = set()
            for word_data in data.values():
                all_keys.update(word_data.keys())
            print(f"\nAll possible keys in the data: {sorted(all_keys)}")
                
    except Exception as e:
        print(f"Error inspecting gold standard file: {e}")

def inspect_wordnet_words():
    """Inspect the wordnet_words.json file"""
    print("\n=== WORDNET WORDS FILE ===")
    try:
        with open(WORDNET_WORDS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        print(f"Total words: {len(data)}")
        
        # Sample words
        print("\nSample words from wordnet_words.json:")
        for i, (word, word_data) in enumerate(data.items()):
            if i >= 1:
                break
            print(f"  {word}: {word_data}")
            
            # Print keys in the data
            print(f"\nKeys in the first word data: {list(word_data.keys())}")
            
        # Check how many words have arpa_phonemes
        words_with_arpa = sum(1 for word_data in data.values() if 'arpa_phonemes' in word_data)
        print(f"\nWords with arpa_phonemes key: {words_with_arpa} ({words_with_arpa/len(data)*100:.1f}%)")
        
        # Check how many words have phonemes
        words_with_phonemes = sum(1 for word_data in data.values() if 'phonemes' in word_data)
        print(f"Words with phonemes key: {words_with_phonemes} ({words_with_phonemes/len(data)*100:.1f}%)")
        
        # Check for alternate phoneme keys
        if words_with_phonemes == 0:
            # Look for all possible keys 
            all_keys = set()
            for word_data in data.values():
                all_keys.update(word_data.keys())
            print(f"\nAll possible keys in the data: {sorted(all_keys)}")
        
    except Exception as e:
        print(f"Error inspecting wordnet_words file: {e}")
    
def inspect_vectors():
    """Inspect the phoneme vectors file"""
    print("\n=== PHONEME VECTORS FILE ===")
    try:
        with open(VECTORS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        print(f"Total phonemes with vectors: {len(data)}")
        
        # Sample phonemes
        print("\nSample phonemes from vectors file:")
        for i, (phoneme, vector) in enumerate(data.items()):
            if i >= 5:
                break
            print(f"  '{phoneme}': {vector[:3]}...")  # Show just first 3 values
            
    except Exception as e:
        print(f"Error inspecting vectors file: {e}")
        
def compare_phonemes_with_vectors():
    """Compare phonemes from words with available vectors"""
    print("\n=== PHONEME COMPARISON ===")
    try:
        # Load vectors
        with open(VECTORS_FILE, 'r', encoding='utf-8') as f:
            vectors = json.load(f)
            
        # Load words
        with open(WORDNET_WORDS_FILE, 'r', encoding='utf-8') as f:
            words = json.load(f)
        
        # Extract unique phonemes from words
        all_phonemes = set()
        for word_data in words.values():
            # Try both possible keys
            phonemes = word_data.get('phonemes', word_data.get('arpa_phonemes', []))
            for p in phonemes:
                # Remove stress marker if present
                base_p = p[:-1] if p[-1] in '012' else p
                all_phonemes.add(base_p)
        
        print(f"Unique phonemes in words: {len(all_phonemes)}")
        print(f"Phonemes with vectors: {len(vectors)}")
        
        # Show sample phonemes
        if all_phonemes:
            print("\nSample phonemes from words:")
            for p in list(all_phonemes)[:10]:
                print(f"  '{p}'")
        
        # Find phonemes without vectors
        missing_vectors = all_phonemes - set(vectors.keys())
        print(f"\nPhonemes without vectors: {len(missing_vectors)}")
        if missing_vectors:
            print("Sample missing phonemes:")
            for p in list(missing_vectors)[:10]:
                print(f"  '{p}'")
                
        # Check if any words have all their phonemes in the vector set
        words_with_all_phonemes = 0
        for word, word_data in words.items():
            # Try both possible keys
            phonemes = word_data.get('phonemes', word_data.get('arpa_phonemes', []))
            if not phonemes:
                continue
                
            all_present = True
            for p in phonemes:
                base_p = p[:-1] if p[-1] in '012' else p
                if base_p not in vectors:
                    all_present = False
                    break
                    
            if all_present:
                words_with_all_phonemes += 1
                
        print(f"Words with all phonemes in vector set: {words_with_all_phonemes}")
        
    except Exception as e:
        print(f"Error comparing phonemes: {e}")

def main():
    """Main function"""
    print("Inspecting data files...")
    
    inspect_gold_standard()
    inspect_wordnet_words()
    inspect_vectors()
    compare_phonemes_with_vectors()
    
    print("\nInspection complete!")
    
if __name__ == "__main__":
    main() 