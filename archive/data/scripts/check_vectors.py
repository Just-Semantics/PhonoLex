#!/usr/bin/env python3
"""
Check for fuzzy values in phoneme vectors.
"""

import json
import numpy as np

# Paths
FEATURES_FILE = "data/processed/english_phoneme_features_complete.json"
VECTORS_FILE = "data/processed/english_phoneme_vectors_complete.json"

def main():
    """Check for fuzzy values in vectors"""
    print("Loading phoneme data...")
    
    # Load feature data
    with open(FEATURES_FILE, 'r', encoding='utf-8') as f:
        features_data = json.load(f)
        
    # Load vector data
    with open(VECTORS_FILE, 'r', encoding='utf-8') as f:
        vectors_data = json.load(f)
    
    # Get feature names for reference
    feature_names = features_data.get('metadata', {}).get('feature_names', [])
    
    print(f"Loaded {len(vectors_data)} phoneme vectors")
    print(f"Using {len(feature_names)} features: {', '.join(feature_names)}")
    print("\nChecking for fuzzy values in vectors...")
    
    # Count phonemes with fuzzy values
    phonemes_with_fuzzy = 0
    fuzzy_pairs = []
    
    for phoneme, vector in vectors_data.items():
        # Find values that are between 0 and 1 (but not exactly 0.5)
        fuzzy_indices = [i for i, v in enumerate(vector) if 0 < v < 1 and abs(v - 0.5) > 0.01]
        
        if fuzzy_indices:
            phonemes_with_fuzzy += 1
            print(f"\n{phoneme}: has {len(fuzzy_indices)} fuzzy values")
            
            # Print the fuzzy feature values
            for idx in fuzzy_indices[:5]:  # Limit to first 5 for readability
                feature_name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
                print(f"  {feature_name}: {vector[idx]:.4f}")
            
            # Keep track for analysis
            if phoneme in features_data.get('phonemes', {}):
                features = features_data['phonemes'][phoneme].get('features', {})
                for idx in fuzzy_indices[:3]:
                    if idx < len(feature_names):
                        feature = feature_names[idx]
                        value = features.get(feature, "N/A")
                        fuzzy_pairs.append((phoneme, feature, value, vector[idx]))
    
    print(f"\nFound {phonemes_with_fuzzy} phonemes with fuzzy values out of {len(vectors_data)}")
    
    # Show some examples of fuzzy features
    if fuzzy_pairs:
        print("\nExample fuzzy feature to vector mappings:")
        for phoneme, feature, orig_value, vector_value in fuzzy_pairs[:10]:
            print(f"{phoneme}.{feature}: '{orig_value}' -> {vector_value:.4f}")

if __name__ == "__main__":
    main() 