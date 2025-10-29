#!/usr/bin/env python3
"""
Run PhonoLex Data Processing Pipeline

This script runs the full data processing pipeline for PhonoLex, including:
1. Extracting English phoneme features from PHOIBLE
2. Adding missing phonemes not found in PHOIBLE
3. Processing the CMU dictionary with these features

Usage:
    python run_processing.py
"""

import os
import subprocess
import sys
from pathlib import Path

# Configuration
OUTPUT_DIR = "data/processed"

def main():
    """Run the full PhonoLex data processing pipeline"""
    print("Starting PhonoLex data processing pipeline...")
    
    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Created output directory: {OUTPUT_DIR}")
    
    # Run the PHOIBLE feature extraction
    print("\n1. Extracting English phoneme features from PHOIBLE...")
    result = subprocess.run(["python", "extract_english_features.py"], 
                          capture_output=True, text=True)
    
    if result.returncode != 0:
        print("ERROR: Failed to extract PHOIBLE features")
        print(f"Error output: {result.stderr}")
        sys.exit(1)
    else:
        print(result.stdout)
    
    # Add missing phonemes
    print("\n2. Adding missing phonemes not found in PHOIBLE...")
    result = subprocess.run(["python", "missing_phonemes.py"],
                          capture_output=True, text=True)
    
    if result.returncode != 0:
        print("ERROR: Failed to add missing phonemes")
        print(f"Error output: {result.stderr}")
        sys.exit(1)
    else:
        print(result.stdout)
        
    # Run the CMU dictionary processing
    print("\n3. Processing CMU dictionary with phoneme features...")
    result = subprocess.run(["python", "process_cmu.py"], 
                          capture_output=True, text=True)
    
    if result.returncode != 0:
        print("ERROR: Failed to process CMU dictionary")
        print(f"Error output: {result.stderr}")
        sys.exit(1)
    else:
        print(result.stdout)
    
    # Check if output files exist
    output_files = [
        "english_phoneme_features.json",
        "english_phoneme_vectors.json",
        "english_phoneme_features_complete.json",
        "english_phoneme_vectors_complete.json",
        "cmu_words.json",
        "cmu_phonemes.json"
    ]
    
    print("\nVerifying output files...")
    for file in output_files:
        path = os.path.join(OUTPUT_DIR, file)
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024  # KB
            print(f"✓ {file} ({size:.1f} KB)")
        else:
            print(f"✗ {file} (missing)")
    
    print("\nPhonoLex data processing complete!")
    print(f"All processed data is available in the {OUTPUT_DIR} directory")


if __name__ == "__main__":
    main() 