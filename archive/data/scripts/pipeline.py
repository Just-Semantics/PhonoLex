#!/usr/bin/env python3
"""
PhonoLex Data Processing Pipeline

This script runs the complete data processing pipeline:
1. Extract English phoneme features from PHOIBLE
2. Add missing phonemes not in PHOIBLE
3. Process CMU dictionary to create a phonological dataset
4. Filter CMU data to only include WordNet words
5. Create a comprehensive dataset with phonological and semantic information

Usage:
    python pipeline.py
"""

import os
import sys
import time
import subprocess
from pathlib import Path

# Ensure we're in the right directory
os.chdir(Path(__file__).parent)

# Configuration
PROCESSED_DIR = "../data/processed"
SCRIPTS = [
    "extract_english_features.py",
    "missing_phonemes.py", 
    "process_cmu.py",
    "filter_wordnet.py",
    "create_wordnet_dataset.py"
]

def run_script(script_name):
    """Run a Python script and return whether it succeeded"""
    print(f"\n{'='*80}")
    print(f"Running {script_name}...")
    print(f"{'='*80}\n")
    
    result = subprocess.run([sys.executable, script_name], 
                            capture_output=False,
                            text=True)
    
    if result.returncode != 0:
        print(f"Error running {script_name}! Pipeline halted.")
        return False
    
    return True

def verify_outputs():
    """Verify that expected output files exist"""
    expected_files = [
        "english_phoneme_features.json",
        "english_phoneme_vectors.json", 
        "english_phoneme_features_complete.json",
        "english_phoneme_vectors_complete.json",
        "cmu_words.json",
        "cmu_phonemes.json",
        "wordnet_words.json",
        "wordnet_stats.json",
        "wordnet_dataset.json"
    ]
    
    print("\nVerifying output files...")
    missing_files = []
    
    for filename in expected_files:
        path = os.path.join(PROCESSED_DIR, filename)
        if not os.path.exists(path):
            missing_files.append(filename)
            print(f"❌ Missing: {filename}")
        else:
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"✓ Found: {filename} ({size_mb:.1f} MB)")
    
    if missing_files:
        print(f"\nWarning: {len(missing_files)} expected files are missing!")
        return False
    else:
        print("\nAll expected output files were generated successfully.")
        return True

def main():
    """Main function to run the complete pipeline"""
    start_time = time.time()
    print("Starting PhonoLex data processing pipeline...")
    
    # Create output directory
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    print(f"Output directory: {PROCESSED_DIR}")
    
    # Run each script in sequence
    for i, script in enumerate(SCRIPTS):
        print(f"\nStep {i+1}/{len(SCRIPTS)}: {script}")
        if not run_script(script):
            print("Pipeline failed!")
            return 1
    
    # Verify outputs
    verify_outputs()
    
    # Done!
    elapsed = time.time() - start_time
    print(f"\nPipeline completed in {elapsed:.1f} seconds.")
    print("All processed data is available in the data/processed directory")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 