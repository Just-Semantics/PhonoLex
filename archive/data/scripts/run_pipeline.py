#!/usr/bin/env python3
"""
PhonoLex Data Processing Pipeline

This script runs the complete data processing pipeline:
1. Extract English phoneme features from PHOIBLE
2. Add missing phonemes not in PHOIBLE
3. Process CMU dictionary to create a phonological dataset
4. Filter words to create a gold standard dataset
5. Create a comprehensive dataset with phonological and semantic information

Usage:
    cd data/scripts
    python run_pipeline.py
"""

import os
import sys
import time
import subprocess
import shutil
from pathlib import Path

# Configuration
SCRIPT_DIR = Path(__file__).parent.absolute()
DATA_DIR = SCRIPT_DIR.parent.absolute()
PROCESSED_DIR = DATA_DIR / "processed"

# Ensure processed directory exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Steps in the pipeline
SCRIPTS = [
    "extract_english_features.py",
    "missing_phonemes.py", 
    "process_cmu.py",
    "filter_gold_standard.py",  # Renamed from filter_wordnet.py
    "create_dataset.py",        # Renamed from create_wordnet_dataset.py
    "filter_supported_phonemes.py"  # Add this as the final step
]

# File name mapping - old to new
FILE_MAPPING = {
    "wordnet_words.json": "gold_standard.json",
    "wordnet_stats.json": "gold_standard_stats.json",
    "wordnet_dataset.json": "phonolex_dataset.json"
}

def check_source_files():
    """Check that all required source files exist"""
    print("Checking source files...")
    
    required_files = [
        DATA_DIR / "phoible/phoible.csv",
        DATA_DIR / "phoible/phoible-segments-features.tsv",
        DATA_DIR / "cmu/cmudict-0.7b",
        DATA_DIR / "mappings/arpa_to_ipa.json"
    ]
    
    missing = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing.append(file_path)
            print(f"❌ Missing: {file_path}")
        else:
            print(f"✓ Found: {file_path}")
    
    if missing:
        print(f"\nERROR: {len(missing)} required source files are missing!")
        print("Please ensure all source files are present before running the pipeline.")
        return False
        
    return True

def rename_files():
    """Rename output files according to FILE_MAPPING"""
    print("\nRenaming files to use clearer names...")
    for old_name, new_name in FILE_MAPPING.items():
        old_path = PROCESSED_DIR / old_name
        new_path = PROCESSED_DIR / new_name
        
        if os.path.exists(old_path):
            # If the new file already exists, remove it first
            if os.path.exists(new_path):
                os.remove(new_path)
                
            # Copy the file instead of renaming to keep the original for compatibility
            shutil.copy2(old_path, new_path)
            print(f"✓ Copied: {old_name} → {new_name}")
        else:
            print(f"❌ Missing: {old_name} (cannot rename)")

def run_script(script_name):
    """Run a Python script and return whether it succeeded"""
    script_path = SCRIPT_DIR / script_name
    
    print(f"\n{'='*80}")
    print(f"Running {script_name}...")
    print(f"{'='*80}\n")
    
    result = subprocess.run([sys.executable, script_path], 
                            capture_output=False,
                            text=True,
                            cwd=SCRIPT_DIR)  # Run from scripts directory
    
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
        "wordnet_words.json",             # Keep checking old files 
        "wordnet_stats.json",             # to ensure backward compatibility
        "wordnet_dataset.json",
        # Check new files
        "gold_standard.json",
        "gold_standard_stats.json",
        "phonolex_dataset.json",
        "phoneme_vectors.json",           # Filtered phoneme vectors
        "phoneme_features.json",          # Filtered phoneme features
        "phoneme_vectors_full.json",      # Backup of full phoneme vectors
        "phoneme_features_full.json"      # Backup of full phoneme features
    ]
    
    print("\nVerifying output files...")
    missing_files = []
    
    for filename in expected_files:
        path = PROCESSED_DIR / filename
        if not os.path.exists(path):
            # For renamed files, only warn if both versions are missing
            if filename in FILE_MAPPING:
                new_path = PROCESSED_DIR / FILE_MAPPING[filename]
                if os.path.exists(new_path):
                    size_mb = os.path.getsize(new_path) / (1024 * 1024)
                    print(f"✓ Found: {filename} → {FILE_MAPPING[filename]} ({size_mb:.1f} MB)")
                    continue
            
            missing_files.append(filename)
            print(f"❌ Missing: {filename}")
        else:
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"✓ Found: {filename} ({size_mb:.1f} MB)")
    
    if missing_files and all(f not in FILE_MAPPING.values() for f in missing_files):
        print(f"\nWarning: {len(missing_files)} expected files are missing!")
        return False
    else:
        print("\nAll expected output files were generated successfully.")
        return True

def show_dataset_summary():
    """Show a summary of the generated dataset"""
    try:
        import json
        dataset_file = PROCESSED_DIR / "phonolex_dataset.json"
        
        if not os.path.exists(dataset_file):
            # Try the old name if new one doesn't exist
            dataset_file = PROCESSED_DIR / "wordnet_dataset.json"
            if not os.path.exists(dataset_file):
                print("Dataset file not found.")
                return
        
        with open(dataset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        metadata = data.get('metadata', {})
        words = data.get('words', {})
        
        print("\nDATASET SUMMARY:")
        print(f"Total words: {len(words)}")
        print(f"Vector dimensions: {metadata.get('vector_dimensions', 0)}")
        
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

def clean_processed_directory():
    """Remove any existing processed files to start fresh"""
    print("Cleaning processed directory...")
    
    # Check if directory exists
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
        print("Created empty processed directory.")
        return
        
    # Count files
    files = list(PROCESSED_DIR.glob("*.json"))
    if not files:
        print("Processed directory is already empty.")
        return
        
    # Ask for confirmation if files exist
    print(f"Found {len(files)} existing files in {PROCESSED_DIR}")
    response = input("Do you want to delete these files and start fresh? (y/n): ")
    
    if response.lower() != 'y':
        print("Keeping existing files.")
        return
        
    # Delete files
    for file in files:
        try:
            os.remove(file)
            print(f"Deleted {file.name}")
        except Exception as e:
            print(f"Error deleting {file.name}: {e}")
    
    print("Processed directory cleaned.")

def main():
    """Main function to run the complete pipeline"""
    start_time = time.time()
    print("Starting PhonoLex data processing pipeline...")
    
    # Check if all source files exist
    if not check_source_files():
        print("Pipeline aborted due to missing source files.")
        return 1
    
    # Optionally clean processed directory
    if "--clean" in sys.argv:
        clean_processed_directory()
    
    # Create output directory
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    print(f"Output directory: {PROCESSED_DIR}")
    
    # Run each script in sequence
    for i, script in enumerate(SCRIPTS):
        print(f"\nStep {i+1}/{len(SCRIPTS)}: {script}")
        if not run_script(script):
            print("Pipeline failed!")
            return 1
    
    # Rename files to use clearer names
    rename_files()
    
    # Verify outputs
    verify_outputs()
    
    # Show dataset summary
    show_dataset_summary()
    
    # Done!
    elapsed = time.time() - start_time
    print(f"\nPipeline completed in {elapsed:.1f} seconds.")
    print("All processed data is available in the data/processed directory")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 