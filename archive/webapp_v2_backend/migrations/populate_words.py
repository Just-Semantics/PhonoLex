#!/usr/bin/env python3
"""
Populate the words table with data from CMU dictionary and embeddings.

This script:
1. Loads words from the CMU Pronouncing Dictionary
2. Generates syllable embeddings using the hierarchical model
3. Computes derived properties (word_length, complexity)
4. Inserts all data into PostgreSQL
"""

import sys
import os
import json
import psycopg2
from psycopg2.extras import execute_batch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.phonolex.embeddings.english_data_loader import EnglishPhonologyLoader
from src.phonolex.utils.syllabification import syllabify
import torch

# Import word filter for vocabulary selection
sys.path.insert(0, str(Path(__file__).parent))
from word_filter import WordFilter

# Connection string (can be overridden with environment variable)
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/phonolex")


def load_hierarchical_model(model_path="models/hierarchical/final.pt"):
    """Load the trained hierarchical syllable embedding model"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from train_hierarchical_final import HierarchicalPhonemeEncoder

    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu')

    # Get phoneme to ID mapping from checkpoint
    phoneme_to_id = checkpoint.get('phoneme_to_id', {})
    num_phonemes = len(phoneme_to_id)

    # Get model configuration from state dict shape
    embedding_weight = checkpoint['model_state_dict']['phoneme_embeddings.weight']
    num_phonemes = embedding_weight.shape[0]
    d_model = embedding_weight.shape[1]

    print(f"Model config: num_phonemes={num_phonemes}, d_model={d_model}")

    # Model configuration (matches training)
    model = HierarchicalPhonemeEncoder(
        num_phonemes=num_phonemes,
        d_model=d_model,
        nhead=4,  # Default from training
        num_layers=3  # Default from training
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Extract metadata if available
    if 'epoch' in checkpoint:
        print(f"✓ Model loaded (trained for {checkpoint.get('epoch', '?')} epochs)")
    else:
        print(f"✓ Model loaded")

    return model, phoneme_to_id


def compute_syllable_embedding(word_phonemes, model, ipa_to_idx):
    """
    Compute 384-dim syllable embedding for a word.

    Args:
        word_phonemes: List of IPA phonemes
        model: Trained HierarchicalPhonemeEncoder
        ipa_to_idx: Dict mapping IPA phonemes to indices

    Returns:
        numpy array of shape (384,) or None if computation fails
    """
    try:
        # Syllabify word
        syllables = syllabify(word_phonemes)
        if not syllables:
            return None

        # Convert phonemes to indices
        phoneme_indices = []
        for phoneme in word_phonemes:
            if phoneme in ipa_to_idx:
                phoneme_indices.append(ipa_to_idx[phoneme])
            else:
                # Unknown phoneme - skip this word
                return None

        # Create input tensor
        input_tensor = torch.tensor([phoneme_indices], dtype=torch.long)

        # Forward pass
        with torch.no_grad():
            syllable_embeddings = model.encode_syllables(input_tensor)  # Shape: (1, num_syllables, 384)

        # Average syllable embeddings to get word embedding
        word_embedding = syllable_embeddings.mean(dim=1).squeeze(0).numpy()  # Shape: (384,)

        return word_embedding

    except Exception as e:
        # Silently skip words with errors
        return None


def compute_wcm_score(phonemes, syllables):
    """
    Compute Word Complexity Measure (WCM) score.

    Simplified version based on:
    - Phoneme count
    - Syllable count
    - Cluster complexity

    Args:
        phonemes: List of phoneme strings (IPA)
        syllables: List of Syllable objects (already computed)

    Returns:
        int (1-10 scale)
    """
    num_syllables = len(syllables)
    num_phonemes = len(phonemes)

    # Base score from phoneme count
    score = min(num_phonemes, 10)

    # Penalty for consonant clusters
    for syl in syllables:
        if len(syl.onset) > 1:
            score += 1
        if len(syl.coda) > 1:
            score += 1

    return min(int(score), 10)


def categorize_word_length(phoneme_count):
    """Categorize word length based on phoneme count"""
    if phoneme_count <= 3:
        return 'short'
    elif phoneme_count <= 6:
        return 'medium'
    else:
        return 'long'


def categorize_complexity(wcm_score):
    """Categorize complexity based on WCM score"""
    if wcm_score <= 3:
        return 'low'
    elif wcm_score <= 6:
        return 'medium'
    else:
        return 'high'


def prepare_word_data(loader, model, phoneme_to_id, word_filter, limit=None):
    """
    Prepare all word data for insertion.

    Args:
        loader: EnglishPhonologyLoader instance
        model: Trained hierarchical model
        phoneme_to_id: Dict mapping phonemes to model indices (from checkpoint)
        word_filter: WordFilter instance for vocabulary selection
        limit: Optional limit on number of words (for testing)

    Returns:
        List of tuples ready for database insertion
    """
    words_data = []
    filtered_count = 0

    print(f"Using phoneme mapping with {len(phoneme_to_id)} phonemes from model checkpoint")

    print("Preparing word data with filtering criterion (freq + any norm)...")
    word_list = list(loader.lexicon_with_stress.items())
    if limit:
        word_list = word_list[:limit]

    for i, (word, phonemes_with_stress) in enumerate(word_list):
        if i % 1000 == 0:
            print(f"Processing word {i}/{len(word_list)} (included: {len(words_data)}, filtered: {filtered_count})...")

        # FILTER: Apply vocabulary selection criterion
        if not word_filter.should_include_word(word):
            filtered_count += 1
            continue

        # Get syllables
        syllables = syllabify(phonemes_with_stress)

        # Extract plain phonemes (without stress) for embedding computation
        phonemes = [p.phoneme for p in phonemes_with_stress]
        if not syllables:
            continue

        # Convert to IPA string
        ipa = ' '.join(phonemes)

        # Prepare phonemes JSON
        phonemes_json = json.dumps([
            {"ipa": p, "position": pos}
            for pos, p in enumerate(phonemes)
        ])

        # Prepare syllables JSON
        syllables_json = json.dumps([
            {
                "onset": syl.onset,
                "nucleus": syl.nucleus,
                "coda": syl.coda
            }
            for syl in syllables
        ])

        # Compute counts
        phoneme_count = len(phonemes)
        syllable_count = len(syllables)

        # Compute WCM score
        wcm_score = compute_wcm_score(phonemes, syllables)

        # Categorize
        word_length = categorize_word_length(phoneme_count)
        complexity = categorize_complexity(wcm_score)

        # Compute syllable embedding
        syllable_embedding = compute_syllable_embedding(phonemes, model, phoneme_to_id)

        # For now, we'll skip psycholinguistic properties (can add later)
        # These would come from external datasets like SUBTLEXus, Age of Acquisition norms, etc.

        # Prepare tuple for insertion
        word_tuple = (
            word,
            ipa,
            phonemes_json,
            syllables_json,
            phoneme_count,
            syllable_count,
            None,  # frequency (TODO: add from SUBTLEXus)
            None,  # log_frequency
            None,  # aoa
            None,  # imageability
            None,  # familiarity
            None,  # concreteness
            None,  # valence
            None,  # arousal
            None,  # dominance
            wcm_score,
            None,  # msh_stage (TODO: compute)
            syllable_embedding.tolist() if syllable_embedding is not None else None,
            None,  # word_embedding_flat (optional)
            word_length,
            complexity
        )

        words_data.append(word_tuple)

    print(f"✓ Prepared {len(words_data)} words for insertion")
    print(f"  Filtered out: {filtered_count} words (no psycholinguistic norms)")
    print(f"  Reduction: {100*filtered_count/(len(words_data)+filtered_count):.1f}%")
    return words_data


def insert_words(conn, words_data, batch_size=1000):
    """
    Insert words into database in batches.

    Args:
        conn: psycopg2 connection
        words_data: List of word tuples
        batch_size: Number of rows per batch
    """
    cursor = conn.cursor()

    insert_query = """
        INSERT INTO words (
            word, ipa, phonemes_json, syllables_json,
            phoneme_count, syllable_count,
            frequency, log_frequency, aoa, imageability, familiarity, concreteness,
            valence, arousal, dominance,
            wcm_score, msh_stage,
            syllable_embedding, word_embedding_flat,
            word_length, complexity
        ) VALUES (
            %s, %s, %s, %s,
            %s, %s,
            %s, %s, %s, %s, %s, %s,
            %s, %s, %s,
            %s, %s,
            %s, %s,
            %s, %s
        )
        ON CONFLICT (word) DO NOTHING;
    """

    print(f"Inserting {len(words_data)} words in batches of {batch_size}...")
    execute_batch(cursor, insert_query, words_data, page_size=batch_size)
    conn.commit()

    # Get count
    cursor.execute("SELECT COUNT(*) FROM words;")
    count = cursor.fetchone()[0]
    print(f"✓ Inserted words. Total in database: {count}")

    cursor.close()


def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(description="Populate words table")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of words (for testing)")
    parser.add_argument("--model", type=str, default="models/hierarchical/final.pt", help="Path to model")
    args = parser.parse_args()

    print("=" * 80)
    print("PhonoLex v2.0 - Words Table Population")
    print("=" * 80)

    # Load data
    print("\n[1/4] Loading CMU dictionary...")
    loader = EnglishPhonologyLoader()
    print(f"✓ Loaded {len(loader.lexicon)} words")

    # Load model
    print("\n[2/4] Loading hierarchical model...")
    model, phoneme_to_id = load_hierarchical_model(args.model)

    # Load word filter
    print("\n[3/5] Loading word filter (psycholinguistic norms)...")
    word_filter = WordFilter()
    word_filter.load_all_norms()

    # Prepare data
    print("\n[4/5] Preparing word data...")
    words_data = prepare_word_data(loader, model, phoneme_to_id, word_filter, limit=args.limit)

    # Insert into database
    print("\n[5/5] Inserting into database...")
    conn = psycopg2.connect(DATABASE_URL)
    try:
        insert_words(conn, words_data)
        print("\n" + "=" * 80)
        print("✓ SUCCESS: Words table populated")
        print("=" * 80)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
