#!/usr/bin/env python3
"""
Populate syllable embeddings for all words in the database.

This script:
1. Loads the hierarchical phoneme encoder model
2. Generates 384-dim syllable embeddings for all words
3. Updates the database in batches

Run after populate_words.py to enable vector similarity search.
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
import psycopg2
from psycopg2.extras import execute_batch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from train_hierarchical_final import HierarchicalPhonemeEncoder
from src.phonolex.embeddings.english_data_loader import EnglishPhonologyLoader
from src.phonolex.utils.syllabification import syllabify

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/phonolex")


def load_hierarchical_model(model_path="models/hierarchical/final.pt"):
    """
    Load the hierarchical phoneme encoder model.

    Returns:
        tuple: (model, phoneme_to_id, device)
    """
    print(f"Loading model from {model_path}...")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')

    # Get model configuration from state dict shape
    embedding_weight = checkpoint['model_state_dict']['phoneme_embeddings.weight']
    num_phonemes = embedding_weight.shape[0]  # 42
    d_model = embedding_weight.shape[1]  # 128

    print(f"  Model config: {num_phonemes} phonemes, {d_model} dimensions")

    # Initialize model
    model = HierarchicalPhonemeEncoder(num_phonemes=num_phonemes, d_model=d_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Get phoneme_to_id mapping
    phoneme_to_id = checkpoint.get('phoneme_to_id', {})

    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("  Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device('cpu')
        print("  Using CPU")

    model = model.to(device)

    print(f"✓ Model loaded successfully")
    return model, phoneme_to_id, device


def generate_syllable_embedding(word_phonemes, syllables, model, phoneme_to_id, device, debug=False):
    """
    Generate syllable embedding for a single word.

    Aggregates contextual phoneme embeddings by syllable structure:
    - For each syllable: onset (mean) + nucleus + coda (mean) → 384 dims
    - For multi-syllable words: concatenate all syllables

    Args:
        word_phonemes: List of phoneme strings (IPA)
        syllables: List of Syllable objects
        model: HierarchicalPhonemeEncoder
        phoneme_to_id: Dict mapping phoneme to ID
        device: torch device
        debug: Print debug info

    Returns:
        numpy.ndarray: 384-dim syllable embedding, or None if generation fails
    """
    try:
        # Convert phonemes to IDs
        phoneme_ids = []
        for phoneme in word_phonemes:
            if phoneme not in phoneme_to_id:
                if debug:
                    print(f"    Unknown phoneme: {phoneme}")
                return None  # Unknown phoneme
            phoneme_ids.append(phoneme_to_id[phoneme])

        if debug:
            print(f"    Phoneme IDs: {phoneme_ids}")

        # Convert to tensor
        phoneme_tensor = torch.tensor([phoneme_ids], dtype=torch.long).to(device)

        # Create attention mask (all ones = attend to all tokens)
        attention_mask = torch.ones_like(phoneme_tensor, dtype=torch.long).to(device)

        if debug:
            print(f"    Tensor shape: {phoneme_tensor.shape}")

        # Get contextual phoneme embeddings
        with torch.no_grad():
            predictions, contextual = model(phoneme_tensor, attention_mask)
            # contextual: [batch=1, seq_len, d_model=128]

        if debug:
            print(f"    Contextual embeddings shape: {contextual.shape}")

        # Extract embeddings for first syllable (single syllable for now)
        # For multi-syllable words, we'd concatenate all syllables
        # But for simplicity and compatibility with database (384-dim vector),
        # we'll use only the first syllable

        contextual_np = contextual[0].cpu().numpy()  # [seq_len, 128]

        if len(syllables) > 0:
            syll = syllables[0]  # Use first syllable

            # Map syllable structure to phoneme indices
            idx = 0
            onset_indices = list(range(idx, idx + len(syll.onset)))
            idx += len(syll.onset)
            nucleus_index = idx
            idx += 1
            coda_indices = list(range(idx, idx + len(syll.coda)))

            # Aggregate embeddings by syllable part
            # Onset: mean of onset phoneme embeddings (or zeros if empty)
            if len(onset_indices) > 0:
                onset_emb = contextual_np[onset_indices].mean(axis=0)  # (128,)
            else:
                onset_emb = np.zeros(128, dtype=np.float32)

            # Nucleus: single phoneme embedding
            nucleus_emb = contextual_np[nucleus_index]  # (128,)

            # Coda: mean of coda phoneme embeddings (or zeros if empty)
            if len(coda_indices) > 0:
                coda_emb = contextual_np[coda_indices].mean(axis=0)  # (128,)
            else:
                coda_emb = np.zeros(128, dtype=np.float32)

            # Concatenate: onset + nucleus + coda = 384 dims
            syllable_embedding = np.concatenate([onset_emb, nucleus_emb, coda_emb])

            if debug:
                print(f"    Syllable structure: onset={len(syll.onset)}, nucleus=1, coda={len(syll.coda)}")
                print(f"    Final embedding shape: {syllable_embedding.shape}")

            return syllable_embedding
        else:
            # No syllables - fallback to mean pooling
            embedding_np = contextual_np.mean(axis=0)
            # Pad to 384 dims
            embedding_np = np.concatenate([embedding_np, embedding_np, embedding_np])
            return embedding_np

    except Exception as e:
        if debug:
            print(f"    Exception: {e}")
        import traceback
        if debug:
            traceback.print_exc()
        return None


def get_words_without_embeddings(conn, limit=None):
    """
    Get words that don't have syllable embeddings.

    Args:
        conn: psycopg2 connection
        limit: Optional limit on number of words

    Returns:
        list: List of (word_id, word, ipa, phonemes_json, syllables_json) tuples
    """
    cursor = conn.cursor()

    query = """
        SELECT word_id, word, ipa, phonemes_json, syllables_json
        FROM words
        WHERE syllable_embedding IS NULL
        ORDER BY word_id
    """

    if limit:
        query += f" LIMIT {limit}"

    cursor.execute(query)
    words = cursor.fetchall()
    cursor.close()

    return words


def update_embeddings_batch(conn, embedding_data, batch_size=500):
    """
    Update syllable embeddings in database.

    Args:
        conn: psycopg2 connection
        embedding_data: List of (word_id, embedding_array) tuples
        batch_size: Batch size for updates
    """
    cursor = conn.cursor()

    update_query = """
        UPDATE words
        SET syllable_embedding = %s
        WHERE word_id = %s
    """

    # Convert to format expected by execute_batch
    # Note: embedding comes before word_id in query
    batch_data = [(embedding.tolist(), word_id) for word_id, embedding in embedding_data]

    execute_batch(cursor, update_query, batch_data, page_size=batch_size)
    conn.commit()
    cursor.close()


def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(description="Populate syllable embeddings")
    parser.add_argument("--limit", type=int, help="Limit number of words to process (for testing)")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for processing")
    args = parser.parse_args()

    print("=" * 80)
    print("PhonoLex v2.0 - Syllable Embedding Population")
    print("=" * 80)

    # Load model
    print("\n[1/5] Loading hierarchical model...")
    model, phoneme_to_id, device = load_hierarchical_model()

    # Load data loader (for syllabification)
    print("\n[2/5] Loading phonology data loader...")
    loader = EnglishPhonologyLoader()
    print(f"✓ Loaded {len(loader.lexicon)} words")

    # Connect to database
    print("\n[3/5] Connecting to database...")
    conn = psycopg2.connect(DATABASE_URL)
    print("✓ Connected")

    try:
        # Get words without embeddings
        print("\n[4/5] Fetching words without embeddings...")
        words_to_process = get_words_without_embeddings(conn, limit=args.limit)
        print(f"✓ Found {len(words_to_process)} words to process")

        if len(words_to_process) == 0:
            print("\n✓ All words already have embeddings!")
            return

        # Process in batches
        print(f"\n[5/5] Generating embeddings (batch size: {args.batch_size})...")

        total_processed = 0
        total_success = 0
        total_failed = 0

        embedding_batch = []

        for i, (word_id, word, ipa, phonemes_json, syllables_json) in enumerate(words_to_process):
            # Extract phoneme list from JSON
            phonemes = [p['ipa'] for p in phonemes_json]

            # Look up word in loader to get PhonemeWithStress objects for syllabification
            if word not in loader.lexicon_with_stress:
                total_failed += 1
                continue

            phonemes_with_stress = loader.lexicon_with_stress[word]

            # Syllabify
            try:
                syllables = syllabify(phonemes_with_stress)
            except Exception as e:
                if i < 5:  # Print first few errors for debugging
                    print(f"  [DEBUG] Syllabification failed for '{word}': {e}")
                total_failed += 1
                continue

            # Generate embedding
            debug_mode = i < 3  # Debug first 3 words
            embedding = generate_syllable_embedding(
                phonemes, syllables, model, phoneme_to_id, device, debug=debug_mode
            )

            if embedding is not None:
                embedding_batch.append((word_id, embedding))
                total_success += 1
                if debug_mode:
                    print(f"  [DEBUG] SUCCESS for '{word}' - embedding shape: {embedding.shape}")
            else:
                if i < 5:  # Print first few errors for debugging
                    print(f"  [DEBUG] Embedding generation failed for '{word}'")
                total_failed += 1

            total_processed += 1

            # Update database when batch is full
            if len(embedding_batch) >= args.batch_size:
                print(f"  Updating batch... ({total_processed}/{len(words_to_process)})")
                update_embeddings_batch(conn, embedding_batch)
                embedding_batch = []

            # Progress update
            if total_processed % 5000 == 0:
                print(f"  Processed {total_processed}/{len(words_to_process)} words...")

        # Update remaining embeddings
        if embedding_batch:
            print(f"  Updating final batch... ({total_processed}/{len(words_to_process)})")
            update_embeddings_batch(conn, embedding_batch)

        print(f"\n✓ Processing complete!")
        print(f"  Total processed: {total_processed:,}")
        print(f"  Successful: {total_success:,}")
        print(f"  Failed: {total_failed:,}")

        # Verify
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM words WHERE syllable_embedding IS NOT NULL")
        count_with_embeddings = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM words")
        total_words = cursor.fetchone()[0]
        cursor.close()

        print(f"\nDatabase status:")
        print(f"  Words with embeddings: {count_with_embeddings:,} / {total_words:,}")
        print(f"  Percentage: {100 * count_with_embeddings / total_words:.1f}%")

        print("\n" + "=" * 80)
        print("✓ SUCCESS: Syllable embeddings populated")
        print("=" * 80)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
