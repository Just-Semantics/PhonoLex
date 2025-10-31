"""
Populate syllables table from Layer 4 checkpoint.

Uses the pre-computed syllable embeddings from embeddings/layer4/syllable_embeddings.pt
and creates a lookup table of unique syllables.
"""

import sys
import os
import torch
import json
import psycopg2
from psycopg2.extras import execute_batch
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import DatabaseService


def load_layer4_embeddings(checkpoint_path):
    """Load Layer 4 syllable embeddings from checkpoint file."""
    print(f"Loading Layer 4 checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    word_to_syllable_embeddings = checkpoint['word_to_syllable_embeddings']
    print(f"✓ Loaded embeddings for {len(word_to_syllable_embeddings):,} words")
    return word_to_syllable_embeddings


def extract_unique_syllables(db):
    """
    Extract unique syllables from words table.

    Returns:
        dict: syllable_key -> {structure, ipa, onset_count, coda_count, total_phonemes, word_count}
    """
    print("\nExtracting unique syllables from database...")

    with db.get_session() as session:
        from models import Word
        words = session.query(Word).all()

        syllable_data = {}  # syllable_key -> data
        syllable_words = defaultdict(list)  # syllable_key -> [word_ids]

        for word in words:
            if not word.syllables_json:
                continue

            for syll_struct in word.syllables_json:
                # Create unique key from structure
                onset = tuple(syll_struct.get('onset', []))
                nucleus = syll_struct.get('nucleus', '')
                coda = tuple(syll_struct.get('coda', []))

                # IPA representation
                onset_str = ''.join(onset)
                coda_str = ''.join(coda)
                ipa = onset_str + nucleus + coda_str

                syllable_key = (onset, nucleus, coda)

                if syllable_key not in syllable_data:
                    syllable_data[syllable_key] = {
                        'structure': syll_struct,
                        'ipa': ipa,
                        'onset_count': len(onset),
                        'coda_count': len(coda),
                        'total_phonemes': len(onset) + 1 + len(coda),
                    }

                syllable_words[syllable_key].append(word.word_id)

        # Add frequency counts
        for key in syllable_data:
            syllable_data[key]['frequency'] = len(syllable_words[key])

        print(f"✓ Found {len(syllable_data):,} unique syllables")
        print(f"  Most common syllables:")
        sorted_sylls = sorted(syllable_data.items(), key=lambda x: x[1]['frequency'], reverse=True)
        for (key, data) in sorted_sylls[:10]:
            print(f"    {data['ipa']}: {data['frequency']} words")

        return syllable_data


def get_syllable_embedding(word, syllable_index, word_to_syllable_embeddings):
    """Get embedding for a specific syllable from Layer 4 checkpoint."""
    if word not in word_to_syllable_embeddings:
        return None

    syllables = word_to_syllable_embeddings[word]
    if syllable_index >= len(syllables):
        return None

    return syllables[syllable_index]


def populate_syllables_table(db, syllable_data, word_to_syllable_embeddings):
    """
    Populate syllables table with unique syllables and their embeddings.

    Strategy:
    1. For each unique syllable structure, find ANY word that contains it
    2. Extract the EXACT embedding from Layer 4 checkpoint for that syllable position
    3. Store in syllables table

    CRITICAL: We must use the exact embeddings from the checkpoint, not regenerate them.
    """
    print("\nPopulating syllables table...")

    # Get database connection
    conn = psycopg2.connect(os.getenv("DATABASE_URL", "postgresql://localhost/phonolex"))
    cursor = conn.cursor()

    # Build mapping from syllable structure to (word, syllable_index) from checkpoint
    print("Finding source words for syllables from checkpoint...")
    syllable_sources = {}  # syllable_key -> (word, syllable_index)

    # Iterate through checkpoint to ensure we're using the right word/index pairs
    for word, syll_embs in word_to_syllable_embeddings.items():
        # Get syllables_json for this word from database
        with db.get_session() as session:
            from models import Word as WordModel
            word_obj = session.query(WordModel).filter(WordModel.word == word).first()
            if not word_obj or not word_obj.syllables_json:
                continue

            syllables_json = word_obj.syllables_json

            # Match checkpoint syllables to database syllables
            for syll_idx, syll_struct in enumerate(syllables_json):
                if syll_idx >= len(syll_embs):
                    continue  # Skip if mismatch

                onset = tuple(syll_struct.get('onset', []))
                nucleus = syll_struct.get('nucleus', '')
                coda = tuple(syll_struct.get('coda', []))
                syllable_key = (onset, nucleus, coda)

                # Only store first occurrence of each unique syllable
                if syllable_key not in syllable_sources:
                    syllable_sources[syllable_key] = (word, syll_idx)

    print(f"✓ Found source words for {len(syllable_sources):,} syllables")

    # Insert syllables with embeddings
    insert_query = """
        INSERT INTO syllables (ipa, structure, onset_count, coda_count, total_phonemes, embedding, frequency)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING syllable_id
    """

    syllable_id_map = {}  # syllable_key -> syllable_id
    successful = 0
    failed = 0

    for syllable_key, data in syllable_data.items():
        if syllable_key not in syllable_sources:
            print(f"  WARNING: No source word for syllable {data['ipa']}")
            failed += 1
            continue

        source_word, syll_idx = syllable_sources[syllable_key]
        embedding = get_syllable_embedding(source_word, syll_idx, word_to_syllable_embeddings)

        if embedding is None:
            print(f"  WARNING: No embedding for syllable {data['ipa']} from {source_word}")
            failed += 1
            continue

        # Convert to list for PostgreSQL
        embedding_list = embedding.tolist()

        try:
            cursor.execute(insert_query, (
                data['ipa'],
                json.dumps(data['structure']),
                data['onset_count'],
                data['coda_count'],
                data['total_phonemes'],
                embedding_list,
                data['frequency']
            ))
            syllable_id = cursor.fetchone()[0]
            syllable_id_map[syllable_key] = syllable_id
            successful += 1

            if successful % 100 == 0:
                print(f"  Inserted {successful} syllables...")

        except Exception as e:
            print(f"  ERROR inserting syllable {data['ipa']}: {e}")
            failed += 1
            continue

    conn.commit()
    cursor.close()
    conn.close()

    print(f"\n✓ Populated syllables table")
    print(f"  Successful: {successful:,}")
    print(f"  Failed: {failed:,}")

    return syllable_id_map


def populate_word_syllables_junction(db, syllable_id_map):
    """
    Populate word_syllables junction table.
    """
    print("\nPopulating word_syllables junction table...")

    conn = psycopg2.connect(os.getenv("DATABASE_URL", "postgresql://localhost/phonolex"))
    cursor = conn.cursor()

    with db.get_session() as session:
        from models import Word
        words = session.query(Word).all()

        insert_batch = []

        for word in words:
            if not word.syllables_json:
                continue

            for position, syll_struct in enumerate(word.syllables_json):
                onset = tuple(syll_struct.get('onset', []))
                nucleus = syll_struct.get('nucleus', '')
                coda = tuple(syll_struct.get('coda', []))
                syllable_key = (onset, nucleus, coda)

                if syllable_key not in syllable_id_map:
                    continue

                syllable_id = syllable_id_map[syllable_key]
                insert_batch.append((word.word_id, syllable_id, position))

        # Batch insert
        insert_query = """
            INSERT INTO word_syllables (word_id, syllable_id, position)
            VALUES (%s, %s, %s)
        """
        execute_batch(cursor, insert_query, insert_batch, page_size=1000)
        conn.commit()

    cursor.close()
    conn.close()

    print(f"✓ Populated word_syllables junction table")
    print(f"  Total mappings: {len(insert_batch):,}")


def main():
    # Paths (absolute path from repo root)
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    checkpoint_path = os.path.join(repo_root, 'embeddings', 'layer4', 'syllable_embeddings.pt')

    # Load Layer 4 embeddings
    word_to_syllable_embeddings = load_layer4_embeddings(checkpoint_path)

    # Initialize database
    db = DatabaseService()

    # Extract unique syllables
    syllable_data = extract_unique_syllables(db)

    # Populate syllables table
    syllable_id_map = populate_syllables_table(db, syllable_data, word_to_syllable_embeddings)

    # Populate junction table
    populate_word_syllables_junction(db, syllable_id_map)

    print("\n✅ Syllables population complete!")


if __name__ == '__main__':
    main()
