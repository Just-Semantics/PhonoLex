#!/usr/bin/env python3
"""
Populate typed phonological graph edges using direct algorithms.

Simpler approach that doesn't require the PhonologicalGraph class.

Edge Types:
- MINIMAL_PAIR: Words differing by exactly 1 phoneme in same position
- RHYME: Words with matching final syllable
- NEIGHBOR: Words with edit distance ≤ 2

Run after populate_words.py and populate_embeddings.py.
"""

import sys
import os
from pathlib import Path
import psycopg2
from psycopg2.extras import execute_batch
import json

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/phonolex")


def levenshtein_distance(seq1, seq2):
    """Compute edit distance between two sequences."""
    if len(seq1) == 0:
        return len(seq2)
    if len(seq2) == 0:
        return len(seq1)

    # Create distance matrix
    d = [[0] * (len(seq2) + 1) for _ in range(len(seq1) + 1)]

    for i in range(len(seq1) + 1):
        d[i][0] = i
    for j in range(len(seq2) + 1):
        d[0][j] = j

    for i in range(1, len(seq1) + 1):
        for j in range(1, len(seq2) + 1):
            cost = 0 if seq1[i-1] == seq2[j-1] else 1
            d[i][j] = min(
                d[i-1][j] + 1,      # deletion
                d[i][j-1] + 1,      # insertion
                d[i-1][j-1] + cost  # substitution
            )

    return d[len(seq1)][len(seq2)]


def get_database_words(conn):
    """
    Get all words from database.

    Returns:
        list: [(word_id, word, phonemes, syllables), ...]
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT word_id, word, phonemes_json, syllables_json
        FROM words
        ORDER BY word_id
    """)

    words = []
    for word_id, word, phonemes_json, syllables_json in cursor.fetchall():
        phonemes = [p['ipa'] for p in phonemes_json]
        words.append((word_id, word, phonemes, syllables_json))

    cursor.close()
    return words


def find_minimal_pairs_for_word(word_data, all_words):
    """
    Find minimal pairs for a word.

    A minimal pair differs by exactly 1 phoneme in the same position.
    """
    word_id, word, phonemes, syllables = word_data
    pairs = []

    for other_id, other_word, other_phonemes, other_syllables in all_words:
        if other_id <= word_id:  # Avoid duplicates
            continue

        # Must have same length
        if len(phonemes) != len(other_phonemes):
            continue

        # Count differences
        diffs = 0
        diff_pos = -1
        for i, (p1, p2) in enumerate(zip(phonemes, other_phonemes)):
            if p1 != p2:
                diffs += 1
                diff_pos = i

        # Exactly 1 difference = minimal pair
        if diffs == 1:
            metadata = {
                'position': diff_pos,
                'phoneme1': phonemes[diff_pos],
                'phoneme2': other_phonemes[diff_pos]
            }
            pairs.append((other_id, metadata))

    return pairs


def find_rhymes_for_word(word_data, all_words):
    """
    Find rhymes for a word.

    Words rhyme if they share the final syllable structure.
    """
    word_id, word, phonemes, syllables = word_data
    rhymes = []

    if not syllables or len(syllables) == 0:
        return rhymes

    # Get last syllable
    last_syll = syllables[-1]
    last_nucleus = last_syll.get('nucleus', '')
    last_coda = tuple(last_syll.get('coda', []))

    for other_id, other_word, other_phonemes, other_syllables in all_words:
        if other_id <= word_id:  # Avoid duplicates
            continue

        if not other_syllables or len(other_syllables) == 0:
            continue

        # Get last syllable of other word
        other_last_syll = other_syllables[-1]
        other_nucleus = other_last_syll.get('nucleus', '')
        other_coda = tuple(other_last_syll.get('coda', []))

        # Check if nucleus and coda match
        if last_nucleus == other_nucleus and last_coda == other_coda:
            metadata = {
                'rhyme_type': 'last_syllable',
                'nucleus': last_nucleus,
                'coda': list(last_coda),
                'quality': 1.0  # perfect rhyme
            }
            rhymes.append((other_id, metadata))

    return rhymes


def find_neighbors_for_word(word_data, all_words, max_distance=2):
    """
    Find phonological neighbors.

    Neighbors have edit distance ≤ max_distance.
    """
    word_id, word, phonemes, syllables = word_data
    neighbors = []

    for other_id, other_word, other_phonemes, other_syllables in all_words:
        if other_id <= word_id:  # Avoid duplicates
            continue

        # Calculate edit distance
        dist = levenshtein_distance(phonemes, other_phonemes)

        if 0 < dist <= max_distance:
            metadata = {
                'edit_distance': dist,
                'phoneme_diff': dist
            }
            neighbors.append((other_id, metadata))

    return neighbors


def populate_edges(conn, batch_size=1000):
    """
    Populate all typed edges.
    """
    print("\n[1/4] Fetching words from database...")
    all_words = get_database_words(conn)
    print(f"✓ Loaded {len(all_words):,} words")

    # Delete existing typed edges
    print("\n[2/4] Clearing existing typed edges...")
    cursor = conn.cursor()
    cursor.execute("""
        DELETE FROM word_edges
        WHERE relation_type IN ('MINIMAL_PAIR', 'RHYME', 'NEIGHBOR')
    """)
    deleted = cursor.rowcount
    conn.commit()
    cursor.close()
    print(f"✓ Deleted {deleted:,} existing typed edges")

    # Collect all edges
    print("\n[3/4] Generating edges...")
    all_edges = []

    for idx, word_data in enumerate(all_words):
        if (idx + 1) % 500 == 0:
            print(f"  Progress: {idx + 1:,}/{len(all_words):,} words ({100*(idx+1)/len(all_words):.1f}%)")

        word_id = word_data[0]

        # Find minimal pairs
        pairs = find_minimal_pairs_for_word(word_data, all_words)
        for other_id, metadata in pairs:
            all_edges.append((word_id, other_id, 'MINIMAL_PAIR', json.dumps(metadata), 1.0))

        # Find rhymes
        rhymes = find_rhymes_for_word(word_data, all_words)
        for other_id, metadata in rhymes:
            all_edges.append((word_id, other_id, 'RHYME', json.dumps(metadata), 1.0))

        # Find neighbors (limit to save time)
        if len(all_words) < 10000:  # Only for smaller datasets
            neighbors = find_neighbors_for_word(word_data, all_words, max_distance=2)
            for other_id, metadata in neighbors:
                weight = 1.0 / (metadata['edit_distance'] + 1)
                all_edges.append((word_id, other_id, 'NEIGHBOR', json.dumps(metadata), weight))

    print(f"✓ Generated {len(all_edges):,} edges")

    # Insert edges
    print(f"\n[4/4] Inserting edges in batches of {batch_size}...")
    cursor = conn.cursor()

    insert_query = """
        INSERT INTO word_edges (word1_id, word2_id, relation_type, edge_metadata, weight)
        VALUES (%s, %s, %s, %s::jsonb, %s)
    """

    execute_batch(cursor, insert_query, all_edges, page_size=batch_size)
    conn.commit()
    cursor.close()

    print(f"✓ Inserted {len(all_edges):,} edges")


def main():
    """Main execution"""
    print("=" * 80)
    print("PhonoLex v2.0 - Typed Edge Population (Simple)")
    print("=" * 80)

    # Connect to database
    print("\nConnecting to database...")
    conn = psycopg2.connect(DATABASE_URL)
    print("✓ Connected")

    try:
        # Populate edges
        populate_edges(conn)

        # Get final statistics
        cursor = conn.cursor()
        cursor.execute("""
            SELECT relation_type, COUNT(*) as count
            FROM word_edges
            GROUP BY relation_type
            ORDER BY count DESC
        """)

        print("\n" + "=" * 80)
        print("Edge Statistics:")
        print("=" * 80)

        for relation_type, count in cursor.fetchall():
            print(f"  {relation_type:20s}: {count:>10,} edges")

        cursor.execute("SELECT COUNT(*) FROM word_edges")
        total_edges = cursor.fetchone()[0]
        print(f"  {'TOTAL':20s}: {total_edges:>10,} edges")

        cursor.close()

        print("\n" + "=" * 80)
        print("✓ SUCCESS: Typed edges populated")
        print("=" * 80)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
