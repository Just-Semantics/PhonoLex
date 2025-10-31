#!/usr/bin/env python3
"""
Populate typed phonological graph edges in the database.

Edge Types:
- MINIMAL_PAIR: Words differing by exactly 1 phoneme
- RHYME: Words sharing final syllable structure
- NEIGHBOR: Phonological neighbors (edit distance ≤ 2)
- SIMILAR: High embedding similarity (cosine > 0.85)
- MAXIMAL_OPP: Maximal phonological oppositions

Run after populate_words.py and populate_embeddings.py.
"""

import sys
import os
from pathlib import Path
import psycopg2
from psycopg2.extras import execute_batch
import pickle

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.phonolex.build_phonological_graph import PhonologicalGraph

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/phonolex")


def get_database_words(conn):
    """
    Get all words from database with their IPA and phonemes.

    Returns:
        dict: word_id -> (word, ipa, phonemes)
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT word_id, word, ipa, phonemes_json
        FROM words
        ORDER BY word_id
    """)

    words_by_id = {}
    words_by_word = {}

    for word_id, word, ipa, phonemes_json in cursor.fetchall():
        phonemes = [p['ipa'] for p in phonemes_json]
        words_by_id[word_id] = (word, ipa, phonemes)
        words_by_word[word] = word_id

    cursor.close()
    return words_by_id, words_by_word


def clear_existing_typed_edges(conn):
    """Delete existing typed edges (keep SIMILAR edges for now)."""
    cursor = conn.cursor()

    print("\nClearing existing typed edges...")
    cursor.execute("""
        DELETE FROM word_edges
        WHERE relation_type IN ('MINIMAL_PAIR', 'RHYME', 'NEIGHBOR', 'MAXIMAL_OPP')
    """)
    deleted = cursor.rowcount
    conn.commit()
    cursor.close()

    print(f"✓ Deleted {deleted:,} existing typed edges")


def populate_minimal_pairs(graph, conn, words_by_id, words_by_word, batch_size=1000):
    """
    Populate MINIMAL_PAIR edges.

    Minimal pairs: words differing by exactly 1 phoneme.
    """
    print("\n[1/3] Generating MINIMAL_PAIR edges...")

    edges = []
    total_pairs = 0
    total_words = len(words_by_id)

    for idx, (word_id, (word, ipa, phonemes)) in enumerate(words_by_id.items()):
        if (idx + 1) % 1000 == 0:
            print(f"  Progress: {idx + 1:,}/{total_words:,} words ({100*(idx+1)/total_words:.1f}%)")
        try:
            pairs = graph.find_minimal_pairs(word, max_results=50)

            for pair_word, metadata in pairs:
                if pair_word in words_by_word:
                    pair_id = words_by_word[pair_word]

                    # Only add edge if pair_id > word_id to avoid duplicates
                    if pair_id > word_id:
                        edges.append((
                            word_id,
                            pair_id,
                            'MINIMAL_PAIR',
                            metadata,
                            1.0  # weight
                        ))
                        total_pairs += 1
        except Exception as e:
            # Skip words not in graph
            continue

    print(f"  Found {total_pairs:,} minimal pair relationships")

    # Insert edges
    if edges:
        print(f"  Inserting {len(edges):,} edges in batches of {batch_size}...")
        cursor = conn.cursor()

        insert_query = """
            INSERT INTO word_edges (word1_id, word2_id, relation_type, edge_metadata, weight)
            VALUES (%s, %s, %s, %s::jsonb, %s)
        """

        # Convert metadata to JSON strings
        edge_data = [
            (w1, w2, rel, str(meta).replace("'", '"'), weight)
            for w1, w2, rel, meta, weight in edges
        ]

        execute_batch(cursor, insert_query, edge_data, page_size=batch_size)
        conn.commit()
        cursor.close()

        print(f"✓ Inserted {len(edges):,} MINIMAL_PAIR edges")


def populate_rhymes(graph, conn, words_by_id, words_by_word, batch_size=1000):
    """
    Populate RHYME edges.

    Rhymes: words sharing final syllable structure.
    """
    print("\n[2/3] Generating RHYME edges...")

    edges = []
    total_rhymes = 0
    total_words = len(words_by_id)

    for idx, (word_id, (word, ipa, phonemes)) in enumerate(words_by_id.items()):
        if (idx + 1) % 1000 == 0:
            print(f"  Progress: {idx + 1:,}/{total_words:,} words ({100*(idx+1)/total_words:.1f}%)")
        try:
            rhymes = graph.find_rhymes(word, rhyme_type='last_syllable', max_results=50)

            for rhyme_word, metadata in rhymes:
                if rhyme_word in words_by_word:
                    rhyme_id = words_by_word[rhyme_word]

                    # Only add edge if rhyme_id > word_id to avoid duplicates
                    if rhyme_id > word_id:
                        edges.append((
                            word_id,
                            rhyme_id,
                            'RHYME',
                            metadata,
                            metadata.get('quality', 1.0)  # use rhyme quality as weight
                        ))
                        total_rhymes += 1
        except Exception as e:
            # Skip words not in graph
            continue

    print(f"  Found {total_rhymes:,} rhyme relationships")

    # Insert edges
    if edges:
        print(f"  Inserting {len(edges):,} edges in batches of {batch_size}...")
        cursor = conn.cursor()

        insert_query = """
            INSERT INTO word_edges (word1_id, word2_id, relation_type, edge_metadata, weight)
            VALUES (%s, %s, %s, %s::jsonb, %s)
        """

        # Convert metadata to JSON strings
        edge_data = [
            (w1, w2, rel, str(meta).replace("'", '"'), weight)
            for w1, w2, rel, meta, weight in edges
        ]

        execute_batch(cursor, insert_query, edge_data, page_size=batch_size)
        conn.commit()
        cursor.close()

        print(f"✓ Inserted {len(edges):,} RHYME edges")


def populate_neighbors(graph, conn, words_by_id, words_by_word, batch_size=1000):
    """
    Populate NEIGHBOR edges.

    Neighbors: words with edit distance ≤ 2.
    """
    print("\n[3/3] Generating NEIGHBOR edges...")

    edges = []
    total_neighbors = 0
    total_words = len(words_by_id)

    for idx, (word_id, (word, ipa, phonemes)) in enumerate(words_by_id.items()):
        if (idx + 1) % 1000 == 0:
            print(f"  Progress: {idx + 1:,}/{total_words:,} words ({100*(idx+1)/total_words:.1f}%)")
        try:
            neighbors = graph.find_phoneme_neighbors(word, max_edit_distance=2, max_results=50)

            for neighbor_word, metadata in neighbors:
                if neighbor_word in words_by_word:
                    neighbor_id = words_by_word[neighbor_word]

                    # Only add edge if neighbor_id > word_id to avoid duplicates
                    if neighbor_id > word_id:
                        edit_dist = metadata['edit_distance']
                        # Weight inversely proportional to edit distance
                        weight = 1.0 / (edit_dist + 1)

                        edges.append((
                            word_id,
                            neighbor_id,
                            'NEIGHBOR',
                            metadata,
                            weight
                        ))
                        total_neighbors += 1
        except Exception as e:
            # Skip words not in graph
            continue

    print(f"  Found {total_neighbors:,} neighbor relationships")

    # Insert edges
    if edges:
        print(f"  Inserting {len(edges):,} edges in batches of {batch_size}...")
        cursor = conn.cursor()

        insert_query = """
            INSERT INTO word_edges (word1_id, word2_id, relation_type, edge_metadata, weight)
            VALUES (%s, %s, %s, %s::jsonb, %s)
        """

        # Convert metadata to JSON strings
        edge_data = [
            (w1, w2, rel, str(meta).replace("'", '"'), weight)
            for w1, w2, rel, meta, weight in edges
        ]

        execute_batch(cursor, insert_query, edge_data, page_size=batch_size)
        conn.commit()
        cursor.close()

        print(f"✓ Inserted {len(edges):,} NEIGHBOR edges")


def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(description="Populate typed phonological edges")
    parser.add_argument("--graph-path", default="data/phonological_graph.pkl",
                       help="Path to phonological graph pickle file")
    parser.add_argument("--batch-size", type=int, default=1000,
                       help="Batch size for database inserts")
    args = parser.parse_args()

    print("=" * 80)
    print("PhonoLex v2.0 - Typed Edge Population")
    print("=" * 80)

    # Load phonological graph
    print(f"\n[1/5] Loading phonological graph from {args.graph_path}...")
    graph_path = Path(__file__).parent.parent.parent.parent / args.graph_path

    if not graph_path.exists():
        print(f"  Error: Graph file not found at {graph_path}")
        print(f"  Please run build_phonological_graph.py first to create the graph.")
        return

    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)

    print(f"✓ Loaded graph with {len(graph.G.nodes)} nodes")

    # Connect to database
    print("\n[2/5] Connecting to database...")
    conn = psycopg2.connect(DATABASE_URL)
    print("✓ Connected")

    try:
        # Get database words
        print("\n[3/5] Fetching words from database...")
        words_by_id, words_by_word = get_database_words(conn)
        print(f"✓ Loaded {len(words_by_id):,} words from database")

        # Check how many words are in both graph and database
        db_words_set = set(words_by_word.keys())
        graph_words_set = set(graph.G.nodes)
        common_words = db_words_set & graph_words_set
        print(f"  {len(common_words):,} words found in both graph and database ({100*len(common_words)/len(db_words_set):.1f}%)")

        # Clear existing typed edges
        print("\n[4/5] Clearing existing typed edges...")
        clear_existing_typed_edges(conn)

        # Populate typed edges
        print("\n[5/5] Populating typed edges...")
        populate_minimal_pairs(graph, conn, words_by_id, words_by_word, args.batch_size)
        populate_rhymes(graph, conn, words_by_id, words_by_word, args.batch_size)
        populate_neighbors(graph, conn, words_by_id, words_by_word, args.batch_size)

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
