#!/usr/bin/env python3
"""
Populate the word_edges table with typed graph relationships.

This script:
1. Loads the pre-built phonological graph (networkx pickle)
2. Extracts typed edges with metadata
3. Inserts into PostgreSQL word_edges table
"""

import sys
import os
import json
import pickle
import psycopg2
from psycopg2.extras import execute_batch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Connection string
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/phonolex")


def load_graph():
    """Load pre-built phonological graph"""
    graph_path = Path(__file__).parent.parent.parent.parent / "data" / "phonological_graph.pkl"

    print(f"Loading phonological graph from {graph_path}...")
    with open(graph_path, 'rb') as f:
        data = pickle.load(f)

    # The pickle contains a dictionary with 'graph' key
    if isinstance(data, dict) and 'graph' in data:
        graph = data['graph']
        print(f"✓ Loaded graph from dict: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    else:
        # Assume it's the graph itself
        graph = data
        print(f"✓ Loaded graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    return graph


def get_word_id_mapping(conn):
    """
    Get mapping of word string to word_id from database.

    Args:
        conn: psycopg2 connection

    Returns:
        dict: word -> word_id
    """
    cursor = conn.cursor()
    cursor.execute("SELECT word_id, word FROM words;")
    word_to_id = {word: word_id for word_id, word in cursor.fetchall()}
    cursor.close()

    print(f"✓ Loaded {len(word_to_id)} word IDs from database")
    return word_to_id


def extract_edges_from_graph(graph, word_to_id):
    """
    Extract typed edges from NetworkX graph.

    Args:
        graph: NetworkX graph
        word_to_id: Dict mapping word -> word_id

    Returns:
        List of tuples (word1_id, word2_id, relation_type, metadata, weight)
    """
    edges_data = []

    print("Extracting edges from graph...")

    for i, (word1, word2, edge_data) in enumerate(graph.edges(data=True)):
        if i % 5000 == 0 and i > 0:
            print(f"  Processed {i} edges...")

        # Skip if either word not in database
        if word1 not in word_to_id or word2 not in word_to_id:
            continue

        word1_id = word_to_id[word1]
        word2_id = word_to_id[word2]

        # Ensure word1_id < word2_id (database constraint)
        if word1_id > word2_id:
            word1_id, word2_id = word2_id, word1_id

        # Extract edge type (relation)
        relation_type = edge_data.get('relation', 'SIMILAR')

        # Extract metadata (everything except 'relation' and 'weight')
        metadata = {k: v for k, v in edge_data.items() if k not in ['relation', 'weight']}

        # Extract weight (default to 1.0 for unweighted edges)
        weight = edge_data.get('weight', 1.0)

        # Prepare tuple
        edge_tuple = (
            word1_id,
            word2_id,
            relation_type,
            json.dumps(metadata),
            float(weight)
        )

        edges_data.append(edge_tuple)

    print(f"✓ Extracted {len(edges_data)} edges")
    return edges_data


def insert_edges(conn, edges_data, batch_size=1000):
    """
    Insert edges into database.

    Args:
        conn: psycopg2 connection
        edges_data: List of edge tuples
        batch_size: Number of rows per batch
    """
    cursor = conn.cursor()

    insert_query = """
        INSERT INTO word_edges (
            word1_id, word2_id, relation_type, metadata, weight
        ) VALUES (
            %s, %s, %s, %s, %s
        )
        ON CONFLICT (word1_id, word2_id, relation_type) DO NOTHING;
    """

    print(f"Inserting {len(edges_data)} edges in batches of {batch_size}...")
    execute_batch(cursor, insert_query, edges_data, page_size=batch_size)
    conn.commit()

    # Get count
    cursor.execute("SELECT COUNT(*) FROM word_edges;")
    count = cursor.fetchone()[0]
    print(f"✓ Inserted edges. Total in database: {count}")

    # Show breakdown by relation type
    cursor.execute("""
        SELECT relation_type, COUNT(*)
        FROM word_edges
        GROUP BY relation_type
        ORDER BY COUNT(*) DESC;
    """)
    breakdown = cursor.fetchall()
    print("\nEdge type breakdown:")
    for relation_type, count in breakdown:
        print(f"  {relation_type}: {count:,}")

    cursor.close()


def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(description="Populate word_edges table")
    parser.add_argument("--graph", type=str, default="data/phonological_graph.pkl",
                        help="Path to phonological graph pickle")
    args = parser.parse_args()

    print("=" * 80)
    print("PhonoLex v2.0 - Word Edges Table Population")
    print("=" * 80)

    # Load graph
    print("\n[1/4] Loading phonological graph...")
    graph = load_graph()

    # Connect to database
    print("\n[2/4] Connecting to database...")
    conn = psycopg2.connect(DATABASE_URL)

    try:
        # Get word ID mapping
        print("\n[3/4] Loading word IDs from database...")
        word_to_id = get_word_id_mapping(conn)

        # Extract edges
        print("\n[4/4] Extracting and inserting edges...")
        edges_data = extract_edges_from_graph(graph, word_to_id)

        # Insert edges
        insert_edges(conn, edges_data)

        print("\n" + "=" * 80)
        print("✓ SUCCESS: Word edges table populated")
        print("=" * 80)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
