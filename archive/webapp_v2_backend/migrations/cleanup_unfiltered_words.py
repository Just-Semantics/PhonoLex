#!/usr/bin/env python3
"""
Clean up database by removing words that don't meet the filtering criterion.

This script removes words that have frequency data but lack other
psycholinguistic norms. This is useful for migrating existing databases
to the new filtering standard.

WARNING: This permanently deletes data. Backup your database first!
"""

import sys
import os
import psycopg2
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from word_filter import WordFilter

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/phonolex")


def get_database_stats(conn):
    """Get current database statistics"""
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            COUNT(*) as total,
            COUNT(frequency) as has_freq,
            COUNT(aoa) as has_aoa,
            COUNT(imageability) as has_img,
            COUNT(concreteness) as has_conc,
            COUNT(valence) as has_val
        FROM words
    """)

    stats = cursor.fetchone()
    cursor.close()

    return {
        'total': stats[0],
        'has_freq': stats[1],
        'has_aoa': stats[2],
        'has_img': stats[3],
        'has_conc': stats[4],
        'has_val': stats[5]
    }


def identify_words_to_remove(conn, word_filter):
    """
    Identify words in database that don't meet filtering criterion.

    Args:
        conn: psycopg2 connection
        word_filter: WordFilter instance

    Returns:
        List of word_ids to remove
    """
    cursor = conn.cursor()

    # Get all words from database
    cursor.execute("SELECT word_id, word FROM words")
    db_words = cursor.fetchall()
    cursor.close()

    print(f"Analyzing {len(db_words):,} words in database...")

    to_remove = []
    for word_id, word in db_words:
        if not word_filter.should_include_word(word):
            to_remove.append(word_id)

    return to_remove


def remove_words(conn, word_ids_to_remove, batch_size=1000):
    """
    Remove words from database.

    This will cascade to related tables (word_edges, word_syllables, etc.)
    due to foreign key constraints.

    Args:
        conn: psycopg2 connection
        word_ids_to_remove: List of word_ids
        batch_size: Batch size for deletion
    """
    if len(word_ids_to_remove) == 0:
        print("No words to remove!")
        return

    cursor = conn.cursor()

    print(f"Removing {len(word_ids_to_remove):,} words in batches of {batch_size}...")

    for i in range(0, len(word_ids_to_remove), batch_size):
        batch = word_ids_to_remove[i:i+batch_size]

        # Delete words (cascades to related tables)
        cursor.execute(
            "DELETE FROM words WHERE word_id = ANY(%s)",
            (batch,)
        )

        if (i + batch_size) % 10000 == 0:
            print(f"  Processed {i + batch_size:,} deletions...")

    conn.commit()
    cursor.close()

    print(f"✓ Removed {len(word_ids_to_remove):,} words")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Clean up database by removing words without psycholinguistic norms"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without actually deleting"
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Database Cleanup: Remove Unfiltered Words")
    print("=" * 80)

    # Load word filter
    print("\n[1/5] Loading word filter...")
    word_filter = WordFilter()
    word_filter.load_all_norms()

    # Connect to database
    print("\n[2/5] Connecting to database...")
    conn = psycopg2.connect(DATABASE_URL)
    print("✓ Connected")

    try:
        # Get current stats
        print("\n[3/5] Analyzing current database...")
        stats_before = get_database_stats(conn)

        print(f"\nCurrent database:")
        print(f"  Total words: {stats_before['total']:,}")
        print(f"  With frequency: {stats_before['has_freq']:,}")
        print(f"  With AoA: {stats_before['has_aoa']:,}")
        print(f"  With imageability: {stats_before['has_img']:,}")
        print(f"  With concreteness: {stats_before['has_conc']:,}")
        print(f"  With valence: {stats_before['has_val']:,}")

        # Identify words to remove
        print("\n[4/5] Identifying words to remove...")
        word_ids_to_remove = identify_words_to_remove(conn, word_filter)

        words_to_keep = stats_before['total'] - len(word_ids_to_remove)
        reduction_pct = 100 * len(word_ids_to_remove) / stats_before['total']

        print(f"\nPlanned changes:")
        print(f"  Words to remove: {len(word_ids_to_remove):,}")
        print(f"  Words to keep: {words_to_keep:,}")
        print(f"  Reduction: {reduction_pct:.1f}%")

        if args.dry_run:
            print("\n" + "=" * 80)
            print("DRY RUN: No changes made")
            print("=" * 80)
            return

        # Confirm deletion
        if not args.yes:
            print("\n" + "⚠️  WARNING: This will permanently delete data! ⚠️")
            print("Make sure you have a database backup before proceeding.")
            response = input("\nContinue? (type 'yes' to confirm): ")
            if response.lower() != 'yes':
                print("Cancelled.")
                return

        # Remove words
        print("\n[5/5] Removing words...")
        remove_words(conn, word_ids_to_remove)

        # Get final stats
        stats_after = get_database_stats(conn)

        print("\n" + "=" * 80)
        print("✓ SUCCESS: Database cleaned")
        print("=" * 80)

        print(f"\nFinal database:")
        print(f"  Total words: {stats_after['total']:,}")
        print(f"  Removed: {stats_before['total'] - stats_after['total']:,}")
        print(f"  Reduction: {100*(stats_before['total'] - stats_after['total'])/stats_before['total']:.1f}%")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
