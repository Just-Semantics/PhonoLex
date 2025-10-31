#!/usr/bin/env python3
"""
Populate psycholinguistic norm data for all words in the database.

Data sources:
- SUBTLEXus: Word frequency
- Concreteness ratings: Brysbaert et al. (2014)
- Glasgow Norms: Imageability, Familiarity, AoA
- Warriner et al.: Valence, Arousal, Dominance

Run after populate_words.py to add psycholinguistic properties.
"""

import sys
import os
from pathlib import Path
import csv
import psycopg2
from psycopg2.extras import execute_batch
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/phonolex")


def load_subtlex_frequency():
    """
    Load SUBTLEXus frequency data.

    Returns:
        dict: word -> (frequency, log_frequency)
    """
    print("Loading SUBTLEXus frequency data...")
    freq_path = Path(__file__).parent.parent.parent.parent / "data" / "subtlex_frequency.txt"

    freq_data = {}
    with open(freq_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            word = row['Word'].lower()
            freq = float(row['SUBTLWF']) if row['SUBTLWF'] else None
            log_freq = float(row['Lg10WF']) if row['Lg10WF'] else None
            freq_data[word] = (freq, log_freq)

    print(f"✓ Loaded {len(freq_data)} frequency entries")
    return freq_data


def load_concreteness():
    """
    Load concreteness ratings (Brysbaert et al. 2014).

    Returns:
        dict: word -> concreteness_rating
    """
    print("Loading concreteness ratings...")
    conc_path = Path(__file__).parent.parent.parent.parent / "data" / "norms" / "concreteness.txt"

    conc_data = {}
    with open(conc_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            word = row['Word'].lower()
            conc = float(row['Conc.M']) if row['Conc.M'] else None
            conc_data[word] = conc

    print(f"✓ Loaded {len(conc_data)} concreteness entries")
    return conc_data


def load_glasgow_norms():
    """
    Load Glasgow Norms: AoA, Imageability, Familiarity.

    Returns:
        dict: word -> (aoa, imageability, familiarity)
    """
    print("Loading Glasgow Norms (AoA, Imageability, Familiarity)...")
    glasgow_path = Path(__file__).parent.parent.parent.parent / "data" / "norms" / "GlasgowNorms.xlsx"

    try:
        # Skip first row (measure names), use second row as header
        # Row structure: word, length, M, SD, N (for each measure)
        # Measures: AROU, VAL, DOM, CNC, IMAG, FAM, AOA, SIZE, GEND
        df = pd.read_excel(glasgow_path, header=1)

        glasgow_data = {}
        for _, row in df.iterrows():
            word = str(row['word']).lower()
            # Column mapping:
            # M.6 = AOA (age of acquisition)
            # M.4 = IMAG (imageability)
            # M.5 = FAM (familiarity)
            aoa = float(row['M.6']) if pd.notna(row['M.6']) else None
            imageability = float(row['M.4']) if pd.notna(row['M.4']) else None
            familiarity = float(row['M.5']) if pd.notna(row['M.5']) else None
            glasgow_data[word] = (aoa, imageability, familiarity)

        print(f"✓ Loaded {len(glasgow_data)} Glasgow norm entries")
        return glasgow_data
    except Exception as e:
        print(f"  Warning: Could not load Glasgow norms: {e}")
        import traceback
        traceback.print_exc()
        return {}


def load_vad_ratings():
    """
    Load Valence-Arousal-Dominance ratings (Warriner et al.).

    Returns:
        dict: word -> (valence, arousal, dominance)
    """
    print("Loading VAD ratings (Valence, Arousal, Dominance)...")
    vad_path = Path(__file__).parent.parent.parent.parent / "data" / "norms" / "Ratings_VAD_WarrinerEtAl.csv"

    vad_data = {}
    with open(vad_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row['word'].lower()  # lowercase 'word'
            valence = float(row['valence']) if row['valence'] else None
            arousal = float(row['arousal']) if row['arousal'] else None
            dominance = float(row['dominance']) if row['dominance'] else None
            vad_data[word] = (valence, arousal, dominance)

    print(f"✓ Loaded {len(vad_data)} VAD entries")
    return vad_data


def merge_norm_data(freq_data, conc_data, glasgow_data, vad_data):
    """
    Merge all norm data sources into a single dictionary.

    Returns:
        dict: word -> (frequency, log_frequency, aoa, imageability, familiarity,
                       concreteness, valence, arousal, dominance)
    """
    print("\nMerging all norm data sources...")

    # Get all unique words
    all_words = set()
    all_words.update(freq_data.keys())
    all_words.update(conc_data.keys())
    all_words.update(glasgow_data.keys())
    all_words.update(vad_data.keys())

    merged_data = {}
    for word in all_words:
        freq, log_freq = freq_data.get(word, (None, None))
        conc = conc_data.get(word, None)
        aoa, imageability, familiarity = glasgow_data.get(word, (None, None, None))
        valence, arousal, dominance = vad_data.get(word, (None, None, None))

        merged_data[word] = (
            freq, log_freq, aoa, imageability, familiarity,
            conc, valence, arousal, dominance
        )

    print(f"✓ Merged data for {len(merged_data)} unique words")
    return merged_data


def update_database(conn, norm_data, batch_size=1000):
    """
    Update database with norm data.

    Args:
        conn: psycopg2 connection
        norm_data: Dict of word -> norm values
        batch_size: Batch size for updates
    """
    cursor = conn.cursor()

    # Get words from database
    print("\nFetching words from database...")
    cursor.execute("SELECT word_id, word FROM words")
    db_words = {word.lower(): word_id for word_id, word in cursor.fetchall()}
    print(f"✓ Found {len(db_words)} words in database")

    # Prepare update data
    print("\nPreparing update data...")
    update_data = []
    matched = 0

    for word, (freq, log_freq, aoa, imageability, familiarity,
               conc, valence, arousal, dominance) in norm_data.items():
        if word in db_words:
            word_id = db_words[word]
            update_data.append((
                freq, log_freq, aoa, imageability, familiarity,
                conc, valence, arousal, dominance, word_id
            ))
            matched += 1

    print(f"✓ Matched {matched} words to database ({100*matched/len(db_words):.1f}%)")

    if len(update_data) == 0:
        print("  No data to update!")
        return

    # Update database
    print(f"\nUpdating database in batches of {batch_size}...")

    update_query = """
        UPDATE words
        SET
            frequency = %s,
            log_frequency = %s,
            aoa = %s,
            imageability = %s,
            familiarity = %s,
            concreteness = %s,
            valence = %s,
            arousal = %s,
            dominance = %s
        WHERE word_id = %s
    """

    execute_batch(cursor, update_query, update_data, page_size=batch_size)
    conn.commit()

    print(f"✓ Updated {len(update_data)} words")

    # Get statistics
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
    total, has_freq, has_aoa, has_img, has_conc, has_val = stats

    print("\nDatabase statistics:")
    print(f"  Total words: {total:,}")
    print(f"  With frequency: {has_freq:,} ({100*has_freq/total:.1f}%)")
    print(f"  With AoA: {has_aoa:,} ({100*has_aoa/total:.1f}%)")
    print(f"  With imageability: {has_img:,} ({100*has_img/total:.1f}%)")
    print(f"  With concreteness: {has_conc:,} ({100*has_conc/total:.1f}%)")
    print(f"  With valence: {has_val:,} ({100*has_val/total:.1f}%)")

    cursor.close()


def main():
    """Main execution"""
    print("=" * 80)
    print("PhonoLex v2.0 - Psycholinguistic Norms Population")
    print("=" * 80)

    # Load all data sources
    print("\n[1/5] Loading data sources...")
    freq_data = load_subtlex_frequency()
    conc_data = load_concreteness()
    glasgow_data = load_glasgow_norms()
    vad_data = load_vad_ratings()

    # Merge data
    print("\n[2/5] Merging data sources...")
    norm_data = merge_norm_data(freq_data, conc_data, glasgow_data, vad_data)

    # Connect to database
    print("\n[3/5] Connecting to database...")
    conn = psycopg2.connect(DATABASE_URL)
    print("✓ Connected")

    try:
        # Update database
        print("\n[4/5] Updating database...")
        update_database(conn, norm_data)

        print("\n[5/5] Verification complete!")

        print("\n" + "=" * 80)
        print("✓ SUCCESS: Psycholinguistic norms populated")
        print("=" * 80)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
