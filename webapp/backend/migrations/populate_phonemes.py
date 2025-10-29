#!/usr/bin/env python3
"""
Populate the phonemes table with Phoible features and embeddings.

This script:
1. Loads phoneme features from Phoible JSON
2. Generates multiple embedding granularities
3. Inserts into PostgreSQL
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

# Connection string
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/phonolex")


def load_phoible_features():
    """Load Phoible features from TSV file"""
    import csv

    # Use the segments-features TSV file
    features_path = Path(__file__).parent.parent.parent.parent / "data" / "phoible" / "phoible-segments-features.tsv"

    print(f"Loading Phoible features from {features_path}...")

    features_data = {}

    with open(features_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')

        for row in reader:
            segment = row['segment']

            # Extract all feature columns (everything except 'segment')
            features = {key: value for key, value in row.items() if key != 'segment'}

            features_data[segment] = features

    print(f"✓ Loaded {len(features_data)} phonemes from Phoible")
    return features_data


def normalize_raw_features(features_dict):
    """
    Convert Phoible features (+/-/0) to normalized 37-dim vector.

    Note: The TSV file has 37 features (missing 'lenis'), not 38.

    Args:
        features_dict: Dict of feature names to values

    Returns:
        numpy array of shape (37,) with values in {-1, 0, +1}
    """
    # Expected features from the TSV file (37 features - 'lenis' is not in this TSV)
    # Note: The PhonemeVectorizer expects 38 but the TSV only has 37
    expected_features = [
        'tone', 'stress', 'syllabic', 'short', 'long',
        'consonantal', 'sonorant', 'continuant', 'delayedRelease', 'approximant',
        'tap', 'trill', 'nasal', 'lateral',
        'labial', 'round', 'labiodental',
        'coronal', 'anterior', 'distributed', 'strident',
        'dorsal', 'high', 'low', 'front', 'back', 'tense',
        'retractedTongueRoot', 'advancedTongueRoot',
        'periodicGlottalSource', 'epilaryngealSource',
        'spreadGlottis', 'constrictedGlottis',
        'fortis',  # Note: 'lenis' is missing from this TSV
        'raisedLarynxEjective', 'loweredLarynxImplosive', 'click'
    ]

    vector = []
    for feature in expected_features:
        value = features_dict.get(feature, '0')  # Default to '0' if missing
        if value == '+':
            vector.append(1.0)
        elif value == '-':
            vector.append(-1.0)
        else:  # '0' or missing
            vector.append(0.0)

    return np.array(vector, dtype=np.float32)


def normalize_feature_vector(features_dict):
    """
    Convert Phoible features (+/-/0) to normalized vector.

    Args:
        features_dict: Dict of feature names to values

    Returns:
        numpy array of shape (38,) with values in {-1, 0, +1}
    """
    # Phoible has 38 features
    feature_order = sorted(features_dict.keys())

    vector = []
    for feature in feature_order:
        value = features_dict[feature]
        if value == '+':
            vector.append(1.0)
        elif value == '-':
            vector.append(-1.0)
        else:  # '0'
            vector.append(0.0)

    return np.array(vector, dtype=np.float32)


def determine_segment_class(features_dict):
    """
    Determine segment class (consonant/vowel) from features.

    Args:
        features_dict: Dict of Phoible features

    Returns:
        str: 'consonant', 'vowel', or 'tone'
    """
    # Simplified heuristic: syllabic = + -> vowel, else consonant
    if features_dict.get('syllabic') == '+':
        return 'vowel'
    elif features_dict.get('tone') == '+':
        return 'tone'
    else:
        return 'consonant'


def prepare_phoneme_data(features_data):
    """
    Prepare all phoneme data for insertion.

    Args:
        features_data: Dict of IPA -> features from Phoible

    Returns:
        List of tuples ready for database insertion
    """
    phonemes_data = []

    print("Preparing phoneme data...")

    for ipa, features_dict in features_data.items():
        # Determine segment class
        segment_class = determine_segment_class(features_dict)

        # Features as JSONB
        features_json = json.dumps(features_dict)

        # Raw features vector (38-dim)
        raw_features = normalize_raw_features(features_dict)

        # For now, we'll skip the PhonemeVectorizer embeddings and contextual embeddings
        # These can be populated later with a separate script
        # The important thing is to get the basic phonemes and features in the database
        endpoints_76d = None
        trajectories_152d = None
        has_trajectory = False
        contextual_128d = None

        # Prepare tuple
        phoneme_tuple = (
            ipa,
            segment_class,
            features_json,
            raw_features.tolist() if raw_features is not None else None,
            endpoints_76d,
            trajectories_152d,
            contextual_128d,  # Will add later
            has_trajectory,
            None  # trajectory_features (optional metadata)
        )

        phonemes_data.append(phoneme_tuple)

    print(f"✓ Prepared {len(phonemes_data)} phonemes for insertion")
    return phonemes_data


def insert_phonemes(conn, phonemes_data, batch_size=100):
    """
    Insert phonemes into database.

    Args:
        conn: psycopg2 connection
        phonemes_data: List of phoneme tuples
        batch_size: Number of rows per batch
    """
    cursor = conn.cursor()

    insert_query = """
        INSERT INTO phonemes (
            ipa, segment_class, features,
            raw_features, endpoints_76d, trajectories_152d, contextual_128d,
            has_trajectory, trajectory_features
        ) VALUES (
            %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s
        )
        ON CONFLICT (ipa) DO NOTHING;
    """

    print(f"Inserting {len(phonemes_data)} phonemes in batches of {batch_size}...")
    execute_batch(cursor, insert_query, phonemes_data, page_size=batch_size)
    conn.commit()

    # Get count
    cursor.execute("SELECT COUNT(*) FROM phonemes;")
    count = cursor.fetchone()[0]
    print(f"✓ Inserted phonemes. Total in database: {count}")

    # Show breakdown by segment class
    cursor.execute("""
        SELECT segment_class, COUNT(*)
        FROM phonemes
        GROUP BY segment_class
        ORDER BY segment_class;
    """)
    breakdown = cursor.fetchall()
    print("\nPhoneme breakdown:")
    for segment_class, count in breakdown:
        print(f"  {segment_class}: {count}")

    cursor.close()


def main():
    """Main execution"""
    print("=" * 80)
    print("PhonoLex v2.0 - Phonemes Table Population")
    print("=" * 80)

    # Load Phoible features
    print("\n[1/3] Loading Phoible features...")
    features_data = load_phoible_features()

    # Prepare data
    print("\n[2/3] Preparing phoneme data...")
    phonemes_data = prepare_phoneme_data(features_data)

    # Insert into database
    print("\n[3/3] Inserting into database...")
    conn = psycopg2.connect(DATABASE_URL)
    try:
        insert_phonemes(conn, phonemes_data)
        print("\n" + "=" * 80)
        print("✓ SUCCESS: Phonemes table populated")
        print("=" * 80)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
