"""
Populate words.syllable_embeddings array column from Layer 4 checkpoint.

This stores the exact syllable embedding sequences (list of 384-dim vectors)
for each word, preserving the contextual information from Layer 3.
"""

import sys
import os
import torch
import psycopg2
from psycopg2.extras import execute_batch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import DatabaseService


def load_layer4_embeddings(checkpoint_path):
    """Load Layer 4 syllable embeddings from checkpoint."""
    print(f"Loading Layer 4 checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    word_to_syllable_embeddings = checkpoint['word_to_syllable_embeddings']
    print(f"✓ Loaded embeddings for {len(word_to_syllable_embeddings):,} words")
    return word_to_syllable_embeddings


def populate_syllable_embeddings_array(db, word_to_syllable_embeddings):
    """
    Populate words.syllable_embeddings array column.

    Stores exact syllable sequences from Layer 4 checkpoint.
    """
    print("\nPopulating words.syllable_embeddings array...")

    conn = psycopg2.connect(os.getenv("DATABASE_URL", "postgresql://localhost/phonolex"))
    cursor = conn.cursor()

    # Get all words from database
    with db.get_session() as session:
        from models import Word
        words = session.query(Word).all()

        update_batch = []
        matched = 0
        missing = 0

        for word in words:
            if word.word in word_to_syllable_embeddings:
                # Get syllable embeddings from checkpoint
                syll_embs = word_to_syllable_embeddings[word.word]

                # Convert each embedding to pgvector string format: '[val1,val2,...]'
                syll_embs_str = [f"[{','.join(map(str, emb.tolist()))}]" for emb in syll_embs]

                update_batch.append((syll_embs_str, word.word_id))
                matched += 1
            else:
                missing += 1

        print(f"  Matched {matched:,} words")
        print(f"  Missing {missing:,} words")

        # Batch update (construct ARRAY manually)
        if update_batch:
            for i, (embs_str, word_id) in enumerate(update_batch):
                # Construct array literal: ARRAY['[...]'::vector, '[...]'::vector, ...]
                array_literal = "ARRAY[" + ",".join([f"'{e}'::vector(384)" for e in embs_str]) + "]"

                update_query = f"""
                    UPDATE words
                    SET syllable_embeddings = {array_literal}
                    WHERE word_id = %s
                """

                cursor.execute(update_query, (word_id,))

                if (i + 1) % 1000 == 0:
                    print(f"  Updated {i+1:,} words...")
                    conn.commit()

            conn.commit()

    cursor.close()
    conn.close()

    print(f"✓ Populated syllable_embeddings array for {matched:,} words")


def main():
    # Paths
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    checkpoint_path = os.path.join(repo_root, 'embeddings', 'layer4', 'syllable_embeddings.pt')

    # Load Layer 4 embeddings
    word_to_syllable_embeddings = load_layer4_embeddings(checkpoint_path)

    # Initialize database
    db = DatabaseService()

    # Populate array column
    populate_syllable_embeddings_array(db, word_to_syllable_embeddings)

    print("\n✅ Syllable embeddings array population complete!")


if __name__ == '__main__':
    main()
