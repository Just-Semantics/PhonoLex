#!/usr/bin/env python3
"""
Phase 2: Normalized Feature Vectors Demo

This script demonstrates how to compute Phase 2 normalized vectors from Phase 1
Phoible features.

Phase 2 is NOT learned - it's a deterministic transformation:
- 76-dim: Start + end positions for each of 38 features
- 152-dim: 4-timestep interpolation for articulation dynamics

Use cases:
- Continuous phoneme similarity
- Diphthong modeling (trajectories capture glides)
- Vector database initialization
"""

from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path.cwd()))
from data.mappings.phoneme_vectorizer import PhonemeVectorizer, load_phoible_csv


def demo_normalized_vectors():
    """Demonstrate Phase 2 normalization."""

    print("=== Phase 2: Normalized Feature Vectors Demo ===\n")

    # Load Phoible data (Phase 1)
    print("Loading Phoible features (Phase 1)...")
    phoible_path = "data/phoible/phoible.csv"
    phoible_data = load_phoible_csv(phoible_path)

    # Get English phonemes
    english_data = [p for p in phoible_data if p.get("ISO6393") == "eng"]

    # Normalize Phoible symbols to standard IPA (strip diacritics)
    # Phoible uses retraction diacritics for affricates, we want clean symbols
    phoible_to_standard_ipa = {
        "d̠ʒ": "dʒ",  # Voiced postalveolar affricate
        "t̠ʃ": "tʃ",  # Voiceless postalveolar affricate
        "ɚ": "ɚ",  # Unstressed r-colored schwa (already standard)
        "ɡ": "ɡ",  # Voiced velar stop (already standard)
    }

    # Normalize phoneme symbols in data
    for entry in english_data:
        orig = entry["Phoneme"]
        if orig in phoible_to_standard_ipa:
            entry["Phoneme"] = phoible_to_standard_ipa[orig]

    # Add ɝ (stressed r-colored vowel) as fallback - use ɚ features
    # Phoible doesn't distinguish ɝ/ɚ, but CMU does
    schwa_r_entry = [p for p in english_data if p["Phoneme"] == "ɚ"]
    if schwa_r_entry:
        stressed_schwa_r = schwa_r_entry[0].copy()
        stressed_schwa_r["Phoneme"] = "ɝ"
        english_data.append(stressed_schwa_r)

    # Initialize vectorizer
    vectorizer = PhonemeVectorizer(encoding_scheme="three_way")

    # Example: Vectorize /t/ and /d/ (minimal pair: voicing)
    print("\n--- Example 1: Minimal Pair /t/ vs /d/ ---")
    t_data = [p for p in english_data if p["Phoneme"] == "t"][0]
    d_data = [p for p in english_data if p["Phoneme"] == "d"][0]

    t_vec = vectorizer.vectorize(t_data)
    d_vec = vectorizer.vectorize(d_data)

    print(f"/t/ endpoints (76-dim): shape {t_vec.endpoints_76d.shape}")
    print(f"/d/ endpoints (76-dim): shape {d_vec.endpoints_76d.shape}")

    # Compute similarity
    t_norm = t_vec.endpoints_76d / np.linalg.norm(t_vec.endpoints_76d)
    d_norm = d_vec.endpoints_76d / np.linalg.norm(d_vec.endpoints_76d)
    similarity = np.dot(t_norm, d_norm)

    print(f"Cosine similarity: {similarity:.3f} (high - only differ in voicing)")

    # Example 2: Diphthong trajectory
    print("\n--- Example 2: Diphthong Trajectory ---")
    # Note: This is conceptual - Phoible has monophthongs
    # In practice, you'd create diphthong data with start/end features

    print("76-dim encoding: [start_f1, end_f1, start_f2, end_f2, ...]")
    print("152-dim encoding: [t0_f1, t1_f1, t2_f1, t3_f1, t0_f2, ...]")
    print("  where t0=start, t3=end, t1/t2=interpolated")

    # Example 3: Vector database use case
    print("\n--- Example 3: Vector Database Storage ---")
    print("For ChromaDB/pgvector storage:")
    print("  - Use 76-dim endpoints for simple similarity")
    print("  - Use 152-dim trajectories for diphthongs/dynamics")
    print("  - All vectors are continuous (not ternary)")

    # Show all English phonemes
    print("\n--- English Phoneme Coverage ---")
    english_phonemes = list(set(p["Phoneme"] for p in english_data))
    print(f"Total phonemes: {len(english_phonemes)}")
    print(f"Sample: {english_phonemes[:10]}")

    # Save normalized vectors for all English phonemes
    print("\n--- Saving Phase 2 Embeddings ---")
    from pathlib import Path
    import pickle

    embeddings_76d = {}
    embeddings_152d = {}

    for phoneme_data in english_data:
        vec = vectorizer.vectorize(phoneme_data)
        phoneme = phoneme_data["Phoneme"]
        if phoneme not in embeddings_76d:  # Avoid duplicates
            embeddings_76d[phoneme] = vec.endpoints_76d
            embeddings_152d[phoneme] = vec.trajectory_152d

    output_dir = Path("embeddings/phase2")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "normalized_76d.pkl", "wb") as f:
        pickle.dump(embeddings_76d, f)
    with open(output_dir / "normalized_152d.pkl", "wb") as f:
        pickle.dump(embeddings_152d, f)

    print(f"Saved {len(embeddings_76d)} phoneme embeddings to embeddings/phase2/")
    print("  - normalized_76d.pkl (endpoints)")
    print("  - normalized_152d.pkl (trajectories)")


if __name__ == "__main__":
    demo_normalized_vectors()
