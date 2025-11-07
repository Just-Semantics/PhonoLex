#!/usr/bin/env python3
"""
Demo: Phoneme Learning Difficulty Based on Flege's Speech Learning Model

This demonstrates the core algorithm for the phoneme difficulty tool:
1. Load L1 and L2 phoneme inventories
2. For each L2 phoneme, find closest L1 phoneme
3. Compute phonetic distance (1 - cosine similarity)
4. Classify difficulty: identical, similar (HARD!), or new (easier)

Based on: Flege, J. E. (1995). Second language speech learning.
"""

import pickle
import json
import numpy as np
from typing import Dict, List, Tuple


def load_data():
    """Load PHOIBLE phoneme vectors and language inventories."""
    with open('embeddings/phase2/phoible_all_phonemes_76d.pkl', 'rb') as f:
        vectors = pickle.load(f)

    with open('embeddings/phase2/phoible_language_inventories.json') as f:
        languages = json.load(f)

    return vectors, languages


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)


def classify_difficulty(distance: float) -> Tuple[str, int, str]:
    """
    Classify L2 phoneme difficulty based on phonetic distance (Flege's SLM).

    Returns:
        (category, difficulty_score, explanation)

    Categories:
    - IDENTICAL (distance < 0.1): Easy - perfect transfer from L1
    - SIMILAR (0.1 <= distance < 0.5): HARDEST - equivalence classification
    - NEW (distance >= 0.5): Easier - differences are discernible
    """
    if distance < 0.1:
        return ("identical", 1, "Perfect transfer - identical to L1 sound")
    elif distance < 0.3:
        return ("similar", 5, "VERY HARD - equivalence classification (Flege H5)")
    elif distance < 0.5:
        return ("similar", 4, "HARD - perceived as same but actually different")
    elif distance < 0.7:
        return ("new", 2, "Easier - differences discernible")
    else:
        return ("new", 1, "Easy - clearly different, new category formed")


def find_closest_phoneme(
    target_phoneme: str,
    candidate_phonemes: List[str],
    vectors: Dict[str, np.ndarray]
) -> Tuple[str, float]:
    """
    Find the closest phoneme to target from candidates.

    Returns:
        (closest_phoneme, distance)
    """
    if target_phoneme not in vectors:
        return (None, None)

    target_vec = vectors[target_phoneme]
    best_phoneme = None
    best_distance = float('inf')

    for candidate in candidate_phonemes:
        if candidate not in vectors:
            continue

        similarity = cosine_similarity(target_vec, vectors[candidate])
        distance = 1.0 - similarity

        if distance < best_distance:
            best_distance = distance
            best_phoneme = candidate

    return (best_phoneme, best_distance)


def analyze_l1_to_l2(
    l1_code: str,
    l2_code: str,
    vectors: Dict[str, np.ndarray],
    languages: Dict[str, dict]
) -> List[dict]:
    """
    Analyze difficulty of learning L2 phonemes for L1 speakers.

    Returns list of results, sorted by difficulty (hardest first).
    """
    if l1_code not in languages or l2_code not in languages:
        print(f"Error: Language not found")
        return []

    l1 = languages[l1_code]
    l2 = languages[l2_code]

    results = []

    for l2_phoneme in l2['phonemes']:
        # Check if identical phoneme exists in L1
        if l2_phoneme in l1['phonemes']:
            category = "identical"
            difficulty = 1
            explanation = "Perfect transfer - exists in L1"
            closest_l1 = l2_phoneme
            distance = 0.0
        else:
            # Find closest L1 phoneme
            closest_l1, distance = find_closest_phoneme(
                l2_phoneme, l1['phonemes'], vectors
            )

            if closest_l1 is None:
                continue

            category, difficulty, explanation = classify_difficulty(distance)

        results.append({
            'l2_phoneme': l2_phoneme,
            'closest_l1': closest_l1,
            'distance': distance,
            'category': category,
            'difficulty': difficulty,
            'explanation': explanation
        })

    # Sort by difficulty (hardest first)
    results.sort(key=lambda x: (x['difficulty'], x['distance']), reverse=True)

    return results


def main():
    """Demonstrate phoneme difficulty analysis."""
    print("=== Phoneme Learning Difficulty Demo ===\n")
    print("Based on Flege's Speech Learning Model (1995)")
    print("Key insight: SIMILAR sounds are HARDER than completely NEW sounds!\n")

    # Load data
    print("Loading PHOIBLE data...")
    vectors, languages = load_data()
    print(f"  {len(vectors):,} phonemes")
    print(f"  {len(languages):,} languages\n")

    # Example 1: English speakers learning Spanish
    print("=" * 60)
    print("Example 1: English → Spanish")
    print("=" * 60)

    results = analyze_l1_to_l2('eng', 'spa', vectors, languages)

    print(f"\nSpanish has {languages['spa']['phoneme_count']} phonemes")
    print("Showing hardest 10 phonemes for English speakers:\n")

    for i, r in enumerate(results[:10], 1):
        print(f"{i:2}. /{r['l2_phoneme']:4}/ - Difficulty: {r['difficulty']}/5 ({r['category'].upper()})")
        print(f"    Closest English: /{r['closest_l1']}/ (distance: {r['distance']:.3f})")
        print(f"    {r['explanation']}")
        print()

    # Example 2: Spanish speakers learning English
    print("=" * 60)
    print("Example 2: Spanish → English")
    print("=" * 60)

    results = analyze_l1_to_l2('spa', 'eng', vectors, languages)

    print(f"\nEnglish has {languages['eng']['phoneme_count']} phonemes")
    print("Showing hardest 10 phonemes for Spanish speakers:\n")

    for i, r in enumerate(results[:10], 1):
        print(f"{i:2}. /{r['l2_phoneme']:4}/ - Difficulty: {r['difficulty']}/5 ({r['category'].upper()})")
        print(f"    Closest Spanish: /{r['closest_l1']}/ (distance: {r['distance']:.3f})")
        print(f"    {r['explanation']}")
        print()

    # Summary statistics
    print("=" * 60)
    print("Summary: Spanish → English Difficulty Distribution")
    print("=" * 60)

    category_counts = {}
    for r in results:
        category_counts[r['category']] = category_counts.get(r['category'], 0) + 1

    total = len(results)
    print(f"\nTotal L2 phonemes: {total}")
    print(f"  Identical: {category_counts.get('identical', 0):3} ({category_counts.get('identical', 0)/total*100:5.1f}%) - Easy transfer")
    print(f"  Similar:   {category_counts.get('similar', 0):3} ({category_counts.get('similar', 0)/total*100:5.1f}%) - HARDEST (equivalence classification)")
    print(f"  New:       {category_counts.get('new', 0):3} ({category_counts.get('new', 0)/total*100:5.1f}%) - Easier (differences discernible)")

    # Show the "danger zone" - similar sounds
    similar_sounds = [r for r in results if r['category'] == 'similar']
    if similar_sounds:
        print(f"\n⚠️  DANGER ZONE - {len(similar_sounds)} SIMILAR sounds (hardest to learn):")
        for r in similar_sounds:
            print(f"  /{r['l2_phoneme']:4}/ ← /{r['closest_l1']:4}/ (distance: {r['distance']:.3f})")

    print("\n" + "=" * 60)
    print("Implementation Note:")
    print("=" * 60)
    print("This algorithm will power the frontend phoneme difficulty tool.")
    print("Users can select any L1 → L2 language pair from 2,095 languages.")
    print("\nThe 'similar sounds' are where Flege's equivalence classification")
    print("(H5) causes maximum difficulty - learners perceive L1 and L2 sounds")
    print("as 'the same' when they're actually different.")


if __name__ == "__main__":
    main()
