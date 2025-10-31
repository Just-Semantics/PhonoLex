#!/usr/bin/env python3
"""
Analyze the impact of different word filtering criteria based on psycholinguistic norms.

Current: Words with frequency data only
Proposed: Words with frequency + at least one additional norm (AoA, imageability, concreteness, VAD)
"""

import sys
from pathlib import Path
import csv
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.phonolex.embeddings.english_data_loader import EnglishPhonologyLoader


def load_subtlex_frequency():
    """Load SUBTLEXus frequency data"""
    print("Loading SUBTLEXus frequency data...")
    freq_path = Path("data/subtlex_frequency.txt")

    freq_words = set()
    with open(freq_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            word = row['Word'].lower()
            if row['SUBTLWF']:  # Has frequency data
                freq_words.add(word)

    print(f"  ✓ {len(freq_words):,} words with frequency")
    return freq_words


def load_concreteness():
    """Load concreteness ratings"""
    print("Loading concreteness ratings...")
    conc_path = Path("data/norms/concreteness.txt")

    conc_words = set()
    with open(conc_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            word = row['Word'].lower()
            if row['Conc.M']:  # Has concreteness rating
                conc_words.add(word)

    print(f"  ✓ {len(conc_words):,} words with concreteness")
    return conc_words


def load_glasgow_norms():
    """Load Glasgow Norms: AoA, Imageability, Familiarity"""
    print("Loading Glasgow Norms...")
    glasgow_path = Path("data/norms/GlasgowNorms.xlsx")

    try:
        df = pd.read_excel(glasgow_path, header=1)

        aoa_words = set()
        img_words = set()
        fam_words = set()

        for _, row in df.iterrows():
            word = str(row['word']).lower()
            if pd.notna(row['M.6']):  # AoA
                aoa_words.add(word)
            if pd.notna(row['M.4']):  # Imageability
                img_words.add(word)
            if pd.notna(row['M.5']):  # Familiarity
                fam_words.add(word)

        print(f"  ✓ {len(aoa_words):,} words with AoA")
        print(f"  ✓ {len(img_words):,} words with imageability")
        print(f"  ✓ {len(fam_words):,} words with familiarity")

        return aoa_words, img_words, fam_words
    except Exception as e:
        print(f"  Warning: Could not load Glasgow norms: {e}")
        return set(), set(), set()


def load_vad_ratings():
    """Load Valence-Arousal-Dominance ratings"""
    print("Loading VAD ratings...")
    vad_path = Path("data/norms/Ratings_VAD_WarrinerEtAl.csv")

    vad_words = set()
    with open(vad_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row['word'].lower()
            if row['valence'] or row['arousal'] or row['dominance']:
                vad_words.add(word)

    print(f"  ✓ {len(vad_words):,} words with VAD")
    return vad_words


def main():
    print("=" * 80)
    print("Psycholinguistic Norm Filtering Analysis")
    print("=" * 80)

    # Load CMU dictionary
    print("\n[1/3] Loading CMU Dictionary...")
    loader = EnglishPhonologyLoader()
    cmu_words = set(loader.lexicon.keys())
    print(f"  ✓ {len(cmu_words):,} total words in CMU dictionary")

    # Load all norm datasets
    print("\n[2/3] Loading psycholinguistic norm datasets...")
    freq_words = load_subtlex_frequency()
    conc_words = load_concreteness()
    aoa_words, img_words, fam_words = load_glasgow_norms()
    vad_words = load_vad_ratings()

    # Combine "other norms" (non-frequency)
    other_norms = conc_words | aoa_words | img_words | fam_words | vad_words
    print(f"\n  → {len(other_norms):,} unique words with at least one non-frequency norm")

    # Analyze filtering criteria
    print("\n[3/3] Analyzing filtering criteria...")
    print("=" * 80)

    # Current: frequency only
    current_filtered = freq_words & cmu_words
    print(f"\n1. CURRENT CRITERION: Frequency only")
    print(f"   Words in database: {len(current_filtered):,}")
    print(f"   Coverage of CMU: {100*len(current_filtered)/len(cmu_words):.1f}%")

    # Proposed: frequency + at least one other norm
    proposed_filtered = (freq_words & other_norms) & cmu_words
    print(f"\n2. PROPOSED CRITERION: Frequency + at least one other norm")
    print(f"   Words in database: {len(proposed_filtered):,}")
    print(f"   Coverage of CMU: {100*len(proposed_filtered)/len(cmu_words):.1f}%")

    # Size reduction
    reduction_count = len(current_filtered) - len(proposed_filtered)
    reduction_pct = 100 * reduction_count / len(current_filtered)
    print(f"\n3. IMPACT:")
    print(f"   Words removed: {reduction_count:,}")
    print(f"   Reduction: {reduction_pct:.1f}%")
    print(f"   Embedding size reduction: ~{reduction_pct:.0f}% (1.0GB → ~{1.0*(1-reduction_pct/100):.1f}GB)")

    # Detailed breakdown by norm
    print(f"\n4. BREAKDOWN (of proposed {len(proposed_filtered):,} words):")

    for_breakdown = proposed_filtered
    with_conc = conc_words & for_breakdown
    with_aoa = aoa_words & for_breakdown
    with_img = img_words & for_breakdown
    with_fam = fam_words & for_breakdown
    with_vad = vad_words & for_breakdown

    print(f"   With concreteness: {len(with_conc):,} ({100*len(with_conc)/len(for_breakdown):.1f}%)")
    print(f"   With AoA: {len(with_aoa):,} ({100*len(with_aoa)/len(for_breakdown):.1f}%)")
    print(f"   With imageability: {len(with_img):,} ({100*len(with_img)/len(for_breakdown):.1f}%)")
    print(f"   With familiarity: {len(with_fam):,} ({100*len(with_fam)/len(for_breakdown):.1f}%)")
    print(f"   With VAD: {len(with_vad):,} ({100*len(with_vad)/len(for_breakdown):.1f}%)")

    # Alternative: frequency + 2+ norms
    print(f"\n5. ALTERNATIVE CRITERION: Frequency + 2+ other norms")
    multi_norm_words = set()
    for word in freq_words & cmu_words:
        norm_count = sum([
            word in conc_words,
            word in aoa_words,
            word in img_words,
            word in fam_words,
            word in vad_words
        ])
        if norm_count >= 2:
            multi_norm_words.add(word)

    print(f"   Words in database: {len(multi_norm_words):,}")
    print(f"   Reduction: {100*(len(current_filtered)-len(multi_norm_words))/len(current_filtered):.1f}%")

    # Quality assessment
    print(f"\n6. QUALITY ASSESSMENT:")
    print(f"   Words with frequency ONLY (no other norms): {len(current_filtered - other_norms):,}")
    print(f"   → These are likely proper nouns, technical terms, or rare words")
    print(f"   → Removing them improves data quality for research/clinical use")

    print("\n" + "=" * 80)
    print("RECOMMENDATION:")
    print("=" * 80)
    print(f"✅ Implement 'frequency + at least one other norm' criterion")
    print(f"   - Reduces database/embedding size by ~{reduction_pct:.0f}%")
    print(f"   - Retains {len(proposed_filtered):,} high-quality words")
    print(f"   - Improves data quality (removes words without psycholinguistic properties)")
    print(f"   - Better suited for research and clinical applications")
    print("=" * 80)


if __name__ == "__main__":
    main()
