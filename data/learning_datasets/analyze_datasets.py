#!/usr/bin/env python3
"""
Comprehensive Analysis of Learning Datasets

Analyzes all downloaded datasets and their potential for phonological learning.
Creates a summary with statistics, overlap analysis, and recommendations.
"""

from pathlib import Path
from collections import defaultdict, Counter
import json

def analyze_sigmorphon():
    """Analyze SIGMORPHON 2020 English data"""
    print("\n" + "=" * 70)
    print("1. SIGMORPHON 2020 - English Morphological Inflection")
    print("=" * 70)

    base = Path("sigmorphon2020/task0/data/germanic")
    trn_file = base / "eng.trn"
    dev_file = base / "eng.dev"

    if not trn_file.exists():
        print("  NOT FOUND")
        return None

    # Count lines
    with open(trn_file) as f:
        trn_lines = [line.strip() for line in f if line.strip()]
    with open(dev_file) as f:
        dev_lines = [line.strip() for line in f if line.strip()]

    print(f"\n  Files:")
    print(f"    Training: eng.trn ({len(trn_lines):,} examples)")
    print(f"    Development: eng.dev ({len(dev_lines):,} examples)")
    print(f"    Total: {len(trn_lines) + len(dev_lines):,} examples")

    # Analyze features
    feature_counts = Counter()
    lemma_set = set()
    inflected_set = set()

    for line in trn_lines + dev_lines:
        parts = line.split('\t')
        if len(parts) == 3:
            lemma, inflected, features = parts
            lemma_set.add(lemma)
            inflected_set.add(inflected)
            feature_counts[features] += 1

    print(f"\n  Vocabulary:")
    print(f"    Unique lemmas: {len(lemma_set):,}")
    print(f"    Unique inflected forms: {len(inflected_set):,}")

    print(f"\n  Most common morphological features (top 10):")
    for feat, count in feature_counts.most_common(10):
        print(f"    {feat:30s} {count:6,} ({100*count/len(trn_lines+dev_lines):.1f}%)")

    print(f"\n  Sample entries:")
    for i, line in enumerate(trn_lines[:5]):
        parts = line.split('\t')
        if len(parts) == 3:
            lemma, inflected, features = parts
            print(f"    {lemma:20s} → {inflected:20s} [{features}]")

    return {
        'trn_count': len(trn_lines),
        'dev_count': len(dev_lines),
        'lemmas': lemma_set,
        'inflected': inflected_set,
        'features': feature_counts
    }


def analyze_unimorph():
    """Analyze UniMorph English data"""
    print("\n" + "=" * 70)
    print("2. UniMorph - English")
    print("=" * 70)

    base = Path("unimorph-eng")
    if not base.exists():
        print("  NOT FOUND")
        return None

    # Find all data files
    data_files = list(base.glob("eng*")) + list(base.glob("*.txt"))

    if not data_files:
        print("  No data files found")
        return None

    total_lines = 0
    all_lemmas = set()
    all_features = Counter()

    for f in data_files:
        with open(f, encoding='utf-8', errors='ignore') as file:
            lines = [line.strip() for line in file if line.strip() and not line.startswith('#')]
            total_lines += len(lines)

            for line in lines:
                parts = line.split('\t')
                if len(parts) >= 3:
                    lemma, inflected, features = parts[0], parts[1], parts[2]
                    all_lemmas.add(lemma)
                    all_features[features] += 1

    print(f"\n  Files: {len(data_files)} files")
    print(f"  Total entries: {total_lines:,}")
    print(f"  Unique lemmas: {len(all_lemmas):,}")

    if all_features:
        print(f"\n  Most common features (top 10):")
        for feat, count in all_features.most_common(10):
            print(f"    {feat:30s} {count:6,}")

    return {
        'total': total_lines,
        'lemmas': all_lemmas,
        'features': all_features
    }


def analyze_ipa_dict():
    """Analyze ipa-dict English pronunciations"""
    print("\n" + "=" * 70)
    print("3. ipa-dict - English Pronunciations")
    print("=" * 70)

    en_us = Path("ipa-dict/data/en_US.txt")
    en_uk = Path("ipa-dict/data/en_UK.txt")

    datasets = []
    if en_us.exists():
        datasets.append(('en_US (American)', en_us))
    if en_uk.exists():
        datasets.append(('en_UK (British)', en_uk))

    if not datasets:
        print("  NOT FOUND")
        return None

    all_data = {}
    for name, filepath in datasets:
        with open(filepath, encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]

        words = set()
        pronunciations = []
        for line in lines:
            parts = line.split('\t')
            if len(parts) == 2:
                word, ipa = parts
                words.add(word)
                pronunciations.append(ipa)

        all_data[name] = {
            'count': len(lines),
            'words': words,
            'pronunciations': pronunciations
        }

        print(f"\n  {name}:")
        print(f"    Total entries: {len(lines):,}")
        print(f"    Unique words: {len(words):,}")
        print(f"\n    Sample:")
        for line in lines[:5]:
            parts = line.split('\t')
            if len(parts) == 2:
                word, ipa = parts
                print(f"      {word:20s} /{ipa}/")

    return all_data


def analyze_cmu_dict():
    """Analyze CMU Pronouncing Dictionary for comparison"""
    print("\n" + "=" * 70)
    print("4. CMU Pronouncing Dictionary (for comparison)")
    print("=" * 70)

    cmu_file = Path("../cmu/cmudict-0.7b")
    if not cmu_file.exists():
        print("  NOT FOUND")
        return None

    words = set()
    pronunciations = []

    with open(cmu_file, encoding='latin-1') as f:
        for line in f:
            if line.startswith(';;;'):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                word = parts[0]
                phones = ' '.join(parts[1:])
                words.add(word)
                pronunciations.append(phones)

    print(f"\n  Total entries: {len(pronunciations):,}")
    print(f"  Unique words: {len(words):,}")
    print(f"\n  Sample:")
    with open(cmu_file, encoding='latin-1') as f:
        count = 0
        for line in f:
            if not line.startswith(';;;'):
                print(f"    {line.strip()}")
                count += 1
                if count >= 5:
                    break

    return {
        'count': len(pronunciations),
        'words': words
    }


def analyze_overlap(sig_data, uni_data, ipa_data, cmu_data):
    """Analyze overlap between datasets"""
    print("\n" + "=" * 70)
    print("DATASET OVERLAP ANALYSIS")
    print("=" * 70)

    if sig_data and uni_data:
        sig_lemmas = sig_data['lemmas']
        uni_lemmas = uni_data['lemmas']
        overlap = sig_lemmas & uni_lemmas
        print(f"\n  SIGMORPHON ∩ UniMorph:")
        print(f"    Overlapping lemmas: {len(overlap):,}")
        print(f"    SIGMORPHON only: {len(sig_lemmas - uni_lemmas):,}")
        print(f"    UniMorph only: {len(uni_lemmas - sig_lemmas):,}")

    if sig_data and cmu_data:
        sig_words = sig_data['lemmas'] | sig_data['inflected']
        cmu_words = cmu_data['words']
        overlap = sig_words & cmu_words
        print(f"\n  SIGMORPHON ∩ CMU Dict:")
        print(f"    Overlapping words: {len(overlap):,}")
        print(f"    Coverage: {100*len(overlap)/len(sig_words):.1f}% of SIGMORPHON words in CMU")

    if ipa_data and cmu_data:
        if 'en_US (American)' in ipa_data:
            ipa_words = ipa_data['en_US (American)']['words']
            cmu_words = cmu_data['words']
            overlap = ipa_words & cmu_words
            print(f"\n  ipa-dict (US) ∩ CMU Dict:")
            print(f"    Overlapping words: {len(overlap):,}")
            print(f"    ipa-dict only: {len(ipa_words - cmu_words):,}")
            print(f"    CMU only: {len(cmu_words - ipa_words):,}")


def create_summary():
    """Create comprehensive summary"""
    print("\n" + "=" * 70)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 70)

    print("""
  DATASETS READY FOR LEARNING:

  1. SIGMORPHON 2020 (92,418 examples)
     ✅ Morphological inflection with features
     ✅ Train/dev splits provided
     ✅ Learn: allomorphy, phonological conditioning

  2. UniMorph (varies)
     ✅ Additional morphological data
     ✅ Can be used for augmentation or held-out test

  3. ipa-dict (pronunciation data)
     ✅ IPA transcriptions for words
     ✅ Multiple dialects (US, UK)
     ✅ Can link to morphology datasets

  4. CMU Dict (134K words)
     ✅ ARPAbet pronunciations
     ✅ Already have ARPAbet→IPA mapping
     ✅ Baseline for comparison

  RECOMMENDED APPROACH:

  Train/Dev/Test Strategy:
  - TRAIN: SIGMORPHON train split + augment with Phoible features
  - DEV: SIGMORPHON dev split
  - TEST1: UniMorph (held-out, different distribution)
  - TEST2: Novel words from ipa-dict not in SIGMORPHON

  Learning Tasks:
  1. Plural allomorphy: -s vs -z vs -ɪz (phonologically conditioned)
  2. Past tense: -t vs -d vs -ɪd (phonologically conditioned)
  3. Irregular forms: sing→sang (lexical)
  4. Stress shifts: photograph→photográphic

  Next Steps:
  1. Create combined dataset: SIGMORPHON + CMU pronunciations + Phoible features
  2. Build baseline model (simple rules)
  3. Train neural model
  4. Evaluate on held-out test sets
    """)


def main():
    print("=" * 70)
    print("PHONOLEX LEARNING DATASETS ANALYSIS")
    print("=" * 70)
    print("\nAnalyzing all downloaded datasets...")

    # Analyze each dataset
    sig_data = analyze_sigmorphon()
    uni_data = analyze_unimorph()
    ipa_data = analyze_ipa_dict()
    cmu_data = analyze_cmu_dict()

    # Analyze overlap
    analyze_overlap(sig_data, uni_data, ipa_data, cmu_data)

    # Create summary
    create_summary()

    print("\n" + "=" * 70)
    print("✓ Analysis complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
