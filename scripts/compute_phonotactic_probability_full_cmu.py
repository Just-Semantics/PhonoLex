#!/usr/bin/env python3
"""
Phonotactic Probability Calculator - Full CMU Dictionary

Computes on ALL 117K CMU words for accurate probability estimates,
then maps to our 24K subset.

Key improvement: Using full corpus eliminates bias from frequency-filtered subset!
"""

import json
from collections import defaultdict, Counter
from pathlib import Path
import math
from typing import Dict, List, Tuple
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.phonolex.embeddings.english_data_loader import PhonemeWithStress
from src.phonolex.utils.syllabification import syllabify

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
CMU_JSON_PATH = DATA_DIR / "cmu.json"
WORD_METADATA_PATH = Path(__file__).parent.parent / "webapp" / "frontend" / "public" / "data" / "word_metadata.json"
OUTPUT_PATH = DATA_DIR / "phonotactic_probability_full.json"


def load_full_cmu():
    """Load full CMU dictionary (117K words)"""
    print(f"Loading full CMU dictionary from {CMU_JSON_PATH}...")
    with open(CMU_JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"  ✓ Loaded {len(data)} words from CMU")
    return data


def convert_to_syllables(cmu_entry: dict) -> List[Dict]:
    """
    Convert CMU entry to syllable structures using PhonoLex syllabification

    Returns list of syllables with onset/nucleus/coda
    """
    ipa_phonemes = cmu_entry['ipa_phonemes']

    # Convert to PhonemeWithStress objects (extract stress from IPA)
    phonemes_with_stress = []
    for ipa in ipa_phonemes:
        # Strip stress markers
        phoneme = ipa.lstrip('ˈˌ')

        # Detect stress level from marker
        if ipa.startswith('ˈ'):
            stress = 1  # Primary
        elif ipa.startswith('ˌ'):
            stress = 2  # Secondary
        else:
            stress = 0 if phoneme in {'i', 'ɪ', 'ɛ', 'æ', 'ɑ', 'ɔ', 'ʊ', 'u', 'ʌ', 'ə', 'ɝ', 'ɚ', 'eɪ', 'aɪ', 'ɔɪ', 'aʊ', 'oʊ'} else None

        phonemes_with_stress.append(PhonemeWithStress(phoneme=phoneme, stress=stress))

    # Syllabify
    syllables = syllabify(phonemes_with_stress)

    # Convert to dicts
    return [
        {
            'onset': syll.onset,
            'nucleus': syll.nucleus,
            'coda': syll.coda,
            'stress': syll.stress
        }
        for syll in syllables
    ]


def extract_phoneme_sequences(cmu_data: dict) -> Dict[str, List[Tuple[str, str]]]:
    """
    Extract phoneme sequences from ALL CMU words

    Returns biphone sequences by position
    """
    sequences = {
        'onset': [],
        'coda': [],
        'onset_nucleus': [],
        'nucleus_coda': [],
    }

    print("Syllabifying all words...")
    syllabified = {}

    for i, (word, entry) in enumerate(cmu_data.items()):
        if i % 10000 == 0:
            print(f"  {i}/{len(cmu_data)} words...")

        try:
            syllables = convert_to_syllables(entry)
            syllabified[word] = syllables

            for syll in syllables:
                onset = syll['onset']
                nucleus = syll['nucleus']
                coda = syll['coda']

                # Onset biphones
                for j in range(len(onset) - 1):
                    sequences['onset'].append((onset[j], onset[j + 1]))

                # Coda biphones
                for j in range(len(coda) - 1):
                    sequences['coda'].append((coda[j], coda[j + 1]))

                # Cross-component transitions
                if onset and nucleus:
                    sequences['onset_nucleus'].append((onset[-1], nucleus))
                if nucleus and coda:
                    sequences['nucleus_coda'].append((nucleus, coda[0]))
        except Exception as e:
            print(f"  Warning: Failed to syllabify {word}: {e}")
            continue

    print(f"  ✓ Syllabified {len(syllabified)} words")
    return sequences, syllabified


def compute_positional_frequencies(syllabified: dict) -> Tuple[Dict[str, Counter], Dict[str, int]]:
    """
    Compute positional segment frequency from syllabified words
    """
    position_counts = {
        'onset': Counter(),
        'nucleus': Counter(),
        'coda': Counter()
    }

    position_totals = {
        'onset': 0,
        'nucleus': 0,
        'coda': 0
    }

    for word, syllables in syllabified.items():
        for syll in syllables:
            # Count onset phonemes
            for phoneme in syll['onset']:
                position_counts['onset'][phoneme] += 1
                position_totals['onset'] += 1

            # Count nucleus
            nucleus = syll['nucleus']
            if nucleus:
                position_counts['nucleus'][nucleus] += 1
                position_totals['nucleus'] += 1

            # Count coda phonemes
            for phoneme in syll['coda']:
                position_counts['coda'][phoneme] += 1
                position_totals['coda'] += 1

    return position_counts, position_totals


def compute_biphone_probabilities(sequences: Dict[str, List[Tuple[str, str]]]) -> Dict[str, Dict[Tuple[str, str], float]]:
    """
    Compute biphone probabilities
    """
    biphone_probs = {}

    for position, biphones in sequences.items():
        biphone_counts = Counter(biphones)
        first_phoneme_counts = Counter(p1 for p1, p2 in biphones)

        biphone_probs[position] = {}
        for (p1, p2), count in biphone_counts.items():
            prob = count / first_phoneme_counts[p1] if first_phoneme_counts[p1] > 0 else 0
            biphone_probs[position][(p1, p2)] = prob

    return biphone_probs


def compute_word_probs(
    syllabified: dict,
    position_counts: Dict[str, Counter],
    position_totals: Dict[str, int],
    biphone_probs: Dict[str, Dict[Tuple[str, str], float]]
) -> Dict[str, Dict[str, float]]:
    """
    Compute phonotactic probability for each word
    """
    word_probs = {}

    for word, syllables in syllabified.items():
        biphone_prob_list = []
        positional_prob_list = []

        for syll in syllables:
            onset = syll['onset']
            nucleus = syll['nucleus']
            coda = syll['coda']

            # Positional probabilities
            for phoneme in onset:
                total = position_totals['onset']
                count = position_counts['onset'][phoneme]
                prob = count / total if total > 0 else 0
                positional_prob_list.append(prob)

            if nucleus:
                total = position_totals['nucleus']
                count = position_counts['nucleus'][nucleus]
                prob = count / total if total > 0 else 0
                positional_prob_list.append(prob)

            for phoneme in coda:
                total = position_totals['coda']
                count = position_counts['coda'][phoneme]
                prob = count / total if total > 0 else 0
                positional_prob_list.append(prob)

            # Biphone probabilities
            for i in range(len(onset) - 1):
                biphone = (onset[i], onset[i + 1])
                prob = biphone_probs['onset'].get(biphone, 0)
                if prob > 0:
                    biphone_prob_list.append(prob)

            if onset and nucleus:
                transition = (onset[-1], nucleus)
                prob = biphone_probs['onset_nucleus'].get(transition, 0)
                if prob > 0:
                    biphone_prob_list.append(prob)

            if nucleus and coda:
                transition = (nucleus, coda[0])
                prob = biphone_probs['nucleus_coda'].get(transition, 0)
                if prob > 0:
                    biphone_prob_list.append(prob)

            for i in range(len(coda) - 1):
                biphone = (coda[i], coda[i + 1])
                prob = biphone_probs['coda'].get(biphone, 0)
                if prob > 0:
                    biphone_prob_list.append(prob)

        # Summary statistics
        if biphone_prob_list:
            phono_prob_sum_log = sum(math.log10(p) if p > 0 else -10 for p in biphone_prob_list)
            phono_prob_product = math.exp(sum(math.log(p) if p > 0 else -10 for p in biphone_prob_list))
            phono_prob_avg = sum(biphone_prob_list) / len(biphone_prob_list)
        else:
            phono_prob_sum_log = 0
            phono_prob_product = 1.0
            phono_prob_avg = 1.0

        if positional_prob_list:
            positional_prob_avg = sum(positional_prob_list) / len(positional_prob_list)
        else:
            positional_prob_avg = 0

        word_probs[word] = {
            'phono_prob_sum_log': phono_prob_sum_log,
            'phono_prob_product': phono_prob_product,
            'phono_prob_avg': phono_prob_avg,
            'positional_prob_avg': positional_prob_avg,
            'num_biphones': len(biphone_prob_list),
            'num_segments': len(positional_prob_list)
        }

    return word_probs


def main():
    print("=" * 70)
    print("Phonotactic Probability - FULL CMU Dictionary (117K words)")
    print("=" * 70)
    print()

    # Load full CMU
    cmu_data = load_full_cmu()
    print()

    # Extract sequences
    print("Extracting phoneme sequences...")
    sequences, syllabified = extract_phoneme_sequences(cmu_data)
    for pos, seqs in sequences.items():
        print(f"  {pos}: {len(seqs)} biphones")
    print()

    # Compute positional frequencies
    print("Computing positional frequencies...")
    position_counts, position_totals = compute_positional_frequencies(syllabified)
    for pos, counter in position_counts.items():
        print(f"  {pos}: {len(counter)} unique phonemes, {position_totals[pos]} total tokens")
    print()

    # Compute biphone probabilities
    print("Computing biphone probabilities...")
    biphone_probs = compute_biphone_probabilities(sequences)
    for pos, probs in biphone_probs.items():
        print(f"  {pos}: {len(probs)} unique biphones")
    print()

    # Compute word probabilities
    print("Computing word-level probabilities...")
    word_probs = compute_word_probs(syllabified, position_counts, position_totals, biphone_probs)
    print(f"  ✓ Computed for {len(word_probs)} words")
    print()

    # Show examples
    print("=" * 70)
    print("Examples (sorted by avg biphone probability)")
    print("=" * 70)

    sorted_words = sorted(word_probs.items(), key=lambda x: x[1]['phono_prob_avg'], reverse=True)

    print("\n=== HIGH phonotactic probability ===")
    print(f"{'Word':<15} {'Avg Biphone':>12} {'Sum Log':>10} {'Positional':>12}")
    print("-" * 70)
    for word, probs in sorted_words[:10]:
        print(f"{word:<15} {probs['phono_prob_avg']:>12.6f} {probs['phono_prob_sum_log']:>10.4f} {probs['positional_prob_avg']:>12.6f}")

    print("\n=== LOW phonotactic probability ===")
    print(f"{'Word':<15} {'Avg Biphone':>12} {'Sum Log':>10} {'Positional':>12}")
    print("-" * 70)
    multi_phoneme = [(w, p) for w, p in sorted_words if p['num_biphones'] > 0]
    for word, probs in multi_phoneme[-10:]:
        print(f"{word:<15} {probs['phono_prob_avg']:>12.6f} {probs['phono_prob_sum_log']:>10.4f} {probs['positional_prob_avg']:>12.6f}")

    # Save full results
    print()
    print(f"Saving results to {OUTPUT_PATH}...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'metadata': {
            'description': 'Phonotactic probability computed on full CMU dictionary',
            'source': 'CMU Pronouncing Dictionary',
            'num_words': len(word_probs),
            'reference': 'Vitevitch & Luce (2004)',
        },
        'word_probabilities': word_probs
    }

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"  ✓ Saved {len(word_probs)} word probabilities")

    # Map to 24K subset
    print()
    print("Mapping to 24K subset...")
    print(f"Loading subset from {WORD_METADATA_PATH}...")
    with open(WORD_METADATA_PATH, 'r') as f:
        subset_metadata = json.load(f)

    matched = 0
    missing = 0
    subset_probs = {}

    for word in subset_metadata.keys():
        if word in word_probs:
            subset_probs[word] = word_probs[word]
            matched += 1
        else:
            missing += 1
            print(f"  Warning: '{word}' not in full CMU")

    print(f"  ✓ Matched: {matched}/{len(subset_metadata)} words")
    print(f"  ⚠ Missing: {missing} words")

    # Save subset
    subset_output_path = DATA_DIR / "phonotactic_probability_24k.json"
    with open(subset_output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {
                'description': 'Phonotactic probability for 24K subset (computed on full CMU)',
                'source': 'Full CMU dictionary (117K words)',
                'subset_size': len(subset_probs),
                'reference': 'Vitevitch & Luce (2004)',
            },
            'word_probabilities': subset_probs
        }, f, indent=2, ensure_ascii=False)

    print(f"  ✓ Saved subset to {subset_output_path}")
    print()
    print("=" * 70)
    print("✓ Complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Merge into word_metadata.json")
    print("2. Add to Builder filters")
    print("3. Validate against Vitevitch & Luce calculator")


if __name__ == '__main__':
    main()
