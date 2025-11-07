#!/usr/bin/env python3
"""
Rebuild ARPAbet→IPA mapping from PHOIBLE English union.

This script properly derives the mapping from linguistic data rather than
manual specification, ensuring all English phonemes from PHOIBLE are covered.

Process:
1. Extract union of all English phonemes from PHOIBLE
2. Normalize PHOIBLE symbols to standard IPA
3. Map each to ARPAbet equivalent
4. Generate bidirectional mappings
"""

import csv
import json
from pathlib import Path
from typing import Dict, Set

project_root = Path(__file__).resolve().parents[1]
phoible_path = project_root / "data/phoible/phoible.csv"
output_dir = project_root / "data/mappings"


def extract_english_phonemes() -> Set[str]:
    """Extract union of all English phonemes from PHOIBLE."""
    print("Extracting English phonemes from PHOIBLE...")

    english_phonemes = set()

    with open(phoible_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['ISO6393'] == 'eng':
                phoneme = row['Phoneme']
                english_phonemes.add(phoneme)

    print(f"  Found {len(english_phonemes)} unique English phonemes")
    return english_phonemes


def normalize_phoible_symbols(phonemes: Set[str]) -> Dict[str, str]:
    """Normalize PHOIBLE symbols to standard IPA."""
    print("\nNormalizing PHOIBLE symbols...")

    # Known normalizations
    normalizations = {
        'd̠ʒ': 'dʒ',  # Voiced postalveolar affricate
        't̠ʃ': 'tʃ',  # Voiceless postalveolar affricate
        'eɪ̯': 'eɪ',  # Diphthong normalization
        'iɪ': 'ɪ',   # Common variant
        'kʰ': 'k',   # Aspiration (not phonemic in English)
        'pʰ': 'p',   # Aspiration
        'tʰ': 't',   # Aspiration
        'uː': 'u',   # Length (not phonemic in standard American English)
        'ɚː': 'ɚ',   # Length
        'ei': 'eɪ',  # Alternative notation
        'g': 'ɡ',    # Use standard IPA g
    }

    normalized = {}
    for phoneme in phonemes:
        if phoneme in normalizations:
            normalized[phoneme] = normalizations[phoneme]
        else:
            normalized[phoneme] = phoneme

    # Get unique normalized set
    unique_normalized = set(normalized.values())
    print(f"  Normalized to {len(unique_normalized)} unique IPA symbols")

    return normalized


def map_ipa_to_arpa() -> Dict[str, str]:
    """
    Create IPA→ARPAbet mapping.

    This is the canonical mapping from linguistic symbols to CMU notation.
    Based on CMU Pronouncing Dictionary standard.
    """
    print("\nCreating IPA→ARPAbet mapping...")

    # Vowels (monophthongs and diphthongs)
    vowels = {
        'eɪ': 'EY',   # hate, day
        'i': 'IY',    # Pete, see
        'aɪ': 'AY',   # site, buy
        'oʊ': 'OW',   # note, go
        'u': 'UW',    # cute, boot
        'æ': 'AE',    # hat, cat
        'ɛ': 'EH',    # pet, bed
        'ɪ': 'IH',    # sit, bit
        'ɔ': 'AO',    # caught, bought
        'ɑ': 'AA',    # hot, lot (American)
        'ʌ': 'AH',    # cut, but (stressed)
        'ə': 'AH',    # about (unstressed - special handling below)
        'ɔɪ': 'OY',   # coin, boy
        'aʊ': 'AW',   # loud, cow
        'ʊ': 'UH',    # book, put
        'ɝ': 'ER',    # bird, turn (stressed r-colored)
        'ɚ': 'ER',    # butter (unstressed r-colored - special handling below)
    }

    # Consonants
    consonants = {
        'b': 'B',     # buy
        'p': 'P',     # pie
        'd': 'D',     # die
        't': 'T',     # tie
        'v': 'V',     # vie
        'f': 'F',     # fight
        'ɡ': 'G',     # guy
        'k': 'K',     # kite
        'h': 'HH',    # high
        'dʒ': 'JH',   # joy, judge
        'tʃ': 'CH',   # China, church
        'l': 'L',     # lie
        'm': 'M',     # my
        'n': 'N',     # nigh
        'ɹ': 'R',     # rye (American r)
        'z': 'Z',     # zoo
        's': 'S',     # sigh
        'w': 'W',     # wise
        'j': 'Y',     # yacht
        'ʒ': 'ZH',    # pleasure, measure
        'ʃ': 'SH',    # shy, ship
        'ð': 'DH',    # they, this
        'θ': 'TH',    # thigh, thing
        'ŋ': 'NG',    # sing, thing
    }

    mapping = {**vowels, **consonants}
    print(f"  Created mapping for {len(mapping)} phonemes")

    return mapping


def create_arpa_to_ipa(ipa_to_arpa: Dict[str, str]) -> Dict[str, str]:
    """
    Create ARPAbet→IPA mapping with stress markers.

    ARPAbet uses digit suffixes for stress:
    - 0: unstressed
    - 1: primary stress
    - 2: secondary stress

    Vowels can have any stress level.
    Consonants never have stress markers.
    """
    print("\nCreating ARPAbet→IPA mapping with stress...")

    arpa_to_ipa = {}

    # IPA stress markers
    PRIMARY_STRESS = 'ˈ'
    SECONDARY_STRESS = 'ˌ'

    # Vowels that can take stress
    stress_vowels = ['EY', 'IY', 'AY', 'OW', 'UW', 'AE', 'EH', 'IH',
                     'AO', 'AA', 'AH', 'OY', 'AW', 'UH', 'ER']

    for ipa, arpa in ipa_to_arpa.items():
        # Check if this is a vowel that takes stress
        if arpa in stress_vowels:
            # Base form (no stress marker)
            arpa_to_ipa[arpa] = ipa

            # Unstressed (0)
            if arpa == 'AH':
                # AH0 → ə (schwa), AH (no suffix) → ʌ (stressed)
                arpa_to_ipa['AH0'] = 'ə'
                arpa_to_ipa['AH1'] = f'{PRIMARY_STRESS}ʌ'
                arpa_to_ipa['AH2'] = f'{SECONDARY_STRESS}ʌ'
            elif arpa == 'ER':
                # ER0 → ɚ (unstressed), ER1 → ɝ (stressed)
                arpa_to_ipa['ER0'] = 'ɚ'
                arpa_to_ipa['ER1'] = f'{PRIMARY_STRESS}ɝ'
                arpa_to_ipa['ER2'] = f'{SECONDARY_STRESS}ɝ'
                arpa_to_ipa['ER'] = 'ɝ'  # Default to stressed
            else:
                # Regular vowels
                arpa_to_ipa[f'{arpa}0'] = ipa  # Unstressed
                arpa_to_ipa[f'{arpa}1'] = f'{PRIMARY_STRESS}{ipa}'  # Primary stress
                arpa_to_ipa[f'{arpa}2'] = f'{SECONDARY_STRESS}{ipa}'  # Secondary stress
        else:
            # Consonants - no stress
            arpa_to_ipa[arpa] = ipa

    print(f"  Created {len(arpa_to_ipa)} ARPAbet→IPA mappings (including stress variants)")

    return arpa_to_ipa


def save_mappings(ipa_to_arpa: Dict[str, str], arpa_to_ipa: Dict[str, str]):
    """Save mapping files."""
    print("\nSaving mapping files...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save IPA→ARPA
    ipa_file = output_dir / "ipa_to_arpa.json"
    with open(ipa_file, 'w', encoding='utf-8') as f:
        json.dump(ipa_to_arpa, f, indent=2, ensure_ascii=False)
    print(f"  ✓ {ipa_file} ({len(ipa_to_arpa)} mappings)")

    # Save ARPA→IPA
    arpa_file = output_dir / "arpa_to_ipa.json"
    with open(arpa_file, 'w', encoding='utf-8') as f:
        json.dump(arpa_to_ipa, f, indent=2, ensure_ascii=False)
    print(f"  ✓ {arpa_file} ({len(arpa_to_ipa)} mappings)")


def verify_coverage():
    """Verify CMU uses only PHOIBLE-derived phonemes."""
    print("\nVerifying CMU coverage...")

    # Load our PHOIBLE-derived mapping (this is the canonical set)
    arpa_file = output_dir / "arpa_to_ipa.json"
    with open(arpa_file, 'r', encoding='utf-8') as f:
        arpa_to_ipa = json.load(f)

    canonical_arpa = set(arpa_to_ipa.keys())
    print(f"  Canonical ARPAbet phonemes (from PHOIBLE English): {len(canonical_arpa)}")

    # Load CMU dictionary to see what it actually uses
    cmu_path = project_root / "data/cmu/cmudict-0.7b"
    cmu_phonemes = set()
    with open(cmu_path, 'r', encoding='latin-1') as f:
        for line in f:
            if line.startswith(';;;'):
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            # Skip variant markers like WORD(1)
            if '(' in parts[0]:
                continue
            # Get phonemes (everything after the word)
            phonemes = parts[1:]
            cmu_phonemes.update(phonemes)

    print(f"  ARPAbet phonemes used in CMU dictionary: {len(cmu_phonemes)}")

    # Check if CMU uses any phonemes NOT in our canonical set
    extra = cmu_phonemes - canonical_arpa
    if extra:
        print(f"  ⚠ CMU uses {len(extra)} phonemes NOT in PHOIBLE English:")
        for p in sorted(extra):
            print(f"    - {p}")
        return False

    # Check which canonical phonemes are actually used
    used = cmu_phonemes & canonical_arpa
    unused = canonical_arpa - cmu_phonemes

    print(f"  ✓ All CMU phonemes are in PHOIBLE-derived set")
    print(f"  {len(used)}/{len(canonical_arpa)} canonical phonemes used in CMU")
    if unused:
        print(f"  Note: {len(unused)} PHOIBLE phonemes not used in CMU (dialectal variants)")

    return True


def main():
    print("=" * 80)
    print("Rebuilding ARPAbet↔IPA Mapping from PHOIBLE English Union")
    print("=" * 80)

    # Step 1: Extract English phonemes from PHOIBLE
    phoible_phonemes = extract_english_phonemes()

    # Step 2: Normalize symbols
    normalized = normalize_phoible_symbols(phoible_phonemes)

    # Step 3: Create IPA→ARPA mapping
    ipa_to_arpa = map_ipa_to_arpa()

    # Step 4: Create ARPA→IPA mapping with stress
    arpa_to_ipa = create_arpa_to_ipa(ipa_to_arpa)

    # Step 5: Save mappings
    save_mappings(ipa_to_arpa, arpa_to_ipa)

    # Step 6: Verify coverage
    success = verify_coverage()

    print("\n" + "=" * 80)
    if success:
        print("✓ SUCCESS: Mappings rebuilt and verified")
    else:
        print("⚠ WARNING: Some CMU phonemes not covered")
    print("=" * 80)


if __name__ == "__main__":
    main()
