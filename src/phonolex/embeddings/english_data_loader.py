#!/usr/bin/env python3
"""
English-Only Phonological Data Loader

Focused on English phonology for cleaner results.
"""

import csv
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class WordPronunciation:
    """A word with its pronunciation"""
    word: str
    phonemes: List[str]  # IPA phonemes


@dataclass
class PhonemeWithStress:
    """Phoneme with stress marker"""
    phoneme: str  # IPA
    stress: Optional[int] = None  # 0=unstressed, 1=primary, 2=secondary, None=consonant

    def __str__(self):
        if self.stress is not None:
            return f"{self.phoneme}{self.stress}"
        return self.phoneme


@dataclass
class MorphologicalPair:
    """Lemma + inflection for learning allomorphy"""
    lemma: str
    inflected: str
    features: str  # e.g., "N;PL", "V;PST"
    lemma_phonemes: Optional[List[str]] = None
    inflected_phonemes: Optional[List[str]] = None


class EnglishPhonologyLoader:
    """
    Load English phonological data only

    Sources:
    - CMU Dict (134K words)
    - SIGMORPHON English morphology
    - English phoneme set from Phoible
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)

        # Mappings
        self.arpa_to_ipa = self._load_arpa_mappings()

        # Data
        self.lexicon: Dict[str, List[str]] = {}  # word -> phonemes (IPA, stress stripped)
        self.lexicon_with_stress: Dict[str, List[PhonemeWithStress]] = {}  # word -> phonemes with stress
        self.morphology: List[MorphologicalPair] = []
        self.english_phonemes: set = set()

        print("\n" + "=" * 70)
        print("ENGLISH PHONOLOGY DATA LOADER")
        print("=" * 70)

        self._load_cmu_dict()
        self._load_sigmorphon()
        self._get_phoneme_inventory()

    def _load_arpa_mappings(self) -> Dict[str, str]:
        """Load ARPAbet to IPA mappings"""
        with open(self.data_dir / "mappings" / "arpa_to_ipa.json") as f:
            return json.load(f)

    def _arpa_to_ipa_sequence(self, arpa_phones: List[str]) -> List[str]:
        """Convert ARPAbet sequence to IPA (stress stripped)"""
        ipa = []
        for phone in arpa_phones:
            # Remove stress markers
            base = phone.rstrip('012')
            if base in self.arpa_to_ipa:
                ipa.append(self.arpa_to_ipa[base])
        return ipa

    def _arpa_to_ipa_with_stress(self, arpa_phones: List[str]) -> List[PhonemeWithStress]:
        """Convert ARPAbet sequence to IPA with stress preserved"""
        ipa_with_stress = []
        for phone in arpa_phones:
            # Extract stress marker if present
            stress = None
            if phone and phone[-1] in '012':
                stress = int(phone[-1])
                base = phone[:-1]
            else:
                base = phone

            if base in self.arpa_to_ipa:
                ipa_with_stress.append(PhonemeWithStress(
                    phoneme=self.arpa_to_ipa[base],
                    stress=stress
                ))
        return ipa_with_stress

    def _load_cmu_dict(self):
        """Load CMU pronouncing dictionary"""
        print("\nLoading CMU Dict...")

        cmu_path = self.data_dir / "cmu" / "cmudict-0.7b"

        with open(cmu_path, encoding='latin-1') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(';;;'):
                    continue

                parts = line.split()
                word = parts[0].lower()

                # Remove variant markers (word(2), etc.)
                if '(' in word:
                    word = word.split('(')[0]

                # Convert ARPAbet to IPA (both with and without stress)
                arpa_phones = parts[1:]
                ipa_phones = self._arpa_to_ipa_sequence(arpa_phones)
                ipa_phones_with_stress = self._arpa_to_ipa_with_stress(arpa_phones)

                if ipa_phones:
                    self.lexicon[word] = ipa_phones
                    self.lexicon_with_stress[word] = ipa_phones_with_stress

        print(f"  ✓ Loaded {len(self.lexicon):,} words")

    def _load_sigmorphon(self):
        """Load SIGMORPHON English morphology"""
        print("\nLoading SIGMORPHON English morphology...")

        sig_path = self.data_dir / "learning_datasets" / "sigmorphon2020" / "task0" / "data" / "germanic" / "eng.trn"

        if not sig_path.exists():
            print("  ✗ SIGMORPHON data not found")
            return

        with open(sig_path) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) != 3:
                    continue

                lemma, inflected, features = parts

                # Get pronunciations from lexicon
                lemma_phones = self.lexicon.get(lemma.lower())
                inflected_phones = self.lexicon.get(inflected.lower())

                self.morphology.append(MorphologicalPair(
                    lemma=lemma.lower(),
                    inflected=inflected.lower(),
                    features=features,
                    lemma_phonemes=lemma_phones,
                    inflected_phonemes=inflected_phones
                ))

        print(f"  ✓ Loaded {len(self.morphology):,} morphological pairs")

        # Count how many have pronunciations
        with_phones = sum(1 for m in self.morphology if m.lemma_phonemes and m.inflected_phonemes)
        print(f"  ✓ {with_phones:,} pairs have pronunciations ({100*with_phones/len(self.morphology):.1f}%)")

    def _get_phoneme_inventory(self):
        """Get English phoneme inventory"""
        print("\nExtracting English phoneme inventory...")

        # Get all phonemes from lexicon
        for phonemes in self.lexicon.values():
            self.english_phonemes.update(phonemes)

        print(f"  ✓ Found {len(self.english_phonemes)} unique phonemes in English lexicon")
        print(f"  Phonemes: {sorted(self.english_phonemes)[:20]}...")

    def get_allomorph_data(self, feature_type: str = "N;PL") -> List[Tuple[str, str, str]]:
        """
        Get data for allomorph prediction

        Args:
            feature_type: Morphological feature (e.g., "N;PL" for plural, "V;PST" for past tense)

        Returns:
            List of (lemma, final_phoneme, allomorph) tuples
        """
        allomorph_data = []

        for pair in self.morphology:
            if pair.features != feature_type:
                continue

            if not pair.lemma_phonemes or not pair.inflected_phonemes:
                continue

            # Get final phoneme of lemma
            final_phoneme = pair.lemma_phonemes[-1]

            # Extract allomorph (difference between inflected and lemma)
            lemma_len = len(pair.lemma_phonemes)
            inflected_len = len(pair.inflected_phonemes)

            if inflected_len > lemma_len:
                # Simple suffixation
                allomorph_phones = pair.inflected_phonemes[lemma_len:]
                allomorph = ''.join(allomorph_phones)

                allomorph_data.append((pair.lemma, final_phoneme, allomorph))

        return allomorph_data

    def get_phonological_neighbors(self, word: str, max_distance: int = 1) -> List[Tuple[str, int]]:
        """
        Find phonological neighbors (differ by max_distance phonemes)

        Args:
            word: Target word
            max_distance: Maximum edit distance

        Returns:
            List of (neighbor_word, distance) tuples
        """
        if word not in self.lexicon:
            return []

        target_phones = self.lexicon[word]
        neighbors = []

        for other_word, other_phones in self.lexicon.items():
            if other_word == word:
                continue

            # Simple edit distance
            dist = self._edit_distance(target_phones, other_phones)
            if dist <= max_distance:
                neighbors.append((other_word, dist))

        return sorted(neighbors, key=lambda x: x[1])

    def _edit_distance(self, seq1: List[str], seq2: List[str]) -> int:
        """Compute edit distance between two sequences"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

        return dp[m][n]

    def summary(self):
        """Print data summary"""
        print("\n" + "=" * 70)
        print("DATA SUMMARY")
        print("=" * 70)
        print(f"\nLexicon: {len(self.lexicon):,} words")
        print(f"Morphology: {len(self.morphology):,} pairs")
        print(f"Phonemes: {len(self.english_phonemes)} unique")

        # Allomorph counts
        print("\nAllomorph data:")
        for feat in ["N;PL", "V;PST", "V;3;SG;PRS"]:
            data = self.get_allomorph_data(feat)
            if data:
                allomorphs = set(a for _, _, a in data)
                print(f"  {feat}: {len(data)} examples, {len(allomorphs)} unique allomorphs")


if __name__ == '__main__':
    # Demo
    loader = EnglishPhonologyLoader()
    loader.summary()

    # Test allomorph extraction
    print("\n" + "=" * 70)
    print("PLURAL ALLOMORPH EXAMPLES")
    print("=" * 70)

    plural_data = loader.get_allomorph_data("N;PL")[:20]
    for lemma, final_phone, allomorph in plural_data:
        print(f"  {lemma:15s} [/{final_phone}/] → +{allomorph}")

    # Test phonological neighbors
    print("\n" + "=" * 70)
    print("PHONOLOGICAL NEIGHBORS OF 'cat'")
    print("=" * 70)

    neighbors = loader.get_phonological_neighbors('cat', max_distance=1)[:10]
    for neighbor, dist in neighbors:
        phones = ''.join(loader.lexicon[neighbor])
        print(f"  {neighbor:15s} /{phones}/ (distance: {dist})")
