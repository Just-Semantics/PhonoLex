#!/usr/bin/env python3
"""
Maximal Opposition Intervention Approach

Based on Gierut's research (1989-1992) as described in Storkel (2022).

Key principles:
1. Select TWO UNKNOWN sounds (not target + substitute)
2. Sounds must differ by MAJOR CLASS (sonorant vs. obstruent)
3. Sounds should differ by MAXIMAL number of distinctive features

This leads to system-wide phonological learning beyond the specific targets.

References:
- Gierut, J. A. (1990). Differential learning of phonological oppositions.
  Journal of Speech and Hearing Research, 33(3), 540-549.
- Storkel, H. L. (2022). Minimal, Maximal, or Multiple: Which Contrastive
  Intervention Approach to Use With Children With Speech Sound Disorders?
  Language, Speech, and Hearing Services in Schools, 53, 632-645.

Author: PhonoLex
Date: 2025-01-06
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np


# Feature order from Phoible (38 features)
PHOIBLE_FEATURES = [
    'tone', 'stress', 'syllabic', 'short', 'long',
    'consonantal', 'sonorant', 'continuant', 'delayedRelease', 'approximant',
    'tap', 'trill', 'nasal', 'lateral',
    'labial', 'round', 'labiodental',
    'coronal', 'anterior', 'distributed', 'strident',
    'dorsal', 'high', 'low', 'front', 'back', 'tense',
    'retractedTongueRoot', 'advancedTongueRoot',
    'periodicGlottalSource', 'epilaryngealSource',
    'spreadGlottis', 'constrictedGlottis', 'fortis',
    'raisedLarynxEjective', 'loweredLarynxImplosive', 'click'
]


@dataclass
class MaximalOppositionPair:
    """A pair of phonemes suitable for maximal opposition intervention"""
    phoneme1: str
    phoneme2: str

    # Classification
    major_class_diff: bool  # True if one is sonorant, other is obstruent

    # Feature differences
    num_feature_diffs: int  # Total number of differing features
    feature_diffs: List[str]  # Names of differing features

    # Similarity score (higher = better for maximal opposition)
    maximal_opposition_score: float

    def __repr__(self):
        return (f"MaximalOppositionPair(/{self.phoneme1}/ - /{self.phoneme2}/, "
                f"major_class={self.major_class_diff}, "
                f"feature_diffs={self.num_feature_diffs}, "
                f"score={self.maximal_opposition_score:.2f})")


class MaximalOppositionGenerator:
    """
    Generate maximal opposition pairs for phonological intervention

    Based on Gierut's research showing that pairing two unknown sounds
    that differ by major class and maximal features leads to better
    system-wide generalization than conventional minimal pairs.
    """

    def __init__(self, phoible_data_path: Optional[str] = None):
        """
        Initialize generator with phonological features

        Args:
            phoible_data_path: Path to phoible-segments-features.tsv
                             If None, uses default location
        """
        if phoible_data_path is None:
            # Default to bundled data
            repo_root = Path(__file__).parent.parent.parent.parent
            phoible_data_path = repo_root / 'data' / 'phoible' / 'phoible-segments-features.tsv'

        self.phoneme_features = self._load_features(phoible_data_path)

    def _load_features(self, tsv_path: Path) -> Dict[str, Dict[str, str]]:
        """Load Phoible features for all phonemes"""
        features = {}

        with open(tsv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                segment = row['segment']
                # Store all features
                features[segment] = {feat: row[feat] for feat in PHOIBLE_FEATURES}

        return features

    def is_consonant(self, phoneme: str) -> bool:
        """Check if phoneme is a consonant (not a vowel)"""
        if phoneme not in self.phoneme_features:
            return False

        # Consonantal = + means it's a consonant
        return self.phoneme_features[phoneme]['consonantal'] == '+'

    def is_sonorant(self, phoneme: str) -> bool:
        """
        Check if consonant is a sonorant (nasals, liquids, glides)

        Note: This returns the raw sonorant feature. For therapy purposes,
        we typically focus on sonorant CONSONANTS (not vowels).
        """
        if phoneme not in self.phoneme_features:
            return False

        return self.phoneme_features[phoneme]['sonorant'] == '+'

    def is_obstruent(self, phoneme: str) -> bool:
        """
        Check if consonant is an obstruent (stops, fricatives, affricates)

        Obstruents are non-sonorant consonants.
        """
        if phoneme not in self.phoneme_features:
            return False

        feats = self.phoneme_features[phoneme]
        return feats['consonantal'] == '+' and feats['sonorant'] == '-'

    def has_major_class_difference(self, phoneme1: str, phoneme2: str) -> bool:
        """
        Check if two phonemes differ by major class

        For therapy purposes (Gierut's research):
        - Major class difference = one sonorant consonant, one obstruent

        This distinction is clinically meaningful because it captures
        a wide range of phonological patterns children struggle with.
        """
        if phoneme1 not in self.phoneme_features or phoneme2 not in self.phoneme_features:
            return False

        # Both must be consonants
        if not (self.is_consonant(phoneme1) and self.is_consonant(phoneme2)):
            return False

        # One must be sonorant, the other obstruent
        son1 = self.is_sonorant(phoneme1)
        son2 = self.is_sonorant(phoneme2)

        return son1 != son2

    def count_feature_differences(self, phoneme1: str, phoneme2: str) -> Tuple[int, List[str]]:
        """
        Count how many distinctive features differ between two phonemes

        Returns:
            (num_diffs, list_of_differing_features)
        """
        if phoneme1 not in self.phoneme_features or phoneme2 not in self.phoneme_features:
            return (0, [])

        feats1 = self.phoneme_features[phoneme1]
        feats2 = self.phoneme_features[phoneme2]

        diffs = []
        for feat_name in PHOIBLE_FEATURES:
            val1 = feats1[feat_name]
            val2 = feats2[feat_name]

            # Count as different if values don't match
            # Note: '0' (not applicable) vs. '+' or '-' counts as different
            if val1 != val2:
                diffs.append(feat_name)

        return (len(diffs), diffs)

    def calculate_maximal_opposition_score(
        self,
        phoneme1: str,
        phoneme2: str
    ) -> float:
        """
        Calculate a score indicating how suitable this pair is for
        maximal opposition intervention

        Higher scores = better candidates for maximal opposition

        Scoring:
        - Major class difference: +100 points (REQUIRED)
        - Feature differences: +1 point per difference

        Returns:
            Score (0-138 range, where 138 = major class + all 38 features different)
        """
        score = 0.0

        # CRITICAL: Major class difference
        if self.has_major_class_difference(phoneme1, phoneme2):
            score += 100.0
        else:
            # Without major class difference, not suitable for maximal opposition
            return 0.0

        # Feature differences
        num_diffs, _ = self.count_feature_differences(phoneme1, phoneme2)
        score += num_diffs

        return score

    def generate_pairs(
        self,
        unknown_sounds: List[str],
        known_sounds: Optional[List[str]] = None,
        min_score: float = 100.0,
        top_n: int = 10
    ) -> List[MaximalOppositionPair]:
        """
        Generate maximal opposition pairs from unknown sounds

        Args:
            unknown_sounds: Sounds the child cannot produce
            known_sounds: Sounds the child can produce (optional, used to filter)
            min_score: Minimum maximal opposition score (default 100 = major class required)
            top_n: Return top N pairs by score

        Returns:
            List of MaximalOppositionPair objects, sorted by score (best first)
        """
        pairs = []

        # Only consider pairs where BOTH sounds are unknown
        for i, p1 in enumerate(unknown_sounds):
            for p2 in unknown_sounds[i+1:]:
                # Skip if either is known
                if known_sounds and (p1 in known_sounds or p2 in known_sounds):
                    continue

                # Calculate score
                score = self.calculate_maximal_opposition_score(p1, p2)

                if score >= min_score:
                    num_diffs, diff_list = self.count_feature_differences(p1, p2)
                    major_class = self.has_major_class_difference(p1, p2)

                    pair = MaximalOppositionPair(
                        phoneme1=p1,
                        phoneme2=p2,
                        major_class_diff=major_class,
                        num_feature_diffs=num_diffs,
                        feature_diffs=diff_list,
                        maximal_opposition_score=score
                    )
                    pairs.append(pair)

        # Sort by score (best first)
        pairs.sort(key=lambda p: p.maximal_opposition_score, reverse=True)

        return pairs[:top_n]

    def analyze_pair(self, phoneme1: str, phoneme2: str) -> None:
        """
        Print detailed analysis of a phoneme pair

        Useful for understanding why a pair is/isn't suitable for
        maximal opposition intervention.
        """
        print(f"\n{'='*70}")
        print(f"MAXIMAL OPPOSITION ANALYSIS: /{phoneme1}/ vs. /{phoneme2}/")
        print(f"{'='*70}")

        # Check existence
        if phoneme1 not in self.phoneme_features:
            print(f"ERROR: /{phoneme1}/ not found in Phoible database")
            return
        if phoneme2 not in self.phoneme_features:
            print(f"ERROR: /{phoneme2}/ not found in Phoible database")
            return

        # Classification
        print(f"\nCLASSIFICATION:")
        print(f"  /{phoneme1}/: ", end="")
        if self.is_consonant(phoneme1):
            if self.is_sonorant(phoneme1):
                print("Sonorant consonant (nasal, liquid, glide)")
            else:
                print("Obstruent (stop, fricative, affricate)")
        else:
            print("Vowel (not suitable for maximal opposition)")

        print(f"  /{phoneme2}/: ", end="")
        if self.is_consonant(phoneme2):
            if self.is_sonorant(phoneme2):
                print("Sonorant consonant (nasal, liquid, glide)")
            else:
                print("Obstruent (stop, fricative, affricate)")
        else:
            print("Vowel (not suitable for maximal opposition)")

        # Major class difference
        major_class = self.has_major_class_difference(phoneme1, phoneme2)
        print(f"\nMAJOR CLASS DIFFERENCE: {'✓ YES' if major_class else '✗ NO'}")
        if not major_class:
            print("  → Not suitable for maximal opposition (requires major class difference)")

        # Feature differences
        num_diffs, diff_list = self.count_feature_differences(phoneme1, phoneme2)
        print(f"\nFEATURE DIFFERENCES: {num_diffs}/38")
        if num_diffs > 0:
            print(f"  Differing features: {', '.join(diff_list[:10])}")
            if len(diff_list) > 10:
                print(f"  ... and {len(diff_list) - 10} more")

        # Overall score
        score = self.calculate_maximal_opposition_score(phoneme1, phoneme2)
        print(f"\nMAXIMAL OPPOSITION SCORE: {score:.1f}")
        if score >= 100:
            print(f"  → ✓ EXCELLENT candidate for maximal opposition")
            print(f"     (Major class difference + {num_diffs} feature differences)")
        else:
            print(f"  → ✗ NOT recommended for maximal opposition")
            print(f"     (Lacks major class difference)")


@dataclass
class MinimalPairWordList:
    """Word lists for maximal opposition intervention"""
    phoneme1: str
    phoneme2: str
    position: str  # 'initial', 'medial', 'final'
    word_pairs: List[Tuple[str, str]] = field(default_factory=list)

    def __repr__(self):
        return (f"MinimalPairWordList(/{self.phoneme1}/ - /{self.phoneme2}/ "
                f"in {self.position} position, {len(self.word_pairs)} pairs)")


def generate_word_lists(
    phoneme_pair: MaximalOppositionPair,
    lexicon: Dict[str, List[str]],
    position: str = 'initial',
    max_pairs: int = 10
) -> MinimalPairWordList:
    """
    Generate minimal pair word lists for a maximal opposition phoneme pair

    Args:
        phoneme_pair: The maximal opposition pair to generate words for
        lexicon: Dictionary mapping words to phoneme lists (IPA)
        position: 'initial', 'medial', or 'final'
        max_pairs: Maximum number of word pairs to generate

    Returns:
        MinimalPairWordList with word pairs

    Example:
        For /θ/-/r/ in initial position:
        - thick/rick, thank/rank, thought/rot, etc.
    """
    p1 = phoneme_pair.phoneme1
    p2 = phoneme_pair.phoneme2

    # Group words by length for efficient comparison
    by_length = defaultdict(list)
    for word, phonemes in lexicon.items():
        if len(phonemes) > 0:
            by_length[len(phonemes)].append((word, phonemes))

    word_pairs = []

    # Find minimal pairs
    for length, words in by_length.items():
        if length < 2:  # Need at least 2 phonemes
            continue

        # Compare all pairs of same length
        for i, (word1, phonemes1) in enumerate(words):
            for word2, phonemes2 in words[i+1:]:
                # Check if they differ by exactly one phoneme
                diff_positions = [
                    idx for idx, (ph1, ph2) in enumerate(zip(phonemes1, phonemes2))
                    if ph1 != ph2
                ]

                # Must differ by exactly one phoneme
                if len(diff_positions) != 1:
                    continue

                diff_pos = diff_positions[0]

                # Check position match
                if position == 'initial' and diff_pos != 0:
                    continue
                elif position == 'final' and diff_pos != len(phonemes1) - 1:
                    continue
                elif position == 'medial' and (diff_pos == 0 or diff_pos == len(phonemes1) - 1):
                    continue

                # Check if the differing phonemes match our target pair
                ph1_at_pos = phonemes1[diff_pos]
                ph2_at_pos = phonemes2[diff_pos]

                if (ph1_at_pos == p1 and ph2_at_pos == p2) or \
                   (ph1_at_pos == p2 and ph2_at_pos == p1):
                    word_pairs.append((word1, word2))

                    if len(word_pairs) >= max_pairs:
                        break

            if len(word_pairs) >= max_pairs:
                break

        if len(word_pairs) >= max_pairs:
            break

    return MinimalPairWordList(
        phoneme1=p1,
        phoneme2=p2,
        position=position,
        word_pairs=word_pairs
    )


def main():
    """Demo: Generate maximal opposition pairs and word lists"""

    print("="*70)
    print("MAXIMAL OPPOSITION GENERATOR")
    print("Based on Gierut (1989-1992) and Storkel (2022)")
    print("="*70)

    # Initialize generator
    generator = MaximalOppositionGenerator()

    # Example: Child with severe SSD (from Storkel 2022 paper)
    # These are sounds the child produces incorrectly
    unknown_sounds = ['g', 'θ', 'ð', 'ʃ', 'ʤ', 'ŋ', 'l', 'r']

    print(f"\nUnknown sounds (child cannot produce): {', '.join(unknown_sounds)}")
    print(f"\nGenerating maximal opposition pairs...")

    # Generate pairs
    pairs = generator.generate_pairs(
        unknown_sounds=unknown_sounds,
        top_n=10
    )

    print(f"\n{'='*70}")
    print(f"TOP MAXIMAL OPPOSITION PAIRS")
    print(f"{'='*70}\n")

    for i, pair in enumerate(pairs, 1):
        print(f"{i}. /{pair.phoneme1}/ - /{pair.phoneme2}/")
        print(f"   Score: {pair.maximal_opposition_score:.1f}")
        print(f"   Major class difference: {'✓' if pair.major_class_diff else '✗'}")
        print(f"   Feature differences: {pair.num_feature_diffs}/38")
        print()

    # Detailed analysis of top pair
    if pairs:
        top_pair = pairs[0]
        generator.analyze_pair(top_pair.phoneme1, top_pair.phoneme2)

    # Compare to conventional minimal pair (for illustration)
    print(f"\n{'='*70}")
    print("COMPARISON: Conventional Minimal Pair")
    print(f"{'='*70}")
    print("\nConventional minimal pair would pair target with its substitute:")
    print("Example: /θ/ (target) with /t/ (substitute)")
    generator.analyze_pair('θ', 't')

    # Generate word lists
    print(f"\n{'='*70}")
    print("WORD LIST GENERATION")
    print(f"{'='*70}")
    print("\nLoading CMU dictionary...")

    try:
        # Load lexicon
        from ..embeddings.english_data_loader import EnglishPhonologyLoader
        loader = EnglishPhonologyLoader()
        lexicon = loader.lexicon  # word -> List[str] (IPA phonemes)

        print(f"Loaded {len(lexicon):,} words")

        # Generate word lists for top 3 pairs
        print(f"\nGenerating word lists for top 3 maximal opposition pairs...\n")

        for i, pair in enumerate(pairs[:3], 1):
            print(f"\n{i}. /{pair.phoneme1}/ - /{pair.phoneme2}/ (Score: {pair.maximal_opposition_score:.1f})")

            # Try initial position first
            word_list = generate_word_lists(
                phoneme_pair=pair,
                lexicon=lexicon,
                position='initial',
                max_pairs=5
            )

            if len(word_list.word_pairs) > 0:
                print(f"   Initial position ({len(word_list.word_pairs)} pairs):")
                for w1, w2 in word_list.word_pairs[:5]:
                    print(f"     {w1} - {w2}")
            else:
                print(f"   Initial position: No pairs found")

                # Try medial if initial fails
                word_list = generate_word_lists(
                    phoneme_pair=pair,
                    lexicon=lexicon,
                    position='medial',
                    max_pairs=5
                )

                if len(word_list.word_pairs) > 0:
                    print(f"   Medial position ({len(word_list.word_pairs)} pairs):")
                    for w1, w2 in word_list.word_pairs[:5]:
                        print(f"     {w1} - {w2}")

    except ImportError:
        print("Could not load English lexicon (EnglishPhonologyLoader not available)")

    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    print("\nMaximal opposition pairs (like /g/-/l/ or /θ/-/r/) are predicted to")
    print("produce better system-wide generalization than conventional minimal")
    print("pairs (like /θ/-/t/) for children with moderate-to-severe SSD.")


if __name__ == '__main__':
    main()
