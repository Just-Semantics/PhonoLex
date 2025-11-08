"""
Central word filtering module for PhonoLex v2.3.

Implements the filtering criterion:
- Word must have frequency data (SUBTLEXus)

This ensures we only include real words from contemporary English,
filtering out archaic terms, typos, and nonsense while maximizing
vocabulary coverage (~50K words from CMU dictionary).

Additional psycholinguistic norms (concreteness, AoA, etc.) are loaded
and available when present, but not required for inclusion.
"""

import csv
from pathlib import Path

import pandas as pd


class WordFilter:
    """
    Central word filter for PhonoLex vocabulary selection.

    Usage:
        filter = WordFilter()
        filter.load_all_norms()
        if filter.should_include_word("cat"):
            # Include this word in database
    """

    def __init__(self):
        self.freq_words: set[str] = set()
        self.conc_words: set[str] = set()
        self.aoa_words: set[str] = set()
        self.img_words: set[str] = set()
        self.fam_words: set[str] = set()
        self.vad_words: set[str] = set()

        # Path from src/phonolex/ up to project root, then to data/
        self.data_dir = Path(__file__).parent.parent.parent / "data"
        self._loaded = False

    def load_all_norms(self):
        """Load all norm datasets"""
        if self._loaded:
            return

        print("Loading psycholinguistic norms for filtering...")
        self.load_frequency()
        self.load_concreteness()
        self.load_glasgow_norms()
        self.load_vad_ratings()
        self._loaded = True

        # Report statistics
        total_eligible = self.get_eligible_words()
        print("✓ Filtering criterion loaded:")
        print(f"  - {len(self.freq_words):,} words with frequency")
        print(f"  - {len(total_eligible):,} words meeting criterion (freq + any norm)")

    def load_frequency(self):
        """Load SUBTLEXus frequency data"""
        freq_path = self.data_dir / "subtlex_frequency.txt"

        with open(freq_path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                word = row["Word"].lower()
                if row["SUBTLWF"]:  # Has frequency data
                    self.freq_words.add(word)

    def load_concreteness(self):
        """Load concreteness ratings"""
        conc_path = self.data_dir / "norms" / "concreteness.txt"

        with open(conc_path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                word = row["Word"].lower()
                if row["Conc.M"]:  # Has concreteness rating
                    self.conc_words.add(word)

    def load_glasgow_norms(self):
        """Load Glasgow Norms: AoA, Imageability, Familiarity"""
        glasgow_path = self.data_dir / "norms" / "GlasgowNorms.xlsx"

        try:
            df = pd.read_excel(glasgow_path, header=1)

            for _, row in df.iterrows():
                word = str(row["word"]).lower()
                if pd.notna(row["M.6"]):  # AoA
                    self.aoa_words.add(word)
                if pd.notna(row["M.4"]):  # Imageability
                    self.img_words.add(word)
                if pd.notna(row["M.5"]):  # Familiarity
                    self.fam_words.add(word)
        except Exception as e:
            print(f"  Warning: Could not load Glasgow norms: {e}")

    def load_vad_ratings(self):
        """Load Valence-Arousal-Dominance ratings"""
        vad_path = self.data_dir / "norms" / "Ratings_VAD_WarrinerEtAl.csv"

        with open(vad_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                word = row["word"].lower()
                if row["valence"] or row["arousal"] or row["dominance"]:
                    self.vad_words.add(word)

    def should_include_word(self, word: str) -> bool:
        """
        Check if word meets inclusion criteria.

        Criterion: Must have frequency data (relaxed from v2.0).

        Args:
            word: Word string (will be lowercased)

        Returns:
            True if word should be included in database
        """
        if not self._loaded:
            raise RuntimeError("Must call load_all_norms() first")

        word = word.lower()

        # Only requirement: must have frequency data
        # This filters out archaic/nonsense words while maximizing coverage
        return word in self.freq_words

    def get_eligible_words(self) -> set[str]:
        """Get all words that meet the inclusion criterion"""
        if not self._loaded:
            raise RuntimeError("Must call load_all_norms() first")

        eligible = set()
        for word in self.freq_words:
            if self.should_include_word(word):
                eligible.add(word)

        return eligible

    def get_norm_coverage(self, word: str) -> dict[str, bool]:
        """
        Get which norms are available for a word.

        Args:
            word: Word string

        Returns:
            Dict mapping norm name to availability
        """
        word = word.lower()

        return {
            "frequency": word in self.freq_words,
            "concreteness": word in self.conc_words,
            "aoa": word in self.aoa_words,
            "imageability": word in self.img_words,
            "familiarity": word in self.fam_words,
            "vad": word in self.vad_words,
        }


# Singleton instance for convenience
_filter_instance = None


def get_word_filter() -> WordFilter:
    """Get singleton WordFilter instance"""
    global _filter_instance
    if _filter_instance is None:
        _filter_instance = WordFilter()
        _filter_instance.load_all_norms()
    return _filter_instance


def should_include_word(word: str) -> bool:
    """
    Convenience function to check if word should be included.

    Uses singleton filter instance.
    """
    return get_word_filter().should_include_word(word)


if __name__ == "__main__":
    # Test the filter
    print("=" * 80)
    print("Word Filter Test")
    print("=" * 80)

    filter = WordFilter()
    filter.load_all_norms()

    # Test some words
    test_words = [
        "cat",  # Should pass (common word with norms)
        "dog",  # Should pass
        "zygote",  # Might fail (technical, might lack norms)
        "john",  # Should fail (proper noun, likely no norms)
        "computerization",  # Might fail (rare, might lack norms)
    ]

    print("\nTesting sample words:")
    for word in test_words:
        included = filter.should_include_word(word)
        coverage = filter.get_norm_coverage(word)
        norm_count = sum(coverage.values()) - 1  # Exclude frequency from count

        print(f"\n  {word}:")
        print(f"    Include: {included}")
        print(f"    Norms available: {norm_count}")
        for norm, has_it in coverage.items():
            if has_it:
                print(f"      - {norm}")

    print("\n" + "=" * 80)
