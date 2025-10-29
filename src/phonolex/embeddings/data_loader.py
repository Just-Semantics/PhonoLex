#!/usr/bin/env python3
"""
Unified Data Loader for Phoneme Embedding Training

Combines multiple data sources to create training examples:
1. Phonological context (CMU Dict) - like Word2Vec
2. Morphological patterns (SIGMORPHON) - allomorph prediction
3. Phonological features (Phoible) - initialization + reconstruction
4. Cross-linguistic structure (Phoible inventories) - universals

This is THE foundation for learning phoneme embeddings.
"""

import csv
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterator
from collections import defaultdict, Counter
from dataclasses import dataclass


@dataclass
class PhonemeContextExample:
    """Example for context prediction (skip-gram style)"""
    center_phoneme: str
    context_phonemes: List[str]
    word: str  # For debugging


@dataclass
class MorphologyExample:
    """Example for morphological pattern learning"""
    lemma: str
    inflected: str
    lemma_phonemes: List[str]
    inflected_phonemes: List[str]
    stem_final_phoneme: str  # Key for allomorph selection
    morphological_features: str  # e.g., "V;PST", "N;PL"
    allomorph: str  # e.g., "-ed", "-s"


@dataclass
class ContrastivePair:
    """Pair of phonemes with similarity label"""
    phoneme1: str
    phoneme2: str
    similarity: float  # 0.0 to 1.0


@dataclass
class InventoryExample:
    """Phoneme inventory for a language"""
    language: str
    inventory_id: int
    phonemes: List[str]


class PhonemeEmbeddingDataLoader:
    """
    Unified data loader for phoneme embedding training

    Loads and combines:
    - Phoible phoneme features (38-dim)
    - CMU Dict pronunciations (context)
    - SIGMORPHON morphology (patterns)
    - ARPAbet <-> IPA mappings
    """

    def __init__(
        self,
        data_dir: str = "data",
        learning_datasets_dir: str = "data/learning_datasets"
    ):
        self.data_dir = Path(data_dir)
        self.learning_dir = Path(learning_datasets_dir)

        # Storage
        self.phoneme_to_features = {}  # phoneme -> 38-dim features
        self.phoneme_to_id = {}  # phoneme -> unique ID
        self.id_to_phoneme = {}
        self.arpa_to_ipa = {}
        self.ipa_to_arpa = {}

        # Statistics
        self.stats = {
            'total_phonemes': 0,
            'total_languages': 0,
            'context_examples': 0,
            'morphology_examples': 0,
            'contrastive_pairs': 0,
            'inventories': 0
        }

        print("=" * 70)
        print("PHONEME EMBEDDING DATA LOADER")
        print("=" * 70)

        # Load everything
        self._load_mappings()
        self._load_phoible()
        self._build_phoneme_vocabulary()

        print(f"\n✓ Loaded {self.stats['total_phonemes']} unique phonemes")
        print(f"✓ From {self.stats['total_languages']} languages")

    def _load_mappings(self):
        """Load ARPAbet <-> IPA mappings"""
        print("\nLoading ARPAbet <-> IPA mappings...")

        arpa_file = self.data_dir / "mappings" / "arpa_to_ipa.json"
        ipa_file = self.data_dir / "mappings" / "ipa_to_arpa.json"

        with open(arpa_file) as f:
            self.arpa_to_ipa = json.load(f)

        with open(ipa_file) as f:
            self.ipa_to_arpa = json.load(f)

        print(f"  ✓ Loaded {len(self.arpa_to_ipa)} ARPAbet->IPA mappings")

    def _load_phoible(self):
        """Load Phoible phoneme features"""
        print("\nLoading Phoible phoneme features...")

        phoible_file = self.data_dir / "phoible" / "phoible.csv"

        # Feature columns (38 features)
        feature_cols = [
            'tone', 'stress', 'syllabic', 'short', 'long',
            'consonantal', 'sonorant', 'continuant', 'delayedRelease', 'approximant',
            'tap', 'trill', 'nasal', 'lateral',
            'labial', 'round', 'labiodental',
            'coronal', 'anterior', 'distributed', 'strident',
            'dorsal', 'high', 'low', 'front', 'back', 'tense',
            'retractedTongueRoot', 'advancedTongueRoot',
            'periodicGlottalSource', 'epilaryngealSource',
            'spreadGlottis', 'constrictedGlottis',
            'fortis', 'lenis',
            'raisedLarynxEjective', 'loweredLarynxImplosive', 'click'
        ]

        languages = set()
        phoneme_count = 0

        with open(phoible_file, encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                phoneme = row['Phoneme']
                lang = row['LanguageName']
                languages.add(lang)

                # Extract features (handle trajectories by taking first value)
                features = []
                for feat in feature_cols:
                    value = row[feat]
                    # Handle trajectory features (e.g., "+,-")
                    if ',' in value:
                        value = value.split(',')[0]  # Take first value for now

                    # Convert to float: + -> 1.0, - -> -1.0, 0 -> 0.0
                    if value == '+':
                        features.append(1.0)
                    elif value == '-':
                        features.append(-1.0)
                    elif value == '0':
                        features.append(0.0)
                    else:
                        features.append(0.0)  # Default

                # Store (use first occurrence per phoneme for now)
                if phoneme not in self.phoneme_to_features:
                    self.phoneme_to_features[phoneme] = np.array(features, dtype=np.float32)
                    phoneme_count += 1

        self.stats['total_phonemes'] = phoneme_count
        self.stats['total_languages'] = len(languages)

        print(f"  ✓ Loaded features for {phoneme_count} unique phonemes")
        print(f"  ✓ From {len(languages)} languages")

    def _build_phoneme_vocabulary(self):
        """Build phoneme -> ID mapping"""
        print("\nBuilding phoneme vocabulary...")

        for i, phoneme in enumerate(sorted(self.phoneme_to_features.keys())):
            self.phoneme_to_id[phoneme] = i
            self.id_to_phoneme[i] = phoneme

        print(f"  ✓ Vocabulary size: {len(self.phoneme_to_id)}")

    def get_context_examples(self, window_size: int = 2) -> Iterator[PhonemeContextExample]:
        """
        Generate phonological context examples from CMU Dict

        Like Word2Vec skip-gram: predict context phonemes from center phoneme

        Args:
            window_size: Context window on each side

        Yields:
            PhonemeContextExample objects
        """
        print(f"\nGenerating context examples (window={window_size})...")

        cmu_file = self.data_dir / "cmu" / "cmudict-0.7b"
        count = 0

        with open(cmu_file, encoding='latin-1') as f:
            for line in f:
                if line.startswith(';;;'):
                    continue

                parts = line.strip().split()
                if len(parts) < 2:
                    continue

                word = parts[0]
                arpa_phones = parts[1:]

                # Convert ARPAbet to IPA
                ipa_phones = []
                for arpa in arpa_phones:
                    # Remove stress markers
                    base = arpa.rstrip('012')
                    if base in self.arpa_to_ipa:
                        ipa = self.arpa_to_ipa[base]
                        # Only keep if in our vocabulary
                        if ipa in self.phoneme_to_id:
                            ipa_phones.append(ipa)

                if len(ipa_phones) < 2:
                    continue

                # Generate context examples
                for i, center in enumerate(ipa_phones):
                    # Get context window
                    start = max(0, i - window_size)
                    end = min(len(ipa_phones), i + window_size + 1)
                    context = ipa_phones[start:i] + ipa_phones[i+1:end]

                    if context:
                        yield PhonemeContextExample(
                            center_phoneme=center,
                            context_phonemes=context,
                            word=word
                        )
                        count += 1

        self.stats['context_examples'] = count
        print(f"  ✓ Generated {count:,} context examples")

    def get_morphology_examples(self) -> Iterator[MorphologyExample]:
        """
        Generate morphological pattern examples from SIGMORPHON

        For learning allomorph selection rules

        Yields:
            MorphologyExample objects
        """
        print("\nLoading SIGMORPHON morphology examples...")

        sig_file = self.learning_dir / "sigmorphon2020/task0/data/germanic/eng.trn"

        if not sig_file.exists():
            print("  ⚠ SIGMORPHON data not found, skipping")
            return

        count = 0

        with open(sig_file) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) != 3:
                    continue

                lemma, inflected, features = parts

                # TODO: Get pronunciations and extract allomorph
                # For now, just yield the structure
                # This will be enhanced with pronunciation lookup

                yield MorphologyExample(
                    lemma=lemma,
                    inflected=inflected,
                    lemma_phonemes=[],  # Will add pronunciation lookup
                    inflected_phonemes=[],
                    stem_final_phoneme="",
                    morphological_features=features,
                    allomorph=""
                )
                count += 1

        self.stats['morphology_examples'] = count
        print(f"  ✓ Loaded {count:,} morphology examples")

    def get_contrastive_pairs(self, num_pairs: int = 100000) -> Iterator[ContrastivePair]:
        """
        Generate contrastive phoneme pairs for metric learning

        Positive pairs: similar phonemes (same natural class)
        Negative pairs: dissimilar phonemes (different classes)

        Args:
            num_pairs: Number of pairs to generate

        Yields:
            ContrastivePair objects
        """
        print(f"\nGenerating {num_pairs:,} contrastive pairs...")

        phonemes = list(self.phoneme_to_features.keys())
        count = 0

        for _ in range(num_pairs):
            # Sample two random phonemes
            p1, p2 = np.random.choice(phonemes, size=2, replace=False)

            # Compute similarity from features (cosine similarity)
            f1 = self.phoneme_to_features[p1]
            f2 = self.phoneme_to_features[p2]

            # Cosine similarity
            sim = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-8)
            # Map from [-1, 1] to [0, 1]
            sim = (sim + 1.0) / 2.0

            yield ContrastivePair(
                phoneme1=p1,
                phoneme2=p2,
                similarity=float(sim)
            )
            count += 1

        self.stats['contrastive_pairs'] = count
        print(f"  ✓ Generated {count:,} contrastive pairs")

    def get_inventory_examples(self) -> Iterator[InventoryExample]:
        """
        Generate phoneme inventory examples for cross-linguistic learning

        Yields:
            InventoryExample objects
        """
        print("\nLoading phoneme inventories...")

        phoible_file = self.data_dir / "phoible" / "phoible.csv"

        # Group by inventory
        inventories = defaultdict(lambda: {'language': '', 'phonemes': []})

        with open(phoible_file, encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                inv_id = int(row['InventoryID'])
                lang = row['LanguageName']
                phoneme = row['Phoneme']

                if phoneme in self.phoneme_to_id:
                    inventories[inv_id]['language'] = lang
                    inventories[inv_id]['phonemes'].append(phoneme)

        count = 0
        for inv_id, data in inventories.items():
            yield InventoryExample(
                language=data['language'],
                inventory_id=inv_id,
                phonemes=data['phonemes']
            )
            count += 1

        self.stats['inventories'] = count
        print(f"  ✓ Loaded {count} phoneme inventories")

    def get_phoneme_features(self, phoneme: str) -> Optional[np.ndarray]:
        """Get 38-dim Phoible features for a phoneme"""
        return self.phoneme_to_features.get(phoneme)

    def get_phoneme_id(self, phoneme: str) -> Optional[int]:
        """Get unique ID for a phoneme"""
        return self.phoneme_to_id.get(phoneme)

    def get_all_phonemes(self) -> List[str]:
        """Get list of all phonemes"""
        return list(self.phoneme_to_id.keys())

    def get_feature_matrix(self) -> np.ndarray:
        """
        Get feature matrix for all phonemes

        Returns:
            Array of shape (num_phonemes, 38)
        """
        num_phonemes = len(self.phoneme_to_id)
        features = np.zeros((num_phonemes, 38), dtype=np.float32)

        for phoneme, idx in self.phoneme_to_id.items():
            features[idx] = self.phoneme_to_features[phoneme]

        return features

    def print_stats(self):
        """Print dataset statistics"""
        print("\n" + "=" * 70)
        print("DATASET STATISTICS")
        print("=" * 70)
        print(f"\nPhonemes:")
        print(f"  Total unique phonemes: {self.stats['total_phonemes']:,}")
        print(f"  Languages: {self.stats['total_languages']:,}")

        print(f"\nTraining Examples:")
        print(f"  Context examples: {self.stats.get('context_examples', 0):,}")
        print(f"  Morphology examples: {self.stats.get('morphology_examples', 0):,}")
        print(f"  Contrastive pairs: {self.stats.get('contrastive_pairs', 0):,}")
        print(f"  Inventories: {self.stats.get('inventories', 0):,}")


def main():
    """Demo the data loader"""

    # Initialize
    loader = PhonemeEmbeddingDataLoader()

    # Generate all data sources
    context_examples = list(loader.get_context_examples(window_size=2))
    morphology_examples = list(loader.get_morphology_examples())
    contrastive_pairs = list(loader.get_contrastive_pairs(num_pairs=10000))
    inventories = list(loader.get_inventory_examples())

    # Print stats
    loader.print_stats()

    # Show examples
    print("\n" + "=" * 70)
    print("SAMPLE EXAMPLES")
    print("=" * 70)

    print("\n1. Context Examples (first 5):")
    for i, ex in enumerate(context_examples[:5]):
        print(f"   {ex.center_phoneme} → context: {ex.context_phonemes} (word: {ex.word})")

    print("\n2. Contrastive Pairs (first 5):")
    for i, pair in enumerate(contrastive_pairs[:5]):
        print(f"   /{pair.phoneme1}/ vs /{pair.phoneme2}/ → similarity: {pair.similarity:.3f}")

    print("\n3. Sample Inventories (first 3):")
    for i, inv in enumerate(inventories[:3]):
        print(f"   {inv.language}: {len(inv.phonemes)} phonemes")
        print(f"      {', '.join(inv.phonemes[:10])}...")

    # Get feature matrix
    features = loader.get_feature_matrix()
    print(f"\n4. Feature Matrix:")
    print(f"   Shape: {features.shape}")
    print(f"   Sample (first phoneme): {features[0][:10]}...")

    print("\n" + "=" * 70)
    print("✓ Data loader ready for training!")
    print("=" * 70)


if __name__ == '__main__':
    main()
