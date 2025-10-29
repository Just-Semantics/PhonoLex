#!/usr/bin/env python3
"""
Phoneme Vectorization for Phoible Feature System

This module converts Phoible's phonological features (including trajectories)
into vector representations suitable for similarity metrics and ML.

Supports multiple encoding strategies:
- Strategy 9: Endpoints (76-dim) - start and end states
- Strategy 10: Trajectory sequence (152-dim) - 4 timesteps with interpolation

Author: PhonoLex
Date: 2025-10-26
"""

import csv
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict


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
    'spreadGlottis', 'constrictedGlottis',
    'fortis', 'lenis',
    'raisedLarynxEjective', 'loweredLarynxImplosive', 'click'
]


@dataclass
class PhonemeVector:
    """Container for multiple vector representations of a phoneme"""
    phoneme: str
    language: str
    inventory_id: int
    segment_class: str

    # Feature data
    features_raw: Dict[str, str]  # Original feature values from Phoible

    # Vector representations
    endpoints_76d: np.ndarray  # Strategy 9: start + end (76-dim)
    trajectory_152d: np.ndarray  # Strategy 10: 4 timesteps (152-dim)

    # Trajectory metadata
    has_trajectory: bool
    trajectory_features: List[str]  # Features with multi-values

    def to_dict(self):
        """Convert to JSON-serializable dict"""
        return {
            'phoneme': self.phoneme,
            'language': self.language,
            'inventory_id': self.inventory_id,
            'segment_class': self.segment_class,
            'features_raw': self.features_raw,
            'endpoints_76d': self.endpoints_76d.tolist(),
            'trajectory_152d': self.trajectory_152d.tolist(),
            'has_trajectory': self.has_trajectory,
            'trajectory_features': self.trajectory_features
        }


class PhonemeVectorizer:
    """
    Convert Phoible phonological features to vector representations

    Handles ternary features (+, -, 0) and trajectory features (comma-separated)
    """

    def __init__(self, encoding_scheme: str = 'three_way'):
        """
        Initialize vectorizer

        Args:
            encoding_scheme: How to encode single values
                - 'three_way': +→1.0, -→-1.0, 0→0.0 (recommended)
                - 'binary': +→1.0, -→0.0, 0→0.5
        """
        self.encoding_scheme = encoding_scheme

        if encoding_scheme == 'three_way':
            self.value_map = {'+': 1.0, '-': -1.0, '0': 0.0}
        elif encoding_scheme == 'binary':
            self.value_map = {'+': 1.0, '-': 0.0, '0': 0.5}
        else:
            raise ValueError(f"Unknown encoding scheme: {encoding_scheme}")

    def encode_value(self, value: str) -> float:
        """Encode a single feature value to float"""
        return self.value_map[value.strip()]

    def parse_trajectory(self, value: str) -> List[float]:
        """
        Parse a feature value (possibly with trajectory) into list of states

        Examples:
            '+' → [1.0]
            '-,+' → [-1.0, 1.0]
            '+,-,+' → [1.0, -1.0, 1.0]
        """
        if ',' not in value:
            # Single value (static feature)
            return [self.encode_value(value)]

        # Trajectory: split and encode each phase
        parts = value.split(',')
        return [self.encode_value(p) for p in parts]

    def interpolate_trajectory(self, states: List[float], num_points: int = 4) -> np.ndarray:
        """
        Interpolate a trajectory to have exactly num_points timesteps

        Args:
            states: List of feature values (1-4 values typically)
            num_points: Target number of timesteps (default 4)

        Returns:
            Array of shape (num_points,) with interpolated values

        Examples:
            [1.0] → [1.0, 1.0, 1.0, 1.0]  # Static: repeat
            [-1.0, 1.0] → [-1.0, -0.33, 0.33, 1.0]  # 2-phase: interpolate
            [1.0, -1.0, 1.0] → [1.0, -1.0, 1.0, 1.0]  # 3-phase: last repeated
        """
        states = np.array(states)
        n = len(states)

        if n == 1:
            # Static feature: repeat same value
            return np.full(num_points, states[0])

        if n >= num_points:
            # Already have enough points: sample evenly
            indices = np.linspace(0, n-1, num_points).astype(int)
            return states[indices]

        # Interpolate: create num_points evenly spaced along trajectory
        old_indices = np.arange(n)
        new_indices = np.linspace(0, n-1, num_points)
        return np.interp(new_indices, old_indices, states)

    def encode_endpoints_76d(self, features: Dict[str, str]) -> np.ndarray:
        """
        Strategy 9: Encode as 76-dim vector (38 features × 2: start + end)

        For static features (no trajectory): start == end
        For trajectories: start = first value, end = last value

        Returns:
            Array of shape (76,) = [feat1_start, feat1_end, feat2_start, feat2_end, ...]
        """
        vector = []

        for feature_name in PHOIBLE_FEATURES:
            value = features[feature_name]
            trajectory = self.parse_trajectory(value)

            if len(trajectory) == 1:
                # Static: same start and end
                start = end = trajectory[0]
            else:
                # Dynamic: first and last
                start = trajectory[0]
                end = trajectory[-1]

            vector.extend([start, end])

        return np.array(vector)

    def encode_trajectory_152d(self, features: Dict[str, str], num_timesteps: int = 4) -> np.ndarray:
        """
        Strategy 10: Encode as 152-dim vector (38 features × 4 timesteps)

        Each feature is interpolated to have exactly num_timesteps values,
        representing the feature's state at different points in the articulation.

        Returns:
            Array of shape (152,) = [feat1_t0, feat1_t1, feat1_t2, feat1_t3,
                                      feat2_t0, feat2_t1, feat2_t2, feat2_t3, ...]
        """
        vector = []

        for feature_name in PHOIBLE_FEATURES:
            value = features[feature_name]
            trajectory = self.parse_trajectory(value)
            interpolated = self.interpolate_trajectory(trajectory, num_timesteps)
            vector.extend(interpolated)

        return np.array(vector)

    def detect_trajectory_features(self, features: Dict[str, str]) -> List[str]:
        """
        Identify which features have trajectories (comma-separated values)

        Returns:
            List of feature names that have multi-value (trajectory) encoding
        """
        trajectory_feats = []
        for feature_name in PHOIBLE_FEATURES:
            value = features[feature_name]
            if ',' in value:
                trajectory_feats.append(feature_name)
        return trajectory_feats

    def vectorize(self, phoneme_data: Dict[str, str]) -> PhonemeVector:
        """
        Convert a Phoible phoneme row to multiple vector representations

        Args:
            phoneme_data: Dictionary with keys matching Phoible CSV columns

        Returns:
            PhonemeVector with all representations
        """
        # Extract feature values
        features = {feat: phoneme_data[feat] for feat in PHOIBLE_FEATURES}

        # Detect trajectories
        trajectory_features = self.detect_trajectory_features(features)
        has_trajectory = len(trajectory_features) > 0

        # Generate vectors
        endpoints = self.encode_endpoints_76d(features)
        trajectory = self.encode_trajectory_152d(features)

        return PhonemeVector(
            phoneme=phoneme_data['Phoneme'],
            language=phoneme_data['LanguageName'],
            inventory_id=int(phoneme_data['InventoryID']),
            segment_class=phoneme_data['SegmentClass'],
            features_raw=features,
            endpoints_76d=endpoints,
            trajectory_152d=trajectory,
            has_trajectory=has_trajectory,
            trajectory_features=trajectory_features
        )


def load_phoible_csv(csv_path: str) -> List[Dict[str, str]]:
    """Load Phoible CSV file and return list of phoneme dictionaries"""
    phonemes = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            phonemes.append(row)
    return phonemes


def main():
    """Demo: vectorize English phonemes from Phoible"""

    # Initialize vectorizer
    vectorizer = PhonemeVectorizer(encoding_scheme='three_way')

    # Load English data
    data_dir = Path(__file__).parent.parent
    english_csv = data_dir / 'phoible' / 'english' / 'phoible-english.csv'

    print(f"Loading English phonemes from {english_csv}")
    phonemes = load_phoible_csv(str(english_csv))
    print(f"Loaded {len(phonemes)} phoneme entries\n")

    # Vectorize all phonemes
    vectors = []
    for phoneme_data in phonemes:
        vec = vectorizer.vectorize(phoneme_data)
        vectors.append(vec)

    # Show some examples
    print("=" * 70)
    print("EXAMPLES")
    print("=" * 70)

    # Find a monophthong
    monophthong = next(v for v in vectors if v.phoneme == 'i' and not v.has_trajectory)
    print(f"\n1. Monophthong: /{monophthong.phoneme}/")
    print(f"   Language: {monophthong.language}")
    print(f"   Has trajectory: {monophthong.has_trajectory}")
    print(f"   Endpoints (76d): shape={monophthong.endpoints_76d.shape}, sample={monophthong.endpoints_76d[:10]}")
    print(f"   Trajectory (152d): shape={monophthong.trajectory_152d.shape}, sample={monophthong.trajectory_152d[:10]}")

    # Find a diphthong
    diphthong = next((v for v in vectors if v.phoneme == 'aɪ'), None)
    if diphthong:
        print(f"\n2. Diphthong: /{diphthong.phoneme}/")
        print(f"   Language: {diphthong.language}")
        print(f"   Has trajectory: {diphthong.has_trajectory}")
        print(f"   Trajectory features: {diphthong.trajectory_features}")
        print(f"   Endpoints (76d): shape={diphthong.endpoints_76d.shape}")
        print(f"   Trajectory (152d): shape={diphthong.trajectory_152d.shape}")

        # Show trajectory detail for 'high' feature
        high_idx = PHOIBLE_FEATURES.index('high')
        print(f"\n   Feature 'high' trajectory:")
        print(f"     Raw value: {diphthong.features_raw['high']}")
        print(f"     Endpoints: start={diphthong.endpoints_76d[high_idx*2]:.1f}, end={diphthong.endpoints_76d[high_idx*2+1]:.1f}")
        print(f"     Full trajectory (4 steps): {diphthong.trajectory_152d[high_idx*4:high_idx*4+4]}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    total = len(vectors)
    with_trajectory = sum(1 for v in vectors if v.has_trajectory)
    vowels = sum(1 for v in vectors if v.segment_class == 'vowel')
    vowels_with_traj = sum(1 for v in vectors if v.segment_class == 'vowel' and v.has_trajectory)

    print(f"Total phonemes: {total}")
    print(f"Phonemes with trajectories: {with_trajectory} ({100*with_trajectory/total:.1f}%)")
    print(f"Vowels: {vowels}")
    print(f"Vowels with trajectories (diphthongs): {vowels_with_traj} ({100*vowels_with_traj/vowels:.1f}%)")

    # Save sample output
    output_file = data_dir / 'mappings' / 'sample_vectors.json'
    sample_data = [v.to_dict() for v in vectors[:10]]
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved sample vectors to {output_file}")


if __name__ == '__main__':
    main()
