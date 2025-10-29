#!/usr/bin/env python3
"""
Missing Phonemes Supplement

This script adds manually defined feature vectors for English phonemes
that are missing from the PHOIBLE dataset, particularly diphthongs
and other complex vowels specific to English dialects.
"""

import json
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional, Union

# Configuration
DATA_DIR = Path(__file__).parent.parent
INPUT_FEATURES = DATA_DIR / "processed/english_phoneme_features.json"
INPUT_VECTORS = DATA_DIR / "processed/english_phoneme_vectors.json"
OUTPUT_FEATURES = DATA_DIR / "processed/english_phoneme_features_complete.json"
OUTPUT_VECTORS = DATA_DIR / "processed/english_phoneme_vectors_complete.json" 