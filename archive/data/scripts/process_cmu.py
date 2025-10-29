#!/usr/bin/env python3
"""
Process CMU Dictionary

This script processes the CMU Pronouncing Dictionary and generates:
1. A JSON file mapping words to phonetic transcriptions
2. A structured database of Word objects with phonetic vectors
"""

import os
import json
import re
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional

# Configuration
DATA_DIR = Path(__file__).parent.parent
INPUT_FILE = DATA_DIR / "cmu/cmudict-0.7b"
OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_FILE = OUTPUT_DIR / "cmu_words.json"
PHONEMES_FILE = OUTPUT_DIR / "cmu_phonemes.json"
FEATURES_FILE = OUTPUT_DIR / "english_phoneme_features_complete.json"
VECTORS_FILE = OUTPUT_DIR / "english_phoneme_vectors_complete.json" 