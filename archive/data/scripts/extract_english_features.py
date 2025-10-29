#!/usr/bin/env python3
"""
Extract English Phoneme Features

This script extracts phoneme features for English from the PHOIBLE database.
"""

import os
import csv
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional, Union

# Import our phoneme mapping module
from phoneme_mappings import IPA_TO_ARPA

# Paths
DATA_DIR = Path(__file__).parent.parent
PHOIBLE_DATA = DATA_DIR / "phoible/phoible.csv"
FEATURES_TSV = DATA_DIR / "phoible/phoible-segments-features.tsv"
OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_FEATURES = OUTPUT_DIR / "english_phoneme_features.json"
OUTPUT_VECTORS = OUTPUT_DIR / "english_phoneme_vectors.json" 