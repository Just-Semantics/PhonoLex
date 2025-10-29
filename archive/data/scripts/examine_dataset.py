#!/usr/bin/env python3
"""
Examine WordNet Dataset

This script examines the WordNet dataset and displays sample entries
to verify the data structure and content.
"""

import json
import random
from pathlib import Path
from pprint import pprint

# Paths
DATA_DIR = Path(__file__).parent.parent
DATASET_FILE = DATA_DIR / "processed/wordnet_dataset.json" 