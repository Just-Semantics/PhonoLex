#!/usr/bin/env python3
"""
Phoneme Mapping Definitions

This module contains the mappings between different phonetic notation systems:
- IPA (International Phonetic Alphabet)
- ARPAbet (CMU Pronouncing Dictionary standard)

These mappings are used throughout PhonoLex for phonetic conversions
and maintaining consistent phonological representations.
"""

import json
import os
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field

# Directory for saving/loading mappings
SCRIPT_DIR = Path(__file__).parent.absolute()
DATA_DIR = SCRIPT_DIR.parent / "mappings"
os.makedirs(DATA_DIR, exist_ok=True)

# Mapping files
IPA_TO_ARPA_FILE = DATA_DIR / "ipa_to_arpa.json"
ARPA_TO_IPA_FILE = DATA_DIR / "arpa_to_ipa.json"

# Phoneme Categories
class PhonemeType(Enum):
    """Classification of phonemes by type"""
    VOWEL = auto()
    CONSONANT = auto()
    DIPHTHONG = auto()
    AFFRICATE = auto()
    STRESS_MARKER = auto()


class PhonemeSystem(Enum):
    """Phonetic notation systems"""
    IPA = "IPA"
    ARPA = "ARPAbet"
    XSAMPA = "X-SAMPA"  # Reserved for future use


class Phoneme(BaseModel):
    """
    A phoneme with its representations in different notation systems
    and phonological properties
    """
    symbol: str
    system: PhonemeSystem
    phoneme_type: PhonemeType
    example_word: Optional[str] = None
    description: Optional[str] = None
    
    class Config:
        frozen = True


class PhonemeMapping(BaseModel):
    """Mapping between a phoneme in one system and its equivalent in another"""
    source: Phoneme
    target: Phoneme
    is_primary: bool = True  # Whether this is the primary mapping
    
    class Config:
        frozen = True


# Define vowels in IPA with their ARPAbet equivalents
VOWELS_IPA_TO_ARPA = {
    "eɪ": "EY",  # hate
    "i": "IY",   # Pete
    "aɪ": "AY",  # site
    "oʊ": "OW",  # note
    "u": "UW",   # cute
    "æ": "AE",   # hat
    "ɛ": "EH",   # pet
    "ɪ": "IH",   # sit
    "ɔ": "AO",   # not (variant 1)
    "ɑ": "AA",   # not (variant 2)
    "ʌ": "AH",   # cut (stressed)
    "ə": "AH0",  # cut (unstressed)
    "ɔɪ": "OY",  # coin
    "aʊ": "AW",  # loud
    "ʊ": "UH",   # book
    "ɝ": "ER",   # turn (stressed)
    "ɚ": "ER0",  # turn (unstressed)
}

# Define consonants in IPA with their ARPAbet equivalents
CONSONANTS_IPA_TO_ARPA = {
    "b": "B",    # buy
    "p": "P",    # pie
    "d": "D",    # die
    "t": "T",    # tie
    "v": "V",    # vie
    "f": "F",    # fight
    "g": "G",    # guy
    "k": "K",    # kite
    "h": "HH",   # high
    "dʒ": "JH",  # joy
    "tʃ": "CH",  # China
    "l": "L",    # lie
    "m": "M",    # my
    "n": "N",    # nigh
    "ɹ": "R",    # rye
    "z": "Z",    # zoo
    "s": "S",    # sigh
    "w": "W",    # wise
    "j": "Y",    # yacht
    "ʒ": "ZH",   # pleasure
    "ʃ": "SH",   # shy
    "ð": "DH",   # they
    "θ": "TH",   # thigh
    "ŋ": "NG",   # sing
}

# Stress markers
STRESS_MARKERS = {
    "ˈ": "1",    # Primary stress
    "ˌ": "2",    # Secondary stress
}

# Combine all mappings
IPA_TO_ARPA = {**VOWELS_IPA_TO_ARPA, **CONSONANTS_IPA_TO_ARPA}

# Create the reverse mapping
ARPA_TO_IPA = {}
for ipa, arpa in IPA_TO_ARPA.items():
    # Handle basic phonemes
    if not any(char.isdigit() for char in arpa):
        ARPA_TO_IPA[arpa] = ipa
    
    # Handle stress levels for vowels
    if arpa in ["AH", "ER"]:  # Special cases with different symbols for stressed/unstressed
        ARPA_TO_IPA[f"{arpa}0"] = "ə" if arpa == "AH" else "ɚ"  # Unstressed
        ARPA_TO_IPA[f"{arpa}1"] = f"ˈ{ipa}"  # Primary stress
        ARPA_TO_IPA[f"{arpa}2"] = f"ˌ{ipa}"  # Secondary stress
    elif arpa in ["AA", "AE", "AO", "AW", "AY", "EH", "EY", "IH", "IY", "OW", "OY", "UH", "UW"]:
        ARPA_TO_IPA[f"{arpa}0"] = ipa  # Unstressed
        ARPA_TO_IPA[f"{arpa}1"] = f"ˈ{ipa}"  # Primary stress
        ARPA_TO_IPA[f"{arpa}2"] = f"ˌ{ipa}"  # Secondary stress


def save_mapping_files():
    """Save the mappings to JSON files"""
    with open(IPA_TO_ARPA_FILE, 'w', encoding='utf-8') as f:
        json.dump(IPA_TO_ARPA, f, indent=2, ensure_ascii=False)
    
    with open(ARPA_TO_IPA_FILE, 'w', encoding='utf-8') as f:
        json.dump(ARPA_TO_IPA, f, indent=2, ensure_ascii=False)
    
    print(f"Saved mapping files to {DATA_DIR}")


def convert_ipa_to_arpa(ipa_text: str) -> str:
    """Convert IPA text to ARPAbet"""
    # Placeholder for more sophisticated conversion logic
    # This is a simple direct mapping for now
    result = []
    i = 0
    while i < len(ipa_text):
        # Try to match longest possible phoneme first
        matched = False
        for length in range(min(3, len(ipa_text) - i), 0, -1):
            substr = ipa_text[i:i+length]
            if substr in IPA_TO_ARPA:
                result.append(IPA_TO_ARPA[substr])
                i += length
                matched = True
                break
        
        # If no match, just keep the character
        if not matched:
            # Handle stress markers
            if ipa_text[i] in STRESS_MARKERS and i+1 < len(ipa_text):
                # Skip for now, we'll handle stress in the next iteration
                i += 1
            else:
                result.append(ipa_text[i])
                i += 1
    
    return " ".join(result)


def convert_arpa_to_ipa(arpa_text: str) -> str:
    """Convert ARPAbet text to IPA"""
    # Split into phonemes
    phonemes = arpa_text.strip().split()
    result = []
    
    for phoneme in phonemes:
        if phoneme in ARPA_TO_IPA:
            result.append(ARPA_TO_IPA[phoneme])
        else:
            # Handle stress markers
            match = None
            # Placeholder for regex matching
            # For now, use simple check for digit at end
            if phoneme[-1].isdigit():
                base = phoneme[:-1]
                stress = phoneme[-1]
                
                if f"{base}{stress}" in ARPA_TO_IPA:
                    result.append(ARPA_TO_IPA[f"{base}{stress}"])
                else:
                    # Fallback: just keep as is
                    result.append(phoneme)
            else:
                # Keep as is if no mapping
                result.append(phoneme)
    
    return "".join(result)


if __name__ == "__main__":
    # Save the mappings when run directly
    save_mapping_files()
    
    # Test the conversion functions
    test_ipa = "ˈhɛloʊ"
    test_arpa = "HH EH1 L OW0"
    
    print(f"IPA: {test_ipa} -> ARPAbet: {convert_ipa_to_arpa(test_ipa)}")
    print(f"ARPAbet: {test_arpa} -> IPA: {convert_arpa_to_ipa(test_arpa)}") 