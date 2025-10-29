"""
Pattern matching module for PhonoLex.

This module provides tools for matching phonetic patterns in words, supporting:
- Exact matches (e.g., "bat")
- Feature-based patterns (e.g., "any consonant + a + any stop")
- Natural class patterns (e.g., "stop + vowel + fricative")
- Wildcards and quantifiers (e.g., "any number of consonants + vowel")
"""

from typing import List, Optional, Union, Dict
from dataclasses import dataclass
from enum import Enum

from .rules import Phoneme, NATURAL_CLASSES, FEATURE_VALUES

class PatternType(Enum):
    """Types of pattern elements."""
    EXACT = "exact"           # Match exact phoneme
    FEATURE = "feature"       # Match by features
    NATURAL_CLASS = "class"   # Match by natural class
    WILDCARD = "wildcard"     # Match any phoneme
    REPEAT = "repeat"         # Repeat previous pattern

@dataclass
class PatternElement:
    """A single element in a phonetic pattern."""
    pattern_type: PatternType
    value: Union[str, Dict[str, bool], str, None]  # IPA string, features dict, or class name
    min_repeat: int = 1
    max_repeat: Optional[int] = 1
    
    def matches(self, phoneme: Phoneme) -> bool:
        """Check if a phoneme matches this pattern element."""
        if self.pattern_type == PatternType.EXACT:
            return phoneme.ipa == self.value
        
        elif self.pattern_type == PatternType.FEATURE:
            return all(
                phoneme.features.get(feature) == value 
                for feature, value in self.value.items()
            )
        
        elif self.pattern_type == PatternType.NATURAL_CLASS:
            class_features = NATURAL_CLASSES[self.value]
            return all(
                phoneme.features.get(feature) == value 
                for feature, value in class_features.items()
            )
        
        elif self.pattern_type == PatternType.WILDCARD:
            return True
        
        return False

@dataclass
class Pattern:
    """A phonetic pattern for matching words."""
    elements: List[PatternElement]
    name: Optional[str] = None
    description: Optional[str] = None
    
    def matches(self, phonemes: List[Phoneme]) -> bool:
        """Check if a sequence of phonemes matches this pattern."""
        if not phonemes:
            return False
            
        # Current position in the phoneme sequence
        pos = 0
        
        # Try to match each pattern element
        for element in self.elements:
            # Handle repetition
            matched = 0
            while pos < len(phonemes) and matched < (element.max_repeat or float('inf')):
                if not element.matches(phonemes[pos]):
                    break
                matched += 1
                pos += 1
            
            # Check if we matched enough times
            if matched < element.min_repeat:
                return False
        
        # Pattern matches if we consumed all phonemes
        return pos == len(phonemes)

def create_exact_pattern(ipa: str) -> Pattern:
    """Create a pattern that matches an exact IPA string."""
    return Pattern([
        PatternElement(PatternType.EXACT, ipa)
    ], f"Match exact '{ipa}'")

def create_feature_pattern(features: Dict[str, bool]) -> Pattern:
    """Create a pattern that matches phonemes with specific features."""
    return Pattern([
        PatternElement(PatternType.FEATURE, features)
    ], f"Match features {features}")

def create_natural_class_pattern(class_name: str) -> Pattern:
    """Create a pattern that matches phonemes in a natural class."""
    if class_name not in NATURAL_CLASSES:
        raise ValueError(f"Unknown natural class: {class_name}")
    
    return Pattern([
        PatternElement(PatternType.NATURAL_CLASS, class_name)
    ], f"Match {class_name}")

def create_sequence_pattern(elements: List[PatternElement], name: Optional[str] = None) -> Pattern:
    """Create a pattern from a sequence of pattern elements."""
    return Pattern(elements, name)

def create_cv_pattern() -> Pattern:
    """Create a pattern for CV syllable structure."""
    return Pattern([
        PatternElement(PatternType.NATURAL_CLASS, "consonant"),
        PatternElement(PatternType.NATURAL_CLASS, "vowel")
    ], "CV syllable")

def create_cvc_pattern() -> Pattern:
    """Create a pattern for CVC syllable structure."""
    return Pattern([
        PatternElement(PatternType.NATURAL_CLASS, "consonant"),
        PatternElement(PatternType.NATURAL_CLASS, "vowel"),
        PatternElement(PatternType.NATURAL_CLASS, "consonant")
    ], "CVC syllable")

def create_custom_pattern(
    elements: List[Union[str, Dict[str, bool], str]],
    repeats: Optional[List[Dict[str, int]]] = None
) -> Pattern:
    """Create a custom pattern from a list of elements.
    
    Args:
        elements: List of pattern elements, each can be:
            - IPA string for exact match
            - Feature dictionary for feature match
            - Natural class name for class match
            - None for wildcard
        repeats: Optional list of repeat specifications for each element
            Each dict can contain 'min' and 'max' keys
    """
    pattern_elements = []
    
    for i, element in enumerate(elements):
        repeat_spec = repeats[i] if repeats and i < len(repeats) else None
        
        if isinstance(element, str):
            if element in NATURAL_CLASSES:
                pattern_type = PatternType.NATURAL_CLASS
            else:
                pattern_type = PatternType.EXACT
        elif isinstance(element, dict):
            pattern_type = PatternType.FEATURE
        elif element is None:
            pattern_type = PatternType.WILDCARD
        else:
            raise ValueError(f"Invalid pattern element: {element}")
        
        min_repeat = repeat_spec.get('min', 1) if repeat_spec else 1
        max_repeat = repeat_spec.get('max', 1) if repeat_spec else 1
        
        pattern_elements.append(PatternElement(
            pattern_type=pattern_type,
            value=element,
            min_repeat=min_repeat,
            max_repeat=max_repeat
        ))
    
    return Pattern(pattern_elements)

__all__ = [
    "PatternType",
    "PatternElement",
    "Pattern",
    "create_exact_pattern",
    "create_feature_pattern",
    "create_natural_class_pattern",
    "create_sequence_pattern",
    "create_cv_pattern",
    "create_cvc_pattern",
    "create_custom_pattern"
] 