"""
Base classes and utilities for the phonological rule system.
"""

from typing import Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum

class RuleType(Enum):
    """Types of phonological rules."""
    FEATURE_MATCH = "feature_match"
    NATURAL_CLASS = "natural_class"
    ENVIRONMENT = "environment"
    ASSIMILATION = "assimilation"
    DISSIMILATION = "dissimilation"
    SYLLABLE_STRUCTURE = "syllable_structure"
    SONORITY = "sonority"
    HARMONIZATION = "harmonization"
    PHONOTACTIC = "phonotactic"
    CUSTOM = "custom"

@dataclass
class Phoneme:
    """Represents a phoneme with its features."""
    ipa: str
    features: Dict[str, bool]

@dataclass
class Rule:
    """Base class for phonological rules."""
    name: str
    rule_type: RuleType
    description: str
    check: Callable[[List[Phoneme], int], bool]
    
    def __call__(self, sequence: List[Phoneme], index: int) -> bool:
        """Check if the rule applies at the given position."""
        return self.check(sequence, index)

def check_sequence(sequence: List[Phoneme], rule: Rule) -> bool:
    """Check if a sequence satisfies a rule at any position."""
    return any(rule(sequence, i) for i in range(len(sequence)))

def check_all_rules(sequence: List[Phoneme], rules: List[Rule]) -> bool:
    """Check if a sequence satisfies all rules."""
    return all(check_sequence(sequence, rule) for rule in rules)

__all__ = [
    "RuleType",
    "Phoneme",
    "Rule",
    "check_sequence",
    "check_all_rules"
] 