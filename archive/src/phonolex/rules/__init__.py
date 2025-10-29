"""
PhonoLex rules package.

This package provides a comprehensive system for working with phonological rules,
including feature matching, natural classes, and pattern matching.
"""

from .base import (
    RuleType,
    Phoneme,
    Rule,
    check_sequence,
    check_all_rules
)

from .features import (
    FEATURE_VALUES,
    NATURAL_CLASSES,
    SONORITY_HIERARCHY,
    get_sonority_rank,
    get_natural_class,
    matches_natural_class
)

from .phonological import (
    create_feature_match_rule,
    create_natural_class_rule,
    create_environment_rule,
    create_assimilation_rule
)

__all__ = [
    # Base classes and utilities
    "RuleType",
    "Phoneme",
    "Rule",
    "check_sequence",
    "check_all_rules",
    
    # Features and natural classes
    "FEATURE_VALUES",
    "NATURAL_CLASSES",
    "SONORITY_HIERARCHY",
    "get_sonority_rank",
    "get_natural_class",
    "matches_natural_class",
    
    # Rule creation functions
    "create_feature_match_rule",
    "create_natural_class_rule",
    "create_environment_rule",
    "create_assimilation_rule"
]
