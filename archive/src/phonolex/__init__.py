"""
PhonoLex - A phonological lexicon with vector-based representation for phoneme features.
"""

__version__ = "0.1.0"

# Make key modules available at package level
from .rules import (
    # Constants
    FEATURE_VALUES,
    NATURAL_CLASSES,
    
    # Rule creation
    create_feature_match_rule,
    create_natural_class_rule,
    create_environment_rule,
    create_assimilation_rule,
)

__all__ = [
    # Constants
    "FEATURE_VALUES",
    "NATURAL_CLASSES",
    
    # Rule creation
    "create_feature_match_rule",
    "create_natural_class_rule",
    "create_environment_rule",
    "create_assimilation_rule",
]
