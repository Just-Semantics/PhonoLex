"""
Core phonological rules.

This module provides the basic building blocks for phonological rules:
- Feature matching rules
- Natural class rules
- Environment rules
- Assimilation rules
"""

from typing import Dict, List, Optional
from .base import Rule, Phoneme, RuleType
from .features import get_natural_class, matches_natural_class

def create_feature_match_rule(features: Dict[str, bool], name: Optional[str] = None) -> Rule:
    """Create a rule that matches phonemes with specific features."""
    def check(sequence: List[Phoneme], index: int) -> bool:
        phoneme = sequence[index]
        return all(
            phoneme.features.get(feature) == value 
            for feature, value in features.items()
        )
    
    return Rule(
        name=name or f"Match features: {features}",
        rule_type=RuleType.FEATURE_MATCH,
        description=f"Match phonemes with features: {features}",
        check=check
    )

def create_natural_class_rule(class_name: str, name: Optional[str] = None) -> Rule:
    """Create a rule that matches phonemes belonging to a natural class."""
    class_features = get_natural_class(class_name)
    
    def check(sequence: List[Phoneme], index: int) -> bool:
        return matches_natural_class(sequence[index].features, class_name)
    
    return Rule(
        name=name or f"Match {class_name}",
        rule_type=RuleType.NATURAL_CLASS,
        description=f"Match phonemes in natural class: {class_name}",
        check=check
    )

def create_environment_rule(
    target: Dict[str, bool],
    left_context: Optional[Dict[str, bool]] = None,
    right_context: Optional[Dict[str, bool]] = None,
    name: Optional[str] = None
) -> Rule:
    """Create a rule that matches phonemes in specific environments."""
    def check(sequence: List[Phoneme], index: int) -> bool:
        # Check target
        if not all(sequence[index].features.get(feature) == value 
                  for feature, value in target.items()):
            return False
        
        # Check left context
        if left_context and index > 0:
            if not all(sequence[index-1].features.get(feature) == value 
                      for feature, value in left_context.items()):
                return False
        
        # Check right context
        if right_context and index < len(sequence) - 1:
            if not all(sequence[index+1].features.get(feature) == value 
                      for feature, value in right_context.items()):
                return False
        
        return True
    
    return Rule(
        name=name or f"Match {target} between {left_context} and {right_context}",
        rule_type=RuleType.ENVIRONMENT,
        description=f"Match phonemes with features {target} between {left_context} and {right_context}",
        check=check
    )

def create_assimilation_rule(
    target_class: str,
    assimilate_to: str,
    feature: str,
    direction: str = "progressive",
    name: Optional[str] = None
) -> Rule:
    """Create a rule that checks for feature assimilation."""
    def check(sequence: List[Phoneme], index: int) -> bool:
        # Check if current phoneme belongs to target class
        if not matches_natural_class(sequence[index].features, target_class):
            return False
        
        # Find trigger phoneme based on direction
        if direction == "progressive":
            if index == len(sequence) - 1:
                return False
            trigger = sequence[index + 1]
        else:  # regressive
            if index == 0:
                return False
            trigger = sequence[index - 1]
        
        # Check if trigger belongs to assimilate_to class
        if not matches_natural_class(trigger.features, assimilate_to):
            return False
        
        # Check if feature values match
        return sequence[index].features.get(feature) == trigger.features.get(feature)
    
    return Rule(
        name=name or f"{direction.capitalize()} {feature} assimilation",
        rule_type=RuleType.ASSIMILATION,
        description=f"Check for {direction} {feature} assimilation between {target_class} and {assimilate_to}",
        check=check
    )

__all__ = [
    "create_feature_match_rule",
    "create_natural_class_rule",
    "create_environment_rule",
    "create_assimilation_rule"
] 