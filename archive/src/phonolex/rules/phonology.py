"""
Phonological rule definitions using seqrule.

This module provides phonological rules for analyzing and processing
phoneme sequences. It includes rules for:
- Feature-based matching
- Phonological environments
- Common sound changes
- Syllable structure constraints
- Phonotactic patterns
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union

from seqrule import AbstractObject, DSLRule, check_sequence
from seqrule import And, Or, Not


# Constants for phonological features
FEATURE_VALUES = {
    "consonantal": [True, False],
    "sonorant": [True, False],
    "syllabic": [True, False],
    "voice": [True, False],
    "continuant": [True, False],
    "nasal": [True, False], 
    "strident": [True, False],
    "lateral": [True, False],
    "delayed_release": [True, False],
    "spread_glottis": [True, False],
    "constricted_glottis": [True, False],
    "labial": [True, False],
    "coronal": [True, False],
    "dorsal": [True, False],
    "pharyngeal": [True, False],
    "laryngeal": [True, False],
    "high": [True, False],
    "low": [True, False],
    "back": [True, False],
    "round": [True, False],
    "tense": [True, False],
    "anterior": [True, False],
    "distributed": [True, False]
}

# Natural classes for common sound groups
NATURAL_CLASSES = {
    "vowel": {"syllabic": True},
    "consonant": {"consonantal": True},
    "stop": {"consonantal": True, "sonorant": False, "continuant": False},
    "fricative": {"consonantal": True, "sonorant": False, "continuant": True},
    "nasal": {"consonantal": True, "sonorant": True, "nasal": True},
    "liquid": {"consonantal": True, "sonorant": True, "lateral": True},
    "glide": {"consonantal": False, "sonorant": True, "syllabic": False},
    "obstruent": {"consonantal": True, "sonorant": False},
    "sibilant": {"consonantal": True, "strident": True},
    "high_vowel": {"syllabic": True, "high": True, "low": False},
    "mid_vowel": {"syllabic": True, "high": False, "low": False},
    "low_vowel": {"syllabic": True, "low": True},
    "front_vowel": {"syllabic": True, "back": False},
    "back_vowel": {"syllabic": True, "back": True},
    "round_vowel": {"syllabic": True, "round": True},
    "labial": {"labial": True},
    "coronal": {"coronal": True},
    "dorsal": {"dorsal": True},
    "voiced_obstruent": {"consonantal": True, "sonorant": False, "voice": True},
    "voiceless_obstruent": {"consonantal": True, "sonorant": False, "voice": False}
}


def create_feature_match_rule(features: Dict[str, bool]) -> DSLRule:
    """
    Creates a rule that checks if phonemes match specified features.
    
    Args:
        features: Dictionary of features and their required values
        
    Returns:
        A rule that checks if all phonemes match the feature specification
        
    Example:
        vowel_rule = create_feature_match_rule({"syllabic": True})
    """
    def check_features(seq):
        for phoneme in seq:
            for feature, value in features.items():
                if feature not in phoneme:
                    return False
                if phoneme[feature] != value:
                    return False
        return True
    
    feature_desc = ", ".join(f"{f}={v}" for f, v in features.items())
    return DSLRule(check_features, f"all phonemes must have features: {feature_desc}")


def create_natural_class_rule(class_name: str) -> DSLRule:
    """
    Creates a rule that checks if phonemes belong to a natural class.
    
    Args:
        class_name: Name of the natural class (e.g., "vowel", "stop")
        
    Returns:
        A rule that checks if all phonemes belong to the specified class
        
    Example:
        nasal_rule = create_natural_class_rule("nasal")
    """
    if class_name not in NATURAL_CLASSES:
        raise ValueError(f"Unknown natural class: {class_name}")
        
    return create_feature_match_rule(NATURAL_CLASSES[class_name])


def create_environment_rule(
    target_features: Dict[str, bool],
    left_context: Optional[Dict[str, bool]] = None,
    right_context: Optional[Dict[str, bool]] = None
) -> DSLRule:
    """
    Creates a rule that checks if phonemes appear in specific environments.
    
    Args:
        target_features: Features of the target phoneme
        left_context: Features of the preceding context (optional)
        right_context: Features of the following context (optional)
        
    Returns:
        A rule that checks if phonemes appear in the specified environment
        
    Example:
        # Nasals appearing between vowels
        intervocalic_nasals = create_environment_rule(
            {"nasal": True},
            {"syllabic": True},
            {"syllabic": True}
        )
    """
    def check_environment(seq):
        if len(seq) < (1 + (left_context is not None) + (right_context is not None)):
            return True  # Too short to check
            
        for i in range(len(seq)):
            # Skip if this isn't a potential target
            target_match = True
            for feature, value in target_features.items():
                if feature not in seq[i] or seq[i][feature] != value:
                    target_match = False
                    break
                    
            if not target_match:
                continue
                
            # Check left context if specified
            left_match = True
            if left_context is not None:
                if i == 0:  # No left context
                    left_match = False
                else:
                    for feature, value in left_context.items():
                        if feature not in seq[i-1] or seq[i-1][feature] != value:
                            left_match = False
                            break
                            
            # Check right context if specified
            right_match = True
            if right_context is not None:
                if i == len(seq) - 1:  # No right context
                    right_match = False
                else:
                    for feature, value in right_context.items():
                        if feature not in seq[i+1] or seq[i+1][feature] != value:
                            right_match = False
                            break
                            
            # If both contexts match, we found a valid environment
            if left_match and right_match:
                return True
                
        return False
    
    # Create description
    desc_parts = []
    desc = f"phonemes with {list(target_features.items())}"
    if left_context:
        desc += f" after {list(left_context.items())}"
    if right_context:
        desc += f" before {list(right_context.items())}"
        
    return DSLRule(check_environment, desc)


def create_assimilation_rule(
    target_class: str,
    assimilate_to: str,
    feature: str,
    direction: str = "progressive"
) -> DSLRule:
    """
    Creates a rule that checks for feature assimilation.
    
    Args:
        target_class: Natural class that undergoes assimilation
        assimilate_to: Natural class that triggers assimilation
        feature: Feature that assimilates
        direction: "progressive" or "regressive"
        
    Returns:
        A rule that checks for proper assimilation
        
    Example:
        # Voicing assimilation: obstruents take voicing from following sonorant
        voicing_assimilation = create_assimilation_rule(
            "obstruent", "sonorant", "voice", "regressive"
        )
    """
    if target_class not in NATURAL_CLASSES or assimilate_to not in NATURAL_CLASSES:
        raise ValueError(f"Unknown natural class: {target_class} or {assimilate_to}")
        
    if feature not in FEATURE_VALUES:
        raise ValueError(f"Unknown feature: {feature}")
        
    target_features = NATURAL_CLASSES[target_class]
    trigger_features = NATURAL_CLASSES[assimilate_to]
    
    def check_assimilation(seq):
        if len(seq) < 2:
            return True
            
        for i in range(len(seq) - 1):
            # For progressive assimilation: first = trigger, second = target
            # For regressive assimilation: first = target, second = trigger
            if direction == "progressive":
                trigger_idx, target_idx = i, i+1
            else:  # regressive
                trigger_idx, target_idx = i+1, i
                
            # Check if the phonemes match the respective classes
            is_trigger = True
            for f, v in trigger_features.items():
                if f not in seq[trigger_idx] or seq[trigger_idx][f] != v:
                    is_trigger = False
                    break
                    
            is_target = True
            for f, v in target_features.items():
                if f not in seq[target_idx] or seq[target_idx][f] != v:
                    is_target = False
                    break
                    
            if not (is_trigger and is_target):
                continue
                
            # Check if the target has assimilated to the trigger for the specified feature
            if feature not in seq[trigger_idx] or feature not in seq[target_idx]:
                return False
                
            if seq[target_idx][feature] != seq[trigger_idx][feature]:
                return False
                
        return True
    
    return DSLRule(
        check_assimilation,
        f"{direction} assimilation of {feature} from {assimilate_to} to {target_class}"
    )


def create_dissimilation_rule(
    target_class: str,
    dissimilate_from: str,
    feature: str,
    direction: str = "progressive"
) -> DSLRule:
    """
    Creates a rule that checks for feature dissimilation.
    
    Args:
        target_class: Natural class that undergoes dissimilation
        dissimilate_from: Natural class that triggers dissimilation
        feature: Feature that dissimilates
        direction: "progressive" or "regressive"
        
    Returns:
        A rule that checks for proper dissimilation
        
    Example:
        # Lateral dissimilation: liquids lose laterality after another lateral
        lateral_dissimilation = create_dissimilation_rule(
            "liquid", "liquid", "lateral", "progressive"
        )
    """
    if target_class not in NATURAL_CLASSES or dissimilate_from not in NATURAL_CLASSES:
        raise ValueError(f"Unknown natural class: {target_class} or {dissimilate_from}")
        
    if feature not in FEATURE_VALUES:
        raise ValueError(f"Unknown feature: {feature}")
        
    target_features = NATURAL_CLASSES[target_class]
    trigger_features = NATURAL_CLASSES[dissimilate_from]
    
    def check_dissimilation(seq):
        if len(seq) < 2:
            return True
            
        for i in range(len(seq) - 1):
            # For progressive dissimilation: first = trigger, second = target
            # For regressive dissimilation: first = target, second = trigger
            if direction == "progressive":
                trigger_idx, target_idx = i, i+1
            else:  # regressive
                trigger_idx, target_idx = i+1, i
                
            # Check if the phonemes match the respective classes
            is_trigger = True
            for f, v in trigger_features.items():
                if f not in seq[trigger_idx] or seq[trigger_idx][f] != v:
                    is_trigger = False
                    break
                    
            is_target = True
            for f, v in target_features.items():
                if f not in seq[target_idx] or seq[target_idx][f] != v:
                    is_target = False
                    break
                    
            if not (is_trigger and is_target):
                continue
                
            # Check if the target has dissimilated from the trigger for the specified feature
            if feature not in seq[trigger_idx] or feature not in seq[target_idx]:
                return False
                
            if seq[target_idx][feature] == seq[trigger_idx][feature]:
                return False
                
        return True
    
    return DSLRule(
        check_dissimilation,
        f"{direction} dissimilation of {feature} between {target_class} and {dissimilate_from}"
    )


def create_syllable_structure_rule(
    allowed_patterns: List[str]
) -> DSLRule:
    """
    Creates a rule that checks syllable structure patterns.
    
    Args:
        allowed_patterns: List of allowed syllable patterns (e.g., ["CV", "CVC"])
        
    Returns:
        A rule that checks if syllables follow allowed patterns
        
    Example:
        syllable_rule = create_syllable_structure_rule(["CV", "CVC"])
    """
    def check_syllable_structure(seq):
        # First classify each phoneme as C or V
        cv_pattern = ""
        for phoneme in seq:
            if "syllabic" in phoneme and phoneme["syllabic"]:
                cv_pattern += "V"
            else:
                cv_pattern += "C"
                
        # Check if the pattern matches any allowed pattern
        for pattern in allowed_patterns:
            if cv_pattern == pattern:
                return True
                
        return False
    
    patterns_str = ", ".join(allowed_patterns)
    return DSLRule(check_syllable_structure, f"syllable must match patterns: {patterns_str}")


def create_sonority_rule(sonority_scale: Dict[str, int]) -> DSLRule:
    """
    Creates a rule that checks sonority sequencing.
    
    Args:
        sonority_scale: Dictionary mapping feature combinations to sonority values
        
    Returns:
        A rule that checks if sonority increases toward nucleus and decreases afterward
        
    Example:
        sonority_rule = create_sonority_rule({
            "stop": 1,
            "fricative": 2,
            "nasal": 3,
            "liquid": 4,
            "glide": 5,
            "vowel": 6
        })
    """
    def get_sonority(phoneme):
        for class_name, sonority in sonority_scale.items():
            if class_name not in NATURAL_CLASSES:
                continue
                
            features = NATURAL_CLASSES[class_name]
            matches = True
            for feature, value in features.items():
                if feature not in phoneme or phoneme[feature] != value:
                    matches = False
                    break
                    
            if matches:
                return sonority
                
        return 0  # Default for unknown
    
    def check_sonority(seq):
        if len(seq) <= 1:
            return True
            
        # Find nucleus (most sonorous element)
        sonorities = [get_sonority(phoneme) for phoneme in seq]
        nucleus_idx = sonorities.index(max(sonorities))
        
        # Check rising sonority before nucleus
        for i in range(nucleus_idx):
            if sonorities[i] > sonorities[i+1]:
                return False
                
        # Check falling sonority after nucleus
        for i in range(nucleus_idx, len(seq)-1):
            if sonorities[i] < sonorities[i+1]:
                return False
                
        return True
    
    return DSLRule(check_sonority, "sonority must rise toward nucleus and fall afterward")


def create_harmonization_rule(
    feature: str,
    domain: str = "word"
) -> DSLRule:
    """
    Creates a rule that checks for vowel harmony.
    
    Args:
        feature: Feature that must harmonize (e.g., "back", "round")
        domain: Domain over which harmony applies ("word", "syllable", etc.)
        
    Returns:
        A rule that checks if vowels harmonize for the specified feature
        
    Example:
        backness_harmony = create_harmonization_rule("back")
    """
    if feature not in FEATURE_VALUES:
        raise ValueError(f"Unknown feature: {feature}")
        
    def check_harmony(seq):
        # Get all vowels in the sequence
        vowels = []
        for phoneme in seq:
            if "syllabic" in phoneme and phoneme["syllabic"]:
                if feature in phoneme:
                    vowels.append(phoneme)
                    
        if not vowels:
            return True  # No vowels to check
            
        # Check if all vowels have the same value for the feature
        first_value = vowels[0][feature]
        return all(v[feature] == first_value for v in vowels)
    
    return DSLRule(check_harmony, f"vowels must harmonize for {feature}")


def create_phonotactic_rule(
    forbidden_sequences: List[Dict[str, Dict[str, bool]]]
) -> DSLRule:
    """
    Creates a rule that prohibits specific phoneme sequences.
    
    Args:
        forbidden_sequences: List of feature dictionaries representing forbidden sequences
        
    Returns:
        A rule that checks if the sequence contains no forbidden sequences
        
    Example:
        # No voiceless obstruent + voiced obstruent sequences
        phonotactic_rule = create_phonotactic_rule([
            {
                "first": {"sonorant": False, "voice": False},
                "second": {"sonorant": False, "voice": True}
            }
        ])
    """
    def check_phonotactics(seq):
        if len(seq) < 2:
            return True
            
        for i in range(len(seq) - 1):
            for forbidden in forbidden_sequences:
                if "first" in forbidden and "second" in forbidden:
                    first_match = True
                    for feature, value in forbidden["first"].items():
                        if feature not in seq[i] or seq[i][feature] != value:
                            first_match = False
                            break
                            
                    if not first_match:
                        continue
                        
                    second_match = True
                    for feature, value in forbidden["second"].items():
                        if feature not in seq[i+1] or seq[i+1][feature] != value:
                            second_match = False
                            break
                            
                    if first_match and second_match:
                        return False
                        
        return True
    
    return DSLRule(check_phonotactics, "sequence must not contain forbidden phoneme combinations") 