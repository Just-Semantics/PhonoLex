"""
Example demonstrating phonological rules using PhonoLex.

This example demonstrates how to:
1. Create phonemes with features
2. Define phonological rules
3. Check if sequences satisfy rules
4. Create custom rules
"""

from typing import List
from phonolex.rules import (
    Phoneme,
    Rule,
    create_feature_match_rule,
    create_natural_class_rule,
    create_environment_rule,
    create_assimilation_rule,
    create_custom_rule,
    check_sequence,
    check_all_rules,
    NATURAL_CLASSES
)

# Create some phonemes with features
phonemes = [
    # Word: "bat"
    Phoneme(
        ipa="b",
        features={
            "consonantal": True,
            "sonorant": False,
            "continuant": False,
            "voice": True,
            "labial": True
        }
    ),
    Phoneme(
        ipa="æ",
        features={
            "syllabic": True,
            "consonantal": False,
            "sonorant": True,
            "low": True,
            "front": True
        }
    ),
    Phoneme(
        ipa="t",
        features={
            "consonantal": True,
            "sonorant": False,
            "continuant": False,
            "voice": False,
            "coronal": True
        }
    ),
    
    # Word: "dog"
    Phoneme(
        ipa="d",
        features={
            "consonantal": True,
            "sonorant": False,
            "continuant": False,
            "voice": True,
            "coronal": True
        }
    ),
    Phoneme(
        ipa="ɔ",
        features={
            "syllabic": True,
            "consonantal": False,
            "sonorant": True,
            "low": True,
            "back": True,
            "round": True
        }
    ),
    Phoneme(
        ipa="g",
        features={
            "consonantal": True,
            "sonorant": False,
            "continuant": False,
            "voice": True,
            "dorsal": True
        }
    ),
    
    # Word: "cat"
    Phoneme(
        ipa="k",
        features={
            "consonantal": True,
            "sonorant": False,
            "continuant": False,
            "voice": False,
            "dorsal": True
        }
    ),
    Phoneme(
        ipa="æ",
        features={
            "syllabic": True,
            "consonantal": False,
            "sonorant": True,
            "low": True,
            "front": True
        }
    ),
    Phoneme(
        ipa="t",
        features={
            "consonantal": True,
            "sonorant": False,
            "continuant": False,
            "voice": False,
            "coronal": True
        }
    )
]

# Group phonemes into words
bat = phonemes[0:3]
dog = phonemes[3:6]
cat = phonemes[6:9]

print("=== Example 1: Feature-Based Rules ===")
# Create a rule for vowels
vowel_rule = create_feature_match_rule({"syllabic": True})
print("Vowels only?", check_sequence([p for p in bat if p.features.get("syllabic", False)], vowel_rule))  # True

# Create a natural class rule for stops
stop_rule = create_natural_class_rule("stop")
print("Stops only?", check_sequence([bat[0], bat[2]], stop_rule))  # True

print("\n=== Example 2: Environment Rules ===")
# Create a rule for voiced obstruents between vowels
intervocalic_voicing = create_environment_rule(
    {"consonantal": True, "sonorant": False},
    {"syllabic": True},
    {"syllabic": True}
)

# Check if any consonants in the sequence appear between vowels
bat_dog = bat + dog
print("Contains intervocalic obstruents?", check_sequence(bat_dog, intervocalic_voicing))  # True (t in bat + dog)

print("\n=== Example 3: Custom Rules ===")
# Create a custom rule for checking syllable structure (CV)
def check_cv_structure(sequence: List[Phoneme], index: int) -> bool:
    if index == 0 and len(sequence) >= 2:
        # Check if first phoneme is a consonant
        if not sequence[0].features.get("consonantal", False):
            return False
        # Check if second phoneme is a vowel
        if not sequence[1].features.get("syllabic", False):
            return False
        return True
    return False

cv_rule = create_custom_rule(
    check_cv_structure,
    "CV Structure",
    "Check if the sequence starts with a consonant followed by a vowel"
)

print("Starts with CV?", check_sequence(bat, cv_rule))  # True
print("Starts with CV?", check_sequence(dog, cv_rule))  # True
print("Starts with CV?", check_sequence(cat, cv_rule))  # True

print("\n=== Example 4: Multiple Rules ===")
# Create a set of rules for a valid word
word_rules = [
    create_natural_class_rule("consonant"),  # Must contain consonants
    create_natural_class_rule("vowel"),      # Must contain vowels
    create_feature_match_rule({"voice": True})  # Must contain voiced sounds
]

# Check if words satisfy all rules
all_words = [bat, dog, cat]
for i, word in enumerate(["bat", "dog", "cat"]):
    print(f"{word} satisfies all rules?", check_all_rules(all_words[i], word_rules))

print("\n=== Example 5: Displaying Features ===")
# Print natural classes and their features
print("Natural Classes:")
for class_name, features in NATURAL_CLASSES.items():
    print(f"  {class_name}: {features}")

print("\nDone!") 