"""
Example API client for PhonoLex.

This example demonstrates how to use the PhonoLex API to check phonological rules.
"""

import requests
import json

# API endpoint
API_URL = "http://localhost:8000"

def print_separator():
    """Print a separator line."""
    print("\n" + "=" * 60 + "\n")

def get_features():
    """Get all possible feature values."""
    response = requests.get(f"{API_URL}/features")
    return response.json()["features"]

def get_natural_classes():
    """Get all defined natural classes."""
    response = requests.get(f"{API_URL}/natural-classes")
    return response.json()["natural_classes"]

def check_rule(phonemes, rule_type, parameters):
    """Check if phonemes satisfy a rule."""
    data = {
        "phonemes": [
            {
                "ipa": p["ipa"],
                "features": {k: v for k, v in p.items() if k != "ipa"}
            }
            for p in phonemes
        ],
        "rule": {
            "rule_type": rule_type,
            "parameters": parameters
        }
    }
    
    response = requests.post(f"{API_URL}/check-rule", json=data)
    return response.json()

def main():
    """Run API client examples."""
    print("PhonoLex API Client Example")
    print_separator()
    
    # Example 1: Check feature match rule
    print("Example 1: Feature Match Rule")
    result = check_rule(
        # Create vowels
        phonemes=[
            {"ipa": "a", "syllabic": True, "consonantal": False, "sonorant": True},
            {"ipa": "e", "syllabic": True, "consonantal": False, "sonorant": True},
            {"ipa": "i", "syllabic": True, "consonantal": False, "sonorant": True}
        ],
        rule_type="feature_match",
        parameters={"features": {"syllabic": True}}
    )
    print(f"Rule: {result['rule_description']}")
    print(f"Valid: {result['valid']}")
    print_separator()
    
    # Example 2: Check natural class rule
    print("Example 2: Natural Class Rule")
    result = check_rule(
        # Create stops
        phonemes=[
            {"ipa": "p", "consonantal": True, "sonorant": False, "continuant": False},
            {"ipa": "t", "consonantal": True, "sonorant": False, "continuant": False},
            {"ipa": "k", "consonantal": True, "sonorant": False, "continuant": False}
        ],
        rule_type="natural_class",
        parameters={"class_name": "stop"}
    )
    print(f"Rule: {result['rule_description']}")
    print(f"Valid: {result['valid']}")
    print_separator()
    
    # Example 3: Check environment rule
    print("Example 3: Environment Rule")
    result = check_rule(
        # Create word "ata"
        phonemes=[
            {"ipa": "a", "syllabic": True, "consonantal": False, "sonorant": True},
            {"ipa": "t", "consonantal": True, "sonorant": False, "continuant": False},
            {"ipa": "a", "syllabic": True, "consonantal": False, "sonorant": True}
        ],
        rule_type="environment",
        parameters={
            "target_features": {"consonantal": True, "sonorant": False},
            "left_context": {"syllabic": True},
            "right_context": {"syllabic": True}
        }
    )
    print(f"Rule: {result['rule_description']}")
    print(f"Valid: {result['valid']}")
    print_separator()
    
    # Example 4: Check syllable structure rule
    print("Example 4: Syllable Structure Rule")
    result = check_rule(
        # Create word "pa"
        phonemes=[
            {"ipa": "p", "consonantal": True, "syllabic": False},
            {"ipa": "a", "consonantal": False, "syllabic": True}
        ],
        rule_type="syllable_structure",
        parameters={"allowed_patterns": ["CV", "CVC"]}
    )
    print(f"Rule: {result['rule_description']}")
    print(f"Valid: {result['valid']}")
    print_separator()
    
    print("Done!")

if __name__ == "__main__":
    main() 