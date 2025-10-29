"""
Phonological features and natural classes.
"""

# Define phonological features
FEATURE_VALUES = {
    "syllabic": [True, False],
    "consonantal": [True, False],
    "sonorant": [True, False],
    "continuant": [True, False],
    "voice": [True, False],
    "nasal": [True, False],
    "strident": [True, False],
    "lateral": [True, False],
    "labial": [True, False],
    "coronal": [True, False],
    "dorsal": [True, False],
    "pharyngeal": [True, False],
    "glottal": [True, False],
    "high": [True, False],
    "low": [True, False],
    "back": [True, False],
    "round": [True, False],
    "tense": [True, False],
    "long": [True, False],
    "stress": [True, False]
}

# Define natural classes
NATURAL_CLASSES = {
    "vowel": {
        "syllabic": True,
        "consonantal": False,
        "sonorant": True
    },
    "consonant": {
        "consonantal": True
    },
    "stop": {
        "consonantal": True,
        "continuant": False
    },
    "fricative": {
        "consonantal": True,
        "continuant": True,
        "strident": True
    },
    "obstruent": {
        "consonantal": True,
        "sonorant": False
    },
    "sonorant": {
        "sonorant": True
    },
    "nasal": {
        "consonantal": True,
        "sonorant": True,
        "nasal": True
    },
    "liquid": {
        "consonantal": True,
        "sonorant": True,
        "continuant": True
    },
    "labial": {
        "labial": True
    },
    "coronal": {
        "coronal": True
    },
    "dorsal": {
        "dorsal": True
    },
    "voiced": {
        "voice": True
    },
    "voiceless": {
        "voice": False
    }
}

# Sonority hierarchy (from most to least sonorous)
SONORITY_HIERARCHY = [
    "vowel",
    "glide",
    "liquid",
    "nasal",
    "fricative",
    "stop"
]

def get_sonority_rank(phoneme_class: str) -> int:
    """Get the sonority rank of a phoneme class."""
    try:
        return SONORITY_HIERARCHY.index(phoneme_class)
    except ValueError:
        return -1

def get_natural_class(class_name: str) -> dict:
    """Get the features of a natural class."""
    if class_name not in NATURAL_CLASSES:
        raise ValueError(f"Unknown natural class: {class_name}")
    return NATURAL_CLASSES[class_name]

def matches_natural_class(features: dict, class_name: str) -> bool:
    """Check if a set of features matches a natural class."""
    class_features = get_natural_class(class_name)
    return all(
        features.get(feature) == value 
        for feature, value in class_features.items()
    )

__all__ = [
    "FEATURE_VALUES",
    "NATURAL_CLASSES",
    "SONORITY_HIERARCHY",
    "get_sonority_rank",
    "get_natural_class",
    "matches_natural_class"
] 