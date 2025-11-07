#!/usr/bin/env python3
"""
Verify all examples from docs/getting-started/examples.md
Tests examples programmatically using the exported client-side data.
"""

import json
from pathlib import Path

# Load word metadata
data_dir = Path(__file__).parent.parent / "webapp/frontend/public/data"
with open(data_dir / "word_metadata.json") as f:
    words = json.load(f)


def filter_words(filters):
    """Filter words based on criteria"""
    results = []
    for word, data in words.items():
        # Check all filters
        if "syllables" in filters:
            if (
                filters["syllables"][0] > data["syllable_count"]
                or filters["syllables"][1] < data["syllable_count"]
            ):
                continue

        if "phonemes" in filters:
            if (
                filters["phonemes"][0] > data["phoneme_count"]
                or filters["phonemes"][1] < data["phoneme_count"]
            ):
                continue

        if "wcm" in filters and data.get("wcm_score") is not None:
            if (
                filters["wcm"][0] > data["wcm_score"]
                or filters["wcm"][1] < data["wcm_score"]
            ):
                continue

        if "frequency" in filters and data.get("frequency") is not None:
            if filters["frequency"][0] > data["frequency"]:
                continue

        if "imageability" in filters and data.get("imageability") is not None:
            if filters["imageability"][0] > data["imageability"]:
                continue

        if "aoa" in filters and data.get("aoa") is not None:
            if filters["aoa"][0] > data["aoa"] or filters["aoa"][1] < data["aoa"]:
                continue

        if "msh" in filters and data.get("msh_stage") is not None:
            if (
                filters["msh"][0] > data["msh_stage"]
                or filters["msh"][1] < data["msh_stage"]
            ):
                continue

        if "valence" in filters and data.get("valence") is not None:
            if (
                filters["valence"][0] > data["valence"]
                or filters["valence"][1] < data["valence"]
            ):
                continue

        if "arousal" in filters and data.get("arousal") is not None:
            if (
                filters["arousal"][0] > data["arousal"]
                or filters["arousal"][1] < data["arousal"]
            ):
                continue

        if "concreteness" in filters and data.get("concreteness") is not None:
            if (
                filters["concreteness"][0] > data["concreteness"]
                or filters["concreteness"][1] < data["concreteness"]
            ):
                continue

        # Pattern matching
        if "starts_with" in filters:
            phonemes = data["phonemes"]  # Already a list
            pattern = (
                filters["starts_with"]
                if isinstance(filters["starts_with"], list)
                else [filters["starts_with"]]
            )
            if not phonemes[: len(pattern)] == pattern:
                continue

        if "ends_with" in filters:
            phonemes = data["phonemes"]
            pattern = (
                filters["ends_with"]
                if isinstance(filters["ends_with"], list)
                else [filters["ends_with"]]
            )
            if not phonemes[-len(pattern) :] == pattern:
                continue

        if "contains" in filters:
            phonemes = data["phonemes"]
            pattern = (
                filters["contains"]
                if isinstance(filters["contains"], list)
                else [filters["contains"]]
            )
            # Check if pattern appears anywhere in phonemes
            found = False
            for i in range(len(phonemes) - len(pattern) + 1):
                if phonemes[i : i + len(pattern)] == pattern:
                    found = True
                    break
            if not found:
                continue

        # Exclusions
        if "exclude" in filters:
            phonemes = set(data["phonemes"])
            excluded = set(
                filters["exclude"]
                if isinstance(filters["exclude"], list)
                else [filters["exclude"]]
            )
            if phonemes & excluded:  # If any overlap
                continue

        results.append(data)

    return sorted(results, key=lambda x: x["word"])


def test_example(name, filters, expected_keywords):
    """Test an example and return results"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")

    results = filter_words(filters)
    print(f"Found {len(results)} words")

    if len(results) > 0:
        print("\nFirst 10 results:")
        for word_data in results[:10]:
            wcm = word_data.get("wcm_score", "N/A")
            freq = word_data.get("frequency")
            freq_str = f"{freq:.1f}" if freq else "N/A"
            print(
                f"  {word_data['word']} /{word_data['ipa']}/ "
                f"(syl={word_data['syllable_count']}, ph={word_data['phoneme_count']}, "
                f"wcm={wcm}, freq={freq_str})"
            )

        # Check if expected keywords are in results
        result_words = [w["word"] for w in results]
        found = [kw for kw in expected_keywords if kw in result_words]
        missing = [kw for kw in expected_keywords if kw not in result_words]

        if found:
            print(f"\n✓ Found expected words: {', '.join(found)}")
        if missing:
            print(f"\n✗ Missing expected words: {', '.join(missing)}")

        return len(results), found, missing
    else:
        print("✗ No results found!")
        return 0, [], expected_keywords


# Run all Custom Word List examples
print("\n" + "=" * 60)
print("CUSTOM WORD LISTS EXAMPLES")
print("=" * 60)

# Example 1: Simple CVC Words
test_example(
    "Example 1: Simple CVC Words for Early Intervention",
    {"syllables": [1, 1], "phonemes": [1, 4], "wcm": [0, 3], "frequency": [10, 999999]},
    ["cat", "dog", "bed", "cup", "hat", "sit", "run"],
)

# Example 2: Initial /s/ Words with Semantic Scaffolding
test_example(
    "Example 2: Initial /s/ Words with Semantic Scaffolding",
    {"starts_with": "s", "imageability": [5.0, 7.0], "frequency": [5, 999999]},
    ["sun", "snake", "sock", "sand", "soap", "snow"],
)

# Example 3: Late-Developing Sounds in Simple Contexts
test_example(
    "Example 3: Late-Developing Sounds in Simple Contexts",
    {"contains": "ɹ", "syllables": [1, 1], "msh": [0, 4.0], "aoa": [0, 4.0]},
    ["hear", "hair", "air", "fair", "bear", "chair", "fear"],
)

# Example 4: Negative Valence Words
test_example(
    "Example 4: Negative Valence Words for Emotional Language",
    {
        "valence": [0, 3.0],
        "arousal": [5.0, 9.0],
        "frequency": [10, 999999],
        "imageability": [4.0, 7.0],
    },
    ["afraid", "scared", "mad", "evil", "dangerous", "attack"],
)

# Example 5: Excluding Problematic Phonemes
test_example(
    "Example 5: Excluding Problematic Phonemes",
    {"starts_with": "k", "exclude": "g", "syllables": [1, 2]},
    ["cat", "car", "cut", "candy"],
)

print("\n" + "=" * 60)
print("ADVANCED QUERIES EXAMPLES")
print("=" * 60)

# Example 16: High-Frequency Abstract Words
test_example(
    "Example 16: High-Frequency Abstract Words",
    {"frequency": [20, 999999], "concreteness": [0, 2.5], "aoa": [5.0, 7.0]},
    [
        "justice",
        "theory",
        "particular",
        "professional",
        "affair",
        "destiny",
        "former",
        "instance",
    ],
)

# Example 17: Emotionally Neutral Words
test_example(
    "Example 17: Emotionally Neutral Words",
    {"valence": [4.0, 6.0], "arousal": [3.0, 5.0], "frequency": [10, 999999]},
    ["table", "paper", "floor", "time", "thing", "work", "wait", "put"],
)

# Example 18: Multi-Pattern Complex Query
test_example(
    "Example 18: Multi-Pattern Complex Query",
    {"starts_with": "k", "ends_with": "t", "exclude": "s", "frequency": [5, 999999]},
    ["cat", "kit", "court", "caught", "coat"],
)

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("\nAll Custom Word Lists examples tested!")
print("Note: Some expected words may be missing due to:")
print("  - Different property thresholds in actual data")
print("  - Missing psycholinguistic norms for some words")
print("  - IPA transcription differences")
print("\nReview results above to update documentation examples with actual data.")
