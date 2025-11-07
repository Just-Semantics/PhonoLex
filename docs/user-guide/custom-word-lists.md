# Custom Word Lists

The Custom Word Lists tool is the most powerful feature in PhonoLex, allowing you to build targeted word lists using multiple criteria.

## Overview

Build word lists by combining:
- Phoneme patterns (STARTS_WITH, ENDS_WITH, CONTAINS)
- Property filters (12 psycholinguistic properties)
- Phoneme exclusions

## Basic Usage

1. **Add a pattern**: Click "Add Pattern" and select a pattern type
2. **Choose a phoneme**: Use the IPA keyboard or type directly
3. **Add filters** (optional): Set ranges for frequency, imageability, etc.
4. **Generate**: Click "Generate List" to see results
5. **Export**: Download as CSV or copy individual words

## Pattern Types

- **STARTS_WITH**: Words beginning with the phoneme
- **ENDS_WITH**: Words ending with the phoneme
- **CONTAINS**: Words containing the phoneme anywhere
- **CONTAINS_MEDIAL**: Words containing the phoneme only in medial position (not initial or final)

## Property Filters

### Phonological Complexity
- Syllable count (1-5)
- Phoneme count (1-10+)
- WCM - Word Complexity Measure (0-15)
- MSH - Mean Syllable Height (1-6)

### Lexical Properties
- Frequency (SUBTLEX-US: 0-1000+)
- Age of Acquisition (1-7 scale)

### Semantic Properties
- Imageability (1-7 scale)
- Familiarity (1-7 scale)
- Concreteness (1-5 scale)

### Affective Properties
- Valence (1-9: negative to positive)
- Arousal (1-9: calm to excited)
- Dominance (1-9: weak to powerful)

## Examples

### Example 1: Simple /k/ initial words
- Pattern: STARTS_WITH /k/
- No filters
- Result: cat, car, kit, etc.

### Example 2: High-frequency /s/ words
- Pattern: STARTS_WITH /s/
- Filter: Frequency > 50
- Result: see, say, so, etc.

### Example 3: Concrete CVC words
- Patterns: STARTS_WITH (any consonant), ENDS_WITH (any consonant)
- Filter: Syllables = 1, Concreteness > 4.0
- Result: cat, dog, bed, etc.

## Tips

- **Start simple**: Begin with one pattern, then add filters
- **Check the count**: Preview shows how many words match before generating
- **Combine criteria**: Use multiple patterns and filters for precise lists
- **Export for later**: Save your results as CSV for use in therapy or research
