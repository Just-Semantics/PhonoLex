# Pattern Builder Component

The Pattern Builder is a powerful tool for creating and searching phonological patterns in the PhonoLex system. It allows users to construct complex patterns based on phonemes, their features, and positions within words.

## Features

- Create patterns with multiple elements
- Support for three matching types:
  - **Exact Match**: Find words containing specific phonemes
  - **Similar Match**: Find words with phonemes similar to a target (based on vector similarity)
  - **Feature-Based**: Find words with phonemes matching certain feature patterns
- Position-based filtering (word-initial, word-final, middle, or any position)
- Interactive results display
- Detailed word information for matched items

## Pattern Elements

Each pattern consists of one or more elements, where each element defines:

1. **Type**: How to match phonemes (exact, similar, or feature-based)
2. **Position**: Where in the word to look for the pattern
3. **Value**: The target phoneme or feature values to match against

### Example Patterns

#### Simple Consonant Pattern

- Type: Exact
- Position: Initial
- Value: "p"

This will find all words that start with the phoneme "p".

#### Similar Vowel Pattern

- Type: Similar
- Position: Any
- Value: "æ" (with similarity threshold 0.8)

This will find all words containing vowels that are similar to "æ" based on vector similarity.

#### Complex Feature Pattern

- Type: Feature-based
- Position: Final
- Features: +voice, +nasal (weighted appropriately)

This will find all words ending with voiced nasal consonants.

## Implementation Details

The Pattern Builder leverages the vector-based phoneme representation in PhonoLex to enable powerful pattern matching capabilities. It uses cosine similarity for vector comparisons and allows for gradient feature matching.

## Future Enhancements

- Pattern saving and loading
- Pattern combination (AND/OR operations)
- More advanced position specifications (syllable-based, stress-based)
- Visualization of pattern matches in vector space
- Support for cross-linguistic pattern matching 