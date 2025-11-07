# Lookup

Look up words and phonemes, compare phoneme features, or search by distinctive features.

## Word Lookup

Search for any word to view comprehensive information:

### Phonological Properties
- IPA transcription
- Syllable count
- Phoneme count
- Word Complexity Measure (WCM)
- Mean Syllable Height (MSH)

### Psycholinguistic Properties

**Lexical:**
- Frequency (SUBTLEX-US)
- Age of Acquisition

**Semantic:**
- Imageability
- Familiarity
- Concreteness

**Affective:**
- Valence (negative to positive)
- Arousal (calm to excited)
- Dominance (weak to powerful)

### Example
Search "cat" to see:
- IPA: /kæt/
- Syllables: 1
- Phonemes: 3
- WCM: 3
- Frequency: 182.5
- Concreteness: 4.93
- etc.

## Phoneme Lookup

View detailed features for any IPA phoneme:

- **Phoneme type**: Vowel or consonant
- **38 distinctive features** from PHOIBLE database
- Feature values: '+' (present), '-' (absent), '0' (not applicable)

### Features Include
- consonantal, sonorant, continuant
- voice, nasal, lateral
- labial, coronal, dorsal
- high, low, front, back
- and 27 more...

### Example
Lookup /k/:
- Type: Consonant
- consonantal: +
- voice: -
- dorsal: +
- etc.

## Phoneme Comparison

Compare two phonemes feature-by-feature:

1. Enter first phoneme (e.g., /t/)
2. Enter second phoneme (e.g., /d/)
3. View comparison showing:
   - **Shared features**: Features with same values
   - **Different features**: Features that differ
   - **Similarity score**: Overall similarity (0.0-1.0)

### Example: /t/ vs /d/
- **Shared**: consonantal (+), sonorant (-), continuant (-), dorsal (-), ...
- **Different**: voice (- vs +), periodicGlottalSource (- vs +)
- **Score**: 0.94 (minimal pair - only voice differs)

## Search by Features

Find phonemes matching specific feature combinations:

1. Select features from dropdown
2. Set values (+, -, or 0)
3. View all matching phonemes

### Example Searches

**Find all voiced stops:**
- consonantal: +
- sonorant: -
- continuant: -
- voice: +
- Result: /b, d, g/

**Find all front vowels:**
- consonantal: -
- syllabic: +
- front: +
- Result: /i, ɪ, e, ɛ, æ/

## Tips

- **Word lookup** is case-insensitive
- **Phoneme lookup** requires exact IPA characters (use the IPA keyboard)
- **Feature search** shows only features with variation in English
- **Comparison** is useful for understanding minimal pairs and maximal opposition
