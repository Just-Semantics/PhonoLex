# Phonological Similarity

Find phonologically similar words with adjustable weights for onset, nucleus, and coda components.

## Overview

The Phonological Similarity tool uses **phoneme-sequence soft Levenshtein distance** to find similar words while preserving:
- Consonant clusters (e.g., /kr/, /st/)
- Diphthongs (e.g., /aɪ/, /oʊ/)
- Syllable structure

## Basic Usage

1. Enter a **target word**
2. Choose a **preset** or set custom weights
3. Adjust **threshold** (how similar words must be)
4. Set **limit** (maximum results)
5. Click **Find Similar Words**

## Weight Presets

### Rhymes
- Onset: 0.0, Nucleus: 0.5, Coda: 0.5
- Matches nucleus and coda sounds
- Example: "cat" → bat, hat, sat, mat

### Alliteration
- Onset: 1.0, Nucleus: 0.0, Coda: 0.0
- Matches initial sounds only
- Example: "cat" → can, cap, cast, kit

### Assonance
- Onset: 0.0, Nucleus: 1.0, Coda: 0.0
- Matches vowel sounds only
- Example: "cat" → bad, had, slam

### Consonance
- Onset: 0.5, Nucleus: 0.0, Coda: 0.5
- Matches consonants, ignores vowels
- Example: "cat" → kit, cot, cut

### Balanced
- Onset: 0.33, Nucleus: 0.33, Coda: 0.33
- Considers all components equally
- Example: "cat" → similar overall sound

## Understanding Similarity Scores

Scores range from 0.0 (completely different) to 1.0 (identical):

- **0.90+**: Perfect rhymes (cat-bat)
- **0.75-0.89**: Very similar (cat-cap)
- **0.60-0.74**: Moderately similar (cat-crest)
- **< 0.60**: Somewhat different

## Custom Weights

Adjust sliders to create your own similarity definition:

- **Increase onset** for more initial sound matching
- **Increase nucleus** for more vowel matching
- **Increase coda** for more final sound matching

## Threshold Control

- **High threshold (0.85+)**: Only very similar words
- **Medium threshold (0.70-0.84)**: Moderately similar words
- **Low threshold (< 0.70)**: Broader matches

## Advanced: Phoneme-Sequence Architecture

Unlike traditional approaches that average phonemes, PhonoLex preserves sequences:

- **Traditional**: /kr/ → average of /k/ and /r/ (loses information)
- **PhonoLex**: /kr/ → sequence of [k], [r] (preserves cluster)

This correctly discriminates:
- "cat" vs "crest": 0.74 (different cluster lengths penalized)
- "cat" vs "act": 0.20 (different syllable structures properly distinguished)
