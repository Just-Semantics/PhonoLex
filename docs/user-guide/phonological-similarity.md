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

## Technical Architecture

### Phoneme-Sequence Representation

PhonoLex represents words as **sequences of syllables**, with each syllable containing three components:

```
Word: "cat" /kæt/
Syllable 1:
  onset = [[k]]      (sequence of 1 phoneme vector: 76-dim)
  nucleus = [[æ]]    (sequence of 1 phoneme vector: 76-dim)
  coda = [[t]]       (sequence of 1 phoneme vector: 76-dim)

Word: "crest" /kɹɛst/
Syllable 1:
  onset = [[k], [ɹ]]   (sequence of 2 phoneme vectors)
  nucleus = [[ɛ]]      (sequence of 1 phoneme vector)
  coda = [[s], [t]]    (sequence of 2 phoneme vectors)
```

**Key Insight**: Consonant clusters and diphthongs are preserved as **sequences** of phoneme vectors, not averaged into single vectors. This allows proper discrimination of:
- Different cluster lengths ("cat" vs "crest")
- Different syllable structures ("cat" vs "act")
- Complex phoneme patterns ("spray" vs "say")

### Similarity Computation Algorithm

Phonological similarity is computed using **soft Levenshtein distance** on phoneme sequences with **weighted syllable components**.

**Algorithm Overview**:
1. Decompose both words into syllables (onset-nucleus-coda)
2. For each syllable pair, compute component distances:
   - Onset distance (soft Levenshtein on onset phoneme sequences)
   - Nucleus distance (soft Levenshtein on nucleus phoneme sequences)
   - Coda distance (soft Levenshtein on coda phoneme sequences)
3. Combine component distances using user-specified weights
4. Average across all syllable pairs
5. Convert distance to similarity: `similarity = 1 - distance`

### Soft Levenshtein Distance

Soft Levenshtein extends standard edit distance to use **phoneme vector similarity** instead of exact matches.

**Standard Levenshtein**: Counts insertions, deletions, substitutions
- /k/ vs /t/: cost = 1 (substitution)
- /k/ vs /k/: cost = 0 (match)

**Soft Levenshtein**: Uses cosine similarity between phoneme vectors
- /k/ vs /t/: cost = 1 - sim(k, t) = 1 - 0.65 = 0.35
- /k/ vs /g/: cost = 1 - sim(k, g) = 1 - 0.92 = 0.08 (voicing difference only)
- /k/ vs /k/: cost = 1 - 1.0 = 0.0 (identical)

**Algorithm** (for two phoneme sequences A and B):
```
def soft_levenshtein(A, B):
    # Initialize distance matrix
    D[i][0] = sum of insertion costs for A[0:i]
    D[0][j] = sum of insertion costs for B[0:j]

    # Fill matrix
    for i in 1 to len(A):
        for j in 1 to len(B):
            # Substitution cost based on phoneme similarity
            subst_cost = 1 - cosine_similarity(A[i], B[j])

            D[i][j] = min(
                D[i-1][j] + 1,              # deletion
                D[i][j-1] + 1,              # insertion
                D[i-1][j-1] + subst_cost    # substitution
            )

    # Normalize by maximum possible distance
    return D[len(A)][len(B)] / max(len(A), len(B))
```

**Why This Works**:
- Insertion/deletion cost = 1.0 (penalizes length differences)
- Substitution cost varies (0.0 to 1.0) based on phoneme similarity
- Normalization ensures output in [0, 1] range
- Similar phonemes have lower substitution costs

### Weight Application

User-specified weights control the importance of each syllable component.

**Weight Formula**:
```
weighted_distance = (w_onset × onset_dist + w_nucleus × nucleus_dist + w_coda × coda_dist) / (w_onset + w_nucleus + w_coda)
```

**Presets**:

| Preset | Onset | Nucleus | Coda | Effect |
|--------|-------|---------|------|--------|
| Rhymes | 0.0 | 0.5 | 0.5 | Ignores onset, prioritizes nucleus+coda |
| Alliteration | 1.0 | 0.0 | 0.0 | Only onset matters |
| Assonance | 0.0 | 1.0 | 0.0 | Only nucleus (vowel) matters |
| Consonance | 0.5 | 0.0 | 0.5 | Ignores vowels, prioritizes consonants |
| Balanced | 0.33 | 0.33 | 0.33 | All components equally weighted |

### Worked Example: "cat" vs "bat"

**Step 1: Syllabification**
```
"cat" /kæt/: onset=[k], nucleus=[æ], coda=[t]
"bat" /bæt/: onset=[b], nucleus=[æ], coda=[t]
```

**Step 2: Component Distances** (using Rhymes preset: onset=0.0, nucleus=0.5, coda=0.5)

*Onset distance*:
- Soft Levenshtein([k], [b]) = 0.35 (some similarity, both stops)

*Nucleus distance*:
- Soft Levenshtein([æ], [æ]) = 0.0 (identical)

*Coda distance*:
- Soft Levenshtein([t], [t]) = 0.0 (identical)

**Step 3: Weighted Distance**
```
weighted_distance = (0.0 × 0.35 + 0.5 × 0.0 + 0.5 × 0.0) / (0.0 + 0.5 + 0.5)
                  = 0.0 / 1.0
                  = 0.0
```

**Step 4: Convert to Similarity**
```
similarity = 1 - weighted_distance = 1 - 0.0 = 1.0
```

**Result**: "cat" and "bat" have similarity 1.0 with Rhymes preset (perfect rhyme, onset ignored).

### Worked Example: "cat" vs "crest"

**Step 1: Syllabification**
```
"cat" /kæt/: onset=[k], nucleus=[æ], coda=[t]
"crest" /kɹɛst/: onset=[k, ɹ], nucleus=[ɛ], coda=[s, t]
```

**Step 2: Component Distances** (using Balanced preset: all weights = 0.33)

*Onset distance*:
- Soft Levenshtein([k], [k, ɹ])
- Matrix calculation:
  ```
  D[0][0] = 0
  D[1][0] = 1 (delete k from A)
  D[0][1] = 1 (insert k to B)
  D[0][2] = 2 (insert k, ɹ to B)
  D[1][1] = min(D[0][1]+1, D[1][0]+1, D[0][0]+(1-sim(k,k))) = min(2, 2, 0) = 0
  D[1][2] = min(D[0][2]+1, D[1][1]+1, D[0][1]+(1-sim(k,ɹ))) = min(3, 1, 1.75) = 1
  ```
- Distance = 1 / max(1, 2) = 0.5

*Nucleus distance*:
- Soft Levenshtein([æ], [ɛ])
- Both are front vowels, high similarity
- Distance ≈ 0.15

*Coda distance*:
- Soft Levenshtein([t], [s, t])
- Matrix calculation:
  ```
  D[1][2] = 1 (insert s)
  ```
- Distance = 1 / max(1, 2) = 0.5

**Step 3: Weighted Distance**
```
weighted_distance = (0.33 × 0.5 + 0.33 × 0.15 + 0.33 × 0.5) / (0.33 + 0.33 + 0.33)
                  = (0.165 + 0.0495 + 0.165) / 0.99
                  = 0.3795 / 0.99
                  ≈ 0.38
```

**Step 4: Convert to Similarity**
```
similarity = 1 - 0.38 = 0.62
```

**Result**: "cat" and "crest" have similarity ~0.62 (moderately similar, penalized for cluster differences).

### Worked Example: "cat" vs "act"

**Step 1: Syllabification**
```
"cat" /kæt/: onset=[k], nucleus=[æ], coda=[t]
"act" /ækt/: onset=[], nucleus=[æ], coda=[k, t]
```

**Step 2: Component Distances** (using Balanced preset)

*Onset distance*:
- Soft Levenshtein([k], [])
- Distance = 1 / max(1, 0) = 1.0 (complete mismatch)

*Nucleus distance*:
- Soft Levenshtein([æ], [æ]) = 0.0 (identical)

*Coda distance*:
- Soft Levenshtein([t], [k, t])
- Distance ≈ 0.5 (extra phoneme)

**Step 3: Weighted Distance**
```
weighted_distance = (0.33 × 1.0 + 0.33 × 0.0 + 0.33 × 0.5) / 0.99
                  = (0.33 + 0 + 0.165) / 0.99
                  = 0.495 / 0.99
                  ≈ 0.50
```

**Step 4: Convert to Similarity**
```
similarity = 1 - 0.50 = 0.50
```

**Result**: "cat" and "act" have similarity ~0.50 (anagrams properly distinguished due to different syllable structures).

## Score Interpretation Guide

Understanding what similarity scores mean in different contexts:

### Perfect Matches (1.0)
- Identical words (same phonemes, same positions)
- Perfect rhymes with Rhymes preset (cat-bat: 1.0)
- Perfect alliteration with Alliteration preset (cat-cap: ~0.95+)

### Very High Similarity (0.85-0.99)
- Perfect rhymes (cat-hat: 0.95)
- One-phoneme substitutions in similar classes (cat-cap: 0.88)
- Minimal vowel differences (cat-cot: 0.90)

### High Similarity (0.70-0.84)
- Imperfect rhymes (cat-cap: 0.78)
- Similar clusters (spray-splay: 0.75)
- Vowel + coda matches (cat-bad: 0.72)

### Moderate Similarity (0.55-0.69)
- Shared onset + nucleus (cat-cab: 0.65)
- One component matches well (cat-crest: 0.62)
- Similar phoneme distributions (cat-cut: 0.60)

### Low Similarity (0.40-0.54)
- Anagrams with different syllable structures (cat-act: 0.50)
- Some shared phonemes (cat-talk: 0.45)
- Length differences (cat-catastrophe: 0.42)

### Very Low Similarity (< 0.40)
- Completely different phonemes (cat-dog: 0.25)
- Different syllable counts (cat-computer: 0.20)
- No shared components (cat-box: 0.15)

**Note**: Scores depend heavily on weight settings. The above ranges assume Balanced preset (all weights = 0.33).

## Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| Word lookup | 5-10 ms | O(1) hash lookup |
| Syllabification | 1-2 ms | O(n) where n = phoneme count |
| Soft Levenshtein (per component) | 0.1-0.5 ms | O(m × n) where m, n = sequence lengths |
| Full vocabulary scan | 50-100 ms | 48,720 comparisons |
| Top-20 results | 60-120 ms | Full scan + sort |

**Factors Affecting Speed**:
- Target word complexity (more syllables = slower)
- Vocabulary size (more words = slower scan)
- Browser performance (Chrome/Edge slightly faster)

**Optimization**: Results are computed on-the-fly (not cached). Changing weights requires recomputation.

## Limitations and Edge Cases

### Monosyllabic vs. Multisyllabic Comparisons

Comparing words with different syllable counts can produce unexpected results:

```
"cat" /kæt/ (1 syllable) vs "catalog" /kætəlɔg/ (3 syllables)

Result: Low similarity (~0.35) due to length penalty
```

**Reason**: Soft Levenshtein normalizes by maximum length, but syllable count differences are heavily penalized.

### Empty Components

Words with missing onset or coda:

```
"act" /ækt/: onset=[] (no onset)
"see" /si/: coda=[] (no coda)

Comparing "act" vs "cat":
- Onset distance = 1.0 (empty vs [k])
- Nucleus distance = 0.0 (both [æ])
- Coda distance = 0.5 ([k,t] vs [t])
```

**Effect**: Empty components are treated as complete mismatches (distance = 1.0).

### Diphthongs

Diphthongs are represented as sequences:

```
"time" /taɪm/: nucleus = [a, ɪ] (2 vectors)
"team" /tim/: nucleus = [i] (1 vector)

Nucleus distance ≈ 0.5 (length difference penalized)
```

**Effect**: Diphthongs vs. monophthongs incur insertion/deletion penalties.

### Weight = 0.0 Does Not Mean "Ignore"

Zero weights still contribute to normalization:

```
Rhymes preset: onset=0.0, nucleus=0.5, coda=0.5

weighted_distance = (0.0 × onset_dist + 0.5 × nucleus_dist + 0.5 × coda_dist) / 1.0
```

**Interpretation**: Onset distance is computed but multiplied by 0, effectively ignoring it.

### Stress and Syllable Alignment

Multi-syllabic words compare syllables pairwise:

```
"computer" /kəmpjutɚ/ (3 syllables)
"commuter" /kəmjutɚ/ (3 syllables)

Syllable 1: /kəm/ vs /kəm/ → high similarity
Syllable 2: /pju/ vs /ju/ → moderate similarity (extra /p/)
Syllable 3: /tɚ/ vs /tɚ/ → high similarity

Overall: ~0.80 (high similarity)
```

**Note**: Syllable alignment is position-based (1st syllable vs 1st syllable, etc.). No dynamic alignment.

### Vocabulary Coverage

- **Full vocabulary**: 48,720 words from CMU Pronouncing Dictionary
- **Dialect**: General American English (primary pronunciations only)
- **Excluded**: Pronunciation variants, proper nouns, non-English loanwords

**Implication**: Some words you search for may not be in the vocabulary.

## Advanced Use Cases

### Finding Imperfect Rhymes (Slant Rhymes)

**Goal**: Find words that rhyme approximately but not perfectly

**Settings**:
- Target: "love"
- Preset: Rhymes (onset=0.0, nucleus=0.5, coda=0.5)
- Threshold: 0.70-0.85 (exclude perfect rhymes)

**Expected Results**: of, above, dove, shove, glove (varying vowel quality)

### Finding Consonant Frames

**Goal**: Find words with similar consonant structure but different vowels

**Settings**:
- Target: "big"
- Preset: Consonance (onset=0.5, nucleus=0.0, coda=0.5)
- Threshold: 0.75+

**Expected Results**: bag, bug, bog, beg (same /b_g/ frame)

### Finding Phonetic Neighbors

**Goal**: Find words that sound overall similar (useful for phonological neighborhood density studies)

**Settings**:
- Target: any word
- Preset: Balanced (all=0.33)
- Threshold: 0.70+
- Limit: 50

**Expected Results**: One-phoneme substitutions, additions, deletions

### Custom Weight Exploration

**Goal**: Test hypotheses about phonological similarity

**Example**: "Does onset matter more than coda for word recognition?"

**Settings**:
- Try onset=0.5, nucleus=0.25, coda=0.25
- Compare results to coda=0.5, nucleus=0.25, onset=0.25
- Observe which words appear in top results

## Data Sources and Phoneme Vectors

**Phoneme Vectors**: 76-dimensional or 152-dimensional continuous vectors derived from PHOIBLE distinctive features:
- 38 ternary features (+, -, 0) from PHOIBLE database
- Normalized to continuous values (0.0, 0.5, 1.0)
- Consonants/monophthongs: 76-dim (38 features × 2 for positive/negative)
- Diphthongs: 152-dim (concatenation of two 76-dim vectors)

**Syllabification**: Based on English phonotactic constraints
- Maximal onset principle
- Sonority sequencing
- Legal cluster identification

**Coverage**: 39 English phonemes (General American English from CMU Pronouncing Dictionary)

## See Also

- [Technical Architecture](../technical/architecture.md) - Complete phoneme-sequence architecture documentation
- [Practical Examples](../getting-started/examples.md) - Examples 9-11 for hands-on practice
- [Custom Word Lists](custom-word-lists.md) - Alternative pattern-based word finding
- [Lookup](lookup.md) - View phoneme features and compare phonemes directly
