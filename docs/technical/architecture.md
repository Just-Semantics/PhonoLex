# Technical Architecture

This document provides technical details on PhonoLex's architecture and computational methods.

## System Overview

PhonoLex is a fully client-side web application that performs phonological analysis using:

- **48,720 English words** from the CMU Pronouncing Dictionary
- **38 distinctive phonological features** from the PHOIBLE database (covering 2,716 languages)
- **12 psycholinguistic properties** from 4 major research datasets
- **Phoneme-sequence similarity** using soft Levenshtein distance

### Client-Side Architecture

All computation happens in the browser using static JSON data files:

```
Frontend (React + TypeScript)
    ↓
Data Service (clientSideData.ts)
    ↓
Static JSON Files (~525 MB uncompressed, ~10.4 MB gzipped)
    ├── word_metadata.json (48,720 words with all properties)
    ├── embeddings.json.gz (phoneme-sequence syllable structures)
    ├── minimal_pairs.json.gz (112,964 precomputed minimal pairs)
    ├── phoneme_features.json.gz (38 PHOIBLE features for 35 phonemes)
    ├── syllable_structures.json.gz (syllable decompositions)
    └── arpa_to_ipa.json.gz (CMU ARPA to IPA mapping)
```

**Benefits:**
- Zero server costs
- No network latency
- Works offline
- 98% compression (525 MB → 10.4 MB with gzip)
- Instant queries after initial load

## Three-Phase Pipeline

PhonoLex uses a deterministic 3-phase pipeline to transform raw phonological features into similarity computations. All phases are pure computation from linguistic databases.

**Pipeline workflow:**
```bash
# Phase 1: Extract PHOIBLE features
python scripts/compute_phase1_features.py         # ~1 sec

# Phase 2: Normalize to continuous vectors
python scripts/compute_phase2_normalized_vectors.py  # ~5 sec

# Phase 3: Build phoneme-sequence syllable structures
python scripts/build_phase3_syllable_embeddings.py   # ~5 min

# Export to client-side data (SEPARATE step - recomputes minimal pairs)
python scripts/export_clientside_data.py            # ~5-10 min
```

**Important:** The export script is a **separate manual step** that must be run after Phase 3. This is where minimal pairs are computed from scratch.

### Phase 1: Feature Extraction

**Purpose:** Extract universal phonological features from PHOIBLE database

**Input:** PHOIBLE database CSV (2,716 languages, 38 distinctive features)

**Output:** 38-dimensional ternary feature vectors for English phonemes

**Format:** Ternary values (+, -, 0)
- `+` = feature is present
- `-` = feature is absent
- `0` = feature is not applicable

**Computation time:** <1 second

**Script:** `scripts/compute_phase1_features.py`

**Output file:** `embeddings/phase1/phoible_features.csv`

**Example - /k/ (voiceless velar stop):**
```
syllabic: -
consonantal: +
sonorant: -
continuant: -
voice: -
nasal: -
dorsal: +
high: +
back: +
... (29 more features)
```

### Phase 2: Normalization

**Purpose:** Convert ternary features to continuous normalized vectors

**Input:** Phase 1 ternary features (38-dimensional)

**Output:** Continuous normalized vectors (76-dimensional)

**Transformation:**
1. Ternary (+, -, 0) → Continuous (+1.0, -1.0, 0.0)
2. Each feature gets 2 dimensions: [start, end] to model articulation dynamics
3. 38 features × 2 = 76 dimensions
4. Normalized to unit length per phoneme

**Why 76 dimensions?**
- Supports diphthong modeling (vowel trajectories from start to end)
- [start, end] captures articulatory movement
- Enables interpolation for phonological processes

**Computation time:** ~5 seconds

**Script:** `scripts/compute_phase2_normalized_vectors.py`

**Output file:** `embeddings/phase2/normalized_76d.pkl`

**Special case - Diphthongs:**
- Diphthongs use 152 dimensions (76 for each vowel endpoint)
- Examples: /aɪ/ (as in "time"), /oʊ/ (as in "go"), /aʊ/ (as in "cow")

### Phase 3: Phoneme-Sequence Syllable Structures (v2.3)

**Purpose:** Build phoneme-sequence syllable representations

**Input:** Phase 2 normalized phoneme vectors (76-dim)

**Output:** Syllable structures with onset-nucleus-coda as **sequences of phoneme vectors**

**Technical approach:** Each syllable component (onset, nucleus, coda) is represented as a **sequence of phoneme vectors** rather than a single averaged vector. This preserves the sequential structure of consonant clusters and diphthongs.

**Data structure:**
```typescript
interface SyllableStructure {
  onset: number[][];   // Sequence of 76-dim phoneme vectors
  nucleus: number[][]; // Sequence of 76-dim or 152-dim vectors
  coda: number[][];    // Sequence of 76-dim phoneme vectors
}
```

**Examples:**

| Word | Onset | Nucleus | Coda |
|------|-------|---------|------|
| cat | [[k]] (1×76) | [[æ]] (1×76) | [[t]] (1×76) |
| crest | [[k], [ɹ]] (2×76) | [[ɛ]] (1×76) | [[s], [t]] (2×76) |
| time | [[t]] (1×76) | [[aɪ]] (1×152) | [[m]] (1×76) |
| spray | [[s], [p], [ɹ]] (3×76) | [[eɪ]] (1×152) | [] (0×76) |

**Algorithm:**
1. Syllabify word using English phonotactic constraints
2. For each syllable, extract onset/nucleus/coda phoneme sequences
3. Map each phoneme to its Phase 2 normalized vector
4. Store as sequences (preserves cluster length and order)

**Computation time:** ~5 minutes for full vocabulary

**Script:** `scripts/build_phase3_syllable_embeddings.py`

**Output file:** `embeddings/phase3/syllable_embeddings_phoible.pt`

**Example: Comparing cat vs crest:**

```
cat:   onset = [[k]]       (length = 1)
crest: onset = [[k], [ɹ]]  (length = 2)

Soft Levenshtein distance properly accounts for:
- Phoneme-level similarity: cosine(k, k) = 1.0, cosine(k, ɹ) = 0.15
- Sequence-level alignment: edit distance penalizes length difference
- Result: onset similarity = 0.50 (reflects structural difference)
```

This sequence-based approach correctly distinguishes words with different consonant cluster lengths, preserving phonotactic information critical for phonological analysis.

## Soft Levenshtein Distance

PhonoLex uses soft (fuzzy) Levenshtein distance to compare phoneme sequences, where phoneme similarity is measured using cosine distance between feature vectors rather than binary matching.

### Algorithm

Soft Levenshtein extends edit distance by allowing partial phoneme matches with continuous similarity scores (0.0 to 1.0) instead of binary matches (0 or 1).

**Step 1: Phoneme-level similarity**
```
sim(phoneme1, phoneme2) = cosine_similarity(vec1, vec2)
```

**Step 2: Component-level similarity (onset, nucleus, coda)**

Build dynamic programming table:

```
For sequences A = [a₁, a₂, ...] and B = [b₁, b₂, ...]:

DP[i][j] = minimum edit cost to align A[0:i] with B[0:j]

DP[i][j] = min(
    DP[i-1][j] + 1.0,              # delete from A
    DP[i][j-1] + 1.0,              # insert into A
    DP[i-1][j-1] + (1 - sim(aᵢ, bⱼ)) # substitute
)
```

**Step 3: Normalize to similarity score**
```
edit_distance = DP[len(A)][len(B)]
max_length = max(len(A), len(B))
similarity = 1 - (edit_distance / max_length)
```

**Step 4: Weighted component average**
```
syllable_sim = w_onset * sim_onset +
               w_nucleus * sim_nucleus +
               w_coda * sim_coda
```

Where weights sum to 1.0.

### Example: cat vs crest

**Syllable structures:**
- cat: onset=[k], nucleus=[æ], coda=[t]
- crest: onset=[k, ɹ], nucleus=[ɛ], coda=[s, t]

**Onset comparison ([k] vs [k, ɹ]):**

| DP | ∅ | k | ɹ |
|----|---|---|---|
| ∅  | 0 | 1 | 2 |
| k  | 1 | 0.0 | 1.0 |

Edit distance = 1.0
Max length = 2
Similarity = 1 - (1.0 / 2) = **0.50**

**Nucleus comparison ([æ] vs [ɛ]):**

| DP | ∅ | ɛ |
|----|---|---|
| ∅  | 0 | 1 |
| æ  | 1 | 0.15 |

Cosine sim(æ, ɛ) = 0.85
Edit distance = 0.15
Similarity = **0.85**

**Coda comparison ([t] vs [s, t]):**

| DP | ∅ | s | t |
|----|---|---|---|
| ∅  | 0 | 1 | 2 |
| t  | 1 | 0.70 | 0.70 |

Edit distance = 0.70
Max length = 2
Similarity = **0.65**

**Overall (balanced weights: onset=0.33, nucleus=0.33, coda=0.33):**
```
similarity = 0.33 * 0.50 + 0.33 * 0.85 + 0.33 * 0.65
           = 0.165 + 0.281 + 0.215
           = 0.66
```

**With rhyme weights (onset=0.0, nucleus=0.5, coda=0.5):**
```
similarity = 0.0 * 0.50 + 0.5 * 0.85 + 0.5 * 0.65
           = 0.0 + 0.425 + 0.325
           = 0.75
```

### Multi-Syllable Words

For words with multiple syllables, apply soft Levenshtein at the syllable level:

```
word_sim = soft_levenshtein(syllables1, syllables2)
```

Where each syllable is compared using the weighted component similarity from above.

## Data Coverage

### Vocabulary

**Source:** CMU Pronouncing Dictionary (primary pronunciations only)

**Size:** 48,720 words

**Selection criteria:**
- Primary pronunciations only (no variants)
- Valid IPA mapping available
- Frequency data available (relaxed from v2.0's "frequency + 1 norm" requirement)

**Phoneme inventory:** 35 English phonemes (General American dialect)

**Consonants (24):**
- Stops: p, b, t, d, k, g
- Fricatives: f, v, θ, ð, s, z, ʃ, ʒ, h
- Affricates: ʧ, ʤ
- Nasals: m, n, ŋ
- Liquids: l, ɹ
- Glides: w, j

**Vowels (11 monophthongs + 5 diphthongs = 16, but some overlap, total 15 distinct):**
- Monophthongs: i, ɪ, e, ɛ, æ, ɑ, ɔ, o, ʊ, u, ʌ, ə, ɝ
- Diphthongs: aɪ, aʊ, ɔɪ, eɪ, oʊ (represented as 152-dim sequences)

### Minimal Pairs

**Precomputed:** 31,399 minimal pair relationships

**Definition:** Words that differ by exactly one phoneme **at the same position**

**Key constraint:** Phoneme contrast occurs within the same position (e.g., both at position 0, or both at position 2). Words like cat /kæt/ and act /ækt/ are NOT minimal pairs (different positions and structures).

**Examples:**
- cat /kæt/ → bat /bæt/ (position 0: /k/ → /b/)
- cat /kæt/ → cot /kɑt/ (position 1: /æ/ → /ɑ/)
- cat /kæt/ → cap /kæp/ (position 2: /t/ → /p/)

**When computed:** During `scripts/export_clientside_data.py` (after Phase 3 embeddings)
- **Regeneration:** Automatically recomputed from scratch each time export script runs
- **Manual step:** You must run the export script manually after Phase 3

**Algorithm:**
```
Group words by phoneme length
For each length:
    For each pair of words (i, j) where i < j:
        Count phoneme differences
        If exactly 1 difference:
            Store as minimal pair
```

**Complexity:** O(n² × L) where n = vocabulary size, L = phoneme length
- Computation time: ~5-10 minutes
- Output: 2.4 MB uncompressed, 210 KB gzipped

**Why precomputed:** Word-level comparisons are expensive; results are static

**Usage:** Loaded on-demand in browser for minimal pairs intervention tool

**Note:** If you modify phoneme data or fix phoneme issues, re-run the full pipeline (Phase 1-2-3) and then the export script to regenerate minimal pairs.

### Maximal Opposition & Multiple Opposition

**Computed:** On-the-fly in browser (not precomputed)

**Maximal Opposition Algorithm:**
```
For each pair of unknown phonemes (p1, p2):
    score = count_feature_differences(p1, p2)
    if has_major_class_difference(p1, p2):
        score += 100  // Major class bonus
    Store pair with score
```

**Complexity:** O(p²) where p = number of unknown phonemes (typically < 10)
- Computation time: ~20-50 ms in browser
- Major class check: consonantal:+ AND (sonorant:+ XOR sonorant:-)

**Multiple Opposition Algorithm:**
```
// Index words by position-phoneme
For each word length:
    For each position:
        index[position][phoneme] = words_with_phoneme_at_position

// Find minimal sets (triplets, quadruplets, quintuplets)
For each position:
    For each combination of phonemes:
        If all phonemes have words at this position:
            Create minimal set
```

**Complexity:** O(n + k) where n = vocabulary size (indexing), k = result sets
- Computation time: ~30-80 ms in browser

**Why on-the-fly:**
- Small phoneme inventory (35 total) makes computations fast
- User-specific parameters (which phonemes are unknown/problematic)
- Flexible position filtering (initial/medial/final)

### Psycholinguistic Norms

**Total properties:** 12 (8 psycholinguistic + 4 phonological complexity)

**Coverage by property:**

| Property | Coverage | Source | Range |
|----------|----------|--------|-------|
| **Lexical Properties** |
| Frequency | ~99% | SUBTLEX-US | 0-1000+ per million |
| Age of Acquisition | ~75% | Glasgow Norms | 1-7 (1=earliest, 7=latest) |
| **Semantic Properties** |
| Imageability | ~40% | Glasgow Norms | 1-7 (ease of mental imagery) |
| Familiarity | ~40% | Glasgow Norms | 1-7 (word familiarity) |
| Concreteness | ~60% | Brysbaert et al. | 1-5 (concrete vs. abstract) |
| **Affective Properties** |
| Valence | ~50% | Warriner et al. | 1-9 (negative to positive) |
| Arousal | ~50% | Warriner et al. | 1-9 (calm to excited) |
| Dominance | ~50% | Warriner et al. | 1-9 (weak to powerful) |
| **Phonological Complexity** |
| Syllables | 100% | CMU Dict | 1-5 |
| Phonemes | 100% | CMU Dict | 1-10+ |
| WCM | ~95% | Stoel-Gammon 2010 | 0-15 |
| MSH | ~95% | Computed | 1-6 |

**Note:** Not all words have all properties. Filters only apply to words with available data.

## Performance Characteristics

### Data Loading

- **Initial load:** ~1-2 seconds (loads gzipped files)
- **Subsequent queries:** Instant (data cached in memory)
- **Memory usage:** ~60 MB in browser
- **Compression ratio:** 99% (56.7 MB → 0.6 MB gzipped)

### Query Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Pattern search | 10-50 ms | Full vocabulary scan with regex |
| Property filter | 5-20 ms | In-memory array filtering |
| Similarity search | 50-100 ms | Soft Levenshtein on full vocab |
| Minimal pairs | 1-5 ms | Precomputed lookup |
| Maximal opposition | 20-50 ms | Feature comparison matrix (n² pairs) |
| Multiple opposition | 30-80 ms | Set generation algorithm |

### Export Performance

| Format | Time | Size |
|--------|------|------|
| CSV | ~50 ms | ~1 KB per word |
| JSON | ~30 ms | ~2 KB per word |
| Copy to clipboard | ~10 ms | Direct text |

## Phonological Complexity Measures

### Word Complexity Measure (WCM)

**Source:** Stoel-Gammon (2010)

**Range:** 0-15

**8 Parameters:**
1. More than 2 syllables: +1
2. Non-initial stress: +1
3. Word-final consonant: +1
4. Consonant cluster: +1 per cluster (onset or coda)
5. Velar consonant (k, g, ŋ): +1 per occurrence
6. Liquid/rhotic (l, ɹ, r, ɚ, ɝ): +1 per occurrence
7. Fricative/affricate (f, v, θ, ð, s, z, ʃ, ʒ, h, tʃ, dʒ): +1 per occurrence
8. Voiced fricative/affricate: +1 additional per occurrence

**Examples:**
- "cat" (/kæt/): WCM = 2 (velar /k/, final consonant)
- "spray" (/spreɪ/): WCM = 5 (3-consonant cluster, fricative /s/, liquid /ɹ/)
- "strength" (/strɛŋkθ/): WCM = 11

### Mean Syllable Height (MSH)

**Source:** Motor Speech Hierarchy stages (Namasivayam et al., 2021)

**Range:** 1-6 (average of all syllable stages)

**Syllable Stages:**
- Stage I-II: Vowels only, /h/
- Stage III: Bilabials (p, b, m), nasals (n, ŋ)
- Stage IV: Stops/glides (t, d, k, g, w, j)
- Stage V: Fricatives (f, v, s, z, θ, ð, ʃ, ʒ)
- Stage VI: Liquids/affricates (l, ɹ, ʧ, ʤ)

**Calculation:**
1. Determine stage for each syllable (based on most complex consonant)
2. Average across all syllables
3. Round to 1 decimal place

**Examples:**
- "cat" (/kæt/): Stage IV (stop /k/, stop /t/) → MSH = 4.0
- "fish" (/fɪʃ/): Stage V (fricatives /f/, /ʃ/) → MSH = 5.0
- "splash" (/splæʃ/): Stage VI (liquid /l/ in cluster) → MSH = 6.0

## Technical Limitations

### 1. Dialect Coverage

**Current:** General American English only (CMU Dictionary primary pronunciations)

**Not supported:**
- British English (e.g., /ɑː/ vs /æ/ in "bath")
- Regional dialects (Southern, Boston, etc.)
- Pronunciation variants (CMU dict entries with (1), (2), etc. are excluded)

### 2. Psycholinguistic Norm Coverage

Not all words have all psycholinguistic properties. Coverage varies:
- **High coverage (>90%):** Frequency, syllables, phonemes, WCM, MSH
- **Medium coverage (50-75%):** AoA, concreteness, valence, arousal, dominance
- **Lower coverage (40-50%):** Imageability, familiarity

When filtering by properties, only words with available data are considered.

### 3. Syllabification

Uses rule-based English phonotactic constraints. May not optimally handle:
- Loanwords with non-English phonotactics (e.g., "tsunami" /tsuːnɑːmi/)
- Proper nouns with unusual structures
- Ambisyllabic consonants (treated as coda of first syllable)
- Syllabic consonants beyond typical patterns

### 4. Browser Compatibility

Requires modern browser with:
- ES6+ JavaScript support (arrow functions, async/await, etc.)
- Gzip decompression (automatic in all modern browsers)
- ~100 MB available memory
- Local storage for caching (optional but recommended)

**Tested on:**
- Chrome 90+ ✓
- Firefox 88+ ✓
- Safari 14+ ✓
- Edge 90+ ✓

### 5. Computational Complexity

**Similarity search:** O(n) where n = vocabulary size (48,720)
- Each comparison requires soft Levenshtein on syllable sequences
- Threshold filtering helps reduce result set

**Maximal opposition:** O(p²) where p = number of unknown phonemes
- All pairwise phoneme comparisons for feature differences
- Typically p < 10, so < 100 comparisons

**Multiple opposition:** O(w) where w = words matching phoneme patterns
- Linear scan to find minimal sets
- May need multiple passes to find complete sets

## Data Export Format

### Client-Side JSON Files

All data files are available in `webapp/frontend/public/data/`:

**1. word_metadata.json (12 MB uncompressed, 1.1 MB gzipped)**
```json
{
  "words": [
    {
      "word": "cat",
      "ipa": "kæt",
      "syllable_count": 1,
      "phoneme_count": 3,
      "wcm": 2,
      "msh": 4.0,
      "frequency": 182.5,
      "aoa": 2.1,
      "imageability": 6.8,
      "familiarity": 6.9,
      "concreteness": 4.93,
      "valence": 7.2,
      "arousal": 3.8,
      "dominance": 5.2
    }
    // ... 24,743 more words
  ]
}
```

**2. embeddings.json.gz (2.1 MB gzipped)**
```json
{
  "cat": {
    "syllables": [
      {
        "onset": [[0.12, -0.45, ..., 0.67]],  // [k] (76-dim)
        "nucleus": [[0.33, 0.21, ..., -0.12]], // [æ] (76-dim)
        "coda": [[-0.56, 0.78, ..., 0.34]]     // [t] (76-dim)
      }
    ]
  }
  // ... 24,743 more words
}
```

**3. minimal_pairs.json.gz (210 KB gzipped)**
```json
{
  "k": {
    "b": [
      {"word1": "cat", "word2": "bat", "position": "initial"},
      {"word1": "cap", "word2": "bap", "position": "initial"}
      // ... more pairs
    ]
    // ... other phoneme substitutions
  }
  // ... other source phonemes
}
```

**4. phoneme_features.json.gz (1 KB gzipped)**
```json
{
  "k": {
    "ipa": "k",
    "type": "consonant",
    "features": {
      "syllabic": "-",
      "consonantal": "+",
      "sonorant": "-",
      "voice": "-",
      "dorsal": "+",
      // ... 33 more features
    }
  }
  // ... 34 more phonemes
}
```

**5. syllable_structures.json.gz (576 KB gzipped)**
```json
{
  "cat": {
    "syllables": [
      {
        "onset": ["k"],
        "nucleus": "æ",
        "coda": ["t"]
      }
    ]
  }
  // ... 24,743 more words
}
```

**6. manifest.json.gz**
```json
{
  "version": "2.0.0",
  "created": "1762511838.527301",
  "vocabulary_size": 48720,
  "minimal_pairs_count": 112964,
  "phoneme_count": 35,
  "filter_criterion": "frequency only (relaxed in v2.3)",
  "files": {
    "word_metadata.json": "Word properties, IPA, syllables, psycholinguistic norms",
    "embeddings.json": "Phoible-based syllable embeddings",
    "minimal_pairs.json": "Precomputed minimal pair relationships",
    "phoneme_features.json": "Phoneme inventory with Phoible features"
  }
}
```

## References

**Phonological Features:**
- Moran, S., & McCloy, D. (2019). PHOIBLE 2.0. Max Planck Institute for the Science of Human History. https://phoible.org/
- Hayes, B. (2009). *Introductory Phonology*. Wiley-Blackwell.

**Phonological Complexity:**
- Stoel-Gammon, C. (2010). The Word Complexity Measure: Description and application to developmental phonology and disorders. *Clinical Linguistics & Phonetics*, 24(4-5), 271-282.
- Namasivayam, A. K., et al. (2021). Milestones of speech production in children. *Journal of Speech, Language, and Hearing Research*.

**Psycholinguistic Norms:**
- Brysbaert, M., & New, B. (2009). Moving beyond Kučera and Francis: A critical evaluation of current word frequency norms. *Behavior Research Methods*, 41(4), 977-990.
- Brysbaert, M., Warriner, A. B., & Kuperman, V. (2014). Concreteness ratings for 40 thousand generally known English word lemmas. *Behavior Research Methods*, 46, 904-911.
- Scott, G. G., et al. (2019). The Glasgow Norms: Ratings of 5,500 words on nine scales. *Behavior Research Methods*, 51, 1258-1270.
- Warriner, A. B., Kuperman, V., & Brysbaert, M. (2013). Norms of valence, arousal, and dominance for 13,915 English lemmas. *Behavior Research Methods*, 45, 1191-1207.

**Clinical Interventions:**
- Gierut, J. A. (1989). Maximal opposition approach to phonological treatment. *Journal of Speech and Hearing Disorders*, 54(1), 9-19.
- Gierut, J. A. (1992). The conditions and course of clinically induced phonological change. *Journal of Speech and Hearing Research*, 35(5), 1049-1063.
- Storkel, H. L. (2022). Minimal, Maximal, or Multiple Oppositions: A review of phonological intervention approaches. *Language, Speech, and Hearing Services in Schools*, 53(2), 421-437.

**Similarity Metrics:**
- Levenshtein, V. I. (1966). Binary codes capable of correcting deletions, insertions, and reversals. *Soviet Physics Doklady*, 10(8), 707-710.
