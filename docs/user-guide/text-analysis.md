# Text Analysis

Analyze passages for readability across phonological, lexical, semantic, and affective dimensions with interactive color-coded highlighting.

## Overview

The Text Analysis tool provides comprehensive psycholinguistic profiling of text passages, computing aggregate percentile statistics across 14 properties and enabling interactive word-level highlighting.

**Key Features**:
- Aggregate percentile statistics for entire passages
- Interactive highlighting by property with color gradients
- Three preset phonetics passages (Grandfather, Rainbow, Caterpillar)
- Coverage tracking (percentage of words in vocabulary)
- Unknown word identification with dotted underlines

**Use Cases**:
- **Clinical**: Assess therapy script complexity, select developmentally appropriate reading materials
- **Education**: Evaluate text readability for students at different levels
- **Research**: Analyze stimulus materials for controlled psycholinguistic properties
- **SLP**: Track phonological, lexical, and semantic complexity of treatment materials

## Basic Usage

1. **Enter or paste text** into the input field (or load a preset passage)
2. **Click "Analyze Text"** to compute aggregate statistics
3. **Select a feature** from the dropdown to highlight words by percentile
4. **Review aggregate percentiles** to understand overall passage characteristics
5. **Hover over highlighted words** to see individual percentile ranks

## Analyzed Properties (14 Total)

### Phonological Complexity (3 properties)

| Property | Interpretation | Color Scheme |
|----------|----------------|--------------|
| **Syllables** | Lower = simpler | Orange gradient (light → dark) |
| **Phonemes** | Lower = simpler | Orange gradient (light → dark) |
| **WCM** (Word Complexity Measure) | Lower = simpler | Orange gradient (light → dark) |

### Phonotactic Probability (3 properties)

| Property | Interpretation | Color Scheme |
|----------|----------------|--------------|
| **Phono Prob (Avg)** | Higher = more typical | Red (rare) → Green (common) |
| **Phono Prob (Sum Log)** | Higher = more typical | Red (rare) → Green (common) |
| **Positional Prob** | Higher = more typical | Red (rare) → Green (common) |

### Lexical Properties (2 properties)

| Property | Interpretation | Color Scheme |
|----------|----------------|--------------|
| **Frequency** | Higher = more common | Red (rare) → Green (common) |
| **Age of Acquisition** | Lower = learned earlier | Orange gradient (early → late) |

### Semantic Properties (3 properties)

| Property | Interpretation | Color Scheme |
|----------|----------------|--------------|
| **Imageability** | Higher = easier to visualize | Blue gradient (low → high) |
| **Familiarity** | Higher = more familiar | Blue gradient (low → high) |
| **Concreteness** | Higher = more concrete | Blue gradient (low → high) |

### Affective Properties (3 properties)

| Property | Interpretation | Color Scheme |
|----------|----------------|--------------|
| **Valence** | Higher = more positive | Red (negative) → Green (positive) |
| **Arousal** | Higher = more exciting | Purple gradient (calm → excited) |
| **Dominance** | Higher = more powerful | Purple gradient (weak → powerful) |

## Preset Passages

### Grandfather Passage
**Type**: Standard phonetics passage
**Purpose**: Adult speech assessment, voice/resonance evaluation
**Length**: 113 words
**Source**: Standardized diagnostic passage for speech-language pathology

### Rainbow Passage
**Type**: Classic phonetics passage
**Purpose**: Voice quality assessment, speech rate analysis
**Length**: 85 words
**Source**: Fairbanks (1960) - widely used in clinical and research settings

### Caterpillar Passage
**Type**: Pediatric speech sample
**Purpose**: Child speech assessment, narrative language evaluation
**Length**: 181 words
**Source**: Standardized passage for pediatric speech assessment

## Understanding Aggregate Percentiles

Aggregate percentiles represent the **average percentile rank** of all known words in the passage:

- **0-25th percentile**: Very low on this property (e.g., very simple, very rare, very concrete)
- **25-50th percentile**: Below average
- **50-75th percentile**: Above average
- **75-100th percentile**: Very high on this property (e.g., very complex, very common, very abstract)

**Example Interpretation**:
```
Syllable Count: 35th percentile → Passage uses words with fewer syllables than average
Frequency: 72nd percentile → Passage uses more common words than average
Concreteness: 20th percentile → Passage uses more abstract words than average
```

## Interactive Highlighting

### Color Schemes

**Difficulty/Complexity** (Syllables, Phonemes, WCM, AoA):
- Light orange → Words at low percentiles (simple/early)
- Dark orange → Words at high percentiles (complex/late)

**Frequency** (Frequency, Phonotactic Probability):
- Red → Rare words (low percentiles)
- Yellow → Medium frequency
- Green → Common words (high percentiles)

**Semantic** (Imageability, Familiarity, Concreteness):
- Light blue → Low ratings (abstract, unfamiliar, low imageability)
- Dark blue → High ratings (concrete, familiar, high imageability)

**Diverging** (Valence):
- Red → Negative valence (low percentiles)
- Gray → Neutral
- Green → Positive valence (high percentiles)

**Intensity** (Arousal, Dominance):
- Light purple → Low intensity (calm, weak)
- Dark purple → High intensity (excited, powerful)

### Unknown Words

Words not in PhonoLex's vocabulary (48,720 words) are marked with:
- Dotted gray underline
- Reduced opacity (60%)
- Hover tooltip: "Unknown word (not in vocabulary)"

Words with no data for the selected property are marked similarly:
- Dotted gray underline
- Hover tooltip: "[word]: No data for this property"

## Clinical Applications

### Therapy Script Assessment

**Goal**: Ensure therapy materials are developmentally appropriate

**Steps**:
1. Paste therapy script into Text Analysis
2. Review **Syllable Count**, **WCM**, and **AoA** percentiles
3. Target 25-50th percentiles for early intervention (simple words)
4. Target 50-75th percentiles for intermediate stages
5. Adjust script by replacing high-percentile words with simpler alternatives

**Example**:
```
Original: "The enormous elephant enthusiastically embraced exercise."
Analysis: Syllables (85th), WCM (92nd), AoA (88th) → Too complex for early intervention

Revised: "The big elephant liked to run and play."
Analysis: Syllables (35th), WCM (28th), AoA (15th) → Appropriate
```

### Reading Material Selection

**Goal**: Match text complexity to student reading level

**Steps**:
1. Analyze candidate passages
2. Compare **Frequency**, **Imageability**, and **Concreteness** percentiles
3. Select texts with properties matching target level:
   - Early readers: High imageability (60-80th), high concreteness (60-80th)
   - Intermediate: Medium imageability (40-60th), mixed concreteness
   - Advanced: Lower imageability (20-40th), abstract concepts (20-40th)

### Stimulus Control for Research

**Goal**: Match experimental stimuli on psycholinguistic properties

**Steps**:
1. Analyze Condition A passage
2. Analyze Condition B passage
3. Compare aggregate percentiles across all 14 properties
4. Adjust passages to minimize differences in confounding variables
5. Re-analyze to confirm matching

## Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Text analysis | 50-100 ms | Depends on passage length |
| Percentile lookup | <1 ms | Pre-computed during export |
| Highlighting render | 5-20 ms | Depends on passage length |

**Scalability**: Tested on passages up to 1,000 words without performance issues.

## Data Coverage

**Vocabulary**: 48,720 English words from CMU Pronouncing Dictionary

**Property Coverage** (% of vocabulary with data):
- Phonological properties: 100% (all words)
- Phonotactic probability: ~51% (24,831 words)
- Frequency: 100% (all words - filtering criterion)
- AoA: ~9% (4,366 words - Glasgow Norms)
- Imageability: ~9% (4,366 words - Glasgow Norms)
- Familiarity: ~9% (4,366 words - Glasgow Norms)
- Concreteness: ~50% (24,366 words - Brysbaert et al.)
- VAD (Valence/Arousal/Dominance): ~27% (13,131 words - Warriner et al.)

**Coverage Impact**: Passages with technical/specialized vocabulary may have lower coverage. Unknown words and words without specific property data are clearly marked.

## Limitations

### Vocabulary Coverage

- **48,720 words** covers 99%+ of typical English text
- Proper nouns, neologisms, and specialized jargon may not be included
- Non-English words and code-switching not supported

### Property Availability

- Not all words have all 14 properties (see coverage above)
- Missing properties marked with dotted underline when selected
- Aggregate percentiles computed only from words with data

### Interpretation Caveats

- Percentiles are **corpus-relative**, not absolute measures
- High-frequency words may be simple OR common depending on context
- Semantic properties (imageability, concreteness) have lower coverage than phonological properties

### Linguistic Limitations

- General American English pronunciation only (CMU Dictionary)
- Single-word analysis (no multi-word expressions or idioms)
- No syntactic or discourse-level analysis

## References

**Data Sources**:
- Brysbaert, M., & New, B. (2009). SUBTLEX-US word frequencies. *Behavior Research Methods*, 41(4), 977-990.
- Brysbaert, M., Warriner, A. B., & Kuperman, V. (2014). Concreteness ratings for 40 thousand English words. *Behavior Research Methods*, 46(3), 904-911.
- Scott, G. G., et al. (2019). The Glasgow Norms. *Behavior Research Methods*, 51(3), 1258-1270.
- Vitevitch, M. S., & Luce, P. A. (2004). Phonotactic probability for words and nonwords in English. *Behavior Research Methods*, 36(3), 481-487.
- Warriner, A. B., Kuperman, V., & Brysbaert, M. (2013). Norms of valence, arousal, and dominance. *Behavior Research Methods*, 45(4), 1191-1207.

**Standard Passages**:
- Fairbanks, G. (1960). *Voice and articulation drillbook* (2nd ed.). New York: Harper & Row. (Rainbow Passage)
- Darley, F. L., Aronson, A. E., & Brown, J. R. (1975). *Motor speech disorders*. Philadelphia: Saunders. (Grandfather Passage)

## See Also

- [Custom Word Lists](custom-word-lists.md) - Build targeted word lists with property filtering
- [Lookup](lookup.md) - View detailed psycholinguistic properties for individual words
- [Psycholinguistic Norms Reference](../reference/psycholinguistic-norms.md) - Complete property documentation
