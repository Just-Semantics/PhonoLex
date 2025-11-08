# Custom Word Lists

The Custom Word Lists tool is the most powerful feature in PhonoLex, allowing you to build targeted word lists using multiple criteria across phonological, lexical, semantic, and affective domains.

## Overview

Build word lists by combining:
- **Phoneme patterns** (STARTS_WITH, ENDS_WITH, CONTAINS, CONTAINS_MEDIAL)
- **Property filters** (15 psycholinguistic and phonological properties)
- **Phoneme exclusions** (exclude words containing specific phonemes)
- **AND logic** (words must match ALL criteria)

**Vocabulary size:** 48,720 English words from the CMU Pronouncing Dictionary

## Basic Usage

1. **Add a pattern**: Click "Add Pattern" and select a pattern type
2. **Choose a phoneme**: Use the IPA keyboard or type directly
3. **Add filters** (optional): Set ranges for frequency, imageability, etc.
4. **Generate**: Click "Generate List" to see results
5. **Export**: Download as CSV or copy individual words

## Pattern Types

### Pattern Matching Algorithm

Patterns use IPA transcriptions to match phoneme sequences:

**STARTS_WITH /k/**
```
Matches: cat /kæt/, king /kɪŋ/, crest /kɹɛst/
Does not match: back /bæk/, attack /ətæk/
```

**ENDS_WITH /t/**
```
Matches: cat /kæt/, fight /faɪt/, rest /ɹɛst/
Does not match: cats /kæts/ (ends with /s/)
```

**CONTAINS /s/**
```
Matches: sit /sɪt/, pass /pæs/, outside /aʊtsaɪd/
Matches any position: initial, medial, or final
```

**CONTAINS_MEDIAL /s/**
```
Matches: missile /mɪsəl/ (medial /s/)
Does not match: sit /sɪt/ (initial), pass /pæs/ (final)
```

### Technical Details

**Implementation:**
- Uses regular expression matching on IPA strings
- Case-sensitive IPA matching (e.g., /i/ ≠ /ɪ/)
- Matches exact phoneme boundaries (e.g., /s/ won't match /ʃ/)
- Diphthongs treated as single units (e.g., /aɪ/ is one phoneme)

**Performance:** ~10-50ms for pattern search across full vocabulary

**Limitations:**
- Cannot match phoneme features directly (use Lookup tool for feature-based search)
- Cannot use wildcards or phonological classes (e.g., cannot search "any fricative")

## Property Filters

### Complete Property Reference

#### Phonological Complexity (4 properties)

| Property | Range | Source | Description | Coverage |
|----------|-------|--------|-------------|----------|
| **Syllables** | 1-5 | CMU Dictionary | Number of syllables | 100% |
| **Phonemes** | 1-10+ | CMU Dictionary | Number of phonemes (IPA segments) | 100% |
| **WCM** | 0-15 | Stoel-Gammon (2010) | Word Complexity Measure (8 parameters) | ~95% |
| **MSH** | 1-6 | Motor Speech Hierarchy | Mean Syllable Height (motor complexity) | ~95% |

**Syllables:**
- Counted from syllabification algorithm
- Example: "cat" = 1, "window" = 2, "computer" = 3

**Phonemes:**
- Counted from IPA transcription
- Diphthongs count as 1 phoneme (e.g., /aɪ/ in "time")
- Example: "cat" /kæt/ = 3, "spray" /spreɪ/ = 4

**WCM (Word Complexity Measure):**

8 parameters from Stoel-Gammon (2010):
1. More than 2 syllables: +1
2. Non-initial stress: +1
3. Word-final consonant: +1
4. Consonant cluster: +1 per cluster
5. Velar (k, g, ŋ): +1 per occurrence
6. Liquid/rhotic (l, ɹ): +1 per occurrence
7. Fricative/affricate (f, v, θ, ð, s, z, ʃ, ʒ, h, tʃ, dʒ): +1 per occurrence
8. Voiced fricative/affricate: +1 additional

Examples:
- "cat" /kæt/ = 2 (velar /k/, final consonant)
- "spray" /spreɪ/ = 5 (cluster, fricative /s/, liquid /ɹ/)
- "strength" /strɛŋkθ/ = 11 (very high complexity)

**MSH (Mean Syllable Height):**

Stages from Motor Speech Hierarchy (Namasivayam et al., 2021):
- Stage I-II: Vowels, /h/
- Stage III: Bilabials (p, b, m), nasals (n, ŋ)
- Stage IV: Stops/glides (t, d, k, g, w, j)
- Stage V: Fricatives (f, v, s, z, θ, ð, ʃ, ʒ)
- Stage VI: Liquids/affricates (l, ɹ, ʧ, ʤ)

Calculation: Average stage across all syllables

Examples:
- "cat" = 4.0 (stops only)
- "fish" = 5.0 (fricatives)
- "splash" = 6.0 (liquid in cluster)

#### Phonotactic Probability (3 properties)

| Property | Range | Source | Description | Coverage |
|----------|-------|--------|-------------|----------|
| **Biphone Probability** | 0-1 | Vitevitch & Luce (2004) | Mean probability of phoneme sequences (higher = more typical) | ~100% |
| **Sum Log Probability** | -10 to 0 | Vitevitch & Luce (2004) | Sum of log biphone probabilities (less negative = more typical) | ~100% |
| **Positional Probability** | 0-1 | Vitevitch & Luce (2004) | Mean probability of phonemes in their syllable positions | ~100% |

**Biphone Probability:**
- Measures how typical the sound sequences are in English
- Computed on full CMU Pronouncing Dictionary (117K words) for unbiased estimates
- Higher values = more phonotactically "legal" or common sequences

Interpretation:
- 0.00-0.02: Very low (unusual sequences like "strengths")
- 0.02-0.05: Low-moderate (e.g., "splash", "squid")
- 0.05-0.10: Moderate-high (e.g., "cat", "dog", "jump")
- 0.10+: Very high (very typical sequences like "mama", "see")

**Sum Log Probability:**
- Standard metric from Vitevitch & Luce (2004)
- Negative values (more negative = less typical sequences)
- Useful for replicating published research studies

**Positional Probability:**
- Measures individual phoneme frequencies in onset/nucleus/coda positions
- Independent of sequence probability (biphone)
- Higher values = phonemes that occur frequently in their positions

**Clinical/Research use:**
- High phonotactic probability correlates with faster word learning
- Children acquire high-probability patterns before low-probability patterns
- Useful for controlling word learning difficulty in intervention or research

#### Lexical Properties (2 properties)

| Property | Range | Source | Description | Coverage |
|----------|-------|--------|-------------|----------|
| **Frequency** | 0-1000+ | SUBTLEX-US (Brysbaert & New, 2009) | Occurrences per million words in film subtitles | ~99% |
| **Age of Acquisition (AoA)** | 1-7 | Glasgow Norms (Scott et al., 2019) | Age when word is typically learned (1=earliest, 7=latest) | ~75% |

**Frequency:**
- Based on 51 million words from film and television subtitles
- More representative of spoken language than written corpora
- Log-transformed for UI (actual values are log10 per million)

Interpretation:
- 0-5: Very rare words
- 5-20: Uncommon words
- 20-100: Common words
- 100+: Very high frequency words

**Age of Acquisition:**
- Based on adult ratings of when they learned each word
- Scale: 1 (very early, <3 years) to 7 (late, adult years)
- Correlates with processing speed and naming accuracy

Interpretation:
- 1-2: Early childhood words (mommy, cat, eat)
- 3-4: Elementary school words (book, teacher, happy)
- 5-6: Middle/high school words (concept, analyze, determine)
- 7: Late acquisition words (arcane, ephemeral, juxtapose)

#### Semantic Properties (3 properties)

| Property | Range | Source | Description | Coverage |
|----------|-------|--------|-------------|----------|
| **Imageability** | 1-7 | Glasgow Norms | Ease of mental imagery (1=hard to imagine, 7=easy) | ~40% |
| **Familiarity** | 1-7 | Glasgow Norms | Word familiarity (1=unfamiliar, 7=very familiar) | ~40% |
| **Concreteness** | 1-5 | Brysbaert et al. (2014) | Concrete vs. abstract (1=abstract, 5=concrete) | ~60% |

**Imageability:**
- Measures how easily a word evokes a mental image
- High imageability: cat, tree, house (tangible objects)
- Low imageability: truth, concept, democracy (abstract ideas)

**Familiarity:**
- Self-reported familiarity ratings from adults
- Distinct from frequency (can be familiar but rarely used)
- Example: "elephant" = high familiarity, moderate frequency

**Concreteness:**
- Measures how concrete (physical) vs. abstract a concept is
- Based on ratings from 40,000 English words
- High concreteness: table, water, run
- Low concreteness: truth, love, think

#### Affective Properties (3 properties)

| Property | Range | Source | Description | Coverage |
|----------|-------|--------|-------------|----------|
| **Valence** | 1-9 | Warriner et al. (2013) | Emotional valence (1=very negative, 9=very positive) | ~50% |
| **Arousal** | 1-9 | Warriner et al. (2013) | Emotional arousal (1=calm, 9=excited/intense) | ~50% |
| **Dominance** | 1-9 | Warriner et al. (2013) | Sense of control (1=weak/submissive, 9=powerful/in-control) | ~50% |

**Valence:**
- Emotional positivity/negativity
- Negative (1-3): war, death, hate, fear
- Neutral (4-6): table, walk, window
- Positive (7-9): love, happy, success, joy

**Arousal:**
- Emotional intensity/activation
- Low arousal (1-3): calm, sleep, relax, quiet
- Medium arousal (4-6): walk, think, read
- High arousal (7-9): excited, angry, panic, thrill

**Dominance:**
- Sense of power or control
- Low dominance (1-3): helpless, weak, afraid, victim
- Medium dominance (4-6): walk, see, think
- High dominance (7-9): powerful, boss, control, leader

### Filter Logic

**AND Logic:** All filters must be satisfied

Example query:
```
Pattern: STARTS_WITH /s/
Filter: Frequency ≥ 20
Filter: Syllables = 1
Filter: Concreteness ≥ 4.0

Result: Words that start with /s/ AND are high-frequency
        AND are monosyllabic AND are concrete

Matches: sun, sea, sock, snow
Does not match: sad (concreteness too low),
                 seven (two syllables),
                 see (frequency too low)
```

**Missing Data:** Words without a property are excluded when filtering by that property

Example:
```
Filter: Imageability ≥ 5.0

Only words with imageability data are considered.
Words without imageability ratings are excluded from results.
```

## Phoneme Exclusions

**Purpose:** Exclude words containing specific phonemes

**Use cases:**
- Avoiding error sounds (e.g., exclude /s/ when child substitutes s→θ)
- Creating phoneme-specific lists (e.g., /k/ words without /g/)
- Controlling phonological context

**Example:**
```
Pattern: STARTS_WITH /k/
Exclude: /g/
Exclude: /s/

Result: /k/ words without /g/ or /s/
Matches: cat, car, cut, candy
Does not match: cat+s (has /s/), big (has /g/)
```

**Technical details:**
- Exclusions apply to entire IPA transcription
- Multiple exclusions create additional AND conditions
- Case-sensitive (e.g., excluding /i/ won't exclude /ɪ/)

## Export Formats

### CSV Export

**Format:**
```csv
word,ipa,syllables,phonemes,wcm,msh,phono_prob_avg,phono_prob_sum_log,positional_prob_avg,frequency,aoa,imageability,familiarity,concreteness,valence,arousal,dominance
cat,kæt,1,3,2,4.0,0.062,-2.46,0.081,182.5,2.1,6.8,6.9,4.93,7.2,3.8,5.2
dog,dɔg,1,3,1,4.0,0.054,-2.89,0.073,245.3,1.8,6.9,7.0,5.0,7.5,4.2,5.5
```

**Details:**
- Header row with property names
- One word per row
- Empty cells for missing properties
- IPA in standard Unicode characters
- Numbers use decimal notation

**File size:** ~1 KB per word

### Copy Individual Words

**Format:** Plain text, one word per line
```
cat
dog
house
```

**Use case:** Quick copying for clinical materials

## Advanced Query Examples

See [Practical Examples](../getting-started/examples.md) for detailed walkthroughs, including:

- Example 1: Simple CVC words for early intervention
- Example 2: Initial /s/ words with semantic scaffolding
- Example 3: Late-developing sounds in simple contexts
- Example 4: Negative valence words for emotional language
- Example 5: Excluding problematic phonemes

## Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| Pattern matching | 10-50 ms | Full vocabulary regex scan |
| Property filtering | 5-20 ms | In-memory array filter |
| Combined query | 15-70 ms | Pattern + multiple filters |
| Export to CSV | ~50 ms | Serialization + download |
| Copy to clipboard | ~10 ms | Direct text copy |

**Factors affecting speed:**
- Number of filters (more filters = slightly slower)
- Result set size (larger results = slower export)
- Browser performance (Chrome/Edge slightly faster)

## Data Coverage & Limitations

### Coverage by Property

| Category | Properties | Average Coverage |
|----------|-----------|------------------|
| Phonological | Syllables, Phonemes, WCM, MSH | 98% |
| Phonotactic | Biphone Prob, Sum Log Prob, Positional Prob | 100% |
| Lexical | Frequency, AoA | 87% |
| Semantic | Imageability, Familiarity, Concreteness | 47% |
| Affective | Valence, Arousal, Dominance | 50% |

**Important:** Filtering by properties with lower coverage (e.g., imageability) will reduce result set size significantly.

### Vocabulary Limitations

**Included:**
- Primary pronunciations only (no variants)
- General American English dialect
- Common words with established psycholinguistic norms

**Excluded:**
- Pronunciation variants (CMU entries with (1), (2), etc.)
- Proper nouns
- Words without any psycholinguistic properties
- Non-English loanwords without standard pronunciations

**Total vocabulary:** 48,720 words

### Technical Limitations

**Cannot filter by:**
- Phoneme features directly (e.g., "all fricatives") - use pattern matching or Lookup tool
- Orthographic properties (spelling patterns)
- Grammatical category (noun, verb, etc.)
- Morphological complexity (prefixes, suffixes)

**Pattern matching:**
- Exact phoneme matching only (no fuzzy matching)
- Cannot match phonological classes (e.g., "any stop")
- Cannot use regular expressions directly

## Tips & Best Practices

### Getting Started
- **Start simple:** Begin with one pattern, then add filters incrementally
- **Check the count:** Preview shows how many words match before generating
- **Iterate:** Adjust filters to get desired word count (aim for 20-50 words for clinical use)

### Optimization
- **Use frequency filters:** Ensures functional, commonly-used words
- **Combine complexity measures:** Use WCM + MSH for precise developmental targeting
- **Consider coverage:** Properties with lower coverage (imageability, familiarity) will reduce results more

### Clinical Applications
- **Early intervention:** Low WCM + high frequency + high imageability
- **Phoneme-specific practice:** Pattern matching + exclusions + frequency
- **Semantic therapy:** High imageability + concreteness + valence/arousal filters
- **Literacy support:** Monosyllabic + high frequency + moderate AoA

### Research Applications
- **Stimulus control:** Match WCM, frequency, AoA across conditions
- **Semantic variables:** Control concreteness, imageability, familiarity
- **Affective content:** Select words by valence, arousal, dominance
- **Phonological complexity:** Systematic manipulation of WCM, MSH, syllable/phoneme counts

## References

**Phonological Complexity:**
- Stoel-Gammon, C. (2010). The Word Complexity Measure: Description and application to developmental phonology and disorders. *Clinical Linguistics & Phonetics*, 24(4-5), 271-282.
- Namasivayam, A. K., et al. (2021). Milestones of speech production in children. *Journal of Speech, Language, and Hearing Research*.

**Phonotactic Probability:**
- Vitevitch, M. S., & Luce, P. A. (2004). A Web-based interface to calculate phonotactic probability for words and nonwords in English. *Behavior Research Methods, Instruments, & Computers*, 36(3), 481-487.

**Psycholinguistic Norms:**
- Brysbaert, M., & New, B. (2009). Moving beyond Kučera and Francis. *Behavior Research Methods*, 41(4), 977-990.
- Brysbaert, M., et al. (2014). Concreteness ratings for 40 thousand English words. *Behavior Research Methods*, 46, 904-911.
- Scott, G. G., et al. (2019). The Glasgow Norms: Ratings of 5,500 words. *Behavior Research Methods*, 51, 1258-1270.
- Warriner, A. B., et al. (2013). Norms of valence, arousal, and dominance. *Behavior Research Methods*, 45, 1191-1207.

## See Also

- [Practical Examples](../getting-started/examples.md) - Hands-on examples with expected results
- [Technical Architecture](../technical/architecture.md) - Pattern matching implementation details
- [Psycholinguistic Norms Reference](../reference/psycholinguistic-norms.md) - Complete property documentation
