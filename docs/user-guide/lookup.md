# Lookup

Look up words and phonemes, compare phoneme features, or search by distinctive features.

## Overview

The Lookup tool provides four modes:
1. **Word Lookup** - Complete phonological and psycholinguistic profile for any word
2. **Phoneme Lookup** - View all 38 PHOIBLE distinctive features for any phoneme
3. **Phoneme Comparison** - Feature-by-feature comparison of two phonemes with similarity score
4. **Search by Features** - Find all phonemes matching specific feature combinations

## Word Lookup

Search for any word (48,720 words available) to view comprehensive information across 4 domains.

### Phonological Complexity (4 properties)

| Property | Source | Description | Coverage |
|----------|--------|-------------|----------|
| **IPA** | CMU Dictionary | International Phonetic Alphabet transcription | 100% |
| **Syllables** | Syllabification algorithm | Number of syllables (1-5) | 100% |
| **Phonemes** | CMU Dictionary | Number of phoneme segments (1-10+) | 100% |
| **WCM** | Stoel-Gammon (2010) | Word Complexity Measure (0-15) | ~95% |
| **MSH** | Motor Speech Hierarchy | Mean Syllable Height (1-6) | ~95% |

**Word Complexity Measure (WCM)**: 8-parameter measure of phonological complexity:
1. More than 2 syllables: +1
2. Non-initial stress: +1
3. Word-final consonant: +1
4. Consonant cluster: +1 per cluster
5. Velar (k, g, ŋ): +1 per occurrence
6. Liquid/rhotic (l, ɹ): +1 per occurrence
7. Fricative/affricate (f, v, θ, ð, s, z, ʃ, ʒ, h, tʃ, dʒ): +1 per occurrence
8. Voiced fricative/affricate: +1 additional

**Mean Syllable Height (MSH)**: Motor complexity based on developmental stages:
- Stage I-II (1-2): Vowels, /h/
- Stage III (3): Bilabials (p, b, m), nasals (n, ŋ)
- Stage IV (4): Stops/glides (t, d, k, g, w, j)
- Stage V (5): Fricatives (f, v, s, z, θ, ð, ʃ, ʒ)
- Stage VI (6): Liquids/affricates (l, ɹ, ʧ, ʤ)

### Psycholinguistic Properties (8 properties)

**Lexical (2 properties):**

| Property | Source | Range | Description | Coverage |
|----------|--------|-------|-------------|----------|
| **Frequency** | SUBTLEX-US (Brysbaert & New, 2009) | 0-1000+ | Occurrences per million words | ~99% |
| **Age of Acquisition** | Glasgow Norms (Scott et al., 2019) | 1-7 | Age when typically learned (1=earliest) | ~75% |

**Semantic (3 properties):**

| Property | Source | Range | Description | Coverage |
|----------|--------|-------|-------------|----------|
| **Imageability** | Glasgow Norms | 1-7 | Ease of mental imagery | ~40% |
| **Familiarity** | Glasgow Norms | 1-7 | Word familiarity | ~40% |
| **Concreteness** | Brysbaert et al. (2014) | 1-5 | Concrete vs. abstract | ~60% |

**Affective (3 properties):**

| Property | Source | Range | Description | Coverage |
|----------|--------|-------|-------------|----------|
| **Valence** | Warriner et al. (2013) | 1-9 | Negative to positive emotion | ~50% |
| **Arousal** | Warriner et al. (2013) | 1-9 | Calm to excited/intense | ~50% |
| **Dominance** | Warriner et al. (2013) | 1-9 | Weak to powerful/in-control | ~50% |

See [Psycholinguistic Norms Reference](../reference/psycholinguistic-norms.md) for complete documentation of all 8 properties.

### Example: Word Lookup for "strength"

```
Word: strength
IPA: /strɛŋkθ/

Phonological Complexity:
  Syllables: 1
  Phonemes: 7
  WCM: 11 (very high - 3-consonant cluster, velars, fricatives)
  MSH: 5.5 (high motor complexity - fricatives and velar nasal)

Lexical:
  Frequency: 28.5 (fairly common)
  Age of Acquisition: 5.8 (learned later)

Semantic:
  Imageability: 3.2 (abstract concept)
  Familiarity: 6.1 (familiar word)
  Concreteness: 2.5 (abstract)

Affective:
  Valence: 6.8 (positive)
  Arousal: 5.2 (moderately arousing)
  Dominance: 7.1 (high dominance/power)
```

**Interpretation**: "Strength" has very high phonological complexity (WCM=11) due to the initial 3-consonant cluster /str/ and multiple late-developing phonemes. Despite this complexity, it's a familiar word learned in middle childhood (AoA=5.8). The semantic profile shows it's an abstract concept (low imageability and concreteness) with positive emotional valence and strong associations with power (high dominance).

## Phoneme Lookup

View detailed features for any IPA phoneme using the **PHOIBLE distinctive feature system** (Hayes 2009; Moisik & Esling 2011).

### Feature System Overview

**38 ternary features** across 7 categories:
- **Major class features** (4): consonantal, syllabic, sonorant, approximant
- **Laryngeal features** (11): voice, spreadGlottis, constrictedGlottis, etc.
- **Manner features** (6): continuant, nasal, strident, lateral, delayedRelease, tap
- **Place features (articulator)** (3): labial, coronal, dorsal
- **Place features (tongue body)** (8): high, low, front, back, tense, etc.
- **Place features (detailed)** (5): anterior, distributed, etc.
- **Vowel-specific** (1): retractedTongueRoot

**Feature values:**
- `+` : Feature is present
- `-` : Feature is absent
- `0` : Feature is not applicable (e.g., "nasal" for vowels)

**Coverage**: 39 English phonemes (General American English)

### Complete PHOIBLE Features Reference

#### Major Class Features (4 features)

| Feature | + | - | 0 | Definition |
|---------|---|---|---|------------|
| **consonantal** | Obstruents, nasals, liquids (p, t, k, s, n, l, ɹ) | Vowels, glides (i, u, w, j) | - | Constriction in oral cavity |
| **syllabic** | Vowels (i, æ, u) | Consonants (p, t, k, s) | - | Can form syllable nucleus |
| **sonorant** | Vowels, nasals, liquids, glides (i, n, l, w) | Obstruents (p, t, k, s, f) | - | Spontaneous voicing possible |
| **approximant** | Glides, liquids (w, j, l, ɹ) | Obstruents, vowels | - | Close approximation without turbulence |

**Key distinction**: Obstruents have `sonorant:-`, sonorants have `sonorant:+`. This is the **major class difference** used in maximal opposition.

#### Laryngeal Features (11 features)

| Feature | + | - | 0 | Definition |
|---------|---|---|---|------------|
| **voice** | Voiced sounds (b, d, g, z, v, m, n, all vowels) | Voiceless sounds (p, t, k, s, f, θ) | - | Vocal fold vibration during articulation |
| **spreadGlottis** | Aspirated sounds (pʰ, tʰ, kʰ, h) | Non-aspirated | - | Glottis spread for aspiration |
| **constrictedGlottis** | Glottalized sounds (ʔ) | Non-glottalized | - | Glottis constricted |
| **stiffVocalFolds** | Voiceless obstruents (p, t, k, s, f) | Voiced obstruents, sonorants | Vowels, sonorants | Vocal folds stiffened |
| **slackVocalFolds** | Breathy voiced (rare in English) | Most sounds | - | Vocal folds slackened |
| **periodicGlottalSource** | Voiced sounds (all vowels, b, d, g, m, n) | Voiceless sounds (p, t, k, s, f) | - | Periodic vibration of vocal folds |
| **epilaryngealSource** | Pharyngealized sounds | Most English sounds | - | Epilaryngeal constriction |
| **raisedLarynxEjective** | Ejectives (not in English) | Most sounds | - | Raised larynx for ejection |
| **loweredLarynxImplosive** | Implosives (not in English) | Most sounds | - | Lowered larynx for implosion |
| **constricted** | Tense vowels (i, u) | Lax vowels (ɪ, ʊ) | Consonants | Pharyngeal constriction |
| **fortis** | Voiceless obstruents (p, t, k, s) | Voiced obstruents (b, d, g, z) | Sonorants | Greater articulatory force |

**Note**: Many laryngeal features correlate with voicing. Voiceless obstruents typically have `stiffVocalFolds:+` and `fortis:+`, while voiced obstruents have these features as `-`.

#### Manner Features (6 features)

| Feature | + | - | 0 | Definition |
|---------|---|---|---|------------|
| **continuant** | Fricatives, approximants (f, s, w, ɹ, vowels) | Stops, affricates, nasals (p, t, k, tʃ, m) | - | Airflow continues through oral cavity |
| **nasal** | Nasal consonants (m, n, ŋ) | Oral sounds (all others) | - | Airflow through nasal cavity |
| **strident** | Sibilants, labiodentals (s, z, ʃ, ʒ, f, v, tʃ, dʒ) | Non-sibilants (θ, ð, p, t) | Sonorants, vowels | High-amplitude frication noise |
| **lateral** | Lateral approximants (l) | Central sounds (all others) | - | Airflow along sides of tongue |
| **delayedRelease** | Affricates (tʃ, dʒ) | Stops, fricatives (p, t, s) | - | Gradual release from closure |
| **tap** | Flaps/taps (rare in standard American English) | All others | - | Ballistic tongue movement |

**Key for intervention**: `continuant` distinguishes stops/affricates (`-`) from fricatives (`+`). `strident` identifies high-noise fricatives.

#### Place Features - Articulator (3 features)

| Feature | + | - | Definition |
|---------|---|---|------------|
| **labial** | Bilabials, labiodentals (p, b, m, f, v, w) | Non-labials (t, d, k, s) | Lips involved in articulation |
| **coronal** | Alveolars, dentals, palatals (t, d, s, z, θ, ð, ʃ, ʒ, n, l, ɹ) | Non-coronals (p, k, m) | Tongue blade/tip raised |
| **dorsal** | Velars, back vowels (k, g, ŋ, u, ʊ, o, ɔ, ɑ) | Non-dorsals (p, t, s, i, e) | Tongue body raised toward velum |

**Note**: Multiple place features can be active simultaneously:
- /w/: `labial:+, dorsal:+` (both lips and tongue back)
- /j/: `coronal:+, dorsal:+` (both tongue blade and body)

#### Place Features - Tongue Body (8 features)

| Feature | + | - | 0 | Definition |
|---------|---|---|---|------------|
| **high** | High vowels, palatals (i, ɪ, u, ʊ, j) | Mid/low vowels (e, ɛ, æ, ɑ) | Most consonants | Tongue body raised |
| **low** | Low vowels (æ, ɑ, ɔ) | Mid/high vowels (i, e, u) | Most consonants | Tongue body lowered |
| **front** | Front vowels, palatals (i, ɪ, e, ɛ, æ, j) | Back vowels (u, ʊ, o, ɔ, ɑ) | Most consonants | Tongue body fronted |
| **back** | Back vowels, velars (u, ʊ, o, ɔ, ɑ, k, g, ŋ, w) | Front vowels (i, e, æ) | Most consonants | Tongue body backed |
| **tense** | Tense vowels (i, e, u, o) | Lax vowels (ɪ, ɛ, ʊ, ɔ) | Consonants | Greater muscular tension |
| **retractedTongueRoot** | RTR vowels (rare in English) | Most English vowels | - | Tongue root retracted |
| **advancedTongueRoot** | ATR vowels (tense vowels in some analyses) | Lax vowels | - | Tongue root advanced |
| **raisedLarynx** | Some tense vowels | Most sounds | - | Larynx raised during articulation |

**Vowel space**: The features `high`, `low`, `front`, `back`, and `tense` define the vowel space:
- /i/: `high:+, front:+, tense:+` (high front tense)
- /ɪ/: `high:+, front:+, tense:-` (high front lax)
- /æ/: `low:+, front:+, tense:-` (low front lax)
- /ɑ/: `low:+, back:+, tense:-` (low back lax)

#### Place Features - Detailed (5 features)

| Feature | + | - | 0 | Definition |
|---------|---|---|---|------------|
| **anterior** | Labials, dentals, alveolars (p, t, s, θ, m, n, l) | Palatals, velars (ʃ, k, g, ŋ, j) | Vowels | Articulated at or in front of alveolar ridge |
| **distributed** | Palatals, dentals, laterals (ʃ, θ, l) | Alveolars (t, s) | Vowels | Longer constriction along midline |
| **strident** | (see Manner features) | | | |
| **labialDental** | Labiodentals (f, v) | All others | - | Lower lip against upper teeth |
| **retroflexed** | Rhotic sounds (/ɹ/ in some dialects) | Most sounds | - | Tongue tip curled back |

**Note**: `anterior` distinguishes sounds articulated further forward (`+`) from those further back (`-`).

### Example: Phoneme Lookup for /k/

```
Phoneme: /k/
Type: Consonant (stop)

Major Class:
  consonantal: +
  syllabic: -
  sonorant: -
  approximant: -

Laryngeal:
  voice: -
  spreadGlottis: + (in word-initial position)
  periodicGlottalSource: -
  stiffVocalFolds: +
  fortis: +

Manner:
  continuant: -
  nasal: -
  strident: 0
  lateral: -
  delayedRelease: -

Place (Articulator):
  labial: -
  coronal: -
  dorsal: +

Place (Tongue Body):
  high: +
  low: -
  front: -
  back: +

Place (Detailed):
  anterior: -
```

**Interpretation**: /k/ is a voiceless velar stop. It's an obstruent (`sonorant:-`), not continuous (`continuant:-`), and articulated with the tongue body raised toward the velum (`dorsal:+, high:+, back:+`). In word-initial position, it's typically aspirated (`spreadGlottis:+`).

## Phoneme Comparison

Compare two phonemes feature-by-feature to understand minimal pairs, maximal opposition, and phonological relationships.

### Comparison Algorithm

1. **Feature-by-feature comparison**: Compare all 38 PHOIBLE features
2. **Count agreements**: Features where both phonemes have the same value (+, -, or 0)
3. **Count differences**: Features where phonemes have different values
4. **Compute similarity**: `similarity = agreements / (agreements + differences)`
5. **Identify major class difference**: Check if `sonorant` features differ

### Similarity Score Interpretation

| Score Range | Interpretation | Typical Relationships |
|-------------|----------------|------------------------|
| **0.95-1.0** | Minimal pair | Voicing pairs (t/d, k/g), single feature difference |
| **0.85-0.94** | Very similar | Same place/manner, minor differences |
| **0.70-0.84** | Similar | Same major class, different place or manner |
| **0.60-0.69** | Moderate | Different manner within same major class |
| **< 0.60** | Very different | Different major class (obstruent vs. sonorant) |

**Major class difference**: If `sonorant` values differ, phonemes belong to different major classes (obstruent vs. sonorant). This receives +100 bonus in maximal opposition scoring.

### Example 1: /t/ vs /d/ (Minimal Pair - Voicing)

```
Phoneme 1: /t/ (voiceless alveolar stop)
Phoneme 2: /d/ (voiced alveolar stop)

Shared Features (36 matching):
  consonantal: +
  syllabic: -
  sonorant: -
  approximant: -
  continuant: -
  nasal: -
  labial: -
  coronal: +
  dorsal: -
  anterior: +
  ... (and 26 more)

Different Features (2):
  voice: /t/ = - | /d/ = +
  periodicGlottalSource: /t/ = - | /d/ = +

Similarity Score: 36 / (36 + 2) = 0.947

Major Class Difference: NO (both sonorant:-)
```

**Interpretation**: /t/ and /d/ are a canonical minimal pair differing only in voicing. They share 36 of 38 features, resulting in very high similarity (0.947). Because they're both obstruents (`sonorant:-`), there's no major class difference.

### Example 2: /s/ vs /l/ (Maximal Opposition)

```
Phoneme 1: /s/ (voiceless alveolar fricative)
Phoneme 2: /l/ (voiced alveolar lateral approximant)

Shared Features (24 matching):
  consonantal: +
  syllabic: -
  labial: -
  coronal: +
  dorsal: -
  anterior: +
  ... (and 18 more)

Different Features (14):
  sonorant: /s/ = - | /l/ = + ⭐ MAJOR CLASS DIFFERENCE
  approximant: /s/ = - | /l/ = +
  voice: /s/ = - | /l/ = +
  continuant: /s/ = + | /l/ = +
  strident: /s/ = + | /l/ = -
  lateral: /s/ = - | /l/ = +
  periodicGlottalSource: /s/ = - | /l/ = +
  stiffVocalFolds: /s/ = + | /l/ = -
  fortis: /s/ = + | /l/ = -
  ... (and 5 more)

Similarity Score: 24 / (24 + 14) = 0.632

Major Class Difference: YES (sonorant: - vs +)

Maximal Opposition Score: 14 + 100 = 114
```

**Interpretation**: /s/ and /l/ differ in 14 features AND belong to different major classes (obstruent vs. sonorant). This qualifies as maximal opposition - many feature differences plus a major class distinction. Research suggests this promotes broader phonological system reorganization.

### Example 3: /i/ vs /ɪ/ (Tense/Lax Vowel Pair)

```
Phoneme 1: /i/ (high front tense vowel)
Phoneme 2: /ɪ/ (high front lax vowel)

Shared Features (35 matching):
  consonantal: -
  syllabic: +
  sonorant: +
  voice: +
  high: +
  front: +
  low: -
  back: -
  ... (and 27 more)

Different Features (3):
  tense: /i/ = + | /ɪ/ = -
  advancedTongueRoot: /i/ = + | /ɪ/ = -
  constricted: /i/ = + | /ɪ/ = -

Similarity Score: 35 / (35 + 3) = 0.921

Major Class Difference: NO (both sonorant:+)
```

**Interpretation**: /i/ and /ɪ/ differ primarily in tenseness, with /i/ being more tense and having advanced tongue root. They share most features, resulting in high similarity (0.921).

## Search by Features

Find all phonemes matching specific feature combinations. Useful for identifying sound classes, planning intervention hierarchies, or understanding phonological patterns.

### Search Algorithm

1. **Select features**: Choose from 38 PHOIBLE features
2. **Set values**: Specify +, -, or 0 for each selected feature
3. **Filter phonemes**: Return all phonemes where ALL specified features match (AND logic)
4. **Display results**: Show matching phonemes with full feature matrices

**Logic**: All specified features must match (conjunction). Unspecified features are ignored.

### Common Sound Class Searches

#### Obstruents

**Find all obstruents:**
```
consonantal: +
sonorant: -

Results (16): p, b, t, d, k, g, f, v, θ, ð, s, z, ʃ, ʒ, tʃ, dʒ
```

#### Sonorants

**Find all sonorants (non-vowels):**
```
consonantal: +
sonorant: +

Results (6): m, n, ŋ, l, ɹ, (and possibly w, j depending on analysis)
```

#### Fricatives

**Find all fricatives:**
```
consonantal: +
sonorant: -
continuant: +

Results (10): f, v, θ, ð, s, z, ʃ, ʒ, h, (and possibly more)
```

#### Stops

**Find all stops:**
```
consonantal: +
sonorant: -
continuant: -
delayedRelease: -

Results (6): p, b, t, d, k, g
```

#### Voiced Stops

**Find all voiced stops:**
```
consonantal: +
sonorant: -
continuant: -
delayedRelease: -
voice: +

Results (3): b, d, g
```

#### Voiceless Stops

**Find all voiceless stops:**
```
consonantal: +
sonorant: -
continuant: -
delayedRelease: -
voice: -

Results (3): p, t, k
```

#### Nasals

**Find all nasals:**
```
consonantal: +
sonorant: +
nasal: +

Results (3): m, n, ŋ
```

#### Liquids

**Find all liquids:**
```
consonantal: +
sonorant: +
approximant: +
nasal: -

Results (2): l, ɹ
```

#### Front Vowels

**Find all front vowels:**
```
consonantal: -
syllabic: +
front: +

Results (5): i, ɪ, e, ɛ, æ
```

#### High Vowels

**Find all high vowels:**
```
consonantal: -
syllabic: +
high: +

Results (4): i, ɪ, u, ʊ
```

#### Tense Vowels

**Find all tense vowels:**
```
consonantal: -
syllabic: +
tense: +

Results (5): i, e, u, o, ɔ (varies by dialect)
```

#### Sibilants

**Find all sibilants:**
```
consonantal: +
strident: +
continuant: +

Results (6): s, z, ʃ, ʒ, (plus affricates tʃ, dʒ with delayedRelease:+)
```

### Advanced Searches

#### Alveolar Obstruents

**Find all alveolar obstruents:**
```
consonantal: +
sonorant: -
coronal: +
anterior: +

Results (6): t, d, s, z (and potentially θ, ð depending on features)
```

#### Back Rounded Vowels

**Find all back rounded vowels:**
```
consonantal: -
syllabic: +
back: +

Results (4): u, ʊ, o, ɔ (Note: rounding not directly encoded, inferred from backness in English)
```

#### Late-Developing Sounds (Approximation)

**Find fricatives and affricates (late-developing classes):**
```
consonantal: +
sonorant: -
continuant: +

Results: All fricatives

OR

delayedRelease: +

Results: All affricates (tʃ, dʒ)
```

## Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| Word lookup | < 5 ms | O(1) hash table lookup |
| Phoneme lookup | < 1 ms | O(1) direct access to feature matrix |
| Phoneme comparison | < 1 ms | O(38) feature-by-feature comparison |
| Search by features | < 10 ms | O(39 × F) where F = number of features |
| Full feature matrix display | < 5 ms | Rendering 38 features |

**Factors Affecting Speed**:
- Number of features specified (more features = slightly faster filtering)
- Browser rendering (MUI table rendering is the bottleneck)
- Result set size (larger results = slower display)

**Data Size**:
- Phoneme features: ~2 KB (39 phonemes × 38 features × 1 byte)
- Word metadata: ~12 MB uncompressed (48,720 words)
- Minimal pairs: ~600 KB gzipped

## Vocabulary and Coverage

| Category | Coverage | Notes |
|----------|----------|-------|
| **Words** | 48,720 | CMU Pronouncing Dictionary (primary pronunciations) |
| **Phonemes** | 39 | General American English |
| **Phonological properties** | 100% | Syllables, phonemes, IPA |
| **WCM/MSH** | ~95% | Computed for most words |
| **Psycholinguistic norms** | 40-99% | Varies by property (see tables above) |
| **Dialect** | General American | Primary pronunciations only |

**Excluded**:
- Pronunciation variants (CMU entries with (1), (2), etc.)
- Proper nouns
- Non-English loanwords without standard pronunciations

## Tips and Best Practices

### Word Lookup
- **Case-insensitive**: "cat", "Cat", "CAT" all work
- **Exact match required**: Searches for "cat" won't find "cats"
- **Missing data**: Properties without values display as "—" or "N/A"
- **IPA display**: Uses Unicode IPA characters

### Phoneme Lookup
- **Exact IPA required**: Use the IPA keyboard or copy-paste
- **Case-sensitive**: /i/ ≠ /ɪ/ (different phonemes)
- **Coverage**: 39 English phonemes (vowels, consonants, glides)

### Phoneme Comparison
- **Use for minimal pairs**: Compare sounds that differ in one feature (e.g., /t/ vs /d/)
- **Use for maximal opposition**: Compare sounds with major class difference (e.g., /s/ vs /l/)
- **Similarity score**: 0.0-1.0, higher = more similar
- **Major class difference**: Look for `sonorant` value difference

### Search by Features
- **Start broad**: Begin with major class features (consonantal, sonorant)
- **Refine incrementally**: Add features one at a time
- **Use AND logic**: All specified features must match
- **Check coverage**: Some features are not applicable to all phonemes

## Use Cases

### Clinical Practice

1. **Understanding error patterns**: Compare target phoneme to substitution (e.g., /k/ vs /t/ for fronting)
2. **Planning intervention**: Find maximal opposition pairs (major class difference + many features)
3. **Identifying sound classes**: Search for fricatives, stops, nasals for treatment hierarchies
4. **Word selection**: Look up words to ensure appropriate complexity (WCM, MSH) and frequency

### Research

1. **Stimulus control**: Match words on frequency, AoA, imageability, concreteness
2. **Phonological variables**: Select words by phoneme count, syllable count, complexity
3. **Semantic variables**: Control for concreteness, imageability, familiarity
4. **Affective content**: Select words by valence, arousal, dominance
5. **Feature analysis**: Compare phoneme inventories across sound classes

### Education

1. **Phonology instruction**: Explore distinctive features and sound classes
2. **Minimal pairs**: Find pairs differing in single features
3. **Vowel space**: Examine front/back, high/low, tense/lax distinctions
4. **Consonant inventory**: Understand place, manner, voicing contrasts

## References

**PHOIBLE Distinctive Features**:
- Moran, S., & McCloy, D. (2019). PHOIBLE 2.0. Max Planck Institute. https://phoible.org/
- Hayes, B. (2009). Introductory Phonology. Wiley-Blackwell.
- Moisik, S. R., & Esling, J. H. (2011). The 'whole larynx' approach to laryngeal features. *ICPhS XVII*, 1406-1409.

**Phonological Complexity**:
- Stoel-Gammon, C. (2010). The Word Complexity Measure: Description and application to developmental phonology and disorders. *Clinical Linguistics & Phonetics*, 24(4-5), 271-282.
- Namasivayam, A. K., et al. (2021). Milestones of speech production in children. *Journal of Speech, Language, and Hearing Research*.

**Psycholinguistic Norms**:
- Brysbaert, M., & New, B. (2009). Moving beyond Kučera and Francis. *Behavior Research Methods*, 41(4), 977-990.
- Brysbaert, M., et al. (2014). Concreteness ratings for 40 thousand English words. *Behavior Research Methods*, 46, 904-911.
- Scott, G. G., et al. (2019). The Glasgow Norms: Ratings of 5,500 words. *Behavior Research Methods*, 51, 1258-1270.
- Warriner, A. B., et al. (2013). Norms of valence, arousal, and dominance. *Behavior Research Methods*, 45, 1191-1207.

## See Also

- [PHOIBLE Features Reference](../reference/phoible-features.md) - Complete documentation of all 38 features
- [Psycholinguistic Norms Reference](../reference/psycholinguistic-norms.md) - Complete property documentation
- [Contrastive Sets](contrastive-sets.md) - Use phoneme comparisons for intervention planning
- [Custom Word Lists](custom-word-lists.md) - Filter words by properties shown in Word Lookup
- [Technical Architecture](../technical/architecture.md) - How features are extracted and compared
