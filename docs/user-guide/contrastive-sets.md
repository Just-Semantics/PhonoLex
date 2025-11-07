# Contrastive Sets

Generate research-based phonological intervention word lists using three evidence-based approaches.

## Overview

The Contrastive Sets tool provides three clinical intervention methods:
1. **Minimal Pairs** - Single-feature contrast (target vs. substitute)
2. **Maximal Opposition** - Two unknown phonemes with major class difference (Gierut 1989-1992)
3. **Multiple Opposition** - Global collapse treatment with minimal sets (Gierut 1989-1992, Storkel 2022)

**Vocabulary:** 24,744 English words with 31,399 precomputed minimal pair relationships

## Minimal Pairs

Phonological intervention using single-feature contrasts between a target phoneme (correct production) and substitute phoneme (child's production) **at the same position within words**.

**Key constraint:** The phoneme difference must occur at the **same position** in both words (e.g., initial, medial, or final position).

### Usage

1. Select "Minimal Pairs"
2. Enter **Target** phoneme (correct production, e.g., /θ/)
3. Enter **Substitute** phoneme (child's production, e.g., /t/)
4. Choose position: **initial**, **medial**, **final**, or **any**
5. Optional: Set word length (short/medium/long) and complexity (low/medium/high)
6. Generate pairs

### Algorithm

**Data source:** Precomputed minimal pairs (31,399 relationships)

**Search process:**
```
1. Filter pairs where phoneme1 = target AND phoneme2 = substitute
2. Filter by position (initial/medial/final) if specified
3. Filter by word length if specified:
   - Short: ≤4 phonemes
   - Medium: 5-6 phonemes
   - Long: ≥7 phonemes
4. Filter by complexity (WCM) if specified:
   - Low: WCM ≤ 4
   - Medium: WCM 5-8
   - High: WCM ≥ 9
5. Return matched pairs
```

**Performance:** ~1-5 ms (precomputed lookup)

### Example

**Input:**
- Target: /θ/ (correct: "think")
- Substitute: /t/ (child says: "tink")
- Position: Initial
- Length: Short
- Complexity: Low

**Expected output:**
- thin /θɪn/ → tin /tɪn/
- thick /θɪk/ → tick /tɪk/
- theme /θim/ → team /tim/

### Research Context

**Typical clinical application:**
- Single phoneme error pattern
- Traditional phonological intervention approach
- Focused practice on specific contrast

**Research findings on generalization:**
- Addresses single error pattern
- Generalization to other phonemes varies by individual
- May require extended intervention for broader phonological change

---

## Maximal Opposition

Targets two **unknown phonemes** (neither produced correctly) that differ maximally in phonological features, including major class differences (obstruent vs. sonorant). Words contrast these phonemes **at the same position**.

**Key constraint:** Like minimal pairs, maximal opposition requires phoneme contrast **within the same position** in word pairs.

### Theory (Gierut 1989-1992)

**Research basis:**
- Pairing two unknown sounds promotes broader phonological learning than one unknown + one known
- Major class differences (obstruent vs. sonorant) show greater generalization
- Maximal feature differences highlight phonological diversity
- System-wide changes extend beyond trained sounds

**Clinical evidence:**
- More efficient than minimal pairs for moderate-severe SSD
- Promotes generalization to untrained phonemes
- Requires fewer treatment sessions

### Scoring Algorithm

PhonoLex automatically ranks phoneme pairs using the **maximal opposition score**:

```
score = feature_differences + major_class_bonus

where:
- feature_differences = count of differing PHOIBLE features (0-38)
- major_class_bonus = 100 if major class difference exists, 0 otherwise
```

**Major class difference:**
```
Sonorants: consonantal:+ AND sonorant:+
  Examples: /m, n, ŋ, l, ɹ, w, j/

Obstruents: consonantal:+ AND sonorant:-
  Examples: /p, t, k, b, d, g, f, v, s, z, θ, ð, ʃ, ʒ, h, ʧ, ʤ/

Major class difference = (p1 is sonorant AND p2 is obstruent) OR
                        (p1 is obstruent AND p2 is sonorant)
```

**Scoring examples:**

| Pair | Major Class? | Feature Diffs | Score | Interpretation |
|------|--------------|---------------|-------|----------------|
| /θ/ - /l/ | ✓ Yes | 15 | **115** | Excellent (obstruent vs. sonorant) |
| /s/ - /l/ | ✓ Yes | 14 | **114** | Excellent (obstruent vs. sonorant) |
| /ʃ/ - /ɹ/ | ✓ Yes | 13 | **113** | Excellent (obstruent vs. sonorant) |
| /s/ - /ʃ/ | ✗ No | 8 | 8 | Poor (both obstruents) |
| /l/ - /ɹ/ | ✗ No | 5 | 5 | Poor (both sonorants) |

**Threshold:** PhonoLex only suggests pairs with major class differences (score ≥ 100)

### Usage

1. Select "Maximal Opposition"
2. Enter **unknown phonemes** (comma-separated, e.g., /s, ʃ, θ, l, ɹ/)
3. Click **Generate Phoneme Pairs**
4. System automatically ranks all pairs by score
5. Review top-ranked pairs with feature differences
6. Select a pair
7. Click **Generate Word Lists** to find minimal pairs

### Algorithm

**Step 1: Generate all pairs**
```
For each phoneme pair (i, j) where i < j:
    Load PHOIBLE features for both phonemes
    Count feature differences (38 total features)
    Check major class difference (sonorant feature)
    Calculate score = diffs + (100 if major class else 0)
    Store pair if score ≥ 100
```

**Complexity:** O(p²) where p = number of unknown phonemes (typically < 10)
- ~20-50 ms for 5-8 phonemes

**Step 2: Find word lists**
```
For selected pair (phoneme1, phoneme2):
    Load precomputed minimal pairs
    Filter pairs where BOTH phonemes occur at the SAME position:
        (word1 has phoneme1 at position X AND word2 has phoneme2 at position X)
        OR
        (word1 has phoneme2 at position X AND word2 has phoneme1 at position X)
    Group by position (initial/medial/final)
    Return pairs

Note: Position X must be identical in both words (e.g., both at position 0,
or both at position 2). This ensures true opposition within the same context.
```

**Complexity:** O(n) where n = minimal pairs count
- ~5-10 ms lookup

### Example

**Input:**
- Unknown phonemes: /s, ʃ, θ, l, ɹ/

**System output (top 3 pairs):**

| Pair | Score | Major Class | Feature Differences |
|------|-------|-------------|---------------------|
| /θ/ - /l/ | 115 | ✓ | strident, continuant, lateral, sonorant, voice, +12 more |
| /s/ - /l/ | 114 | ✓ | strident, lateral, sonorant, voice, anterior, +9 more |
| /ʃ/ - /ɹ/ | 113 | ✓ | strident, distributed, sonorant, voice, approximant, +8 more |

**Selected pair:** /θ/ - /l/

**Word lists:**
- Initial: thin/Lynn, think/link, thumb/lumb
- Final: mouth/mole, bath/ball, math/mall

### PHOIBLE Features

PhonoLex uses 38 distinctive features from the PHOIBLE database (Moran & McCloy 2019, Hayes 2009):

**Major features (commonly different):**

| Feature | Values | Description | Examples |
|---------|--------|-------------|----------|
| **consonantal** | + / - | Constriction in vocal tract | +: /t, s, l/  -: /a, i/ |
| **sonorant** | + / - | Spontaneous voicing | +: /l, m, a/  -: /t, s/ |
| **continuant** | + / - | Airflow continues | +: /s, f, l/  -: /t, p/ |
| **voice** | + / - | Vocal fold vibration | +: /b, d, z/  -: /p, t, s/ |
| **nasal** | + / - | Nasal cavity open | +: /m, n, ŋ/  -: /p, t/ |
| **lateral** | + / - | Air flows around tongue sides | +: /l/  -: all others |
| **strident** | + / - | High-frequency noise | +: /s, z, ʃ, ʒ/  -: /f, θ/ |

**Place features:**

| Feature | Values | Description | Examples |
|---------|--------|-------------|----------|
| **labial** | + / - | Lips involved | +: /p, b, m, f, v/  -: /t, k/ |
| **coronal** | + / - | Tongue blade/tip | +: /t, d, s, l/  -: /p, k/ |
| **dorsal** | + / - | Tongue body | +: /k, g, ŋ/  -: /t, p/ |
| **anterior** | + / - | Front of mouth | +: /t, s/  -: /k, ʃ/ |
| **distributed** | + / - | Broad tongue contact | +: /ʃ, ʒ/  -: /s, z/ |

**Vowel features:**

| Feature | Values | Description | Examples |
|---------|--------|-------------|----------|
| **syllabic** | + / - | Forms syllable nucleus | +: /a, i, o/  -: /t, k/ |
| **high** | + / - | Tongue raised | +: /i, u/  -: /a/ |
| **low** | + / - | Tongue lowered | +: /a, æ/  -: /i, u/ |
| **front** | + / - | Tongue forward | +: /i, e/  -: /u, o/ |
| **back** | + / - | Tongue back | +: /u, o/  -: /i, e/ |
| **tense** | + / - | Muscular tension | +: /i, u/  -: /ɪ, ʊ/ |
| **round** | + / - | Lips rounded | +: /u, o/  -: /i, a/ |

**Complete feature list:** See [PHOIBLE Features Reference](../reference/phoible-features.md) for all 38 features with definitions and examples.

### Clinical Research Context

**Assessment considerations:**
- Maximal opposition typically targets phonemes not produced in any context
- Research studies used assessment data from single-word naming and connected speech
- Stimulability testing may inform phoneme selection

**Phoneme pair selection:**
- Research prioritized pairs with major class differences (score ≥ 100)
- Studies balanced feature maximization with learnability
- Feature distance alone doesn't predict treatment success

**Word list characteristics in research:**
- Studies typically used high-frequency words
- Imageability considerations appeared in studies with young children
- Initial position frequently targeted due to perceptual salience
- Study word lists ranged from 8-20 pairs per position

**Generalization patterns in research:**
- Studies documented system-wide phonological changes
- Generalization to untrained phonemes varied by individual
- Some studies reported generalization within 8-12 weeks

---

## Multiple Opposition

Treats global phoneme collapse by contrasting the substitute phoneme with multiple target phonemes simultaneously using minimal sets (triplets, quadruplets, quintuplets). All phonemes in a set contrast **at the same position**.

**Key constraint:** All words in a minimal set must differ at **exactly the same position** (e.g., all at initial position, or all at position 2). This creates a clear phonological contrast point.

### Theory (Gierut 1989-1992, Storkel 2022)

**When to use:**
- Child collapses multiple phonemes to one substitute
- Example: /t, d, k, g/ all produced as [t]
- Global phonological patterns need addressing

**Research basis:**
- Addresses entire collapse pattern simultaneously
- Minimal sets force attention to multiple contrasts
- Faster generalization than sequential minimal pairs
- More efficient than treating each phoneme individually

**Clinical evidence:**
- Effective for severe phonological disorders
- Promotes awareness of phonological system
- Requires cognitive engagement (more complex than minimal pairs)

### Algorithms

PhonoLex uses two complementary algorithms:

#### 1. Maximal Classification

**Purpose:** Select representative target phonemes from the collapsed set

**Algorithm:**
```
Given: substitute phoneme, list of target phonemes
Goal: Select subset that maximizes phonological diversity

For each target phoneme:
    Compute feature distance from substitute
    Compute feature distance from other targets
Select targets that:
    1. Differ maximally from substitute
    2. Differ maximally from each other
```

**Example:**
- Substitute: /t/
- Candidates: /d, k, g, b, p/
- Selected: /d/ (voice), /k/ (dorsal), /g/ (voice + dorsal)
- Rationale: Maximizes feature spread (voice, place)

#### 2. Maximal Distinction

**Purpose:** Find minimal sets where all phonemes contrast at the same position

**Algorithm:**
```
Given: substitute + selected targets
Goal: Find words that form minimal sets (differ at the SAME POSITION only)

1. Index words by position and phoneme
   For each word:
       For each position P:
           index[P][phoneme].add(word)

2. Find minimal sets
   For each position P:
       For each phoneme in [substitute + targets]:
           candidates[phoneme] = words with this phoneme at position P

       If all phonemes have candidates at position P:
           Create minimal set by selecting one word per phoneme
           All words differ at position P only
           Example: tie, die, kite, guy (all differ at position 0)
```

**Critical constraints:**
- All words must have same phoneme length
- Must differ at exactly **one position** (the same position for all)
- That position can be initial, medial, or final
- **Counter-example:** tie /taɪ/, die /daɪ/, sky /skaɪ/ is NOT valid because /k/ in "sky" is at a different position than /t/ and /d/

**Complexity:** O(n) where n = vocabulary size (indexing is linear)
- ~30-80 ms for full vocabulary

### Usage

1. Select "Multiple Opposition"
2. Enter **substitute phoneme** (what child says, e.g., /t/)
3. Enter **target phonemes** (what should be said, e.g., /d, k, g/)
4. Optional: Choose position (initial/medial/final/any)
5. Click **Generate Sets**
6. System uses Maximal Classification + Maximal Distinction
7. Review minimal sets (triplets, quadruplets, quintuplets)

### Example

**Input:**
- Substitute: /t/
- Targets: /d, k, g/
- Position: Initial

**System process:**

**Step 1: Maximal Classification**
- All targets selected (small set, all differ from /t/)

**Step 2: Maximal Distinction**
- Find 4-word minimal sets (substitute + 3 targets)

**Output (minimal sets):**

| Set | /t/ (substitute) | /d/ (target) | /k/ (target) | /g/ (target) |
|-----|------------------|--------------|--------------|--------------|
| Set 1 | tie | die | kite* | guy |
| Set 2 | tore | door | core | gore |
| Set 3 | tear | dear | care | gear |

*Note: "kite" has different structure but contrasts /t/ vs /k/ in different positions - system will flag non-perfect sets

**Clinical use:**
- Present all 4 words simultaneously
- Child must produce correct phoneme for each word
- Contrasts highlight phonological differences
- Promotes system-level awareness

### Set Size in Research

| Set Size | Name | Typical Application |
|----------|------|---------------------|
| 3 words | Triplet | Substitute + 2 targets |
| 4 words | Quadruplet | Substitute + 3 targets |
| 5 words | Quintuplet | Substitute + 4 targets |

**Research findings:**
- Studies used various set sizes depending on collapse pattern
- Larger sets (4-5 words) appeared in studies with older children
- Set size often determined by availability of appropriate word sets rather than predetermined choice

### Word Selection in Research

**Common characteristics in published studies:**
- Words typically matched on phoneme length
- High-frequency words often preferred
- Imageability considered in some studies with younger populations

**Factors noted in research:**
- Word familiarity relevant to treatment success
- Some studies used semantically related sets
- Phonological complexity varied across studies

---

## Research Comparison

| Approach | Typical Application | Participant Profile | Generalization Findings | Task Complexity |
|----------|---------------------|---------------------|-------------------------|-----------------|
| **Minimal Pairs** | Single phoneme contrast | Varied | Focused on trained contrast | Single contrast |
| **Maximal Opposition** | Two unknown phonemes | Moderate-severe SSD | Broader system changes | Dual contrast |
| **Multiple Opposition** | Phoneme collapse | Severe SSD | System-wide changes | Multiple contrasts |

**Application context from research:**

1. **Minimal Pairs** in research
   - Studies targeted specific phoneme contrasts
   - Example: /s/ vs. /θ/ contrast training
   - Traditional approach with extensive research base

2. **Maximal Opposition** in research
   - Studies selected phoneme pairs with major class differences
   - Example: /θ/ - /l/ pairing in Gierut (1989)
   - Research showed broader generalization than minimal pairs

3. **Multiple Opposition** in research
   - Studies addressed global collapse patterns
   - Example: /t, d, k, g/ → [t] collapse
   - Research found efficient system-wide phonological change

## Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| Minimal pairs lookup | 1-5 ms | Precomputed database |
| Maximal opposition pairs | 20-50 ms | O(p²) phoneme comparisons |
| Word list generation | 5-10 ms | Filtered minimal pairs |
| Multiple opposition sets | 30-80 ms | O(n) vocabulary indexing |

## Data Sources

**Minimal pairs:** 31,399 precomputed relationships from 24,744 words

**Features:** 38 PHOIBLE distinctive features (Moran & McCloy 2019, Hayes 2009)

**Research basis:** Clinical studies by Gierut (1989, 1990, 1992) and Storkel (2022)

## References

**Clinical Research:**

- Gierut, J. A. (1989). Maximal opposition approach to phonological treatment. *Journal of Speech and Hearing Disorders*, 54(1), 9-19.
- Gierut, J. A. (1990). Differential learning of phonological oppositions. *Journal of Speech and Hearing Research*, 33(3), 540-549.
- Gierut, J. A. (1992). The conditions and course of clinically induced phonological change. *Journal of Speech and Hearing Research*, 35(5), 1049-1063.
- Gierut, J. A., & Neumann, H. J. (1992). Teaching and learning /θ/: A non-confound. *Clinical Linguistics & Phonetics*, 6(3), 191-200.
- Storkel, H. L. (2022). Minimal, Maximal, or Multiple Oppositions: A review of phonological intervention approaches. *Language, Speech, and Hearing Services in Schools*, 53(2), 421-437.

**Phonological Features:**

- Moran, S., & McCloy, D. (2019). PHOIBLE 2.0. Max Planck Institute for the Science of Human History. https://phoible.org/
- Hayes, B. (2009). *Introductory Phonology*. Wiley-Blackwell.

## See Also

- [Practical Examples](../getting-started/examples.md) - Worked examples for all three approaches
- [PHOIBLE Features Reference](../reference/phoible-features.md) - Complete 38-feature system
- [Technical Architecture](../technical/architecture.md) - Algorithm implementation details
