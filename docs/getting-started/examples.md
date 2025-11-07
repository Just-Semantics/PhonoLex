# Practical Examples

This guide provides hands-on examples you can try in PhonoLex to understand each tool's capabilities.

## Custom Word Lists Examples

### Example 1: Simple CVC Words for Early Intervention

**Goal:** Find simple consonant-vowel-consonant words for early speech therapy

**Steps:**
1. Open **Custom Word Lists**
2. Add filter: **Syllables** = 1
3. Add filter: **Phonemes** ≤ 4
4. Add filter: **WCM** ≤ 3 (low complexity)
5. Add filter: **Frequency** ≥ 10 (common words)
6. Generate

**Expected Results:**
- Words like: cat, dog, bed, cup, hat, sit, run
- All monosyllabic, simple structure, commonly used

**Why This Works:**
- Syllable count = 1 ensures no multisyllabic words
- Phoneme count ≤ 4 keeps words short
- Low WCM means fewer complex features (clusters, velars, etc.)
- High frequency ensures familiar, functional words

---

### Example 2: Initial /s/ Words with Semantic Scaffolding

**Goal:** Find /s/ initial words that are highly imageable for articulation practice

**Steps:**
1. Open **Custom Word Lists**
2. Add pattern: **STARTS_WITH** → /s/
3. Add filter: **Imageability** ≥ 5.0 (highly imageable)
4. Add filter: **Frequency** ≥ 5 (reasonably common)
5. Generate

**Expected Results:**
- Words like: sun, snake, sock, sand, soap, snow
- All start with /s/, easy to visualize

**Why This Works:**
- STARTS_WITH /s/ targets specific phoneme position
- High imageability helps children create mental images
- Frequency filter ensures words are useful in daily life

---

### Example 3: Late-Developing Sounds in Simple Contexts

**Goal:** Find words with /ɹ/ (r-sound) in simple, early-acquired contexts

**Steps:**
1. Open **Custom Word Lists**
2. Add pattern: **CONTAINS** → /ɹ/
3. Add filter: **Syllables** = 1
4. Add filter: **MSH** ≤ 4.0 (lower motor complexity)
5. Add filter: **Age of Acquisition** ≤ 4.0 (learned early)
6. Generate

**Expected Results:**
- Words like: hear, hair, air, fair, bear, chair, fear
- Late phoneme (/ɹ/) in otherwise simple words

**Why This Works:**
- CONTAINS /ɹ/ finds all positions (initial, medial, final)
- Monosyllabic keeps structure simple
- Low MSH means other phonemes are early-developing
- Low AoA ensures words are familiar to children

---

### Example 4: Negative Valence Words for Emotional Language

**Goal:** Find words with negative emotional content for social-emotional work

**Steps:**
1. Open **Custom Word Lists**
2. Add filter: **Valence** ≤ 3.0 (negative valence)
3. Add filter: **Arousal** ≥ 5.0 (emotionally activating)
4. Add filter: **Frequency** ≥ 10 (common words)
5. Add filter: **Imageability** ≥ 4.0 (can be visualized)
6. Generate

**Expected Results:**
- Words like: afraid, scared, mad, evil, dangerous, attack
- Negative emotional tone, high arousal, common usage

**Why This Works:**
- Low valence targets negative emotions
- High arousal ensures emotionally salient words
- Frequency + imageability make words functional and learnable

---

### Example 5: Excluding Problematic Phonemes

**Goal:** Find /k/ words without /g/ (for children substituting k→g)

**Steps:**
1. Open **Custom Word Lists**
2. Add pattern: **STARTS_WITH** → /k/
3. Add exclusion: **Exclude phoneme** → /g/
4. Add filter: **Syllables** ≤ 2
5. Generate

**Expected Results:**
- Words like: cat, car, cut, candy, carrot
- All have initial /k/, none contain /g/

**Why This Works:**
- STARTS_WITH /k/ targets the sound we're working on
- Exclusion prevents words with the substitution pattern
- Syllable limit keeps words manageable

---

## Contrastive Sets Examples

### Example 6: Minimal Pairs - Fronting (k→t)

**Goal:** Generate minimal pairs for a child who fronts /k/ to /t/

**Steps:**
1. Open **Contrastive Sets** → **Minimal Pairs**
2. Target phoneme: /k/
3. Substitute phoneme: /t/
4. Position: **Initial**
5. Word length: **Short**
6. Complexity: **Low**
7. Generate

**Expected Results:**
Minimal pairs like:
- cat / tat (if "tat" exists)
- key / tea
- came / tame
- cold / told

**Why This Works:**
- Contrasts exactly one phoneme (/k/ vs /t/)
- Initial position targets the error position
- Short, simple words make practice easier

---

### Example 7: Maximal Opposition - Major Class Difference

**Goal:** Find maximal opposition pairs for a child with moderate-severe SSD

**Unknown phonemes:** /s, ʃ, θ, l, ɹ/

**Steps:**
1. Open **Contrastive Sets** → **Maximal Opposition**
2. Enter unknown phonemes: /s, ʃ, θ, l, ɹ/ (comma-separated)
3. Click **Generate Phoneme Pairs**

**Expected Results:**

Top-ranked pairs (system calculates automatically):

| Pair | Major Class? | Feature Differences | Score |
|------|--------------|---------------------|-------|
| /θ/ - /l/ | ✓ Yes (obstruent-sonorant) | 15 | 115 |
| /s/ - /l/ | ✓ Yes (obstruent-sonorant) | 14 | 114 |
| /ʃ/ - /ɹ/ | ✓ Yes (obstruent-sonorant) | 13 | 113 |

**Why This Works:**
- Major class difference (obstruent vs. sonorant) earns +100 bonus
- Many feature differences maximize phonological contrast
- Research shows this promotes system-wide learning

**Next Step:** Select /θ/ - /l/ pair, then click **Generate Word Lists** to get minimal pairs like:
- thin / Lynn
- think / link
- mouth / mole (word-final)

---

### Example 8: Multiple Opposition - Stopping

**Goal:** Treat a child who produces all of {/s, ʃ, θ, f/} as [t]

**Steps:**
1. Open **Contrastive Sets** → **Multiple Opposition**
2. Substitute phoneme: /t/ (what child says)
3. Target phonemes: /s, ʃ, θ, f/ (what should be said)
4. Click **Generate Sets**

**Expected Results:**

**Minimal sets (4-word sets):**
- tie / sigh / shy / thigh / fie
- tore / sore / shore / Thor / four
- tee / see / she / thee / fee

**Why This Works:**
- Addresses entire phonological collapse pattern simultaneously
- Maximal Classification algorithm selects representative targets
- Maximal Distinction algorithm finds minimal sets that contrast all targets

---

## Phonological Similarity Examples

### Example 9: Finding Perfect Rhymes

**Goal:** Find words that rhyme with "cat" for phonological awareness activities

**Steps:**
1. Open **Phonological Similarity**
2. Enter target word: **cat**
3. Select preset: **Rhymes** (onset=0.0, nucleus=0.5, coda=0.5)
4. Threshold: **0.85** (high similarity)
5. Limit: **20** results
6. Click **Find Similar Words**

**Expected Results:**
- bat (0.95), hat (0.94), sat (0.93), mat (0.92), rat (0.91), pat (0.90)
- All share /æt/ ending (nucleus + coda)

**Why This Works:**
- Zero onset weight ignores initial consonants
- High nucleus/coda weights prioritize vowel and final consonant match
- High threshold (0.85) returns only near-perfect rhymes

---

### Example 10: Finding Alliteration

**Goal:** Find words that alliterate with "snap" for phonological awareness

**Steps:**
1. Open **Phonological Similarity**
2. Enter target word: **snap**
3. Select preset: **Alliteration** (onset=1.0, nucleus=0.0, coda=0.0)
4. Threshold: **0.70** (moderate similarity)
5. Limit: **20** results
6. Click **Find Similar Words**

**Expected Results:**
- Words starting with /sn/, /s/, or similar clusters:
  - snake (0.85), snack (0.83), snow (0.80)
  - slip, slide, slow (lower scores, similar onset)

**Why This Works:**
- Onset weight = 1.0 prioritizes initial sounds
- Zero nucleus/coda weights ignore vowels and endings
- Moderate threshold allows some variation in cluster structure

---

### Example 11: Custom Weights - Consonant Frames

**Goal:** Find words with similar consonant frame to "cat" (ignore vowel)

**Steps:**
1. Open **Phonological Similarity**
2. Enter target word: **cat**
3. Select preset: **Consonance** (onset=0.5, nucleus=0.0, coda=0.5)
4. Adjust threshold: **0.75**
5. Limit: **15** results
6. Click **Find Similar Words**

**Expected Results:**
- kit (0.82), cut (0.78), cot (0.78), coat (0.76)
- All have /k_t/ or similar consonant frame

**Why This Works:**
- Zero nucleus weight completely ignores vowel differences
- High onset/coda weights prioritize consonant matches
- Useful for identifying phoneme confusions based on consonant context

---

## Lookup Examples

### Example 12: Word Properties Lookup

**Goal:** Get complete phonological and psycholinguistic profile for "strength"

**Steps:**
1. Open **Lookup** → **Word Lookup**
2. Enter: **strength**
3. View results

**Expected Results:**
```
Word: strength
IPA: /strɛŋkθ/
Syllables: 1
Phonemes: 7

Phonological Complexity:
- WCM: 11 (very high - 3-consonant cluster, velars, fricatives)
- MSH: 5.5 (high motor complexity - fricatives and velar nasal)

Lexical:
- Frequency: 28.5 (fairly common)
- AoA: 5.8 (learned later)

Semantic:
- Imageability: 3.2 (abstract concept)
- Familiarity: 6.1 (familiar word)
- Concreteness: 2.5 (abstract)

Affective:
- Valence: 6.8 (positive)
- Arousal: 5.2 (moderately arousing)
- Dominance: 7.1 (high dominance/power)
```

**Why This is Useful:**
- High WCM/MSH indicates late-developing word structure
- Abstract (low imageability/concreteness) makes it harder for children
- Late AoA aligns with complexity
- Positive valence and high dominance fit semantic associations with "power"

---

### Example 13: Phoneme Feature Comparison

**Goal:** Understand why /t/ and /d/ are a minimal pair (voice contrast)

**Steps:**
1. Open **Lookup** → **Phoneme Comparison**
2. Phoneme 1: **/t/**
3. Phoneme 2: **/d/**
4. Click **Compare**

**Expected Results:**

**Shared Features (36 matching):**
- consonantal: +
- sonorant: -
- continuant: -
- nasal: -
- labial: -
- coronal: +
- anterior: +
- dorsal: -
- (etc.)

**Different Features (2):**
- **voice:** /t/ = **-** (voiceless) | /d/ = **+** (voiced)
- **periodicGlottalSource:** /t/ = **-** | /d/ = **+**

**Similarity Score: 0.947** (very similar - only voice differs)

**Why This is Useful:**
- Shows exactly which features distinguish minimal pairs
- Explains why /t/ - /d/ is a common substitution (only 1 major feature differs)
- Demonstrates that voiced/voiceless pairs share almost all features

---

### Example 14: Finding Phonemes by Features

**Goal:** Find all fricatives for articulation hierarchy planning

**Steps:**
1. Open **Lookup** → **Search by Features**
2. Add feature: **consonantal** = **+**
3. Add feature: **sonorant** = **-**
4. Add feature: **continuant** = **+**
5. Click **Search**

**Expected Results:**

**Phonemes found (9):**
- Voiceless fricatives: /f, θ, s, ʃ, h/
- Voiced fricatives: /v, ð, z, ʒ/

**Why This is Useful:**
- Identifies entire sound classes systematically
- Helps plan treatment hierarchies (fricatives often later-developing)
- Shows relationships between sounds (voice pairs)

---

### Example 15: Comparing Maximal Opposition Phonemes

**Goal:** Verify that /s/ and /l/ differ maximally (obstruent vs. sonorant)

**Steps:**
1. Open **Lookup** → **Phoneme Comparison**
2. Phoneme 1: **/s/**
3. Phoneme 2: **/l/**
4. Click **Compare**

**Expected Results:**

**Key Differences (14 features):**
- **sonorant:** /s/ = **-** (obstruent) | /l/ = **+** (sonorant) ⭐
- **continuant:** /s/ = **+** | /l/ = **+** ✓
- **strident:** /s/ = **+** | /l/ = **-**
- **lateral:** /s/ = **-** | /l/ = **+**
- **coronal:** /s/ = **+** | /l/ = **+** ✓
- **anterior:** /s/ = **+** | /l/ = **+** ✓
- **distributed:** /s/ = **-** | /l/ = **+**
- **voice:** /s/ = **-** | /l/ = **+**
- (6 more differences)

**Similarity Score: 0.632** (very different - 14 features differ)

**Major Class Difference: YES** (sonorant: - vs +)

**Why This is Useful:**
- Confirms maximal opposition (major class + many features)
- Explains why /s/ - /l/ is effective for intervention
- Shows that even though both are coronal and anterior, they differ greatly

---

## Advanced Queries

### Example 16: High-Frequency Abstract Words

**Goal:** Find abstract words for older children working on vocabulary

**Steps:**
1. Open **Custom Word Lists**
2. Add filter: **Frequency** ≥ 20 (high frequency)
3. Add filter: **Concreteness** ≤ 2.5 (abstract)
4. Add filter: **Age of Acquisition** ≥ 5.0 (learned later)
5. Generate

**Expected Results:**
- Words like: justice, theory, particular, professional, affair, destiny, former, instance
- Common abstract concepts learned in later school years

---

### Example 17: Emotionally Neutral Words

**Goal:** Find emotionally neutral words for baseline testing

**Steps:**
1. Open **Custom Word Lists**
2. Add filter: **Valence** = 4.0 - 6.0 (neutral range)
3. Add filter: **Arousal** = 3.0 - 5.0 (low arousal)
4. Add filter: **Frequency** ≥ 10
5. Generate

**Expected Results:**
- Words like: table, paper, floor, time, thing, work, wait, put
- Everyday words without emotional content

---

### Example 18: Multi-Pattern Complex Query

**Goal:** Find words with initial /k/ AND final /t/, excluding /s/

**Steps:**
1. Open **Custom Word Lists**
2. Add pattern: **STARTS_WITH** → /k/
3. Add pattern: **ENDS_WITH** → /t/
4. Add exclusion: **Exclude phoneme** → /s/
5. Add filter: **Frequency** ≥ 5
6. Generate

**Expected Results:**
- Words like: cat, kit, court, caught, coat
- All start with /k/, end with /t/, contain no /s/

**Why This is Useful:**
- Demonstrates AND logic (must match ALL patterns)
- Exclusions refine word lists for specific therapy goals
- Complex queries help find exactly the right stimuli

---

## Tips for Using Examples

1. **Start Simple:** Try examples 1-4 first to understand basic filtering
2. **Experiment:** Adjust thresholds and limits to see how results change
3. **Compare Tools:** Try the same goal across different tools (e.g., finding rhymes in Custom Word Lists vs Phonological Similarity)
4. **Export Results:** Use CSV export to save word lists for therapy materials
5. **Combine Filters:** Use multiple filters to create highly specific lists

## Next Steps

- Review [Custom Word Lists](../user-guide/custom-word-lists.md) for detailed feature explanations
- Read [Technical Architecture](../technical/architecture.md) to understand how similarity scores are calculated
- Check [Psycholinguistic Norms Reference](../reference/psycholinguistic-norms.md) for complete property descriptions
