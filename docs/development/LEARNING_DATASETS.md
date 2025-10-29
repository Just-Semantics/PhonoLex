# Phonological Learning Datasets

**What data can we actually LEARN from?**

This document catalogs datasets that contain **phonological patterns, rules, and alternations** - not just static pronunciations, but data where we can learn systematic relationships.

---

## 🎯 What Makes a Good "Learning" Dataset?

### Static Pronunciation Data (What we have)
- **CMU Dict**: word → pronunciation mapping
- **Phoible**: phoneme → features mapping
- **Use**: Lookup, similarity search

### Learning Data (What we want)
- **Morphological alternations**: sing → sang, cat → cats
- **Phonological rules**: /s/ vs /z/ vs /ɪz/ for plurals
- **Sound changes**: historical evolution
- **Contextual variation**: allophones, co-articulation
- **Use**: Learn patterns, predict novel forms, generalize

---

## 📊 Available Datasets

### 1. SIGMORPHON Shared Tasks ⭐⭐⭐

**What**: Morphological inflection across 90+ languages
**URL**: https://sigmorphon.github.io/sharedtasks/

**Data Format**:
```
lemma: sing
features: V;PST
inflected: sang

lemma: cat
features: N;PL
inflected: cats
```

**Coverage**:
- 90+ languages across 13 language families
- 100K+ inflection examples per language (varies)
- Past tense, plurals, case marking, verb conjugation, etc.

**What you can learn**:
- ✅ Morphophonological alternations (sing/sang, cat/cats)
- ✅ Phonological conditioning (plural: -s vs -z vs -ɪz)
- ✅ Cross-linguistic patterns
- ✅ Rare/irregular forms

**Phonological content**:
- Includes phonological transcriptions for some languages
- Learn: stem changes, affix selection, vowel harmony, etc.

**Download**: GitHub repos linked from sigmorphon.github.io
**Format**: TSV files (lemma, features, inflected_form)

**Example use cases**:
1. Learn English plural rule: cat → cats, dog → dogs, bus → buses
2. Learn past tense: walk → walked, sing → sang, go → went
3. Generalize to novel words (wug → wugs)

---

### 2. CELEX (English/Dutch/German) ⭐⭐⭐

**What**: Comprehensive lexical database with morphology + phonology
**URL**: https://catalog.ldc.upenn.edu/LDC96L14 (Linguistic Data Consortium)
**Free access**: https://webcelex.ivdnt.org/

**Content for English**:
- 52,447 word forms
- 160,595 lemmas
- Orthography, phonology (SAMPA), morphology, syntax, frequency

**Morphology**:
- Derivational structure (teach → teacher)
- Compositional structure (blackboard)
- Inflectional paradigms (sing/sang/sung)

**Phonology**:
- Phonetic transcriptions (SAMPA notation)
- Syllable structure
- Primary stress marking
- Pronunciation variations

**What you can learn**:
- ✅ Morphological decomposition (un-believ-able)
- ✅ Stress patterns in derived words
- ✅ Phonological changes in morphology
- ✅ Frequency effects

**Example**:
```
Word: unbelievable
Morphology: [[un-][[[believe]V]V]Adj]Adj]
Phonology: @n-bI-"li-v@-b@l (SAMPA)
Syllables: @n.bI."li.v@.b@l
Stress: secondary on "li"
```

**Download**: Requires LDC membership ($) or use WebCelex (free web interface)

---

### 3. Wiktionary-Derived Datasets ⭐⭐

#### 3a. WikiPron

**What**: 1.7 million pronunciations from 165 languages
**URL**: https://github.com/LREC-SIGMORPHON-Workshop/wikipron (mentioned in papers)

**Format**: word → IPA transcription
**Languages**: 165 (massively multilingual)

**What you can learn**:
- ✅ Grapheme-to-phoneme rules
- ✅ Cross-linguistic phonotactics
- ✅ Loan word adaptation

#### 3b. ipa-dict

**What**: Monolingual wordlists with IPA
**URL**: https://github.com/open-dict-data/ipa-dict

**Languages**: English, German, French, Spanish, etc.
**Format**: TSV (word \\t IPA)

**Example**:
```
hello	həˈloʊ
world	wɝld
```

**What you can learn**:
- ✅ Pronunciation patterns
- ✅ Stress assignment
- ✅ Syllabification

#### 3c. MorphyNet

**What**: 13.5M inflectional + 696K derivational forms across 15 languages
**URL**: https://github.com/kbatsuren/MorphyNet

**Content**:
- Inflections: sing → sang, cat → cats
- Derivations: teach → teacher → teachable

**Format**: TSV with morphological relationships

**What you can learn**:
- ✅ Inflectional paradigms
- ✅ Derivational patterns
- ✅ Productivity of affixes

#### 3d. wiktextract

**What**: Python tool to extract structured data from Wiktionary dumps
**URL**: https://pypi.org/project/wiktextract/

**Extracts**:
- Pronunciations (IPA + audio links)
- Inflection tables
- Etymologies
- Translations

**What you can learn**:
- ✅ Build custom datasets from Wiktionary
- ✅ Extract pronunciation + morphology together

---

### 4. English Past Tense & Pluralization Datasets ⭐⭐

**Classic psycholinguistic datasets**:

#### Rumelhart & McClelland (1986) Past Tense
- ~500 English verbs
- Regular (walk → walked) vs irregular (sing → sang)
- Used in classic neural network debate

#### Wug Test Extensions
- Berko (1958) extended with modern data
- Novel words (wug → wugs, gling → glinged)
- Tests phonological conditioning

**Where to find**:
- Research papers often include data in appendices
- Supplementary materials on journal websites
- Contact authors for datasets

**What you can learn**:
- ✅ Regular vs irregular patterns
- ✅ Phonological conditioning of allomorphs
- ✅ Generalization to novel forms

**Example patterns**:
```
# English Plural
cat [t] → cats [s]     (voiceless)
dog [g] → dogs [z]     (voiced)
bus [s] → buses [ɪz]   (sibilant)

# English Past Tense
walk [k] → walked [t]   (voiceless)
hug [g] → hugged [d]    (voiced)
pat [t] → patted [ɪd]   (alveolar stop)
```

---

### 5. UniMorph ⭐⭐⭐

**What**: Universal morphological annotation across 150+ languages
**URL**: https://unimorph.github.io/

**Content**:
- Inflectional paradigms
- Standardized feature schema
- Linked to Universal Dependencies

**Format**:
```
lemma \t inflected_form \t feature_bundle
sing  \t sang            \t V;PST
cat   \t cats            \t N;PL
```

**Languages**: 150+
**Size**: Millions of inflected forms

**What you can learn**:
- ✅ Cross-linguistic morphological patterns
- ✅ Paradigm completion
- ✅ Typological generalizations

---

### 6. Spoken Corpora (with phonetic transcription)

#### TIMIT
- Phonetically-transcribed speech
- 630 speakers, 8 dialects of American English
- Phoneme-level annotations

#### Buckeye Corpus
- Conversational speech
- Time-aligned phonetic transcriptions
- Captures real phonological variation

**What you can learn**:
- ✅ Allophonic variation
- ✅ Co-articulation effects
- ✅ Fast speech processes (reduction, deletion)

**Limitation**: Acoustic data, not just phonology

---

## 🎯 Recommended Learning Tasks

### Task 1: English Morphophonology
**Dataset**: SIGMORPHON English + CELEX
**Learn**:
- Plural allomorphy (-s, -z, -ɪz)
- Past tense allomorphy (-t, -d, -ɪd)
- Irregular patterns (sing/sang, go/went)

**Method**: Neural sequence-to-sequence with attention

### Task 2: Cross-Linguistic Sound Patterns
**Dataset**: Phoible + UniMorph
**Learn**:
- Which phonological features co-occur?
- Universal tendencies vs language-specific
- Predict phoneme inventory from typological features

**Method**: Statistical analysis, clustering, PCA

### Task 3: Grapheme-to-Phoneme
**Dataset**: CMU Dict + WikiPron
**Learn**:
- Spelling → pronunciation rules
- Handle irregularities (through, though, tough)

**Method**: Transformer with character-level attention

### Task 4: Stress Assignment
**Dataset**: CELEX
**Learn**:
- Primary stress placement in derived words
- Stress shifts: photograph → photográphic

**Method**: LSTM with syllable-level features

### Task 5: Phoneme Distribution Patterns
**Dataset**: Phoible feature vectors
**Learn**:
- Which features predict which others?
- Implicational universals
- Typological clusters

**Method**: Graph neural networks, Bayesian models

---

## 💡 Novel Dataset Ideas

### 1. CMU Dict + Morphology
**Create**: Add morphological decomposition to CMU entries
```
teachable → teach + able
pronunciation: [tʃiːʃtʃəbəl] → [tiːtʃ] + [əbəl]
```

**Learn**: How morphology affects pronunciation

### 2. Phoible + Historical Sound Changes
**Create**: Link related languages in Phoible
```
Latin /k/ → Italian /tʃ/ before front vowels
centum → cento [tʃɛnto]
```

**Learn**: Predict sound changes, directionality

### 3. English Dialect Variation
**Create**: Map CMU (General American) to other dialects
```
General American /r/ → British RP /ɹ/ deletion
car: /kɑr/ → /kɑː/
```

**Learn**: Systematic dialectal differences

---

## 🔧 Quick Start: Download & Explore

### SIGMORPHON 2020
```bash
git clone https://github.com/sigmorphon/2020.git
cd 2020/task0-data

# Explore English data
head eng/eng.trn  # Training data
# Output: lemma \t inflected \t features
# walk  walked   V;PST
# cat   cats     N;PL
```

### WikiPron (IPA-dict)
```bash
git clone https://github.com/open-dict-data/ipa-dict.git
cd ipa-dict

# English pronunciations
head data/en_US.txt
# Output: word \t IPA
# hello	həˈloʊ
```

### MorphyNet
```bash
git clone https://github.com/kbatsuren/MorphyNet.git
cd MorphyNet

# English inflections
head eng.infl.txt
# Output: lemma \t inflected \t features
```

---

## 📈 Comparison Matrix

| Dataset | Size | Languages | Morphology | Phonology | Free | Learning Potential |
|---------|------|-----------|------------|-----------|------|--------------------|
| **SIGMORPHON** | 100K+ | 90+ | ✅ Strong | ⚠️ Partial | ✅ Yes | ⭐⭐⭐ Excellent |
| **CELEX** | 160K | 3 | ✅ Strong | ✅ Strong | ⚠️ LDC/Web | ⭐⭐⭐ Excellent |
| **UniMorph** | Millions | 150+ | ✅ Strong | ❌ None | ✅ Yes | ⭐⭐ Good |
| **WikiPron** | 1.7M | 165 | ❌ None | ✅ IPA | ✅ Yes | ⭐⭐ Good |
| **MorphyNet** | 13.5M | 15 | ✅ Strong | ❌ None | ✅ Yes | ⭐⭐ Good |
| **CMU Dict** | 134K | 1 | ❌ None | ✅ ARPAbet | ✅ Yes | ⭐ Lookup only |
| **Phoible** | 105K | 2,716 | ❌ None | ✅ Features | ✅ Yes | ⭐⭐ Typology |

---

## 🎯 My Recommendation

### Start with: **SIGMORPHON + Phoible Features**

**Why?**
1. **Free and readily available** (just git clone)
2. **English data** is well-developed (tens of thousands of examples)
3. **Clear learning task**: lemma + features → inflected form
4. **Phonological patterns**: You can learn plural/past tense allomorphy
5. **Can enhance with Phoible vectors**: Add phonological features to SIGMORPHON data

**Example pipeline**:
```python
# 1. Get SIGMORPHON data
lemma: "cat", features: "N;PL" → inflected: "cats"

# 2. Convert to phonemes (via CMU dict)
"cat" → /k æ t/ → "cats" → /k æ t s/

# 3. Add Phoible features
/t/ → [consonantal:+, voiced:-, ...] (38 features)
/s/ → [consonantal:+, voiced:-, sibilant:+, ...]

# 4. LEARN: Which suffix allomorph to use?
/t/ (voiceless) → /-s/ (voiceless)
/g/ (voiced) → /-z/ (voiced)
/s/ (sibilant) → /-ɪz/ (epenthetic)
```

This combines:
- ✅ Morphological patterns (SIGMORPHON)
- ✅ Phonological features (Phoible)
- ✅ Pronunciation data (CMU)

You could **learn phonologically-conditioned allomorphy**!

---

## Next Steps

Want me to:
1. Download SIGMORPHON English data and analyze it?
2. Create a combined dataset: SIGMORPHON + CMU + Phoible?
3. Build a phonological rule learner?

