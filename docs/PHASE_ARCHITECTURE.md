# Phase Architecture (v2.2.1+)

## Overview

PhonoLex uses a **3-phase deterministic pipeline** to transform raw phonological features into syllable embeddings for similarity calculations.

**No training required** - all transformations are deterministic!

## The 3 Phases

```
Phase 1: Extract       Phase 2: Normalize     Phase 3: Aggregate
(Phoible DB)          (Continuous)           (Syllables)
   ↓                      ↓                      ↓
38-dim ternary  →  76-dim normalized  →  228-dim syllables
(+, -, 0)            (float32)              (onset/nucleus/coda)
  <1 sec                <5 sec                 <30 sec
```

### Phase 1: Feature Extraction

**Purpose**: Extract universal phonological features from Phoible database

**Input**: Phoible database CSV (2,716 languages, 38 distinctive features)
**Output**: 38-dim ternary feature vectors (+, -, 0) for 39 English phonemes
**Format**: CSV file
**Location**: `embeddings/phase1/phoible_features.csv`
**Script**: `scripts/compute_phase1_features.py`
**Time**: <1 second

**Features**: 38 universal phonological features from Hayes (2009) + Moisik & Esling (2011):
- Consonant: syllabic, consonantal, sonorant, continuant, delayedRelease, approximant, tap, trill, nasal, voice, spreadGlottis, constrictedGlottis, labial, round, labiodental, coronal, anterior, distributed, strident, lateral, dorsal, high, low, front, back, tense
- Laryngeal: spreadGlottis, constrictedGlottis, voice
- Place: labial, labiodental, coronal, dorsal
- Manner: nasal, lateral, continuant, delayedRelease, strident
- Vowel: syllabic, high, low, back, front, tense, round

### Phase 2: Normalization

**Purpose**: Convert ternary features to continuous normalized vectors

**Input**: Phase 1 ternary features (38-dim)
**Output**: Normalized continuous vectors (76-dim)
**Format**: Pickle file (dict: IPA → numpy array)
**Location**: `embeddings/phase2/normalized_76d.pkl`
**Script**: `scripts/compute_phase2_normalized_vectors.py`
**Time**: <5 seconds

**Transformation**:
- Ternary (+, -, 0) → Continuous (+1.0, -1.0, 0.0)
- Each feature gets 2 dimensions: [start, end]
- 38 features × 2 = 76 dimensions
- Normalized to unit length per phoneme

**Why 76-dim?**
- Supports diphthong modeling (vowel trajectories)
- [start, end] captures articulation dynamics
- Can interpolate intermediate positions

### Phase 3: Syllable Aggregation

**Purpose**: Build syllable embeddings with onset-nucleus-coda structure

**Input**: Phase 2 normalized phoneme vectors (76-dim)
**Output**: Syllable embeddings (228-dim per syllable)
**Format**: PyTorch checkpoint
**Location**: `embeddings/phase3/syllable_embeddings_phoible.pt`
**Script**: `scripts/build_phase3_syllable_embeddings.py`
**Time**: <30 seconds

**Structure**: onset(76) + nucleus(76) + coda(76) = 228 dims

**Algorithm**:
1. Syllabify word using English phonotactic constraints
2. For each syllable:
   - **Onset**: Average consonant cluster embeddings → normalize to unit length
   - **Nucleus**: Single vowel embedding → normalize to unit length
   - **Coda**: Average consonant cluster embeddings → normalize to unit length
3. Concatenate [onset | nucleus | coda]
4. Store as list of syllables per word

**Component Normalization**: Each component individually normalized ensures equal weighting by default. Users can adjust weights at query time.

## Pipeline Commands

### Full Rebuild

Run all three phases in sequence:

```bash
# Phase 1: Extract Phoible features
python scripts/compute_phase1_features.py

# Phase 2: Normalize features
python scripts/compute_phase2_normalized_vectors.py

# Phase 3: Build syllable embeddings
python scripts/build_phase3_syllable_embeddings.py

# Export for web app
python scripts/export_clientside_data.py
```

**Total time**: ~45 seconds

### Quick Rebuild (Skip Phase 1 & 2)

If you only modified syllabification or filtering:

```bash
# Just rebuild Phase 3
python scripts/build_phase3_syllable_embeddings.py

# Re-export
python scripts/export_clientside_data.py
```

**Time**: ~40 seconds

## File Sizes

| Phase | Uncompressed | Compressed (gz) | Coverage |
|-------|--------------|-----------------|----------|
| Phase 1 | 59 KB | 18 KB | 94 phonemes |
| Phase 2 | 174 KB | 52 KB | 94 phonemes |
| Phase 3 | 111.6 MB | 1.5 MB | 17,920 words |

**Key Insight**: Phoible features compress 99% due to sparse structure!

## Advantages of Phase Architecture

### 1. No Training Required
- **Old**: 10 minutes to train transformer (Layer 3)
- **New**: 0 seconds - purely deterministic transformations

### 2. Instant Deployment
- Run 3 scripts → export → deploy
- No GPU, no training data, no hyperparameters

### 3. Interpretable
- Each phase has clear linguistic meaning
- Feature-based, not learned
- Easy to debug and explain

### 4. Extensible
- Add new languages by adding Phoible entries
- Modify syllabification rules easily
- Adjust component weights at runtime

### 5. Reproducible
- Same input always produces same output
- No random initialization
- No training variance

## Comparison: Phases vs Layers

### Old Architecture (Layers 1-4)

```
Layer 1 (Extract)  → Layer 2 (Normalize) → Layer 3 (TRAIN 10min) → Layer 4 (Aggregate)
   Phoible             Continuous              Transformer           Syllables
   38-dim              76-dim                  128-dim               384-dim
   <1 sec              <5 sec                  10 minutes            <30 sec
```

**Problem**: Layer 3 training created confusion - was it part of the deterministic pipeline or a separate ML step?

### New Architecture (Phases 1-3)

```
Phase 1 (Extract)  → Phase 2 (Normalize) → Phase 3 (Aggregate)
   Phoible             Continuous              Syllables
   38-dim              76-dim                  228-dim
   <1 sec              <5 sec                  <30 sec
```

**Solution**: All phases are deterministic transformations. No training, no confusion!

## What Happened to Layer 3?

**Deprecated**: The transformer training step has been removed entirely.

**Why?**
- 10-minute training for marginal improvement
- Black box (not interpretable)
- 16x larger files (23.8 MB vs 1.5 MB)
- Not adjustable by users

**Archived**: Old Layer 3 training scripts moved to `archive/deprecated_layer_scripts/`

## Technical Details

### Syllabification Algorithm

Based on English phonotactic constraints (Hayes 2009):

1. **Onset Maximization**: Assign consonants to onset where legal
2. **Sonority Sequencing**: Respect sonority hierarchy
3. **Special Rules**:
   - /ŋ/ only allowed in coda
   - /h/ only allowed in onset
   - Affricates /tʃ/, /dʒ/ treated as single units

**Example**: "strength" /stɹɛŋkθ/
- Syllable 1: onset=/stɹ/, nucleus=/ɛ/, coda=/ŋ/
- Syllable 2: onset=/k/, nucleus=∅, coda=/θ/

### Component Normalization Math

```python
# For each syllable component:
raw_vector = average(phoneme_embeddings)  # If multiple phonemes
normalized_vector = raw_vector / ||raw_vector||  # Unit length

# Result: each component has norm 1.0
# Default similarity: (onset_sim + nucleus_sim + coda_sim) / 3
```

### User-Adjustable Weighting

At query time, users can adjust component weights:

```python
# Default (balanced)
weights = { onset: 0.33, nucleus: 0.33, coda: 0.33 }

# Rhymes (ignore onset)
weights = { onset: 0.0, nucleus: 0.5, coda: 0.5 }

# Alliteration (only onset)
weights = { onset: 1.0, nucleus: 0.0, coda: 0.0 }
```

## Directory Structure

```
embeddings/
├── phase1/
│   └── phoible_features.csv          # 38-dim ternary features
├── phase2/
│   ├── normalized_76d.pkl             # 76-dim endpoints
│   └── normalized_152d.pkl            # 152-dim trajectories (optional)
└── phase3/
    └── syllable_embeddings_phoible.pt # 228-dim syllables (17,920 words)

scripts/
├── compute_phase1_features.py         # Extract from Phoible
├── compute_phase2_normalized_vectors.py # Normalize features
├── build_phase3_syllable_embeddings.py # Build syllables
└── export_clientside_data.py          # Export for web app

archive/
└── deprecated_layer_scripts/          # Old training scripts
    ├── train_layer3_contextual_embeddings.py
    ├── train_layer3_mlm_only.py
    ├── build_filtered_layer4_embeddings.py
    └── build_layer4_syllable_embeddings.py
```

## Migration from Layers to Phases

**Breaking Changes**: File and directory names changed

**Backward Compatibility**: None - this is a clean break from the old architecture

**Migration**: Already complete in v2.2.1-beta

**Old Paths** → **New Paths**:
- `embeddings/layer1/` → `embeddings/phase1/`
- `embeddings/layer2/` → `embeddings/phase2/`
- `embeddings/layer4/` → `embeddings/phase3/`
- `scripts/compute_layer1_*` → `scripts/compute_phase1_*`
- `scripts/compute_layer2_*` → `scripts/compute_phase2_*`
- `scripts/build_*_layer4_*` → `scripts/build_phase3_*`

## FAQs

**Q: Why phases instead of layers?**
A: "Layers" implies ML model layers (like neural network layers). "Phases" better describes deterministic processing stages.

**Q: Can I skip Phase 1 or Phase 2?**
A: Yes! Phase 1 and 2 outputs are checked into the repo. You only need to rebuild if modifying Phoible features.

**Q: What if I want to train a model?**
A: The old Layer 3 scripts are archived if needed. But the Phoible-only approach is recommended for production.

**Q: How do I add a new language?**
A: Add phonemes to Phoible database → rerun Phase 1 → Phase 2 → Phase 3 with new language's CMU-equivalent dict.

**Q: Can I use Phase 2 embeddings directly?**
A: Yes! Phase 2 gives you 76-dim phoneme embeddings suitable for phoneme-level analysis.

## References

- **Phoible**: Moran & McCloy (2019). PHOIBLE 2.0. https://phoible.org/
- **Phonotactics**: Hayes (2009). *Introductory Phonology*
- **Features**: Hayes (2009) + Moisik & Esling (2011)
- **Syllabification**: Based on sonority sequencing principle

---

**Version**: 2.2.1-beta
**Status**: Production
**Last Updated**: 2025-01-06
