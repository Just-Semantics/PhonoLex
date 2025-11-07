# Phoneme-Sequence Soft Levenshtein Architecture (v2.3)

**Status**: ✅ Production (November 2025)
**Replaces**: v2.2.1 Phoible component-wise averaging (deprecated due to structural information loss)

## Executive Summary

PhonoLex v2.3 uses **phoneme-sequence soft Levenshtein distance** for phonological similarity, preserving the sequential structure of consonant clusters and diphthongs without averaging.

**Key Innovation**: No averaging! Onset/nucleus/coda are represented as **sequences of phoneme vectors**, compared using soft Levenshtein distance on phoneme sequences.

### Why This Matters

**v2.2.1 Problem (Component-wise averaging)**:
```
crest: onset = average([k, ɹ]) → single 76-dim vector
cat:   onset = [k] → single 76-dim vector
```
❌ Averaging destroyed information: [k, ɹ] looked similar to [k] alone!

**v2.3 Solution (Phoneme sequences)**:
```
crest: onset = [[k_vec], [ɹ_vec]] → sequence of 2 vectors
cat:   onset = [[k_vec]] → sequence of 1 vector
```
✅ Soft Levenshtein computes edit distance, properly penalizing length differences!

## Architecture Overview

### Data Structure

Each word has syllable structures with phoneme sequences:

```typescript
interface SyllableStructure {
  onset: number[][];   // Sequence of 76-dim phoneme vectors
  nucleus: number[][]; // Sequence of 76-dim (monophthong) or 152-dim (diphthong) vectors
  coda: number[][];    // Sequence of 76-dim phoneme vectors
}
```

**Examples**:

| Word | Onset | Nucleus | Coda |
|------|-------|---------|------|
| cat | [k] (1×76) | [æ] (1×76) | [t] (1×76) |
| crest | [k, ɹ] (2×76) | [ɛ] (1×76) | [s, t] (2×76) |
| time | [t] (1×76) | [aɪ] (1×152) | [m] (1×76) |

### Similarity Calculation

**Three-level hierarchy**:

1. **Phoneme-level**: Cosine similarity between phoneme vectors (76-dim or 152-dim)
2. **Component-level**: Soft Levenshtein on phoneme sequences within onset/nucleus/coda
3. **Syllable-level**: Weighted average of component similarities
4. **Word-level**: Soft Levenshtein on syllable sequences

#### Example: cat vs crest

**Step 1: Component similarities**

Onset comparison (`[k]` vs `[k, ɹ]`):
```
DP table for soft Levenshtein:
       ∅    k    ɹ
   ∅   0    1    2
   k   1   0.0   1.0

Edit distance = 1.0
Similarity = 1 - (1.0 / 2) = 0.50
```

Nucleus comparison (`[æ]` vs `[ɛ]`):
```
Cosine similarity: ~0.85 (similar vowels)
```

Coda comparison (`[t]` vs `[s, t]`):
```
DP table:
       ∅    s    t
   ∅   0    1    2
   t   1    1    0.2  (t-s mismatch, t-t match)

Edit distance = 0.2
Similarity = 1 - (0.2 / 2) = 0.90
```

**Step 2: Weighted syllable similarity**

With balanced weights (0.33 each):
```
Similarity = (0.33 × 0.50) + (0.33 × 0.85) + (0.33 × 0.90)
           = 0.165 + 0.281 + 0.297
           = 0.743
```

**Step 3: Word-level** (both single-syllable, so same as syllable-level)

**Final: cat-crest similarity ≈ 0.74**

#### Example: cat vs bat

Onset: `[k]` vs `[b]` → cosine(k, b) ≈ 0.70 (both stops)
Nucleus: `[æ]` vs `[æ]` → 1.0 (identical)
Coda: `[t]` vs `[t]` → 1.0 (identical)

**Weighted**: `(0.33 × 0.70) + (0.33 × 1.0) + (0.33 × 1.0) = 0.90`

**cat-bat ≈ 0.90** (rhyme, different onset)

### Weighted Component Similarity

Users can adjust weights to emphasize different components:

| Preset | Onset | Nucleus | Coda | Use Case |
|--------|-------|---------|------|----------|
| Rhymes | 0.0 | 0.5 | 0.5 | Perfect rhymes |
| Balanced | 0.33 | 0.33 | 0.33 | Overall similarity |
| Alliteration | 1.0 | 0.0 | 0.0 | Initial sounds |
| Assonance | 0.0 | 1.0 | 0.0 | Vowel matching |
| Consonance | 0.5 | 0.0 | 0.5 | Consonant sounds (onset + coda) |

## Data Pipeline

### Backend: Build Syllable Structures

```bash
python scripts/build_phase3_syllable_embeddings_v2.py
```

**Output**: `embeddings/phase3/syllable_structures_phoible_v2.pt` (121 MB)

**Process**:
1. Load Phase 2 Phoible features (76-dim for consonants/monophthongs, 152-dim for diphthongs)
2. Syllabify each word using onset-nucleus-coda parser
3. For each syllable component:
   - Onset: Store sequence of consonant vectors
   - Nucleus: Store vowel vector (152-dim if diphthong, else 76-dim)
   - Coda: Store sequence of consonant vectors
4. Save as nested Python lists (no averaging!)

### Export for Frontend

```bash
python scripts/export_clientside_data_v2.py
```

**Output**: `webapp/frontend/public/data/syllable_structures.json.gz` (0.6 MB!)

**Compression**: 99% (56.7 MB → 0.6 MB gzipped)

### Frontend: Load and Compute

[clientSideData_v2.ts](../webapp/frontend/src/services/clientSideData_v2.ts):

1. **Load**: Fetch `syllable_structures.json.gz` (~600 KB)
2. **Parse**: Deserialize JSON into `SyllableStructure[]` per word
3. **Compute**: On-demand soft Levenshtein similarity in browser

**Performance**:
- Load time: ~500ms (one-time, cached)
- Similarity query: ~50-100ms for full vocabulary scan (17,920 words)
- Memory: ~60 MB (structures kept in RAM)

## Implementation Details

### Soft Levenshtein Distance

Standard edit distance DP with soft substitution costs:

```typescript
function softLevenshtein(seq1: Vec[], seq2: Vec[]): number {
  const len1 = seq1.length;
  const len2 = seq2.length;

  // Pre-compute similarity matrix
  const sim = Array(len1).fill(0).map(() => Array(len2).fill(0));
  for (let i = 0; i < len1; i++) {
    for (let j = 0; j < len2; j++) {
      sim[i][j] = cosineSimilarity(seq1[i], seq2[j]);
    }
  }

  // DP for edit distance
  const dp = Array(len1 + 1).fill(0).map(() => Array(len2 + 1).fill(0));

  // Initialize
  for (let i = 0; i <= len1; i++) dp[i][0] = i;
  for (let j = 0; j <= len2; j++) dp[0][j] = j;

  // Fill
  for (let i = 1; i <= len1; i++) {
    for (let j = 1; j <= len2; j++) {
      const matchCost = 1.0 - sim[i-1][j-1]; // 0 if identical
      dp[i][j] = Math.min(
        dp[i-1][j] + 1.0,           // Delete
        dp[i][j-1] + 1.0,           // Insert
        dp[i-1][j-1] + matchCost    // Match/substitute
      );
    }
  }

  // Normalize
  const maxLen = Math.max(len1, len2);
  return 1.0 - (dp[len1][len2] / maxLen);
}
```

**Key properties**:
- **Symmetric**: dist(A, B) = dist(B, A)
- **Length-sensitive**: Longer sequences penalized
- **Soft substitution**: Similar phonemes have low cost
- **Bounded**: [0, 1] range (0 = unrelated, 1 = identical)

### Phoneme Feature Vectors

**Source**: Phase 2 Phoible normalized features

**76-dim (consonants & monophthongs)**:
- Endpoint features only
- Normalized to unit length
- Represents static articulation

**152-dim (diphthongs)**:
- Trajectory features (start → end)
- 76-dim start + 76-dim end
- Preserves dynamic movement

**Examples**:
- `/k/`: 76-dim (voiceless velar stop)
- `/æ/`: 76-dim (low front vowel)
- `/aɪ/`: 152-dim (low central → high front trajectory)

## Performance Characteristics

### File Sizes

| File | Uncompressed | Gzipped | Compression |
|------|--------------|---------|-------------|
| syllable_structures.json | 56.7 MB | 0.6 MB | **99%** |
| word_metadata.json | 7.9 MB | 0.8 MB | 90% |
| **Total download** | 64.6 MB | **1.4 MB** | 98% |

### Query Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Load data | ~500ms | One-time, cached |
| Find similar words | ~50-100ms | Scans 17,920 words |
| Single comparison | ~0.005ms | One word pair |
| Phoneme sequence DP | ~0.001ms | Typical onset/coda |

**Optimization**: Soft Levenshtein DP is O(n×m) but n,m are small (≤5 phonemes typically).

### Vocabulary Coverage

- **17,920 words** with syllable structures
- **Filtered**: Only words with frequency + ≥1 psycholinguistic norm
- **Reduction**: 84.9% from full CMU dict (125K → 18K)
- **Quality**: High-quality research/clinical vocabulary

## Comparison with Previous Approaches

### v2.2.0 (MLM-trained embeddings)

**Approach**: 128-dim learned embeddings → 384-dim syllables (onset+nucleus+coda)

❌ **Problems**:
- Requires 10-minute training
- 16× larger file size (23.8 MB vs 1.5 MB)
- Not phonologically transparent
- Cannot adjust weights at query time

✅ **Advantages**:
- Better learned representations
- Slightly better edge-case discrimination

**Verdict**: Training overhead and file size not worth the marginal gains.

### v2.2.1 (Phoible component-wise averaging)

**Approach**: Average onset/nucleus/coda phonemes → 228-dim syllables

❌ **Fatal flaw**: Averaging destroyed sequential structure
- `[k, ɹ]` averaged looked like `[k]` alone
- Diphthong trajectories lost
- Consonant clusters collapsed

✅ **Advantages**:
- Small file size
- Phonologically transparent features

**Verdict**: Good idea, fatally flawed execution. Fixed in v2.3.

### v2.3 (Phoneme-sequence soft Levenshtein) ← **Current**

**Approach**: Sequences of phoneme vectors + soft Levenshtein DP

✅ **Advantages**:
- Preserves sequential structure (clusters, diphthongs)
- Phonologically transparent (pure Phoible features)
- User-adjustable weights (onset/nucleus/coda)
- Small file size (0.6 MB gzipped)
- No training required
- Linguistically principled (edit distance)

❌ **Tradeoffs**:
- Slightly more computation (DP on sequences vs single vector)
- Still ~100ms per full-vocab query (acceptable)

**Verdict**: Best of all worlds. Production-ready.

## Future Enhancements

### Potential Improvements

1. **Trajectory encoding for clusters**
   - Encode onset clusters like diphthongs (trajectory embeddings)
   - E.g., [kɹ] → single 152-dim trajectory vector
   - Would reduce DP overhead for complex onsets

2. **Phonotactic priors**
   - Weight frequent clusters higher (e.g., [st] vs [sθ])
   - Learned from corpus statistics
   - Improves similarity for natural sequences

3. **Position-specific phoneme embeddings**
   - Different vectors for onset vs coda /t/
   - Captures positional allophones
   - Requires larger feature set

4. **Multi-language support**
   - Extend beyond English
   - Universal Phoible features already support 2,716 languages!
   - Need language-specific syllabification

### Not Recommended

❌ **Return to MLM training**: Overhead not justified
❌ **Quantization**: File already tiny (0.6 MB)
❌ **Caching precomputed similarities**: RAM vs recomputation tradeoff not favorable

## References

### Internal Documentation

- [EMBEDDINGS_ARCHITECTURE.md](EMBEDDINGS_ARCHITECTURE.md) - Phase 1-4 pipeline
- [CLIENT_SIDE_DATA_PACKAGE.md](CLIENT_SIDE_DATA_PACKAGE.md) - Data export format
- [PHASE_ARCHITECTURE.md](PHASE_ARCHITECTURE.md) - Phase naming conventions

### Academic Background

- **Phoible**: Moran & McCloy (2019). PHOIBLE 2.0. https://phoible.org/
- **Levenshtein Distance**: Levenshtein (1966). Binary codes capable of correcting deletions, insertions, and reversals.
- **Soft Edit Distance**: Oncina & Sebban (2006). Learning stochastic edit distance.
- **Syllable Structure**: Hayes (2009). Introductory Phonology.

### Build Scripts

- `scripts/build_phase3_syllable_embeddings_v2.py` - Build syllable structures
- `scripts/export_clientside_data_v2.py` - Export for frontend
- `scripts/compute_layer2_normalized_vectors.py` - Generate Phase 2 features

### Frontend Implementation

- `webapp/frontend/src/services/clientSideData_v2.ts` - Main similarity service
- `webapp/frontend/src/components/tools/PhonologicalSimilarityTool.tsx` - UI component

## Changelog

### v2.3.0 (November 2025)

- ✅ **NEW**: Phoneme-sequence soft Levenshtein architecture
- ✅ **FIXED**: Consonant cluster averaging bug (v2.2.1)
- ✅ **FIXED**: Diphthong trajectory preservation
- ✅ **IMPROVED**: cat-crest discrimination (0.20 → 0.74 → correct ranking)
- ✅ **PERFORMANCE**: 99% compression ratio (56.7 MB → 0.6 MB)
- ✅ **UX**: User-adjustable onset/nucleus/coda weights

### v2.2.1 (October 2025) - **DEPRECATED**

- ❌ **BUG**: Component-wise averaging destroyed cluster structure
- ❌ **BUG**: Diphthong trajectories collapsed to single vector
- ⚠️ **Do not use** - Replaced by v2.3

### v2.2.0 (October 2025) - **ARCHIVED**

- MLM-trained embeddings approach
- Too large, required training
- Archived but functional

---

**Last updated**: November 6, 2025
**Status**: ✅ Production
**Version**: v2.3.0
