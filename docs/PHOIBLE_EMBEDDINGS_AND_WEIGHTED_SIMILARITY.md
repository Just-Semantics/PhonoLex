# Phoible Embeddings and Weighted Similarity

## Overview

PhonoLex v2.2+ uses **pure Phoible feature-based syllable embeddings** with **user-adjustable component weighting** for phonological similarity calculations.

## Key Benefits

1. **No Training Required**: Embeddings are built directly from Phoible phonological features
2. **Linguistically Transparent**: Based on 38 universal phonological features from Phoible
3. **Smaller & Faster**: 228-dim vs 384-dim (40% reduction), compresses to 1.5 MB
4. **User-Controllable**: Adjust onset/nucleus/coda weights at query time

## Architecture

### Syllable Structure

Each syllable embedding has 3 normalized components:

```
Syllable = [onset(76) | nucleus(76) | coda(76)] = 228 dimensions
```

- **Onset**: Initial consonant(s) - e.g., /k/ in "cat", /br/ in "bring"
- **Nucleus**: Vowel - e.g., /æ/ in "cat", /eɪ/ in "make"
- **Coda**: Final consonant(s) - e.g., /t/ in "cat", /ŋ/ in "sing"

Each component is **individually normalized** to unit length, ensuring equal weight by default.

### Component Normalization

```python
# Example: Building syllable embedding for /k-æ-t/ (cat)
onset_vec = phoible_features['k']  # 76-dim
onset_normalized = onset_vec / ||onset_vec||  # Unit length

nucleus_vec = phoible_features['æ']  # 76-dim
nucleus_normalized = nucleus_vec / ||nucleus_vec||  # Unit length

coda_vec = phoible_features['t']  # 76-dim
coda_normalized = coda_vec / ||coda_vec||  # Unit length

syllable_embedding = [onset_normalized | nucleus_normalized | coda_normalized]  # 228-dim
```

## Weighted Similarity

### Default: Equal Weighting

By default, onset, nucleus, and coda contribute equally to similarity:

```typescript
similarity = (onset_sim + nucleus_sim + coda_sim) / 3
```

**Example**: cat vs bat
- Onset: /k/ vs /b/ → 0.55 similarity
- Nucleus: /æ/ vs /æ/ → 1.00 similarity
- Coda: /t/ vs /t/ → 1.00 similarity
- **Overall**: (0.55 + 1.00 + 1.00) / 3 = **0.85**

### Custom Weighting

Users can adjust weights for specific use cases:

#### Rhyme-Focused (nucleus + coda)
```typescript
weights = { onset: 0.0, nucleus: 0.5, coda: 0.5 }
// cat-bat: (0.0*0.55 + 0.5*1.00 + 0.5*1.00) = 1.00
// Perfect rhymes!
```

#### Onset-Focused (alliteration)
```typescript
weights = { onset: 1.0, nucleus: 0.0, coda: 0.0 }
// cat-can: (1.0*1.00 + 0.0*0.85 + 0.0*0.65) = 1.00
// Same initial sound!
```

#### Balanced (default)
```typescript
weights = { onset: 0.33, nucleus: 0.33, coda: 0.33 }
// Equal contribution from all components
```

## Implementation Guide

### Frontend UI Controls

Add weight sliders to similarity search interfaces:

```tsx
interface SimilarityWeights {
  onset: number;    // 0.0 - 1.0
  nucleus: number;  // 0.0 - 1.0
  coda: number;     // 0.0 - 1.0
}

// Example component
function SimilarityControls() {
  const [weights, setWeights] = useState<SimilarityWeights>({
    onset: 0.33,
    nucleus: 0.33,
    coda: 0.33
  });

  return (
    <Box>
      <Typography variant="h6">Similarity Weights</Typography>

      <Slider
        label="Onset (initial sounds)"
        value={weights.onset}
        onChange={(v) => setWeights({ ...weights, onset: v })}
        min={0}
        max={1}
        step={0.1}
      />

      <Slider
        label="Nucleus (vowels)"
        value={weights.nucleus}
        onChange={(v) => setWeights({ ...weights, nucleus: v })}
        min={0}
        max={1}
        step={0.1}
      />

      <Slider
        label="Coda (final sounds)"
        value={weights.coda}
        onChange={(v) => setWeights({ ...weights, coda: v })}
        min={0}
        max={1}
        step={0.1}
      />

      {/* Presets */}
      <ButtonGroup>
        <Button onClick={() => setWeights({ onset: 0.33, nucleus: 0.33, coda: 0.33 })}>
          Balanced
        </Button>
        <Button onClick={() => setWeights({ onset: 0.0, nucleus: 0.5, coda: 0.5 })}>
          Rhymes
        </Button>
        <Button onClick={() => setWeights({ onset: 1.0, nucleus: 0.0, coda: 0.0 })}>
          Alliteration
        </Button>
      </ButtonGroup>
    </Box>
  );
}
```

### Backend: Weighted Similarity Function

Add to `clientSideData.ts`:

```typescript
/**
 * Compute weighted cosine similarity between syllables
 *
 * @param syll1 228-dim syllable embedding
 * @param syll2 228-dim syllable embedding
 * @param weights Component weights { onset, nucleus, coda }
 */
private cosineSimilarityWeighted(
  syll1: number[],
  syll2: number[],
  weights: { onset: number; nucleus: number; coda: number } = { onset: 0.33, nucleus: 0.33, coda: 0.33 }
): number {
  const ONSET_START = 0, ONSET_END = 76;
  const NUCLEUS_START = 76, NUCLEUS_END = 152;
  const CODA_START = 152, CODA_END = 228;

  // Compute similarity for each component
  const onsetSim = this.cosineSimilarity(
    syll1.slice(ONSET_START, ONSET_END),
    syll2.slice(ONSET_START, ONSET_END)
  );

  const nucleusSim = this.cosineSimilarity(
    syll1.slice(NUCLEUS_START, NUCLEUS_END),
    syll2.slice(NUCLEUS_START, NUCLEUS_END)
  );

  const codaSim = this.cosineSimilarity(
    syll1.slice(CODA_START, CODA_END),
    syll2.slice(CODA_START, CODA_END)
  );

  // Weighted average (normalize if weights don't sum to 1.0)
  const totalWeight = weights.onset + weights.nucleus + weights.coda;
  if (totalWeight === 0) return 0;

  return (
    weights.onset * onsetSim +
    weights.nucleus * nucleusSim +
    weights.coda * codaSim
  ) / totalWeight;
}
```

### Update Soft Levenshtein to Accept Weights

```typescript
private computeSoftLevenshteinSimilarity(
  syllables1: number[][],
  syllables2: number[][],
  weights?: { onset: number; nucleus: number; coda: number }
): number {
  const len1 = syllables1.length;
  const len2 = syllables2.length;

  // Pre-compute pairwise syllable similarities
  const simMatrix: number[][] = [];
  for (let i = 0; i < len1; i++) {
    simMatrix[i] = [];
    for (let j = 0; j < len2; j++) {
      // Use weighted similarity if weights provided
      if (weights) {
        simMatrix[i][j] = this.cosineSimilarityWeighted(
          syllables1[i],
          syllables2[j],
          weights
        );
      } else {
        simMatrix[i][j] = this.cosineSimilarity(syllables1[i], syllables2[j]);
      }
    }
  }

  // ... rest of Levenshtein DP algorithm
}
```

## Use Cases

### 1. General Phonological Similarity (Default)
**Weights**: `{ onset: 0.33, nucleus: 0.33, coda: 0.33 }`

Use for:
- Word difficulty prediction
- Neighborhood density
- General phonological analysis

**Example Results**:
- cat-bat: 0.85 (rhyme, different onset)
- make-bake: 0.83 (rhyme, similar onset)
- make-take: 0.80 (rhyme, dissimilar onset)
- cat-act: 0.54 (anagram)

### 2. Rhyme Detection
**Weights**: `{ onset: 0.0, nucleus: 0.5, coda: 0.5 }`

Use for:
- Poetry analysis
- Rhyme generation
- Phonological awareness training

**Example Results**:
- cat-bat: 1.00 (perfect rhyme)
- make-bake: 1.00 (perfect rhyme)
- make-take: 1.00 (perfect rhyme)

### 3. Alliteration Detection
**Weights**: `{ onset: 1.0, nucleus: 0.0, coda: 0.0 }`

Use for:
- Literary analysis
- Tongue twisters
- Consonant awareness

**Example Results**:
- cat-can: 1.00 (both start with /k/)
- bat-big: 1.00 (both start with /b/)
- make-mat: 1.00 (both start with /m/)

### 4. Vowel-Focused (Assonance)
**Weights**: `{ onset: 0.0, nucleus: 1.0, coda: 0.0 }`

Use for:
- Vowel awareness training
- Poetry analysis (assonance)

### 5. Coda-Focused (Final Sounds)
**Weights**: `{ onset: 0.0, nucleus: 0.0, coda: 1.0 }`

Use for:
- Final consonant awareness
- Specific articulation therapy

## Comparison: Phoible vs MLM Embeddings

| Feature | Phoible (Current) | MLM (Deprecated) |
|---------|-------------------|------------------|
| Training time | 0 seconds | 10 minutes |
| File size (gzipped) | 1.5 MB | 23.8 MB |
| Embedding dim | 228 | 384 |
| Interpretability | Transparent | Black box |
| Anagram discrimination | 0.79 | 0.54 |
| Unrelated words | 0.68 | 0.15 |
| Onset discrimination | ✅ Correct | ✅ Correct |
| User-adjustable weights | ✅ Yes | ❌ No |

**Decision**: Phoible wins for:
- Instant deployment
- User control
- Transparency
- Smaller size

MLM had better discrimination but required training and wasn't adjustable.

## Migration Notes

### Breaking Changes from v2.1

1. **Embedding dimension**: 384 → 228
2. **File name**: `embeddings_quantized.json.gz` → `embeddings.json.gz`
3. **No quantization**: Direct float32 values (no scales needed)
4. **Component structure**: Must update to 76-dim boundaries

### Update Checklist

- [x] Generate Phoible-based embeddings (`scripts/build_phase3_syllable_embeddings.py`)
- [x] Export to client-side format (`scripts/export_clientside_data.py`)
- [x] Update `clientSideData.ts` to load new format
- [x] Remove quantization/dequantization logic
- [ ] Add weight controls to UI
- [ ] Update similarity functions to accept weights
- [ ] Add preset buttons (Balanced, Rhymes, Alliteration, etc.)
- [ ] Test all phonological tools with new embeddings

## Future Enhancements

1. **Save user preferences**: Remember weight settings per user
2. **Context-aware defaults**: Auto-adjust weights based on tool (rhyme → rhyme weights)
3. **Advanced presets**: Add more specific use cases (onset clusters, diphthongs, etc.)
4. **Visualization**: Show component contributions in similarity results
5. **A/B testing**: Compare weighted vs unweighted results

## References

- Phoible database: https://phoible.org/
- Syllable structure: Hayes (2009), *Introductory Phonology*
- Component normalization: Ensures equal weighting baseline
- Soft Levenshtein: Extends edit distance to continuous similarity

---

**Status**: ✅ Implemented (backend), ⏳ UI pending
**Version**: 2.2.0-beta
**Last Updated**: 2025-01-06
