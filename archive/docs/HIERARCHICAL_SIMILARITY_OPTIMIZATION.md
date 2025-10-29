# Hierarchical Syllable Similarity: The REAL Bottleneck

**Date:** 2025-10-27
**Analysis:** Syllable frequency in PhonoLex (125K words, 308K syllables)

---

## The Problem

Hierarchical similarity is **50x slower** than phoneme-level Levenshtein:
- Phoneme edit distance: **500K ops/sec** (2 μs per call)
- Syllable similarity: **10K ops/sec** (100 μs per call)

**Why?** The cosine distance computation dominates:
- 3 syllables × 3 syllables = 9 cosine distances
- Each cosine: ~17 μs (scipy) or ~2.4 μs (manual)
- Total: 9 × 2.4 μs = **22 μs** just for similarity!
- DP algorithm: <1 μs

---

## Syllable Frequency Analysis

### Key Findings

1. **Total syllables:** 308,244 tokens
2. **Unique syllables:** 22,993 types
3. **Type/Token ratio:** 7.5% (very low → high repetition!)
4. **Distribution:** Follows Zipf's law (R² = 0.971)

### Most Frequent Syllables

| Rank | Syllable | Count | Coverage |
|------|----------|-------|----------|
| 1 | l\|i\|- | 3,297 | 1.07% |
| 2 | -\|ʌ\|- | 3,131 | 1.02% |
| 3 | ɹ\|i\|- | 2,754 | 0.89% |
| 4 | l\|ʌ\|- | 2,314 | 0.75% |
| 5 | t\|ɝ\|- | 2,306 | 0.75% |
| ... | ... | ... | ... |

### Coverage by Cache Size

| Cache Size | Token Coverage | Memory | # Similarities |
|------------|---------------|--------|----------------|
| **50** | **21%** | **10 KB** | **2,500** |
| **100** | **30%** | **39 KB** | **10,000** |
| **200** | **41%** | **156 KB** | **40,000** |
| **500** | **57%** | **1.0 MB** | **250,000** |
| 1000 | 68% | 3.9 MB | 1,000,000 |

---

## Optimization Strategy

### 1. Pre-Normalize ALL Syllable Embeddings

**Impact: 60x speedup** on cosine similarity

```python
def get_word_embedding(self, word: str) -> Optional[List[np.ndarray]]:
    """
    Return NORMALIZED syllable embeddings.

    Normalize ONCE when creating, then use dot product (not cosine).
    """
    # ... get contextual phoneme embeddings ...
    # ... aggregate into syllables ...

    syllable_embeddings = []
    for syl_emb in raw_syllable_embeddings:
        # CRITICAL: Normalize here!
        syl_emb_norm = syl_emb / np.linalg.norm(syl_emb)
        syllable_embeddings.append(syl_emb_norm)

    return syllable_embeddings


def hierarchical_similarity_fast(syllables1_norm, syllables2_norm):
    """
    Syllables are pre-normalized → use dot product instead of cosine.

    60x faster: 0.3 μs vs 17 μs per similarity
    """
    # Precompute similarity matrix with dot product
    sim_matrix = np.zeros((len(syllables1_norm), len(syllables2_norm)))
    for i in range(len(syllables1_norm)):
        for j in range(len(syllables2_norm)):
            sim_matrix[i, j] = np.dot(syllables1_norm[i], syllables2_norm[j])

    # ... DP with precomputed similarities ...
```

**Before:** `cosine_dist(s1, s2)` → 17 μs
**After:** `np.dot(s1_norm, s2_norm)` → 0.3 μs
**Speedup:** 60x

---

### 2. Cache Top 200 Frequent Syllables

**Impact: 41% of comparisons are instant lookups**

```python
class SyllableCache:
    """
    Pre-compute similarities for top-N frequent syllables.

    Strategy:
    - Top 200 syllables cover 41% of tokens
    - 200×200 = 40K precomputed similarities
    - Memory: 156 KB (negligible)
    - Lookup: O(1)
    """

    def __init__(self, top_n=200):
        self.top_n = top_n
        self.freq_syllables = []  # List of (syllable_emb, frequency)
        self.similarity_matrix = None  # Precomputed [N×N]
        self.syllable_to_idx = {}  # {syllable_hash: index}

    def build_from_lexicon(self, graph):
        """
        Build cache from graph word embeddings.

        1. Count syllable frequencies
        2. Identify top N
        3. Precompute N×N similarity matrix
        """
        from collections import Counter

        syllable_counts = Counter()
        syllable_map = {}  # hash → embedding

        # Count frequencies
        for word in graph.graph.nodes():
            if word in graph.word_embeddings:
                for syl_emb in graph.word_embeddings[word]:
                    # Hash by first few values
                    syl_hash = tuple(syl_emb[:4].round(3))
                    syllable_counts[syl_hash] += 1
                    if syl_hash not in syllable_map:
                        syllable_map[syl_hash] = syl_emb

        # Get top N
        top_hashes = [h for h, _ in syllable_counts.most_common(self.top_n)]

        # Build index
        self.freq_syllables = []
        for idx, syl_hash in enumerate(top_hashes):
            self.freq_syllables.append(syllable_map[syl_hash])
            self.syllable_to_idx[syl_hash] = idx

        # Precompute similarity matrix
        n = len(self.freq_syllables)
        self.similarity_matrix = np.zeros((n, n), dtype=np.float32)

        for i in range(n):
            for j in range(i, n):
                # Fast dot product (already normalized!)
                sim = np.dot(self.freq_syllables[i], self.freq_syllables[j])
                self.similarity_matrix[i, j] = sim
                self.similarity_matrix[j, i] = sim

        print(f"✓ Cached {n} syllables ({syllable_counts[top_hashes[0]]} to {syllable_counts[top_hashes[-1]]} occurrences)")
        print(f"✓ Precomputed {n*n:,} similarities ({self.similarity_matrix.nbytes/1024:.1f} KB)")

    def get_similarity(self, syl1_emb, syl2_emb):
        """Get similarity (from cache or compute)."""
        hash1 = tuple(syl1_emb[:4].round(3))
        hash2 = tuple(syl2_emb[:4].round(3))

        idx1 = self.syllable_to_idx.get(hash1)
        idx2 = self.syllable_to_idx.get(hash2)

        if idx1 is not None and idx2 is not None:
            # Cache hit! O(1) lookup
            return self.similarity_matrix[idx1, idx2], True

        # Cache miss - compute (still fast with pre-normalized!)
        return np.dot(syl1_emb, syl2_emb), False


# Usage in hierarchical_similarity:
def hierarchical_similarity_optimized(syllables1, syllables2, cache=None):
    """
    Optimized with cache + pre-normalization.
    """
    # Precompute similarity matrix (from cache where possible)
    sim_matrix = np.zeros((len(syllables1), len(syllables2)))
    cache_hits = 0

    for i in range(len(syllables1)):
        for j in range(len(syllables2)):
            if cache:
                sim, was_cached = cache.get_similarity(syllables1[i], syllables2[j])
                if was_cached:
                    cache_hits += 1
            else:
                sim = np.dot(syllables1[i], syllables2[j])  # Pre-normalized!

            sim_matrix[i, j] = sim

    # ... DP with precomputed similarities ...
```

---

### 3. Pre-Compute Identities

**Impact: Free 1.0 similarity for same syllables**

```python
def hierarchical_similarity_optimized(syllables1, syllables2, cache=None):
    """Add identity check."""

    for i in range(len(syllables1)):
        for j in range(len(syllables2)):
            # Check if same syllable (by reference or hash)
            if np.array_equal(syllables1[i], syllables2[j]):
                sim_matrix[i, j] = 1.0
                continue

            # Otherwise, lookup/compute
            if cache:
                sim, _ = cache.get_similarity(syllables1[i], syllables2[j])
            else:
                sim = np.dot(syllables1[i], syllables2[j])

            sim_matrix[i, j] = sim
```

---

## Combined Speedup Analysis

### Baseline (Current)
- Cosine with scipy: 17 μs × 9 comparisons = **153 μs**
- DP algorithm: 1 μs
- **Total: 154 μs per pair**

### Optimization 1: Manual Cosine
- Manual cosine: 2.4 μs × 9 = **22 μs**
- **Speedup: 7x**

### Optimization 2: Pre-Normalized
- Dot product: 0.3 μs × 9 = **3 μs**
- **Speedup: 60x from baseline, 8x from manual**

### Optimization 3: Cache (200 syllables, 41% coverage)
- Cached lookups: 0 μs × 3.7 comparisons (41% of 9)
- Dot products: 0.3 μs × 5.3 comparisons (59% of 9) = 1.6 μs
- **Total: 1.6 μs** (vs 153 μs baseline)
- **Speedup: 96x**

### Optimization 4: Identity Check
For comparing word to itself (common during graph building):
- All 9 comparisons are identities → 0 μs
- **Speedup: ∞**

---

## Expected Performance

### Before Optimization
```
Hierarchical similarity: 10,000 ops/sec (100 μs/op)
```

### After Optimization (Pre-normalized + Cache)
```
Hierarchical similarity: 500,000 ops/sec (2 μs/op)

Breakdown:
- Cached (41%): ~0 μs
- Dot product (59%): 1.6 μs
- Identity check: 0 μs
- DP: 1 μs
Total: ~2-3 μs
```

**Overall speedup: 50x**

---

## Implementation Checklist

### Phase 1: Critical (1 day)
- [ ] Fix syllable aggregation bug in `get_word_embedding()`
- [ ] Add normalization when creating syllable embeddings
- [ ] Replace `scipy.cosine_dist` with `np.dot` (pre-normalized)

### Phase 2: Caching (1 day)
- [ ] Implement `SyllableCache` class
- [ ] Build cache from top 200 syllables at graph initialization
- [ ] Integrate cache into `hierarchical_similarity()`

### Phase 3: Polish (1 day)
- [ ] Add identity checks
- [ ] Profile and verify speedup
- [ ] Add cache hit/miss statistics

---

## Memory Analysis

### Syllable Embeddings
- 22,993 unique syllables
- 384 dimensions per syllable (3 × 128)
- 384 × 4 bytes (float32) = 1.5 KB per syllable
- **Total: 35 MB** (all unique syllables, normalized)

### Similarity Cache
- Top 200 syllables
- 200 × 200 similarities
- 4 bytes (float32) per similarity
- **Total: 156 KB**

### Trade-off
- **Cost:** 156 KB memory
- **Benefit:** 41% of comparisons are instant (96x faster)
- **Worth it?** Absolutely!

---

## Conclusion

The hierarchical similarity bottleneck can be solved with 3 simple changes:

1. **Pre-normalize syllables** → 60x faster cosine
2. **Cache top 200** → 41% coverage, 156 KB
3. **Check identities** → Free for self-comparisons

**Expected total speedup: 50-100x**

This brings hierarchical similarity from 100 μs/op to 2 μs/op, making it competitive with phoneme-level Levenshtein!

---

## Bonus: Vectorized Similarity Matrix

For even more speed, compute all pairwise similarities at once:

```python
def compute_similarity_matrix_vectorized(syllables1_norm, syllables2_norm):
    """
    Vectorized dot product: compute all similarities at once.

    syllables1_norm: [n1, 384] array
    syllables2_norm: [n2, 384] array

    Returns: [n1, n2] similarity matrix
    """
    syll1_matrix = np.array(syllables1_norm)  # [n1, 384]
    syll2_matrix = np.array(syllables2_norm)  # [n2, 384]

    # Matrix multiplication: [n1, 384] @ [384, n2] = [n1, n2]
    sim_matrix = syll1_matrix @ syll2_matrix.T

    return sim_matrix
```

This uses optimized BLAS routines and can be **10x faster** than looping!

**Final speedup: 500-1000x** 🚀
