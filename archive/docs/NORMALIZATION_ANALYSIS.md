# Vector Normalization Analysis: [-1, 1] vs [0, 1]

## Current Encoding (Three-Way: -1, 0, 1)

```python
'+' → 1.0
'-' → -1.0
'0' → 0.0
```

## Alternative: Binary-Style (0, 0.5, 1)

```python
'+' → 1.0
'-' → 0.0
'0' → 0.5
```

## Mathematical Comparison

### Distance Metrics

#### Example: /p/ vs /b/ (differ only in voicing)

**Feature**: `voiced`
- /p/: `voiced: -` (voiceless)
- /b/: `voiced: +` (voiced)

**Three-way encoding (-1, 0, 1):**
```python
# /p/: voiced = -1
# /b/: voiced = +1
euclidean_distance = |1 - (-1)| = 2
squared_distance = (1 - (-1))^2 = 4
```

**Binary-style (0, 0.5, 1):**
```python
# /p/: voiced = 0
# /b/: voiced = 1
euclidean_distance = |1 - 0| = 1
squared_distance = (1 - 0)^2 = 1
```

**Observation:** Three-way encoding gives **2× the distance** for feature oppositions!

---

### Cosine Similarity

**Three-way encoding:**
```python
# Two phonemes differing in 1 feature (out of 38)
v1 = [1, 1, 1, ..., -1, ...]  # 37 matching, 1 opposite
v2 = [1, 1, 1, ..., +1, ...]

cosine_sim = dot(v1, v2) / (||v1|| * ||v2||)
# The -1 vs +1 creates negative contribution to dot product
```

**Binary-style:**
```python
v1 = [1, 1, 1, ..., 0, ...]
v2 = [1, 1, 1, ..., 1, ...]

# The 0 vs 1 creates smaller (but still positive) contribution
```

**Observation:** Three-way can produce negative cosine similarities (phonemes are "opposite"), binary-style stays more positive.

---

## Semantic Interpretation

### Three-Way (-1, 0, 1)

**Linguistic meaning:**
- `+1` and `-1` are **opposites** (voiced vs voiceless, high vs low)
- `0` is **orthogonal/neutral** (feature doesn't apply)
- Distance between `+1` and `-1` is **maximized**

**Example:**
- `high: +1` (high vowel like /i/)
- `high: -1` (not high, like /a/)
- `high: 0` (doesn't apply to consonants)

**Geometric interpretation:** Features live in a signed space where opposites are far apart.

---

### Binary-Style (0, 0.5, 1)

**Linguistic meaning:**
- `1` = feature present
- `0` = feature absent
- `0.5` = feature neutral/NA (arbitrary midpoint)

**Problem:** Is "absent" really the opposite of "present"? Or are they just different states?

**Example:**
- `high: 1` (high vowel)
- `high: 0` (not high)
- `high: 0.5` (NA for consonants)

**Issue:** The `0.5` for NA is **arbitrary**. Why not `0.3` or `0.7`? It has no linguistic meaning.

---

## Vector Magnitude Effects

### Three-Way Encoding

```python
# Vowel with many + features
v = [1, 1, 1, 0, 0, -1, -1, ...]
||v|| = sqrt(1² + 1² + 1² + 0² + 0² + 1² + 1² + ...)
     = sqrt(sum of 1s and -1s)
```

- Magnitude is **symmetric** for `+1` and `-1`
- All phonemes have roughly similar magnitudes (if similar # of features apply)

### Binary-Style

```python
# Same vowel
v = [1, 1, 1, 0.5, 0.5, 0, 0, ...]
||v|| = sqrt(1² + 1² + 1² + 0.5² + 0.5² + 0² + 0² + ...)
     = sqrt(more 1s → larger magnitude)
```

- Phonemes with more `+` features have **larger magnitudes**
- Biases cosine similarity toward phonemes with many `+` features

---

## Practical Considerations

### 1. ChromaDB / Vector Database Compatibility

**ChromaDB uses L2 (Euclidean) distance by default**, but also supports:
- Cosine similarity
- Inner product

**Both encodings work fine**, but:
- Three-way is more **semantically meaningful**
- Binary-style is more **ML-standard**

---

### 2. Machine Learning Models

**Neural networks:**
- Often prefer `[0, 1]` range for **sigmoid/ReLU activations**
- But modern models (transformers) don't care much

**Classical ML (SVM, kNN):**
- Usually **normalize/standardize** anyway
- Three-way already has nice properties (centered at 0)

---

### 3. Interpretability

**Three-way (-1, 0, 1):**
- ✅ Clear linguistic meaning
- ✅ `+` vs `-` are opposites (high vs low, voiced vs voiceless)
- ✅ `0` is truly neutral
- ✅ Symmetric

**Binary-style (0, 0.5, 1):**
- ❌ `0.5` for NA is arbitrary
- ❌ Asymmetric (0 and 1 not symmetric around 0.5 in meaningful way)
- ✅ Standard ML range

---

## Recommendation

### **Keep Three-Way Encoding (-1, 0, 1)** ⭐

**Why:**

1. **Linguistically motivated**: `+` and `-` are true opposites in phonology
2. **Symmetric**: Equal treatment of `+1` and `-1`
3. **Meaningful neutral point**: `0` is orthogonal to `±1`
4. **Better for cosine similarity**: Signed vectors capture opposition
5. **No arbitrary choices**: `0` is the natural neutral point, not `0.5`

### When to normalize to [0, 1]

**Only if:**
- You're feeding into a neural network that expects `[0, 1]` input
- You're using a library that requires positive values
- You want to treat features as "presence/absence" rather than "opposition"

**How to convert later if needed:**
```python
def normalize_to_01(vector):
    """Convert from [-1, 1] to [0, 1]"""
    return (vector + 1) / 2

# Example:
# -1 → 0
#  0 → 0.5
# +1 → 1.0
```

---

## Advanced: L2 Normalization (Unit Vectors)

**Different question:** Should we normalize vectors to **unit length**?

```python
v_normalized = v / ||v||
```

**Pros:**
- All vectors have magnitude 1
- Cosine similarity = dot product (faster)
- Removes magnitude bias

**Cons:**
- Loses information about # of features that apply
- Diphthongs vs monophthongs might differ in natural magnitude

**Verdict:** Probably **not necessary** for phonemes, since they all have similar numbers of applicable features (38 features each).

---

## Example Comparison

### Query: Find phonemes similar to /b/

**Three-way encoding:**
```python
/b/: voiced=+1, labial=+1, stop=-1 (continuant)
/p/: voiced=-1, labial=+1, stop=-1

Distance: sqrt((1-(-1))^2 + 0 + 0) = 2.0
# Clear: differs by voicing
```

**Binary-style:**
```python
/b/: voiced=1, labial=1, continuant=0
/p/: voiced=0, labial=1, continuant=0

Distance: sqrt(1 + 0 + 0) = 1.0
# Less discriminative
```

**Three-way gives more separation for feature oppositions!**

---

## Final Recommendation

✅ **Stick with three-way encoding (-1, 0, 1)**

It's:
- Linguistically principled
- Mathematically sound
- Better for capturing phonological oppositions
- Easy to convert to [0, 1] later if needed

The current implementation is correct! 🎯
