# Vector Encoding Strategies for Phonological Features

## Overview

This document outlines strategies for converting Phoible's ternary phonological features into vector representations suitable for similarity metrics and machine learning.

## Phoible Feature System

### Source & Theoretical Basis
- **Based on**: Hayes (2009) *Introductory Phonology* + Moisik & Esling (2011)
- **Design goal**: Descriptively adequate cross-linguistically
- **Feature count**: 38 features per phoneme
- **Feature values**: Ternary system: `+`, `-`, `0`

### Value Semantics

| Value | Meaning | Interpretation | Examples |
|-------|---------|----------------|----------|
| `+` | Feature present/active | Phoneme has this property | `voiced: +` for /b/ |
| `-` | Feature absent/inactive | Phoneme explicitly lacks this property | `voiced: -` for /p/ |
| `0` | Not applicable/neutral | Feature doesn't apply to this segment type | `consonantal: 0` for glides |

### **CRITICAL: Multi-Value Features (Trajectories)**

**Phoible represents dynamic segments (diphthongs, affricates) with comma-separated values showing articulatory TRAJECTORIES:**

| Pattern | Meaning | Example |
|---------|---------|---------|
| `"+,-"` | Starts with feature, ends without | `syllabic: +,-` in /eɪ̯/ (starts vowel-like, ends glide-like) |
| `"-,+"` | Starts without, ends with | `high: -,+` in /aɪ/ (starts low, ends high) |
| `"+,-,+"` | Three-phase trajectory | Rare, complex segments |
| `"+,-,+,-"` | Four-phase trajectory | Very rare |

**Real examples from English:**
- **`/aɪ/`** ("buy"): `high: -,+` `low: +,-` `front: -,+`
  - Starts low+back (`[a]`), ends high+front (`[ɪ]`)
- **`/aʊ/`** ("how"): `high: -,+` `labial: -,+` `back: -,+`
  - Starts low (`[a]`), ends high+back+rounded (`[ʊ]`)
- **`/oʊ/`** ("go"): `high: -,+` `tense: +,-`
  - Starts mid-back (`[o]`), glides higher to lax offglide

**Distribution across FULL Phoible:**
- **2-value trajectories**: 18,021 occurrences (most diphthongs)
- **3-value trajectories**: 815 occurrences (complex segments)
- **4-value trajectories**: 9 occurrences (very rare)

### Distribution in English Dataset
From [data/phoible/english/phoible-english.csv](data/phoible/english/phoible-english.csv):
- **`-`**: 8,991 occurrences (58.8%)
- **`0`**: 3,436 occurrences (22.5%)
- **`+`**: 2,729 occurrences (17.8%)
- **Multi-value** (trajectories): present in all diphthongs and some complex segments

Total: 406 phonemes × 38 features = 15,428 feature values

## Vector Encoding Strategies

### Strategy 1: Binary Encoding (Ignore `0`)
**Approach**: Map only `+`/`-` to binary, treat `0` as special case

```python
encoding = {
    '+': 1.0,
    '-': 0.0,
    '0': 0.5  # or np.nan, or separate handling
}
```

**Pros:**
- Simple, standard ML approach
- Works with most distance metrics (cosine, euclidean)
- Matches Hayes' binary theoretical framework

**Cons:**
- Loses semantic distinction between "absent" and "not applicable"
- `0` → 0.5 is arbitrary (why not 0.0 or 1.0?)
- May create false similarities between vowels and consonants

**Best for:** Quick prototyping, traditional ML models

---

### Strategy 2: Three-Way Encoding (Fuzzy Values)
**Approach**: Map to three distinct scalar values

```python
encoding = {
    '+': 1.0,
    '-': -1.0,
    '0': 0.0
}
```

**Pros:**
- Preserves semantic distance: `+` and `-` are opposites
- `0` is truly neutral/central
- Works with cosine similarity (signed vectors)
- Mathematically interpretable

**Cons:**
- Distance metrics may need tuning
- `0` still creates middle ground (is that meaningful?)
- Euclidean distance treats `0` as "halfway" between `+`/`-`

**Best for:** Gradient-based models, signed similarity metrics

---

### Strategy 3: One-Hot Encoding (Categorical)
**Approach**: Each feature becomes 3 dimensions

```python
# Original: 38 features → Vector: 114 dimensions
'+' → [1, 0, 0]
'-' → [0, 1, 0]
'0' → [0, 0, 1]
```

**Pros:**
- No assumptions about feature value similarity
- Categorical feature properly represented
- Clear semantic separation
- Hamming distance is meaningful

**Cons:**
- 3× vector dimensionality (38 → 114 dimensions)
- Sparse representation
- May need dimensionality reduction for some applications

**Best for:** Neural networks, tree-based models, when semantics matter most

---

### Strategy 4: Segment-Type Aware Encoding
**Approach**: Different encoding per segment class (vowel/consonant)

```python
# For consonants:
consonant_encoding = {
    '+': 1.0,
    '-': 0.0,
    '0': np.nan  # Mark as missing, then impute or mask
}

# For vowels (different feature applicability):
vowel_encoding = {
    '+': 1.0,
    '-': 0.0,
    '0': np.nan
}
```

**Pros:**
- Respects linguistic reality (features apply differently)
- Can use feature masking or attention mechanisms
- Doesn't force false comparisons (e.g., vowels vs consonants)

**Cons:**
- More complex implementation
- Need to handle NaN/masking in distance calculations
- Requires segment type metadata

**Best for:** Linguistically-informed models, attention mechanisms

---

### Strategy 5: Multi-Vector Per Phoneme
**Approach**: Create multiple vectors per phoneme for different interpretations

```python
# Phoneme /b/ gets 3 vectors:
vectors = {
    'binary': [1, 0, 1, ...],           # Binary features only
    'ternary': [1.0, -1.0, 0.0, ...],   # Three-way encoding
    'onehot': [1,0,0, 0,1,0, ...],      # Categorical encoding
}
```

**Pros:**
- Flexibility for different similarity metrics
- Can ensemble across representations
- Supports multi-view learning
- No information loss

**Cons:**
- Storage overhead (3× vectors)
- Complexity in querying
- Need to decide which representation for which task

**Best for:** Research, exploration, multi-task learning

---

### Strategy 6: Learned Embeddings
**Approach**: Use features as input to learn dense representations

```python
# Input: 38-dim ternary features
# Train autoencoder or use contrastive learning
# Output: k-dim dense embedding (e.g., 16, 32, 64 dims)

# Example with allophone data:
# /b/ → [bⁿ, p͉, b̚, b] → learn shared embedding
```

**Pros:**
- Captures phonological relationships in data
- Compact representation
- Can incorporate allophone information
- Task-optimized (if supervised)

**Cons:**
- Requires training data and labels (if supervised)
- Less interpretable
- Need to decide on architecture
- Computational overhead

**Best for:** Deep learning applications, when you have related tasks

---

## Recommendations by Use Case

### For Phoneme Similarity Search
**Recommended**: **Strategy 2 (Three-Way)** or **Strategy 4 (Segment-Aware)**

```python
# Quick start with three-way
vector = np.array([1.0 if f=='+' else -1.0 if f=='-' else 0.0
                   for f in features])
similarity = cosine_similarity(v1, v2)
```

### For Classification/ML
**Recommended**: **Strategy 3 (One-Hot)** or **Strategy 6 (Learned)**

Neural networks handle categorical features well via one-hot encoding

### For Linguistic Analysis
**Recommended**: **Strategy 4 (Segment-Aware)** or **Strategy 5 (Multi-Vector)**

Preserve linguistic distinctions, allow for segment-class comparisons

### For Vector Databases
**Recommended**: Start with **Strategy 2**, optionally add **Strategy 6**

Most vector DBs expect dense float vectors; three-way encoding is simple and effective

---

## Vector Database Options

### Option 1: PostgreSQL + pgvector
```sql
CREATE TABLE phonemes (
    id SERIAL PRIMARY KEY,
    phoneme TEXT,
    inventory_id INT,
    language TEXT,
    features VECTOR(38),  -- Three-way encoding
    segment_class TEXT
);

CREATE INDEX ON phonemes USING ivfflat (features vector_cosine_ops);
```

**Pros:**
- SQL interface
- ACID transactions
- Metadata queries + vector search
- Mature ecosystem

**Cons:**
- Limited to cosine/L2/inner product
- Not optimized for very large scale
- Need separate Postgres instance

---

### Option 2: ChromaDB
```python
import chromadb
client = chromadb.Client()

collection = client.create_collection(
    name="phonemes",
    metadata={"description": "Phoible phoneme features"}
)

collection.add(
    embeddings=[[1.0, -1.0, 0.0, ...]],  # Feature vectors
    metadatas=[{"phoneme": "b", "language": "English", "inventory": 160}],
    ids=["eng_160_b"]
)

# Query
results = collection.query(
    query_embeddings=[[1.0, -1.0, ...]],
    n_results=10
)
```

**Pros:**
- Lightweight, embedded
- Easy Python API
- Built-in metadata filtering
- Good for prototyping

**Cons:**
- Less scalable than dedicated solutions
- Limited deployment options
- Newer, less mature

---

### Option 3: FAISS (Facebook AI Similarity Search)
```python
import faiss
import numpy as np

# Build index
dimension = 38
index = faiss.IndexFlatL2(dimension)  # or IndexIVFFlat, IndexHNSW

# Add vectors
vectors = np.array([[1.0, -1.0, 0.0, ...], ...])
index.add(vectors.astype('float32'))

# Search
D, I = index.search(query_vector, k=10)  # distances, indices
```

**Pros:**
- Blazing fast
- Scales to billions of vectors
- Many index types (flat, IVF, HNSW, etc.)
- GPU support

**Cons:**
- No metadata storage (need separate DB)
- No transactions
- Python-only interface
- More complex setup for production

---

### Option 4: Simple JSON + NumPy (Baseline)
```python
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load data
with open('data/phoneme_vectors.json') as f:
    data = json.load(f)

vectors = np.array([d['features'] for d in data])
metadata = [{'phoneme': d['phoneme'], 'lang': d['lang']} for d in data]

# Search
sims = cosine_similarity([query_vector], vectors)[0]
top_k = np.argsort(sims)[-10:][::-1]
```

**Pros:**
- Zero setup
- Full control
- Portable
- Good for <10K vectors

**Cons:**
- Linear search (O(n))
- No indexing
- Manual memory management
- Not production-ready

---

## Recommended Starting Point

### Phase 1: Exploration (Now)
1. **Storage**: JSON files with NumPy arrays
2. **Encoding**: Strategy 2 (Three-way: +1, -1, 0)
3. **Similarity**: Cosine similarity
4. **Scale**: ~500 English phonemes across 10 dialects

### Phase 2: Prototype
1. **Storage**: ChromaDB (embedded)
2. **Encoding**: Multi-vector (Strategy 5) - three-way + one-hot
3. **Similarity**: Cosine + custom weighted features
4. **Scale**: Full Phoible (2,716 languages, ~40K+ phonemes)

### Phase 3: Production
1. **Storage**: PostgreSQL + pgvector
2. **Encoding**: Learned embeddings (Strategy 6) from phoneme context
3. **Similarity**: Learned metric
4. **Scale**: All languages + derived features

---

## Next Steps

1. **Create sample vectors** from English Phoible data
2. **Test encoding strategies** with known similar phonemes
3. **Evaluate similarity metrics** (cosine vs euclidean vs custom)
4. **Integrate CMU dict** pronunciation data with feature vectors
5. **Build prototype search** interface

---

## References

- **Hayes (2009)**: *Introductory Phonology* - Binary feature theory
- **Moisik & Esling (2011)**: Laryngeal features
- **Phoible**: https://phoible.org/ - Feature system source
- **Phoible GitHub**: https://github.com/phoible/dev - Raw data and documentation

---

## Handling Multi-Value Trajectories

The presence of trajectory features (comma-separated values) fundamentally changes vector encoding! Here are strategies:

### Strategy 7: Average Trajectory Values
**Approach**: For `"+,-"` → average to `0.0`, for `"-,+"` → average to `0.0`

```python
def encode_feature(value):
    if ',' not in value:
        return {'+': 1.0, '-': -1.0, '0': 0.0}[value]
    
    # Average trajectory
    parts = value.split(',')
    values = [{'+': 1.0, '-': -1.0, '0': 0.0}[p] for p in parts]
    return np.mean(values)

# Example: /aɪ/ with high: -,+
# → (-1.0 + 1.0) / 2 = 0.0
```

**Pros:**
- Simple, single vector per phoneme
- Smooth representation
- Works with standard distance metrics

**Cons:**
- **LOSES ALL TRAJECTORY INFORMATION!** 
- `/aɪ/` (low→high) and `/oʊ/` (mid→high) both average to 0.5
- Diphthongs collapse to something like monophthongs
- Linguistically incorrect

**Verdict:** ❌ **Not recommended** - loses too much information

---

### Strategy 8: Separate Start/End Vectors
**Approach**: Each phoneme gets TWO vectors - one for start state, one for end state

```python
# For /aɪ/ with "high: -,+" "low: +,-" "front: -,+"
start_vector = [-1.0, 1.0, -1.0, ...]  # low, back
end_vector = [1.0, -1.0, 1.0, ...]     # high, front

# For monophthongs (no trajectory), start == end
# For /i/ with "high: +" "front: +"
start_vector = [1.0, 1.0, ...]
end_vector = [1.0, 1.0, ...]  # same
```

**Pros:**
- Preserves trajectory information
- Can compute similarity on start, end, or both
- Natural for dynamic segments
- Linguistically motivated

**Cons:**
- 2× storage
- Need to handle 2-value, 3-value, 4-value trajectories differently
- Querying is more complex (search start? end? both?)

**Best for:** Diphthong analysis, trajectory-aware similarity

---

### Strategy 9: Trajectory Feature Expansion
**Approach**: Expand each feature into START and END dimensions

```python
# Original: 38 features → New: 76 dimensions (38_start + 38_end)

# For /aɪ/:
features = {
    'high_start': -1.0, 'high_end': 1.0,
    'low_start': 1.0, 'low_end': -1.0,
    'front_start': -1.0, 'front_end': 1.0,
    # ... monophthong features same start/end
    'voiced_start': 1.0, 'voiced_end': 1.0,  # no trajectory
}

vector = [high_start, high_end, low_start, low_end, ...]  # 76-dim
```

**Pros:**
- Single vector per phoneme
- Full trajectory information preserved
- Standard ML tools work
- Can learn which features (start vs end) matter for task

**Cons:**
- 2× dimensionality
- Some redundancy for monophthongs
- Need to handle 3+ value trajectories (use midpoints?)

**Best for:** ML models, when you want single vector with trajectory info

---

### Strategy 10: Temporal Sequence Vectors
**Approach**: Represent as a SEQUENCE of states (like RNN input)

```python
# For /aɪ/ with 2-phase trajectory:
sequence = [
    [-1.0, 1.0, -1.0, ...],  # t=0: low, back
    [1.0, -1.0, 1.0, ...]     # t=1: high, front
]  # Shape: (2, 38)

# For /i/ (no trajectory):
sequence = [
    [1.0, 1.0, ...],  # t=0: high, front
    [1.0, 1.0, ...]   # t=1: same
]  # Shape: (2, 38)

# Pad to max length (4 for 4-phase trajectories)
sequence = pad(sequence, max_len=4)  # Shape: (4, 38)
```

**Pros:**
- Handles arbitrary trajectory lengths
- Natural for RNNs/LSTMs/Transformers
- Can model phoneme as temporal process
- Preserves full dynamics

**Cons:**
- Requires sequence-aware models or pooling
- More complex
- Need padding/masking for variable lengths

**Best for:** Deep learning, when modeling phonemes as temporal sequences

---

### Strategy 11: Delta Features (Trajectory Encoding)
**Approach**: Store STATIC features + DELTA (change) features

```python
# For /aɪ/:
static = [-1.0, 1.0, -1.0, ...]  # Start state
delta = [2.0, -2.0, 2.0, ...]    # Change: -1→1 = Δ2.0

# Combined vector: [static features, delta features]
vector = np.concatenate([static, delta])  # 76-dim

# For monophthongs:
static = [1.0, 1.0, ...]
delta = [0.0, 0.0, ...]  # No change
```

**Pros:**
- Single vector
- Separates static vs dynamic information
- Can weight static vs dynamic differently in similarity
- Linguistically interpretable

**Cons:**
- 2× dimensionality
- Need to handle 3+ phase trajectories (multiple deltas?)
- Arbitrary choice of "start" as reference

**Best for:** When you want to weight static vs dynamic features differently

---

### Strategy 12: Trajectory Type Encoding
**Approach**: Classify trajectory patterns and use as additional features

```python
# Classify each feature's trajectory type:
trajectory_types = {
    'high': 'RISING',      # -,+
    'low': 'FALLING',      # +,-
    'front': 'RISING',     # -,+
    'voiced': 'STATIC',    # + (no comma)
    # ...
}

# Encode as one-hot or numerical:
trajectory_encoding = {
    'STATIC': 0,
    'RISING': 1,
    'FALLING': -1,
    'COMPLEX': 2  # for 3+ values
}

# Vector includes both values AND trajectory types
feature_vector = [start_values... end_values... trajectory_types...]
```

**Pros:**
- Explicit trajectory representation
- Can query "find vowels with rising height"
- Interpretable
- Combines well with other strategies

**Cons:**
- Increases dimensionality
- Loses some granularity (rising vs how much?)

**Best for:** Feature-based querying, interpretability

---

## Recommended Approach for Trajectories

### **Recommendation: Strategy 9 (Feature Expansion) + Strategy 2 (Three-way)**

```python
def encode_phoneme(features_dict):
    """
    Convert Phoible features to 76-dim vector (38 features × 2 states)
    """
    vector = []
    
    for feature_name in FEATURE_ORDER:  # 38 features
        value = features_dict[feature_name]
        
        if ',' in value:
            # Trajectory: split into start and end
            parts = value.split(',')
            start = encode_value(parts[0])
            end = encode_value(parts[-1])  # Use last for end
            # For 3+ values, could use first/middle/last or interpolate
        else:
            # Static: same start and end
            start = end = encode_value(value)
        
        vector.extend([start, end])
    
    return np.array(vector)  # 76-dim

def encode_value(v):
    return {'+': 1.0, '-': -1.0, '0': 0.0}[v]

# Example:
# /aɪ/: high:-,+ → high_start=-1.0, high_end=1.0
# /i/: high:+ → high_start=1.0, high_end=1.0
```

**Why this works:**
- Single vector per phoneme (easy storage/querying)
- Preserves trajectory information
- Works with standard similarity metrics
- Can learn feature importance (model may learn to ignore _end features for consonants)
- Handles 3-4 phase trajectories (use first and last, or average middle)

---

## Updated Database Recommendation

### ChromaDB with 76-dim vectors

```python
import chromadb
import numpy as np

client = chromadb.Client()
collection = client.create_collection("phonemes_with_trajectories")

# Add phoneme with trajectory encoding
collection.add(
    embeddings=[[  # 76-dim: 38 features × 2 (start, end)
        -1.0, 1.0,  # high: start, end
        1.0, -1.0,  # low: start, end
        -1.0, 1.0,  # front: start, end
        # ... etc
    ]],
    metadatas=[{
        "phoneme": "aɪ",
        "language": "English",
        "inventory_id": 160,
        "segment_class": "vowel",
        "has_trajectory": True,
        "trajectory_features": ["high", "low", "front"]
    }],
    ids=["eng_160_aɪ"]
)

# Query: find similar diphthongs
results = collection.query(
    query_embeddings=[phoneme_vector],
    n_results=10,
    where={"segment_class": "vowel"}  # Filter metadata
)
```

---

## Next Implementation Steps

1. **Parse Phoible data** → extract trajectory features
2. **Implement Strategy 9 encoder** → 76-dim vectors
3. **Build ChromaDB collection** → store all phonemes
4. **Test similarity queries**:
   - Find phonemes similar to /aɪ/
   - Find monophthongs similar to /i/
   - Find diphthongs with rising height
5. **Evaluate** different weighting schemes for start vs end features
6. **Integrate CMU dict** → map ARPAbet to Phoible vectors

