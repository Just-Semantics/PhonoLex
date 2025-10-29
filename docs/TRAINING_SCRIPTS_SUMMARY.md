# Training Scripts Summary

Complete set of canonical scripts for the four-layer architecture.

---

## Layer 1: Raw Phoible Features

**Script**: `compute_layer1_phoible_features.py`  
**Type**: Data extraction (not training)  
**Output**: `data/phoible/english/phoible-english.csv`

```bash
python scripts/compute_layer1_phoible_features.py
```

Extracts 38 distinctive features from Phoible database.  
Ternary encoding: `+` (present), `-` (absent), `0` (not applicable)

---

## Layer 2: Normalized Feature Vectors

**Script**: `demo_layer2_normalized_vectors.py`  
**Type**: Demo/usage (not training - deterministic computation)  
**Computation**: Via `PhonemeVectorizer` class

```bash
python scripts/demo_layer2_normalized_vectors.py  # Shows usage examples
```

Normalizes Layer 1 features to continuous vectors:
- 76-dim endpoints (start + end for 38 features)
- 152-dim trajectories (4 timesteps for articulation dynamics)

---

## Layer 3: Contextual Phoneme Embeddings

**Script**: `train_layer3_contextual_embeddings.py` ⭐  
**Type**: Training (transformer + next-phoneme prediction)  
**Output**: `models/curriculum/phoible_initialized_final.pt`

```bash
python scripts/train_layer3_contextual_embeddings.py
```

**Training details:**
- Data: CMU Dictionary (125K) + ipa-dict (147K total)
- Architecture: 3-layer transformer, 4 heads, 128-dim
- Task: Next-phoneme prediction
- Accuracy: 99.98%
- Time: ~10 minutes on Apple Silicon

**Results:**
- cat-bat (rhyme): 0.993
- cat-act (anagram): 0.677
- cat-dog (unrelated): 0.300

---

## Layer 4: Hierarchical Syllable Embeddings

**Script**: `train_layer4_hierarchical.py` ⭐  
**Type**: Building (aggregates frozen Layer 3)  
**Output**: `embeddings/layer4/syllable_embeddings.pt`

```bash
python scripts/train_layer4_hierarchical.py
```

**Building details:**
- Requires: Pre-trained Layer 3 (frozen)
- Process: Syllabification + aggregation by onset-nucleus-coda
- Data: CMU Dictionary (125K words)
- Time: ~5 minutes on CPU

**Results:**
- cat-bat (rhyme): 0.993
- cat-act (anagram): 0.200 ⭐ (excellent!)
- cat-dog (unrelated): 0.298

---

## Complete Pipeline

To build all layers from scratch:

```bash
# Layer 1: Extract Phoible features
python scripts/compute_layer1_phoible_features.py

# Layer 2: Demo (computation is automatic via PhonemeVectorizer)
python scripts/demo_layer2_normalized_vectors.py

# Layer 3: Train contextual embeddings (~10 min)
python scripts/train_layer3_contextual_embeddings.py

# Layer 4: Build syllable embeddings (~5 min)
python scripts/train_layer4_hierarchical.py
```

**Total time**: ~15 minutes on Apple Silicon

---

## File Organization

```
PhonoLex/
├── compute_layer1_phoible_features.py      # Layer 1 extraction
├── demo_layer2_normalized_vectors.py       # Layer 2 demo
├── train_layer3_contextual_embeddings.py   # Layer 3 training ⭐
├── train_layer4_hierarchical.py            # Layer 4 building ⭐
│
├── data/
│   └── phoible/
│       └── english/phoible-english.csv     # Layer 1 output
│
└── models/
    ├── curriculum/
    │   └── phoible_initialized_final.pt    # Layer 3 output ⭐
    └── hierarchical/
        └── final.pt                         # Layer 4 output ⭐
```

---

## Quick Reference

| Layer | Script | Type | Time | Output |
|-------|--------|------|------|--------|
| 1 | `compute_layer1_phoible_features.py` | Extract | <1 min | CSV file |
| 2 | `demo_layer2_normalized_vectors.py` | Demo | N/A | (computed) |
| 3 | `train_layer3_contextual_embeddings.py` | Train | ~10 min | 2.3 MB model |
| 4 | `train_layer4_hierarchical.py` | Build | ~5 min | Model + embeddings |

---

See **EMBEDDINGS_ARCHITECTURE.md** for complete documentation.
