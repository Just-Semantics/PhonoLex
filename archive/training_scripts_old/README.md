# Archived Training Scripts

These scripts have been superseded by canonical versions.

## Archived Files

### train_hierarchical_final.py
**Superseded by**: `../../train_layer4_hierarchical.py`

**Issue**: This version trained Layer 3 from scratch during the hierarchical model building, instead of loading the pre-trained Layer 3 model.

**Date archived**: 2025-10-28

---

### train_sequential_all_datasets.py
**Superseded by**: `../../train_layer3_contextual_embeddings.py`

**Issue**: This version included SIGMORPHON and UniMorph datasets, which mostly add suffix patterns rather than genuine new phoneme sequences. The canonical version uses only CMU + ipa-dict.

**Date archived**: 2025-10-28

---

### train_sequential_cmu_ipadict.py
**Superseded by**: `../../train_layer3_contextual_embeddings.py`

**Status**: This WAS the correct version! It was copied and renamed to the canonical script name.

**Date archived**: 2025-10-28

---

## Why Archived?

During the canonization process (2025-10-28), we established clear naming conventions:
- Layer 1: `compute_layer1_phoible_features.py`
- Layer 2: `demo_layer2_normalized_vectors.py`
- Layer 3: `train_layer3_contextual_embeddings.py`
- Layer 4: `train_layer4_hierarchical.py`

These old scripts had inconsistent naming and/or incorrect architectures.

See `../../EMBEDDINGS_ARCHITECTURE.md` for the canonical specification.
