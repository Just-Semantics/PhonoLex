# Training Scripts

This directory contains the canonical training scripts for building all 4 layers of phonological embeddings.

## Quick Start

Run in order:

```bash
# Layer 1: Extract Phoible features (38-dim ternary)
python compute_layer1_phoible_features.py

# Layer 2: Demo normalized vectors (76/152-dim computed)
python demo_layer2_normalized_vectors.py

# Layer 3: Train contextual phoneme embeddings (128-dim, ~10 min)
python train_layer3_contextual_embeddings.py

# Layer 4: Build hierarchical syllable embeddings (384-dim, ~5 min)
python train_layer4_hierarchical.py
```

## Outputs

- **Layer 1**: `data/phoible/english/phoible-english.csv`
- **Layer 2**: No output (demonstration only)
- **Layer 3**: `models/layer3/model.pt` (2.4M, PhonoLexBERT model)
- **Layer 4**: `embeddings/layer4/syllable_embeddings.pt` (1.0G, syllable embeddings dictionary)

## Architecture

```
Layer 1: Raw Phoible Features (38-dim ternary)
    ↓ normalization & interpolation
Layer 2: Normalized Feature Vectors (76-dim / 152-dim)
    ↓ contextual learning via transformer
Layer 3: Contextual Phoneme Embeddings (128-dim)
    ↓ syllable aggregation (onset-nucleus-coda)
Layer 4: Hierarchical Syllable Embeddings (384-dim)
```

## Layer Details

### Layer 1: Raw Phoible Features
- **Not trained** - extracted from Phoible database
- 38 distinctive features (Hayes 2009 + Moisik & Esling 2011)
- Ternary encoding: `+` (present), `-` (absent), `0` (not applicable)

### Layer 2: Normalized Vectors
- **Not trained** - deterministic transformation
- Converts ternary features to continuous vectors
- 76-dim: start + end for each feature
- 152-dim: 4-timestep trajectory for diphthongs

### Layer 3: Contextual Phoneme Embeddings
- **TRAINED** - PhonoLexBERT transformer
- Training data: CMU Dictionary (125K) + ipa-dict (22K) = 147K words
- Training task: Next-phoneme prediction
- Training time: ~10 minutes on Apple Silicon
- Accuracy: 99.98%
- Output: 128-dim contextual embedding per phoneme
- Model size: 2.4M

### Layer 4: Hierarchical Syllable Embeddings
- **NOT trained** - uses frozen Layer 3
- Process: Load Layer 3 → get contextual phoneme embeddings → aggregate by syllable structure
- Syllable structure: onset-nucleus-coda (128-dim each)
- Building time: ~5 minutes on CPU
- Output: Dictionary of pre-computed 384-dim syllable embeddings per word
- Dictionary size: 1.0G (125K words)

## Performance

- cat-bat (rhyme): 0.995 ✅
- cat-act (anagram): 0.288 ✅ (excellent discrimination)
- cat-dog (unrelated): 0.225 ✅
- make-take (rhyme): 0.987 ✅
- computer-commuter (sound-alike): 0.797 ✅

## Dependencies

- `train_phonology_bert.py`: Contains PhonoLexBERT model class used by Layer 3

## See Also

- [EMBEDDINGS_ARCHITECTURE.md](../../docs/EMBEDDINGS_ARCHITECTURE.md) - Complete architecture documentation
- [TRAINING_SCRIPTS_SUMMARY.md](../../docs/TRAINING_SCRIPTS_SUMMARY.md) - Quick reference
- [CLAUDE.md](../../CLAUDE.md) - Development instructions
