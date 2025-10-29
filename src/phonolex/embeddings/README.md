# Phoneme Embeddings

**Learning dense vector representations of phonemes through multi-task learning.**

Like Word2Vec/BERT but for sounds.

---

## 🎯 Overview

This module trains **phoneme embeddings** - dense vector representations that capture phonological similarity and natural class structure.

### Why Phoneme Embeddings?

Just as word embeddings (Word2Vec, BERT) revolutionized NLP by learning semantic representations, phoneme embeddings provide a universal representation for phonology that enables:

1. **Rule Learning**: Predict allomorph selection (plural -s vs -z vs -ɪz)
2. **Similarity Tasks**: Rhyme detection, minimal pair identification
3. **Typology**: Discover cross-linguistic universals
4. **Dialectology**: Quantify systematic pronunciation differences
5. **G2P**: Improve grapheme-to-phoneme models

---

## 🏗️ Architecture

### Multi-Task Learning Framework

The model combines **5 training objectives**:

```
┌─────────────────────────────────────────────────────────┐
│                  Phoneme Embeddings (32-dim)            │
│              Initialized from Phoible features          │
└───────────┬─────────────────────────────────────────────┘
            │
    ┌───────┴───────┬───────────┬────────────┬────────────┐
    │               │           │            │            │
    ▼               ▼           ▼            ▼            ▼
┌──────┐     ┌──────────┐  ┌──────┐    ┌────────┐  ┌──────────┐
│ Skip │     │ Morpho-  │  │Metric│    │Feature │  │Inventory │
│ Gram │     │  logy    │  │Learn │    │Recon-  │  │Co-occur  │
│      │     │          │  │      │    │struct  │  │          │
└──────┘     └──────────┘  └──────┘    └────────┘  └──────────┘
Context      Allomorph     Similar     Preserve    Cross-ling
Prediction   Selection     Phonemes    Features    Structure
```

### Training Objectives

1. **Context Prediction** (like Word2Vec skip-gram)
   - Predict neighboring phonemes in words
   - Data: CMU Dict (803K examples)
   - Loss: Cross-entropy

2. **Morphology Prediction**
   - Predict allomorph from stem-final phoneme
   - Example: /t/ → /-s/ (cats), /g/ → /-z/ (dogs)
   - Data: SIGMORPHON (80K examples)
   - Loss: Cross-entropy (currently disabled - needs pronunciation lookup)

3. **Contrastive Learning**
   - Similar phonemes → similar embeddings
   - Data: Phoible feature similarity (100K pairs)
   - Loss: MSE on cosine similarity

4. **Feature Reconstruction**
   - Embeddings must preserve phonological features
   - Forces retention of linguistic knowledge
   - Data: Phoible 38-dim features
   - Loss: MSE

5. **Inventory Co-occurrence**
   - Phonemes in same language → related embeddings
   - Cross-linguistic regularization
   - Data: Phoible 3,020 inventories
   - Loss: Binary cross-entropy

---

## 📊 Data Sources

The model learns from **multiple datasets**:

| Dataset | Size | Purpose |
|---------|------|---------|
| **Phoible** | 3,142 phonemes, 2,716 languages | Features + inventories |
| **CMU Dict** | 134K words | Context (skip-gram) |
| **SIGMORPHON** | 80K examples | Morphology (disabled for now) |

All connected via ARPAbet↔IPA mappings.

---

## 🚀 Quick Start

### 1. Train Embeddings

```bash
cd /Users/jneumann/Repos/PhonoLex

# Train for 10 epochs (default)
python3 src/phonolex/embeddings/train.py

# Custom configuration
python3 src/phonolex/embeddings/train.py \
  --embedding-dim 64 \
  --batch-size 128 \
  --num-epochs 20 \
  --learning-rate 0.001 \
  --device cpu \
  --output-dir models/phoneme_embeddings
```

**Output**:
- `models/phoneme_embeddings/best_model.pt` - Best checkpoint
- `models/phoneme_embeddings/final_model.pt` - Final model
- `models/phoneme_embeddings/training_history.json` - Loss curves
- `models/phoneme_embeddings/phoneme_vocab.json` - Phoneme vocabulary

### 2. Evaluate Embeddings

```bash
python3 src/phonolex/embeddings/evaluate.py \
  --model-path models/phoneme_embeddings/best_model.pt \
  --output-dir outputs/evaluation
```

**Evaluation Tasks**:
- Feature reconstruction (how well preserved?)
- Similarity correlation (match feature similarity?)
- Natural class clustering (do classes emerge?)
- t-SNE visualization

### 3. Use Embeddings in Your Code

```python
from phonolex.embeddings.model import create_model
from phonolex.embeddings.data_loader import PhonemeEmbeddingDataLoader
import torch

# Load data
loader = PhonemeEmbeddingDataLoader()

# Load trained model
checkpoint = torch.load('models/phoneme_embeddings/best_model.pt')
model = create_model(
    num_phonemes=checkpoint['config']['num_phonemes'],
    embedding_dim=checkpoint['config']['embedding_dim']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Get embedding for /p/
p_id = loader.get_phoneme_id('p')
p_embedding = model.get_embeddings(torch.tensor([p_id]))[0]

# Find similar phonemes (cosine similarity)
all_ids = torch.arange(len(loader.phoneme_to_id))
all_embeddings = model.get_embeddings(all_ids)

# Cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
sims = cosine_similarity([p_embedding], all_embeddings)[0]

# Top 10 similar
top_indices = sims.argsort()[-10:][::-1]
for idx in top_indices:
    phoneme = loader.id_to_phoneme[idx]
    print(f"/{phoneme}/: {sims[idx]:.3f}")
```

---

## 📁 Module Structure

```
src/phonolex/embeddings/
├── data_loader.py      # Unified data loading (THE foundation)
├── model.py            # Multi-task embedding model
├── train.py            # Training script
├── evaluate.py         # Evaluation script
└── README.md           # This file
```

---

## 🧪 Expected Results

After training, you should see:

### Feature Reconstruction
- MSE: < 0.5 (good preservation)
- Correlation: > 0.7 (strong alignment)

### Similarity
- Spearman ρ: > 0.6 (embeddings match feature similarity)

### Examples
Similar phonemes should cluster:
- **/p/, /b/, /t/, /d/, /k/, /g/** (stops)
- **/s/, /z/, /ʃ/, /ʒ/** (sibilants)
- **/a/, /ɑ/, /æ/** (low vowels)
- **/i/, /iː/, /ɪ/** (high front vowels)

---

## 🔬 How It Works

### 1. Initialization

Embeddings are initialized from Phoible features using SVD:

```python
# 38-dim features → 32-dim embeddings (via SVD)
U, S, V = torch.svd(phoible_features)
embeddings = phoible_features @ V[:, :32]
```

This ensures the model starts with phonological knowledge.

### 2. Multi-Task Training

Each batch includes multiple tasks:
- Context prediction (always)
- Feature reconstruction (always)
- Contrastive learning (50% of batches)
- Inventory co-occurrence (30% of batches)
- Morphology (disabled for now)

Weighted loss:
```python
loss = 1.0 * context_loss \
     + 1.0 * feature_loss \
     + 0.5 * contrastive_loss \
     + 0.3 * inventory_loss
```

### 3. Learning Dynamics

- **Context prediction** pulls similar phonemes together (distributional)
- **Feature reconstruction** preserves linguistic structure
- **Contrastive learning** explicitly encodes similarity
- **Inventory co-occurrence** adds cross-linguistic signal

The combination creates embeddings that are both:
- **Linguistically grounded** (from features)
- **Functionally useful** (from distributional data)

---

## 🎨 Visualization

After training, create t-SNE plots:

```bash
python3 src/phonolex/embeddings/evaluate.py
```

This generates `outputs/visualizations/embeddings_tsne.png`:
- Each point = one phoneme
- Distance = phonological similarity
- Clusters = natural classes

---

## 🔧 Advanced Usage

### Custom Task Weights

Edit `train.py` to adjust task weights:

```python
task_weights = {
    'context': 2.0,      # Emphasize distributional learning
    'morphology': 0.0,   # Disable (for now)
    'contrastive': 1.0,  # Moderate similarity learning
    'feature': 0.5,      # Lighter feature preservation
    'inventory': 0.1     # Minimal cross-linguistic
}
```

### Different Embedding Dimensions

```bash
# Larger embeddings (more capacity)
python3 src/phonolex/embeddings/train.py --embedding-dim 128

# Smaller embeddings (faster, more constrained)
python3 src/phonolex/embeddings/train.py --embedding-dim 16
```

### Checkpointing

Models are saved after each epoch if they improve:

```python
if val_loss < best_loss:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': val_loss
    }, 'models/best_model.pt')
```

---

## 🐛 Known Issues

### MPS (Apple Silicon) Dtype Error

MPS currently has issues with mixed int64/float32 operations. Use CPU for now:

```bash
python3 src/phonolex/embeddings/train.py --device cpu
```

### Morphology Task Disabled

Needs pronunciation lookup to extract actual allomorphs from SIGMORPHON data. Currently returns empty lists.

**TODO**: Integrate CMU Dict lookup in `MorphologyExample` generation.

---

## 📈 Future Improvements

1. **Enable morphology task**
   - Add pronunciation lookup
   - Extract allomorphs automatically
   - Train allomorph classifier

2. **Add negative sampling** for context prediction
   - Currently uses full softmax (slow)
   - Negative sampling would speed up training

3. **Pre-trained embeddings**
   - Release pre-trained checkpoints
   - Different sizes (16d, 32d, 64d, 128d)

4. **Downstream task evaluation**
   - Rhyme detection accuracy
   - Minimal pair classification
   - Allomorph prediction accuracy

5. **Multilingual embeddings**
   - Train on all 2,716 languages
   - Cross-lingual phoneme transfer

---

## 📚 References

### Conceptual Inspiration
- **Word2Vec**: Mikolov et al. (2013) - Distributional semantics
- **BERT**: Devlin et al. (2019) - Multi-task pre-training
- **Phonological Features**: Chomsky & Halle (1968) - SPE

### Data Sources
- **Phoible**: Moran & McCloy (2019) - Cross-linguistic phonology
- **CMU Dict**: Carnegie Mellon University - Pronunciations
- **SIGMORPHON**: Morphological inflection shared tasks

---

## 💡 Citation

If you use these phoneme embeddings, please cite:

```bibtex
@software{phonolex_embeddings,
  title={PhonoLex: Phoneme Embeddings via Multi-Task Learning},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/PhonoLex}
}
```

---

## ✨ Summary

**Phoneme embeddings are the universal representation for phonology.**

Just as word embeddings enabled modern NLP, phoneme embeddings enable:
- Phonological rule learning
- Cross-linguistic analysis
- Pronunciation modeling
- Dialectology
- And more!

This module provides a complete pipeline:
1. **Data loading** from multiple sources
2. **Multi-task training** with 5 objectives
3. **Evaluation** on phonological tasks
4. **Pre-trained models** (coming soon)

**Let's go! Train your first model:**

```bash
python3 src/phonolex/embeddings/train.py --num-epochs 10
```
