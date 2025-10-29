# Phoneme Embeddings as Universal Framework

## 💡 Core Insight

**Phoneme embeddings are like word embeddings (Word2Vec, GloVe) but for sounds.**

Just as word embeddings learned from context enable:
- Similarity (king - man + woman ≈ queen)
- Analogies
- Classification
- Clustering
- Transfer learning

**Phoneme embeddings enable ALL of our phonological tasks!**

---

## 🎯 The Universal Task: Learn Phoneme Embeddings

### Input Data Sources

1. **Phonological Context** (like word context in Word2Vec)
   ```python
   # From CMU Dict:
   "cat" → /k æ t/
   # Context: /æ/ appears between /k/ and /t/

   # Learn embeddings where:
   # - Similar phonemes have similar embeddings
   # - Context-appropriate phonemes are close
   ```

2. **Phonological Features** (initialization/supervision)
   ```python
   # Start with Phoible 38-dim features
   /t/ → [consonantal:+, voiced:-, ...]

   # Learn to refine these into dense embeddings
   # Maybe 16-dim, 32-dim, or 64-dim
   ```

3. **Morphological Patterns** (additional signal)
   ```python
   # From SIGMORPHON:
   # "walk" /wɔk/ + past tense → "walked" /wɔkt/
   # Learn that /k/ → /t/ is a common pattern
   ```

4. **Cross-Linguistic Context** (universals)
   ```python
   # From Phoible 2,716 languages:
   # Which phonemes co-occur in inventories?
   # /p/ and /b/ often occur together
   # Learn universal phonological structure
   ```

---

## 🏗️ Unified Architecture

```
┌─────────────────────────────────────────────────────┐
│         PHONEME EMBEDDING MODEL                      │
│                                                       │
│  Input: Phoneme /t/                                  │
│         ↓                                            │
│  Initialize: Phoible features (38-dim)               │
│         ↓                                            │
│  Learn: Dense embedding (32-dim)                     │
│         ↓                                            │
│  Objectives (multi-task):                            │
│    1. Predict context phonemes                       │
│    2. Predict allomorph selection                    │
│    3. Cluster by natural classes                     │
│    4. Preserve Phoible feature info                  │
│         ↓                                            │
│  Output: Learned phoneme embedding                   │
└─────────────────────────────────────────────────────┘
```

---

## 🎓 Training Objectives (Multi-Task Learning)

### Objective 1: Phonological Context Prediction
**Like Word2Vec skip-gram for phonemes**

```python
# From CMU Dict pronunciations:
# Given center phoneme, predict context

# Example: "cat" /k æ t/
center = /æ/
context = [/k/, /t/]

# Loss: maximize P(context | center)
```

**What this learns**: Phonemes that appear in similar contexts

---

### Objective 2: Morphological Pattern Prediction
**Predict allomorph selection from phoneme features**

```python
# From SIGMORPHON:
stem_final = /k/  # from "walk"
inflection_type = "past_tense"

# Predict allomorph
predict(/k/, "past_tense") → /-t/ (not /-d/ or /-ɪd/)

# Loss: cross-entropy over allomorph choices
```

**What this learns**: Phonologically-conditioned rules

---

### Objective 3: Natural Class Clustering
**Embeddings should cluster by phonological classes**

```python
# Contrastive learning:
# Pull together: /p, t, k/ (voiceless stops)
# Push apart: /p, b/ (differ in voicing)

# Loss: triplet loss or contrastive loss
```

**What this learns**: Distinctive feature structure

---

### Objective 4: Feature Reconstruction
**Preserve Phoible feature information**

```python
# Given embedding, reconstruct features
embedding(/t/) → [consonantal:+, voiced:-, ...]

# Loss: MSE or BCE on feature reconstruction
```

**What this learns**: Interpretable dimensions

---

### Objective 5: Cross-Linguistic Regularization
**Phonemes that co-occur in languages should be similar**

```python
# From Phoible inventories:
# /p/ and /b/ often co-occur
# /k/ and /g/ often co-occur
# /θ/ and /ð/ sometimes co-occur (English, Greek, ...)

# Loss: encourage co-occurring phonemes to be close
```

**What this learns**: Typological universals

---

## 🔬 Architecture Options

### Option A: Multi-Task Neural Network

```python
import torch
import torch.nn as nn

class PhonemeEmbedding(nn.Module):
    def __init__(self, num_phonemes, phoible_features_dim=38, embedding_dim=32):
        super().__init__()

        # Initialize from Phoible features
        self.feature_encoder = nn.Linear(phoible_features_dim, embedding_dim)

        # Learnable embeddings
        self.phoneme_embeddings = nn.Embedding(num_phonemes, embedding_dim)

        # Task-specific heads
        self.context_predictor = nn.Linear(embedding_dim, num_phonemes)
        self.allomorph_predictor = nn.Linear(embedding_dim, num_allomorphs)
        self.feature_reconstructor = nn.Linear(embedding_dim, phoible_features_dim)

    def forward(self, phoneme_id, phoible_features):
        # Combine learned embedding with feature-based encoding
        learned_emb = self.phoneme_embeddings(phoneme_id)
        feature_emb = self.feature_encoder(phoible_features)
        embedding = learned_emb + feature_emb

        return embedding

    def predict_context(self, embedding):
        return self.context_predictor(embedding)

    def predict_allomorph(self, embedding, inflection_type):
        return self.allomorph_predictor(embedding)

    def reconstruct_features(self, embedding):
        return self.feature_reconstructor(embedding)
```

**Training:**
```python
# Multi-task loss
loss = (
    lambda_1 * context_loss +
    lambda_2 * allomorph_loss +
    lambda_3 * clustering_loss +
    lambda_4 * reconstruction_loss +
    lambda_5 * crossling_loss
)
```

---

### Option B: Contrastive Learning (Simpler)

```python
from sentence_transformers import SentenceTransformer, losses

# Use Phoible features as initial representation
# Learn refinement through contrastive learning

model = SentenceTransformer(modules=[
    nn.Linear(38, 64),  # Phoible features → hidden
    nn.ReLU(),
    nn.Linear(64, 32),  # hidden → embedding
])

# Contrastive examples:
# Positive: phonemes in same natural class
# Negative: phonemes in different classes

train_examples = [
    # (phoneme1_features, phoneme2_features, similarity_score)
    ([features_p], [features_t], 0.8),  # both voiceless stops
    ([features_p], [features_b], 0.3),  # differ in voicing
    ([features_p], [features_a], 0.0),  # consonant vs vowel
]
```

---

## 📊 Evaluation: Intrinsic + Extrinsic

### Intrinsic (Embedding Quality)

1. **Phonological Similarity**
   ```python
   # Are similar phonemes close?
   assert cosine_sim(emb(/p/), emb(/t/)) > cosine_sim(emb(/p/), emb(/a/))
   ```

2. **Natural Class Recovery**
   ```python
   # Do embeddings cluster by natural classes?
   voiceless_stops = cluster([/p/, /t/, /k/])
   assert coherence(voiceless_stops) > threshold
   ```

3. **Feature Reconstruction**
   ```python
   # Can we recover Phoible features?
   predicted_features = decode(embedding)
   assert accuracy(predicted_features, true_features) > baseline
   ```

### Extrinsic (Downstream Tasks)

1. **Allomorph Prediction** (Option A: Rule Learning)
   ```python
   # Use embeddings to predict plural/past tense
   allomorph = predict(stem_final_embedding, "plural")
   accuracy = eval_on_sigmorphon()
   ```

2. **Rhyme Detection** (Option C: Similarity)
   ```python
   # Do rhyming words have similar final phoneme embeddings?
   rhyme_accuracy = eval_rhyme_detection(embeddings)
   ```

3. **Dialect Classification** (Option D: Variation)
   ```python
   # Can we classify dialect from phoneme inventory embeddings?
   dialect = classify([emb for emb in inventory])
   ```

4. **G2P Error** (Option E: Pronunciation)
   ```python
   # Use embeddings as targets for G2P
   g2p_model.train(chars, target_embeddings)
   phoneme_error_rate = evaluate()
   ```

5. **Typological Prediction** (Option B: Universals)
   ```python
   # Predict if language has phoneme X given inventory
   has_phoneme = predict_from_inventory_embeddings()
   ```

---

## 🎯 Why This is Powerful

### 1. **Unified Representation**
   - Learn once, use everywhere
   - Transfer across tasks
   - Consistent semantics

### 2. **Interpretable**
   - Can visualize (t-SNE)
   - Can probe (what do dimensions mean?)
   - Can compare to Phoible features

### 3. **Efficient**
   - 32-dim embeddings << 38-dim sparse features
   - Faster computation
   - Better generalization

### 4. **Learnable**
   - Automatically discover patterns
   - Adapt to data
   - Not limited by linguistic theory

### 5. **Extensible**
   - Add new objectives easily
   - Incorporate acoustic features later
   - Multi-lingual transfer

---

## 🚀 Implementation Plan

### Phase 1: Simple Baseline (1 week)
```python
# Learn embeddings from Phoible features only
# Use PCA or autoencoder: 38-dim → 32-dim
# Evaluate on similarity tasks

from sklearn.decomposition import PCA
embeddings = PCA(n_components=32).fit_transform(phoible_features)
```

**Deliverable**: Baseline embeddings for all Phoible phonemes

---

### Phase 2: Context-Based Learning (1 week)
```python
# Add phonological context from CMU Dict
# Train skip-gram style model
# Phoneme + context → predict neighbors

model = train_skipgram(cmu_pronunciations, init=phoible_features)
```

**Deliverable**: Context-aware embeddings

---

### Phase 3: Multi-Task Learning (2 weeks)
```python
# Add morphological task (SIGMORPHON)
# Add contrastive learning
# Add feature reconstruction

model = train_multitask(
    context_data=cmu_pronunciations,
    morphology_data=sigmorphon,
    features=phoible_features,
    cross_ling=phoible_inventories
)
```

**Deliverable**: Fully-trained phoneme embeddings

---

### Phase 4: Evaluation & Analysis (1 week)
```python
# Intrinsic: clustering, similarity, reconstruction
# Extrinsic: downstream tasks (rule learning, rhyme, etc.)
# Visualization: t-SNE, feature importance
# Comparison: embeddings vs Phoible features

results = evaluate_all_tasks(embeddings)
```

**Deliverable**: Paper-ready results, plots, analysis

---

## 📦 Code Structure

```python
phonolex/
├── embeddings/
│   ├── models.py                # Embedding architectures
│   ├── training.py              # Multi-task training loop
│   ├── objectives.py            # Loss functions
│   └── data_loaders.py          # Unified data loading
│
├── evaluation/
│   ├── intrinsic.py             # Similarity, clustering
│   ├── extrinsic.py             # Downstream tasks
│   └── visualization.py         # t-SNE, plots
│
├── tasks/                       # Downstream applications
│   ├── allomorph_prediction.py  # Rule learning
│   ├── rhyme_detection.py       # Similarity
│   ├── dialect_classification.py
│   └── g2p.py
│
└── core/                        # Reusable (already built)
    ├── phoneme_vectorizer.py
    ├── phoible_loader.py
    └── mappings.py
```

---

## 💡 Key Insight: Embeddings ARE the Product

Instead of building separate systems for each task, we build:

**ONE**: Phoneme embedding model
**MANY**: Downstream applications that use the embeddings

This is exactly how modern NLP works:
- Train embeddings (Word2Vec, BERT, GPT)
- Use for everything (classification, similarity, generation)

**Phoneme embeddings are the foundation for computational phonology!**

---

## 🎯 Recommended Next Step

**Build the unified data loader** that creates training examples for embedding learning:

```python
class PhonemeEmbeddingDataset:
    """
    Loads and combines all data sources for phoneme embedding training
    """
    def __init__(self):
        self.phoible = load_phoible_features()
        self.cmu_dict = load_cmu_dict()
        self.sigmorphon = load_sigmorphon()
        self.arpa_to_ipa = load_mappings()

    def get_context_examples(self):
        """Phonological context pairs for skip-gram"""
        # From CMU pronunciations
        pass

    def get_morphology_examples(self):
        """Stem + inflection for allomorph prediction"""
        # From SIGMORPHON
        pass

    def get_contrastive_pairs(self):
        """Positive/negative phoneme pairs"""
        # From Phoible features
        pass

    def get_inventory_examples(self):
        """Cross-linguistic co-occurrence"""
        # From Phoible inventories
        pass
```

This data loader feeds into ANY embedding model we want to try!

**Want me to build this?** It's the foundation for everything.
