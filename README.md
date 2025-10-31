# PhonoLex

**Hierarchical phonological word embeddings learned from natural phoneme sequences**

A complete toolkit for phonological analysis combining universal phonological features (Phoible) with learned contextual representations.

---

## What is PhonoLex?

PhonoLex provides:
- **Hierarchical word embeddings** that capture phonological similarity naturally
- **Universal phonological features** covering 2,716 languages
- **Position-aware representations** that discriminate anagrams (cat ≠ act)
- **No contrastive learning required** - learns from sequences alone

### Key Results

| Word Pair | Similarity | Type |
|-----------|------------|------|
| cat - bat | 0.993 | rhyme ✅ |
| **cat - act** | **0.200** | **anagram (discriminated!)** ✅ |
| computer - commuter | 0.794 | sound-alike ✅ |
| cat - dog | 0.298 | unrelated ✅ |

---

## Quick Start

### Installation (For Embedding Generation)

**Note**: The web application is fully client-side and doesn't require Python. Only install Python dependencies if you're building embeddings from scratch.

```bash
git clone https://github.com/yourusername/PhonoLex.git
cd PhonoLex

# Install for embedding generation
pip install -e ./python[build]

# Or for full development
pip install -e ./python[all]
```

### Web Application (No Installation Required)

The interactive web app runs entirely in your browser:

```bash
cd webapp/frontend
npm install
npm run dev
```

Visit http://localhost:3000 to explore phonological relationships, find minimal pairs, and analyze word similarity - all client-side!

### Demo: Find Similar Words

```python
import torch

# Load Layer 4 syllable embeddings
checkpoint = torch.load('embeddings/layer4/syllable_embeddings.pt')
word_to_syllable_embeddings = checkpoint['word_to_syllable_embeddings']

# Get embeddings
cat = word_to_syllable_embeddings['cat']
bat = word_to_syllable_embeddings['bat']
act = word_to_syllable_embeddings['act']

# Compute similarity (see scripts/build_layer4_syllable_embeddings.py)
# bat:  0.993 (rhyme)
# mat:  0.994 (rhyme)
# fat:  0.992 (rhyme)
# act:  0.200 (anagram - properly low!)
```

### Building the Layer Pipeline

```bash
# Extract Layer 1 (Phoible features)
python scripts/compute_layer1_phoible_features.py

# Compute Layer 2 (normalized vectors)
python scripts/compute_layer2_normalized_vectors.py

# Train Layer 3 (contextual phoneme embeddings, ~10 minutes)
python scripts/train_layer3_contextual_embeddings.py

# Build Layer 4 (syllable embeddings, ~5 minutes)
python scripts/build_layer4_syllable_embeddings.py
```

---

## Architecture

For detailed architecture documentation, see [docs/EMBEDDINGS_ARCHITECTURE.md](docs/EMBEDDINGS_ARCHITECTURE.md).

### Core Representation Hierarchy

```
Layer 1: Raw Phoible Features (38-dim ternary)
    ↓ normalization & interpolation
Layer 2: Normalized Feature Vectors (76-dim / 152-dim)
    ↓ transformer learning + phonotactic patterns
Layer 3: Contextual Phoneme Embeddings (128-dim)
    ↓ syllable aggregation (onset-nucleus-coda)
Layer 4: Hierarchical Syllable Embeddings (384-dim)
    ↓ soft Levenshtein on syllable sequences
Word Similarity
```

### Four Embedding Layers

PhonoLex provides 4 sequential layers of phonological representation:

**Layer 1: Raw Phoible Features (38-dim)**
- Output: `embeddings/layer1/phoible_features.csv`
- Ternary features: +, -, 0 (positive, negative, not applicable)
- 94 English phonemes extracted from universal database
- **Use for**: Cross-linguistic comparison, feature analysis

**Layer 2: Normalized Feature Vectors (76-dim / 152-dim)**
- Output: `embeddings/layer2/normalized_76d.pkl`, `normalized_152d.pkl`
- 76-dim endpoints: Start + end positions for each feature
- 152-dim trajectories: 4-timestep interpolation for articulation dynamics
- **Use for**: Continuous phoneme similarity, Layer 3 initialization

**Layer 3: Contextual Phoneme Embeddings (128-dim)** ⭐ Only trained layer
- Model: `models/layer3/model.pt` (PhonoLexBERT transformer)
- Initialized from Layer 2, trained on 147K words
- Next-phoneme prediction (99.98% accuracy)
- **Use for**: Contextual phoneme analysis, foundation for Layer 4

**Layer 4: Hierarchical Syllable Embeddings (384-dim)** ⭐ Main production embeddings
- Output: `embeddings/layer4/syllable_embeddings.pt`
- Built from frozen Layer 3 + syllable structure
- Onset-nucleus-coda (128-dim each)
- **Use for**: Word similarity, rhyme detection, anagram discrimination

### What Makes This Work

**Universal Phonological Prior**: Initialize with 76-dim articulatory features from Phoible covering voicing, place, manner, height, backness, etc.

**Next-Phoneme Prediction**: Model learns phonotactics naturally from sequences (99.98% accuracy) without artificial similarity targets.

**Syllable Structure**: Extracts onset-nucleus-coda for each syllable - `cat: [k-æ-t]` vs `act: [∅-æ-kt]` (different structures!)

**Hierarchical Similarity**: Compares syllable sequences using soft Levenshtein distance, naturally handling insertions/deletions.

---

## Models & Embeddings

### Production Outputs

**Layer 3 Model** (trained): `models/layer3/model.pt`
- PhonoLexBERT transformer (3 layers, 4 attention heads)
- 128-dim contextual phoneme embeddings
- Next-phoneme prediction (99.98% accuracy on 147K words)
- Initialized from Layer 2 normalized vectors
- **Use for**: Contextual phoneme analysis, foundation for Layer 4

**Layer 4 Embeddings** (computed): `embeddings/layer4/syllable_embeddings.pt` ⭐
- 384-dim syllable embeddings for 125K words
- Onset-nucleus-coda structure (128-dim each)
- Built from frozen Layer 3 model
- Best anagram discrimination (cat-act: 0.20)
- **Use for**: Word similarity, rhyme detection, pronunciation analysis

**Layer 1 & 2 Embeddings** (extracted/computed):
- `embeddings/layer1/phoible_features.csv` - Raw 38-dim ternary features
- `embeddings/layer2/normalized_76d.pkl` - 76-dim continuous vectors
- `embeddings/layer2/normalized_152d.pkl` - 152-dim trajectory vectors

### Archived Models

Historical experiments and training checkpoints are in `archive/` for reproducibility.

---

## Data Coverage

### English Pronunciations
- **125,764 words** from CMU Pronouncing Dictionary
- **Stress markers** preserved (primary, secondary, unstressed)
- **IPA transcriptions** with syllabification
- **39 phonemes** (ARPAbet → IPA)

### Universal Features (Phoible)
- **2,716 languages** worldwide
- **105,484 phonemes** total
- **38 distinctive features** (Hayes 2009 + Moisik & Esling 2011)
- **Cross-linguistic vector database** for similarity search

---

## Repository Structure

```
PhonoLex/
├── scripts/                       # Layer generation scripts
│   ├── compute_layer1_phoible_features.py      # Extract Layer 1
│   ├── compute_layer2_normalized_vectors.py    # Compute Layer 2
│   ├── train_layer3_contextual_embeddings.py   # Train Layer 3 (~10 min)
│   └── build_layer4_syllable_embeddings.py     # Build Layer 4 (~5 min)
│
├── embeddings/                    # Pre-computed embeddings (Layers 1, 2, 4)
│   ├── layer1/
│   │   └── phoible_features.csv               # 38-dim ternary (94 phonemes)
│   ├── layer2/
│   │   ├── normalized_76d.pkl                 # 76-dim endpoints
│   │   └── normalized_152d.pkl                # 152-dim trajectories
│   └── layer4/
│       └── syllable_embeddings.pt             # 384-dim syllables (1.0GB) ⭐
│
├── models/                        # Trained models (Layer 3 only)
│   └── layer3/
│       └── model.pt                           # PhonoLexBERT (2.4MB) ⭐
│
├── src/phonolex/                  # Core library
│   ├── embeddings/
│   │   └── english_data_loader.py             # CMU + ipa-dict loaders
│   ├── models/
│   │   └── phonolex_bert.py                   # PhonoLexBERT transformer
│   ├── utils/
│   │   └── syllabification.py                 # Onset-nucleus-coda parser
│   └── build_phonological_graph.py            # Graph construction
│
├── data/                          # Datasets & resources
│   ├── cmu/                                   # CMU Dictionary (125K words)
│   ├── phoible/                               # Phoible (2,716 languages)
│   ├── mappings/                              # ARPAbet ↔ IPA conversion
│   └── phonological_graph.pkl                 # 26K words, 56K typed edges
│
├── webapp/                        # Web application (v2.1 - Client-Side)
│   └── frontend/                              # React UI (static site)
│       ├── src/services/
│       │   ├── clientSideData.ts              # Main data service
│       │   └── clientSideApiAdapter.ts        # API compatibility layer
│       └── public/data/                       # Static JSON data (~88MB)
│
├── docs/                          # Documentation
│   ├── EMBEDDINGS_ARCHITECTURE.md             # ⭐ Complete 4-layer architecture
│   ├── CLIENT_SIDE_DATA_PACKAGE.md            # Client-side data format
│   ├── MIGRATION_TO_CLIENT_SIDE.md            # Migration guide
│   ├── ARCHITECTURE_V2.md                     # Archived v2.0 backend design
│   └── development/
│       └── LEARNING_DATASETS.md               # Dataset reference
│
└── archive/                       # Historical code
    ├── webapp_v1/                             # Flask backend (deprecated)
    ├── webapp_v2_backend/                     # FastAPI backend (archived Oct 2025)
    └── training_scripts_old/                  # Experimental scripts
```

---

## Applications

### 1. Phonological Similarity Search
```python
# Find words that sound like "computer"
similar_words = find_similar(model, "computer", threshold=0.8)
# → commuter, recruiter, disputer
```

### 2. Rhyme Detection
```python
# Find rhymes for poetry generation
rhymes = find_rhymes(model, "cat")
# → bat, mat, fat, sat, hat, rat
```

### 3. Pronunciation Error Detection
```python
# Compare learner vs native pronunciation
similarity = compare_pronunciations(learner_audio, native_word)
# Low similarity indicates mispronunciation
```

### 4. Phonotactic Validation
```python
# Check if phoneme sequence is valid in English
prob = model.next_phoneme_probability(["/s/", "/t/", "/r/"])  # High
prob = model.next_phoneme_probability(["/ŋ/", "/k/", "/r/"])  # Low
```

### 5. Morphological Analysis
```python
# Find related forms using phonological overlap
related = find_morphological_variants(model, "run")
# → running, runs, runner
```

---

## Technical Details

### Model Architecture
```python
HierarchicalPhonemeEncoder(
    num_phonemes=42,
    d_model=128,
    nhead=4,
    num_layers=3,
    max_len=36
)
```

**Training:**
- Task: Next-phoneme prediction
- Loss: CrossEntropyLoss
- Optimizer: Adam (lr=0.001)
- Epochs: 20
- Time: ~5 minutes on Apple Silicon

### Syllable Embeddings
```python
syllable_emb = concat([
    mean(onset_phonemes),   # 128-dim
    nucleus_phoneme,        # 128-dim
    mean(coda_phonemes)     # 128-dim
])  # Total: 384-dim per syllable
```

### Similarity Metric
Soft Levenshtein distance on syllable embedding sequences:
- Match cost: `1 - cosine_similarity(syll1, syll2)`
- Insert/delete cost: `1.0`
- Final similarity: `1 - (edit_distance / max_length)`

---

## Why Hierarchical Works Better

### Comparison with Flat Approaches

| Approach | cat-act | Why |
|----------|---------|-----|
| Skip-gram (bag-of-phonemes) | 0.99 | ❌ Position-invariant |
| Mean pooling | 0.94 | ❌ Order lost |
| First-mid-last concat | 0.31 | ⚠️ Better but lossy |
| Sequential (Levenshtein) | 0.68 | ⚠️ No syllable structure |
| **Hierarchical (Layer 4)** | **0.20** | ✅ **Captures syllable boundaries** |

### Key Insight
Syllable structure naturally discriminates anagrams:
- `cat`: [C-V-C] with single onset
- `act`: [V-CC] with zero onset, consonant cluster coda
- Different syllable templates → different representations

---

## Performance Metrics

### Phonotactic Learning
- Next-phoneme prediction: **99.98% accuracy**
- Model perfectly learned English phoneme sequences
- Captures valid/invalid consonant clusters

### Similarity Patterns (Layer 4)
**Rhymes** (should be high): ✅ 0.99+
**Anagrams** (should be low): ✅ 0.20
**Sound-alikes** (should be high): ✅ 0.79
**Unrelated** (should be low): ✅ 0.30

---

## Future Work

### Immediate Extensions
1. **Stress-aware weighting**: Emphasize stressed syllables
2. **Cross-lingual transfer**: Train on all Phoible languages
3. **Perceptual validation**: Compare with human judgments

### Research Directions
1. **Generative models**: Sample valid words
2. **Sub-phonemic features**: Learn distinctive features
3. **Prosody integration**: Rhythm and intonation
4. **Semantic-phonological joint embeddings**: Sound + meaning

---

## References

### Data Sources
- **Phoible**: Moran & McCloy (2019). PHOIBLE 2.0. Max Planck Institute. https://phoible.org/
- **CMU Dict**: http://www.speech.cs.cmu.edu/cgi-bin/cmudict
- **SIGMORPHON**: Morphological paradigms dataset

### Theoretical Foundations
- **Hayes, Bruce** (2009). *Introductory Phonology*. Wiley-Blackwell.
- **Moisik & Esling** (2011). The 'whole larynx' approach to laryngeal features.
- **Chomsky & Halle** (1968). *The Sound Pattern of English*.

---

## Citation

```bibtex
@software{phonolex2025,
  title = {PhonoLex: Hierarchical Phonological Embeddings},
  author = {[Your Name]},
  year = {2025},
  url = {https://github.com/yourusername/PhonoLex},
  note = {Phoneme → Syllable → Word hierarchy learned from sequences}
}
```

---

## License

- **Code**: MIT License (see [LICENSE.txt](LICENSE.txt))
- **Phoible data**: Creative Commons Attribution-ShareAlike 3.0
- **CMU Dict**: Public domain

---

## Acknowledgments

Built with data from:
- The Phoible Project (Max Planck Institute)
- CMU Speech Group
- SIGMORPHON shared tasks

Powered by:
- PyTorch (model training)
- ChromaDB (vector similarity search)
- NumPy & SciPy (numerical operations)

---

**PhonoLex** - Making phonology computable through hierarchical structure 🔊 → 🧠
