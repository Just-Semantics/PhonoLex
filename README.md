# PhonoLex

Phonological analysis toolkit with hierarchical embeddings and client-side web application.

[![Version](https://img.shields.io/badge/version-2.1.0--beta-blue.svg)](CHANGELOG.md)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE.txt)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](python/pyproject.toml)
[![TypeScript](https://img.shields.io/badge/typescript-5.0+-blue.svg)](webapp/frontend/package.json)

## Overview

PhonoLex implements a 4-layer embedding pipeline for phonological analysis, combining universal phonological features from Phoible with learned contextual representations. The system produces position-aware syllable embeddings that discriminate anagrams using onset-nucleus-coda structure, trained via next-phoneme prediction.

**Components**:
- 4-layer hierarchical embeddings (raw features → contextual syllable representations)
- Client-side web application (no backend required)
- 24,744 words with psycholinguistic properties and embeddings
- Phonological features for 2,716 languages from Phoible
- Tools for minimal pairs, rhyme detection, and similarity search

### Word Similarity Results

| Word Pair | Similarity | Type |
|-----------|------------|------|
| cat - bat | 0.995 | Rhyme |
| make - take | 0.987 | Rhyme |
| computer - commuter | 0.797 | Phonologically similar |
| cat - act | 0.288 | Anagram |
| cat - dog | 0.225 | Unrelated |

The hierarchical syllable structure discriminates anagrams: `cat` [k-æ-t] has a single onset consonant while `act` [∅-æ-kt] has zero onset and a consonant cluster coda, producing different 384-dimensional embeddings despite identical phoneme content

---

## Quick Start

### Web Application

The web application is fully client-side and runs in the browser without server dependencies.

**Run locally**:
```bash
cd webapp/frontend
npm install
npm run dev
```

Access at http://localhost:3000. Available tools:
- Word search with phonological and psycholinguistic filters
- Minimal pair generation for phoneme contrasts
- Rhyme detection (perfect, slant, syllable-based)
- Word list export for research or clinical applications
- Phonological similarity comparison

### Python Library

The Python library is required only for building embeddings from scratch. The web application uses pre-computed embeddings

```bash
git clone https://github.com/yourusername/PhonoLex.git
cd PhonoLex

# Install for embedding generation
pip install -e ./python[build]

# Or for development (includes testing, linting)
pip install -e ./python[dev]
```

### Using Pre-computed Embeddings

```python
import torch

# Load Layer 4 syllable embeddings
checkpoint = torch.load('embeddings/layer4/syllable_embeddings_filtered.pt')
word_to_syllable_embeddings = checkpoint['word_to_syllable_embeddings']

# Get embeddings
cat = word_to_syllable_embeddings['cat']  # List of 384-dim numpy arrays (one per syllable)
bat = word_to_syllable_embeddings['bat']
act = word_to_syllable_embeddings['act']

# Compute similarity with hierarchical soft Levenshtein distance
# See scripts/build_filtered_layer4_embeddings.py for implementation
```

---

## Four-Layer Architecture

The embedding pipeline consists of four sequential layers. Only Layer 3 requires training; other layers are deterministic transformations

```
Layer 1: Raw Phoible Features (38-dim ternary: +, -, 0)
    ↓ normalization & interpolation
Layer 2: Normalized Feature Vectors (76-dim endpoints / 152-dim trajectories)
    ↓ transformer learning + phonotactic patterns
Layer 3: Contextual Phoneme Embeddings (128-dim) ⭐ ONLY TRAINED LAYER
    ↓ syllable aggregation (onset-nucleus-coda structure)
Layer 4: Hierarchical Syllable Embeddings (384-dim) ⭐ MAIN PRODUCTION EMBEDDINGS
    ↓ soft Levenshtein distance on syllable sequences
Word Similarity
```

### Layer Details

#### Layer 1: Raw Phoible Features (38-dim ternary)
- **NOT trained** - extracted from Phoible database
- **Output**: `embeddings/layer1/phoible_features.csv` (59 KB)
- **Coverage**: 94 English phonemes
- **Format**: Ternary features (+, -, 0)
- **Build time**: <1 second
- **Use for**: Cross-linguistic comparison, feature analysis

#### Layer 2: Normalized Feature Vectors (76-dim / 152-dim)
- **NOT trained** - deterministic transformation
- **Output**: `embeddings/layer2/normalized_76d.pkl` (59 KB), `normalized_152d.pkl` (115 KB)
- **76-dim**: Endpoint vectors (start + end positions)
- **152-dim**: Trajectory vectors (4-timestep interpolation for diphthongs)
- **Build time**: ~5 seconds
- **Use for**: Continuous phoneme similarity, diphthong modeling, Layer 3 initialization

#### Layer 3: Contextual Phoneme Embeddings (128-dim) ⭐
- **TRAINED** - only layer that requires training
- **Model**: `models/layer3/model.pt` (2.4 MB)
- **Architecture**: PhonoLexBERT transformer (3 layers, 4 attention heads)
- **Task**: Next-phoneme prediction (self-supervised)
- **Accuracy**: 99.98% on 147K words
- **Training data**: CMU Dictionary (125K) + ipa-dict (22K)
- **Initialization**: Layer 2 normalized vectors
- **Training time**: ~10 minutes on Apple Silicon
- **Use for**: Contextual phoneme analysis, foundation for Layer 4

#### Layer 4: Hierarchical Syllable Embeddings (384-dim) ⭐
- **NOT trained** - built from frozen Layer 3 model
- **Output**:
  - `embeddings/layer4/syllable_embeddings_filtered.pt` (137 MB) - **RECOMMENDED**
  - `embeddings/layer4/syllable_embeddings_filtered_quantized.pt` (42 MB) - Int8 quantized
  - `embeddings/layer4/syllable_embeddings.pt` (1.0 GB) - Unfiltered (deprecated)
- **Structure**: Onset (128-dim) + Nucleus (128-dim) + Coda (128-dim) = 384-dim per syllable
- **Vocabulary**: 24,744 words (filtered with psycholinguistic norms)
- **Build time**: ~5 minutes on CPU
- **Use for**: Word similarity, rhyme detection, anagram discrimination

### Building the Pipeline

```bash
# Layer 1: Extract Phoible features (<1 second)
python scripts/compute_layer1_phoible_features.py

# Layer 2: Compute normalized vectors (~5 seconds)
python scripts/compute_layer2_normalized_vectors.py

# Layer 3: Train contextual embeddings (~10 minutes on Apple Silicon)
python scripts/train_layer3_contextual_embeddings.py

# Layer 4: Build syllable embeddings (~5 minutes on CPU) - RECOMMENDED
python scripts/build_filtered_layer4_embeddings.py

# Optional: Quantize to int8 (75% additional size reduction)
python scripts/quantize_embeddings.py

# Export data for web application
python scripts/export_clientside_data.py
```

**Complete documentation**: See [docs/EMBEDDINGS_ARCHITECTURE.md](docs/EMBEDDINGS_ARCHITECTURE.md)

---

## Web Application (v2.1 - Client-Side)

### Architecture

The web application runs entirely client-side without backend or database dependencies.

**Stack**:
- Frontend: React 18 + TypeScript + Material-UI
- Data: Static JSON files (~90 MB, gzips to ~45 MB)
- Computation: In-browser (cosine similarity, pattern matching)
- Deployment: Static host (Netlify, Cloudflare Pages, Vercel, GitHub Pages)

**Architecture implications**:
- No server costs or maintenance
- No database (PostgreSQL, pgvector) required
- In-memory queries without network latency
- Deployable as static files

### Features

**Search**:
- Phonological property filters (phoneme count, syllable count, complexity)
- Psycholinguistic norm filters (frequency, AoA, concreteness, VAD)
- Pattern matching with wildcards
- Vector similarity search

**Minimal Pairs Generator**:
- Phoneme contrast specification (e.g., /t/ vs /d/)
- Word length and complexity filters
- Export for SLP/clinical applications

**Rhyme Detection**:
- Perfect rhymes (cat - bat)
- Slant rhymes (cat - cut)
- Syllable-based rhymes
- Configurable matching strictness

**Norm-Filtered Lists**:
- Export vocabularies with psycholinguistic properties
- Developmental appropriateness filtering (AoA)
- Concreteness/abstractness filtering
- Emotional valence filtering

**Word Comparison**:
- Phonological similarity computation
- Feature-level analysis
- Syllable structure visualization

### Data Coverage

- 24,744 words (filtered vocabulary)
- 100% frequency data (SUBTLEXus)
- 97.8% concreteness ratings (Brysbaert)
- 54.0% VAD ratings (valence, arousal, dominance)
- 18.6% Glasgow norms (AoA, imageability, familiarity)

### Development

```bash
cd webapp/frontend

# Install dependencies
npm install

# Development server (http://localhost:3000)
npm run dev

# Production build
npm run build

# Preview production build
npm run preview

# Type checking
npm run type-check

# Linting
npm run lint
npm run lint:fix
```

### Deployment

**Netlify** (recommended - already configured):
```bash
netlify deploy --prod
```

**Manual deployment**:
```bash
cd webapp/frontend
npm run build
# Deploy dist/ folder to any static host
```

**Configuration**: See [netlify.toml](netlify.toml) for build settings.

---

## Repository Structure

```
PhonoLex/
├── README.md                        # This file
├── CLAUDE.md                        # Project instructions for Claude Code
├── CHANGELOG.md                     # Version history
├── LICENSE.txt                      # MIT License
├── netlify.toml                     # Netlify deployment config
│
├── python/                          # Python dependencies (local dev only)
│   ├── pyproject.toml              # Modern Python packaging (PEP 621)
│   ├── requirements.txt            # Legacy pip requirements
│   └── README.md                   # Python setup guide
│
├── src/phonolex/                    # Core library (embeddings, models, utils)
│   ├── embeddings/
│   │   ├── english_data_loader.py  # CMU + ipa-dict loader (147K words)
│   │   └── ...                     # Other data loaders
│   ├── models/
│   │   └── phonolex_bert.py        # PhonoLexBERT transformer (Layer 3)
│   └── utils/
│       └── syllabification.py      # Onset-nucleus-coda parser
│
├── scripts/                         # Layer generation pipeline
│   ├── compute_layer1_phoible_features.py       # Layer 1 (<1 sec)
│   ├── compute_layer2_normalized_vectors.py     # Layer 2 (~5 sec)
│   ├── train_layer3_contextual_embeddings.py    # Layer 3 (~10 min) ⭐
│   ├── build_filtered_layer4_embeddings.py      # Layer 4 (~5 min) - RECOMMENDED
│   ├── build_layer4_syllable_embeddings.py      # Layer 4 unfiltered (deprecated)
│   ├── quantize_embeddings.py                   # Int8 quantization
│   └── export_clientside_data.py                # Export for webapp
│
├── embeddings/                      # Pre-computed embeddings
│   ├── layer1/phoible_features.csv              # 38-dim ternary (59 KB)
│   ├── layer2/normalized_76d.pkl                # 76-dim continuous (59 KB)
│   ├── layer2/normalized_152d.pkl               # 152-dim trajectories (115 KB)
│   ├── layer4/syllable_embeddings_filtered.pt   # 384-dim filtered (137 MB) ⭐
│   └── layer4/syllable_embeddings_filtered_quantized.pt  # Int8 (42 MB)
│
├── models/                          # Trained models
│   └── layer3/model.pt             # PhonoLexBERT (2.4 MB) ⭐
│
├── data/                            # Source data
│   ├── cmu/                        # CMU Dictionary (125K words)
│   ├── phoible/                    # Phoible database (2,716 languages)
│   ├── mappings/                   # ARPAbet ↔ IPA conversion
│   ├── learning_datasets/          # ipa-dict, SIGMORPHON
│   └── norms/                      # Psycholinguistic norms
│
├── webapp/                          # Web application (v2.1)
│   └── frontend/                   # React + TypeScript
│       ├── src/
│       │   ├── services/
│       │   │   ├── clientSideData.ts           # Main data service ⭐
│       │   │   ├── clientSideApiAdapter.ts     # API compatibility
│       │   │   └── phonolexApi.ts              # Exports adapter
│       │   ├── components/                     # UI components
│       │   └── types/phonology.ts              # TypeScript types
│       ├── public/data/                        # Static JSON data (~90 MB)
│       │   ├── word_metadata.json              # 24K words (14 MB)
│       │   ├── embeddings_quantized.json       # Int8 embeddings (75 MB)
│       │   ├── phonemes.json                   # Phoneme features (37 KB)
│       │   └── manifest.json                   # Data package metadata
│       └── dist/                               # Build output
│
├── docs/                            # Documentation
│   ├── INDEX.md                                # Documentation index
│   ├── EMBEDDINGS_ARCHITECTURE.md              # ⭐ Complete architecture
│   ├── CLIENT_SIDE_DATA_PACKAGE.md             # Client-side data format
│   ├── MIGRATION_TO_CLIENT_SIDE.md             # v2.0 → v2.1 migration
│   ├── VOCABULARY_FILTERING.md                 # Filtering strategy
│   ├── TRAINING_SCRIPTS_SUMMARY.md             # Script reference
│   └── development/
│       └── LEARNING_DATASETS.md                # Dataset documentation
│
└── archive/                         # Historical code (not for active use)
    ├── webapp_v1/                  # Flask backend (deprecated)
    └── webapp_v2_backend/          # FastAPI backend (archived Oct 2025)
```

---

## Data Sources & Coverage

### English Pronunciations

**CMU Pronouncing Dictionary**:
- 125,764 words with ARPAbet transcriptions
- Primary pronunciations only (no variants)
- Stress markers: 0 (unstressed), 1 (primary), 2 (secondary)
- Dialect: General American English
- Converted to IPA for processing

**ipa-dict**:
- 22,000 additional words (US + UK variants)
- IPA transcriptions
- Supplements CMU coverage

**Total training corpus**: 147,000 words

### Universal Phonological Features (Phoible)

- **2,716 languages** worldwide
- **105,484 phonemes** across all languages
- **38 distinctive features** (Hayes 2009 + Moisik & Esling 2011)
- **94 English phonemes** extracted for Layer 1

### Psycholinguistic Norms

**SUBTLEXus** (100% coverage):
- Word frequency from subtitles
- Most ecologically valid frequency measure

**Brysbaert Concreteness** (97.8% coverage):
- Concreteness ratings (1-5 scale)
- Based on 40,000+ words

**Glasgow Norms** (18.6% coverage):
- Age of Acquisition (AoA)
- Imageability
- Familiarity

**VAD Ratings** (54.0% coverage):
- Valence (positive/negative emotion)
- Arousal (activation level)
- Dominance (control/power)

### Morphology

**SIGMORPHON 2020**:
- English inflectional paradigms
- Task 0: Lemma + features → inflected form

---

## Applications

### 1. Speech-Language Pathology (SLP)

**Minimal Pairs for Articulation Therapy**:
```python
# Generate /s/ vs /θ/ minimal pairs for therapy
pairs = find_minimal_pairs(phoneme1='s', phoneme2='θ',
                          word_length='short', complexity='low')
# → sank/thank, sick/thick, sink/think
```

**Developmentally Appropriate Word Lists**:
```python
# Export words acquired by age 5
early_words = filter_words(aoa_max=5.0, frequency_min=10.0)
```

### 2. Phonological Research

**Similarity Studies**:
```python
# Compare similarity metrics
similarity = compute_similarity('computer', 'commuter')
# Layer 4: 0.794 (hierarchical syllable-based)
```

**Phonotactic Analysis**:
```python
# Check if phoneme sequence is valid
prob = model.predict_next_phoneme(['s', 't', 'r'])  # High (English allows)
prob = model.predict_next_phoneme(['ŋ', 'k', 'r'])  # Low (invalid in English)
```

### 3. Natural Language Processing

**Rhyme Generation for Poetry**:
```python
# Find rhymes for word generation
rhymes = find_rhymes('cat', mode='perfect', limit=20)
# → bat, mat, fat, sat, hat, rat, pat, chat, flat, that
```

**Spelling Error Correction**:
```python
# Find phonologically similar words for spell-check
suggestions = find_similar('recieve', threshold=0.8)
# → receive, deceive
```

### 4. Language Learning

**Pronunciation Comparison**:
```python
# Compare learner vs target pronunciation
similarity = compare_pronunciations(
    learner=['k', 'o', 'm', 'p', 'j', 'u', 't', 'ɚ'],  # Learner's /kompjutɚ/
    target=['k', 'ə', 'm', 'p', 'j', 'u', 't', 'ɚ']     # Native /kəmpjutɚ/
)
# Low similarity indicates mispronunciation
```

**Minimal Pair Drills**:
```python
# Generate contrast sets for phoneme training
pairs = find_minimal_pairs(phoneme1='r', phoneme2='l')
# → right/light, read/lead, rock/lock
```

### 5. Computational Linguistics

**Phonological Embeddings for Downstream Tasks**:
```python
# Use Layer 4 embeddings as features
word_embedding = get_layer4_embedding('cat')  # 384-dim per syllable
# Concatenate or pool for classification tasks
```

**Cross-Linguistic Phonological Transfer**:
```python
# Compare phonemes across languages using Layer 1
english_t = get_phoible_features('t', language='eng')
spanish_t = get_phoible_features('t', language='spa')
# Analyze feature differences
```

---

## Performance Metrics

### Layer 3 Training Results

- **Next-phoneme prediction**: 99.98% accuracy
- **Training time**: ~10 minutes on Apple Silicon (M1/M2)
- **Dataset**: 147K words (CMU + ipa-dict)
- **Model size**: 2.4 MB
- **Inference speed**: ~0.1 ms per word (CPU)

### Layer 4 Similarity Performance

Verified results on Layer 4 embeddings:

| Type | Examples | Similarity Range |
|------|----------|------------------|
| Rhymes | cat-bat: 0.995, make-take: 0.987 | 0.99+ |
| Phonologically similar | computer-commuter: 0.797 | 0.75-0.85 |
| Anagrams | cat-act: 0.288 | ~0.30 |
| Unrelated | cat-dog: 0.225 | 0.20-0.30 |

### Vocabulary Filtering Impact (v2.1)

| Metric | Unfiltered | Filtered | Reduction |
|--------|-----------|----------|-----------|
| Words | 48,000 | 24,744 | 49% |
| Embeddings size | 1.0 GB | 137 MB | 86% |
| Quantized size | 333 MB | 42 MB | 87% |
| Frequency coverage | 100% | 100% | - |
| Concreteness coverage | 84% | 97.8% | +13.8% |

Filtering criterion: Frequency + at least one additional psycholinguistic norm (concreteness, AoA, imageability, familiarity, or VAD). This ensures research and clinical applications have high-quality psycholinguistic data beyond frequency alone.

See [docs/VOCABULARY_FILTERING.md](docs/VOCABULARY_FILTERING.md) for analysis.

---

## Comparison with Alternative Approaches

### Anagram Discrimination (cat vs act)

| Approach | Similarity | Note |
|----------|-----------|------|
| Skip-gram (bag-of-phonemes) | 0.99 | Position-invariant |
| Mean pooling | 0.94 | Order information lost |
| First-mid-last concatenation | 0.31 | Partial position encoding |
| Sequential (Levenshtein on phonemes) | 0.68 | No syllable structure |
| Hierarchical syllable (Layer 4) | 0.288 | Syllable boundary encoding |

### Mechanism

Syllable structure provides discrimination:

**cat** [kæt]:
- onset=[k], nucleus=æ, coda=[t]
- Template: C-V-C

**act** [ækt]:
- onset=[], nucleus=æ, coda=[k,t]
- Template: V-CC

Different syllable structures produce different 384-dim embeddings (128-dim for onset, nucleus, coda each). The model learns these distinctions from the 4-layer hierarchy without explicit syllable boundary labels

---

## Technical Details

### Model Architecture (Layer 3)

```python
PhonoLexBERT(
    vocab_size=42,              # 39 phonemes + special tokens
    d_model=128,                # Embedding dimension
    nhead=4,                    # Attention heads
    num_layers=3,               # Transformer layers
    dim_feedforward=512,        # FFN hidden size
    max_len=36,                 # Max phoneme sequence length
    dropout=0.1
)
```

**Training Configuration**:
- Task: Next-phoneme prediction (self-supervised)
- Loss: CrossEntropyLoss
- Optimizer: Adam (lr=0.001)
- Batch size: 64
- Epochs: 20
- Time: ~10 minutes on Apple Silicon

**Initialization**:
- Phoneme embeddings: Layer 2 normalized vectors (76-dim → 128-dim)
- Positional encoding: Sinusoidal (learnable option available)

### Syllable Embeddings (Layer 4)

```python
# For each syllable in word
syllable_embedding = concat([
    mean_pool(onset_phonemes),     # 128-dim (or zeros if no onset)
    nucleus_phoneme,               # 128-dim (always present)
    mean_pool(coda_phonemes)       # 128-dim (or zeros if no coda)
])  # Total: 384-dim per syllable

# Word representation
word = [syllable_1, syllable_2, ..., syllable_n]  # Variable length
```

**Similarity Computation**:
Soft Levenshtein distance on syllable embedding sequences:
- Match cost: `1 - cosine_similarity(syll1, syll2)`
- Insert/delete cost: `1.0`
- Final similarity: `1 - (edit_distance / max(len1, len2))`

### Client-Side Data Format (v2.1)

**word_metadata.json** (14 MB):
```json
{
  "words": [
    {
      "word": "cat",
      "ipa": "kæt",
      "syllables": [{"onset": ["k"], "nucleus": "æ", "coda": ["t"]}],
      "phoneme_count": 3,
      "syllable_count": 1,
      "freq": 123.45,
      "concreteness": 4.82,
      "aoa": 2.5,
      "valence": 6.2,
      "arousal": 3.8,
      "dominance": 5.1
    }
  ]
}
```

**embeddings_quantized.json** (75 MB):
```json
{
  "cat": {
    "syllables": [
      {
        "onset": [23, 45, 67, ...],    // 128 int8 values
        "nucleus": [12, 34, 56, ...],   // 128 int8 values
        "coda": [78, 90, 12, ...]       // 128 int8 values
      }
    ],
    "scale": 0.0123  // Dequantization: value * scale
  }
}
```

See [docs/CLIENT_SIDE_DATA_PACKAGE.md](docs/CLIENT_SIDE_DATA_PACKAGE.md) for complete format specification.

---

## Potential Extensions

### Implementation

1. Stress-aware weighting for stressed syllables in similarity computation
2. Cross-lingual training on multiple languages from Phoible
3. Perceptual validation against human similarity judgments
4. Progressive Web App features (offline support, caching)

### Research

1. Generative models for valid phoneme sequences
2. Learning sub-phonemic distinctive features from data
3. Prosody integration (rhythm, intonation, tone)
4. Joint semantic-phonological embeddings
5. End-to-end syllabification with transformers

---

## Documentation

### Primary Documentation

- [docs/EMBEDDINGS_ARCHITECTURE.md](docs/EMBEDDINGS_ARCHITECTURE.md) - Complete 4-layer architecture
- [docs/CLIENT_SIDE_DATA_PACKAGE.md](docs/CLIENT_SIDE_DATA_PACKAGE.md) - Client-side data format
- [docs/VOCABULARY_FILTERING.md](docs/VOCABULARY_FILTERING.md) - Filtering strategy
- [docs/TRAINING_SCRIPTS_SUMMARY.md](docs/TRAINING_SCRIPTS_SUMMARY.md) - Script reference

### Development

- [docs/MIGRATION_TO_CLIENT_SIDE.md](docs/MIGRATION_TO_CLIENT_SIDE.md) - v2.0 → v2.1 migration
- [docs/development/LEARNING_DATASETS.md](docs/development/LEARNING_DATASETS.md) - Dataset documentation
- [docs/INDEX.md](docs/INDEX.md) - Documentation index

### Historical

- [docs/ARCHITECTURE_V2.md](docs/ARCHITECTURE_V2.md) - Archived v2.0 backend architecture

---

## Contributing

Areas for contribution:

- Multilingual support (extend beyond English)
- Additional psycholinguistic norms (integrate more datasets)
- Performance optimization (similarity computation)
- New phonological tools (stress patterns, syllable complexity metrics)
- Testing (unit tests, integration tests, user testing)

Open an issue to discuss major changes before submitting pull requests.

---

## References

### Data Sources

- **Phoible**: Moran, Steven & McCloy, Daniel (eds.) 2019. PHOIBLE 2.0. Jena: Max Planck Institute for the Science of Human History. https://phoible.org/
- **CMU Pronouncing Dictionary**: http://www.speech.cs.cmu.edu/cgi-bin/cmudict
- **ipa-dict**: https://github.com/open-dict-data/ipa-dict
- **SIGMORPHON**: https://sigmorphon.github.io/sharedtasks/2020/
- **SUBTLEXus**: Brysbaert, M., & New, B. (2009). Moving beyond Kučera and Francis.
- **Concreteness**: Brysbaert, M., Warriner, A.B., & Kuperman, V. (2014). Concreteness ratings.
- **Glasgow Norms**: Scott, G.G., Keitelova, S., Houghton, J. & The Glasgow Norms Associates (2019).
- **VAD**: Warriner, A.B., Kuperman, V., & Brysbaert, M. (2013). Norms of valence, arousal, and dominance.

### Theoretical Foundations

- **Hayes, Bruce** (2009). *Introductory Phonology*. Wiley-Blackwell.
- **Moisik, Scott R. & Esling, John H.** (2011). The 'whole larynx' approach to laryngeal features. *Proceedings of the International Congress of Phonetic Sciences XVII*, 1406-1409.
- **Chomsky, Noam & Halle, Morris** (1968). *The Sound Pattern of English*. Harper & Row.

---

## Citation

If you use PhonoLex in your research, please cite:

```bibtex
@software{phonolex2025,
  title = {PhonoLex: Hierarchical Phonological Embeddings with Client-Side Analysis},
  author = {[Your Name]},
  year = {2025},
  version = {2.1.0-beta},
  url = {https://github.com/yourusername/PhonoLex},
  note = {Four-layer phonological embedding pipeline with onset-nucleus-coda structure}
}
```

---

## License

- **Code**: MIT License (see [LICENSE.txt](LICENSE.txt))
- **Phoible data**: Creative Commons Attribution-ShareAlike 3.0 Unported License
- **CMU Pronouncing Dictionary**: Public domain
- **Other datasets**: See individual dataset licenses in [docs/development/LEARNING_DATASETS.md](docs/development/LEARNING_DATASETS.md)

---

## Acknowledgments

**Data sources**:
- Phoible Project (Max Planck Institute for the Science of Human History)
- CMU Speech Group (Carnegie Mellon University)
- SIGMORPHON shared tasks
- Glasgow Norms Associates
- Marc Brysbaert's psycholinguistics lab

**Technologies**:
- PyTorch (model training)
- React + TypeScript (web application)
- Material-UI (UI components)
- NumPy & SciPy (numerical operations)
- Vite (frontend build tool)

---

**Version 2.1.0-beta** | [Changelog](CHANGELOG.md) | [Documentation](docs/INDEX.md) | [GitHub](https://github.com/yourusername/PhonoLex)
