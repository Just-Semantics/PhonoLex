# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PhonoLex is a phonological analysis toolkit that combines universal phonological features (Phoible) with learned contextual representations. It provides hierarchical word embeddings learned from natural phoneme sequences, enabling phonological similarity analysis, rhyme detection, and pronunciation comparison.

**Key Innovation**: Position-aware syllable embeddings that properly discriminate anagrams (cat ≠ act) using onset-nucleus-coda structure, learned from next-phoneme prediction without contrastive learning.

## Project Structure (v2.1 - Client-Side)

The project uses a **modern Python package structure** with a **fully client-side web application**:

```
PhonoLex/
├── pyproject.toml           # Modern Python packaging config
├── README.md                # Main documentation
├── CLAUDE.md                # This file
├── requirements.txt         # Root dev dependencies
│
├── src/phonolex/            # Core library (phoneme embeddings, models, utils)
│   ├── __init__.py
│   ├── embeddings/          # Data loaders
│   ├── models/              # PhonoLexBERT and other models
│   └── utils/               # Syllabification, utilities
│
├── webapp/                  # Client-side web application (v2.1)
│   ├── __init__.py
│   └── frontend/            # React + TypeScript + MUI (static site)
│       ├── src/
│       │   ├── services/
│       │   │   ├── clientSideData.ts      # Main data service
│       │   │   ├── clientSideApiAdapter.ts # API compatibility layer
│       │   │   └── phonolexApi.ts         # Exports client-side adapter
│       │   └── components/
│       └── public/
│           └── data/        # Static JSON data files (~88MB, gzips to ~45MB)
│
├── scripts/                 # Build/training scripts
│   ├── compute_layer1_phoible_features.py
│   ├── compute_layer2_normalized_vectors.py
│   ├── train_layer3_contextual_embeddings.py
│   ├── build_filtered_layer4_embeddings.py  # Recommended (filtered vocab)
│   └── export_clientside_data.py            # Export data to webapp/frontend/public/data/
│
├── docs/                    # All documentation
│   ├── EMBEDDINGS_ARCHITECTURE.md
│   ├── CLIENT_SIDE_DATA_PACKAGE.md  # Client-side data format
│   ├── MIGRATION_TO_CLIENT_SIDE.md  # Migration guide from v2.0
│   ├── VOCABULARY_FILTERING.md      # Filtering strategy
│   └── ARCHITECTURE_V2.md   # v2.0 architecture (archived backend design)
│
├── data/                    # Source data (CMU, Phoible, mappings)
├── embeddings/              # Pre-computed embeddings
├── models/                  # Trained models
├── research/                # Research notebooks
└── archive/                 # Old code (v1 backend, v2.0 backend)
    ├── webapp_v1/           # Flask backend (deprecated)
    └── webapp_v2_backend/   # FastAPI + PostgreSQL (archived Oct 2025)
```

**Key Points**:
- **No backend required** - Fully static site deployment
- **Client-side computation** - All features run in browser
- Data pre-exported to JSON files in `webapp/frontend/public/data/`
- Backend code archived in `archive/webapp_v2_backend/`
- See [docs/MIGRATION_TO_CLIENT_SIDE.md](docs/MIGRATION_TO_CLIENT_SIDE.md) for migration details

## Development Environment Setup

### Python Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
```

### Web Application Setup

The web application is **fully client-side** (no backend required):

**Frontend (React + TypeScript)**:
```bash
cd webapp/frontend
npm install

# Run development server (default port 3000)
npm run dev

# Build for production (static files)
npm run build

# Preview production build
npm run preview

# Type checking
npm run type-check

# Linting
npm run lint
npm run lint:fix
```

**Deployment**: The built static files (`dist/`) can be deployed to any static hosting:
- Netlify (recommended)
- Cloudflare Pages
- GitHub Pages
- Vercel
- Any CDN or web server

## Core Commands

### Building Embeddings

The project has a 4-layer embedding pipeline. Only Layer 3 requires training:

```bash
# Layer 1: Extract Phoible features (38-dim ternary)
python scripts/compute_layer1_phoible_features.py
# Output: embeddings/layer1/phoible_features.csv

# Layer 2: Compute normalized vectors (76-dim / 152-dim)
python scripts/compute_layer2_normalized_vectors.py
# Output: embeddings/layer2/normalized_76d.pkl, normalized_152d.pkl

# Layer 3: Train contextual phoneme embeddings (128-dim)
# ~10 minutes on Apple Silicon
python scripts/train_layer3_contextual_embeddings.py
# Output: models/layer3/model.pt

# Layer 4: Build syllable embeddings (384-dim)
# ~5 minutes on CPU
# RECOMMENDED: Use filtered version (49% size reduction)
python scripts/build_filtered_layer4_embeddings.py
# Output: embeddings/layer4/syllable_embeddings_filtered.pt (~0.5GB vs 1.0GB)

# Optional: Quantize to int8 for deployment (75% additional reduction)
python scripts/quantize_embeddings.py
# Output: embeddings/layer4/syllable_embeddings_filtered_quantized.pt (~60MB compressed!)

# Alternative: Build with all words (deprecated)
python scripts/build_layer4_syllable_embeddings.py
# Output: embeddings/layer4/syllable_embeddings.pt (1.0GB)
```

**Vocabulary Filtering (v2.1+)**: By default, only words with **frequency + at least one additional psycholinguistic norm** (concreteness, AoA, imageability, familiarity, or VAD) are included. This reduces vocabulary from 48K → 24K words (49% reduction) while improving data quality for research and clinical applications.

See [docs/EMBEDDINGS_ARCHITECTURE.md](docs/EMBEDDINGS_ARCHITECTURE.md) for complete documentation.

### Exporting Client-Side Data

After building embeddings, export data for the web app:

```bash
# Export all data to webapp/frontend/public/data/
python scripts/export_clientside_data.py

# This creates:
# - words.json (~88 MB, gzips to ~45 MB)
# - phonemes.json (if needed)
# - Other supporting data files
```

### Running the Web Application

```bash
# Single terminal (no backend needed!)
cd webapp/frontend
npm run dev
# Runs on http://localhost:3000 (or 5173 depending on Vite version)

# Build for production
npm run build

# Preview production build
npm run preview
```

### Demo Scripts

```bash
# Quick similarity demo
python quick_demo.py

# Detailed word similarity analysis
python demo_word_similarity.py

# Check phonological graph structure
python check_graph_structure.py
```

## Architecture Overview

### Four-Layer Hierarchy

PhonoLex uses a hierarchical embedding approach with 4 distinct layers:

```
Layer 1: Raw Phoible Features (38-dim ternary: +, -, 0)
    ↓ normalization & interpolation
Layer 2: Normalized Feature Vectors (76-dim endpoints / 152-dim trajectories)
    ↓ transformer learning + phonotactic patterns
Layer 3: Contextual Phoneme Embeddings (128-dim, position-aware)
    ↓ syllable aggregation (onset-nucleus-coda structure)
Layer 4: Hierarchical Syllable Embeddings (384-dim)
    ↓ soft Levenshtein distance on syllable sequences
Word Similarity
```

**Key Insight**: Syllable structure naturally discriminates anagrams:
- `cat`: [k-æ-t] (single onset 'k')
- `act`: [∅-æ-kt] (zero onset, consonant cluster coda)
- Different syllable templates → low similarity (0.20)

### Four Embedding Layers

1. **Layer 1: Raw Phoible Features** (38-dim ternary: +, -, 0)
   - Output: `embeddings/layer1/phoible_features.csv`
   - Coverage: 94 English phonemes
   - Generated by: `scripts/compute_layer1_phoible_features.py`
   - Use: Cross-linguistic comparison, feature analysis

2. **Layer 2: Normalized Feature Vectors** (76-dim endpoints, 152-dim trajectories)
   - Output: `embeddings/layer2/normalized_76d.pkl`, `normalized_152d.pkl`
   - Generated by: `scripts/compute_layer2_normalized_vectors.py`
   - Use: Continuous phoneme similarity, diphthong modeling, Layer 3 initialization

3. **Layer 3: Contextual Phoneme Embeddings** (128-dim) ⭐ Only trained layer
   - Model: `models/layer3/model.pt` (PhonoLexBERT transformer)
   - Initialized from: Layer 2 embeddings
   - Training: Next-phoneme prediction (99.98% accuracy)
   - Data: CMU Dictionary (125K) + ipa-dict (22K) = 147K total words
   - Training script: `scripts/train_layer3_contextual_embeddings.py`
   - Use: Contextual phoneme analysis, foundation for Layer 4

4. **Layer 4: Hierarchical Syllable Embeddings** (384-dim) ⭐ Main production embeddings
   - Output: `embeddings/layer4/syllable_embeddings.pt`
   - Structure: Onset-nucleus-coda (128-dim each)
   - Built from: Frozen Layer 3 model + syllabification
   - Building script: `scripts/build_layer4_syllable_embeddings.py`
   - Use: Word similarity, rhyme detection, anagram discrimination

### Web Application Architecture (v2.1 - Client-Side)

The current production architecture is **fully client-side**:

**Stack**:
- **Frontend**: React 18 + TypeScript + MUI
- **Data Storage**: Static JSON files (~88 MB, gzips to ~45 MB)
- **Computation**: In-browser JavaScript (no backend)
- **Deployment**: Any static host (Netlify, Cloudflare Pages, etc.)

**Data Files** (in `webapp/frontend/public/data/`):
- `words.json`: 24K words with syllable structure, psycholinguistic properties, embeddings
- `phonemes.json`: Phoneme features and representations (if needed)

**Benefits**:
- Zero server costs
- No database maintenance
- Faster queries (no network latency)
- Offline-capable (PWA-ready)
- Simple deployment

See [docs/CLIENT_SIDE_DATA_PACKAGE.md](docs/CLIENT_SIDE_DATA_PACKAGE.md) for data format details.

**Historical Note**: The v2.0 backend (FastAPI + PostgreSQL + pgvector) was archived in October 2025. See `archive/webapp_v2_backend/` and [docs/ARCHITECTURE_V2.md](docs/ARCHITECTURE_V2.md) for the legacy architecture.

## Project Structure

### Core Library (`src/phonolex/`)

- `embeddings/`: Data loaders and embedding models
  - `english_data_loader.py`: CMU dictionary loader with stress markers
- `utils/`: Core utilities
  - `syllabification.py`: Onset-nucleus-coda syllable parser
- `build_phonological_graph.py`: Graph construction with typed edges

### Data (`data/`)

- `cmu/`: CMU Pronouncing Dictionary (125,764 words, ARPAbet)
- `phoible/`: Phoible database (2,716 languages, 38 distinctive features)
- `mappings/`: ARPAbet ↔ IPA conversion, phoneme vectorization
- `phonological_graph.pkl`: Pre-built graph (26K words, 56K edges)

### Embeddings (`embeddings/`)

Pre-computed embeddings from Layers 1, 2, and 4:
- `layer1/phoible_features.csv`: Raw Phoible features (38-dim ternary, 94 phonemes)
- `layer2/normalized_76d.pkl`: Normalized endpoint vectors (76-dim)
- `layer2/normalized_152d.pkl`: Normalized trajectory vectors (152-dim)
- `layer4/syllable_embeddings.pt`: Hierarchical syllable embeddings (384-dim, 1.0GB)

### Models (`models/`)

Trained models (only Layer 3 requires training):
- `layer3/model.pt`: PhonoLexBERT transformer (128-dim contextual phoneme embeddings)

### Scripts (`scripts/`)

Layer generation scripts:
- `compute_layer1_phoible_features.py`: Extract Layer 1 from Phoible database
- `compute_layer2_normalized_vectors.py`: Compute Layer 2 normalized vectors
- `train_layer3_contextual_embeddings.py`: Train Layer 3 transformer model
- `build_layer4_syllable_embeddings.py`: Build Layer 4 syllable embeddings

### Web Application (`webapp/`)

**Frontend (`webapp/frontend/`)**:
- React + TypeScript + MUI
- State management: Zustand
- Build tool: Vite
- Data: Static JSON files in `public/data/`
- Services:
  - `clientSideData.ts`: Main data service (loads and queries JSON)
  - `clientSideApiAdapter.ts`: API compatibility wrapper
  - `phonolexApi.ts`: Exports client-side adapter

**Note**: Backend was archived in October 2025. See `archive/webapp_v2_backend/` for historical FastAPI code.

### Documentation (`docs/`)

- `EMBEDDINGS_ARCHITECTURE.md`: Complete 4-layer embedding architecture documentation
- `CLIENT_SIDE_DATA_PACKAGE.md`: Client-side data format and structure
- `MIGRATION_TO_CLIENT_SIDE.md`: Migration guide from v2.0 backend to v2.1 client-side
- `VOCABULARY_FILTERING.md`: Word filtering strategy and psycholinguistic norms
- `ARCHITECTURE_V2.md`: Archived v2.0 database-centric architecture (historical)
- `development/LEARNING_DATASETS.md`: Dataset reference

### Source Code (`src/phonolex/`)

- `embeddings/english_data_loader.py`: CMU dictionary and ipa-dict loaders
- `models/phonolex_bert.py`: PhonoLexBERT transformer model class
- `utils/syllabification.py`: Onset-nucleus-coda syllable parser
- `build_phonological_graph.py`: Graph construction with typed edges

### Archive (`archive/`)

Historical experiments and old documentation (do not use for new work)

## Important Development Notes

### Layer Pipeline

The 4-layer pipeline is sequential - each layer depends on the previous:

1. **Layer 1** (Extract): Extracts English phonemes from Phoible database (~1 second)
2. **Layer 2** (Compute): Normalizes Layer 1 features to continuous vectors (~5 seconds)
3. **Layer 3** (Train): Learns contextual embeddings initialized from Layer 2 (~10 minutes on Apple Silicon)
4. **Layer 4** (Build): Aggregates Layer 3 embeddings into syllable structure (~5 minutes on CPU)

Only Layer 3 requires training:
- **Training time**: ~10 minutes on Apple Silicon (M1/M2/M3)
- **Device**: Automatically uses MPS (Metal) if available, else CPU
- **Dataset**: CMU Dictionary (125K) + ipa-dict (22K) = 147K words
- **Task**: Next-phoneme prediction (self-supervised)
- **Accuracy**: 99.98%
- **No contrastive learning**: Learns phonological structure from sequences alone

### Syllabification

The `syllabification.py` module implements English syllable parsing:
- Returns `List[Syllable]` with `.onset`, `.nucleus`, `.coda` attributes
- Handles edge cases: vowel-only words, consonant clusters
- Respects phonotactic constraints (e.g., /ŋ/ only in coda)

**Example**:
```python
from src.phonolex.utils.syllabification import syllabify

syllables = syllabify(['k', 'æ', 't'])  # Returns: [Syllable(onset=['k'], nucleus='æ', coda=['t'])]
syllables = syllabify(['æ', 'k', 't'])  # Returns: [Syllable(onset=[], nucleus='æ', coda=['k', 't'])]
```

### Data Formats

**ARPAbet to IPA mapping**: Use `data/mappings/arpa_to_ipa.json`
- CMU dict uses ARPAbet (e.g., "K AE1 T")
- Models use IPA (e.g., "k æ t")
- Stress markers: 0 (unstressed), 1 (primary), 2 (secondary)
- **Dialect**: General American English (CMU primary pronunciations only)
- **Variants**: The loader skips variant pronunciations (entries with parentheses like "GOOD(1)")
  to ensure consistent, standard pronunciations based on the primary CMU dialect

**Phoible features**: 38 ternary features (Hayes 2009 + Moisik & Esling 2011)
- Values: '+' (present), '-' (absent), '0' (not applicable)
- Stored in JSON: `data/phoible/phoible_features.json`

### Client-Side Data Service

The client-side data service (`webapp/frontend/src/services/clientSideData.ts`) provides:

1. **Data Loading**: Lazy-load JSON files on first use
2. **In-Memory Search**: Fast filtering and pattern matching
3. **Vector Similarity**: Cosine similarity computed in-browser
4. **API Compatibility**: Adapter layer for backward compatibility

**Key Functions**:
- `loadData()`: Load all data files once
- `getWord(word)`: Get word details
- `filterWords(filters)`: Filter by properties
- `patternSearch(patterns)`: Find words by phoneme patterns
- `findSimilarWords(word, threshold, limit)`: Vector similarity search
- `findMinimalPairs(phoneme1, phoneme2, limit)`: Generate minimal pairs
- `findRhymes(word, mode, limit)`: Generate rhyme sets

## Common Tasks

### Rebuilding the Layer Pipeline

If you modify Layer 1 or Layer 2, you'll need to rebuild downstream layers:

```bash
# Modified Phoible features? Rebuild all 4 layers
python scripts/compute_layer1_phoible_features.py
python scripts/compute_layer2_normalized_vectors.py
python scripts/train_layer3_contextual_embeddings.py  # ~10 min
python scripts/build_layer4_syllable_embeddings.py    # ~5 min
```

If you only retrain Layer 3 (e.g., different hyperparameters), rebuild Layer 4:

```bash
python scripts/train_layer3_contextual_embeddings.py
python scripts/build_layer4_syllable_embeddings.py
```

### Working with Layer 4 Embeddings

```python
import torch

# Load Layer 4 syllable embeddings
checkpoint = torch.load('embeddings/layer4/syllable_embeddings.pt')
word_to_syllable_embeddings = checkpoint['word_to_syllable_embeddings']

# Get syllable embeddings for a word
cat_syllables = word_to_syllable_embeddings['cat']  # List of 384-dim numpy arrays
# Each element is one syllable: [onset (128-dim) + nucleus (128-dim) + coda (128-dim)]

# Compute similarity (soft Levenshtein)
# See scripts/build_layer4_syllable_embeddings.py for hierarchical_similarity()
```

### Working with the Phonological Graph

The graph is stored as NetworkX pickle in `data/phonological_graph.pkl`:
- 26,076 nodes (words)
- 56,433 edges (typed relationships)
- Node attributes: phonemes, syllables, WCM, frequency, psycholinguistics
- Edge types: MINIMAL_PAIR, RHYME, NEIGHBOR, MAXIMAL_OPP, SIMILAR, MORPHOLOGICAL

**Edge metadata** (relationship-specific):
- MINIMAL_PAIR: `position`, `phoneme1`, `phoneme2`, `feature_diff`
- RHYME: `rhyme_type`, `syllables_matched`, `quality`
- NEIGHBOR: `edit_distance`, `phoneme_diff`

## Key Files to Read

When starting work on different aspects:

- **Layer architecture**: [docs/EMBEDDINGS_ARCHITECTURE.md](docs/EMBEDDINGS_ARCHITECTURE.md)
- **Layer 1 extraction**: [scripts/compute_layer1_phoible_features.py](scripts/compute_layer1_phoible_features.py)
- **Layer 2 computation**: [scripts/compute_layer2_normalized_vectors.py](scripts/compute_layer2_normalized_vectors.py)
- **Layer 3 training**: [scripts/train_layer3_contextual_embeddings.py](scripts/train_layer3_contextual_embeddings.py)
- **Layer 4 building**: [scripts/build_filtered_layer4_embeddings.py](scripts/build_filtered_layer4_embeddings.py)
- **Client-side export**: [scripts/export_clientside_data.py](scripts/export_clientside_data.py)
- **Model class**: [src/phonolex/models/phonolex_bert.py](src/phonolex/models/phonolex_bert.py)
- **Syllable parsing**: [src/phonolex/utils/syllabification.py](src/phonolex/utils/syllabification.py)
- **Data loading**: [src/phonolex/embeddings/english_data_loader.py](src/phonolex/embeddings/english_data_loader.py)
- **Phoneme features**: [data/mappings/phoneme_vectorizer.py](data/mappings/phoneme_vectorizer.py)
- **Client-side data service**: [webapp/frontend/src/services/clientSideData.ts](webapp/frontend/src/services/clientSideData.ts)
- **API adapter**: [webapp/frontend/src/services/clientSideApiAdapter.ts](webapp/frontend/src/services/clientSideApiAdapter.ts)
- **Data format**: [docs/CLIENT_SIDE_DATA_PACKAGE.md](docs/CLIENT_SIDE_DATA_PACKAGE.md)
- **Migration guide**: [docs/MIGRATION_TO_CLIENT_SIDE.md](docs/MIGRATION_TO_CLIENT_SIDE.md)
- **Archived backend**: [archive/webapp_v2_backend/](archive/webapp_v2_backend/)
- **Graph construction**: [src/phonolex/build_phonological_graph.py](src/phonolex/build_phonological_graph.py)

## Performance Characteristics

### Layer Performance

| Layer | Computation | Time | Size |
|-------|-------------|------|------|
| Layer 1 | Database lookup | <1 second | 59KB |
| Layer 2 | Vectorization | ~5 seconds | 174KB |
| Layer 3 | Transformer forward | ~0.1ms/word | 2.4MB |
| Layer 4 | Syllable aggregation | ~0.5ms/word | 1.0GB |

### Model Inference

- Layer 3 embedding: ~0.1ms per word (CPU)
- Layer 4 similarity: ~1-5ms per word pair (Levenshtein DP)
- Batch processing: ~1000 words/second
- Memory: Layer 3 model ~3MB, Layer 4 embeddings ~1GB

### Data Coverage

- **Layer 1**: 94 English phonemes (extracted from Phoible)
- **Layer 2**: 94 phonemes with continuous vectors
- **Layer 3**: Trained on 147K words (CMU 125K + ipa-dict 22K, primary pronunciations only)
- **Layer 4**: 24K words with syllable embeddings (filtered, v2.1+) or 125K (unfiltered, deprecated)
- **Client-side data (v2.1)**: 24K words with full psycholinguistic norms (~88 MB JSON, gzips to ~45 MB)
- **Dialect**: General American English (CMU primary pronunciations only, variant pronunciations excluded)
- **Universal**: 105,484 phonemes across 2,716 languages (Phoible)
- **Graph**: Pre-computed graph archived (see `archive/webapp_v2_backend/` for historical graph data)

### Expected Similarity Scores (Layer 4)

- Rhymes (cat-bat): 0.99+
- Anagrams (cat-act): ~0.20 (excellent discrimination!)
- Sound-alikes (computer-commuter): ~0.79
- Unrelated (cat-dog): ~0.30

## Testing Philosophy

**Note**: Backend tests were archived with the v2.0 backend. Frontend testing can be added using:
- **Vitest** for unit tests
- **React Testing Library** for component tests
- **Playwright** or **Cypress** for E2E tests

General testing principles:
- Keep tests fast and isolated
- Mock external dependencies
- Test user-facing behavior, not implementation details

## Code Style

- Follow PEP 8 for Python code
- Use type hints for function signatures
- Document complex algorithms with inline comments
- Keep functions focused and single-purpose
- Prefer explicit over implicit (e.g., named arguments)

## Git Workflow

The repository uses a standard Git workflow. Key branches:
- `main`: Production-ready code

Typical workflow:
```bash
# Create feature branch
git checkout -b feature/your-feature

# Make changes, commit
git add .
git commit -m "Description of changes"

# Push and create PR
git push origin feature/your-feature
```

## References

- **Phoible**: Moran & McCloy (2019). PHOIBLE 2.0. https://phoible.org/
- **CMU Dictionary**: http://www.speech.cs.cmu.edu/cgi-bin/cmudict
- **Syllabification**: Based on English phonotactic constraints (Hayes 2009)
- **Embeddings**: Hierarchical structure inspired by linguistic theory

## Future Roadmap

**Current Version**: v2.1 (Client-Side) - Fully static, no backend

Potential future enhancements:
- Multi-language support (extend beyond English)
- Progressive Web App (PWA) features for offline use
- Web Workers for computationally intensive operations
- Improved syllabification for edge cases
- Additional phonological tools (stress patterns, tone analysis)

**Archived**: v2.0 database backend plans archived in October 2025. See `archive/webapp_v2_backend/README.md` for historical context.
