# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PhonoLex is a phonological analysis toolkit that combines universal phonological features (Phoible) with learned contextual representations. It provides hierarchical word embeddings learned from natural phoneme sequences, enabling phonological similarity analysis, rhyme detection, and pronunciation comparison.

**Key Innovation**: Position-aware syllable embeddings that properly discriminate anagrams (cat ≠ act) using onset-nucleus-coda structure, learned from next-phoneme prediction without contrastive learning.

## Project Structure (v2.0)

The project uses a **modern Python package structure** with proper organization:

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
├── webapp/                  # Full-stack web application (v2.0)
│   ├── __init__.py
│   ├── backend/             # FastAPI backend (PostgreSQL + pgvector)
│   │   ├── __init__.py
│   │   ├── main.py          # Entry point (run with: python main.py)
│   │   ├── database.py      # DatabaseService class
│   │   ├── models.py        # SQLAlchemy ORM models
│   │   ├── routers/         # API route modules
│   │   ├── migrations/      # Database population scripts
│   │   └── tests/           # Backend test suite
│   └── frontend/            # React + TypeScript + MUI
│       ├── src/
│       └── ...
│
├── scripts/                 # Build/training scripts
│   ├── compute_layer1_phoible_features.py
│   ├── compute_layer2_normalized_vectors.py
│   ├── train_layer3_contextual_embeddings.py
│   └── build_layer4_syllable_embeddings.py
│
├── docs/                    # All documentation
│   ├── EMBEDDINGS_ARCHITECTURE.md
│   ├── ARCHITECTURE_V2.md   # Web app v2.0 architecture
│   ├── development/         # Planning docs
│   └── webapp/              # Web app docs
│       ├── backend/
│       └── frontend/
│
├── data/                    # Source data (CMU, Phoible, mappings)
├── embeddings/              # Pre-computed embeddings
├── models/                  # Trained models
├── research/                # Research notebooks
└── archive/                 # Old code (deprecated v1 backend, etc.)
```

**Key Points**:
- `webapp/` is now a proper Python package with `__init__.py` files throughout
- Backend is at `webapp/backend/` (v2.0 is canonical, v1 archived)
- Documentation organized in `docs/` with subdirectories
- Use `pyproject.toml` for modern Python packaging

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

The project has two separate web application setups:

**Backend (FastAPI)**:
```bash
cd webapp/backend
pip install -r requirements.txt

# Run development server
uvicorn main:app --reload --port 8000
# Or: python main.py
```

**Frontend (React + TypeScript)**:
```bash
cd webapp/frontend
npm install

# Run development server (default port 3000)
npm run dev

# Build for production
npm run build

# Type checking
npm run type-check

# Linting
npm run lint
npm run lint:fix
```

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
python scripts/build_layer4_syllable_embeddings.py
# Output: embeddings/layer4/syllable_embeddings.pt
```

See [docs/EMBEDDINGS_ARCHITECTURE.md](docs/EMBEDDINGS_ARCHITECTURE.md) for complete documentation.

### Testing

**Backend Tests** (pytest):
```bash
cd webapp/backend

# Run all tests
venv_test/bin/pytest tests/ -v

# Run with coverage
venv_test/bin/pytest tests/ --cov

# Run specific test categories (using markers from pytest.ini)
venv_test/bin/pytest tests/ -m unit        # Fast unit tests only
venv_test/bin/pytest tests/ -m integration # Integration tests (need DB)
venv_test/bin/pytest tests/ -m api         # API endpoint tests

# Run single test file
venv_test/bin/pytest tests/test_specific.py -v

# Run single test function
venv_test/bin/pytest tests/test_specific.py::test_function_name -v

# Verbose output with short traceback
venv_test/bin/pytest tests/ -v --tb=short

# Stop on first failure
venv_test/bin/pytest tests/ -x
```

Test markers (defined in `webapp/backend/pytest.ini`):
- `unit`: Fast unit tests (no DB required)
- `integration`: Integration tests (require DB)
- `service`: Service layer tests
- `api`: API endpoint tests
- `performance`: Performance benchmark tests
- `slow`: Tests taking > 1 second
- `critical`: Critical path tests

### Running the Web Application

```bash
# Terminal 1: Backend
cd webapp/backend
python main.py
# Runs on http://localhost:8000

# Terminal 2: Frontend
cd webapp/frontend
npm run dev
# Runs on http://localhost:3000
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

### Database Architecture (v2.0 - Future)

The planned v2.0 architecture uses PostgreSQL + pgvector for production deployment. See `docs/ARCHITECTURE_V2.md` for complete details.

**Stack**:
- **Database**: PostgreSQL 15+ with pgvector extension
- **Backend**: FastAPI (Python 3.10+)
- **Frontend**: React 18 + TypeScript + MUI
- **Deployment**: Netlify + Neon (recommended) or Cloudflare + Supabase

**Key Tables**:
- `words`: 26K words with syllable structure, psycholinguistic properties, embeddings
- `phonemes`: 103 phonemes with Phoible features and multiple embedding granularities
- `word_edges`: 56K typed edges (MINIMAL_PAIR, RHYME, NEIGHBOR, etc.)

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

**Backend (`webapp/backend/`)**:
- `main.py`: FastAPI application entry point
- `services/`: Business logic (currently empty, being migrated)
- `api/`: API routes (currently empty, being migrated)
- `tests/`: Pytest test suite
- `pytest.ini`: Test configuration with markers

**Frontend (`webapp/frontend/`)**:
- React + TypeScript + MUI
- State management: Zustand
- Build tool: Vite

### Documentation (`docs/`)

- `EMBEDDINGS_ARCHITECTURE.md`: Complete 4-layer embedding architecture documentation
- `ARCHITECTURE_V2.md`: Complete v2.0 database-centric architecture
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

**ARPAbet to IPA mapping**: Use `data/mappings/arpabet_to_ipa.json`
- CMU dict uses ARPAbet (e.g., "K AE1 T")
- Models use IPA (e.g., "k æ t")
- Stress markers: 0 (unstressed), 1 (primary), 2 (secondary)

**Phoible features**: 38 ternary features (Hayes 2009 + Moisik & Esling 2011)
- Values: '+' (present), '-' (absent), '0' (not applicable)
- Stored in JSON: `data/phoible/phoible_features.json`

### Web API Patterns

The FastAPI backend follows these patterns:

1. **Service Layer**: Business logic separate from routes (being migrated to `services/`)
2. **Pydantic Schemas**: Request/response models in route files or dedicated schemas
3. **CORS**: Configured for `localhost:3000` (Vite) and `localhost:5173` (older Vite)
4. **Startup Events**: Load data once on startup (see `@app.on_event("startup")`)

### Database (Future v2.0)

When implementing database features:

- Use **PostgreSQL with pgvector** for vector similarity
- Store **typed edges** in `word_edges` table (not separate graph DB)
- Use **GIN indexes** on JSONB columns for pattern matching
- Use **HNSW indexes** on vector columns for similarity search
- Push computation to database (not client) - leverage SQL indexes

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
- **Layer 4 building**: [scripts/build_layer4_syllable_embeddings.py](scripts/build_layer4_syllable_embeddings.py)
- **Model class**: [src/phonolex/models/phonolex_bert.py](src/phonolex/models/phonolex_bert.py)
- **Syllable parsing**: [src/phonolex/utils/syllabification.py](src/phonolex/utils/syllabification.py)
- **Data loading**: [src/phonolex/embeddings/english_data_loader.py](src/phonolex/embeddings/english_data_loader.py)
- **Phoneme features**: [data/mappings/phoneme_vectorizer.py](data/mappings/phoneme_vectorizer.py)
- **API structure**: [webapp/backend/main.py](webapp/backend/main.py)
- **Database design**: [docs/ARCHITECTURE_V2.md](docs/ARCHITECTURE_V2.md)
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
- **Layer 3**: Trained on 147K words (CMU 125K + ipa-dict 22K)
- **Layer 4**: 125K words with syllable embeddings
- **Universal**: 105,484 phonemes across 2,716 languages (Phoible)
- **Graph**: 26K words, 56K typed edges

### Expected Similarity Scores (Layer 4)

- Rhymes (cat-bat): 0.99+
- Anagrams (cat-act): ~0.20 (excellent discrimination!)
- Sound-alikes (computer-commuter): ~0.79
- Unrelated (cat-dog): ~0.30

## Testing Philosophy

- **Unit tests** should be fast (<10ms) and require no external resources
- **Integration tests** may use database or file I/O
- **Mark slow tests** with `@pytest.mark.slow` for optional exclusion
- **Critical path tests** should be marked with `@pytest.mark.critical`
- Use **fixtures** for shared test data (define in `conftest.py`)
- Mock external services in unit tests

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

See `docs/ARCHITECTURE_V2.md` for v2.0 database-centric architecture plans.

Short-term priorities:
- Database migration (PostgreSQL + pgvector)
- Web app v2.0 with database backend
- Multi-language support
- Improved syllabification for edge cases
