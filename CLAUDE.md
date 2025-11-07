# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PhonoLex** is a comprehensive word analysis and list generation tool for phonological research, speech-language pathology, and language education. It uniquely combines:

1. **Universal Phonological Features** - PHOIBLE database (38 distinctive features across 2,716 languages)
2. **Psycholinguistic Norms** - 8 properties from 4 major research datasets
3. **Phoneme-Sequence Similarity** - v2.3 soft Levenshtein distance preserving consonant clusters and diphthongs

###  Four Core Tools

1. **Custom Word List Builder** ⭐ THE POWER TOOL
   - Pattern matching (STARTS_WITH, ENDS_WITH, CONTAINS) with IPA phonemes
   - Multi-dimensional property filtering across 4 domains:
     - **Phonological Complexity**: Syllable count, phoneme count, WCM, MSH
     - **Lexical Properties**: Frequency (SUBTLEX-US), Age of Acquisition
     - **Semantic Properties**: Imageability, familiarity, concreteness
     - **Affective Properties**: Valence, arousal, dominance (Warriner norms)
   - Phoneme exclusion rules
   - AND logic for combined queries
   - Export results with all properties

2. **Contrastive Intervention**
   - Research-based phonological intervention word lists
   - Three modes:
     - **Minimal Pairs**: Traditional target/substitute contrast (e.g., θ/t)
     - **Maximal Opposition**: Two unknowns with major class difference (Gierut 1989-1992)
     - **Multiple Opposition**: Global phoneme collapse treatment (e.g., t→d,k,g)
   - Position filtering (initial/medial/final)
   - IPA keyboard integration

3. **Phonological Similarity Explorer**
   - Adjustable onset/nucleus/coda weights for similarity computation
   - Weight presets:
     - Rhymes (onset: 0.0, nucleus: 0.5, coda: 0.5)
     - Alliteration (onset: 1.0, nucleus: 0.0, coda: 0.0)
     - Assonance (onset: 0.0, nucleus: 1.0, coda: 0.0)
     - Consonance (onset: 0.5, nucleus: 0.0, coda: 0.5)
     - Balanced (all: 0.33)
   - Real-time weight sliders
   - Threshold selection

4. **Lookup**
   - Word details with all phonological and psycholinguistic properties
   - Phoneme feature lookup (38 distinctive features)
   - Phoneme comparison (feature-by-feature diff)
   - Feature-based phoneme search

### Psycholinguistic Norms (8 Properties)

PhonoLex integrates psycholinguistic norms from 4 major research datasets:

| Property | Source | Range | Description |
|----------|--------|-------|-------------|
| **Lexical Properties** |
| Frequency | SUBTLEX-US (Brysbaert & New, 2009) | 0-1000+ | Per million words in film subtitles |
| Age of Acquisition | Glasgow Norms (Scott et al., 2019) | 1-7 | 1=earliest, 7=latest |
| **Semantic Properties** |
| Imageability | Glasgow Norms | 1-7 | Ease of mental imagery |
| Familiarity | Glasgow Norms | 1-7 | Word familiarity |
| Concreteness | Brysbaert et al. (2014) | 1-5 | Concrete vs. abstract |
| **Affective Properties** |
| Valence | Warriner et al. (2013) | 1-9 | Negative to positive |
| Arousal | Warriner et al. (2013) | 1-9 | Calm to excited |
| Dominance | Warriner et al. (2013) | 1-9 | Weak to powerful |

**Plus 4 Phonological Complexity Measures**:
- Syllables (CMU Dictionary): 1-5 - Number of syllables
- Phonemes (CMU Dictionary): 1-10+ - Number of phonemes
- WCM (Stoel-Gammon 2010): 0-15 - Word Complexity Measure
- MSH (Phonological analysis): 1-6 - Mean Syllable Height

**Vocabulary**: 24,744 English words from the CMU Pronouncing Dictionary.

### Key Innovation (v2.3 Architecture)

**Phoneme-sequence soft Levenshtein distance** preserves consonant clusters and diphthongs without averaging:

Each syllable component (onset/nucleus/coda) is represented as a **sequence of phoneme vectors**, not a single averaged vector. This properly discriminates complex structures:

- `cat`: onset = [[k]], nucleus = [[æ]], coda = [[t]]
- `crest`: onset = [[k], [ɹ]], nucleus = [[ɛ]], coda = [[s], [t]]
- **Similarity**: 0.74 (correct!) vs old averaging: 0.99 (incorrect!)

## Project Structure (v2.2.0-beta - Client-Side)

The project uses a **modern Python package structure** with a **fully client-side web application**:

```
PhonoLex/
├── README.md                # Main documentation
├── CLAUDE.md                # This file
│
├── python/                  # Python dependencies (local dev only)
│   ├── pyproject.toml       # Modern Python packaging config
│   ├── requirements.txt     # Legacy pip requirements
│   └── README.md            # Python setup guide
│
├── src/phonolex/            # Core library
│   ├── __init__.py
│   ├── embeddings/          # Data loaders (CMU, Phoible, psycholinguistic norms)
│   ├── tools/               # Python implementations (maximal opposition, etc.)
│   └── utils/               # Syllabification, utilities
│
├── webapp/                  # Client-side web application (v2.3)
│   ├── __init__.py
│   └── frontend/            # React + TypeScript + MUI (static site)
│       ├── src/
│       │   ├── services/
│       │   │   ├── clientSideData.ts      # Main data service
│       │   │   ├── clientSideApiAdapter.ts # API compatibility layer
│       │   │   └── phonolexApi.ts         # Exports client-side adapter
│       │   └── components/
│       │       ├── Builder.tsx            # Custom Word List Builder
│       │       ├── tools/
│       │       │   ├── ContrastiveInterventionTool.tsx  # Unified intervention tool
│       │       │   ├── PhonologicalSimilarityTool.tsx   # Similarity explorer
│       │       │   └── SearchTool.tsx                   # Lookup
│       │       └── ...
│       └── public/
│           └── data/        # Static JSON data files (~56MB, gzips to ~600KB!)
│               ├── word_metadata.json      # 24,744 words with all properties
│               ├── embeddings.json.gz      # Phoneme-sequence syllable structures
│               ├── minimal_pairs.json.gz   # Precomputed minimal pairs
│               ├── phoneme_features.json.gz # PHOIBLE features for all phonemes
│               └── ...
│
├── scripts/                 # Build scripts (no training required!)
│   ├── compute_phase1_features.py          # Extract Phoible features (38-dim ternary)
│   ├── compute_phase2_normalized_vectors.py # Normalize to continuous vectors (76-dim/152-dim)
│   ├── build_phase3_syllable_embeddings.py # Build syllable sequences (v2.3 architecture)
│   └── export_clientside_data.py           # Export data to webapp/frontend/public/data/
│
├── docs/                    # All documentation
│   ├── PHONEME_SEQUENCE_ARCHITECTURE_V2.3.md  # v2.3 phoneme-sequence architecture
│   ├── PHASE_ARCHITECTURE.md                  # Phase 1-2-3 pipeline explanation
│   ├── MIGRATION_SUMMARY_V2.2.1.md           # v2.2.1 migration notes
│   ├── MAXIMAL_OPPOSITION_TOOL.md            # Maximal opposition tool guide
│   ├── CONTRASTIVE_INTERVENTION_UNIFIED_ARCHITECTURE.md # Unified intervention design
│   ├── EMBEDDINGS_ARCHITECTURE.md            # Legacy embedding architecture
│   ├── CLIENT_SIDE_DATA_PACKAGE.md           # Client-side data format
│   ├── VOCABULARY_FILTERING.md               # Filtering strategy
│   └── ARCHITECTURE_V2.md                    # v2.0 architecture (archived backend)
│
├── data/                    # Source data (CMU, Phoible, mappings, psycholinguistic norms)
├── embeddings/              # Pre-computed embeddings
├── research/                # Research notebooks and analysis scripts
│   └── v2.3_development/    # v2.3 development and testing scripts
└── archive/                 # Old code (v1 backend, v2.0 backend)
    ├── webapp_v1/           # Flask backend (deprecated)
    └── webapp_v2_backend/   # FastAPI + PostgreSQL (archived Oct 2025)
```

**Key Points**:
- **No backend required** - Fully static site deployment
- **Client-side computation** - All features run in browser
- Data pre-exported to JSON files in `webapp/frontend/public/data/`
- Backend code archived in `archive/webapp_v2_backend/`
- **24,744 words** with comprehensive psycholinguistic norms
- **99% compression**: 56.7 MB → 0.6 MB gzipped!

## Development Environment Setup

### Python Environment (Only for Embedding Generation)

**Note**: The webapp is 100% client-side JavaScript. You only need Python if you're building embeddings from scratch or working with data processing.

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies for embedding generation
pip install -e ./python[build]

# Or for development (includes testing, linting)
pip install -e ./python[dev]

# Or for everything
pip install -e ./python[all]
```

### Web Application Setup

The web application is **fully client-side** (no backend required):

**Frontend (React + TypeScript)**:
```bash
cd webapp/frontend
npm install

# Run development server (default port 5173)
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

The project has a 3-phase embedding pipeline. **No training required** - all phases are pure computation:

```bash
# Phase 1: Extract Phoible features (38-dim ternary)
python scripts/compute_phase1_features.py
# Output: embeddings/phase1/phoible_features.csv
# Time: < 1 second

# Phase 2: Compute normalized vectors (76-dim / 152-dim)
python scripts/compute_phase2_normalized_vectors.py
# Output: embeddings/phase2/normalized_76d.pkl, normalized_152d.pkl
# Time: ~5 seconds

# Phase 3: Build phoneme-sequence syllable structures (v2.3)
python scripts/build_phase3_syllable_embeddings.py
# Output: embeddings/phase3/syllable_embeddings_phoible.pt
# Time: ~5 minutes on CPU
# Processes: 24,744 words from CMU Pronouncing Dictionary
```

See [docs/PHONEME_SEQUENCE_ARCHITECTURE_V2.3.md](docs/PHONEME_SEQUENCE_ARCHITECTURE_V2.3.md) and [docs/PHASE_ARCHITECTURE.md](docs/PHASE_ARCHITECTURE.md) for complete documentation.

### Exporting Client-Side Data

After building embeddings, export data for the web app:

```bash
# Export all word data to webapp/frontend/public/data/
python scripts/export_clientside_data.py
# This creates:
# - word_metadata.json - 24,744 words with all properties and psycholinguistic norms
# - embeddings.json.gz - Phoneme-sequence syllable structures (v2.3)
# - minimal_pairs.json.gz - Precomputed minimal pairs
# - phoneme_features.json.gz - PHOIBLE features for phoneme comparison
# - syllable_structures.json.gz - Syllable decompositions
# Total size: ~56 MB uncompressed, ~0.6 MB gzipped (99% compression!)
```

### Running the Web Application

```bash
# Single terminal (no backend needed!)
cd webapp/frontend
npm run dev
# Runs on http://localhost:5173

# Build for production
npm run build

# Preview production build
npm run preview
```

## Architecture Overview

### Three-Phase Pipeline (v2.3)

PhonoLex v2.3 uses a 3-phase pipeline with **no training required**:

```
Phase 1: Raw Phoible Features (38-dim ternary: +, -, 0)
    ↓ normalization & interpolation
Phase 2: Normalized Phoneme Vectors (76-dim / 152-dim)
    ↓ syllabification (onset-nucleus-coda decomposition)
Phase 3: Phoneme-Sequence Syllable Structures
    ↓ soft Levenshtein distance on phoneme sequences
Word Similarity
```

**Key Insight (v2.3)**: No averaging! Consonant clusters and diphthongs preserved as sequences:
- `cat`: onset = [[k]], nucleus = [[æ]], coda = [[t]]
- `crest`: onset = [[k], [ɹ]], nucleus = [[ɛ]], coda = [[s], [t]]
- Soft Levenshtein properly penalizes length differences → cat-crest similarity: 0.74 (was 0.99 with averaging!)

### Three Phases

1. **Phase 1: Raw Phoible Features** (38-dim ternary: +, -, 0)
   - Output: `embeddings/phase1/phoible_features.csv`
   - Coverage: 39 English phonemes
   - Generated by: `scripts/compute_phase1_features.py`
   - Use: Cross-linguistic comparison, feature analysis
   - Time: < 1 second

2. **Phase 2: Normalized Feature Vectors** (76-dim / 152-dim)
   - Output: `embeddings/phase2/normalized_76d.pkl`, `normalized_152d.pkl`
   - Coverage: 39 phonemes (76-dim for consonants/monophthongs, 152-dim for diphthongs)
   - Generated by: `scripts/compute_phase2_normalized_vectors.py`
   - Use: Continuous phoneme similarity, diphthong modeling
   - Time: ~5 seconds

3. **Phase 3: Phoneme-Sequence Syllable Structures** ⭐ Main production representation
   - Output: `embeddings/phase3/syllable_embeddings_phoible.pt`
   - Structure: Onset/nucleus/coda as **sequences of Phase 2 vectors** (no averaging!)
   - Built from: Phase 2 normalized vectors + syllabification
   - Building script: `scripts/build_phase3_syllable_embeddings.py`
   - Use: Word similarity, rhyme detection, phonological interventions
   - Time: ~5 minutes

### Web Application Architecture (v2.3 - Client-Side)

The current production architecture is **fully client-side**:

**Stack**:
- **Frontend**: React 18 + TypeScript + MUI
- **Data Storage**: Static JSON files (~56 MB uncompressed, ~0.6 MB gzipped)
- **Computation**: In-browser JavaScript (no backend)
- **Deployment**: Any static host (Netlify, Cloudflare Pages, etc.)

**Data Files** (in `webapp/frontend/public/data/`):
- `word_metadata.json`: 24,744 words with phonological properties and psycholinguistic norms
- `embeddings.json.gz`: Phoneme-sequence syllable structures (v2.3)
- `minimal_pairs.json.gz`: Precomputed minimal pairs
- `phoneme_features.json.gz`: PHOIBLE features for 39 English phonemes
- `syllable_structures.json.gz`: Onset-nucleus-coda decompositions

**Benefits**:
- Zero server costs
- No database maintenance
- Faster queries (no network latency)
- Offline-capable (PWA-ready)
- Simple deployment
- 99% compression with gzip

See [docs/CLIENT_SIDE_DATA_PACKAGE.md](docs/CLIENT_SIDE_DATA_PACKAGE.md) for data format details.

**Historical Note**: The v2.0 backend (FastAPI + PostgreSQL + pgvector) was archived in October 2025. See `archive/webapp_v2_backend/` and [docs/ARCHITECTURE_V2.md](docs/ARCHITECTURE_V2.md) for the legacy architecture.

## Important Development Notes

### Phase Pipeline

The 3-phase pipeline is sequential - each phase depends on the previous:

1. **Phase 1** (Extract): Extracts English phonemes from Phoible database (~1 second)
2. **Phase 2** (Compute): Normalizes Phase 1 features to continuous vectors (~5 seconds)
3. **Phase 3** (Build): Constructs phoneme-sequence syllable structures (~5 minutes)

**No training required!** All phases are deterministic computations from linguistic data.

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
  to ensure consistent, standard pronunciations

**Phoible features**: 38 ternary features (Hayes 2009 + Moisik & Esling 2011)
- Values: '+' (present), '-' (absent), '0' (not applicable)
- Stored in JSON: `data/phoible/phoible_features.json`

**Psycholinguistic norms**: Merged from multiple datasets
- SUBTLEX-US: Word frequency
- Glasgow Norms: AoA, imageability, familiarity
- Brysbaert et al.: Concreteness
- Warriner et al.: Valence, arousal, dominance

### Client-Side Data Service

The client-side data service (`webapp/frontend/src/services/clientSideData.ts`) provides:

1. **Data Loading**: Lazy-load JSON files on first use
2. **In-Memory Search**: Fast filtering and pattern matching
3. **Vector Similarity**: Soft Levenshtein on phoneme sequences computed in-browser
4. **API Compatibility**: Adapter layer for backward compatibility

**Key Functions**:
- `loadData()`: Load all data files once
- `getWord(word)`: Get word details with all properties
- `filterWords(filters)`: Filter by phonological/lexical/semantic/affective properties
- `patternSearch(patterns)`: Find words by phoneme patterns (STARTS_WITH/ENDS_WITH/CONTAINS)
- `findSimilarWords(word, threshold, limit)`: Vector similarity search
- `findMinimalPairs(phoneme1, phoneme2, limit)`: Generate minimal pairs
- `generateMaximalOppositionPairs()`: Find phoneme pairs with major class differences
- `selectRepresentativeTargets()`: Maximal Classification + Maximal Distinction algorithms
- `generateMultipleOppositionSets()`: Generate minimal sets (triplets/quadruplets/quintuplets)

## Common Tasks

### Rebuilding the Phase Pipeline

If you modify Phase 1 or Phase 2, you'll need to rebuild downstream phases:

```bash
# Modified Phoible features? Rebuild all 3 phases
python scripts/compute_phase1_features.py
python scripts/compute_phase2_normalized_vectors.py
python scripts/build_phase3_syllable_embeddings.py  # ~5 min
```

### Working with Phase 3 Embeddings

```python
import torch

# Load Phase 3 syllable embeddings
checkpoint = torch.load('embeddings/phase3/syllable_embeddings_phoible.pt')
word_to_syllable_embeddings = checkpoint['word_to_syllable_embeddings']

# Get syllable embeddings for a word
cat_syllables = word_to_syllable_embeddings['cat']  # List of syllable structures
# Each syllable contains: onset (list of 76-dim vectors), nucleus (list of 76/152-dim), coda (list of 76-dim)

# Compute similarity using soft Levenshtein
# See webapp/frontend/src/services/clientSideData.ts for similarity computation
```

### Working with Psycholinguistic Norms

All psycholinguistic norms are stored in `word_metadata.json`:

```typescript
interface Word {
  word: string;
  ipa: string;

  // Phonological
  syllable_count: number;
  phoneme_count: number;
  wcm: number | null;
  msh: number | null;

  // Lexical
  frequency: number | null;
  aoa: number | null;

  // Semantic
  imageability: number | null;
  familiarity: number | null;
  concreteness: number | null;

  // Affective
  valence: number | null;
  arousal: number | null;
  dominance: number | null;
}
```

## Key Files to Read

When starting work on different aspects:

**Architecture & Design**:
- [docs/PHONEME_SEQUENCE_ARCHITECTURE_V2.3.md](docs/PHONEME_SEQUENCE_ARCHITECTURE_V2.3.md) - v2.3 phoneme-sequence architecture
- [docs/PHASE_ARCHITECTURE.md](docs/PHASE_ARCHITECTURE.md) - Phase 1-2-3 pipeline
- [docs/MAXIMAL_OPPOSITION_TOOL.md](docs/MAXIMAL_OPPOSITION_TOOL.md) - Maximal opposition algorithms
- [docs/CONTRASTIVE_INTERVENTION_UNIFIED_ARCHITECTURE.md](docs/CONTRASTIVE_INTERVENTION_UNIFIED_ARCHITECTURE.md) - Unified intervention design

**Scripts**:
- [scripts/compute_phase1_features.py](scripts/compute_phase1_features.py) - Phase 1 extraction
- [scripts/compute_phase2_normalized_vectors.py](scripts/compute_phase2_normalized_vectors.py) - Phase 2 computation
- [scripts/build_phase3_syllable_embeddings.py](scripts/build_phase3_syllable_embeddings.py) - Phase 3 building
- [scripts/export_clientside_data.py](scripts/export_clientside_data.py) - Client-side export

**Core Library**:
- [src/phonolex/utils/syllabification.py](src/phonolex/utils/syllabification.py) - Syllable parsing
- [src/phonolex/embeddings/english_data_loader.py](src/phonolex/embeddings/english_data_loader.py) - Data loading
- [src/phonolex/tools/maximal_opposition.py](src/phonolex/tools/maximal_opposition.py) - Maximal opposition backend

**Frontend**:
- [webapp/frontend/src/components/Builder.tsx](webapp/frontend/src/components/Builder.tsx) - Custom Word List Builder
- [webapp/frontend/src/components/tools/ContrastiveInterventionTool.tsx](webapp/frontend/src/components/tools/ContrastiveInterventionTool.tsx) - Unified intervention tool
- [webapp/frontend/src/components/tools/PhonologicalSimilarityTool.tsx](webapp/frontend/src/components/tools/PhonologicalSimilarityTool.tsx) - Similarity explorer
- [webapp/frontend/src/services/clientSideData.ts](webapp/frontend/src/services/clientSideData.ts) - Main data service
- [webapp/frontend/src/services/clientSideApiAdapter.ts](webapp/frontend/src/services/clientSideApiAdapter.ts) - API adapter

**Documentation**:
- [docs/CLIENT_SIDE_DATA_PACKAGE.md](docs/CLIENT_SIDE_DATA_PACKAGE.md) - Data format
- [docs/VOCABULARY_FILTERING.md](docs/VOCABULARY_FILTERING.md) - Filtering strategy

## Performance Characteristics

### Phase Performance

| Phase | Computation | Time | Size |
|-------|-------------|------|------|
| Phase 1 | Database lookup | <1 second | 59KB |
| Phase 2 | Vectorization | ~5 seconds | 174KB |
| Phase 3 | Syllable structures | ~5 minutes | ~112MB |

### Client-Side Performance

- **Data Loading**: ~1-2 seconds (initial load, cached thereafter)
- **Pattern Search**: ~10-50ms for full vocabulary scan
- **Similarity Search**: ~50-100ms for full vocabulary comparison
- **Filtering**: ~5-20ms for multi-property filters
- **Memory**: ~60MB for all data in browser

### Data Coverage

- **Phase 1**: 39 English phonemes (extracted from Phoible)
- **Phase 2**: 39 phonemes with continuous vectors
- **Phase 3**: 24,744 words with phoneme-sequence syllable structures (v2.3)
- **Client-side data**: 24,744 words with comprehensive psycholinguistic norms
- **Dialect**: General American English (CMU primary pronunciations only)
- **Universal**: 105,484 phonemes across 2,716 languages (Phoible database)
- **Compression**: 56.7 MB → 0.6 MB gzipped (99% reduction!)

### Expected Similarity Scores (v2.3)

- Perfect rhymes (cat-bat): 0.90+ (nucleus+coda match)
- Anagrams (cat-act): ~0.20 (different syllable structures - excellent discrimination!)
- Onset clusters (cat-crest): ~0.74 (proper length penalty from soft Levenshtein)
- Sound-alikes (computer-commuter): ~0.75-0.85
- Unrelated (cat-dog): ~0.20-0.30

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

**Phonological Features**:
- Moran, S., & McCloy, D. (2019). PHOIBLE 2.0. Max Planck Institute. https://phoible.org/
- Hayes, B. (2009). Introductory Phonology. Wiley-Blackwell.
- Stoel-Gammon, C. (2010). The Word Complexity Measure: Description and application to developmental phonology and disorders. Clinical Linguistics & Phonetics.

**Psycholinguistic Norms**:
- Brysbaert, M., & New, B. (2009). Moving beyond Kučera and Francis: A critical evaluation of current word frequency norms and the introduction of a new and improved word frequency measure for American English. Behavior Research Methods, 41(4), 977-990.
- Brysbaert, M., Warriner, A. B., & Kuperman, V. (2014). Concreteness ratings for 40 thousand generally known English word lemmas. Behavior Research Methods, 46, 904-911.
- Kuperman, V., Stadthagen-Gonzalez, H., & Brysbaert, M. (2012). Age-of-acquisition ratings for 30,000 English words. Behavior Research Methods, 44, 978-990.
- Scott, G. G., Keitel, A., Becirspahic, M., Yao, B., & Sereno, S. C. (2019). The Glasgow Norms: Ratings of 5,500 words on nine scales. Behavior Research Methods, 51, 1258-1270.
- Warriner, A. B., Kuperman, V., & Brysbaert, M. (2013). Norms of valence, arousal, and dominance for 13,915 English lemmas. Behavior Research Methods, 45, 1191-1207.

**Clinical Interventions**:
- Gierut, J. A. (1989). Maximal opposition approach to phonological treatment. Journal of Speech and Hearing Disorders, 54(1), 9-19.
- Gierut, J. A. (1992). The conditions and course of clinically induced phonological change. Journal of Speech and Hearing Research, 35(5), 1049-1063.
- Storkel, H. L. (2022). Minimal, Maximal, or Multiple Oppositions: A review of phonological intervention approaches. Language, Speech, and Hearing Services in Schools, 53(2), 421-437.

**Data Sources**:
- CMU Pronouncing Dictionary: http://www.speech.cs.cmu.edu/cgi-bin/cmudict
- Syllabification: Based on English phonotactic constraints

## Future Roadmap

**Current Version**: v2.2.0-beta (Client-Side, Phoneme-Sequence Architecture)

Potential future enhancements:
- Multi-language support (extend beyond English using Phoible's cross-linguistic features)
- Progressive Web App (PWA) features for offline use
- Web Workers for computationally intensive operations
- Additional phonological tools (stress patterns, tone analysis, phonotactic probability)
- Export formats (CSV, Excel, plain text with customizable templates)

**Archived**: v2.0 database backend plans archived in October 2025. See `archive/webapp_v2_backend/README.md` for historical context.
