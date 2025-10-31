# PhonoLex Web App v2.0 ✨

Interactive web application for phonological exploration and word list generation.

**Architecture**: Database-centric design with PostgreSQL + pgvector
**Documentation**: See [../docs/ARCHITECTURE_V2.md](../docs/ARCHITECTURE_V2.md) for complete architecture

## Overview

PhonoLex v2.0 provides four modalities for phonological analysis:

1. **Quick Tools**: Pre-built utilities (minimal pairs, rhyme finder, phoneme neighbors)
2. **Search**: Word lookup with detailed phonological and psycholinguistic information
3. **Builder**: Custom pattern-based word list generation
4. **Compare**: Multi-word phonological comparison

## Features (v2.0 - Current)
- **Database-centric architecture**: PostgreSQL + pgvector for efficient similarity search
- **Four embedding granularities**: Raw features, normalized vectors, tuned phonemes, syllable embeddings
- **Phonological graph**: 26K words with 56K typed edges (MINIMAL_PAIR, RHYME, NEIGHBOR, SIMILAR, MAXIMAL_OPP)
- **Psycholinguistic properties**: Age of acquisition, imageability, familiarity, concreteness, VAD (valence-arousal-dominance)
- **Fast vector similarity**: HNSW indexes for approximate nearest neighbor search
- **Graph traversal**: Recursive CTEs for finding word paths and neighborhoods

## Architecture

```
webapp/
├── backend/          # FastAPI server (Python)
│   ├── main.py       # API routes & thin query layer
│   └── services/     # Database query builders
└── frontend/         # React + TypeScript + MUI
    └── src/
        ├── components/   # UI components
        ├── stores/       # Zustand state management
        ├── services/     # API client
        ├── types/        # TypeScript types
        └── theme/        # MUI theme config

DATABASE (v2.0)
PostgreSQL + pgvector (Neon on Netlify)
├── words table          # Word properties + embeddings
├── phonemes table       # Phoneme features + vectors
├── word_edges table     # Graph relationships
└── Indexes
    ├── GIN (JSONB pattern matching)
    ├── HNSW (vector similarity)
    └── B-tree (property filtering)
```

## Quick Start

### 1. Start Backend

```bash
cd webapp/backend

# Install dependencies
pip install -r requirements.txt

# Install parent dependencies (if not already installed)
cd ../..
pip install -r requirements.txt

# Run server
cd webapp/backend
python main.py
```

Backend will run on `http://localhost:8000`

### 2. Start Frontend

```bash
cd webapp/frontend

# Install dependencies
npm install

# Run dev server
npm run dev
```

Frontend will run on `http://localhost:3000`

## Usage

### Building Patterns

1. Click **"Add Position"** to add a phoneme position
2. Select phoneme type: **Any**, **Vowel**, or **Consonant**
3. (Optional) Choose specific phonemes from dropdown
4. Results update automatically as you build

### Example Patterns

**CVC words with /æ/ vowel** (cat, bat, mat):
```
Position 1: Consonant
Position 2: Vowel → Select /æ/
Position 3: Consonant
```

**Words starting with voiced stops**:
```
Position 1: Consonant → Select /b/, /d/, /g/
```

**Two-syllable words**:
```
(No position constraints)
Syllables: Min 2, Max 2
```

## API Endpoints

- `GET /api/phonemes` - Get all available phonemes
- `GET /api/phonemes/{phoneme}` - Get phoneme features
- `POST /api/phonemes/by-features` - Find phonemes by features
- `POST /api/filter` - Filter words by pattern
- `GET /api/stats` - Get corpus statistics

## Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **Pydantic** - Data validation
- **Uvicorn** - ASGI server

### Frontend
- **React 18** - UI framework
- **TypeScript** - Type safety
- **Zustand** - State management (lightweight, performant)
- **MUI (Material-UI)** - Component library
- **Vite** - Build tool (fast HMR)

## Development

### Backend Development

```bash
# Run with auto-reload
cd webapp/backend
python main.py

# Test filtering service
python services/phoneme_filter.py
```

### Frontend Development

```bash
# Type checking
npm run type-check

# Linting
npm run lint
npm run lint:fix

# Build for production
npm run build
```

## Future Enhancements

- [ ] Advanced feature filtering (Phoible features UI)
- [ ] Minimal pair generator
- [ ] Syllable structure constraints (onset-nucleus-coda)
- [ ] Word frequency filtering
- [ ] Imageability/concreteness filtering
- [ ] Saved pattern templates
- [ ] User accounts and saved lists
- [ ] Rhyme finder integration
- [ ] Word similarity search

## Performance

- **Backend**: Inverted index for O(1) position-phoneme lookups
- **Frontend**: Zustand for minimal re-renders
- **Real-time**: Filters 125k words in <100ms

## License

MIT
