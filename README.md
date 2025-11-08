# PhonoLex

**Comprehensive word analysis and list generation for phonological research, speech-language pathology, and language education.**

[![Version](https://img.shields.io/badge/version-2.3.0--beta-blue.svg)](CHANGELOG.md)
[![License](https://img.shields.io/badge/license-CC%20BY--SA%203.0-green.svg)](LICENSE.txt)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](python/pyproject.toml)
[![TypeScript](https://img.shields.io/badge/typescript-5.0+-blue.svg)](webapp/frontend/package.json)

## Overview

PhonoLex uniquely combines three powerful dimensions for word analysis:

1. **Universal Phonological Features** - PHOIBLE database (38 distinctive features across 2,716 languages)
2. **Psycholinguistic Norms** - 8 properties from 4 major research datasets (frequency, imageability, valence, etc.)
3. **Phoneme-Sequence Similarity** - v2.3 soft Levenshtein distance preserving consonant clusters and diphthongs

**Key Features**:
- ⭐ **Custom Word List Builder** with multi-dimensional filtering
- **Contrastive Intervention** for speech therapy (minimal pairs, maximal opposition, multiple opposition)
- **Phonological Similarity Explorer** with adjustable onset/nucleus/coda weights
- **Lookup** for word details, phoneme features, and phoneme comparison
- **48,720 words** with comprehensive psycholinguistic norms
- **Fully client-side** - no backend required, deploy anywhere
- **98% compression** - 525 MB → 10.4 MB gzipped

### Five Core Tools

#### 1. Custom Word List Builder ⭐ THE POWER TOOL

Pattern matching with multi-dimensional property filtering:

**Phonological Complexity**:
- Syllable count, phoneme count
- WCM (Word Complexity Measure - Stoel-Gammon 2010)
- MSH (Mean Syllable Height)

**Lexical Properties**:
- Frequency (SUBTLEX-US - per million words in film subtitles)
- Age of Acquisition (Glasgow Norms)

**Semantic Properties**:
- Imageability (ease of mental imagery)
- Familiarity (word familiarity)
- Concreteness (concrete vs. abstract)

**Affective Properties** (Warriner et al. 2013):
- Valence (negative to positive)
- Arousal (calm to excited)
- Dominance (weak to powerful)

**Example**: Find 1-syllable words starting with /s/, high imageability (>5), positive valence (>6), excluding words with /r/:

```
Pattern: STARTS_WITH "s"
Filters: syllables=1, imageability=5-7, valence=6-9
Exclusions: r
Result: "sun", "smile", "sky", "sand", "sea"...
```

#### 2. Contrastive Intervention

Research-based phonological intervention word lists (Gierut 1989-1992, Storkel 2022):

- **Minimal Pairs**: Traditional target/substitute contrast (e.g., θ/t → "thin/tin", "bath/bat")
- **Maximal Opposition**: Two unknowns with major class difference (sonorant vs. obstruent)
- **Multiple Opposition**: Global phoneme collapse treatment (e.g., t→d,k,g)

Position filtering (initial/medial/final) and IPA keyboard integration.

#### 3. Phonological Similarity Explorer

Adjustable onset/nucleus/coda weights for similarity computation:

**Weight Presets**:
- **Rhymes**: onset=0.0, nucleus=0.5, coda=0.5 (ignores onset)
- **Alliteration**: onset=1.0, nucleus=0.0, coda=0.0 (onset only)
- **Assonance**: onset=0.0, nucleus=1.0, coda=0.0 (vowels only)
- **Consonance**: onset=0.5, nucleus=0.0, coda=0.5 (consonants only)
- **Balanced**: all=0.33 (equal weighting)

Real-time weight sliders and threshold selection.

#### 4. Text Analysis

Analyze passages for readability across phonological, lexical, semantic, and affective dimensions:

- **Aggregate percentile statistics** across 14 psycholinguistic properties
- **Interactive highlighting** by feature with color gradients
- **Preset passages**: Grandfather, Rainbow, and Caterpillar (standard phonetics samples)
- **Coverage tracking**: percentage of words in vocabulary
- **Unknown word marking**: dotted underlines for out-of-vocabulary words

**Use cases**: Assess therapy script complexity, select reading materials, control stimulus properties for research

#### 5. Lookup

- Word details with all phonological and psycholinguistic properties
- Phoneme feature lookup (38 PHOIBLE distinctive features)
- Phoneme comparison (feature-by-feature diff)
- Feature-based phoneme search

### Psycholinguistic Norms (8 Properties)

PhonoLex integrates norms from 4 major research datasets:

| Property | Source | Range | Description |
|----------|--------|-------|-------------|
| **Lexical** |
| Frequency | SUBTLEX-US (Brysbaert & New, 2009) | 0-1000+ | Per million words |
| Age of Acquisition | Glasgow Norms (Scott et al., 2019) | 1-7 | 1=earliest |
| **Semantic** |
| Imageability | Glasgow Norms | 1-7 | Mental imagery |
| Familiarity | Glasgow Norms | 1-7 | Word familiarity |
| Concreteness | Brysbaert et al. (2014) | 1-5 | Concrete vs. abstract |
| **Affective** |
| Valence | Warriner et al. (2013) | 1-9 | Negative→positive |
| Arousal | Warriner et al. (2013) | 1-9 | Calm→excited |
| Dominance | Warriner et al. (2013) | 1-9 | Weak→powerful |

**Plus 4 Phonological Complexity Measures**:
- Syllables (CMU Dictionary): 1-5
- Phonemes (CMU Dictionary): 1-10+
- WCM (Stoel-Gammon 2010): 0-15
- MSH (Phonological analysis): 1-6

**Vocabulary**: 48,720 English words from the CMU Pronouncing Dictionary.

### Word Similarity Results (v2.3)

| Word Pair | Similarity | Type | Notes |
|-----------|------------|------|-------|
| cat - bat | 0.90+ | Perfect rhyme | Nucleus + coda match |
| cat - crest | 0.74 | Onset cluster | Proper length penalty |
| cat - act | 0.20 | Anagram | Different syllable structures |
| computer - commuter | 0.75-0.85 | Sound-alike | Multi-syllable similarity |
| cat - dog | 0.20-0.30 | Unrelated | Low similarity |

**v2.3 Innovation**: Phoneme-sequence soft Levenshtein distance preserves consonant clusters and diphthongs:
- `cat`: onset=[[k]], nucleus=[[æ]], coda=[[t]]
- `crest`: onset=[[k],[ɹ]], nucleus=[[ɛ]], coda=[[s],[t]]
- **No averaging!** Each component is a sequence of phoneme vectors

---

## Quick Start

### Web Application (Client-Side)

The web application runs entirely in the browser without server dependencies.

**Run locally**:
```bash
cd webapp/frontend
npm install
npm run dev
```

Access at http://localhost:5173

**Deploy**: Build static files and deploy to any host (Netlify, Cloudflare Pages, GitHub Pages, Vercel):
```bash
npm run build
# Upload dist/ folder to your static host
```

### Python Library (Optional - Only for Building Embeddings)

**Note**: The web application uses pre-computed embeddings. You only need Python if you're building embeddings from scratch.

```bash
git clone https://github.com/Just-Semantics/PhonoLex.git
cd PhonoLex

# Install for embedding generation
pip install -e ./python[build]

# Or for development (includes testing, linting)
pip install -e ./python[dev]
```

---

## Three-Phase Pipeline (v2.3)

**No training required!** All phases are deterministic computations from linguistic data.

```
Phase 1: Raw Phoible Features (38-dim ternary: +, -, 0)
    ↓ normalization & interpolation
Phase 2: Normalized Phoneme Vectors (76-dim / 152-dim)
    ↓ syllabification (onset-nucleus-coda decomposition)
Phase 3: Phoneme-Sequence Syllable Structures
    ↓ soft Levenshtein distance on phoneme sequences
Word Similarity
```

### Phase Details

#### Phase 1: Raw Phoible Features (38-dim ternary)
- **Output**: `embeddings/phase1/phoible_features.csv` (59 KB)
- **Coverage**: 94 English phonemes
- **Format**: Ternary features (+, -, 0)
- **Build time**: <1 second
- **Use for**: Cross-linguistic comparison, feature analysis

#### Phase 2: Normalized Feature Vectors (76-dim / 152-dim)
- **Output**: `embeddings/phase2/normalized_76d.pkl` (59 KB), `normalized_152d.pkl` (115 KB)
- **76-dim**: Monophthongs and consonants
- **152-dim**: Diphthongs (trajectory vectors)
- **Build time**: ~5 seconds
- **Use for**: Continuous phoneme similarity, diphthong modeling

#### Phase 3: Phoneme-Sequence Syllable Structures ⭐ Main Production
- **Output**: `embeddings/phase3/syllable_embeddings_phoible.pt` (~304 MB)
- **Structure**: Onset/nucleus/coda as **sequences of Phase 2 vectors** (no averaging!)
- **Vocabulary**: 48,720 English words
- **Build time**: ~5 minutes
- **Use for**: Word similarity, rhyme detection, phonological interventions

### Building the Pipeline

```bash
# Phase 1: Extract Phoible features (<1 second)
python scripts/compute_phase1_features.py

# Phase 2: Compute normalized vectors (~5 seconds)
python scripts/compute_phase2_normalized_vectors.py

# Phase 3: Build phoneme-sequence syllable structures (~5 minutes)
python scripts/build_phase3_syllable_embeddings.py

# Export data for web application
python scripts/export_clientside_data.py
# Creates: word_metadata.json, embeddings.json.gz, minimal_pairs.json.gz, etc.
# Total: ~525 MB uncompressed, ~10.4 MB gzipped (98% compression!)
```

**Complete documentation**: See [docs/PHONEME_SEQUENCE_ARCHITECTURE_V2.3.md](docs/PHONEME_SEQUENCE_ARCHITECTURE_V2.3.md) and [docs/PHASE_ARCHITECTURE.md](docs/PHASE_ARCHITECTURE.md)

---

## Usage Examples

### Python: Load Phase 3 Embeddings

```python
import torch

# Load Phase 3 syllable embeddings
checkpoint = torch.load('embeddings/phase3/syllable_embeddings_phoible.pt')
word_to_syllable_embeddings = checkpoint['word_to_syllable_embeddings']

# Get embeddings
cat = word_to_syllable_embeddings['cat']  # List of syllable structures
# Each syllable contains:
#   onset: list of 76-dim vectors
#   nucleus: list of 76-dim (monophthong) or 152-dim (diphthong) vectors
#   coda: list of 76-dim vectors

# Compute similarity using soft Levenshtein distance
# See webapp/frontend/src/services/clientSideData.ts for implementation
```

### Web Application: Custom Word List

1. Open **Custom Word List Builder**
2. Add pattern: STARTS_WITH "s"
3. Set filters:
   - Syllables: 1-2
   - Frequency: 100+ (common words)
   - Imageability: 5-7 (high)
   - Valence: 6-9 (positive)
4. Add exclusion: "r" (no /r/ sound)
5. Click **Build** → Get results with all properties
6. Export to CSV/text

### Web Application: Minimal Pairs for Therapy

1. Open **Contrastive Intervention**
2. Select **Minimal Pairs** mode
3. Enter target phoneme: θ (IPA keyboard available)
4. Enter substitute phoneme: t
5. Select position: **Word-Initial**
6. Generate pairs: "thin/tin", "thick/tick", "thought/taught"...

---

## Data Sources & References

### Phonological Features
- **PHOIBLE 2.0**: Moran & McCloy (2019) - 38 distinctive features, 2,716 languages
- **CMU Pronouncing Dictionary**: 125K words, General American English
- **Syllabification**: English phonotactic constraints (Hayes 2009)
- **WCM**: Stoel-Gammon (2010) - Word Complexity Measure

### Psycholinguistic Norms
- **SUBTLEX-US**: Brysbaert & New (2009) - Word frequency from film subtitles
- **Glasgow Norms**: Scott et al. (2019) - Imageability, familiarity, AoA
- **Concreteness**: Brysbaert et al. (2014) - 40K English words
- **Valence/Arousal/Dominance**: Warriner et al. (2013) - 13,915 English lemmas

### Clinical Interventions
- **Maximal Opposition**: Gierut (1989, 1992) - Phonological treatment approach
- **Multiple Opposition**: Storkel (2022) - Review of intervention approaches

---

## Project Structure

```
PhonoLex/
├── webapp/frontend/          # React + TypeScript web application
│   ├── src/
│   │   ├── components/
│   │   │   ├── Builder.tsx                      # Custom Word List Builder
│   │   │   └── tools/
│   │   │       ├── ContrastiveInterventionTool.tsx  # Minimal/Maximal/Multiple Opposition
│   │   │       ├── PhonologicalSimilarityTool.tsx   # Similarity explorer
│   │   │       └── SearchTool.tsx                   # Lookup
│   │   └── services/
│   │       ├── clientSideData.ts           # Main data service
│   │       └── clientSideApiAdapter.ts     # API compatibility layer
│   └── public/data/                        # Static JSON data (~0.6 MB gzipped)
│
├── scripts/                  # Build scripts (no training required!)
│   ├── compute_phase1_features.py
│   ├── compute_phase2_normalized_vectors.py
│   ├── build_phase3_syllable_embeddings.py
│   └── export_clientside_data.py
│
├── src/phonolex/             # Core library
│   ├── embeddings/           # Data loaders (CMU, Phoible, norms)
│   ├── tools/                # Maximal opposition algorithms
│   └── utils/                # Syllabification, utilities
│
├── docs/                     # Documentation
│   ├── PHONEME_SEQUENCE_ARCHITECTURE_V2.3.md  # v2.3 architecture
│   ├── PHASE_ARCHITECTURE.md                  # Phase pipeline
│   ├── MAXIMAL_OPPOSITION_TOOL.md            # Clinical intervention guide
│   └── ...
│
└── data/                     # Source data (CMU, Phoible, mappings, norms)
```

---

## Performance

### Client-Side Performance
- **Data Loading**: 1-2 seconds (initial load, cached thereafter)
- **Pattern Search**: 10-50ms for full vocabulary scan
- **Similarity Search**: 50-100ms for full vocabulary comparison
- **Filtering**: 5-20ms for multi-property filters
- **Memory**: ~60MB for all data in browser

### Data Coverage
- **48,720 words** with comprehensive psycholinguistic norms
- **112,964 minimal pairs** precomputed for contrastive intervention
- **39 English phonemes** with PHOIBLE features
- **General American English** dialect (CMU primary pronunciations)
- **98% compression**: 525 MB → 10.4 MB gzipped

---

## Architecture History

- **v2.3.0-beta (Jan 2025)**: Phonotactic probability integration + vocabulary expansion (current)
  - No averaging - sequences preserved
  - 98% compression (525 MB → 10.4 MB gzipped)
  - 48,720 words with comprehensive norms (relaxed filtering: frequency-only requirement)

- **v2.2.1 (Oct 2025)**: Phoible component-wise averaging (deprecated)
  - Averaging destroyed structural information
  - 228-dim syllable embeddings

- **v2.1 (Sep 2025)**: Client-side migration
  - Removed backend (FastAPI + PostgreSQL)
  - Fully static site deployment

- **v2.0 (Aug 2025)**: Database-centric (archived)
  - FastAPI + PostgreSQL + pgvector
  - See `archive/webapp_v2_backend/`

---

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

See [CLAUDE.md](CLAUDE.md) for development guide.

---

## License

**CC BY-SA 3.0** (Creative Commons Attribution-ShareAlike 3.0)

ShareAlike license required due to PHOIBLE data usage.

---

## Citation

If you use PhonoLex in research or clinical work, please cite:

**PhonoLex**:
```
Neumann, J. (2025). PhonoLex: Comprehensive word analysis with phonological features
and psycholinguistic norms (Version 2.3.0-beta). https://github.com/Just-Semantics/PhonoLex
```

**PHOIBLE**:
```
Moran, Steven & McCloy, Daniel (eds.) 2019.
PHOIBLE 2.0. Jena: Max Planck Institute for the Science of Human History.
(Available online at http://phoible.org)
```

**Psycholinguistic Norms**: See individual dataset citations in [webapp/frontend/src/components/CitationDialog.tsx](webapp/frontend/src/components/CitationDialog.tsx)

---

## Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/Just-Semantics/PhonoLex/issues)
- **Email**: contact@justsemantics.net

---

## Acknowledgments

Built with data from:
- PHOIBLE 2.0 (Moran & McCloy 2019)
- CMU Pronouncing Dictionary
- SUBTLEX-US (Brysbaert & New 2009)
- Glasgow Norms (Scott et al. 2019)
- Concreteness ratings (Brysbaert et al. 2014)
- Valence/Arousal/Dominance norms (Warriner et al. 2013)

Clinical intervention approaches based on research by Gierut (1989, 1992) and Storkel (2022).
