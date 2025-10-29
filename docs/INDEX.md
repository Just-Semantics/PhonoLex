# PhonoLex Documentation Index

Welcome to the PhonoLex documentation. This index will guide you to the right documentation for your needs.

---

## Quick Navigation

### 🚀 Getting Started
- **[Main README](../README.md)** - Project overview, quick start, installation
- **[ARCHITECTURE_V2.md](ARCHITECTURE_V2.md)** - ⭐ **Single source of truth** for v2.0 architecture

### 📊 Data & Models
- **[EMBEDDINGS.md](EMBEDDINGS.md)** - Four embedding granularities explained
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Deployment guide for v2.0 (Netlify + Neon)
- **[development/LEARNING_DATASETS.md](development/LEARNING_DATASETS.md)** - Dataset reference

### 💻 Web Application
- **[webapp/README.md](../webapp/README.md)** - Web app documentation (v1.0 → v2.0)

### 📦 Archive
- **[../archive/docs/](../archive/docs/)** - Historical documentation
- **[../CLEANUP_PLAN.md](../CLEANUP_PLAN.md)** - Documentation cleanup rationale

---

## Documentation Structure

```
docs/
├── INDEX.md                 ← You are here
├── ARCHITECTURE_V2.md       ← Primary reference for v2.0
├── EMBEDDINGS.md            ← Embedding types & usage
├── DEPLOYMENT.md            ← How to deploy v2.0
└── development/
    └── LEARNING_DATASETS.md ← Dataset information
```

---

## By Use Case

### I want to understand the project architecture
→ Read [ARCHITECTURE_V2.md](ARCHITECTURE_V2.md)

### I want to use embeddings in my code
→ Read [EMBEDDINGS.md](EMBEDDINGS.md)

### I want to deploy the web app
→ Read [DEPLOYMENT.md](DEPLOYMENT.md)

### I want to understand the datasets
→ Read [development/LEARNING_DATASETS.md](development/LEARNING_DATASETS.md)

### I want to train models
→ Read [Main README](../README.md) → Training section

### I want to use the Python library
→ Read [Main README](../README.md) → Quick Start section

### I want to understand historical decisions
→ Browse [../archive/docs/](../archive/docs/)

---

## Key Concepts

### Four Embedding Granularities
PhonoLex provides multiple levels of phonological representation:
1. **Raw Phoible Features** (38-dim) - Ternary distinctive features
2. **Normalized Vectors** (76-dim/152-dim) - Continuous feature vectors
3. **Tuned Phoneme Embeddings** (128-dim) - Contextual phoneme representations
4. **Hierarchical Syllable Embeddings** (384-dim) - Onset-nucleus-coda structure

→ See [EMBEDDINGS.md](EMBEDDINGS.md) for details

### Database-Centric Architecture
v2.0 uses a database-centric approach:
- **PostgreSQL + pgvector** for vector similarity
- **GIN indexes** for JSONB pattern matching
- **HNSW indexes** for fast ANN search
- **Recursive CTEs** for graph traversal
- **Thin API layer** (FastAPI)
- **Thin client** (React)

→ See [ARCHITECTURE_V2.md](ARCHITECTURE_V2.md) for details

### Phonological Graph
26,076 words with 56,433 typed edges:
- **MINIMAL_PAIR**: Words differing by one phoneme
- **RHYME**: Words with matching final syllables
- **NEIGHBOR**: Phonologically similar words
- **SIMILAR**: Vector similarity (embeddings)
- **MAXIMAL_OPP**: Words maximally different

Includes psycholinguistic properties: AoA, imageability, familiarity, concreteness, VAD

→ See [ARCHITECTURE_V2.md](ARCHITECTURE_V2.md) → Data Schema section

---

## Contributing

When adding new documentation:
1. Keep [ARCHITECTURE_V2.md](ARCHITECTURE_V2.md) as the single source of truth
2. Add specific guides as separate files (like EMBEDDINGS.md, DEPLOYMENT.md)
3. Update this INDEX.md to link to new docs
4. Archive outdated docs to `../archive/docs/`

---

## Version History

- **v2.0** (Current) - Database-centric architecture with four embedding granularities
- **v1.0** - Client-side filtering with position-based pattern builder

---

**Last Updated**: 2025-10-28
