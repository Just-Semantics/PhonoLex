# Vocabulary Filtering Strategy (v2.0+)

**Last Updated**: 2025-10-29
**Status**: ✅ Implemented

---

## Overview

PhonoLex v2.0 implements **systematic vocabulary filtering** to reduce database size while improving data quality. Only words with comprehensive psycholinguistic properties are included.

### Filtering Criterion

**Words must have**:
1. ✅ **Frequency data** (SUBTLEXus)
2. ✅ **At least one additional psycholinguistic norm**:
   - Concreteness (Brysbaert et al. 2014)
   - Age of Acquisition (Glasgow Norms)
   - Imageability (Glasgow Norms)
   - Familiarity (Glasgow Norms)
   - Valence/Arousal/Dominance (Warriner et al.)

---

## Impact

### Size Reduction

| Metric | Before (Freq Only) | After (Freq + Norm) | Change |
|--------|-------------------|---------------------|---------|
| **Word Count** | 48,720 | 24,744 | **-49.2%** ⬇️ |
| **Layer 4 Embeddings** | ~1.0 GB | **~0.5 GB** | **-49%** ⬇️ |
| **Database Size** | ~500 MB | **~250 MB** | **-50%** ⬇️ |

### Norm Coverage (Filtered 24K Words)

- **97.8%** have concreteness ratings
- **54.0%** have VAD ratings (emotional properties)
- **18.6%** have Glasgow norms (AoA, imageability, familiarity)
- **100%** have frequency data

---

## Quality Improvement

### Words Removed (23,976)

Words with frequency but **no other psycholinguistic properties**:

❌ Proper nouns (personal names, place names)
❌ Technical/specialized jargon
❌ Rare/obscure words
❌ Compound words with limited research utility

**Example removed**: "zygomorphically", "eigenvalue", "john", "mcdonald"

### Words Retained (24,744)

Words with **comprehensive psycholinguistic characterization**:

✅ Common vocabulary with clinical utility
✅ Research-grade words with multiple norms
✅ Age-appropriate for therapy targets
✅ Characterized for emotional content

**Example retained**: "cat", "dog", "happy", "run", "computer"

---

## User Impact

### ✅ Clinical/SLP Applications

**POSITIVE** - Better therapy vocabulary:
- All words have evidence-based properties
- Can select targets by:
  - Frequency (high-frequency first)
  - AoA (developmentally appropriate)
  - Concreteness (AAC considerations)
  - Emotional valence (engagement)
- No wasted words without usable properties

### ✅ Research Applications

**POSITIVE** - Higher-quality datasets:
- Every word has controllable variables
- Can match stimuli on multiple dimensions
- No words lacking critical norms
- Better for psycholinguistic experiments

### ⚠️ General Linguistics

**NEUTRAL** - Core vocabulary maintained:
- 24K words covers everyday language
- Rare/technical words accessible via CMU dictionary
- Can add custom wordsets if needed

---

## Implementation

### Central Filter Module

[`webapp/backend/migrations/word_filter.py`](../webapp/backend/migrations/word_filter.py) provides:

```python
from word_filter import WordFilter

# Initialize filter
word_filter = WordFilter()
word_filter.load_all_norms()

# Check if word should be included
if word_filter.should_include_word("cat"):
    # Include in database
    pass
```

### Integration Points

1. **[`populate_words.py`](../webapp/backend/migrations/populate_words.py)**: Filters during database population
2. **[`build_filtered_layer4_embeddings.py`](../scripts/build_filtered_layer4_embeddings.py)**: Creates embeddings only for filtered words
3. **[`populate_edges.py`](../webapp/backend/migrations/populate_edges.py)**: Automatically uses filtered words (looks up from database)
4. **[`populate_norms.py`](../webapp/backend/migrations/populate_norms.py)**: Automatically updates only filtered words

---

## Usage

### Building Filtered Dataset

```bash
# 1. Analyze filtering impact (optional)
python scripts/analyze_norm_filtering.py

# 2. Populate database with filtered words
python webapp/backend/migrations/populate_words.py

# 3. Add psycholinguistic norms
python webapp/backend/migrations/populate_norms.py

# 4. Build filtered embeddings
python scripts/build_filtered_layer4_embeddings.py

# 5. Generate graph edges
python webapp/backend/migrations/populate_edges.py
```

### Cleaning Existing Database

If you have an existing database with unfiltered words:

```bash
# Dry run (see what would be removed)
python webapp/backend/migrations/cleanup_unfiltered_words.py --dry-run

# Actually remove words (requires confirmation)
python webapp/backend/migrations/cleanup_unfiltered_words.py

# Skip confirmation (be careful!)
python webapp/backend/migrations/cleanup_unfiltered_words.py --yes
```

---

## Cloudflare Hosting Benefits

This filtering strategy **enables pure Cloudflare deployment**:

### Before Filtering (48K words)
- Embeddings: 1.0 GB
- Compressed (gzip): ~330 MB
- ❌ **Too large** for Cloudflare Workers (128 MB memory limit)

### After Filtering (24K words)
- Embeddings: 0.5 GB
- Compressed (gzip): ~170 MB
- With int8 quantization: **~60 MB**
- ✅ **Fits** in Cloudflare Workers bundle (10 MB script + 60 MB data)

### Deployment Options

1. **Cloudflare Vectorize** (Recommended)
   - Upload 24K × 384-dim vectors
   - Workers query via API (no memory constraint)
   - Metadata in D1 SQLite

2. **Bundled Embeddings** (Alternative)
   - Quantize to int8: ~60 MB compressed
   - Include in Worker bundle
   - Fast query (no network roundtrip)

---

## Data Sources

### Frequency
- **Source**: SUBTLEXus (Brysbaert & New, 2009)
- **Coverage**: 74,286 words
- **File**: `data/subtlex_frequency.txt`

### Concreteness
- **Source**: Brysbaert et al. (2014)
- **Coverage**: 39,954 words
- **File**: `data/norms/concreteness.txt`

### Glasgow Norms (AoA, Imageability, Familiarity)
- **Source**: Scott et al. (2019)
- **Coverage**: 5,553 words
- **File**: `data/norms/GlasgowNorms.xlsx`

### VAD (Valence, Arousal, Dominance)
- **Source**: Warriner et al. (2013)
- **Coverage**: 13,905 words
- **File**: `data/norms/Ratings_VAD_WarrinerEtAl.csv`

---

## References

- Brysbaert, M., & New, B. (2009). Moving beyond Kučera and Francis: A critical evaluation of current word frequency norms and the introduction of a new and improved word frequency measure for American English. *Behavior Research Methods*, 41(4), 977-990.

- Brysbaert, M., Warriner, A. B., & Kuperman, V. (2014). Concreteness ratings for 40 thousand generally known English word lemmas. *Behavior Research Methods*, 46(3), 904-911.

- Scott, G. G., Keitel, A., Becirspahic, M., Yao, B., & Sereno, S. C. (2019). The Glasgow Norms: Ratings of 5,500 words on nine scales. *Behavior Research Methods*, 51(3), 1258-1270.

- Warriner, A. B., Kuperman, V., & Brysbaert, M. (2013). Norms of valence, arousal, and dominance for 13,915 English lemmas. *Behavior Research Methods*, 45(4), 1191-1207.

---

## Alternatives Considered

### Option 1: Frequency Only (Current System, Pre-v2.0)
- **Words**: 48,720
- **Issue**: 49% lack research-grade properties
- **Status**: ❌ Deprecated

### Option 2: Frequency + Any Norm (Implemented)
- **Words**: 24,744
- **Benefit**: All words fully characterized
- **Status**: ✅ **Recommended**

### Option 3: Frequency + 2+ Norms (Too Strict)
- **Words**: 13,542
- **Benefit**: Highest quality
- **Issue**: Loses valuable words (e.g., many action verbs)
- **Status**: ⚠️ Optional for specialized applications

---

## Migration Guide

### For Existing Projects

1. **Backup your database**:
   ```bash
   pg_dump phonolex > phonolex_backup.sql
   ```

2. **Run cleanup script**:
   ```bash
   python webapp/backend/migrations/cleanup_unfiltered_words.py
   ```

3. **Rebuild embeddings**:
   ```bash
   python scripts/build_filtered_layer4_embeddings.py
   ```

4. **Update application code** to use filtered embeddings:
   ```python
   # Old
   checkpoint = torch.load('embeddings/layer4/syllable_embeddings.pt')

   # New
   checkpoint = torch.load('embeddings/layer4/syllable_embeddings_filtered.pt')
   ```

### For New Projects

Just use the default commands - filtering is automatic:

```bash
python webapp/backend/migrations/populate_words.py
python scripts/build_filtered_layer4_embeddings.py
```

---

## FAQ

**Q: Can I still access the full 125K word vocabulary?**
A: Yes! The CMU dictionary and Layer 3 model support all 125K words. The filtering only affects the database and Layer 4 embeddings. You can always compute embeddings on-demand for any word.

**Q: What if I need a word that was filtered out?**
A: You can:
1. Add it manually to the database with custom norms
2. Compute its embedding on-demand using Layer 3 model
3. Create a custom build without filtering (set `limit=None` in scripts)

**Q: Does this affect model training?**
A: No! Layer 3 is still trained on the full 147K word corpus. Filtering only affects the production database and pre-computed embeddings.

**Q: Will this break my existing application?**
A: Migration is required. Follow the [Migration Guide](#migration-guide) above. The API remains compatible - just fewer words are available.

---

**Status**: ✅ Production Ready
**Next Steps**: Deploy to Cloudflare with filtered dataset
