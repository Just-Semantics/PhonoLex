# Client-Side Data Package for PhonoLex

**Generated:** October 31, 2025  
**Version:** 2.0.0  
**Location:** `webapp/frontend/public/data/`

## Overview

All data needed to run PhonoLex entirely client-side (in the browser), eliminating the need for backend servers and PostgreSQL databases. This enables deployment as a static site on Netlify, Cloudflare Pages, or GitHub Pages.

## Data Files

### 1. word_metadata.json (12 MB)
Complete metadata for 24,743 filtered words including:
- `word`: Word string (e.g., "cat")
- `ipa`: IPA transcription (e.g., "k æ t")
- `phonemes`: Array of IPA phonemes
- `syllables`: Array of syllable structures with onset/nucleus/coda
- `phoneme_count`: Number of phonemes
- `syllable_count`: Number of syllables
- `wcm_score`: Word Complexity Measure
- `msh_stage`: MacArthur-Bates stage
- Psycholinguistic norms:
  - `frequency`: SUBTLEXus word frequency
  - `log_frequency`: Log-transformed frequency
  - `concreteness`: Brysbaert concreteness rating
  - `aoa`: Age of Acquisition (Glasgow Norms)
  - `imageability`: Glasgow imageability rating
  - `familiarity`: Glasgow familiarity rating
  - `valence`: VAD valence rating
  - `arousal`: VAD arousal rating
  - `dominance`: VAD dominance rating

### 2. embeddings_quantized.json (75 MB)
Int8-quantized syllable embeddings for similarity computation:
- `embeddings`: Dict mapping words to arrays of quantized syllable vectors
  - Each word maps to a list of syllables
  - Each syllable is a 384-dim int8 array
- `scales`: Quantization scales for dequantization
- `embedding_dim`: 384 (onset=128, nucleus=128, coda=128)
- `quantization`: "int8_symmetric"

**Size optimization:**
- Original float32 embeddings: 1.0 GB (unfiltered) / 137 MB (filtered)
- Quantized int8: 75 MB (45% reduction from filtered float32)
- Gzip compressed: Could reduce to ~30-40 MB for network transfer

### 3. manifest.json (312 bytes)
Metadata about the data package:
- Version, vocabulary size, filter criterion
- File descriptions

## Vocabulary Filtering

Words included meet this criterion:
✅ **Has frequency data (SUBTLEXus) AND at least one additional psycholinguistic norm**

Additional norms include:
- Concreteness (Brysbaert et al. 2014)
- Age of Acquisition (Glasgow Norms)
- Imageability (Glasgow Norms)  
- Familiarity (Glasgow Norms)
- Valence/Arousal/Dominance (Warriner et al.)

**Result:** 24,743 words (49% reduction from 48K CMU vocabulary)

## Total Package Size

- **Uncompressed:** 87 MB
- **Gzip compressed (estimated):** ~45-50 MB
- **Brotli compressed (estimated):** ~40-45 MB

Modern browsers will automatically decompress gzip/brotli, making this very efficient for web delivery.

## Usage in Browser

### Loading Data

```javascript
// Load word metadata
const metadataResponse = await fetch('/data/word_metadata.json');
const wordMetadata = await metadataResponse.json();

// Load quantized embeddings  
const embeddingsResponse = await fetch('/data/embeddings_quantized.json');
const embeddingsData = await embeddingsResponse.json();
```

### Dequantizing Embeddings

```javascript
function dequantize(quantizedArray, scale) {
  return quantizedArray.map(val => val * scale);
}

// Get dequantized embeddings for a word
function getWordEmbeddings(word, embeddingsData) {
  const quantizedSyllables = embeddingsData.embeddings[word];
  const scale = embeddingsData.scales[word]; // Per-word scale
  
  return quantizedSyllables.map(syllable => 
    dequantize(syllable, scale)
  );
}
```

### Computing Similarity

The embeddings use soft Levenshtein distance on syllable sequences:

```javascript
function cosineSimilarity(vec1, vec2) {
  const dot = vec1.reduce((sum, val, i) => sum + val * vec2[i], 0);
  const mag1 = Math.sqrt(vec1.reduce((sum, val) => sum + val * val, 0));
  const mag2 = Math.sqrt(vec2.reduce((sum, val) => sum + val * val, 0));
  return dot / (mag1 * mag2);
}

function softLevenshteinSimilarity(syllables1, syllables2) {
  // Dynamic programming with soft costs based on syllable similarity
  // See docs/EMBEDDINGS_ARCHITECTURE.md for algorithm details
  // ... implementation ...
}
```

## Next Steps

1. **Frontend Integration:**
   - Create data loading service in `src/services/dataLoader.ts`
   - Implement similarity computation in JavaScript/TypeScript
   - Add Web Worker for heavy similarity computations
   - Consider using WASM for performance-critical parts

2. **Optimization:**
   - Enable gzip/brotli compression on static host
   - Implement lazy loading (load embeddings only when needed)
   - Consider splitting data by word frequency (load high-frequency words first)

3. **Deployment:**
   - Deploy to Netlify / Cloudflare Pages / Vercel as static site
   - No backend required!
   - No database required!
   - No server costs!

## References

- Vocabulary filtering: `webapp/backend/migrations/word_filter.py`
- Export script: `scripts/export_clientside_simple.py`
- Quantization script: `scripts/quantize_embeddings.py`
- Embeddings architecture: `docs/EMBEDDINGS_ARCHITECTURE.md`
