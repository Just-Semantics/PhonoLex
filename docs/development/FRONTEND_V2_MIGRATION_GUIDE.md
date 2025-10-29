# Frontend V2.0 Migration Complete - Summary & Guide

## Executive Summary

The frontend has been successfully updated to support the v2.0 database-centric backend architecture. All TypeScript types now match the backend schemas, and all v2.0 API endpoints are accessible through the updated API client.

**Status**: ✅ Complete - All type checking passes

## What Changed

### 1. Type Definitions ([src/types/phonology.ts](webapp/frontend/src/types/phonology.ts))

**Added Complete Word Interface**:
```typescript
export interface Word {
  word_id: number;
  word: string;
  ipa: string;

  // Phonological structure
  phonemes: PhonemePosition[];
  syllables: Syllable[];
  phoneme_count: number;
  syllable_count: number;

  // Categorical
  word_length: 'short' | 'medium' | 'long' | null;
  complexity: 'low' | 'medium' | 'high' | null;

  // Psycholinguistic (NEW!)
  frequency: number | null;
  log_frequency: number | null;
  aoa: number | null;  // Age of Acquisition
  imageability: number | null;
  familiarity: number | null;
  concreteness: number | null;
  valence: number | null;
  arousal: number | null;
  dominance: number | null;

  // Clinical (NEW!)
  wcm_score: number | null;
  msh_stage: number | null;  // Motor Speech Hierarchy
}
```

**Added Edge/Graph Types**:
```typescript
export interface WordEdge {
  word1: string;
  word2: string;
  relation_type: EdgeType;
  metadata: EdgeMetadata;
  weight: number;
}

export interface MinimalPairResult {
  word1: Word;
  word2: Word;
  position: number;
  phoneme1: string;
  phoneme2: string;
  feature_diff?: number;
}

export interface RhymeResult {
  rhyme: Word;
  rhyme_type: string;
  nucleus: string;
  coda: string[];
  quality: number;
}
```

### 2. API Client ([src/services/phonolexApi.ts](webapp/frontend/src/services/phonolexApi.ts))

**New v2.0 Endpoints Added**:
- ✅ `getWord(word)` - Full word data with all properties
- ✅ `filterWords(request)` - Filter by syllables, complexity, length, etc.
- ✅ `patternSearch(request)` - Search by phoneme patterns (starts_with, ends_with, contains)
- ✅ `findSimilarWords(word, threshold, limit)` - pgvector similarity search
- ✅ `similaritySearch(request)` - Similarity with filters
- ✅ `getNeighbors(word, relationType, limit)` - Graph neighbors by edge type
- ✅ `getMinimalPairs(params)` - Generate minimal pairs for phoneme contrasts
- ✅ `getRhymes(params)` - Find rhyming words with metadata
- ✅ `exportGraph(includeEmbeddings)` - Export full graph data

**Backward Compatibility**:
All old v1.0 methods still work but are marked `@deprecated`:
- `generateMinimalPairs()` → use `getMinimalPairs()`
- `generateRhymeSet()` → use `getRhymes()`
- `generateComplexityList()` → use `filterWords()`
- `findPhonemePosition()` → use `patternSearch()`
- `buildWordList()` → use `filterWords()`
- `comparePhonemes()` → works but uses client-side comparison
- `searchPhonemes()` → use `searchPhonemesByFeatures()`

### 3. Component Updates

**Fixed Type Errors**:
- [Compare.tsx](webapp/frontend/src/components/Compare.tsx:72) - Convert API response to PhonemeDetail format
- [Search.tsx](webapp/frontend/src/components/Search.tsx:458) - Use `phoneme_id` instead of `id`

**No Breaking Changes**:
All existing components continue to work through backward-compatible API methods.

## New Data Available

### Word Properties You Can Now Access

**Psycholinguistic Properties**:
```typescript
word.aoa          // Age of Acquisition (when kids learn it)
word.imageability // How easily visualized (concrete vs abstract)
word.familiarity  // How common/familiar
word.concreteness // Concrete vs abstract
word.valence      // Emotional positivity/negativity
word.arousal      // Emotional intensity
word.dominance    // Sense of control/power
```

**Clinical Properties**:
```typescript
word.msh_stage    // Motor Speech Hierarchy stage (1-8)
```

**Phonological Structure**:
```typescript
word.phonemes     // Array of {ipa, position}
word.syllables    // Array of {onset, nucleus, coda}
```

### Graph Relationships You Can Now Query

```typescript
// Get all neighbors by edge type
const neighbors = await api.getNeighbors('cat', 'MINIMAL_PAIR');
const rhymes = await api.getNeighbors('cat', 'RHYME');
const similar = await api.getNeighbors('cat', 'SIMILAR');

// Get typed edges with metadata
neighbors[0].edge.relation_type  // 'MINIMAL_PAIR'
neighbors[0].edge.metadata       // {position: 0, phoneme1: 'k', phoneme2: 'b', ...}

// Generate minimal pairs for specific contrasts
const pairs = await api.getMinimalPairs({
  phoneme1: 't',
  phoneme2: 'd',
  word_length: 'short',
  complexity: 'low',
  limit: 50
});

// Find rhymes with quality scores
const rhymes = await api.getRhymes({
  word: 'cat',
  limit: 50
});
```

### Vector Similarity Search

```typescript
// Find similar words using 384-dim syllable embeddings
const similar = await api.findSimilarWords('cat', 0.85, 50);

similar.forEach(result => {
  console.log(result.word.word, result.similarity);
  // Access full word data including psycholinguistics
  console.log('AoA:', result.word.aoa);
  console.log('Imageability:', result.word.imageability);
});

// Similarity with filters
const similarFiltered = await api.similaritySearch({
  word: 'cat',
  threshold: 0.85,
  limit: 50,
  filters: {
    complexity: 'low',
    max_wcm: 5
  }
});
```

## Migration Path for New Features

### Example 1: Display Psycholinguistic Data in Word Cards

```typescript
// In your WordCard component
const WordCard: React.FC<{ word: Word }> = ({ word }) => {
  return (
    <Card>
      <CardContent>
        <Typography variant="h5">{word.word}</Typography>
        <Typography>IPA: {word.ipa}</Typography>

        {/* NEW: Show psycholinguistic properties */}
        {word.aoa && (
          <Chip label={`AoA: ${word.aoa.toFixed(1)}`} />
        )}
        {word.imageability && (
          <Chip label={`Imageability: ${word.imageability.toFixed(2)}`} />
        )}
        {word.concreteness && (
          <Chip label={`Concreteness: ${word.concreteness.toFixed(2)}`} />
        )}

        {/* NEW: Show clinical measures */}
        {word.msh_stage && (
          <Chip label={`MSH Stage: ${word.msh_stage}`} color="primary" />
        )}
        {word.wcm_score !== null && (
          <Chip label={`WCM: ${word.wcm_score}`} />
        )}
      </CardContent>
    </Card>
  );
};
```

### Example 2: Graph Visualization

```typescript
// Fetch graph neighbors and visualize edges
const GraphViewer: React.FC<{ word: string }> = ({ word }) => {
  const [neighbors, setNeighbors] = useState<NeighborResult[]>([]);

  useEffect(() => {
    api.getNeighbors(word, undefined, 100).then(setNeighbors);
  }, [word]);

  return (
    <Box>
      {neighbors.map(neighbor => (
        <Box key={neighbor.neighbor.word_id}>
          <Typography>
            {word} → {neighbor.neighbor.word}
          </Typography>
          <Chip
            label={neighbor.edge.relation_type}
            size="small"
          />
          {/* Display edge-specific metadata */}
          {neighbor.edge.relation_type === 'MINIMAL_PAIR' && (
            <Typography variant="caption">
              Position {neighbor.edge.metadata.position}:
              {neighbor.edge.metadata.phoneme1} → {neighbor.edge.metadata.phoneme2}
            </Typography>
          )}
        </Box>
      ))}
    </Box>
  );
};
```

### Example 3: Advanced Filtering

```typescript
// Use new filtering capabilities
const AdvancedFilter: React.FC = () => {
  const [results, setResults] = useState<Word[]>([]);

  const handleFilter = async () => {
    const words = await api.filterWords({
      min_syllables: 1,
      max_syllables: 2,
      word_length: 'short',
      complexity: 'low',
      limit: 100
    });

    // Filter further by psycholinguistic properties
    const filtered = words.filter(w =>
      w.aoa !== null && w.aoa < 5 &&  // Early acquisition
      w.imageability !== null && w.imageability > 5  // Highly imageable
    );

    setResults(filtered);
  };

  return (
    <Box>
      <Button onClick={handleFilter}>
        Find Early-Acquired Imageable Words
      </Button>
      {/* Display results */}
    </Box>
  );
};
```

## Backend Requirements

The frontend now expects the v2.0 backend to be running. You have two options:

### Option 1: Use v2.0 Backend with Database (Recommended)

```bash
cd webapp/backend

# Ensure database is populated
python migrations/populate_phonemes.py
python migrations/populate_words.py
python migrations/populate_edges.py

# Run v2.0 backend
python main_v2.py
# Runs on http://localhost:8000
```

### Option 2: Use v1.0 Backend (Temporary Fallback)

The old v1.0 endpoints still work through backward-compatible methods, but you won't have access to:
- Psycholinguistic properties (aoa, imageability, etc.)
- MSH stage
- Graph operations (neighbors, edges)
- pgvector similarity search

```bash
cd webapp/backend
python main.py  # Old v1.0 backend
```

## Testing Checklist

- ✅ TypeScript compilation passes (`npm run type-check`)
- ✅ All component imports resolved
- ✅ Backward compatibility maintained
- ⏳ Runtime testing needed:
  - [ ] Test word fetching with new properties
  - [ ] Test similarity search
  - [ ] Test graph neighbors
  - [ ] Test minimal pairs generation
  - [ ] Test rhyme finding
  - [ ] Test filtering by new properties

## Next Steps

### Immediate (Optional - UI Enhancements)

1. **Update Word Display Components** to show new psycholinguistic properties
2. **Add Graph Visualization** using edge data
3. **Enhance Filtering UI** to support AoA, imageability, etc.
4. **Add MSH Stage Filter** for clinical users

### Future (v3.0)

1. Remove deprecated v1.0 methods
2. Implement full Builder with pattern matching
3. Add client-side graph caching using `exportGraph()`
4. Build interactive graph explorer

## Files Modified

1. ✅ [src/types/phonology.ts](webapp/frontend/src/types/phonology.ts) - Complete type system
2. ✅ [src/services/phonolexApi.ts](webapp/frontend/src/services/phonolexApi.ts) - v2.0 API client
3. ✅ [src/components/Compare.tsx](webapp/frontend/src/components/Compare.tsx:72) - Type fix
4. ✅ [src/components/Search.tsx](webapp/frontend/src/components/Search.tsx:458) - Type fix

## Documentation

- [FRONTEND_UPDATE_PLAN.md](FRONTEND_UPDATE_PLAN.md) - Detailed update plan
- [docs/ARCHITECTURE_V2.md](docs/ARCHITECTURE_V2.md) - Backend v2.0 architecture
- [docs/EMBEDDINGS_ARCHITECTURE.md](docs/EMBEDDINGS_ARCHITECTURE.md) - 4-layer embeddings

## Support

For questions or issues:
1. Check type definitions in `src/types/phonology.ts`
2. Review API client methods in `src/services/phonolexApi.ts`
3. Consult backend API docs at http://localhost:8000/docs (when backend is running)

---

**Migration Status**: ✅ **COMPLETE**

All data from the backend is now accessible in the frontend!
