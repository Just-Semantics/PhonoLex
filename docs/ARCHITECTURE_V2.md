# PhonoLex Architecture v2.0

**Status**: Implementation Plan
**Date**: 2025-10-28
**Version**: 2.0.0

---

## Executive Summary

PhonoLex v2.0 uses a **database-centric architecture** that offloads computation to a fast remote database with smart indexing. The key insight: modern serverless databases (Netlify, Cloudflare, Neon) with proper indexes can handle complex queries faster than client-side JavaScript.

### Key Design Decisions

1. **Database Does the Work**: Leverage PostgreSQL + pgvector + graph extensions for all computation
2. **Smart Algorithm Design**: Bottlenecks live in algorithms, not data transfer - optimize at the database level
3. **Offload to Database**: Push filtering, pattern matching, graph traversal to SQL/graph queries
4. **Cloud-Native**: Built for Netlify/Cloudflare/Neon serverless databases from day one

### Performance Targets

- Database queries: <50ms (optimized indexes + edge caching)
- Pattern matching: <100ms (JSONB GIN indexes + denormalization)
- Vector similarity: <150ms (pgvector HNSW index)
- Graph traversal: <100ms (recursive CTEs or graph extension)
- Client rendering: <20ms (minimal processing, just display)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│               React Frontend (Netlify)                       │
│  - Thin client: just rendering and user interaction         │
│  - Minimal caching: only UI state and recent results        │
│  - All queries go to backend API                            │
└───────────────────┬─────────────────────────────────────────┘
                    │ (All queries via API)
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│          FastAPI Backend (Netlify Functions/Cloudflare)     │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Service Layer (Python)                                │ │
│  │  - Thin orchestration layer                           │ │
│  │  - Translates user queries to optimized SQL           │ │
│  │  - Returns JSON responses                             │ │
│  └────────────────────────────────────────────────────────┘ │
└───────────────────┬─────────────────────────────────────────┘
                    │ (All computation pushed to DB)
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│      PostgreSQL + Extensions (Neon/Supabase/Cloudflare)    │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Core Data Store                                       │ │
│  │  - 26K words with full properties                     │ │
│  │  - 56K typed edges (graph relationships)              │ │
│  │  - 103 phonemes with features                         │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  pgvector Extension                                    │ │
│  │  - HNSW indexes for fast ANN similarity search        │ │
│  │  - Stores 384-dim syllable embeddings                 │ │
│  │  - Stores 76-dim phoneme endpoint vectors             │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────���───────────────────────────────────────────┐ │
│  │  Apache AGE Extension (Graph Queries) [OPTIONAL]      │ │
│  │  - Native graph traversal (BFS, shortest path)        │ │
│  │  - OR: Use recursive CTEs (standard SQL)              │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Smart Indexing                                        │ │
│  │  - GIN indexes on JSONB (pattern matching)            │ │
│  │  - B-tree indexes on properties (WCM, frequency)      │ │
│  │  - Partial indexes for common queries                 │ │
│  │  - Materialized views for expensive aggregations      │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### What Lives Where

#### Client-Side (React Frontend)
- **UI components** (forms, tables, visualizations)
- **UI state** (selected filters, current view)
- **Recent results cache** (~10 most recent queries, <1MB)
- **No graph data** (everything fetched from API)

#### Backend API (FastAPI on Netlify/Cloudflare)
- **Query translation** (user request → optimized SQL)
- **Response formatting** (SQL results → JSON)
- **Caching layer** (optional: Redis/KV store for hot queries)
- **Rate limiting** and authentication

#### Database (PostgreSQL + Extensions)
- **All data** (words, phonemes, edges, embeddings)
- **All computation** (filtering, matching, graph traversal, similarity)
- **All indexing** (GIN, B-tree, HNSW for fast queries)
- **Materialized views** (pre-computed aggregations)

---

## Layer 1: PostgreSQL + pgvector

### Role
Persistent storage for structured data and vector similarity search.

### Schema

```sql
-- ============================================================================
-- WORDS TABLE
-- ============================================================================
CREATE TABLE words (
    word_id SERIAL PRIMARY KEY,
    word VARCHAR(100) UNIQUE NOT NULL,
    ipa TEXT NOT NULL,

    -- Phonological structure (denormalized for speed)
    phonemes_json JSONB NOT NULL,  -- [{"ipa": "k", "position": 0}, ...]
    syllables_json JSONB NOT NULL, -- [{"onset": ["k"], "nucleus": "æ", ...}]

    -- Counts
    phoneme_count INT NOT NULL,
    syllable_count INT NOT NULL,

    -- Psycholinguistic properties
    frequency FLOAT,
    log_frequency FLOAT,
    aoa FLOAT,                    -- Age of acquisition
    imageability FLOAT,
    familiarity FLOAT,
    concreteness FLOAT,
    valence FLOAT,                -- Emotional valence
    arousal FLOAT,                -- Emotional arousal
    dominance FLOAT,              -- Emotional dominance

    -- Clinical measures
    wcm_score INT,                -- Word Complexity Measure
    msh_stage INT,                -- Motor Speech Hierarchy

    -- Embeddings (pgvector columns)
    syllable_embedding vector(384),     -- Hierarchical model
    word_embedding_flat vector(64),     -- Simple word embedding

    -- Categorical properties
    word_length VARCHAR(10),      -- 'short', 'medium', 'long'
    complexity VARCHAR(10)        -- 'low', 'medium', 'high' (from WCM)
);

-- Indexes for fast filtering
CREATE INDEX idx_words_phoneme_count ON words(phoneme_count);
CREATE INDEX idx_words_syllable_count ON words(syllable_count);
CREATE INDEX idx_words_wcm ON words(wcm_score);
CREATE INDEX idx_words_aoa ON words(aoa);
CREATE INDEX idx_words_frequency ON words(frequency);

-- JSONB indexes for pattern matching
CREATE INDEX idx_words_phonemes_gin ON words USING gin (phonemes_json jsonb_path_ops);

-- pgvector HNSW indexes for fast ANN
CREATE INDEX idx_words_syllable_embedding ON words
    USING hnsw (syllable_embedding vector_cosine_ops);


-- ============================================================================
-- PHONEMES TABLE
-- ============================================================================
CREATE TABLE phonemes (
    phoneme_id SERIAL PRIMARY KEY,
    ipa VARCHAR(10) UNIQUE NOT NULL,
    segment_class VARCHAR(20) NOT NULL,  -- 'consonant', 'vowel', 'tone'

    -- PHOIBLE features (38 dimensions)
    features JSONB NOT NULL,

    -- Vector representations (multiple granularities)
    raw_features vector(38),           -- Raw ternary features
    endpoints_76d vector(76),           -- Start + end states
    trajectories_152d vector(152),     -- 4 timesteps (for diphthongs)
    contextual_128d vector(128),       -- From hierarchical model

    -- Trajectory metadata
    has_trajectory BOOLEAN DEFAULT FALSE,
    trajectory_features TEXT[]
);


-- ============================================================================
-- WORD_EDGES TABLE (Graph relationships)
-- ============================================================================
CREATE TABLE word_edges (
    edge_id SERIAL PRIMARY KEY,
    word1_id INT NOT NULL REFERENCES words(word_id),
    word2_id INT NOT NULL REFERENCES words(word_id),

    -- Edge type (supports multiple relationship types)
    relation_type VARCHAR(50) NOT NULL,

    -- Relationship-specific metadata
    metadata JSONB NOT NULL,

    -- Precomputed weight for graph algorithms
    weight FLOAT,

    CONSTRAINT no_self_loops CHECK (word1_id != word2_id),
    CONSTRAINT ordered_pair CHECK (word1_id < word2_id),
    UNIQUE (word1_id, word2_id, relation_type)
);

-- Indexes for fast edge queries
CREATE INDEX idx_edges_word1 ON word_edges(word1_id);
CREATE INDEX idx_edges_word2 ON word_edges(word2_id);
CREATE INDEX idx_edges_relation ON word_edges(relation_type);
CREATE INDEX idx_edges_word1_relation ON word_edges(word1_id, relation_type);
```

### Edge Types

All edges are **typed relationships** with rich metadata:

| Relation Type | Description | Metadata |
|--------------|-------------|----------|
| `MINIMAL_PAIR` | Differ by exactly 1 phoneme | `position`, `phoneme1`, `phoneme2`, `feature_diff` |
| `RHYME` | Rhyming words | `rhyme_type`, `syllables_matched`, `quality` |
| `NEIGHBOR` | Edit distance 1-3 | `edit_distance`, `phoneme_diff` |
| `MAXIMAL_OPP` | Maximally different | `phoneme_differences`, `feature_differences` |
| `SIMILAR` | Embedding similarity >0.85 | `similarity`, `embedding_type` |
| `MORPHOLOGICAL` | Morphologically related | `relationship`, `suffix` |

**Why typed edges?**
- Graph has **semantic meaning** (not just similarity scores)
- Enables edge-type filtering in graph algorithms
- Supports multiple relationship types between same word pair
- Metadata makes edges interpretable

---

## Layer 2: NetworkX Graph (Python Backend + Client Cache)

### Backend: Python NetworkX

Used for:
- **Initial graph construction** from database edges
- **Complex graph algorithms** (community detection, centrality)
- **Serving initial graph data** to client
- **Backup for client-side graph operations**

```python
# Load graph from database
edges = db.query("SELECT word1, word2, relation_type, metadata, weight FROM word_edges")
graph = nx.Graph()

for word1, word2, rel_type, metadata, weight in edges:
    graph.add_edge(word1, word2,
                   relation=rel_type,
                   weight=weight,
                   **metadata)

# Graph statistics
print(f"Nodes: {graph.number_of_nodes()}")  # 26,076
print(f"Edges: {graph.number_of_edges()}")  # 56,433
print(f"Memory: ~100MB")
```

### Client: JavaScript Graph (Browser Cache)

**Critical Design Decision**: Load entire graph into browser memory on startup.

#### Why This Works

1. **Size is manageable**: 26K nodes + 56K edges ≈ 50-100MB compressed
2. **Modern browsers handle it**: 8GB RAM typical, 100MB is <2% of available memory
3. **Instant queries**: No network latency for 90% of operations
4. **Offline capable**: Can work without server once loaded
5. **Service worker cache**: Persists across sessions

#### Client-Side Graph Structure

```javascript
// In-memory graph (loaded on startup)
const phonolexGraph = {
  nodes: new Map(),  // word -> node data
  edges: new Map(),  // word -> [neighbors]

  // Node data structure
  // {
  //   word: "cat",
  //   ipa: "k æ t",
  //   phonemes: ["k", "æ", "t"],
  //   syllable_count: 1,
  //   wcm_score: 3,
  //   frequency: 66.33,
  //   aoa: 1.74,
  //   ...all properties
  // }

  // Edge data structure
  // {
  //   neighbor: "bat",
  //   relation: "MINIMAL_PAIR",
  //   metadata: {position: 0, phoneme1: "k", phoneme2: "b"},
  //   weight: 1.0
  // }
};

// Example: Client-side minimal pairs query (instant!)
function findMinimalPairs(word, phoneme1, phoneme2) {
  const edges = phonolexGraph.edges.get(word) || [];
  return edges.filter(e =>
    e.relation === 'MINIMAL_PAIR' &&
    e.metadata.phoneme1 === phoneme1 &&
    e.metadata.phoneme2 === phoneme2
  );
}
// Time: <1ms (pure JavaScript array operations)
```

#### Loading Strategy

```javascript
// On app initialization (loading screen)
async function initializePhonoLex() {
  showLoadingScreen("Loading phonological graph...");

  // Fetch compressed graph data
  const response = await fetch('/api/graph/export');
  const compressed = await response.arrayBuffer();

  // Decompress (using WASM for speed)
  const graphData = await decompressGraph(compressed);

  // Build in-memory structure
  buildGraphStructure(graphData);

  // Cache in service worker for next session
  await cacheGraph(compressed);

  hideLoadingScreen();
}
```

---

## Layer 3: Service Layer (FastAPI)

### Role
Orchestrate database and graph operations, handle complex queries, serve initial graph data.

### Key Services

```python
class PhonoLexService:
    """Unified service combining DB, graph, and embeddings"""

    def __init__(self, connection_string: str):
        self.db = DatabaseService(connection_string)
        self.graph = GraphService(self.db)  # For complex algorithms
        self.embedding_models = EmbeddingService()  # For on-demand computation
```

### API Endpoints

#### Graph Export (Initial Load)
```python
@app.get("/api/graph/export")
async def export_graph():
    """Export full graph for client caching"""
    graph_data = {
        'nodes': [...],  # All 26K words with properties
        'edges': [...],  # All 56K edges with metadata
        'phonemes': [...],  # 103 phonemes with features
        'version': '2.0.0'
    }

    # Compress with gzip
    compressed = gzip.compress(json.dumps(graph_data).encode())

    return Response(
        content=compressed,
        media_type='application/gzip',
        headers={'Content-Encoding': 'gzip'}
    )
```

#### Vector Similarity (Server-Side)
```python
@app.get("/api/similarity/word/{word}")
async def similarity_search(word: str, threshold: float = 0.85):
    """Vector similarity using pgvector (too expensive for client)"""

    # Get word embedding from DB
    vec = db.get_word_embedding(word)

    # pgvector ANN search
    similar = db.query("""
        SELECT word, 1 - (syllable_embedding <=> %s) as similarity
        FROM words
        WHERE 1 - (syllable_embedding <=> %s) > %s
        ORDER BY syllable_embedding <=> %s
        LIMIT 50
    """, vec, vec, threshold, vec)

    return similar
```

#### Pattern Search (Hybrid: Client Pre-Filter + Server Finalize)
```python
@app.post("/api/builder/generate")
async def pattern_search(patterns: List[Pattern], properties: Dict):
    """
    Pattern matching strategy:
    1. Client filters by simple patterns (starts/ends/contains)
    2. Server finalizes with property constraints
    3. Returns only matching words
    """

    # Server-side property filtering (when needed)
    if properties.get('requires_db_query'):
        return db.pattern_search(patterns, properties)
    else:
        # Return instruction for client to handle locally
        return {"client_filter": True, "patterns": patterns}
```

---

## Algorithm Design: Database-Centric Computation

### Core Principle

**All computation happens in the database**. The database is optimized for these operations with indexes, extensions, and query planning. The client just renders results.

### Why Database Over Client?

1. **Indexes are faster than JavaScript loops**: B-tree, GIN, HNSW indexes beat any client-side algorithm
2. **No data transfer bottleneck**: Query 26K records, return 50 results (not 26K → client → filter → 50)
3. **Parallel execution**: Database query planner uses multiple cores
4. **Edge CDN caching**: Cloudflare/Netlify cache frequently-accessed queries
5. **Serverless scaling**: Database auto-scales, client doesn't need to

### Database Query Strategies

| Operation | Implementation | Time | Optimization |
|-----------|----------------|------|--------------|
| Minimal pairs | JOIN on word_edges with filters | <50ms | Index on (word1_id, relation_type, metadata) |
| Rhyme set | JOIN on word_edges WHERE relation='RHYME' | <30ms | Partial index on rhyme edges |
| Property filtering | WHERE clauses with B-tree indexes | <20ms | Composite indexes on common filters |
| Pattern matching | JSONB operators with GIN index | <80ms | Denormalized phonemes_json |
| Vector similarity | pgvector <=> operator with HNSW | <150ms | HNSW index on embeddings |
| Graph traversal | Recursive CTE or AGE queries | <100ms | Edge table indexes + temp tables |

### Example 1: Minimal Pairs (Pure SQL)

```sql
-- Find minimal pairs for /t/ vs /d/ with filters
-- Time: <50ms with proper indexes

SELECT
    w1.word as word1,
    w2.word as word2,
    e.metadata->'position' as position,
    w1.wcm_score,
    w1.syllable_count
FROM word_edges e
JOIN words w1 ON w1.word_id = e.word1_id
JOIN words w2 ON w2.word_id = e.word2_id
WHERE e.relation_type = 'MINIMAL_PAIR'
  AND e.metadata->>'phoneme1' = 't'
  AND e.metadata->>'phoneme2' = 'd'
  AND w1.word_length = 'short'
  AND w1.complexity = 'low'
LIMIT 30;

-- Why this is fast:
-- 1. Index on (relation_type, metadata) filters 56K edges to ~500
-- 2. Index on word_length/complexity filters to ~200
-- 3. No data transfer until final 30 results
```

### Example 2: Pattern Matching (JSONB + GIN)

```sql
-- Find words: starts with /b/, ends with /t/, has /æ/, 1 syllable, low WCM
-- Time: <80ms with GIN index

SELECT word, phonemes_json, wcm_score
FROM words
WHERE phonemes_json->0->>'ipa' = 'b'           -- Starts with /b/
  AND phonemes_json->-1->>'ipa' = 't'          -- Ends with /t/
  AND phonemes_json @> '[{"ipa": "æ"}]'::jsonb -- Contains /æ/ (GIN index!)
  AND syllable_count = 1
  AND wcm_score <= 5
LIMIT 100;

-- Why GIN index is critical:
-- The @> containment operator uses GIN index for fast lookup
-- Without GIN: full table scan (slow)
-- With GIN: index scan → returns matching rows instantly
```

### Example 3: Vector Similarity (pgvector)

```sql
-- Find words similar to 'cat' using syllable embeddings
-- Time: <150ms with HNSW index

WITH target AS (
    SELECT syllable_embedding FROM words WHERE word = 'cat'
)
SELECT
    w.word,
    1 - (w.syllable_embedding <=> t.syllable_embedding) as similarity,
    w.wcm_score,
    w.frequency
FROM words w, target t
WHERE w.syllable_embedding IS NOT NULL
  AND 1 - (w.syllable_embedding <=> t.syllable_embedding) > 0.85
  AND w.wcm_score <= 5  -- Can combine with property filters!
ORDER BY w.syllable_embedding <=> t.syllable_embedding
LIMIT 50;

-- Why HNSW index is essential:
-- Without HNSW: compute cosine similarity for all 26K words (slow!)
-- With HNSW: approximate nearest neighbor search (sub-linear time)
```

### Example 4: Graph Traversal (Recursive CTE)

```sql
-- Find phonological path from 'cat' to 'dog' (treatment progression)
-- Time: <100ms

WITH RECURSIVE path AS (
    -- Base case: start at 'cat'
    SELECT
        w1.word_id,
        w1.word,
        ARRAY[w1.word] as path,
        0 as depth
    FROM words w1
    WHERE w1.word = 'cat'

    UNION ALL

    -- Recursive case: follow edges
    SELECT
        w2.word_id,
        w2.word,
        p.path || w2.word,
        p.depth + 1
    FROM path p
    JOIN word_edges e ON e.word1_id = p.word_id
    JOIN words w2 ON w2.word_id = e.word2_id
    WHERE e.relation_type IN ('MINIMAL_PAIR', 'NEIGHBOR')
      AND p.depth < 5  -- Max depth
      AND NOT w2.word = ANY(p.path)  -- Avoid cycles
)
SELECT * FROM path WHERE word = 'dog' LIMIT 1;

-- Alternative: Apache AGE extension (if available)
-- SELECT * FROM cypher('phonolex', $$
--     MATCH path = (a:Word {word: 'cat'})-[:MINIMAL_PAIR*1..5]-(b:Word {word: 'dog'})
--     RETURN path
-- $$) as (path agtype);
```

### Backend API: Thin Translation Layer

The FastAPI backend just translates user requests to SQL:

```python
class PhonoLexService:
    """Thin service layer - all computation in database"""

    def minimal_pairs(self, phoneme1: str, phoneme2: str, filters: Dict):
        """Translate to SQL, execute, return results"""

        query = """
            SELECT w1.word, w2.word, e.metadata
            FROM word_edges e
            JOIN words w1 ON w1.word_id = e.word1_id
            JOIN words w2 ON w2.word_id = e.word2_id
            WHERE e.relation_type = 'MINIMAL_PAIR'
              AND e.metadata->>'phoneme1' = %s
              AND e.metadata->>'phoneme2' = %s
        """

        params = [phoneme1, phoneme2]

        # Add filters dynamically
        if filters.get('word_length'):
            query += " AND w1.word_length = %s"
            params.append(filters['word_length'])

        if filters.get('complexity'):
            query += " AND w1.complexity = %s"
            params.append(filters['complexity'])

        query += " LIMIT %s"
        params.append(filters.get('limit', 30))

        # Execute and return
        return self.db.execute(query, params)
```

---

## Four Modalities: Implementation Strategy

### 1. Quick Tools (Mostly Client-Side)

**Operations**: Minimal pairs, rhyme sets, maximal oppositions, phoneme position

**Strategy**: Pure client-side using cached graph

```javascript
// All handled client-side, no server calls needed
class QuickTools {
  minimalPairs(phoneme1, phoneme2, filters) {
    return this.graph.findEdges('MINIMAL_PAIR', filters);
  }

  rhymeSet(word, perfectOnly) {
    return this.graph.findEdges('RHYME', {
      source: word,
      filter: e => !perfectOnly || e.metadata.quality === 'perfect'
    });
  }

  maximalOppositions(word, excludedPhonemes) {
    const neighbors = this.graph.neighbors(word);
    return neighbors
      .filter(n => !n.phonemes.some(p => excludedPhonemes.includes(p)))
      .sort((a, b) => b.phoneme_diff - a.phoneme_diff)
      .slice(0, 20);
  }
}
```

**Performance**: <10ms per query (all in-memory)

### 2. Search (Hybrid: Client Filter + Server Similarity)

**Operations**: Word lookup, phoneme search, similarity search

**Strategy**:
- Word/phoneme lookup: Client-side (instant)
- Similarity search: Server-side (requires vectors)

```javascript
class Search {
  wordLookup(word) {
    // Pure client lookup
    return this.graph.nodes.get(word);
  }

  async similaritySearch(word, threshold) {
    // Step 1: Check cache for recent queries
    const cached = this.cache.get(`sim_${word}_${threshold}`);
    if (cached) return cached;

    // Step 2: Server call for vector similarity
    const results = await fetch(`/api/similarity/word/${word}?threshold=${threshold}`);

    // Step 3: Enrich with client-side graph data
    return results.map(r => ({
      ...r,
      neighborhood_density: this.graph.neighbors(r.word).length,
      edges: this.graph.edges.get(r.word)
    }));
  }
}
```

**Performance**:
- Word lookup: <1ms (client)
- Similarity search: <150ms (server) + <5ms (client enrichment)

### 3. Builder (Client-Side with Server Fallback)

**Operations**: Pattern matching (starts/ends/contains), property filtering

**Strategy**: Do 90% on client, use server for edge cases

```javascript
class Builder {
  patternSearch(patterns, properties) {
    // Step 1: Pattern matching (client-side)
    let candidates = Array.from(this.graph.nodes.values());

    patterns.forEach(p => {
      if (p.type === 'STARTS_WITH') {
        candidates = candidates.filter(w => w.phonemes[0] === p.phoneme);
      } else if (p.type === 'ENDS_WITH') {
        candidates = candidates.filter(w =>
          w.phonemes[w.phonemes.length - 1] === p.phoneme
        );
      } else if (p.type === 'CONTAINS') {
        candidates = candidates.filter(w =>
          w.phonemes.includes(p.phoneme)
        );
      }
    });

    // Step 2: Property filtering (client-side)
    if (properties.wcm_score) {
      const [min, max] = properties.wcm_score;
      candidates = candidates.filter(w =>
        w.wcm_score >= min && w.wcm_score <= max
      );
    }

    return candidates;
  }
}
```

**Performance**: <20ms for complex patterns (all client-side)

### 4. Compare (Client-Side)

**Operations**: Phoneme feature comparison

**Strategy**: Pure client lookup with cached phoneme data

```javascript
class Compare {
  comparePhonemes(ipa1, ipa2) {
    const p1 = this.phonemes.get(ipa1);
    const p2 = this.phonemes.get(ipa2);

    // Calculate feature distance
    const features = Object.keys(p1.features);
    const differences = features.filter(f =>
      p1.features[f] !== p2.features[f]
    );

    return {
      phoneme1: p1,
      phoneme2: p2,
      distance: differences.length / features.length,
      differences: differences
    };
  }
}
```

**Performance**: <1ms (pure JavaScript object comparison)

---

## Data Flow: Initial Load → Cached Operations

### Startup Sequence

```
User opens app
    ↓
Show loading screen (3-5 seconds)
    ↓
┌─────────────────────────────────────────────────┐
│ 1. Fetch compressed graph from server          │
│    GET /api/graph/export                        │
│    Returns: ~20-50MB gzipped JSON               │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ 2. Decompress in browser (WASM)                │
│    Uses pako.js or similar                      │
│    Time: ~500ms                                  │
└───────────────────────────��─────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ 3. Build in-memory graph structure              │
│    Create Map() for nodes and edges             │
│    Time: ~1000ms                                 │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ 4. Cache in service worker                      │
│    Persist for next session                     │
│    Time: ~500ms                                  │
└─────────────────────────────────────────────────┘
    ↓
Hide loading screen
    ↓
App ready (90% of queries now instant!)
```

### Query Flow (After Initial Load)

```
User requests minimal pairs for /t/ vs /d/
    ↓
Check cached graph (in-memory)
    ↓
Filter edges by relation type = 'MINIMAL_PAIR'
    ↓
Filter by phoneme1 = 't' and phoneme2 = 'd'
    ↓
Apply property filters (wcm_score, syllable_count)
    ↓
Return results
    ↓
Total time: <5ms (no network call!)
```

### When Server Is Needed

```
User requests similarity search for "cat"
    ↓
Check similarity cache (might be empty)
    ↓
POST /api/similarity/word/cat
    ↓
Server: pgvector ANN search (~150ms)
    ↓
Return top 50 similar words
    ↓
Client: Enrich with cached graph data (~5ms)
    ↓
Cache results for this query
    ↓
Total time: ~160ms (one-time server call, then cached)
```

---

## Deployment Architecture

### Development (Local)
```
localhost:3000 (React dev server)
    ↓
localhost:8000 (FastAPI with uvicorn --reload)
    ↓
localhost:5432 (PostgreSQL + pgvector in Docker)
```

### Production: Option 1 (Netlify + Neon) - **RECOMMENDED**
```
phonolex.netlify.app (React static site + Netlify Functions)
    ↓
Netlify Functions (Python/FastAPI serverless)
    ↓
Neon PostgreSQL (Native Netlify integration!)
```

**Why Netlify + Neon**:
- ✅ **Native integration**: Netlify has built-in Neon support (one-click setup!)
- ✅ Free tier: 3GB storage, 1 compute
- ✅ pgvector extension supported
- ✅ Serverless: scales to zero when not in use
- ✅ Connection pooling built-in
- ✅ Branch database for testing
- ✅ Environment variables auto-configured in Netlify
- ✅ Deploy previews get their own database branches

**Setup**: Literally click "Add Neon Database" in Netlify dashboard → done!

### Production: Option 2 (Cloudflare + Neon)
```
phonolex.pages.dev (React on Cloudflare Pages)
    ↓
Cloudflare Workers (FastAPI via Pyodide/WASM)
    ↓
neon.tech/phonolex (Serverless PostgreSQL with pgvector)
```

**Why Cloudflare**:
- ✅ Edge computing: API runs close to users globally
- ✅ KV storage: Cache hot queries at edge
- ✅ D1 database: Optional SQLite for simple queries
- ✅ Workers KV: Store phoneme lookup tables

### Production: Option 3 (Cloudflare + D1 + Neon Hybrid)
```
phonolex.pages.dev (React on Cloudflare Pages)
    ↓
Cloudflare Workers (Thin routing layer)
    ↓
┌─────────────────┬──────────────────┐
│ Cloudflare D1   │ Neon PostgreSQL  │
│ (Simple queries)│ (Vector search)  │
└─────────────────┴──────────────────┘
```

**Hybrid Strategy**:
- **D1 (SQLite)**: Handle 80% of queries (minimal pairs, rhymes, property filtering)
- **Neon (PostgreSQL)**: Handle 20% (vector similarity, complex graph queries)
- **Why**: D1 is free, instant at edge, perfect for simple graph queries
- **Tradeoff**: Need to keep D1 and Neon in sync (but worth it for performance)

### Production: Option 4 (Supabase - All-in-One)
```
phonolex.netlify.app (React static site)
    ↓
supabase.co/phonolex (Supabase API - built-in REST/GraphQL)
    ↓
supabase.co/phonolex (PostgreSQL with pgvector + Row Level Security)
```

**Why Supabase**:
- ✅ Free tier: 500MB database, 2GB bandwidth
- ✅ Instant REST API (no FastAPI needed!)
- ✅ pgvector extension supported
- ✅ Real-time subscriptions (if we want live updates)
- ✅ Row Level Security (authentication built-in)
- ✅ Edge Functions (Deno-based, deploy alongside DB)

**Simplest Option**: Supabase eliminates the need for a custom backend - it auto-generates REST/GraphQL APIs from your schema!

### Recommended: Netlify + Neon (MVP & Production)

**Why Netlify + Neon is the Winner**:
1. **Native integration**: One-click database setup in Netlify dashboard
2. **Preview branches**: Each deploy preview gets its own database branch
3. **Environment variables**: Auto-configured, no manual setup
4. **Generous free tier**: 3GB Neon + Netlify's free tier = perfect for MVP
5. **Production-ready**: Same stack for MVP → scale (no migration needed)

**Setup Flow**:
```bash
# 1. Deploy React to Netlify (drag & drop or Git)
# 2. Click "Add Neon Database" in Netlify dashboard
# 3. Netlify automatically:
#    - Creates Neon project
#    - Provisions PostgreSQL database
#    - Sets DATABASE_URL environment variable
#    - Configures connection pooling
# 4. Deploy Netlify Functions (FastAPI)
# 5. Done! Total time: <1 hour
```

**Alternative for Different Use Cases**:
- **Need auto-generated API?** → Supabase (eliminates backend code)
- **Need edge performance?** → Cloudflare + Neon (global edge computing)
- **Need hybrid DB?** → Cloudflare D1 + Neon (SQLite at edge + PostgreSQL for vectors)

### Database Provider Comparison

| Provider | Free Tier | pgvector | Edge | Netlify Integration | Setup |
|----------|-----------|----------|------|---------------------|-------|
| **Netlify + Neon** | 3GB, always on | ✅ | ❌ | ✅ **Native!** | 1-click |
| **Supabase** | 500MB, 2GB bandwidth | ✅ | ✅ | Manual | 5 min |
| **Cloudflare D1** | 5GB, unlimited reads | ❌ (SQLite) | ✅ | Manual | 10 min |
| **Railway** | $5/month credit | ✅ | ❌ | Manual | 5 min |
| **Render** | Free 90 days, then $7/mo | ✅ | ❌ | Manual | 10 min |

**Winner for Netlify users**: **Neon** (native integration + generous free tier + pgvector + deploy preview branches)

**Winner for non-Netlify**: **Supabase** (auto-generated API + edge functions)

### Scaling Considerations

**Current Scale** (MVP):
- 26K words, 56K edges
- ~100MB in-memory graph
- Single PostgreSQL instance handles it easily

**Future Scale** (10x growth):
- 260K words, 560K edges
- ~1GB in-memory graph
- **Client-side**: Still feasible (chunked loading, lazy edges)
- **Server-side**: Add read replicas, Redis cache layer

---

## Performance Budget

### Initial Load
- **Target**: <5 seconds (with loading screen)
- **Breakdown**:
  - Fetch graph: 2 seconds (20-50MB over network)
  - Decompress: 0.5 seconds (WASM)
  - Build structures: 1 second (JavaScript)
  - Cache: 0.5 seconds (Service Worker)
  - Buffer: 1 second

### Client-Side Queries (90% of operations)
- **Target**: <10ms
- **Operations**: Minimal pairs, rhymes, property filtering, pattern matching, word lookup

### Server-Side Queries (10% of operations)
- **Target**: <200ms
- **Operations**: Vector similarity, embedding computation, complex multi-layer queries

### Memory Budget
- **Client**: 100MB for graph (acceptable for modern browsers)
- **Server**: 1GB RAM per instance (supports 100+ concurrent users)

---

## Migration Plan: Remote Database

### Phase 1: Local Development (Current)
- PostgreSQL on localhost
- NetworkX graph in FastAPI process
- React dev server

### Phase 2: Staging (Netlify + Fly.io + Neon)
- Deploy React to Netlify
- Deploy FastAPI to Fly.io
- Migrate PostgreSQL to Neon.tech (or Supabase)
- Test end-to-end with remote DB

### Phase 3: Production (Optimize)
- Enable CDN caching for /api/graph/export
- Add Redis cache for frequent queries
- Monitor query performance
- Optimize slow queries

### Phase 4: Scale (If Needed)
- Add read replicas for PostgreSQL
- Implement query result caching
- Consider edge computing (Cloudflare Workers)

---

## Technology Stack

### Frontend
- **React 18**: UI framework
- **TypeScript**: Type safety
- **Zustand**: State management (lightweight)
- **MUI**: Component library
- **pako.js**: Gzip compression/decompression
- **Service Worker**: Graph caching

### Backend
- **FastAPI**: Python web framework
- **PostgreSQL 15+**: Relational database
- **pgvector**: Vector similarity extension
- **NetworkX**: Graph algorithms (Python)
- **PyTorch**: Embedding model inference
- **Uvicorn**: ASGI server

### Deployment
- **Netlify**: Static site hosting + CDN
- **Fly.io**: Docker container hosting (FastAPI)
- **Neon.tech**: Serverless PostgreSQL with pgvector
- **Sentry**: Error tracking
- **Vercel Analytics**: Usage monitoring

---

## Implementation Phases

### Week 1-2: Database Schema + Population
- [ ] Create PostgreSQL schema with pgvector
- [ ] Populate phonemes table (103 phonemes)
- [ ] Populate words table (26,076 words)
- [ ] Generate typed edges (56,433 edges)
- [ ] Create indexes (GIN, HNSW, B-tree)

### Week 3-4: Backend Service Layer
- [ ] Implement DatabaseService (query methods)
- [ ] Implement GraphService (NetworkX wrapper)
- [ ] Implement PhonoLexService (unified API)
- [ ] Create FastAPI endpoints (4 modalities)
- [ ] Add /api/graph/export endpoint

### Week 5-6: Frontend Client Graph
- [ ] Implement graph loading screen
- [ ] Build in-memory graph structure (JavaScript)
- [ ] Implement client-side algorithms
- [ ] Add Service Worker caching
- [ ] Integrate with existing React components

### Week 7-8: Testing + Optimization
- [ ] Performance testing (query times)
- [ ] Memory profiling (client graph size)
- [ ] Algorithm optimization (bottleneck analysis)
- [ ] End-to-end testing (all modalities)
- [ ] Load testing (concurrent users)

### Week 9-10: Deployment + Migration
- [ ] Deploy to Netlify (frontend)
- [ ] Deploy to Fly.io (backend)
- [ ] Migrate to Neon.tech (database)
- [ ] Configure CDN caching
- [ ] Monitor performance in production

---

## Key Metrics

### Success Criteria
- ✅ Initial load: <5 seconds
- ✅ 90% of queries: <10ms (client-side)
- ✅ Vector similarity: <200ms
- ✅ Memory usage: <150MB (client)
- ✅ Server response: <200ms (p95)
- ✅ Concurrent users: 100+ (single instance)

### Monitoring
- Query latency (p50, p95, p99)
- Cache hit rate (client-side)
- API call frequency (% client vs server)
- Memory usage (client graph size)
- Error rate (API failures)

---

## Advantages of This Architecture

### 1. Speed
- 90% of queries instant (client-side cache)
- No network latency for common operations
- Offline-capable after initial load

### 2. Scalability
- Server only handles 10% of traffic
- Client does heavy lifting (graph traversal)
- Cheap to host (low server load)

### 3. User Experience
- Instant feedback for most interactions
- Loading screen only once per session
- Works offline after initial load

### 4. Developer Experience
- Clear separation of concerns (client vs server)
- Easy to test (graph operations in pure JS)
- Simple deployment (static frontend + API)

### 5. Cost Efficiency
- Low server costs (minimal API calls)
- No need for expensive infrastructure
- CDN caching reduces bandwidth

---

## Risks & Mitigations

### Risk 1: Client Memory Issues
**Problem**: 100MB graph might crash on low-memory devices

**Mitigation**:
- Lazy-load edges (only load when needed)
- Implement chunked loading (load graph in pieces)
- Provide "lite mode" with reduced graph

### Risk 2: Initial Load Too Slow
**Problem**: 5-second load time might frustrate users

**Mitigation**:
- Compress graph aggressively (gzip + msgpack)
- Use CDN for /api/graph/export (edge caching)
- Progressive loading (load core data first, edges later)
- Show progress bar with % loaded

### Risk 3: Graph Update Complexity
**Problem**: How to update client cache when server data changes?

**Mitigation**:
- Version graph exports (v2.0.0, v2.0.1, etc.)
- Check version on startup, reload if outdated
- Implement incremental updates (delta patches)

### Risk 4: Algorithm Bottlenecks
**Problem**: Some client-side algorithms might be slow

**Mitigation**:
- Profile all algorithms (identify slow operations)
- Use Web Workers for heavy computation
- Implement WASM for critical paths
- Server fallback for expensive operations

---

## Future Enhancements

### Short-Term (Months 1-3)
- [ ] Add morphological edges (plural, past tense)
- [ ] Implement treatment progression planner
- [ ] Add word difficulty scorer
- [ ] Create saved query templates

### Medium-Term (Months 4-6)
- [ ] Multi-language support (Spanish, Mandarin)
- [ ] User accounts & saved lists
- [ ] Custom edge types (user-defined relationships)
- [ ] Real-time collaborative filtering

### Long-Term (Months 7-12)
- [ ] Mobile app (React Native)
- [ ] Offline-first PWA
- [ ] API for third-party integrations
- [ ] Machine learning features (smart suggestions)

---

## Conclusion

This database-centric architecture leverages:
- **PostgreSQL + pgvector** for all data storage and vector similarity
- **Smart indexing** (GIN, B-tree, HNSW) for fast queries
- **Typed graph edges** stored as relational data (not separate graph DB)
- **Thin API layer** (FastAPI or Supabase auto-generated) for query translation
- **Serverless deployment** (Netlify/Cloudflare + Neon/Supabase)

**Key Innovation**: Push all computation to the database with proper indexes, avoid transferring data to client for processing.

**Result**:
- ✅ <100ms query times for all operations
- ✅ Minimal backend code (API is mostly SQL translation)
- ✅ Low cost (serverless databases with generous free tiers)
- ✅ Scales automatically (Cloudflare edge + serverless DB)
- ✅ Simple deployment (no complex infrastructure)

**Trade-off Accepted**:
- ❌ Requires network call for every query (no offline mode)
- ✅ But: With edge caching (Cloudflare), queries are still <100ms globally
- ✅ And: Simpler architecture, easier to maintain

**Recommended Stack for MVP**:
1. **Frontend**: React on Netlify
2. **Backend**: Netlify Functions (FastAPI)
3. **Database**: Neon PostgreSQL with pgvector (native Netlify integration!)
4. **Deployment time**: <1 hour (one-click Neon setup!)

**Why This Stack**:
- Native integration means zero configuration
- Deploy preview branches get their own database
- Free tier is generous enough for production
- Scale seamlessly without platform changes

---

## Appendix: Embedding Granularities

From our earlier verification, we support **four embedding granularities** on graph nodes:

### 1. Raw Phoible Features (38-dim)
- **Location**: `phonemes.raw_features`
- **Computable**: No (raw input data)
- **Use**: Feature lookups, exact matching

### 2. Normalized Feature Vectors (76-dim + 152-dim)
- **Location**: `phonemes.endpoints_76d`, `phonemes.trajectories_152d`
- **Computable**: Yes (via `PhonemeVectorizer`)
- **Use**: Phoneme similarity, trajectory analysis

### 3. Tuned Phoneme Embeddings (128-dim)
- **Location**: `phonemes.contextual_128d`
- **Source**: `models/curriculum/phoible_initialized_final.pt`
- **Computable**: Yes (model inference)
- **Use**: Contextual phoneme representations

### 4. Hierarchical Word/Syllable Embeddings (384-dim)
- **Location**: `words.syllable_embedding`
- **Source**: `embeddings/layer4/syllable_embeddings.pt`
- **Computable**: Yes (model inference)
- **Use**: Word similarity, rhyme detection

**Storage Strategy**:
- Store in PostgreSQL with pgvector
- Cache in client for frequently accessed words
- Compute on-demand for rare queries

---

**Document Version**: 2.0.0
**Last Updated**: 2025-10-28
**Status**: Ready for Implementation
