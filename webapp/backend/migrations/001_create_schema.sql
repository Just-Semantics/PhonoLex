-- ============================================================================
-- PhonoLex v2.0 Database Schema
-- PostgreSQL 14+ with pgvector extension
-- ============================================================================

-- Enable pgvector extension (idempotent)
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- WORDS TABLE
-- Core word lexicon with phonological structure and embeddings
-- ============================================================================
CREATE TABLE IF NOT EXISTS words (
    word_id SERIAL PRIMARY KEY,
    word VARCHAR(100) UNIQUE NOT NULL,
    ipa TEXT NOT NULL,

    -- Phonological structure (denormalized for speed)
    phonemes_json JSONB NOT NULL,  -- [{"ipa": "k", "position": 0}, ...]
    syllables_json JSONB NOT NULL, -- [{"onset": ["k"], "nucleus": "æ", "coda": ["t"]}]

    -- Counts
    phoneme_count INT NOT NULL,
    syllable_count INT NOT NULL,

    -- Psycholinguistic properties (nullable - not all words have these)
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
    syllable_embedding vector(384),     -- Hierarchical model (384-dim)
    word_embedding_flat vector(64),     -- Simple word embedding (64-dim, optional)

    -- Categorical properties (derived from numerical scores)
    word_length VARCHAR(10),      -- 'short', 'medium', 'long'
    complexity VARCHAR(10),       -- 'low', 'medium', 'high' (from WCM)

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for fast filtering
CREATE INDEX IF NOT EXISTS idx_words_phoneme_count ON words(phoneme_count);
CREATE INDEX IF NOT EXISTS idx_words_syllable_count ON words(syllable_count);
CREATE INDEX IF NOT EXISTS idx_words_wcm ON words(wcm_score);
CREATE INDEX IF NOT EXISTS idx_words_aoa ON words(aoa);
CREATE INDEX IF NOT EXISTS idx_words_frequency ON words(frequency);
CREATE INDEX IF NOT EXISTS idx_words_word_length ON words(word_length);
CREATE INDEX IF NOT EXISTS idx_words_complexity ON words(complexity);

-- JSONB indexes for pattern matching
CREATE INDEX IF NOT EXISTS idx_words_phonemes_gin ON words USING gin (phonemes_json jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_words_syllables_gin ON words USING gin (syllables_json jsonb_path_ops);

-- pgvector HNSW indexes for fast ANN similarity search
-- Using cosine distance (most common for normalized embeddings)
CREATE INDEX IF NOT EXISTS idx_words_syllable_embedding
    ON words USING hnsw (syllable_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);  -- m=16: neighbors, ef_construction=64: build quality

-- ============================================================================
-- PHONEMES TABLE
-- Phoneme inventory with Phoible features and multiple embedding granularities
-- ============================================================================
CREATE TABLE IF NOT EXISTS phonemes (
    phoneme_id SERIAL PRIMARY KEY,
    ipa VARCHAR(10) UNIQUE NOT NULL,
    segment_class VARCHAR(20) NOT NULL,  -- 'consonant', 'vowel', 'tone'

    -- PHOIBLE features (38 dimensions, ternary: +/-/0)
    features JSONB NOT NULL,

    -- Vector representations (multiple granularities)
    raw_features vector(38),              -- Raw ternary features (normalized to -1/0/+1)
    endpoints_76d vector(76),              -- Start + end states (for simple phonemes)
    trajectories_152d vector(152),         -- 4 timesteps (for diphthongs/complex)
    contextual_128d vector(128),           -- From hierarchical model (context-aware)

    -- Trajectory metadata
    has_trajectory BOOLEAN DEFAULT FALSE,
    trajectory_features TEXT[],

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for phoneme lookups
CREATE INDEX IF NOT EXISTS idx_phonemes_segment_class ON phonemes(segment_class);
CREATE INDEX IF NOT EXISTS idx_phonemes_features_gin ON phonemes USING gin (features jsonb_path_ops);

-- ============================================================================
-- WORD_EDGES TABLE
-- Graph relationships between words (typed edges with metadata)
-- ============================================================================
CREATE TABLE IF NOT EXISTS word_edges (
    edge_id SERIAL PRIMARY KEY,
    word1_id INT NOT NULL REFERENCES words(word_id) ON DELETE CASCADE,
    word2_id INT NOT NULL REFERENCES words(word_id) ON DELETE CASCADE,

    -- Edge type (supports multiple relationship types)
    relation_type VARCHAR(50) NOT NULL,

    -- Relationship-specific metadata (stored as JSONB for flexibility)
    -- Examples:
    --   MINIMAL_PAIR: {"position": 0, "phoneme1": "t", "phoneme2": "d", "feature_diff": 1}
    --   RHYME: {"rhyme_type": "perfect", "syllables_matched": 2, "quality": 0.99}
    --   NEIGHBOR: {"edit_distance": 1, "phoneme_diff": [{"pos": 2, "from": "t", "to": "d"}]}
    metadata JSONB NOT NULL DEFAULT '{}',

    -- Precomputed weight for graph algorithms
    -- Lower = closer/more similar (for shortest path algorithms)
    weight FLOAT DEFAULT 1.0,

    -- Ensure data integrity
    CONSTRAINT no_self_loops CHECK (word1_id != word2_id),
    CONSTRAINT ordered_pair CHECK (word1_id < word2_id),  -- Avoid duplicate pairs
    CONSTRAINT unique_edge UNIQUE (word1_id, word2_id, relation_type),

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for fast edge queries (critical for graph traversal)
CREATE INDEX IF NOT EXISTS idx_edges_word1 ON word_edges(word1_id);
CREATE INDEX IF NOT EXISTS idx_edges_word2 ON word_edges(word2_id);
CREATE INDEX IF NOT EXISTS idx_edges_relation ON word_edges(relation_type);
CREATE INDEX IF NOT EXISTS idx_edges_word1_relation ON word_edges(word1_id, relation_type);
CREATE INDEX IF NOT EXISTS idx_edges_word2_relation ON word_edges(word2_id, relation_type);
CREATE INDEX IF NOT EXISTS idx_edges_metadata_gin ON word_edges USING gin (metadata jsonb_path_ops);

-- Composite index for common query pattern: find edges of specific type between word pairs
CREATE INDEX IF NOT EXISTS idx_edges_word1_word2_relation
    ON word_edges(word1_id, word2_id, relation_type);

-- ============================================================================
-- VIEWS
-- Convenience views for common queries
-- ============================================================================

-- View: All edges (bidirectional) - makes graph queries easier
CREATE OR REPLACE VIEW word_edges_bidirectional AS
    SELECT edge_id, word1_id as source_id, word2_id as target_id, relation_type, metadata, weight
    FROM word_edges
    UNION ALL
    SELECT edge_id, word2_id as source_id, word1_id as target_id, relation_type, metadata, weight
    FROM word_edges;

-- ============================================================================
-- FUNCTIONS
-- Helper functions for common operations
-- ============================================================================

-- Function: Update updated_at timestamp automatically
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger: Auto-update updated_at for words table
DROP TRIGGER IF EXISTS update_words_updated_at ON words;
CREATE TRIGGER update_words_updated_at
    BEFORE UPDATE ON words
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Trigger: Auto-update updated_at for phonemes table
DROP TRIGGER IF EXISTS update_phonemes_updated_at ON phonemes;
CREATE TRIGGER update_phonemes_updated_at
    BEFORE UPDATE ON phonemes
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- COMMENTS
-- Document the schema for future reference
-- ============================================================================

COMMENT ON TABLE words IS 'Core word lexicon with phonological structure, psycholinguistic properties, and embeddings';
COMMENT ON TABLE phonemes IS 'Phoneme inventory with Phoible features and multiple embedding granularities';
COMMENT ON TABLE word_edges IS 'Typed graph relationships between words (minimal pairs, rhymes, neighbors, etc.)';

COMMENT ON COLUMN words.phonemes_json IS 'Array of phonemes with position: [{"ipa": "k", "position": 0}, ...]';
COMMENT ON COLUMN words.syllables_json IS 'Array of syllables with onset-nucleus-coda structure';
COMMENT ON COLUMN words.syllable_embedding IS '384-dim hierarchical syllable embedding from trained model';
COMMENT ON COLUMN words.wcm_score IS 'Word Complexity Measure (1-10 scale)';

COMMENT ON COLUMN word_edges.relation_type IS 'Edge type: MINIMAL_PAIR, RHYME, NEIGHBOR, MAXIMAL_OPP, SIMILAR, MORPHOLOGICAL';
COMMENT ON COLUMN word_edges.metadata IS 'Relationship-specific metadata (position, phoneme differences, similarity scores, etc.)';
COMMENT ON COLUMN word_edges.weight IS 'Precomputed edge weight for graph algorithms (lower = closer)';

-- ============================================================================
-- VALIDATION
-- Verify schema was created successfully
-- ============================================================================

-- Check table counts
SELECT
    schemaname,
    tablename,
    CASE
        WHEN tablename IN ('words', 'phonemes', 'word_edges') THEN '✓'
        ELSE '?'
    END as status
FROM pg_tables
WHERE schemaname = 'public'
    AND tablename IN ('words', 'phonemes', 'word_edges')
ORDER BY tablename;

-- Check indexes
SELECT
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE schemaname = 'public'
    AND tablename IN ('words', 'phonemes', 'word_edges')
ORDER BY tablename, indexname;
