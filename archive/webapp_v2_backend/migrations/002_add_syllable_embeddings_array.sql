-- Migration: Add syllable_embeddings array column for proper word similarity
-- Date: 2025-10-29
-- Purpose: Store per-syllable embeddings for soft Levenshtein word similarity

-- Add new column for array of syllable embeddings
ALTER TABLE words
ADD COLUMN IF NOT EXISTS syllable_embeddings vector(384)[];

-- Add comment
COMMENT ON COLUMN words.syllable_embeddings IS 'Array of 384-dim syllable embeddings (one per syllable) for soft Levenshtein word similarity';
COMMENT ON COLUMN words.syllable_embedding IS 'DEPRECATED: Single syllable embedding (first syllable only), will be removed in future';
