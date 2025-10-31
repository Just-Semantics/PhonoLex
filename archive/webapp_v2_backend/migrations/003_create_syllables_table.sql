-- Migration: Create syllables lookup table
-- Date: 2025-10-29
-- Purpose: Store unique syllables with embeddings for efficient word similarity computation

-- Create syllables table
CREATE TABLE IF NOT EXISTS syllables (
    syllable_id SERIAL PRIMARY KEY,

    -- Syllable structure (JSONB for fast queries)
    structure JSONB NOT NULL,  -- {onset: [...], nucleus: "...", coda: [...]}

    -- Text representations
    ipa TEXT NOT NULL,  -- IPA string representation

    -- Phoneme counts
    onset_count INTEGER NOT NULL DEFAULT 0,
    coda_count INTEGER NOT NULL DEFAULT 0,
    total_phonemes INTEGER NOT NULL DEFAULT 1,  -- At least nucleus

    -- Embedding (384-dim: onset + nucleus + coda)
    embedding vector(384) NOT NULL,

    -- Statistics
    frequency INTEGER NOT NULL DEFAULT 0,  -- How many words contain this syllable

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Unique constraint on structure
    UNIQUE(ipa)
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_syllables_ipa ON syllables(ipa);
CREATE INDEX IF NOT EXISTS idx_syllables_onset_count ON syllables(onset_count);
CREATE INDEX IF NOT EXISTS idx_syllables_coda_count ON syllables(coda_count);
CREATE INDEX IF NOT EXISTS idx_syllables_frequency ON syllables(frequency DESC);

-- Create GIN index on structure JSONB for fast queries
CREATE INDEX IF NOT EXISTS idx_syllables_structure_gin ON syllables USING GIN (structure);

-- Create word_syllables junction table (maintains syllable order)
CREATE TABLE IF NOT EXISTS word_syllables (
    word_id INTEGER NOT NULL REFERENCES words(word_id) ON DELETE CASCADE,
    syllable_id INTEGER NOT NULL REFERENCES syllables(syllable_id) ON DELETE CASCADE,
    position INTEGER NOT NULL,  -- 0-indexed position in word

    PRIMARY KEY (word_id, position),
    FOREIGN KEY (word_id) REFERENCES words(word_id) ON DELETE CASCADE,
    FOREIGN KEY (syllable_id) REFERENCES syllables(syllable_id) ON DELETE CASCADE
);

-- Create indexes on junction table
CREATE INDEX IF NOT EXISTS idx_word_syllables_word_id ON word_syllables(word_id);
CREATE INDEX IF NOT EXISTS idx_word_syllables_syllable_id ON word_syllables(syllable_id);

-- Add comments
COMMENT ON TABLE syllables IS 'Unique syllables with 384-dim embeddings for soft Levenshtein word similarity';
COMMENT ON TABLE word_syllables IS 'Junction table mapping words to ordered syllables';
COMMENT ON COLUMN syllables.embedding IS '384-dim syllable embedding (onset + nucleus + coda)';
COMMENT ON COLUMN word_syllables.position IS '0-indexed syllable position within word';
