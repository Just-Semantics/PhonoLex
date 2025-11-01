/**
 * Type definitions for phonological data structures (v2.0)
 * Matches backend SQLAlchemy models and Pydantic schemas
 */

// ============================================================================
// Phoneme Types
// ============================================================================

export type PhonemeType = 'vowel' | 'consonant';
export type SegmentClass = 'consonant' | 'vowel' | 'tone';

export interface PhonemeConstraint {
  position?: number | null;  // null/undefined = "contains" mode (phoneme anywhere in word)
  phoneme_type?: PhonemeType;  // snake_case to match backend API
  allowed_phonemes?: string[];  // snake_case to match backend API
  required_features?: Record<string, string>;  // snake_case to match backend API
}

export interface PhonemeInfo {
  phoneme: string;
  type: PhonemeType;
  features: Record<string, string>;
}

export interface PhonemeDetail {
  phoneme_id: number;
  ipa: string;
  segment_class: SegmentClass;
  features: Record<string, string>;
  has_trajectory: boolean;
  trajectory_features?: string[];
}

export interface PhonemeListResponse {
  vowels: string[];
  consonants: string[];
}

// ============================================================================
// Word Types
// ============================================================================

export interface Syllable {
  onset: string[];
  nucleus: string;
  coda: string[];
}

export interface PhonemePosition {
  ipa: string;
  arpa?: string;  // ARPAbet (client-side only)
  position: number;
}

/**
 * Complete word data structure matching backend v2.0 schema
 */
export interface Word {
  word_id: number;
  word: string;
  ipa: string;
  arpa?: string;  // ARPAbet pronunciation (client-side only)

  // Phonological structure
  phonemes: PhonemePosition[];  // Array of {ipa, position}
  syllables: Syllable[];  // Array of {onset, nucleus, coda}
  phoneme_count: number;
  syllable_count: number;

  // Psycholinguistic properties
  frequency: number | null;
  log_frequency: number | null;
  aoa: number | null;  // Age of Acquisition
  imageability: number | null;
  familiarity: number | null;
  concreteness: number | null;
  valence: number | null;  // Emotional valence
  arousal: number | null;  // Emotional arousal
  dominance: number | null;  // Emotional dominance

  // Clinical measures
  wcm_score: number | null;  // Word Complexity Measure
  msh_stage: number | null;  // Motor Speech Hierarchy stage
}

/**
 * Legacy word result (for backward compatibility with v1.0)
 */
export interface WordResult {
  word: string;
  phonemes: string[];
  stress: (number | null)[];
  syllables: number;
  ipa: string;
  frequency: number;  // Log10 word frequency from SUBTLEX
}

// ============================================================================
// Filter Types
// ============================================================================

export interface WordFilter {
  pattern: PhonemeConstraint[];
  minSyllables?: number;
  maxSyllables?: number;
  minPhonemes?: number;
  maxPhonemes?: number;
  limit?: number;  // Max results for pagination
  offset?: number; // Pagination offset
}

export interface WordFilterRequest {
  min_syllables?: number;
  max_syllables?: number;
  min_phonemes?: number;
  max_phonemes?: number;

  // Psycholinguistic properties
  min_frequency?: number;
  max_frequency?: number;
  min_aoa?: number;
  max_aoa?: number;
  min_imageability?: number;
  max_imageability?: number;
  min_familiarity?: number;
  max_familiarity?: number;
  min_concreteness?: number;
  max_concreteness?: number;
  min_valence?: number;
  max_valence?: number;
  min_arousal?: number;
  max_arousal?: number;
  min_dominance?: number;
  max_dominance?: number;

  // Clinical measures
  min_wcm?: number;
  max_wcm?: number;
  min_msh?: number;
  max_msh?: number;

  limit?: number;
  offset?: number;
}

export interface PatternSearchRequest {
  starts_with?: string;
  ends_with?: string;
  contains?: string;
  contains_medial_only?: boolean;  // For CONTAINS: exclude word edges
  min_syllables?: number;
  max_syllables?: number;
  filters?: WordFilterRequest;  // Additional filters
  limit?: number;
}

export interface FilterResponse {
  count: number;
  words: WordResult[];
}

// ============================================================================
// Edge / Graph Types
// ============================================================================

export type EdgeType =
  | 'MINIMAL_PAIR'
  | 'RHYME'
  | 'NEIGHBOR'
  | 'MAXIMAL_OPP'
  | 'SIMILAR'
  | 'MORPHOLOGICAL'
  | 'embedding_similarity';  // Legacy from pickle graph

export interface MinimalPairMetadata {
  position: number;
  phoneme1: string;
  phoneme2: string;
  feature_diff?: number;
}

export interface RhymeMetadata {
  rhyme_type: 'perfect' | 'slant' | 'assonance';
  nucleus: string;
  coda: string[];
  syllables_matched?: number;
  quality: number;
}

export interface NeighborMetadata {
  edit_distance: number;
  phoneme_diff: string[];
}

export type EdgeMetadata = MinimalPairMetadata | RhymeMetadata | NeighborMetadata | Record<string, unknown>;

export interface WordEdge {
  word1: string;
  word2: string;
  relation_type: EdgeType;
  metadata: EdgeMetadata;
  weight: number;
}

export interface NeighborResult {
  neighbor: Word;
  edge: WordEdge;
}

export interface MinimalPairResult {
  word1: Word;
  word2: Word;
  position?: number;
  phoneme1?: string;
  phoneme2?: string;
  feature_diff?: number;
  metadata?: {
    position: number;
    phoneme1: string;
    phoneme2: string;
  };
}

export interface RhymeResult {
  rhyme?: Word;
  word?: Word;  // Alternative field name for client-side
  rhyme_type?: string;
  nucleus?: string;
  coda?: string[];
  quality?: number;
  metadata?: {
    rhyme_type: string;
    nucleus: string;
    coda: string[];
    quality: number;
  };
}

// ============================================================================
// Similarity Types
// ============================================================================

export interface SimilarityResult {
  word: Word;
  similarity: number;
}

export interface SimilaritySearchRequest {
  word: string;
  threshold?: number;
  limit?: number;
  filters?: {
    max_wcm?: number;
  };
}

// ============================================================================
// Response Types
// ============================================================================

export interface StatsResponse {
  total_words: number;
  total_phonemes: number;
  vowels?: number;
  consonants?: number;
  indexed_positions?: number;
  total_edges?: number;
  edge_types?: Record<string, number>;
}
