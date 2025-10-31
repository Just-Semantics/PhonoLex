/**
 * PhonoLex API (Client-Side Mode)
 *
 * This file now exports the client-side data adapter instead of the backend API.
 * All data is loaded from static JSON files and computed in the browser.
 */

import type { WordFilterRequest } from '../types/phonology';

// Re-export types from phonology.ts
export type {
  Word,
  PhonemeDetail as Phoneme,
  SimilarityResult as SimilarWord,
  MinimalPairResult as MinimalPair,
  PhonemeDetail,
  EdgeType,
  WordFilterRequest,
} from '../types/phonology';

// Define types previously from backend
export type PatternType = 'STARTS_WITH' | 'ENDS_WITH' | 'CONTAINS' | 'CONTAINS_MEDIAL';

export interface Pattern {
  type: PatternType;
  phoneme: string;
  medial_only?: boolean;  // For CONTAINS type
}

export interface BuilderRequest {
  patterns: Pattern[];
  filters?: WordFilterRequest;
  exclusions?: {
    exclude_phonemes?: string[];
  };
  limit?: number;
}

export interface PhonemeComparison {
  phoneme1: string;
  phoneme2: string;
  shared_features: Record<string, string>;
  different_features: Record<string, [string, string]>;
  similarity_score: number;
}

// Export the client-side adapter as the API
export { api, api as default } from './clientSideApiAdapter';
