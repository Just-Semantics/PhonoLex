/**
 * PhonoLex API Client (v2.0)
 *
 * Connects to FastAPI backend (http://localhost:8000)
 * Full v2.0 endpoints with complete data structures
 */

import type {
  Word,
  WordFilterRequest,
  PatternSearchRequest,
  SimilarityResult,
  SimilaritySearchRequest,
  NeighborResult,
  MinimalPairResult,
  RhymeResult,
  StatsResponse,
  PhonemeListResponse,
  EdgeType,
  PhonemeDetail,
} from '../types/phonology';

// Re-export types for backward compatibility with old components
export type {
  Word,
  PhonemeDetail as Phoneme,
  SimilarityResult as SimilarWord,
  MinimalPairResult as MinimalPair,
  PhonemeDetail,
  EdgeType,
};

// Legacy types for Builder component (v1.0)
export type PatternType = 'STARTS_WITH' | 'ENDS_WITH' | 'CONTAINS';

export interface Pattern {
  type: PatternType;
  phoneme: string;
  position_range?: [number, number];
  medial_only?: boolean;  // For CONTAINS: if true, excludes initial/final positions
}

export interface BuilderRequest {
  patterns?: Pattern[];
  filters?: {
    min_syllables?: number;
    max_syllables?: number;
    min_wcm?: number;
    max_wcm?: number;
    min_msh?: number;
    max_msh?: number;
    min_aoa?: number;
    max_aoa?: number;
  };
  exclusions?: {
    exclude_phonemes?: string[];
    features?: Record<string, string>;
  };
  limit?: number;
}

export interface PhonemeComparison {
  phoneme1: PhonemeDetail;
  phoneme2: PhonemeDetail;
  feature_distance: number;
  differing_features: string[];
  shared_features: string[];
}

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// ============================================================================
// API Client
// ============================================================================

class PhonoLexAPI {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    endpoint: string,
    options?: RequestInit
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`API Error: ${response.status} - ${error}`);
    }

    return response.json();
  }

  // ==========================================================================
  // Health Check
  // ==========================================================================

  /**
   * Health check endpoint
   */
  async healthCheck(): Promise<{
    status: string;
    database: string;
    pgvector: string;
    words: number;
    edges: number;
  }> {
    return this.request('/health');
  }

  // ==========================================================================
  // Word Operations
  // ==========================================================================

  /**
   * Get word by name (v2.0)
   */
  async getWord(word: string): Promise<Word> {
    return this.request(`/api/words/${encodeURIComponent(word)}`);
  }

  /**
   * Filter words by properties (v2.0)
   */
  async filterWords(request: WordFilterRequest): Promise<Word[]> {
    return this.request('/api/words/filter', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  /**
   * Search words by phoneme patterns (v2.0)
   *
   * Supports:
   * - starts_with: Word must start with this phoneme (IPA)
   * - ends_with: Word must end with this phoneme (IPA)
   * - contains: Word must contain this phoneme (IPA)
   */
  async patternSearch(request: PatternSearchRequest): Promise<Word[]> {
    return this.request('/api/words/pattern-search', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  /**
   * Get database statistics (v2.0)
   */
  async getStats(): Promise<StatsResponse> {
    return this.request('/api/words/stats/summary');
  }

  // ==========================================================================
  // Similarity Operations (v2.0 with pgvector)
  // ==========================================================================

  /**
   * Find similar words using hierarchical syllable embeddings (v2.0)
   *
   * Uses pgvector's HNSW index for fast approximate nearest neighbor search
   */
  async findSimilarWords(
    word: string,
    threshold: number = 0.85,
    limit: number = 50
  ): Promise<SimilarityResult[]> {
    const params = new URLSearchParams({
      threshold: threshold.toString(),
      limit: limit.toString(),
    });
    return this.request(`/api/similarity/word/${encodeURIComponent(word)}?${params}`);
  }

  /**
   * Similarity search with additional filters (v2.0)
   */
  async similaritySearch(request: SimilaritySearchRequest): Promise<SimilarityResult[]> {
    return this.request('/api/similarity/search', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  // ==========================================================================
  // Graph Operations (v2.0)
  // ==========================================================================

  /**
   * Get neighboring words in phonological graph (v2.0)
   *
   * Returns all words connected by typed edges (MINIMAL_PAIR, RHYME, NEIGHBOR, etc.)
   */
  async getNeighbors(
    word: string,
    relationType?: EdgeType,
    limit: number = 100
  ): Promise<NeighborResult[]> {
    const params = new URLSearchParams({ limit: limit.toString() });
    if (relationType) {
      params.append('relation_type', relationType);
    }
    return this.request(`/api/graph/neighbors/${encodeURIComponent(word)}?${params}`);
  }

  /**
   * Get minimal pairs for specific phoneme contrast (v2.0)
   *
   * Example: Find minimal pairs for /t/ vs /d/ contrast
   */
  async getMinimalPairs(params: {
    phoneme1: string;
    phoneme2: string;
    // Legacy categorical filters (deprecated)
    word_length?: 'short' | 'medium' | 'long';
    complexity?: 'low' | 'medium' | 'high';
    // New numeric range filters
    min_syllables?: number;
    max_syllables?: number;
    min_wcm?: number;
    max_wcm?: number;
    min_frequency?: number;
    max_frequency?: number;
    limit?: number;
  }): Promise<MinimalPairResult[]> {
    const searchParams = new URLSearchParams({
      phoneme1: params.phoneme1,
      phoneme2: params.phoneme2,
    });

    // Legacy filters (backward compatibility)
    if (params.word_length) searchParams.append('word_length', params.word_length);
    if (params.complexity) searchParams.append('complexity', params.complexity);

    // Numeric range filters
    if (params.min_syllables !== undefined) searchParams.append('min_syllables', params.min_syllables.toString());
    if (params.max_syllables !== undefined) searchParams.append('max_syllables', params.max_syllables.toString());
    if (params.min_wcm !== undefined) searchParams.append('min_wcm', params.min_wcm.toString());
    if (params.max_wcm !== undefined) searchParams.append('max_wcm', params.max_wcm.toString());
    if (params.min_frequency !== undefined) searchParams.append('min_frequency', params.min_frequency.toString());
    if (params.max_frequency !== undefined) searchParams.append('max_frequency', params.max_frequency.toString());

    if (params.limit) searchParams.append('limit', params.limit.toString());

    return this.request(`/api/graph/minimal-pairs?${searchParams}`);
  }

  /**
   * Get rhyming words (v2.0)
   *
   * Returns words that rhyme with the target word, with rhyme metadata
   */
  async getRhymes(params: {
    word: string;
    rhyme_mode?: 'last_1' | 'last_2' | 'last_3' | 'assonance' | 'consonance';
    use_embeddings?: boolean;
    word_length?: 'short' | 'medium' | 'long';
    complexity?: 'low' | 'medium' | 'high';
    limit?: number;
  }): Promise<RhymeResult[]> {
    const searchParams = new URLSearchParams();
    if (params.rhyme_mode) searchParams.append('rhyme_mode', params.rhyme_mode);
    if (params.use_embeddings !== undefined) searchParams.append('use_embeddings', params.use_embeddings.toString());
    if (params.word_length) searchParams.append('word_length', params.word_length);
    if (params.complexity) searchParams.append('complexity', params.complexity);
    if (params.limit) searchParams.append('limit', params.limit.toString());

    const query = searchParams.toString() ? `?${searchParams}` : '';
    return this.request(`/api/graph/rhymes/${encodeURIComponent(params.word)}${query}`);
  }

  /**
   * Export full graph data for client caching (v2.0)
   *
   * Returns compressed JSON with:
   * - All words with properties
   * - All typed edges
   * - All phonemes with features
   * - Database statistics
   */
  async exportGraph(includeEmbeddings: boolean = false): Promise<Blob> {
    const params = new URLSearchParams({
      include_embeddings: includeEmbeddings.toString(),
    });

    const response = await fetch(`${this.baseUrl}/api/graph/export?${params}`);
    if (!response.ok) {
      throw new Error(`Export failed: ${response.status}`);
    }

    return response.blob();
  }

  // ==========================================================================
  // Phoneme Operations (Legacy v1.0 - may be deprecated)
  // ==========================================================================

  /**
   * List all phonemes (v1.0)
   */
  async listPhonemes(): Promise<PhonemeListResponse> {
    return this.request('/api/phonemes');
  }

  /**
   * Get phoneme by IPA symbol (v1.0)
   */
  async getPhoneme(ipa: string): Promise<{
    phoneme: string;
    type: 'vowel' | 'consonant';
    features: Record<string, string>;
  }> {
    return this.request(`/api/phonemes/${encodeURIComponent(ipa)}`);
  }

  /**
   * Search phonemes by Phoible features (v1.0)
   */
  async searchPhonemesByFeatures(features: Record<string, string>): Promise<{
    features: Record<string, string>;
    matching_phonemes: string[];
    count: number;
  }> {
    return this.request('/api/phonemes/by-features', {
      method: 'POST',
      body: JSON.stringify(features),
    });
  }

  // ==========================================================================
  // Legacy v1.0 Methods (for backward compatibility with existing components)
  // ==========================================================================

  /**
   * @deprecated Use getMinimalPairs instead
   */
  async generateMinimalPairs(params: {
    phoneme1: string;
    phoneme2: string;
    word_length?: 'short' | 'medium' | 'long';
    complexity?: 'low' | 'medium' | 'high';
    limit?: number;
  }): Promise<MinimalPairResult[]> {
    return this.getMinimalPairs(params);
  }

  /**
   * @deprecated Use getRhymes instead
   */
  async generateRhymeSet(params: {
    target_word: string;
    rhyme_mode?: 'last_1' | 'last_2' | 'last_3' | 'assonance' | 'consonance';
    use_embeddings?: boolean;
    perfect_only?: boolean;  // deprecated, kept for backwards compat
    limit?: number;
  }): Promise<Array<{ word: string; similarity: number }>> {
    // Map to new API with rhyme_mode
    const results = await this.getRhymes({
      word: params.target_word,
      rhyme_mode: params.rhyme_mode || 'last_1',
      use_embeddings: params.use_embeddings,
      limit: params.limit,
    });

    // Convert to legacy format
    return results.map(r => ({
      word: r.rhyme.word,
      similarity: r.quality,
    }));
  }

  /**
   * @deprecated Use filterWords with appropriate filters instead
   */
  async generateComplexityList(params: {
    min_wcm?: number;
    max_wcm?: number;
    min_msh?: number;
    max_msh?: number;
    limit?: number;
  }): Promise<Word[]> {
    // Map min_msh/max_msh to complexity categories for now
    let complexity: 'low' | 'medium' | 'high' | undefined;
    if (params.min_msh !== undefined && params.min_msh <= 2) {
      complexity = 'low';
    } else if (params.min_msh !== undefined && params.min_msh <= 4) {
      complexity = 'medium';
    } else if (params.min_msh !== undefined) {
      complexity = 'high';
    }

    return this.filterWords({
      complexity,
      limit: params.limit || 100,
    });
  }

  /**
   * @deprecated Use patternSearch instead
   */
  async findPhonemePosition(params: {
    phoneme: string;
    position: 'initial' | 'medial' | 'final' | 'any';
    limit?: number;
  }): Promise<Word[]> {
    const request: PatternSearchRequest = {
      limit: params.limit || 100,
    };

    // Map position to pattern
    if (params.position === 'initial') {
      request.starts_with = params.phoneme;
    } else if (params.position === 'final') {
      request.ends_with = params.phoneme;
    } else {
      request.contains = params.phoneme;
    }

    return this.patternSearch(request);
  }

  /**
   * @deprecated Use filterWords with patterns instead
   */
  async buildWordList(request: BuilderRequest): Promise<Word[]> {
    // This is a complex mapping - for now, return filtered results
    // In the future, implement full builder logic
    const filters: WordFilterRequest = {
      min_syllables: request.filters?.min_syllables,
      max_syllables: request.filters?.max_syllables,
      limit: request.limit || 100,
    };

    return this.filterWords(filters);
  }

  /**
   * @deprecated Use getPhoneme for both phonemes and compare manually
   */
  async comparePhonemes(
    ipa1: string,
    ipa2: string
  ): Promise<PhonemeComparison> {
    // Get both phonemes
    const [p1, p2] = await Promise.all([
      this.getPhoneme(ipa1),
      this.getPhoneme(ipa2),
    ]);

    // Convert to PhonemeDetail format
    const phoneme1: PhonemeDetail = {
      phoneme_id: 0,
      ipa: p1.phoneme,
      segment_class: p1.type === 'vowel' ? 'vowel' : 'consonant',
      features: p1.features,
      has_trajectory: false,
    };

    const phoneme2: PhonemeDetail = {
      phoneme_id: 0,
      ipa: p2.phoneme,
      segment_class: p2.type === 'vowel' ? 'vowel' : 'consonant',
      features: p2.features,
      has_trajectory: false,
    };

    // Compare features
    const allFeatures = new Set([...Object.keys(p1.features), ...Object.keys(p2.features)]);
    const differing: string[] = [];
    const shared: string[] = [];

    for (const feature of allFeatures) {
      if (p1.features[feature] === p2.features[feature]) {
        shared.push(feature);
      } else {
        differing.push(feature);
      }
    }

    return {
      phoneme1,
      phoneme2,
      feature_distance: differing.length,
      differing_features: differing,
      shared_features: shared,
    };
  }

  /**
   * @deprecated Use listPhonemes instead
   */
  async searchPhonemes(features: Record<string, string>): Promise<PhonemeDetail[]> {
    const result = await this.searchPhonemesByFeatures(features);

    // Convert to PhonemeDetail format
    return result.matching_phonemes.map(ipa => ({
      phoneme_id: 0,
      ipa,
      segment_class: 'consonant' as const,
      features: {},
      has_trajectory: false,
    }));
  }

  /**
   * Get property ranges from the database
   */
  async getPropertyRanges(): Promise<Record<string, [number, number]>> {
    const response = await fetch(`${this.baseUrl}/api/words/stats/property-ranges`);
    if (!response.ok) {
      throw new Error(`Failed to fetch property ranges: ${response.statusText}`);
    }
    return response.json();
  }
}

// Export singleton instance
export const api = new PhonoLexAPI();
export default api;
