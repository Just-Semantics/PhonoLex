/**
 * Client-Side API Adapter
 *
 * Wraps the client-side data service to match the PhonoLexAPI interface,
 * allowing existing components to work without modification.
 */

import { clientSideData } from './clientSideData';
import { applyExclusions } from '../utils/phonemeUtils';
import type {
  Word,
  WordFilterRequest,
  PatternSearchRequest,
  SimilarityResult,
  SimilaritySearchRequest,
  MinimalPairResult,
  RhymeResult,
  StatsResponse,
  PhonemeDetail,
} from '../types/phonology';

class ClientSideAPIAdapter {
  private dataLoaded = false;

  /**
   * Ensure data is loaded before any operation
   */
  private async ensureLoaded() {
    if (!this.dataLoaded) {
      await clientSideData.loadData();
      this.dataLoaded = true;
    }
  }

  /**
   * Health check (always returns healthy for client-side)
   */
  async healthCheck() {
    return {
      status: 'healthy',
      database: 'client-side',
      pgvector: 'n/a',
      words: 24743,
      edges: 0,
    };
  }

  /**
   * Get word by name
   */
  async getWord(word: string): Promise<Word> {
    await this.ensureLoaded();
    const result = await clientSideData.getWord(word);
    if (!result) {
      throw new Error(`Word not found: ${word}`);
    }
    return result;
  }

  /**
   * Filter words by properties
   */
  async filterWords(request: WordFilterRequest): Promise<Word[]> {
    await this.ensureLoaded();
    return clientSideData.filterWords(request);
  }

  /**
   * Search words by phoneme patterns
   */
  async patternSearch(request: PatternSearchRequest): Promise<Word[]> {
    await this.ensureLoaded();
    return clientSideData.patternSearch(request);
  }

  /**
   * Get database statistics
   */
  async getStats(): Promise<StatsResponse> {
    await this.ensureLoaded();
    return clientSideData.getStats();
  }

  /**
   * Find similar words
   */
  async findSimilarWords(
    word: string,
    options?: { threshold?: number; limit?: number }
  ): Promise<SimilarityResult[]> {
    await this.ensureLoaded();
    return clientSideData.findSimilarWords(
      word,
      options?.threshold || 0.85,
      options?.limit || 50
    );
  }

  /**
   * Similarity search (alias for findSimilarWords)
   */
  async similaritySearch(request: SimilaritySearchRequest): Promise<SimilarityResult[]> {
    return this.findSimilarWords(request.word, {
      threshold: request.threshold,
      limit: request.limit,
    });
  }

  /**
   * Get neighbors (not implemented - returns empty for now)
   */
  async getNeighbors(): Promise<never[]> {
    await this.ensureLoaded();
    console.warn('getNeighbors not implemented in client-side mode');
    return [];
  }

  /**
   * Get minimal pairs
   */
  async getMinimalPairs(params: {
    phoneme1: string;
    phoneme2: string;
    word_length?: string;
    complexity?: string;
    limit?: number;
  }): Promise<MinimalPairResult[]> {
    await this.ensureLoaded();
    return clientSideData.findMinimalPairs(
      params.phoneme1,
      params.phoneme2,
      params.limit || 50
    );
  }

  /**
   * Get rhymes
   */
  async getRhymes(params: {
    word?: string;
    target_word?: string;
    rhyme_mode?: 'last_1' | 'last_2' | 'last_3' | 'assonance' | 'consonance';
    perfect_only?: boolean;
    use_embeddings?: boolean;
    word_length?: string;
    complexity?: string;
    limit?: number;
  }): Promise<RhymeResult[]> {
    await this.ensureLoaded();
    const word = params.target_word || params.word;
    if (!word) {
      return [];
    }
    // perfect_only maps to use_embeddings (if perfect_only is true, don't use embeddings)
    const useEmbeddings = params.perfect_only === true ? false : (params.use_embeddings !== undefined ? params.use_embeddings : true);
    return clientSideData.findRhymes(
      word,
      params.rhyme_mode || 'last_1',
      params.limit || 50,
      useEmbeddings
    );
  }

  /**
   * Export graph (not applicable in client-side mode)
   */
  async exportGraph(): Promise<Blob> {
    throw new Error('Graph export not available in client-side mode');
  }

  /**
   * List all phonemes
   */
  async listPhonemes(): Promise<{ phonemes: Array<{ ipa: string; type: string; features: Record<string, string> }> }> {
    await this.ensureLoaded();
    return clientSideData.listPhonemes();
  }

  /**
   * Get phoneme details
   */
  async getPhoneme(ipa: string): Promise<{
    phoneme: string;
    type: 'vowel' | 'consonant';
    features: Record<string, string>;
  } | null> {
    await this.ensureLoaded();
    return clientSideData.getPhoneme(ipa);
  }

  /**
   * Search phonemes by features
   */
  async searchPhonemesByFeatures(features: Record<string, string>): Promise<{
    features: Record<string, string>;
    matching_phonemes: string[];
    count: number;
  }> {
    await this.ensureLoaded();
    return clientSideData.searchPhonemesByFeatures(features);
  }

  /**
   * Generate minimal pairs (alias for getMinimalPairs)
   */
  async generateMinimalPairs(params: {
    phoneme1: string;
    phoneme2: string;
    word_length?: string;
    complexity?: string;
    limit?: number;
  }): Promise<MinimalPairResult[]> {
    return this.getMinimalPairs(params);
  }

  /**
   * Generate rhyme set (alias for getRhymes)
   */
  async generateRhymeSet(params: {
    word?: string;
    target_word?: string;
    rhyme_mode?: 'last_1' | 'last_2' | 'last_3' | 'assonance' | 'consonance';
    use_embeddings?: boolean;
    word_length?: string;
    complexity?: string;
    limit?: number;
  }): Promise<RhymeResult[]> {
    return this.getRhymes(params);
  }

  /**
   * Generate complexity list
   */
  async generateComplexityList(params: {
    min_wcm?: number;
    max_wcm?: number;
    min_msh?: number;
    max_msh?: number;
    limit?: number;
  }): Promise<Word[]> {
    await this.ensureLoaded();
    return this.filterWords({
      ...params,
      limit: params.limit || 50,
    });
  }

  /**
   * Find phoneme position
   */
  async findPhonemePosition(params: {
    phoneme: string;
    position: 'initial' | 'medial' | 'final' | 'any';
    limit?: number;
  }): Promise<Word[]> {
    await this.ensureLoaded();
    return this.patternSearch(params);
  }

  /**
   * Build word list
   */
  async buildWordList(request: {
    patterns?: Array<{
      type: 'STARTS_WITH' | 'ENDS_WITH' | 'CONTAINS' | 'CONTAINS_MEDIAL';
      phoneme: string;
      medial_only?: boolean;
    }>;
    filters?: WordFilterRequest;
    exclusions?: {
      exclude_phonemes?: string[];
    };
    limit?: number;
  }): Promise<Word[]> {
    await this.ensureLoaded();
    // Convert builder request to pattern search
    const patterns = request.patterns || [];
    const filters = request.filters || {};
    const exclusions = request.exclusions || {};
    let results: Word[] = [];

    if (patterns.length > 0) {
      // Build search request with ALL patterns (AND logic)
      const searchReq: PatternSearchRequest = {
        limit: 5000, // Get more results for filtering
        filters: filters,
      };

      // Process all patterns and combine them
      for (const pattern of patterns) {
        if (pattern.type === 'STARTS_WITH') {
          searchReq.starts_with = pattern.phoneme;
        } else if (pattern.type === 'ENDS_WITH') {
          searchReq.ends_with = pattern.phoneme;
        } else if (pattern.type === 'CONTAINS') {
          searchReq.contains = pattern.phoneme;
          searchReq.contains_medial_only = pattern.medial_only;
        } else if (pattern.type === 'CONTAINS_MEDIAL') {
          searchReq.contains = pattern.phoneme;
          searchReq.contains_medial_only = true;
        }
      }

      results = await this.patternSearch(searchReq);
    } else {
      // No patterns - just filter
      results = await this.filterWords({
        ...filters,
        limit: 5000,
      });
    }

    // Apply exclusions (phonemes to exclude)
    results = applyExclusions(results, exclusions.exclude_phonemes);

    // Limit final results
    return results.slice(0, request.limit || 200);
  }

  /**
   * Compare phonemes
   */
  async comparePhonemes(phoneme1: string, phoneme2: string): Promise<{
    phoneme1: {
      phoneme_id: number;
      ipa: string;
      segment_class: 'vowel' | 'consonant';
      features: Record<string, string>;
      has_trajectory: boolean;
    };
    phoneme2: {
      phoneme_id: number;
      ipa: string;
      segment_class: 'vowel' | 'consonant';
      features: Record<string, string>;
      has_trajectory: boolean;
    };
    similarity_score: number;
    different_features: Record<string, [string, string]>;
    shared_features: Record<string, string>;
  }> {
    await this.ensureLoaded();
    return clientSideData.comparePhonemes(phoneme1, phoneme2);
  }

  /**
   * Search phonemes
   */
  async searchPhonemes(): Promise<PhonemeDetail[]> {
    await this.ensureLoaded();
    console.warn('searchPhonemes not implemented in client-side mode');
    return [];
  }

  /**
   * Get property ranges
   */
  async getPropertyRanges(): Promise<Record<string, [number, number]>> {
    await this.ensureLoaded();
    // Get computed ranges from clientSideData
    return clientSideData.getPropertyRanges();
  }
}

// Export singleton instance
export const api = new ClientSideAPIAdapter();
export default api;
