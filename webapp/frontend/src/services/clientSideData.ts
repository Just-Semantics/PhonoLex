/**
 * Client-Side Data Service for PhonoLex
 *
 * Loads all data from static JSON files and implements phonological operations
 * entirely in the browser - no backend required!
 *
 * Data loaded:
 * - word_metadata.json (~8 MB) - All word properties
 * - embeddings.json (~138 MB → 1.5 MB gzipped) - Phoible syllable embeddings
 * - arpa_to_ipa.json (1.2 KB) - ARPAbet mapping reference
 *
 * Embeddings are pure Phoible features - no training or quantization needed!
 * Structure: onset(76) + nucleus(76) + coda(76) = 228 dims per syllable
 */

import type {
  Word,
  WordFilterRequest,
  PatternSearchRequest,
  SimilarityResult,
  MinimalPairResult,
  RhymeResult,
  StatsResponse,
} from '../types/phonology';
import { tokenizePhonemes as tokenize, containsSequence } from '../utils/phonemeUtils';

// ============================================================================
// Types
// ============================================================================

interface WordMetadata {
  word: string;
  ipa: string;
  arpa?: string;  // Optional - not present in Phase 3 exports
  phonemes: string[];
  phonemes_arpa?: string[];  // Optional - not present in Phase 3 exports
  syllables: Array<{
    onset: string[];
    nucleus: string;
    coda: string[];
    stress?: number;
  }>;
  phoneme_count: number;
  syllable_count: number;
  wcm_score: number | null;
  msh_stage: number | null;
  frequency: number | null;
  log_frequency: number | null;
  concreteness: number | null;
  aoa: number | null;
  imageability: number | null;
  familiarity: number | null;
  valence: number | null;
  arousal: number | null;
  dominance: number | null;
}

interface EmbeddingsData {
  embeddings: Record<string, number[][]>; // word -> syllables -> float32 values
  embedding_dim: number;
  syllable_structure: string;
  source: string;
  normalization: string;
}

interface PhonemeData {
  ipa: string;
  type: 'vowel' | 'consonant';
  features: Record<string, string>; // Phoible features (+, -, 0)
}

interface PhonemesFile {
  phonemes: PhonemeData[];
  count: number;
}

// ============================================================================
// Client-Side Data Loader
// ============================================================================

class ClientSideDataService {
  private wordMetadata: Map<string, WordMetadata> = new Map();
  private embeddings: EmbeddingsData | null = null;
  private phonemes: Map<string, PhonemeData> = new Map(); // IPA -> phoneme data
  private loaded: boolean = false;
  private loading: Promise<void> | null = null;
  private cachedRanges: Record<string, [number, number]> | null = null;

  /**
   * Load all data files from public/data/
   */
  async loadData(): Promise<void> {
    // Return existing promise if already loading
    if (this.loading) {
      return this.loading;
    }

    // Return immediately if already loaded
    if (this.loaded) {
      return;
    }

    // Start loading
    this.loading = this._loadDataInternal();
    await this.loading;
    this.loaded = true;
  }

  private async _loadDataInternal(): Promise<void> {
    console.log('[ClientSideData] Loading data files...');
    const startTime = performance.now();

    try {
      // Load all data in parallel (gzipped files)
      const [metadataRes, embeddingsRes, arpaRes, phonemesRes] = await Promise.all([
        fetch('/data/word_metadata.json.gz'),
        fetch('/data/embeddings.json.gz'),
        fetch('/data/arpa_to_ipa.json.gz'),
        fetch('/data/phonemes.json.gz'),
      ]);

      // Check each response individually for better error messages
      if (!metadataRes.ok) {
        throw new Error(`Failed to load word_metadata.json.gz: ${metadataRes.status} ${metadataRes.statusText}`);
      }
      if (!embeddingsRes.ok) {
        throw new Error(`Failed to load embeddings.json.gz: ${embeddingsRes.status} ${embeddingsRes.statusText}`);
      }
      if (!arpaRes.ok) {
        throw new Error(`Failed to load arpa_to_ipa.json.gz: ${arpaRes.status} ${arpaRes.statusText}`);
      }
      if (!phonemesRes.ok) {
        throw new Error(`Failed to load phonemes.json.gz: ${phonemesRes.status} ${phonemesRes.statusText}`);
      }

      // Parse JSON with individual error handling
      let metadataJson: Record<string, WordMetadata>;
      let embeddingsJson: EmbeddingsData;
      let _arpaJson: Record<string, string>;
      let phonemesJson: { phonemes: PhonemeData[] };

      try {
        metadataJson = await metadataRes.json();
      } catch (error) {
        throw new Error(`Failed to parse word_metadata.json.gz: ${error instanceof Error ? error.message : String(error)}`);
      }

      try {
        embeddingsJson = await embeddingsRes.json();
      } catch (error) {
        throw new Error(`Failed to parse embeddings.json.gz: ${error instanceof Error ? error.message : String(error)}`);
      }

      try {
        _arpaJson = await arpaRes.json();
        void _arpaJson; // Loaded but not used (available in other data structures if needed)
      } catch (error) {
        throw new Error(`Failed to parse arpa_to_ipa.json.gz: ${error instanceof Error ? error.message : String(error)}`);
      }

      try {
        phonemesJson = await phonemesRes.json();
      } catch (error) {
        throw new Error(`Failed to parse phonemes.json.gz: ${error instanceof Error ? error.message : String(error)}`);
      }

      // Store metadata in Map for fast lookup
      Object.entries(metadataJson).forEach(
        ([word, data]) => {
          this.wordMetadata.set(word, data);
        }
      );

      // Store phonemes in Map for fast lookup
      const phonemesFile = phonemesJson as PhonemesFile;
      phonemesFile.phonemes.forEach((phoneme) => {
        this.phonemes.set(phoneme.ipa, phoneme);
      });

      this.embeddings = embeddingsJson as EmbeddingsData;
      // ARPAbet mapping loaded but not stored (available in embeddingsJson if needed)

      // Compute property ranges immediately after loading
      this.cachedRanges = this._computePropertyRanges();

      const loadTime = ((performance.now() - startTime) / 1000).toFixed(2);
      console.log(
        `[ClientSideData] ✓ Loaded ${this.wordMetadata.size} words in ${loadTime}s`
      );
    } catch (error) {
      console.error('[ClientSideData] Failed to load data:', error);
      throw error;
    }
  }

  /**
   * Ensure data is loaded before operation
   */
  private async ensureLoaded(): Promise<void> {
    if (!this.loaded) {
      await this.loadData();
    }
  }

  // ==========================================================================
  // Word Queries
  // ==========================================================================

  /**
   * Get word by string
   */
  async getWord(word: string): Promise<Word | null> {
    await this.ensureLoaded();

    const metadata = this.wordMetadata.get(word.toLowerCase());
    if (!metadata) {
      return null;
    }

    return this.metadataToWord(metadata);
  }

  /**
   * Filter words by properties
   */
  async filterWords(request: WordFilterRequest): Promise<Word[]> {
    await this.ensureLoaded();

    const results: Word[] = [];

    for (const metadata of this.wordMetadata.values()) {
      let matches = true;

      // Apply filters (AND logic - all must match)
      // Syllable count
      if (request.min_syllables !== undefined && metadata.syllable_count < request.min_syllables) matches = false;
      if (request.max_syllables !== undefined && metadata.syllable_count > request.max_syllables) matches = false;

      // Phoneme count
      if (request.min_phonemes !== undefined && metadata.phoneme_count < request.min_phonemes) matches = false;
      if (request.max_phonemes !== undefined && metadata.phoneme_count > request.max_phonemes) matches = false;

      // WCM
      if (request.min_wcm !== undefined && (metadata.wcm_score === null || metadata.wcm_score < request.min_wcm)) matches = false;
      if (request.max_wcm !== undefined && (metadata.wcm_score === null || metadata.wcm_score > request.max_wcm)) matches = false;

      // MSH
      if (request.min_msh !== undefined && (metadata.msh_stage === null || metadata.msh_stage < request.min_msh)) matches = false;
      if (request.max_msh !== undefined && (metadata.msh_stage === null || metadata.msh_stage > request.max_msh)) matches = false;

      // Frequency
      if (request.min_frequency !== undefined && (metadata.frequency === null || metadata.frequency < request.min_frequency)) matches = false;
      if (request.max_frequency !== undefined && (metadata.frequency === null || metadata.frequency > request.max_frequency)) matches = false;

      // Age of Acquisition
      if (request.min_aoa !== undefined && (metadata.aoa === null || metadata.aoa < request.min_aoa)) matches = false;
      if (request.max_aoa !== undefined && (metadata.aoa === null || metadata.aoa > request.max_aoa)) matches = false;

      // Imageability
      if (request.min_imageability !== undefined && (metadata.imageability === null || metadata.imageability < request.min_imageability)) matches = false;
      if (request.max_imageability !== undefined && (metadata.imageability === null || metadata.imageability > request.max_imageability)) matches = false;

      // Familiarity
      if (request.min_familiarity !== undefined && (metadata.familiarity === null || metadata.familiarity < request.min_familiarity)) matches = false;
      if (request.max_familiarity !== undefined && (metadata.familiarity === null || metadata.familiarity > request.max_familiarity)) matches = false;

      // Concreteness
      if (request.min_concreteness !== undefined && (metadata.concreteness === null || metadata.concreteness < request.min_concreteness)) matches = false;
      if (request.max_concreteness !== undefined && (metadata.concreteness === null || metadata.concreteness > request.max_concreteness)) matches = false;

      // Valence
      if (request.min_valence !== undefined && (metadata.valence === null || metadata.valence < request.min_valence)) matches = false;
      if (request.max_valence !== undefined && (metadata.valence === null || metadata.valence > request.max_valence)) matches = false;

      // Arousal
      if (request.min_arousal !== undefined && (metadata.arousal === null || metadata.arousal < request.min_arousal)) matches = false;
      if (request.max_arousal !== undefined && (metadata.arousal === null || metadata.arousal > request.max_arousal)) matches = false;

      // Dominance
      if (request.min_dominance !== undefined && (metadata.dominance === null || metadata.dominance < request.min_dominance)) matches = false;
      if (request.max_dominance !== undefined && (metadata.dominance === null || metadata.dominance > request.max_dominance)) matches = false;

      if (matches) {
        results.push(this.metadataToWord(metadata));

        // Limit results
        if (results.length >= (request.limit || 500)) {
          break;
        }
      }
    }

    return results;
  }

  /**
   * Tokenize a phoneme string into individual phonemes
   * REQUIRES space-separated input (e.g., "k æ t" or "dʒ ʌ dʒ")
   */
  private tokenizePhonemes(input: string): string[] {
    return tokenize(input);
  }

  /**
   * Pattern search (starts with, ends with, contains)
   */
  async patternSearch(request: PatternSearchRequest): Promise<Word[]> {
    await this.ensureLoaded();

    const results: Word[] = [];

    for (const metadata of this.wordMetadata.values()) {
      let matches = true;

      // Check starts_with (support both space-separated and concatenated)
      if (request.starts_with) {
        const targetPhonemes = this.tokenizePhonemes(request.starts_with);
        const startPhonemes = metadata.phonemes.slice(0, targetPhonemes.length);

        if (JSON.stringify(startPhonemes) !== JSON.stringify(targetPhonemes)) {
          matches = false;
        }
      }

      // Check ends_with (support both space-separated and concatenated)
      if (request.ends_with) {
        const targetPhonemes = this.tokenizePhonemes(request.ends_with);
        const endPhonemes = metadata.phonemes.slice(-targetPhonemes.length);
        if (JSON.stringify(endPhonemes) !== JSON.stringify(targetPhonemes)) {
          matches = false;
        }
      }

      // Check contains (support both space-separated and concatenated)
      if (request.contains) {
        const targetPhonemes = this.tokenizePhonemes(request.contains);

        if (request.contains_medial_only) {
          // Exclude first and last positions
          const medialPhonemes = metadata.phonemes.slice(1, -1);
          if (!containsSequence(medialPhonemes, targetPhonemes)) {
            matches = false;
          }
        } else {
          if (!containsSequence(metadata.phonemes, targetPhonemes)) {
            matches = false;
          }
        }
      }

      if (matches) {
        // Apply additional filters (AND logic - all must match)
        if (request.filters) {
          const f = request.filters;

          // Syllable count
          if (f.min_syllables !== undefined && metadata.syllable_count < f.min_syllables) matches = false;
          if (f.max_syllables !== undefined && metadata.syllable_count > f.max_syllables) matches = false;

          // Phoneme count
          if (f.min_phonemes !== undefined && metadata.phoneme_count < f.min_phonemes) matches = false;
          if (f.max_phonemes !== undefined && metadata.phoneme_count > f.max_phonemes) matches = false;

          // WCM
          if (f.min_wcm !== undefined && (metadata.wcm_score === null || metadata.wcm_score < f.min_wcm)) matches = false;
          if (f.max_wcm !== undefined && (metadata.wcm_score === null || metadata.wcm_score > f.max_wcm)) matches = false;

          // MSH
          if (f.min_msh !== undefined && (metadata.msh_stage === null || metadata.msh_stage < f.min_msh)) matches = false;
          if (f.max_msh !== undefined && (metadata.msh_stage === null || metadata.msh_stage > f.max_msh)) matches = false;

          // Frequency
          if (f.min_frequency !== undefined && (metadata.frequency === null || metadata.frequency < f.min_frequency)) matches = false;
          if (f.max_frequency !== undefined && (metadata.frequency === null || metadata.frequency > f.max_frequency)) matches = false;

          // Age of Acquisition
          if (f.min_aoa !== undefined && (metadata.aoa === null || metadata.aoa < f.min_aoa)) matches = false;
          if (f.max_aoa !== undefined && (metadata.aoa === null || metadata.aoa > f.max_aoa)) matches = false;

          // Imageability
          if (f.min_imageability !== undefined && (metadata.imageability === null || metadata.imageability < f.min_imageability)) matches = false;
          if (f.max_imageability !== undefined && (metadata.imageability === null || metadata.imageability > f.max_imageability)) matches = false;

          // Familiarity
          if (f.min_familiarity !== undefined && (metadata.familiarity === null || metadata.familiarity < f.min_familiarity)) matches = false;
          if (f.max_familiarity !== undefined && (metadata.familiarity === null || metadata.familiarity > f.max_familiarity)) matches = false;

          // Concreteness
          if (f.min_concreteness !== undefined && (metadata.concreteness === null || metadata.concreteness < f.min_concreteness)) matches = false;
          if (f.max_concreteness !== undefined && (metadata.concreteness === null || metadata.concreteness > f.max_concreteness)) matches = false;

          // Valence
          if (f.min_valence !== undefined && (metadata.valence === null || metadata.valence < f.min_valence)) matches = false;
          if (f.max_valence !== undefined && (metadata.valence === null || metadata.valence > f.max_valence)) matches = false;

          // Arousal
          if (f.min_arousal !== undefined && (metadata.arousal === null || metadata.arousal < f.min_arousal)) matches = false;
          if (f.max_arousal !== undefined && (metadata.arousal === null || metadata.arousal > f.max_arousal)) matches = false;

          // Dominance
          if (f.min_dominance !== undefined && (metadata.dominance === null || metadata.dominance < f.min_dominance)) matches = false;
          if (f.max_dominance !== undefined && (metadata.dominance === null || metadata.dominance > f.max_dominance)) matches = false;
        }

        if (matches) {
          results.push(this.metadataToWord(metadata));

          if (results.length >= (request.limit || 500)) {
            break;
          }
        }
      }
    }

    return results;
  }

  // ==========================================================================
  // Phoneme Queries
  // ==========================================================================

  /**
   * Get phoneme details by IPA symbol
   */
  async getPhoneme(ipa: string): Promise<{
    phoneme: string;
    type: 'vowel' | 'consonant';
    features: Record<string, string>;
  } | null> {
    await this.ensureLoaded();
    const phoneme = this.phonemes.get(ipa);
    if (!phoneme) {
      return null;
    }
    return {
      phoneme: phoneme.ipa,
      type: phoneme.type,
      features: phoneme.features
    };
  }

  /**
   * List all phonemes
   */
  async listPhonemes(): Promise<{ phonemes: Array<{ ipa: string; type: string; features: Record<string, string> }> }> {
    await this.ensureLoaded();
    return {
      phonemes: Array.from(this.phonemes.values()).map(p => ({
        ipa: p.ipa,
        type: p.type,
        features: p.features
      }))
    };
  }

  /**
   * Search phonemes by Phoible features
   */
  async searchPhonemesByFeatures(features: Record<string, string>): Promise<{
    features: Record<string, string>;
    matching_phonemes: string[];
    count: number;
  }> {
    await this.ensureLoaded();

    // Filter phonemes that match ALL specified features
    const matching: string[] = [];
    for (const phoneme of this.phonemes.values()) {
      let matches = true;
      for (const [feature, value] of Object.entries(features)) {
        if (phoneme.features[feature] !== value) {
          matches = false;
          break;
        }
      }
      if (matches) {
        matching.push(phoneme.ipa);
      }
    }

    return {
      features,
      matching_phonemes: matching,
      count: matching.length
    };
  }

  /**
   * Compare two phonemes and compute feature differences
   */
  async comparePhonemes(ipa1: string, ipa2: string): Promise<{
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

    const p1 = this.phonemes.get(ipa1);
    const p2 = this.phonemes.get(ipa2);

    if (!p1 || !p2) {
      throw new Error(`Phoneme not found: ${!p1 ? ipa1 : ipa2}`);
    }

    // Get all unique features
    const allFeatures = new Set([...Object.keys(p1.features), ...Object.keys(p2.features)]);
    const different: Record<string, [string, string]> = {};
    const shared: Record<string, string> = {};

    // Compare features
    for (const feature of allFeatures) {
      const val1 = p1.features[feature] || '0';
      const val2 = p2.features[feature] || '0';

      if (val1 === val2) {
        shared[feature] = val1;
      } else {
        different[feature] = [val1, val2];
      }
    }

    // Calculate similarity score (1 - feature distance / total features)
    const totalFeatures = allFeatures.size;
    const differingCount = Object.keys(different).length;
    const similarity_score = totalFeatures > 0 ? 1 - (differingCount / totalFeatures) : 1;

    return {
      phoneme1: {
        phoneme_id: 0,
        ipa: p1.ipa,
        segment_class: p1.type,
        features: p1.features,
        has_trajectory: false,
      },
      phoneme2: {
        phoneme_id: 0,
        ipa: p2.ipa,
        segment_class: p2.type,
        features: p2.features,
        has_trajectory: false,
      },
      similarity_score: similarity_score,
      different_features: different,
      shared_features: shared,
    };
  }

  // ==========================================================================
  // Similarity Search (using embeddings)
  // ==========================================================================

  /**
   * Find words similar to a given word using syllable embeddings
   */
  async findSimilarWords(
    word: string,
    threshold: number = 0.85,
    limit: number = 50,
    weights?: { onset: number; nucleus: number; coda: number }
  ): Promise<SimilarityResult[]> {
    await this.ensureLoaded();

    const targetMetadata = this.wordMetadata.get(word.toLowerCase());
    if (!targetMetadata || !this.embeddings) {
      return [];
    }

    const targetEmbeddings = this.getWordEmbeddings(word);
    if (!targetEmbeddings) {
      return [];
    }

    // Compute similarity with all other words
    const results: Array<{ word: Word; similarity: number }> = [];

    for (const [candidateWord, candidateMetadata] of this.wordMetadata.entries()) {
      if (candidateWord === word.toLowerCase()) {
        continue; // Skip self
      }

      const candidateEmbeddings = this.getWordEmbeddings(candidateWord);
      if (!candidateEmbeddings) {
        continue;
      }

      const similarity = this.computeSoftLevenshteinSimilarity(
        targetEmbeddings,
        candidateEmbeddings,
        weights // Pass custom weights if provided
      );

      // Debug logging for first few results
      if (results.length < 3) {
        console.log(`Similarity for ${candidateWord}:`, similarity, 'with weights:', weights);
      }

      if (similarity >= threshold) {
        results.push({
          word: this.metadataToWord(candidateMetadata),
          similarity,
        });
      }
    }

    // Sort by similarity (highest first) and limit
    results.sort((a, b) => b.similarity - a.similarity);
    return results.slice(0, limit);
  }

  // ==========================================================================
  // Minimal Pairs
  // ==========================================================================

  /**
   * Find minimal pairs for phoneme contrast
   */
  async findMinimalPairs(
    phoneme1: string,
    phoneme2: string,
    limit: number = 50
  ): Promise<MinimalPairResult[]> {
    await this.ensureLoaded();
    console.log('[findMinimalPairs] phoneme1:', phoneme1, 'hex:', phoneme1.split('').map(c => '0x' + c.charCodeAt(0).toString(16)).join(' '));
    console.log('[findMinimalPairs] phoneme2:', phoneme2, 'hex:', phoneme2.split('').map(c => '0x' + c.charCodeAt(0).toString(16)).join(' '));

    const results: MinimalPairResult[] = [];

    // Group words by length for efficiency
    const wordsByLength = new Map<number, string[]>();
    for (const word of this.wordMetadata.keys()) {
      const metadata = this.wordMetadata.get(word)!;
      const length = metadata.phoneme_count;
      if (!wordsByLength.has(length)) {
        wordsByLength.set(length, []);
      }
      wordsByLength.get(length)!.push(word);
    }

    // Find minimal pairs within each length group
    for (const words of wordsByLength.values()) {
      for (let i = 0; i < words.length; i++) {
        for (let j = i + 1; j < words.length; j++) {
          const word1 = words[i];
          const word2 = words[j];

          const metadata1 = this.wordMetadata.get(word1)!;
          const metadata2 = this.wordMetadata.get(word2)!;

          // Count differences
          let diffCount = 0;
          let diffPosition = -1;
          let diffPhoneme1 = '';
          let diffPhoneme2 = '';

          for (let k = 0; k < metadata1.phonemes.length; k++) {
            if (metadata1.phonemes[k] !== metadata2.phonemes[k]) {
              diffCount++;
              diffPosition = k;
              diffPhoneme1 = metadata1.phonemes[k];
              diffPhoneme2 = metadata2.phonemes[k];
            }
          }

          // Check if minimal pair with requested contrast
          if (diffCount === 1) {
            // Log potential matches for debugging
            if ((diffPhoneme1 === phoneme1 || diffPhoneme1 === phoneme2 || diffPhoneme2 === phoneme1 || diffPhoneme2 === phoneme2)) {
              console.log('[findMinimalPairs] Potential match:', word1, 'vs', word2, 'diff:', diffPhoneme1, '<->', diffPhoneme2);
              console.log('  diffPhoneme1 hex:', diffPhoneme1.split('').map(c => '0x' + c.charCodeAt(0).toString(16)).join(' '));
              console.log('  diffPhoneme2 hex:', diffPhoneme2.split('').map(c => '0x' + c.charCodeAt(0).toString(16)).join(' '));
              console.log('  Match1:', diffPhoneme1 === phoneme1, diffPhoneme2 === phoneme2);
              console.log('  Match2:', diffPhoneme1 === phoneme2, diffPhoneme2 === phoneme1);
            }
          }

          if (
            diffCount === 1 &&
            ((diffPhoneme1 === phoneme1 && diffPhoneme2 === phoneme2) ||
              (diffPhoneme1 === phoneme2 && diffPhoneme2 === phoneme1))
          ) {
            results.push({
              word1: this.metadataToWord(metadata1),
              word2: this.metadataToWord(metadata2),
              position: diffPosition,
              phoneme1: diffPhoneme1,
              phoneme2: diffPhoneme2,
              metadata: {
                position: diffPosition,
                phoneme1: diffPhoneme1,
                phoneme2: diffPhoneme2,
              },
            });

            if (results.length >= limit) {
              return results;
            }
          }
        }
      }
    }

    console.log('[findMinimalPairs] Found', results.length, 'minimal pairs');
    return results;
  }

  // ==========================================================================
  // Rhyme Detection
  // ==========================================================================

  /**
   * Find rhymes for a word
   */
  async findRhymes(
    word: string,
    rhymeMode: 'last_1' | 'last_2' | 'last_3' | 'assonance' | 'consonance' = 'last_1',
    limit: number = 50,
    useEmbeddings: boolean = false
  ): Promise<RhymeResult[]> {
    await this.ensureLoaded();

    const targetMetadata = this.wordMetadata.get(word.toLowerCase());
    if (!targetMetadata) {
      return [];
    }

    // Get exact phoneme matches
    const exactMatches = this._findRhymesByPhonemes(targetMetadata, rhymeMode, limit);

    if (!useEmbeddings) {
      return exactMatches;
    }

    // Get embedding-based near-matches
    const nearMatches = this._findRhymesByEmbeddings(targetMetadata, rhymeMode, limit * 2, exactMatches);

    // Combine and sort by quality
    const combined = [...exactMatches, ...nearMatches];
    combined.sort((a, b) => (b.metadata?.quality || 0) - (a.metadata?.quality || 0));

    return combined.slice(0, limit);
  }

  /**
   * Find rhymes using exact phoneme matching (quality=1.0)
   */
  private _findRhymesByPhonemes(
    targetMetadata: WordMetadata,
    rhymeMode: 'last_1' | 'last_2' | 'last_3' | 'assonance' | 'consonance',
    limit: number
  ): RhymeResult[] {
    const targetSyllables = targetMetadata.syllables;
    const results: RhymeResult[] = [];

    for (const [candidateWord, candidateMetadata] of this.wordMetadata.entries()) {
      if (candidateWord === targetMetadata.word.toLowerCase()) {
        continue; // Skip self
      }

      const candidateSyllables = candidateMetadata.syllables;
      let matches = false;
      let quality = 0;

      // Check rhyme based on mode
      if (rhymeMode === 'last_1') {
        // Match last syllable (nucleus + coda)
        const targetLast = targetSyllables[targetSyllables.length - 1];
        const candidateLast = candidateSyllables[candidateSyllables.length - 1];

        if (
          targetLast.nucleus === candidateLast.nucleus &&
          JSON.stringify(targetLast.coda) === JSON.stringify(candidateLast.coda)
        ) {
          matches = true;
          quality = 1.0; // Perfect rhyme
        }
      } else if (rhymeMode === 'last_2') {
        // Match last 2 syllables
        if (targetSyllables.length >= 2 && candidateSyllables.length >= 2) {
          const targetLast2 = targetSyllables.slice(-2);
          const candidateLast2 = candidateSyllables.slice(-2);

          if (JSON.stringify(targetLast2) === JSON.stringify(candidateLast2)) {
            matches = true;
            quality = 1.0;
          }
        }
      } else if (rhymeMode === 'last_3') {
        // Match last 3 syllables
        if (targetSyllables.length >= 3 && candidateSyllables.length >= 3) {
          const targetLast3 = targetSyllables.slice(-3);
          const candidateLast3 = candidateSyllables.slice(-3);

          if (JSON.stringify(targetLast3) === JSON.stringify(candidateLast3)) {
            matches = true;
            quality = 1.0;
          }
        }
      } else if (rhymeMode === 'assonance') {
        // Match nucleus (vowel) only from last syllable
        const targetLast = targetSyllables[targetSyllables.length - 1];
        const candidateLast = candidateSyllables[candidateSyllables.length - 1];

        if (targetLast.nucleus === candidateLast.nucleus) {
          matches = true;
          quality = 1.0;
        }
      } else if (rhymeMode === 'consonance') {
        // Match coda (final consonants) only from last syllable
        const targetLast = targetSyllables[targetSyllables.length - 1];
        const candidateLast = candidateSyllables[candidateSyllables.length - 1];

        if (JSON.stringify(targetLast.coda) === JSON.stringify(candidateLast.coda)) {
          matches = true;
          quality = 1.0;
        }
      }

      if (matches) {
        results.push({
          word: this.metadataToWord(candidateMetadata),
          metadata: {
            rhyme_type: rhymeMode,
            quality,
            nucleus: targetSyllables[targetSyllables.length - 1].nucleus,
            coda: targetSyllables[targetSyllables.length - 1].coda,
          },
        });

        if (results.length >= limit) {
          break;
        }
      }
    }

    return results;
  }

  /**
   * Find rhymes using syllable embedding similarity (near-rhymes with quality<1.0)
   */
  private _findRhymesByEmbeddings(
    targetMetadata: WordMetadata,
    rhymeMode: 'last_1' | 'last_2' | 'last_3' | 'assonance' | 'consonance',
    limit: number,
    exactMatches: RhymeResult[]
  ): RhymeResult[] {
    if (!this.embeddings || !this.embeddings[targetMetadata.word]) {
      return [];
    }

    const targetEmbeddings = this.embeddings[targetMetadata.word];
    const targetSyllables = targetMetadata.syllables;
    const exactWordIds = new Set(exactMatches.map(r => r.word?.word));
    const results: Array<{ word: Word; similarity: number }> = [];

    const threshold = 0.7; // Min similarity for near-rhymes

    // Calculate hierarchical soft Levenshtein similarity for all candidates
    for (const [candidateWord, candidateMetadata] of this.wordMetadata.entries()) {
      if (candidateWord === targetMetadata.word.toLowerCase() || exactWordIds.has(candidateWord)) {
        continue; // Skip self and exact matches
      }

      const candidateEmbeddings = this.embeddings[candidateWord];
      if (!candidateEmbeddings) {
        continue;
      }

      // Calculate hierarchical soft Levenshtein similarity
      const similarity = this.computeSoftLevenshteinSimilarity(targetEmbeddings, candidateEmbeddings);
      if (similarity < threshold) {
        continue;
      }

      // Apply rhyme mode constraint (relaxed - match nuclei, allow coda variation)
      const candidateSyllables = candidateMetadata.syllables;
      let matchesConstraint = false;

      if (rhymeMode === 'last_1') {
        // Match nucleus of last syllable
        const targetLast = targetSyllables[targetSyllables.length - 1];
        const candidateLast = candidateSyllables[candidateSyllables.length - 1];
        matchesConstraint = targetLast.nucleus === candidateLast.nucleus;
      } else if (rhymeMode === 'last_2') {
        // Match nuclei of last 2 syllables
        if (targetSyllables.length >= 2 && candidateSyllables.length >= 2) {
          const targetNuclei = targetSyllables.slice(-2).map(s => s.nucleus);
          const candidateNuclei = candidateSyllables.slice(-2).map(s => s.nucleus);
          matchesConstraint = JSON.stringify(targetNuclei) === JSON.stringify(candidateNuclei);
        }
      } else if (rhymeMode === 'last_3') {
        // Match nuclei of last 3 syllables
        if (targetSyllables.length >= 3 && candidateSyllables.length >= 3) {
          const targetNuclei = targetSyllables.slice(-3).map(s => s.nucleus);
          const candidateNuclei = candidateSyllables.slice(-3).map(s => s.nucleus);
          matchesConstraint = JSON.stringify(targetNuclei) === JSON.stringify(candidateNuclei);
        }
      } else if (rhymeMode === 'assonance') {
        // Match nucleus of last syllable
        const targetLast = targetSyllables[targetSyllables.length - 1];
        const candidateLast = candidateSyllables[candidateSyllables.length - 1];
        matchesConstraint = targetLast.nucleus === candidateLast.nucleus;
      } else if (rhymeMode === 'consonance') {
        // Match coda of last syllable (exact for consonance)
        const targetLast = targetSyllables[targetSyllables.length - 1];
        const candidateLast = candidateSyllables[candidateSyllables.length - 1];
        matchesConstraint = JSON.stringify(targetLast.coda) === JSON.stringify(candidateLast.coda);
      }

      if (matchesConstraint) {
        results.push({
          word: this.metadataToWord(candidateMetadata),
          similarity
        });
      }
    }

    // Sort by similarity and take top results
    results.sort((a, b) => b.similarity - a.similarity);
    const topResults = results.slice(0, limit);

    // Convert to RhymeResult format
    return topResults.map(r => ({
      word: r.word,
      metadata: {
        rhyme_type: `${rhymeMode}_near`,
        quality: r.similarity,
        nucleus: targetSyllables[targetSyllables.length - 1]?.nucleus || '',
        coda: targetSyllables[targetSyllables.length - 1]?.coda || [],
      }
    }));
  }


  // ==========================================================================
  // Maximal Opposition (Gierut 1989-1992, Storkel 2022)
  // ==========================================================================

  /**
   * Check if two phonemes differ by major class (sonorant vs. obstruent)
   * This is the critical criterion for maximal opposition intervention
   */
  private hasMajorClassDifference(ipa1: string, ipa2: string): boolean {
    const p1 = this.phonemes.get(ipa1);
    const p2 = this.phonemes.get(ipa2);

    if (!p1 || !p2) return false;

    // Both must be consonants
    if (p1.type !== 'consonant' || p2.type !== 'consonant') {
      return false;
    }

    // One must be sonorant, the other obstruent
    const son1 = p1.features.sonorant === '+';
    const son2 = p2.features.sonorant === '+';

    return son1 !== son2;
  }

  /**
   * Count how many distinctive features differ between two phonemes
   */
  private countFeatureDifferences(ipa1: string, ipa2: string): number {
    const p1 = this.phonemes.get(ipa1);
    const p2 = this.phonemes.get(ipa2);

    if (!p1 || !p2) return 0;

    let diffCount = 0;
    const allFeatures = new Set([...Object.keys(p1.features), ...Object.keys(p2.features)]);

    for (const feature of allFeatures) {
      const val1 = p1.features[feature] || '0';
      const val2 = p2.features[feature] || '0';

      if (val1 !== val2) {
        diffCount++;
      }
    }

    return diffCount;
  }

  /**
   * Calculate maximal opposition score for a phoneme pair
   * Higher scores = better candidates for maximal opposition intervention
   *
   * Scoring:
   * - Major class difference: +100 (REQUIRED)
   * - Each feature difference: +1
   */
  private calculateMaximalOppositionScore(ipa1: string, ipa2: string): number {
    let score = 0;

    // CRITICAL: Major class difference (required)
    if (this.hasMajorClassDifference(ipa1, ipa2)) {
      score += 100;
    } else {
      return 0; // Not suitable for maximal opposition without major class difference
    }

    // Feature differences
    score += this.countFeatureDifferences(ipa1, ipa2);

    return score;
  }

  /**
   * Generate maximal opposition pairs from a list of unknown phonemes
   * Based on Gierut's research showing better generalization than minimal pairs
   */
  async generateMaximalOppositionPairs(
    unknownPhonemes: string[],
    topN: number = 10
  ): Promise<Array<{
    phoneme1: string;
    phoneme2: string;
    score: number;
    major_class_diff: boolean;
    feature_diffs: number;
  }>> {
    await this.ensureLoaded();

    const pairs: Array<{
      phoneme1: string;
      phoneme2: string;
      score: number;
      major_class_diff: boolean;
      feature_diffs: number;
    }> = [];

    // Generate all pairs of unknown phonemes
    for (let i = 0; i < unknownPhonemes.length; i++) {
      for (let j = i + 1; j < unknownPhonemes.length; j++) {
        const p1 = unknownPhonemes[i];
        const p2 = unknownPhonemes[j];

        const score = this.calculateMaximalOppositionScore(p1, p2);

        // Only include pairs with major class difference (score >= 100)
        if (score >= 100) {
          pairs.push({
            phoneme1: p1,
            phoneme2: p2,
            score,
            major_class_diff: true,
            feature_diffs: score - 100, // Subtract the 100 pt bonus
          });
        }
      }
    }

    // Sort by score (best first) and return top N
    pairs.sort((a, b) => b.score - a.score);
    return pairs.slice(0, topN);
  }

  /**
   * Find minimal pair word lists for a maximal opposition phoneme pair
   * These are used for intervention activities
   */
  async findMaximalOppositionWordLists(
    phoneme1: string,
    phoneme2: string,
    position: 'initial' | 'medial' | 'final' | 'any' = 'initial',
    maxPairs: number = 10
  ): Promise<Array<{
    word1: Word;
    word2: Word;
    position: number;
  }>> {
    await this.ensureLoaded();

    const pairs: Array<{
      word1: Word;
      word2: Word;
      position: number;
    }> = [];

    // Group words by phoneme length for efficient comparison
    const byLength: Map<number, Array<[string, WordMetadata]>> = new Map();
    for (const [word, metadata] of this.wordMetadata.entries()) {
      const len = metadata.phoneme_count;
      if (!byLength.has(len)) {
        byLength.set(len, []);
      }
      byLength.get(len)!.push([word, metadata]);
    }

    // Find minimal pairs
    for (const [length, words] of byLength.entries()) {
      if (length < 2) continue; // Need at least 2 phonemes

      for (let i = 0; i < words.length && pairs.length < maxPairs; i++) {
        const [, meta1] = words[i];

        for (let j = i + 1; j < words.length && pairs.length < maxPairs; j++) {
          const [, meta2] = words[j];

          // Find differences
          let diffPos = -1;
          let diffCount = 0;
          for (let k = 0; k < meta1.phonemes.length; k++) {
            if (meta1.phonemes[k] !== meta2.phonemes[k]) {
              diffPos = k;
              diffCount++;
            }
          }

          // Must differ by exactly one phoneme
          if (diffCount !== 1) continue;

          // Check position constraint
          if (position === 'initial' && diffPos !== 0) continue;
          if (position === 'final' && diffPos !== length - 1) continue;
          if (position === 'medial' && (diffPos === 0 || diffPos === length - 1)) continue;

          // Check if the differing phonemes match our target pair
          const ph1 = meta1.phonemes[diffPos];
          const ph2 = meta2.phonemes[diffPos];

          if ((ph1 === phoneme1 && ph2 === phoneme2) ||
              (ph1 === phoneme2 && ph2 === phoneme1)) {
            pairs.push({
              word1: this.metadataToWord(meta1),
              word2: this.metadataToWord(meta2),
              position: diffPos,
            });

            if (pairs.length >= maxPairs) break;
          }
        }
        if (pairs.length >= maxPairs) break;
      }
      if (pairs.length >= maxPairs) break;
    }

    return pairs;
  }

  // ==========================================================================
  // Multiple Opposition (Gierut 1989-1992, Storkel 2022)
  // ==========================================================================

  /**
   * Select representative targets from a collapse using Maximal Classification + Maximal Distinction
   *
   * Maximal Classification: Select targets that represent the breadth of the phonological collapse
   * Maximal Distinction: Maximize phonological distance from the substitute phoneme
   *
   * @param substitutePhoneme - The phoneme the child produces (e.g., 't')
   * @param targetPhonemes - All phonemes the child should produce (e.g., ['θ', 'k', 'l', 'kr'])
   * @param count - How many representative targets to select (typically 3-5)
   * @returns Selected representative target phonemes
   */
  selectRepresentativeTargets(
    substitutePhoneme: string,
    targetPhonemes: string[],
    count: number = 3
  ): string[] {
    if (targetPhonemes.length === 0) return [];
    if (targetPhonemes.length <= count) return [...targetPhonemes];

    // Calculate distance from substitute for each target (Maximal Distinction)
    const targetDistances = targetPhonemes.map(target => ({
      phoneme: target,
      distanceFromSubstitute: this.countFeatureDifferences(substitutePhoneme, target),
    }));

    // Sort by distance from substitute (furthest first)
    targetDistances.sort((a, b) => b.distanceFromSubstitute - a.distanceFromSubstitute);

    // Greedy selection algorithm for Maximal Classification:
    // 1. Start with the target most distant from substitute
    // 2. Iteratively add targets that maximize diversity (distance from already-selected targets)
    const selected: string[] = [targetDistances[0].phoneme];
    const remaining = targetDistances.slice(1);

    while (selected.length < count && remaining.length > 0) {
      let bestIdx = -1;
      let bestAvgDistance = -1;

      // Find remaining target with maximum average distance from already-selected targets
      for (let i = 0; i < remaining.length; i++) {
        const candidate = remaining[i].phoneme;

        // Calculate average distance from all selected targets
        let totalDistance = 0;
        for (const selectedTarget of selected) {
          totalDistance += this.countFeatureDifferences(candidate, selectedTarget);
        }
        const avgDistance = totalDistance / selected.length;

        if (avgDistance > bestAvgDistance) {
          bestAvgDistance = avgDistance;
          bestIdx = i;
        }
      }

      if (bestIdx >= 0) {
        selected.push(remaining[bestIdx].phoneme);
        remaining.splice(bestIdx, 1);
      } else {
        break;
      }
    }

    return selected;
  }

  /**
   * Generate minimal sets (triplets/quadruplets/quintuplets) for Multiple Opposition intervention
   *
   * Finds groups of 3-5 words that:
   * - All have the same length
   * - Differ at exactly one position
   * - One word has the substitute phoneme at that position
   * - Other words have different target phonemes at that position
   *
   * @param substitutePhoneme - The phoneme the child produces (e.g., 't')
   * @param targetPhonemes - Selected representative targets (e.g., ['θ', 'l', 'kr'])
   * @param position - Where contrast should occur ('initial', 'medial', 'final', 'any')
   * @param maxSets - Maximum number of minimal sets to return
   * @returns Array of minimal sets, each containing 3-5 words
   */
  async generateMultipleOppositionSets(
    substitutePhoneme: string,
    targetPhonemes: string[],
    position: 'initial' | 'medial' | 'final' | 'any' = 'initial',
    maxSets: number = 10
  ): Promise<Array<{
    words: Array<{ word: Word; phoneme: string }>;
    position: number;
  }>> {
    await this.ensureLoaded();

    const allPhonemes = [substitutePhoneme, ...targetPhonemes];
    const minSetSize = 3; // At least triplets
    const maxSetSize = Math.min(5, allPhonemes.length); // Up to quintuplets, or all phonemes

    const sets: Array<{
      words: Array<{ word: Word; phoneme: string }>;
      position: number;
    }> = [];

    // Group words by length for efficient comparison
    const byLength: Map<number, Array<[string, WordMetadata]>> = new Map();
    for (const [word, metadata] of this.wordMetadata.entries()) {
      const len = metadata.phoneme_count;
      if (!byLength.has(len)) {
        byLength.set(len, []);
      }
      byLength.get(len)!.push([word, metadata]);
    }

    // For each word length
    for (const [length, words] of byLength.entries()) {
      if (length < 2) continue;

      // Build position-phoneme index for fast lookup
      // Map: position -> phoneme -> [words with that phoneme at that position]
      const positionIndex = new Map<number, Map<string, WordMetadata[]>>();

      for (let pos = 0; pos < length; pos++) {
        // Skip positions that don't match constraint
        if (position === 'initial' && pos !== 0) continue;
        if (position === 'final' && pos !== length - 1) continue;
        if (position === 'medial' && (pos === 0 || pos === length - 1)) continue;

        const phonemeMap = new Map<string, WordMetadata[]>();

        for (const [, meta] of words) {
          const phoneme = meta.phonemes[pos];
          if (allPhonemes.includes(phoneme)) {
            if (!phonemeMap.has(phoneme)) {
              phonemeMap.set(phoneme, []);
            }
            phonemeMap.get(phoneme)!.push(meta);
          }
        }

        positionIndex.set(pos, phonemeMap);
      }

      // Find minimal sets at each position
      for (const [pos, phonemeMap] of positionIndex.entries()) {
        // Need at least minSetSize different phonemes represented
        if (phonemeMap.size < minSetSize) continue;

        // Get words for each phoneme
        const phonemeWords = new Map<string, WordMetadata[]>();
        for (const phoneme of allPhonemes) {
          if (phonemeMap.has(phoneme)) {
            phonemeWords.set(phoneme, phonemeMap.get(phoneme)!);
          }
        }

        // Must have substitute + at least 2 targets
        if (!phonemeWords.has(substitutePhoneme)) continue;
        if (phonemeWords.size < minSetSize) continue;

        // Try to build minimal sets
        // For each word with substitute phoneme, find matching words with target phonemes
        const substituteWords = phonemeWords.get(substitutePhoneme)!;

        for (const subWord of substituteWords) {
          if (sets.length >= maxSets) break;

          // Find words that differ ONLY at position pos
          const matchingWords: Array<{ meta: WordMetadata; phoneme: string }> = [];
          const usedPhonemes = new Set<string>([substitutePhoneme]); // Track phonemes already in the set

          for (const [targetPhoneme, targetWords] of phonemeWords.entries()) {
            if (targetPhoneme === substitutePhoneme) continue;
            if (usedPhonemes.has(targetPhoneme)) continue; // Skip if phoneme already used

            for (const targetWord of targetWords) {
              // Check if words differ only at position pos
              let differenceCount = 0;
              for (let k = 0; k < length; k++) {
                if (subWord.phonemes[k] !== targetWord.phonemes[k]) {
                  if (k === pos) {
                    differenceCount++;
                  } else {
                    // Different at another position - not a minimal pair
                    differenceCount = 999;
                    break;
                  }
                }
              }

              if (differenceCount === 1) {
                matchingWords.push({ meta: targetWord, phoneme: targetPhoneme });
                usedPhonemes.add(targetPhoneme); // Mark this phoneme as used
                break; // Only take one word per phoneme
              }
            }
          }

          // Need at least 2 different targets (for triplet: substitute + 2 targets)
          if (matchingWords.length >= minSetSize - 1) {
            // Take best targets (up to maxSetSize - 1)
            const selectedTargets = matchingWords.slice(0, maxSetSize - 1);

            sets.push({
              words: [
                { word: this.metadataToWord(subWord), phoneme: substitutePhoneme },
                ...selectedTargets.map(t => ({ word: this.metadataToWord(t.meta), phoneme: t.phoneme }))
              ],
              position: pos,
            });

            if (sets.length >= maxSets) break;
          }
        }

        if (sets.length >= maxSets) break;
      }

      if (sets.length >= maxSets) break;
    }

    return sets;
  }

  // ==========================================================================
  // Statistics
  // ==========================================================================

  /**
   * Get database statistics
   */
  async getStats(): Promise<StatsResponse> {
    await this.ensureLoaded();

    return {
      total_words: this.wordMetadata.size,
      total_phonemes: 39, // From English phoneme inventory
      total_edges: 0, // Not applicable in client-side mode
      edge_types: {},
    };
  }

  /**
   * Get property ranges from loaded word data (returns cached version)
   */
  async getPropertyRanges(): Promise<Record<string, [number, number]>> {
    await this.ensureLoaded();

    // Return cached ranges (computed during data load)
    if (this.cachedRanges) {
      return this.cachedRanges;
    }

    // Fallback: compute if not cached
    return this._computePropertyRanges();
  }

  /**
   * Internal method to compute property ranges
   */
  private _computePropertyRanges(): Record<string, [number, number]> {
    // Compute actual ranges from loaded data
    const ranges: Record<string, { min: number; max: number }> = {
      syllables: { min: Infinity, max: -Infinity },
      phonemes: { min: Infinity, max: -Infinity },
      wcm: { min: Infinity, max: -Infinity },
      msh: { min: Infinity, max: -Infinity },
      frequency: { min: Infinity, max: -Infinity },
      aoa: { min: Infinity, max: -Infinity },
      imageability: { min: Infinity, max: -Infinity },
      familiarity: { min: Infinity, max: -Infinity },
      concreteness: { min: Infinity, max: -Infinity },
      valence: { min: Infinity, max: -Infinity },
      arousal: { min: Infinity, max: -Infinity },
      dominance: { min: Infinity, max: -Infinity },
    };

    // Scan all words to find actual min/max
    for (const metadata of this.wordMetadata.values()) {
      // Phonological
      if (metadata.syllable_count) {
        ranges.syllables.min = Math.min(ranges.syllables.min, metadata.syllable_count);
        ranges.syllables.max = Math.max(ranges.syllables.max, metadata.syllable_count);
      }
      if (metadata.phoneme_count) {
        ranges.phonemes.min = Math.min(ranges.phonemes.min, metadata.phoneme_count);
        ranges.phonemes.max = Math.max(ranges.phonemes.max, metadata.phoneme_count);
      }
      if (metadata.wcm_score !== null && metadata.wcm_score !== undefined) {
        ranges.wcm.min = Math.min(ranges.wcm.min, metadata.wcm_score);
        ranges.wcm.max = Math.max(ranges.wcm.max, metadata.wcm_score);
      }
      if (metadata.msh_stage !== null && metadata.msh_stage !== undefined) {
        ranges.msh.min = Math.min(ranges.msh.min, metadata.msh_stage);
        ranges.msh.max = Math.max(ranges.msh.max, metadata.msh_stage);
      }

      // Lexical
      if (metadata.frequency !== null && metadata.frequency !== undefined) {
        ranges.frequency.min = Math.min(ranges.frequency.min, metadata.frequency);
        ranges.frequency.max = Math.max(ranges.frequency.max, metadata.frequency);
      }
      if (metadata.aoa !== null && metadata.aoa !== undefined) {
        ranges.aoa.min = Math.min(ranges.aoa.min, metadata.aoa);
        ranges.aoa.max = Math.max(ranges.aoa.max, metadata.aoa);
      }

      // Semantic
      if (metadata.imageability !== null && metadata.imageability !== undefined) {
        ranges.imageability.min = Math.min(ranges.imageability.min, metadata.imageability);
        ranges.imageability.max = Math.max(ranges.imageability.max, metadata.imageability);
      }
      if (metadata.familiarity !== null && metadata.familiarity !== undefined) {
        ranges.familiarity.min = Math.min(ranges.familiarity.min, metadata.familiarity);
        ranges.familiarity.max = Math.max(ranges.familiarity.max, metadata.familiarity);
      }
      if (metadata.concreteness !== null && metadata.concreteness !== undefined) {
        ranges.concreteness.min = Math.min(ranges.concreteness.min, metadata.concreteness);
        ranges.concreteness.max = Math.max(ranges.concreteness.max, metadata.concreteness);
      }

      // Affective
      if (metadata.valence !== null && metadata.valence !== undefined) {
        ranges.valence.min = Math.min(ranges.valence.min, metadata.valence);
        ranges.valence.max = Math.max(ranges.valence.max, metadata.valence);
      }
      if (metadata.arousal !== null && metadata.arousal !== undefined) {
        ranges.arousal.min = Math.min(ranges.arousal.min, metadata.arousal);
        ranges.arousal.max = Math.max(ranges.arousal.max, metadata.arousal);
      }
      if (metadata.dominance !== null && metadata.dominance !== undefined) {
        ranges.dominance.min = Math.min(ranges.dominance.min, metadata.dominance);
        ranges.dominance.max = Math.max(ranges.dominance.max, metadata.dominance);
      }
    }

    // Convert to [min, max] tuples, rounding nicely
    // Handle edge case where no words have a particular property (all null)
    const result: Record<string, [number, number]> = {};
    for (const [key, range] of Object.entries(ranges)) {
      if (range.min === Infinity || range.max === -Infinity) {
        // No valid values found - use sensible defaults
        result[key] = [0, 10];
      } else {
        result[key] = [
          Math.floor(range.min),
          Math.ceil(range.max)
        ];
      }
    }

    return result;
  }

  // ==========================================================================
  // Helper Functions
  // ==========================================================================

  /**
   * Convert metadata to Word type
   */
  private metadataToWord(metadata: WordMetadata): Word {
    return {
      word_id: 0, // Not applicable
      word: metadata.word,
      ipa: metadata.ipa,
      arpa: metadata.arpa,
      phonemes: metadata.phonemes.map((ipa, i) => ({
        ipa,
        arpa: metadata.phonemes_arpa?.[i],  // Optional chaining for Phase 3 data
        position: i,
      })),
      syllables: metadata.syllables,
      phoneme_count: metadata.phoneme_count,
      syllable_count: metadata.syllable_count,
      wcm_score: metadata.wcm_score,
      msh_stage: metadata.msh_stage,
      frequency: metadata.frequency,
      log_frequency: metadata.log_frequency,
      aoa: metadata.aoa,
      imageability: metadata.imageability,
      familiarity: metadata.familiarity,
      concreteness: metadata.concreteness,
      valence: metadata.valence,
      arousal: metadata.arousal,
      dominance: metadata.dominance,
    };
  }

  /**
   * Get Phoible embeddings for a word (no dequantization needed)
   */
  private getWordEmbeddings(word: string): number[][] | null {
    if (!this.embeddings) return null;

    const embeddings = this.embeddings.embeddings[word.toLowerCase()];
    if (!embeddings) return null;

    return embeddings;
  }

  /**
   * Compute soft Levenshtein similarity between syllable sequences
   *
   * Uses dynamic programming with soft costs based on syllable similarity.
   * See docs/EMBEDDINGS_ARCHITECTURE.md for algorithm details.
   *
   * @param syllables1 First word's syllable embeddings
   * @param syllables2 Second word's syllable embeddings
   * @param weights Optional component weights for similarity calculation
   */
  private computeSoftLevenshteinSimilarity(
    syllables1: number[][],
    syllables2: number[][],
    weights?: { onset: number; nucleus: number; coda: number }
  ): number {
    // Null/undefined check
    if (!syllables1 || !syllables2 || !Array.isArray(syllables1) || !Array.isArray(syllables2)) {
      console.warn('Invalid syllables arrays passed to computeSoftLevenshteinSimilarity:', { syllables1, syllables2 });
      return 0;
    }

    const len1 = syllables1.length;
    const len2 = syllables2.length;

    // Pre-compute pairwise syllable similarities
    const simMatrix: number[][] = Array(len1)
      .fill(0)
      .map(() => Array(len2).fill(0));

    for (let i = 0; i < len1; i++) {
      for (let j = 0; j < len2; j++) {
        // Validate syllable embeddings exist
        if (!syllables1[i] || !syllables2[j]) {
          console.warn(`Missing syllable embedding: syllables1[${i}]=${!!syllables1[i]}, syllables2[${j}]=${!!syllables2[j]}`);
          simMatrix[i][j] = 0;
          continue;
        }
        // Use weighted similarity with custom or default weights
        simMatrix[i][j] = this.cosineSimilarityWeighted(syllables1[i], syllables2[j], weights);
      }
    }

    // Dynamic programming for edit distance
    const dp: number[][] = Array(len1 + 1)
      .fill(0)
      .map(() => Array(len2 + 1).fill(0));

    // Initialize: cost of insertions/deletions
    for (let i = 0; i <= len1; i++) {
      dp[i][0] = i;
    }
    for (let j = 0; j <= len2; j++) {
      dp[0][j] = j;
    }

    // Fill DP table
    for (let i = 1; i <= len1; i++) {
      for (let j = 1; j <= len2; j++) {
        const matchCost = 1.0 - simMatrix[i - 1][j - 1]; // 0 if identical, 2 if opposite

        dp[i][j] = Math.min(
          dp[i - 1][j] + 1.0, // Delete
          dp[i][j - 1] + 1.0, // Insert
          dp[i - 1][j - 1] + matchCost // Match/substitute
        );
      }
    }

    // Convert to similarity [0, 1]
    const maxLen = Math.max(len1, len2);
    if (maxLen === 0) return 1.0;

    const editDistance = dp[len1][len2];
    const similarity = 1.0 - editDistance / maxLen;

    return Math.max(0, Math.min(1, similarity));
  }

  /**
   * Cosine similarity between two vectors
   */
  private cosineSimilarity(vec1: number[], vec2: number[]): number {
    // Null/undefined check
    if (!vec1 || !vec2 || !Array.isArray(vec1) || !Array.isArray(vec2)) {
      console.warn('Invalid vectors passed to cosineSimilarity:', { vec1, vec2 });
      return 0;
    }

    // Length check
    if (vec1.length !== vec2.length) {
      console.warn(`Vector length mismatch: vec1=${vec1.length}, vec2=${vec2.length}`);
      return 0;
    }

    let dot = 0;
    let mag1 = 0;
    let mag2 = 0;

    for (let i = 0; i < vec1.length; i++) {
      dot += vec1[i] * vec2[i];
      mag1 += vec1[i] * vec1[i];
      mag2 += vec2[i] * vec2[i];
    }

    mag1 = Math.sqrt(mag1);
    mag2 = Math.sqrt(mag2);

    if (mag1 === 0 || mag2 === 0) return 0;

    return dot / (mag1 * mag2);
  }

  /**
   * Weighted component cosine similarity for syllable embeddings.
   *
   * Computes similarity with user-adjustable weights for onset, nucleus, and coda.
   * Default equal weights (1/3 each) ensure balanced comparison.
   *
   * @param vec1 228-dim syllable embedding [onset(76) + nucleus(76) + coda(76)]
   * @param vec2 228-dim syllable embedding [onset(76) + nucleus(76) + coda(76)]
   * @param weights Optional weights { onset, nucleus, coda }. Defaults to equal (0.33 each)
   * @returns Weighted average similarity
   */
  private cosineSimilarityWeighted(
    vec1: number[],
    vec2: number[],
    weights: { onset: number; nucleus: number; coda: number } = { onset: 0.33, nucleus: 0.33, coda: 0.33 }
  ): number {
    // Null/undefined check
    if (!vec1 || !vec2 || !Array.isArray(vec1) || !Array.isArray(vec2)) {
      console.warn('Invalid vectors passed to cosineSimilarityWeighted:', { vec1, vec2 });
      return 0;
    }

    // Dimension check
    if (vec1.length !== 228 || vec2.length !== 228) {
      console.warn(`Unexpected embedding dimensions: vec1=${vec1.length}, vec2=${vec2.length}. Expected 228.`);
      return 0;
    }

    // Component boundaries for Phase 3 Phoible embeddings (76 dims each)
    const ONSET_START = 0;
    const ONSET_END = 76;
    const NUCLEUS_START = 76;
    const NUCLEUS_END = 152;
    const CODA_START = 152;
    const CODA_END = 228;

    // Create weighted copies of the vectors by scaling each component
    // Use sqrt of weights to preserve magnitude relationships
    const weighted1 = new Float32Array(228);
    const weighted2 = new Float32Array(228);

    const sqrtOnset = Math.sqrt(weights.onset);
    const sqrtNucleus = Math.sqrt(weights.nucleus);
    const sqrtCoda = Math.sqrt(weights.coda);

    // Scale onset component
    for (let i = ONSET_START; i < ONSET_END; i++) {
      weighted1[i] = vec1[i] * sqrtOnset;
      weighted2[i] = vec2[i] * sqrtOnset;
    }

    // Scale nucleus component
    for (let i = NUCLEUS_START; i < NUCLEUS_END; i++) {
      weighted1[i] = vec1[i] * sqrtNucleus;
      weighted2[i] = vec2[i] * sqrtNucleus;
    }

    // Scale coda component
    for (let i = CODA_START; i < CODA_END; i++) {
      weighted1[i] = vec1[i] * sqrtCoda;
      weighted2[i] = vec2[i] * sqrtCoda;
    }

    // Compute cosine similarity on weighted vectors
    return this.cosineSimilarity(Array.from(weighted1), Array.from(weighted2));
  }
}

// Export singleton instance
export const clientSideData = new ClientSideDataService();
