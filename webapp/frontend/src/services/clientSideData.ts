/**
 * Client-Side Data Service for PhonoLex
 *
 * Loads all data from static JSON files and implements phonological operations
 * entirely in the browser - no backend required!
 *
 * Data loaded:
 * - word_metadata.json (14 MB) - All word properties
 * - embeddings_quantized.json (75 MB) - Syllable embeddings for similarity
 * - arpa_to_ipa.json (1.2 KB) - ARPAbet mapping reference
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

// ============================================================================
// Types
// ============================================================================

interface WordMetadata {
  word: string;
  ipa: string;
  arpa: string;
  phonemes: string[];
  phonemes_arpa: string[];
  syllables: Array<{
    onset: string[];
    nucleus: string;
    coda: string[];
    stress?: number;
  }>;
  phoneme_count: number;
  syllable_count: number;
  word_length?: 'short' | 'medium' | 'long' | null;
  complexity?: 'low' | 'medium' | 'high' | null;
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
  embeddings: Record<string, number[][]>; // word -> syllables -> int8 values
  scales: Record<string, number | number[]>; // dequantization scales
  embedding_dim: number;
  quantization: string;
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
        fetch('/data/embeddings_quantized.json.gz'),
        fetch('/data/arpa_to_ipa.json.gz'),
        fetch('/data/phonemes.json.gz'),
      ]);

      if (!metadataRes.ok || !embeddingsRes.ok || !arpaRes.ok || !phonemesRes.ok) {
        throw new Error('Failed to load data files');
      }

      // Parse JSON
      const [metadataJson, embeddingsJson, _arpaJson, phonemesJson] = await Promise.all([
        metadataRes.json(),
        embeddingsRes.json(),
        arpaRes.json(),
        phonemesRes.json(),
      ]);

      // Store metadata in Map for fast lookup
      Object.entries(metadataJson as Record<string, WordMetadata>).forEach(
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

      // Word length category
      if (request.word_length && metadata.word_length !== request.word_length) matches = false;

      // Complexity category
      if (request.complexity && metadata.complexity !== request.complexity) matches = false;

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
   * Supports both space-separated ("k æ t") and concatenated ("kæt") input
   * Uses greedy longest-match for multi-character phonemes
   */
  private tokenizePhonemes(input: string): string[] {
    const trimmed = input.trim();

    // If input contains spaces, use space-separated tokens
    if (/\s/.test(trimmed)) {
      return trimmed.split(/\s+/).filter(p => p.length > 0);
    }

    // Otherwise, greedily match against known phonemes
    const phonemes: string[] = [];
    let i = 0;

    while (i < trimmed.length) {
      let matched = false;

      // Try to match longest phoneme first (up to 3 characters)
      for (let len = Math.min(3, trimmed.length - i); len >= 1; len--) {
        const candidate = trimmed.substring(i, i + len);
        if (this.phonemes.has(candidate)) {
          phonemes.push(candidate);
          i += len;
          matched = true;
          break;
        }
      }

      // If no match, treat as single character (might be invalid, but let it through)
      if (!matched) {
        phonemes.push(trimmed[i]);
        i++;
      }
    }

    return phonemes;
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

        // Helper to check if sequence exists in array
        const containsSequence = (haystack: string[], needle: string[]): boolean => {
          for (let i = 0; i <= haystack.length - needle.length; i++) {
            if (JSON.stringify(haystack.slice(i, i + needle.length)) === JSON.stringify(needle)) {
              return true;
            }
          }
          return false;
        };

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
    phoneme1: any;
    phoneme2: any;
    feature_distance: number;
    differing_features: string[];
    shared_features: string[];
  }> {
    await this.ensureLoaded();

    const p1 = this.phonemes.get(ipa1);
    const p2 = this.phonemes.get(ipa2);

    if (!p1 || !p2) {
      throw new Error(`Phoneme not found: ${!p1 ? ipa1 : ipa2}`);
    }

    // Get all unique features
    const allFeatures = new Set([...Object.keys(p1.features), ...Object.keys(p2.features)]);
    const differing: string[] = [];
    const shared: string[] = [];

    // Compare features
    for (const feature of allFeatures) {
      if (p1.features[feature] === p2.features[feature]) {
        shared.push(feature);
      } else {
        differing.push(feature);
      }
    }

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
      feature_distance: differing.length,
      differing_features: differing,
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
    limit: number = 50
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
        candidateEmbeddings
      );

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
          if (
            diffCount === 1 &&
            ((diffPhoneme1 === phoneme1 && diffPhoneme2 === phoneme2) ||
              (diffPhoneme1 === phoneme2 && diffPhoneme2 === phoneme1))
          ) {
            results.push({
              word1: this.metadataToWord(metadata1),
              word2: this.metadataToWord(metadata2),
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
      const similarity = this._hierarchicalSoftLevenshtein(targetEmbeddings, candidateEmbeddings);
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

  /**
   * Calculate hierarchical soft Levenshtein similarity between syllable sequences
   * Implementation from scripts/build_layer4_syllable_embeddings.py
   */
  private _hierarchicalSoftLevenshtein(syllables1: number[][], syllables2: number[][]): number {
    const len1 = syllables1.length;
    const len2 = syllables2.length;

    if (len1 === 0 && len2 === 0) {
      return 1.0;
    }

    // Pre-compute all pairwise cosine similarities (vectorized)
    const simMatrix: number[][] = [];
    for (let i = 0; i < len1; i++) {
      simMatrix[i] = [];
      for (let j = 0; j < len2; j++) {
        simMatrix[i][j] = this._cosineSimilarity(syllables1[i], syllables2[j]);
      }
    }

    // Dynamic programming for edit distance with soft costs
    const dp: number[][] = [];
    for (let i = 0; i <= len1; i++) {
      dp[i] = [];
      dp[i][0] = i * 1.0; // Deletion cost
    }
    for (let j = 0; j <= len2; j++) {
      dp[0][j] = j * 1.0; // Insertion cost
    }

    // Fill DP table
    for (let i = 1; i <= len1; i++) {
      for (let j = 1; j <= len2; j++) {
        // Match/substitute cost: 1 - similarity (0 if identical, 2 if opposite)
        const matchCost = 1.0 - simMatrix[i - 1][j - 1];

        dp[i][j] = Math.min(
          dp[i - 1][j] + 1.0,        // Delete from s1
          dp[i][j - 1] + 1.0,        // Insert from s2
          dp[i - 1][j - 1] + matchCost  // Match/substitute
        );
      }
    }

    // Normalize to [0, 1] similarity
    const maxLen = Math.max(len1, len2);
    if (maxLen === 0) {
      return 1.0;
    }

    const editDistance = dp[len1][len2];
    const similarity = 1.0 - (editDistance / maxLen);
    return Math.max(0.0, Math.min(1.0, similarity));
  }

  /**
   * Calculate cosine similarity between two embedding vectors
   */
  private _cosineSimilarity(vec1: number[], vec2: number[]): number {
    if (vec1.length !== vec2.length) {
      return 0;
    }

    let dotProduct = 0;
    let norm1 = 0;
    let norm2 = 0;

    for (let i = 0; i < vec1.length; i++) {
      dotProduct += vec1[i] * vec2[i];
      norm1 += vec1[i] * vec1[i];
      norm2 += vec2[i] * vec2[i];
    }

    const magnitude = Math.sqrt(norm1) * Math.sqrt(norm2);
    return magnitude === 0 ? 0 : dotProduct / magnitude;
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
        arpa: metadata.phonemes_arpa[i],
        position: i,
      })),
      syllables: metadata.syllables,
      phoneme_count: metadata.phoneme_count,
      syllable_count: metadata.syllable_count,
      wcm_score: metadata.wcm_score,
      msh_stage: metadata.msh_stage,
      word_length: this.categorizeWordLength(metadata.phoneme_count),
      complexity: this.categorizeComplexity(metadata.wcm_score),
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
   * Categorize word length
   */
  private categorizeWordLength(phonemeCount: number): 'short' | 'medium' | 'long' {
    if (phonemeCount <= 4) return 'short';
    if (phonemeCount <= 7) return 'medium';
    return 'long';
  }

  /**
   * Categorize complexity
   */
  private categorizeComplexity(wcm: number | null): 'low' | 'medium' | 'high' {
    if (wcm === null) return 'medium';
    if (wcm <= 2) return 'low';
    if (wcm <= 5) return 'medium';
    return 'high';
  }

  /**
   * Get dequantized embeddings for a word
   */
  private getWordEmbeddings(word: string): number[][] | null {
    if (!this.embeddings) return null;

    const quantized = this.embeddings.embeddings[word.toLowerCase()];
    if (!quantized) return null;

    const scale = this.embeddings.scales[word.toLowerCase()];
    if (!scale) return null;

    // Dequantize each syllable
    return quantized.map((syllable) =>
      syllable.map((val) => val * (Array.isArray(scale) ? scale[0] : scale))
    );
  }

  /**
   * Compute soft Levenshtein similarity between syllable sequences
   *
   * Uses dynamic programming with soft costs based on syllable similarity.
   * See docs/EMBEDDINGS_ARCHITECTURE.md for algorithm details.
   */
  private computeSoftLevenshteinSimilarity(
    syllables1: number[][],
    syllables2: number[][]
  ): number {
    const len1 = syllables1.length;
    const len2 = syllables2.length;

    // Pre-compute pairwise syllable similarities
    const simMatrix: number[][] = Array(len1)
      .fill(0)
      .map(() => Array(len2).fill(0));

    for (let i = 0; i < len1; i++) {
      for (let j = 0; j < len2; j++) {
        simMatrix[i][j] = this.cosineSimilarity(syllables1[i], syllables2[j]);
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
}

// Export singleton instance
export const clientSideData = new ClientSideDataService();
