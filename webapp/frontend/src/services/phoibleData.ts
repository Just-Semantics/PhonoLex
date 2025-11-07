/**
 * PHOIBLE Data Service
 *
 * Loads and manages phoneme vectors and language inventories for the
 * Phoneme Difficulty Tool (based on Flege's Speech Learning Model).
 *
 * Data: 3,142 phonemes across 2,095 languages
 */

export interface PhoiblePhoneme {
  ipa: string;
  vector: number[];  // 76-dim feature vector
}

export interface PhoiblePhonemeMetadata {
  ipa: string;
  languages: string[];  // ISO 639-3 codes
  language_count: number;
}

export interface PhoibleLanguage {
  iso: string;
  name: string;
  glottocode: string;
  phonemes: string[];  // IPA symbols
  phoneme_count: number;
}

export interface PhoibleData {
  phonemes: Record<string, PhoiblePhoneme>;
  metadata: Record<string, PhoiblePhonemeMetadata>;
  languages: Record<string, PhoibleLanguage>;
  isLoaded: boolean;
}

export interface DifficultyResult {
  l2Phoneme: string;
  closestL1: string;
  distance: number;
  category: 'identical' | 'similar' | 'new';
  difficulty: number;  // 1-5
  explanation: string;
}

// Singleton instance
let phoibleDataCache: PhoibleData | null = null;

/**
 * Load PHOIBLE data from static JSON files.
 * Returns cached data if already loaded.
 */
export async function loadPhoibleData(): Promise<PhoibleData> {
  if (phoibleDataCache?.isLoaded) {
    return phoibleDataCache;
  }

  console.log('Loading PHOIBLE data...');

  try {
    // Load both files in parallel
    const [phonemesResponse, languagesResponse] = await Promise.all([
      fetch('/data/phoible_phonemes.json.gz'),
      fetch('/data/phoible_languages.json.gz')
    ]);

    if (!phonemesResponse.ok || !languagesResponse.ok) {
      throw new Error('Failed to load PHOIBLE data files');
    }

    const phonemesData = await phonemesResponse.json();
    const languagesData = await languagesResponse.json();

    // Transform data structure
    const phonemes: Record<string, PhoiblePhoneme> = {};
    for (const [ipa, vector] of Object.entries(phonemesData.phonemes)) {
      phonemes[ipa] = { ipa, vector: vector as number[] };
    }

    phoibleDataCache = {
      phonemes,
      metadata: phonemesData.metadata,
      languages: languagesData.languages,
      isLoaded: true
    };

    console.log(`✓ Loaded ${Object.keys(phonemes).length} phonemes`);
    console.log(`✓ Loaded ${Object.keys(languagesData.languages).length} languages`);

    return phoibleDataCache;
  } catch (error) {
    console.error('Error loading PHOIBLE data:', error);
    throw error;
  }
}

/**
 * Compute cosine similarity between two vectors.
 */
function cosineSimilarity(vec1: number[], vec2: number[]): number {
  if (vec1.length !== vec2.length) {
    throw new Error('Vectors must have same length');
  }

  const dotProduct = vec1.reduce((sum, v, i) => sum + v * vec2[i], 0);
  const norm1 = Math.sqrt(vec1.reduce((sum, v) => sum + v * v, 0));
  const norm2 = Math.sqrt(vec2.reduce((sum, v) => sum + v * v, 0));

  if (norm1 === 0 || norm2 === 0) return 0;
  return dotProduct / (norm1 * norm2);
}

/**
 * Classify L2 phoneme difficulty based on phonetic distance.
 *
 * Based on Flege's Speech Learning Model (1995):
 * - Identical (< 0.1): Easy transfer from L1
 * - Similar (0.1-0.5): HARDEST - equivalence classification (H5)
 * - New (> 0.5): Easier - differences discernible (H3)
 */
function classifyDifficulty(distance: number): {
  category: 'identical' | 'similar' | 'new';
  difficulty: number;
  explanation: string;
} {
  if (distance < 0.1) {
    return {
      category: 'identical',
      difficulty: 1,
      explanation: 'Perfect transfer - identical to L1 sound'
    };
  } else if (distance < 0.3) {
    return {
      category: 'similar',
      difficulty: 5,
      explanation: 'VERY HARD - equivalence classification (Flege H5)'
    };
  } else if (distance < 0.5) {
    return {
      category: 'similar',
      difficulty: 4,
      explanation: 'HARD - perceived as same but actually different'
    };
  } else if (distance < 0.7) {
    return {
      category: 'new',
      difficulty: 2,
      explanation: 'Easier - differences discernible'
    };
  } else {
    return {
      category: 'new',
      difficulty: 1,
      explanation: 'Easy - clearly different, new category formed'
    };
  }
}

/**
 * Find the closest L1 phoneme to a target L2 phoneme.
 */
function findClosestPhoneme(
  targetPhoneme: string,
  candidatePhonemes: string[],
  phoibleData: PhoibleData
): { phoneme: string; distance: number } {
  const targetVec = phoibleData.phonemes[targetPhoneme]?.vector;
  if (!targetVec) {
    return { phoneme: '', distance: Infinity };
  }

  let bestPhoneme = '';
  let bestDistance = Infinity;

  for (const candidate of candidatePhonemes) {
    const candidateVec = phoibleData.phonemes[candidate]?.vector;
    if (!candidateVec) continue;

    const similarity = cosineSimilarity(targetVec, candidateVec);
    const distance = 1 - similarity;

    if (distance < bestDistance) {
      bestDistance = distance;
      bestPhoneme = candidate;
    }
  }

  return { phoneme: bestPhoneme, distance: bestDistance };
}

/**
 * Analyze L1 → L2 phoneme learning difficulty.
 *
 * For each L2 phoneme:
 * 1. Check if identical phoneme exists in L1 (easy transfer)
 * 2. Otherwise, find closest L1 phoneme
 * 3. Compute phonetic distance
 * 4. Classify difficulty (identical/similar/new)
 *
 * Returns results sorted by difficulty (hardest first).
 */
export async function analyzeL1toL2(
  l1Iso: string,
  l2Iso: string
): Promise<DifficultyResult[]> {
  const phoibleData = await loadPhoibleData();

  const l1 = phoibleData.languages[l1Iso];
  const l2 = phoibleData.languages[l2Iso];

  if (!l1 || !l2) {
    throw new Error(`Language not found: ${!l1 ? l1Iso : l2Iso}`);
  }

  const results: DifficultyResult[] = [];

  for (const l2Phoneme of l2.phonemes) {
    // Check if identical phoneme exists in L1
    if (l1.phonemes.includes(l2Phoneme)) {
      results.push({
        l2Phoneme,
        closestL1: l2Phoneme,
        distance: 0,
        category: 'identical',
        difficulty: 1,
        explanation: 'Perfect transfer - exists in L1'
      });
    } else {
      // Find closest L1 phoneme
      const { phoneme: closestL1, distance } = findClosestPhoneme(
        l2Phoneme,
        l1.phonemes,
        phoibleData
      );

      if (!closestL1 || distance === Infinity) continue;

      const { category, difficulty, explanation } = classifyDifficulty(distance);

      results.push({
        l2Phoneme,
        closestL1,
        distance,
        category,
        difficulty,
        explanation
      });
    }
  }

  // Sort by difficulty (hardest first), then by distance
  results.sort((a, b) => {
    if (b.difficulty !== a.difficulty) {
      return b.difficulty - a.difficulty;
    }
    return b.distance - a.distance;
  });

  return results;
}

/**
 * Get summary statistics for difficulty analysis.
 */
export function getDifficultyStats(results: DifficultyResult[]): {
  total: number;
  identical: number;
  similar: number;
  new: number;
  identicalPercent: number;
  similarPercent: number;
  newPercent: number;
} {
  const total = results.length;
  const identical = results.filter(r => r.category === 'identical').length;
  const similar = results.filter(r => r.category === 'similar').length;
  const newCount = results.filter(r => r.category === 'new').length;

  return {
    total,
    identical,
    similar,
    new: newCount,
    identicalPercent: total > 0 ? (identical / total) * 100 : 0,
    similarPercent: total > 0 ? (similar / total) * 100 : 0,
    newPercent: total > 0 ? (newCount / total) * 100 : 0
  };
}

/**
 * Get all available languages sorted by name.
 */
export async function getAvailableLanguages(): Promise<PhoibleLanguage[]> {
  const data = await loadPhoibleData();
  return Object.values(data.languages).sort((a, b) => a.name.localeCompare(b.name));
}
