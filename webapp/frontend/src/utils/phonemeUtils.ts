/**
 * Phoneme utility functions for tokenization and sequence matching
 *
 * These are extracted for testing and reuse across the application.
 */

/**
 * Tokenize a phoneme string into individual phonemes
 * REQUIRES space-separated input (e.g., "k æ t" or "dʒ ʌ dʒ")
 */
export function tokenizePhonemes(input: string): string[] {
  const trimmed = input.trim();
  if (!trimmed) return [];

  // Space-separated tokens only - no greedy matching
  return trimmed.split(/\s+/).filter(p => p.length > 0);
}

/**
 * Check if a sequence of phonemes exists in an array
 */
export function containsSequence(haystack: string[], needle: string[]): boolean {
  for (let i = 0; i <= haystack.length - needle.length; i++) {
    if (JSON.stringify(haystack.slice(i, i + needle.length)) === JSON.stringify(needle)) {
      return true;
    }
  }
  return false;
}

/**
 * Apply exclusion filter to words
 */
export function applyExclusions<T extends { phonemes: Array<{ ipa: string }> }>(
  words: T[],
  excludePhonemes: string[] | undefined
): T[] {
  if (!excludePhonemes || excludePhonemes.length === 0) {
    return words;
  }

  return words.filter(word => {
    const phonemes = word.phonemes.map(p => p.ipa);
    const hasExcluded = excludePhonemes.some(excluded => phonemes.includes(excluded));
    return !hasExcluded;
  });
}
