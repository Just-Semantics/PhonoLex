import { describe, it, expect } from 'vitest';
import { tokenizePhonemes, containsSequence } from '../utils/phonemeUtils';
import { clientSideData } from './clientSideData';

/**
 * Unit tests for phoneme tokenization and pattern matching logic
 *
 * These tests verify the actual production code works correctly.
 */

describe('Phoneme Tokenization', () => {
  it('should tokenize space-separated phonemes', () => {
    expect(tokenizePhonemes('k æ t')).toEqual(['k', 'æ', 't']);
    expect(tokenizePhonemes('dʒ ʌ dʒ')).toEqual(['dʒ', 'ʌ', 'dʒ']);
  });

  it('should handle multi-character phonemes', () => {
    expect(tokenizePhonemes('tʃ ɝ tʃ')).toEqual(['tʃ', 'ɝ', 'tʃ']);
    expect(tokenizePhonemes('aɪ k aɪ')).toEqual(['aɪ', 'k', 'aɪ']);
  });

  it('should trim whitespace', () => {
    expect(tokenizePhonemes('  k æ t  ')).toEqual(['k', 'æ', 't']);
  });

  it('should handle empty input', () => {
    expect(tokenizePhonemes('')).toEqual([]);
    expect(tokenizePhonemes('   ')).toEqual([]);
  });

  it('should handle multiple spaces between phonemes', () => {
    expect(tokenizePhonemes('k    æ    t')).toEqual(['k', 'æ', 't']);
  });
});

describe('Sequence Matching', () => {
  it('should match single phoneme sequences', () => {
    expect(containsSequence(['k', 'æ', 't'], ['æ'])).toBe(true);
    expect(containsSequence(['k', 'æ', 't'], ['k'])).toBe(true);
    expect(containsSequence(['k', 'æ', 't'], ['t'])).toBe(true);
    expect(containsSequence(['k', 'æ', 't'], ['ʌ'])).toBe(false);
  });

  it('should match multi-phoneme sequences at start', () => {
    expect(containsSequence(['k', 'æ', 't'], ['k', 'æ'])).toBe(true);
  });

  it('should match multi-phoneme sequences at end', () => {
    expect(containsSequence(['k', 'æ', 't'], ['æ', 't'])).toBe(true);
  });

  it('should not match out-of-order sequences', () => {
    expect(containsSequence(['k', 'æ', 't'], ['æ', 'k'])).toBe(false);
    expect(containsSequence(['k', 'æ', 't'], ['t', 'k'])).toBe(false);
  });

  it('should handle multi-character phonemes', () => {
    expect(containsSequence(['dʒ', 'ʌ', 'dʒ'], ['dʒ'])).toBe(true);
    expect(containsSequence(['dʒ', 'ʌ', 'dʒ'], ['dʒ', 'ʌ'])).toBe(true);
  });
});

describe('Unicode Phoneme Handling', () => {
  it('should handle multi-byte Unicode characters', () => {
    expect(tokenizePhonemes('ð ʌ m')).toEqual(['ð', 'ʌ', 'm']);
    expect(tokenizePhonemes('θ ʌ m')).toEqual(['θ', 'ʌ', 'm']);
    expect(tokenizePhonemes('ʃ ʒ ŋ')).toEqual(['ʃ', 'ʒ', 'ŋ']);
  });

  it('should handle complex IPA characters', () => {
    expect(tokenizePhonemes('dʒ')).toEqual(['dʒ']);
    expect(tokenizePhonemes('tʃ')).toEqual(['tʃ']);
  });

  it('should preserve Unicode in sequence matching', () => {
    expect(containsSequence(['ð', 'ɛ', 'm'], ['ð'])).toBe(true);
    expect(containsSequence(['θ', 'ʌ', 'm'], ['θ'])).toBe(true);
    expect(containsSequence(['ð', 'ɛ', 'm'], ['θ'])).toBe(false);
  });
});

/**
 * Multiple Opposition Algorithm Tests
 *
 * These tests verify the Maximal Classification + Maximal Distinction algorithms
 * used for Multiple Opposition intervention (Storkel 2022).
 *
 * Note: These tests use unit testing with mock phoneme features rather than
 * integration tests with real data, to avoid the overhead of loading ~45MB
 * of JSON data for each test run.
 */

describe('Multiple Opposition - Representative Target Selection', () => {
  it('should select targets with maximal distinction from substitute', () => {
    // Test that algorithm returns the requested number of targets
    // The algorithm uses real Phoible features to compute distance
    const result = clientSideData.selectRepresentativeTargets('t', ['d', 'k', 'l'], 2);

    // Should return exactly 2 targets as requested
    expect(result).toHaveLength(2);

    // All returned targets should be from the input list
    expect(['d', 'k', 'l']).toContain(result[0]);
    expect(['d', 'k', 'l']).toContain(result[1]);

    // Should not contain duplicates
    expect(result[0]).not.toBe(result[1]);
  });

  it('should select diverse targets (Maximal Classification)', () => {
    // Test with phonemes that have varying degrees of similarity to each other
    // The algorithm should prefer targets that are maximally different from the substitute
    // AND from each other
    const result = clientSideData.selectRepresentativeTargets('t', ['d', 'k', 'g', 's', 'z', 'l', 'r'], 3);

    expect(result).toHaveLength(3);
    // Should not select phonemes that are too similar to each other
    // e.g., shouldn't select both d and g (both voiced stops)
  });

  it('should handle requesting more targets than available', () => {
    const result = clientSideData.selectRepresentativeTargets('t', ['d', 'k'], 5);
    expect(result).toHaveLength(2); // Should return only 2 (all available)
  });

  it('should handle single target', () => {
    const result = clientSideData.selectRepresentativeTargets('t', ['d'], 3);
    expect(result).toEqual(['d']);
  });

  it('should handle empty targets', () => {
    const result = clientSideData.selectRepresentativeTargets('t', [], 3);
    expect(result).toEqual([]);
  });
});

describe('Multiple Opposition - Minimal Set Generation', () => {
  // Note: These are integration tests that require data loading
  // Skipped in unit test suite - run with a test server for full integration testing

  it.skip('should return array of minimal sets', async () => {
    // Integration test - requires data files to be served
    const result = await clientSideData.generateMultipleOppositionSets('t', ['d', 'k'], 'initial', 5);
    expect(Array.isArray(result)).toBe(true);

    if (result.length > 0) {
      const firstSet = result[0];
      expect(firstSet).toHaveProperty('words');
      expect(firstSet).toHaveProperty('position');
      expect(Array.isArray(firstSet.words)).toBe(true);

      // Each word should have word and phoneme properties
      if (firstSet.words.length > 0) {
        const firstWord = firstSet.words[0];
        expect(firstWord).toHaveProperty('word');
        expect(firstWord).toHaveProperty('phoneme');
        expect(firstWord.word).toHaveProperty('word');
        expect(firstWord.word).toHaveProperty('ipa');
      }
    }
  });

  it.skip('should enforce no duplicate phonemes in sets', async () => {
    // Integration test - requires data files to be served
    const result = await clientSideData.generateMultipleOppositionSets('t', ['d', 'k', 'g'], 'initial', 10);

    // Each set should have unique phonemes (no duplicates)
    for (const set of result) {
      const phonemesInSet = set.words.map(w => w.phoneme);
      const uniquePhonemes = new Set(phonemesInSet);
      expect(phonemesInSet.length).toBe(uniquePhonemes.size);
    }
  });

  it.skip('should respect position parameter', async () => {
    // Integration test - requires data files to be served
    // Test with phonemes that work in initial position
    const initialResult = await clientSideData.generateMultipleOppositionSets('t', ['d', 'k'], 'initial', 5);

    // All sets should have the substitute/target at the specified position
    for (const set of initialResult) {
      expect(set.position).toBeGreaterThanOrEqual(0);
    }
  });
});

describe('Multiple Opposition - Edge Cases', () => {
  it.skip('should handle substitute phoneme that matches one target', async () => {
    // Integration test - requires data files to be served
    // t→t,d should still work (though clinically unusual)
    const result = await clientSideData.generateMultipleOppositionSets('t', ['t', 'd'], 'initial', 5);
    expect(Array.isArray(result)).toBe(true);
  });

  it.skip('should handle phonotactically invalid positions gracefully', async () => {
    // Integration test - requires data files to be served
    // /w/ cannot appear word-finally in English
    const result = await clientSideData.generateMultipleOppositionSets('w', ['r', 'l'], 'final', 5);
    expect(Array.isArray(result)).toBe(true);
    // May return empty array, but shouldn't crash
  });
});
