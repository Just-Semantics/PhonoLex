import { describe, it, expect } from 'vitest';
import { tokenizePhonemes, containsSequence } from '../utils/phonemeUtils';

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
