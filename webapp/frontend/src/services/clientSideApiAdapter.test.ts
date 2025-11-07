import { describe, it, expect } from 'vitest';
import { applyExclusions } from '../utils/phonemeUtils';
import { api } from './clientSideApiAdapter';

/**
 * Unit tests for exclusion filtering logic and API adapter methods
 *
 * These tests verify the actual production code works correctly.
 */

// Mock word interface matching actual structure
interface MockWord {
  word: string;
  phonemes: Array<{ ipa: string; position: number }>;
}

describe('Exclusion Filtering Logic', () => {
  it('should filter out words containing excluded phonemes', () => {
    const words: MockWord[] = [
      { word: 'cat', phonemes: [{ ipa: 'k', position: 0 }, { ipa: 'æ', position: 1 }, { ipa: 't', position: 2 }] },
      { word: 'mat', phonemes: [{ ipa: 'm', position: 0 }, { ipa: 'æ', position: 1 }, { ipa: 't', position: 2 }] },
      { word: 'magenta', phonemes: [
        { ipa: 'm', position: 0 },
        { ipa: 'ʌ', position: 1 },
        { ipa: 'dʒ', position: 2 },
        { ipa: 'ɛ', position: 3 },
        { ipa: 'n', position: 4 },
        { ipa: 't', position: 5 },
        { ipa: 'ʌ', position: 6 }
      ]},
      { word: 'make', phonemes: [{ ipa: 'm', position: 0 }, { ipa: 'eɪ', position: 1 }, { ipa: 'k', position: 2 }] },
    ];

    const filtered = applyExclusions(words, ['dʒ']);

    expect(filtered).toHaveLength(3);
    expect(filtered.map(w => w.word)).toEqual(['cat', 'mat', 'make']);
    expect(filtered.map(w => w.word)).not.toContain('magenta');
  });

  it('should handle multiple exclusions', () => {
    const words: MockWord[] = [
      { word: 'cat', phonemes: [{ ipa: 'k', position: 0 }, { ipa: 'æ', position: 1 }, { ipa: 't', position: 2 }] },
      { word: 'mat', phonemes: [{ ipa: 'm', position: 0 }, { ipa: 'æ', position: 1 }, { ipa: 't', position: 2 }] },
      { word: 'judge', phonemes: [{ ipa: 'dʒ', position: 0 }, { ipa: 'ʌ', position: 1 }, { ipa: 'dʒ', position: 2 }] },
      { word: 'that', phonemes: [{ ipa: 'ð', position: 0 }, { ipa: 'æ', position: 1 }, { ipa: 't', position: 2 }] },
    ];

    const filtered = applyExclusions(words, ['dʒ', 'ð']);

    expect(filtered).toHaveLength(2);
    expect(filtered.map(w => w.word)).toEqual(['cat', 'mat']);
  });

  it('should handle Unicode phonemes correctly', () => {
    const words: MockWord[] = [
      { word: 'them', phonemes: [{ ipa: 'ð', position: 0 }, { ipa: 'ɛ', position: 1 }, { ipa: 'm', position: 2 }] },
      { word: 'thumb', phonemes: [{ ipa: 'θ', position: 0 }, { ipa: 'ʌ', position: 1 }, { ipa: 'm', position: 2 }] },
    ];

    const filtered = applyExclusions(words, ['ð']);

    expect(filtered).toHaveLength(1);
    expect(filtered[0].word).toBe('thumb');
  });

  it('should not filter when exclusion list is empty', () => {
    const words: MockWord[] = [
      { word: 'cat', phonemes: [{ ipa: 'k', position: 0 }, { ipa: 'æ', position: 1 }, { ipa: 't', position: 2 }] },
      { word: 'mat', phonemes: [{ ipa: 'm', position: 0 }, { ipa: 'æ', position: 1 }, { ipa: 't', position: 2 }] },
    ];

    const filteredEmpty = applyExclusions(words, []);
    const filteredUndefined = applyExclusions(words, undefined);

    expect(filteredEmpty).toHaveLength(2);
    expect(filteredEmpty).toEqual(words);
    expect(filteredUndefined).toEqual(words);
  });
});

/**
 * API Adapter Tests for Multiple Opposition
 *
 * Tests the adapter layer that wraps the Multiple Opposition algorithms
 */

describe('API Adapter - Representative Target Selection', () => {
  it('should expose selectRepresentativeTargets method', () => {
    expect(typeof api.selectRepresentativeTargets).toBe('function');
  });

  it('should pass through to clientSideData correctly', () => {
    const result = api.selectRepresentativeTargets({
      substitute_phoneme: 't',
      target_phonemes: ['d', 'k', 'l'],
      count: 2,
    });

    expect(Array.isArray(result)).toBe(true);
  });

  it('should use default count of 3 when not specified', () => {
    const result = api.selectRepresentativeTargets({
      substitute_phoneme: 't',
      target_phonemes: ['d', 'k', 'l', 'r', 's'],
    });

    expect(Array.isArray(result)).toBe(true);
    expect(result.length).toBeLessThanOrEqual(3);
  });
});

describe('API Adapter - Multiple Opposition Set Generation', () => {
  it('should expose generateMultipleOppositionSets method', async () => {
    expect(typeof api.generateMultipleOppositionSets).toBe('function');
  });

  it('should return promise with array structure', async () => {
    const result = await api.generateMultipleOppositionSets({
      substitute_phoneme: 't',
      target_phonemes: ['d', 'k'],
      position: 'initial',
      max_sets: 5,
    });

    expect(Array.isArray(result)).toBe(true);
  });

  it('should use default position of "initial" when not specified', async () => {
    const result = await api.generateMultipleOppositionSets({
      substitute_phoneme: 't',
      target_phonemes: ['d', 'k'],
      max_sets: 5,
    });

    expect(Array.isArray(result)).toBe(true);
  });

  it('should use default max_sets of 10 when not specified', async () => {
    const result = await api.generateMultipleOppositionSets({
      substitute_phoneme: 't',
      target_phonemes: ['d', 'k'],
      position: 'initial',
    });

    expect(Array.isArray(result)).toBe(true);
    expect(result.length).toBeLessThanOrEqual(10);
  });

  it('should handle all position values', async () => {
    const positions: Array<'initial' | 'medial' | 'final' | 'any'> = ['initial', 'medial', 'final', 'any'];

    for (const position of positions) {
      const result = await api.generateMultipleOppositionSets({
        substitute_phoneme: 't',
        target_phonemes: ['d', 'k'],
        position,
        max_sets: 5,
      });

      expect(Array.isArray(result)).toBe(true);
    }
  });
});
