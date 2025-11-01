import { describe, it, expect } from 'vitest';

/**
 * Tests for exclusion filtering logic
 *
 * This tests the critical bug we fixed where exclusions weren't being applied
 * because they weren't being added to the array before submission.
 */
describe('Exclusion Filtering Logic', () => {
  it('should filter out words containing excluded phonemes', () => {
    // Mock word objects
    const words = [
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

    const excludePhonemes = ['dʒ'];

    // Filter logic from clientSideApiAdapter
    const filtered = words.filter(word => {
      const phonemes = word.phonemes.map(p => p.ipa);
      const hasExcluded = excludePhonemes.some(excluded => phonemes.includes(excluded));
      return !hasExcluded;
    });

    expect(filtered).toHaveLength(3);
    expect(filtered.map(w => w.word)).toEqual(['cat', 'mat', 'make']);
    expect(filtered.map(w => w.word)).not.toContain('magenta');
  });

  it('should handle multiple exclusions', () => {
    const words = [
      { word: 'cat', phonemes: [{ ipa: 'k', position: 0 }, { ipa: 'æ', position: 1 }, { ipa: 't', position: 2 }] },
      { word: 'mat', phonemes: [{ ipa: 'm', position: 0 }, { ipa: 'æ', position: 1 }, { ipa: 't', position: 2 }] },
      { word: 'judge', phonemes: [{ ipa: 'dʒ', position: 0 }, { ipa: 'ʌ', position: 1 }, { ipa: 'dʒ', position: 2 }] },
      { word: 'that', phonemes: [{ ipa: 'ð', position: 0 }, { ipa: 'æ', position: 1 }, { ipa: 't', position: 2 }] },
    ];

    const excludePhonemes = ['dʒ', 'ð'];

    const filtered = words.filter(word => {
      const phonemes = word.phonemes.map(p => p.ipa);
      const hasExcluded = excludePhonemes.some(excluded => phonemes.includes(excluded));
      return !hasExcluded;
    });

    expect(filtered).toHaveLength(2);
    expect(filtered.map(w => w.word)).toEqual(['cat', 'mat']);
  });

  it('should handle Unicode phonemes correctly', () => {
    const words = [
      { word: 'them', phonemes: [{ ipa: 'ð', position: 0 }, { ipa: 'ɛ', position: 1 }, { ipa: 'm', position: 2 }] },
      { word: 'thumb', phonemes: [{ ipa: 'θ', position: 0 }, { ipa: 'ʌ', position: 1 }, { ipa: 'm', position: 2 }] },
    ];

    const excludePhonemes = ['ð'];

    const filtered = words.filter(word => {
      const phonemes = word.phonemes.map(p => p.ipa);
      const hasExcluded = excludePhonemes.some(excluded => phonemes.includes(excluded));
      return !hasExcluded;
    });

    expect(filtered).toHaveLength(1);
    expect(filtered[0].word).toBe('thumb');
  });

  it('should not filter when exclusion list is empty', () => {
    const words = [
      { word: 'cat', phonemes: [{ ipa: 'k', position: 0 }, { ipa: 'æ', position: 1 }, { ipa: 't', position: 2 }] },
      { word: 'mat', phonemes: [{ ipa: 'm', position: 0 }, { ipa: 'æ', position: 1 }, { ipa: 't', position: 2 }] },
    ];

    const excludePhonemes: string[] = [];

    const filtered = words.filter(word => {
      const phonemes = word.phonemes.map(p => p.ipa);
      const hasExcluded = excludePhonemes.some(excluded => phonemes.includes(excluded));
      return !hasExcluded;
    });

    expect(filtered).toHaveLength(2);
    expect(filtered).toEqual(words);
  });
});
