/**
 * IPA Input Validation Utilities
 *
 * Detects when users accidentally type ASCII characters instead of IPA symbols
 * and provides helpful warnings to use the phoneme picker.
 */

/**
 * Common IPA phonemes used in English phonology
 * This list is not exhaustive but covers the most common cases
 */
const VALID_IPA_PHONEMES = new Set([
  // Consonants
  'p', 'b', 't', 'd', 'k', 'g', 'f', 'v', 'θ', 'ð', 's', 'z', 'ʃ', 'ʒ',
  'h', 'm', 'n', 'ŋ', 'l', 'r', 'w', 'j', 'ɹ', 'tʃ', 'dʒ',

  // Vowels (monophthongs)
  'i', 'ɪ', 'e', 'ɛ', 'æ', 'ɑ', 'ɔ', 'o', 'ʊ', 'u', 'ʌ', 'ə', 'ɚ', 'ɝ',

  // Diphthongs
  'aɪ', 'aʊ', 'ɔɪ', 'eɪ', 'oʊ',

  // Stress markers (sometimes included)
  'ˈ', 'ˌ',
]);

/**
 * ASCII characters that look like IPA but aren't
 * Maps ASCII -> IPA for common mistakes
 */
const ASCII_TO_IPA_SUGGESTIONS: Record<string, string> = {
  // Common consonant confusions
  'th': 'θ or ð',
  'sh': 'ʃ',
  'zh': 'ʒ',
  'ng': 'ŋ',
  'ch': 'tʃ',

  // Common vowel confusions (single letters that might be wrong)
  'a': 'æ, ɑ, or ə',
  'e': 'ɛ or ə',
  'i': 'ɪ or i',
  'o': 'ɔ or o',
  'u': 'ʌ, ʊ, or u',
};

/**
 * Checks if a string contains only valid IPA characters
 */
export function isValidIPA(input: string): boolean {
  if (!input || input.trim() === '') return true;

  // Remove whitespace and stress markers for validation
  const cleaned = input.trim().replace(/[ˈˌ]/g, '');

  // Check if it's a known valid IPA phoneme
  if (VALID_IPA_PHONEMES.has(cleaned)) return true;

  // Check for common multi-character phonemes (diphthongs, affricates)
  const multiChar = ['tʃ', 'dʒ', 'aɪ', 'aʊ', 'ɔɪ', 'eɪ', 'oʊ'];
  if (multiChar.includes(cleaned)) return true;

  // If it contains only ASCII letters (no IPA characters), it's likely wrong
  const hasOnlyASCII = /^[a-zA-Z]+$/.test(cleaned);
  if (hasOnlyASCII && cleaned.length > 0) {
    // Single ASCII letters might be valid (p, t, k, etc.)
    // But longer sequences are likely mistakes
    return cleaned.length === 1 && /[ptbdkgfvszhmнlrwj]/.test(cleaned);
  }

  // Otherwise, assume it's valid (user might be entering valid IPA we don't recognize)
  return true;
}

/**
 * Gets a helpful suggestion for correcting invalid IPA input
 */
export function getIPASuggestion(input: string): string | null {
  if (!input || input.trim() === '') return null;

  const cleaned = input.trim().toLowerCase();

  // Check for common multi-character mistakes
  for (const [ascii, ipa] of Object.entries(ASCII_TO_IPA_SUGGESTIONS)) {
    if (cleaned === ascii || cleaned.includes(ascii)) {
      return `Did you mean "${ipa}"? Use the keyboard icon to select IPA symbols.`;
    }
  }

  // Check if it's all ASCII (likely a mistake)
  if (/^[a-zA-Z]+$/.test(cleaned) && cleaned.length > 1) {
    return `This looks like ASCII text. Use the keyboard icon (⌨️) to select IPA phonemes.`;
  }

  return null;
}

/**
 * Validates phoneme input and returns an error message if invalid
 */
export interface IPAValidationResult {
  isValid: boolean;
  suggestion?: string;
}

export function validatePhonemeInput(input: string): IPAValidationResult {
  const isValid = isValidIPA(input);

  if (!isValid) {
    const suggestion = getIPASuggestion(input);
    return { isValid: false, suggestion: suggestion || undefined };
  }

  return { isValid: true };
}
