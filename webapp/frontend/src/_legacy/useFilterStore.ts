/**
 * Zustand store for word filtering state
 *
 * Manages:
 * - Pattern builder state (phoneme constraints)
 * - Filter results
 * - Loading states
 */

import { create } from 'zustand';
import type { PhonemeConstraint, WordFilter, WordResult } from '@types/phonology';
import { api } from '@services/api';

interface FilterState {
  // Pattern builder
  pattern: PhonemeConstraint[];
  minSyllables?: number;
  maxSyllables?: number;
  minPhonemes?: number;
  maxPhonemes?: number;

  // Results
  results: WordResult[];
  resultCount: number;
  isLoading: boolean;
  error: string | null;
  maxResults: number; // Limit to prevent loading too many

  // Available phonemes (loaded on init)
  availableVowels: string[];
  availableConsonants: string[];
  isPhonemesLoaded: boolean;

  // Actions
  addPosition: (position: number) => void;
  addContains: () => void;  // Add "contains" constraint (no position)
  removeConstraint: (index: number) => void;
  updateConstraint: (index: number, constraint: Partial<PhonemeConstraint>) => void;
  setSyllableRange: (min?: number, max?: number) => void;
  setPhonemeRange: (min?: number, max?: number) => void;
  executeFilter: () => Promise<void>;
  clearFilter: () => void;
  loadPhonemes: () => Promise<void>;
}

export const useFilterStore = create<FilterState>((set, get) => ({
  // Initial state
  pattern: [],
  minSyllables: undefined,
  maxSyllables: undefined,
  minPhonemes: undefined,
  maxPhonemes: undefined,
  results: [],
  resultCount: 0,
  isLoading: false,
  error: null,
  maxResults: 500, // Only load first 500 results for performance
  availableVowels: [],
  availableConsonants: [],
  isPhonemesLoaded: false,

  // Actions
  addPosition: (position) => {
    const { pattern } = get();

    // Add new constraint at position
    const newConstraint: PhonemeConstraint = {
      position,
      phoneme_type: undefined,
      allowed_phonemes: undefined,
      required_features: undefined,
    };

    const newPattern = [...pattern, newConstraint].sort((a, b) => {
      // Sort: position-based first, then contains constraints
      if (a.position === undefined || a.position === null) return 1;
      if (b.position === undefined || b.position === null) return -1;
      return a.position - b.position;
    });
    set({ pattern: newPattern });

    // Auto-execute filter
    get().executeFilter();
  },

  addContains: () => {
    const { pattern } = get();

    // Add "contains" constraint (no position)
    const newConstraint: PhonemeConstraint = {
      position: null,
      phoneme_type: undefined,
      allowed_phonemes: undefined,
      required_features: undefined,
    };

    set({ pattern: [...pattern, newConstraint] });

    // Auto-execute filter
    get().executeFilter();
  },

  removeConstraint: (index) => {
    const { pattern } = get();
    const newPattern = pattern.filter((_, i) => i !== index);
    set({ pattern: newPattern });

    // Auto-execute filter
    get().executeFilter();
  },

  updateConstraint: (index, updates) => {
    const { pattern } = get();
    const newPattern = pattern.map((c, i) =>
      i === index ? { ...c, ...updates } : c
    );
    set({ pattern: newPattern });

    // Auto-execute filter
    get().executeFilter();
  },

  setSyllableRange: (min, max) => {
    set({ minSyllables: min, maxSyllables: max });
    get().executeFilter();
  },

  setPhonemeRange: (min, max) => {
    set({ minPhonemes: min, maxPhonemes: max });
    get().executeFilter();
  },

  executeFilter: async () => {
    const state = get();

    // Build filter request
    const filter: WordFilter = {
      pattern: state.pattern,
      minSyllables: state.minSyllables,
      maxSyllables: state.maxSyllables,
      minPhonemes: state.minPhonemes,
      maxPhonemes: state.maxPhonemes,
      limit: state.maxResults, // Send limit to backend
      offset: 0,
    };

    // Check if pattern has any actual constraints (not just empty positions)
    const hasPatternConstraints = filter.pattern.some(
      (c) => c.phoneme_type || c.allowed_phonemes || c.required_features
    );

    // If no meaningful constraints, don't filter (avoid loading all 125k words)
    if (
      !hasPatternConstraints &&
      !filter.minSyllables &&
      !filter.maxSyllables &&
      !filter.minPhonemes &&
      !filter.maxPhonemes
    ) {
      set({ results: [], resultCount: 0, error: null });
      return;
    }

    set({ isLoading: true, error: null });

    // DEBUG: Log what we're sending
    console.log('=== SENDING FILTER REQUEST ===');
    console.log('Pattern:', JSON.stringify(filter.pattern, null, 2));
    console.log('===============================');

    try {
      const response = await api.filterWords(filter);

      set({
        results: response.words, // Backend already limited results
        resultCount: response.count, // Full count for display
        isLoading: false,
      });
    } catch (error) {
      set({
        error: (error as Error).message,
        isLoading: false,
        results: [],
        resultCount: 0,
      });
    }
  },

  clearFilter: () => {
    set({
      pattern: [],
      minSyllables: undefined,
      maxSyllables: undefined,
      minPhonemes: undefined,
      maxPhonemes: undefined,
      results: [],
      resultCount: 0,
      error: null,
    });
  },

  loadPhonemes: async () => {
    try {
      const phonemes = await api.getPhonemes();
      set({
        availableVowels: phonemes.vowels,
        availableConsonants: phonemes.consonants,
        isPhonemesLoaded: true,
      });
    } catch (error) {
      console.error('Failed to load phonemes:', error);
      set({ error: (error as Error).message });
    }
  },
}));
