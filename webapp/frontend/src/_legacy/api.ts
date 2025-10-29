/**
 * API client for PhonoLex backend
 */

import type {
  WordFilter,
  FilterResponse,
  PhonemeListResponse,
  PhonemeInfo,
  StatsResponse,
} from '@types/phonology';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

class ApiError extends Error {
  constructor(
    message: string,
    public status?: number,
    public response?: unknown
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

async function fetchApi<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;

  try {
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
      ...options,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new ApiError(
        errorData.detail || `HTTP ${response.status}: ${response.statusText}`,
        response.status,
        errorData
      );
    }

    return await response.json();
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }
    throw new ApiError(`Network error: ${(error as Error).message}`);
  }
}

export const api = {
  /**
   * Health check
   */
  async healthCheck() {
    return fetchApi<{ service: string; status: string; version: string }>('/');
  },

  /**
   * Get all phonemes (vowels and consonants)
   */
  async getPhonemes() {
    return fetchApi<PhonemeListResponse>('/api/phonemes');
  },

  /**
   * Get detailed info about a specific phoneme
   */
  async getPhonemeInfo(phoneme: string) {
    return fetchApi<PhonemeInfo>(`/api/phonemes/${encodeURIComponent(phoneme)}`);
  },

  /**
   * Find phonemes by Phoible features
   */
  async getPhonemesByFeatures(features: Record<string, string>) {
    return fetchApi<{ features: Record<string, string>; matching_phonemes: string[]; count: number }>(
      '/api/phonemes/by-features',
      {
        method: 'POST',
        body: JSON.stringify(features),
      }
    );
  },

  /**
   * Filter words by phonological pattern
   */
  async filterWords(filter: WordFilter) {
    return fetchApi<FilterResponse>('/api/filter', {
      method: 'POST',
      body: JSON.stringify(filter),
    });
  },

  /**
   * Get corpus statistics
   */
  async getStats() {
    return fetchApi<StatsResponse>('/api/stats');
  },
};

export { ApiError };
