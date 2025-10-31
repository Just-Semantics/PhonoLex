/**
 * Search Component
 *
 * Three search modes:
 * 1. Word Lookup - Get detailed info about a specific word
 * 2. Similarity Search - Find words similar to a target
 * 3. Phoneme Search - Search by phonological features
 */

import React, { useState } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Stack,
  Alert,
  CircularProgress,
  Divider,
  Paper,
} from '@mui/material';
import {
  Search as SearchIcon,
  Clear as ClearIcon,
} from '@mui/icons-material';
import api from '../services/phonolexApi';
import type { Word, Phoneme, SimilarWord } from '../services/phonolexApi';
import WordResultsDisplay from './WordResultsDisplay';

const Search: React.FC = () => {
  // Search mode
  const [mode, setMode] = useState<'word' | 'similarity' | 'phoneme'>('similarity');

  // Word lookup state
  const [wordQuery, setWordQuery] = useState('');
  const [wordResult, setWordResult] = useState<Word | null>(null);

  // Similarity search state
  const [similarityQuery, setSimilarityQuery] = useState('cat');
  const [similarityThreshold, setSimilarityThreshold] = useState(0.85);
  const [similarityResults, setSimilarityResults] = useState<SimilarWord[] | null>(null);

  // Phoneme search state
  const [phonemeQuery, setPhonemeQuery] = useState('');
  const [phonemeFeatures, setPhonemeFeatures] = useState<{
    segmentClass?: string;
    consonantal?: string;
    voice?: string;
  }>({});
  const [phonemeResults, setPhonemeResults] = useState<Phoneme[] | null>(null);

  // Loading/error state
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Word lookup
  const handleWordLookup = async () => {
    if (!wordQuery.trim()) return;

    setLoading(true);
    setError(null);
    try {
      const result = await api.getWord(wordQuery.trim());
      setWordResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Word not found');
      setWordResult(null);
    } finally {
      setLoading(false);
    }
  };

  // Similarity search
  const handleSimilaritySearch = async () => {
    if (!similarityQuery.trim()) return;

    setLoading(true);
    setError(null);
    try {
      const results = await api.findSimilarWords(
        similarityQuery.trim(),
        { threshold: similarityThreshold, limit: 100 }
      );
      setSimilarityResults(results);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed');
      setSimilarityResults(null);
    } finally {
      setLoading(false);
    }
  };

  // Phoneme search
  const handlePhonemeSearch = async () => {
    setLoading(true);
    setError(null);
    try {
      const features: Record<string, string> = {};
      if (phonemeFeatures.segmentClass) features.segment_class = phonemeFeatures.segmentClass;
      if (phonemeFeatures.consonantal) features.consonantal = phonemeFeatures.consonantal;
      if (phonemeFeatures.voice) features.voice = phonemeFeatures.voice;

      // TODO: Implement phoneme search
      const results = await api.searchPhonemes();
      setPhonemeResults(results);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed');
      setPhonemeResults(null);
    } finally {
      setLoading(false);
    }
  };

  // Clear all
  const handleClear = () => {
    setWordQuery('');
    setSimilarityQuery('cat');
    setPhonemeQuery('');
    setPhonemeFeatures({});
    setWordResult(null);
    setSimilarityResults(null);
    setPhonemeResults(null);
    setError(null);
  };

  return (
    <Box>
      {/* Mode Selector */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Stack direction="row" spacing={2} alignItems="center">
          <Typography variant="subtitle1" fontWeight={500}>
            Search Mode:
          </Typography>
          <Button
            variant={mode === 'similarity' ? 'contained' : 'outlined'}
            onClick={() => setMode('similarity')}
          >
            Similarity Search
          </Button>
          <Button
            variant={mode === 'word' ? 'contained' : 'outlined'}
            onClick={() => setMode('word')}
          >
            Word Lookup
          </Button>
          <Button
            variant={mode === 'phoneme' ? 'contained' : 'outlined'}
            onClick={() => setMode('phoneme')}
          >
            Phoneme Search
          </Button>
        </Stack>
      </Paper>

      {/* Similarity Search */}
      {mode === 'similarity' && (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Find Similar Words
            </Typography>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Uses hierarchical phonological similarity (discriminates anagrams)
            </Typography>

            <Stack spacing={3} sx={{ mt: 3 }}>
              <TextField
                label="Target Word"
                value={similarityQuery}
                onChange={(e) => setSimilarityQuery(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSimilaritySearch()}
                placeholder="e.g., cat, dog, house"
                fullWidth
              />

              <Box>
                <Typography gutterBottom>
                  Similarity Threshold: {similarityThreshold.toFixed(2)}
                </Typography>
                <Slider
                  value={similarityThreshold}
                  onChange={(_, value) => setSimilarityThreshold(value as number)}
                  min={0.5}
                  max={1.0}
                  step={0.05}
                  marks={[
                    { value: 0.5, label: '0.5' },
                    { value: 0.75, label: '0.75' },
                    { value: 0.85, label: '0.85' },
                    { value: 1.0, label: '1.0' },
                  ]}
                />
                <Typography variant="caption" color="text.secondary">
                  Lower = more diverse results, Higher = more similar results
                </Typography>
              </Box>

              <Stack direction="row" spacing={2}>
                <Button
                  variant="contained"
                  startIcon={<SearchIcon />}
                  onClick={handleSimilaritySearch}
                  disabled={loading}
                  fullWidth
                >
                  Search
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<ClearIcon />}
                  onClick={handleClear}
                >
                  Clear
                </Button>
              </Stack>
            </Stack>
          </CardContent>
        </Card>
      )}

      {/* Word Lookup */}
      {mode === 'word' && (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Word Lookup
            </Typography>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Get detailed phonological information about a specific word
            </Typography>

            <Stack spacing={3} sx={{ mt: 3 }}>
              <TextField
                label="Word"
                value={wordQuery}
                onChange={(e) => setWordQuery(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleWordLookup()}
                placeholder="e.g., elephant"
                fullWidth
              />

              <Stack direction="row" spacing={2}>
                <Button
                  variant="contained"
                  startIcon={<SearchIcon />}
                  onClick={handleWordLookup}
                  disabled={loading}
                  fullWidth
                >
                  Lookup
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<ClearIcon />}
                  onClick={handleClear}
                >
                  Clear
                </Button>
              </Stack>
            </Stack>
          </CardContent>
        </Card>
      )}

      {/* Phoneme Search */}
      {mode === 'phoneme' && (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Phoneme Search
            </Typography>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Search phonemes by IPA symbol or distinctive features
            </Typography>

            <Stack spacing={3} sx={{ mt: 3 }}>
              <TextField
                label="IPA Symbol"
                value={phonemeQuery}
                onChange={(e) => setPhonemeQuery(e.target.value)}
                placeholder="e.g., t, k, ʃ"
                fullWidth
              />

              <Grid container spacing={2}>
                <Grid item xs={12} sm={4}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Segment Class</InputLabel>
                    <Select
                      value={phonemeFeatures.segmentClass || ''}
                      label="Segment Class"
                      onChange={(e) =>
                        setPhonemeFeatures({
                          ...phonemeFeatures,
                          segmentClass: e.target.value || undefined,
                        })
                      }
                    >
                      <MenuItem value="">Any</MenuItem>
                      <MenuItem value="consonant">Consonant</MenuItem>
                      <MenuItem value="vowel">Vowel</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>

                <Grid item xs={12} sm={4}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Consonantal</InputLabel>
                    <Select
                      value={phonemeFeatures.consonantal || ''}
                      label="Consonantal"
                      onChange={(e) =>
                        setPhonemeFeatures({
                          ...phonemeFeatures,
                          consonantal: e.target.value || undefined,
                        })
                      }
                    >
                      <MenuItem value="">Any</MenuItem>
                      <MenuItem value="+">+</MenuItem>
                      <MenuItem value="-">-</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>

                <Grid item xs={12} sm={4}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Voice</InputLabel>
                    <Select
                      value={phonemeFeatures.voice || ''}
                      label="Voice"
                      onChange={(e) =>
                        setPhonemeFeatures({
                          ...phonemeFeatures,
                          voice: e.target.value || undefined,
                        })
                      }
                    >
                      <MenuItem value="">Any</MenuItem>
                      <MenuItem value="+">+</MenuItem>
                      <MenuItem value="-">-</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
              </Grid>

              <Stack direction="row" spacing={2}>
                <Button
                  variant="contained"
                  startIcon={<SearchIcon />}
                  onClick={handlePhonemeSearch}
                  disabled={loading}
                  fullWidth
                >
                  Search
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<ClearIcon />}
                  onClick={handleClear}
                >
                  Clear
                </Button>
              </Stack>
            </Stack>
          </CardContent>
        </Card>
      )}

      {/* Status Messages */}
      <Box sx={{ mt: 3 }}>
        {loading && (
          <Alert severity="info" icon={<CircularProgress size={20} />}>
            Searching...
          </Alert>
        )}
        {error && <Alert severity="error">{error}</Alert>}
      </Box>

      {/* Word Lookup Result */}
      {wordResult && !loading && (
        <Paper sx={{ p: 3, mt: 3 }}>
          <Typography variant="h5" gutterBottom>
            {wordResult.word}
          </Typography>
          <Typography variant="h6" fontFamily="monospace" color="text.secondary" gutterBottom>
            /{wordResult.ipa}/
          </Typography>

          <Divider sx={{ my: 2 }} />

          <Grid container spacing={2}>
            <Grid item xs={6} sm={3}>
              <Typography variant="caption" color="text.secondary">
                Syllables
              </Typography>
              <Typography variant="h6">{wordResult.syllable_count}</Typography>
            </Grid>
            <Grid item xs={6} sm={3}>
              <Typography variant="caption" color="text.secondary">
                WCM
              </Typography>
              <Typography variant="h6">{wordResult.wcm_score?.toFixed(1) || 'N/A'}</Typography>
            </Grid>
            <Grid item xs={6} sm={3}>
              <Typography variant="caption" color="text.secondary">
                Complexity
              </Typography>
              <Typography variant="h6">{wordResult.complexity || 'N/A'}</Typography>
            </Grid>
            <Grid item xs={6} sm={3}>
              <Typography variant="caption" color="text.secondary">
                AoA
              </Typography>
              <Typography variant="h6">{wordResult.aoa?.toFixed(1) || 'N/A'}</Typography>
            </Grid>
          </Grid>

          {wordResult.phonemes && wordResult.phonemes.length > 0 && (
            <Box sx={{ mt: 3 }}>
              <Typography variant="subtitle2" gutterBottom>
                Phoneme Breakdown
              </Typography>
              <Stack direction="row" spacing={1} flexWrap="wrap">
                {wordResult.phonemes.map((p, idx) => (
                  <Chip key={idx} label={p.ipa} size="small" />
                ))}
              </Stack>
            </Box>
          )}
        </Paper>
      )}

      {/* Similarity Results */}
      {similarityResults && !loading && (
        <Box sx={{ mt: 3 }}>
          <WordResultsDisplay results={similarityResults} showSimilarity />
        </Box>
      )}

      {/* Phoneme Results */}
      {phonemeResults && !loading && (
        <Paper sx={{ p: 3, mt: 3 }}>
          <Typography variant="h6" gutterBottom>
            {phonemeResults.length} Phonemes Found
          </Typography>
          <Grid container spacing={2}>
            {phonemeResults.map((phoneme) => (
              <Grid item xs={12} sm={6} md={4} key={phoneme.phoneme_id}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="h4" gutterBottom>
                      {phoneme.ipa}
                    </Typography>
                    <Chip
                      label={phoneme.segment_class}
                      size="small"
                      color="primary"
                      sx={{ mb: 1 }}
                    />
                    {phoneme.features && (
                      <Box sx={{ mt: 2 }}>
                        <Typography variant="caption" color="text.secondary">
                          Key Features:
                        </Typography>
                        <Stack direction="row" spacing={0.5} flexWrap="wrap" sx={{ mt: 0.5 }}>
                          {Object.entries(phoneme.features)
                            .filter(([_, value]) => value === '+')
                            .slice(0, 5)
                            .map(([key]) => (
                              <Chip key={key} label={key} size="small" variant="outlined" />
                            ))}
                        </Stack>
                      </Box>
                    )}
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Paper>
      )}
    </Box>
  );
};

export default Search;
