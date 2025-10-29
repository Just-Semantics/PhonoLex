/**
 * Minimal Pairs Tool Component
 *
 * Generate word pairs differing by a single phoneme for discrimination therapy
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  TextField,
  Button,
  Stack,
  Alert,
  CircularProgress,
  IconButton,
  Typography,
  Slider,
} from '@mui/material';
import {
  PlayArrow as RunIcon,
  Keyboard as KeyboardIcon,
} from '@mui/icons-material';
import api from '../../services/phonolexApi';
import type { MinimalPair } from '../../services/phonolexApi';
import WordResultsDisplay from '../WordResultsDisplay';
import PhonemePickerDialog from '../PhonemePickerDialog';

const MinimalPairsTool: React.FC = () => {
  const [state, setState] = useState<{
    phoneme1: string;
    phoneme2: string;
  }>({
    phoneme1: 't',
    phoneme2: 'd',
  });

  // Granular filters
  const [filters, setFilters] = useState<{
    syllables: [number, number];
    wcm: [number, number];
    frequency: [number, number];
  }>({
    syllables: [1, 5],
    wcm: [0, 15],
    frequency: [0, 10],
  });

  // Database property ranges
  const [dbRanges, setDbRanges] = useState<Record<string, [number, number]>>({
    syllables: [1, 5],
    wcm: [0, 15],
    frequency: [0, 10],
  });

  const [results, setResults] = useState<MinimalPair[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [phonemePickerOpen, setPhonemePickerOpen] = useState(false);
  const [activePhonemeField, setActivePhonemeField] = useState<'phoneme1' | 'phoneme2'>('phoneme1');

  // Fetch property ranges from database on mount
  useEffect(() => {
    const fetchRanges = async () => {
      try {
        const ranges = await api.getPropertyRanges();
        setDbRanges(ranges);
        setFilters({
          syllables: ranges.syllables as [number, number],
          wcm: ranges.wcm as [number, number],
          frequency: ranges.frequency as [number, number],
        });
      } catch (error) {
        console.error('Failed to fetch property ranges:', error);
      }
    };
    fetchRanges();
  }, []);

  const handleGenerate = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.getMinimalPairs({
        phoneme1: state.phoneme1,
        phoneme2: state.phoneme2,
        min_syllables: filters.syllables[0],
        max_syllables: filters.syllables[1],
        min_wcm: filters.wcm[0],
        max_wcm: filters.wcm[1],
        min_frequency: filters.frequency[0],
        max_frequency: filters.frequency[1],
        limit: 50,
      });
      setResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      setResults(null);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setFilters({
      syllables: dbRanges.syllables as [number, number],
      wcm: dbRanges.wcm as [number, number],
      frequency: dbRanges.frequency as [number, number],
    });
    setResults(null);
    setError(null);
  };

  const handlePhonemeSelect = (phoneme: string) => {
    setState({ ...state, [activePhonemeField]: phoneme });
  };

  const openPhonemePicker = (field: 'phoneme1' | 'phoneme2') => {
    setActivePhonemeField(field);
    setPhonemePickerOpen(true);
  };

  return (
    <Box sx={{ p: { xs: 0, sm: 0 } }}>
      <Stack spacing={{ xs: 2, sm: 2 }}>
        <TextField
          label="Phoneme 1 (IPA)"
          value={state.phoneme1}
          onChange={(e) => setState({ ...state, phoneme1: e.target.value })}
          size="small"
          placeholder="Tap keyboard to select"
          fullWidth
          InputProps={{
            endAdornment: (
              <IconButton
                onClick={() => openPhonemePicker('phoneme1')}
                edge="end"
                color="primary"
                size="small"
                aria-label="Open phoneme picker"
              >
                <KeyboardIcon />
              </IconButton>
            )
          }}
        />
        <TextField
          label="Phoneme 2 (IPA)"
          value={state.phoneme2}
          onChange={(e) => setState({ ...state, phoneme2: e.target.value })}
          size="small"
          placeholder="Tap keyboard to select"
          fullWidth
          InputProps={{
            endAdornment: (
              <IconButton
                onClick={() => openPhonemePicker('phoneme2')}
                edge="end"
                color="primary"
                size="small"
                aria-label="Open phoneme picker"
              >
                <KeyboardIcon />
              </IconButton>
            )
          }}
        />

        {/* Property Filters */}
        <Box>
          <Typography variant="subtitle2" gutterBottom>
            Property Filters
          </Typography>

          {/* Syllables */}
          <Box sx={{ mb: 2 }}>
            <Typography variant="caption" color="text.secondary">
              Syllables: {filters.syllables[0]} - {filters.syllables[1]}
            </Typography>
            <Slider
              value={filters.syllables}
              onChange={(_, newValue) => setFilters({ ...filters, syllables: newValue as [number, number] })}
              min={dbRanges.syllables[0]}
              max={dbRanges.syllables[1]}
              valueLabelDisplay="auto"
              size="small"
            />
          </Box>

          {/* WCM Score */}
          <Box sx={{ mb: 2 }}>
            <Typography variant="caption" color="text.secondary">
              WCM Score: {filters.wcm[0]} - {filters.wcm[1]}
            </Typography>
            <Slider
              value={filters.wcm}
              onChange={(_, newValue) => setFilters({ ...filters, wcm: newValue as [number, number] })}
              min={dbRanges.wcm[0]}
              max={dbRanges.wcm[1]}
              valueLabelDisplay="auto"
              size="small"
            />
          </Box>

          {/* Frequency */}
          <Box sx={{ mb: 2 }}>
            <Typography variant="caption" color="text.secondary">
              Frequency: {filters.frequency[0]} - {filters.frequency[1]}
            </Typography>
            <Slider
              value={filters.frequency}
              onChange={(_, newValue) => setFilters({ ...filters, frequency: newValue as [number, number] })}
              min={dbRanges.frequency[0]}
              max={dbRanges.frequency[1]}
              valueLabelDisplay="auto"
              size="small"
            />
          </Box>
        </Box>

        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2}>
          <Button
            variant="outlined"
            onClick={handleClear}
            fullWidth
            size="large"
          >
            Clear Filters
          </Button>
          <Button
            variant="contained"
            startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <RunIcon />}
            onClick={handleGenerate}
            disabled={loading}
            fullWidth
            size="large"
          >
            {loading ? 'Generating...' : 'Generate'}
          </Button>
        </Stack>
      </Stack>

      {error && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {error}
        </Alert>
      )}

      {results && results.length > 0 && (
        <Box sx={{ mt: 3 }}>
          <WordResultsDisplay results={results} />
        </Box>
      )}

      {results && results.length === 0 && (
        <Alert severity="info" sx={{ mt: 2 }}>
          No minimal pairs found for this phoneme contrast.
        </Alert>
      )}

      {/* Phoneme Picker Dialog */}
      <PhonemePickerDialog
        open={phonemePickerOpen}
        onClose={() => setPhonemePickerOpen(false)}
        onSelect={handlePhonemeSelect}
      />
    </Box>
  );
};

export default MinimalPairsTool;
