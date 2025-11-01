/**
 * Minimal Pairs Tool Component
 *
 * Generate word pairs differing by a single phoneme for discrimination therapy
 */

import React, { useState } from 'react';
import {
  Box,
  TextField,
  Button,
  Stack,
  Alert,
  CircularProgress,
  IconButton,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import {
  PlayArrow as RunIcon,
  Keyboard as KeyboardIcon,
} from '@mui/icons-material';
import api from '../../services/phonolexApi';
import type { MinimalPair } from '../../services/phonolexApi';
import WordResultsDisplay from '../WordResultsDisplay';
import PhonemePickerDialog from '../PhonemePickerDialog';
import { validatePhonemeInput } from '../../utils/ipaValidation';

const MinimalPairsTool: React.FC = () => {
  const [state, setState] = useState<{
    phoneme1: string;
    phoneme2: string;
  }>({
    phoneme1: '',
    phoneme2: '',
  });

  const [position, setPosition] = useState<'any' | 'initial' | 'medial' | 'final'>('any');
  const [results, setResults] = useState<MinimalPair[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [phonemePickerOpen, setPhonemePickerOpen] = useState(false);
  const [activePhonemeField, setActivePhonemeField] = useState<'phoneme1' | 'phoneme2'>('phoneme1');
  const [ipaWarning1, setIpaWarning1] = useState<string | null>(null);
  const [ipaWarning2, setIpaWarning2] = useState<string | null>(null);

  const handleGenerate = async () => {
    setLoading(true);
    setError(null);
    try {
      let data = await api.getMinimalPairs({
        phoneme1: state.phoneme1,
        phoneme2: state.phoneme2,
        limit: 200,  // Get more for filtering
      });

      // Filter by position if specified
      if (position !== 'any') {
        data = data.filter(pair => {
          const pos = pair.position ?? pair.metadata?.position;
          if (pos === undefined) return true;  // Include if position unknown

          const wordLength = pair.word1.phoneme_count;

          if (position === 'initial') {
            return pos === 0;
          } else if (position === 'final') {
            return pos === wordLength - 1;
          } else if (position === 'medial') {
            return pos > 0 && pos < wordLength - 1;
          }
          return true;
        });
      }

      setResults(data.slice(0, 50));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      setResults(null);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setState({ phoneme1: '', phoneme2: '' });
    setPosition('any');
    setResults(null);
    setError(null);
  };

  const handlePhonemeSelect = (phoneme: string) => {
    setState({ ...state, [activePhonemeField]: state[activePhonemeField] + phoneme });
  };

  const openPhonemePicker = (field: 'phoneme1' | 'phoneme2') => {
    setActivePhonemeField(field);
    setPhonemePickerOpen(true);
  };

  return (
    <Box sx={{ p: { xs: 0, sm: 0 } }}>
      <Stack spacing={{ xs: 2, sm: 2 }}>
        <Box>
          <TextField
            label="Phoneme 1 (IPA)"
            value={state.phoneme1}
            onChange={(e) => {
              const newValue = e.target.value;
              setState({ ...state, phoneme1: newValue });

              // Validate IPA input
              if (newValue.trim()) {
                const validation = validatePhonemeInput(newValue);
                if (!validation.isValid && validation.suggestion) {
                  setIpaWarning1(validation.suggestion);
                } else {
                  setIpaWarning1(null);
                }
              } else {
                setIpaWarning1(null);
              }
            }}
            size="small"
            placeholder="e.g., t, k, s"
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
          {ipaWarning1 && (
            <Alert severity="warning" sx={{ mt: 1 }}>
              {ipaWarning1}
            </Alert>
          )}
        </Box>

        <Box>
          <TextField
            label="Phoneme 2 (IPA)"
            value={state.phoneme2}
            onChange={(e) => {
              const newValue = e.target.value;
              setState({ ...state, phoneme2: newValue });

              // Validate IPA input
              if (newValue.trim()) {
                const validation = validatePhonemeInput(newValue);
                if (!validation.isValid && validation.suggestion) {
                  setIpaWarning2(validation.suggestion);
                } else {
                  setIpaWarning2(null);
                }
              } else {
                setIpaWarning2(null);
              }
            }}
            size="small"
            placeholder="e.g., d, g, z"
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
          {ipaWarning2 && (
            <Alert severity="warning" sx={{ mt: 1 }}>
              {ipaWarning2}
            </Alert>
          )}
        </Box>

        {/* Position Filter */}
        <FormControl fullWidth size="small">
          <InputLabel>Position in Word</InputLabel>
          <Select
            value={position}
            label="Position in Word"
            onChange={(e) => setPosition(e.target.value as 'any' | 'initial' | 'medial' | 'final')}
          >
            <MenuItem value="any">Any Position</MenuItem>
            <MenuItem value="initial">Word-Initial</MenuItem>
            <MenuItem value="medial">Word-Medial</MenuItem>
            <MenuItem value="final">Word-Final</MenuItem>
          </Select>
        </FormControl>

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
