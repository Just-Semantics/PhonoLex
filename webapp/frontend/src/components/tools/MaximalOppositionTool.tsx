/**
 * Maximal Opposition Tool Component
 *
 * Generate intervention targets based on Gierut's maximal opposition research.
 * Pairs two UNKNOWN sounds differing by major class and maximal features
 * for better system-wide phonological generalization.
 *
 * References:
 * - Gierut, J. A. (1990). Differential learning of phonological oppositions.
 * - Storkel, H. L. (2022). Minimal, Maximal, or Multiple...
 */

import React, { useState } from 'react';
import {
  Box,
  TextField,
  Button,
  Stack,
  Alert,
  CircularProgress,
  Typography,
  Paper,
  Chip,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  IconButton,
} from '@mui/material';
import {
  PlayArrow as RunIcon,
  Clear as ClearIcon,
  Keyboard as KeyboardIcon,
} from '@mui/icons-material';
import api from '../../services/phonolexApi';
import type { Word } from '../../services/phonolexApi';
import PhonemePickerDialog from '../PhonemePickerDialog';

interface MaximalOppositionPair {
  phoneme1: string;
  phoneme2: string;
  score: number;
  major_class_diff: boolean;
  feature_diffs: number;
}

interface WordPair {
  word1: Word;
  word2: Word;
  position: number;
}

const MaximalOppositionTool: React.FC = () => {
  const [sonorants, setSonorants] = useState<string>('');
  const [obstruents, setObstruents] = useState<string>('');
  const [pairs, setPairs] = useState<MaximalOppositionPair[] | null>(null);
  const [selectedPair, setSelectedPair] = useState<MaximalOppositionPair | null>(null);
  const [wordLists, setWordLists] = useState<WordPair[] | null>(null);
  const [position, setPosition] = useState<'initial' | 'medial' | 'final' | 'any'>('initial');

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [phonemePickerOpen, setPhonemePickerOpen] = useState(false);
  const [phonemePickerTarget, setPhonemePickerTarget] = useState<'sonorants' | 'obstruents'>('sonorants');

  const handleGeneratePairs = async () => {
    setLoading(true);
    setError(null);
    setPairs(null);
    setSelectedPair(null);
    setWordLists(null);

    try {
      // Parse phonemes from both fields
      const sonorantList = sonorants
        .split(/[\s,]+/)
        .map(p => p.trim())
        .filter(p => p.length > 0);

      const obstruentList = obstruents
        .split(/[\s,]+/)
        .map(p => p.trim())
        .filter(p => p.length > 0);

      if (sonorantList.length === 0 && obstruentList.length === 0) {
        throw new Error('Please enter at least one phoneme in each field');
      }

      if (sonorantList.length === 0) {
        throw new Error('Please enter at least one sonorant (m, n, ŋ, l, r, w, j)');
      }

      if (obstruentList.length === 0) {
        throw new Error('Please enter at least one obstruent (p, t, k, f, s, ʃ, etc.)');
      }

      // Combine both lists for API call
      const phonemeList = [...sonorantList, ...obstruentList];

      const data = await api.generateMaximalOppositionPairs({
        unknown_phonemes: phonemeList,
        top_n: 10,
      });

      if (data.length === 0) {
        setError('No maximal opposition pairs found. Verify phonemes are from different major classes.');
      } else {
        setPairs(data);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateWordLists = async (pair: MaximalOppositionPair) => {
    setLoading(true);
    setError(null);
    setSelectedPair(pair);
    setWordLists(null);

    try {
      const data = await api.findMaximalOppositionWordLists({
        phoneme1: pair.phoneme1,
        phoneme2: pair.phoneme2,
        position,
        max_pairs: 10,
      });

      if (data.length === 0) {
        setError(`No word pairs found for /${pair.phoneme1}/-/${pair.phoneme2}/ in ${position} position. Try a different position.`);
      } else {
        setWordLists(data);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setSonorants('');
    setObstruents('');
    setPairs(null);
    setSelectedPair(null);
    setWordLists(null);
    setError(null);
  };

  const handlePhonemeSelect = (phoneme: string) => {
    if (phonemePickerTarget === 'sonorants') {
      setSonorants(prev => prev + (prev ? ' ' : '') + phoneme);
    } else {
      setObstruents(prev => prev + (prev ? ' ' : '') + phoneme);
    }
  };

  return (
    <Box>
      {/* Input Section */}
      <Paper sx={{ p: { xs: 2, sm: 3 }, mb: { xs: 2, sm: 3 } }}>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2, fontSize: { xs: '0.8125rem', sm: '0.875rem' } }}>
          Enter unknown phonemes from each major class. Pairs one sonorant with one obstruent for maximal feature contrast.
        </Typography>

        <Stack spacing={2}>
          {/* Sonorants Field */}
          <Box>
            <TextField
              fullWidth
              label="Sonorants (m, n, ŋ, l, r, w, j)"
              value={sonorants}
              onChange={(e) => setSonorants(e.target.value)}
              placeholder="e.g., l r ŋ"
              size="small"
              helperText="Nasals, liquids, glides"
              InputProps={{
                endAdornment: (
                  <IconButton
                    onClick={() => {
                      setPhonemePickerTarget('sonorants');
                      setPhonemePickerOpen(true);
                    }}
                    edge="end"
                    color="primary"
                    size="small"
                    sx={{ minWidth: 40, minHeight: 40 }}
                  >
                    <KeyboardIcon />
                  </IconButton>
                )
              }}
            />
          </Box>

          {/* Obstruents Field */}
          <Box>
            <TextField
              fullWidth
              label="Obstruents (p, t, k, b, d, g, f, s, ʃ, θ, etc.)"
              value={obstruents}
              onChange={(e) => setObstruents(e.target.value)}
              placeholder="e.g., g θ ʃ"
              size="small"
              helperText="Stops, fricatives, affricates"
              InputProps={{
                endAdornment: (
                  <IconButton
                    onClick={() => {
                      setPhonemePickerTarget('obstruents');
                      setPhonemePickerOpen(true);
                    }}
                    edge="end"
                    color="primary"
                    size="small"
                    sx={{ minWidth: 40, minHeight: 40 }}
                  >
                    <KeyboardIcon />
                  </IconButton>
                )
              }}
            />
          </Box>

          <Stack direction={{ xs: 'column', sm: 'row' }} spacing={{ xs: 1.5, sm: 2 }}>
            <Button
              variant="outlined"
              startIcon={<ClearIcon />}
              onClick={handleClear}
              disabled={loading}
              fullWidth
              size="large"
            >
              Clear
            </Button>
            <Button
              variant="contained"
              startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <RunIcon />}
              onClick={handleGeneratePairs}
              disabled={loading || sonorants.trim().length === 0 || obstruents.trim().length === 0}
              fullWidth
              size="large"
            >
              {loading ? 'Generating...' : 'Generate Pairs'}
            </Button>
          </Stack>
        </Stack>
      </Paper>

      {/* Error Display */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Results: Maximal Opposition Pairs */}
      {pairs && pairs.length > 0 && (
        <Paper sx={{ p: { xs: 2, sm: 3 }, mb: { xs: 2, sm: 3 } }}>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2, fontSize: { xs: '0.8125rem', sm: '0.875rem' } }}>
            Select a pair to generate word lists. Click pair to continue.
          </Typography>

          {/* Position Selection */}
          <FormControl size="small" fullWidth sx={{ mb: 2 }}>
            <InputLabel>Position in Word</InputLabel>
            <Select
              value={position}
              label="Position in Word"
              onChange={(e) => setPosition(e.target.value as 'initial' | 'medial' | 'final' | 'any')}
            >
              <MenuItem value="initial">Word-Initial</MenuItem>
              <MenuItem value="medial">Word-Medial</MenuItem>
              <MenuItem value="final">Word-Final</MenuItem>
              <MenuItem value="any">Any Position</MenuItem>
            </Select>
          </FormControl>

          <Stack spacing={1}>
            {pairs.map((pair, index) => (
              <Paper
                key={index}
                sx={{
                  p: { xs: 1.5, sm: 2 },
                  cursor: 'pointer',
                  border: selectedPair === pair ? '2px solid' : '1px solid',
                  borderColor: selectedPair === pair ? 'primary.main' : 'divider',
                  '&:hover': { bgcolor: 'action.hover' },
                }}
                onClick={() => handleGenerateWordLists(pair)}
              >
                <Stack direction="row" spacing={2} alignItems="center" justifyContent="space-between">
                  <Box>
                    <Typography variant="h6" component="span" sx={{ fontSize: { xs: '1rem', sm: '1.25rem' } }}>
                      /{pair.phoneme1}/ - /{pair.phoneme2}/
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5, fontSize: { xs: '0.75rem', sm: '0.875rem' } }}>
                      Score: {pair.score} ({pair.feature_diffs} features)
                    </Typography>
                  </Box>
                  <Chip
                    label={`${pair.feature_diffs} features`}
                    color="primary"
                    size="small"
                  />
                </Stack>
              </Paper>
            ))}
          </Stack>
        </Paper>
      )}

      {/* Results: Word Lists */}
      {selectedPair && (
        <Paper sx={{ p: { xs: 2, sm: 3 } }}>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2, fontSize: { xs: '0.8125rem', sm: '0.875rem' } }}>
            Word pairs for /{selectedPair.phoneme1}/ - /{selectedPair.phoneme2}/ in {position} position
          </Typography>

          {loading && (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
              <CircularProgress />
            </Box>
          )}

          {wordLists && wordLists.length > 0 && (
            <>
              <Stack spacing={1}>
                {wordLists.map((pair, index) => (
                  <Paper key={index} sx={{ p: { xs: 1.5, sm: 2 }, bgcolor: 'action.hover' }}>
                    <Stack direction="row" spacing={{ xs: 1.5, sm: 3 }} alignItems="center">
                      <Typography variant="body1" sx={{ minWidth: { xs: 20, sm: 30 }, fontSize: { xs: '0.9375rem', sm: '1rem' } }}>
                        {index + 1}.
                      </Typography>
                      <Box sx={{ flexGrow: 1 }}>
                        <Stack direction="row" spacing={2} divider={<Typography>-</Typography>}>
                          <Box>
                            <Typography variant="body1" fontWeight="bold" sx={{ fontSize: { xs: '0.9375rem', sm: '1rem' } }}>
                              {pair.word1.word}
                            </Typography>
                            <Typography variant="caption" color="text.secondary" sx={{ fontSize: { xs: '0.6875rem', sm: '0.75rem' } }}>
                              /{pair.word1.ipa}/
                            </Typography>
                          </Box>
                          <Box>
                            <Typography variant="body1" fontWeight="bold" sx={{ fontSize: { xs: '0.9375rem', sm: '1rem' } }}>
                              {pair.word2.word}
                            </Typography>
                            <Typography variant="caption" color="text.secondary" sx={{ fontSize: { xs: '0.6875rem', sm: '0.75rem' } }}>
                              /{pair.word2.ipa}/
                            </Typography>
                          </Box>
                        </Stack>
                      </Box>
                    </Stack>
                  </Paper>
                ))}
              </Stack>

            </>
          )}

          {!loading && wordLists && wordLists.length === 0 && (
            <Alert severity="warning">
              No word pairs found for this phoneme combination in {position} position. Try selecting a different position.
            </Alert>
          )}
        </Paper>
      )}

      {/* Phoneme Picker Dialog */}
      <PhonemePickerDialog
        open={phonemePickerOpen}
        onClose={() => setPhonemePickerOpen(false)}
        onSelect={handlePhonemeSelect}
        filter={phonemePickerTarget}
      />
    </Box>
  );
};

export default MaximalOppositionTool;
