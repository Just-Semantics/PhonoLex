/**
 * Phonological Similarity Explorer
 *
 * Unified tool for phonological similarity search with adjustable component weights.
 * Replaces separate RhymeSetsTool and similarity search.
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
  Slider,
  Paper,
  Chip,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import {
  PlayArrow as RunIcon,
  Refresh as ResetIcon,
} from '@mui/icons-material';
import api from '../../services/phonolexApi';
import { SimilarityResult } from '../../types/phonology';
import WordListTable from '../shared/WordListTable';

interface SimilarityWeights {
  onset: number;
  nucleus: number;
  coda: number;
}

interface PresetConfig {
  name: string;
  weights: SimilarityWeights;
  description: string;
}

const PRESETS: PresetConfig[] = [
  {
    name: 'Rhymes',
    weights: { onset: 0.0, nucleus: 0.5, coda: 0.5 },
    description: 'Focus on nucleus + coda (perfect for rhyme detection)',
  },
  {
    name: 'Balanced',
    weights: { onset: 0.33, nucleus: 0.33, coda: 0.33 },
    description: 'Equal weight to all syllable components',
  },
  {
    name: 'Alliteration',
    weights: { onset: 1.0, nucleus: 0.0, coda: 0.0 },
    description: 'Focus on initial sounds only',
  },
  {
    name: 'Assonance',
    weights: { onset: 0.0, nucleus: 1.0, coda: 0.0 },
    description: 'Focus on vowel sounds only',
  },
  {
    name: 'Consonance',
    weights: { onset: 0.5, nucleus: 0.0, coda: 0.5 },
    description: 'Focus on consonant sounds (onset + coda)',
  },
];

const PhonologicalSimilarityTool: React.FC = () => {
  const [targetWord, setTargetWord] = useState('cat');
  const [weights, setWeights] = useState<SimilarityWeights>(PRESETS[0].weights); // Default to Rhymes
  const [threshold, setThreshold] = useState(0.75);
  const [results, setResults] = useState<SimilarityResult[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleGenerate = async () => {
    setLoading(true);
    setError(null);
    try {
      console.log('Searching for:', targetWord, 'with weights:', weights, 'threshold:', threshold);
      const data = await api.findSimilarWords(targetWord, {
        threshold,
        limit: 200,
        weights,
      });
      console.log('Received results:', data?.length, 'words');
      console.log('First result:', data?.[0]);
      setResults(data);
    } catch (err) {
      console.error('Error in handleGenerate:', err);
      setError(err instanceof Error ? err.message : 'An error occurred');
      setResults(null);
    } finally {
      setLoading(false);
    }
  };

  const applyPreset = (preset: PresetConfig) => {
    setWeights(preset.weights);
  };

  const resetWeights = () => {
    setWeights(PRESETS[0].weights); // Reset to Rhymes preset
  };

  // Sort results by similarity (highest first)
  const sortedResults = results
    ? [...results]
        .filter(result => result && result.word) // Defensive null check
        .sort((a, b) => b.similarity - a.similarity) // Sort by similarity descending
    : null;

  return (
    <Box>
      <Stack spacing={3}>
        {/* Target Word Input */}
        <TextField
          label="Target Word"
          value={targetWord}
          onChange={(e) => setTargetWord(e.target.value)}
          size="small"
          placeholder="e.g., cat, make, computer"
          fullWidth
        />

        {/* Preset Buttons */}
        <Box>
          <Typography variant="subtitle2" gutterBottom sx={{ mb: 1 }}>
            Presets
          </Typography>
          <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
            {PRESETS.map((preset) => (
              <Chip
                key={preset.name}
                label={preset.name}
                onClick={() => applyPreset(preset)}
                color={
                  weights.onset === preset.weights.onset &&
                  weights.nucleus === preset.weights.nucleus &&
                  weights.coda === preset.weights.coda
                    ? 'primary'
                    : 'default'
                }
                sx={{ mb: 1 }}
              />
            ))}
          </Stack>
        </Box>

        {/* Weight Sliders */}
        <Paper variant="outlined" sx={{ p: 2 }}>
          <Stack spacing={2}>
            <Box display="flex" justifyContent="space-between" alignItems="center">
              <Typography variant="subtitle2">Component Weights</Typography>
              <Button size="small" startIcon={<ResetIcon />} onClick={resetWeights}>
                Reset
              </Button>
            </Box>

            {/* Onset Slider */}
            <Box>
              <Typography variant="body2" gutterBottom>
                Onset (initial sounds): {weights.onset.toFixed(2)}
              </Typography>
              <Slider
                value={weights.onset}
                onChange={(_, value) => setWeights({ ...weights, onset: value as number })}
                min={0}
                max={1}
                step={0.05}
                marks={[
                  { value: 0, label: '0' },
                  { value: 0.5, label: '0.5' },
                  { value: 1, label: '1' },
                ]}
                valueLabelDisplay="auto"
              />
            </Box>

            {/* Nucleus Slider */}
            <Box>
              <Typography variant="body2" gutterBottom>
                Nucleus (vowels): {weights.nucleus.toFixed(2)}
              </Typography>
              <Slider
                value={weights.nucleus}
                onChange={(_, value) => setWeights({ ...weights, nucleus: value as number })}
                min={0}
                max={1}
                step={0.05}
                marks={[
                  { value: 0, label: '0' },
                  { value: 0.5, label: '0.5' },
                  { value: 1, label: '1' },
                ]}
                valueLabelDisplay="auto"
              />
            </Box>

            {/* Coda Slider */}
            <Box>
              <Typography variant="body2" gutterBottom>
                Coda (final sounds): {weights.coda.toFixed(2)}
              </Typography>
              <Slider
                value={weights.coda}
                onChange={(_, value) => setWeights({ ...weights, coda: value as number })}
                min={0}
                max={1}
                step={0.05}
                marks={[
                  { value: 0, label: '0' },
                  { value: 0.5, label: '0.5' },
                  { value: 1, label: '1' },
                ]}
                valueLabelDisplay="auto"
              />
            </Box>
          </Stack>
        </Paper>

        {/* Similarity Threshold */}
        <FormControl size="small" fullWidth>
          <InputLabel>Minimum Similarity</InputLabel>
          <Select
            value={threshold}
            label="Minimum Similarity"
            onChange={(e) => setThreshold(e.target.value as number)}
          >
            <MenuItem value={0.95}>Very High (0.95)</MenuItem>
            <MenuItem value={0.85}>High (0.85)</MenuItem>
            <MenuItem value={0.75}>Medium (0.75)</MenuItem>
            <MenuItem value={0.65}>Low (0.65)</MenuItem>
            <MenuItem value={0.50}>Very Low (0.50)</MenuItem>
          </Select>
        </FormControl>

        {/* Generate Button */}
        <Button
          variant="contained"
          size="large"
          startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <RunIcon />}
          onClick={handleGenerate}
          disabled={loading}
          fullWidth
          sx={{ minHeight: 48 }}
        >
          {loading ? 'Searching...' : 'Find Similar Words'}
        </Button>
      </Stack>

      {/* Error Display */}
      {error && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {error}
        </Alert>
      )}

      {/* Results Display */}
      {results && results.length > 0 && sortedResults && (
        <Box sx={{ mt: 3 }}>
          <WordListTable
            words={sortedResults}
            showSimilarity={true}
            enableSelection={true}
            defaultSort="similarity"
            exportFilename={`phonolex_similar_to_${targetWord}.csv`}
          />
        </Box>
      )}

      {/* No Results */}
      {results && results.length === 0 && (
        <Alert severity="info" sx={{ mt: 2 }}>
          No similar words found for "{targetWord}" with threshold {threshold.toFixed(2)}. Try lowering the threshold or adjusting the weights.
        </Alert>
      )}
    </Box>
  );
};

export default PhonologicalSimilarityTool;
