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
  position: 'all' | 'initial' | 'final' | 'medial';
  syllableCount: number;
  description: string;
}

const PRESETS: PresetConfig[] = [
  {
    name: 'Rhymes',
    weights: { onset: 0.0, nucleus: 0.5, coda: 0.5 },
    position: 'final',
    syllableCount: 1,
    description: 'Final syllable',
  },
  {
    name: 'Balanced',
    weights: { onset: 0.33, nucleus: 0.33, coda: 0.33 },
    position: 'all',
    syllableCount: 1,
    description: 'Equal weighting',
  },
  {
    name: 'Alliteration',
    weights: { onset: 1.0, nucleus: 0.5, coda: 0.0 },
    position: 'initial',
    syllableCount: 1,
    description: 'Initial syllable',
  },
  {
    name: 'Assonance',
    weights: { onset: 0.0, nucleus: 1.0, coda: 0.0 },
    position: 'all',
    syllableCount: 1,
    description: 'Vowels only',
  },
  {
    name: 'Consonance',
    weights: { onset: 0.5, nucleus: 0.0, coda: 0.5 },
    position: 'all',
    syllableCount: 1,
    description: 'Consonants only',
  },
];

const PhonologicalSimilarityTool: React.FC = () => {
  const [targetWord, setTargetWord] = useState('cat');
  const [weights, setWeights] = useState<SimilarityWeights>(PRESETS[0].weights); // Default to Rhymes
  const [threshold, setThreshold] = useState(0.75);
  const [position, setPosition] = useState<'all' | 'initial' | 'final' | 'medial'>(PRESETS[0].position); // Default to Rhymes
  const [syllableCount, setSyllableCount] = useState(PRESETS[0].syllableCount); // Default to Rhymes
  const [results, setResults] = useState<SimilarityResult[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleGenerate = async () => {
    setLoading(true);
    setError(null);
    try {
      console.log('Searching for:', targetWord, 'with weights:', weights, 'threshold:', threshold, 'position:', position, 'count:', syllableCount);
      const data = await api.findSimilarWords(targetWord, {
        threshold,
        limit: 200,
        weights,
        position,
        syllableCount,
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
    setPosition(preset.position);
    setSyllableCount(preset.syllableCount);
  };

  const resetWeights = () => {
    const rhymesPreset = PRESETS[0];
    setWeights(rhymesPreset.weights);
    setPosition(rhymesPreset.position);
    setSyllableCount(rhymesPreset.syllableCount);
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
                  weights.coda === preset.weights.coda &&
                  position === preset.position &&
                  syllableCount === preset.syllableCount
                    ? 'primary'
                    : 'default'
                }
                sx={{ mb: { xs: 1.5, sm: 1 } }}
              />
            ))}
          </Stack>
        </Box>

        {/* Syllable Position Selection */}
        <Paper variant="outlined" sx={{ p: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Syllable Position
          </Typography>
          <Stack direction="row" spacing={2}>
            <FormControl size="small" fullWidth>
              <InputLabel>Position</InputLabel>
              <Select
                value={position}
                label="Position"
                onChange={(e) => setPosition(e.target.value as 'all' | 'initial' | 'final' | 'medial')}
              >
                <MenuItem value="all">All syllables</MenuItem>
                <MenuItem value="final">Final (rhyme detection)</MenuItem>
                <MenuItem value="initial">Initial (alliteration)</MenuItem>
                <MenuItem value="medial">Medial (exclude first & last)</MenuItem>
              </Select>
            </FormControl>

            <FormControl size="small" sx={{ minWidth: 120 }} disabled={position === 'all' || position === 'medial'}>
              <InputLabel>Count</InputLabel>
              <Select
                value={syllableCount}
                label="Count"
                onChange={(e) => setSyllableCount(e.target.value as number)}
              >
                <MenuItem value={1}>1 syllable</MenuItem>
                <MenuItem value={2}>2 syllables</MenuItem>
                <MenuItem value={3}>3 syllables</MenuItem>
              </Select>
            </FormControl>
          </Stack>
          <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
            {position === 'final' && 'Final syllable(s) only'}
            {position === 'initial' && 'Initial syllable(s) only'}
            {position === 'medial' && 'Middle syllables only'}
            {position === 'all' && 'All syllables'}
          </Typography>
        </Paper>

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
