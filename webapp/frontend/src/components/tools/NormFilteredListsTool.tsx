/**
 * Norm-Filtered Lists Tool Component
 *
 * Filter words by comprehensive psycholinguistic norms:
 * - Frequency, Age of Acquisition
 * - Imageability, Familiarity, Concreteness
 * - Emotional norms (Valence, Arousal, Dominance)
 * - Phonological complexity (WCM, syllables)
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Button,
  Stack,
  Alert,
  CircularProgress,
  Typography,
  Slider,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
} from '@mui/material';
import {
  PlayArrow as RunIcon,
  ExpandMore as ExpandMoreIcon,
} from '@mui/icons-material';
import api from '../../services/phonolexApi';
import type { Word } from '../../services/phonolexApi';
import WordResultsDisplay from '../WordResultsDisplay';

interface NormFilters {
  frequency: [number, number];
  aoa: [number, number];
  imageability: [number, number];
  familiarity: [number, number];
  concreteness: [number, number];
  valence: [number, number];
  arousal: [number, number];
  dominance: [number, number];
  wcm: [number, number];
  syllables: [number, number];
  phonemes: [number, number];
  msh: [number, number];
}

const NormFilteredListsTool: React.FC = () => {
  // Store database ranges separately for slider min/max
  const [dbRanges, setDbRanges] = useState<Record<string, [number, number]>>({
    syllables: [1, 5],
    phonemes: [1, 10],
    wcm: [0, 15],
    msh: [1, 6],
    frequency: [0, 1000],
    aoa: [2, 10],
    imageability: [1, 7],
    familiarity: [1, 7],
    concreteness: [1, 5],
    valence: [1, 9],
    arousal: [1, 9],
    dominance: [1, 9],
  });

  // Initial values are fallbacks - will be replaced with database values
  const [filters, setFilters] = useState<NormFilters>({
    frequency: [0, 1000],
    aoa: [2, 10],
    imageability: [1, 7],
    familiarity: [1, 7],
    concreteness: [1, 5],
    valence: [1, 9],
    arousal: [1, 9],
    dominance: [1, 9],
    wcm: [0, 15],
    syllables: [1, 5],
    phonemes: [1, 10],
    msh: [1, 6],
  });

  const [results, setResults] = useState<Word[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch property ranges from database on mount
  useEffect(() => {
    const fetchRanges = async () => {
      try {
        const ranges = await api.getPropertyRanges();
        setDbRanges(ranges);
        setFilters({
          syllables: ranges.syllables as [number, number],
          phonemes: ranges.phonemes as [number, number],
          wcm: ranges.wcm as [number, number],
          msh: ranges.msh as [number, number],
          frequency: ranges.frequency as [number, number],
          aoa: ranges.aoa as [number, number],
          imageability: ranges.imageability as [number, number],
          familiarity: ranges.familiarity as [number, number],
          concreteness: ranges.concreteness as [number, number],
          valence: ranges.valence as [number, number],
          arousal: ranges.arousal as [number, number],
          dominance: ranges.dominance as [number, number],
        });
      } catch (error) {
        console.error('Failed to fetch property ranges:', error);
        // Keep hardcoded defaults as fallback
      }
    };
    fetchRanges();
  }, []);

  const handleGenerate = async () => {
    setLoading(true);
    setError(null);
    try {
      // For now, just filter by syllables and WCM since the API doesn't support all filters yet
      // TODO: Update API to support all psycholinguistic filters
      const data = await api.filterWords({
        min_syllables: filters.syllables[0],
        max_syllables: filters.syllables[1],
        limit: 100,
      });
      setResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      setResults(null);
    } finally {
      setLoading(false);
    }
  };

  const handleFilterChange = (name: keyof NormFilters, value: [number, number]) => {
    setFilters({ ...filters, [name]: value });
  };

  const getActiveFilterCount = () => {
    // Count how many filters are not at their default ranges
    let count = 0;
    if (filters.syllables[0] !== 1 || filters.syllables[1] !== 5) count++;
    if (filters.wcm[0] !== 0 || filters.wcm[1] !== 15) count++;
    if (filters.frequency[0] !== 0 || filters.frequency[1] !== 1000) count++;
    if (filters.aoa[0] !== 2 || filters.aoa[1] !== 10) count++;
    return count;
  };

  return (
    <Box>
      <Stack spacing={2}>
        {/* Phonological Complexity */}
        <Accordion defaultExpanded>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="subtitle2" fontWeight={600}>
              Phonological Complexity
            </Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Stack spacing={3}>
              <Box>
                <Typography variant="body2" gutterBottom>
                  Syllable Count: {filters.syllables[0]} - {filters.syllables[1]}
                </Typography>
                <Slider
                  value={filters.syllables}
                  onChange={(_, value) => handleFilterChange('syllables', value as [number, number])}
                  min={dbRanges.syllables[0]}
                  max={dbRanges.syllables[1]}
                  step={1}
                  marks
                  valueLabelDisplay="auto"
                />
              </Box>
              <Box>
                <Typography variant="body2" gutterBottom>
                  Phoneme Count: {filters.phonemes[0]} - {filters.phonemes[1]}
                </Typography>
                <Slider
                  value={filters.phonemes}
                  onChange={(_, value) => handleFilterChange('phonemes', value as [number, number])}
                  min={dbRanges.phonemes[0]}
                  max={dbRanges.phonemes[1]}
                  step={1}
                  marks
                  valueLabelDisplay="auto"
                />
              </Box>
              <Box>
                <Typography variant="body2" gutterBottom>
                  WCM Score: {filters.wcm[0]} - {filters.wcm[1]}
                  <Typography variant="caption" color="text.secondary" display="block">
                    Word Complexity Measure (Stoel-Gammon, 2010)
                  </Typography>
                </Typography>
                <Slider
                  value={filters.wcm}
                  onChange={(_, value) => handleFilterChange('wcm', value as [number, number])}
                  min={dbRanges.wcm[0]}
                  max={dbRanges.wcm[1]}
                  step={1}
                  valueLabelDisplay="auto"
                />
              </Box>
              <Box>
                <Typography variant="body2" gutterBottom>
                  MSH Stage: {filters.msh[0]} - {filters.msh[1]}
                  <Typography variant="caption" color="text.secondary" display="block">
                    Motor Speech Hierarchy - Developmental complexity
                  </Typography>
                </Typography>
                <Slider
                  value={filters.msh}
                  onChange={(_, value) => handleFilterChange('msh', value as [number, number])}
                  min={dbRanges.msh[0]}
                  max={dbRanges.msh[1]}
                  step={1}
                  marks
                  valueLabelDisplay="auto"
                />
              </Box>
            </Stack>
          </AccordionDetails>
        </Accordion>

        {/* Lexical Properties */}
        <Accordion>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="subtitle2" fontWeight={600}>
              Lexical Properties
            </Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Stack spacing={3}>
              <Box>
                <Typography variant="body2" gutterBottom>
                  Frequency: {filters.frequency[0]} - {filters.frequency[1]} per million
                  <Typography variant="caption" color="text.secondary" display="block">
                    SUBTLEX-US corpus
                  </Typography>
                </Typography>
                <Slider
                  value={filters.frequency}
                  onChange={(_, value) => handleFilterChange('frequency', value as [number, number])}
                  min={dbRanges.frequency[0]}
                  max={dbRanges.frequency[1]}
                  step={10}
                  valueLabelDisplay="auto"
                />
              </Box>
              <Box>
                <Typography variant="body2" gutterBottom>
                  Age of Acquisition: {filters.aoa[0]} - {filters.aoa[1]} years
                  <Typography variant="caption" color="text.secondary" display="block">
                    Kuperman et al. (2012) - Age typically learned
                  </Typography>
                </Typography>
                <Slider
                  value={filters.aoa}
                  onChange={(_, value) => handleFilterChange('aoa', value as [number, number])}
                  min={dbRanges.aoa[0]}
                  max={dbRanges.aoa[1]}
                  step={0.5}
                  valueLabelDisplay="auto"
                />
              </Box>
            </Stack>
          </AccordionDetails>
        </Accordion>

        {/* Semantic Properties */}
        <Accordion>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="subtitle2" fontWeight={600}>
              Semantic Properties
            </Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Stack spacing={3}>
              <Box>
                <Typography variant="body2" gutterBottom>
                  Imageability: {filters.imageability[0]} - {filters.imageability[1]}
                  <Typography variant="caption" color="text.secondary" display="block">
                    How easily a word evokes a mental image
                  </Typography>
                </Typography>
                <Slider
                  value={filters.imageability}
                  onChange={(_, value) => handleFilterChange('imageability', value as [number, number])}
                  min={dbRanges.imageability[0]}
                  max={dbRanges.imageability[1]}
                  step={0.5}
                  valueLabelDisplay="auto"
                />
              </Box>
              <Box>
                <Typography variant="body2" gutterBottom>
                  Familiarity: {filters.familiarity[0]} - {filters.familiarity[1]}
                  <Typography variant="caption" color="text.secondary" display="block">
                    How familiar the word is
                  </Typography>
                </Typography>
                <Slider
                  value={filters.familiarity}
                  onChange={(_, value) => handleFilterChange('familiarity', value as [number, number])}
                  min={dbRanges.familiarity[0]}
                  max={dbRanges.familiarity[1]}
                  step={0.5}
                  valueLabelDisplay="auto"
                />
              </Box>
              <Box>
                <Typography variant="body2" gutterBottom>
                  Concreteness: {filters.concreteness[0]} - {filters.concreteness[1]}
                  <Typography variant="caption" color="text.secondary" display="block">
                    How concrete vs. abstract (higher = more concrete)
                  </Typography>
                </Typography>
                <Slider
                  value={filters.concreteness}
                  onChange={(_, value) => handleFilterChange('concreteness', value as [number, number])}
                  min={dbRanges.concreteness[0]}
                  max={dbRanges.concreteness[1]}
                  step={0.5}
                  valueLabelDisplay="auto"
                />
              </Box>
            </Stack>
          </AccordionDetails>
        </Accordion>

        {/* Affective Properties */}
        <Accordion>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="subtitle2" fontWeight={600}>
              Affective Properties
            </Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Stack spacing={3}>
              <Box>
                <Typography variant="body2" gutterBottom>
                  Valence: {filters.valence[0]} - {filters.valence[1]}
                  <Typography variant="caption" color="text.secondary" display="block">
                    Warriner et al. (2013) - Positive vs. negative emotion
                  </Typography>
                </Typography>
                <Slider
                  value={filters.valence}
                  onChange={(_, value) => handleFilterChange('valence', value as [number, number])}
                  min={dbRanges.valence[0]}
                  max={dbRanges.valence[1]}
                  step={0.5}
                  valueLabelDisplay="auto"
                />
              </Box>
              <Box>
                <Typography variant="body2" gutterBottom>
                  Arousal: {filters.arousal[0]} - {filters.arousal[1]}
                  <Typography variant="caption" color="text.secondary" display="block">
                    Emotional intensity (calm vs. excited)
                  </Typography>
                </Typography>
                <Slider
                  value={filters.arousal}
                  onChange={(_, value) => handleFilterChange('arousal', value as [number, number])}
                  min={dbRanges.arousal[0]}
                  max={dbRanges.arousal[1]}
                  step={0.5}
                  valueLabelDisplay="auto"
                />
              </Box>
              <Box>
                <Typography variant="body2" gutterBottom>
                  Dominance: {filters.dominance[0]} - {filters.dominance[1]}
                  <Typography variant="caption" color="text.secondary" display="block">
                    Perceived control (weak vs. powerful)
                  </Typography>
                </Typography>
                <Slider
                  value={filters.dominance}
                  onChange={(_, value) => handleFilterChange('dominance', value as [number, number])}
                  min={dbRanges.dominance[0]}
                  max={dbRanges.dominance[1]}
                  step={0.5}
                  valueLabelDisplay="auto"
                />
              </Box>
            </Stack>
          </AccordionDetails>
        </Accordion>

        <Button
          variant="contained"
          startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <RunIcon />}
          onClick={handleGenerate}
          disabled={loading}
          fullWidth
          size="large"
        >
          {loading ? 'Generating List...' : 'Generate Filtered Word List'}
        </Button>
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
          No words found matching these filters. Try expanding your ranges.
        </Alert>
      )}
    </Box>
  );
};

export default NormFilteredListsTool;
