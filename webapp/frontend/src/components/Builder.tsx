/**
 * Builder Component - THE POWER TOOL
 *
 * Custom word list builder with:
 * - Pattern matching (STARTS_WITH, ENDS_WITH, CONTAINS)
 * - Property filters (syllables, WCM, MSH, AoA)
 * - Exclusion rules (phoneme blacklist, feature blacklist)
 * - Combined queries with AND logic
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Stack,
  Alert,
  CircularProgress,
  IconButton,
  Paper,
  Checkbox,
  FormControlLabel,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Slider,
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  Build as BuildIcon,
  Clear as ClearIcon,
  Keyboard as KeyboardIcon,
  ExpandMore as ExpandMoreIcon,
} from '@mui/icons-material';
import api from '../services/phonolexApi';
import type { BuilderRequest, Pattern, PatternType, Word } from '../services/phonolexApi';
import WordResultsDisplay from './WordResultsDisplay';
import PhonemePickerDialog from './PhonemePickerDialog';

const Builder: React.FC = () => {
  // Patterns state
  const [patterns, setPatterns] = useState<Pattern[]>([
    { type: 'STARTS_WITH', phoneme: 'k' },
  ]);

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

  // Property filters state (using ranges like NormFilteredListsTool)
  // Initial values are fallbacks - will be replaced with database values
  const [filters, setFilters] = useState({
    // Phonological Complexity
    syllables: [1, 5] as [number, number],
    phonemes: [1, 10] as [number, number],
    wcm: [0, 15] as [number, number],
    msh: [1, 6] as [number, number],

    // Lexical Properties
    frequency: [0, 1000] as [number, number],
    aoa: [2, 10] as [number, number],

    // Semantic Properties
    imageability: [1, 7] as [number, number],
    familiarity: [1, 7] as [number, number],
    concreteness: [1, 5] as [number, number],

    // Affective Properties
    valence: [1, 9] as [number, number],
    arousal: [1, 9] as [number, number],
    dominance: [1, 9] as [number, number],
  });

  const handleFilterChange = (key: keyof typeof filters, value: [number, number]) => {
    setFilters({ ...filters, [key]: value });
  };

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

  // Exclusions state
  const [excludePhonemes, setExcludePhonemes] = useState<string[]>([]);
  const [excludePhonemeInput, setExcludePhonemeInput] = useState('');

  // Results state
  const [results, setResults] = useState<Word[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Phoneme picker state
  const [phonemePickerOpen, setPhonemePickerOpen] = useState(false);
  const [phonemePickerTarget, setPhonemePickerTarget] = useState<
    { type: 'pattern'; index: number } | { type: 'exclusion' } | null
  >(null);

  // Handle phoneme selection
  const handlePhonemeSelect = (phoneme: string) => {
    if (phonemePickerTarget?.type === 'pattern') {
      updatePattern(phonemePickerTarget.index, 'phoneme', phoneme);
    } else if (phonemePickerTarget?.type === 'exclusion') {
      setExcludePhonemeInput(phoneme);
    }
    setPhonemePickerOpen(false);
    setPhonemePickerTarget(null);
  };

  // Open phoneme picker
  const openPhonemePicker = (target: { type: 'pattern'; index: number } | { type: 'exclusion' }) => {
    setPhonemePickerTarget(target);
    setPhonemePickerOpen(true);
  };

  // Add pattern
  const addPattern = () => {
    setPatterns([...patterns, { type: 'STARTS_WITH', phoneme: '' }]);
  };

  // Remove pattern
  const removePattern = (index: number) => {
    setPatterns(patterns.filter((_, i) => i !== index));
  };

  // Update pattern
  const updatePattern = (index: number, field: keyof Pattern, value: any) => {
    const updated = [...patterns];
    updated[index] = { ...updated[index], [field]: value };
    setPatterns(updated);
  };

  // Add exclusion
  const addExclusion = () => {
    if (excludePhonemeInput.trim() && !excludePhonemes.includes(excludePhonemeInput.trim())) {
      setExcludePhonemes([...excludePhonemes, excludePhonemeInput.trim()]);
      setExcludePhonemeInput('');
    }
  };

  // Remove exclusion
  const removeExclusion = (phoneme: string) => {
    setExcludePhonemes(excludePhonemes.filter((p) => p !== phoneme));
  };

  // Build word list
  const handleBuild = async () => {
    setLoading(true);
    setError(null);

    try {
      const request: BuilderRequest = {
        patterns: patterns.filter((p) => p.phoneme.trim() !== ''),
        filters: {
          min_syllables: filters.syllables[0],
          max_syllables: filters.syllables[1],
          min_phonemes: filters.phonemes[0],
          max_phonemes: filters.phonemes[1],
          min_wcm: filters.wcm[0],
          max_wcm: filters.wcm[1],
          min_msh: filters.msh[0],
          max_msh: filters.msh[1],
          min_frequency: filters.frequency[0],
          max_frequency: filters.frequency[1],
          min_aoa: filters.aoa[0],
          max_aoa: filters.aoa[1],
          min_imageability: filters.imageability[0],
          max_imageability: filters.imageability[1],
          min_familiarity: filters.familiarity[0],
          max_familiarity: filters.familiarity[1],
          min_concreteness: filters.concreteness[0],
          max_concreteness: filters.concreteness[1],
          min_valence: filters.valence[0],
          max_valence: filters.valence[1],
          min_arousal: filters.arousal[0],
          max_arousal: filters.arousal[1],
          min_dominance: filters.dominance[0],
          max_dominance: filters.dominance[1],
        },
        exclusions: {
          exclude_phonemes: excludePhonemes.length > 0 ? excludePhonemes : undefined,
        },
        limit: 200,
      };

      const data = await api.buildWordList(request);
      setResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Build failed');
      setResults(null);
    } finally {
      setLoading(false);
    }
  };

  // Clear all - reset to database ranges
  const handleClear = () => {
    setPatterns([{ type: 'STARTS_WITH', phoneme: 'k' }]);
    setFilters({
      syllables: dbRanges.syllables as [number, number],
      phonemes: dbRanges.phonemes as [number, number],
      wcm: dbRanges.wcm as [number, number],
      msh: dbRanges.msh as [number, number],
      frequency: dbRanges.frequency as [number, number],
      aoa: dbRanges.aoa as [number, number],
      imageability: dbRanges.imageability as [number, number],
      familiarity: dbRanges.familiarity as [number, number],
      concreteness: dbRanges.concreteness as [number, number],
      valence: dbRanges.valence as [number, number],
      arousal: dbRanges.arousal as [number, number],
      dominance: dbRanges.dominance as [number, number],
    });
    setExcludePhonemes([]);
    setExcludePhonemeInput('');
    setResults(null);
    setError(null);
  };

  return (
    <Box>
      <Stack spacing={2}>
        {/* Pattern Matching */}
        <Accordion defaultExpanded>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="h6">Patterns</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Stack spacing={2}>
              <Typography variant="body2" color="text.secondary">
                AND logic: all patterns must match. Space-separate phonemes (e.g., "s t" for /st/)
              </Typography>

              <Stack spacing={2}>
                {patterns.map((pattern, idx) => (
                  <Paper key={idx} variant="outlined" sx={{ p: 2 }}>
                    <Stack spacing={1}>
                      <Stack direction="row" spacing={2} alignItems="center">
                        <FormControl size="small" sx={{ minWidth: 140 }}>
                          <InputLabel>Type</InputLabel>
                          <Select
                            value={pattern.type}
                            label="Type"
                            onChange={(e) =>
                              updatePattern(idx, 'type', e.target.value as PatternType)
                            }
                          >
                            <MenuItem value="STARTS_WITH">Starts With</MenuItem>
                            <MenuItem value="ENDS_WITH">Ends With</MenuItem>
                            <MenuItem value="CONTAINS">Contains</MenuItem>
                          </Select>
                        </FormControl>

                        <TextField
                          label="Phoneme(s)"
                          value={pattern.phoneme}
                          onChange={(e) => updatePattern(idx, 'phoneme', e.target.value)}
                          size="small"
                          placeholder="Click keyboard to select"
                          fullWidth
                          InputProps={{
                            endAdornment: (
                              <IconButton
                                onClick={() => openPhonemePicker({ type: 'pattern', index: idx })}
                                edge="end"
                                color="primary"
                                size="small"
                              >
                                <KeyboardIcon />
                              </IconButton>
                            ),
                          }}
                        />

                        <IconButton
                          size="small"
                          color="error"
                          onClick={() => removePattern(idx)}
                          disabled={patterns.length === 1}
                        >
                          <DeleteIcon />
                        </IconButton>
                      </Stack>

                      {/* Medial Only checkbox - only shown for CONTAINS patterns */}
                      {pattern.type === 'CONTAINS' && (
                        <FormControlLabel
                          control={
                            <Checkbox
                              checked={pattern.medial_only || false}
                              onChange={(e) => updatePattern(idx, 'medial_only', e.target.checked)}
                              size="small"
                            />
                          }
                          label={
                            <Typography variant="body2" color="text.secondary">
                              Medial only (excludes word edges)
                            </Typography>
                          }
                          sx={{ ml: 0.5 }}
                        />
                      )}
                    </Stack>
                  </Paper>
                ))}
              </Stack>

              <Button
                size="small"
                startIcon={<AddIcon />}
                onClick={addPattern}
                variant="outlined"
                sx={{ alignSelf: 'flex-start' }}
              >
                Add Pattern
              </Button>
            </Stack>
          </AccordionDetails>
        </Accordion>

        {/* Property Filters */}
        <Accordion>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Box>
              <Typography variant="h6">Property Filters</Typography>
              <Typography variant="caption" color="text.secondary">
                Phonological, lexical, semantic, and affective properties
              </Typography>
            </Box>
          </AccordionSummary>
          <AccordionDetails>
            <Stack spacing={1}>
                {/* Phonological Complexity */}
                <Accordion defaultExpanded>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography variant="subtitle2" fontWeight={600}>
                      Phonological Complexity
                    </Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Stack spacing={3}>
                      {/* Syllables */}
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

                      {/* Phonemes */}
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

                      {/* WCM */}
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

                      {/* MSH */}
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
                      {/* Frequency */}
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

                      {/* Age of Acquisition */}
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
                      {/* Imageability */}
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

                      {/* Familiarity */}
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

                      {/* Concreteness */}
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
                      {/* Valence */}
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

                      {/* Arousal */}
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

                      {/* Dominance */}
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
              </Stack>
          </AccordionDetails>
        </Accordion>

        {/* Exclusions */}
        <Accordion>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Box>
              <Typography variant="h6">Exclusions</Typography>
              <Typography variant="caption" color="text.secondary">
                Exclude words containing specific phonemes
              </Typography>
            </Box>
          </AccordionSummary>
          <AccordionDetails>
            <Stack spacing={2}>
              <Stack direction="row" spacing={2}>
                <TextField
                  label="Phoneme to exclude"
                  value={excludePhonemeInput}
                  onChange={(e) => setExcludePhonemeInput(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && addExclusion()}
                  size="small"
                  placeholder="Click keyboard to select"
                  fullWidth
                  InputProps={{
                    endAdornment: (
                      <IconButton
                        onClick={() => openPhonemePicker({ type: 'exclusion' })}
                        edge="end"
                        color="primary"
                        size="small"
                      >
                        <KeyboardIcon />
                      </IconButton>
                    ),
                  }}
                />
                <Button
                  variant="outlined"
                  startIcon={<AddIcon />}
                  onClick={addExclusion}
                >
                  Add
                </Button>
              </Stack>

              {excludePhonemes.length > 0 && (
                <Stack direction="row" spacing={1} flexWrap="wrap">
                  {excludePhonemes.map((phoneme) => (
                    <Chip
                      key={phoneme}
                      label={phoneme}
                      onDelete={() => removeExclusion(phoneme)}
                      color="error"
                      variant="outlined"
                    />
                  ))}
                </Stack>
              )}
            </Stack>
          </AccordionDetails>
        </Accordion>
      </Stack>

      {/* Actions */}
      <Stack direction="row" spacing={2} sx={{ mt: 3 }}>
        <Button
          variant="contained"
          size="large"
          startIcon={<BuildIcon />}
          onClick={handleBuild}
          disabled={loading || patterns.every((p) => !p.phoneme.trim())}
          fullWidth
        >
          Build Word List
        </Button>
        <Button
          variant="outlined"
          startIcon={<ClearIcon />}
          onClick={handleClear}
        >
          Clear
        </Button>
      </Stack>

      {/* Status Messages */}
      <Box sx={{ mt: 3 }}>
        {loading && (
          <Alert severity="info" icon={<CircularProgress size={20} />}>
            Building word list...
          </Alert>
        )}
        {error && <Alert severity="error">{error}</Alert>}
      </Box>

      {/* Results */}
      {results && !loading && (
        <Box sx={{ mt: 3 }}>
          <WordResultsDisplay results={results} />
        </Box>
      )}

      {/* Phoneme Picker Dialog */}
      <PhonemePickerDialog
        open={phonemePickerOpen}
        onClose={() => {
          setPhonemePickerOpen(false);
          setPhonemePickerTarget(null);
        }}
        onSelect={handlePhonemeSelect}
      />
    </Box>
  );
};

export default Builder;
