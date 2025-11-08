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
import WordListTable from './shared/WordListTable';
import PhonemePickerDialog from './PhonemePickerDialog';
import { validatePhonemeInput } from '../utils/ipaValidation';

const Builder: React.FC = () => {
  // Patterns state
  const [patterns, setPatterns] = useState<Pattern[]>([
    { type: 'STARTS_WITH', phoneme: '' },
  ]);

  // Store database ranges separately for slider min/max
  const [dbRanges, setDbRanges] = useState<Record<string, [number, number]>>({
    syllables: [1, 5],
    phonemes: [1, 10],
    wcm: [0, 15],
    msh: [1, 6],
    phono_prob_avg: [0, 1],
    frequency: [0, 1000],
    aoa: [2, 10],
    imageability: [1, 7],
    familiarity: [1, 7],
    concreteness: [1, 5],
    valence: [1, 9],
    arousal: [1, 9],
    dominance: [1, 9],
  });

  // Property filters state
  // Initial values are fallbacks - will be replaced with database values
  const [filters, setFilters] = useState({
    // Phonological Complexity
    syllables: [1, 5] as [number, number],
    phonemes: [1, 10] as [number, number],
    wcm: [0, 15] as [number, number],
    msh: [1, 6] as [number, number],
    phono_prob_avg: [0, 1] as [number, number],

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
          phono_prob_avg: ranges.phono_prob_avg as [number, number],
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

  // IPA validation warnings (one per pattern + one for exclusion)
  const [patternWarnings, setPatternWarnings] = useState<Map<number, string>>(new Map());
  const [exclusionWarning, setExclusionWarning] = useState<string | null>(null);

  // Handle phoneme selection
  const handlePhonemeSelect = (phoneme: string) => {
    if (phonemePickerTarget?.type === 'pattern') {
      // Append to existing phoneme value with auto-spacing
      const currentPattern = patterns[phonemePickerTarget.index];
      const newValue = currentPattern.phoneme
        ? currentPattern.phoneme + ' ' + phoneme
        : phoneme;
      updatePattern(phonemePickerTarget.index, 'phoneme', newValue);
    } else if (phonemePickerTarget?.type === 'exclusion') {
      // Append to exclusion input with auto-spacing
      setExcludePhonemeInput((prev) => prev ? prev + ' ' + phoneme : phoneme);
    }
    // Don't close - allow multiple selections
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
  const updatePattern = (index: number, field: keyof Pattern, value: string | boolean) => {
    const updated = [...patterns];
    updated[index] = { ...updated[index], [field]: value };
    setPatterns(updated);

    // Validate IPA input if updating phoneme field
    if (field === 'phoneme' && typeof value === 'string') {
      if (value.trim()) {
        const validation = validatePhonemeInput(value);
        if (!validation.isValid && validation.suggestion) {
          setPatternWarnings(prev => new Map(prev).set(index, validation.suggestion!));
        } else {
          setPatternWarnings(prev => {
            const next = new Map(prev);
            next.delete(index);
            return next;
          });
        }
      } else {
        setPatternWarnings(prev => {
          const next = new Map(prev);
          next.delete(index);
          return next;
        });
      }
    }
  };


  // Build word list
  const handleBuild = async () => {
    setLoading(true);
    setError(null);

    // Parse space-separated exclusions from input field
    const finalExclusions = excludePhonemeInput.trim()
      ? excludePhonemeInput.trim().split(/\s+/).filter(p => p)
      : [];

    try {
      // Only include filter values if they differ from the full DB range
      // This allows words with null values to be included when filters are at defaults
      const filtersObj: Record<string, number> = {};

      if (dbRanges) {
        // Syllables and phonemes (required properties - always filter)
        filtersObj.min_syllables = filters.syllables[0];
        filtersObj.max_syllables = filters.syllables[1];
        filtersObj.min_phonemes = filters.phonemes[0];
        filtersObj.max_phonemes = filters.phonemes[1];

        // Optional properties - only filter if not at full range
        if (filters.wcm[0] !== dbRanges.wcm[0] || filters.wcm[1] !== dbRanges.wcm[1]) {
          filtersObj.min_wcm = filters.wcm[0];
          filtersObj.max_wcm = filters.wcm[1];
        }
        if (filters.msh[0] !== dbRanges.msh[0] || filters.msh[1] !== dbRanges.msh[1]) {
          filtersObj.min_msh = filters.msh[0];
          filtersObj.max_msh = filters.msh[1];
        }
        if (filters.frequency[0] !== dbRanges.frequency[0] || filters.frequency[1] !== dbRanges.frequency[1]) {
          filtersObj.min_frequency = filters.frequency[0];
          filtersObj.max_frequency = filters.frequency[1];
        }
        if (filters.aoa[0] !== dbRanges.aoa[0] || filters.aoa[1] !== dbRanges.aoa[1]) {
          filtersObj.min_aoa = filters.aoa[0];
          filtersObj.max_aoa = filters.aoa[1];
        }
        if (filters.imageability[0] !== dbRanges.imageability[0] || filters.imageability[1] !== dbRanges.imageability[1]) {
          filtersObj.min_imageability = filters.imageability[0];
          filtersObj.max_imageability = filters.imageability[1];
        }
        if (filters.familiarity[0] !== dbRanges.familiarity[0] || filters.familiarity[1] !== dbRanges.familiarity[1]) {
          filtersObj.min_familiarity = filters.familiarity[0];
          filtersObj.max_familiarity = filters.familiarity[1];
        }
        if (filters.concreteness[0] !== dbRanges.concreteness[0] || filters.concreteness[1] !== dbRanges.concreteness[1]) {
          filtersObj.min_concreteness = filters.concreteness[0];
          filtersObj.max_concreteness = filters.concreteness[1];
        }
        if (filters.valence[0] !== dbRanges.valence[0] || filters.valence[1] !== dbRanges.valence[1]) {
          filtersObj.min_valence = filters.valence[0];
          filtersObj.max_valence = filters.valence[1];
        }
        if (filters.arousal[0] !== dbRanges.arousal[0] || filters.arousal[1] !== dbRanges.arousal[1]) {
          filtersObj.min_arousal = filters.arousal[0];
          filtersObj.max_arousal = filters.arousal[1];
        }
        if (filters.dominance[0] !== dbRanges.dominance[0] || filters.dominance[1] !== dbRanges.dominance[1]) {
          filtersObj.min_dominance = filters.dominance[0];
          filtersObj.max_dominance = filters.dominance[1];
        }
        if (filters.phono_prob_avg[0] !== dbRanges.phono_prob_avg[0] || filters.phono_prob_avg[1] !== dbRanges.phono_prob_avg[1]) {
          filtersObj.min_phono_prob_avg = filters.phono_prob_avg[0];
          filtersObj.max_phono_prob_avg = filters.phono_prob_avg[1];
        }
      }

      const request: BuilderRequest = {
        patterns: patterns.filter((p) => p.phoneme.trim() !== ''),
        filters: filtersObj,
        exclusions: {
          exclude_phonemes: finalExclusions.length > 0 ? finalExclusions : undefined,
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
    setPatterns([{ type: 'STARTS_WITH', phoneme: '' }]);
    setFilters({
      syllables: dbRanges.syllables as [number, number],
      phonemes: dbRanges.phonemes as [number, number],
      wcm: dbRanges.wcm as [number, number],
      msh: dbRanges.msh as [number, number],
      phono_prob_avg: dbRanges.phono_prob_avg as [number, number],
      frequency: dbRanges.frequency as [number, number],
      aoa: dbRanges.aoa as [number, number],
      imageability: dbRanges.imageability as [number, number],
      familiarity: dbRanges.familiarity as [number, number],
      concreteness: dbRanges.concreteness as [number, number],
      valence: dbRanges.valence as [number, number],
      arousal: dbRanges.arousal as [number, number],
      dominance: dbRanges.dominance as [number, number],
    });
    setExcludePhonemeInput('');
    setResults(null);
    setError(null);
  };

  return (
    <Box>
      <Stack spacing={{ xs: 1.5, sm: 2 }}>
        {/* Pattern Matching */}
        <Accordion defaultExpanded>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="h6" sx={{ fontSize: { xs: '1rem', sm: '1.25rem' } }}>
              Patterns
            </Typography>
          </AccordionSummary>
          <AccordionDetails sx={{ px: { xs: 1.5, sm: 2 }, py: { xs: 1, sm: 2 } }}>
            <Stack spacing={{ xs: 1.5, sm: 2 }}>
              <Typography variant="body2" color="text.secondary" sx={{ fontSize: { xs: '0.8125rem', sm: '0.875rem' } }}>
                AND logic: all patterns must match. Space-separate phonemes (e.g., "s t" for /st/).
              </Typography>

              <Stack spacing={{ xs: 1.5, sm: 2 }}>
                {patterns.map((pattern, idx) => (
                  <Paper key={idx} variant="outlined" sx={{ p: { xs: 1.5, sm: 2 } }}>
                    <Stack spacing={{ xs: 1, sm: 1 }}>
                      <Stack
                        direction={{ xs: 'column', sm: 'row' }}
                        spacing={{ xs: 1, sm: 2 }}
                        alignItems={{ xs: 'stretch', sm: 'center' }}
                      >
                        <FormControl size="small" sx={{ minWidth: { xs: '100%', sm: 140 } }}>
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

                        <Box sx={{ display: 'flex', gap: 1, flex: 1, flexDirection: 'column' }}>
                          <Box sx={{ display: 'flex', gap: 1 }}>
                            <TextField
                              label="Phoneme(s)"
                              value={pattern.phoneme}
                              onChange={(e) => updatePattern(idx, 'phoneme', e.target.value)}
                              size="small"
                              placeholder="Use IPA →"
                              fullWidth
                              InputProps={{
                                endAdornment: (
                                  <IconButton
                                    onClick={() => openPhonemePicker({ type: 'pattern', index: idx })}
                                    edge="end"
                                    color="primary"
                                    size="small"
                                    sx={{ minWidth: { xs: 48, sm: 44 }, minHeight: { xs: 48, sm: 44 } }}
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
                              sx={{ minWidth: 44, minHeight: 44, flexShrink: 0 }}
                            >
                              <DeleteIcon />
                            </IconButton>
                          </Box>
                          {patternWarnings.has(idx) && (
                            <Alert severity="warning" sx={{ mt: 0.5 }}>
                              {patternWarnings.get(idx)}
                            </Alert>
                          )}
                        </Box>
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
              <Typography variant="h6" sx={{ fontSize: { xs: '1rem', sm: '1.25rem' } }}>
                Property Filters
              </Typography>
              <Typography variant="caption" color="text.secondary" sx={{ fontSize: { xs: '0.75rem', sm: '0.8125rem' } }}>
                Phonological, lexical, semantic, and affective properties
              </Typography>
            </Box>
          </AccordionSummary>
          <AccordionDetails sx={{ px: { xs: 1.5, sm: 2 }, py: { xs: 1, sm: 2 } }}>
            <Stack spacing={{ xs: 0.75, sm: 1 }}>
                {/* Phonological Complexity */}
                <Accordion defaultExpanded>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography variant="subtitle2" fontWeight={600} sx={{ fontSize: { xs: '0.875rem', sm: '0.9375rem' } }}>
                      Phonological Complexity
                    </Typography>
                  </AccordionSummary>
                  <AccordionDetails sx={{ px: { xs: 1.5, sm: 2 }, py: { xs: 1.5, sm: 2 } }}>
                    <Stack spacing={{ xs: 2.5, sm: 3 }}>
                      {/* Syllables */}
                      <Box>
                        <Typography variant="body2" gutterBottom sx={{ fontSize: { xs: '0.8125rem', sm: '0.875rem' } }}>
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
                          sx={{ '& .MuiSlider-markLabel': { fontSize: { xs: '0.625rem', sm: '0.75rem' } } }}
                        />
                      </Box>

                      {/* Phonemes */}
                      <Box>
                        <Typography variant="body2" gutterBottom sx={{ fontSize: { xs: '0.8125rem', sm: '0.875rem' } }}>
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
                          sx={{ '& .MuiSlider-markLabel': { fontSize: { xs: '0.625rem', sm: '0.75rem' } } }}
                        />
                      </Box>

                      {/* WCM */}
                      <Box>
                        <Typography variant="body2" gutterBottom sx={{ fontSize: { xs: '0.8125rem', sm: '0.875rem' } }}>
                          WCM Score: {filters.wcm[0]} - {filters.wcm[1]}
                          <Typography variant="caption" color="text.secondary" display="block" sx={{ fontSize: { xs: '0.6875rem', sm: '0.75rem' } }}>
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
                          sx={{ '& .MuiSlider-markLabel': { fontSize: { xs: '0.625rem', sm: '0.75rem' } } }}
                        />
                      </Box>

                      {/* MSH */}
                      <Box>
                        <Typography variant="body2" gutterBottom sx={{ fontSize: { xs: '0.8125rem', sm: '0.875rem' } }}>
                          MSH Stage: {filters.msh[0]} - {filters.msh[1]}
                          <Typography variant="caption" color="text.secondary" display="block" sx={{ fontSize: { xs: '0.6875rem', sm: '0.75rem' } }}>
                            Motor Speech Hierarchy (Namasivayam et al., 2021)
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
                          sx={{ '& .MuiSlider-markLabel': { fontSize: { xs: '0.625rem', sm: '0.75rem' } } }}
                        />
                      </Box>

                      {/* Phonotactic Probability */}
                      <Box>
                        <Typography variant="body2" gutterBottom sx={{ fontSize: { xs: '0.8125rem', sm: '0.875rem' } }}>
                          Phonotactic Probability: {filters.phono_prob_avg[0].toFixed(3)} - {filters.phono_prob_avg[1].toFixed(3)}
                          <Typography variant="caption" color="text.secondary" display="block" sx={{ fontSize: { xs: '0.6875rem', sm: '0.75rem' } }}>
                            Sound sequence typicality (Vitevitch & Luce, 2004)
                          </Typography>
                        </Typography>
                        <Slider
                          value={filters.phono_prob_avg}
                          onChange={(_, value) => handleFilterChange('phono_prob_avg', value as [number, number])}
                          min={dbRanges.phono_prob_avg[0]}
                          max={dbRanges.phono_prob_avg[1]}
                          step={0.001}
                          valueLabelDisplay="auto"
                          valueLabelFormat={(value) => value.toFixed(3)}
                          sx={{ '& .MuiSlider-markLabel': { fontSize: { xs: '0.625rem', sm: '0.75rem' } } }}
                        />
                      </Box>
                    </Stack>
                  </AccordionDetails>
                </Accordion>

                {/* Lexical Properties */}
                <Accordion>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography variant="subtitle2" fontWeight={600} sx={{ fontSize: { xs: '0.875rem', sm: '0.9375rem' } }}>
                      Lexical Properties
                    </Typography>
                  </AccordionSummary>
                  <AccordionDetails sx={{ px: { xs: 1.5, sm: 2 }, py: { xs: 1.5, sm: 2 } }}>
                    <Stack spacing={{ xs: 2.5, sm: 3 }}>
                      {/* Frequency */}
                      <Box>
                        <Typography variant="body2" gutterBottom sx={{ fontSize: { xs: '0.8125rem', sm: '0.875rem' } }}>
                          Frequency: {Math.round(filters.frequency[0])} - {Math.round(filters.frequency[1])}
                          <Typography variant="caption" color="text.secondary" display="block" sx={{ fontSize: { xs: '0.6875rem', sm: '0.75rem' } }}>
                            SUBTLEX-US (per million words, log scale)
                          </Typography>
                        </Typography>
                        <Slider
                          value={[
                            filters.frequency[0] > 0 ? Math.log10(filters.frequency[0]) : 0,
                            Math.log10(filters.frequency[1])
                          ]}
                          onChange={(_, value) => {
                            const [minLog, maxLog] = value as [number, number];
                            handleFilterChange('frequency', [
                              minLog > 0 ? Math.pow(10, minLog) : 0,
                              Math.pow(10, maxLog)
                            ]);
                          }}
                          min={0}
                          max={Math.log10(dbRanges.frequency[1])}
                          step={0.01}
                          valueLabelDisplay="auto"
                          valueLabelFormat={(value) => Math.round(Math.pow(10, value)).toString()}
                          marks={[
                            { value: 0, label: '0' },
                            { value: 1, label: '10' },
                            { value: 2, label: '100' },
                            { value: 3, label: '1K' },
                            { value: 4, label: '10K' },
                          ].filter(mark => mark.value <= Math.log10(dbRanges.frequency[1]))}
                          sx={{ '& .MuiSlider-markLabel': { fontSize: { xs: '0.625rem', sm: '0.75rem' } } }}
                        />
                      </Box>

                      {/* Age of Acquisition */}
                      <Box>
                        <Typography variant="body2" gutterBottom sx={{ fontSize: { xs: '0.8125rem', sm: '0.875rem' } }}>
                          Age of Acquisition: {filters.aoa[0]} - {filters.aoa[1]}
                          <Typography variant="caption" color="text.secondary" display="block" sx={{ fontSize: { xs: '0.6875rem', sm: '0.75rem' } }}>
                            Glasgow Norms (1-7: 1=earliest, 7=latest)
                          </Typography>
                        </Typography>
                        <Slider
                          value={filters.aoa}
                          onChange={(_, value) => handleFilterChange('aoa', value as [number, number])}
                          min={dbRanges.aoa[0]}
                          max={dbRanges.aoa[1]}
                          step={0.5}
                          valueLabelDisplay="auto"
                          sx={{ '& .MuiSlider-markLabel': { fontSize: { xs: '0.625rem', sm: '0.75rem' } } }}
                        />
                      </Box>
                    </Stack>
                  </AccordionDetails>
                </Accordion>

                {/* Semantic Properties */}
                <Accordion>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography variant="subtitle2" fontWeight={600} sx={{ fontSize: { xs: '0.875rem', sm: '0.9375rem' } }}>
                      Semantic Properties
                    </Typography>
                  </AccordionSummary>
                  <AccordionDetails sx={{ px: { xs: 1.5, sm: 2 }, py: { xs: 1.5, sm: 2 } }}>
                    <Stack spacing={{ xs: 2.5, sm: 3 }}>
                      {/* Imageability */}
                      <Box>
                        <Typography variant="body2" gutterBottom sx={{ fontSize: { xs: '0.8125rem', sm: '0.875rem' } }}>
                          Imageability: {filters.imageability[0]} - {filters.imageability[1]}
                          <Typography variant="caption" color="text.secondary" display="block" sx={{ fontSize: { xs: '0.6875rem', sm: '0.75rem' } }}>
                            Glasgow Norms (1-7: ease of mental imagery)
                          </Typography>
                        </Typography>
                        <Slider
                          value={filters.imageability}
                          onChange={(_, value) => handleFilterChange('imageability', value as [number, number])}
                          min={dbRanges.imageability[0]}
                          max={dbRanges.imageability[1]}
                          step={0.5}
                          valueLabelDisplay="auto"
                          sx={{ '& .MuiSlider-markLabel': { fontSize: { xs: '0.625rem', sm: '0.75rem' } } }}
                        />
                      </Box>

                      {/* Familiarity */}
                      <Box>
                        <Typography variant="body2" gutterBottom sx={{ fontSize: { xs: '0.8125rem', sm: '0.875rem' } }}>
                          Familiarity: {filters.familiarity[0]} - {filters.familiarity[1]}
                          <Typography variant="caption" color="text.secondary" display="block" sx={{ fontSize: { xs: '0.6875rem', sm: '0.75rem' } }}>
                            Glasgow Norms (1-7: word familiarity)
                          </Typography>
                        </Typography>
                        <Slider
                          value={filters.familiarity}
                          onChange={(_, value) => handleFilterChange('familiarity', value as [number, number])}
                          min={dbRanges.familiarity[0]}
                          max={dbRanges.familiarity[1]}
                          step={0.5}
                          valueLabelDisplay="auto"
                          sx={{ '& .MuiSlider-markLabel': { fontSize: { xs: '0.625rem', sm: '0.75rem' } } }}
                        />
                      </Box>

                      {/* Concreteness */}
                      <Box>
                        <Typography variant="body2" gutterBottom sx={{ fontSize: { xs: '0.8125rem', sm: '0.875rem' } }}>
                          Concreteness: {filters.concreteness[0]} - {filters.concreteness[1]}
                          <Typography variant="caption" color="text.secondary" display="block" sx={{ fontSize: { xs: '0.6875rem', sm: '0.75rem' } }}>
                            Brysbaert et al. (1-5: concrete vs. abstract)
                          </Typography>
                        </Typography>
                        <Slider
                          value={filters.concreteness}
                          onChange={(_, value) => handleFilterChange('concreteness', value as [number, number])}
                          min={dbRanges.concreteness[0]}
                          max={dbRanges.concreteness[1]}
                          step={0.5}
                          valueLabelDisplay="auto"
                          sx={{ '& .MuiSlider-markLabel': { fontSize: { xs: '0.625rem', sm: '0.75rem' } } }}
                        />
                      </Box>
                    </Stack>
                  </AccordionDetails>
                </Accordion>

                {/* Affective Properties */}
                <Accordion>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography variant="subtitle2" fontWeight={600} sx={{ fontSize: { xs: '0.875rem', sm: '0.9375rem' } }}>
                      Affective Properties
                    </Typography>
                  </AccordionSummary>
                  <AccordionDetails sx={{ px: { xs: 1.5, sm: 2 }, py: { xs: 1.5, sm: 2 } }}>
                    <Stack spacing={{ xs: 2.5, sm: 3 }}>
                      {/* Valence */}
                      <Box>
                        <Typography variant="body2" gutterBottom sx={{ fontSize: { xs: '0.8125rem', sm: '0.875rem' } }}>
                          Valence: {filters.valence[0]} - {filters.valence[1]}
                          <Typography variant="caption" color="text.secondary" display="block" sx={{ fontSize: { xs: '0.6875rem', sm: '0.75rem' } }}>
                            Warriner et al. (1-9: negative to positive)
                          </Typography>
                        </Typography>
                        <Slider
                          value={filters.valence}
                          onChange={(_, value) => handleFilterChange('valence', value as [number, number])}
                          min={dbRanges.valence[0]}
                          max={dbRanges.valence[1]}
                          step={0.5}
                          valueLabelDisplay="auto"
                          sx={{ '& .MuiSlider-markLabel': { fontSize: { xs: '0.625rem', sm: '0.75rem' } } }}
                        />
                      </Box>

                      {/* Arousal */}
                      <Box>
                        <Typography variant="body2" gutterBottom sx={{ fontSize: { xs: '0.8125rem', sm: '0.875rem' } }}>
                          Arousal: {filters.arousal[0]} - {filters.arousal[1]}
                          <Typography variant="caption" color="text.secondary" display="block" sx={{ fontSize: { xs: '0.6875rem', sm: '0.75rem' } }}>
                            Warriner et al. (1-9: calm to excited)
                          </Typography>
                        </Typography>
                        <Slider
                          value={filters.arousal}
                          onChange={(_, value) => handleFilterChange('arousal', value as [number, number])}
                          min={dbRanges.arousal[0]}
                          max={dbRanges.arousal[1]}
                          step={0.5}
                          valueLabelDisplay="auto"
                          sx={{ '& .MuiSlider-markLabel': { fontSize: { xs: '0.625rem', sm: '0.75rem' } } }}
                        />
                      </Box>

                      {/* Dominance */}
                      <Box>
                        <Typography variant="body2" gutterBottom sx={{ fontSize: { xs: '0.8125rem', sm: '0.875rem' } }}>
                          Dominance: {filters.dominance[0]} - {filters.dominance[1]}
                          <Typography variant="caption" color="text.secondary" display="block" sx={{ fontSize: { xs: '0.6875rem', sm: '0.75rem' } }}>
                            Warriner et al. (1-9: weak to powerful)
                          </Typography>
                        </Typography>
                        <Slider
                          value={filters.dominance}
                          onChange={(_, value) => handleFilterChange('dominance', value as [number, number])}
                          min={dbRanges.dominance[0]}
                          max={dbRanges.dominance[1]}
                          step={0.5}
                          valueLabelDisplay="auto"
                          sx={{ '& .MuiSlider-markLabel': { fontSize: { xs: '0.625rem', sm: '0.75rem' } } }}
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
              <Typography variant="h6" sx={{ fontSize: { xs: '1rem', sm: '1.25rem' } }}>
                Exclusions
              </Typography>
              <Typography variant="caption" color="text.secondary" sx={{ fontSize: { xs: '0.75rem', sm: '0.8125rem' } }}>
                Exclude words containing specific phonemes
              </Typography>
            </Box>
          </AccordionSummary>
          <AccordionDetails sx={{ px: { xs: 1.5, sm: 2 }, py: { xs: 1, sm: 2 } }}>
            <Stack spacing={{ xs: 1.5, sm: 2 }}>
              <Stack spacing={{ xs: 1, sm: 0 }}>
                <TextField
                  label="Phonemes to exclude"
                  value={excludePhonemeInput}
                  onChange={(e) => {
                    const newValue = e.target.value;
                    setExcludePhonemeInput(newValue);

                    // Validate IPA input
                    if (newValue.trim()) {
                      const validation = validatePhonemeInput(newValue);
                      if (!validation.isValid && validation.suggestion) {
                        setExclusionWarning(validation.suggestion);
                      } else {
                        setExclusionWarning(null);
                      }
                    } else {
                      setExclusionWarning(null);
                    }
                  }}
                  size="small"
                  placeholder="Use IPA →"
                  fullWidth
                  InputProps={{
                    endAdornment: (
                      <IconButton
                        onClick={() => openPhonemePicker({ type: 'exclusion' })}
                        edge="end"
                        color="primary"
                        size="small"
                        sx={{ minWidth: { xs: 48, sm: 44 }, minHeight: { xs: 48, sm: 44 } }}
                      >
                        <KeyboardIcon />
                      </IconButton>
                    ),
                  }}
                />
                {exclusionWarning && (
                  <Alert severity="warning" sx={{ mt: 1 }}>
                    {exclusionWarning}
                  </Alert>
                )}
              </Stack>
            </Stack>
          </AccordionDetails>
        </Accordion>
      </Stack>

      {/* Actions */}
      <Stack
        direction={{ xs: 'column', sm: 'row' }}
        spacing={{ xs: 1.5, sm: 2 }}
        sx={{ mt: { xs: 2, sm: 3 } }}
      >
        <Button
          variant="contained"
          size="large"
          startIcon={<BuildIcon />}
          onClick={handleBuild}
          disabled={loading}
          fullWidth
          sx={{ minHeight: 48 }}
        >
          Build Word List
        </Button>
        <Button
          variant="outlined"
          size="large"
          startIcon={<ClearIcon />}
          onClick={handleClear}
          sx={{
            minHeight: 48,
            minWidth: { sm: 120 },
            width: { xs: '100%', sm: 'auto' },
          }}
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
          <WordListTable
            words={results}
            showSimilarity={false}
            enableSelection={true}
            defaultSort="word"
            exportFilename="phonolex_custom_word_list.csv"
          />
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
