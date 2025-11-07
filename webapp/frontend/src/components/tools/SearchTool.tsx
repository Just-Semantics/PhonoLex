/**
 * Lookup Tool Component
 *
 * Informational lookup for words and phonemes, including phoneme comparison
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
  Card,
  CardContent,
  Grid,
  Chip,
  Divider,
  ToggleButtonGroup,
  ToggleButton,
  Table,
  TableBody,
  TableRow,
  TableCell,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  IconButton,
  TableContainer,
  TableHead,
  Paper,
} from '@mui/material';
import {
  Search as SearchIcon,
  TextFields as WordIcon,
  GraphicEq as PhonemeIcon,
  FilterList as FilterIcon,
  Add as AddIcon,
  Remove as RemoveIcon,
  Keyboard as KeyboardIcon,
  SwapHoriz as SwapIcon,
} from '@mui/icons-material';
import api from '../../services/phonolexApi';
import type { Word, PhonemeComparison } from '../../services/phonolexApi';
import PhonemePickerDialog from '../PhonemePickerDialog';
import { validatePhonemeInput } from '../../utils/ipaValidation';

type SearchMode = 'word' | 'phoneme' | 'phoneme-features';

interface PhonemeResult {
  phoneme: string;
  type: 'vowel' | 'consonant';
  features: Record<string, string>;
}

interface PhonemeSearchResult {
  features: Record<string, string>;
  matching_phonemes: string[];
  count: number;
}

interface SimilarityResult {
  word: string;
  ipa: string;
  similarity: number;
  syllable_count: number;
  wcm_score: number | null;
}

const SearchTool: React.FC = () => {
  const [mode, setMode] = useState<SearchMode>('word');
  const [query, setQuery] = useState('');
  const [query2, setQuery2] = useState(''); // Second phoneme for comparison
  const [wordResult, setWordResult] = useState<Word | null>(null);
  const [phonemeResult, setPhonemeResult] = useState<PhonemeResult | null>(null);
  const [phoneme2Result, setPhoneme2Result] = useState<PhonemeResult | null>(null);
  const [comparisonResult, setComparisonResult] = useState<PhonemeComparison | null>(null);
  const [phonemeSearchResult, setPhonemeSearchResult] = useState<PhonemeSearchResult | null>(null);
  const [similarityResults, setSimilarityResults] = useState<SimilarityResult[] | null>(null);
  const [availableFeatures, setAvailableFeatures] = useState<string[]>([]);
  const [featureFilters, setFeatureFilters] = useState<Array<{ feature: string; value: string }>>([
    { feature: '', value: '' }
  ]);
  const [similarityThreshold] = useState(0.85);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [phonemePickerOpen, setPhonemePickerOpen] = useState(false);
  const [activePhonemeField, setActivePhonemeField] = useState<'phoneme1' | 'phoneme2'>('phoneme1');
  const [ipaWarning, setIpaWarning] = useState<string | null>(null);
  const [ipaWarning2, setIpaWarning2] = useState<string | null>(null);

  // Load available features from phoneme data on mount
  React.useEffect(() => {
    const loadFeatures = async () => {
      try {
        const phonemeList = await api.listPhonemes();
        // Extract feature names that have non-zero values ('+' or '-') for at least one English phoneme
        const featureValues = new Map<string, Set<string>>();

        phonemeList.phonemes.forEach(p => {
          Object.entries(p.features || {}).forEach(([feat, value]) => {
            if (!featureValues.has(feat)) {
              featureValues.set(feat, new Set());
            }
            featureValues.get(feat)!.add(value);
          });
        });

        // Filter to features that have at least one non-'0' value
        const relevantFeatures = Array.from(featureValues.entries())
          .filter(([_feat, values]) => values.has('+') || values.has('-'))
          .map(([feat, _values]) => feat)
          .sort();

        setAvailableFeatures(relevantFeatures);
      } catch (err) {
        console.error('Failed to load phoneme features:', err);
        // Fallback to a basic set if API fails
        setAvailableFeatures(['consonantal', 'sonorant', 'voice', 'nasal', 'continuant']);
      }
    };
    loadFeatures();
  }, []);

  const handleSearch = async () => {
    setLoading(true);
    setError(null);
    setWordResult(null);
    setPhonemeResult(null);
    setPhoneme2Result(null);
    setComparisonResult(null);
    setPhonemeSearchResult(null);
    setSimilarityResults(null);

    try {
      if (mode === 'word') {
        if (!query.trim()) {
          setError('Please enter a word');
          return;
        }
        const wordData = await api.getWord(query.trim().toLowerCase());
        setWordResult(wordData);

        // Also fetch similar words
        try {
          const similarData = await api.findSimilarWords(query.trim().toLowerCase(), { threshold: similarityThreshold, limit: 20 });
          setSimilarityResults(similarData.map(r => ({
            word: r.word.word,
            ipa: r.word.ipa,
            similarity: r.similarity,
            syllable_count: r.word.syllable_count,
            wcm_score: r.word.wcm_score
          })));
        } catch (err) {
          // Don't fail the whole search if similarity fails
          console.error('Failed to fetch similar words:', err);
        }
      } else if (mode === 'phoneme') {
        if (!query.trim()) {
          setError('Please enter a phoneme (IPA)');
          return;
        }

        // Get first phoneme
        const p1Raw = await api.getPhoneme(query.trim());
        setPhonemeResult(p1Raw);

        // If second phoneme is provided, get it and compare
        if (query2.trim()) {
          const p2Raw = await api.getPhoneme(query2.trim());
          setPhoneme2Result(p2Raw);

          // Get comparison
          const comp = await api.comparePhonemes(query.trim(), query2.trim());
          setComparisonResult(comp as unknown as PhonemeComparison);
        }
      } else if (mode === 'phoneme-features') {
        // phoneme-features mode
        const validFilters = featureFilters.filter(f => f.feature && f.value);
        if (validFilters.length === 0) {
          setError('Please select at least one feature filter');
          return;
        }
        const featuresObj = Object.fromEntries(
          validFilters.map(f => [f.feature, f.value])
        );
        const data = await api.searchPhonemesByFeatures(featuresObj);
        setPhonemeSearchResult(data);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed');
    } finally {
      setLoading(false);
    }
  };

  const handleModeChange = (_: React.MouseEvent<HTMLElement>, newMode: SearchMode | null) => {
    if (newMode !== null) {
      setMode(newMode);
      setQuery('');
      setQuery2('');
      setWordResult(null);
      setPhonemeResult(null);
      setPhoneme2Result(null);
      setComparisonResult(null);
      setPhonemeSearchResult(null);
      setSimilarityResults(null);
      setError(null);
    }
  };

  const addFeatureFilter = () => {
    setFeatureFilters([...featureFilters, { feature: '', value: '' }]);
  };

  const removeFeatureFilter = (index: number) => {
    setFeatureFilters(featureFilters.filter((_, i) => i !== index));
  };

  const updateFeatureFilter = (index: number, field: 'feature' | 'value', value: string) => {
    const updated = [...featureFilters];
    updated[index][field] = value;
    setFeatureFilters(updated);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  const handleSwap = () => {
    const temp = query;
    setQuery(query2);
    setQuery2(temp);

    if (phonemeResult && phoneme2Result) {
      const tempPhoneme = phonemeResult;
      setPhonemeResult(phoneme2Result);
      setPhoneme2Result(tempPhoneme);
    }
  };

  const openPhonemePicker = (field: 'phoneme1' | 'phoneme2') => {
    setActivePhonemeField(field);
    setPhonemePickerOpen(true);
  };

  const handlePhonemeSelect = (phoneme: string) => {
    if (activePhonemeField === 'phoneme1') {
      setQuery((prev) => prev + phoneme);
    } else {
      setQuery2((prev) => prev + phoneme);
    }
  };

  // Get feature comparison data
  const getFeatureComparison = () => {
    if (!phonemeResult || !phoneme2Result || !phonemeResult.features || !phoneme2Result.features) return [];

    const allFeatures = new Set([
      ...Object.keys(phonemeResult.features),
      ...Object.keys(phoneme2Result.features),
    ]);

    return Array.from(allFeatures).map((feature) => ({
      feature,
      phoneme1Value: phonemeResult.features?.[feature] || '',
      phoneme2Value: phoneme2Result.features?.[feature] || '',
      match: phonemeResult.features?.[feature] === phoneme2Result.features?.[feature],
    }));
  };

  return (
    <Box>
      <Stack spacing={2}>
        {/* Mode Toggle */}
        <ToggleButtonGroup
          value={mode}
          exclusive
          onChange={handleModeChange}
          fullWidth
          color="primary"
        >
          <ToggleButton value="word">
            <WordIcon sx={{ mr: 0.5, fontSize: '1.1rem' }} />
            <Typography variant="body2">Word</Typography>
          </ToggleButton>
          <ToggleButton value="phoneme">
            <PhonemeIcon sx={{ mr: 0.5, fontSize: '1.1rem' }} />
            <Typography variant="body2">Phoneme</Typography>
          </ToggleButton>
          <ToggleButton value="phoneme-features">
            <FilterIcon sx={{ mr: 0.5, fontSize: '1.1rem' }} />
            <Typography variant="body2">Features</Typography>
          </ToggleButton>
        </ToggleButtonGroup>

        {/* Word Search Input */}
        {mode === 'word' && (
          <Box sx={{ position: 'relative' }}>
            <TextField
              label="Enter a word"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={handleKeyPress}
              size="medium"
              placeholder="e.g., cat, computer, beautiful"
              fullWidth
              autoFocus
            />
          </Box>
        )}

        {/* Phoneme Search Inputs (with optional second phoneme for comparison) */}
        {mode === 'phoneme' && (
          <Stack spacing={2}>
            <Box sx={{ position: 'relative' }}>
              <TextField
                label="Select or type phoneme (IPA)"
                value={query}
                onChange={(e) => {
                  const newValue = e.target.value;
                  setQuery(newValue);

                  // Validate IPA input
                  if (newValue.trim()) {
                    const validation = validatePhonemeInput(newValue);
                    if (!validation.isValid && validation.suggestion) {
                      setIpaWarning(validation.suggestion);
                    } else {
                      setIpaWarning(null);
                    }
                  } else {
                    setIpaWarning(null);
                  }
                }}
                onKeyPress={handleKeyPress}
                size="medium"
                placeholder="Use keyboard icon → to select IPA"
                fullWidth
                autoFocus
                InputProps={{
                  endAdornment: (
                    <IconButton
                      onClick={() => openPhonemePicker('phoneme1')}
                      edge="end"
                      color="primary"
                    >
                      <KeyboardIcon />
                    </IconButton>
                  )
                }}
              />

              {/* IPA Warning */}
              {ipaWarning && (
                <Alert severity="warning" sx={{ mt: 1 }}>
                  {ipaWarning}
                </Alert>
              )}
            </Box>

            {/* Optional second phoneme for comparison */}
            <Box>
              <Typography variant="caption" color="text.secondary" sx={{ mb: 0.5, display: 'block' }}>
                Optional: Compare with second phoneme
              </Typography>
              <Stack direction="row" spacing={1} alignItems="center">
                <TextField
                  label="Second phoneme (optional)"
                  value={query2}
                  onChange={(e) => {
                    const newValue = e.target.value;
                    setQuery2(newValue);

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
                  onKeyPress={handleKeyPress}
                  size="medium"
                  placeholder="Use keyboard icon → to select IPA"
                  fullWidth
                  InputProps={{
                    endAdornment: (
                      <IconButton
                        onClick={() => openPhonemePicker('phoneme2')}
                        edge="end"
                        color="primary"
                      >
                        <KeyboardIcon />
                      </IconButton>
                    )
                  }}
                />
                {query2.trim() && (
                  <IconButton
                    onClick={handleSwap}
                    color="primary"
                    size="small"
                    title="Swap phonemes"
                  >
                    <SwapIcon />
                  </IconButton>
                )}
              </Stack>

              {/* IPA Warning for second phoneme */}
              {ipaWarning2 && (
                <Alert severity="warning" sx={{ mt: 1 }}>
                  {ipaWarning2}
                </Alert>
              )}
            </Box>
          </Stack>
        )}

        {/* Feature Filters */}
        {mode === 'phoneme-features' && (
          <Box>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
              Search for phonemes by Phoible features
            </Typography>
            {featureFilters.map((filter, index) => (
              <Stack key={index} direction="row" spacing={1} sx={{ mb: 1 }}>
                <FormControl size="small" sx={{ flex: 1 }}>
                  <InputLabel>Feature</InputLabel>
                  <Select
                    value={filter.feature}
                    label="Feature"
                    onChange={(e) => updateFeatureFilter(index, 'feature', e.target.value)}
                  >
                    {availableFeatures.map(feat => (
                      <MenuItem key={feat} value={feat}>{feat}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <FormControl size="small" sx={{ width: 100 }}>
                  <InputLabel>Value</InputLabel>
                  <Select
                    value={filter.value}
                    label="Value"
                    onChange={(e) => updateFeatureFilter(index, 'value', e.target.value)}
                  >
                    <MenuItem value="+">+</MenuItem>
                    <MenuItem value="-">-</MenuItem>
                    <MenuItem value="0">0</MenuItem>
                  </Select>
                </FormControl>
                <IconButton
                  onClick={() => removeFeatureFilter(index)}
                  disabled={featureFilters.length === 1}
                  size="small"
                >
                  <RemoveIcon />
                </IconButton>
              </Stack>
            ))}
            <Button
              startIcon={<AddIcon />}
              onClick={addFeatureFilter}
              size="small"
              variant="outlined"
            >
              Add Feature
            </Button>
          </Box>
        )}

        <Button
          variant="contained"
          startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <SearchIcon />}
          onClick={handleSearch}
          disabled={loading}
          fullWidth
        >
          {loading ? 'Searching...' : mode === 'phoneme' && query2.trim() ? 'Compare Phonemes' : 'Lookup'}
        </Button>
      </Stack>

      {error && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {error}
        </Alert>
      )}

      {/* Word Result */}
      {wordResult && (
        <Card sx={{ mt: 3 }} elevation={2}>
          <CardContent>
            <Typography variant="h5" gutterBottom fontWeight={600}>
              {wordResult.word}
            </Typography>
            <Typography variant="h6" color="text.secondary" fontFamily="monospace" gutterBottom>
              /{wordResult.ipa}/
            </Typography>

            <Divider sx={{ my: 2 }} />

            {/* Phonological Structure */}
            <Box sx={{ mb: 3 }}>
              <Typography variant="subtitle2" fontWeight={600} gutterBottom>
                Phonological Structure
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={6} sm={3}>
                  <Typography variant="caption" color="text.secondary">Phonemes</Typography>
                  <Typography variant="body1">{wordResult.phoneme_count}</Typography>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Typography variant="caption" color="text.secondary">Syllables</Typography>
                  <Typography variant="body1">{wordResult.syllable_count}</Typography>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Typography variant="caption" color="text.secondary">WCM Score</Typography>
                  <Typography variant="body1">{wordResult.wcm_score || 'N/A'}</Typography>
                </Grid>
              </Grid>
            </Box>

            <Divider sx={{ my: 2 }} />

            {/* Lexical Properties */}
            <Box sx={{ mb: 3 }}>
              <Typography variant="subtitle2" fontWeight={600} gutterBottom>
                Lexical Properties
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={6} sm={4}>
                  <Typography variant="caption" color="text.secondary">Frequency</Typography>
                  <Typography variant="body1">
                    {wordResult.frequency ? `${wordResult.frequency.toFixed(1)} per million` : '-'}
                  </Typography>
                </Grid>
                <Grid item xs={6} sm={4}>
                  <Typography variant="caption" color="text.secondary">
                    Age of Acquisition (1-7)
                  </Typography>
                  <Typography variant="body1">
                    {wordResult.aoa ? wordResult.aoa.toFixed(1) : '-'}
                  </Typography>
                </Grid>
              </Grid>
            </Box>

            <Divider sx={{ my: 2 }} />

            {/* Semantic Properties */}
            <Box sx={{ mb: 3 }}>
              <Typography variant="subtitle2" fontWeight={600} gutterBottom>
                Semantic Properties
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={4}>
                  <Typography variant="caption" color="text.secondary">Imageability</Typography>
                  <Typography variant="body1">{wordResult.imageability ? wordResult.imageability.toFixed(1) : '-'}</Typography>
                </Grid>
                <Grid item xs={4}>
                  <Typography variant="caption" color="text.secondary">Familiarity</Typography>
                  <Typography variant="body1">{wordResult.familiarity ? wordResult.familiarity.toFixed(1) : '-'}</Typography>
                </Grid>
                <Grid item xs={4}>
                  <Typography variant="caption" color="text.secondary">Concreteness</Typography>
                  <Typography variant="body1">{wordResult.concreteness ? wordResult.concreteness.toFixed(1) : '-'}</Typography>
                </Grid>
              </Grid>
            </Box>

            <Divider sx={{ my: 2 }} />

            {/* Emotional Norms */}
            <Box>
              <Typography variant="subtitle2" fontWeight={600} gutterBottom>
                Emotional Norms
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={4}>
                  <Typography variant="caption" color="text.secondary">Valence</Typography>
                  <Typography variant="body1">{wordResult.valence ? wordResult.valence.toFixed(1) : '-'}</Typography>
                </Grid>
                <Grid item xs={4}>
                  <Typography variant="caption" color="text.secondary">Arousal</Typography>
                  <Typography variant="body1">{wordResult.arousal ? wordResult.arousal.toFixed(1) : '-'}</Typography>
                </Grid>
                <Grid item xs={4}>
                  <Typography variant="caption" color="text.secondary">Dominance</Typography>
                  <Typography variant="body1">{wordResult.dominance ? wordResult.dominance.toFixed(1) : '-'}</Typography>
                </Grid>
              </Grid>
            </Box>

            {/* Syllable Breakdown */}
            {wordResult.syllables && wordResult.syllables.length > 0 && (
              <>
                <Divider sx={{ my: 2 }} />
                <Box>
                  <Typography variant="subtitle2" fontWeight={600} gutterBottom>
                    Syllable Structure
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                    {wordResult.syllables.map((syl, i) => (
                      <Chip
                        key={i}
                        label={`${syl.onset.join('')}-${syl.nucleus}-${syl.coda.join('')}`}
                        variant="outlined"
                      />
                    ))}
                  </Box>
                </Box>
              </>
            )}

            {/* Similar Words */}
            {similarityResults && similarityResults.length > 0 && (
              <>
                <Divider sx={{ my: 2 }} />
                <Box>
                  <Typography variant="subtitle2" fontWeight={600} gutterBottom>
                    Similar Words (by phonological structure)
                  </Typography>
                  <Box
                    sx={{
                      maxHeight: 300,
                      overflowY: 'auto',
                      mt: 1,
                      border: 1,
                      borderColor: 'divider',
                      borderRadius: 1,
                      bgcolor: 'grey.50'
                    }}
                  >
                    <Stack spacing={0}>
                      {similarityResults.map((result, i) => (
                        <Box
                          key={i}
                          sx={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'center',
                            p: 1.5,
                            borderBottom: i < similarityResults.length - 1 ? 1 : 0,
                            borderColor: 'divider',
                            '&:hover': {
                              bgcolor: 'grey.100'
                            }
                          }}
                        >
                          <Box sx={{ display: 'flex', alignItems: 'baseline', gap: 1.5 }}>
                            <Typography variant="body1" fontWeight={600}>
                              {result.word}
                            </Typography>
                            <Typography variant="body2" color="text.secondary" fontFamily="monospace">
                              /{result.ipa}/
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              {result.syllable_count} syl
                            </Typography>
                          </Box>
                          <Chip
                            label={result.similarity != null ? `${(result.similarity * 100).toFixed(0)}%` : 'N/A'}
                            size="small"
                            color="primary"
                            variant="outlined"
                          />
                        </Box>
                      ))}
                    </Stack>
                  </Box>
                </Box>
              </>
            )}
          </CardContent>
        </Card>
      )}

      {/* Phoneme Result (single phoneme) */}
      {phonemeResult && phonemeResult.phoneme && !phoneme2Result && (
        <Card sx={{ mt: 3 }} elevation={2}>
          <CardContent>
            <Typography variant="h5" gutterBottom fontWeight={600} fontFamily="monospace">
              /{phonemeResult.phoneme}/
            </Typography>
            <Chip
              label={phonemeResult.type.toUpperCase()}
              color={phonemeResult.type === 'vowel' ? 'secondary' : 'primary'}
              size="small"
              sx={{ mb: 2 }}
            />

            <Divider sx={{ my: 2 }} />

            <Typography variant="subtitle2" fontWeight={600} gutterBottom>
              Phoible Features
            </Typography>
            <Table size="small">
              <TableBody>
                {Object.entries(phonemeResult.features).map(([feature, value]) => (
                  <TableRow key={feature}>
                    <TableCell>{feature}</TableCell>
                    <TableCell align="right">
                      <Chip
                        label={value}
                        size="small"
                        color={value === '+' ? 'success' : value === '-' ? 'error' : 'default'}
                        variant="outlined"
                      />
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}

      {/* Phoneme Comparison Result (two phonemes) */}
      {phonemeResult && phoneme2Result && comparisonResult && (
        <Box sx={{ mt: 3 }}>
          {/* Summary */}
          <Paper sx={{ p: { xs: 2, sm: 3 }, mb: 3 }}>
            <Stack spacing={3}>
              {/* Phoneme 1 */}
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h2" sx={{ fontSize: { xs: '2.5rem', sm: '3rem' } }}>
                  {phonemeResult.phoneme}
                </Typography>
                <Typography variant="subtitle1" color="text.secondary">
                  {phonemeResult.type}
                </Typography>
              </Box>

              {/* Feature Distance */}
              <Box
                sx={{
                  textAlign: 'center',
                  py: 2,
                  bgcolor: 'primary.50',
                  borderRadius: 2,
                }}
              >
                <Typography variant="caption" color="text.secondary" display="block">
                  Feature Distance (0-1 scale)
                </Typography>
                <Typography variant="h3" color="primary" sx={{ fontSize: { xs: '2rem', sm: '2.5rem' } }}>
                  {comparisonResult.similarity_score.toFixed(2)}
                </Typography>
                <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 1 }}>
                  {Object.keys(comparisonResult.different_features).length} differences
                </Typography>
              </Box>

              {/* Phoneme 2 */}
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h2" sx={{ fontSize: { xs: '2.5rem', sm: '3rem' } }}>
                  {phoneme2Result.phoneme}
                </Typography>
                <Typography variant="subtitle1" color="text.secondary">
                  {phoneme2Result.type}
                </Typography>
              </Box>
            </Stack>

            {Object.keys(comparisonResult.different_features).length > 0 && (
              <Box sx={{ mt: 3 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Differing Features:
                </Typography>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                  {Object.entries(comparisonResult.different_features).map(([feature]) => (
                    <Chip
                      key={feature}
                      label={feature}
                      size="small"
                      color="warning"
                    />
                  ))}
                </Box>
              </Box>
            )}

            {Object.keys(comparisonResult.shared_features).length > 0 && (
              <Box sx={{ mt: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Shared Features:
                </Typography>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                  {Object.entries(comparisonResult.shared_features).map(([feature]) => (
                    <Chip
                      key={feature}
                      label={feature}
                      size="small"
                      color="success"
                      variant="outlined"
                    />
                  ))}
                </Box>
              </Box>
            )}
          </Paper>

          {/* Feature Table */}
          <Card>
            <CardContent sx={{ p: { xs: 1, sm: 2 } }}>
              <Typography variant="h6" gutterBottom sx={{ px: { xs: 1, sm: 0 } }}>
                Feature Comparison
              </Typography>

              <TableContainer sx={{ overflowX: 'auto' }}>
                <Table size="small" sx={{ minWidth: 400 }}>
                  <TableHead>
                    <TableRow>
                      <TableCell>Feature</TableCell>
                      <TableCell align="center">{phonemeResult.phoneme}</TableCell>
                      <TableCell align="center">{phoneme2Result.phoneme}</TableCell>
                      <TableCell align="center">Match</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {getFeatureComparison()
                      .sort((a, b) => {
                        // Sort: differences first, then matches
                        if (a.match !== b.match) return a.match ? 1 : -1;
                        return a.feature.localeCompare(b.feature);
                      })
                      .map((row) => (
                        <TableRow
                          key={row.feature}
                          sx={{
                            backgroundColor: row.match ? undefined : 'warning.light',
                            opacity: row.match ? 0.7 : 1,
                          }}
                        >
                          <TableCell>
                            <Typography variant="body2" fontFamily="monospace">
                              {row.feature}
                            </Typography>
                          </TableCell>
                          <TableCell align="center">
                            <Chip
                              label={row.phoneme1Value || 'N/A'}
                              size="small"
                              color={row.phoneme1Value === '+' ? 'primary' : 'default'}
                            />
                          </TableCell>
                          <TableCell align="center">
                            <Chip
                              label={row.phoneme2Value || 'N/A'}
                              size="small"
                              color={row.phoneme2Value === '+' ? 'primary' : 'default'}
                            />
                          </TableCell>
                          <TableCell align="center">
                            {row.match ? (
                              <Chip label="✓" size="small" color="success" />
                            ) : (
                              <Chip label="✗" size="small" color="warning" />
                            )}
                          </TableCell>
                        </TableRow>
                      ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Box>
      )}

      {/* Phoneme Search Results */}
      {phonemeSearchResult && (
        <Box sx={{ mt: 3 }}>
          <Alert severity="success" sx={{ mb: 2 }}>
            Found {phonemeSearchResult.count} phonemes matching the selected features
          </Alert>
          <Card elevation={2}>
            <CardContent>
              <Typography variant="subtitle2" fontWeight={600} gutterBottom>
                Matching Phonemes
              </Typography>
              <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mt: 1 }}>
                {phonemeSearchResult.matching_phonemes.map((phoneme, i) => (
                  <Chip
                    key={i}
                    label={phoneme}
                    variant="outlined"
                    sx={{ fontFamily: 'monospace', fontSize: '1rem' }}
                  />
                ))}
              </Box>
            </CardContent>
          </Card>
        </Box>
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

export default SearchTool;
