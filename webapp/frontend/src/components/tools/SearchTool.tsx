/**
 * Search Tool Component
 *
 * Informational lookup for words and phonemes
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
} from '@mui/material';
import {
  Search as SearchIcon,
  TextFields as WordIcon,
  GraphicEq as PhonemeIcon,
  FilterList as FilterIcon,
  Add as AddIcon,
  Remove as RemoveIcon,
  Keyboard as KeyboardIcon,
} from '@mui/icons-material';
import api from '../../services/phonolexApi';
import type { Word } from '../../services/phonolexApi';
import PhonemePickerDialog from '../PhonemePickerDialog';

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
  const [wordResult, setWordResult] = useState<Word | null>(null);
  const [phonemeResult, setPhonemeResult] = useState<PhonemeResult | null>(null);
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

  // Load available features from phoneme data on mount
  React.useEffect(() => {
    const loadFeatures = async () => {
      try {
        const phonemeList = await api.listPhonemes();
        // Extract all unique feature names from the phonemes
        const featureSet = new Set<string>();
        phonemeList.phonemes.forEach(p => {
          Object.keys(p.features || {}).forEach(feat => featureSet.add(feat));
        });
        setAvailableFeatures(Array.from(featureSet).sort());
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
        const data = await api.getPhoneme(query.trim());
        setPhonemeResult(data);
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
      setWordResult(null);
      setPhonemeResult(null);
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

        {/* Word/Phoneme Search Input */}
        {mode !== 'phoneme-features' && (
          <Box sx={{ position: 'relative' }}>
            <TextField
              label={mode === 'word' ? 'Enter a word' : 'Select or type phoneme (IPA)'}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={handleKeyPress}
              size="medium"
              placeholder={mode === 'word' ? 'e.g., cat, computer, beautiful' : 'Click the keyboard icon to select'}
              fullWidth
              autoFocus
              InputProps={mode === 'phoneme' ? {
                endAdornment: (
                  <IconButton
                    onClick={() => setPhonemePickerOpen(true)}
                    edge="end"
                    color="primary"
                  >
                    <KeyboardIcon />
                  </IconButton>
                )
              } : undefined}
            />
          </Box>
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
          {loading ? 'Searching...' : 'Search'}
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
                <Grid item xs={6} sm={3}>
                  <Typography variant="caption" color="text.secondary">Complexity</Typography>
                  <Chip label={wordResult.complexity || 'N/A'} size="small" />
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
                  <Typography variant="caption" color="text.secondary">Age of Acquisition</Typography>
                  <Typography variant="body1">
                    {wordResult.aoa ? `${wordResult.aoa.toFixed(1)} years` : '-'}
                  </Typography>
                </Grid>
                <Grid item xs={6} sm={4}>
                  <Typography variant="caption" color="text.secondary">Word Length</Typography>
                  <Chip label={wordResult.word_length || 'N/A'} size="small" />
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

      {/* Phoneme Result */}
      {phonemeResult && phonemeResult.phoneme && (
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
        onSelect={(phoneme) => setQuery((prev) => prev + phoneme)}
      />
    </Box>
  );
};

export default SearchTool;
