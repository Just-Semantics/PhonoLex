/**
 * Compare Component
 *
 * Phoneme comparison tool:
 * - Compare two phonemes side-by-side
 * - Show PHOIBLE distinctive features
 * - Calculate feature distance
 * - Find phonologically similar phonemes
 */

import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Chip,
  Stack,
  Alert,
  CircularProgress,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
} from '@mui/material';
import {
  Compare as CompareIcon,
  Clear as ClearIcon,
  SwapHoriz as SwapIcon,
  Keyboard as KeyboardIcon,
} from '@mui/icons-material';
import api from '../services/phonolexApi';
import type { Phoneme, PhonemeComparison } from '../services/phonolexApi';
import PhonemePickerDialog from './PhonemePickerDialog';
import { validatePhonemeInput } from '../utils/ipaValidation';

const Compare: React.FC = () => {
  // Input state
  const [phoneme1Input, setPhoneme1Input] = useState('');
  const [phoneme2Input, setPhoneme2Input] = useState('');

  // Data state
  const [phoneme1, setPhoneme1] = useState<Phoneme | null>(null);
  const [phoneme2, setPhoneme2] = useState<Phoneme | null>(null);
  const [comparison, setComparison] = useState<PhonemeComparison | null>(null);

  // Loading/error state
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Phoneme picker state
  const [phonemePickerOpen, setPhonemePickerOpen] = useState(false);
  const [activePhonemeField, setActivePhonemeField] = useState<'phoneme1' | 'phoneme2'>('phoneme1');

  // IPA validation warnings
  const [ipaWarning1, setIpaWarning1] = useState<string | null>(null);
  const [ipaWarning2, setIpaWarning2] = useState<string | null>(null);

  // Compare phonemes
  const handleCompare = async () => {
    if (!phoneme1Input.trim() || !phoneme2Input.trim()) {
      setError('Please enter both phonemes');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Fetch both phonemes - convert to PhonemeDetail format
      const [p1Raw, p2Raw] = await Promise.all([
        api.getPhoneme(phoneme1Input.trim()),
        api.getPhoneme(phoneme2Input.trim()),
      ]);

      if (!p1Raw || !p2Raw) {
        setError('One or both phonemes not found');
        setLoading(false);
        return;
      }

      // Convert to PhonemeDetail format
      const p1: Phoneme = {
        phoneme_id: 0,
        ipa: p1Raw.phoneme,
        segment_class: p1Raw.type === 'vowel' ? 'vowel' : 'consonant',
        features: p1Raw.features,
        has_trajectory: false,
      };

      const p2: Phoneme = {
        phoneme_id: 0,
        ipa: p2Raw.phoneme,
        segment_class: p2Raw.type === 'vowel' ? 'vowel' : 'consonant',
        features: p2Raw.features,
        has_trajectory: false,
      };

      setPhoneme1(p1);
      setPhoneme2(p2);

      // Get comparison
      const comp = await api.comparePhonemes(phoneme1Input.trim(), phoneme2Input.trim());
      setComparison(comp as unknown as PhonemeComparison);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Comparison failed');
      setPhoneme1(null);
      setPhoneme2(null);
      setComparison(null);
    } finally {
      setLoading(false);
    }
  };

  // Swap phonemes
  const handleSwap = () => {
    const temp = phoneme1Input;
    setPhoneme1Input(phoneme2Input);
    setPhoneme2Input(temp);

    if (phoneme1 && phoneme2) {
      const tempPhoneme = phoneme1;
      setPhoneme1(phoneme2);
      setPhoneme2(tempPhoneme);
    }
  };

  // Clear all
  const handleClear = () => {
    setPhoneme1Input('');
    setPhoneme2Input('');
    setPhoneme1(null);
    setPhoneme2(null);
    setComparison(null);
    setError(null);
  };

  // Handle phoneme picker
  const handlePhonemeSelect = (phoneme: string) => {
    if (activePhonemeField === 'phoneme1') {
      setPhoneme1Input((prev) => prev + phoneme);
    } else {
      setPhoneme2Input((prev) => prev + phoneme);
    }
    // Don't close - allow multiple selections
  };

  const openPhonemePicker = (field: 'phoneme1' | 'phoneme2') => {
    setActivePhonemeField(field);
    setPhonemePickerOpen(true);
  };

  // Get feature comparison data
  const getFeatureComparison = () => {
    if (!phoneme1 || !phoneme2 || !phoneme1.features || !phoneme2.features) return [];

    const allFeatures = new Set([
      ...Object.keys(phoneme1.features),
      ...Object.keys(phoneme2.features),
    ]);

    return Array.from(allFeatures).map((feature) => ({
      feature,
      phoneme1Value: phoneme1.features?.[feature] || '',
      phoneme2Value: phoneme2.features?.[feature] || '',
      match: phoneme1.features?.[feature] === phoneme2.features?.[feature],
    }));
  };

  return (
    <Box>
      {/* Input Section */}
      <Paper sx={{ p: { xs: 2, sm: 3 } }}>
        <Stack spacing={2}>
          <TextField
            label="Phoneme 1 (IPA)"
            value={phoneme1Input}
            onChange={(e) => {
              const newValue = e.target.value;
              setPhoneme1Input(newValue);

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
            onKeyPress={(e) => e.key === 'Enter' && handleCompare()}
            placeholder="e.g., t, k, s"
            fullWidth
            size="small"
            InputProps={{
              endAdornment: (
                <IconButton
                  onClick={() => openPhonemePicker('phoneme1')}
                  edge="end"
                  color="primary"
                  size="small"
                  aria-label="Open phoneme picker for phoneme 1"
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

          <Button
            variant="outlined"
            startIcon={<SwapIcon />}
            onClick={handleSwap}
            size="small"
            sx={{ alignSelf: 'center', minWidth: 120 }}
          >
            Swap
          </Button>

          <TextField
            label="Phoneme 2 (IPA)"
            value={phoneme2Input}
            onChange={(e) => {
              const newValue = e.target.value;
              setPhoneme2Input(newValue);

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
            onKeyPress={(e) => e.key === 'Enter' && handleCompare()}
            placeholder="e.g., d, g, z"
            fullWidth
            size="small"
            InputProps={{
              endAdornment: (
                <IconButton
                  onClick={() => openPhonemePicker('phoneme2')}
                  edge="end"
                  color="primary"
                  size="small"
                  aria-label="Open phoneme picker for phoneme 2"
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

          <Stack direction="row" spacing={2} sx={{ mt: 2 }}>
            <Button
              variant="contained"
              startIcon={<CompareIcon />}
              onClick={handleCompare}
              disabled={loading}
              fullWidth
            >
              Compare
            </Button>
            <Button
              variant="outlined"
              startIcon={<ClearIcon />}
              onClick={handleClear}
              sx={{ minWidth: { xs: 80, sm: 100 } }}
            >
              Clear
            </Button>
          </Stack>
        </Stack>
      </Paper>

      {/* Status Messages */}
      <Box sx={{ mt: 3 }}>
        {loading && (
          <Alert severity="info" icon={<CircularProgress size={20} />}>
            Comparing phonemes...
          </Alert>
        )}
        {error && <Alert severity="error">{error}</Alert>}
      </Box>

      {/* Comparison Results */}
      {phoneme1 && phoneme2 && comparison && !loading && (
        <Box sx={{ mt: 3 }}>
          {/* Summary */}
          <Paper sx={{ p: { xs: 2, sm: 3 }, mb: 3 }}>
            <Stack spacing={3}>
              {/* Phoneme 1 */}
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h2" sx={{ fontSize: { xs: '2.5rem', sm: '3rem' } }}>
                  {phoneme1.ipa}
                </Typography>
                <Typography variant="subtitle1" color="text.secondary">
                  {phoneme1.segment_class}
                </Typography>
                {phoneme1.has_trajectory && (
                  <Box sx={{ mt: 1 }}>
                    <Chip label="Has trajectory" size="small" color="primary" />
                  </Box>
                )}
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
                  {comparison.similarity_score.toFixed(2)}
                </Typography>
                <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 1 }}>
                  {Object.keys(comparison.different_features).length} differences
                </Typography>
              </Box>

              {/* Phoneme 2 */}
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h2" sx={{ fontSize: { xs: '2.5rem', sm: '3rem' } }}>
                  {phoneme2.ipa}
                </Typography>
                <Typography variant="subtitle1" color="text.secondary">
                  {phoneme2.segment_class}
                </Typography>
                {phoneme2.has_trajectory && (
                  <Box sx={{ mt: 1 }}>
                    <Chip label="Has trajectory" size="small" color="primary" />
                  </Box>
                )}
              </Box>
            </Stack>

            {Object.keys(comparison.different_features).length > 0 && (
              <Box sx={{ mt: 3 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Differing Features:
                </Typography>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                  {Object.entries(comparison.different_features).map(([feature]) => (
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

            {Object.keys(comparison.shared_features).length > 0 && (
              <Box sx={{ mt: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Shared Features:
                </Typography>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                  {Object.entries(comparison.shared_features).map(([feature]) => (
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
                      <TableCell align="center">{phoneme1.ipa}</TableCell>
                      <TableCell align="center">{phoneme2.ipa}</TableCell>
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

      {/* Phoneme Picker Dialog */}
      <PhonemePickerDialog
        open={phonemePickerOpen}
        onClose={() => setPhonemePickerOpen(false)}
        onSelect={handlePhonemeSelect}
      />
    </Box>
  );
};

export default Compare;
