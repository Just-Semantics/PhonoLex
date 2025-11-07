/**
 * Phoneme Learning Difficulty Tool
 *
 * Based on Flege's Speech Learning Model (1995):
 * Analyzes which L2 phonemes are hardest for L1 speakers to learn.
 *
 * Key insight: SIMILAR sounds are HARDER than completely NEW sounds!
 * - Similar sounds trigger "equivalence classification" (Flege H5)
 * - Learners perceive L1 and L2 sounds as "the same" when they're actually different
 * - This blocks formation of a new L2 category
 */

import { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Autocomplete,
  TextField,
  Button,
  Card,
  CardContent,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  LinearProgress,
  Alert,
  Stack,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Psychology as PsychologyIcon,
  Download as DownloadIcon,
  Info as InfoIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  FiberNew as NewIcon
} from '@mui/icons-material';
import {
  analyzeL1toL2,
  getAvailableLanguages,
  getDifficultyStats,
  type PhoibleLanguage,
  type DifficultyResult
} from '../../services/phoibleData';

export default function PhonemeDifficultyTool() {
  const [languages, setLanguages] = useState<PhoibleLanguage[]>([]);
  const [l1Language, setL1Language] = useState<PhoibleLanguage | null>(null);
  const [l2Language, setL2Language] = useState<PhoibleLanguage | null>(null);
  const [results, setResults] = useState<DifficultyResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadingLanguages, setLoadingLanguages] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load available languages on mount
  useEffect(() => {
    loadLanguages();
  }, []);

  const loadLanguages = async () => {
    try {
      setLoadingLanguages(true);
      const langs = await getAvailableLanguages();
      setLanguages(langs);

      // Set English as default L1
      const english = langs.find(l => l.iso === 'eng');
      if (english) setL1Language(english);
    } catch (err) {
      console.error('Failed to load languages:', err);
      setError('Failed to load language data');
    } finally {
      setLoadingLanguages(false);
    }
  };

  const handleAnalyze = async () => {
    if (!l1Language || !l2Language) return;

    try {
      setLoading(true);
      setError(null);
      const analysisResults = await analyzeL1toL2(l1Language.iso, l2Language.iso);
      setResults(analysisResults);
    } catch (err) {
      console.error('Analysis failed:', err);
      setError('Failed to analyze phoneme difficulty');
    } finally {
      setLoading(false);
    }
  };

  const handleExport = () => {
    if (results.length === 0) return;

    // Create CSV content
    const headers = ['L2 Phoneme', 'Closest L1', 'Distance', 'Category', 'Difficulty', 'Explanation'];
    const rows = results.map(r => [
      r.l2Phoneme,
      r.closestL1,
      r.distance.toFixed(3),
      r.category,
      r.difficulty.toString(),
      r.explanation
    ]);

    const csv = [
      headers.join(','),
      ...rows.map(row => row.map(cell => `"${cell}"`).join(','))
    ].join('\n');

    // Download
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `phoneme-difficulty-${l1Language?.iso}-to-${l2Language?.iso}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const stats = results.length > 0 ? getDifficultyStats(results) : null;

  const identicalResults = results.filter(r => r.category === 'identical');
  const similarResults = results.filter(r => r.category === 'similar');
  const newResults = results.filter(r => r.category === 'new');

  if (loadingLanguages) {
    return (
      <Box sx={{ p: 3 }}>
        <Typography>Loading language data...</Typography>
        <LinearProgress sx={{ mt: 2 }} />
      </Box>
    );
  }

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h5" fontWeight={700} gutterBottom>
          <PsychologyIcon sx={{ verticalAlign: 'middle', mr: 1 }} />
          Phoneme Learning Difficulty
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Based on Flege's Speech Learning Model (1995) · 2,095 languages · 3,142 phonemes
        </Typography>
      </Box>

      {/* Theory Alert */}
      <Alert severity="info" sx={{ mb: 3 }} icon={<InfoIcon />}>
        <Typography variant="body2" fontWeight={600} gutterBottom>
          Key Insight: Similar sounds are HARDER than completely new sounds!
        </Typography>
        <Typography variant="body2">
          Learners perceive L1 and L2 sounds as "the same" when they're actually different,
          blocking formation of a new L2 category (equivalence classification).
        </Typography>
      </Alert>

      {/* Language Selection */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" fontWeight={600} gutterBottom>
            Select Languages
          </Typography>

          <Stack spacing={2}>
            <Autocomplete
              value={l1Language}
              onChange={(_, newValue) => setL1Language(newValue)}
              options={languages}
              getOptionLabel={(option) => `${option.name} (${option.iso})`}
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="L1 (Native Language)"
                  helperText="The learner's native language"
                />
              )}
              isOptionEqualToValue={(option, value) => option.iso === value.iso}
            />

            <Autocomplete
              value={l2Language}
              onChange={(_, newValue) => setL2Language(newValue)}
              options={languages}
              getOptionLabel={(option) => `${option.name} (${option.iso})`}
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="L2 (Target Language)"
                  helperText="The language being learned"
                />
              )}
              isOptionEqualToValue={(option, value) => option.iso === value.iso}
            />

            <Button
              variant="contained"
              size="large"
              onClick={handleAnalyze}
              disabled={!l1Language || !l2Language || loading}
              fullWidth
            >
              {loading ? 'Analyzing...' : 'Analyze Difficulty'}
            </Button>
          </Stack>
        </CardContent>
      </Card>

      {/* Error */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {/* Loading */}
      {loading && <LinearProgress sx={{ mb: 3 }} />}

      {/* Results */}
      {results.length > 0 && stats && (
        <Box>
          {/* Summary */}
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6" fontWeight={600}>
                  Results: {l2Language?.name} ({stats.total} phonemes)
                </Typography>
                <Tooltip title="Export to CSV">
                  <IconButton onClick={handleExport} color="primary">
                    <DownloadIcon />
                  </IconButton>
                </Tooltip>
              </Box>

              <Stack direction="row" spacing={2} flexWrap="wrap" useFlexGap>
                <Chip
                  icon={<WarningIcon />}
                  label={`${similarResults.length} Similar (Hardest)`}
                  color="error"
                  variant="filled"
                />
                <Chip
                  icon={<CheckCircleIcon />}
                  label={`${identicalResults.length} Identical (Easy)`}
                  color="success"
                  variant="filled"
                />
                <Chip
                  icon={<NewIcon />}
                  label={`${newResults.length} New (Easier)`}
                  color="info"
                  variant="filled"
                />
              </Stack>
            </CardContent>
          </Card>

          {/* Danger Zone - Similar Sounds */}
          {similarResults.length > 0 && (
            <Accordion defaultExpanded sx={{ mb: 2 }}>
              <AccordionSummary expandIcon={<ExpandMoreIcon />} sx={{ bgcolor: 'error.light', color: 'error.dark' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <WarningIcon />
                  <Typography fontWeight={600}>
                    Danger Zone - {similarResults.length} SIMILAR sounds (hardest to learn)
                  </Typography>
                </Box>
              </AccordionSummary>
              <AccordionDetails>
                <Alert severity="warning" sx={{ mb: 2 }}>
                  <Typography variant="body2">
                    These sounds trigger <strong>equivalence classification</strong> (Flege H5).
                    Learners perceive them as "the same" as L1 sounds, causing bidirectional interference.
                  </Typography>
                </Alert>

                <Stack spacing={1.5}>
                  {similarResults.map((result, idx) => (
                    <Card key={idx} variant="outlined" sx={{ bgcolor: 'error.50' }}>
                      <CardContent sx={{ py: 1.5, '&:last-child': { pb: 1.5 } }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <Box>
                            <Typography variant="body1" fontWeight={600}>
                              /{result.l2Phoneme}/ ← /{result.closestL1}/
                              <Chip
                                label={`Difficulty: ${result.difficulty}/5`}
                                size="small"
                                color="error"
                                sx={{ ml: 1 }}
                              />
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                              Distance: {result.distance.toFixed(3)} · {result.explanation}
                            </Typography>
                          </Box>
                        </Box>
                      </CardContent>
                    </Card>
                  ))}
                </Stack>
              </AccordionDetails>
            </Accordion>
          )}

          {/* Easy Transfer - Identical Sounds */}
          {identicalResults.length > 0 && (
            <Accordion sx={{ mb: 2 }}>
              <AccordionSummary expandIcon={<ExpandMoreIcon />} sx={{ bgcolor: 'success.light', color: 'success.dark' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <CheckCircleIcon />
                  <Typography fontWeight={600}>
                    Easy Transfer - {identicalResults.length} IDENTICAL sounds
                  </Typography>
                </Box>
              </AccordionSummary>
              <AccordionDetails>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  These phonemes exist in both L1 and L2. Perfect transfer from native language.
                </Typography>

                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                  {identicalResults.slice(0, 50).map((result, idx) => (
                    <Chip
                      key={idx}
                      label={`/${result.l2Phoneme}/`}
                      size="small"
                      color="success"
                      variant="outlined"
                    />
                  ))}
                  {identicalResults.length > 50 && (
                    <Chip
                      label={`+${identicalResults.length - 50} more`}
                      size="small"
                      variant="outlined"
                    />
                  )}
                </Box>
              </AccordionDetails>
            </Accordion>
          )}

          {/* New Sounds */}
          {newResults.length > 0 && (
            <Accordion sx={{ mb: 2 }}>
              <AccordionSummary expandIcon={<ExpandMoreIcon />} sx={{ bgcolor: 'info.light', color: 'info.dark' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <NewIcon />
                  <Typography fontWeight={600}>
                    New Sounds - {newResults.length} sounds (easier than similar)
                  </Typography>
                </Box>
              </AccordionSummary>
              <AccordionDetails>
                <Alert severity="info" sx={{ mb: 2 }}>
                  <Typography variant="body2">
                    These sounds are clearly different from L1. Learners can <strong>discern the differences</strong> and
                    form new L2 categories (Flege H2, H3).
                  </Typography>
                </Alert>

                <Stack spacing={1.5}>
                  {newResults.map((result, idx) => (
                    <Card key={idx} variant="outlined" sx={{ bgcolor: 'info.50' }}>
                      <CardContent sx={{ py: 1.5, '&:last-child': { pb: 1.5 } }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <Box>
                            <Typography variant="body1" fontWeight={600}>
                              /{result.l2Phoneme}/ ← /{result.closestL1}/
                              <Chip
                                label={`Difficulty: ${result.difficulty}/5`}
                                size="small"
                                color="info"
                                sx={{ ml: 1 }}
                              />
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                              Distance: {result.distance.toFixed(3)} · {result.explanation}
                            </Typography>
                          </Box>
                        </Box>
                      </CardContent>
                    </Card>
                  ))}
                </Stack>
              </AccordionDetails>
            </Accordion>
          )}

          {/* Visual Distance Chart */}
          <Card>
            <CardContent>
              <Typography variant="h6" fontWeight={600} gutterBottom>
                Distance Distribution
              </Typography>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                The "similarity valley" (0.1-0.5) is where equivalence classification causes maximum difficulty
              </Typography>

              <Box sx={{ mt: 3, position: 'relative', height: 60 }}>
                {/* Scale */}
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="caption">0.0</Typography>
                  <Typography variant="caption">0.3</Typography>
                  <Typography variant="caption">0.5</Typography>
                  <Typography variant="caption">0.7</Typography>
                  <Typography variant="caption">1.0</Typography>
                </Box>

                {/* Gradient bar */}
                <Box
                  sx={{
                    height: 40,
                    background: 'linear-gradient(to right, #4caf50 0%, #4caf50 10%, #f44336 10%, #f44336 50%, #2196f3 50%, #2196f3 100%)',
                    borderRadius: 1,
                    position: 'relative',
                    display: 'flex',
                    alignItems: 'center'
                  }}
                >
                  <Typography variant="caption" sx={{ position: 'absolute', left: '5%', color: 'white', fontWeight: 600 }}>
                    Identical
                  </Typography>
                  <Typography variant="caption" sx={{ position: 'absolute', left: '30%', color: 'white', fontWeight: 600 }}>
                    Similar (HARD!)
                  </Typography>
                  <Typography variant="caption" sx={{ position: 'absolute', left: '75%', color: 'white', fontWeight: 600 }}>
                    New (Easier)
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>

          {/* Citation */}
          <Box sx={{ mt: 3, p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
            <Typography variant="caption" color="text.secondary">
              <strong>Reference:</strong> Flege, J. E. (1995). Second language speech learning: Theory, findings, and problems.
              In W. Strange (Ed.), <em>Speech Perception and Linguistic Experience: Issues in Cross-Language Research</em>.
              York Press. (2,152 citations)
            </Typography>
          </Box>
        </Box>
      )}
    </Box>
  );
}
