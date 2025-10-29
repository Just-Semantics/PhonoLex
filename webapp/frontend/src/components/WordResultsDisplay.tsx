/**
 * Word Results Display Component
 *
 * Displays word results from API calls with:
 * - Table view with sortable columns
 * - Export to CSV functionality
 * - Phonological property display (WCM, MSH, syllables)
 * - Similarity scores (when applicable)
 */

import React, { useState, useMemo } from 'react';
import {
  Box,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TableSortLabel,
  Button,
  Chip,
  Typography,
  Stack,
  Tooltip,
  Card,
  CardContent,
  Grid,
  Divider,
  useMediaQuery,
  useTheme,
  ToggleButton,
  ToggleButtonGroup,
} from '@mui/material';
import {
  Download as ExportIcon,
  ContentCopy as CopyIcon,
  ViewList as TableViewIcon,
  ViewModule as CardViewIcon,
} from '@mui/icons-material';
import type { Word, MinimalPair, SimilarWord } from '../services/phonolexApi';

type SortField = 'word' | 'wcm' | 'msh' | 'syllable_count' | 'similarity' | 'frequency' | 'aoa' | 'imageability' | 'familiarity' | 'concreteness' | 'valence' | 'arousal' | 'dominance';
type SortDirection = 'asc' | 'desc';

interface Props {
  results: Word[] | MinimalPair[] | SimilarWord[];
  showSimilarity?: boolean;
}

const WordResultsDisplay: React.FC<Props> = ({ results, showSimilarity = false }) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [sortField, setSortField] = useState<SortField>('word');
  const [sortDirection, setSortDirection] = useState<SortDirection>('asc');
  const [viewMode, setViewMode] = useState<'table' | 'cards'>(isMobile ? 'cards' : 'table');

  console.log('WordResultsDisplay:', { results, length: results.length });

  // Determine result type
  const isMinimalPairs = results.length > 0 && 'word1' in results[0];
  const isSimilarWords = results.length > 0 && 'word' in results[0] && 'similarity' in results[0];

  console.log('Type detection:', { isMinimalPairs, isSimilarWords });

  // Helper function to average two values (for sorting minimal pairs)
  const avg = (a: number | null | undefined, b: number | null | undefined): number | null => {
    const validA = a ?? null;
    const validB = b ?? null;
    if (validA !== null && validB !== null) return (validA + validB) / 2;
    if (validA !== null) return validA;
    if (validB !== null) return validB;
    return null;
  };

  // Helper function to format pair values for display
  const formatPair = (a: number | null | undefined, b: number | null | undefined, decimals: number = 1): string => {
    const valA = a !== null && a !== undefined ? a.toFixed(decimals) : '-';
    const valB = b !== null && b !== undefined ? b.toFixed(decimals) : '-';
    return `${valA} / ${valB}`;
  };

  // Extract words for display
  const displayWords = useMemo(() => {
    if (isMinimalPairs) {
      // Show pairs as single row: word1 / word2
      const pairs = results as MinimalPair[];
      return pairs.map(pair => ({
        word: `${pair.word1.word} / ${pair.word2.word}`,
        ipa: `${pair.word1.ipa?.replace(/\n/g, ' ')} / ${pair.word2.ipa?.replace(/\n/g, ' ')}`,
        wcm_score: avg(pair.word1.wcm_score, pair.word2.wcm_score),
        syllable_count: avg(pair.word1.syllable_count, pair.word2.syllable_count),
        complexity: pair.word1.complexity || pair.word2.complexity,
        frequency: avg(pair.word1.frequency, pair.word2.frequency),
        aoa: avg(pair.word1.aoa, pair.word2.aoa),
        imageability: avg(pair.word1.imageability, pair.word2.imageability),
        familiarity: avg(pair.word1.familiarity, pair.word2.familiarity),
        concreteness: avg(pair.word1.concreteness, pair.word2.concreteness),
        valence: avg(pair.word1.valence, pair.word2.valence),
        arousal: avg(pair.word1.arousal, pair.word2.arousal),
        dominance: avg(pair.word1.dominance, pair.word2.dominance),
        // Store raw values for display
        _raw: {
          wcm_score: [pair.word1.wcm_score, pair.word2.wcm_score],
          syllable_count: [pair.word1.syllable_count, pair.word2.syllable_count],
          frequency: [pair.word1.frequency, pair.word2.frequency],
          aoa: [pair.word1.aoa, pair.word2.aoa],
          imageability: [pair.word1.imageability, pair.word2.imageability],
          familiarity: [pair.word1.familiarity, pair.word2.familiarity],
          concreteness: [pair.word1.concreteness, pair.word2.concreteness],
          valence: [pair.word1.valence, pair.word2.valence],
          arousal: [pair.word1.arousal, pair.word2.arousal],
          dominance: [pair.word1.dominance, pair.word2.dominance],
        },
        phoneme1: pair.phoneme1,
        phoneme2: pair.phoneme2,
        position: pair.position,
        isPair: true
      }));
    } else if (isSimilarWords) {
      return (results as SimilarWord[]).map(sw => ({ ...sw.word, similarity: sw.similarity, isPair: false }));
    } else {
      return (results as Word[]).map(w => ({ ...w, isPair: false }));
    }
  }, [results, isMinimalPairs, isSimilarWords]);

  // Sort words
  const sortedWords = useMemo(() => {
    const sorted = [...displayWords];
    sorted.sort((a: any, b: any) => {
      let aVal: any, bVal: any;

      switch (sortField) {
        case 'word':
          aVal = a.word || '';
          bVal = b.word || '';
          break;
        case 'wcm':
          aVal = a.wcm_score || 0;
          bVal = b.wcm_score || 0;
          break;
        case 'msh':
          aVal = a.complexity || '';
          bVal = b.complexity || '';
          break;
        case 'syllable_count':
          aVal = a.syllable_count || 0;
          bVal = b.syllable_count || 0;
          break;
        case 'frequency':
          aVal = a.frequency || 0;
          bVal = b.frequency || 0;
          break;
        case 'aoa':
          aVal = a.aoa || 999;
          bVal = b.aoa || 999;
          break;
        case 'imageability':
          aVal = a.imageability || 0;
          bVal = b.imageability || 0;
          break;
        case 'familiarity':
          aVal = a.familiarity || 0;
          bVal = b.familiarity || 0;
          break;
        case 'concreteness':
          aVal = a.concreteness || 0;
          bVal = b.concreteness || 0;
          break;
        case 'valence':
          aVal = a.valence || 0;
          bVal = b.valence || 0;
          break;
        case 'arousal':
          aVal = a.arousal || 0;
          bVal = b.arousal || 0;
          break;
        case 'dominance':
          aVal = a.dominance || 0;
          bVal = b.dominance || 0;
          break;
        case 'similarity':
          aVal = a.similarity || 0;
          bVal = b.similarity || 0;
          break;
        default:
          return 0;
      }

      if (sortDirection === 'asc') {
        return aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
      } else {
        return aVal > bVal ? -1 : aVal < bVal ? 1 : 0;
      }
    });
    return sorted;
  }, [displayWords, sortField, sortDirection]);

  // Handle sort
  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('asc');
    }
  };

  // Export to CSV
  const exportCSV = () => {
    const headers = [
      'Word', 'IPA', 'Syllables', 'WCM', 'Complexity', 'Frequency',
      'AoA', 'Imageability', 'Familiarity', 'Concreteness',
      'Valence', 'Arousal', 'Dominance'
    ];
    if (showSimilarity || isSimilarWords) headers.push('Similarity');

    const rows = sortedWords.map((w: any) => [
      w.word || '',
      w.ipa || '',
      w.syllable_count || '',
      w.wcm_score?.toFixed(2) || '',
      w.complexity || '',
      w.frequency?.toFixed(1) || '',
      w.aoa?.toFixed(1) || '',
      w.imageability?.toFixed(1) || '',
      w.familiarity?.toFixed(1) || '',
      w.concreteness?.toFixed(1) || '',
      w.valence?.toFixed(1) || '',
      w.arousal?.toFixed(1) || '',
      w.dominance?.toFixed(1) || '',
      ...(showSimilarity || isSimilarWords ? [w.similarity?.toFixed(3) || ''] : []),
    ]);

    const csv = [headers, ...rows].map(row => row.join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `phonolex_results_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Copy words to clipboard
  const copyWords = () => {
    const text = sortedWords.map((w: any) => w.word).join(', ');
    navigator.clipboard.writeText(text);
  };

  if (results.length === 0) {
    return (
      <Paper sx={{ p: 3, textAlign: 'center' }}>
        <Typography color="text.secondary">No results found</Typography>
      </Paper>
    );
  }

  return (
    <Box>
      <Stack
        direction={{ xs: 'column', sm: 'row' }}
        justifyContent="space-between"
        alignItems={{ xs: 'stretch', sm: 'center' }}
        spacing={2}
        sx={{ mb: 2 }}
      >
        <Typography variant="h6">
          {results.length} {isMinimalPairs ? 'Minimal Pairs' : 'Words'} Found
        </Typography>
        <Stack direction="row" spacing={1} justifyContent={{ xs: 'space-between', sm: 'flex-end' }}>
          <ToggleButtonGroup
            value={viewMode}
            exclusive
            onChange={(_, newMode) => newMode && setViewMode(newMode)}
            size="small"
            sx={{ display: { xs: 'flex', md: 'none' } }}
          >
            <ToggleButton value="table">
              <Tooltip title="Table view">
                <TableViewIcon fontSize="small" />
              </Tooltip>
            </ToggleButton>
            <ToggleButton value="cards">
              <Tooltip title="Card view">
                <CardViewIcon fontSize="small" />
              </Tooltip>
            </ToggleButton>
          </ToggleButtonGroup>

          <Stack direction="row" spacing={1}>
            <Tooltip title="Copy words to clipboard">
              <Button
                size="small"
                startIcon={<CopyIcon />}
                onClick={copyWords}
              >
                Copy
              </Button>
            </Tooltip>
            <Tooltip title="Export to CSV">
              <Button
                size="small"
                startIcon={<ExportIcon />}
                onClick={exportCSV}
              >
                Export
              </Button>
            </Tooltip>
          </Stack>
        </Stack>
      </Stack>

      {/* Card View for Mobile */}
      {viewMode === 'cards' ? (
        <Stack spacing={2}>
          {sortedWords.map((word: any, idx) => (
            <Card key={idx} variant="outlined">
              <CardContent>
                <Stack spacing={2}>
                  {/* Header: Word + IPA */}
                  <Box>
                    <Typography variant="h6" fontWeight={600}>
                      {word.word}
                    </Typography>
                    <Typography variant="body2" fontFamily="monospace" color="text.secondary">
                      {word.ipa}
                    </Typography>
                  </Box>

                  <Divider />

                  {/* Key Metrics */}
                  <Grid container spacing={1}>
                    <Grid item xs={6} sm={3}>
                      <Typography variant="caption" color="text.secondary">
                        Syllables
                      </Typography>
                      <Box>
                        {word.isPair ? (
                          <Typography variant="body2" fontFamily="monospace">
                            {formatPair(word._raw.syllable_count[0], word._raw.syllable_count[1], 0)}
                          </Typography>
                        ) : (
                          <Chip label={word.syllable_count || 0} size="small" />
                        )}
                      </Box>
                    </Grid>
                    <Grid item xs={6} sm={3}>
                      <Typography variant="caption" color="text.secondary">
                        WCM
                      </Typography>
                      <Box>
                        {word.isPair ? (
                          <Typography variant="body2" fontFamily="monospace">
                            {formatPair(word._raw.wcm_score[0], word._raw.wcm_score[1], 1)}
                          </Typography>
                        ) : (
                          <Chip
                            label={word.wcm_score?.toFixed(1) || '0.0'}
                            size="small"
                            color={
                              (word.wcm_score || 0) < 5 ? 'success' :
                              (word.wcm_score || 0) < 10 ? 'warning' : 'error'
                            }
                          />
                        )}
                      </Box>
                    </Grid>
                    <Grid item xs={6} sm={3}>
                      <Typography variant="caption" color="text.secondary">
                        Complexity
                      </Typography>
                      <Box>
                        <Chip label={word.complexity || 'N/A'} size="small" variant="outlined" />
                      </Box>
                    </Grid>
                    {(showSimilarity || isSimilarWords) && (
                      <Grid item xs={6} sm={3}>
                        <Typography variant="caption" color="text.secondary">
                          Similarity
                        </Typography>
                        <Box>
                          <Chip
                            label={word.similarity?.toFixed(3) || 'N/A'}
                            size="small"
                            color="primary"
                          />
                        </Box>
                      </Grid>
                    )}
                  </Grid>

                  {/* Lexical Properties */}
                  {(word.frequency || word.aoa) && (
                    <>
                      <Divider />
                      <Grid container spacing={1}>
                        {word.frequency !== null && word.frequency !== undefined && (
                          <Grid item xs={6}>
                            <Typography variant="caption" color="text.secondary">
                              Frequency
                            </Typography>
                            <Typography variant="body2" fontFamily="monospace">
                              {word.isPair ? formatPair(word._raw.frequency[0], word._raw.frequency[1]) : word.frequency.toFixed(1)}
                            </Typography>
                          </Grid>
                        )}
                        {word.aoa !== null && word.aoa !== undefined && (
                          <Grid item xs={6}>
                            <Typography variant="caption" color="text.secondary">
                              AoA
                            </Typography>
                            <Typography variant="body2" fontFamily="monospace">
                              {word.isPair ? formatPair(word._raw.aoa[0], word._raw.aoa[1]) : word.aoa.toFixed(1)} yrs
                            </Typography>
                          </Grid>
                        )}
                      </Grid>
                    </>
                  )}

                  {/* Semantic Properties (Collapsed) */}
                  {(word.imageability || word.familiarity || word.concreteness) && (
                    <>
                      <Divider />
                      <Typography variant="caption" color="text.secondary" fontWeight={600}>
                        Semantic Properties
                      </Typography>
                      <Grid container spacing={1}>
                        {word.imageability !== null && word.imageability !== undefined && (
                          <Grid item xs={4}>
                            <Typography variant="caption" color="text.secondary">
                              Image
                            </Typography>
                            <Typography variant="body2" fontFamily="monospace">
                              {word.isPair ? formatPair(word._raw.imageability[0], word._raw.imageability[1]) : word.imageability.toFixed(1)}
                            </Typography>
                          </Grid>
                        )}
                        {word.familiarity !== null && word.familiarity !== undefined && (
                          <Grid item xs={4}>
                            <Typography variant="caption" color="text.secondary">
                              Famil
                            </Typography>
                            <Typography variant="body2" fontFamily="monospace">
                              {word.isPair ? formatPair(word._raw.familiarity[0], word._raw.familiarity[1]) : word.familiarity.toFixed(1)}
                            </Typography>
                          </Grid>
                        )}
                        {word.concreteness !== null && word.concreteness !== undefined && (
                          <Grid item xs={4}>
                            <Typography variant="caption" color="text.secondary">
                              Concr
                            </Typography>
                            <Typography variant="body2" fontFamily="monospace">
                              {word.isPair ? formatPair(word._raw.concreteness[0], word._raw.concreteness[1]) : word.concreteness.toFixed(1)}
                            </Typography>
                          </Grid>
                        )}
                      </Grid>
                    </>
                  )}
                </Stack>
              </CardContent>
            </Card>
          ))}
        </Stack>
      ) : (
        /* Table View */
        <TableContainer component={Paper} sx={{ overflowX: 'auto' }}>
          <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>
                <TableSortLabel
                  active={sortField === 'word'}
                  direction={sortField === 'word' ? sortDirection : 'asc'}
                  onClick={() => handleSort('word')}
                >
                  Word
                </TableSortLabel>
              </TableCell>
              <TableCell>IPA</TableCell>
              <TableCell align="center">
                <TableSortLabel
                  active={sortField === 'syllable_count'}
                  direction={sortField === 'syllable_count' ? sortDirection : 'asc'}
                  onClick={() => handleSort('syllable_count')}
                >
                  Syllables
                </TableSortLabel>
              </TableCell>
              <TableCell align="center">
                <TableSortLabel
                  active={sortField === 'wcm'}
                  direction={sortField === 'wcm' ? sortDirection : 'asc'}
                  onClick={() => handleSort('wcm')}
                >
                  WCM
                </TableSortLabel>
              </TableCell>
              <TableCell align="center">
                <TableSortLabel
                  active={sortField === 'msh'}
                  direction={sortField === 'msh' ? sortDirection : 'asc'}
                  onClick={() => handleSort('msh')}
                >
                  Complexity
                </TableSortLabel>
              </TableCell>
              <TableCell align="center">
                <Tooltip title="Word frequency (per million)">
                  <TableSortLabel
                    active={sortField === 'frequency'}
                    direction={sortField === 'frequency' ? sortDirection : 'asc'}
                    onClick={() => handleSort('frequency')}
                  >
                    Freq
                  </TableSortLabel>
                </Tooltip>
              </TableCell>
              <TableCell align="center">
                <Tooltip title="Age of Acquisition (years)">
                  <TableSortLabel
                    active={sortField === 'aoa'}
                    direction={sortField === 'aoa' ? sortDirection : 'asc'}
                    onClick={() => handleSort('aoa')}
                  >
                    AoA
                  </TableSortLabel>
                </Tooltip>
              </TableCell>
              <TableCell align="center">
                <Tooltip title="Imageability (1-7 scale)">
                  <TableSortLabel
                    active={sortField === 'imageability'}
                    direction={sortField === 'imageability' ? sortDirection : 'asc'}
                    onClick={() => handleSort('imageability')}
                  >
                    Image
                  </TableSortLabel>
                </Tooltip>
              </TableCell>
              <TableCell align="center">
                <Tooltip title="Familiarity (1-7 scale)">
                  <TableSortLabel
                    active={sortField === 'familiarity'}
                    direction={sortField === 'familiarity' ? sortDirection : 'asc'}
                    onClick={() => handleSort('familiarity')}
                  >
                    Famil
                  </TableSortLabel>
                </Tooltip>
              </TableCell>
              <TableCell align="center">
                <Tooltip title="Concreteness (1-5 scale)">
                  <TableSortLabel
                    active={sortField === 'concreteness'}
                    direction={sortField === 'concreteness' ? sortDirection : 'asc'}
                    onClick={() => handleSort('concreteness')}
                  >
                    Concr
                  </TableSortLabel>
                </Tooltip>
              </TableCell>
              <TableCell align="center">
                <Tooltip title="Emotional Valence (1-9 scale)">
                  <TableSortLabel
                    active={sortField === 'valence'}
                    direction={sortField === 'valence' ? sortDirection : 'asc'}
                    onClick={() => handleSort('valence')}
                  >
                    Val
                  </TableSortLabel>
                </Tooltip>
              </TableCell>
              <TableCell align="center">
                <Tooltip title="Emotional Arousal (1-9 scale)">
                  <TableSortLabel
                    active={sortField === 'arousal'}
                    direction={sortField === 'arousal' ? sortDirection : 'asc'}
                    onClick={() => handleSort('arousal')}
                  >
                    Aro
                  </TableSortLabel>
                </Tooltip>
              </TableCell>
              <TableCell align="center">
                <Tooltip title="Emotional Dominance (1-9 scale)">
                  <TableSortLabel
                    active={sortField === 'dominance'}
                    direction={sortField === 'dominance' ? sortDirection : 'asc'}
                    onClick={() => handleSort('dominance')}
                  >
                    Dom
                  </TableSortLabel>
                </Tooltip>
              </TableCell>
              {(showSimilarity || isSimilarWords) && (
                <TableCell align="center">
                  <TableSortLabel
                    active={sortField === 'similarity'}
                    direction={sortField === 'similarity' ? sortDirection : 'asc'}
                    onClick={() => handleSort('similarity')}
                  >
                    Similarity
                  </TableSortLabel>
                </TableCell>
              )}
            </TableRow>
          </TableHead>
          <TableBody>
            {sortedWords.map((word: any, idx) => (
              <TableRow key={idx} hover>
                <TableCell>
                  <Typography variant="body2" fontWeight={500}>
                    {word.word}
                  </Typography>
                </TableCell>
                <TableCell>
                  <Typography variant="body2" fontFamily="monospace" color="text.secondary">
                    {word.ipa}
                  </Typography>
                </TableCell>
                <TableCell align="center">
                  {word.isPair ? (
                    <Typography variant="body2" color="text.secondary" sx={{ fontFamily: 'monospace' }}>
                      {formatPair(word._raw.syllable_count[0], word._raw.syllable_count[1], 0)}
                    </Typography>
                  ) : (
                    <Chip
                      label={word.syllable_count || 0}
                      size="small"
                      color="default"
                    />
                  )}
                </TableCell>
                <TableCell align="center">
                  {word.isPair ? (
                    <Typography variant="body2" color="text.secondary" sx={{ fontFamily: 'monospace' }}>
                      {formatPair(word._raw.wcm_score[0], word._raw.wcm_score[1], 1)}
                    </Typography>
                  ) : (
                    <Chip
                      label={word.wcm_score?.toFixed(1) || '0.0'}
                      size="small"
                      color={
                        (word.wcm_score || 0) < 5 ? 'success' :
                        (word.wcm_score || 0) < 10 ? 'warning' : 'error'
                      }
                    />
                  )}
                </TableCell>
                <TableCell align="center">
                  <Chip
                    label={word.complexity || 'N/A'}
                    size="small"
                    variant="outlined"
                  />
                </TableCell>
                <TableCell align="center">
                  <Typography variant="body2" color="text.secondary" sx={{ fontFamily: 'monospace' }}>
                    {word.isPair ? formatPair(word._raw.frequency[0], word._raw.frequency[1]) : (word.frequency ? word.frequency.toFixed(1) : '-')}
                  </Typography>
                </TableCell>
                <TableCell align="center">
                  <Typography variant="body2" color="text.secondary" sx={{ fontFamily: 'monospace' }}>
                    {word.isPair ? formatPair(word._raw.aoa[0], word._raw.aoa[1]) : (word.aoa ? word.aoa.toFixed(1) : '-')}
                  </Typography>
                </TableCell>
                <TableCell align="center">
                  <Typography variant="body2" color="text.secondary" sx={{ fontFamily: 'monospace' }}>
                    {word.isPair ? formatPair(word._raw.imageability[0], word._raw.imageability[1]) : (word.imageability ? word.imageability.toFixed(1) : '-')}
                  </Typography>
                </TableCell>
                <TableCell align="center">
                  <Typography variant="body2" color="text.secondary" sx={{ fontFamily: 'monospace' }}>
                    {word.isPair ? formatPair(word._raw.familiarity[0], word._raw.familiarity[1]) : (word.familiarity ? word.familiarity.toFixed(1) : '-')}
                  </Typography>
                </TableCell>
                <TableCell align="center">
                  <Typography variant="body2" color="text.secondary" sx={{ fontFamily: 'monospace' }}>
                    {word.isPair ? formatPair(word._raw.concreteness[0], word._raw.concreteness[1]) : (word.concreteness ? word.concreteness.toFixed(1) : '-')}
                  </Typography>
                </TableCell>
                <TableCell align="center">
                  <Typography variant="body2" color="text.secondary" sx={{ fontFamily: 'monospace' }}>
                    {word.isPair ? formatPair(word._raw.valence[0], word._raw.valence[1]) : (word.valence ? word.valence.toFixed(1) : '-')}
                  </Typography>
                </TableCell>
                <TableCell align="center">
                  <Typography variant="body2" color="text.secondary" sx={{ fontFamily: 'monospace' }}>
                    {word.isPair ? formatPair(word._raw.arousal[0], word._raw.arousal[1]) : (word.arousal ? word.arousal.toFixed(1) : '-')}
                  </Typography>
                </TableCell>
                <TableCell align="center">
                  <Typography variant="body2" color="text.secondary" sx={{ fontFamily: 'monospace' }}>
                    {word.isPair ? formatPair(word._raw.dominance[0], word._raw.dominance[1]) : (word.dominance ? word.dominance.toFixed(1) : '-')}
                  </Typography>
                </TableCell>
                {(showSimilarity || isSimilarWords) && (
                  <TableCell align="center">
                    <Chip
                      label={word.similarity?.toFixed(3) || 'N/A'}
                      size="small"
                      color="primary"
                    />
                  </TableCell>
                )}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
      )}
    </Box>
  );
};

export default WordResultsDisplay;
