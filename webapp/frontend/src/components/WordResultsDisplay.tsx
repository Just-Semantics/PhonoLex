/**
 * Word Results Display Component
 *
 * Displays word results from API calls with:
 * - Table view with sortable columns
 * - Export to CSV functionality
 * - Phonological property display (WCM, MSH, syllables)
 * - Similarity scores (when applicable)
 */

import React, { useState, useMemo, useRef, useEffect } from 'react';
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
  Fade,
} from '@mui/material';
import {
  Download as ExportIcon,
  ContentCopy as CopyIcon,
  ViewList as TableViewIcon,
  ViewModule as CardViewIcon,
  SwipeRounded as ScrollIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
} from '@mui/icons-material';
import type { Word, MinimalPair, SimilarWord } from '../services/phonolexApi';

type SortField = 'word' | 'wcm' | 'msh' | 'syllable_count' | 'similarity' | 'frequency' | 'aoa' | 'imageability' | 'familiarity' | 'concreteness' | 'valence' | 'arousal' | 'dominance';
type SortDirection = 'asc' | 'desc';

// Display word types with discriminated union
type DisplayWordBase = {
  word: string;
  ipa: string;
  wcm_score: number | null;
  msh_stage: number | null;
  syllable_count: number | null;
  frequency: number | null;
  aoa: number | null;
  imageability: number | null;
  familiarity: number | null;
  concreteness: number | null;
  valence: number | null;
  arousal: number | null;
  dominance: number | null;
};

type DisplayWordPair = DisplayWordBase & {
  isPair: true;
  _raw: {
    wcm_score: [number | null, number | null];
    msh_stage: [number | null, number | null];
    syllable_count: [number | null, number | null];
    frequency: [number | null, number | null];
    aoa: [number | null, number | null];
    imageability: [number | null, number | null];
    familiarity: [number | null, number | null];
    concreteness: [number | null, number | null];
    valence: [number | null, number | null];
    arousal: [number | null, number | null];
    dominance: [number | null, number | null];
  };
  phoneme1?: string;
  phoneme2?: string;
  position?: number;
};

type DisplayWordSimilar = DisplayWordBase & {
  isPair: false;
  similarity: number;
};

type DisplayWordRegular = DisplayWordBase & {
  isPair: false;
};

type DisplayWord = DisplayWordPair | DisplayWordSimilar | DisplayWordRegular;

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
  const [showScrollHint, setShowScrollHint] = useState(true);
  const [isExpanded, setIsExpanded] = useState(false);
  const tableContainerRef = useRef<HTMLDivElement>(null);

  // Threshold for showing collapse/expand - show if more than 50 results
  const COLLAPSE_THRESHOLD = 50;
  const COLLAPSED_DISPLAY_COUNT = 25;

  console.log('WordResultsDisplay:', { results, length: results.length });

  // Hide scroll hint after user scrolls
  useEffect(() => {
    const container = tableContainerRef.current;
    if (!container) return;

    const handleScroll = () => {
      setShowScrollHint(false);
    };

    container.addEventListener('scroll', handleScroll);
    return () => container.removeEventListener('scroll', handleScroll);
  }, []);

  // Reset scroll hint when view mode changes
  useEffect(() => {
    if (viewMode === 'table') {
      setShowScrollHint(true);
    }
  }, [viewMode]);

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
  const displayWords = useMemo((): DisplayWord[] => {
    if (isMinimalPairs) {
      // Show pairs as single row: word1 / word2
      const pairs = results as MinimalPair[];
      return pairs.map(pair => ({
        word: `${pair.word1.word} / ${pair.word2.word}`,
        ipa: `${pair.word1.ipa?.replace(/\n/g, ' ')} / ${pair.word2.ipa?.replace(/\n/g, ' ')}`,
        wcm_score: avg(pair.word1.wcm_score, pair.word2.wcm_score),
        msh_stage: avg(pair.word1.msh_stage, pair.word2.msh_stage),
        syllable_count: avg(pair.word1.syllable_count, pair.word2.syllable_count),
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
          msh_stage: [pair.word1.msh_stage, pair.word2.msh_stage],
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
    sorted.sort((a, b) => {
      let aVal: string | number | null, bVal: string | number | null;

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
          aVal = a.msh_stage || 0;
          bVal = b.msh_stage || 0;
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
          aVal = (a as DisplayWordSimilar).similarity || 0;
          bVal = (b as DisplayWordSimilar).similarity || 0;
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

  // Display words (with collapse functionality for long lists)
  const displayedWords = useMemo(() => {
    const shouldCollapse = sortedWords.length > COLLAPSE_THRESHOLD;
    if (shouldCollapse && !isExpanded) {
      return sortedWords.slice(0, COLLAPSED_DISPLAY_COUNT);
    }
    return sortedWords;
  }, [sortedWords, isExpanded, COLLAPSE_THRESHOLD, COLLAPSED_DISPLAY_COUNT]);

  const showCollapseControls = sortedWords.length > COLLAPSE_THRESHOLD;

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
      'Word', 'IPA', 'Syllables', 'WCM', 'Frequency',
      'AoA', 'Imageability', 'Familiarity', 'Concreteness',
      'Valence', 'Arousal', 'Dominance'
    ];
    if (showSimilarity || isSimilarWords) headers.push('Similarity');

    const rows = sortedWords.map((w) => [
      w.word || '',
      w.ipa || '',
      w.syllable_count || '',
      w.wcm_score?.toFixed(2) || '',
      w.frequency?.toFixed(1) || '',
      w.aoa?.toFixed(1) || '',
      w.imageability?.toFixed(1) || '',
      w.familiarity?.toFixed(1) || '',
      w.concreteness?.toFixed(1) || '',
      w.valence?.toFixed(1) || '',
      w.arousal?.toFixed(1) || '',
      w.dominance?.toFixed(1) || '',
      ...(showSimilarity || isSimilarWords ? [(w as DisplayWordSimilar).similarity?.toFixed(3) || ''] : []),
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

  // Copy words to clipboard (line-delimited)
  const copyWords = () => {
    const text = sortedWords.map((w) => w.word).join('\n');
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
        spacing={{ xs: 1.5, sm: 2 }}
        sx={{ mb: { xs: 1.5, sm: 2 } }}
      >
        <Stack direction="row" alignItems="center" spacing={1}>
          <Typography variant="h6" sx={{ fontSize: { xs: '1rem', sm: '1.25rem' } }}>
            {results.length} {isMinimalPairs ? 'Minimal Pairs' : 'Words'} Found
          </Typography>
          {showCollapseControls && (
            <Chip
              label={isExpanded ? `Showing all ${results.length}` : `Showing ${COLLAPSED_DISPLAY_COUNT} of ${results.length}`}
              size="small"
              color={isExpanded ? 'primary' : 'default'}
              sx={{ fontSize: '0.75rem' }}
            />
          )}
        </Stack>
        <Stack
          direction={{ xs: 'column', sm: 'row' }}
          spacing={{ xs: 1, sm: 1 }}
          alignItems={{ xs: 'stretch', sm: 'center' }}
        >
          <ToggleButtonGroup
            value={viewMode}
            exclusive
            onChange={(_, newMode) => newMode && setViewMode(newMode)}
            size="small"
            fullWidth
            sx={{
              display: { xs: 'flex', md: 'none' },
              '& .MuiToggleButton-root': {
                minHeight: 44,
                flex: 1,
              },
            }}
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

          <Stack direction="row" spacing={{ xs: 1, sm: 1 }} sx={{ width: { xs: '100%', sm: 'auto' } }}>
            <Tooltip title="Copy words to clipboard">
              <Button
                size="small"
                startIcon={<CopyIcon />}
                onClick={copyWords}
                sx={{
                  minHeight: 44,
                  width: { xs: '100%', sm: 'auto' },
                }}
              >
                Copy
              </Button>
            </Tooltip>
            <Tooltip title="Export to CSV">
              <Button
                size="small"
                startIcon={<ExportIcon />}
                onClick={exportCSV}
                sx={{
                  minHeight: 44,
                  width: { xs: '100%', sm: 'auto' },
                }}
              >
                Export
              </Button>
            </Tooltip>
          </Stack>
        </Stack>
      </Stack>

      {/* Card View for Mobile */}
      {viewMode === 'cards' ? (
        <Stack spacing={{ xs: 1.5, sm: 2 }}>
          {displayedWords.map((word, idx) => (
            <Card
              key={idx}
              variant="outlined"
              sx={{
                transition: 'all 0.2s ease',
                '&:hover': {
                  boxShadow: 2,
                  borderColor: 'primary.main',
                },
              }}
            >
              <CardContent sx={{ px: { xs: 1.5, sm: 2 }, py: { xs: 1.5, sm: 2 }, '&:last-child': { pb: { xs: 1.5, sm: 2 } } }}>
                <Stack spacing={{ xs: 1.5, sm: 2 }}>
                  {/* Header: Word + IPA */}
                  <Box>
                    <Typography variant="h6" fontWeight={600} sx={{ fontSize: { xs: '1.125rem', sm: '1.25rem' }, color: 'primary.main' }}>
                      {word.word}
                    </Typography>
                    <Typography
                      variant="body2"
                      fontFamily="monospace"
                      color="text.secondary"
                      sx={{ fontSize: { xs: '0.8125rem', sm: '0.875rem' }, mt: 0.25 }}
                    >
                      {word.ipa}
                    </Typography>
                  </Box>

                  <Divider />

                  {/* Key Metrics */}
                  <Grid container spacing={{ xs: 1, sm: 1.5 }}>
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
                    {(showSimilarity || isSimilarWords) && (
                      <Grid item xs={6} sm={3}>
                        <Typography variant="caption" color="text.secondary">
                          Similarity
                        </Typography>
                        <Box>
                          <Chip
                            label={(word as DisplayWordSimilar).similarity?.toFixed(3) || 'N/A'}
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
                              {word.isPair ? formatPair(word._raw.frequency[0], word._raw.frequency[1]) : (word.frequency != null ? word.frequency.toFixed(1) : '-')}
                            </Typography>
                          </Grid>
                        )}
                        {word.aoa !== null && word.aoa !== undefined && (
                          <Grid item xs={6}>
                            <Typography variant="caption" color="text.secondary">
                              AoA
                            </Typography>
                            <Typography variant="body2" fontFamily="monospace">
                              {word.isPair ? formatPair(word._raw.aoa[0], word._raw.aoa[1]) : (word.aoa != null ? word.aoa.toFixed(1) : '-')}
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
                              {word.isPair ? formatPair(word._raw.imageability[0], word._raw.imageability[1]) : (word.imageability != null ? word.imageability.toFixed(1) : '-')}
                            </Typography>
                          </Grid>
                        )}
                        {word.familiarity !== null && word.familiarity !== undefined && (
                          <Grid item xs={4}>
                            <Typography variant="caption" color="text.secondary">
                              Famil
                            </Typography>
                            <Typography variant="body2" fontFamily="monospace">
                              {word.isPair ? formatPair(word._raw.familiarity[0], word._raw.familiarity[1]) : (word.familiarity != null ? word.familiarity.toFixed(1) : '-')}
                            </Typography>
                          </Grid>
                        )}
                        {word.concreteness !== null && word.concreteness !== undefined && (
                          <Grid item xs={4}>
                            <Typography variant="caption" color="text.secondary">
                              Concr
                            </Typography>
                            <Typography variant="body2" fontFamily="monospace">
                              {word.isPair ? formatPair(word._raw.concreteness[0], word._raw.concreteness[1]) : (word.concreteness != null ? word.concreteness.toFixed(1) : '-')}
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

          {/* Expand/Collapse Button for Cards View */}
          {showCollapseControls && (
            <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
              <Button
                variant="outlined"
                onClick={() => setIsExpanded(!isExpanded)}
                startIcon={isExpanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                size="small"
              >
                {isExpanded
                  ? `Collapse (showing ${COLLAPSED_DISPLAY_COUNT})`
                  : `Show All ${sortedWords.length} Results`}
              </Button>
            </Box>
          )}
        </Stack>
      ) : (
        /* Table View with Sticky Header */
        <Box sx={{ position: 'relative' }}>
          {/* Scroll Hint Overlay */}
          <Fade in={showScrollHint && isMobile} timeout={1000}>
            <Box
              sx={{
                position: 'absolute',
                top: '50%',
                right: 16,
                transform: 'translateY(-50%)',
                zIndex: 10,
                bgcolor: 'primary.main',
                color: 'white',
                px: 2,
                py: 1,
                borderRadius: 2,
                display: 'flex',
                alignItems: 'center',
                gap: 1,
                boxShadow: 3,
                pointerEvents: 'none',
              }}
            >
              <ScrollIcon />
              <Typography variant="caption" fontWeight={600}>
                Scroll to see more
              </Typography>
            </Box>
          </Fade>

          <TableContainer
            ref={tableContainerRef}
            component={Paper}
            sx={{
              overflowX: 'auto',
              overflowY: 'auto',
              maxHeight: '70vh', // Limit height to enable vertical scrolling
              WebkitOverflowScrolling: 'touch', // Smooth scrolling on iOS
              position: 'relative', // For sticky positioning context
              '&::-webkit-scrollbar': {
                height: 8,
                width: 8,
              },
              '&::-webkit-scrollbar-thumb': {
                backgroundColor: 'rgba(0,0,0,0.2)',
                borderRadius: 4,
              },
              '&::-webkit-scrollbar-track': {
                backgroundColor: 'rgba(0,0,0,0.05)',
              },
            }}
          >
          <Table size="small" sx={{ minWidth: { xs: 800, sm: 'auto' } }} stickyHeader>
          <TableHead>
            <TableRow>
              <TableCell
                sx={{
                  position: 'sticky',
                  left: 0,
                  zIndex: 3, // Above other cells but below sticky header
                  bgcolor: 'background.paper',
                  borderRight: 1,
                  borderColor: 'divider',
                  boxShadow: '2px 0 4px rgba(0,0,0,0.05)',
                }}
              >
                <TableSortLabel
                  active={sortField === 'word'}
                  direction={sortField === 'word' ? sortDirection : 'asc'}
                  onClick={() => handleSort('word')}
                >
                  Word
                </TableSortLabel>
              </TableCell>
              <TableCell sx={{ whiteSpace: 'nowrap' }}>IPA</TableCell>
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
                <Tooltip title="Age of Acquisition (1-7 rating: 1=earliest, 7=latest)">
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
            {displayedWords.map((word, idx) => (
              <TableRow key={idx} hover>
                <TableCell
                  sx={{
                    position: 'sticky',
                    left: 0,
                    zIndex: 1, // Above other body cells
                    bgcolor: 'background.paper',
                    borderRight: 1,
                    borderColor: 'divider',
                    boxShadow: '2px 0 4px rgba(0,0,0,0.05)',
                  }}
                >
                  <Typography variant="body2" fontWeight={500}>
                    {word.word}
                  </Typography>
                </TableCell>
                <TableCell sx={{ whiteSpace: 'nowrap' }}>
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
                      label={(word as DisplayWordSimilar).similarity?.toFixed(3) || 'N/A'}
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

      {/* Expand/Collapse Button */}
      {showCollapseControls && (
        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
          <Button
            variant="outlined"
            onClick={() => setIsExpanded(!isExpanded)}
            startIcon={isExpanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
            size="small"
          >
            {isExpanded
              ? `Collapse (showing ${COLLAPSED_DISPLAY_COUNT})`
              : `Show All ${sortedWords.length} Results`}
          </Button>
        </Box>
      )}
        </Box>
      )}
    </Box>
  );
};

export default WordResultsDisplay;
